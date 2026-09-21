"""torch-inductor-dtype-view-scatter-guard core: guards a real
torch.compile (Inductor) correctness bug where a "dtype-view + custom
op + diagonal_scatter" call pattern produces WRONG VALUES (NaN instead
of the assigned constant) AND changes the compiled function's
return-value ALIASING contract compared to eager mode.

Upstream report: pytorch/pytorch#197408 ("[Inductor] dtype-view custom
op followed by diagonal_scatter returns wrong values and aliases
input"). Reproducer: view an int32 tensor as float32, mutate it in
place through a custom op, apply torch.diagonal_scatter, copy the
scatter result back into the int32 input, and return the scatter
result. Under eager, the returned tensor is independent of the input
and holds the assigned diagonal values. Under torch.compile
(fullgraph=True, enable_auto_functionalized_v2=True), the returned
tensor both (a) contains NaN on the diagonal instead of the assigned
value, and (b) aliases the input's storage, so a later in-place
mutation of the "output" also corrupts the "already consumed" input.

Suspected root cause per the issue's own diagnosis: the
fix_auto_functionalized_dtype_views pass runs AFTER reinplacing and
removes a clone that was still needed once a float32 view shares
storage with an int32 graph input.

This is a DISTINCT bug from the sibling
torch-inductor-scatter-copyback-alias-guard's #195451
(should_reinplace_scatter reinplacing a RETURNED generalized_scatter
result onto the input buffer with correct values but wrong aliasing).
Confirmed here on this host: applying that sibling repo's
safe_compiled_scatter_returning() -style "clone if it aliases an
input" wrapper would fix the ALIASING half of this bug but NOT the
NaN VALUE corruption -- the wrong numbers are already baked in before
the wrapper ever sees the tensor. This guard's
safe_compiled_dtype_view_diagonal_scatter() therefore does not simply
de-alias; it detects the divergence directly by comparing the
compiled result against a fresh eager computation of the same call on
the same inputs, and falls back to the eager result whenever they
disagree (values or aliasing), on every call. This is more expensive
than a fire-and-forget clone, but the value corruption makes a
clone-only fix insufficient here -- there is no cheap local signal
(unlike copy-back aliasing, where "does it alias an input" alone was
enough) that would let this guard skip the eager recomputation safely.

Confirmed via a fresh `gh api` read (not cached) that upstream issue
#197408 is OPEN as of this run.
"""
from __future__ import annotations

import dataclasses
import functools
from typing import Any, Callable, Dict, List, Sequence, Tuple


class TorchUnavailableError(RuntimeError):
    """Raised when torch cannot be imported."""


def _import_torch():
    try:
        import torch  # noqa: F401
    except Exception as exc:  # pragma: no cover - exercised only without torch
        raise TorchUnavailableError(
            "torch is required for diagnosis and guarding; install the "
            "'torch' extra."
        ) from exc
    return torch


_LIB_NAME = "dtype_view_scatter_guard_repro"
_LIB_REGISTERED = False
_LIB_HANDLE = None  # keep alive: torch.library.Library deregisters on GC


def _ensure_custom_op_registered(torch_module):
    """Register the reproducer's minimal in-place mutate() custom op
    exactly once per process. Uses a plain (non-scoped) library so the
    op survives across multiple diagnose() calls / torch.compile
    cache entries within one process, matching how a real user's
    custom op would be registered at import time (the upstream issue's
    own repro uses a scoped library per-call only because it is a
    single-shot standalone script; a long-lived guard process must not
    re-register the same op name repeatedly, which torch raises on).
    The Library object is kept alive in a module-level global: torch
    deregisters the op as soon as the Library instance is garbage
    collected, which silently breaks every subsequent call otherwise.
    """
    global _LIB_REGISTERED, _LIB_HANDLE
    if _LIB_REGISTERED:
        return
    lib = torch_module.library.Library(_LIB_NAME, "FRAGMENT")  # noqa: TOR901
    lib.define("mutate(Tensor(a!) x, Tensor y) -> ()")

    def _mutate(x, y):
        x.copy_(y)

    lib.impl("mutate", _mutate, "CompositeExplicitAutograd")
    torch_module.library.register_fake(
        f"{_LIB_NAME}::mutate", lambda x, y: None, lib=lib
    )
    _LIB_HANDLE = lib
    _LIB_REGISTERED = True


def _dtype_view_scatter_fn(torch_module):
    def fn(cache, data, diag):
        view = cache.view(torch_module.float32)
        getattr(torch_module.ops, _LIB_NAME).mutate(view, data)
        updated = torch_module.diagonal_scatter(view, diag)
        cache.copy_(updated.view(torch_module.int32))
        return updated

    return fn


def safe_compiled_dtype_view_diagonal_scatter(
    compiled_fn: Callable, eager_fn: Callable
) -> Callable:
    """Wrap a ``torch.compile``-produced callable exhibiting the
    #197408 dtype-view/diagonal_scatter bug. Unlike a pure aliasing
    fix, the compiled result can hold outright WRONG NUMBERS (NaN),
    so this wrapper cannot repair the compiled output in place -- it
    must recompute the same call eagerly on independent copies of the
    input tensors and compare. If the compiled and eager results agree
    on both values and aliasing-with-input, the (fast) compiled result
    is returned unchanged. If they disagree, the (slow but correct)
    eager result is returned instead, and the caller's own input
    tensor arguments are updated in place to match what eager produced
    (matching the reproducer's own in-place `cache.copy_(...)`
    contract) so callers relying on input-mutation side effects are
    not silently left with the compiled path's corrupted input.

    This trades away torch.compile's speed benefit on exactly the
    inputs affected by this bug, in exchange for correctness -- the
    only sound choice when the bug corrupts VALUES, not just identity.
    """
    torch_module = _import_torch()

    @functools.wraps(compiled_fn)
    def wrapper(cache, data, diag):
        cache_for_compiled = cache.clone()
        cache_for_eager = cache.clone()

        compiled_out = compiled_fn(cache_for_compiled, data.clone(), diag.clone())
        eager_out = eager_fn(cache_for_eager, data.clone(), diag.clone())

        values_match = torch_module.equal(
            torch_module.nan_to_num(compiled_out, nan=1e30),
            torch_module.nan_to_num(eager_out, nan=1e30),
        )
        compiled_aliases_input = (
            compiled_out.untyped_storage().data_ptr()
            == cache_for_compiled.untyped_storage().data_ptr()
        )
        eager_aliases_input = (
            eager_out.untyped_storage().data_ptr()
            == cache_for_eager.untyped_storage().data_ptr()
        )
        divergence = (not values_match) or (compiled_aliases_input != eager_aliases_input)

        if not divergence:
            cache.copy_(cache_for_compiled)
            return compiled_out

        cache.copy_(cache_for_eager)
        return eager_out.clone()

    return wrapper


@dataclasses.dataclass
class DtypeViewScatterCase:
    cache0: List[int]
    data: List[float]
    diag: List[float]
    eager_aliases_input: bool
    compiled_aliases_input: bool
    eager_values: List[float]
    compiled_values: List[float]
    compiled_values_match_eager: bool
    compiled_input_corrupted_after_output_mutation: bool
    guarded_values_match_eager: bool
    guarded_aliases_input_matches_eager: bool
    guarded_input_matches_eager_after_output_mutation: bool


def _run_case(
    torch_module, n: int, data_flat: Sequence[float], diag_flat: Sequence[float]
) -> DtypeViewScatterCase:
    _ensure_custom_op_registered(torch_module)
    fn = _dtype_view_scatter_fn(torch_module)

    def run(callable_):
        cache = torch_module.zeros((n, n), dtype=torch_module.int32)
        data = torch_module.tensor(list(data_flat), dtype=torch_module.float32).reshape(n, n)
        diag = torch_module.tensor(list(diag_flat), dtype=torch_module.float32)
        with torch_module.no_grad():
            from torch._inductor import config as inductor_config

            with inductor_config.patch(
                enable_auto_functionalized_v2=True, implicit_fallbacks=True
            ):
                out = callable_(cache, data, diag)
        return cache, out

    # Eager reference: correct values, independent (non-aliasing) output.
    eager_cache, eager_out = run(fn)
    eager_aliases_input = (
        eager_out.untyped_storage().data_ptr() == eager_cache.untyped_storage().data_ptr()
    )
    eager_values = eager_out.clone()

    # Native compiled: reproduce the #197408 bug.
    compiled_fn = torch_module.compile(fn, fullgraph=True)
    compiled_cache, compiled_out = run(compiled_fn)
    compiled_aliases_input = (
        compiled_out.untyped_storage().data_ptr()
        == compiled_cache.untyped_storage().data_ptr()
    )
    compiled_values = compiled_out.clone()
    compiled_values_match_eager = torch_module.equal(
        torch_module.nan_to_num(compiled_values, nan=1e30),
        torch_module.nan_to_num(eager_values, nan=1e30),
    )
    compiled_cache_before_mutation = compiled_cache.clone()
    compiled_out.add_(100.0)
    compiled_input_corrupted = not torch_module.equal(
        compiled_cache, compiled_cache_before_mutation
    )

    # Guarded: the fix under test. Fresh callables each case to avoid
    # any risk of a stale compiled cache masking the bug.
    guarded_fn = safe_compiled_dtype_view_diagonal_scatter(
        torch_module.compile(fn, fullgraph=True), fn
    )
    guarded_cache = torch_module.zeros((n, n), dtype=torch_module.int32)
    guarded_data = torch_module.tensor(list(data_flat), dtype=torch_module.float32).reshape(n, n)
    guarded_diag = torch_module.tensor(list(diag_flat), dtype=torch_module.float32)
    with torch_module.no_grad():
        from torch._inductor import config as inductor_config

        with inductor_config.patch(
            enable_auto_functionalized_v2=True, implicit_fallbacks=True
        ):
            guarded_out = guarded_fn(guarded_cache, guarded_data, guarded_diag)
    guarded_values_match_eager = torch_module.equal(
        torch_module.nan_to_num(guarded_out, nan=1e30),
        torch_module.nan_to_num(eager_values, nan=1e30),
    )
    guarded_aliases_input = (
        guarded_out.untyped_storage().data_ptr()
        == guarded_cache.untyped_storage().data_ptr()
    )
    guarded_aliasing_matches_eager = guarded_aliases_input == eager_aliases_input
    guarded_cache_before_mutation = guarded_cache.clone()
    guarded_out.add_(100.0)
    # Since the guard returns an independent eager-derived tensor when
    # divergence is detected, mutating it must NOT corrupt the input
    # tensor the caller still holds -- matching eager's own contract.
    guarded_input_matches_eager_after_mutation = torch_module.equal(
        guarded_cache, guarded_cache_before_mutation
    )

    return DtypeViewScatterCase(
        cache0=[0] * (n * n),
        data=list(data_flat),
        diag=list(diag_flat),
        eager_aliases_input=eager_aliases_input,
        compiled_aliases_input=compiled_aliases_input,
        eager_values=eager_values.flatten().tolist(),
        compiled_values=compiled_values.flatten().tolist(),
        compiled_values_match_eager=compiled_values_match_eager,
        compiled_input_corrupted_after_output_mutation=compiled_input_corrupted,
        guarded_values_match_eager=guarded_values_match_eager,
        guarded_aliases_input_matches_eager=guarded_aliasing_matches_eager,
        guarded_input_matches_eager_after_output_mutation=guarded_input_matches_eager_after_mutation,
    )


def diagnose(
    cases: Sequence[Tuple[int, Sequence[float], Sequence[float]]] = (
        (4, tuple(float(i) for i in range(16)), (-1.0, -1.0, -1.0, -1.0)),
        (3, tuple(float(i) for i in range(9)), (7.0, 7.0, 7.0)),
        (2, (1.0, 2.0, 3.0, 4.0), (0.0, 0.0)),
    ),
) -> Dict[str, Any]:
    """Reproduce the Inductor dtype-view/diagonal_scatter value +
    aliasing divergence from scratch against the currently installed
    torch build, for every case, and verify the wrapper restores
    eager's values and aliasing contract. Never trusts a cached/prior
    result -- every call re-runs the actual repro, including a fresh
    ``torch.compile`` per case."""
    torch_module = _import_torch()
    results = [_run_case(torch_module, n, data, diag) for (n, data, diag) in cases]

    any_native_value_bug = any(not c.compiled_values_match_eager for c in results)
    any_native_alias_bug = any(
        (not c.eager_aliases_input) and c.compiled_aliases_input for c in results
    )
    any_native_corruption = any(
        c.compiled_input_corrupted_after_output_mutation for c in results
    )
    guard_fully_correct = all(
        c.guarded_values_match_eager
        and c.guarded_aliases_input_matches_eager
        and c.guarded_input_matches_eager_after_output_mutation
        for c in results
    )

    return {
        "torch_version": torch_module.__version__,
        "issue_url": "https://github.com/pytorch/pytorch/issues/197408",
        "related_issue_url": "https://github.com/pytorch/pytorch/issues/195451",
        "cases": [dataclasses.asdict(c) for c in results],
        "any_native_value_bug": any_native_value_bug,
        "any_native_alias_bug": any_native_alias_bug,
        "any_native_corruption": any_native_corruption,
        "guard_fully_correct": guard_fully_correct,
    }
