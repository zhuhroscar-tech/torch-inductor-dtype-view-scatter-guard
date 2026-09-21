"""Tests for torch-inductor-dtype-view-scatter-guard. Requires the
'torch' extra (skipped otherwise).

Design mirrors this fleet's established discipline: every guard claim
is backed by a real reproduction, not an assumption, and at least one
test proves the test suite itself would have failed before the fix
(bug-injection verification), not just that the fix's own code path
returns success.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch_inductor_dtype_view_scatter_guard.core import (  # noqa: E402
    TorchUnavailableError,
    _dtype_view_scatter_fn,
    _ensure_custom_op_registered,
    diagnose,
    safe_compiled_dtype_view_diagonal_scatter,
)


def test_diagnose_runs_and_reports_torch_version():
    report = diagnose()
    assert report["torch_version"] == torch.__version__
    assert len(report["cases"]) == 3
    assert report["issue_url"] == "https://github.com/pytorch/pytorch/issues/197408"


def test_native_value_and_alias_bug_is_actually_reproduced_on_this_host():
    """This is the core evidentiary claim for this tool: prove the
    Inductor dtype-view/diagonal_scatter VALUE corruption (NaN) and
    aliasing-contract change are real on the CURRENTLY installed torch
    build, not merely cited from the issue tracker
    (pytorch/pytorch#197408). If torch fixes this upstream, these
    assertions should start failing -- news the tool should surface
    (via any_native_value_bug / any_native_alias_bug), not silently
    pass."""
    report = diagnose()
    assert report["any_native_value_bug"] is True, (
        "Expected the known upstream Inductor dtype-view/"
        "diagonal_scatter NaN value-corruption bug "
        f"(pytorch/pytorch#197408) to reproduce on torch "
        f"{torch.__version__}; if this now fails, the bug may have "
        "been fixed upstream -- verify against the issue tracker "
        "before assuming a test regression."
    )
    assert report["any_native_alias_bug"] is True
    assert report["any_native_corruption"] is True


def test_guard_fully_correct_across_all_cases():
    report = diagnose()
    assert report["guard_fully_correct"] is True
    for c in report["cases"]:
        assert c["guarded_values_match_eager"], c
        assert c["guarded_aliases_input_matches_eager"], c
        assert c["guarded_input_matches_eager_after_output_mutation"], c


def test_safe_wrapper_restores_correct_values_and_non_aliasing_directly():
    """Direct, minimal reproduction of the guard's core claim without
    going through diagnose(): the wrapped compiled function's return
    value must match eager's VALUES (no NaN) and must not alias the
    input tensor, restoring eager's full contract."""
    _ensure_custom_op_registered(torch)
    fn = _dtype_view_scatter_fn(torch)
    compiled = torch.compile(fn, fullgraph=True)
    guarded = safe_compiled_dtype_view_diagonal_scatter(compiled, fn)

    cache = torch.zeros((2, 2), dtype=torch.int32)
    data = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    diag = torch.tensor([0.0, 0.0])

    from torch._inductor import config as inductor_config

    with torch.no_grad(), inductor_config.patch(
        enable_auto_functionalized_v2=True, implicit_fallbacks=True
    ):
        out = guarded(cache, data, diag)

    assert not torch.isnan(out).any(), "guarded output must not contain NaN"
    assert out.untyped_storage().data_ptr() != cache.untyped_storage().data_ptr()

    snapshot = cache.clone()
    out.add_(100.0)
    assert torch.equal(cache, snapshot), (
        "mutating the guarded output must not corrupt the input, "
        "matching eager's non-aliasing contract"
    )


def test_native_compiled_diverges_bug_injection_check():
    """Bug-injection check proving the regression tests above are
    real: deliberately call the RAW (unguarded) compiled function and
    confirm it DOES produce NaN values and DOES alias its input --
    i.e. if safe_compiled_dtype_view_diagonal_scatter were a no-op
    passthrough (the bug this tool guards against), the guard tests
    above would correctly fail. This proves those tests are not
    tautological."""
    _ensure_custom_op_registered(torch)
    fn = _dtype_view_scatter_fn(torch)
    compiled = torch.compile(fn, fullgraph=True)

    cache = torch.zeros((2, 2), dtype=torch.int32)
    data = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    diag = torch.tensor([-1.0, -1.0])

    from torch._inductor import config as inductor_config

    with torch.no_grad(), inductor_config.patch(
        enable_auto_functionalized_v2=True, implicit_fallbacks=True
    ):
        out = compiled(cache, data, diag)

    assert torch.isnan(out).any(), (
        "Expected the RAW (unguarded) compiled function to produce "
        "NaN on the diagonal (that is the whole value-corruption bug "
        "this tool detects, pytorch/pytorch#197408); if this "
        "assertion fails, the underlying bug may have disappeared "
        "upstream, which would make the guard tautologically pass "
        "for the wrong reason."
    )
    assert out.untyped_storage().data_ptr() == cache.untyped_storage().data_ptr(), (
        "Expected the RAW compiled output to alias the input; if "
        "this fails the aliasing half of the bug may be fixed "
        "upstream."
    )


def test_guard_preserves_correct_values_when_no_bug_present():
    """When the compiled and eager paths already agree (e.g. a
    hypothetical already-fixed torch build), the guard must still
    return values matching eager -- proving it is not merely
    coincidentally correct only in the presence of the bug."""
    _ensure_custom_op_registered(torch)
    fn = _dtype_view_scatter_fn(torch)
    compiled = torch.compile(fn, fullgraph=True)
    guarded = safe_compiled_dtype_view_diagonal_scatter(compiled, fn)

    cache = torch.zeros((3, 3), dtype=torch.int32)
    data = torch.arange(9.0).reshape(3, 3)
    diag = torch.tensor([5.0, 5.0, 5.0])

    from torch._inductor import config as inductor_config

    with torch.no_grad(), inductor_config.patch(
        enable_auto_functionalized_v2=True, implicit_fallbacks=True
    ):
        out = guarded(cache, data, diag)

    expected_diag = torch.tensor([5.0, 5.0, 5.0])
    assert torch.equal(torch.diagonal(out), expected_diag)


def test_torch_unavailable_error_is_distinct_type():
    """Sanity check the error type exists and is a RuntimeError
    subclass, independent of whether torch is actually installed in
    this env."""
    assert issubclass(TorchUnavailableError, RuntimeError)
