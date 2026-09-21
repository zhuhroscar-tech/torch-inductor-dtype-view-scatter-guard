[![English](https://img.shields.io/badge/English-555555?style=flat)](README.md) [![简体中文](https://img.shields.io/badge/简体中文-555555?style=flat)](README.zh-CN.md)

# torch-inductor-dtype-view-scatter-guard

Detect and guard a real `torch.compile` (Inductor) correctness bug: a
**dtype-view + custom-op + `diagonal_scatter`** call pattern produces
**wrong values (NaN)** and changes the compiled function's
**return-value aliasing contract**, compared to eager mode.

```python
def fn(cache, data, diag):
    view = cache.view(torch.float32)
    torch.ops.my_lib.mutate(view, data)      # in-place custom op
    updated = torch.diagonal_scatter(view, diag)
    cache.copy_(updated.view(torch.int32))
    return updated
```

- **Eager**: `fn(cache, data, diag)` returns an independent tensor
  holding the assigned diagonal values.
- **`torch.compile` (Inductor, `enable_auto_functionalized_v2=True`)**:
  the returned tensor holds **`NaN`** on the diagonal instead of the
  assigned value, **and** aliases `cache`'s storage. A later in-place
  mutation of the "output" also corrupts the "already consumed" input.

Reproduced on this host (torch 2.14.0, CPU):

| Observation | Eager | Compiled |
|---|---|---|
| Diagonal values | assigned value (e.g. `-1`) | `NaN` |
| Output aliases input | `False` | `True` |
| Input unchanged after `out.add_(100)` | `True` | `False` |

Upstream: [pytorch/pytorch#197408](https://github.com/pytorch/pytorch/issues/197408)
("[Inductor] dtype-view custom op followed by diagonal_scatter returns
wrong values and aliases input"). Suspected root cause per the issue's
own diagnosis: the `fix_auto_functionalized_dtype_views` pass runs
*after* reinplacing and removes a clone that is still needed once a
float32 view shares storage with an int32 graph input. Open and
unresolved as of this tool's last verification.

**Distinct from the sibling `torch-inductor-scatter-copyback-alias-guard`
(pytorch#195451):** that bug only changes the aliasing contract — the
*values* returned are correct. This bug corrupts *both* values and
aliasing. Confirmed on this host that applying the sibling repo's
"clone if it aliases an input" fix alone would repair the aliasing
half but leave the NaN value corruption untouched — proving this is a
genuinely different code path, not a duplicate.

## Install and diagnose

Requires Python 3.9+ and a PyTorch build. From source:

```bash
git clone https://github.com/zhuhroscar-tech/torch-inductor-dtype-view-scatter-guard.git
cd torch-inductor-dtype-view-scatter-guard
python3 -m venv .venv
source .venv/bin/activate
python -m pip install '.[torch]'
torch-inductor-dtype-view-scatter-guard
torch-inductor-dtype-view-scatter-guard --json
```

If you already manage a compatible PyTorch installation, install `.`
without the extra. Use `--no-color` for plain text output.

Exit codes: **0** means the guard restored eager's values and
non-aliasing contract for every test case (and prints an "info" line,
not a warning, if the underlying bug also happened not to reproduce on
your build — e.g. after an upstream fix lands), **1** means the guard
failed to restore the contract for at least one case, and **2** means
PyTorch could not be imported. A successful guard check does not by
itself mean the upstream bug was reproduced on your build; inspect
`any_native_value_bug` / `any_native_alias_bug` separately.

## Python API

Wrap a `torch.compile`-produced callable together with its (uncompiled)
eager equivalent:

```python
from torch_inductor_dtype_view_scatter_guard import safe_compiled_dtype_view_diagonal_scatter

compiled_fn = torch.compile(fn, fullgraph=True)
safe_fn = safe_compiled_dtype_view_diagonal_scatter(compiled_fn, fn)

out = safe_fn(cache, data, diag)   # matches eager's values and non-aliasing contract
out.add_(100.0)                    # safe: does NOT corrupt cache, unlike the raw compiled_fn
```

**Important — this is not a cheap fix-in-place wrapper.** Because the
bug corrupts *values* (not just object identity), there is no local
signal cheap enough to detect the corruption without recomputation.
`safe_compiled_dtype_view_diagonal_scatter()` therefore calls **both**
the compiled and the eager function on independent copies of the
inputs, compares values and aliasing, and returns the eager result
whenever they disagree — trading away `torch.compile`'s speed
advantage on exactly the inputs affected by this bug, in exchange for
correctness. It also copies the eager path's input mutation back into
your original `cache` tensor argument so callers relying on
`cache.copy_(...)`'s in-place side effect are not left with the
compiled path's corrupted input.

## Scope and limitations

- Reproduced and tested on CPU only (torch 2.14.0, macOS arm64 +
  Ubuntu CI). Not separately verified on CUDA/MPS; the upstream issue's
  own report is also CPU-only (`USE_CUDA=0` build) as of this writing.
- Requires `config.patch(enable_auto_functionalized_v2=True,
  implicit_fallbacks=True)` to reproduce — matches the upstream
  issue's exact reproduction configuration. This is not the Inductor
  default in every torch version; the guard's `diagnose()` always
  applies this config internally for its own reproduction, but if your
  own code path does not use `enable_auto_functionalized_v2`, this
  specific bug shape may not apply to you.
- This is a **userspace workaround**, not an upstream fix, and unlike
  a pure aliasing guard it always pays the cost of an eager
  recomputation — it does not attempt to salvage the compiled path's
  speed. If your workload's dtype-view/custom-op/diagonal_scatter
  pattern is on a hot path, consider avoiding the pattern (e.g.
  avoiding the dtype view, or not returning the scatter result)
  instead of wrapping it, until the upstream fix lands.
- `diagnose()` always re-runs the actual reproduction against whatever
  torch build is installed — it never assumes a specific PyTorch
  version is or isn't affected. If the upstream fix lands and ships,
  `any_native_value_bug` and `any_native_alias_bug` will correctly
  report `false`.
- Does not attempt to patch or monkeypatch Inductor internals — it is
  a call-boundary wrapper only, safe to use regardless of which torch
  version is installed.

## Development

```bash
python -m pip install -e '.[dev,torch]'
python -m pytest --cov=torch_inductor_dtype_view_scatter_guard --cov-report=term-missing
```

## License

MIT
