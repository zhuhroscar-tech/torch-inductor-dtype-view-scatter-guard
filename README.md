# torch-inductor-dtype-view-scatter-guard

This repository has been consolidated into **torch-correctness-guards**.

Use the maintained umbrella package instead:

```bash
git clone https://github.com/zhuhroscar-tech/torch-correctness-guards.git
cd torch-correctness-guards
python -m pip install -e ".[torch]"
torch-guard run dtype-view-scatter
```

Python API:

```python
from torch_correctness_guards import safe_compiled_dtype_view_diagonal_scatter
```

The original functionality from this repo now lives in:

- CLI: `torch-guard run dtype-view-scatter`
- Module: `torch_correctness_guards.guards.dtype_view_scatter`
- Umbrella repo: https://github.com/zhuhroscar-tech/torch-correctness-guards

This source repo is archived to prevent split maintenance; history remains available for reference.
