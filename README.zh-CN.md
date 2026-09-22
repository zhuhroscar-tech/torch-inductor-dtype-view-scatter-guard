# torch-inductor-dtype-view-scatter-guard

本仓库已合并到 **torch-correctness-guards** 统一包中维护。

请改用新的 umbrella package：

```bash
git clone https://github.com/zhuhroscar-tech/torch-correctness-guards.git
cd torch-correctness-guards
python -m pip install -e ".[torch]"
torch-guard run dtype-view-scatter
```

Python API：

```python
from torch_correctness_guards import safe_compiled_dtype_view_diagonal_scatter
```

原仓库功能现在位于：

- CLI：`torch-guard run dtype-view-scatter`
- 模块：`torch_correctness_guards.guards.dtype_view_scatter`
- 统一仓库：https://github.com/zhuhroscar-tech/torch-correctness-guards

本源仓库将归档，以避免重复维护；历史记录仍可查阅。
