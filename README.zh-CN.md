[![English](https://img.shields.io/badge/English-555555?style=flat)](README.md) [![简体中文](https://img.shields.io/badge/简体中文-555555?style=flat)](README.zh-CN.md)

# torch-inductor-dtype-view-scatter-guard

诊断并规避一个真实存在的 `torch.compile`（Inductor）正确性缺陷：
**dtype 视图 + 自定义算子 + `diagonal_scatter`** 的调用模式会产生
**错误的数值（NaN）**，并且改变了编译函数的**返回值别名（aliasing）
约定**，与 eager 模式不一致。

```python
def fn(cache, data, diag):
    view = cache.view(torch.float32)
    torch.ops.my_lib.mutate(view, data)      # 原地自定义算子
    updated = torch.diagonal_scatter(view, diag)
    cache.copy_(updated.view(torch.int32))
    return updated
```

- **Eager**：`fn(cache, data, diag)` 返回一个独立的张量，其中保存着
  正确赋值的对角线数值。
- **`torch.compile`（Inductor，`enable_auto_functionalized_v2=True`）**：
  返回张量的对角线为 **`NaN`**，而不是赋的值，**并且**该张量与
  `cache` 共享存储。之后对"输出"进行原地修改，会连带破坏"本应已被
  消费"的输入张量——没有任何报错或警告。

已在本机复现（torch 2.14.0，CPU）：

| 观测项 | Eager | 编译后 |
|---|---|---|
| 对角线数值 | 赋的值（如 `-1`） | `NaN` |
| 输出是否与输入共享内存 | `False` | `True` |
| `out.add_(100)` 后输入是否不变 | `True` | `False` |

上游 issue：[pytorch/pytorch#197408](https://github.com/pytorch/pytorch/issues/197408)
（"[Inductor] dtype-view custom op followed by diagonal_scatter returns
wrong values and aliases input"）。按 issue 原文的诊断，疑似根因是：
`fix_auto_functionalized_dtype_views` 这个 pass 在 reinplacing 之后
才运行，当一个 float32 视图与某个 int32 图输入共享存储时，它会移除
一个本应保留的 clone。截至本工具最近一次验证，该 issue 仍处于 open
且未解决状态。

**与同项目下的 `torch-inductor-scatter-copyback-alias-guard`
（pytorch#195451）明显不同：** 那个缺陷只改变了别名约定——返回的
*数值*本身是正确的。而这个缺陷同时破坏了*数值*和*别名*两方面。已在
本机验证：单独套用那个仓库"如果与输入共享内存就 clone"的修复方式，
只能修复别名部分，NaN 数值损坏依然存在——证明这是一条完全不同的代码
路径，而非重复缺陷。

## 安装与诊断

需要 Python 3.9+ 和一份 PyTorch。从源码安装：

```bash
git clone https://github.com/zhuhroscar-tech/torch-inductor-dtype-view-scatter-guard.git
cd torch-inductor-dtype-view-scatter-guard
python3 -m venv .venv
source .venv/bin/activate
python -m pip install '.[torch]'
torch-inductor-dtype-view-scatter-guard
torch-inductor-dtype-view-scatter-guard --json
```

如果你已经有可用的 PyTorch 环境，可以不装 `torch` extra，直接
`pip install .`。使用 `--no-color` 输出纯文本。

退出码：**0** 表示规避方案在所有测试用例上都成功恢复了 eager 的数值
与非别名约定（如果本机安装的版本恰好未复现该缺陷——例如上游修复已
发布——会打印"info"提示而非警告）；**1** 表示至少有一个用例未能恢复；
**2** 表示无法导入 PyTorch。规避检查通过本身并不代表上游缺陷在你的
版本上被复现了，需单独查看 `any_native_value_bug` /
`any_native_alias_bug` 字段。

## Python API

同时传入编译后的可调用对象和它对应的（未编译）eager 版本：

```python
from torch_inductor_dtype_view_scatter_guard import safe_compiled_dtype_view_diagonal_scatter

compiled_fn = torch.compile(fn, fullgraph=True)
safe_fn = safe_compiled_dtype_view_diagonal_scatter(compiled_fn, fn)

out = safe_fn(cache, data, diag)   # 数值与非别名约定均与 eager 一致
out.add_(100.0)                    # 安全：不会像原始 compiled_fn 那样破坏 cache
```

**重要提示——这不是一个廉价的"原地修补"包装器。** 由于该缺陷会破坏
*数值本身*（而不仅仅是对象身份），没有足够廉价的局部信号能够在不
重新计算的情况下检测出数值损坏。因此
`safe_compiled_dtype_view_diagonal_scatter()` 会在输入的独立副本上
**同时**调用编译版本和 eager 版本，比较数值与别名情况，一旦不一致
就返回 eager 的结果——这是用牺牲 `torch.compile` 在受影响输入上的
速度优势，换取正确性。它还会把 eager 路径对输入的原地修改同步写回
调用方原始的 `cache` 参数，避免依赖 `cache.copy_(...)` 原地副作用的
调用方，最终却拿到编译路径产生的、已损坏的输入。

## 真实局限性

- 仅在 CPU 上复现和测试过（torch 2.14.0，macOS arm64 本机 + Ubuntu
  CI）。未在 CUDA/MPS 上单独验证；上游 issue 本身的报告也是纯 CPU
  构建（`USE_CUDA=0`）。
- 需要 `config.patch(enable_auto_functionalized_v2=True,
  implicit_fallbacks=True)` 才能复现——这与上游 issue 的精确复现配置
  一致。这并非所有 torch 版本的 Inductor 默认配置；本工具的
  `diagnose()` 内部总是自行套用这个配置来做复现，但如果你自己的代码
  路径没有使用 `enable_auto_functionalized_v2`，这个具体的缺陷形态
  可能并不适用于你。
- 这是一个**用户侧规避方案**，不是上游修复；而且与纯别名规避不同，
  它总是要为一次 eager 重算付出代价——并不试图挽救编译路径的速度。
  如果你的工作负载中 dtype-view/自定义算子/diagonal_scatter 这种模式
  处于热路径上，在上游修复发布之前，更推荐直接避开这个模式（例如
  避免使用 dtype 视图，或不要返回 scatter 的结果），而不是套用这个
  包装器。
- `diagnose()` 每次调用都会针对当前安装的 torch 版本重新真实复现，
  从不假设某个特定版本受影响或不受影响。一旦上游修复发布，
  `any_native_value_bug` 和 `any_native_alias_bug` 会正确地报告为
  `false`。
- 不会尝试修补或 monkeypatch Inductor 内部实现——只是一个调用边界的
  包装器，无论安装了哪个 torch 版本都可以安全使用。

## 开发

```bash
python -m pip install -e '.[dev,torch]'
python -m pytest --cov=torch_inductor_dtype_view_scatter_guard --cov-report=term-missing
```

## 许可证

MIT
