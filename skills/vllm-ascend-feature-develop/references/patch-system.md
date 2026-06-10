# vllm-ascend Patch System

## 目录

1. [架构概览](#架构概览)
2. [两阶段 Patch 应用](#两阶段-patch-应用)
3. [Patch 模式分类](#patch-模式分类)
4. [条件 Patch](#条件-patch)
5. [Patch 文档规范](#patch-文档规范)
6. [新增 Patch 流程](#新增-patch-流程)
7. [何时使用 Patch vs 继承](#何时使用-patch-vs-继承)

## 架构概览

vllm-ascend 通过 **运行时 monkey-patching** 修改上游 vLLM 类，无需 fork vLLM。所有 patch 集中在 `vllm_ascend/patch/` 目录下。

```
vllm_ascend/patch/
├── __init__.py          # 所有 patch 的文档记录（900+ 行）
├── platform/            # ~25 个文件，worker 启动前应用
│   ├── patch_config.py
│   ├── patch_scheduler.py
│   ├── patch_distributed.py
│   └── ...
└── worker/              # ~24 个文件，worker 启动时应用
    ├── patch_attention.py
    ├── patch_model.py
    ├── patch_rejection_sampler.py
    └── patch_v2/         # v2 model runner 专用 patch
        └── ...
```

## 两阶段 Patch 应用

调度函数位于 `vllm_ascend/utils.py`:

```python
def adapt_patch(is_global_patch: bool = False):
    if is_global_patch:
        from vllm_ascend.patch import platform  # noqa: F401
    else:
        from vllm_ascend.patch import worker  # noqa: F401
```

| 阶段 | 触发时机 | 调用位置 | 目录 |
|------|---------|---------|------|
| Platform patch | `NPUPlatform.pre_register_and_update()` | `adapt_patch(is_global_patch=True)` | `patch/platform/` |
| Worker patch | Worker `__init__` | `adapt_patch(is_global_patch=False)` | `patch/worker/` |

## Patch 模式分类

### 1. 直接属性替换

替换上游类的单个方法或属性：

```python
from vllm.v1.core.sched.scheduler import Scheduler
Scheduler.schedule = patched_schedule_method
```

### 2. 类替换

用 vllm-ascend 的实现替换整个上游类：

```python
import vllm.v1.core.sched.scheduler as scheduler_module
scheduler_module.Scheduler = BalanceScheduler
```

### 3. 子类化 + 替换

定义上游类的子类，然后替换引用：

```python
class AscendSampler(Sampler):
    def forward(self, ...):
        # NPU-specific sampling logic
        ...

import vllm.v1.sample.sampler as sampler_module
sampler_module.Sampler = AscendSampler
```

### 4. 模块级注入

添加上游模块缺少的函数或属性：

```python
import triton
triton.next_power_of_2 = lambda x: 1 << (x - 1).bit_length()
```

## 条件 Patch

`patch/platform/__init__.py` 和 `patch/worker/__init__.py` 使用条件判断：

```python
from vllm_ascend.utils import is_310p, vllm_version_is

# 硬件条件
if is_310p():
    from vllm_ascend.patch.worker import patch_310p_ops  # noqa

# 版本条件
if vllm_version_is("0.21.0"):
    from vllm_ascend.patch.worker.patch_v2 import patch_v2_triton  # noqa

# Triton 可用性
if HAS_TRITON:
    from vllm_ascend.patch.worker import patch_triton_ops  # noqa

# 环境变量
if envs.VLLM_ASCEND_ENABLE_DFLASH:
    from vllm_ascend.patch.worker import patch_qwen3_dflash  # noqa
```

## Patch 文档规范

每个 patch 必须在 `patch/__init__.py` 中记录，格式：

```python
# ============================================================================
# Patch: patch_file_name.py
# What: 描述被 patch 的对象
# Why: 为什么需要 patch
# How: patch 的实现方式
# Upstream PR: 相关的上游 PR（如果有）
# Future Plan: 何时可以移除这个 patch
# ============================================================================
```

## 新增 Patch 流程

1. 确认 patch 属于 platform 还是 worker 阶段
2. 在对应目录创建 `patch_xxx.py` 文件
3. 在对应目录的 `__init__.py` 中 import 该模块
4. 在 `patch/__init__.py` 中添加文档记录
5. 如需条件应用，在 `__init__.py` 中添加条件判断

## 何时使用 Patch vs 继承

| 场景 | 推荐方式 | 原因 |
|------|---------|------|
| 修改上游类的单个方法 | Patch（属性替换） | 最小侵入 |
| 需要替换整个类的行为 | 继承 + 类替换 | 可维护性好 |
| 添加上游没有的函数 | 模块注入 | 简单直接 |
| 修改上游类的 `__init__` | Patch 或子类化 | 视复杂度而定 |
| 平台级 hook（Platform 已提供） | 继承 Platform 方法 | 使用正式 hook |
| Worker 级扩展（WorkerBase 已提供） | 继承 NPUWorker | 使用正式 hook |
| Attention backend | 注册新 backend | vLLM 提供注册机制 |
| Quantization config | 注册新 config | vLLM 提供注册机制 |

**原则**：优先使用 vLLM 提供的正式扩展点（entry points、注册机制、Platform hooks）。仅在没有正式扩展点时才使用 patch。
