# vllm-ascend Feature Implementation Patterns

## 目录

1. [Attention Backend](#attention-backend)
2. [Custom Operator](#custom-operator)
3. [Worker Extension](#worker-extension)
4. [Scheduler Extension](#scheduler-extension)
5. [Platform Hook](#platform-hook)
6. [Entry Point Registration](#entry-point-registration)
7. [Environment Variables](#environment-variables)
8. [NPU Development Constraints](#npu-development-constraints)

## Attention Backend

vLLM 提供 attention backend 注册机制。vllm-ascend 实现了 5 个 backend。

### 实现模板

```python
from vllm.v1.attention.backends.abstract import AttentionBackend, AttentionImpl, AttentionMetadata
from vllm.attention.backends.registry import register_backend, AttentionBackendEnum

@register_backend(AttentionBackendEnum.CUSTOM, "ASCEND_MLA")
class AscendMLABackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "ASCEND_MLA"

    @staticmethod
    def get_impl_cls() -> type:
        return AscendMLAImpl

    @staticmethod
    def get_metadata_cls() -> type:
        return AscendMLAMetadata

    @staticmethod
    def get_builder_cls() -> type:
        return AscendMLAMetadataBuilder

class AscendMLAImpl(AttentionImpl):
    def __init__(self, ...):
        # NPU-specific initialization
        ...

    def forward(self, query, key, value, kv_cache, attn_metadata, ...):
        # NPU attention computation (torch_npu ops)
        ...
```

### Backend 选择逻辑

`NPUPlatform.get_attn_backend_cls()` 根据模型架构选择 backend：

| 模型类型 | Backend | 文件 |
|---------|---------|------|
| 标准 GQA/MHA | `AscendAttentionBackend` | `attention/attention_v1.py` |
| MLA (DeepSeek) | `AscendMLABackend` | `attention/mla_v1.py` |
| SFA | `AscendSFABackend` | `attention/sfa_v1.py` |
| DSA (DeepSeek Sparse) | `AscendDSABackend` | `attention/dsa_v1.py` |
| FA3 | `AscendFABackend` | `attention/fa3_v1.py` |

### 文件组织

```
attention/
├── abstract.py          # DSAAttentionImpl 抽象基类
├── attention_v1.py      # 标准 attention
├── mla_v1.py            # MLA
├── sfa_v1.py            # SFA
├── dsa_v1.py            # DSA
├── fa3_v1.py            # FA3
├── attention_mask.py    # Mask 构建工具
├── context_parallel/    # 上下文并行
└── utils.py
```

## Custom Operator

NPU 自定义算子放在 `ops/` 目录。

### 实现模板

```python
import torch
import torch_npu

class AscendRMSNorm:
    def __init__(self, hidden_size, eps=1e-6):
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        # 优先使用 torch_npu 融合算子
        return torch_npu.npu_rms_norm(x, self.weight, self.variance_epsilon)[0]
```

### 注册方式

通过 `ops/register_custom_ops.py` 注册，或通过 patch 替换上游 vLLM 的 op 实现：

```python
# 在 patch 中替换
from vllm.model_executor.layers.layernorm import RMSNorm
RMSNorm.forward = ascend_rms_norm_forward
```

### Triton-Ascend 算子

对于需要 Triton 的场景，使用 `triton-ascend` 后端：

```python
import triton
import triton.language as tl

@triton.jit
def my_npu_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = x * 2  # example computation
    tl.store(output_ptr + offsets, output, mask=mask)
```

## Worker Extension

### NPUWorker 继承模式

```python
from vllm.v1.worker.worker_base import WorkerBase

class NPUWorker(WorkerBase):
    def __init__(self, vllm_config, local_rank, rank, ...):
        super().__init__(vllm_config, local_rank, rank, ...)
        # NPU-specific init: torch_npu, ATB, CPU binding, distributed
        ...

    def load_model(self):
        # NPU model loading
        ...

    def get_supported_pooling_tasks(self):
        return self.model_runner.get_supported_pooling_tasks()
```

### Model Runner 扩展

`NPUModelRunner` 是最大的组件（~5000 行），处理：
- 模型加载与 KV cache 初始化
- Forward pass 执行
- ACL graph capture（NPU 版 CUDA graph）
- 输入 batch 管理

## Scheduler Extension

通过 patch 或子类化扩展 scheduler：

```python
from vllm.v1.core.sched.scheduler import Scheduler

class BalanceScheduler(Scheduler):
    def schedule(self):
        # NPU-specific scheduling with load balancing
        ...

# 替换上游 Scheduler
import vllm.v1.core.sched.scheduler as scheduler_module
scheduler_module.Scheduler = BalanceScheduler
```

其他 scheduler 扩展在 `core/` 目录：
- `scheduler_dynamic_batch.py` — 动态 batch 调度
- `scheduler_profiling_chunk.py` — Profiling chunk 调度
- `recompute_scheduler.py` — Recompute 调度

## Platform Hook

`NPUPlatform(Platform)` 通过 vLLM 的正式 hook 扩展平台行为：

| Hook 方法 | 用途 |
|----------|------|
| `check_and_update_config()` | 验证和修改 VllmConfig |
| `get_attn_backend_cls()` | 选择 attention backend |
| `pre_register_and_update()` | 应用全局 patch、注册量化 |
| `import_kernels()` | 引导自定义算子环境 |
| `set_additional_forward_context()` | 注入 NPU forward 上下文 |
| `get_device_communicator_cls()` | 返回 NPUCommunicator |
| `get_static_graph_wrapper_cls()` | 返回 ACLGraphWrapper |
| `apply_config_platform_defaults()` | 设置 Ascend 默认参数 |

## Entry Point Registration

在 `setup.py` 中注册：

```python
entry_points={
    "vllm.platform_plugins": ["ascend = vllm_ascend:register"],
    "vllm.general_plugins": [
        "ascend_kv_connector = vllm_ascend:register_connector",
        "ascend_model_loader = vllm_ascend:register_model_loader",
        "ascend_service_profiling = vllm_ascend:register_service_profiling",
        "ascend_model = vllm_ascend:register_model",
    ],
}
```

新增 connector/loader/model 时，在 `vllm_ascend/__init__.py` 的对应 `register_xxx()` 函数中添加注册逻辑。

## Environment Variables

所有环境变量集中定义在 `vllm_ascend/envs.py`：

```python
env_variables = {
    "VLLM_ASCEND_ENABLE_TOPK_TOPP": ("false", bool),
    "VLLM_ASCEND_MC2_COMM": ("0", int),
    "VLLM_ASCEND_FLASHCOMM2_ENABLE": ("false", bool),
    # ...
}
```

**规则**：
- 命名必须使用 `VLLM_ASCEND_*` 前缀
- 在 `envs.py` 中集中定义，其他文件通过 `from vllm_ascend import envs` 访问
- 禁止在其他文件中硬编码环境变量名

## NPU Development Constraints

### 性能陷阱

| 陷阱 | 说明 | 替代方案 |
|------|------|---------|
| `tensor.item()` | 触发 NPU→CPU 同步 | 使用 `tensor.cpu()` 或避免在热路径中使用 |
| 非 in-place 操作 | `x = x + 1` 产生额外内存拷贝 | 使用 `x.add_(1)` |
| CPU-NPU 内存传输 | 频繁的小数据传输成为瓶颈 | 批量传输或使用 pinned memory |
| `torch.cuda.*` 调用 | NPU 不支持 CUDA API | 使用 `torch_npu.npu.*` 对应 API |

### 编码规范

- 行宽：**120 字符**（非上游的 88 字符）
- Lint：`bash format.sh ci`
- 类型检查：`mypy --config-file mypy.ini vllm_ascend/`
- 测试：`pytest -sv tests/ut/path/to/test.py`
- Commit：必须 signed-off (`git commit -s`)

### torch_npu 常用 API 映射

| CUDA API | NPU 等价 API |
|----------|-------------|
| `torch.cuda.synchronize()` | `torch.npu.synchronize()` |
| `torch.cuda.current_device()` | `torch.npu.current_device()` |
| `torch.cuda.Event()` | `torch.npu.Event()` |
| `torch.cuda.Stream()` | `torch.npu.Stream()` |
| CUDA Graph | ACL Graph (`ACLGraphWrapper`) |
| NCCL | HCCL (`NPUCommunicator`) |
