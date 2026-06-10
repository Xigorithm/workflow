# vllm-ascend Focus Areas: Quantization, Pooling, Speculative Decoding

## 目录

1. [Quantization](#quantization)
2. [Pooling Models](#pooling-models)
3. [Speculative Decoding](#speculative-decoding)

---

## Quantization

### 架构概览

```
quantization/
├── __init__.py                # Lazy imports
├── compressed_tensors_config.py  # LLM-Compressor 适配
├── fp8_config.py              # FP8 量化（含 deepseek_v4_fp8）
├── modelslim_config.py        # ModelSlim（Ascend 原生量化）
├── method_adapters.py         # 适配层：AscendLinearMethod, AscendKVCacheMethod, AscendFusedMoEMethod
├── quant_parser.py            # MXFP 量化类型映射
├── quant_type.py              # QuantType 枚举
├── utils.py                   # 自动检测、模型文件获取
└── methods/
    ├── base.py                # ABC：AscendLinearScheme, AscendAttentionScheme, AscendMoEScheme
    ├── registry.py            # @register_scheme(quant_type, layer_type) 装饰器
    ├── w8a8_dynamic.py        # W8A8 动态量化
    ├── w4a8.py                # W4A8 量化
    ├── w4a16.py               # W4A16 MoE 量化
    ├── fp8.py                 # FP8 量化
    ├── kv_c8.py               # KV cache INT8 量化
    └── ...                    # 更多量化方案
```

### 类层次

```
QuantizationConfig (vLLM base)
├── AscendModelSlimConfig          # "ascend" — 解析 quant_model_description.json
├── AscendCompressedTensorsConfig  # "compressed-tensors" — 解析 config.json config_groups
└── AscendFp8Config                # "fp8" / "deepseek_v4_fp8"

LinearMethodBase (vLLM)
└── AscendLinearMethod             # 委托给 AscendLinearScheme
    └── AscendEmbeddingMethod

BaseKVCacheMethod (vLLM)
└── AscendKVCacheMethod            # 委托给 AscendAttentionScheme

FusedMoEMethodBase (vLLM)
└── AscendFusedMoEMethod           # 委托给 AscendMoEScheme

AscendLinearScheme (ABC)           # 抽象：get_weight(), apply()
├── AscendW8A8LinearMethod
├── AscendW8A8DynamicLinearMethod
├── AscendW4A8DynamicLinearMethod
└── ... (10+ 变体)

AscendMoEScheme (ABC)             # 抽象：get_weight(), get_dynamic_quant_param(), apply()
├── AscendW8A8DynamicFusedMoEMethod
├── AscendW4A8DynamicFusedMoEMethod
└── ... (5+ 变体)

AscendAttentionScheme (ABC)       # 抽象：apply()
├── AscendFAQuantAttentionMethod
└── AscendC8KVCacheAttentionMethod
```

### Scheme 注册机制

```python
# methods/registry.py
_SCHEME_REGISTRY: dict[tuple[str, str], type[Any]] = {}

def register_scheme(quant_type: str, layer_type: str):
    def decorator(cls):
        _SCHEME_REGISTRY[(quant_type, layer_type)] = cls
        return cls
    return decorator

# 使用示例
@register_scheme("W8A8_DYNAMIC", "linear")
class AscendW8A8DynamicLinearMethod(AscendLinearScheme):
    ...
```

### 新增量化方案流程

1. 在 `quant_type.py` 的 `QuantType` 枚举中添加新类型
2. 在 `methods/` 下创建实现文件，继承对应的 Scheme ABC
3. 使用 `@register_scheme(quant_type, layer_type)` 注册
4. 在 `methods/__init__.py` 中导出
5. 如需新的 QuantizationConfig，在对应 config 文件中注册

### 自动检测

`utils.py:detect_quantization_method` 在 `NPUPlatform.check_and_update_config()` 中调用：
1. 检查 `quant_model_description.json` → ModelSlim ("ascend")
2. 检查 `config.json` 中的 `"quant_method"` 字段
3. 无需显式 `--quantization` 参数

### ModelSlim packed_modules_model_mapping

ModelSlim 为 20+ 模型架构维护了 `packed_modules_model_mapping`，描述融合模块（qkv_proj、gate_up_proj、experts）到组件层的映射关系。新增模型架构时需要在此添加映射。

---

## Pooling Models

### 实现方式

vllm-ascend 没有独立的 pooling 模块，通过以下路径集成：

| 组件 | 文件 | 关键代码 |
|------|------|---------|
| Worker | `worker/worker.py` | `get_supported_pooling_tasks()` 委托给 model_runner |
| Input Batch | `worker/npu_input_batch.py` | `is_pooling_model` 标志 + `pooling_params`/`pooling_states` 字典 |
| Model Runner | `worker/model_runner_v1.py` | 根据 `is_pooling_model` 分支处理 |
| Scheduler | `core/recompute_scheduler.py` | 提取 `pooler_output` 并路由 |
| Attention | `worker/v2/attn_utils.py` | `runner_type == "pooling"` 特殊处理 |

### 关键模式

Pooling 是 vLLM 的 `runner_type` 之一。vllm-ascend 的适配：

```python
# npu_input_batch.py
class NPUInputBatch:
    def __init__(self, ..., is_pooling_model: bool = False, ...):
        self.is_pooling_model = is_pooling_model
        self.pooling_params: dict[str, PoolingParams] = {}
        self.pooling_states: dict[str, PoolingStates] = {}

# model_runner_v1.py
if self.is_pooling_model:
    # pooling-specific forward path
    pooler_output = self._execute_pooling(...)
```

### 扩展 Pooling 支持

新增 pooling 特性时：
1. 在 `NPUInputBatch` 中添加 pooling 相关状态
2. 在 `NPUModelRunner` 中实现 pooling forward 逻辑
3. 在 scheduler 中处理 pooling 输出路由
4. 无需 patch — 通过继承和 runner_type 分支实现

---

## Speculative Decoding

### 架构概览

```
spec_decode/
├── __init__.py                       # get_spec_decode_method() 分发器
├── llm_base_proposer.py             # AscendSpecDecodeBaseProposer（核心基类，~1800 行）
├── eagle_proposer.py                # Eagle/Eagle3/MTP
├── dflash_proposer.py               # DFlash（cross-attention 变体）
├── draft_proposer.py                # 独立 draft model
├── extract_hidden_states_proposer.py # 提取隐藏状态
├── medusa_proposer.py               # Medusa
├── ngram_proposer.py                # CPU ngram
├── ngram_proposer_npu.py            # NPU ngram
├── suffix_proposer.py               # Suffix decoding
└── utils.py
```

### 类层次

```
SpecDecodeBaseProposer (vLLM)
└── AscendSpecDecodeBaseProposer     # NPU 核心基类（~1800 行）
    ├── AscendEagleProposer          # eagle/eagle3/mtp
    │   └── AscendDflashProposer     # dflash（cross-attention）
    └── AscendDraftModelProposer     # 独立 draft model

MedusaProposer (vLLM)
└── AscendMedusaProposer

NgramProposer (vLLM)
└── AscendNgramProposer              # CPU-based

NgramProposerGPU (vLLM)
└── AscendNgramProposerNPU           # NPU-based

SuffixDecodingProposer (vLLM)
└── AscendSuffixDecodingProposer

ExtractHiddenStatesProposer (vLLM)
└── AscendExtractHiddenStatesProposer
```

### 分发器

```python
# spec_decode/__init__.py
def get_spec_decode_method(method, vllm_config, device, runner):
    mapping = {
        "ngram": AscendNgramProposer,
        "ngram_gpu": AscendNgramProposerNPU,
        "suffix": AscendSuffixDecodingProposer,
        "medusa": AscendMedusaProposer,
        "eagle": AscendEagleProposer,
        "eagle3": AscendEagleProposer,
        "mtp": AscendEagleProposer,
        "dflash": AscendDflashProposer,
        "draft_model": AscendDraftModelProposer,
        "extract_hidden_states": AscendExtractHiddenStatesProposer,
    }
    return mapping[method](vllm_config, device, runner)
```

### AscendSpecDecodeBaseProposer 核心实现

**构造函数**：初始化 ACLGraphWrapper、TP group、PCP/DCP 状态、预分配 slot_mapping

**模型加载** (`load_model`)：
- 通过 `get_model()` 加载 draft model，tag 为 `"eagle_head"`
- 共享 target model 的 embedding 和 lm_head（节省内存）
- 用 `ACLGraphWrapper` 包装 `_run_merged_draft` 实现全图模式

**推测循环** (`_propose`)：
1. First pass：准备 input_ids、positions、slot_mappings
2. Multi-step drafting：循环执行 draft forward
3. 每步更新 attention metadata（`attn_update_stack_num_spec_norm`）
4. 通过 `greedy_sample()` + TP all-gather 获取 draft tokens

### NPU 特有优化

| 优化 | 说明 |
|------|------|
| ACL Graph | `ACLGraphWrapper` 包装 draft forward，避免重复编译 |
| Ascend Forward Context | `set_ascend_forward_context(is_draft_model=True, draft_attn_metadatas=...)` |
| Triton Kernels | `prepare_inputs_padded_kernel`、`copy_and_expand_dflash_inputs_kernel` |
| Greedy + TP All-Gather | 分布式 vocab 下的 greedy sampling |
| PCP/DCP | 长序列推测的上下文并行，block table 克隆和 slot mapping 管理 |
| Embedding 共享 | Target model 的 embedding/lm_head 与 draft model 共享 |
| Draft TP=1 Override | Draft model TP 与 target 不同时，创建单 rank TP group |

### 新增 Spec Decode 方法流程

1. 在 `spec_decode/` 下创建 `xxx_proposer.py`
2. 继承合适的基类（`AscendSpecDecodeBaseProposer` 或 vLLM 的 proposer）
3. 实现 `propose()` 和 `dummy_run()` 方法
4. 在 `spec_decode/__init__.py` 的 `get_spec_decode_method()` 中注册
5. 如需 NPU 特定 patch（如 rejection sampler），在 `patch/worker/` 中添加

### 相关 Patch

| Patch 文件 | 功能 |
|-----------|------|
| `patch_rejection_sampler.py` | NPU rejection sampler（top_k_top_p、expand_batch_to_tokens） |
| `patch_draft_quarot.py` | Draft model 的 Ascend 量化权重加载 |
| `patch_minimax_m2.py` | MiniMax-M2 Eagle3 辅助隐藏状态 |
| `patch_deepseek_mtp.py` | MTP 层权重加载（rotary quant） |
| `patch_qwen3_dflash.py` | DFlash NPU 算子适配 |
| `patch_v2/patch_triton.py` | V2 worker 的 rejection sampler triton ops |
