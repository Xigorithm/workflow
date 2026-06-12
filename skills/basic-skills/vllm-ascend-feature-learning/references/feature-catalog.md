# vLLM Ascend 特性目录

本文件列出 vllm-ascend 仓库中所有主要特性/模块，供生成学习文档时快速定位。

> **仓库路径**: `/home/y00884608/code/vllm-ascend`
> **源码根目录**: `vllm_ascend/`

## 插件架构

| 特性 | 说明 | 核心文件 |
|------|------|----------|
| 插件注册机制 | 通过 entry_points 注册 NPU 平台 | `setup.py`, `vllm_ascend/__init__.py` |
| NPUPlatform | 平台抽象层，设备管理/配置校验/Attention 后端选择 | `vllm_ascend/platform.py` |
| Patch 系统 | 运行时 monkey-patch 上游 vLLM 类 | `vllm_ascend/patch/` |
| Platform Patch | 全局级 patch（分布式、调度、KV Cache 等） | `vllm_ascend/patch/platform/` |
| Worker Patch | Worker 级 patch（模型特定行为、Triton 算子） | `vllm_ascend/patch/worker/` |
| AscendConfig | 分层配置系统 | `vllm_ascend/ascend_config.py` |
| 环境变量 | 集中式环境变量管理 | `vllm_ascend/envs.py` |

## Worker 与模型执行

| 特性 | 说明 | 核心文件 |
|------|------|----------|
| NPUWorker | NPU Worker 实现 | `vllm_ascend/worker/worker.py` |
| NPUModelRunner (v1) | v1 模型执行器（4900+ 行） | `vllm_ascend/worker/model_runner_v1.py` |
| v2 ModelRunner | v2 模型执行器 | `vllm_ascend/worker/v2/model_runner.py` |
| v2 InputBatch | NPU 输入批处理 | `vllm_ascend/worker/v2/input_batch.py` |
| v2 BlockTable | NPU Block Table（int32 slot mapping） | `vllm_ascend/worker/v2/block_table.py` |
| ACL Graph | NPU 版 CUDA Graph（torch.npu.NPUGraph） | `vllm_ascend/compilation/acl_graph.py` |
| npugraph_ex 后端 | NPU 图编译后端 | `vllm_ascend/compilation/compiler_interface.py` |
| 图融合 Pass | Norm+Quant、QKNorm+RoPE、AllReduce+RMSNorm 等融合 | `vllm_ascend/compilation/passes/` |

## Attention 后端

| 特性 | 说明 | 核心文件 |
|------|------|----------|
| 标准 Attention | Paged Attention + Context Parallelism | `vllm_ascend/attention/attention_v1.py` |
| MLA (Multi-head Latent Attention) | DeepSeek 系列 MLA 实现 | `vllm_ascend/attention/mla_v1.py` |
| SFA (Sparse Fine-grained Attention) | 稀疏细粒度注意力 | `vllm_ascend/attention/sfa_v1.py` |
| DSA (DeepSeek Sparse Attention) | DeepSeek 稀疏注意力 | `vllm_ascend/attention/dsa_v1.py` |
| FA3 (Flash Attention 3) | 训推一致性 Flash Attention | `vllm_ascend/attention/fa3_v1.py` |
| Context Parallelism | PCP/DCP 上下文并行 | `vllm_ascend/attention/context_parallel/` |
| KV Compression | Hamming Sparse KV 压缩注意力 | `vllm_ascend/attention/kvcomp_attn/` |
| Attention Mask | Attention Mask 构建工具 | `vllm_ascend/attention/attention_mask.py` |

## 分布式与通信

| 特性 | 说明 | 核心文件 |
|------|------|----------|
| NPUCommunicator | NPU 设备通信（all_to_all, all_reduce） | `vllm_ascend/distributed/device_communicators/npu_communicator.py` |
| HCCL Wrapper | 华为集合通信库封装 | `vllm_ascend/distributed/device_communicators/pyhccl.py` |
| Parallel State | Ascend 并行组管理（MC2、MLP TP、FlashComm2） | `vllm_ascend/distributed/parallel_state.py` |
| FlashComm1 | Ascend 通信优化（高并发场景） | 通过 `AscendConfig.enable_flashcomm1` 启用 |
| FlashComm2 | O-matrix TP 分组通信优化 | 通过 `VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE` 配置 |
| MC2 算子 | Ascend MoE dispatch/combine 算子 | 通过 `VLLM_ASCEND_ENABLE_FUSED_MC2` 启用 |
| KV Transfer | PD 分离 KV Cache 传输（P2P + Pool） | `vllm_ascend/distributed/kv_transfer/` |
| PD Disaggregation | Prefill-Decode 分离部署 | `vllm_ascend/distributed/kv_transfer/` |

## 量化

| 特性 | 说明 | 核心文件 |
|------|------|----------|
| W8A8 Dynamic | W8A8 动态量化 | `vllm_ascend/quantization/methods/w8a8_dynamic.py` |
| W8A8 Static | W8A8 静态量化 | `vllm_ascend/quantization/methods/w8a8_static.py` |
| W8A8 MXFP8 | W8A8 MX FP8 量化 | `vllm_ascend/quantization/methods/w8a8_mxfp8.py` |
| W4A16 | W4A16 量化 | `vllm_ascend/quantization/methods/w4a16.py` |
| W4A8 | W4A8 量化 | `vllm_ascend/quantization/methods/w4a8.py` |
| W4A4 系列 | FlatQuant/LAOS/MXFP4 等 W4A4 变体 | `vllm_ascend/quantization/methods/w4a4_*.py` |
| FP8 | FP8 量化 | `vllm_ascend/quantization/methods/fp8.py` |
| KV C8 | INT8 KV Cache 量化 | `vllm_ascend/quantization/methods/kv_c8.py` |
| 量化注册表 | 量化方法注册与自动检测 | `vllm_ascend/quantization/methods/registry.py` |

## 算子与 Triton

| 特性 | 说明 | 核心文件 |
|------|------|----------|
| AscendRMSNorm | NPU 优化的 RMSNorm | `vllm_ascend/ops/layernorm.py` |
| Rotary Embedding | NPU 旋转位置编码 | `vllm_ascend/ops/rotary_embedding.py` |
| Fused MoE | 完整 MoE 融合实现（12 个文件） | `vllm_ascend/ops/fused_moe/` |
| Triton NPU 适配 | 19+ Triton 内核 NPU 适配 | `vllm_ascend/ops/triton/` |
| MLA 算子 | MLA 算子实现 | `vllm_ascend/ops/mla.py` |
| DSA 算子 | DeepSeek Sparse Attention 算子 | `vllm_ascend/ops/dsa.py` |
| GDN 算子 | Gated Delta Net 算子 | `vllm_ascend/ops/gdn.py` |
| Weight Prefetch | 权重预取（计算重叠） | `vllm_ascend/ops/weight_prefetch.py` |
| Layer Shard Linear | 层分片线性运算 | `vllm_ascend/ops/layer_shard_linear.py` |

## 推测解码 (Speculative Decoding)

| 特性 | 说明 | 核心文件 |
|------|------|----------|
| Eagle/Eagle3 Proposer | Eagle 推测解码 | `vllm_ascend/spec_decode/eagle_proposer.py` |
| Medusa Proposer | Medusa 推测解码 | `vllm_ascend/spec_decode/medusa_proposer.py` |
| N-gram Proposer | N-gram 推测 | `vllm_ascend/spec_decode/ngram_proposer.py` |
| DFlash Proposer | DFlash 推测解码 | `vllm_ascend/spec_decode/dflash_proposer.py` |
| Rejection Sampler | NPU 优化拒绝采样 | `vllm_ascend/sample/rejection_sampler.py` |
| Block Verify | 块验证增强拒绝采样 | 通过 `RejectionSamplerConfig` 配置 |

## 采样 (Sampling)

| 特性 | 说明 | 核心文件 |
|------|------|----------|
| AscendSampler | NPU 优化采样器（避免 CPU-NPU 同步） | `vllm_ascend/sample/sampler.py` |
| Penalties | NPU 适配惩罚（frequency/presence/repetition） | `vllm_ascend/sample/penalties.py` |

## 调度与批处理

| 特性 | 说明 | 核心文件 |
|------|------|----------|
| Dynamic Batch | SLO 感知动态 Token 预算 | `vllm_ascend/core/scheduler_dynamic_batch.py` |
| Profiling Chunk | 基于 profiling 的动态 chunk（二次模型预测） | `vllm_ascend/core/scheduler_profiling_chunk.py` |
| Recompute Scheduler | PD 分离重计算调度 | `vllm_ascend/core/recompute_scheduler.py` |
| Balance Scheduling | Prefill/Decode 分离调度 | 通过 `AscendConfig.enable_balance_scheduling` 启用 |

## 内存管理

| 特性 | 说明 | 核心文件 |
|------|------|----------|
| CaMem Allocator | CANN 内存分配器（Sleep Mode/RL 场景） | `vllm_ascend/device_allocator/camem.py` |
| KV Cache Offload | KV Cache CPU-NPU 卸载 | `vllm_ascend/kv_offload/` |
| Simple KV Offload | 简单 KV 卸载后端 | `vllm_ascend/simple_kv_offload/` |
| NZ Weight Format | Ascend FRACTAL_NZ 矩阵格式 | 通过 `VLLM_ASCEND_ENABLE_NZ` 配置 |
| KV Cache NZ | MLA 模型 PD 场景 NZ 格式 KV Cache | 通过 `AscendConfig.enable_kv_nz` 启用 |

## MoE 专项

| 特性 | 说明 | 核心文件 |
|------|------|----------|
| Fused MoE Layer | 融合 MoE 层 | `vllm_ascend/ops/fused_moe/fused_moe.py` |
| Token Dispatcher | Token 分发（ALLTOALL/MC2） | `vllm_ascend/ops/fused_moe/token_dispatcher.py` |
| Expert Selector | 专家选择 | `vllm_ascend/ops/fused_moe/experts_selector.py` |
| EPLB | 动态专家级负载均衡 | `vllm_ascend/eplb/` |
| Shared Expert DP | 共享专家数据并行 | 通过 `AscendConfig.enable_shared_expert_dp` 启用 |

## LoRA

| 特性 | 说明 | 核心文件 |
|------|------|----------|
| PunicaWrapperNPU | NPU 适配 Punica（BGMV/SGMV） | `vllm_ascend/lora/punica_npu.py` |
| LoRA Ops | NPU LoRA 算子 | `vllm_ascend/lora/lora_ops.py` |

## 模型实现

| 特性 | 说明 | 核心文件 |
|------|------|----------|
| DeepSeek V4 | Ascend DSV4 模型实现 | `vllm_ascend/models/deepseek_v4.py` |
| DSV4 MTP | DSV4 Multi-Token Prediction | `vllm_ascend/models/deepseek_v4_mtp.py` |

## Profiling

| 特性 | 说明 | 核心文件 |
|------|------|----------|
| TorchNPU Profiler | NPU Profiling 封装 | `vllm_ascend/profiler/torch_npu_profiler.py` |

## 硬件特定

| 特性 | 说明 | 核心文件 |
|------|------|----------|
| Ascend 310P | 310P 推理芯片专用支持 | `vllm_ascend/_310p/` |
| Xlite Graph Mode | openEuler Xlite 图模式 | `vllm_ascend/xlite/` |
| CPU Binding | Worker 进程 CPU 核绑定 | `vllm_ascend/cpu_binding.py` |

## NPU 独有特性

以下特性为 Ascend NPU 独有，学习文档中应重点阐述其与 GPU 版的差异：

1. **ACL Graph** vs CUDA Graph — `torch.npu.NPUGraph` 替代 `torch.cuda.CUDAGraph`
2. **npugraph_ex** — NPU 专用 Dynamo 图编译后端
3. **FlashComm1/2** — Ascend 通信优化（GPU 无对应）
4. **MC2 算子** — Ascend MoE 专用 dispatch/combine
5. **CaMem Allocator** — CANN 内存管理（Sleep Mode）
6. **NZ 格式** — FRACTAL_NZ 矩阵存储格式
7. **MLAPO** — DeepSeek W8A8 MLA Prefill 优化
8. **MatmulAllReduce** — A2 芯片融合算子
9. **Sequence Parallelism** — 基于 FlashComm 的序列并行
10. **EPLB** — MoE 动态专家负载均衡
