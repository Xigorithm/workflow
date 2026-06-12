# vllm-ascend Workflow Skills

**[English](#english)** | **[中文](#中文)**

---

## English

A collection of [OpenCode](https://opencode.ai) Agent Skills providing development workflow support for the [vllm-ascend](https://github.com/vllm-project/vllm-ascend) (vLLM Ascend NPU plugin) project.

### Skills Overview

| Skill | Status | Description |
|-------|--------|-------------|
| [vllm-ascend-feature-learning](#vllm-ascend-feature-learning-1) | Available | Generate Chinese technical learning docs for vllm-ascend features |
| [vllm-ascend-feature-develop](#vllm-ascend-feature-develop-1) | Available | Assist vllm-ascend feature development and code implementation |

### vllm-ascend-feature-learning

Generate comprehensive Chinese code walkthrough learning documents for features/modules in the vllm-ascend repository.

#### Trigger

Request to learn a vllm-ascend feature in the OpenCode TUI, e.g.:

```
我想了解 vllm-ascend 的 ACL Graph
帮我生成 vllm-ascend patch 系统的学习文档
学习 vllm-ascend 的 MLA attention 后端
```

#### Output Format

Generated learning documents include:

- **Mermaid architecture/flow/sequence/class diagrams** — Visualize system architecture and execution flow
- **GPU vs NPU comparison tables** — Compare upstream vLLM and vllm-ascend implementation differences
- **Source code walkthrough** — Reference real vllm-ascend code with file paths
- **Configuration reference** — List all relevant environment variables and config parameters
- **Plugin integration analysis** — Analyze patch/inheritance/entry point integration mechanisms

Documents are output to the `docs/feature_learning/` directory.

#### Covered Features

| Domain | Features |
|--------|----------|
| Plugin Architecture | Plugin Registration, Patch System, NPUPlatform, AscendConfig |
| Attention Backends | MLA, SFA, DSA, FA3, Context Parallelism |
| Distributed | FlashComm, MC2, HCCL, PD Disaggregation, KV Transfer |
| Graph Optimization | ACL Graph, npugraph_ex, Graph Fusion Pass |
| Quantization | W8A8, W4A16, FP8, KV C8, W4A4 series |
| MoE | Fused MoE, EPLB, Token Dispatcher |
| Speculative Decoding | Eagle, Medusa, N-gram, Rejection Sampler |
| Memory Management | CaMem Allocator, KV Offload, NZ Format |
| Scheduling | Dynamic Batch, Profiling Chunk, Balance Scheduling |
| Hardware-Specific | 310P Support, Xlite, CPU Binding |

Full feature list: [feature-catalog.md](skills/basic-skills/vllm-ascend-feature-learning/references/feature-catalog.md).

### vllm-ascend-feature-develop

Assist vllm-ascend feature development with code implementation, design document generation, and technical solutions.

#### Trigger

```
帮我在 vllm-ascend 中实现 W4A16 量化支持
为 vllm-ascend 添加 xxx 特性
vllm-ascend xxx 特性开发
```

#### References

- [patch-system.md](skills/basic-skills/vllm-ascend-feature-develop/references/patch-system.md) — Patch system usage guide
- [feature-patterns.md](skills/basic-skills/vllm-ascend-feature-develop/references/feature-patterns.md) — Feature development patterns and examples
- [focus-areas.md](skills/basic-skills/vllm-ascend-feature-develop/references/focus-areas.md) — Key focus areas

Design documents are output to the `docs/feature_develop/` directory.

### Generated Documents

#### Learning Documents (`docs/feature_learning/`)

| Document | Topic |
|----------|-------|
| llm_inference_pipeline.md | LLM Inference Pipeline |
| vllm_inference_pipeline.md | vLLM Inference Pipeline |
| pipeline_step.md | Inference Pipeline Steps |
| pooling.md | Pooling Feature |
| quantization.md | Quantization Overview |
| quantization_vllm_ascend.md | vllm-ascend Quantization |
| speculative_decoding.md | Speculative Decoding |

#### Design Documents (`docs/feature_develop/`)

| Document | Topic |
|----------|-------|
| design-w4a16-quantization-adaptation-guide.md | W4A16 Quantization Adaptation Design |

### Directory Structure

```
workflow/
├── code/                                       # Code workspace (reserved)
├── docs/                                       # Generated document output
│   ├── feature_develop/                        # Feature design docs
│   ├── feature_insight/                        # Feature deep insights
│   └── feature_learning/                       # Feature learning docs
├── sessions/                                   # Session records
├── skills/                                     # Agent Skills definitions
│   ├── basic-skills/                           # Basic skills
│   │   ├── vllm-ascend-feature-develop/        # Feature development
│   │   │   ├── SKILL.md
│   │   │   └── references/
│   │   │       ├── feature-patterns.md
│   │   │       ├── focus-areas.md
│   │   │       └── patch-system.md
│   │   └── vllm-ascend-feature-learning/       # Feature learning docs
│   │       ├── SKILL.md
│   │       └── references/
│   │           ├── feature-catalog.md
│   │           └── style-guide.md
│   ├── kv-pooling-skills/                      # KV pooling skills (reserved)
│   ├── quantization-skills/                    # Quantization skills (reserved)
│   └── speculative-decoding-skills/            # Spec decode skills (reserved)
└── README.md
```

### Prerequisites

- Install [OpenCode](https://opencode.ai/docs/)
- Configure LLM Provider API Key
- Clone [vllm-ascend](https://github.com/vllm-project/vllm-ascend) repository locally (for source code walkthrough)

### Installation

Configure the skills directory in OpenCode's skills path. Add to `opencode.json`:

```json
{
  "skills": {
    "paths": ["./skills"]
  }
}
```

### License

This project follows the same license as [vllm-ascend](https://github.com/vllm-project/vllm-ascend).

---

## 中文

[OpenCode](https://opencode.ai) Agent Skills 集合，为 [vllm-ascend](https://github.com/vllm-project/vllm-ascend)（vLLM Ascend NPU 插件）项目提供开发工作流支持。

### Skills 概览

| Skill | 状态 | 说明 |
|-------|------|------|
| [vllm-ascend-feature-learning](#vllm-ascend-feature-learning-2) | 可用 | 生成 vllm-ascend 特性的中文技术学习文档 |
| [vllm-ascend-feature-develop](#vllm-ascend-feature-develop-2) | 可用 | 辅助 vllm-ascend 特性开发与代码实现 |

### vllm-ascend-feature-learning

为 vllm-ascend 仓库的特性/模块生成全面的中文代码走读学习文档。

#### 触发方式

在 OpenCode TUI 中直接请求学习 vllm-ascend 的某个特性即可自动触发，例如：

```
我想了解 vllm-ascend 的 ACL Graph
帮我生成 vllm-ascend patch 系统的学习文档
学习 vllm-ascend 的 MLA attention 后端
```

#### 输出格式

生成的学习文档包含：

- **Mermaid 架构图/流程图/时序图/类图** — 可视化展示系统架构和执行流程
- **GPU vs NPU 对比表** — 对比上游 vLLM 与 vllm-ascend 的实现差异
- **源码走读** — 引用 vllm-ascend 真实代码并标注文件路径
- **配置参考** — 列出所有相关环境变量和配置参数
- **插件集成分析** — 分析 patch/继承/entry point 等集成机制

文档输出到 `docs/feature_learning/` 目录。

#### 覆盖特性

| 领域 | 特性 |
|------|------|
| 插件架构 | 插件注册、Patch 系统、NPUPlatform、AscendConfig |
| Attention 后端 | MLA、SFA、DSA、FA3、Context Parallelism |
| 分布式 | FlashComm、MC2、HCCL、PD Disaggregation、KV Transfer |
| 图优化 | ACL Graph、npugraph_ex、图融合 Pass |
| 量化 | W8A8、W4A16、FP8、KV C8、W4A4 系列 |
| MoE | Fused MoE、EPLB、Token Dispatcher |
| 推测解码 | Eagle、Medusa、N-gram、Rejection Sampler |
| 内存管理 | CaMem Allocator、KV Offload、NZ 格式 |
| 调度 | Dynamic Batch、Profiling Chunk、Balance Scheduling |
| 硬件特定 | 310P 支持、Xlite、CPU Binding |

完整特性列表见 [feature-catalog.md](skills/basic-skills/vllm-ascend-feature-learning/references/feature-catalog.md)。

### vllm-ascend-feature-develop

辅助 vllm-ascend 特性开发，提供代码实现、设计文档生成和技术方案输出。

#### 触发方式

```
帮我在 vllm-ascend 中实现 W4A16 量化支持
为 vllm-ascend 添加 xxx 特性
vllm-ascend xxx 特性开发
```

#### 参考资料

- [patch-system.md](skills/basic-skills/vllm-ascend-feature-develop/references/patch-system.md) — Patch 系统使用指南
- [feature-patterns.md](skills/basic-skills/vllm-ascend-feature-develop/references/feature-patterns.md) — 特性开发模式与范例
- [focus-areas.md](skills/basic-skills/vllm-ascend-feature-develop/references/focus-areas.md) — 重点关注领域

设计文档输出到 `docs/feature_develop/` 目录。

### 已生成的文档

#### 学习文档 (`docs/feature_learning/`)

| 文档 | 主题 |
|------|------|
| llm_inference_pipeline.md | LLM 推理流水线 |
| vllm_inference_pipeline.md | vLLM 推理流水线 |
| pipeline_step.md | 推理流水线各步骤详解 |
| pooling.md | 池化（Pooling）特性 |
| quantization.md | 量化技术概述 |
| quantization_vllm_ascend.md | vllm-ascend 量化实现 |
| speculative_decoding.md | 推测解码 |

#### 设计文档 (`docs/feature_develop/`)

| 文档 | 主题 |
|------|------|
| design-w4a16-quantization-adaptation-guide.md | W4A16 量化适配设计 |

### 目录结构

```
workflow/
├── code/                                       # 代码工作区（预留）
├── docs/                                       # 生成的文档输出
│   ├── feature_develop/                        # 特性设计文档
│   ├── feature_insight/                        # 特性深度洞察
│   └── feature_learning/                       # 特性学习文档
├── sessions/                                   # Session 记录
├── skills/                                     # Agent Skills 定义
│   ├── basic-skills/                           # 基础技能
│   │   ├── vllm-ascend-feature-develop/        # 特性开发辅助
│   │   │   ├── SKILL.md
│   │   │   └── references/
│   │   │       ├── feature-patterns.md
│   │   │       ├── focus-areas.md
│   │   │       └── patch-system.md
│   │   └── vllm-ascend-feature-learning/       # 特性学习文档生成
│   │       ├── SKILL.md
│   │       └── references/
│   │           ├── feature-catalog.md
│   │           └── style-guide.md
│   ├── kv-pooling-skills/                      # KV 池化技能（预留）
│   ├── quantization-skills/                    # 量化技能（预留）
│   └── speculative-decoding-skills/            # 推测解码技能（预留）
└── README.md
```

### 使用前提

- 安装 [OpenCode](https://opencode.ai/docs/)
- 配置 LLM Provider API Key
- 本地克隆 [vllm-ascend](https://github.com/vllm-project/vllm-ascend) 仓库（用于源码走读）

### 安装

将本仓库的 skills 目录配置到 OpenCode 的 skills 路径中。在 `opencode.json` 中添加：

```json
{
  "skills": {
    "paths": ["./skills"]
  }
}
```

### 许可证

本项目遵循与 [vllm-ascend](https://github.com/vllm-project/vllm-ascend) 相同的许可证。
