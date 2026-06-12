# vllm-ascend Workflow Skills

**English** | [中文](README.zh-CN.md)

A collection of [OpenCode](https://opencode.ai) Agent Skills providing development workflow support for the [vllm-ascend](https://github.com/vllm-project/vllm-ascend) (vLLM Ascend NPU plugin) project.

## Skills Overview

| Skill | Status | Description |
|-------|--------|-------------|
| [vllm-ascend-feature-learning](#vllm-ascend-feature-learning) | Available | Generate Chinese technical learning docs for vllm-ascend features |
| [vllm-ascend-feature-develop](#vllm-ascend-feature-develop) | Available | Assist vllm-ascend feature development and code implementation |

## vllm-ascend-feature-learning

Generate comprehensive Chinese code walkthrough learning documents for features/modules in the vllm-ascend repository.

### Trigger

Request to learn a vllm-ascend feature in the OpenCode TUI, e.g.:

```
我想了解 vllm-ascend 的 ACL Graph
帮我生成 vllm-ascend patch 系统的学习文档
学习 vllm-ascend 的 MLA attention 后端
```

### Output Format

Generated learning documents include:

- **Mermaid architecture/flow/sequence/class diagrams** — Visualize system architecture and execution flow
- **GPU vs NPU comparison tables** — Compare upstream vLLM and vllm-ascend implementation differences
- **Source code walkthrough** — Reference real vllm-ascend code with file paths
- **Configuration reference** — List all relevant environment variables and config parameters
- **Plugin integration analysis** — Analyze patch/inheritance/entry point integration mechanisms

Documents are output to the `docs/feature_learning/` directory.

### Covered Features

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

## vllm-ascend-feature-develop

Assist vllm-ascend feature development with code implementation, design document generation, and technical solutions.

### Trigger

```
帮我在 vllm-ascend 中实现 W4A16 量化支持
为 vllm-ascend 添加 xxx 特性
vllm-ascend xxx 特性开发
```

### References

- [patch-system.md](skills/basic-skills/vllm-ascend-feature-develop/references/patch-system.md) — Patch system usage guide
- [feature-patterns.md](skills/basic-skills/vllm-ascend-feature-develop/references/feature-patterns.md) — Feature development patterns and examples
- [focus-areas.md](skills/basic-skills/vllm-ascend-feature-develop/references/focus-areas.md) — Key focus areas

Design documents are output to the `docs/feature_develop/` directory.

## Generated Documents

### Learning Documents (`docs/feature_learning/`)

| Document | Topic |
|----------|-------|
| llm_inference_pipeline.md | LLM Inference Pipeline |
| vllm_inference_pipeline.md | vLLM Inference Pipeline |
| pipeline_step.md | Inference Pipeline Steps |
| pooling.md | Pooling Feature |
| quantization.md | Quantization Overview |
| quantization_vllm_ascend.md | vllm-ascend Quantization |
| speculative_decoding.md | Speculative Decoding |

### Design Documents (`docs/feature_develop/`)

| Document | Topic |
|----------|-------|
| design-w4a16-quantization-adaptation-guide.md | W4A16 Quantization Adaptation Design |

## Directory Structure

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

## Prerequisites

- Install [OpenCode](https://opencode.ai/docs/)
- Configure LLM Provider API Key
- Clone [vllm-ascend](https://github.com/vllm-project/vllm-ascend) repository locally (for source code walkthrough)

## Installation

Configure the skills directory in OpenCode's skills path. Add to `opencode.json`:

```json
{
  "skills": {
    "paths": ["./skills"]
  }
}
```

## License

This project follows the same license as [vllm-ascend](https://github.com/vllm-project/vllm-ascend).
