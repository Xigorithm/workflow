# vllm-ascend Workflow Skills

[OpenCode](https://opencode.ai) Agent Skills 集合，为 [vllm-ascend](https://github.com/vllm-project/vllm-ascend)（vLLM Ascend NPU 插件）项目提供开发工作流支持。

## Skills 概览

| Skill | 状态 | 说明 |
|-------|------|------|
| [vllm-ascend-feature-learning](#vllm-ascend-feature-learning) | 可用 | 生成 vllm-ascend 特性的中文技术学习文档 |
| vllm-ascend-feature-develop | 开发中 | 辅助 vllm-ascend 特性开发 |
| vllm-ascend-pr-review | 开发中 | 辅助 vllm-ascend PR 代码审查 |

## vllm-ascend-feature-learning

为 vllm-ascend 仓库的特性/模块生成全面的中文代码走读学习文档。

### 触发方式

在 OpenCode TUI 中直接请求学习 vllm-ascend 的某个特性即可自动触发，例如：

```
我想了解 vllm-ascend 的 ACL Graph
帮我生成 vllm-ascend patch 系统的学习文档
学习 vllm-ascend 的 MLA attention 后端
```

### 输出格式

生成的学习文档包含：

- **Mermaid 架构图/流程图/时序图/类图** — 可视化展示系统架构和执行流程
- **GPU vs NPU 对比表** — 对比上游 vLLM 与 vllm-ascend 的实现差异
- **源码走读** — 引用 vllm-ascend 真实代码并标注文件路径
- **配置参考** — 列出所有相关环境变量和配置参数
- **插件集成分析** — 分析 patch/继承/entry point 等集成机制

文档输出到 skill 目录下的 `outputs/` 子目录。

### 覆盖特性

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

完整特性列表见 [feature-catalog.md](skills/vllm-ascend-feature-learning/references/feature-catalog.md)。

## 目录结构

```
workflow/
├── skills/
│   ├── vllm-ascend-feature-learning/   # 特性学习文档生成
│   │   ├── SKILL.md                    # Skill 定义与工作流
│   │   └── references/
│   │       ├── feature-catalog.md      # 特性目录与文件路径索引
│   │       └── style-guide.md          # 文档结构与格式规范
│   ├── vllm-ascend-feature-develop/    # 特性开发辅助（开发中）
│   └── vllm-ascend-pr-review/          # PR 审查辅助（开发中）
└── README.md
```

## 使用前提

- 安装 [OpenCode](https://opencode.ai/docs/)
- 配置 LLM Provider API Key
- 本地克隆 [vllm-ascend](https://github.com/vllm-project/vllm-ascend) 仓库（用于源码走读）

## 安装

将本仓库的 skills 目录配置到 OpenCode 的 skills 路径中。在 `opencode.json` 中添加：

```json
{
  "skills": {
    "paths": ["./skills"]
  }
}
```

## 许可证

本项目遵循与 [vllm-ascend](https://github.com/vllm-project/vllm-ascend) 相同的许可证。
