# Unified Platform Design

本文档是 `snn_online` 侧记录的跨仓协作草案，作用是对齐角色边界、协议方向和验收口径。
它不是 `dynn` 或 `neuralsoup` 的权威内部设计文档；凡属引擎内部调度、数据结构或渲染实现的细节，都应回到各自仓库维护。

本文档把三个相关仓库重新定义为彼此独立、通过协议协作的实验平台组件：

- `neuralsoup`：交互式 `Studio`，负责拓扑编辑、实验配置、运行监控、结果回放与分析。
- `dynn`：事件驱动、面向通用稀疏图的执行引擎，负责网络执行、局部可塑性更新、轻量事件输出与产物落盘。
- `snn_online`：研究与实验项目，负责定义任务、组合学习规则、发起对照实验，并接入 `dynn` 与 `neuralsoup`。

三者目标是协作，但不应直接互相依赖实现细节。共享层只保留协议与产物格式。

## 1. 重定义后的角色

### 1.1 snn_online

`snn_online` 不再承担“平台本体”的职责，而是聚焦为：

- 研究问题的载体；
- 实验模板与基线算法集合；
- 任务定义、对照实验、分析与报告入口；
- `dynn` 的主要早期调用方；
- `neuralsoup` 的主要早期展示数据提供方。

它应该继续保持：

- 问题定义清楚；
- 原型简单；
- 可解释；
- 对局部学习规则敏感；
- 实验脚本可直接运行。

### 1.2 neuralsoup

`neuralsoup` 重新定义为独立的 `Studio`：

- 拓扑编辑器；
- 实验配置与模板选择界面；
- 运行状态查看器；
- 结果回放器；
- 神经活动、连接结构、环境轨迹的可视化分析工具。

它不负责：

- 作为权威训练引擎；
- 决定实验协议；
- 内嵌依赖某个后端内部实现。

`neuralsoup` 只依赖公开协议，例如：

- `ExperimentSpec`
- `TopologyIR`
- `RunEventSchema`
- `ArtifactSchema`

### 1.3 dynn

`dynn` 在这里被视为独立执行引擎。

目标能力应是：

- 结构灵活；
- 支持稀疏连接；
- 支持事件驱动传播；
- 支持局部学习与 trace；
- 支持多种神经元动力学；
- 支持实验事件流与标准化产物输出。

因此 `dynn` 的长期定位是：

- `runtime / engine`
- 不再是若干单实验脚本的集合

## 2. 总体协作方式

三者之间的关系应是：

```text
snn_online --(ExperimentSpec)--> dynn
dynn --(RunEventSchema + ArtifactSchema)--> neuralsoup
snn_online --(runs / reports / topology)--> neuralsoup
```

更完整地说：

1. `snn_online` 生成实验定义；
2. `dynn` 读取实验定义并执行；
3. `dynn` 产出标准运行事件与实验产物；
4. `neuralsoup` 读取这些标准数据做可视化与交互分析。

这里的约束非常重要：

- `neuralsoup` 不调用 `dynn` 内部对象；
- `dynn` 不关心 `neuralsoup` 前端如何渲染；
- `snn_online` 不依赖任一前端或引擎的私有数据结构。

## 3. 共享协议层

三者之间只共享四类协议：

1. `ExperimentSpec`
2. `TopologyIR`
3. `RunEventSchema`
4. `ArtifactSchema`

### 3.1 ExperimentSpec

负责定义一次实验“要做什么”，不描述具体 UI，也不描述底层内存布局。

建议最小字段：

- `experiment`
  - `id`
  - `name`
  - `description`
  - `seed`
  - `tags`
- `task`
  - `task_id`
  - `env_name`
  - `observation_mode`
  - `reward_config`
  - `episode_horizon`
- `model`
  - `family`
  - `topology_ref`
  - `neuron_type`
  - `state_layout`
- `plasticity`
  - `rule_family`
  - `trace_config`
  - `modulation_sources`
  - `plastic_targets`
- `training`
  - `episodes`
  - `eval_schedule`
  - `exploration`
  - `checkpoint_policy`
- `runtime`
  - `engine`
  - `device`
  - `step_mode`
  - `profiling`

### 3.2 TopologyIR

负责统一描述网络结构，供 `snn_online`、`dynn` 与 `neuralsoup` 共享。它应是通用图 IR，而不是为多层二维网络或局部连接特化的格式。

建议至少表达：

- `node_sets`：节点集合，可表示神经元群、输入编码器、读出头、调制节点或外部状态；
- `edge_sets`：边集合，可显式给出，也可由规则生成；
- `ports`：输入、输出、调制和探针端口；
- `annotations`：布局、分组、实验标签和可视化注解；
- `parameters`：节点和边的参数或随机初始化分布；
- `plasticity`：边集合绑定的可塑性标签、规则和更新权限；
- `representation`：边的表示方式，例如显式边表、生成规则、稀疏矩阵或外部引用。

多层二维网格、局部邻域连接、双向连接和层级结构应通过 `coordinate_space`、`edge_sets.representation.rule` 与 `annotations` 表达。它们是重要实验模板，但不能成为协议的中心假设。

### 3.3 RunEventSchema

负责运行中的流式观测与调试，不要求保存全部内部状态，并且应主动避免大字段导致文件尺寸暴增。事件本体只保存轻量摘要，大型数组和长序列通过 artifact 引用。

必须支持：

- 训练起止；
- episode 起止；
- step 级 reward；
- spike 活动摘要；
- 神经元状态摘要；
- 权重更新摘要；
- 调制信号摘要；
- 环境状态摘要；
- checkpoint 事件；
- 错误与中断事件。

建议把它设计为 append-only 的轻量事件流，便于：

- 实时监控；
- 回放索引；
- 故障恢复；
- 后处理分析。

不应在单条事件里直接写入完整 spike train、完整权重矩阵、完整环境轨迹或大体积 tensor。这类数据统一落到 `traces/`、`episodes/`、`checkpoints/` 或 `artifacts/`，事件中只保留 `artifact_refs`。

### 3.4 ArtifactSchema

负责统一落盘结构，最小建议：

- `spec.yaml`
- `summary.json`
- `metrics.jsonl`
- `topology.json`
- `episodes/`
- `checkpoints/`
- `traces/`
- `reports/`

这样 `neuralsoup` 可以只读产物目录做展示，不必依赖引擎在线连接。

## 4. snn_online 的目标架构

`snn_online` 后续应作为实验编排层，重点不是再造引擎，而是：

- 定义任务；
- 组合算法；
- 发起对照；
- 记录问题、结果和解释。

建议后续把代码逐步整理为：

- `tasks/`
- `experiments/`
- `specs/`
- `reports/`
- `analysis/`

其中：

- `tasks/` 定义研究任务；
- `experiments/` 负责具体实验流程；
- `specs/` 放实验 spec 模板；
- `reports/` 放实验结果与结论。

## 5. 近期优先级

为了给后续详细设计、开发和验收做准备，建议按以下顺序推进：

### Phase A: 先统一文档与边界

产出：

- 三仓 README 全部改成新角色定义；
- 每仓新增架构说明文档；
- 统一术语表。

### Phase B: 先定义协议

产出：

- `ExperimentSpec` 草案；
- `TopologyIR` 草案；
- `RunEventSchema` 草案；
- `ArtifactSchema` 草案。

### Phase C: 再做引擎 MVP

目标：

- `dynn` 支持通用 `TopologyIR` 编译，至少包含显式边表、规则生成边和稀疏矩阵引用；
- 支持事件驱动或事件稀疏传播；
- 支持 ETLP 类 trace；
- 支持基础调制学习；
- 支持标准产物输出。

### Phase D: 再做 Studio 接入

目标：

- `neuralsoup` 能导入导出 `TopologyIR`；
- 能监控 run；
- 能加载产物回放。

### Phase E: 再迁移 snn_online 实验

目标：

- 当前 `point_robot`、`partial_goal_cue`、后续 ETLP/CML 实验都走统一 spec；
- `snn_online` 不再直接耦合私有执行实现。

这里仍然只记录 `snn_online` 对外部协作方的接口期待，不展开它们的内部实现分层。

## 6. 验收口径

为了避免后续开发发散，三仓应提前使用统一验收口径。

### 6.1 dynn 验收重点

- 能表达通用节点集合、边集合、端口和注解；
- 能表达显式边表、规则生成边、稀疏矩阵引用和双向边；
- 能支持局部 trace 与 modulation；
- 能以标准事件流输出运行状态；
- 能稳定执行 `snn_online` 给出的至少一个闭环任务。

### 6.2 neuralsoup 验收重点

- 能导入导出统一拓扑格式；
- 能配置并展示一次实验；
- 能读取 `dynn` 的事件与产物；
- 能完成至少一次环境回放与 spike/weight 可视化。

### 6.3 snn_online 验收重点

- 能用统一 spec 发起实验；
- 能把任务和学习规则解释清楚；
- 能做结构或模型对照；
- 能把结果交给 `neuralsoup` 回放展示。

## 7. 当前结论

新的总体关系应明确为：

- `neuralsoup` 是独立 `Studio`
- `dynn` 是独立执行引擎
- `snn_online` 是独立实验项目

三者通过协议协作，而不是通过实现互相嵌套。

这个边界一旦固定，后续详细技术设计、开发拆分和阶段验收才会稳定。
