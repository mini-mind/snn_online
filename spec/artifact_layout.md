# Artifact Layout Draft

本文档定义一次实验 run 的推荐落盘布局。目标是让 `dynn` 输出稳定产物，让 `neuralsoup` 和 `snn_online` 只读目录就能完成复现、分析和展示。

## 1. 顶层目录

一次 run 建议使用如下目录结构：

```text
runs/<run_id>/
├── spec.yaml
├── summary.json
├── metrics.jsonl
├── topology.json
├── stdout.log
├── stderr.log
├── episodes/
├── checkpoints/
├── traces/
├── artifacts/
└── reports/
```

## 2. 顶层文件

### `spec.yaml`

- 本次运行的最终实验定义快照
- 应与提交给 `dynn` 的 `ExperimentSpec` 等价
- 允许在运行前由 `snn_online` 生成，也允许由 `dynn` 写出归一化版本

### `summary.json`

- 单次 run 的汇总结果
- 建议至少包含：
  - `run_id`
  - `task_id`
  - `model_family`
  - `engine`
  - `seed`
  - `status`
  - `mean_eval_reward`
  - `mean_eval_success`
  - `mean_eval_length`
  - `prediction_mse`
  - `wall_clock_sec`

### `metrics.jsonl`

- 逐事件或逐阶段扁平指标
- 每行一条 JSON
- 可视为 `RunEventSchema` 的裁剪版或投影视图

### `topology.json`

- 本次运行使用的 `TopologyIR`
- 即使实验最初通过模板生成，也建议写出解析后的最终拓扑

### `stdout.log` / `stderr.log`

- 原始控制台日志
- 便于调试失败 run

## 3. 子目录

### `episodes/`

建议按 episode 存放轻量结构化结果：

```text
episodes/
├── episode_000000.json
├── episode_000001.json
└── ...
```

单个 episode 文件建议包含：

- `episode_index`
- `phase`
- `reward_sum`
- `success`
- `length`
- `start_state`
- `end_state`
- `trajectory_ref`

### `checkpoints/`

用于中断恢复和阶段性分析。

建议布局：

```text
checkpoints/
├── latest.json
├── ckpt_000040/
├── ckpt_000080/
└── ...
```

每个 checkpoint 目录可包含：

- 神经元状态
- 权重状态
- trace 状态
- learner 状态
- RNG 状态

### `traces/`

用于存放较大的时间序列或稀疏事件数据。

例如：

- spike trains
- 权重变化时间线
- modulation 曲线
- 环境轨迹

建议避免把大体积数组直接塞进 `metrics.jsonl`。

### `artifacts/`

用于存放其他补充产物：

- png / svg 图
- npz / parquet / csv 数据
- 导出的回放包
- 前端消费用预处理缓存

### `reports/`

用于存放：

- 自动生成的 `report.md`
- 对照分析结果
- 失败原因摘要

## 4. 命名与版本建议

- `run_id` 应全局唯一
- `spec_version` 与 `schema_version` 应显式写出
- 顶层目录结构应尽量稳定，避免频繁重命名
- 对大型二进制数据，建议在文件名或元数据中附带格式版本

## 5. 第一阶段最小要求

第一阶段不必一次产出所有内容，但至少应有：

- `spec.yaml`
- `summary.json`
- `metrics.jsonl`
- `topology.json`
- `episodes/`

只有这几项稳定下来，前端回放和跨仓对齐才会真正开始变简单。
