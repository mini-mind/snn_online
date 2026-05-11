# Minimal Experiments

`src/` 现在按三类职责拆分：

- `envs/`：最小环境。
- `models/`：可复用学习器与 `dynn` 适配层。
- `experiments/`：实验入口脚本，只负责组装配置、运行和打印摘要。

## Directory Layout

| Directory / Script | Purpose | Core Signal | Run |
|---|---|---|---|
| `experiments/etlp_continuous_toy.py` | 连续输入 ETLP-like 分类 | teaching signal | `PYTHONPATH=src python src/experiments/etlp_continuous_toy.py` |
| `experiments/cognitive_map_etlp_toy.py` | 学 gridworld 转移图并规划 | prediction error | `PYTHONPATH=src python src/experiments/cognitive_map_etlp_toy.py` |
| `experiments/point_robot_closed_loop.py` | 点机器人完整控制闭环 | prediction error + TD error | `PYTHONPATH=src python src/experiments/point_robot_closed_loop.py` |
| `experiments/compare_lif_vs_izh.py` | 比较 LIF 与 IZ 神经元模型 | reward / success / wall time | `PYTHONPATH=src python src/experiments/compare_lif_vs_izh.py` |
| `experiments/compare_partial_observable_lif_vs_izh.py` | 在部分可观测导航上比较 LIF 与 IZ | reward / success / wall time | `PYTHONPATH=src python src/experiments/compare_partial_observable_lif_vs_izh.py` |

## Dependency Boundary

- `envs/`：纯 Python 最小环境。
- `models/`：学习器和网络执行都尽量统一到 `dynn`。
- `experiments/`：入口脚本在本仓，但运行前需要让 Python 能找到 `src/`，例如使用 `PYTHONPATH=src`。

## Shared Pattern

三个原型都围绕同一个研究假设：

```text
local trace / recurrent state
×
low-dimensional modulation signal
=>
online weight update
```

## Functional Brain Layout

当前 `src/` 仍保持最小实验结构，但后续闭环任务可以按功能性脑区拆分模型职责：

| Brain Region | Engineering Role | Current / Near-term Mapping |
|---|---|---|
| 端脑 / 皮层 | 单层 recurrent SNN，负责特征、状态、记忆、预测和关联推理 | `models/recurrent_spiking.py` |
| 丘脑 | 路由外部特征和皮层区信息，计算价值调制并控制可塑性强度 | 后续 thalamic modulation adapter |
| 小脑 | 记忆并回放动作细节，把动作意图展开为执行参数 | 后续 motor program model |
| 中脑 / 后脑 | 维护饥饿、疼痛、疲劳、能量等内稳态变量 | 后续 physiology environment state |

这个划分不是为了在代码里引入复杂框架，而是为了让实验问题更清楚：

```text
env observation / physiology
  -> thalamic routing + modulation
  -> cortical recurrent state + local learning
  -> action intent
  -> cerebellar motor program
  -> env action
```

在当前阶段，最直接的落地任务是“部分可观测点机器人 + 内稳态变量”。它比静态模式识别更适合检验单层 recurrent SNN、Izhikevich 动力学和局部调制是否真正有用。

## Experiments

## Toy Learning

### ETLP Continuous Toy

核心更新：

$$
\Delta w_{ij} = \eta \cdot \bar{x}_i(t) \cdot f(V_j(t)) \cdot T_j(t)
$$

用途：观察连续输入如何形成 analog eligibility trace，以及教学信号如何调制局部 trace。

快速检查：

```bash
PYTHONPATH=src python src/experiments/etlp_continuous_toy.py --train-steps 600 --eval-every 100 --eval-samples 200
```

可打印指标：

- `online_acc`：从训练开始累计的在线分类准确率。
- `window_acc`：最近一个 `eval_every` 窗口内的在线准确率。
- `eval_accuracy`：在当前步对应数据分布上的独立评估准确率。
- `weight_norm`：权重范数，用于观察是否发散。

### Cognitive Map + ETLP-like Toy

核心更新：

$$
\Delta W_a[o, i] = \eta \cdot \delta^{pred}_o(t) \cdot \bar{x}_i(t)
$$

用途：用 one-step prediction 学动作转移结构，再把 learned transition graph 用于 shortest-path planning。

快速检查：

```bash
PYTHONPATH=src python src/experiments/cognitive_map_etlp_toy.py --train-steps 1000 --eval-every 250
```

导出到 `NeuralSoup` 标准 run 目录：

```bash
PYTHONPATH=src python src/experiments/cognitive_map_etlp_toy.py --train-steps 1000 --eval-every 250 --export-run-dir ../neuralsoup/public/runs
```

导出结构说明：

- `topology.json` 包含通用 `subgraph_tree` 元数据，用嵌套组表达环境、接口、模型和预测器。
- `manifest.json` 会同时列出回合摘要、轨迹、地图、环境说明、子图结构说明和每条连接的权重摘要 artifact。
- 各 node / port / edge / artifact 会携带 `variable_names`、`variables`、`label`、`label_zh` 和必要 `metadata`，便于 NeuralSoup 直接显示中文标签和变量名。

导出内容除了 `summary/events/topology/manifest` 外，还会附带：

- 嵌套子图结构 `topology/subgraph-tree.json`；
- 环境结构 artifact，例如 `maps/grid-world-map.json`、`environment/grid-world-environment.json`；
- 每个 `node_set` / `edge_set` / `port` 的变量名、英文 label、中文 `label_zh` 与分层 metadata；
- 每条显式连接的边文件 `topology/edges/*.json`，包含源变量名与目标变量名。

关键指标：

- `prediction_mse`：最近一个训练窗口内的一步预测均方误差。
- `transition_acc`：learned graph 与环境真实单步转移的一致率。
- `planning_success`：基于 learned graph 做 BFS 规划后，真实执行仍能到达目标的比例。
- `path_efficiency`：成功规划样本上，真实最短路径长度与 learned path 长度的比值，越接近 `1.0` 越好。

## Closed Loop

### Point Robot Closed Loop

组件：

- `models/recurrent_spiking.py`：面向当前实验的 `dynn` 薄适配层；保留 LIF 与 Izhikevich 两种循环脉冲网络接口。
- `envs/point_robot.py`：连续状态、离散动作的 2D point robot，支持 `full` 与 `partial_goal_cue` 两种观测模式。
- `models/point_robot_closed_loop.py`：world model + TD action value 控制闭环。

用途：验证 R-SNN recurrent state 能否进入真正的 `observe -> act -> learn` 控制回路。

快速检查：

```bash
PYTHONPATH=src python src/experiments/point_robot_closed_loop.py --episodes 160 --eval-every 40 --eval-episodes 40
```

第一阶段先进复现入口：

```bash
PYTHONPATH=src python src/experiments/point_robot_closed_loop.py --episodes 160 --eval-every 40 --eval-episodes 40 --plasticity-rule tess_like
```

关键指标：

- `random_baseline reward/success/length`：同评估预算下随机策略的平均回报、成功率与步长，用于判断训练是否优于随机。
- `model_mse`：最近一个训练窗口里 world model 的一步预测均方误差。
- `eval_reward`：关闭学习后评估 episode 的平均总回报。
- `eval_success`：关闭学习后评估成功率。
- `eval_len`：关闭学习后平均 episode 长度。

部分可观测版本：

```bash
PYTHONPATH=src python src/experiments/point_robot_closed_loop.py --observation-mode partial_goal_cue --goal-cue-steps 6
```

导出结构说明：

- `topology.json` 包含通用 `subgraph_tree` 元数据，用嵌套组表达环境、观测接口、脉冲状态模型和预测 / 控制组。
- `manifest.json` 会列出回合摘要、轨迹、任务环境说明、子图结构说明和每条连接的权重摘要 artifact。
- 观测变量、隐藏群变量、世界模型变量和动作价值变量都会在 `topology/manifest/artifacts` 中保留变量名与中英文标签。

这里的设计是：episode 前几步给出目标相对方向提示，之后隐藏方向，只保留自身位置、速度、进度和目标距离。这样就把任务从“瞬时反应控制”推向“需要在 recurrent state 里保留短期目标记忆”的设置。

### LIF vs IZ 对比

用途：检验 Izhikevich 动力学替代 LIF 后，最终控制效果是否改善，以及 wall-clock 速度是否下降。

运行：

```bash
PYTHONPATH=src python src/experiments/compare_lif_vs_izh.py
```

输出指标：

- `mean_eval_reward`
- `mean_eval_success`
- `mean_elapsed_sec`
- `speed_ratio_izh_vs_lif`

说明：

- 该脚本会对 `lif` 和 `izh` 分别在多 seed 上调用 `train_agent(...)`，再汇总均值。
- 历史跑数会随 `dynn`、默认参数和随机种子变化而失效，因此这里不把旧 benchmark 数字当成“当前结论”保留。
- 若要记录新的结论，请同时保存完整命令、seed 范围和输出摘要。

### Partial Observable LIF vs IZ

用途：把任务切到更依赖记忆的 `partial_goal_cue`，观察 `izh` 是否更容易保留早期目标线索。

运行：

```bash
PYTHONPATH=src python src/experiments/compare_partial_observable_lif_vs_izh.py
```

说明：

- 该脚本只是把对比任务固定到 `partial_goal_cue`，输出字段与 `compare_lif_vs_izh.py` 一致。
- 是否出现 “`izh` 在记忆任务里更稳” 这一现象，需要用当前代码和当前依赖重新跑命令核对，不能把旧结果直接视为长期成立。

## 先进复现顺序

1. `tess_like` 多时间尺度 trace：
   当前已接入 `models/recurrent_spiking.py`，作为 `three_factor` baseline 的第一阶段对照。

2. learnable delay：
   下一步在 `partial_goal_cue` 任务上加入权重外的时延可塑性。

3. dreaming / replay：
   在局部规则稳定后，再把 imagined experience 加回闭环系统。
