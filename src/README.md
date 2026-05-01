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
| `experiments/compare_depth_ablation.py` | 固定预算下比较浅层、深层与等总宽结构 | reward / success / wall time | `PYTHONPATH=src python src/experiments/compare_depth_ablation.py` |

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

若希望显式指定二维网格形状，可以额外传：

```bash
PYTHONPATH=src python src/experiments/point_robot_closed_loop.py --n-layers 3 --n-neurons 64 --grid-width 8 --grid-height 8
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

严格深度消融：

- 命令：`PYTHONPATH=src python src/experiments/compare_depth_ablation.py --episodes 40 --eval-every 20 --eval-episodes 10 --seeds 2 --base-width 64 --deep-layers 3`
- 该实验固定训练预算和随机种子，只比较结构：
  - `1x64`：单层、64 神经元
  - `3x64`：三层、每层 64 神经元
  - `1x192`：单层、192 神经元

输出解读：

- 该脚本会分别打印每个结构在 `lif` / `izh` 下的 `mean_eval_reward`、`mean_eval_success` 和 `mean_elapsed_sec`。
- 它适合回答“深度本身是否带来收益”，但结论仍依赖训练预算、观测模式和 seed 范围。
