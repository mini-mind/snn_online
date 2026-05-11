# Minimal Experiments

`src/` 现在按三类职责拆分：

- `envs/`：最小环境。
- `models/`：可复用学习器、读出头和局部学习规则。
- `experiments/`：实验入口脚本，只负责组装配置、运行和打印摘要。

## Directory Layout

| Directory / Script | Purpose | Core Signal | Run |
|---|---|---|---|
| `experiments/etlp_continuous_toy.py` | 连续输入 ETLP-like 分类 | teaching signal | `PYTHONPATH=src python src/experiments/etlp_continuous_toy.py` |
| `experiments/cognitive_map_etlp_toy.py` | 学 gridworld 转移图并规划 | prediction error | `PYTHONPATH=src python src/experiments/cognitive_map_etlp_toy.py` |
| `experiments/point_robot_closed_loop.py` | 点机器人完整控制闭环 | prediction error + TD error | `PYTHONPATH=src python src/experiments/point_robot_closed_loop.py` |
| `experiments/compare_plasticity_rules.py` | 固定预算比较 `three_factor` 与 `tess_like` | reward / success / wall time | `PYTHONPATH=src python src/experiments/compare_plasticity_rules.py` |
| `experiments/compare_mainline_components.py` | 按当前约束拆分主线组件，默认只跑稳定消融链，h4/h5 需显式 opt-in | reward / success / wall time | `PYTHONPATH=src python src/experiments/compare_mainline_components.py` |
| `experiments/compare_mainline_history.py` | 比较主线历史阶段 | reward / success / wall time | `PYTHONPATH=src python src/experiments/compare_mainline_history.py` |
| `experiments/compare_candidate_difficulties.py` | 复测候选机制在 easy / medium / hard 环境难度下的稳定性 | reward / success / wall time | `PYTHONPATH=src python src/experiments/compare_candidate_difficulties.py` |
| `experiments/observe_quiet_dynamics.py` | 训练后进入低输入安静期，观察内部活动是否自然贴近任务活动 | quiet activity / reactivation similarity | `PYTHONPATH=src python src/experiments/observe_quiet_dynamics.py` |
| `experiments/summarize_jsonl_results.py` | 汇总 JSONL benchmark artifact | saved summary rows | `PYTHONPATH=src python src/experiments/summarize_jsonl_results.py runs/*.jsonl` |
| `experiments/compare_lif_vs_izh.py` | 比较 LIF 与 IZ 神经元模型 | reward / success / wall time | `PYTHONPATH=src python src/experiments/compare_lif_vs_izh.py` |
| `experiments/compare_partial_observable_lif_vs_izh.py` | 在部分可观测导航上比较 LIF 与 IZ | reward / success / wall time | `PYTHONPATH=src python src/experiments/compare_partial_observable_lif_vs_izh.py` |

## Dependency Boundary

- `envs/`：纯 Python 最小环境。
- `models/`：纯 Python 学习器和网络执行。
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

实验入口应逐步补齐两个元信息：

- 参数生物对照：关键 decay、阈值、连接度、delay、plasticity rate 要能解释为生物脑里的时间常数、发放阈值、稀疏连接或可塑性速度。
- 环境测试维度：每个 benchmark 要说明它主要测试 observability、horizon、dynamics、goal structure、reward sparsity、action control 或 distribution shift 中的哪一类。

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

- `models/recurrent_spiking.py`：本仓纯 Python 单层 R-SNN；保留 LIF 与 Izhikevich 两种循环脉冲网络接口。
- `envs/point_robot.py`：连续状态、离散动作的 2D point robot，支持 `full` 与 `partial_goal_cue` 两种观测模式，并提供稳定的任务标签（如 `benchmark_id`、`observability_level`、`horizon_level`）。
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

recurrent delay line 入口：

```bash
PYTHONPATH=src python src/experiments/point_robot_closed_loop.py --episodes 160 --eval-every 40 --eval-episodes 40 --plasticity-rule tess_like --recurrent-delay-line
```

第一阶段 benchmark / 对照入口：

```bash
PYTHONPATH=src python src/experiments/compare_plasticity_rules.py
```

关键指标：

- `random_baseline reward/success/length`：同评估预算下随机策略的平均回报、成功率与步长，用于判断训练是否优于随机。
- `model_mse`：最近一个训练窗口里 world model 的一步预测均方误差。
- `eval_reward`：关闭学习后评估 episode 的平均总回报。
- `eval_success`：关闭学习后评估成功率。
- `eval_len`：关闭学习后平均 episode 长度。
- `biological_params`：verbose 输出和 `train_agent(...)` summary 会附带紧凑参数字典，记录关键 decay / threshold / recurrent / delay 开关的生物类比标签和当前 repo 值。
- recurrent plasticity 现在使用 per-neuron modulation：全局 TD error / prediction MSE 仍是低维调制来源，但会按每个神经元的近期活动和活动变化生成 `modulation_j`，再进入 `eligibility_ij * modulation_j` 的局部更新。
- 可用 `--modulation-mode scalar` 切回单一全局调制，用于和 `per_neuron` 做机制对照。

部分可观测版本：

```bash
PYTHONPATH=src python src/experiments/point_robot_closed_loop.py --observation-mode partial_goal_cue --goal-cue-steps 6
```

这里的设计是：episode 前几步给出目标相对方向提示，之后隐藏方向，只保留自身位置、速度、进度和目标距离。这样就把任务从“瞬时反应控制”推向“需要在 recurrent state 里保留短期目标记忆”的设置。

### Plasticity Rule Benchmark

用途：作为第一阶段 benchmark，在固定 seed 范围和训练预算下直接比较 `three_factor` 与 `tess_like`。

运行：

```bash
PYTHONPATH=src python src/experiments/compare_plasticity_rules.py
```

说明：

- 默认任务是 `partial_goal_cue`，因为它更依赖短期目标记忆。
- 可通过 `--observation-mode full` 切回完整观测，不必为 full / partial 另拆第二个脚本。
- 可通过 `--tess-fast-decay`、`--tess-slow-decay`、`--tess-post-decay`、`--tess-eligibility-decay` 直接扫描 `tess_like` 的多时间尺度 trace 参数，同时保持同一 benchmark 入口。
- 可通过 `--output-jsonl /path/to/results.jsonl` 记录每个 seed/rule 的单次结果和最终 summary，便于本地 benchmark 留档或后续汇总；这些 JSONL summary 可以作为本地 artifact，`runs/` 目录保持忽略。
- 脚本会对两个规则分别在多 seed 上调用 `train_agent(...)`，打印每个 seed 的 `eval_reward`、`eval_success` 和 `elapsed_sec`。
- JSONL 每行一个 JSON object：运行行包含 `plasticity_rule`、`seed`、`eval_reward`、`eval_success`、`elapsed_sec` 及本次关键配置；汇总行包含 `three_factor`、`tess_like`、`delta` 和共享 `config`。
- 汇总输出包括：
  - `mean_eval_reward`
  - `mean_eval_success`
  - `mean_elapsed_sec`
  - `reward_gain_tess_like_minus_three_factor`
  - `success_gain_tess_like_minus_three_factor`
  - `speed_ratio_tess_like_vs_three_factor`

当前小基准结论：

- `partial_goal_cue`，5 seeds，80 episodes：`tess_like` 相对 `three_factor` 的 reward gain 约 `+0.359`，success gain 约 `+0.030`，wall time 约慢 `3%`。
- `full`，5 seeds，80 episodes：`tess_like` 相对 `three_factor` 的 reward gain 约 `-0.671`，success gain 约 `-0.190`，wall time 约慢 `6.5%`。
- 这说明 `tess_like` 更像是短期记忆任务的候选机制，而不是全观测控制下的直接替代。

### Mainline History Benchmark

用途：用项目自身发展阶段做对照，避免继续依赖已废弃的外挂式 delay feature。

运行：

```bash
PYTHONPATH=src python src/experiments/compare_mainline_history.py
```

阶段：

- `h1_three_factor_recurrent`：最小三因子 recurrent baseline。
- `h2_tess_recurrent`：加入多时间尺度 `tess_like` 局部 trace。
- `h3_tess_recurrent_delay`：加入真正 recurrent delay line。
- `h4_eprop_like_v0`：加入 per-neuron modulation；目前作为 e-prop-like 对照/诊断分支，不作为默认主线。
- `h5_metaplasticity_v0`：在 e-prop-like 分支上加入突触级慢变量；当前结果为负，只保留用于诊断。

下一步的对照目标，是复现一条尽量贴近当前约束的成熟局部学习方法，把它当作稳定参照，而不是继续把主线往更复杂的方向推。

说明：

- 默认任务是 hard `partial_goal_cue`：cue 3 steps，horizon 80。
- JSONL summary 会记录每个阶段的 reward / success / wall time。
- 外挂式 delay feature 已移除。它曾作为 readout shortcut 有过历史正结果，但不符合当前“内部 recurrent 动力学优先”的约束，也和 per-neuron modulation 有负交互。

当前 5 seed / 80 episodes 结果：

```text
h1_three_factor_recurrent reward=0.433 success=0.200
h2_tess_recurrent reward=-0.011 success=0.170
h3_tess_recurrent_delay reward=0.142 success=0.210
h4_eprop_like_v0 reward=0.289 success=0.210
h5_metaplasticity_v0 reward=-0.265 success=0.190
```

解释：移除外挂式 delay feature 后，`h4_eprop_like_v0` 在 hard preset 上还没有压过最朴素的 `three_factor` reward，只是在 success 上接近 recurrent delay 阶段。`h5_metaplasticity_v0` 的默认参数是负结果，说明简单地给高 eligibility 连接加保护会抑制学习。当前策略不是继续堆改进，而是回到稳定消融：先确认 `three_factor`、`tess_like`、`recurrent_delay_line` 每一步是否可复现、是否跨 seed 稳定。

## Next

候选主线跨难度复测入口：

```bash
PYTHONPATH=src python src/experiments/compare_candidate_difficulties.py
```

当前 preset 只改变现有 point robot 参数，不引入新环境动力学：

- `easy`：`partial_goal_cue`，cue 10 steps，horizon 40。
- `medium`：`partial_goal_cue`，cue 6 steps，horizon 60。
- `hard`：`partial_goal_cue`，cue 3 steps，horizon 80。

当前 5 seed / 80 episodes 结果：

历史候选复测中的 `scalar_delay_rline` 依赖已移除的外挂式 delay feature，不再作为当前结论使用。当前不再把 `h4_eprop_like_v0` 当作 hard 主线；它与 `h5_metaplasticity_v0` 都降级为诊断分支。

### Quiet Internal Dynamics

用途：训练 hard preset 主线 agent 后，进入低输入安静期，只观察 recurrent state 是否自然贴近任务期活动。这个脚本不做显式 replay buffer、不生成 imagined episode、不在 quiet phase 更新权重。

运行：

```bash
PYTHONPATH=src python src/experiments/observe_quiet_dynamics.py
```

输出指标：

- `quiet_mean_activity`：安静期内部活动强度，过低表示基本沉默。
- `quiet_consecutive_similarity`：相邻 quiet step 的相似度，越高表示内部状态越稳定。
- `quiet_max_reference_similarity`：quiet step 与训练后真实任务活动片段的最大相似度均值。
- `quiet_reactivation_fraction`：quiet step 中相似度超过阈值的比例。它只能说明“像不像任务活动”，不能单独证明有 replay / dreaming。
- `untrained_*`：同一 reference set 下的未训练网络 quiet 基线。只有训练后指标明显高于这个基线时，才值得进一步讨论 replay-like reactivation。
- `quiet_*_uplift`：训练后 quiet 指标减去未训练基线。若 uplift 接近或低于 0，应解释为“暂未观察到超出基线的自然重激活”。

当前 hard e-prop-like 诊断分支单 seed 观察结果：

```text
final_eval_reward=-1.834 final_eval_success=0.050
quiet_max_reference_similarity=0.606
untrained_quiet_max_reference_similarity=0.258
quiet_reference_similarity_uplift=0.348
quiet_reactivation_fraction=0.000
```

解释：训练后的 quiet 状态比未训练网络更接近任务期活动，但没有达到 `0.75` 阈值的重激活事件。因此当前只能说“安静期内部状态带有任务活动痕迹”，不能说已经出现 replay / dreaming。

### JSONL Summary

用途：把 `--output-jsonl` 生成的本地 artifact 汇总成短行，便于比较多次实验。

运行：

```bash
PYTHONPATH=src python src/experiments/summarize_jsonl_results.py runs/*.jsonl
```

说明：

- 只读取 JSONL 中 `type == "summary"` 的行，忽略单 seed 运行行和未知字段。
- 输出包括关键配置、各条件均值、reward / success delta 和 speed ratio。
- `runs/` 是忽略目录；这些文件是本地实验记录，不作为源码提交。

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
- 历史跑数会随默认参数和随机种子变化而失效，因此这里不把旧 benchmark 数字当成“当前结论”保留。
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

2. recurrent delay line：
   当前已接入 `--recurrent-delay-line`，用于验证连接本身的传输延迟是否改善短期记忆。

3. e-prop-like v0：
   当前已接入 per-neuron modulation，保持 `eligibility_ij * modulation_j` 的局部更新形式。

4. quiet internal dynamics：
   只观察低输入安静期内部状态，不设计显式 replay 训练。
