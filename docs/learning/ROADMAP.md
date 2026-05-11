# Experiment Roadmap

本路线面向当前纯 Python 实验仓。目标不是直接复刻完整生物脑，也不是追通用 SOTA，而是在可解释小任务中逐步验证在线局部学习机制。

## Guiding Constraints

1. **局部在线优先**：权重更新优先来自 pre/post 活动、局部状态和低维调制信号。
2. **生物参数有参照**：新增关键参数时，应记录它和生物脑量级的关系，哪怕只是粗略映射。
3. **环境分维度和难度**：每个 benchmark 要说明它测试什么能力，以及难度来自哪里。
4. **replay / dreaming 不预设**：先加入会产生离线重激活压力的状态和条件，再观察是否出现 replay-like 行为。
5. **少参数、强对照**：默认只开放少量开关，用固定预算和固定 seed 做对照。

## Biological Parameter Anchors

本仓参数是工程化、离散时间的近似值。它们不需要一开始等于真实生物数值，但需要能解释“对应生物脑里的什么量”。

| Repo Parameter | Biological Analogy | What To Track |
|---|---|---|
| `membrane_decay` | 膜时间常数 / 泄漏速度 | 对应状态保留多久，是否太短或太长 |
| `trace_decay` | 突触前 / eligibility trace 衰减 | 信用分配窗口，能否覆盖 cue 到 reward 的间隔 |
| `threshold` | spike 发放阈值 | 平均 spike rate 是否过稀或过密 |
| `recurrent_degree` | 局部稀疏连接度 | 是否形成足够 recurrent memory，是否过度耦合 |
| `recurrent_scale` | recurrent synaptic strength | 是否造成爆发、沉默或稳定活动 |
| `plastic_lr` | 突触可塑性速度 | 学习是否过快遗忘或过慢无效 |
| `weight_decay` | 稳态 / 突触归一化压力 | 是否限制权重发散 |
| `metaplasticity` | 突触可塑性状态 | 是否让高活动连接更稳定、低活动连接恢复可塑性 |
| `meta_decay` | metaplastic 状态衰减 | 稳定性记忆持续多久 |
| `meta_lr` | metaplastic 状态更新速度 | 连接被保护或释放的速度 |
| `meta_strength` | metaplastic gate 强度 | 更新被抑制的程度 |
| `tess_*_decay` | 多时间尺度 trace | fast / slow trace 是否覆盖不同事件间隔 |
| `recurrent_delay_line` | 突触传输延迟 | 延迟分布是否帮助时序记忆 |

后续每加入一个新参数，应补充：

```text
repo name
biological analogy
expected range in repo units
why it matters for current task
failure mode if too low / too high
```

## Environment Test Matrix

环境不只按名字区分，还要按测试维度和难度等级区分。

| Dimension | Easy | Medium | Hard | Measures |
|---|---|---|---|---|
| Observability | `full` | `partial_goal_cue` | sparse / delayed cue | 短期记忆、cue retention |
| Horizon | short `max_steps` | current 60 steps | longer / delayed reward | 长程信用分配 |
| Dynamics | deterministic | noise in movement | drift / perturbation | world model 和鲁棒性 |
| Goal Structure | single fixed goal | random goal | switching / multi-goal | context binding |
| Reward | dense distance shaping | sparse success bonus | delayed sparse reward | TD / modulation 稳定性 |
| Action | discrete low-level | larger action set | continuous-like discretization | 动作选择和控制 |
| Distribution | train/eval same | shifted start/goal | task switch | 泛化和抗遗忘 |

当前主线是：

```text
point_robot + partial_goal_cue + fixed 5 seeds
```

它主要测试：

```text
短期目标记忆
recurrent state 是否保留 cue
delay / trace 是否帮助闭环控制
```

## Near-Term Plan

1. **主线历史 benchmark**
   - 对照：`three_factor`、`tess_like`、`tess_like + recurrent_delay_line`。
   - 任务：优先 `partial_goal_cue` hard preset。
   - 预算：先保持 5 seeds / 80 episodes。
   - 已移除外挂式 delay feature；它不再作为当前实验入口或核心对照。
   - `eprop_like_v0` 和 `metaplasticity_v0` 现在只作为诊断分支，不再作为默认推进目标。
   - 先进对照应优先选择与当前约束接近、且已形成成熟定义的局部学习路线，而不是继续堆叠未验证的改进。

2. **生物参数表落到代码输出**
   - 在实验 summary 中记录关键时间常数、连接度、delay 开关和 trace decay。
   - 不要求真实单位精确，但要方便回看“这个实验像什么生物量级”。

3. **环境难度标签**
   - 给 point robot 配置增加难度命名或 summary 字段。
   - 初始只做文档和输出层标签，不急着复杂化环境。

4. **先进对照独立化**
   - 把 reward-based e-prop 作为成熟对照优先复现。
   - 再按需要单独引入 TESS、EchoSpike、BrainTrace、DECOLLE、S-TLLR 这类不同路径。
   - 不把这些路线揉成一个主线，不预设 replay / dreaming。

5. **quiet internal dynamics**
   - 不设计显式 replay loop。
   - 只加入低输入 / 低行动压力下的内部滚动条件。
   - 观察是否出现 replay-like reactivation。
   - 当前单 seed 观察：训练后 quiet 状态相对未训练基线更像任务活动，但尚未达到重激活阈值；暂不声称出现 replay / dreaming。

## What Counts As Progress

- 新机制在至少一个明确测试维度上优于 baseline。
- 结果能被 JSONL summary 复现和解释。
- 参数能说明对应的生物启发含义。
- 环境难度变化能解释机制的优势和短板。
- 没有通过外部 replay buffer 或手写规则掩盖局部学习本身。

## Route Coverage

当前教程把路线分成以下几类，后续实验只从这些类别里挑明确对照，不临时混搭：

| Route | Tutorial | Role In This Repo |
|---|---|---|
| three-factor / STDP family | [02](02-three-factor-learning-rules.md) | 最小稳定 baseline |
| e-prop family | [01](01-eprop.md) | 优先复现的成熟先进对照 |
| ETLP | [03](03-etlp.md) | 事件驱动 / 硬件友好参考 |
| local prediction / cognitive map | [04](04-cognitive-map-learner.md) | world model 和 planning 参考 |
| model-based SNN RL / dreaming | [05](05-model-based-snn-rl.md) | 闭环控制与 replay 观察参考 |
| robot control evaluation | [06](06-spiking-q-learning-robot-control.md) | 任务协议参考，不直接当学习规则 |
| TESS | [10](10-tess.md) | 时间 + 空间局部学习对照 |
| EchoSpike | [11](11-echospike.md) | 预测学习 / quiet dynamics 对照 |
| BrainTrace / pp-prop | [12](12-braintrace.md) | 系统化在线学习参考 |
| DECOLLE | [13](13-decolle.md) | 深层本地监督对照 |
| S-TLLR | [14](14-s-tllr.md) | 时间局部规则对照 |
