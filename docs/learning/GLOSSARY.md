# SNN Programmer Glossary

本页面向没有类脑研究背景的程序员，用工程语言解释本仓当前实验里反复出现的术语。

## 基本对象

- **Spike**：离散事件，通常记为 `0/1`。`1` 表示该神经元在当前时间步发放了一个脉冲。
- **Membrane voltage**：神经元内部 accumulator。输入会让它上升，泄漏会让它衰减，超过阈值就发放 spike。
- **LIF**：Leaky Integrate-and-Fire。最简单的脉冲神经元，可理解为“带衰减的积分器 + 阈值触发器”。
- **Izhikevich**：比 LIF 多一个 recovery 状态的神经元模型，能产生更丰富的发放模式，但更慢、更难调。
- **Recurrent SNN**：带循环连接的脉冲网络。当前输出依赖当前输入和过去 spike，因此可以承载短期状态。

## 局部学习

- **Eligibility trace**：突触本地的“最近是否相关”缓存。它不直接决定方向，只记录 pre/post 活动是否让这条连接值得更新。
- **Three-factor rule**：`delta_w = lr * eligibility * modulation`。前两个因子来自突触局部活动，第三个因子是奖励、预测误差或价值反馈。
- **Modulation signal**：低维反馈信号，不是反向传播梯度。当前 point robot 用 TD error 和 prediction error 混合成一个 scalar modulation。
- **Neuron-specific modulation**：每个目标神经元有自己的 modulation。它比单个全局 scalar 更接近 e-prop 的 learning signal。
- **e-prop**：用在线 eligibility trace 和 learning signal 近似替代 BPTT 的 recurrent SNN 学习方法。本仓目前只有简化影子，还不是严格 e-prop。

## 时间结构

- **STDP**：Spike-Timing-Dependent Plasticity。根据 pre spike 和 post spike 的先后顺序与时间间隔调整突触。
- **Synchrony**：pre/post 活动在时间上是否接近同步。`tess_like` 里用多时间尺度 trace 捕捉这种关系。
- **TESS-like**：本仓的简化多时间尺度局部规则，不是完整论文复现。它组合 fast/slow pre trace、post trace、eligibility 和 modulation。
- **Recurrent delay line**：让 recurrent edge 读取过去若干步的 source spike。它改变连接计算本身，接近突触传输延迟，而不是给 readout 额外外挂记忆特征。

## 闭环 agent

- **World model**：预测 `next_observation` 的模型。当前 point robot 用它估计动作后果，并产生 prediction error。
- **TD error**：`reward + gamma * next_value - current_value`。表示结果比预期好还是坏，是强化学习里的核心反馈。
- **Partial observability**：环境状态只给一部分。`partial_goal_cue` 在 episode 前几步给目标方向，之后隐藏方向，用来测试短期记忆。
- **Replay / dreaming**：把过去经验或 world model 生成的 imagined experience 重新喂给学习系统，用于巩固和样本效率提升。

## 本仓当前地图

- `src/models/recurrent_spiking.py`：LIF/Izh RSNN、`three_factor`、`tess_like`、recurrent delay line 和 modulation 接口。
- `src/models/point_robot_closed_loop.py`：RSNN feature extractor + world model head + TD value head。
- `src/experiments/compare_plasticity_rules.py`：比较 `three_factor` 和 `tess_like`。
- `src/experiments/compare_mainline_history.py`：比较主线历史阶段。
- `src/envs/point_robot.py`：当前主要闭环任务环境，尤其是 `partial_goal_cue`。

当前路线是：先在部分可观测点机器人上确认短期记忆机制，再推进 neuron-specific modulation、真正 delay line 和 replay/dreaming。
