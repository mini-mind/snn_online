# SNN Deep RL / Robot Control: 从学习规则走向动作闭环

## 资料

- Title: Exploring spiking neural networks for deep reinforcement learning in robotic tasks
- DOI: https://doi.org/10.1038/s41598-024-77779-8
- Article: https://www.nature.com/articles/s41598-024-77779-8

## 一句话理解

这类工作展示的是：**SNN 可以被放进深度强化学习和机器人控制任务中，用于动作价值估计、策略表示或控制决策。**

它对你的意义不是“已经解决在线局部学习”，而是提供阶段 3 的任务形态：闭环控制、动作选择、价值评估和真实反馈。

## 研究者使用定位

本文把机器人控制论文当成评估协议来源，而不是学习规则答案：

```text
入门：知道 SNN 可以放进 state encoder、Q network、policy 或 controller
掌握：把 TD error 看成 reward modulation 的第三因子
熟练：能拆分控制性能、训练方式、稀疏性和在线性
精通：能把局部学习规则放进 observe-act-learn 闭环并设计对照实验
```


## 1. 为什么机器人控制比分类更重要

如果一个 SNN 学习规则只能做分类，它离智能体还很远。

智能体需要处理：

```text
当前状态不完整
动作会改变未来输入
奖励可能延迟
错误动作可能造成代价
环境持续变化
```

机器人控制任务天然包含这些问题。即使是仿真机械臂，也比静态分类更接近 agent 学习。

## 2. Deep RL 的基本结构

强化学习的基本循环：

```text
observe state
choose action
environment transitions
receive reward
update policy/value
```

Q-learning 学：

$$
Q(s, a) = \mathbb{E}\left[\sum_{k=0}^{\infty} \gamma^k r_{t+k} \mid s_t=s, a_t=a\right]
$$

也就是：在状态 $s$ 下执行动作 $a$ 的长期价值。

策略学习学：

```text
policy(s) = 应该采取哪个动作
```

机器人控制任务可能是：

- 到达目标位置。
- 操作机械臂。
- 避障。
- 平衡控制。
- 连续动作轨迹优化。

## 3. SNN 可以放在哪里

SNN 在 deep RL 里可能承担不同角色：

```text
state encoder:
  把状态编码成 spike 表示

Q network:
  输出每个动作的价值

policy network:
  直接输出动作或动作分布

critic:
  估计状态价值

controller:
  根据 spike 活动生成低层控制命令
```

不同位置对应不同研究问题。如果 SNN 只是替换一个前馈网络，核心学习仍由反向传播完成，那它对你的项目帮助有限。

## 4. 需要警惕的训练方式

很多 SNN + deep RL 工作并不满足你的约束。常见情况：

```text
surrogate gradient:
  前向用 spike，训练仍靠近似反向传播

ANN-to-SNN:
  先训练普通 ANN，再转换为 SNN

hybrid model:
  SNN 只是 encoder，critic 或 learner 仍是 ANN

offline replay:
  训练大量依赖经验回放和 batch 优化
```

这些都不是坏事，但要分清：

```text
SNN 用于控制任务 != 已有在线局部学习规则
```

## 5. 对你的项目有什么价值

它可以作为阶段 3 的测试方向。

你的新学习规则如果想超越玩具任务，最终至少要在某种闭环控制中验证：

```text
输入：连续状态或事件流
内部：SNN recurrent dynamics
学习：local eligibility * modulation
输出：动作选择
反馈：奖励、预测误差、风险
```

评估时可以问：

- 它能否从试错中学会动作？
- 奖励延迟时能否更新正确连接？
- 环境变化时能否快速适应？
- 学新任务时是否忘掉旧控制技能？
- 是否比 STDP/e-prop/ETLP baseline 更好？

## 6. 基底节式门控应该放在哪里

机器人控制暴露了一个核心问题：动作选择。

大脑类比里，基底节不是直接产生肌肉动作，而是参与选择、启动和抑制动作。工程上可以设计：

```text
candidate_generator:
  产生多个候选动作或计划

value_module:
  估计每个候选的价值、风险和不确定性

gate:
  选择一个动作执行，并抑制其他动作

learning_rule:
  根据结果更新候选生成和价值估计
```

如果没有这个门控层，SNN 可能会有很多内部活动，但行为输出不稳定。

## 7. 和 world model 的关系

纯 model-free 控制通常样本效率低。更符合你路线的是：

```text
world model 预测动作后果
value module 评估后果
gate 选择动作
local rule 更新模型和价值
```

所以这类机器人控制资料应和 model-based SNN RL 一起使用：

- 机器人控制给任务形态。
- model-based SNN RL 给样本效率和 dreaming 思路。
- e-prop/ETLP 给底层在线更新规则。

## 8. 研究者检查点

跨领域使用这类论文时重点审查：

1. SNN 在架构里到底负责什么。
2. 权重如何训练。
3. 推理时是否继续学习。
4. 是否有真正在线更新。
5. 动作是离散还是连续。
6. reward 是否稀疏或延迟。
7. 是否和非 SNN baseline 公平比较。

## 9. 如何借用为研究原型

不要从复杂机械臂开始，按阶段建立可证伪控制原型：

```text
任务 1：multi-armed bandit，测试奖励调制
任务 2：gridworld，测试状态-动作-奖励
任务 3：cartpole，测试连续状态控制
任务 4：简化机械臂 reacher，测试连续动作或离散动作组合
```

每个任务都强制限制：

```text
不使用 BPTT
不使用全局梯度
只允许 local trace + modulation
记录在线权重变化
测试任务切换后的遗忘
```

## 10. 局限

机器人控制工作往往更关注“能不能用 SNN 做 RL”，而不是“是否发现了替代反向传播的学习规则”。所以它应放在你的阶段 3 评估集，而不是理论核心。

## 11. 研究者压缩模型：把 SNN 放进深度强化学习控制闭环

这篇工作的可迁移价值是任务形态，而不是底层学习规则革命。它展示的是：

```text
SNN 可以作为 deep RL agent 的一部分
在机器人控制任务中表示状态、价值或策略
```

强化学习闭环是：

```text
observe state s_t
choose action a_t
execute in robot/environment
receive reward r_t and next state s_{t+1}
update policy/value
```

如果用 Q-learning，核心目标是：

$$
Q(s_t, a_t) \approx r_t + \gamma \max_a Q(s_{t+1}, a)
$$

TD error：

$$
\delta_t = r_t + \gamma \max_a Q(s_{t+1}, a) - Q(s_t, a_t)
$$

在你的局部学习路线里，$\delta_t$ 可以变成第三因子：

$$
\Delta w_{ij} = \eta e_{ij}(t) \delta_t
$$

但很多 SNN deep RL 论文并不是这样训练。它们常常仍用 surrogate gradient、ANN-to-SNN conversion 或标准 deep RL 优化器。

## 12. SNN 在机器人 RL 里常见的三种位置

```text
1. spike encoder
   把连续状态转成 spike train
   后面的 learner 可能仍是 ANN/RL 算法

2. Q network / policy network
   SNN 直接输出动作价值或动作概率
   训练可能使用 surrogate gradient

3. low-level controller
   SNN 产生运动控制信号
   上层 RL 决定目标或动作模式
```

对你的项目，最有价值的是第二和第三种，尤其是：

```text
SNN 内部权重能否在交互中持续更新？
动作选择错误后，奖励误差能否调制早先的资格迹？
```

如果 SNN 只是一个静态 encoder，意义就比较有限。

## 13. 机器人控制为什么比分类难

分类任务：

```text
输入样本固定
标签明确
错误不改变下一次输入
```

机器人控制：

```text
动作会改变未来输入
奖励可能延迟
状态可能部分可观测
错误动作会带来代价
探索本身有风险
```

所以它能暴露局部学习规则的真实问题：

```text
信用分配
动作门控
稳定探索
灾难性遗忘
适应环境变化
```

这就是为什么这项工作适合放在阶段 3：它告诉你最终应该把学习规则放到什么任务里检验。

## 14. 实验结论与可迁移判断

分析这类结果时，不要只看 reward 曲线是否上升。要拆开看：

```text
1. 控制性能
   是否能完成到达、避障、平衡、机械臂控制等任务？

2. 与 ANN baseline 比较
   SNN 是否接近或超过同规模 ANN？
   比较是否公平？

3. 能耗/稀疏性
   spike 活动是否真的减少计算？
   是否估算能耗或事件数量？

4. 训练方式
   是在线局部学习，还是 surrogate gradient / deep RL optimizer？

5. 迁移与鲁棒性
   任务变化、噪声、动力学变化时是否仍稳定？
```

如果结果显示 SNN 在机器人任务中能达到不错表现，说明：

```text
SNN 作为控制网络是可行的
```

但它不自动说明：

```text
在线局部学习问题已经解决
```

这两个结论必须分开。

## 15. 如何把它改造成你的实验平台

你可以借用机器人 RL 的任务，但替换训练规则。

```text
常见做法：
  SNN + surrogate gradient / deep RL training

你的版本：
  recurrent SNN + eligibility trace
  TD_error / reward_error 作为第三因子
  prediction_error 更新 world model
  gate 负责动作选择
```

最小动作闭环：

```text
state_encoder -> recurrent SNN -> Q/action readout -> action
reward -> TD_error -> modulate eligibility traces
next_state -> prediction_error -> update world model
```

这样机器人控制工作就从“结果参考”变成“评估任务来源”。

## 16. 最小研究原型路线

不要直接上复杂机械臂：

```text
1. bandit
   没有状态转移，只测 reward modulation

2. gridworld
   有状态转移和延迟奖励，动作离散

3. cartpole
   连续状态，离散动作，反馈密集

4. simple reacher
   连续控制或离散化控制，接近机器人任务
```

每个阶段都记录：

```text
平均回报
学习速度
spike rate
权重更新次数
环境变化后的恢复速度
旧任务遗忘
```

## 17. 你真正要从这篇拿走什么

这篇不是你的底层算法答案，而是一个提醒：

```text
如果学习规则不能闭环控制动作
它还只是表征学习或分类学习
```

所以阶段 3 的目标应该是：

```text
把 local trace × modulation 放进一个真的 observe-act-learn loop
```

而不是只在静态数据集上刷准确率。

## 18. 本项目代码对应

- R-SNN adapter: [../../src/models/recurrent_spiking.py](../../src/models/recurrent_spiking.py)
- Environment: [../../src/envs/point_robot.py](../../src/envs/point_robot.py)
- Training loop: [../../src/models/point_robot_closed_loop.py](../../src/models/point_robot_closed_loop.py)
- Notes: [../../src/README.md](../../src/README.md)

该原型把前面的 ETLP-like 局部更新和 Cognitive Map world model，接到 `observe -> act -> learn` 控制闭环：

```text
continuous observation -> R-SNN recurrent state -> action
prediction_error -> world model readout
TD_error -> action-value readout
```

这里的实验边界是：`snn_online` 负责任务、读出头和对照脚本；循环脉冲网络执行通过 `models/recurrent_spiking.py` 接到外部 `dynn`。文档只保留实验接口，不展开 `dynn` 内部实现细节。
