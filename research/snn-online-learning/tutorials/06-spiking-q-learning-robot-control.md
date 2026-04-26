# SNN Deep RL / Robot Control: 从学习规则走向动作闭环

## 资料

- Title: Exploring spiking neural networks for deep reinforcement learning in robotic tasks
- DOI: https://doi.org/10.1038/s41598-024-77779-8
- Article: https://www.nature.com/articles/s41598-024-77779-8

## 一句话理解

这类工作展示的是：**SNN 可以被放进深度强化学习和机器人控制任务中，用于动作价值估计、策略表示或控制决策。**

它对你的意义不是“已经解决在线局部学习”，而是提供阶段 3 的任务形态：闭环控制、动作选择、价值评估和真实反馈。

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

```text
Q(s, a) = 在状态 s 下执行动作 a 的长期价值
```

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

所以这篇机器人控制资料应和 model-based SNN RL 一起读：

- 机器人控制给任务形态。
- model-based SNN RL 给样本效率和 dreaming 思路。
- e-prop/ETLP 给底层在线更新规则。

## 8. 应该重点读什么

读这类论文时重点审查：

1. SNN 在架构里到底负责什么。
2. 权重如何训练。
3. 推理时是否继续学习。
4. 是否有真正在线更新。
5. 动作是离散还是连续。
6. reward 是否稀疏或延迟。
7. 是否和非 SNN baseline 公平比较。

## 9. 建议你怎么借鉴

不要一开始做复杂机械臂。建议阶段化：

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

机器人控制论文往往更关注“能不能用 SNN 做 RL”，而不是“是否发现了替代反向传播的学习规则”。所以它应放在你的阶段 3 评估集，而不是理论核心。

