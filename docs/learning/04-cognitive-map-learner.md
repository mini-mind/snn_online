# Cognitive Map Learner: 局部预测学习如何支持 planning

## 资料

- Title: Local prediction-learning in high-dimensional spaces enables neural networks to plan
- DOI: https://doi.org/10.1038/s41467-024-46586-0
- Article: https://www.nature.com/articles/s41467-024-46586-0

## 一句话理解

这篇工作的关键思想是：**网络不只是学输入到输出的映射，而是用局部预测学习形成环境转移结构；一旦学到结构，就能在内部表示空间里规划。**

它是你阶段 2 的重要参考：从“会在线改权重”走向“会学习世界模型”。

## 研究者使用定位

本文关注的是“局部预测学习如何产生可规划结构”，不是认知地图概念科普：

```text
入门：把 cognitive map 理解成 learned transition structure
掌握：把 state + action -> next_state 变成预测误差学习
熟练：能区分 reward learning、prediction learning 和 planning
精通：能把局部预测误差接到底层 SNN 可塑性规则上
```


## 1. 为什么预测学习比奖励学习更基础

奖励学习的问题是奖励太少。一个智能体即使没有奖励，也应该不断学习：

```text
这个动作之后会发生什么？
这个状态和哪个状态相邻？
这个观察是否异常？
哪些变化由我的动作造成？
```

这类学习可以由预测误差驱动：

$$
\delta^{pred}_t = s_{t+1} - \hat{s}_{t+1}
$$

预测误差比奖励密集得多。每一个时间步都有下一状态，因此每一步都能学习。

## 2. 什么是认知地图

认知地图不是一张真的图片，而是一种内部结构表示。它让 agent 知道：

```text
状态 A 通过动作 left 到状态 B
状态 B 通过动作 up 到状态 C
目标 G 离当前状态大概在哪个方向
```

在工程上，可以理解为 learned transition graph 或 state embedding space。

## 3. 核心机制

它学习的不是：

```text
state -> label
```

而是：

```text
state + action -> next_state
```

如果状态被编码到高维空间，那么动作可以被看作在这个空间中的某种变换。学到这些变换后，系统就能做规划：

```text
当前状态表示
目标状态表示
尝试动作方向
选择能让表示更接近目标的动作序列
```

## 4. 从非神经动力学视角理解

可以把它和传统机器学习里的 world model 对应起来：

```text
encoder: observation -> latent_state
transition_model: latent_state + action -> predicted_latent_next
planner: search in latent space
```

区别是这篇工作强调局部预测学习和高维表示，而不是端到端反向传播训练一个黑盒世界模型。

## 5. 局部预测学习如何变成学习规则

简化形式：

```text
当前状态 s_t
执行动作 a_t
网络预测下一状态 s_hat_(t+1)
真实下一状态 s_(t+1) 到来
产生 prediction_error
用局部规则更新相关连接
```

伪代码：

```text
for each step:
    z_t = encode(observation_t)
    a_t = selected_action
    z_pred = transition_predictor(z_t, a_t)

    execute a_t
    z_next = encode(observation_next)

    error = z_next - z_pred
    update local synapses using eligibility * error_signal
```

这里的 `error_signal` 可以作为三因子学习规则的第三因子。

## 6. 它和 e-prop / ETLP 的关系

e-prop / ETLP 更关注：

```text
某个突触如何在线更新？
```

Cognitive Map Learner 更关注：

```text
在线更新的目标是什么？
```

答案是：学习环境结构。

所以它们可以组合：

```text
底层：e-prop/ETLP-like local rule
目标：prediction error / transition learning
结果：world model / cognitive map
```

## 7. 对你的项目的架构启发

你的“皮层模拟”不应只做被动表征。可拆出几个模块：

```text
sensory_encoder:
  把外部输入编码成状态表示

transition_model:
  学习 state + action -> next_state

prediction_error_module:
  计算预测误差，作为局部学习调制信号

planner:
  在内部状态空间中搜索动作序列

basal_ganglia_gate:
  在多个候选动作中选择一个执行
```

LLM 可以放在更高层：

```text
解释任务目标
生成候选目标
把语言目标转成状态约束
评价计划是否符合语义
```

但世界模型的低层更新应由 SNN 自己完成。

## 8. 为什么这对“新学习算法”重要

如果一个算法只能根据标签或奖励学习，它很难成为通用智能体学习规则。大脑大量学习不是外部监督，而是预测：

```text
我看到什么
我会听到什么
我移动后会看到什么
我的身体状态会怎么变
```

因此你设计的新规则应至少支持两种第三因子：

```text
reward_error
prediction_error
```

预测误差负责建模世界，奖励误差负责价值选择。

## 9. 研究者检查点

跨领域使用时重点抓：

1. 状态如何表示。
2. 动作如何影响表示。
3. 局部预测误差如何更新网络。
4. planning 如何从 learned representation 中产生。
5. 它和传统 deep RL world model 的差别。
6. 它是否需要大量离线训练或全局反向传播。

## 10. 局限

这类系统通常仍在受控环境中验证。它不自动解决：

- 高维真实感知输入。
- 长期记忆和灾难性遗忘。
- 多目标冲突。
- 语言目标。
- 安全约束。

但它提供了一个非常关键的方向：**局部学习不应只追求拟合输出，还应该学习可用于 planning 的环境结构。**

## 11. 最小研究原型

你可以做一个小 gridworld：

```text
输入：当前位置局部观察
动作：up/down/left/right
目标：预测下一观察
学习：局部 prediction error
评估：给定目标位置，看能否规划路径
```

然后再让 LLM bootstrap：

```text
LLM 只给语言任务描述
SNN 自己学习 transition model
planner 生成动作序列
LLM 评价高层目标是否达成
```

## 12. 研究者压缩模型：学习“动作如何移动内部表示”

这篇工作的可迁移核心不是 spike，而是一个更基础的智能体问题：

```text
如果没有奖励，神经网络能否靠预测学习形成可用于 planning 的认知地图？
```

它学习的对象不是标签：

```text
observation -> class
```

而是环境转移结构：

```text
current_state + action -> next_state
```

更抽象地说：

```text
当前内部表示 z_t
执行动作 a_t
预测下一个内部表示 z_hat_{t+1}
真实下一个表示 z_{t+1} 到来
用 prediction error 更新连接
```

公式骨架：

$$
\delta^{pred}_t = z_{t+1} - \hat z_{t+1}
$$

$$
\Delta w \propto e_{local}(t) \delta^{pred}_t
$$

这就是它和三因子学习的连接：预测误差可以作为第三因子。

## 13. 为什么高维表示能支持 planning

传统地图可以显式写成图：

```text
A --right--> B --up--> C
```

但高维神经表示里没有显式节点表。它靠向量空间组织状态：

```text
相似/相邻状态在表示空间中有结构关系
动作对应从一个表示到另一个表示的变换
目标可以表示成某个目标向量或目标区域
```

planning 的直觉是：

```text
如果我知道 action a 会把 z 往哪个方向推
那么我可以在脑内试几个动作
选一个让 z 更接近 goal 的动作
```

所以这项工作的关键不是“记住所有路径”，而是学习一个可泛化的转移结构。

## 14. 方法可以拆成四个模块

```text
1. encoder
   把外部观察编码成高维内部状态 z

2. transition learner
   学习在动作 a 下，z_t 如何变成 z_{t+1}

3. local prediction update
   用预测误差更新相关连接，不需要每次都有奖励

4. planner
   在学到的表示空间中模拟动作后果，选择通向目标的动作
```

最小伪代码：

```text
observe o_t
z_t = encode(o_t)
choose action a_t
z_pred = predict(z_t, a_t)

execute a_t
observe o_{t+1}
z_next = encode(o_{t+1})

error = z_next - z_pred
update transition connections using local_trace * error

for planning:
    simulate candidate actions in latent space
    choose action whose predicted state is closest to goal
```

## 15. 它和 model-based RL 的区别

它和 model-based RL 都学 world model，但关注点不同。

```text
model-based RL:
  学 next_state / reward
  目标通常是最大化累计奖励
  planning 常和 value/policy 结合

cognitive map learner:
  学状态之间的可达结构
  奖励不是必要条件
  planning 可以直接基于目标状态和表示空间距离
```

对你的路线来说，这项工作提醒你：

```text
世界模型不应该只为奖励服务
它首先应该学“世界如何变化”
```

奖励学习是在世界模型之上的价值层。

## 16. 实验结论与可迁移判断

这项工作的实验重点通常可以按三个问题理解：

```text
1. 是否能只靠局部预测学习形成可导航结构？
2. 学到的结构是否能支持到新目标的规划？
3. 在高维状态空间中是否仍然有效，而不只是在小图上记表？
```

分析结果时不要只问“准确率是多少”，而要问：

```text
预测误差是否下降？
规划成功率是否提高？
遇到新目标时是否能复用旧地图？
环境结构变化后是否能重新适应？
是否比无模型试错更省交互？
```

如果一个系统能在没有直接奖励的情况下，通过预测学习获得可用于 planning 的结构，这对你的阶段 2 非常重要。

## 17. 和你的 SNN 项目怎么接

可以把它翻译成 SNN 版本：

```text
sensory SNN:
  observation -> spike latent state

transition SNN:
  latent_state + action context -> predicted next latent_state

error module:
  actual latent - predicted latent

plasticity:
  local eligibility trace × prediction_error

planner/gate:
  在 latent space 里评估候选动作
```

这里的第三因子不是奖励，而是：

```text
prediction_error_channel[action/context]
```

这会让你的学习体即使没有外部奖励，也能持续学习环境结构。

## 18. 最小研究原型

先做一个最小认知地图原型：

```text
环境：5x5 gridworld
观察：当前位置 one-hot 或局部视野
动作：up/down/left/right
学习目标：预测下一状态表示
规划目标：给定任意目标位置，规划路径
```

对比：

```text
无 world model 的 model-free agent
表格 transition model
MLP world model
SNN local prediction learner
```

核心指标：

```text
next-state prediction error
到达目标成功率
新目标泛化
障碍改变后的重新学习速度
学习是否需要奖励
```

如果这个实验跑通，你就有了从“局部可塑性”走向“可规划世界模型”的最小桥梁。

## 19. 本项目代码对应

- Code: [../../src/toy_learning/cognitive_map_etlp_toy.py](../../src/toy_learning/cognitive_map_etlp_toy.py)
- Notes: [../../src/README.md](../../src/README.md)

该原型把 Cognitive Map 的 `state + action -> next_state` 学习目标，接到 ETLP-like prediction-error 局部更新上：

$$
\Delta W_a[o, i] = \eta \cdot \delta^{pred}_o(t) \cdot \bar{x}_i(t)
$$
