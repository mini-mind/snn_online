# Cognitive Map Learner: 局部预测学习如何支持 planning

## 资料

- Title: Local prediction-learning in high-dimensional spaces enables neural networks to plan
- DOI: https://doi.org/10.1038/s41467-024-46586-0
- Article: https://www.nature.com/articles/s41467-024-46586-0

## 一句话理解

这篇工作的关键思想是：**网络不只是学输入到输出的映射，而是用局部预测学习形成环境转移结构；一旦学到结构，就能在内部表示空间里规划。**

它是你阶段 2 的重要参考：从“会在线改权重”走向“会学习世界模型”。

## 1. 为什么预测学习比奖励学习更基础

奖励学习的问题是奖励太少。一个智能体即使没有奖励，也应该不断学习：

```text
这个动作之后会发生什么？
这个状态和哪个状态相邻？
这个观察是否异常？
哪些变化由我的动作造成？
```

这类学习可以由预测误差驱动：

```text
prediction_error = actual_next_state - predicted_next_state
```

预测误差比奖励密集得多。每一个时间步都有下一状态，因此每一步都能学习。

## 2. 什么是认知地图

认知地图不是一张真的图片，而是一种内部结构表示。它让 agent 知道：

```text
状态 A 通过动作 left 到状态 B
状态 B 通过动作 up 到状态 C
目标 G 离当前状态大概在哪个方向
```

在工程上，可以理解为 learned transition graph 或 state embedding space。

## 3. 这篇论文的核心思想

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

你的“皮层模拟”不应只做被动表征。建议拆出几个模块：

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

## 9. 应该重点读什么

读这篇时重点抓：

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

## 11. 建议复现实验

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

