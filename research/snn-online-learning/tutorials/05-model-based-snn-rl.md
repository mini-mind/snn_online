# Model-based SNN RL: world model、policy 与 dreaming

## 资料

- Title: Towards biologically plausible model-based reinforcement learning in recurrent spiking networks by dreaming new experiences
- DOI: https://doi.org/10.1038/s41598-024-65631-y
- Article: https://www.nature.com/articles/s41598-024-65631-y

## 一句话理解

这篇论文研究的是：**让 recurrent SNN 学一个世界模型，再用这个世界模型生成模拟经验来训练策略，从而提高强化学习样本效率。**

它对应你的阶段 2 到阶段 3 的过渡：从预测世界到用预测帮助行动。

## 1. Model-free RL 和 model-based RL 的区别

Model-free RL 直接学：

```text
在状态 s 下做动作 a 的价值是多少？
```

或者：

```text
在状态 s 下应该采取什么动作？
```

它不显式学习环境如何变化。

Model-based RL 额外学习：

```text
state + action -> next_state
state + action -> reward
```

有了内部模型，就可以在脑内试错：

```text
如果我向左，会发生什么？
如果再向上，会不会得到奖励？
```

这就是 planning 或 dreaming 的基础。

## 2. 为什么 SNN 需要 world model

如果 SNN 只靠稀疏奖励更新权重，它会面临：

- 学习慢。
- 信用分配困难。
- 容易探索无效动作。
- 新环境适应差。

world model 给它一个更密集的学习目标：

```text
每一步都预测下一状态
每一步都能计算预测误差
每一步都能训练内部模型
```

然后 policy 可以通过真实经验和模拟经验共同改进。

## 3. dreaming 是什么

dreaming 不是玄学。工程上它就是内部模拟：

```text
从某个已知状态开始
选择一个动作
world model 预测下一状态和奖励
再选择动作
继续滚动若干步
把这些模拟轨迹当训练数据
```

伪代码：

```text
for real step:
    observe state
    choose action using policy
    execute action in environment
    observe next_state, reward
    update world_model with prediction error
    update policy with real transition

for dream step:
    sample remembered state
    rollout imagined transitions using world_model
    update policy with imagined transitions
```

## 4. recurrent SNN 在这里承担什么角色

recurrent SNN 适合时序任务，因为它有内部状态：

```text
当前 spike 活动
膜电位
突触 trace
循环连接状态
```

这些状态可以携带短期记忆。对于 partially observable environment，当前观察不完整，历史信息很重要。

因此 recurrent SNN 可以同时承担：

- 记忆短期上下文。
- 预测下一状态。
- 产生动作策略。
- 用局部/近似规则持续更新。

## 5. 这篇论文对你的项目的关键启发

你的系统不能只有“在线学习”，还需要“经验管理”。

建议把学习分成三层：

```text
fast online plasticity:
  新经验来时立刻调整短期 trace 或快速权重

replay/dream:
  在内部重放关键经验或生成模拟经验

slow consolidation:
  把稳定模式写入慢速长期权重，降低遗忘
```

这对应生物里的快速适应、睡眠重放和记忆固化，但工程上可以直接实现。

## 6. 和 LLM bootstrap 的关系

LLM 可以帮助构造更丰富的 dreaming，但要小心不要变成纯蒸馏。

合理用法：

```text
LLM 生成任务描述和初始场景
SNN 在环境中真实交互
world model 学习低层转移
LLM 偶尔生成高层反事实或语义评价
SNN 用局部规则吸收反馈
```

不合理用法：

```text
LLM 直接告诉每一步该做什么
SNN 只模仿 LLM 输出
```

后者学到的是蒸馏策略，不一定学到新的在线学习规则。

## 7. 需要重点审查的地方

读这篇时要问：

1. world model 是如何训练的？
2. policy 是如何训练的？
3. SNN 权重更新是否局部在线？
4. dreaming 轨迹如何选择？
5. 模拟经验错误时如何纠偏？
6. 和 model-free baseline 相比提升在哪里？

尤其要检查是否仍然依赖 surrogate gradient、BPTT 或离线优化。如果依赖，它仍有参考价值，但不是你要的最终学习规则。

## 8. 主要风险：模型偏差

world model 错了，dreaming 就会生成错误经验。policy 在错误经验上训练，可能越学越偏。

需要加入：

```text
uncertainty estimate
real feedback correction
dream rollout length limit
prediction error threshold
memory sampling strategy
```

也就是说，不确定时少 dream；真实反馈和模拟反馈冲突时，以真实反馈为准。

## 9. 建议复现实验

阶段化复现：

```text
1. 只学 world model：state + action -> next_state
2. 加 policy：用真实经验训练动作选择
3. 加 dreaming：用模拟经验训练 policy
4. 加持续变化环境：测试能否适应
5. 加局部学习限制：逐步移除反向传播依赖
```

评估指标：

- 达成奖励速度。
- 真实环境交互步数。
- world model 预测误差。
- dream 轨迹质量。
- 环境变化后的恢复速度。
- 旧任务遗忘程度。

## 10. 和四阶段路线的关系

- 阶段 1：需要底层在线 SNN 学习规则支撑。
- 阶段 2：world model 是核心。
- 阶段 3：policy 和 action selection 进入系统。
- 阶段 4：LLM 可以作为任务生成器、教师和语义评价器，但不应替代 world model 学习。

## 11. 原文核心方法速读：SNN 同时学模型、策略，并用 dream 扩增经验

这篇论文的核心是把 model-based RL 的三件事放进 recurrent spiking network 语境：

```text
1. world model
   预测执行动作后的下一状态/奖励

2. policy 或 controller
   根据当前状态选择动作

3. dreaming / imagination
   用 world model 生成模拟经验，减少真实环境试错
```

最小循环：

```text
real interaction:
  s_t -> choose a_t -> observe s_{t+1}, r_t
  update world model with prediction error
  update policy/value with real transition

dreaming:
  sample internal/remembered state
  rollout imagined transitions using world model
  update policy/value with imagined transition
```

这和普通 model-based RL 的区别在于：

```text
网络主体是 recurrent SNN
内部状态由 spike、膜电位和循环动力学承载
作者强调更 biologically plausible 的 experience generation
```

## 12. world model 学什么

world model 至少要学两个预测：

```text
state prediction:
  f(s_t, a_t) -> s_hat_{t+1}

reward prediction:
  g(s_t, a_t) -> r_hat_t
```

训练信号：

$$
\delta^s_t = s_{t+1} - \hat{s}_{t+1}
$$

$$
\delta^r_t = r_t - \hat{r}_t
$$

这些误差比稀疏奖励密集得多。即使没有得到最终奖励，系统每一步也能学习“我对世界的预测哪里错了”。

对你的项目，这意味着：

```text
prediction_error 用来更新 world model
reward_error / TD_error 用来更新 value 和 policy
```

不要把两种误差混成一个信号。

## 13. dreaming 的作用和风险

Dreaming 的价值：

```text
真实环境交互昂贵或危险
内部模拟可以产生更多训练样本
policy 可以在脑内试错
```

但它有一个核心风险：模型偏差。

```text
world model 错
=> dream 轨迹错
=> policy 在错误经验上训练
=> 行为越来越偏
```

所以一个稳健系统需要：

```text
限制 dream rollout 长度
优先从可靠状态开始 dream
用 prediction uncertainty 控制是否 dream
定期用真实反馈纠偏
真实经验优先级高于模拟经验
```

在你的设计里，可以把 dream 当成“巩固和补充”，不要让它完全替代真实交互。

## 14. recurrent SNN 为什么适合这个任务

Model-based RL 需要短期记忆：

```text
当前观察可能不完整
动作后果可能延迟
奖励依赖过去状态
world model 需要记住上下文
```

Recurrent SNN 的内部状态天然包含：

```text
membrane voltage
spike traces
recurrent activity
eligibility traces
adaptive thresholds
```

这些都可以承载短期上下文。问题是如何训练。论文的价值是把 SNN 放进 model-based RL + dreaming 的任务框架，让你看到最终要验证的系统形态。

## 15. 原文实验效果怎么读

这类实验的关键不是“dreaming 看起来像生物睡眠”，而是下面几个指标：

```text
1. 样本效率
   加入 world model / dreaming 后，真实交互步数是否减少？

2. 任务表现
   policy 是否比没有 dream 的 baseline 更快达到高奖励？

3. world model 质量
   预测误差是否足够低，能否支撑有用的模拟？

4. dream 轨迹价值
   用模拟经验训练是否真的改善真实行为？

5. 适应能力
   环境变化后，world model 和 policy 是否能重新调整？
```

你读结果时要特别检查：

```text
SNN 权重是否在线更新？
是否使用 surrogate gradient 或 BPTT？
是否依赖 replay buffer 和 batch 优化？
```

如果仍然依赖较多深度学习训练技巧，它依然有架构参考价值，但不是最终的局部学习答案。

## 16. 和 e-prop / ETLP 的组合方式

可以把论文里的架构目标，接到底层局部学习规则上：

```text
world model recurrent weights:
  用 prediction_error × eligibility_trace 更新

policy/value recurrent weights:
  用 TD_error 或 reward_error × eligibility_trace 更新

readout weights:
  可以先用更简单的 delta rule / RLS / supervised signal

dreaming memory:
  选择高误差、高奖励、高新奇经验进行 replay
```

这会形成一个更完整的智能体：

```text
底层：局部突触可塑性
中层：world model + value/policy
高层：planning / dreaming / consolidation
```

## 17. 最小复现实验

建议按下面顺序做，而不是直接复现完整论文：

```text
阶段 A：只训练 world model
  gridworld 中预测下一状态

阶段 B：加入 model-free policy
  用真实 transition 学动作价值

阶段 C：加入 dream transition
  world model 生成短 rollout，辅助训练 policy

阶段 D：加入局部学习限制
  把 world model 更新改成 trace × prediction_error

阶段 E：测试环境变化
  障碍或奖励位置改变，观察恢复速度和遗忘
```

最重要的对比：

```text
无 dream
有 dream 但 world model 不确定性不过滤
有 dream + 不确定性门控
纯 model-free
表格 model-based
SNN local model-based
```

