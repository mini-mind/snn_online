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

