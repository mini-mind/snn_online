# e-prop: 在线训练 recurrent SNN 的核心入口

## 资料

- Title: A solution to the learning dilemma for recurrent networks of spiking neurons
- DOI: https://doi.org/10.1038/s41467-020-17236-y
- Article: https://www.nature.com/articles/s41467-020-17236-y

## 一句话理解

e-prop 的核心思想是：**不要把时间展开后再反向传播，而是让每个突触在线维护一个“我最近是否该为未来误差负责”的资格迹，然后用任务学习信号调制这个资格迹来更新权重。**

如果你要研究“反向传播之外的在线学习规则”，e-prop 是必须理解的基线。

## 1. 背景问题：为什么 recurrent SNN 难学

普通前馈网络的权重更新依赖当前样本的误差。循环网络不一样，它当前输出可能受很久之前的状态影响。

例如：

```text
t=1 看到线索 A
t=100 才需要输出答案
```

如果 t=100 输出错了，系统要知道 t=1 附近哪些突触活动应该被调整。这就是时间信用分配问题。

深度学习常用 BPTT：

```text
把网络按时间展开成 100 层
从 t=100 的 loss 反传到 t=1
更新所有相关权重
```

但 BPTT 对你的目标有几个问题：

- 它需要保存历史状态，不是真正在线。
- 它依赖全局梯度，不局部。
- 它不适合事件驱动硬件。
- 它和生物突触更新机制距离很远。

e-prop 试图保留“优化任务误差”的能力，同时把更新改成在线局部形式。

## 2. 核心拆解：梯度被拆成两部分

对一个 recurrent 网络，某个权重 `w_ij` 的理想梯度可以粗略拆成：

```text
任务误差对神经元输出的影响
*
权重对神经元状态的局部影响
```

e-prop 把它写成类似：

```text
delta_w_ij(t) = learning_signal_j(t) * eligibility_trace_ij(t)
```

两个部分含义不同：

- `eligibility_trace_ij(t)`：这个突触最近是否参与了相关活动。
- `learning_signal_j(t)`：当前这个神经元/输出通道应该往哪个方向改。

这种拆分非常重要，因为 eligibility trace 可以本地在线维护，而 learning signal 可以通过较少的反馈通道广播或投影进来。

## 3. eligibility trace 到底记录什么

假设突触 `i -> j`：

- `i` 是突触前神经元。
- `j` 是突触后神经元。
- `w_ij` 是权重。

如果 `i` 在某时刻发放 spike，并且它影响了 `j` 的膜电位，那么这个突触就对未来输出有潜在责任。eligibility trace 就增加。

之后它逐渐衰减：

```text
e_ij(t) = decay * e_ij(t-1) + local_contribution_ij(t)
```

这里的 `local_contribution` 不需要知道全局 loss，只需要知道：

- 突触前 spike。
- 突触后膜电位或状态。
- 神经元的泄漏、阈值、重置等本地动力学。

可以把它理解成突触上的短期缓存：

```text
如果未来奖励/误差来了，我可能和它有关。
```

## 4. learning signal 是什么

eligibility trace 只说明“可能有关”，不说明“应该增强还是削弱”。learning signal 给出方向。

在监督任务里，learning signal 可以来自输出误差：

```text
target - output
```

在强化学习里，它可以来自奖励预测误差：

```text
实际奖励 - 预期奖励
```

在生物类比里，它可以类似多巴胺、注意或其他调制信号。但工程上不要急着脑区命名，先把它当成“低维任务反馈”。

## 5. 用伪代码理解 e-prop

下面是简化版，不是论文完整公式：

```text
initialize weights W
initialize neuron states V
initialize eligibility traces E = 0

for each time step t:
    input spikes arrive
    update neuron states V
    emit output spikes or rates

    for each synapse i -> j:
        E[i, j] = decay * E[i, j] + local_trace(pre_i, post_j, state_j)

    compute learning signal L[j] from task feedback

    for each synapse i -> j:
        W[i, j] += eta * L[j] * E[i, j]
```

真正算法会区分不同神经元模型、输出读出方式、学习信号投影方式和符号细节，但骨架就是这样。

## 6. 它为什么比 STDP 更接近你的目标

STDP 通常只看 spike 时间差：

```text
pre before post -> strengthen
post before pre -> weaken
```

这能形成某些时序关联，但很难表达任务目标。e-prop 加入 learning signal，因此可以学习“对任务有用”的关联，而不仅是“时间上接近”的关联。

对比：

```text
STDP:
  局部，但任务目标弱

Backprop/BPTT:
  任务目标强，但不局部、不在线

e-prop:
  试图在任务目标和局部在线之间折中
```

## 7. 它和三因子学习规则的关系

e-prop 可以看作三因子学习规则的一种严谨版本：

```text
因子 1：突触前活动
因子 2：突触后状态
因子 3：learning signal
```

前两个因子形成 eligibility trace，第三因子调制更新。

## 8. 对你的项目怎么用

你可以把 e-prop 作为第一个 baseline：

```text
任务 1：延迟线索分类
任务 2：简单序列预测
任务 3：gridworld 中的短期记忆任务
任务 4：稀疏奖励 bandit
```

先实现：

```text
LIF recurrent SNN
eligibility trace
简单 learning signal
权重归一化
兴奋/抑制约束
```

然后提出你的改进：

- learning signal 不只是输出误差，而是组合预测误差、奖励误差、不确定性和内稳态。
- eligibility trace 不只是单一时间常数，而是多时间尺度。
- 加入门控机制，决定哪些突触可塑、哪些冻结。
- 加入 replay，把在线经验转成离线巩固。

## 9. 读这篇论文时抓什么

重点看四件事：

1. 它如何从 BPTT 梯度推导出 eligibility trace。
2. learning signal 是如何近似真实梯度的。
3. 不同 e-prop 变体之间有什么差别。
4. 实验任务是否真的展示了长时序信用分配能力。

不需要一开始就完全吃透所有公式。先抓住：

```text
local trace handles memory
learning signal handles task direction
their product updates the synapse online
```

## 10. 局限和风险

e-prop 不是完整智能体算法。它仍然需要外部定义学习信号。它也没有自动解决：

- 奖励从哪里来。
- 目标如何生成。
- 多个动作如何竞争。
- 长期记忆如何固化。
- 持续学习如何抗遗忘。

所以它适合作为你的“底层突触更新基线”，而不是最终架构。

## 11. 和四阶段路线的关系

- 阶段 1：核心资料，直接对应小型 recurrent SNN 在线学习。
- 阶段 2：可扩展到预测误差学习，但论文主线不是世界模型。
- 阶段 3：可以接 RL learning signal，但还缺动作门控架构。
- 阶段 4：可作为 LLM bootstrap 环境里被训练的 SNN 学习体底层规则。

