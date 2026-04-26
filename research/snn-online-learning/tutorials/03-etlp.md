# ETLP: 事件驱动三因子局部可塑性

## 资料

- Title: ETLP: Event-based Three-factor Local Plasticity for online learning with neuromorphic hardware
- arXiv: https://arxiv.org/abs/2301.08281
- PDF: https://arxiv.org/pdf/2301.08281
- Publisher page: https://iopscience.iop.org/article/10.1088/2634-4386/ad6f3b

## 一句话理解

ETLP 可以理解为：**把三因子学习规则做成事件驱动、局部、硬件友好的在线学习算法。**

它不是最宏大的智能体理论，但对你很重要，因为它把“局部学习规则”推进到了工程可实现层面。

## 1. 它和 e-prop 的差别

e-prop 更像从 BPTT 梯度出发，寻找在线局部近似。

ETLP 更像从 neuromorphic hardware 出发，问：

```text
如果硬件只能看到局部 spike、膜电位和一个教学信号，能不能在线学习？
```

因此 ETLP 的关键词是：

- event-based：有事件才计算，不做密集逐步更新。
- three-factor：局部活动乘以第三因子。
- local plasticity：权重更新尽量只依赖局部状态。
- online learning：数据流来时直接更新。
- hardware-aware：考虑神经形态硬件或 FPGA 友好性。

## 2. 为什么事件驱动重要

普通深度学习通常每个 batch 都对全网络做矩阵计算。SNN 的优势在于 spike 稀疏：

```text
没有 spike -> 没有事件 -> 不需要更新相关通路
```

如果算法仍然要求每个时间步扫描所有突触，那么它没有充分利用 SNN 的事件驱动优势。

ETLP 强调在事件发生时更新相关 trace 和权重，这更接近：

```text
输入 spike 到来
更新该输入相关突触的 trace
突触后状态变化
如果有教学信号，则局部更新权重
```

## 3. 三个因子分别是什么

简化理解：

```text
因子 1：突触前 spike trace
因子 2：突触后膜电位 / 活动状态
因子 3：教学信号 / 调制信号
```

突触前 trace 表示这个输入最近是否活跃。

突触后膜电位表示这个输入到来时，后神经元处在什么状态。膜电位比“是否发 spike”更连续，能提供更丰富的局部信息。

教学信号表示当前输出方向应该如何调整。它可以来自标签、错误、奖励或外部教师。

## 4. 用伪代码理解 ETLP

简化伪代码：

```text
initialize weights W
initialize pre_traces P = 0

for each event at time t:
    if input neuron i spikes:
        P[i] = update_pre_trace(P[i])

    update postsynaptic membrane voltage V[j]
    maybe generate output spike

    if teaching signal T[j] is available:
        for synapse i -> j touched by recent activity:
            delta = eta * P[i] * f(V[j]) * T[j]
            W[i, j] += delta
            W[i, j] = constrain(W[i, j])
```

注意这个过程不要求完整计算图，也不要求全局反向传播。

## 5. 从工程角度看它的价值

ETLP 逼你回答几个现实问题：

### 每个突触要存多少状态

如果每个突触都要存大量历史，算法无法扩展。ETLP 倾向少量 trace。

### 更新是否稀疏

事件驱动系统不能每个时间步更新所有权重，否则成本退回普通仿真。

### 教学信号如何广播

第三因子不能无限精细，否则又接近反向传播。需要低维、区域化或投影式信号。

### 是否能跑在硬件上

即便你现在只做软件仿真，硬件约束也能帮你避免设计过度复杂的学习规则。

## 6. 对本项目的启发

你的算法原型可以从 ETLP 学到一个原则：

```text
先定义每个突触允许保存的状态，再定义学习规则。
```

例如限制为：

```text
weight
pre_trace_fast
pre_trace_slow
eligibility
plasticity_gate
```

每个神经元限制为：

```text
membrane_voltage
adaptive_threshold
spike_trace
homeostasis_state
```

全局/区域调制器提供：

```text
reward_error
prediction_error
uncertainty
arousal/homeostasis
teacher_signal
```

这样设计出来的规则更容易实现，也更容易和 e-prop、ETLP 做对照。

## 7. 和 LLM bootstrap 的连接

LLM 可以作为 ETLP 的第三因子来源之一，但不应该直接替代学习体。

可行方式：

```text
SNN 执行动作或预测
环境返回真实结果
LLM 对结果做语义解释或产生教学信号
ETLP-like rule 用局部 trace * teacher modulation 更新权重
```

例如在语言化环境中，LLM 可以告诉系统：

```text
这个动作违反目标
这个特征更关键
这个状态和之前某经验相似
```

但权重更新仍由 SNN 的局部 trace 决定。

## 8. 应该重点读什么

读 ETLP 时不要纠结所有硬件实现细节，先抓：

1. 它的 pre trace 如何定义。
2. 突触后膜电位如何进入学习规则。
3. 教学信号如何构造。
4. 更新是事件驱动还是时间步驱动。
5. 实验任务是否真正在线。
6. 它和 surrogate gradient 的边界在哪里。

## 9. 局限

ETLP 更像底层局部可塑性规则，不是完整智能体架构。它没有直接解决：

- 复杂多步 planning。
- 长期记忆固化。
- 多目标冲突。
- 动作门控。
- 奖励自循环。

因此它适合作为阶段 1 的工程基线，以及阶段 4 中“LLM 教学信号如何变成局部更新”的候选实现方式。

## 10. 建议复现实验

不要一开始复现大任务。建议：

```text
任务 A：在线二分类，输入分布会漂移
任务 B：延迟标签，标签晚几个时间步到达
任务 C：少量类别增量学习，测试遗忘
任务 D：简单 event stream 分类
```

对比：

```text
STDP
reward-modulated STDP
e-prop
ETLP-like rule
你的改进规则
```

