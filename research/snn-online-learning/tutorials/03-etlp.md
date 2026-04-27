# ETLP: 事件驱动三因子局部可塑性

## 资料

- Title: ETLP: Event-based Three-factor Local Plasticity for online learning with neuromorphic hardware
- arXiv: https://arxiv.org/abs/2301.08281
- PDF: https://arxiv.org/pdf/2301.08281
- Publisher page: https://iopscience.iop.org/article/10.1088/2634-4386/ad6f3b

## 一句话理解

ETLP 可以理解为：**把三因子学习规则做成事件驱动、局部、硬件友好的在线学习算法。**

它不是最宏大的智能体理论，但对你很重要，因为它把“局部学习规则”推进到了工程可实现层面。

## 研究者使用定位

本文把 ETLP 当成“硬件约束下的学习规则设计样板”，而不是单篇论文复述：

```text
入门：理解 event-based learning 为什么不能全量扫描突触
掌握：写出 pre trace × post factor × teaching signal
熟练：能估算每个突触/神经元需要保存多少状态
精通：能把 e-prop 的理论 trace 改写成 neuromorphic-friendly 更新
```


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

## 8. 研究者检查点

跨领域使用 ETLP 时，不必先纠结所有硬件细节，优先抓：

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

## 10. 最小研究原型

不要从大任务开始，先建立最小可证伪原型：

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

## 11. 研究者压缩模型：把三因子学习做成事件驱动硬件规则

ETLP 的问题意识很工程化：

```text
如果我们想在 neuromorphic hardware 上在线学习
就不能每个时间步扫描全网络
也不能保存完整计算图
更不能依赖全局反向传播
```

所以它把学习规则限制在事件驱动条件下：

```text
只有 spike/event 发生时，才更新相关 trace
只有教学/调制信号到来时，才把 trace 写入权重
```

它的核心形式仍然是三因子：

$$
\Delta w_{ij} = \eta \bar x_i f(V_j) T_j
$$

其中 $\bar x_i$ 是突触前 trace，$f(V_j)$ 是突触后膜电位相关因子，$T_j$ 是教学信号。

其中：

```text
pre_trace_i:
  输入神经元 i 最近是否发放过 spike

post_factor_j:
  突触后神经元 j 的膜电位、输出状态或可塑性函数

teaching_signal_j:
  外部教师/标签/误差给神经元 j 的方向信号
```

和 e-prop 相比，它不强调从 BPTT 推导一个精确近似，而强调：

```text
规则能否在事件流中本地计算？
每个突触要存多少状态？
是否适合低功耗硬件？
```

## 12. ETLP 的一次更新过程

你可以把一次 ETLP 更新想成下面这个过程。

```text
1. 输入 spike 到来
   更新该输入通道的 pre trace

2. 网络前向运行
   突触电流改变膜电位
   神经元可能发放输出 spike

3. 教学信号到来
   比如目标类别、期望输出、错误方向

4. 只更新最近相关的连接
   delta_w = learning_rate * pre_trace * post_factor * teaching_signal

5. 做约束
   权重裁剪、归一化、符号限制或活动稳态
```

用更接近实现的伪代码：

```text
on_input_spike(i, t):
    pre_trace[i] = pre_trace[i] + 1

on_time_decay():
    pre_trace *= decay

on_post_update(j):
    post_factor[j] = f(membrane_voltage[j], output_spike[j])

on_teacher_signal(j, signal):
    for i in recently_active_inputs:
        W[j, i] += eta * pre_trace[i] * post_factor[j] * signal[j]
        W[j, i] = constrain(W[j, i])
```

关键点是：

```text
recently_active_inputs
```

而不是全部输入。否则它就失去了事件驱动优势。

## 13. 它和 e-prop 的公式差异

可以这样对照：

```text
e-prop:
  eligibility trace 来自 recurrent 神经元状态对权重的导数
  更像 BPTT 的在线分解

ETLP:
  eligibility-like trace 来自事件驱动 pre trace + post membrane factor
  更像硬件可实现的三因子规则
```

也就是说：

```text
e-prop 问：这个局部规则能否近似真实梯度？
ETLP 问：这个局部规则能否在事件硬件上高效运行？
```

你的项目可以把两者合并：

```text
用 e-prop 提供理论基线
用 ETLP 提供实现约束
设计一个既有资格迹又事件稀疏的更新规则
```

## 14. post_factor 为什么用膜电位很重要

只看 post spike 太粗糙，因为 spike 是 0/1 事件。

膜电位提供更细的局部信息：

```text
膜电位远低于阈值：
  当前输入影响很小，更新可弱一些

膜电位接近阈值：
  当前输入可能决定是否发放，更新应更强

膜电位已经强烈超过阈值：
  可能需要饱和或归一化，避免权重爆炸
```

这和 surrogate derivative 的直觉相似：神经元越接近阈值，输入对输出 spike 的影响越敏感。

## 15. 实验结论与可迁移判断

ETLP 类工作的实验通常要看三件事，而不是只看最终准确率。

```text
1. 在线学习效果
   数据流到来时是否边推理边更新？

2. 事件驱动效率
   是否只在事件发生时计算和更新？
   是否减少乘加和内存访问？

3. 硬件友好性
   每个突触/神经元需要多少额外状态？
   教学信号是否可以低维广播？
```

如果一篇 ETLP 实验在分类任务上准确率略低于 surrogate gradient，也不代表失败。它真正想证明的是：

```text
在更严格的局部性和硬件约束下
仍然可以完成在线学习
并且计算/存储模式更适合 neuromorphic 系统
```

## 16. 最小研究原型

从最小可证伪版本开始：

```text
任务：在线 event stream 二分类或 N-MNIST 子集
网络：input spikes -> LIF hidden/output
局部状态：pre_trace, membrane_voltage, output_state
第三因子：target - output
约束：weight clipping + firing-rate homeostasis
```

更新公式：

$$
\Delta w = \eta \cdot \text{pre\_trace} \cdot f(V_{post}) \cdot \text{teaching\_signal}
$$

对比：

```text
只训练 readout
STDP
reward/teacher-modulated STDP
ETLP-like rule
surrogate-gradient SNN
```

评价不要只看准确率，还要记录：

```text
每个样本触发多少权重更新
每个突触需要多少状态
在线适应速度
分布漂移后的恢复速度
是否出现全网沉默或爆发
```

## 17. 对你路线的真正价值

ETLP 最值得借鉴的不是某个固定公式，而是设计纪律：

```text
先规定每个突触和神经元允许保存什么局部状态
再设计学习规则
最后才问任务效果
```

这能防止你的算法不知不觉退化成：

```text
局部学习的名义
全局反传的实现
```

