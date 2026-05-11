# 现代 SNN 局部学习路径总览

这页不是单篇论文笔记，而是给“程序员视角的路线图”。目标很简单：把最近几条前沿路线分清楚，知道它们是同一条线上的不同版本，还是已经换了另一条路。

## 先说结论

这些路线共享一个大约束：

```text
不想把时间完整展开再做 BPTT
不想让每个权重依赖全局反传梯度
希望更新尽量局部、在线、可扩展
```

但它们的侧重点不同：

- `e-prop` 家族：最像“在线局部版的 recurrent 学习”。
- `event-driven e-prop`：还是 e-prop，但更强调事件驱动和大规模稀疏网络。
- `TESS`：把时间局部和空间局部一起做强，不只是 e-prop 的小改。
- `EchoSpike`：更偏预测学习 / 自监督局部规则。
- `BrainTrace / pp-prop`：更像一套把局部在线学习工程化的系统。
- `DECOLLE`：把本地监督铺到每一层，属于深层局部学习路线。
- `S-TLLR`：保留 STDP 的时间局部味道，但更面向训练与可控优化。

## 1. e-prop 家族：同一条母线

### 资料

- [e-prop](https://www.nature.com/articles/s41467-020-17236-y)
- [event-driven eligibility propagation](https://arxiv.org/abs/2511.21674)

### 一句话理解

先让每条突触自己记一笔“我最近是否可能有责任”，等全局学习信号来了，再把这笔账结掉。

### 共同思想

```text
权重更新 = eligibility trace * learning signal
```

人话就是：

- eligibility trace 负责“记录最近发生了什么”。
- learning signal 负责“告诉这次该加还是该减”。

### event-driven 版本改了什么

它不是换了思路，而是换了执行方式：

```text
从按时间步全量更新
变成按 spike / event 触发更新
```

适合：

- 稀疏网络
- 大规模仿真
- 神经形态硬件

不适合：

- 直接拿来当我们现在的第一步主线
- 还没把基础 e-prop 复现稳的时候继续加复杂度

## 2. TESS：时间和空间都要局部

### 资料

- [TESS: A Scalable Temporally and Spatially Local Learning Rule for Spiking Neural Networks](https://arxiv.org/abs/2502.01837)

### 一句话理解

它不是只补 e-prop 的一个小洞，而是把“局部学习”从时间维和空间维一起做强。

### 它在解决什么

e-prop 主要关心：

```text
时间上的 credit assignment
```

TESS 还想同时处理：

```text
空间上的 credit assignment
```

也就是：

- 哪个时间点的活动重要
- 哪个神经元/连接在结构上重要

### 直白理解

可以把它看成：

> 不只给连接记“时间账”，还给每个局部结构记“空间账”。

### 为什么它值得单独看

因为它不是简单的 e-prop 替代品，而是另一种更强的局部约束设计。  
它和我们现在的 `tess_like` 有味道上的相似，但不应当直接当成同一个东西。

## 3. EchoSpike：预测学习，不是奖励优先

### 资料

- [EchoSpike Predictive Plasticity: An Online Local Learning Rule for Spiking Neural Networks](https://arxiv.org/abs/2405.13976)

### 一句话理解

它更像：

```text
先学会预测，再从预测误差里自己调整内部表示
```

### 它和 e-prop 的区别

e-prop 主要问：

```text
这个突触对任务结果有没有责任？
```

EchoSpike 更像在问：

```text
这个网络能不能自己把未来状态预测好？
```

所以它更偏：

- 自监督
- 表征学习
- 预测学习

### 对本项目的意义

它不适合作为第一阶段控制主线，但适合作为：

- quiet / dormant 状态下是否自然形成任务痕迹的对照
- replay / dreaming 是否会涌现的观察对象

## 4. BrainTrace / pp-prop：更像系统，不只是规则

### 资料

- [Model-agnostic linear-memory online learning in spiking neural networks](https://www.nature.com/articles/s41467-026-68453-w)

### 一句话理解

它更像一个在线学习基础设施：

```text
你写模型，系统帮你自动生成线性内存的在线学习路径
```

### 它的重点

- 模型无关
- 线性内存
- 在线学习
- 自动化生成 trace / 更新结构

### 它和前面几条的区别

它不是单纯“再发明一个新规则”，而是试图把很多局部在线学习机制工程化。

对我们项目来说，它的价值主要是：

- 看“在线局部学习”可以被工程系统做到什么程度
- 以后如果要做更严肃的实现，可以借它的系统思路

但它不是我们现在要直接照搬的第一条路线。

## 5. 在本仓里怎么用这些路线

不要把这些东西揉成一个主线。更稳的顺序是：

```text
three_factor
-> tess_like
-> recurrent_delay_line
-> reward-based e-prop comparator
-> TESS / EchoSpike / BrainTrace / DECOLLE / S-TLLR as separate routes
```

对应到实验策略就是：

- 先做稳定消融，确认每一步有没有真实增益
- 再复现一条成熟先进对照，优先考虑 reward-based e-prop
- 再把 TESS、EchoSpike、BrainTrace、DECOLLE、S-TLLR 作为独立分支
- replay / dreaming 只观察是否自然涌现，不手工设计

## 6. 对小白的最简记忆法

你可以把这些路线记成四类：

```text
e-prop：记责任账
TESS：记时间账 + 空间账
EchoSpike：学预测
BrainTrace：把这些东西工程化
DECOLLE：每层自己学
S-TLLR：训练友好的时间局部规则
```

这四类共享“局部、在线、少全局依赖”，但不是同一个路线的名字换法。
