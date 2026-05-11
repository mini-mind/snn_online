# TESS: 时间与空间都局部的 SNN 学习路线

## 资料

- Title: TESS: A Scalable Temporally and Spatially Local Learning Rule for Spiking Neural Networks
- arXiv: https://arxiv.org/abs/2502.01837

## 一句话理解

TESS 不是只补 e-prop 的一个洞，而是把“局部学习”同时做成时间局部和空间局部。

## 研究者使用定位

本文把 TESS 当成“更强的局部学习对照”，而不是 e-prop 的同义词：

```text
入门：知道它为什么同时谈 temporal 和 spatial credit assignment
掌握：看懂它如何只用局部信号完成更新
熟练：能把它和 e-prop / ETLP / 三因子规则区分开
精通：能判断它更适合做分类、控制还是硬件实现对照
```

## 1. 它在补什么问题

e-prop 主要在处理：

```text
时间上，哪个突触该为未来误差负责
```

TESS 还要处理：

```text
空间上，哪个局部结构、哪个神经元群更该负责
```

所以它不是只延长 trace，而是把学习信号组织得更结构化。

## 2. 核心思想

TESS 的关键词是：

- locally available signals
- temporal locality
- spatial locality
- linear scaling in memory and compute

直白说就是：

> 不靠全局反传，不靠长时间展开，让每个神经元只用自己附近能看到的信息更新自己。

## 3. 它和 e-prop 的区别

e-prop 更像：

```text
在线版的 recurrent 信用分配
```

TESS 更像：

```text
把信用分配拆成时间局部 + 空间局部两层
```

这意味着它不只是“又一种 eligibility trace”，而是更系统地处理局部学习的组织方式。

## 4. 为什么它适合做独立对照

它适合当对照的原因不是“更像脑”，而是：

- 定义清楚
- 局部约束明确
- 复杂度目标明确
- 和 BPTT 的分界清楚

所以如果我们要做“先进但仍符合约束”的对照，TESS 是一条很自然的候选。

## 5. 对本项目的意义

对当前仓库来说，TESS 最重要的不是直接照搬结果，而是给出一个标准：

- 我们的 `tess_like` 到底只是味道接近，还是足够像
- 多时间尺度 trace 是否真能带来稳定收益
- 在部分可观测控制里，局部学习到底能推进到哪一步

一句话总结：TESS 是“局部学习路线”的强化版对照，而不是 e-prop 的小注脚。
