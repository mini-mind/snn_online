# S-TLLR: 时间局部脉冲学习路线

## 资料

- Title: STDP-inspired temporal local learning rule for training spiking neural networks
- arXiv: https://arxiv.org/abs/2503.06126

## 一句话理解

S-TLLR 走的是“时间上更像 STDP，但又更适合训练”的路线。

## 研究者使用定位

本文把 S-TLLR 当成“介于 STDP 和更现代局部学习之间”的桥梁：

```text
入门：知道它为什么仍然强调 temporal locality
掌握：知道它为什么不是原始 STDP 的直接复刻
熟练：知道它适合做时序任务对照
精通：知道它和 ETLP、TESS 的边界在哪里
```

## 1. 它想解决什么

原始 STDP 很局部，但太朴素：

```text
只看 pre/post 时序，不一定知道任务目标
```

S-TLLR 想保留“时间局部”的味道，同时让学习规则更适合实际训练。

## 2. 核心思想

它的核心直觉是：

```text
把时间相关的局部活动做成更可训练的规则
```

你可以把它看成：

- 更训练友好的 temporal local rule
- 介于 STDP 和现代局部学习之间

## 3. 它和 ETLP / TESS 的关系

ETLP 更偏事件驱动和硬件友好。

TESS 更偏时间局部 + 空间局部的统一。

S-TLLR 更像把 STDP 这条老路重新整理得更适合训练。

所以它们是邻居，不是同一个东西。

## 4. 为什么它值得单独成章

因为它代表一类很现实的路线：

- 不想直接上 BPTT
- 不想一下子变成复杂系统
- 但也不满足于原始 STDP 的粗糙

这类路线对理解“局部学习到底能走多远”很重要。

## 5. 对本项目的意义

S-TLLR 对我们是一个温和对照：

- 如果 `three_factor` 太粗
- 如果 e-prop 太重
- 那么像 S-TLLR 这样仍然保留时间局部味道的路线就值得比较

一句话总结：S-TLLR 是“更可训练的时间局部学习”，不是单纯 STDP 改名。
