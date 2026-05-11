# BrainTrace / pp-prop: 在线局部学习的工程化路线

## 资料

- Title: Model-agnostic linear-memory online learning in spiking neural networks
- Nature Communications, 2026
- Article: https://www.nature.com/articles/s41467-026-68453-w

## 一句话理解

BrainTrace 更像一个在线学习系统，而不只是一个学习规则：你给它模型，它帮你把在线局部学习做成可扩展实现。

## 研究者使用定位

本文把 BrainTrace 当成“把局部在线学习工程化”的代表：

```text
入门：知道它为什么强调 model-agnostic
掌握：知道它为什么强调 linear-memory
熟练：能看懂它如何服务在线学习而不是离线训练
精通：能判断它适合做系统参考，不适合直接当我们项目的第一实验
```

## 1. 它的关注点和前面几条不同

e-prop、TESS、EchoSpike 多半是在讨论：

```text
规则本身怎么写
```

BrainTrace 更像在讨论：

```text
怎么把这些规则组织成一个可运行、可扩展、可复用的在线学习系统
```

所以它的关键词常常不是“更聪明”，而是：

- 可扩展
- 低内存
- 在线
- 自动化

## 2. 为什么它强调 linear-memory

SNN 在线学习最常见的痛点之一是：

```text
时间一长，状态存储就爆
```

BrainTrace 想做的是把这个问题压到线性级别，让长序列训练不至于把内存开销炸穿。

人话就是：

> 不要把每一步历史都完整攒成一大坨。

## 3. 它和 e-prop 的关系

它不是 e-prop 的简单替身，而更像把 e-prop 这类思想系统化、工程化。

如果 e-prop 像：

```text
手写局部学习规则
```

那 BrainTrace 更像：

```text
把在线学习规则做成系统能力
```

## 4. 对本项目的意义

它给我们的启发主要是两点：

1. 以后如果局部学习规则越来越多，不能只靠手工堆实验入口。
2. 但现在的项目还不该过早系统化，应该先把单条路线的实验约束跑稳。

一句话总结：BrainTrace 更像工程顶层参考，而不是当前主线实现模板。
