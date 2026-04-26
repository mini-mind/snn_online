# 三因子学习规则：局部可塑性的主干框架

## 资料

- Title: A Review of Three-Factor Learning Rules in Spiking Neural Networks
- arXiv: https://arxiv.org/abs/2504.05341
- PDF: https://arxiv.org/pdf/2504.05341

## 一句话理解

三因子学习规则说的是：**突触更新不能只看突触前和突触后是否一起活动，还要看第三个调制信号告诉这次活动是否有价值。**

通用形式：

```text
delta_w = local_pre_post_trace * modulation_signal
```

它是理解 e-prop、ETLP、奖励调制 STDP、神经调质学习和很多 biologically plausible learning 的共同入口。

## 1. 为什么两因子规则不够

最经典的想法是 Hebbian learning：

```text
一起激活的神经元连接增强
```

STDP 则把时间差引入进来：

```text
pre spike 先于 post spike -> 可能增强
post spike 先于 pre spike -> 可能减弱
```

这类规则很局部，但问题是：它们不知道任务目标。

举例：

```text
一个动作导致奖励
另一个动作导致惩罚
```

两个动作内部都可能有大量 pre/post 共激活。只靠共激活，突触不知道哪种共激活应该保留，哪种应该抑制。

所以需要第三因子。

## 2. 第三因子是什么

第三因子是把“局部活动”变成“有意义学习”的调制信号。

常见第三因子：

- reward prediction error：实际奖励比预期好还是差。
- supervised error：输出和目标之间的差。
- prediction error：预测的下一状态和真实下一状态的差。
- novelty：当前状态是否新奇。
- attention：当前信息是否被关注。
- homeostasis：网络是否过度兴奋或过度沉默。
- safety/risk：某个动作是否触发约束。

在工程上，第三因子不必只有一个。你可以设计多个调制通道。

## 3. 三因子规则的最小形式

最小形式：

```text
e_ij(t) = local_trace(pre_i, post_j, state_j)
delta_w_ij(t) = eta * e_ij(t) * M(t)
```

其中：

- `e_ij` 是局部资格迹。
- `M` 是调制信号。
- `eta` 是学习率。

如果 `M > 0`，最近参与的突触被强化。

如果 `M < 0`，最近参与的突触被削弱。

如果 `M = 0`，只记录活动，不改变长期权重。

## 4. 从深度学习视角理解

反向传播给每个参数一个精确或近似梯度：

```text
w = w - eta * dLoss/dw
```

三因子规则不直接计算 `dLoss/dw`，而是用：

```text
dLoss/dw 的替代估计 ~= eligibility * modulation
```

这不是简单地“更生物”，而是换了一种计算约束：

```text
每个突触只存局部 trace
全局系统只广播少量调制信号
```

它牺牲一部分梯度精确性，换取在线性、局部性和硬件友好性。

## 5. 典型算法族

### Reward-modulated STDP

```text
STDP trace 记录 pre/post 时间关系
奖励预测误差到来时调制 trace
```

适合简单强化学习，但长程信用分配和复杂任务上容易不稳定。

### e-prop

```text
eligibility trace 来自 recurrent SNN 的状态导数
learning signal 来自任务误差
```

数学上更接近 BPTT 梯度近似。

### ETLP

```text
pre spike trace + post membrane voltage + teaching signal
```

更偏事件驱动硬件实现。

### Predictive coding style rules

```text
局部活动 trace * 预测误差
```

适合自监督世界模型和认知地图。

## 6. 对你的项目最重要的设计问题

你真正要设计的不是一个“脑区名称系统”，而是下面几个变量：

```text
local_trace:
  记录什么局部活动？
  衰减时间多长？
  是否多时间尺度？

modulation:
  来自奖励、预测误差、风险、内稳态还是 LLM 教师？
  是全局广播还是局部区域广播？
  多个调制信号如何合成？

stability:
  如何防止权重爆炸？
  如何保持兴奋/抑制平衡？
  如何防止所有神经元沉默？

plasticity gating:
  什么情况下允许学习？
  什么情况下冻结？
  什么情况下快速学习，什么情况下慢速固化？
```

## 7. 建议的工程抽象

可以把学习规则写成接口：

```text
class Synapse:
    weight
    eligibility_fast
    eligibility_slow

def on_pre_spike(synapse, t):
    update_pre_trace()

def on_post_state(synapse, membrane, spike):
    update_eligibility()

def apply_modulation(synapse, reward_error, pred_error, homeostasis):
    modulation = combine(...)
    weight += lr * eligibility * modulation
    weight = normalize_or_clip(weight)
```

关键是把“局部 trace”和“调制信号”分离。这样后续可以替换 modulation 设计，而不用重写整个 SNN。

## 8. 常见陷阱

### 陷阱 1：把第三因子只理解成奖励

奖励太稀疏。真正可用的系统需要预测误差、新奇性、不确定性、内稳态等信号。

### 陷阱 2：调制信号太全局

如果一个全局奖励同时更新所有突触，容易污染无关连接。需要区域化、门控或 credit assignment。

### 陷阱 3：只增强不归一

局部学习规则很容易导致权重爆炸、神经元全体兴奋或全体沉默。必须有归一化、权重约束和活动稳态机制。

### 陷阱 4：任务太简单

只在 MNIST 或简单 spike 分类上有效，不代表能成为 agent 学习算法。必须测试时序、延迟奖励和持续学习。

## 9. 和四阶段路线的关系

- 阶段 1：三因子规则是底层学习规则主干。
- 阶段 2：第三因子可换成预测误差，支持世界模型。
- 阶段 3：第三因子可接奖励预测误差和动作价值误差。
- 阶段 4：LLM 可以提供教学信号、解释信号或任务上下文，但不应替代 SNN 自己的局部学习。

## 10. 你应该从综述中提取什么

不要只看术语分类。重点建立一张表：

```text
算法名
局部 trace 用了哪些变量
第三因子是什么
是否在线
是否需要反向传播
是否适合 recurrent SNN
是否做过 RL / continual learning
是否适合硬件
```

这张表会直接变成你后续设计新规则时的对照实验基线。

