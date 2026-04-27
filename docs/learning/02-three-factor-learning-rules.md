# 三因子学习规则：局部可塑性的主干框架

## 资料

- Title: A Review of Three-Factor Learning Rules in Spiking Neural Networks
- arXiv: https://arxiv.org/abs/2504.05341
- PDF: https://arxiv.org/pdf/2504.05341

## 一句话理解

三因子学习规则说的是：**突触更新不能只看突触前和突触后是否一起活动，还要看第三个调制信号告诉这次活动是否有价值。**

通用形式：

$$
\Delta w = e_{local} \cdot M
$$

也就是：局部 pre/post trace 乘以调制信号。

它是理解 e-prop、ETLP、奖励调制 STDP、神经调质学习和很多 biologically plausible learning 的共同入口。

## 研究者使用定位

本文不是术语综述笔记，而是把三因子规则当成一个跨领域设计模式：

```text
入门：把 Hebbian/STDP 看成“无任务目标”的局部规则
掌握：把 eligibility trace 和 modulation signal 解耦
熟练：能为监督、预测、奖励、稳态分别设计第三因子
精通：能用同一接口比较 e-prop、ETLP、R-STDP 和自监督预测学习
```


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

$$
e_{ij}(t) = \operatorname{local\_trace}(\text{pre}_i, \text{post}_j, h_j)
$$

$$
\Delta w_{ij}(t) = \eta M(t) e_{ij}(t)
$$

其中：

- `e_ij` 是局部资格迹。
- `M` 是调制信号。
- `eta` 是学习率。

如果 `M > 0`，最近参与的突触被强化。

如果 `M < 0`，最近参与的突触被削弱。

如果 `M = 0`，只记录活动，不改变长期权重。

## 4. 从深度学习视角理解

反向传播给每个参数一个精确或近似梯度：

$$
w \leftarrow w - \eta \frac{\partial \mathcal{L}}{\partial w}
$$

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

## 10. 从综述中提取什么

不需要追术语谱系，重点建立一张可迁移方法表：

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

## 11. 研究者压缩模型：三因子规则不是一个算法，而是一类算法

这篇综述的价值不在于提出一个新公式，而是把很多看起来不同的 SNN 学习规则统一到同一个框架里：

```text
突触更新 = 局部资格迹 × 第三因子调制
```

最通用形式可以写成：

$$
\Delta w_{ij}(t) = \eta M_j(t) e_{ij}(t)
$$

其中：

```text
e_ij(t):
  由突触前神经元 i 和突触后神经元 j 的局部活动产生
  例如 pre spike trace、post spike trace、膜电位、阈值状态

M_j(t):
  第三因子，告诉这次局部活动是否应该被强化或削弱
  可以是奖励误差、监督误差、预测误差、注意、神经调质或稳态信号
```

如果没有第三因子，规则退化成 Hebbian / STDP：

```text
只要 pre 和 post 时间上相关，就改权重
```

加入第三因子后，规则变成：

```text
pre/post 相关只是“有资格”
真正是否学习，要等任务/奖励/预测误差信号确认
```

这就是为什么三因子规则是 e-prop、ETLP、reward-modulated STDP 和很多 neuromorphic learning 的共同语言。

## 12. 综述里的核心分类

你可以把综述里的方法按“第三因子从哪里来”分类。

```text
1. 奖励调制型
   第三因子 = reward prediction error
   用于强化学习、动作选择、探索-利用

2. 监督误差型
   第三因子 = target - output
   用于分类、序列预测、teacher forcing

3. 预测误差型
   第三因子 = actual_next_state - predicted_next_state
   用于 world model、认知地图、自监督学习

4. 神经调质型
   第三因子 = dopamine / acetylcholine / serotonin-like signal
   生物解释更强，但工程上仍可抽象成 modulation channel

5. 稳态调节型
   第三因子 = network activity too high / too low
   用于防止全网兴奋、全网沉默或权重爆炸
```

对你的项目来说，最重要的是不要把第三因子只等同于奖励。一个更像智能体的系统至少应该有：

```text
reward_error:
  哪些行为带来价值

prediction_error:
  哪些内部世界模型预测错了

novelty / uncertainty:
  哪些经验值得学习更多

homeostasis:
  网络活动是否稳定

teacher_signal:
  LLM 或外部教师是否提供高层纠错
```

## 13. 资格迹的跨领域理解

资格迹不是梯度本身，而是一个“等待被第三因子确认的局部候选信用”。

```text
pre spike 刚发生
post 神经元状态被影响
=> 突触 i -> j 留下一段短期痕迹
```

如果很快来了正向调制：

$$
M(t) > 0 \Rightarrow \Delta w_{ij} > 0
$$

最近相关突触被增强。

如果来了负向调制：

$$
M(t) < 0 \Rightarrow \Delta w_{ij} < 0
$$

最近相关突触被削弱。

如果没有调制：

$$
M(t) \approx 0
$$

痕迹逐渐衰减，长期权重不变。

这解决了一个关键问题：奖励或错误往往延迟到来，但突触活动发生在更早时刻。资格迹把这段时间桥接起来。

## 14. 和反向传播的真正差别

反向传播要做的是：

```text
为每个参数计算尽可能准确的 dLoss/dw
```

三因子规则做的是：

```text
让每个突触自己记录局部“我可能有关”
再由少量第三因子广播“这件事好/坏/错/新奇”
```

它不是免费午餐。代价是：

```text
梯度更粗糙
信用分配更不精确
需要调制信号设计
容易受无关突触污染
```

优势是：

```text
在线
局部
事件驱动
适合 neuromorphic hardware
更接近生物突触可塑性
```

所以三因子规则不是“比 BP 更强”，而是换了约束条件：在不能使用全局反传时，仍然保留任务驱动学习能力。

## 15. 综述结论与可迁移判断

作为综述，它本身不是一篇统一实验工作。使用它时应把结论整理成一张方法地图：

```text
reward-modulated STDP:
  在简单奖励任务可行
  但奖励稀疏时信用分配困难

e-prop:
  更接近 BPTT 梯度分解
  适合 recurrent SNN 和延迟时序任务

surrogate-gradient + local approximation:
  性能通常更好
  但局部性/生物合理性较弱

predictive / self-supervised three-factor rules:
  更适合 world model
  但工程验证通常比分类任务复杂

neuromorphic hardware rules:
  计算和存储更现实
  但算法表达能力受硬件限制
```

重点不是背论文谱系，而是为你的路线建立筛选标准：

```text
这条规则的局部 trace 是什么？
第三因子是什么？
能否在线？
是否需要全局误差反传？
是否能处理延迟奖励/延迟标签？
是否测试持续学习和遗忘？
```

## 16. 你可以直接采用的模板

以后评估任何 SNN 学习规则，都填这个表：

```text
方法名：
学习目标：分类 / 奖励 / 预测 / 控制 / 持续学习
局部变量：pre spike, post spike, membrane, threshold, trace
第三因子：reward, error, prediction error, novelty, teacher signal
更新公式：Delta w = ?
是否在线：是 / 否
是否局部：突触局部 / 神经元局部 / 区域广播 / 全局
是否事件驱动：是 / 否
实验任务：
主要效果：
主要风险：
```

这样无需逐篇精读，也能判断一个方法是否值得纳入自己的研究系统。

