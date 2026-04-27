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

$$
\Delta w_{ij}(t) = L_j(t) \, e_{ij}(t)
$$

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

$$
e_{ij}(t) = \lambda e_{ij}(t-1) + c_{ij}(t)
$$

其中 $\lambda$ 是衰减系数，$c_{ij}(t)$ 是当前局部贡献。

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

## 12. 原文核心方法速读：把 BPTT 拆成在线可算的两项

原文要解决的问题不是“怎么让 SNN 发 spike”，而是：

```text
recurrent SNN 的理想梯度本来要沿时间反传
但生物突触/在线硬件不可能保存完整计算图
所以能否把梯度改写成局部 trace × 学习信号？
```

对 recurrent 连接 `i -> j` 的权重 `W_ji`，原文把梯度写成近似形式：

$$
\frac{\partial E}{\partial W_{ji}} \approx \sum_t L_j^t e_{ji}^t
$$

其中：

```text
L_j^t:
  learning signal
  表示神经元 j 在时刻 t 应该朝哪个方向改变输出

e_ji^t:
  eligibility trace
  表示突触 i -> j 在时刻 t 对神经元 j 的输出有多大潜在责任
```

权重更新就是：

$$
\Delta W_{ji} = -\eta \sum_t L_j^t e_{ji}^t
$$

这句话是整篇论文的核心。你可以把 e-prop 看成：

```text
BPTT:
  等 loss 出现后，沿时间反向追溯所有状态

e-prop:
  每个突触在时间正向运行时保存自己的资格迹
  等学习信号到来时，直接调制这些资格迹
```

所以 e-prop 不是“错误后反思并补链”，而是“事前在线留下可被未来误差使用的局部信用痕迹”。

## 13. 原文公式怎么翻译成直觉

原文先定义神经元隐藏状态：

```text
h_j^t = 神经元 j 在 t 时刻的内部状态
```

对 LIF 神经元，`h_j^t` 主要是膜电位。对 ALIF / LSNN，`h_j^t` 还包括适应变量，比如阈值适应状态。

资格向量递推可以理解为：

```text
ε_ji^t =
  上一时刻资格向量经过神经元动力学传播
  + 当前权重 W_ji 对状态 h_j^t 的直接影响
```

写成公式骨架：

$$
\epsilon_{ji}^t = \frac{\partial h_j^t}{\partial h_j^{t-1}} \epsilon_{ji}^{t-1}
+ \frac{\partial h_j^t}{\partial W_{ji}}
$$

然后把“状态影响”变成“spike 影响”：

$$
e_{ji}^t = \frac{\partial z_j^t}{\partial h_j^t} \epsilon_{ji}^t
$$

这里 `z_j^t` 是 spike 输出。因为 spike 的阶跃函数不可导，原文和大多数 SNN 训练一样，用伪导数近似：

$$
\frac{\partial z_j^t}{\partial h_j^t} \approx \psi_j^t
$$

直觉是：

```text
突触前最近活跃
并且突触后神经元接近阈值
那么这个突触对当前 spike 更有资格负责
```

## 14. LIF 与 ALIF 的区别

对普通 LIF，资格迹主要来自突触前 spike 的低通滤波：

$$
\bar z_i^t = \lambda \bar z_i^{t-1} + z_i^t
$$

$$
e_{ji}^t \approx \psi_j^t \bar z_i^t
$$

它能处理短期时间依赖，但时间尺度有限。

ALIF / LSNN 加入适应变量：

```text
神经元发放越多
适应变量越大
有效阈值越高
```

这让神经元拥有更慢的内部状态，相当于给 recurrent SNN 加入一种 LSTM-like 记忆通道。原文实验里，带适应变量的 LSNN 通常比普通 LIF 更适合延迟线索、工作记忆和长时序信用分配任务。

## 15. learning signal 的三种版本

原文不是只给一个 learning signal，而是讨论了几种近似方式：

```text
symmetric e-prop:
  用输出权重的转置把误差反馈给隐藏神经元
  更接近反向传播，但生物合理性弱一些

random e-prop:
  用固定随机反馈权重把误差投影回隐藏神经元
  类似 feedback alignment，更局部、更生物合理

adaptive / broadcast-like variants:
  用更粗的广播信号近似神经元级 learning signal
  梯度更粗糙，但实现更简单
```

可以把它们理解成同一条轴上的折中：

```text
更精确的 learning signal
  -> 更接近 BPTT
  -> 更不像生物局部机制

更粗糙的 learning signal
  -> 更硬件友好/生物合理
  -> 梯度估计更噪声
```

## 16. 原文实验效果怎么读

实验的重点不是证明 e-prop 全面超过 BPTT，而是证明：

```text
在不完整时间反传的情况下
只靠 eligibility trace × learning signal
recurrent SNN 仍能学会需要时间记忆的任务
```

你读实验时只需要抓四类结果：

1. **延迟任务**：早期线索要在很久以后影响输出，用来测试时间信用分配。
2. **工作记忆任务**：网络需要在内部状态里保存信息，普通瞬时分类器做不好。
3. **真实序列任务**：例如语音、时序模式或类似连续输入，测试方法是否不只适合玩具例子。
4. **RL / 奖励调制任务**：把 supervised error 换成 reward-related signal，说明 e-prop 可接强化学习。

原文的主要结论可以概括为：

```text
e-prop 明显比纯 STDP 更能利用任务误差
比完整 BPTT 更在线、更局部、更硬件友好
但通常仍不如完整 BPTT 稳定和精确
LSNN/ALIF 的慢变量对长时序任务很关键
```

## 17. 如果你只复现一个最小版

建议不要一开始复现完整论文。最小复现版本：

```text
任务：delayed cue classification
网络：input -> recurrent LIF/ALIF -> readout
局部状态：pre_trace, membrane, pseudo_derivative, eligibility
学习信号：readout error 投影到 hidden neuron
对比：BPTT、random e-prop、只训练 readout、STDP
```

更新公式：

$$
\Delta W_{rec} = -\eta L_j^t e_{ji}^t
$$

如果这个任务能跑通，你就真正理解了 e-prop 的方法核心。
