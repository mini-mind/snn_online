# Online Local Learning for SNN Agents

本项目目标是探索一种可以对标反向传播的新学习算法：它应当支持在线、局部、可持续的权重更新，并能在交互过程中完成表征学习、动作选择和长期适应。

## 核心判断

当前 LLM / Transformer 可以被看作一种训练后相对静态的高级认知模块。它们通过预训练和后训练获得大量知识与行为偏好，但在普通推理过程中，本体权重通常不会实时更新。上下文学习、RAG、工具调用、外部记忆和 LoRA 微调可以带来适应性，但这些更接近“被刺激后激活已有能力”或外部状态更新，不等同于生物突触式的持续本体学习。

本项目采用一个工程化假设：

- LLM 可以作为 bootstrap 脚手架，用来模拟环境、教师、评价器、语义解释器或静态认知模块。
- 真正被研究的对象不是 LLM，而是一个可以在线更新的 SNN / 类 SNN 学习体。
- 研究重点不是完整仿真大脑结构，而是寻找可工作的局部学习规则。

## 目标学习规则

理想算法应满足：

- 在线更新：交互过程中持续调整权重。
- 局部可计算：突触更新主要依赖突触前活动、突触后活动、局部状态和少量调制信号。
- 可持续学习：避免快速灾难性遗忘。
- 可用于循环网络：能处理时序、记忆和延迟奖励。
- 可支持动作选择：不仅能分类或预测，还能决定“现在做什么”。
- 可扩展：最终能从小型任务推进到更复杂的 agent 环境。

一个优先研究的形式是三因子学习规则：

```text
delta_w_ij = learning_rate * eligibility_trace_ij * modulation_signal
```

其中：

- `eligibility_trace_ij` 来自突触前 spike、突触后 spike、膜电位、时间差等局部信息。
- `modulation_signal` 可以来自奖励预测误差、预测误差、新奇性、不确定性、注意或内稳态压力。

## 实验路线

项目按四个阶段推进：

1. 小型 recurrent SNN + 在线局部学习
   验证 eligibility trace、三因子规则、稳定性约束和基础时序学习。

2. 预测学习 / 世界模型
   引入下一状态预测、异常检测、局部预测误差和简单 planning。

3. 动作选择 / 门控 / RL
   加入基底节式门控，让系统学习选择动作、抑制动作、切换目标。

4. LLM bootstrap 环境
   使用 LLM 作为静态脚手架，为 SNN 学习体提供任务、解释、反馈和复杂语义环境，但避免把研究退化为单纯蒸馏 LLM。

## 与生物脑的差距和迭代指引

当前原型不是“仿真大脑”，而是一个最小脑式在线学习 agent 骨架：有 recurrent spiking state、局部调制更新、世界模型、奖励调制和动作闭环。后续迭代应优先缩小下表中的关键差距。

| 维度 | 当前原型 | 生物脑 | 后续迭代指引 |
|---|---|---|---|
| 在线学习 | 交互中更新部分权重 | 持续、本体、多区域可塑 | 保持 online-first，避免退回离线 batch 训练 |
| 局部更新 | trace × modulation 的简化三因子规则 | 突触局部变量 + 多神经调质 | 增加区域化 modulation、plasticity gate 和多时间尺度 trace |
| Spike 动力学 | 小型 LIF-like R-SNN | 丰富神经元类型、振荡和回路动力学 | 加入兴奋/抑制约束、活动稳态和多时间常数神经元 |
| 预测误差 | world model / cognitive map toy | 多模态、多层级预测 | 把 prediction error 拆成感觉、动作后果和不确定性通道 |
| 奖励调制 | TD error / reward shaping | 多巴胺等调质系统与动机状态耦合 | 区分 reward、novelty、risk、homeostasis，避免单一全局奖励污染 |
| 世界模型 | gridworld / point robot 的低维模型 | 多尺度、可组合、可想象的环境模型 | 加入 replay/dream、模型不确定性和反事实 rollout |
| 动作控制 | 离散动作 point robot | 连续、多关节、多反馈回路 | 引入动作候选生成、基底节式 gate 和连续控制环境 |
| 记忆系统 | 权重、trace、短期 recurrent state | 海马快速绑定、皮层慢巩固、情景/语义记忆 | 加入 episodic buffer、优先 replay 和慢速 consolidation |
| 稳定性 | 可跑通 toy，但仍需调参 | 强鲁棒、抗噪声、抗遗忘 | 系统评估分布漂移、任务切换和灾难性遗忘 |
| 可扩展性 | 纯 Python 小实验 | 大规模稀疏并行系统 | 在保留可解释性的前提下推进稀疏更新和模块化接口 |

阶段性目标不是追求“像脑一样聪明”，而是逐步验证：局部学习规则能否驱动一个 agent 从预测世界走向稳定行动。

## 关键风险

- 奖励自循环：如果系统自己定义奖励再用奖励训练自己，容易 reward hacking。
- LLM 职责过重：LLM 适合做语义脚手架，不适合直接承担所有情绪、奖励、记忆和控制回路。
- 缺少动作门控：没有基底节式 arbitration 时，多个子模块会给建议，但没有稳定的行动选择机制。
- 只做仿生命名：脑区类比应服务于工程分工，不能替代可验证的算法假设。
- 评估不清：必须用客观任务测试在线学习、局部性、抗遗忘、长程信用分配和样本效率。

## 项目入口

- [Learning docs](docs/learning/INDEX.md)：跨领域研究手册与论文方法压缩。
- [Minimal experiments](src/README.md)：可直接运行的纯 Python 原型。

## 快速运行

```bash
python src/etlp_continuous_toy.py
python src/cognitive_map_etlp_toy.py
python src/rsnn_point_robot_toy.py
```
