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

## 功能性脑区架构

后续实验可以把被模拟对象定义为一个工程化的功能性脑区系统。这里的脑区命名用于划分职责和信号流，不作为严格神经科学复刻。

```text
外部观测 / 生理信号
  -> 丘脑式路由与价值调制
  -> 端脑/皮层式单层 recurrent 学习网络
  -> 动作意图
  -> 小脑式动作程序网络
  -> 环境动作
  -> 生理状态与外部状态变化
```

建议的职责划分：

- 端脑 / 皮层：主学习区，用单层 recurrent SNN 表示特征、状态、上下文、关联记忆、预测和简单推理。它承载 ETLP / CML / TESS-like 一类局部可塑性规则。
- 丘脑：路由、价值计算和调制中心。它接收外部特征、生理信号和皮层状态，输出 `valence`、`arousal`、`attention_gate`、`plasticity_scale` 等低维调制信号。
- 小脑：动作细节参数记忆与回放。它把高层动作意图展开为可执行的连续或离散动作流程，重点学习动作序列、误差修正和复用。
- 中脑 / 后脑：内稳态动力系统。它维护心跳、饥饿、疼痛、疲劳、能量等低维生理变量，并让这些变量反过来影响价值和动作偏置。

关键原则是把“喜恶”从单一 reward 拆开。更适合当前路线的形式是：

```text
physiology + task feedback + prediction error
  -> thalamic modulation
  -> plasticity gate / attention gate / action bias
```

这样同一个外部输入可以在不同内稳态下触发不同学习速度、注意分配和动作选择。它也能把部分可观测机器人任务从简单输入输出映射，推进到需要记忆、价值调制和动作程序复用的闭环任务。

## 关键风险

- 奖励自循环：如果系统自己定义奖励再用奖励训练自己，容易 reward hacking。
- LLM 职责过重：LLM 适合做语义脚手架，不适合直接承担所有情绪、奖励、记忆和控制回路。
- 缺少动作门控：没有基底节式 arbitration 时，多个子模块会给建议，但没有稳定的行动选择机制。
- 只做仿生命名：脑区类比应服务于工程分工，不能替代可验证的算法假设。
- 评估不清：必须用客观任务测试在线学习、局部性、抗遗忘、长程信用分配和样本效率。

## 项目定位

当前仓库是在线局部学习 SNN 的实验仓：

- `snn_online`：负责研究问题、任务设计、学习规则、实验编排、对照和报告。
- 本仓保持纯 Python 实验探索，不承担外部执行引擎或可视化系统适配。

## 项目入口

- [Learning docs](docs/learning/INDEX.md)：跨领域研究手册与论文方法压缩。
- [Experiment roadmap](docs/learning/ROADMAP.md)：后续实验推进、参数生物对照和环境难度矩阵。
- [Minimal experiments](src/README.md)：当前实验入口、目录边界、运行命令和指标说明。

## 当前代码边界

- `src/envs/`：最小环境定义。
- `src/models/`：学习器、读出头和局部可塑性规则。
- `src/experiments/`：实验入口脚本，负责组装配置并打印结果。

顶层文档只保留方向与边界；详细命令、输出字段和指标含义见 [src/README.md](src/README.md)。

## 快速运行

```bash
PYTHONPATH=src python src/experiments/etlp_continuous_toy.py
PYTHONPATH=src python src/experiments/cognitive_map_etlp_toy.py
PYTHONPATH=src python src/experiments/point_robot_closed_loop.py
PYTHONPATH=src python src/experiments/point_robot_closed_loop.py --plasticity-rule tess_like
PYTHONPATH=src python src/experiments/point_robot_closed_loop.py --plasticity-rule tess_like --delay-features
PYTHONPATH=src python src/experiments/point_robot_closed_loop.py --plasticity-rule tess_like --recurrent-delay-line
PYTHONPATH=src python src/experiments/compare_plasticity_rules.py
PYTHONPATH=src python src/experiments/compare_delay_features.py
PYTHONPATH=src python src/experiments/compare_lif_vs_izh.py
PYTHONPATH=src python src/experiments/compare_partial_observable_lif_vs_izh.py
```

## 当前落地顺序

1. `tess_like` 多时间尺度局部 trace
   当前已接入单层 recurrent SNN，作为 `three_factor` baseline 的直接对照。

第一阶段 benchmark 入口是 `src/experiments/compare_plasticity_rules.py`：它在固定 seed / 训练预算下比较 `three_factor` 与 `tess_like`，默认使用更依赖短期记忆的 `partial_goal_cue` 任务。
正式跑数时可加 `--output-jsonl` 把每个 seed/rule 结果与最终 summary 记录为 JSONL。

当前 5 seed 小基准显示：`tess_like` 在 `partial_goal_cue` 下略优于 `three_factor`，但在 `full` 观测下更弱。因此下一步只在部分可观测任务上推进短期记忆机制，不把它视为全局替代规则。

2. delay features
   已加入 `--delay-features`：在 RSNN 输出中拼接多时间尺度 spike delay trace，并用同一个调制信号局部更新 delay mixture。它默认关闭，不改变旧实验；打开后优先用于 `partial_goal_cue`。

当前 5 seed 小基准显示：在 `tess_like + partial_goal_cue` 下，delay feature 相对 plain RSNN 的 reward gain 约 `+1.166`，success gain 约 `+0.110`，但 wall time 约为 `1.61x`。

3. recurrent delay line
   已加入 `--recurrent-delay-line`：recurrent edge 可以读取过去若干步的 source spike，用来区别“额外 delay feature”和“连接本身的传输延迟”。

4. dreaming / replay
   在局部规则稳定后，再加入 model-based imagined experience，避免系统复杂度先失控。

## Next

下一步工程优先级是先降低 delay feature 的计算成本，再确认它是否仍然值得保留。只有在确认 delay 机制对当前部分可观测任务持续有用之后，才进入 replay / dreaming。

`runs/` 保持为忽略目录；使用 `--output-jsonl` 生成的逐 seed 结果和 summary 可以作为本地留档 artifact，不视为仓库内源码或基准真值。

后续所有机制推进都要同时记录两个视角：参数是否能对照生物脑中的时间常数、连接度或调制强度；环境是否明确标注测试维度和难度等级，避免只看单一平均分。
