# SNN Online Learning Research Index

整理日期：2026-04-26

## 总体定位

当前学界大致处在：

- 阶段 1：小型 recurrent SNN + 在线局部学习，已有扎实基础。
- 阶段 2：预测学习 / 世界模型 / 自监督预测，正在前沿推进。
- 阶段 3：SNN + 动作选择 / RL / 控制，有 demo，但很多仍依赖 surrogate gradient、BPTT、ANN-to-SNN 或混合 ANN 模块。
- 阶段 4：LLM 作为 bootstrap 脚手架训练在线 SNN 学习体，还没有形成成熟体系。

## 学习路径

0. [前置概念：从深度学习视角理解 SNN 在线学习](00-foundations.md)
1. [e-prop / 在线局部学习](01-eprop.md)
2. [三因子学习规则综述](02-three-factor-learning-rules.md)
3. [ETLP / 事件驱动三因子局部可塑性](03-etlp.md)
4. [局部预测学习 / 认知地图 / planning](04-cognitive-map-learner.md)
5. [Model-based SNN RL / dreaming](05-model-based-snn-rl.md)
6. [SNN Deep RL / robot control](06-spiking-q-learning-robot-control.md)
7. [SpikingBrain / 大规模类脉冲模型](07-spikingbrain.md)
8. [NSLLM / neuromorphic spiking LLM](08-nsllm.md)

## 核心资料

### 1. e-prop / 在线局部学习

- Title: A solution to the learning dilemma for recurrent networks of spiking neurons
- Authors: Guillaume Bellec et al.
- Venue: Nature Communications, 2020
- DOI: https://doi.org/10.1038/s41467-020-17236-y
- Article: https://www.nature.com/articles/s41467-020-17236-y
- Guide: [01-eprop.md](01-eprop.md)
- Relevance: recurrent SNN 的在线学习核心论文，用 eligibility traces + learning signals 近似替代 BPTT。

### 2. 三因子学习规则综述

- Title: A Review of Three-Factor Learning Rules in Spiking Neural Networks
- arXiv: https://arxiv.org/abs/2504.05341
- PDF: https://arxiv.org/pdf/2504.05341
- Guide: [02-three-factor-learning-rules.md](02-three-factor-learning-rules.md)
- Relevance: 适合作为三因子学习规则入口，关注 local eligibility trace + global/semiglobal modulatory signal。

### 3. ETLP / 事件驱动三因子局部可塑性

- Title: ETLP: Event-based Three-factor Local Plasticity for online learning with neuromorphic hardware
- arXiv: https://arxiv.org/abs/2301.08281
- PDF: https://arxiv.org/pdf/2301.08281
- Publisher page: https://iopscience.iop.org/article/10.1088/2634-4386/ad6f3b
- Guide: [03-etlp.md](03-etlp.md)
- Relevance: 面向在线学习和神经形态硬件的三因子局部规则，可作为工程实现参考。

### 4. 局部预测学习 / 认知地图 / planning

- Title: Local prediction-learning in high-dimensional spaces enables neural networks to plan
- Authors: Christoph Stoeckl et al.
- Venue: Nature Communications, 2024
- DOI: https://doi.org/10.1038/s41467-024-46586-0
- Article: https://www.nature.com/articles/s41467-024-46586-0
- Guide: [04-cognitive-map-learner.md](04-cognitive-map-learner.md)
- Relevance: 使用局部预测学习形成可 planning 的高维表示，接近“预测编码 + 世界模型”的阶段 2。

### 5. Model-based SNN RL / dreaming

- Title: Towards biologically plausible model-based reinforcement learning in recurrent spiking networks by dreaming new experiences
- Authors: Cristiano Capone et al.
- Venue: Scientific Reports, 2024
- DOI: https://doi.org/10.1038/s41598-024-65631-y
- Article: https://www.nature.com/articles/s41598-024-65631-y
- Guide: [05-model-based-snn-rl.md](05-model-based-snn-rl.md)
- Relevance: recurrent SNN 同时学习 world model 和 policy，并用 dream/simulated experience 增强训练。

### 6. SNN Deep RL / robot control

- Title: Exploring spiking neural networks for deep reinforcement learning in robotic tasks
- Venue: Scientific Reports, 2024
- DOI: https://doi.org/10.1038/s41598-024-77779-8
- Article: https://www.nature.com/articles/s41598-024-77779-8
- Guide: [06-spiking-q-learning-robot-control.md](06-spiking-q-learning-robot-control.md)
- Relevance: SNN + RL + 动作控制的代表性工程 demo，但需要检查是否满足“在线、局部、持续更新、不依赖反向传播”的严格要求。

### 7. SpikingBrain / 大规模类脉冲模型

- Title: SpikingBrain 1.0: A Brain-inspired Spiking Large Language Model
- arXiv: https://arxiv.org/abs/2509.05276
- PDF: https://arxiv.org/pdf/2509.05276
- Guide: [07-spikingbrain.md](07-spikingbrain.md)
- Relevance: 更偏大模型高效推理、稀疏激活、类脉冲架构，不等价于在线自更新 SNN 学习算法。

### 8. NSLLM / neuromorphic spiking LLM

- Title: NSLLM: Neuromorphic Spiking Large Language Models
- Venue: National Science Review
- Article: https://academic.oup.com/nsr/article/doi/10.1093/nsr/nwaf551/8365570
- Guide: [08-nsllm.md](08-nsllm.md)
- Relevance: 关注 neuromorphic / spike-based LLM 工程方向，可作为阶段 4 的旁支参考。

## 代码原型

详情见 [src README](../../src/README.md)。

- [ETLP continuous toy](../../src/experiments/etlp_continuous_toy.py)
- [Cognitive Map + ETLP toy](../../src/experiments/cognitive_map_etlp_toy.py)
- [R-SNN point robot closed loop](../../src/experiments/point_robot_closed_loop.py)
