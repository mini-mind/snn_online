# Minimal Experiments

本目录放纯 Python、少依赖、可直接阅读的研究原型。目标不是复刻论文工程，而是把关键学习规则压缩成可改、可证伪的最小闭环。

## Overview

| Script | Purpose | Core Signal | Run |
|---|---|---|---|
| `etlp_continuous_toy.py` | 连续输入 ETLP-like 分类 | teaching signal | `python src/etlp_continuous_toy.py` |
| `cognitive_map_etlp_toy.py` | 学 gridworld 转移图并规划 | prediction error | `python src/cognitive_map_etlp_toy.py` |
| `rsnn_point_robot_toy.py` | R-SNN 控制 2D point robot | prediction error + TD error | `python src/rsnn_point_robot_toy.py` |

## Shared Pattern

三个原型都围绕同一个研究假设：

```text
local trace / recurrent state
×
low-dimensional modulation signal
=>
online weight update
```

## Experiments

### ETLP Continuous Toy

核心更新：

$$
\Delta w_{ij} = \eta \cdot \bar{x}_i(t) \cdot f(V_j(t)) \cdot T_j(t)
$$

用途：观察连续输入如何形成 analog eligibility trace，以及教学信号如何调制局部 trace。

快速检查：

```bash
python src/etlp_continuous_toy.py --train-steps 600 --eval-every 100 --eval-samples 200
```

### Cognitive Map + ETLP-like Toy

核心更新：

$$
\Delta W_a[o, i] = \eta \cdot \delta^{pred}_o(t) \cdot \bar{x}_i(t)
$$

用途：用 one-step prediction 学动作转移结构，再把 learned transition graph 用于 shortest-path planning。

快速检查：

```bash
python src/cognitive_map_etlp_toy.py --train-steps 1000 --eval-every 250
```

关键指标：`prediction_mse`、`transition_acc`、`planning_success`、`path_efficiency`。

### R-SNN Point Robot Toy

组件：

- `rsnn.py`：LIF-like recurrent spiking network。
- `point_robot_env.py`：连续状态、离散动作的 2D point robot。
- `rsnn_point_robot_toy.py`：world model + TD action value 控制闭环。

用途：验证 R-SNN recurrent state 能否进入真正的 `observe -> act -> learn` 控制回路。

快速检查：

```bash
python src/rsnn_point_robot_toy.py --episodes 160 --eval-every 40 --eval-episodes 40
```

关键指标：`random_baseline`、`model_mse`、`eval_reward`、`eval_success`、`eval_len`。
