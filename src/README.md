# Minimal Experiments

本目录放少量代码即可运行的研究原型。目标不是完整复刻论文，而是把关键学习规则压缩成可读、可改、可证伪的实验。

## ETLP Continuous Toy

文件：`etlp_continuous_toy.py`

这是一个纯标准库实现的 ETLP-like 连续输入 toy：

```text
continuous temporal input -> LIF-like output neurons -> online local update
```

核心更新规则：

$$
\Delta w_{ij} = \eta \cdot \bar{x}_i(t) \cdot f(V_j(t)) \cdot T_j(t)
$$

其中：

- `pre_trace` / $\bar{x}_i(t)$：连续输入的低通轨迹，不要求输入是 binary spike。
- `post_membrane_factor` / $f(V_j(t))$：突触后膜电位相关的伪导数/敏感度。
- `teaching_signal` / $T_j(t)$：目标类别与当前输出概率的误差信号。

运行：

```bash
python src/etlp_continuous_toy.py
```

快速检查：

```bash
python src/etlp_continuous_toy.py --train-steps 600 --eval-every 100 --eval-samples 200
```

可调参数：

```bash
python src/etlp_continuous_toy.py --lr 0.02 --noise 0.25 --drift 0.5 --seed 3
```

这个 toy 适合观察：

- 连续输入如何变成 analog eligibility trace。
- 教学信号如何只调制已有局部 trace，而不是直接指定每个权重。
- 分布漂移下，在线局部规则是否还能持续适应。
- 代码层面 ETLP 与 e-prop 的差异：这里不是从 BPTT 导数推 trace，而是工程化的事件/状态三因子更新。


## Cognitive Map + ETLP-like Toy

文件：`cognitive_map_etlp_toy.py`

这是一个组合原型：用 ETLP-like 三因子局部规则学习 Cognitive Map 的动作转移结构。

```text
gridworld state -> continuous place code
state_code + action -> predicted next_state_code
prediction_error -> local transition update
learned transition graph -> shortest-path planning
```

核心更新规则：

$$
\Delta W_a[o, i] = \eta \cdot \delta^{pred}_o(t) \cdot \bar{x}_i(t)
$$

其中：

- $a$：当前动作。
- $\bar{x}_i(t)$：当前状态编码的低通 trace。
- $\delta^{pred}_o(t)$：下一状态预测误差的第 $o$ 个输出通道。
- $W_a$：动作 $a$ 对应的局部 transition model。

运行：

```bash
python src/cognitive_map_etlp_toy.py
```

快速检查：

```bash
python src/cognitive_map_etlp_toy.py --train-steps 1000 --eval-every 250
```

默认输出会显示：

```text
prediction_mse      下一状态预测误差
transition_acc      解码后的动作转移准确率
planning_success    用学到的转移图规划到目标的成功率
path_efficiency     学到路径相对最短路径的效率
```

这个 toy 适合观察：

- prediction error 如何作为第三因子，而不是 reward。
- 局部转移学习如何从 one-step prediction 升级成可规划地图。
- 为什么“世界模型”可以先于奖励学习出现。
- 如何把 ETLP-like 局部更新接到 Cognitive Map Learner。
