"""比较 LIF 与 Izhikevich 神经元模型在点机器人闭环中的表现。

运行方式:
    python src/closed_loop/compare_lif_vs_izh.py

该脚本会在相同训练配置下分别运行 `lif` 与 `izh` 两种神经元模型，记录：

- 最终评估奖励；
- 最终评估成功率；
- 总训练耗时。

目标不是做统计学充分的基准，而是给当前原型提供一个可迭代的“是否值得换模
型”的快速判断。当前默认配置会使用更深的堆叠循环网络，并开启每个神经元
各自的随机内禀参数初始化。脚本也支持切换到 `partial_goal_cue` 任务，用来
测试在更依赖记忆的设置下，`izh` 是否更容易体现优势。
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from typing import Iterable

from point_robot_closed_loop import AgentConfig, PointRobotConfig, train_agent
from recurrent_spiking import resolve_grid_shape


def run_comparison(
    base_config: AgentConfig,
    env_config: PointRobotConfig,
    seeds: list[int],
) -> dict[str, dict[str, float]]:
    """在多个随机种子上比较两种神经元模型。"""
    summaries: dict[str, list[dict[str, float | str]]] = {"lif": [], "izh": []}
    for neuron_model in ("lif", "izh"):
        print(f"model={neuron_model}")
        for seed in seeds:
            agent_config = replace(base_config, neuron_model=neuron_model, seed=seed)
            run_env_config = replace(env_config, seed=seed + 7)
            summary = train_agent(agent_config, run_env_config, verbose=False)
            summaries[neuron_model].append(summary)
            print(
                f"  seed={seed} "
                f"eval_reward={summary['final_eval_reward']:.3f} "
                f"eval_success={summary['final_eval_success']:.3f} "
                f"elapsed_sec={summary['elapsed_sec']:.3f}"
            )
        print()

    aggregated: dict[str, dict[str, float]] = {}
    for neuron_model, rows in summaries.items():
        aggregated[neuron_model] = {
            "mean_eval_reward": mean(row["final_eval_reward"] for row in rows),
            "mean_eval_success": mean(row["final_eval_success"] for row in rows),
            "mean_elapsed_sec": mean(row["elapsed_sec"] for row in rows),
        }

    aggregated["delta"] = {
        "reward_gain_izh_minus_lif": (
            aggregated["izh"]["mean_eval_reward"] - aggregated["lif"]["mean_eval_reward"]
        ),
        "success_gain_izh_minus_lif": (
            aggregated["izh"]["mean_eval_success"] - aggregated["lif"]["mean_eval_success"]
        ),
        "speed_ratio_izh_vs_lif": safe_ratio(
            aggregated["izh"]["mean_elapsed_sec"],
            aggregated["lif"]["mean_elapsed_sec"],
        ),
    }
    return aggregated


def mean(values: Iterable[float]) -> float:
    """计算一组标量的平均值。"""
    value_list = list(values)
    return sum(value_list) / max(1, len(value_list))


def safe_ratio(numerator: float, denominator: float) -> float:
    """安全地计算比值。"""
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def parse_args() -> tuple[AgentConfig, PointRobotConfig, list[int]]:
    """解析对比实验参数。"""
    parser = argparse.ArgumentParser(description="Compare LIF and Izhikevich on point robot control.")
    parser.add_argument("--episodes", type=int, default=80)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--n-neurons", type=int, default=96)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--grid-width", type=int, default=0)
    parser.add_argument("--grid-height", type=int, default=0)
    parser.add_argument("--recurrent-radius", type=int, default=1)
    parser.add_argument("--recurrent-degree", type=int, default=4)
    parser.add_argument("--interlayer-radius", type=int, default=1)
    parser.add_argument("--interlayer-degree", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=PointRobotConfig.max_steps)
    parser.add_argument(
        "--observation-mode",
        choices=["full", "partial_goal_cue"],
        default=PointRobotConfig.observation_mode,
    )
    parser.add_argument("--goal-cue-steps", type=int, default=PointRobotConfig.goal_cue_steps)
    parser.add_argument("--seeds", type=int, default=2, help="number of sequential seeds to run")
    parser.add_argument("--seed-start", type=int, default=31)
    args = parser.parse_args()

    agent_config = AgentConfig(
        episodes=args.episodes,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        n_neurons=args.n_neurons,
        n_layers=args.n_layers,
        grid_width=args.grid_width,
        grid_height=args.grid_height,
        recurrent_radius=args.recurrent_radius,
        recurrent_degree=args.recurrent_degree,
        interlayer_radius=args.interlayer_radius,
        interlayer_degree=args.interlayer_degree,
        neuron_model="lif",
        randomize_intrinsics=True,
        seed=args.seed_start,
    )
    env_config = PointRobotConfig(
        max_steps=args.max_steps,
        observation_mode=args.observation_mode,
        goal_cue_steps=args.goal_cue_steps,
        seed=args.seed_start + 7,
    )
    seeds = [args.seed_start + offset for offset in range(args.seeds)]
    return agent_config, env_config, seeds


def main() -> None:
    """命令行入口。"""
    agent_config, env_config, seeds = parse_args()
    grid_width, grid_height = resolve_grid_shape(
        agent_config.n_neurons,
        agent_config.grid_width,
        agent_config.grid_height,
    )
    print(
        f"task observation_mode={env_config.observation_mode} "
        f"goal_cue_steps={env_config.goal_cue_steps} "
        f"max_steps={env_config.max_steps} "
        f"grid={grid_width}x{grid_height} "
        f"layers={agent_config.n_layers}"
    )
    results = run_comparison(agent_config, env_config, seeds)
    print("summary")
    print(
        f"  lif mean_eval_reward={results['lif']['mean_eval_reward']:.3f} "
        f"mean_eval_success={results['lif']['mean_eval_success']:.3f} "
        f"mean_elapsed_sec={results['lif']['mean_elapsed_sec']:.3f}"
    )
    print(
        f"  izh mean_eval_reward={results['izh']['mean_eval_reward']:.3f} "
        f"mean_eval_success={results['izh']['mean_eval_success']:.3f} "
        f"mean_elapsed_sec={results['izh']['mean_elapsed_sec']:.3f}"
    )
    print(
        f"  delta reward_gain_izh_minus_lif={results['delta']['reward_gain_izh_minus_lif']:.3f} "
        f"success_gain_izh_minus_lif={results['delta']['success_gain_izh_minus_lif']:.3f} "
        f"speed_ratio_izh_vs_lif={results['delta']['speed_ratio_izh_vs_lif']:.3f}"
    )


if __name__ == "__main__":
    main()
