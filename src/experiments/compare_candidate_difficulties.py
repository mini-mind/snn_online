"""Compare shortlisted SNN mechanisms across point-robot difficulty presets."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from envs.point_robot import PointRobotConfig
from models.common import mean
from models.point_robot_closed_loop import AgentConfig, train_agent


DIFFICULTY_PRESETS = {
    "easy": {
        "observation_mode": "partial_goal_cue",
        "goal_cue_steps": 10,
        "max_steps": 40,
    },
    "medium": {
        "observation_mode": "partial_goal_cue",
        "goal_cue_steps": 6,
        "max_steps": 60,
    },
    "hard": {
        "observation_mode": "partial_goal_cue",
        "goal_cue_steps": 3,
        "max_steps": 80,
    },
}

CANDIDATES = {
    "scalar_delay_rline": {
        "modulation_mode": "scalar",
        "delay_features": True,
        "recurrent_delay_line": True,
    },
    "per_neuron_plain_rline": {
        "modulation_mode": "per_neuron",
        "delay_features": False,
        "recurrent_delay_line": True,
    },
}


def append_jsonl(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def prepare_jsonl(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def run_candidate_difficulties(
    base_config: AgentConfig,
    difficulties: list[str],
    seeds: list[int],
    output_jsonl: Path | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    aggregated: dict[str, dict[str, dict[str, float]]] = {}
    for difficulty in difficulties:
        env_config = env_config_for_difficulty(difficulty, seed=base_config.seed + 7)
        aggregated[difficulty] = {}
        print(f"difficulty={difficulty} benchmark_id={env_config.task_metadata()['benchmark_id']}")
        for candidate, settings in CANDIDATES.items():
            print(f"  candidate={candidate}")
            rows = []
            for seed in seeds:
                agent_config = replace(base_config, seed=seed, **settings)
                run_env_config = replace(env_config, seed=seed + 7)
                summary = train_agent(agent_config, run_env_config, verbose=False)
                rows.append(summary)
                print(
                    f"    seed={seed} "
                    f"eval_reward={summary['final_eval_reward']:.3f} "
                    f"eval_success={summary['final_eval_success']:.3f} "
                    f"elapsed_sec={summary['elapsed_sec']:.3f}"
                )
                if output_jsonl is not None:
                    append_jsonl(
                        output_jsonl,
                        run_row(difficulty, candidate, agent_config, run_env_config, summary),
                    )
            aggregated[difficulty][candidate] = {
                "mean_eval_reward": mean(row["final_eval_reward"] for row in rows),
                "mean_eval_success": mean(row["final_eval_success"] for row in rows),
                "mean_elapsed_sec": mean(row["elapsed_sec"] for row in rows),
            }
        print()
    if output_jsonl is not None:
        append_jsonl(
            output_jsonl,
            {
                "type": "summary",
                "config": config_to_dict(base_config, difficulties, seeds),
                "difficulties": aggregated,
                "best_by_difficulty": {
                    difficulty: {
                        "best_by_reward": best_candidate(results, "mean_eval_reward"),
                        "best_by_success": best_candidate(results, "mean_eval_success"),
                    }
                    for difficulty, results in aggregated.items()
                },
            },
        )
    return aggregated


def env_config_for_difficulty(difficulty: str, seed: int) -> PointRobotConfig:
    if difficulty not in DIFFICULTY_PRESETS:
        raise ValueError(f"unknown difficulty {difficulty}; expected one of {sorted(DIFFICULTY_PRESETS)}")
    return PointRobotConfig(seed=seed, **DIFFICULTY_PRESETS[difficulty])


def run_row(
    difficulty: str,
    candidate: str,
    agent_config: AgentConfig,
    env_config: PointRobotConfig,
    summary: dict[str, object],
) -> dict[str, object]:
    return {
        "type": "run",
        "difficulty": difficulty,
        "candidate": candidate,
        "seed": agent_config.seed,
        "eval_reward": summary["final_eval_reward"],
        "eval_success": summary["final_eval_success"],
        "elapsed_sec": summary["elapsed_sec"],
        "episodes": agent_config.episodes,
        "eval_every": agent_config.eval_every,
        "eval_episodes": agent_config.eval_episodes,
        "n_neurons": agent_config.n_neurons,
        "recurrent_degree": agent_config.recurrent_degree,
        "plasticity_rule": agent_config.plasticity_rule,
        "neuron_model": agent_config.neuron_model,
        "modulation_mode": agent_config.modulation_mode,
        "delay_features": agent_config.delay_features,
        "recurrent_delay_line": agent_config.recurrent_delay_line,
        "observation_mode": env_config.observation_mode,
        "goal_cue_steps": env_config.goal_cue_steps,
        "max_steps": env_config.max_steps,
        "task_family": summary["task_family"],
        "benchmark_id": summary["benchmark_id"],
        "observability_level": summary["observability_level"],
        "horizon_level": summary["horizon_level"],
        "reward_level": summary["reward_level"],
        "dynamics_level": summary["dynamics_level"],
        "goal_structure_level": summary["goal_structure_level"],
        "action_level": summary["action_level"],
        "distribution_level": summary["distribution_level"],
        "biological_params": summary["biological_params"],
    }


def best_candidate(results: dict[str, dict[str, float]], metric: str) -> dict[str, object]:
    if not results:
        return {}
    name = max(results, key=lambda key: results[key][metric])
    return {"candidate": name, metric: results[name][metric]}


def config_to_dict(base_config: AgentConfig, difficulties: list[str], seeds: list[int]) -> dict[str, object]:
    return {
        "episodes": base_config.episodes,
        "eval_every": base_config.eval_every,
        "eval_episodes": base_config.eval_episodes,
        "n_neurons": base_config.n_neurons,
        "recurrent_degree": base_config.recurrent_degree,
        "plasticity_rule": base_config.plasticity_rule,
        "neuron_model": base_config.neuron_model,
        "difficulty_presets": difficulties,
        "candidates": sorted(CANDIDATES),
        "seeds": seeds,
    }


def parse_args() -> tuple[AgentConfig, list[str], list[int], Path | None]:
    parser = argparse.ArgumentParser(
        description="Compare shortlisted delay/modulation candidates across task difficulties."
    )
    parser.add_argument("--episodes", type=int, default=80)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--n-neurons", type=int, default=64)
    parser.add_argument("--recurrent-degree", type=int, default=4)
    parser.add_argument("--plasticity-rule", choices=["three_factor", "tess_like"], default="tess_like")
    parser.add_argument("--neuron-model", choices=["lif", "izh"], default=AgentConfig.neuron_model)
    parser.add_argument(
        "--difficulties",
        nargs="+",
        choices=sorted(DIFFICULTY_PRESETS),
        default=["easy", "medium", "hard"],
    )
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--seed-start", type=int, default=31)
    parser.add_argument("--output-jsonl", default="")
    args = parser.parse_args()
    return (
        AgentConfig(
            episodes=args.episodes,
            eval_every=args.eval_every,
            eval_episodes=args.eval_episodes,
            n_neurons=args.n_neurons,
            recurrent_degree=args.recurrent_degree,
            plasticity_rule=args.plasticity_rule,
            neuron_model=args.neuron_model,
            randomize_intrinsics=True,
            seed=args.seed_start,
        ),
        list(args.difficulties),
        [args.seed_start + offset for offset in range(args.seeds)],
        Path(args.output_jsonl) if args.output_jsonl else None,
    )


def main() -> None:
    agent_config, difficulties, seeds, output_jsonl = parse_args()
    print(
        f"difficulties={','.join(difficulties)} "
        f"n_neurons={agent_config.n_neurons} "
        f"recurrent_degree={agent_config.recurrent_degree} "
        f"plasticity_rule={agent_config.plasticity_rule} "
        f"neuron_model={agent_config.neuron_model}"
    )
    if output_jsonl is not None:
        prepare_jsonl(output_jsonl)
        print(f"output_jsonl={output_jsonl}")
    results = run_candidate_difficulties(
        agent_config,
        difficulties,
        seeds,
        output_jsonl=output_jsonl,
    )
    print("summary")
    for difficulty in difficulties:
        for candidate in sorted(results[difficulty]):
            metrics = results[difficulty][candidate]
            print(
                f"  {difficulty}/{candidate} "
                f"mean_eval_reward={metrics['mean_eval_reward']:.3f} "
                f"mean_eval_success={metrics['mean_eval_success']:.3f} "
                f"mean_elapsed_sec={metrics['mean_elapsed_sec']:.3f}"
            )
        print(
            f"  {difficulty} "
            f"best_by_reward={best_candidate(results[difficulty], 'mean_eval_reward')['candidate']} "
            f"best_by_success={best_candidate(results[difficulty], 'mean_eval_success')['candidate']}"
        )


if __name__ == "__main__":
    main()
