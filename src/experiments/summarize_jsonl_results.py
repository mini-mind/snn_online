"""Summarize JSONL experiment outputs from comparison scripts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read JSONL outputs from experiment comparison scripts and print compact summary lines."
        )
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="One or more JSONL files to summarize.",
    )
    return parser.parse_args()


def load_summary_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    print(
                        f"warning: skipping invalid JSON in {path}:{line_number}",
                        file=sys.stderr,
                    )
                    continue
                if isinstance(row, dict) and row.get("type") == "summary":
                    rows.append(row)
    except OSError as error:
        print(f"warning: could not read {path}: {error}", file=sys.stderr)
    return rows


def format_float(value: object) -> str:
    if isinstance(value, int | float):
        return f"{float(value):.3f}"
    return "na"


def format_seeds(value: object) -> str:
    if not isinstance(value, list) or not value:
        return "seeds=na"
    numeric_seeds = [seed for seed in value if isinstance(seed, int)]
    if not numeric_seeds:
        return "seeds=na"
    if len(numeric_seeds) == 1:
        return f"seeds={numeric_seeds[0]}"
    return f"seeds={numeric_seeds[0]}..{numeric_seeds[-1]}(n={len(numeric_seeds)})"


def format_config(config: object) -> str:
    if not isinstance(config, dict):
        return "config=na"
    ordered_keys = (
        "episodes",
        "eval_every",
        "eval_episodes",
        "n_neurons",
        "recurrent_degree",
        "plasticity_rule",
        "neuron_model",
        "modulation_mode",
        "observation_mode",
        "goal_cue_steps",
        "max_steps",
        "include_recurrent_delay_line",
        "tess_fast_decay",
        "tess_slow_decay",
        "tess_post_decay",
        "tess_eligibility_decay",
    )
    parts = []
    for key in ordered_keys:
        if key in config:
            parts.append(f"{key}={config[key]}")
    if "seeds" in config:
        parts.append(format_seeds(config["seeds"]))
    return " ".join(parts) if parts else "config=na"


def is_metric_block(value: object) -> bool:
    return (
        isinstance(value, dict)
        and "mean_eval_reward" in value
        and "mean_eval_success" in value
        and "mean_elapsed_sec" in value
    )


def format_metrics(label: str, value: object) -> str:
    if not isinstance(value, dict):
        return f"{label}=na"
    return (
        f"{label}[reward={format_float(value.get('mean_eval_reward'))} "
        f"success={format_float(value.get('mean_eval_success'))} "
        f"time={format_float(value.get('mean_elapsed_sec'))}]"
    )


def format_deltas(delta: object) -> str:
    if not isinstance(delta, dict):
        return "delta=na"
    reward_parts = []
    success_parts = []
    speed_parts = []
    for key, value in delta.items():
        if "reward" in key:
            reward_parts.append(f"{key}={format_float(value)}")
        elif "success" in key:
            success_parts.append(f"{key}={format_float(value)}")
        elif "speed_ratio" in key:
            speed_parts.append(f"{key}={format_float(value)}")
    ordered = reward_parts + success_parts + speed_parts
    return " ".join(ordered) if ordered else "delta=na"


def summarize_row(path: Path, row: dict[str, object]) -> str:
    metric_labels = [key for key, value in row.items() if key != "delta" and key != "config" and is_metric_block(value)]
    metric_labels.sort()
    parts = [str(path), format_config(row.get("config"))]
    conditions = row.get("conditions")
    if isinstance(conditions, dict):
        condition_labels = [
            key
            for key, value in conditions.items()
            if isinstance(key, str) and is_metric_block(value)
        ]
        condition_labels.sort()
        parts.extend(format_metrics(label, conditions[label]) for label in condition_labels)
    elif isinstance(row.get("difficulties"), dict):
        parts.extend(format_difficulty_metrics(row["difficulties"]))
        parts.extend(format_best_by_difficulty(row.get("best_by_difficulty")))
    elif isinstance(row.get("stages"), dict):
        parts.extend(format_stage_metrics(row["stages"]))
        parts.append(format_best(row.get("best_by_reward"), "best_reward"))
        parts.append(format_best(row.get("best_by_success"), "best_success"))
    else:
        parts.extend(format_metrics(label, row[label]) for label in metric_labels)
    parts.append(format_deltas(row.get("delta")))
    if not isinstance(row.get("stages"), dict):
        parts.append(format_best(row.get("best_by_reward"), "best_reward"))
        parts.append(format_best(row.get("best_by_success"), "best_success"))
    return " | ".join(parts)


def format_best(value: object, label: str) -> str:
    if not isinstance(value, dict):
        return f"{label}=na"
    name = value.get("condition", value.get("stage", value.get("candidate")))
    if name is None:
        return f"{label}=na"
    return f"{label}={name}"


def format_stage_metrics(value: object) -> list[str]:
    if not isinstance(value, dict):
        return ["stages=na"]
    parts = []
    for stage in sorted(value):
        metrics = value[stage]
        if is_metric_block(metrics):
            parts.append(format_metrics(stage, metrics))
    return parts if parts else ["stages=na"]


def format_difficulty_metrics(value: object) -> list[str]:
    if not isinstance(value, dict):
        return ["difficulties=na"]
    parts = []
    for difficulty in sorted(value):
        candidates = value[difficulty]
        if not isinstance(candidates, dict):
            continue
        for candidate in sorted(candidates):
            metrics = candidates[candidate]
            if is_metric_block(metrics):
                parts.append(format_metrics(f"{difficulty}/{candidate}", metrics))
    return parts if parts else ["difficulties=na"]


def format_best_by_difficulty(value: object) -> list[str]:
    if not isinstance(value, dict):
        return ["best_by_difficulty=na"]
    parts = []
    for difficulty in sorted(value):
        row = value[difficulty]
        if not isinstance(row, dict):
            continue
        reward = row.get("best_by_reward")
        success = row.get("best_by_success")
        reward_name = reward.get("candidate") if isinstance(reward, dict) else "na"
        success_name = success.get("candidate") if isinstance(success, dict) else "na"
        parts.append(f"{difficulty}[best_reward={reward_name} best_success={success_name}]")
    return parts if parts else ["best_by_difficulty=na"]


def main() -> int:
    args = parse_args()
    found_summary = False
    for raw_path in args.paths:
        path = Path(raw_path)
        summary_rows = load_summary_rows(path)
        if not summary_rows:
            print(f"warning: no summary rows found in {path}", file=sys.stderr)
            continue
        found_summary = True
        for row in summary_rows:
            print(summarize_row(path, row))
    return 0 if found_summary else 1


if __name__ == "__main__":
    raise SystemExit(main())
