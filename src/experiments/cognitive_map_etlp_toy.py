"""CLI entry for the cognitive-map ETLP-like toy."""

from __future__ import annotations

import argparse

from models.toy_learning import CognitiveMapConfig, train_cognitive_map


def parse_args() -> CognitiveMapConfig:
    parser = argparse.ArgumentParser(description="Run the Cognitive Map + ETLP-like toy.")
    parser.add_argument("--grid-size", type=int, default=CognitiveMapConfig.grid_size)
    parser.add_argument("--feature-dim", type=int, default=CognitiveMapConfig.feature_dim)
    parser.add_argument("--train-steps", type=int, default=CognitiveMapConfig.train_steps)
    parser.add_argument("--eval-every", type=int, default=CognitiveMapConfig.eval_every)
    parser.add_argument("--eval-pairs", type=int, default=CognitiveMapConfig.eval_pairs)
    parser.add_argument("--planning-horizon", type=int, default=CognitiveMapConfig.planning_horizon)
    parser.add_argument("--lr", type=float, default=CognitiveMapConfig.lr)
    parser.add_argument("--trace-decay", type=float, default=CognitiveMapConfig.trace_decay)
    parser.add_argument("--noise", type=float, default=CognitiveMapConfig.noise)
    parser.add_argument("--seed", type=int, default=CognitiveMapConfig.seed)
    args = parser.parse_args()
    return CognitiveMapConfig(
        grid_size=args.grid_size,
        feature_dim=args.feature_dim,
        train_steps=args.train_steps,
        eval_every=args.eval_every,
        eval_pairs=args.eval_pairs,
        planning_horizon=args.planning_horizon,
        lr=args.lr,
        trace_decay=args.trace_decay,
        noise=args.noise,
        seed=args.seed,
    )


def main() -> None:
    train_cognitive_map(parse_args())


if __name__ == "__main__":
    main()
