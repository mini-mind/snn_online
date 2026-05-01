"""CLI entry for the continuous ETLP-like toy."""

from __future__ import annotations

import argparse

from models.toy_learning import ContinuousToyConfig, train_continuous_toy


def parse_args() -> ContinuousToyConfig:
    parser = argparse.ArgumentParser(description="Run the ETLP-like continuous-input toy.")
    parser.add_argument("--train-steps", type=int, default=ContinuousToyConfig.train_steps)
    parser.add_argument("--eval-every", type=int, default=ContinuousToyConfig.eval_every)
    parser.add_argument("--eval-samples", type=int, default=ContinuousToyConfig.eval_samples)
    parser.add_argument("--seq-len", type=int, default=ContinuousToyConfig.seq_len)
    parser.add_argument("--noise", type=float, default=ContinuousToyConfig.noise)
    parser.add_argument("--drift", type=float, default=ContinuousToyConfig.drift)
    parser.add_argument("--seed", type=int, default=ContinuousToyConfig.seed)
    args = parser.parse_args()
    return ContinuousToyConfig(
        train_steps=args.train_steps,
        eval_every=args.eval_every,
        eval_samples=args.eval_samples,
        seq_len=args.seq_len,
        noise=args.noise,
        drift=args.drift,
        seed=args.seed,
    )


def main() -> None:
    train_continuous_toy(parse_args())


if __name__ == "__main__":
    main()
