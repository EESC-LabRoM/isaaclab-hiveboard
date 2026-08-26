#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train a single-step MLP behavior-cloning policy. Does not launch Isaac Sim."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from policy.bc.train import TrainConfig, train


def parse_args() -> argparse.Namespace:
    """Parse training arguments."""
    parser = argparse.ArgumentParser(description="Train MLP behavior cloning on cleaned Spot ball-valve HDF5.")
    parser.add_argument(
        "--dataset_path",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("logs/bc/spot_ball_valve_mlp"),
        help="Run directory for checkpoints, config.json, and train.csv.",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight_decay", type=float, default=1.0e-4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max_train_episodes",
        type=int,
        default=None,
        help="Cap the train split (use 1 to overfit a single demonstration).",
    )
    parser.add_argument("--early_stop_patience", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--eval_demo",
        type=int,
        default=None,
        help="After training, print predicted vs demo actions for this episode.",
    )
    parser.add_argument(
        "--last_action_dropout",
        type=float,
        default=0.0,
        help=(
            "Train-time probability of zeroing last_action so the open/close "
            "command cannot be ignored (eval still uses the real last action)."
        ),
    )
    parser.add_argument(
        "--last_action_shuffle",
        type=float,
        default=0.0,
        help=(
            "Train-time probability of replacing last_action with another "
            "sample's last action (including the opposite task). Try 0.3 if "
            "dropout alone is not enough."
        ),
    )
    parser.add_argument(
        "--transition_weight",
        type=float,
        default=8.0,
        help=(
            "Oversample grasp/rotate change-points inside each demo. 1.0 is "
            "uniform timestep sampling; values > 1 upweight a window around "
            "gripper close and large action jumps."
        ),
    )
    parser.add_argument(
        "--transition_window",
        type=int,
        default=4,
        help="Frames on each side of a detected transition to oversample.",
    )
    parser.add_argument(
        "--transition_action_delta",
        type=float,
        default=0.5,
        help="Minimum ||a_t - a_{t-1}|| counted as a motion jump.",
    )
    parser.add_argument(
        "--transition_ignore_prefix",
        type=int,
        default=2,
        help="Skip this many leading steps so t=0 zeros are not transitions.",
    )
    return parser.parse_args()


def main() -> None:
    """Run BC training."""
    args = parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch_size must be positive")
    if args.epochs <= 0:
        raise SystemExit("--epochs must be positive")
    if not args.dataset_path.is_file():
        raise SystemExit(f"Dataset does not exist: {args.dataset_path}")
    if not 0.0 <= args.last_action_dropout <= 1.0:
        raise SystemExit("--last_action_dropout must be in [0, 1]")
    if not 0.0 <= args.last_action_shuffle <= 1.0:
        raise SystemExit("--last_action_shuffle must be in [0, 1]")
    if args.transition_weight <= 0.0:
        raise SystemExit("--transition_weight must be positive")
    if args.transition_window < 0:
        raise SystemExit("--transition_window must be >= 0")
    if args.transition_action_delta < 0.0:
        raise SystemExit("--transition_action_delta must be >= 0")
    if args.transition_ignore_prefix < 0:
        raise SystemExit("--transition_ignore_prefix must be >= 0")
    config = TrainConfig(
        dataset_path=str(args.dataset_path.resolve()),
        output_dir=str(args.output_dir),
        device=args.device,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        dropout=args.dropout,
        val_ratio=args.val_ratio,
        seed=args.seed,
        max_train_episodes=args.max_train_episodes,
        early_stop_patience=args.early_stop_patience,
        num_workers=args.num_workers,
        eval_demo=args.eval_demo,
        last_action_dropout=args.last_action_dropout,
        last_action_shuffle=args.last_action_shuffle,
        transition_weight=args.transition_weight,
        transition_window=args.transition_window,
        transition_action_delta=args.transition_action_delta,
        transition_ignore_prefix=args.transition_ignore_prefix,
    )
    train(config)


if __name__ == "__main__":
    main()
