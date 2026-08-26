"""Training loop for single-step MLP behavior cloning."""

from __future__ import annotations

import csv
import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .config import ACTION_DIM
from .dataset import (
    SpotBallValveDataset,
    load_episode_arrays,
    split_episode_indices,
)
from .policy import BehaviorCloningPolicy


@dataclass
class TrainConfig:
    """CLI-resolved training hyperparameters."""

    dataset_path: str
    output_dir: str
    device: str = "cpu"
    batch_size: int = 256
    epochs: int = 100
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-4
    hidden_dim: int = 256
    n_layers: int = 2
    dropout: float = 0.1
    val_ratio: float = 0.05
    seed: int = 42
    max_train_episodes: int | None = None
    early_stop_patience: int = 20
    num_workers: int = 0
    eval_demo: int | None = None
    last_action_dropout: float = 0.5
    last_action_shuffle: float = 0.0
    transition_weight: float = 8.0
    transition_window: int = 4
    transition_action_delta: float = 0.5
    transition_ignore_prefix: int = 2


def set_seed(seed: int) -> None:
    """Seed numpy and torch."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def git_hash() -> str | None:
    """Return HEAD if this is a git checkout."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def corrupt_last_action(
    obs: torch.Tensor,
    dropout: float,
    shuffle: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Break the last-action shortcut so direction must carry the rotate sign.

    ``dropout`` zeros the last-action slice (same zeros as t=0 at eval).
    ``shuffle`` replaces it with another row's last action, including the
    opposite task. Applied only during training; validation stays clean.
    """
    if dropout < 0.0 or dropout > 1.0:
        raise ValueError(f"last_action_dropout must be in [0, 1], got {dropout}")
    if shuffle < 0.0 or shuffle > 1.0:
        raise ValueError(f"last_action_shuffle must be in [0, 1], got {shuffle}")
    if obs.ndim != 2 or obs.shape[-1] < ACTION_DIM:
        raise ValueError(
            f"obs must have shape (batch, >= {ACTION_DIM}), got {tuple(obs.shape)}"
        )
    if dropout <= 0.0 and shuffle <= 0.0:
        return obs
    corrupted = obs.clone()
    batch = obs.shape[0]
    last = slice(-ACTION_DIM, None)
    if dropout > 0.0:
        drop_mask = torch.rand(batch, device=obs.device, generator=generator) < dropout
        corrupted[drop_mask, last] = 0.0
    if shuffle > 0.0 and batch > 1:
        shuffle_mask = (
            torch.rand(batch, device=obs.device, generator=generator) < shuffle
        )
        perm = torch.randperm(batch, device=obs.device, generator=generator)
        corrupted[shuffle_mask, last] = obs[perm[shuffle_mask], last]
    return corrupted


def _mean_loss(loader: DataLoader, policy: BehaviorCloningPolicy, device: torch.device) -> float:
    """Mean per-element MSE, matching ``nn.MSELoss()`` used in the train loop."""
    total = 0.0
    count = 0
    criterion = torch.nn.MSELoss()
    with torch.inference_mode():
        for batch in loader:
            obs = batch["obs"].to(device)
            action = batch["action"].to(device)
            assert policy.normalizer is not None
            pred = policy(policy.normalizer.normalize_obs(obs))
            target = policy.normalizer.normalize_action(action)
            batch_size = int(obs.shape[0])
            total += float(criterion(pred, target).item()) * batch_size
            count += batch_size
    return total / max(count, 1)


def dump_demo_predictions(
    policy: BehaviorCloningPolicy,
    dataset_path: str | Path,
    demo_index: int,
    device: torch.device,
    max_steps: int = 8,
) -> None:
    """Print predicted vs recorded actions for one demonstration."""
    observations, actions, _, _ = load_episode_arrays(dataset_path)
    if demo_index < 0 or demo_index >= len(observations):
        raise IndexError(
            f"eval_demo={demo_index} is out of range for {len(observations)} episodes"
        )
    obs = torch.from_numpy(observations[demo_index][:max_steps]).to(device)
    act = torch.from_numpy(actions[demo_index][:max_steps]).to(device)
    policy.eval()
    with torch.inference_mode():
        predicted = policy.predict_action({"obs": obs})["action"][:, 0]
    print(f"[INFO] Demo {demo_index} first {obs.shape[0]} steps:")
    for step in range(obs.shape[0]):
        target = act[step].detach().cpu().tolist()
        pred = predicted[step].detach().cpu().tolist()
        print(f"  t={step} target={target}")
        print(f"       pred  ={pred}")


def train(config: TrainConfig) -> Path:
    """Train an MLP BC policy and write ``best.pt``.

    Args:
        config: Training hyperparameters.

    Returns:
        Path to the best checkpoint.
    """
    if not 0.0 <= config.last_action_dropout <= 1.0:
        raise ValueError(
            "last_action_dropout must be in [0, 1], "
            f"got {config.last_action_dropout}"
        )
    if not 0.0 <= config.last_action_shuffle <= 1.0:
        raise ValueError(
            "last_action_shuffle must be in [0, 1], "
            f"got {config.last_action_shuffle}"
        )
    if config.transition_weight <= 0.0:
        raise ValueError(
            f"transition_weight must be positive, got {config.transition_weight}"
        )
    if config.transition_window < 0:
        raise ValueError(
            f"transition_window must be >= 0, got {config.transition_window}"
        )
    if config.transition_action_delta < 0.0:
        raise ValueError(
            "transition_action_delta must be >= 0, "
            f"got {config.transition_action_delta}"
        )
    if config.transition_ignore_prefix < 0:
        raise ValueError(
            "transition_ignore_prefix must be >= 0, "
            f"got {config.transition_ignore_prefix}"
        )
    set_seed(config.seed)
    device = torch.device(config.device)
    output_dir = Path(config.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    episodes_obs, episodes_act, obs_names, act_names = load_episode_arrays(
        config.dataset_path
    )
    train_ids, val_ids = split_episode_indices(
        len(episodes_obs), val_ratio=config.val_ratio, seed=config.seed
    )
    if config.max_train_episodes is not None:
        train_ids = train_ids[: max(1, int(config.max_train_episodes))]

    train_set = SpotBallValveDataset(
        episodes_obs, episodes_act, train_ids, obs_names, act_names
    )
    val_set = (
        SpotBallValveDataset(episodes_obs, episodes_act, val_ids, obs_names, act_names)
        if val_ids
        else None
    )
    normalizer = train_set.get_normalizer().to(device)

    generator = torch.Generator()
    generator.manual_seed(config.seed)
    train_batch_size = min(config.batch_size, len(train_set))
    sample_weights, transition_mask = train_set.transition_sample_weights(
        weight=config.transition_weight,
        window=config.transition_window,
        ignore_prefix=config.transition_ignore_prefix,
        action_delta_threshold=config.transition_action_delta,
    )
    n_transition = int(transition_mask.sum())
    weight_sum = float(sample_weights.sum())
    transition_mass = (
        float(sample_weights[transition_mask].sum() / weight_sum)
        if weight_sum > 0.0
        else 0.0
    )
    oversample_transitions = config.transition_weight > 1.0 and n_transition > 0
    if config.transition_weight > 1.0 and n_transition == 0:
        print(
            "[WARNING] No grasp/rotate transitions found; "
            "using uniform timestep sampling"
        )
    train_sampler = (
        WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(train_set),
            replacement=True,
            generator=generator,
        )
        if oversample_transitions
        else None
    )
    train_loader = DataLoader(
        train_set,
        batch_size=train_batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=config.num_workers,
        drop_last=False,
        generator=generator,
    )
    val_loader = (
        DataLoader(
            val_set,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        )
        if val_set is not None
        else None
    )

    policy = BehaviorCloningPolicy(
        obs_dim=train_set.obs_dim,
        action_dim=train_set.action_dim,
        hidden_dim=config.hidden_dim,
        n_layers=config.n_layers,
        dropout=config.dropout,
        observation_names=obs_names,
        action_names=act_names,
        normalizer=normalizer,
    ).to(device)
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    criterion = torch.nn.MSELoss()

    metadata: dict[str, Any] = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "git_hash": git_hash(),
        "train": asdict(config),
        "n_train_episodes": len(train_ids),
        "train_episode_indices": train_ids,
        "n_val_episodes": len(val_ids),
        "val_episode_indices": val_ids,
        "n_train_samples": len(train_set),
        "n_val_samples": len(val_set) if val_set is not None else 0,
        "n_train_transition_frames": n_transition,
        "transition_frame_fraction": n_transition / max(len(train_set), 1),
        "transition_sampling_mass": transition_mass,
        "obs_dim": train_set.obs_dim,
        "action_dim": train_set.action_dim,
        "observation_names": list(obs_names),
        "action_names": list(act_names),
    }
    (output_dir / "config.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    csv_path = output_dir / "train.csv"
    best_path = checkpoint_dir / "best.pt"
    last_path = checkpoint_dir / "last.pt"
    best_val = float("inf")
    epochs_without_improve = 0

    print(
        "[INFO] BC train: "
        f"episodes={len(train_ids)}/{len(episodes_obs)}, "
        f"samples={len(train_set)}, obs_dim={train_set.obs_dim}, "
        f"last_action_dropout={config.last_action_dropout}, "
        f"last_action_shuffle={config.last_action_shuffle}, "
        f"transition_frames={n_transition}/{len(train_set)} "
        f"({100.0 * n_transition / max(len(train_set), 1):.1f}%), "
        f"window={config.transition_window}, "
        f"weight={config.transition_weight}, "
        f"expected_mass={100.0 * transition_mass:.1f}%, "
        f"device={device}"
    )

    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file, fieldnames=["epoch", "train_mse", "val_mse", "lr"]
        )
        writer.writeheader()
        for epoch in range(1, config.epochs + 1):
            policy.train()
            running = 0.0
            seen = 0
            for batch in train_loader:
                obs = corrupt_last_action(
                    batch["obs"].to(device),
                    config.last_action_dropout,
                    config.last_action_shuffle,
                )
                action = batch["action"].to(device)
                pred = policy(normalizer.normalize_obs(obs))
                target = normalizer.normalize_action(action)
                loss = criterion(pred, target)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                running += float(loss.item()) * int(obs.shape[0])
                seen += int(obs.shape[0])
            train_mse = running / max(seen, 1)

            policy.eval()
            val_mse = (
                _mean_loss(val_loader, policy, device)
                if val_loader is not None
                else train_mse
            )
            lr = optimizer.param_groups[0]["lr"]
            writer.writerow(
                {
                    "epoch": epoch,
                    "train_mse": f"{train_mse:.8f}",
                    "val_mse": f"{val_mse:.8f}",
                    "lr": f"{lr:.8g}",
                }
            )
            csv_file.flush()
            print(
                f"[INFO] epoch {epoch:4d}/{config.epochs}  "
                f"train_mse={train_mse:.6f}  val_mse={val_mse:.6f}"
            )
            policy.save(last_path)
            if val_mse + 1.0e-12 < best_val:
                best_val = val_mse
                epochs_without_improve = 0
                policy.save(best_path)
            else:
                epochs_without_improve += 1
                if (
                    config.early_stop_patience > 0
                    and epochs_without_improve >= config.early_stop_patience
                ):
                    print(
                        f"[INFO] Early stop at epoch {epoch} "
                        f"(best val_mse={best_val:.6f})"
                    )
                    break

    print(f"[INFO] Best checkpoint: {best_path}")
    eval_demo = config.eval_demo
    if eval_demo is None and len(train_ids) == 1:
        eval_demo = train_ids[0]
    if eval_demo is not None:
        if eval_demo not in train_ids:
            print(
                f"[WARNING] eval_demo={eval_demo} was not in the train split "
                f"(first train episode is {train_ids[0]})"
            )
        dump_demo_predictions(
            BehaviorCloningPolicy.load(best_path, device=device),
            config.dataset_path,
            eval_demo,
            device,
        )
    return best_path
