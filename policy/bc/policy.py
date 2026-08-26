"""Single-step MLP behavior-cloning policy."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from .config import ACTION_DIM
from .normalizer import MinMaxNormalizer


def build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    n_layers: int,
    dropout: float,
) -> nn.Sequential:
    """Build a ReLU MLP with optional dropout after hidden layers."""
    if n_layers < 1:
        raise ValueError("n_layers must be at least 1")
    layers: list[nn.Module] = []
    in_dim = input_dim
    for _ in range(n_layers):
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.ReLU())
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        in_dim = hidden_dim
    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)


class BehaviorCloningPolicy(nn.Module):
    """Map a single observation to a normalized relative-TCP action."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int = ACTION_DIM,
        hidden_dim: int = 256,
        n_layers: int = 2,
        dropout: float = 0.1,
        n_obs_steps: int = 1,
        n_action_steps: int = 1,
        clamp_action: bool = False,
        observation_names: tuple[str, ...] | list[str] | None = None,
        action_names: tuple[str, ...] | list[str] | None = None,
        normalizer: MinMaxNormalizer | None = None,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.n_layers = int(n_layers)
        self.dropout = float(dropout)
        self.n_obs_steps = int(n_obs_steps)
        self.n_action_steps = int(n_action_steps)
        self.clamp_action = bool(clamp_action)
        self.observation_names = (
            tuple(observation_names) if observation_names is not None else None
        )
        self.action_names = tuple(action_names) if action_names is not None else None
        self.normalizer = normalizer
        self.net = build_mlp(
            self.obs_dim, self.action_dim, hidden_dim, n_layers, dropout
        )

    def set_normalizer(self, normalizer: MinMaxNormalizer) -> None:
        """Attach the train-split normalizer used at eval time."""
        self.normalizer = normalizer

    def forward(self, obs_norm: torch.Tensor) -> torch.Tensor:
        """Predict a normalized action from a normalized observation.

        Args:
            obs_norm: Shape ``(batch, obs_dim)``.

        Returns:
            Normalized actions, shape ``(batch, action_dim)``.
        """
        return self.net(obs_norm)

    def predict_action(self, obs_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Match the Diffusion Policy eval contract.

        Args:
            obs_dict: Must contain ``obs`` of shape ``(B, T, D)`` or ``(B, D)``.

        Returns:
            Dict with ``action`` of shape ``(B, n_action_steps, action_dim)``.
        """
        if self.normalizer is None:
            raise RuntimeError("Policy has no normalizer; call set_normalizer() or load()")
        if "obs" not in obs_dict:
            raise KeyError("obs_dict must contain key 'obs'")
        obs = obs_dict["obs"]
        if obs.ndim == 2:
            obs = obs.unsqueeze(1)
        if obs.ndim != 3:
            raise ValueError(f"obs must have rank 2 or 3, got shape {tuple(obs.shape)}")
        if obs.shape[-1] != self.obs_dim:
            raise ValueError(
                f"obs last dim is {obs.shape[-1]}, policy obs_dim={self.obs_dim}"
            )
        latest = obs[:, -1]
        obs_norm = self.normalizer.normalize_obs(latest)
        action_norm = self.forward(obs_norm)
        action = self.normalizer.denormalize_action(action_norm)
        if self.clamp_action:
            action = action.clamp(-1.0, 1.0)
        return {"action": action.unsqueeze(1).expand(-1, self.n_action_steps, -1)}

    def config_dict(self) -> dict[str, Any]:
        """Return JSON-serializable constructor arguments."""
        return {
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "hidden_dim": self.hidden_dim,
            "n_layers": self.n_layers,
            "dropout": self.dropout,
            "n_obs_steps": self.n_obs_steps,
            "n_action_steps": self.n_action_steps,
            "clamp_action": self.clamp_action,
            "observation_names": list(self.observation_names)
            if self.observation_names is not None
            else None,
            "action_names": list(self.action_names)
            if self.action_names is not None
            else None,
        }

    def save(self, path: str | Path) -> None:
        """Serialize weights, config, and normalizer without dill."""
        if self.normalizer is None:
            raise RuntimeError("Cannot save a policy without a normalizer")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": self.config_dict(),
                "normalizer": self.normalizer.state_dict(),
            },
            path,
        )

    @classmethod
    def load(
        cls, path: str | Path, device: str | torch.device = "cpu"
    ) -> BehaviorCloningPolicy:
        """Load a checkpoint written by :meth:`save`."""
        checkpoint_path = Path(path).resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if not isinstance(payload, dict) or "config" not in payload:
            raise RuntimeError(
                f"{checkpoint_path} is not a BehaviorCloningPolicy checkpoint"
            )
        config = dict(payload["config"])
        observation_names = config.get("observation_names")
        action_names = config.get("action_names")
        policy = cls(
            obs_dim=int(config["obs_dim"]),
            action_dim=int(config.get("action_dim", ACTION_DIM)),
            hidden_dim=int(config.get("hidden_dim", 256)),
            n_layers=int(config.get("n_layers", 2)),
            dropout=float(config.get("dropout", 0.1)),
            n_obs_steps=int(config.get("n_obs_steps", 1)),
            n_action_steps=int(config.get("n_action_steps", 1)),
            # Checkpoints created before absolute-pose support were delta policies.
            clamp_action=bool(config.get("clamp_action", True)),
            observation_names=tuple(observation_names)
            if observation_names is not None
            else None,
            action_names=tuple(action_names) if action_names is not None else None,
            normalizer=MinMaxNormalizer.from_state(payload["normalizer"]),
        )
        policy.load_state_dict(payload["state_dict"])
        policy.normalizer.to(device)
        policy.to(device)
        policy.eval()
        return policy
