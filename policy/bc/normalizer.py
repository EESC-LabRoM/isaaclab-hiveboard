"""Min-max normalizer that maps each channel independently to ``[-1, 1]``."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


class MinMaxNormalizer:
    """Scale observations and actions with per-channel min/max statistics."""

    def __init__(
        self,
        obs_min: torch.Tensor,
        obs_max: torch.Tensor,
        act_min: torch.Tensor,
        act_max: torch.Tensor,
        eps: float = 1.0e-6,
    ) -> None:
        if obs_min.shape != obs_max.shape:
            raise ValueError("obs_min and obs_max must have the same shape")
        if act_min.shape != act_max.shape:
            raise ValueError("act_min and act_max must have the same shape")
        self.obs_min = obs_min.float()
        self.obs_max = obs_max.float()
        self.act_min = act_min.float()
        self.act_max = act_max.float()
        self.eps = float(eps)

    @classmethod
    def fit(
        cls,
        observations: np.ndarray,
        actions: np.ndarray,
        eps: float = 1.0e-6,
    ) -> MinMaxNormalizer:
        """Fit channel-wise bounds on a train split.

        Args:
            observations: Train observations, shape ``(N, obs_dim)``.
            actions: Train actions, shape ``(N, action_dim)``.
            eps: Minimum range used for zero-variance channels.

        Returns:
            Fitted normalizer.
        """
        if observations.ndim != 2 or actions.ndim != 2:
            raise ValueError("observations and actions must be rank-2 arrays")
        obs_min = observations.min(axis=0)
        obs_max = observations.max(axis=0)
        act_min = actions.min(axis=0)
        act_max = actions.max(axis=0)
        return cls(
            obs_min=torch.from_numpy(np.asarray(obs_min, dtype=np.float32)),
            obs_max=torch.from_numpy(np.asarray(obs_max, dtype=np.float32)),
            act_min=torch.from_numpy(np.asarray(act_min, dtype=np.float32)),
            act_max=torch.from_numpy(np.asarray(act_max, dtype=np.float32)),
            eps=eps,
        )

    def to(self, device: torch.device | str) -> MinMaxNormalizer:
        """Move statistic tensors onto ``device`` and return self."""
        self.obs_min = self.obs_min.to(device)
        self.obs_max = self.obs_max.to(device)
        self.act_min = self.act_min.to(device)
        self.act_max = self.act_max.to(device)
        return self

    def normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Map observations to ``[-1, 1]``."""
        return self._normalize(obs, self.obs_min, self.obs_max)

    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        """Map actions to ``[-1, 1]``."""
        return self._normalize(action, self.act_min, self.act_max)

    def denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        """Invert action normalization."""
        return self._denormalize(action, self.act_min, self.act_max)

    def denormalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Invert observation normalization."""
        return self._denormalize(obs, self.obs_min, self.obs_max)

    def state_dict(self) -> dict[str, Any]:
        """Return a CPU-serializable state."""
        return {
            "obs_min": self.obs_min.detach().cpu(),
            "obs_max": self.obs_max.detach().cpu(),
            "act_min": self.act_min.detach().cpu(),
            "act_max": self.act_max.detach().cpu(),
            "eps": self.eps,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> MinMaxNormalizer:
        """Restore a normalizer from :meth:`state_dict`."""
        return cls(
            obs_min=torch.as_tensor(state["obs_min"], dtype=torch.float32),
            obs_max=torch.as_tensor(state["obs_max"], dtype=torch.float32),
            act_min=torch.as_tensor(state["act_min"], dtype=torch.float32),
            act_max=torch.as_tensor(state["act_max"], dtype=torch.float32),
            eps=float(state.get("eps", 1.0e-6)),
        )

    def _normalize(
        self, value: torch.Tensor, low: torch.Tensor, high: torch.Tensor
    ) -> torch.Tensor:
        scale = (high - low).clamp_min(self.eps)
        return 2.0 * (value - low) / scale - 1.0

    def _denormalize(
        self, value: torch.Tensor, low: torch.Tensor, high: torch.Tensor
    ) -> torch.Tensor:
        scale = (high - low).clamp_min(self.eps)
        return (value + 1.0) * 0.5 * scale + low
