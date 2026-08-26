"""HDF5 loader for cleaned Spot ball-valve demonstrations."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .config import (
    ACTION_DIM,
    ACTION_NAMES,
    OBSERVATION_NAMES,
    STUDENT_BODY_DIM,
)
from .normalizer import MinMaxNormalizer


def _demo_sort_key(name: str) -> int:
    return int(name.removeprefix("demo_"))


def last_action_from_actions(actions: np.ndarray) -> np.ndarray:
    """Shift actions by one step; zeros at the start of the episode.

    Args:
        actions: Normalized actions, shape ``(T, action_dim)``.

    Returns:
        Last-action channels, shape ``(T, action_dim)``.
    """
    last_action = np.zeros_like(actions)
    if len(actions) > 1:
        last_action[1:] = actions[:-1]
    return last_action


def to_student_observations(
    observations: np.ndarray, actions: np.ndarray
) -> np.ndarray:
    """Keep a regenerated 43-D student recording and refresh last-action.

    The body prefix is proprio, object root, handle pose, and the open/close
    command. Last-action is rebuilt from the action stream so t=0 is zeros.

    Args:
        observations: Recorded student observations, shape ``(T, 43)``.
        actions: Normalized actions, shape ``(T, 7)``.

    Returns:
        Student observations, shape ``(T, 43)``.
    """
    dim = observations.shape[-1]
    expected = len(OBSERVATION_NAMES)
    if dim != expected:
        raise ValueError(
            f"Observations are {dim}-D, expected {expected}-D regenerated "
            "student recordings (EE, arm, object root, handle_pose_ee, "
            "valve_task_direction, last_action). Recollect after adding "
            "the open/close command to DiffusionPolicyCfg."
        )
    body = observations[:, :STUDENT_BODY_DIM]
    last_action = last_action_from_actions(actions)
    return np.concatenate((body, last_action), axis=-1).astype(np.float32)


def parse_name_attribute(value: object) -> tuple[str, ...] | None:
    """Parse an HDF5 attribute that stores a JSON list of channel names."""
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8")
    if isinstance(value, str):
        value = json.loads(value)
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"Unexpected channel-name attribute type: {type(value)!r}")
    return tuple(str(item) for item in value)


def load_episode_arrays(
    dataset_path: str | Path,
) -> tuple[list[np.ndarray], list[np.ndarray], tuple[str, ...], tuple[str, ...]]:
    """Load every valid demonstration from a cleaned HDF5 file.

    Args:
        dataset_path: Path to a file written by ``analyze_dataset.py``.

    Returns:
        Observation episodes, action episodes, observation names, action names.

    Raises:
        FileNotFoundError: If the path does not exist.
        KeyError: If the file has no ``data`` group.
        RuntimeError: If no demonstration is usable.
    """
    path = Path(dataset_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Dataset does not exist: {path}")

    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    observation_names: tuple[str, ...] | None = None
    action_names: tuple[str, ...] | None = None
    skipped = 0
    with h5py.File(path, "r") as file:
        if "data" not in file:
            raise KeyError(f"{path} has no root 'data' group")
        group = file["data"]
        observation_names = parse_name_attribute(group.attrs.get("observation_names"))
        action_names = parse_name_attribute(group.attrs.get("action_names"))
        demo_names = sorted(group.keys(), key=_demo_sort_key)
        for name in demo_names:
            demo = group[name]
            if "observations" not in demo or "actions" not in demo:
                skipped += 1
                continue
            obs = np.asarray(demo["observations"], dtype=np.float32)
            act = np.asarray(demo["actions"], dtype=np.float32)
            if obs.ndim != 2 or act.ndim != 2:
                skipped += 1
                continue
            if act.shape[-1] <= 0:
                raise RuntimeError(f"{name} has no action channels")
            if len(obs) == 0 or len(obs) != len(act):
                skipped += 1
                continue
            if not np.isfinite(obs).all() or not np.isfinite(act).all():
                skipped += 1
                continue
            observations.append(
                to_student_observations(obs, act)
                if obs.shape[-1] == len(OBSERVATION_NAMES)
                and act.shape[-1] == ACTION_DIM
                else obs.astype(np.float32)
            )
            actions.append(act)

    if not observations:
        raise RuntimeError(f"No valid demonstrations in {path}")

    obs_dim = observations[0].shape[-1]
    if any(episode.shape[-1] != obs_dim for episode in observations):
        raise RuntimeError("Demonstrations do not share a single observation dimension")
    action_dim = actions[0].shape[-1]
    if any(episode.shape[-1] != action_dim for episode in actions):
        raise RuntimeError("Demonstrations do not share a single action dimension")
    if observation_names is None:
        observation_names = (
            OBSERVATION_NAMES
            if obs_dim == len(OBSERVATION_NAMES)
            else tuple(f"observation_{index}" for index in range(obs_dim))
        )
    elif len(observation_names) != obs_dim:
        raise RuntimeError(
            f"observation_names has {len(observation_names)} entries, expected {obs_dim}"
        )

    if action_names is None:
        action_names = (
            ACTION_NAMES
            if action_dim == ACTION_DIM
            else tuple(f"action_{index}" for index in range(action_dim))
        )
    elif len(action_names) != action_dim:
        raise RuntimeError(
            f"action_names has {len(action_names)} entries, expected {action_dim}"
        )

    if skipped:
        print(f"[INFO] Skipped {skipped} invalid demonstrations in {path}")
    return observations, actions, observation_names, action_names


def detect_action_transitions(
    actions: np.ndarray,
    *,
    ignore_prefix: int = 2,
    action_delta_threshold: float = 0.5,
) -> np.ndarray:
    """Mark the few frames where the expert switches motion (grasp / start rotating).

    Open vs close is already balanced in the recordings. The rare event is the
    phase change inside each demo: gripper slam and a large jump in the 7-D
    action. The first ``ignore_prefix`` steps are skipped so the dummy zero
    action at t=0 is not treated as a transition.

    Args:
        actions: Episode actions, shape ``(T, action_dim)``.
        ignore_prefix: Leading steps ignored for action-delta hits.
        action_delta_threshold: Minimum ``||a_t - a_{t-1}||`` to count as a
            jump. Gripper open→close is ``≈ 2`` and always fires.

    Returns:
        Boolean mask of shape ``(T,)``.
    """
    if actions.ndim != 2:
        raise ValueError(f"actions must be rank-2, got shape {actions.shape}")
    length = int(actions.shape[0])
    mask = np.zeros(length, dtype=bool)
    if length < 2:
        return mask

    gripper_open = actions[:, 0] >= 0.0
    mask[1:] |= gripper_open[1:] != gripper_open[:-1]

    delta = np.linalg.norm(np.diff(actions, axis=0), axis=1)
    hits = np.where(delta >= action_delta_threshold)[0] + 1
    hits = hits[hits >= ignore_prefix]
    mask[hits] = True
    return mask


def dilate_bool_mask(mask: np.ndarray, window: int) -> np.ndarray:
    """Expand True marks by ``window`` steps on each side, within the episode."""
    if window < 0:
        raise ValueError(f"window must be >= 0, got {window}")
    if mask.dtype != bool:
        mask = mask.astype(bool)
    if window == 0 or not np.any(mask):
        return mask.copy()
    dilated = mask.copy()
    for index in np.flatnonzero(mask):
        start = max(0, int(index) - window)
        end = min(len(mask), int(index) + window + 1)
        dilated[start:end] = True
    return dilated


def transition_sample_weights(
    action_episodes: list[np.ndarray],
    *,
    weight: float,
    window: int = 4,
    ignore_prefix: int = 2,
    action_delta_threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-step sampling weights that oversample dilated transition windows.

    Non-transition steps keep weight 1. ``weight <= 1`` returns uniform weights
    but still reports the transition mask so logs stay meaningful.

    Args:
        action_episodes: Action arrays for the split, one per episode.
        weight: Multiplier on dilated transition frames.
        window: Extra frames on each side of a detected change-point.
        ignore_prefix: See :func:`detect_action_transitions`.
        action_delta_threshold: See :func:`detect_action_transitions`.

    Returns:
        Concatenated weights, shape ``(N,)``, and the dilated boolean mask.
    """
    if weight <= 0.0:
        raise ValueError(f"transition weight must be positive, got {weight}")
    masks = [
        dilate_bool_mask(
            detect_action_transitions(
                episode,
                ignore_prefix=ignore_prefix,
                action_delta_threshold=action_delta_threshold,
            ),
            window,
        )
        for episode in action_episodes
    ]
    mask = (
        np.concatenate(masks, axis=0)
        if masks
        else np.zeros(0, dtype=bool)
    )
    weights = np.ones(len(mask), dtype=np.float64)
    if weight > 1.0:
        weights[mask] = float(weight)
    return weights, mask


class SpotBallValveDataset(Dataset):
    """Flat (observation, action) pairs from a train or validation episode split."""

    def __init__(
        self,
        observations: list[np.ndarray],
        actions: list[np.ndarray],
        episode_indices: list[int],
        observation_names: tuple[str, ...],
        action_names: tuple[str, ...] = ACTION_NAMES,
    ) -> None:
        if not episode_indices:
            raise ValueError("episode_indices must be non-empty")
        obs_arrays = [observations[index] for index in episode_indices]
        act_arrays = [actions[index] for index in episode_indices]
        self.observations = np.concatenate(obs_arrays, axis=0)
        self.actions = np.concatenate(act_arrays, axis=0)
        self.episode_actions = act_arrays
        self.episode_lengths = np.asarray(
            [len(episode) for episode in act_arrays], dtype=np.int32
        )
        self.observation_names = observation_names
        self.action_names = action_names
        self.episode_indices = list(episode_indices)
        self.obs_dim = int(self.observations.shape[-1])
        self.action_dim = int(self.actions.shape[-1])

    def transition_sample_weights(
        self,
        weight: float,
        window: int = 4,
        ignore_prefix: int = 2,
        action_delta_threshold: float = 0.5,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sampling weights aligned with :meth:`__getitem__` indices."""
        return transition_sample_weights(
            self.episode_actions,
            weight=weight,
            window=window,
            ignore_prefix=ignore_prefix,
            action_delta_threshold=action_delta_threshold,
        )

    @classmethod
    def from_hdf5(
        cls,
        dataset_path: str | Path,
        split: str = "train",
        val_ratio: float = 0.05,
        seed: int = 42,
        max_train_episodes: int | None = None,
    ) -> SpotBallValveDataset:
        """Load an HDF5 file and return one split.

        Args:
            dataset_path: Cleaned demonstration file.
            split: ``train`` or ``val``.
            val_ratio: Fraction of episodes reserved for validation.
            seed: Episode shuffle seed.
            max_train_episodes: Optional cap on the train split size.

        Returns:
            Dataset for the requested split.

        Raises:
            ValueError: If ``split`` is unknown or a split is empty.
        """
        if split not in {"train", "val"}:
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")
        episodes_obs, episodes_act, obs_names, act_names = load_episode_arrays(
            dataset_path
        )
        train_ids, val_ids = split_episode_indices(
            len(episodes_obs), val_ratio=val_ratio, seed=seed
        )
        if split == "train":
            if max_train_episodes is not None:
                train_ids = train_ids[: max(1, int(max_train_episodes))]
            indices = train_ids
        else:
            indices = val_ids
        if not indices:
            raise ValueError(f"{split} split is empty")
        return cls(episodes_obs, episodes_act, indices, obs_names, act_names)

    def get_normalizer(self) -> MinMaxNormalizer:
        """Fit a min-max normalizer on this split."""
        return MinMaxNormalizer.fit(self.observations, self.actions)

    def __len__(self) -> int:
        return int(self.observations.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "obs": torch.from_numpy(self.observations[index].copy()),
            "action": torch.from_numpy(self.actions[index].copy()),
        }


def split_episode_indices(
    n_episodes: int, val_ratio: float, seed: int
) -> tuple[list[int], list[int]]:
    """Split episode indices by episode, not by timestep.

    A single-episode dataset is assigned entirely to train so overfit checks
    remain possible.
    """
    if n_episodes <= 0:
        raise ValueError("n_episodes must be positive")
    order = np.random.default_rng(seed).permutation(n_episodes).tolist()
    if n_episodes == 1 or val_ratio <= 0.0:
        return order, []
    n_val = max(1, int(round(n_episodes * val_ratio)))
    n_val = min(n_val, n_episodes - 1)
    return order[n_val:], order[:n_val]
