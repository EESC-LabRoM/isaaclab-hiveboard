# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Inspect and aggregate the HDF5 datasets robomimic trains on.

DAgger produces one dataset per round. Robomimic trains on a single file, so
rounds have to be concatenated; that is what :func:`merge_datasets` does, and
it is the only place demo numbering is allowed to be rewritten.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import h5py


@dataclass
class DatasetStats:
    """Summary of a robomimic-layout HDF5 dataset."""

    path: str
    num_demos: int
    num_samples: int
    num_successes: int
    obs_keys: dict[str, tuple[int, ...]] = field(default_factory=dict)
    action_dim: int | None = None
    env_name: str | None = None

    @property
    def success_rate(self) -> float:
        """Fraction of demos flagged successful, or NaN when none are flagged."""
        if self.num_demos == 0:
            return float("nan")
        return self.num_successes / self.num_demos

    def describe(self) -> str:
        """Render a one-block human-readable summary."""
        lines = [
            f"{self.path}",
            f"  env            : {self.env_name}",
            f"  demos          : {self.num_demos}",
            f"  samples        : {self.num_samples}",
            f"  successful     : {self.num_successes} ({self.success_rate:.1%})",
            f"  action dim     : {self.action_dim}",
            "  observations   :",
        ]
        lines.extend(f"    {key:<18} {shape}" for key, shape in sorted(self.obs_keys.items()))
        return "\n".join(lines)


def dataset_stats(path: str) -> DatasetStats:
    """Summarize a dataset without loading its arrays into memory.

    Args:
        path: Path to the HDF5 dataset.

    Returns:
        The dataset summary.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file has no ``data`` group.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    with h5py.File(path, "r") as handle:
        if "data" not in handle:
            raise ValueError(f"{path} has no 'data' group; it is not a robomimic-layout dataset.")
        data = handle["data"]

        env_name = None
        if "env_args" in data.attrs:
            env_name = json.loads(data.attrs["env_args"]).get("env_name")

        demo_names = [name for name in data if name.startswith("demo_")]
        num_samples = 0
        num_successes = 0
        obs_keys: dict[str, tuple[int, ...]] = {}
        action_dim = None

        for name in demo_names:
            demo = data[name]
            num_samples += int(demo.attrs.get("num_samples", 0))
            num_successes += int(bool(demo.attrs.get("success", False)))
            if not obs_keys and "obs" in demo:
                obs_keys = {key: tuple(demo["obs"][key].shape[1:]) for key in demo["obs"]}
            if action_dim is None and "actions" in demo:
                action_dim = int(demo["actions"].shape[-1])

    return DatasetStats(
        path=path,
        num_demos=len(demo_names),
        num_samples=num_samples,
        num_successes=num_successes,
        obs_keys=obs_keys,
        action_dim=action_dim,
        env_name=env_name,
    )


def merge_datasets(inputs: list[str], output: str, env_name: str | None = None) -> DatasetStats:
    """Concatenate robomimic datasets into one file, renumbering demos.

    Args:
        inputs: Datasets to merge, in order. Later files keep their contents;
            only the ``demo_<i>`` group names are rewritten so ids stay unique.
        output: Path of the merged dataset. Overwritten if it exists.
        env_name: Environment name for ``env_args``. Defaults to the first
            input's, so a merged DAgger dataset stays loadable by robomimic.

    Returns:
        Stats for the merged dataset.

    Raises:
        ValueError: If ``inputs`` is empty, or the inputs disagree on their
            observation keys, which would silently produce an untrainable file.
    """
    if not inputs:
        raise ValueError("merge_datasets needs at least one input dataset.")
    missing = [path for path in inputs if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(f"Datasets not found: {missing}")

    output_dir = os.path.dirname(os.path.abspath(output))
    os.makedirs(output_dir, exist_ok=True)

    reference_keys: set[str] | None = None
    demo_index = 0
    total_samples = 0

    with h5py.File(output, "w") as dst:
        dst.attrs["format_version"] = 1
        data_group = dst.create_group("data")

        for path in inputs:
            with h5py.File(path, "r") as src:
                src_data = src["data"]
                if env_name is None and "env_args" in src_data.attrs:
                    env_name = json.loads(src_data.attrs["env_args"]).get("env_name")

                for name in sorted(src_data, key=_demo_sort_key):
                    if not name.startswith("demo_"):
                        continue
                    demo = src_data[name]

                    keys = set(demo["obs"].keys()) if "obs" in demo else set()
                    if reference_keys is None:
                        reference_keys = keys
                    elif keys != reference_keys:
                        raise ValueError(
                            f"{path}:{name} has observation keys {sorted(keys)}, but earlier demos have "
                            f"{sorted(reference_keys)}. Merging these would produce a dataset robomimic "
                            "cannot train on. Re-collect the rounds with the same observation group."
                        )

                    src_data.copy(demo, data_group, name=f"demo_{demo_index}")
                    total_samples += int(demo.attrs.get("num_samples", 0))
                    demo_index += 1

        data_group.attrs["total"] = total_samples
        data_group.attrs["env_args"] = json.dumps({"env_name": env_name or "", "type": 2})

    return dataset_stats(output)


def rotate_recorder_dataset(env, dataset_dir: str, dataset_name: str) -> str:
    """Point a live recorder at a new dataset file and drop in-flight episodes.

    DAgger writes one dataset per round, but the recorder opens its file in
    ``RecorderManager.__init__`` and a simulation can only host one
    :class:`InteractiveScene`, so rebuilding the environment per round is not an
    option. This closes the open file, opens the next one and clears the episode
    buffers and counters, which does mean reaching into the manager's internals.

    Any episode still in progress is dropped, so half-recorded trajectories from
    the previous round cannot leak into the new file. Callers that want those
    partial trajectories kept should call
    ``recorder_manager.export_episodes()`` before rotating - for behaviour
    cloning they are perfectly usable, since every sample is an independent
    (state, expert action) pair and truncation costs only the episode's tail.

    Args:
        env: The environment whose recorder should be rotated.
        dataset_dir: Directory for the new HDF5 file.
        dataset_name: Filename stem, without the ``.hdf5`` extension.

    Returns:
        Absolute path of the new dataset file.

    Raises:
        RuntimeError: If the environment has no active recorder terms.
    """
    from isaaclab.utils.datasets import EpisodeData

    base = env.unwrapped
    recorder = base.recorder_manager
    if len(recorder.active_terms) == 0:
        raise RuntimeError("Environment has no active recorder terms to rotate.")

    dataset_dir = os.path.abspath(dataset_dir)
    os.makedirs(dataset_dir, exist_ok=True)
    path = os.path.join(dataset_dir, f"{dataset_name}.hdf5")

    recorder.cfg.dataset_export_dir_path = dataset_dir
    recorder.cfg.dataset_filename = dataset_name

    if recorder._dataset_file_handler is not None:
        recorder._dataset_file_handler.close()
        handler = recorder.cfg.dataset_file_handler_class_type()
        handler.create(os.path.join(dataset_dir, dataset_name), env_name=getattr(base.cfg, "env_name", None))
        recorder._dataset_file_handler = handler

    recorder._episodes = {env_id: EpisodeData() for env_id in range(base.num_envs)}
    recorder._exported_successful_episode_count = {}
    recorder._exported_failed_episode_count = {}
    return path


def _demo_sort_key(name: str) -> tuple[int, object]:
    """Sort ``demo_10`` after ``demo_9`` instead of lexicographically."""
    if name.startswith("demo_"):
        suffix = name[len("demo_") :]
        if suffix.isdigit():
            return (0, int(suffix))
    return (1, name)
