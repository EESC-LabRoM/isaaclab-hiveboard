#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Clean Spot ball-valve demonstrations for low-dimensional Diffusion Policy training.

The collector stores named recorder terms and relative scene state in an Isaac Lab
HDF5 file. This script validates those trajectories and writes the flat
``observations`` and ``actions`` datasets expected by the Diffusion Policy dataset
loader used by AutoALMA.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import h5py
import numpy as np

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "isaaclab-experiments-matplotlib")
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# Active observation channels written to the cleaned dataset.
# Comment out any observation channel below to exclude it from the final dataset:
OBSERVATION_NAMES = (
    # End-effector pose in robot base frame (position + wxyz quaternion)
    "ee_pos_b_x",
    "ee_pos_b_y",
    "ee_pos_b_z",
    "ee_quat_b_w",
    "ee_quat_b_x",
    "ee_quat_b_y",
    "ee_quat_b_z",
    # Arm & gripper joint positions
    "arm_joint_pos_0",
    "arm_joint_pos_1",
    "arm_joint_pos_2",
    "arm_joint_pos_3",
    "arm_joint_pos_4",
    "arm_joint_pos_5",
    "gripper_joint_pos",
    # Arm & gripper joint velocities
    "arm_joint_vel_0",
    "arm_joint_vel_1",
    "arm_joint_vel_2",
    "arm_joint_vel_3",
    "arm_joint_vel_4",
    "arm_joint_vel_5",
    "gripper_joint_vel",
    # Valve root pose in robot base frame (position + wxyz quaternion)
    "object_root_pos_b_x",
    "object_root_pos_b_y",
    "object_root_pos_b_z",
    "object_root_quat_b_w",
    "object_root_quat_b_x",
    "object_root_quat_b_y",
    "object_root_quat_b_z",
    # Valve joint position and velocity
    "object_joint_pos",
    "object_joint_vel",
    # Valve task goals and normalized state
    "valve_task_goal",
    "valve_joint_pos_normalized",
    "valve_joint_des_normalized",
)

ACTION_NAMES = (
    "gripper_command",
    "delta_pos_x_normalized",
    "delta_pos_y_normalized",
    "delta_pos_z_normalized",
    "delta_axis_angle_x_normalized",
    "delta_axis_angle_y_normalized",
    "delta_axis_angle_z_normalized",
)

REQUIRED_DATASETS = (
    "actions",
    "processed_actions",
    "obs/ee_pose_b",
    "obs/arm_joint_pos",
    "obs/arm_joint_vel",
    "obs/object_joint_pos",
    "obs/object_joint_vel",
    "obs/valve_task",
    "states/articulation/robot/root_pose",
    "states/articulation/ball_valve/root_pose",
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Clean Spot ball-valve HDF5 demonstrations for Diffusion Policy."
    )
    parser.add_argument(
        "--input_path",
        type=Path,
        default=None,
        help="Raw HDF5 dataset. The newest eligible file is used when omitted.",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=None,
        help="Cleaned HDF5 path. Defaults to '<input>_diffusion.hdf5'.",
    )
    parser.add_argument(
        "--metadata_path",
        type=Path,
        default=None,
        help="Metadata JSON path. Defaults beside the output dataset.",
    )
    parser.add_argument(
        "--plot_path",
        type=Path,
        default=None,
        help="Summary diagnostic plot path. Defaults beside the output dataset.",
    )
    parser.add_argument(
        "--valve_cloud_path",
        type=Path,
        default=None,
        help="Valve reset-pose cloud plot path. Defaults beside the output dataset.",
    )
    parser.add_argument(
        "--action_plot_path",
        type=Path,
        default=None,
        help="First-demo action plot path. Defaults beside the output dataset.",
    )
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=Path("logs/recorded_datasets"),
        help="Directory searched when --input_path is omitted.",
    )
    parser.add_argument(
        "--control_frequency_hz",
        type=float,
        default=20.0,
        help="Action sampling frequency recorded in metadata (default: 20 Hz).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output and diagnostic files.",
    )
    return parser.parse_args()


def find_latest_dataset(dataset_dir: Path) -> Path:
    """Find the most recently modified eligible raw HDF5 dataset.

    Args:
        dataset_dir: Directory containing recorded datasets.

    Returns:
        Path to the newest raw HDF5 file.

    Raises:
        FileNotFoundError: If no eligible HDF5 file exists.
    """
    excluded_suffixes = ("_diffusion.hdf5", "_cleaned.hdf5", "_failed.hdf5")
    candidates = [
        path
        for path in dataset_dir.glob("*.hdf5")
        if not path.name.endswith(excluded_suffixes) and path.stat().st_size > 96
    ]
    if not candidates:
        raise FileNotFoundError(f"No raw HDF5 datasets found in {dataset_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def resolve_paths(args: argparse.Namespace) -> dict[str, Path]:
    """Resolve input and derived output paths.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Mapping containing all input and output paths.
    """
    input_path = args.input_path or find_latest_dataset(args.dataset_dir)
    input_path = input_path.resolve()
    output_path = args.output_path or input_path.with_name(
        f"{input_path.stem}_diffusion.hdf5"
    )
    output_path = output_path.resolve()
    output_stem = output_path.with_suffix("")
    return {
        "input": input_path,
        "output": output_path,
        "metadata": (args.metadata_path or Path(f"{output_stem}_metadata.json")).resolve(),
        "plot": (args.plot_path or Path(f"{output_stem}_diagnostics.png")).resolve(),
        "valve_cloud": (
            args.valve_cloud_path or Path(f"{output_stem}_valve_cloud.png")
        ).resolve(),
        "action_plot": (
            args.action_plot_path or Path(f"{output_stem}_actions.png")
        ).resolve(),
    }


def prepare_outputs(paths: dict[str, Path], overwrite: bool) -> None:
    """Validate output paths and create their parent directories.

    Args:
        paths: Resolved input and output paths.
        overwrite: Whether existing outputs may be replaced.

    Raises:
        FileNotFoundError: If the input dataset does not exist.
        FileExistsError: If an output exists and overwrite is disabled.
        ValueError: If input and cleaned output resolve to the same path.
    """
    if not paths["input"].is_file():
        raise FileNotFoundError(f"Input dataset does not exist: {paths['input']}")
    if paths["input"] == paths["output"]:
        raise ValueError("Input and output HDF5 paths must be different")
    for key, path in paths.items():
        if key == "input":
            continue
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"Refusing to overwrite {path}; pass --overwrite to replace it"
            )
        path.parent.mkdir(parents=True, exist_ok=True)


def canonicalize_quaternions(quaternions: np.ndarray) -> np.ndarray:
    """Select a temporally continuous quaternion sign with non-negative initial w.

    Args:
        quaternions: Quaternion array in ``wxyz`` order, shape ``(T, 4)``.

    Returns:
        Canonicalized float32 quaternion array, shape ``(T, 4)``.
    """
    result = np.asarray(quaternions, dtype=np.float32).copy()
    if result[0, 0] < 0.0:
        result[0] *= -1.0
    for index in range(1, len(result)):
        if np.dot(result[index - 1], result[index]) < 0.0:
            result[index] *= -1.0
    return result


def quaternion_conjugate(quaternion: np.ndarray) -> np.ndarray:
    """Return conjugates of ``wxyz`` quaternions.

    Args:
        quaternion: Quaternion array, shape ``(..., 4)``.

    Returns:
        Conjugated quaternion array, shape ``(..., 4)``.
    """
    result = quaternion.copy()
    result[..., 1:] *= -1.0
    return result


def quaternion_multiply(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Multiply two arrays of ``wxyz`` quaternions.

    Args:
        lhs: Left quaternion array, shape ``(..., 4)``.
        rhs: Right quaternion array, shape ``(..., 4)``.

    Returns:
        Hamilton product array, shape ``(..., 4)``.
    """
    lw, lx, ly, lz = np.moveaxis(lhs, -1, 0)
    rw, rx, ry, rz = np.moveaxis(rhs, -1, 0)
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )


def quaternion_rotate(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Rotate vectors using ``wxyz`` unit quaternions.

    Args:
        quaternion: Quaternion array, shape ``(T, 4)``.
        vector: Vector array, shape ``(T, 3)``.

    Returns:
        Rotated vector array, shape ``(T, 3)``.
    """
    xyz = quaternion[:, 1:]
    twice_cross = 2.0 * np.cross(xyz, vector)
    return vector + quaternion[:, :1] * twice_cross + np.cross(xyz, twice_cross)


def object_pose_in_robot_base(robot_pose: np.ndarray, object_pose: np.ndarray) -> np.ndarray:
    """Express relative-scene object root poses in each robot base frame.

    Args:
        robot_pose: Robot root pose ``(position, quaternion wxyz)``, shape ``(T, 7)``.
        object_pose: Object root pose ``(position, quaternion wxyz)``, shape ``(T, 7)``.

    Returns:
        Object pose in robot base, shape ``(T, 7)``.
    """
    robot_inverse_quat = quaternion_conjugate(robot_pose[:, 3:7])
    position = quaternion_rotate(
        robot_inverse_quat, object_pose[:, :3] - robot_pose[:, :3]
    )
    orientation = quaternion_multiply(robot_inverse_quat, object_pose[:, 3:7])
    orientation = canonicalize_quaternions(orientation)
    return np.concatenate((position, orientation), axis=-1).astype(np.float32)


def read_dataset(group: h5py.Group, path: str) -> np.ndarray:
    """Read a required HDF5 dataset as float32.

    Args:
        group: Demonstration HDF5 group.
        path: Slash-delimited dataset path.

    Returns:
        Dataset values as a float32 NumPy array.

    Raises:
        KeyError: If the path is absent or resolves to a group.
    """
    if path not in group or not isinstance(group[path], h5py.Dataset):
        raise KeyError(f"missing dataset '{path}'")
    return np.asarray(group[path][()], dtype=np.float32)


def extract_raw_observations(demo: h5py.Group) -> dict[str, np.ndarray]:
    """Read and canonicalize all raw demonstration observations and states.

    Args:
        demo: Raw demonstration HDF5 group.

    Returns:
        Dictionary mapping component names to their NumPy arrays.
    """
    values = {path: read_dataset(demo, path) for path in REQUIRED_DATASETS}
    ee_pose = values["obs/ee_pose_b"].copy()
    ee_pose[:, 3:7] = canonicalize_quaternions(ee_pose[:, 3:7])
    object_pose_b = object_pose_in_robot_base(
        values["states/articulation/robot/root_pose"],
        values["states/articulation/ball_valve/root_pose"],
    )
    return {
        "ee_pose": ee_pose,
        "arm_joint_pos": values["obs/arm_joint_pos"],
        "arm_joint_vel": values["obs/arm_joint_vel"],
        "object_pose_b": object_pose_b,
        "object_joint_pos": values["obs/object_joint_pos"],
        "object_joint_vel": values["obs/object_joint_vel"],
        "valve_task": values["obs/valve_task"],
        "actions": values["actions"],
        "processed_actions": values["processed_actions"],
    }


def extract_named_channels(raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Map every individual observation channel name to its (T, 1) array.

    Args:
        raw: Raw demonstration components dictionary from :func:`extract_raw_observations`.

    Returns:
        Dictionary mapping each observation channel name to a 2D array of shape ``(T, 1)``.
    """
    ee_pose = raw["ee_pose"]
    arm_pos = raw["arm_joint_pos"]
    arm_vel = raw["arm_joint_vel"]
    obj_pose = raw["object_pose_b"]
    obj_pos = raw["object_joint_pos"]
    obj_vel = raw["object_joint_vel"]
    valve_task = raw["valve_task"]

    if obj_pos.ndim == 1:
        obj_pos = obj_pos[:, None]
    if obj_vel.ndim == 1:
        obj_vel = obj_vel[:, None]

    return {
        "ee_pos_b_x": ee_pose[:, 0:1],
        "ee_pos_b_y": ee_pose[:, 1:2],
        "ee_pos_b_z": ee_pose[:, 2:3],
        "ee_quat_b_w": ee_pose[:, 3:4],
        "ee_quat_b_x": ee_pose[:, 4:5],
        "ee_quat_b_y": ee_pose[:, 5:6],
        "ee_quat_b_z": ee_pose[:, 6:7],
        "arm_joint_pos_0": arm_pos[:, 0:1],
        "arm_joint_pos_1": arm_pos[:, 1:2],
        "arm_joint_pos_2": arm_pos[:, 2:3],
        "arm_joint_pos_3": arm_pos[:, 3:4],
        "arm_joint_pos_4": arm_pos[:, 4:5],
        "arm_joint_pos_5": arm_pos[:, 5:6],
        "gripper_joint_pos": arm_pos[:, 6:7],
        "arm_joint_vel_0": arm_vel[:, 0:1],
        "arm_joint_vel_1": arm_vel[:, 1:2],
        "arm_joint_vel_2": arm_vel[:, 2:3],
        "arm_joint_vel_3": arm_vel[:, 3:4],
        "arm_joint_vel_4": arm_vel[:, 4:5],
        "arm_joint_vel_5": arm_vel[:, 5:6],
        "gripper_joint_vel": arm_vel[:, 6:7],
        "object_root_pos_b_x": obj_pose[:, 0:1],
        "object_root_pos_b_y": obj_pose[:, 1:2],
        "object_root_pos_b_z": obj_pose[:, 2:3],
        "object_root_quat_b_w": obj_pose[:, 3:4],
        "object_root_quat_b_x": obj_pose[:, 4:5],
        "object_root_quat_b_y": obj_pose[:, 5:6],
        "object_root_quat_b_z": obj_pose[:, 6:7],
        "object_joint_pos": obj_pos[:, 0:1],
        "object_joint_vel": obj_vel[:, 0:1],
        "valve_task_goal": valve_task[:, 0:1],
        "valve_joint_pos_normalized": valve_task[:, 1:2],
        "valve_joint_des_normalized": valve_task[:, 2:3],
    }


def build_training_arrays(
    demo: h5py.Group,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """Extract raw components and assemble selected training observations.

    Args:
        demo: Raw demonstration HDF5 group.

    Returns:
        Tuple of raw components dictionary, flat observations array of shape
        ``(T, len(OBSERVATION_NAMES))``, normalized actions, and processed actions.
    """
    raw = extract_raw_observations(demo)
    channels = extract_named_channels(raw)

    if not OBSERVATION_NAMES:
        raise ValueError("OBSERVATION_NAMES cannot be empty")
    for name in OBSERVATION_NAMES:
        if name not in channels:
            raise KeyError(f"Unknown observation channel '{name}' in OBSERVATION_NAMES")

    observations = np.concatenate(
        [channels[name] for name in OBSERVATION_NAMES], axis=-1
    ).astype(np.float32)
    return raw, observations, raw["actions"], raw["processed_actions"]


def validate_demo(
    demo: h5py.Group,
    raw: dict[str, np.ndarray],
    observations: np.ndarray,
    actions: np.ndarray,
    processed_actions: np.ndarray,
) -> list[str]:
    """Return reasons that a demonstration is unsuitable for training.

    Args:
        demo: Raw demonstration HDF5 group.
        raw: Raw demonstration components dictionary.
        observations: Selected training observations, shape ``(T, len(OBSERVATION_NAMES))``.
        actions: Normalized actions, shape ``(T, 7)``.
        processed_actions: Physical actions, shape ``(T, 7)``.

    Returns:
        Empty list for a valid demonstration, otherwise rejection reasons.
    """
    reasons: list[str] = []
    lengths = {
        path: int(demo[path].shape[0])
        for path in REQUIRED_DATASETS
        if path in demo and isinstance(demo[path], h5py.Dataset)
    }
    if len(set(lengths.values())) != 1:
        reasons.append(f"inconsistent sequence lengths: {lengths}")
    num_samples = int(demo.attrs.get("num_samples", len(actions)))
    if num_samples != len(actions):
        reasons.append(
            f"num_samples attribute {num_samples} does not match actions {len(actions)}"
        )
    if not bool(demo.attrs.get("success", True)):
        reasons.append("episode is not marked successful")
    if observations.shape[1:] != (len(OBSERVATION_NAMES),):
        reasons.append(
            f"observation shape is {observations.shape}, "
            f"expected (T, {len(OBSERVATION_NAMES)})"
        )
    if actions.shape[1:] != (len(ACTION_NAMES),):
        reasons.append(f"action shape is {actions.shape}, expected (T, 7)")
    if processed_actions.shape != actions.shape:
        reasons.append(
            f"processed action shape {processed_actions.shape} != {actions.shape}"
        )
    for label, array in (
        ("observations", observations),
        ("actions", actions),
        ("processed actions", processed_actions),
    ):
        if np.isnan(array).any():
            reasons.append(f"NaN in {label}")
        if np.isinf(array).any():
            reasons.append(f"Inf in {label}")
    if np.any(np.abs(actions) > 1.0 + 1.0e-5):
        reasons.append("normalized action outside [-1, 1]")

    # Validate physical realism directly against canonicalized raw arrays
    ee_position = raw["ee_pose"][:, :3]
    if (
        np.any(ee_position[:, 0] < -0.5)
        or np.any(ee_position[:, 0] > 2.5)
        or np.any(np.abs(ee_position[:, 1]) > 1.5)
        or np.any(ee_position[:, 2] < -0.5)
        or np.any(ee_position[:, 2] > 2.5)
    ):
        reasons.append("EE position outside expected base-frame workspace")
    for label, quaternion in (
        ("EE", raw["ee_pose"][:, 3:7]),
        ("object", raw["object_pose_b"][:, 3:7]),
    ):
        norm_error = np.max(np.abs(np.linalg.norm(quaternion, axis=-1) - 1.0))
        if norm_error > 1.0e-3:
            reasons.append(f"{label} quaternion norm error {norm_error:.3g}")
    if np.any(np.abs(raw["arm_joint_pos"]) > 4.0 * np.pi):
        reasons.append("implausible arm joint position")
    if np.any(np.abs(raw["arm_joint_vel"]) > 1000.0):
        reasons.append("implausible arm joint velocity")
    if np.any(np.abs(raw["object_joint_pos"]) > 20.0 * np.pi):
        reasons.append("implausible valve joint position")
    if np.any(np.abs(raw["object_joint_vel"]) > 1000.0):
        reasons.append("implausible valve joint velocity")
    return reasons


def running_statistics(arrays: list[np.ndarray]) -> dict[str, np.ndarray]:
    """Compute channel-wise statistics for a list of trajectory arrays.

    Args:
        arrays: Arrays with a shared final dimension.

    Returns:
        Mapping containing mean, standard deviation, minimum, and maximum.
    """
    combined = np.concatenate(arrays, axis=0).astype(np.float64)
    return {
        "mean": combined.mean(axis=0),
        "std": combined.std(axis=0),
        "min": combined.min(axis=0),
        "max": combined.max(axis=0),
    }


def serialize_statistics(statistics: dict[str, np.ndarray]) -> dict[str, list[float]]:
    """Convert NumPy statistics to JSON-compatible lists.

    Args:
        statistics: NumPy statistics returned by :func:`running_statistics`.

    Returns:
        JSON-compatible statistics mapping.
    """
    return {key: value.tolist() for key, value in statistics.items()}


def save_summary_plot(
    path: Path,
    sample: dict[str, Any],
    all_actions: np.ndarray,
) -> None:
    """Save summary trajectory and distribution diagnostics.

    Args:
        path: Destination PNG path.
        sample: First clean demonstration arrays.
        all_actions: Concatenated normalized action array, shape ``(N, 7)``.
    """
    raw = sample["raw"]
    ee_pos = raw["ee_pose"][:, :3]
    obj_pos = raw["object_pose_b"][:, :3]

    fig = plt.figure(figsize=(16, 16))
    axis_3d = fig.add_subplot(3, 2, 1, projection="3d")
    axis_3d.plot(*ee_pos.T, label="TCP")
    axis_3d.scatter(*obj_pos[0], marker="*", s=100, label="Valve root")
    axis_3d.set_title("TCP trajectory in robot base")
    axis_3d.set_xlabel("X [m]")
    axis_3d.set_ylabel("Y [m]")
    axis_3d.set_zlabel("Z [m]")
    axis_3d.legend()

    axis_joint_pos = fig.add_subplot(3, 2, 2)
    axis_joint_pos.plot(raw["arm_joint_pos"])
    axis_joint_pos.set_title("Arm and gripper joint positions")
    axis_joint_pos.set_xlabel("Step")
    axis_joint_pos.set_ylabel("Position [rad]")

    axis_valve = fig.add_subplot(3, 2, 3)
    axis_valve.plot(raw["object_joint_pos"], label="position")
    axis_valve.plot(raw["object_joint_vel"], label="velocity")
    axis_valve.set_title("Valve joint state")
    axis_valve.set_xlabel("Step")
    axis_valve.legend()

    axis_delta = fig.add_subplot(3, 2, 4)
    axis_delta.hist(all_actions[:, 1:4], bins=60, label=("x", "y", "z"))
    axis_delta.set_yscale("log")
    axis_delta.set_title("Normalized translation-action distribution")
    axis_delta.legend()

    axis_rotation = fig.add_subplot(3, 2, 5)
    axis_rotation.hist(all_actions[:, 4:7], bins=60, label=("x", "y", "z"))
    axis_rotation.set_yscale("log")
    axis_rotation.set_title("Normalized rotation-action distribution")
    axis_rotation.legend()

    axis_rotation_norm = fig.add_subplot(3, 2, 6)
    axis_rotation_norm.hist(np.linalg.norm(all_actions[:, 4:7], axis=-1), bins=60)
    axis_rotation_norm.set_yscale("log")
    axis_rotation_norm.set_title("Normalized rotation-action magnitude")
    axis_rotation_norm.set_xlabel("Magnitude")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_valve_cloud(path: Path, object_poses: np.ndarray) -> None:
    """Save the distribution of initial valve roots in robot base.

    Args:
        path: Destination PNG path.
        object_poses: Initial object poses, shape ``(num_demos, 7)``.
    """
    fig = plt.figure(figsize=(10, 8))
    axis = fig.add_subplot(111, projection="3d")
    axis.scatter(*object_poses[:, :3].T, alpha=0.6)
    axis.set_title(f"Valve reset positions ({len(object_poses)} demonstrations)")
    axis.set_xlabel("X [m]")
    axis.set_ylabel("Y [m]")
    axis.set_zlabel("Z [m]")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_action_plot(path: Path, sample: dict[str, Any]) -> None:
    """Save normalized and physical action traces for the first clean demo.

    Args:
        path: Destination PNG path.
        sample: First clean demonstration arrays.
    """
    actions = sample["actions"]
    processed = sample["processed_actions"]
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    axes[0].plot(actions[:, 0], label="normalized")
    axes[0].plot(processed[:, 0], label="physical joint target")
    axes[0].set_title("Gripper command")
    axes[0].legend()
    axes[1].plot(actions[:, 1:4])
    axes[1].set_title("Normalized relative translation")
    axes[1].set_ylabel("Normalized action")
    axes[2].plot(actions[:, 4:7])
    axes[2].set_title("Normalized relative axis-angle")
    axes[2].set_ylabel("Normalized action")
    axes[2].set_xlabel("Step")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def analyze_dataset(
    paths: dict[str, Path], control_frequency_hz: float = 20.0
) -> dict[str, Any]:
    """Clean the raw dataset and write the Diffusion Policy HDF5 file.

    Args:
        paths: Resolved input and output paths.
        control_frequency_hz: Dataset action sampling frequency [Hz].

    Returns:
        Metadata dictionary describing the cleaned dataset.

    Raises:
        KeyError: If the input has no root ``data`` group.
        RuntimeError: If no demonstrations pass validation.
    """
    observations_list: list[np.ndarray] = []
    actions_list: list[np.ndarray] = []
    processed_actions_list: list[np.ndarray] = []
    rejected: list[dict[str, str]] = []
    source_names: list[str] = []
    sample: dict[str, Any] | None = None
    object_initial_poses: list[np.ndarray] = []
    temporary_output = paths["output"].with_suffix(".hdf5.tmp")
    if temporary_output.exists():
        temporary_output.unlink()

    try:
        with h5py.File(paths["input"], "r") as input_file:
            if "data" not in input_file or not isinstance(input_file["data"], h5py.Group):
                raise KeyError("Input HDF5 file has no root 'data' group")
            source_group = input_file["data"]
            demo_names = sorted(
                source_group.keys(),
                key=lambda name: int(name.removeprefix("demo_")),
            )
            print(f"[INFO] Found {len(demo_names)} demonstrations in {paths['input']}")
            with h5py.File(temporary_output, "w") as output_file:
                output_group = output_file.create_group("data")
                clean_count = 0
                total_samples = 0
                for source_name in demo_names:
                    demo = source_group[source_name]
                    try:
                        raw, observations, actions, processed_actions = build_training_arrays(demo)
                        reasons = validate_demo(
                            demo, raw, observations, actions, processed_actions
                        )
                    except (KeyError, ValueError, IndexError) as error:
                        reasons = [str(error)]
                    if reasons:
                        reason = "; ".join(reasons)
                        rejected.append({"name": source_name, "reason": reason})
                        print(f"[WARNING] Rejecting {source_name}: {reason}")
                        continue

                    output_demo = output_group.create_group(f"demo_{clean_count}")
                    output_demo.create_dataset(
                        "observations", data=observations, compression="gzip"
                    )
                    output_demo.create_dataset("actions", data=actions, compression="gzip")
                    output_demo.create_dataset(
                        "processed_actions", data=processed_actions, compression="gzip"
                    )
                    output_demo.attrs["num_samples"] = len(actions)
                    output_demo.attrs["success"] = bool(demo.attrs.get("success", True))
                    output_demo.attrs["source_demo"] = source_name
                    observations_list.append(observations)
                    actions_list.append(actions)
                    processed_actions_list.append(processed_actions)
                    source_names.append(source_name)
                    object_initial_poses.append(raw["object_pose_b"][0])
                    if sample is None:
                        sample = {
                            "raw": raw,
                            "observations": observations,
                            "actions": actions,
                            "processed_actions": processed_actions,
                        }
                    clean_count += 1
                    total_samples += len(actions)

                if clean_count == 0:
                    raise RuntimeError("No demonstrations passed validation")
                output_group.attrs["total"] = total_samples
                output_group.attrs["schema_version"] = 2
                output_group.attrs["control_frequency_hz"] = control_frequency_hz
                output_group.attrs["sample_period_s"] = 1.0 / control_frequency_hz
                output_group.attrs["observation_names"] = json.dumps(OBSERVATION_NAMES)
                output_group.attrs["action_names"] = json.dumps(ACTION_NAMES)
                output_group.attrs["env_args"] = json.dumps(
                    {
                        "env_name": "Spot-Manipulation-Ball-Valve-Delta-Play",
                        "type": 2,
                    }
                )
        os.replace(temporary_output, paths["output"])
    except Exception:
        if temporary_output.exists():
            temporary_output.unlink()
        raise

    observation_statistics = running_statistics(observations_list)
    action_statistics = running_statistics(actions_list)
    processed_action_statistics = running_statistics(processed_actions_list)
    zero_observations = np.flatnonzero(observation_statistics["std"] < 1.0e-5)
    zero_actions = np.flatnonzero(action_statistics["std"] < 1.0e-5)
    metadata: dict[str, Any] = {
        "schema_version": 2,
        "source_dataset": str(paths["input"]),
        "output_dataset": str(paths["output"]),
        "control_frequency_hz": control_frequency_hz,
        "sample_period_s": 1.0 / control_frequency_hz,
        "dataset_summary": {
            "raw_demonstrations": len(source_names) + len(rejected),
            "clean_demonstrations": len(source_names),
            "clean_samples": int(sum(len(array) for array in actions_list)),
            "rejected_demonstrations": rejected,
            "source_demo_order": source_names,
        },
        "observation_channels": {
            "dimension": len(OBSERVATION_NAMES),
            "names": list(OBSERVATION_NAMES),
            "layout": " | ".join(OBSERVATION_NAMES),
            "quaternion_convention": "wxyz",
            "zero_variance": [
                {"index": int(index), "name": OBSERVATION_NAMES[index]}
                for index in zero_observations
            ],
            "active_indices": [
                index
                for index in range(len(OBSERVATION_NAMES))
                if index not in zero_observations
            ],
        },
        "action_channels": {
            "dimension": len(ACTION_NAMES),
            "names": list(ACTION_NAMES),
            "layout": "gripper(1)|normalized_delta_position(3)|normalized_axis_angle(3)",
            "zero_variance": [
                {"index": int(index), "name": ACTION_NAMES[index]}
                for index in zero_actions
            ],
            "active_indices": [
                index
                for index in range(len(ACTION_NAMES))
                if index not in zero_actions
            ],
        },
        "statistics": {
            "observations": serialize_statistics(observation_statistics),
            "actions": serialize_statistics(action_statistics),
            "processed_actions": serialize_statistics(processed_action_statistics),
        },
    }

    with paths["metadata"].open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)
        metadata_file.write("\n")
    all_actions = np.concatenate(actions_list, axis=0)
    assert sample is not None
    save_summary_plot(paths["plot"], sample, all_actions)
    save_valve_cloud(paths["valve_cloud"], np.stack(object_initial_poses))
    save_action_plot(paths["action_plot"], sample)
    return metadata


def main() -> None:
    """Run dataset cleaning and diagnostics."""
    args = parse_args()
    if args.control_frequency_hz <= 0.0:
        raise ValueError("--control_frequency_hz must be greater than zero")
    paths = resolve_paths(args)
    prepare_outputs(paths, args.overwrite)
    print(f"[INFO] Input dataset: {paths['input']}")
    print(f"[INFO] Cleaned dataset: {paths['output']}")
    metadata = analyze_dataset(paths, args.control_frequency_hz)
    summary = metadata["dataset_summary"]
    print(
        "[INFO] Complete: "
        f"{summary['clean_demonstrations']}/{summary['raw_demonstrations']} demos, "
        f"{summary['clean_samples']} samples kept"
    )
    for key in ("metadata", "plot", "valve_cloud", "action_plot"):
        print(f"[INFO] {key.replace('_', ' ').title()}: {paths[key]}")


if __name__ == "__main__":
    main()
