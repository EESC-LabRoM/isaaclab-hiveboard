"""Runtime pose diagnostics for scripted manipulation environments.

The diagnostics deliberately distinguish controller tracking error from a bad
configured grasp frame.  A controller can track a target perfectly while the
target makes the gripper approach an object sideways.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import TextIO

import torch

import isaaclab.utils.math as math_utils


_AXIS_NAMES = ("+X", "+Y", "+Z")


def _as_list(tensor: torch.Tensor) -> list[float]:
    return tensor.detach().cpu().tolist()


def _fmt_vector(tensor: torch.Tensor, precision: int = 4) -> str:
    values = _as_list(tensor)
    return "[" + ", ".join(f"{value:+.{precision}f}" for value in values) + "]"


def _shortest_axis_angle_deg(quat: torch.Tensor) -> torch.Tensor:
    """Return the shortest axis-angle rotation vector in degrees."""
    quat = quat / torch.linalg.vector_norm(quat).clamp_min(1.0e-9)
    if quat[0] < 0.0:
        quat = -quat
    return torch.rad2deg(math_utils.axis_angle_from_quat(quat.unsqueeze(0))[0])


def _rotation_angle_deg(quat: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(_shortest_axis_angle_deg(quat)).item())


def _axis_mapping(quat_source_target: torch.Tensor) -> str:
    """Describe target X/Y/Z axes as vectors expressed in the source frame."""
    rotation = math_utils.matrix_from_quat(quat_source_target.unsqueeze(0))[0]
    return " ".join(
        f"{target_axis}={_fmt_vector(rotation[:, index], precision=3)}"
        for index, target_axis in enumerate(_AXIS_NAMES)
    )


class PoseDiagnostics:
    """Print and record end-effector/target pose relationships for one env."""

    _CSV_FIELDS = (
        "step",
        "stage_index",
        "stage",
        "target_frame",
        "position_error_m",
        "rotation_error_deg",
        "rotation_error_x_deg",
        "rotation_error_y_deg",
        "rotation_error_z_deg",
        "facing_score",
        "jaw_alignment_score",
        "command_x_w",
        "command_y_w",
        "command_z_w",
        "command_qw_w",
        "command_qx_w",
        "command_qy_w",
        "command_qz_w",
        "tcp_x_w",
        "tcp_y_w",
        "tcp_z_w",
        "tcp_qw_w",
        "tcp_qx_w",
        "tcp_qy_w",
        "tcp_qz_w",
        "tcp_x_source",
        "tcp_y_source",
        "tcp_z_source",
        "tcp_qw_source",
        "tcp_qx_source",
        "tcp_qy_source",
        "tcp_qz_source",
    )

    def __init__(
        self,
        env,
        *,
        env_index: int = 0,
        print_interval: int = 30,
        csv_path: str | None = None,
        task_name: str | None = None,
    ):
        if not 0 <= env_index < env.num_envs:
            raise ValueError(
                f"pose-debug env index {env_index} is outside [0, {env.num_envs - 1}]"
            )

        self._env = env
        self._env_index = env_index
        self._print_interval = max(1, print_interval)
        self._task_name = task_name
        if task_name in (
            "Franka-Manipulation-Lever-Valve",
            "Franka-Manipulation-Ball-Valve",
        ):
            self._desired_jaw_axis = 2  # source Z, across the +Y valve handle
            self._desired_grasp_quat = (0.5, 0.5, -0.5, -0.5)
            self._grasp_description = "TCP Y -> source Z (across +Y handle)"
        elif task_name == "Franka-Manipulation-Circuit-Breaker":
            self._desired_jaw_axis = 1  # source Y, breaker joint/handle width
            value = math.sqrt(0.5)
            self._desired_grasp_quat = (value, 0.0, -value, 0.0)
            self._grasp_description = "TCP Y -> source Y (across switch width)"
        else:
            self._desired_jaw_axis = None
            self._desired_grasp_quat = None
            self._grasp_description = None
        self._term = env.command_manager.get_term("pose_command")
        self._robot = env.scene[self._term.cfg.asset_name]
        self._target_frame = env.scene["target_frame"]
        self._ee_frame = env.scene["ee_frame"]
        self._tcp_index = self._ee_frame.data.target_frame_names.index("ee_tcp")
        self._last_stage_index: int | None = None

        self._csv_file: TextIO | None = None
        self._csv_writer: csv.DictWriter | None = None
        if csv_path:
            output_path = Path(csv_path).expanduser()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            self._csv_file = output_path.open("w", newline="", encoding="utf-8")
            self._csv_writer = csv.DictWriter(
                self._csv_file, fieldnames=self._CSV_FIELDS
            )
            self._csv_writer.writeheader()
            print(f"[POSE DEBUG] CSV output: {output_path.resolve()}")

        self._print_configuration_report()

    def close(self) -> None:
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None

    def update(self, step: int) -> None:
        sample = self._sample(step)
        self._write_csv(sample)

        stage_changed = sample["stage_index"] != self._last_stage_index
        if stage_changed:
            self._print_stage_report(sample)
            self._last_stage_index = sample["stage_index"]
        elif step % self._print_interval == 0:
            print(
                f"[POSE DEBUG] step={step:04d} stage={sample['stage']} "
                f"tracking={sample['position_error_m'] * 100.0:.2f} cm/"
                f"{sample['rotation_error_deg']:.2f} deg "
                f"facing={sample['facing_score']:+.3f} "
                f"jaw={sample['jaw_alignment_score']:.3f}"
            )

    def _stage(self) -> tuple[int, str, str]:
        stage_index = int(self._term._current_command_idx[self._env_index].item())
        if stage_index >= len(self._term.cfg.commands):
            return stage_index, "complete", ""
        stage_cfg = self._term.cfg.commands[stage_index]
        return (
            stage_index,
            type(stage_cfg).__name__,
            getattr(stage_cfg, "target_frame_name", ""),
        )

    def _sample(self, step: int) -> dict[str, float | int | str]:
        env_index = self._env_index
        stage_index, stage, target_name = self._stage()

        command = self._term.command[env_index]
        command_pos_b = command[1:4]
        command_quat_b = command[4:8]
        command_pos_w, command_quat_w = math_utils.combine_frame_transforms(
            self._robot.data.root_pos_w[env_index : env_index + 1],
            self._robot.data.root_quat_w[env_index : env_index + 1],
            command_pos_b.unsqueeze(0),
            command_quat_b.unsqueeze(0),
        )
        command_pos_w = command_pos_w[0]
        command_quat_w = command_quat_w[0]

        tcp_pos_w = self._ee_frame.data.target_pos_w[env_index, self._tcp_index]
        tcp_quat_w = self._ee_frame.data.target_quat_w[env_index, self._tcp_index]
        error_pos_command, error_quat_command = math_utils.subtract_frame_transforms(
            command_pos_w.unsqueeze(0),
            command_quat_w.unsqueeze(0),
            tcp_pos_w.unsqueeze(0),
            tcp_quat_w.unsqueeze(0),
        )
        error_pos_command = error_pos_command[0]
        error_quat_command = error_quat_command[0]
        error_rotation_deg = _shortest_axis_angle_deg(error_quat_command)

        source_pos_w = self._target_frame.data.source_pos_w[env_index]
        source_quat_w = self._target_frame.data.source_quat_w[env_index]
        tcp_pos_source, tcp_quat_source = math_utils.subtract_frame_transforms(
            source_pos_w.unsqueeze(0),
            source_quat_w.unsqueeze(0),
            tcp_pos_w.unsqueeze(0),
            tcp_quat_w.unsqueeze(0),
        )
        tcp_pos_source = tcp_pos_source[0]
        tcp_quat_source = tcp_quat_source[0]

        # Franka's TCP +Z points forward through the fingers.  The panel-facing
        # convention used by these assets is the opposite of source-frame +X.
        tcp_rotation_source = math_utils.matrix_from_quat(
            tcp_quat_source.unsqueeze(0)
        )[0]
        facing_score = float((-tcp_rotation_source[0, 2]).item())
        jaw_alignment_score = math.nan
        if self._desired_jaw_axis is not None:
            jaw_alignment_score = float(
                torch.abs(tcp_rotation_source[self._desired_jaw_axis, 1]).item()
            )

        row: dict[str, float | int | str] = {
            "step": step,
            "stage_index": stage_index,
            "stage": stage,
            "target_frame": target_name,
            "position_error_m": float(torch.linalg.vector_norm(error_pos_command)),
            "rotation_error_deg": _rotation_angle_deg(error_quat_command),
            "facing_score": facing_score,
            "jaw_alignment_score": jaw_alignment_score,
        }
        for name, value in zip(
            ("rotation_error_x_deg", "rotation_error_y_deg", "rotation_error_z_deg"),
            _as_list(error_rotation_deg),
        ):
            row[name] = value
        self._add_pose(row, "command", command_pos_w, command_quat_w, "w")
        self._add_pose(row, "tcp", tcp_pos_w, tcp_quat_w, "w")
        self._add_pose(row, "tcp", tcp_pos_source, tcp_quat_source, "source")
        return row

    @staticmethod
    def _add_pose(
        row: dict[str, float | int | str],
        prefix: str,
        pos: torch.Tensor,
        quat: torch.Tensor,
        suffix: str,
    ) -> None:
        for axis, value in zip("xyz", _as_list(pos)):
            row[f"{prefix}_{axis}_{suffix}"] = value
        for axis, value in zip(("qw", "qx", "qy", "qz"), _as_list(quat)):
            row[f"{prefix}_{axis}_{suffix}"] = value

    def _write_csv(self, sample: dict[str, float | int | str]) -> None:
        if self._csv_writer is not None:
            self._csv_writer.writerow(sample)
            assert self._csv_file is not None
            self._csv_file.flush()

    def _print_configuration_report(self) -> None:
        print("[POSE DEBUG] Frame convention:")
        print("  Franka TCP +Z = approach direction; TCP +/-Y = jaw closing direction")
        print("  Assumed panel outward normal = target source +X")
        print("  Panel-facing goal: TCP +Z -> source -X (facing score +1.0)")
        if self._grasp_description is not None:
            print(
                f"  Task grasp-roll goal: {self._grasp_description} "
                "(jaw score +1.0)"
            )

        zero_roll_facing_quat = torch.tensor(
            (math.sqrt(0.5), 0.0, -math.sqrt(0.5), 0.0),
            device=self._env.device,
            dtype=torch.float32,
        )
        print("[POSE DEBUG] Configured target frames (pose in target source frame):")
        for index, name in enumerate(self._target_frame.data.target_frame_names):
            target_pos = self._target_frame.data.target_pos_source[
                self._env_index, index
            ]
            target_quat = self._target_frame.data.target_quat_source[
                self._env_index, index
            ]
            target_rotation = math_utils.matrix_from_quat(target_quat.unsqueeze(0))[0]
            facing_score = float((-target_rotation[0, 2]).item())
            print(
                f"  {name}: pos={_fmt_vector(target_pos)} "
                f"quat(wxyz)={_fmt_vector(target_quat)} facing={facing_score:+.3f}"
            )
            print(f"    axes(source): {_axis_mapping(target_quat)}")

            rotation_command = next(
                (
                    cfg
                    for cfg in self._term.cfg.commands
                    if getattr(cfg, "target_frame_name", None) == name
                    and hasattr(cfg, "axis")
                ),
                None,
            )
            if rotation_command is not None:
                axis_in_frame = target_quat.new_tensor(rotation_command.axis)
                axis_in_source = math_utils.quat_apply(
                    target_quat.unsqueeze(0), axis_in_frame.unsqueeze(0)
                )[0]
                print(
                    "    role: mechanical rotation axis; "
                    f"configured axis(source)={_fmt_vector(axis_in_source, 3)}"
                )
                continue
            if name == "rotate_frame":
                print(
                    "    role: dormant mechanical rotation-axis frame "
                    "(no active RotateFrameCfg)"
                )
                continue

            if facing_score > 0.999:
                print(
                    "    facing correction: none "
                    "(roll about the approach axis is unconstrained)"
                )
            else:
                correction_quat = math_utils.quat_mul(
                    zero_roll_facing_quat.unsqueeze(0),
                    math_utils.quat_inv(target_quat.unsqueeze(0)),
                )[0]
                correction_deg = _shortest_axis_angle_deg(correction_quat)
                print(
                    "    one zero-roll facing pose: rot=(0.7071068, 0, -0.7071068, 0); "
                    f"delta rotation-vector(source deg)={_fmt_vector(correction_deg, 2)}"
                )

            if self._desired_grasp_quat is not None:
                desired_grasp_quat = target_quat.new_tensor(self._desired_grasp_quat)
                grasp_correction = math_utils.quat_mul(
                    desired_grasp_quat.unsqueeze(0),
                    math_utils.quat_inv(target_quat.unsqueeze(0)),
                )[0]
                print(
                    f"    task grasp goal: {self._grasp_description}; "
                    f"rot={self._desired_grasp_quat}; "
                    "delta rotation-vector(source deg)="
                    f"{_fmt_vector(_shortest_axis_angle_deg(grasp_correction), 2)}"
                )

    def _print_stage_report(self, sample: dict[str, float | int | str]) -> None:
        print(
            f"[POSE DEBUG] stage change at step={sample['step']}: "
            f"{sample['stage_index']} {sample['stage']} "
            f"target={sample['target_frame'] or '-'}"
        )
        print(
            "  tracking error (command -> actual TCP): "
            f"{sample['position_error_m'] * 100.0:.2f} cm, "
            f"{sample['rotation_error_deg']:.2f} deg; "
            "rotation-vector(command deg)="
            f"[{sample['rotation_error_x_deg']:+.2f}, "
            f"{sample['rotation_error_y_deg']:+.2f}, "
            f"{sample['rotation_error_z_deg']:+.2f}]"
        )
        print(
            "  actual TCP in object/source: "
            f"pos=[{sample['tcp_x_source']:+.4f}, "
            f"{sample['tcp_y_source']:+.4f}, "
            f"{sample['tcp_z_source']:+.4f}] "
            f"quat(wxyz)=[{sample['tcp_qw_source']:+.4f}, "
            f"{sample['tcp_qx_source']:+.4f}, "
            f"{sample['tcp_qy_source']:+.4f}, "
            f"{sample['tcp_qz_source']:+.4f}] "
            f"facing={sample['facing_score']:+.3f} "
            f"jaw={sample['jaw_alignment_score']:.3f}"
        )
