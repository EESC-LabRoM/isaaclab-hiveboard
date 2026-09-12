"""Heuristic conversion from absolute TCP targets to relative pose actions."""

from __future__ import annotations

import isaaclab.utils.math as math_utils
import torch


class RelativeEePoseController:
    """Track a sequential pose command with normalized relative IK actions."""

    def __init__(
        self,
        env,
        command_name: str = "pose_command",
        position_step: float = 0.02,
        rotation_step: float = 0.1,
        position_gain: float = 1.0,
        rotation_gain: float = 1.0,
    ):
        self._env = env.unwrapped
        self._command = self._env.command_manager.get_term(command_name)
        self._position_step = position_step
        self._rotation_step = rotation_step
        self._position_gain = position_gain
        self._rotation_gain = rotation_gain

        terms = list(self._env.action_manager.active_terms)
        dims = list(self._env.action_manager.action_term_dim)
        expected = {"arm_action": 6, "gripper_action": 1}
        if dict(zip(terms, dims)) != expected:
            raise ValueError(
                "Relative EE control requires arm_action(6) and gripper_action(1); "
                f"received terms={terms}, dims={dims}."
            )
        self._action_terms = terms

    @staticmethod
    def _clamp_norm(value: torch.Tensor, maximum: float) -> torch.Tensor:
        norm = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
        scale = torch.clamp(maximum / norm.clamp_min(1.0e-9), max=1.0)
        return value * scale

    def compute(self) -> torch.Tensor:
        """Compute one normalized relative-pose action for every environment."""
        command = self._command.command
        current_pos_b, current_quat_b = self._command._get_ee_in_base_frame(
            slice(None)
        )

        position_error = self._position_gain * (command[:, 1:4] - current_pos_b)
        position_delta = self._clamp_norm(position_error, self._position_step)

        relative_quat = math_utils.quat_mul(
            command[:, 4:8], math_utils.quat_inv(current_quat_b)
        )
        rotation_error = self._rotation_gain * math_utils.axis_angle_from_quat(
            relative_quat
        )
        rotation_delta = self._clamp_norm(rotation_error, self._rotation_step)

        normalized_arm_action = torch.cat(
            (
                position_delta / self._position_step,
                rotation_delta / self._rotation_step,
            ),
            dim=-1,
        )
        actions_by_term = {
            "arm_action": normalized_arm_action,
            "gripper_action": command[:, 0:1],
        }
        return torch.cat(
            [actions_by_term[term_name] for term_name in self._action_terms], dim=-1
        )
