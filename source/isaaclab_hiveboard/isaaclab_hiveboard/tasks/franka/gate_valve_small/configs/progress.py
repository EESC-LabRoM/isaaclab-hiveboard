"""Signed stem travel, independent of the continuous joint's angle wrapping."""

import math

import torch

from isaaclab_hiveboard.mdp.commands.sequential_pose_command import SequentialPoseCommand


def accumulate_rotation(previous, current, accumulated):
    """Unwrap each small physics step; reverse motion subtracts progress."""
    delta = torch.remainder(current - previous + math.pi, math.tau) - math.pi
    return accumulated + delta


class GateValvePoseCommand(SequentialPoseCommand):
    """Track actual stem travel while executing the grasp/release sequence."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._stem = env.scene["small_valve"]
        self._stem_joint = self._stem.find_joints("RevoluteJoint")[0][0]
        self._previous_angle = torch.zeros(self.num_envs, device=self.device)
        self.stem_rotation = torch.zeros_like(self._previous_angle)

    def _resample_command(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        # reset_scene_to_default puts every selected stem at closed zero.
        self._previous_angle[ids] = 0.0
        self.stem_rotation[ids] = 0.0
        super()._resample_command(env_ids)

    def _update_command(self):
        angle = self._stem.data.joint_pos[:, self._stem_joint]
        self.stem_rotation[:] = accumulate_rotation(self._previous_angle, angle, self.stem_rotation)
        self._previous_angle[:] = angle
        super()._update_command()


def full_turn_success(env, command_name: str, tolerance: float):
    """Require completed release/retreat and one measured positive revolution."""
    command = env.command_manager.get_term(command_name)
    return command.is_done() & (torch.abs(command.stem_rotation - math.tau) <= tolerance)
