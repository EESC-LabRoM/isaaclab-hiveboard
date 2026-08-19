from __future__ import annotations

from dataclasses import MISSING
from typing import Sequence

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv
from isaaclab.managers import CommandTerm
from isaaclab.managers.manager_term_cfg import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.markers.visualization_markers import VisualizationMarkers
from isaaclab.sensors.frame_transformer.frame_transformer import FrameTransformer
from isaaclab.utils.configclass import configclass


class FramePoseCommand(CommandTerm):

    cfg: "FramePoseCommandCfg"

    def __init__(self, cfg: "FramePoseCommandCfg", env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        # obtain the robot asset
        # -- robot
        self._asset: Articulation = env.scene[cfg.asset_name]
        body_ids, body_names = self._asset.find_bodies(self.cfg.body_name)
        self._body_idx = body_ids[0]
        self._body_name = body_names[0]
        print("Body names:", body_names)
        # Frames used for pose commands
        self._frame: FrameTransformer = env.scene[self.cfg.frame_name]
        self._n_frames = self._frame.data.target_pos_w.shape[1]

        # Default orientation for the target poses
        self.target_eef_pos = (
            self._frame.data.target_pos_w[:, 0] - self._asset.data.root_pos_w
        )
        self.target_eef_quat = self._frame.data.target_quat_w[:, 0]
        self.target_frame_idx = torch.zeros(
            self._env.num_envs, device=self._env.device, dtype=torch.long
        )

        # Build default command, quat can't be 0s
        self._command = torch.zeros(
            (self._env.num_envs, 8), device=self._env.device, dtype=torch.float32
        )
        self._command[:, 0] = 1  # Close gripper
        self._command[:, 1:4] = self.target_eef_pos
        self._command[:, 4:8] = self.target_eef_quat

        # convert the fixed offsets to torch tensors of batched shape
        if self.cfg.body_offset is not None:
            self._offset_pos = torch.tensor(
                self.cfg.body_offset.pos, device=self.device
            ).repeat(self.num_envs, 1)
            self._offset_rot = torch.tensor(
                self.cfg.body_offset.rot, device=self.device
            ).repeat(self.num_envs, 1)
        else:
            self._offset_pos, self._offset_rot = None, None

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired  command"""

        return self._command

    def _resample_command(
        self, env_ids: Sequence[int] | slice | None | torch.Tensor = None
    ):
        env_origins = self._asset.data.root_pos_w

        # Get the target frame pose in world frame from next frame transformer
        tmp = self._frame.data.target_pos_w[env_ids]
        tmp2 = tmp[:, self.target_frame_idx[env_ids]]

        ee_pose_b = tmp2 - env_origins[env_ids]
        ee_quat_b = self._frame.data.target_quat_w[
            env_ids, self.target_frame_idx[env_ids]
        ]

        if self.cfg.body_offset is not None:
            ee_pose_b, ee_quat_b = math_utils.combine_frame_transforms(
                ee_pose_b, ee_quat_b, self._offset_pos, self._offset_rot
            )

        self.target_eef_pos[env_ids] = ee_pose_b
        self.target_eef_quat[env_ids] = ee_quat_b

        self.target_frame_idx[env_ids] = (
            self.target_frame_idx[env_ids] + 1
        ) % self._n_frames

    def _update_command(self):
        """Post-processes command."""
        # If near the target, resample a new target
        env_ids_to_resample = torch.where(
            torch.norm(
                self.target_eef_pos
                - self._asset.data.body_pos_w[:, self._body_idx]
                + self._asset.data.root_pos_w,
                dim=-1,
            )
            < 0.09,
        )[0]

        if len(env_ids_to_resample) > 0:
            self._resample_command(env_ids_to_resample)

        # Set the command tensor
        self._command[:, 0] = 1  # Close gripper
        self._command[:, 1:4] = self.target_eef_pos
        self._command[:, 4:8] = self.target_eef_quat

    def _update_metrics(self): ...

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first time
            if not hasattr(self, "goal_pose_visualizer"):
                # -- goal
                self.goal_pose_visualizer = VisualizationMarkers(
                    self.cfg.goal_pose_visualizer_cfg
                )
                # -- current
                self.current_pose_visualizer = VisualizationMarkers(
                    self.cfg.current_pose_visualizer_cfg
                )

            # set their visibility to true
            self.goal_pose_visualizer.set_visibility(True)
            self.current_pose_visualizer.set_visibility(True)

        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)
                self.current_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self._asset.is_initialized:
            return

        # goal end-effector pose
        self.goal_pose_visualizer.visualize(
            self.target_eef_pos + self._asset.data.root_pos_w, self.target_eef_quat
        )

        # current end-effector pose
        ee_pos_w = self._asset.data.body_pos_w[:, self._body_idx]
        ee_quat = self._asset.data.body_quat_w[:, self._body_idx]
        self.current_pose_visualizer.visualize(ee_pos_w, ee_quat)


@configclass
class FramePoseCommandCfg(CommandTermCfg):
    """Configuration for the uniform velocity command generator."""

    @configclass
    class OffsetCfg:
        """The offset pose from parent frame to child frame.

        On many robots, end-effector frames are fictitious frames that do not have a corresponding
        rigid body. In such cases, it is easier to define this transform w.r.t. their parent rigid body.
        For instance, for the Franka Emika arm, the end-effector is defined at an offset to the the
        "panda_hand" frame.
        """

        pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Translation w.r.t. the parent frame. Defaults to (0.0, 0.0, 0.0)."""
        rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
        """Quaternion rotation ``(w, x, y, z)`` w.r.t. the parent frame. Defaults to (1.0, 0.0, 0.0, 0.0)."""

    class_type: type = FramePoseCommand

    asset_name: str = MISSING  # type: ignore
    """Name of the asset in the environment for which the commands are generated."""

    body_name: str = MISSING  # type: ignore
    """Name of the end-effector body used for pose commands."""

    frame_name: str = MISSING  # type: ignore
    """Name of the frame used for pose commands."""

    body_offset: OffsetCfg | None = None
    """Offset of target frame w.r.t. to the body frame. Defaults to None, in which case no offset is applied."""

    goal_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(  # type: ignore
        prim_path="/Visuals/Command/pose_goal"
    )

    """The configuration for the goal pose visualization marker. Defaults to GREEN_ARROW_X_MARKER_CFG."""

    current_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(  # type: ignore
        prim_path="/Visuals/Command/current_pose"
    )

    """The configuration for the current pose visualization marker. Defaults to BLUE_ARROW_X_MARKER_CFG."""

    # Set the scale of the visualization markers to (0.5, 0.5, 0.5)
    goal_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)  # type: ignore
    current_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)  # type: ignore
