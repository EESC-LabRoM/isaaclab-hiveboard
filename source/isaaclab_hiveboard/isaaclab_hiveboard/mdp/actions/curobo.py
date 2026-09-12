from __future__ import annotations

import os
import tempfile
from dataclasses import MISSING
from typing import Sequence

import isaaclab.utils.math as PoseUtils
import torch
import yaml
try:
    from curobo.config_io import load_yaml
except Exception:
    from curobo.util_file import load_yaml
from isaaclab.assets import Articulation
from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv
from isaaclab.managers import CommandTerm
from isaaclab.managers.manager_term_cfg import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG, VisualizationMarkersCfg
from isaaclab.markers.visualization_markers import VisualizationMarkers
from isaaclab.sensors import FrameTransformerData
from isaaclab.sensors.frame_transformer.frame_transformer import FrameTransformer
from isaaclab.utils.configclass import configclass
from isaaclab_mimic.datagen.waypoint import (
    Waypoint,
)
from isaaclab_mimic.motion_planners.curobo.curobo_planner import CuroboPlanner
from isaaclab_mimic.motion_planners.curobo.curobo_planner_cfg import CuroboPlannerCfg


class CuroboCommand(CommandTerm):

    cfg: "CuRoboCommandCfg"

    def __init__(self, cfg: "CuRoboCommandCfg", env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        # obtain the robot asset
        # -- robot

        self._asset: Articulation = env.scene[cfg.asset_name]
        body_ids, body_names = self._asset.find_bodies(self.cfg.body_name)
        self._body_idx = body_ids[0]
        self._body_name = body_names[0]

        # -- transformer
        self._frame: FrameTransformer = env.scene[cfg.frame_transformer_name]
        self._n_frames = self._frame.data.target_pos_w.shape[1]

        print(self.cfg.planner_config)
        self.motion_planner = CuroboPlanner(
            env=env,
            robot=env.scene["robot"],
            config=self.cfg.planner_config,  # Pass the config object
            env_id=0,  # Pass environment ID
        )

        # self.targets_eef_pose = torch.ones(self._env.num_envs, 4, 4)
        default_pos = (
            self._asset.data.body_pos_w[0, self._body_idx]
            - self._env.scene.env_origins[0]
        )
        # 2. Define a neutral orientation (Identity Quaternion: w, x, y, z)
        # This usually aligns the gripper with the base frame (pointing forward/flat)
        self.default_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._env.device)

        # 3. Create the 4x4 Homogeneous Matrix
        self.default_pose_4x4 = PoseUtils.make_pose(
            default_pos.unsqueeze(0),
            PoseUtils.matrix_from_quat(self.default_quat.unsqueeze(0)),
        ).squeeze()

        # 4. Repeat for all environments (num_envs, 4, 4)
        self.targets_eef_pose = self.default_pose_4x4.unsqueeze(0).repeat(
            self._env.num_envs, 1, 1
        )
        self.subtask_eef_pose = torch.ones(
            self._env.num_envs, 4, 4, device=self._env.device
        )
        self.trajectories = [
            [
                Waypoint(
                    pose=self.default_pose_4x4,
                    gripper_action=torch.zeros(1, device=self._env.device),
                )
            ]
            for _ in range(self._env.num_envs)
        ]
        self.trajectories_indices = torch.zeros(
            self._env.num_envs, dtype=torch.long, device=self._env.device
        )

        self.toggle_pose = torch.ones(
            self._env.num_envs, device=self._env.device, dtype=torch.bool
        )

        self._command = torch.zeros(self._env.num_envs, 8, device=self._env.device)

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity command in the base frame. Shape is (num_envs, 3)."""

        return self._command

    def _resample_command(self, env_ids):
        env_origins = self._env.scene.env_origins
        tf_data: FrameTransformerData = self._frame.data

        first_tf_pose = tf_data.target_pos_w[..., 0, :] - env_origins
        second_tf_pose = tf_data.target_pos_w[..., 1, :] - env_origins
        # cabinet_tf_quat = cabinet_tf_data.target_quat_w[..., 0, :]

        for env_id in range(self._env.num_envs):

            if self.toggle_pose[env_id]:
                target_pose = PoseUtils.make_pose(
                    first_tf_pose[env_id],
                    PoseUtils.matrix_from_quat(self.default_quat),
                )
            else:
                target_pose = PoseUtils.make_pose(
                    second_tf_pose[env_id],
                    PoseUtils.matrix_from_quat(self.default_quat),
                )

            self.toggle_pose[env_id] = not self.toggle_pose[env_id]

            # print("Target pose: ", target_pose)
            print("Original cabinet pose: ", tf_data.target_pos_w[env_id, 0, :])
            print("Cabinet pose: ", first_tf_pose[env_id])
            # This call updates the planner's world model and computes the trajectory.
            self.targets_eef_pose[env_id] = target_pose
            planning_success = self.motion_planner.update_world_and_plan_motion(
                target_pose=target_pose,
                expected_attached_object=None,
                env_id=env_id,
                step_size=self.motion_planner.step_size,
                enable_retiming=self.motion_planner.step_size is not None,
            )

            # If planning succeeds, execute the planner's trajectory first.
            if not planning_success:
                # If planning fails, abort the data generation trial.
                print(f"Env {env_id}: Motion planning failed")
                self.trajectories[env_id] = [
                    Waypoint(
                        pose=self.default_pose_4x4,
                        gripper_action=-torch.ones(1, device=self._env.device),
                        noise=self.motion_planner.config.motion_noise_scale,
                    )
                    for i in range(10)
                ]
            else:
                # The original subtask trajectory is stored to be executed after the transition.
                print(
                    f"Env {env_id}: Motion planning succeeded with length {len(self.trajectories[env_id])}"
                )
                self.trajectories[env_id] = [
                    Waypoint(
                        pose=planned_pose,
                        gripper_action=-torch.ones(1, device=self._env.device),
                        noise=self.motion_planner.config.motion_noise_scale,
                    )
                    for planned_pose in self.motion_planner.get_planned_poses()
                ]

            self.trajectories_indices[env_id] = 0

    def _update_command(self):
        """Post-processes the velocity command.

        This function sets velocity command to zero for standing environments and computes angular
        velocity command from heading error for heading-based environments.
        """
        for env_id in range(self._env.num_envs):
            if self.trajectories_indices[env_id] < len(self.trajectories[env_id]) - 1:
                self.trajectories_indices[env_id] += 1
            else:
                self._resample_command([env_id])

            # Update visualization if motion planner is available
            if self.motion_planner.visualize_spheres:
                current_joints = self._asset.data.joint_pos[env_id]
                self.motion_planner._update_visualization_at_joint_positions(
                    current_joints
                )

        self._command = torch.cat(
            [
                self.target_eef_pose_to_action(
                    target_eef_pose=self.trajectories[env_id][
                        self.trajectories_indices[env_id]
                    ].pose,
                    gripper_action=self.trajectories[env_id][
                        self.trajectories_indices[env_id]
                    ].gripper_action,
                )
                for env_id in range(self._env.num_envs)
            ],
            dim=0,
        )

    def _update_metrics(self): ...

    def get_robot_eef_pose(
        self, env_ids: Sequence[int] | slice | None = None
    ) -> torch.Tensor:
        """
        Get current robot end effector pose. Should be the same frame as used by the robot end-effector controller.

        Args:
            eef_name: Name of the end effector.
            env_ids: Environment indices to get the pose for. If None, all envs are considered.

        Returns:
            A torch.Tensor eef pose matrix. Shape is (len(env_ids), 4, 4)
        """
        if env_ids is None:
            env_ids = slice(None)

        # Retrieve end effector pose from the observation buffer
        ee_positions = (
            self._asset.data.body_pos_w[:, self._body_idx] - self._asset.data.root_pos_w
        )
        ee_quats = self._asset.data.body_quat_w[:, self._body_idx]

        eef_pos = ee_positions[env_ids]
        eef_quat = ee_quats[env_ids]

        # Quaternion format is w,x,y,z
        return PoseUtils.make_pose(eef_pos, PoseUtils.matrix_from_quat(eef_quat))

    def target_eef_pose_to_action(
        self,
        target_eef_pose: torch.Tensor,
        gripper_action: torch.Tensor,
        env_id: int = 0,
    ) -> torch.Tensor:
        """
        Takes a target pose and gripper action for the end effector controller and returns an action
        (usually a normalized delta pose action) to try and achieve that target pose.
        Noise is added to the target pose action if specified.

        Args:
            target_eef_pose: 4x4 target eef pose for each end-effector.
            gripper_action: gripper actions for each end-effector.
            noise: Noise to add to the action. If None, no noise is added.
            env_id: Environment index to get the action for.

        Returns:
            An action torch.Tensor that's compatible with env.step().
        """
        # target position and rotation
        target_pos, target_rot = PoseUtils.unmake_pose(target_eef_pose)

        # current position and rotation
        # curr_pose = self.get_robot_eef_pose(env_ids=[env_id])[0]
        # curr_pos, curr_rot = PoseUtils.unmake_pose(curr_pose)

        # normalized delta position action
        delta_position = target_pos - self._asset.data.root_pos_w[env_id]

        return torch.cat(
            [gripper_action, delta_position, PoseUtils.quat_from_matrix(target_rot)],
            dim=-1,  #  delta_rotation,
        ).unsqueeze(0)

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first time
            if not hasattr(self, "goal_vel_visualizer"):
                # -- goal
                self.goal_vel_visualizer = VisualizationMarkers(
                    self.cfg.goal_vel_visualizer_cfg
                )
                # -- current
                self.current_vel_visualizer = VisualizationMarkers(
                    self.cfg.current_vel_visualizer_cfg
                )

                # -- next
                self.next_pose_visualizer = VisualizationMarkers(
                    self.cfg.next_pose_visualizer_cfg
                )

            # set their visibility to true
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
            self.next_pose_visualizer.set_visibility(True)

        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)
                self.next_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self._asset.is_initialized:
            return

        # goal end-effector pose
        goal_pos_w, goal_rot = PoseUtils.unmake_pose(self.targets_eef_pose)
        self.goal_vel_visualizer.visualize(
            goal_pos_w, PoseUtils.quat_from_matrix(goal_rot)
        )

        # next end-effector pose
        next_ee_pos_w = self._command[:, 1:4]
        next_ee_quat = self._command[:, 4:8]
        self.next_pose_visualizer.visualize(next_ee_pos_w, next_ee_quat)

        # current end-effector pose
        ee_pos_w = self._asset.data.body_pos_w[:, self._body_idx]
        ee_quat = self._asset.data.body_quat_w[:, self._body_idx]
        self.current_vel_visualizer.visualize(ee_pos_w, ee_quat)


@configclass
class CuRoboCommandCfg(CommandTermCfg):
    """Configuration for the uniform velocity command generator."""

    class_type: type = CuroboCommand

    planner_config: CuroboPlannerCfg = MISSING
    """Configuration for the CuRobo motion planner."""

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    body_name: str = MISSING
    """Name of the end-effector body used for pose commands."""

    frame_transformer_name: str = MISSING
    """Name of the frame transformer used for pose commands."""

    goal_vel_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/pose_goal"
    )

    """The configuration for the goal velocity visualization marker. Defaults to GREEN_ARROW_X_MARKER_CFG."""

    current_vel_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/current_pose"
    )
    next_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/next_pose"
    )
    """The configuration for the current velocity visualization marker. Defaults to BLUE_ARROW_X_MARKER_CFG."""

    # Set the scale of the visualization markers to (0.5, 0.5, 0.5)
    goal_vel_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    current_vel_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    next_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)

    @classmethod
    def _create_temp_robot_yaml(cls, yaml_path: str, urdf_path: str) -> str:
        """Create a temporary robot configuration YAML with custom URDF path.

        Args:
            yaml_path: Absolute path to the robot configuration file
            urdf_path: Absolute path to the URDF file

        Returns:
            Path to the temporary YAML file

        Raises:
            FileNotFoundError: If the URDF file doesn't exist
        """
        # Validate URDF path
        if not os.path.isabs(urdf_path) or not os.path.isfile(urdf_path):
            raise FileNotFoundError(f"URDF must be a local file: {urdf_path}")

        # Load base configuration
        data = load_yaml(yaml_path)
        print(f"urdf_path: {urdf_path}")
        # Update URDF path
        data["robot_cfg"]["kinematics"]["urdf_path"] = urdf_path

        # Write to temporary file
        tmp_dir = tempfile.mkdtemp(prefix="curobo_robot_cfg_")
        out_path = os.path.join(tmp_dir, os.path.basename(yaml_path))
        with open(out_path, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False)

        return out_path
