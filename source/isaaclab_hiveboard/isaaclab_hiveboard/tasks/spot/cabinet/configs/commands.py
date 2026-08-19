try:
    from curobo.geom.sdf.world import CollisionCheckerType
except Exception:
    try:
        from curobo._src.collision.types import CollisionCheckerType
    except Exception:
        CollisionCheckerType = None
from isaaclab.utils import configclass
from isaaclab_mimic.motion_planners.curobo.curobo_planner_cfg import CuroboPlannerCfg

from isaaclab_hiveboard.assets import ASSET_DIR
from isaaclab_hiveboard.assets import ASSET_DIR as SPOT_ASSET_DIR
from isaaclab_hiveboard.mdp.actions.curobo import CuRoboCommandCfg
from isaaclab_hiveboard.mdp.commands.frame_pose_command import (
    FramePoseCommandCfg,
)


@configclass
class CuroboCommandsCfg:
    """Command specifications for the MDP."""

    curobo_command = CuRoboCommandCfg(
        asset_name="robot",
        body_name="arm_link_jaw",
        resampling_time_range=(1e6, 1e6),
        debug_vis=True,
        frame_transformer_name="target_frame",
        planner_config=CuroboPlannerCfg(
            robot_config_file=CuRoboCommandCfg._create_temp_robot_yaml(
                yaml_path=f"{ASSET_DIR}/spot/cumotion/spot.yaml",
                urdf_path=f"{ASSET_DIR}/spot/spot_with_arm.urdf",
            ),
            robot_name="spot",
            gripper_joint_names=["arm_f1x"],
            gripper_open_positions={"arm_f1x": -1.57},
            gripper_closed_positions={"arm_f1x": 0.01},
            hand_link_names=["arm_link_jaw"],
            collision_spheres_file=f"{ASSET_DIR}/spot/cumotion/spot_mesh.yaml",
            collision_activation_distance=0.01,
            grasp_gripper_open_val=-1.57,
            approach_distance=0.0,
            retreat_distance=0.0,
            collision_checker_type=CollisionCheckerType.PRIMITIVE,
            max_planning_attempts=4,
            time_dilation_factor=0.6,
            enable_finetune_trajopt=True,
            n_repeat=50,
            motion_step_size=0.05,
            visualize_spheres=False,
            visualize_plan=False,
            debug_planner=False,
            sphere_update_freq=5,
            motion_noise_scale=0.0,
            world_config_file=f"{SPOT_ASSET_DIR}/example_scenes/cuboid_scene.yaml",
            # World extraction tuning for Spot envs
            world_ignore_substrings=["/World/defaultGroundPlane", "/curobo"],
        ),
    )


@configclass
class FramePoseCommandsCfg:
    """Command specifications for the RMP."""

    curobo_command: FramePoseCommandCfg = FramePoseCommandCfg(
        asset_name="robot",
        body_name="arm_link_jaw",
        frame_name="target_frame",
        resampling_time_range=(1e6, 1e6),
        debug_vis=True,
    )
