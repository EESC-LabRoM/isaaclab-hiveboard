from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.cabinet import mdp


@configclass
class FrankaLeverValveEventCfg:
    """Configuration for Franka lever valve events."""

    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["panda_leftfinger", "panda_rightfinger"]
            ),
            "static_friction_range": (0.4, 0.4),
            "dynamic_friction_range": (0.4, 0.4),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
