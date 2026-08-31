from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.cabinet import mdp

from isaaclab_hiveboard.mdp.terminations import articulation_joint_position_success
from isaaclab_hiveboard.tasks.spot.lamp.configs.scene import LAMP_SEATED_POSITION


@configclass
class TerminationsCfg:
    """The lamp is successful only when it is physically seated."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=articulation_joint_position_success,
        params={
            "command_name": "pose_command",
            "asset_cfg": SceneEntityCfg("lamp", joint_names=["PrismaticJoint"]),
            "target": LAMP_SEATED_POSITION,
            "tolerance": 0.004,
        },
    )
