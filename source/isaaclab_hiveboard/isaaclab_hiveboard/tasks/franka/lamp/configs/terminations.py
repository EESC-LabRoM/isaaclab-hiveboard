from isaaclab.managers import TerminationTermCfg as DoneTerm, SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.cabinet import mdp
from isaaclab_hiveboard.mdp.terminations import articulation_joint_position_success
from isaaclab_hiveboard.tasks.spot.lamp.configs.scene import LAMP_UNSCREWED_POSITION

@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=articulation_joint_position_success,
        params={
            "command_name": "pose_command",
            "asset_cfg": SceneEntityCfg("lamp", joint_names=["PrismaticJoint"]),
            "target": LAMP_UNSCREWED_POSITION,
            "tolerance": 0.002,
        },
    )
