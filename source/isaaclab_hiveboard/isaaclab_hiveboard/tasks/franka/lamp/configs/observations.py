from isaaclab.managers import ObservationGroupCfg as ObsGroup, ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.cabinet import mdp

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "pose_command"})
        def __post_init__(self): self.enable_corruption = False; self.concatenate_terms = False
    policy: PolicyCfg = PolicyCfg()
