from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.cabinet import mdp

@configclass
class FrankaLampEventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
