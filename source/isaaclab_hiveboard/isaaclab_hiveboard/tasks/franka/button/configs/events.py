from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.cabinet import mdp


@configclass
class FrankaButtonEventCfg:
    """Restore the robot, closed cover, and released button each episode."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
