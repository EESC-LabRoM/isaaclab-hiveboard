from dataclasses import MISSING

from isaaclab.envs import ManagerBasedEnv
from isaaclab.envs.mdp.actions.rmpflow_actions_cfg import RMPFlowActionCfg
from isaaclab.envs.mdp.actions.rmpflow_task_space_actions import RMPFlowAction
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

from isaaclab_hiveboard.mdp.actions.rmp_flow_controller_2 import (
    RmpFlowController2,
    RmpFlowControllerCfg2,
)


class RMPFlowAction2(RMPFlowAction):
    cfg: "RMPFlowActionCfg2"  # type: ignore

    def __init__(self, cfg: "RMPFlowActionCfg2", env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self._rmpflow_controller = RmpFlowController2(
            cfg=self.cfg.controller, device=self.device
        )


@configclass
class RMPFlowActionCfg2(RMPFlowActionCfg):
    class_type: type[ActionTerm] = RMPFlowAction2
    controller: RmpFlowControllerCfg2 = MISSING  # type: ignore
