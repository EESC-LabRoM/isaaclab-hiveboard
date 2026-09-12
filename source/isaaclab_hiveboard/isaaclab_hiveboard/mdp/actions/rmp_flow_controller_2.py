# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import isaacsim.core.utils.prims as prim_utils
import torch
from isaaclab.controllers.rmp_flow import RmpFlowController, RmpFlowControllerCfg
from isaaclab.utils import configclass
from isaacsim.core.api.objects import cuboid


@configclass
class RmpFlowControllerCfg2(RmpFlowControllerCfg):
    """Configuration for RMP-Flow controller with dynamic obstacles."""

    fixed_cuboid_prim_expr: str = "/World/envs/env_.*/Cube/geometry/mesh"
    debug_vis: bool = False


class RmpFlowController2(RmpFlowController):
    """Wraps around RMPFlow from IsaacSim for batched environments."""

    cfg: RmpFlowControllerCfg2

    def __init__(self, cfg: RmpFlowControllerCfg2, device: str):
        super().__init__(cfg, device)

    def initialize(self, prim_paths_expr: str):
        """Initialize the controller.

        Args:
            prim_paths_expr: The expression to find the articulation prim paths.
        """
        super().initialize(prim_paths_expr)

        self._obstacles = [
            cuboid.FixedCuboid(prim)
            for prim in prim_utils.find_matching_prim_paths(
                self.cfg.fixed_cuboid_prim_expr
            )
        ]

        for i, obstacle in enumerate(self._obstacles):
            obstacle.initialize()
            self.articulation_policies[i].get_motion_policy().add_obstacle(
                obstacle, static=False
            )

            if self.cfg.debug_vis and i == 0:
                # Visualize collision spheres for the first env only
                self.articulation_policies[
                    i
                ].get_motion_policy().visualize_collision_spheres()

    def compute(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Performs inference with the controller.

        Returns:
            The target joint positions and velocity commands.
        """
        for i, policy in enumerate(self.articulation_policies):
            if len(self._obstacles) > 0:
                policy.get_motion_policy().update_world([self._obstacles[i]])

        return super().compute()
