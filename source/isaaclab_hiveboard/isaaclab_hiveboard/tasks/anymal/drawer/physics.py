# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Contact resolution for the drawer's millimetre-scale guide clearances."""

from isaaclab.utils import configclass

from isaaclab_hiveboard.tasks.anymal.mechanism import MechanismPhysicsCfg

DRAWER_SIM_DT = 1.0 / 1000.0
# Preserve the existing 75 ms command/action period when reducing the tick.
DRAWER_DECIMATION = 75


@configclass
class DrawerPhysicsCfg(MechanismPhysicsCfg):
    newton_mjwarp = MechanismPhysicsCfg().newton_mjwarp.replace(
        class_type=None,
        num_substeps=2,
        collision_decimation=1,
        use_cuda_graph=True,
        default_shape_cfg=MechanismPhysicsCfg().newton_mjwarp.default_shape_cfg.replace(gap=0.0001),
    )
    default = newton_mjwarp
