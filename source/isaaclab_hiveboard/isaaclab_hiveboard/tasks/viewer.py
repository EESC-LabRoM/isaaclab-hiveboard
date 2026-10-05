# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Viewer camera shared by the ``-Play-`` tasks."""

# Environment frame: the robot stands at the origin facing +X and the
# mechanism sits 1 m ahead at about 0.65-0.8 m height (MECHANISM_SPAWN_POS,
# the lever and bench valves). The camera looks over the robot's right
# shoulder: the front of the body and the arm fill the lower left and the
# mechanism face sits right of centre, close enough to see the grasp.
PLAY_VIEWER_EYE = (0.55, -0.55, 1.02)
PLAY_VIEWER_LOOKAT = (0.90, 0.0, 0.68)


def use_play_viewer(env_cfg, shift: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> None:
    """Put the viewer at the shared Play camera in environment 0's frame.

    The camera is fixed rather than tracking a body, so every Play task opens
    on the same view and the live window matches recorded video. The viewer's
    ``asset_name`` is left alone: ``scripts/play.py --pose-debug`` reads it.

    Args:
        env_cfg: The task config, after its own ``__post_init__``.
        shift: Offset added to eye and target, for scenes whose mechanism is
            moved from the shared spawn (the FR3 tasks).
    """
    viewer = env_cfg.viewer
    viewer.origin_type = "env"
    viewer.env_index = 0
    viewer.eye = tuple(round(p + d, 6) for p, d in zip(PLAY_VIEWER_EYE, shift))
    viewer.lookat = tuple(round(p + d, 6) for p, d in zip(PLAY_VIEWER_LOOKAT, shift))
