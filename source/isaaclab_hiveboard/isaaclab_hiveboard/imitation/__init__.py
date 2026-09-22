# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Behaviour cloning and DAgger support for HiveBoard tasks.

The pieces here are deliberately split from ``scripts/imitation`` so that the
expert, the rollout loop and the dataset utilities stay importable without
parsing command-line arguments or launching a simulation.
"""

from .dataset import DatasetStats, dataset_stats, merge_datasets, rotate_recorder_dataset
from .expert import ScriptedExpert
from .policy import RobomimicPolicy
from .rollout import RolloutResult, run_rollout

__all__ = [
    "DatasetStats",
    "RobomimicPolicy",
    "RolloutResult",
    "ScriptedExpert",
    "dataset_stats",
    "merge_datasets",
    "rotate_recorder_dataset",
    "run_rollout",
]
