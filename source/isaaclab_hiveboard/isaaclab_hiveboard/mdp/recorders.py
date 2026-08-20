# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Recorder terms for manipulation demonstrations."""

from __future__ import annotations

from datetime import datetime

import torch
from isaaclab.managers.recorder_manager import (
    DatasetExportMode,
    RecorderManagerBaseCfg,
    RecorderTerm,
    RecorderTermCfg,
)
from isaaclab.utils import configclass
from isaaclab.utils.datasets import HDF5DatasetFileHandler


class PreStepActionsRecorder(RecorderTerm):
    """Record the normalized delta-pose and gripper actions."""

    def record_pre_step(self):
        return "actions", self._env.action_manager.action


class PreStepDiffusionObservationsRecorder(RecorderTerm):
    """Record the flat diffusion_policy observation tensor directly."""

    def record_pre_step(self):
        return "observations", self._env.obs_buf["diffusion_policy"]


class PostStepProcessedActionsRecorder(RecorderTerm):
    """Record physical deltas after action scaling."""

    def record_post_step(self):
        processed_actions = [
            self._env.action_manager.get_term(name).processed_actions for name in self._env.action_manager.active_terms
        ]
        return "processed_actions", torch.cat(processed_actions, dim=-1)


class PostStepStatesRecorder(RecorderTerm):
    """Record the complete relative scene state after each step."""

    def record_post_step(self):
        return "states", self._env.scene.get_state(is_relative=True)


class PreStepRgbCameraRecorder(RecorderTerm):
    """Record RGB from a named scene camera under ``images/<key>``."""

    def record_pre_step(self):
        camera = self._env.scene[self.cfg.sensor_name]
        return f"images/{self.cfg.key}", camera.data.output["rgb"]


@configclass
class PreStepActionsRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = PreStepActionsRecorder


@configclass
class PreStepDiffusionObservationsRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = PreStepDiffusionObservationsRecorder


# Compatibility aliases
PreStepManipulationObservationsRecorder = PreStepDiffusionObservationsRecorder
PreStepManipulationObservationsRecorderCfg = PreStepDiffusionObservationsRecorderCfg


@configclass
class PostStepProcessedActionsRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = PostStepProcessedActionsRecorder


@configclass
class PostStepStatesRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = PostStepStatesRecorder


@configclass
class PreStepRgbCameraRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = PreStepRgbCameraRecorder
    sensor_name: str = "wrist_cam"
    key: str = "wrist"


@configclass
class SpotManipulationRecorderCfg(RecorderManagerBaseCfg):
    """Export successful fixed-base manipulation demonstrations as HDF5."""

    record_actions = PreStepActionsRecorderCfg()
    record_observations = PreStepDiffusionObservationsRecorderCfg()
    record_processed_actions = PostStepProcessedActionsRecorderCfg()

    dataset_file_handler_class_type: type = HDF5DatasetFileHandler
    dataset_export_mode: DatasetExportMode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
    dataset_export_dir_path: str = "logs/recorded_datasets"
    dataset_filename: str = f"spot_manipulation_demo_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    export_in_close: bool = False


@configclass
class SpotManipulationCameraRecorderCfg(SpotManipulationRecorderCfg):
    """Same HDF5 export as :class:`SpotManipulationRecorderCfg`, plus RGB cameras."""

    record_wrist_image = PreStepRgbCameraRecorderCfg(sensor_name="wrist_cam", key="wrist")
    record_scene_image = PreStepRgbCameraRecorderCfg(sensor_name="scene_cam", key="scene")
