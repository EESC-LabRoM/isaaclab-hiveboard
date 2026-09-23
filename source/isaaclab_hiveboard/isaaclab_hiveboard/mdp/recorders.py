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
from isaaclab.utils.configclass import configclass
from isaaclab.utils.datasets import HDF5DatasetFileHandler


class PreStepActionsRecorder(RecorderTerm):
    """Record the normalized delta-pose and gripper actions."""

    def record_pre_step(self):
        return "actions", self._env.action_manager.action


class PreStepDiffusionObservationsRecorder(RecorderTerm):
    """Record the flat diffusion_policy observation tensor directly."""

    def record_pre_step(self):
        return "observations", self._env.obs_buf["diffusion_policy"]


class PreStepEvaluationObservationsRecorder(RecorderTerm):
    """Record named evaluation observations, including contact forces."""

    def record_pre_step(self):
        return "evaluation", self._env.obs_buf["evaluation"]


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


class PreStepObservationGroupRecorder(RecorderTerm):
    """Record every term of an unconcatenated observation group separately.

    Returning the group's dict makes :class:`EpisodeData` recurse into
    ``<key>/<term_name>``, which is the layout robomimic reads back as
    ``data/demo_*/obs/<term_name>``.
    """

    def record_pre_step(self):
        group = self._env.obs_buf[self.cfg.group_name]
        if not isinstance(group, dict):
            raise RuntimeError(
                f"Observation group '{self.cfg.group_name}' is concatenated into a single tensor. "
                "robomimic needs one dataset per observation key: set concatenate_terms=False on the group."
            )
        return self.cfg.key, group


class PreStepExpertActionsRecorder(RecorderTerm):
    """Record the expert's label for the current state instead of the applied action.

    This is what makes DAgger rounds possible. The learner drives the
    environment, but the dataset must store what the expert *would* have done
    at each visited state, so the collection loop stashes that label on the
    environment as ``expert_actions`` before calling ``step``.
    """

    def record_pre_step(self):
        expert_actions = getattr(self._env, "expert_actions", None)
        if expert_actions is None:
            raise RuntimeError(
                "No expert label found on the environment. The rollout loop must set "
                "`env.unwrapped.expert_actions` to the expert's action for the current "
                "observation before every `env.step()` call."
            )
        return "actions", expert_actions


class PreStepExpertFallbackRecorder(RecorderTerm):
    """Record whether the expert's label for this step came from a cuRobo fallback.

    Stored per step as ``data/demo_*/expert_fallback`` so demonstrations and
    DAgger corrections produced by a patched or failed plan can be found and
    filtered later (see :func:`isaaclab_hiveboard.imitation.merge_datasets`).
    Tasks without the command term record nothing.
    """

    def record_pre_step(self):
        manager = self._env.command_manager
        if self.cfg.command_name not in manager.active_terms:
            return None, None
        term = manager.get_term(self.cfg.command_name)
        if not hasattr(term, "expert_fallback"):
            return None, None
        return "expert_fallback", term.expert_fallback()


@configclass
class PreStepActionsRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = PreStepActionsRecorder


@configclass
class PreStepDiffusionObservationsRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = PreStepDiffusionObservationsRecorder


@configclass
class PreStepEvaluationObservationsRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = PreStepEvaluationObservationsRecorder


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
class PreStepObservationGroupRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = PreStepObservationGroupRecorder
    group_name: str = "bc"
    key: str = "obs"


@configclass
class PreStepExpertActionsRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = PreStepExpertActionsRecorder


@configclass
class PreStepExpertFallbackRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = PreStepExpertFallbackRecorder
    command_name: str = "pose_command"


@configclass
class RobomimicRecorderCfg(RecorderManagerBaseCfg):
    """Export demonstrations in the layout robomimic's SequenceDataset expects.

    Writes ``data/demo_<i>/obs/<key>`` and ``data/demo_<i>/actions``. Only the
    applied action is recorded, so this config suits scripted-expert collection
    where the expert is also the actor. Use
    :class:`RobomimicDaggerRecorderCfg` for on-policy rounds.
    """

    record_obs = PreStepObservationGroupRecorderCfg(group_name="bc", key="obs")
    record_actions = PreStepActionsRecorderCfg()
    record_expert_fallback = PreStepExpertFallbackRecorderCfg()

    dataset_file_handler_class_type: type = HDF5DatasetFileHandler
    dataset_export_mode: DatasetExportMode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
    dataset_export_dir_path: str = "logs/imitation/datasets"
    dataset_filename: str = "demos"
    export_in_close: bool = True


@configclass
class RobomimicDaggerRecorderCfg(RobomimicRecorderCfg):
    """Robomimic layout with expert relabelling for on-policy DAgger rounds.

    ``record_actions`` is overridden so the stored action is the expert's label
    for the visited state rather than the learner action that was applied.

    Every episode is exported, successful or not. Filtering to successes would
    discard exactly the data DAgger exists to gather: the off-distribution
    states the learner drifts into and the expert's correction for them.
    """

    record_actions = PreStepExpertActionsRecorderCfg()
    dataset_export_mode: DatasetExportMode = DatasetExportMode.EXPORT_ALL


@configclass
class SpotManipulationRecorderCfg(RecorderManagerBaseCfg):
    """Export successful fixed-base manipulation demonstrations as HDF5."""

    record_actions = PreStepActionsRecorderCfg()
    record_observations = PreStepDiffusionObservationsRecorderCfg()
    record_evaluation = PreStepEvaluationObservationsRecorderCfg()
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
