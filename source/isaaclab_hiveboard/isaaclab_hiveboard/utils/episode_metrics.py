"""Vectorized episode metrics for Spot manipulation policy evaluation."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any

import torch


class EpisodeMetricTracker:
    """Accumulate state, action, and inference metrics for vectorized episodes."""

    def __init__(self, initial_observation: dict[str, torch.Tensor], step_dt: float):
        self.num_envs = initial_observation["valve_angle"].shape[0]
        self.device = initial_observation["valve_angle"].device
        self.step_dt = step_dt
        self.records: list[dict[str, Any]] = []
        self._inference_ms: list[list[float]] = [
            [] for _ in range(self.num_envs)
        ]
        self._allocate()
        self.reset(torch.ones(self.num_envs, dtype=torch.bool, device=self.device), initial_observation)

    def _allocate(self) -> None:
        count_shape = (self.num_envs,)
        float_zeros = lambda: torch.zeros(count_shape, device=self.device)
        long_zeros = lambda: torch.zeros(
            count_shape, dtype=torch.long, device=self.device
        )
        self.steps = long_zeros()
        self.path_length_m = float_zeros()
        self.max_valve_progress_rad = float_zeros()
        self.toward_target_steps = long_zeros()
        self.valve_motion_steps = long_zeros()
        self.action_norm_sum = float_zeros()
        self.action_delta_sum = float_zeros()
        self.action_delta_count = long_zeros()
        self.chunk_delta_sum = float_zeros()
        self.chunk_delta_count = long_zeros()
        self.clipped_components = long_zeros()
        self.saturated_components = long_zeros()
        self.action_components = long_zeros()
        self.has_previous_action = torch.zeros(
            count_shape, dtype=torch.bool, device=self.device
        )
        self.previous_action: torch.Tensor | None = None
        self.initial_valve_angle = float_zeros()
        self.previous_valve_angle = float_zeros()
        self.last_valve_angle = float_zeros()
        self.previous_ee_position = torch.zeros(
            (self.num_envs, 3), device=self.device
        )

    def reset(
        self, env_mask: torch.Tensor, observation: dict[str, torch.Tensor]
    ) -> None:
        """Reset accumulators for selected environments using their reset state."""
        if not env_mask.any():
            return
        valve_angle = observation["valve_angle"][:, 0]
        ee_position = observation["ee_pose_b"][:, :3]
        for tensor in (
            self.steps,
            self.path_length_m,
            self.max_valve_progress_rad,
            self.toward_target_steps,
            self.valve_motion_steps,
            self.action_norm_sum,
            self.action_delta_sum,
            self.action_delta_count,
            self.chunk_delta_sum,
            self.chunk_delta_count,
            self.clipped_components,
            self.saturated_components,
            self.action_components,
        ):
            tensor[env_mask] = 0
        self.has_previous_action[env_mask] = False
        self.initial_valve_angle[env_mask] = valve_angle[env_mask]
        self.previous_valve_angle[env_mask] = valve_angle[env_mask]
        self.last_valve_angle[env_mask] = valve_angle[env_mask]
        self.previous_ee_position[env_mask] = ee_position[env_mask]
        for env_id in env_mask.nonzero(as_tuple=False).flatten().tolist():
            self._inference_ms[env_id].clear()

    def record_step(
        self,
        observation: dict[str, torch.Tensor],
        actions: torch.Tensor,
        raw_actions: torch.Tensor,
        inference_ms: float | None,
        chunk_boundary: bool,
    ) -> None:
        """Record the state and policy command associated with one control step."""
        valve_angle = observation["valve_angle"][:, 0]
        ee_position = observation["ee_pose_b"][:, :3]
        self.path_length_m += torch.linalg.vector_norm(
            ee_position - self.previous_ee_position, dim=-1
        )
        valve_progress = self.initial_valve_angle - valve_angle
        self.max_valve_progress_rad = torch.maximum(
            self.max_valve_progress_rad, valve_progress
        )
        toward_delta = self.previous_valve_angle - valve_angle
        moving = toward_delta.abs() > 1.0e-6
        self.toward_target_steps += moving & (toward_delta > 0.0)
        self.valve_motion_steps += moving

        self.action_norm_sum += torch.linalg.vector_norm(actions, dim=-1)
        self.clipped_components += (raw_actions.abs() > 1.0).sum(dim=-1)
        self.saturated_components += (actions.abs() >= 0.999).sum(dim=-1)
        self.action_components += actions.shape[-1]
        if self.previous_action is None:
            self.previous_action = torch.zeros_like(actions)
        action_delta = torch.linalg.vector_norm(actions - self.previous_action, dim=-1)
        valid_delta = self.has_previous_action
        self.action_delta_sum += action_delta * valid_delta
        self.action_delta_count += valid_delta
        if chunk_boundary:
            self.chunk_delta_sum += action_delta * valid_delta
            self.chunk_delta_count += valid_delta
        self.previous_action.copy_(actions)
        self.has_previous_action[:] = True

        if inference_ms is not None:
            for values in self._inference_ms:
                values.append(inference_ms)
        self.steps += 1
        self.last_valve_angle.copy_(valve_angle)
        self.previous_valve_angle.copy_(valve_angle)
        self.previous_ee_position.copy_(ee_position)

    def finish(
        self,
        done: torch.Tensor,
        successful: torch.Tensor,
        include: torch.Tensor,
    ) -> list[dict[str, Any]]:
        """Finalize selected completed episodes and return their records."""
        completed = done & include
        new_records: list[dict[str, Any]] = []
        for env_id in completed.nonzero(as_tuple=False).flatten().tolist():
            steps = int(self.steps[env_id].item())
            inference = self._inference_ms[env_id]
            sorted_inference = sorted(inference)
            p95_index = max(0, math.ceil(0.95 * len(sorted_inference)) - 1)
            record = {
                "episode": len(self.records),
                "env_id": env_id,
                "success": bool(successful[env_id].item()),
                "steps": steps,
                "duration_s": steps * self.step_dt,
                "final_valve_angle_deg": math.degrees(
                    float(self.last_valve_angle[env_id].item())
                ),
                "max_valve_progress_deg": math.degrees(
                    float(self.max_valve_progress_rad[env_id].item())
                ),
                "tcp_path_length_m": float(self.path_length_m[env_id].item()),
                "valve_motion_toward_target_rate": self._ratio(
                    self.toward_target_steps[env_id],
                    self.valve_motion_steps[env_id],
                ),
                "mean_action_norm": self._ratio(
                    self.action_norm_sum[env_id], self.steps[env_id]
                ),
                "mean_action_delta": self._ratio(
                    self.action_delta_sum[env_id], self.action_delta_count[env_id]
                ),
                "mean_chunk_boundary_delta": self._ratio(
                    self.chunk_delta_sum[env_id], self.chunk_delta_count[env_id]
                ),
                "action_clipping_rate": self._ratio(
                    self.clipped_components[env_id], self.action_components[env_id]
                ),
                "action_saturation_rate": self._ratio(
                    self.saturated_components[env_id], self.action_components[env_id]
                ),
                "inference_mean_ms": fmean(inference) if inference else 0.0,
                "inference_p95_ms": sorted_inference[p95_index]
                if sorted_inference
                else 0.0,
                "inference_max_ms": max(inference) if inference else 0.0,
            }
            self.records.append(record)
            new_records.append(record)
        return new_records

    @staticmethod
    def _ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> float:
        denominator_value = float(denominator.item())
        return (
            float(numerator.item()) / denominator_value
            if denominator_value > 0.0
            else 0.0
        )


def save_evaluation_metrics(
    output_dir: str | Path,
    records: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> tuple[Path, Path]:
    """Write per-episode CSV data and an aggregate JSON summary."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    episodes_path = destination / "episodes.csv"
    summary_path = destination / "summary.json"

    if records:
        with episodes_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    else:
        episodes_path.write_text("", encoding="utf-8")

    successes = sum(bool(record["success"]) for record in records)
    count = len(records)
    rate = successes / count if count else 0.0
    z = 1.959963984540054
    denominator = 1.0 + z * z / count if count else 1.0
    center = (rate + z * z / (2.0 * count)) / denominator if count else 0.0
    margin = (
        z
        * math.sqrt(rate * (1.0 - rate) / count + z * z / (4.0 * count * count))
        / denominator
        if count
        else 0.0
    )
    numeric_fields = (
        "duration_s",
        "final_valve_angle_deg",
        "max_valve_progress_deg",
        "tcp_path_length_m",
        "valve_motion_toward_target_rate",
        "mean_action_norm",
        "mean_action_delta",
        "mean_chunk_boundary_delta",
        "action_clipping_rate",
        "action_saturation_rate",
        "inference_mean_ms",
        "inference_p95_ms",
        "inference_max_ms",
    )
    aggregate = {
        "episodes": count,
        "successes": successes,
        "failures": count - successes,
        "success_rate": rate,
        "success_rate_wilson_95": [max(0.0, center - margin), min(1.0, center + margin)],
        "means": {
            field: fmean(float(record[field]) for record in records)
            for field in numeric_fields
        }
        if records
        else {},
    }
    payload = {"metadata": metadata, "aggregate": aggregate}
    summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return episodes_path, summary_path
