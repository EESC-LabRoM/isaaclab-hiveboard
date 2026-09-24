"""CPU checks for video timing, encoder failures, and batch isolation."""

import json
import sys
from fractions import Fraction

import numpy as np
import pytest
import record_all_envs as batch
from isaaclab_hiveboard.utils.video import VideoWriter, simulation_fps


@pytest.mark.parametrize(
    ("dt", "decimation", "fps"),
    [(1 / 200, 15, Fraction(40, 3)), (1 / 500, 10, Fraction(50)), (1 / 300, 15, Fraction(20))],
)
def test_simulation_fps(dt, decimation, fps):
    assert simulation_fps(dt, decimation) == fps


def test_fractional_fps_round_trip(tmp_path):
    path = tmp_path / "fractional.mp4"
    writer = VideoWriter(path, 32, 24, simulation_fps(1 / 200, 15))
    for index in range(40):
        writer.write(np.full((24, 32, 3), index * 5, dtype=np.uint8))
    writer.close()
    info = batch.inspect_video(path)
    assert info["fps"] == "40/3"
    assert info["frames"] == 40
    assert info["duration_seconds"] == pytest.approx(3.0, abs=0.001)
    assert not writer.partial_path.exists()


def test_encoder_failure_does_not_publish_video(tmp_path):
    path = tmp_path / "failed.mp4"
    # yuv420p rejects odd dimensions; the encoder must report this failure.
    writer = VideoWriter(path, 3, 3, Fraction(50))
    writer.write(np.zeros((3, 3, 3), dtype=np.uint8))
    with pytest.raises(RuntimeError, match="Video encoding failed"):
        writer.close()
    assert not path.exists()


def test_empty_recording_is_not_success(tmp_path):
    path = tmp_path / "empty.mp4"
    writer = VideoWriter(path, 32, 24, Fraction(50))
    with pytest.raises(RuntimeError, match="0 frames"):
        writer.close()
    assert not path.exists()


def test_batch_continues_after_failure(tmp_path, monkeypatch):
    fixture = tmp_path / "fixture.mp4"
    writer = VideoWriter(fixture, 32, 24, Fraction(20))
    writer.write(np.zeros((24, 32, 3), dtype=np.uint8))
    writer.close()

    def command(task, output, args):
        if task == "fails":
            return [sys.executable, "-c", "raise SystemExit(7)"]
        return [
            sys.executable,
            "-c",
            "import shutil, sys; shutil.copyfile(sys.argv[1], sys.argv[2])",
            str(fixture),
            str(output / f"{task}.mp4"),
        ]

    monkeypatch.setattr(batch, "discover_tasks", lambda: ["fails", "succeeds"])
    monkeypatch.setattr(batch, "player_command", command)
    assert batch.main(["--output", str(tmp_path / "runs")]) == 1
    report = json.loads(next((tmp_path / "runs").glob("*/summary.json")).read_text())
    assert [result["status"] for result in report["results"]] == ["failed", "ok"]
    assert report["results"][0]["returncode"] == 7
    assert report["results"][1]["fps"] == "20/1"


def test_task_timeout(tmp_path):
    (tmp_path / "logs").mkdir()
    result = batch.record_task(
        "slow",
        [sys.executable, "-c", "import time; time.sleep(30)"],
        tmp_path,
        timeout=0.1,
    )
    assert result["status"] == "timeout"
    assert result["wall_seconds"] < 10


def test_discovery_includes_alias_and_validation_task():
    tasks = batch.discover_tasks()
    assert "validate_command_spot" in tasks
    assert "Spot-Manipulation-Lamp" in tasks
    assert "Isaac-HiveBoard-Spot-BallValve-Play-v0" in tasks
    assert "Isaac-HiveBoard-Franka-Button-v0" in tasks
    assert "Isaac-HiveBoard-Anymal-BenchValve-Play-v0" in tasks
    assert len(tasks) == len(set(tasks))
