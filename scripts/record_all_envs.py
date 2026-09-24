#!/usr/bin/env python3
"""Record every environment registered by HiveBoard at its simulation-time frame rate."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def discover_tasks() -> list[str]:
    """Load tasks/__init__.py registrations without starting the simulator."""
    import gymnasium as gym
    import isaaclab_hiveboard.tasks  # noqa: F401

    return sorted(
        spec.id
        for spec in gym.registry.values()
        if str(spec.kwargs.get("env_cfg_entry_point", "")).startswith("isaaclab_hiveboard.")
    )


def positive_seconds(value: str) -> float:
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("must be positive and finite")
    return seconds


def player_command(task: str, output: Path, args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "-u",
        str(ROOT / "scripts/play.py"),
        "--task",
        task,
        "--num_envs",
        "1",
        "--device",
        args.device,
        "--seed",
        str(args.seed),
        "--duration",
        str(args.duration),
        "--video",
        "--video-folder",
        str(output),
        "--video-name",
        f"{task}.mp4",
        "--video-source",
        args.video_source,
        "--no-dataset",
        "--no-joint-log",
        "--visualizer",
        "newton" if args.viewer else "none",
        *([] if args.viewer else ["--headless"]),
    ]


def stop_process_group(proc: subprocess.Popen) -> None:
    """Stop a task and its encoder, including children left after a task crash."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            break
        if sig == signal.SIGTERM:
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=5)
    proc.wait()


def inspect_video(path: Path) -> dict:
    """Check that a finalized MP4 contains video and report its playback timing."""
    if not path.is_file():
        raise RuntimeError("No finalized MP4 was produced.")
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate,nb_frames:format=duration",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    info = json.loads(result.stdout)
    streams = info.get("streams", [])
    duration = float(info.get("format", {}).get("duration", 0))
    if not streams or duration <= 0 or int(streams[0].get("nb_frames", 0)) <= 0:
        raise RuntimeError("The MP4 contains no playable video frames.")
    return {
        "video": str(path),
        "fps": streams[0]["avg_frame_rate"],
        "frames": int(streams[0]["nb_frames"]),
        "duration_seconds": duration,
    }


def record_task(task: str, command: list[str], output: Path, timeout: float) -> dict:
    log_path = output / "logs" / f"{task}.log"
    result = {"task": task, "command": command, "log": str(log_path), "status": "failed"}
    start = time.monotonic()
    proc = None
    with log_path.open("w") as log:
        try:
            proc = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            returncode = proc.wait(timeout=timeout)
            result["returncode"] = returncode
            if returncode != 0:
                raise RuntimeError(f"Player exited with status {returncode}.")
            result.update(inspect_video(output / f"{task}.mp4"))
            result["status"] = "ok"
        except subprocess.TimeoutExpired:
            result.update(status="timeout", error=f"Exceeded {timeout:g} seconds of wall time.")
        except KeyboardInterrupt:
            result.update(status="interrupted", error="Recording interrupted by the user.")
        except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
            result["error"] = str(exc)
        finally:
            if proc is not None:
                stop_process_group(proc)
    result["wall_seconds"] = round(time.monotonic() - start, 3)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "videos/environments", help="Parent of dated run folders."
    )
    parser.add_argument(
        "--duration", type=positive_seconds, default=10.0, help="Maximum simulated seconds per clip (10)."
    )
    parser.add_argument("--timeout", type=positive_seconds, default=900.0, help="Maximum wall seconds per task (900).")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task", action="append", help="Exact registered task ID; repeat to select several.")
    parser.add_argument("--match", default="", help="Case-insensitive substring filter on task IDs.")
    parser.add_argument("--viewer", action="store_true", help="Also show the live Newton viewer.")
    parser.add_argument("--video-source", choices=("auto", "scene", "viewer"), default="auto")
    parser.add_argument("--list", action="store_true", help="Print selected task IDs without running simulations.")
    parser.add_argument("--dry-run", action="store_true", help="Print recording commands without running simulations.")
    args = parser.parse_args(argv)

    registered = discover_tasks()
    if args.task:
        unknown = sorted(set(args.task) - set(registered))
        if unknown:
            parser.error(f"Unknown HiveBoard task(s): {', '.join(unknown)}")
    tasks = [task for task in registered if (not args.task or task in args.task) and args.match.lower() in task.lower()]
    if not tasks:
        parser.error("No registered environments match the selection.")
    if args.list:
        print("\n".join(tasks))
        return 0

    output = args.output.resolve() / datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    if args.dry_run:
        for task in tasks:
            print(shlex.join(player_command(task, output, args)))
        return 0
    for binary in ("ffmpeg", "ffprobe"):
        if shutil.which(binary) is None:
            parser.error(f"{binary} must be installed and available on PATH.")
    (output / "logs").mkdir(parents=True)
    report = {"duration_limit_seconds": args.duration, "selected_tasks": tasks, "results": []}
    report_path = output / "summary.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Recording {len(tasks)} environments to {output}", flush=True)
    print("FPS is computed separately from each resolved environment's timestep and decimation.", flush=True)
    for index, task in enumerate(tasks, 1):
        print(f"[{index}/{len(tasks)}] {task}", flush=True)
        result = record_task(task, player_command(task, output, args), output, args.timeout)
        report["results"].append(result)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        if result["status"] == "ok":
            print(f"  saved {result['duration_seconds']:g}s at {result['fps']} FPS", flush=True)
        else:
            print(f"  {result['status']}: {result['error']} Log: {result['log']}", flush=True)
        if result["status"] == "interrupted":
            break
    passed = sum(result["status"] == "ok" for result in report["results"])
    print(f"Saved {passed}/{len(tasks)} videos. Report: {report_path}", flush=True)
    if report["results"] and report["results"][-1]["status"] == "interrupted":
        return 130
    return 0 if passed == len(tasks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
