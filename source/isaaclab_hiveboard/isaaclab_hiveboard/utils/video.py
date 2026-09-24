"""Streaming MP4 output with simulation-time frame rates (no simulator imports)."""

from __future__ import annotations

import contextlib
import math
import shutil
import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path


def simulation_fps(physics_dt: float, decimation: int) -> Fraction:
    """Frame rate for one captured frame per environment step."""
    step_dt = float(physics_dt) * int(decimation)
    if not math.isfinite(step_dt) or step_dt <= 0:
        raise ValueError("The environment step duration must be positive and finite.")
    return 1 / Fraction(step_dt).limit_denominator(1_000_000_000)


class VideoWriter:
    """Encode RGB frames; publish the final filename only after successful encoding."""

    def __init__(self, path: str | Path, width: int, height: int, fps: Fraction):
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("Video recording requires ffmpeg on PATH.")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.partial_path = self.path.with_suffix(".partial.mp4")
        self.frames = 0
        self.width, self.height = width, height
        self._stderr = tempfile.TemporaryFile(mode="w+b")  # noqa: SIM115 - closed by close(), including failures
        try:
            self._proc = subprocess.Popen(
                [
                    ffmpeg,
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "-s",
                    f"{width}x{height}",
                    "-framerate",
                    str(fps),
                    "-i",
                    "-",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-crf",
                    "18",
                    "-movflags",
                    "+faststart",
                    str(self.partial_path),
                ],
                stdin=subprocess.PIPE,
                stderr=self._stderr,
            )
        except BaseException:
            self._stderr.close()
            raise

    def write(self, frame) -> None:
        import numpy as np

        if frame is None:
            raise RuntimeError("The video source returned no RGB frame.")
        if frame.shape != (self.height, self.width, 3) or frame.dtype != np.uint8:
            raise ValueError(f"Expected uint8 RGB frame of shape {(self.height, self.width, 3)}; got {frame.shape}.")
        self._proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        self.frames += 1

    def close(self, *, save: bool = True) -> None:
        try:
            with contextlib.suppress(BrokenPipeError):
                self._proc.stdin.close()
            try:
                returncode = self._proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
                raise RuntimeError("ffmpeg did not finish within 60 seconds.") from None
            self._stderr.seek(0)
            error = self._stderr.read().decode(errors="replace")
            if save:
                if returncode != 0 or self.frames == 0:
                    raise RuntimeError(f"Video encoding failed ({self.frames} frames, exit {returncode}): {error}")
                self.partial_path.replace(self.path)
        finally:
            self._stderr.close()
