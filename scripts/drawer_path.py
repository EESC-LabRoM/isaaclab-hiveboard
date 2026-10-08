"""Shaft-versus-cut overlap for the shipped drawer URDF.

Shapes come from the URDF collision elements. A pose is the free box's
translation in the housing frame, in metres. Nothing here re-states the slot
outline, and nothing stops the box except those shapes.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DRAWER_URDF = (
    REPO_ROOT
    / "source/isaaclab_hiveboard/isaaclab_hiveboard/assets/hiveboard/drawer/Drawer_Assembly.urdf"
)


@dataclass(frozen=True)
class Box:
    """Axis-aligned box, center and full size, metres."""

    name: str
    center: tuple[float, float, float]
    size: tuple[float, float, float]

    def bounds(self) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
        return tuple(
            (self.center[i] - 0.5 * self.size[i], self.center[i] + 0.5 * self.size[i]) for i in range(3)
        )  # type: ignore[return-value]


@dataclass(frozen=True)
class YCylinder:
    """Finite cylinder whose axis is parallel to Y."""

    name: str
    center: tuple[float, float, float]
    radius: float
    length: float

    def bounds(self) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
        x, y, z = self.center
        return (
            (x - self.radius, x + self.radius),
            (y - self.length / 2, y + self.length / 2),
            (z - self.radius, z + self.radius),
        )

    def translated(self, dx: float, dy: float, dz: float) -> YCylinder:
        x, y, z = self.center
        return YCylinder(self.name, (x + dx, y + dy, z + dz), self.radius, self.length)


@dataclass(frozen=True)
class DrawerCollision:
    cuts: tuple[Box | YCylinder, ...]
    shafts: tuple[YCylinder, ...]
    extras: tuple[Box, ...]
    # URDF joint types. The box is free, so this is not a set of limits.
    joint_types: tuple[str, ...]


def _rpy_matrix(roll: float, pitch: float, yaw: float) -> list[list[float]]:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]


def _origin(elem: ET.Element | None) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if elem is None:
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    xyz = tuple(float(v) for v in elem.attrib.get("xyz", "0 0 0").split())
    rpy = tuple(float(v) for v in elem.attrib.get("rpy", "0 0 0").split())
    return xyz, rpy  # type: ignore[return-value]


def _axis_index(rotation: list[list[float]]) -> int:
    """Column of ``rotation`` that maps local Z onto a world axis. Cylinder axis in URDF is local Z."""
    column = [rotation[row][2] for row in range(3)]
    dominant = max(range(3), key=lambda i: abs(column[i]))
    if abs(column[dominant]) < 0.999:
        raise ValueError(f"cylinder axis is not aligned with a world axis: {column}")
    return dominant


def load_drawer_collision(urdf: Path | None = None) -> DrawerCollision:
    """Parse boxes, Y-cylinders and the URDF joint types from ``urdf``."""
    root = ET.parse(str(urdf or DRAWER_URDF)).getroot()
    cuts: list[Box | YCylinder] = []
    shafts: list[YCylinder] = []
    extras: list[Box] = []
    for link in root.findall("link"):
        for collision in link.findall("collision"):
            name = collision.attrib.get("name") or ""
            origin, rpy = _origin(collision.find("origin"))
            geometry = collision.find("geometry")
            if geometry is None:
                continue
            box = geometry.find("box")
            cylinder = geometry.find("cylinder")
            if box is not None:
                if any(abs(v) > 1e-8 for v in rpy):
                    raise ValueError(f"collision {name} box is rotated; the overlap test expects axis-aligned boxes")
                size = tuple(float(v) for v in box.attrib["size"].split())
                shape = Box(name, origin, size)  # type: ignore[arg-type]
                if name.startswith("cut_"):
                    cuts.append(shape)
                else:
                    extras.append(shape)
            elif cylinder is not None:
                rotation = _rpy_matrix(*rpy)
                axis = _axis_index(rotation)
                if axis != 1:
                    raise ValueError(f"collision {name} cylinder axis is not Y")
                (cuts if name.startswith("cut_") else shafts).append(
                    YCylinder(
                        name,
                        origin,
                        float(cylinder.attrib["radius"]),
                        float(cylinder.attrib["length"]),
                    )
                )
            else:
                raise ValueError(f"collision {name} is not a box or a cylinder")
    joint_types = tuple(joint.attrib.get("type", "") for joint in root.findall("joint"))
    if not cuts or not shafts:
        raise ValueError("drawer URDF has no cut boxes or shaft cylinders")
    return DrawerCollision(tuple(cuts), tuple(shafts), tuple(extras), joint_types)


def _intervals_overlap(a0: float, a1: float, b0: float, b1: float, eps: float = 1e-9) -> bool:
    return min(a1, b1) - max(a0, b0) > eps


def _circle_rect(cx: float, cz: float, radius: float, x0: float, x1: float, z0: float, z1: float) -> bool:
    nearest_x = min(max(cx, x0), x1)
    nearest_z = min(max(cz, z0), z1)
    return (cx - nearest_x) ** 2 + (cz - nearest_z) ** 2 < (radius - 1e-9) ** 2


def intersects(shaft: YCylinder, box: Box | YCylinder) -> bool:
    """Positive-volume overlap of a shaft and a box or parallel cylinder."""
    (x0, x1), (y0, y1), (z0, z1) = box.bounds()
    sy0 = shaft.center[1] - 0.5 * shaft.length
    sy1 = shaft.center[1] + 0.5 * shaft.length
    if not _intervals_overlap(sy0, sy1, y0, y1):
        return False
    if isinstance(box, YCylinder):
        return (
            (shaft.center[0] - box.center[0]) ** 2 + (shaft.center[2] - box.center[2]) ** 2
            < (shaft.radius + box.radius - 1e-9) ** 2
        )
    return _circle_rect(shaft.center[0], shaft.center[2], shaft.radius, x0, x1, z0, z1)


Pose = tuple[float, float, float]


def shafts_at(collision: DrawerCollision, pose: Pose) -> tuple[YCylinder, ...]:
    """Shafts after translating the free box by ``pose`` in the housing frame."""
    return tuple(shaft.translated(*pose) for shaft in collision.shafts)


def hitting(collision: DrawerCollision, pose: Pose) -> list[tuple[str, str]]:
    """Pairs (shaft, cut) with positive overlap at ``pose``."""
    pairs = []
    for shaft in shafts_at(collision, pose):
        for cut in collision.cuts:
            if intersects(shaft, cut):
                pairs.append((shaft.name, cut.name))
    return pairs


def y_captured(collision: DrawerCollision, pose: Pose) -> bool:
    """Each shaft overlaps the Y span of the cuts on its own side of the housing."""
    placed = shafts_at(collision, pose)
    for shaft in placed:
        side = "left" if shaft.center[1] < 0.0 else "right"
        cuts = [cut for cut in collision.cuts if side in cut.name]
        if not cuts:
            return False
        sy0 = shaft.center[1] - 0.5 * shaft.length
        sy1 = shaft.center[1] + 0.5 * shaft.length
        if not any(_intervals_overlap(sy0, sy1, *cut.bounds()[1]) for cut in cuts):
            return False
    return len(placed) >= 2


def probe(pose: Pose, axis: int, distance: float) -> Pose:
    shifted = [pose[0], pose[1], pose[2]]
    shifted[axis] = shifted[axis] + distance
    return (shifted[0], shifted[1], shifted[2])


def forward_end(collision: DrawerCollision, probe_distance: float) -> float:
    """First contact along a continuous, level forward pull.

    The search stops at the cuts. There is no joint limit to hide a short lip.
    """
    step = probe_distance / 2.0
    last_free = 0.0
    value = 0.0
    while value <= 0.6 + 1e-12:
        pose = (value, 0.0, 0.0)
        if hitting(collision, pose):
            low, high = last_free, value
            for _ in range(40):
                middle = (low + high) / 2
                if hitting(collision, (middle, 0.0, 0.0)):
                    high = middle
                else:
                    low = middle
            return low
        last_free = value
        value += step
    raise RuntimeError("no forward collision along the guide")


def raised_clear(collision: DrawerCollision, forward: float, probe_distance: float) -> float:
    """Smallest upward move at ``forward`` that frees the next forward step.

    The pose itself must miss every cut. Lift 0 is not a candidate: there the
    lower lip still blocks a straight pull.
    """
    step = probe_distance / 2.0
    value = step
    while value <= 0.05 + 1e-12:
        pose = (forward, 0.0, value)
        ahead = probe(pose, 0, probe_distance)
        if not hitting(collision, pose) and not hitting(collision, ahead):
            return value
        value += step
    raise RuntimeError("raising at the forward end never opens the exit")
