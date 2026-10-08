"""Geometric gate for the free drawer.

Overlaps come from the URDF collision shapes (see drawer_path.py), not from a
second copy of the slot. Open and the pull command are the box's forward
travel. The USD check reads the files the drawer scene spawns.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade

import drawer_path as slot
from isaaclab_hiveboard.tasks.anymal.drawer.configs.commands import DRAWER_OPEN, FramePoseCommandsCfg
from isaaclab_hiveboard.tasks.anymal.drawer.configs.terminations import TerminationsCfg
from isaaclab_hiveboard.tasks.anymal.drawer.slide import done_when_travel_holds, drawer_slide_success
from isaaclab_hiveboard.tasks.anymal.drawer.slide import drawer_clear_of_guides, drawer_removal_success
from isaaclab_hiveboard.tasks.scenes.drawer import DRAWER_PULL, DrawerSceneCfg

_REPO = Path(__file__).resolve().parents[1]
_FORBIDDEN = ("PrismaticJoint", "LiftJoint")
_TASK_FILES = (
    _REPO / "source/isaaclab_hiveboard/isaaclab_hiveboard/tasks/scenes/drawer.py",
    _REPO / "source/isaaclab_hiveboard/isaaclab_hiveboard/tasks/anymal/drawer/env.py",
    _REPO / "source/isaaclab_hiveboard/isaaclab_hiveboard/tasks/anymal/drawer/configs/commands.py",
    _REPO / "source/isaaclab_hiveboard/isaaclab_hiveboard/tasks/anymal/drawer/configs/terminations.py",
    _REPO / "source/isaaclab_hiveboard/isaaclab_hiveboard/tasks/franka/drawer/env.py",
    _REPO / "source/isaaclab_hiveboard/isaaclab_hiveboard/tasks/spot/drawer/env.py",
    _REPO / "configs/Isaac-HiveBoard-Anymal-Drawer-v0.json",
)


def _probe_distance(collision: slot.DrawerCollision) -> float:
    radii = {shaft.radius for shaft in collision.shafts}
    assert len(radii) == 1
    return radii.pop()


def _sides(pairs: list[tuple[str, str]]) -> set[str]:
    return {name.split("_")[1] for name, _cut in pairs}


def _cfg(name: str):
    field = DrawerSceneCfg.__dataclass_fields__[name]
    if field.default_factory is not None:  # type: ignore[attr-defined]
        return field.default_factory()  # type: ignore[misc]
    return field.default


def test_robot_adapters_preserve_drawer_timing():
    """Robot adapters must preserve the fine tick and existing command rate."""
    from isaaclab_hiveboard.tasks.anymal.drawer.env import AnymalDrawerEnvCfg, AnymalDrawerEnvCfg_PLAY
    from isaaclab_hiveboard.tasks.franka.drawer.env import FrankaDrawerEnvCfg, FrankaDrawerEnvCfg_PLAY
    from isaaclab_hiveboard.tasks.spot.drawer.env import SpotDrawerEnvCfg, SpotDrawerEnvCfg_PLAY
    from isaaclab_hiveboard.tasks.anymal.drawer.slide import drawer_removal_success, drawer_slide_success

    removal = (AnymalDrawerEnvCfg, AnymalDrawerEnvCfg_PLAY, FrankaDrawerEnvCfg, FrankaDrawerEnvCfg_PLAY)
    opening = (SpotDrawerEnvCfg, SpotDrawerEnvCfg_PLAY)
    for cls in (*removal, *opening):
        cfg = cls()
        assert cfg.sim.dt <= 0.001, cls.__name__
        assert abs(cfg.sim.dt * cfg.decimation - 0.075) < 1e-9, cls.__name__
        expected_success = drawer_removal_success if cls in removal else drawer_slide_success
        assert cfg.terminations.success.func is expected_success, cls.__name__


def test_runtime_collision_matches_urdf():
    """Check the generated runtime USD, including units and collision schemas."""
    links = {link.attrib["name"]: link for link in ET.parse(slot.DRAWER_URDF).getroot().findall("link")}
    for asset, link in (("drawer_housing", "base"), ("drawer", "drawer")):
        stage = Usd.Stage.Open(_cfg(asset).spawn.usd_path)
        cache = UsdGeom.XformCache()
        prims = {prim.GetName(): prim for prim in stage.Traverse()}
        collisions = links[link].findall("collision")
        expected = {item.attrib["name"] for item in collisions}
        active = {
            prim.GetName() for prim in stage.Traverse()
            if prim.HasAPI(UsdPhysics.CollisionAPI)
            and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False
        }
        assert active == expected, (asset, active, expected)
        for item in collisions:
            name = item.attrib["name"]
            prim = prims[name]
            material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
            mu = UsdPhysics.MaterialAPI(material.GetPrim()).GetDynamicFrictionAttr().Get()
            assert abs(mu - (0.8 if name == "face" else 0.08)) < 1e-6, name
            if name == "face":
                assert prim.GetAttribute("mjc:condim").Get() == 4
            center, _ = slot._origin(item.find("origin"))
            transform = cache.GetLocalToWorldTransform(prim)
            actual = transform.Transform(Gf.Vec3d(0, 0, 0))
            assert all(abs(a - b) < 1e-7 for a, b in zip(actual, center)), name
            geometry = item.find("geometry")
            cube = geometry.find("box")
            if cube is not None:
                size = tuple(float(v) for v in cube.attrib["size"].split())
                for i in range(3):
                    axis = Gf.Vec3d(*(1.0 if j == i else 0.0 for j in range(3)))
                    actual_size = transform.TransformDir(axis).GetLength() * UsdGeom.Cube(prim).GetSizeAttr().Get()
                    assert abs(actual_size - size[i]) < 1e-7, name
            else:
                cylinder = geometry.find("cylinder")
                shape = UsdGeom.Cylinder(prim)
                assert abs(shape.GetRadiusAttr().Get() - float(cylinder.attrib["radius"])) < 1e-7, name
                assert abs(shape.GetHeightAttr().Get() - float(cylinder.attrib["length"])) < 1e-7, name
                axis = shape.GetAxisAttr().Get()
                unit_axis = Gf.Vec3d(*(1.0 if j == "XYZ".index(axis) else 0.0 for j in range(3)))
                direction = transform.TransformDir(unit_axis).GetNormalized()
                assert abs(direction[1]) > 0.999999, name


def test_drawer_leaves_only_forward_then_up():
    collision = slot.load_drawer_collision()
    distance = _probe_distance(collision)
    assert "prismatic" not in collision.joint_types
    assert collision.joint_types == ("floating",)
    closed = (0.0, 0.0, 0.0)

    assert slot.y_captured(collision, closed)
    assert not slot.hitting(collision, closed)
    assert not slot.hitting(collision, slot.probe(closed, 0, distance))
    up_hits = slot.hitting(collision, slot.probe(closed, 2, distance))
    assert _sides(up_hits) == {"left", "right"}

    # The commanded open pose and the pull frame are still under both ceilings.
    forward = slot.forward_end(collision, distance)
    for name, travel in (("open", DRAWER_OPEN), ("pull", DRAWER_PULL)):
        pose = (travel, 0.0, 0.0)
        assert travel < forward, name
        ceiling_hits = [
            pair for pair in slot.hitting(collision, slot.probe(pose, 2, distance)) if "ceiling" in pair[1]
        ]
        assert _sides(ceiling_hits) == {"left", "right"}, (name, ceiling_hits)

    at_lip = (forward, 0.0, 0.0)
    assert forward > distance
    assert slot.y_captured(collision, at_lip)
    assert not slot.hitting(collision, at_lip)
    assert not slot.hitting(collision, slot.probe(at_lip, 2, distance))
    lip_hits = slot.hitting(collision, slot.probe(at_lip, 0, distance))
    assert _sides(lip_hits) == {"left", "right"}

    lift = slot.raised_clear(collision, forward, distance)
    raised = (forward, 0.0, lift)
    assert lift > 0.0
    assert not slot.hitting(collision, raised)
    assert not slot.hitting(collision, slot.probe(raised, 0, distance))
    # The same rise from the closed pose is still inside the ceiling.
    assert slot.hitting(collision, (0.0, 0.0, lift))
    # The finite CAD lip blocks a continuous pull at the face. A teleported
    # pose far beyond the physical housing is collision-free.
    assert slot.hitting(collision, slot.probe(at_lip, 0, distance))
    assert not slot.hitting(collision, (0.2, 0.0, 0.0))

    for pose in (closed, at_lip, raised):
        for extra in collision.extras:
            moved = slot.Box(
                extra.name,
                (extra.center[0] + pose[0], extra.center[1] + pose[1], extra.center[2] + pose[2]),
                extra.size,
            )
            for cut in collision.cuts:
                assert not _boxes_overlap(moved, cut), (pose, extra.name, cut.name)


class _Tensor:
    def __init__(self, value: torch.Tensor):
        self.torch = value


class _Data:
    def __init__(self, pos: torch.Tensor, quat: torch.Tensor):
        self.root_pos_w = _Tensor(pos)
        self.root_quat_w = _Tensor(quat)


class _Asset:
    def __init__(self, pos: torch.Tensor, quat: torch.Tensor):
        self.data = _Data(pos, quat)


class _Command:
    def __init__(self, done: bool):
        self._done = done

    def is_done(self) -> torch.Tensor:
        return torch.tensor([self._done])


class _Env:
    """Housing at the origin. ``local`` is the box origin in the housing frame."""

    def __init__(self, local: tuple[float, float, float], housing_quat: tuple[float, float, float, float], done: bool):
        import isaaclab.utils.math as math_utils

        quat = torch.tensor([housing_quat])
        world = math_utils.quat_apply(quat, torch.tensor([local]))
        self.scene = {
            "drawer": _Asset(world, quat),
            "drawer_housing": _Asset(torch.zeros(1, 3), quat),
        }
        self.command_manager = _Commands(done)


class _Commands:
    def __init__(self, done: bool):
        self._command = _Command(done)

    def get_term(self, name: str) -> _Command:
        assert name == "pose_command"
        return self._command


def _open_cases():
    """(label, box position in the housing frame, housing quat xyzw, expect open)."""
    yaw = (0.0, 0.0, 1.0, 0.0)  # 180 deg, the scene's FACE_QUAT. Local +X is world -X.
    return (
        ("closed", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), False),
        ("lifted", (0.0, 0.0, 0.03), (0.0, 0.0, 0.0, 1.0), False),
        ("open", (DRAWER_OPEN, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), True),
        ("open-yawed", (DRAWER_OPEN, 0.0, 0.0), yaw, True),
        ("lifted-yawed", (0.0, 0.0, 0.03), yaw, False),
    )


def test_open_is_forward_travel():
    """Pull completion and success follow 0.02 m of forward travel, not a joint."""
    for path in _TASK_FILES:
        text = path.read_text()
        for name in _FORBIDDEN:
            assert name not in text, (path.name, name)

    pull = FramePoseCommandsCfg().pose_command.commands[3]
    assert pull.done_when_joint is None
    assert pull.done_when_travel == ("drawer", "drawer_housing", DRAWER_OPEN, 1.0)
    success = TerminationsCfg().success
    assert success.func is drawer_slide_success
    assert success.params["open_distance"] == DRAWER_OPEN
    assert "ranges" not in success.params

    env_ids = torch.tensor([0])
    for label, pos, quat, expect in _open_cases():
        env = _Env(pos, quat, done=True)
        held = done_when_travel_holds(env, env_ids, pull.done_when_travel)
        opened = drawer_slide_success(env, "pose_command", "drawer", "drawer_housing", DRAWER_OPEN)
        assert bool(held.item()) is expect, label
        assert bool(opened.item()) is expect, label

    # Sequence not finished: travel alone is not success.
    finished = _Env((DRAWER_OPEN, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), done=False)
    assert not bool(drawer_slide_success(finished, "pose_command", "drawer", "drawer_housing").item())

    setup = json.loads((_REPO / "configs/Isaac-HiveBoard-Anymal-Drawer-v0.json").read_text())
    travels = []
    for command in setup["commands"]:
        params = command["parameters"]
        blob = json.dumps(params)
        for name in _FORBIDDEN:
            assert name not in blob
        if params.get("done_when_travel") is not None:
            travels.append(params["done_when_travel"])
    assert travels
    assert all(item[2] >= DRAWER_OPEN and item[3] >= item[2] for item in travels)

    for name in ("drawer_housing", "drawer"):
        cfg = _cfg(name)
        stage = Usd.Stage.Open(cfg.spawn.usd_path)
        assert stage is not None, cfg.spawn.usd_path
        prismatic = [
            prim.GetPath().pathString
            for prim in stage.Traverse()
            if prim.IsA(UsdPhysics.PrismaticJoint) or prim.GetTypeName() == "PhysicsPrismaticJoint"
        ]
        assert prismatic == [], (name, prismatic)
    housing_stage = Usd.Stage.Open(_cfg("drawer_housing").spawn.usd_path)
    fixed = [prim.GetName() for prim in housing_stage.Traverse() if prim.IsA(UsdPhysics.FixedJoint)]
    assert sorted(fixed) == ["anchor_joint", "root_joint"]
    box_stage = Usd.Stage.Open(_cfg("drawer").spawn.usd_path)
    assert [prim for prim in box_stage.Traverse() if prim.IsA(UsdPhysics.Joint)] == []
    assert _cfg("drawer").spawn.rigid_props.kinematic_enabled is False
    assert _cfg("drawer_housing").actuators == {}


def test_removal_requires_both_shafts_clear():
    """Opening, lifting inside the slot, or clearing only one pin is insufficient."""
    import isaaclab.utils.math as math_utils

    for housing_quat in ((0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 1.0, 0.0)):
        for pose, expected in (
            ((0.0, 0.0, 0.0), False),
            ((DRAWER_OPEN, 0.0, 0.0), False),
            ((0.0256, 0.0, 0.012), False),
            ((0.070, 0.0, 0.0), True),
            ((0.070, 0.0, 0.030), True),
            ((0.070, 0.0, -0.300), True),
        ):
            env = _Env(pose, housing_quat, done=True)
            assert bool(drawer_removal_success(env, "pose_command", "drawer", "drawer_housing")[0]) is expected

        env = _Env((0.050, 0.0, 0.030), housing_quat, done=True)
        # Yaw puts one pin behind the case even though the box origin moved 50 mm.
        quarter_turn = torch.tensor([[0.0, 0.0, 2 ** -0.5, 2 ** -0.5]])
        env.scene["drawer"].data.root_quat_w.torch = math_utils.quat_mul(torch.tensor([housing_quat]), quarter_turn)
        assert not bool(drawer_clear_of_guides(env, "drawer", "drawer_housing")[0])
        assert not bool(drawer_removal_success(env, "pose_command", "drawer", "drawer_housing")[0])

    unfinished = _Env((0.070, 0.0, 0.030), (0.0, 0.0, 0.0, 1.0), done=False)
    assert not bool(drawer_removal_success(unfinished, "pose_command", "drawer", "drawer_housing")[0])


def _boxes_overlap(a: slot.Box, b: slot.Box) -> bool:
    for i in range(3):
        a0, a1 = a.bounds()[i]
        b0, b1 = b.bounds()[i]
        if min(a1, b1) - max(a0, b0) <= 1e-9:
            return False
    return True
