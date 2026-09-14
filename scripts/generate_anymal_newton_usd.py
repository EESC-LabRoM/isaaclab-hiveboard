# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bake kitless Newton USD assets for the ANYmal-D + DynaArm + Robotiq 2F-140.

The Newton tasks load preconverted USD (gitignored ``**/*.usd*``). This script
is the record of how those files are produced. Everything here is kitless
(pxr / usdex / numpy / pycollada / stdlib urllib) so it runs without Isaac Sim:

1. ``fetch``: mirror the Nucleus ANYmal-D + Robotiq 2F-140 trees from the public
   Omniverse S3 bucket into ``assets/anymal/usd/nucleus/`` (skipped with
   ``--skip-download`` when already vendored; the ANYmal-D tree is also seeded
   from the local Nucleus cache when present).
2. ``dynaarm``: author ``assets/anymal/usd/dynaarm.usd`` from the committed
   arm-only URDF (links, inertials, primitive collisions, revolute/fixed
   joints, best-effort DAE visuals). Mirrors the urdf-usd-converter mapping
   (joint limits in degrees, rpy XYZ composition).
3. ``assembly``: author ``assets/anymal/usd/anymal_d_dynaarm_robotiq.usda``, an
   overlay that references the three assets, welds arm->base and palm->flange
   with fixed joints (same relative-pose math as the Isaac Sim spawn func in
   ``assets/anymal/anymal.py``), and strips nested articulation roots plus the
   gripper's world-fixed joint.
4. ``verify``: open the baked stage with pxr and check joints / bodies /
   articulation roots.

Usage::

    uv run python scripts/generate_anymal_newton_usd.py --verify-only
    uv run python scripts/generate_anymal_newton_usd.py --skip-download
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
EXT_ASSETS = REPO_ROOT / "source/isaaclab_hiveboard/isaaclab_hiveboard/assets"
ANYMAL_USD_DIR = EXT_ASSETS / "anymal" / "usd"
NUCLEUS_DIR = ANYMAL_USD_DIR / "nucleus"
DYNAARM_URDF = EXT_ASSETS / "anymal" / "urdf" / "dynaarm.urdf"
DYNAARM_USD = ANYMAL_USD_DIR / "dynaarm.usd"
ASSEMBLY_USDA = ANYMAL_USD_DIR / "anymal_d_dynaarm_robotiq.usda"

S3_BASE = "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
S3_ANYMAL_PREFIX = "Assets/Isaac/6.0/Isaac/IsaacLab/Robots/ANYbotics/ANYmal-D"
S3_ROBOTIQ_PREFIX = "Assets/Isaac/6.0/Isaac/Robots/Robotiq/2F-140"

# Local Nucleus asset cache (populated by Isaac Sim runs).
NUCLEUS_CACHE = Path("/tmp/Assets/Isaac/6.0")

# arm_mount -> dynaarm_base is a fixed identity joint; camera mounts are
# visual-only 1 cm boxes and are skipped. Everything else is authored.
SKIP_LINKS = {
    "dynaarm_elbow_camera_mount",
    "dynaarm_wrist_1_camera_mount",
}
SKIP_JOINTS = {
    "dynaarm_elbow_camera",
    "dynaarm_wrist_flexion_camera",
}

# Mount of the DynaArm on the ANYmal trunk. anymal.py documents 180 deg yaw so
# the arm faces robot +X (forward, toward the bench valve); the committed
# identity value faces it aft. xyzw (0, 0, 1, 0) = 180 deg about Z.
DYNAARM_MOUNT_POS = (0.0, 0.0, 0.12)
DYNAARM_MOUNT_ROT_XYZW = (0.0, 0.0, 1.0, 0.0)

# Gripper revolute joints kept mobile for Newton; everything else in the
# gripper is frozen at file pose. The 2F-140 is a closed 4-bar loop per side;
# Newton/MJWarp goes unstable on the loop while PhysX tolerates it. Keeping
# the two outer-knuckle swings gives a symmetric 2-DOF jaw (the Isaac Sim
# mapping already drives this pair: finger_joint plus its mimic).
MOBILE_GRIPPER_JOINTS = frozenset({"finger_joint", "right_outer_knuckle_joint"})


def _s3_url(key: str) -> str:
    return f"{S3_BASE}/{key}"


def _fetch(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as src, open(dst, "wb") as fh:
        shutil.copyfileobj(src, fh)


def _s3_list(prefix: str) -> list[str]:
    """List S3 keys under ``prefix`` (public bucket listing)."""
    import xml.etree.ElementTree as XET

    url = f"{S3_BASE}/?list-type=2&prefix={prefix}/"
    with urllib.request.urlopen(url, timeout=60) as resp:
        tree = XET.parse(resp)
    keys = []
    for node in tree.getroot().iter("{http://s3.amazonaws.com/doc/2006-03-01/}Key"):
        key = node.text or ""
        if ".thumbs/" in key:
            continue
        keys.append(key)
    return keys


def fetch_nucleus() -> None:
    """Mirror the ANYmal-D + Robotiq trees into ``assets/anymal/usd/nucleus/``."""
    specs = []
    for key in _s3_list(S3_ANYMAL_PREFIX):
        rel = Path(key).relative_to(Path(S3_ANYMAL_PREFIX).parent)
        specs.append((key, NUCLEUS_DIR / rel))
    for key in _s3_list(S3_ROBOTIQ_PREFIX):
        rel = Path(key).relative_to(Path(S3_ROBOTIQ_PREFIX).parent)
        specs.append((key, NUCLEUS_DIR / rel))
    print(f"[FETCH] {len(specs)} nucleus files")
    for key, dst in specs:
        # Seed ANYmal-D from the local Nucleus cache when present.
        cache_src = NUCLEUS_CACHE / key.split("Isaac/6.0/", 1)[1]
        if cache_src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cache_src, dst)
            print(f"  [cache] {key}")
            continue
        if dst.is_file():
            print(f"  [keep] {key}")
            continue
        _fetch(_s3_url(key), dst)
        print(f"  [s3] {key}")


# ---------------------------------------------------------------------------
# Small math helpers (numpy, no scipy).
# ---------------------------------------------------------------------------


def rpy_to_quat_xyzw(rpy: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """URDF rpy (radians) to xyzw quaternion (UUC convention).

    URDF applies roll, then pitch, then yaw extrinsically, i.e. R = Rz*Rz... so
    q = qz ⊗ qy ⊗ qx. Verified numerically against UUC's Gf.Rotation chain.
    Newton-side note: Newton core orders quaternions (x, y, z, w) in state
    vectors; IsaacLab converts at the boundary. This script only authors pxr
    quats (Gf.Quatf is real-first), so no Newton reordering applies here.
    """
    cx, sx = math.cos(rpy[0] / 2), math.sin(rpy[0] / 2)
    cy, sy = math.cos(rpy[1] / 2), math.sin(rpy[1] / 2)
    cz, sz = math.cos(rpy[2] / 2), math.sin(rpy[2] / 2)
    qx = (sx, 0.0, 0.0, cx)
    qy = (0.0, sy, 0.0, cy)
    qz = (0.0, 0.0, sz, cz)
    return quat_mul(qz, quat_mul(qy, qx))


def quat_mul(a: tuple, b: tuple) -> tuple[float, float, float, float]:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def quat_inv(q: tuple) -> tuple[float, float, float, float]:
    return (-q[0], -q[1], -q[2], q[3])


def quat_to_mat(q: tuple) -> np.ndarray:
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def mat_to_quat_xyzw(m: np.ndarray) -> tuple[float, float, float, float]:
    """Shepperd's method, xyzw output."""
    t = float(np.trace(m))
    if t > 0.0:
        s = 0.5 / math.sqrt(t + 1.0)
        return (
            (m[2, 1] - m[1, 2]) * s,
            (m[0, 2] - m[2, 0]) * s,
            (m[1, 0] - m[0, 1]) * s,
            0.25 / s,
        )
    if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        return (0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s)
    if m[1, 1] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        return ((m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s)
    s = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
    return ((m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s)


def pose_to_mat(pos: tuple, quat_xyzw: tuple) -> np.ndarray:
    """Pose to 4x4 using pxr row-vector storage (translation in row 3)."""
    m = np.eye(4)
    m[:3, :3] = quat_to_mat(quat_xyzw)
    m[3, :3] = np.asarray(pos, dtype=float)
    return m


def mat_to_pose(m: np.ndarray) -> tuple[tuple, tuple]:
    """Extract (pos, xyzw quat) from a pxr row-vector 4x4."""
    return (tuple(float(v) for v in m[3, :3]), mat_to_quat_xyzw(m[:3, :3]))


# ---------------------------------------------------------------------------
# URDF parsing.
# ---------------------------------------------------------------------------


def _vec(text: str | None, n: int, default: float = 0.0) -> tuple:
    if not text:
        return tuple([default] * n)
    return tuple(float(v) for v in text.split())


def parse_urdf(urdf_path: Path) -> tuple[dict, dict]:
    """Return (links, joints) keyed by name with resolved mesh paths."""
    root = ET.parse(str(urdf_path)).getroot()
    links: dict[str, dict] = {}
    for link in root.iter("link"):
        name = link.get("name")
        if name in SKIP_LINKS:
            continue
        inertial = link.find("inertial")
        mass, com, inertia = 1e-3, (0.0, 0.0, 0.0), (1e-6, 1e-6, 1e-6)
        if inertial is not None:
            mass = float(inertial.find("mass").get("value", "0.001"))
            origin = inertial.find("origin")
            if origin is not None:
                com = _vec(origin.get("xyz"), 3)
            inertia_el = inertial.find("inertia")
            if inertia_el is not None:
                inertia = (
                    float(inertia_el.get("ixx", "1e-6")),
                    float(inertia_el.get("iyy", "1e-6")),
                    float(inertia_el.get("izz", "1e-6")),
                )
        collisions, visuals = [], []
        for col in link.iter("collision"):
            origin = col.find("origin")
            pos = _vec(origin.get("xyz") if origin is not None else None, 3)
            rpy = _vec(origin.get("rpy") if origin is not None else None, 3)
            geom = col.find("geometry")
            if geom.find("cylinder") is not None:
                cyl = geom.find("cylinder")
                collisions.append(
                    {"kind": "cylinder", "radius": float(cyl.get("radius")), "height": float(cyl.get("length")), "pos": pos, "rpy": rpy}
                )
            elif geom.find("box") is not None:
                collisions.append(
                    {"kind": "box", "size": _vec(geom.find("box").get("size"), 3), "pos": pos, "rpy": rpy}
                )
            elif geom.find("sphere") is not None:
                collisions.append(
                    {"kind": "sphere", "radius": float(geom.find("sphere").get("radius")), "pos": pos, "rpy": rpy}
                )
        for vis in link.iter("visual"):
            origin = vis.find("origin")
            pos = _vec(origin.get("xyz") if origin is not None else None, 3)
            rpy = _vec(origin.get("rpy") if origin is not None else None, 3)
            mesh = vis.find("geometry/mesh")
            if mesh is not None:
                visuals.append({"mesh": mesh.get("filename"), "pos": pos, "rpy": rpy})
        links[name] = {"mass": mass, "com": com, "inertia": inertia, "collisions": collisions, "visuals": visuals}
    joints: dict[str, dict] = {}
    for joint in root.iter("joint"):
        name = joint.get("name")
        if name in SKIP_JOINTS:
            continue
        origin = joint.find("origin")
        pos = _vec(origin.get("xyz") if origin is not None else None, 3)
        rpy = _vec(origin.get("rpy") if origin is not None else None, 3)
        axis_el = joint.find("axis")
        axis = _vec(axis_el.get("xyz") if axis_el is not None else None, 3, 0.0) or (0.0, 0.0, 1.0)
        limit_el = joint.find("limit")
        lower = float(limit_el.get("lower", "0")) if limit_el is not None else 0.0
        upper = float(limit_el.get("upper", "0")) if limit_el is not None else 0.0
        joints[name] = {
            "type": joint.get("type"),
            "parent": joint.find("parent").get("link"),
            "child": joint.find("child").get("link"),
            "pos": pos,
            "rpy": rpy,
            "axis": axis,
            "lower": lower,
            "upper": upper,
        }
    return links, joints


def resolve_mesh_path(filename: str) -> Path:
    """Resolve a URDF mesh filename to the duatic submodule file."""
    if "://" in filename:
        filename = filename.split("://", 1)[1].split("/", 1)[1]
    candidate = REPO_ROOT / "dependencies" / "duatic_dynaarm" / "duatic_dynaarm_description" / filename
    if candidate.is_file():
        return candidate
    return Path(filename)


# ---------------------------------------------------------------------------
# DynaArm USD authoring (pxr + usdex, kitless).
# ---------------------------------------------------------------------------


def _set_link_transform(prim, pos: tuple, quat_xyzw: tuple) -> None:
    from pxr import Gf, UsdGeom

    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    top = xform.AddTranslateOp()
    topo = xform.AddOrientOp()
    top.Set(Gf.Vec3d(*pos))
    x, y, z, w = quat_xyzw
    topo.Set(Gf.Quatf(float(w), Gf.Vec3f(float(x), float(y), float(z))))


def build_dynaarm_usd() -> Path:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, Vt

    links, joints = parse_urdf(DYNAARM_URDF)
    children: dict[str, list[str]] = {}
    for name, joint in joints.items():
        children.setdefault(joint["parent"], []).append(name)

    # Find the root link (never a child).
    child_links = {j["child"] for j in joints.values()}
    roots = [name for name in links if name not in child_links]
    if len(roots) != 1:
        raise RuntimeError(f"Expected 1 URDF root link, got {roots}")
    root_link = roots[0]

    if DYNAARM_USD.exists():
        DYNAARM_USD.unlink()
    stage = Usd.Stage.CreateNew(str(DYNAARM_USD))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    robot = stage.DefinePrim("/dynaarm", "Xform")

    def _mesh_from_dae(dae_path: Path) -> list[tuple[np.ndarray, np.ndarray]]:
        """Best-effort DAE triangles as (points, faces) in meters.

        Vertices are transformed by the accumulated scene-node matrices: the
        duatic DAEs store millimeters (0.001 node scale) and ignoring the
        nodes bakes 1000x visuals. Cameras/lights carry no geometry and are
        skipped implicitly.
        """
        import collada

        doc = collada.Collada(str(dae_path), ignore=[collada.DaeError])
        unit = doc.assetInfo.unitmeter if doc.assetInfo.unitmeter is not None else 1.0
        out: list[tuple[np.ndarray, np.ndarray]] = []

        def _local(node) -> np.ndarray:
            m = np.eye(4)
            for t in node.transforms:
                cname = type(t).__name__
                if cname == "MatrixTransform":
                    m = m @ np.asarray(t.matrix, dtype=float)
                elif cname == "TranslateTransform":
                    tmat = np.eye(4)
                    tmat[:3, 3] = [t.x, t.y, t.z]
                    m = m @ tmat
                elif cname == "RotateTransform":
                    axis = np.asarray([t.x, t.y, t.z], dtype=float)
                    axis = axis / (np.linalg.norm(axis) or 1.0)
                    a = math.radians(t.angle)
                    kx, ky, kz = axis
                    kmat = np.array([[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]])
                    rot = np.eye(4)
                    rot[:3, :3] = (
                        np.eye(3) * math.cos(a) + math.sin(a) * kmat + (1 - math.cos(a)) * np.outer(axis, axis)
                    )
                    m = m @ rot
                elif cname == "ScaleTransform":
                    m = m @ np.diag([t.x, t.y, t.z, 1.0])
            return m

        def _visit(nodes, parent: np.ndarray) -> None:
            for node in nodes:
                mat = parent @ _local(node)
                for child in node.children:
                    if type(child).__name__ == "GeometryNode":
                        for prim in child.geometry.primitives:
                            if type(prim).__name__ not in ("TriangleSet", "Triangles", "Polylist", "Polygons"):
                                continue
                            pts = np.asarray(prim.vertex, dtype=np.float64)
                            pts4 = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
                            out.append(
                                (
                                    (mat @ pts4.T).T[:, :3],
                                    np.asarray(prim.vertex_index, dtype=np.int64).reshape(-1, 3),
                                )
                            )
                    elif hasattr(child, "children"):
                        _visit([child], mat)

        for scene in doc.scenes:
            _visit(scene.nodes, np.diag([unit, unit, unit, 1.0]))
        return out

    link_prims: dict[str, object] = {}

    def _author_link(name: str, parent_path: str, pos: tuple, quat_xyzw: tuple) -> None:
        spec = links[name]
        path = f"{parent_path}/{name}"
        prim = stage.DefinePrim(path, "Xform")
        _set_link_transform(prim, pos, quat_xyzw)
        link_prims[name] = prim
        UsdPhysics.RigidBodyAPI.Apply(prim)
        mass = UsdPhysics.MassAPI.Apply(prim)
        mass.CreateMassAttr().Set(float(spec["mass"]))
        mass.CreateCenterOfMassAttr().Set(Gf.Vec3f(*spec["com"]))
        mass.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*spec["inertia"]))
        for i, col in enumerate(spec["collisions"]):
            cquat = rpy_to_quat_xyzw(col["rpy"])
            if col["kind"] == "cylinder":
                cprim = UsdGeom.Cylinder.Define(stage, f"{path}/collisions/cyl_{i}").GetPrim()
                cprim.GetAttribute("radius").Set(float(col["radius"]))
                cprim.GetAttribute("height").Set(float(col["height"]))
                cprim.GetAttribute("axis").Set("Z")
            elif col["kind"] == "box":
                cprim = UsdGeom.Cube.Define(stage, f"{path}/collisions/box_{i}").GetPrim()
                cprim.GetAttribute("size").Set(1.0)
                # Unit cube scaled to size via an extra scale op below.
                _set_link_transform(cprim, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
                xform = UsdGeom.Xformable(cprim)
                xform.ClearXformOpOrder()
                xform.AddTranslateOp().Set(Gf.Vec3d(*col["pos"]))
                xform.AddOrientOp().Set(
                    Gf.Quatf(cquat[3], Gf.Vec3f(cquat[0], cquat[1], cquat[2]))
                )
                xform.AddScaleOp().Set(Gf.Vec3f(*col["size"]))
                UsdPhysics.CollisionAPI.Apply(cprim)
                continue
            elif col["kind"] == "sphere":
                cprim = UsdGeom.Sphere.Define(stage, f"{path}/collisions/sphere_{i}").GetPrim()
                cprim.GetAttribute("radius").Set(float(col["radius"]))
            else:
                continue
            _set_link_transform(cprim, col["pos"], cquat)
            UsdPhysics.CollisionAPI.Apply(cprim)
        for i, vis in enumerate(spec["visuals"]):
            try:
                parts = _mesh_from_dae(resolve_mesh_path(vis["mesh"]))
            except Exception as err:  # noqa: BLE001 - visuals must not break physics
                print(f"[DYNAARM] visual skipped ({vis['mesh']}: {err})")
                continue
            vquat = rpy_to_quat_xyzw(vis["rpy"])
            for j, (pts, faces) in enumerate(parts):
                mprim = UsdGeom.Mesh.Define(stage, f"{path}/visuals/vis_{i}_{j}").GetPrim()
                mprim.CreateAttribute("points", Sdf.ValueTypeNames.Point3fArray).Set(
                    Vt.Vec3fArray.FromNumpy(pts.astype(np.float32))
                )
                counts = [3] * len(faces)
                mprim.CreateAttribute("faceVertexCounts", Sdf.ValueTypeNames.IntArray).Set(counts)
                mprim.CreateAttribute("faceVertexIndices", Sdf.ValueTypeNames.IntArray).Set(
                    faces.astype(np.int32).ravel().tolist()
                )
                _set_link_transform(mprim, vis["pos"], vquat)
        for joint_name in children.get(name, []):
            joint = joints[joint_name]
            jquat = rpy_to_quat_xyzw(joint["rpy"])
            if joint["type"] == "fixed":
                # Newton needs an explicit joint between rigid bodies to keep
                # the articulation connected (it does not infer fixed joints
                # from hierarchy nesting like UUC-merged assets do).
                fjprim = stage.DefinePrim(f"{path}/{joint_name}", "PhysicsFixedJoint")
                fjoint = UsdPhysics.Joint(fjprim)
                fjoint.CreateBody0Rel().SetTargets([Sdf.Path(path)])
                fjoint.CreateBody1Rel().SetTargets([Sdf.Path(f"{path}/{joint['child']}")])
                fjoint.CreateLocalPos0Attr().Set(Gf.Vec3f(*joint["pos"]))
                x, y, z, w = jquat
                fjoint.CreateLocalRot0Attr().Set(Gf.Quatf(float(w), Gf.Vec3f(float(x), float(y), float(z))))
                fjoint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
                fjoint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
                _author_link(joint["child"], path, joint["pos"], jquat)
                continue
            # Revolute joint between the link frames. URDF child-link frames sit
            # at the joint origin, so the child local frame is identity.
            jprim = stage.DefinePrim(f"{path}/{joint_name}", "PhysicsRevoluteJoint")
            joint_api = UsdPhysics.Joint(jprim)
            joint_api.CreateBody0Rel().SetTargets([Sdf.Path(path)])
            joint_api.CreateBody1Rel().SetTargets([Sdf.Path(f"{path}/{joint['child']}")])
            joint_api.CreateLocalPos0Attr().Set(Gf.Vec3f(*joint["pos"]))
            x, y, z, w = jquat
            joint_api.CreateLocalRot0Attr().Set(Gf.Quatf(float(w), Gf.Vec3f(float(x), float(y), float(z))))
            joint_api.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
            joint_api.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
            axis = UsdPhysics.RevoluteJoint(jprim)
            ax = joint["axis"]
            axis_token = "XYZ"[int(np.argmax(np.abs(ax)))] if any(ax) else "Z"
            if float(ax[{"X": 0, "Y": 1, "Z": 2}[axis_token]]) < 0:
                axis_token = f"-{axis_token}"
            axis.CreateAxisAttr().Set(axis_token)
            axis.CreateLowerLimitAttr().Set(math.degrees(joint["lower"]))
            axis.CreateUpperLimitAttr().Set(math.degrees(joint["upper"]))
            _author_link(joint["child"], path, joint["pos"], jquat)

    _author_link(root_link, "/dynaarm", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    stage.GetRootLayer().Save()
    print(f"[DYNAARM] wrote {DYNAARM_USD} ({DYNAARM_USD.stat().st_size // 1024} KiB)")
    return DYNAARM_USD


# ---------------------------------------------------------------------------
# Assembly bake (references + welds + strips, kitless).
# ---------------------------------------------------------------------------


def _world_matrix(stage, prim_path: str):
    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    return UsdGeom.XformCache().GetLocalToWorldTransform(prim)


def _find_named_prim(root, name: str):
    if not root.IsValid():
        return None
    if root.GetName() == name:
        return root
    for child in root.GetChildren():
        found = _find_named_prim(child, name)
        if found is not None and found.IsValid():
            return found
    return None


def bake_assembly() -> Path:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    anymal_usd = NUCLEUS_DIR / "ANYmal-D" / "anymal_d.usd"
    gripper_usd = NUCLEUS_DIR / "2F-140" / "Robotiq_2F_140_physics_edit.usd"
    for path in (anymal_usd, gripper_usd, DYNAARM_USD):
        if not path.is_file():
            raise RuntimeError(f"Missing input for assembly: {path}")

    if ASSEMBLY_USDA.exists():
        ASSEMBLY_USDA.unlink()
    stage = Usd.Stage.CreateNew(str(ASSEMBLY_USDA))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    baked = stage.DefinePrim("/anymal_dynaarm_robotiq", "Xform")
    stage.SetDefaultPrim(baked)

    anymal = stage.DefinePrim("/anymal_dynaarm_robotiq/anymal", "Xform")
    anymal.GetReferences().AddReference(
        os.path.relpath(anymal_usd, ASSEMBLY_USDA.parent), "/anymal"
    )

    arm = stage.DefinePrim("/anymal_dynaarm_robotiq/dynaarm", "Xform")
    arm.GetReferences().AddReference(os.path.relpath(DYNAARM_USD, ASSEMBLY_USDA.parent), "/dynaarm")
    _set_link_transform(arm, DYNAARM_MOUNT_POS, DYNAARM_MOUNT_ROT_XYZW)

    grip = stage.DefinePrim("/anymal_dynaarm_robotiq/robotiq_2f_140", "Xform")
    grip.GetReferences().AddReference(os.path.relpath(gripper_usd, ASSEMBLY_USDA.parent), "/Robotiq_2F_140")
    stage.GetRootLayer().Save()

    # Reopen with payloads so world matrices resolve, then weld + strip with
    # the overlay as edit target.
    stage = Usd.Stage.Open(str(ASSEMBLY_USDA))
    stage.SetEditTarget(stage.GetRootLayer())
    baked_path = "/anymal_dynaarm_robotiq"

    base = stage.GetPrimAtPath(f"{baked_path}/anymal/base")
    arm_mount = _find_named_prim(stage.GetPrimAtPath(f"{baked_path}/dynaarm"), "arm_mount")
    flange = _find_named_prim(stage.GetPrimAtPath(f"{baked_path}/dynaarm"), "dynaarm_flange")
    grip_root = stage.GetPrimAtPath(f"{baked_path}/robotiq_2f_140")
    palm = _find_named_prim(grip_root, "robotiq_base_link")
    for label, prim in (("base", base), ("arm_mount", arm_mount), ("flange", flange), ("palm", palm)):
        if prim is None or not prim.IsValid():
            raise RuntimeError(f"Assembly bake: '{label}' prim not found")

    def _weld(joint_path: str, body0_path: str, body1_path: str) -> None:
        w0 = _world_matrix(stage, body0_path)
        w1 = _world_matrix(stage, body1_path)
        m0 = np.array([[w0[i][j] for j in range(4)] for i in range(4)], dtype=float)
        m1 = np.array([[w1[i][j] for j in range(4)] for i in range(4)], dtype=float)
        rel = np.linalg.inv(m0) @ m1
        (pos, quat) = mat_to_pose(rel)
        jprim = UsdPhysics.FixedJoint.Define(stage, joint_path).GetPrim()
        jprim.CreateRelationship("physics:body0").SetTargets([Sdf.Path(body0_path)])
        jprim.CreateRelationship("physics:body1").SetTargets([Sdf.Path(body1_path)])
        jprim.CreateAttribute("physics:localPos0", Sdf.ValueTypeNames.Point3f).Set(Gf.Vec3f(*pos))
        x, y, z, w = quat
        jprim.CreateAttribute("physics:localRot0", Sdf.ValueTypeNames.Quatf).Set(
            Gf.Quatf(float(w), Gf.Vec3f(float(x), float(y), float(z)))
        )
        jprim.CreateAttribute("physics:localPos1", Sdf.ValueTypeNames.Point3f).Set(Gf.Vec3f(0, 0, 0))
        jprim.CreateAttribute("physics:localRot1", Sdf.ValueTypeNames.Quatf).Set(
            Gf.Quatf(1.0, Gf.Vec3f(0, 0, 0))
        )

    # Weld the arm mount to the trunk (mirrors _attach_dynaarm).
    _weld(
        f"{baked_path}/base_to_dynaarm",
        base.GetPath().pathString,
        arm_mount.GetPath().pathString,
    )

    # Put the gripper root so the palm sits on the flange, preserving the
    # palm-from-root offset (mirrors _attach_isaac_robotiq).
    m_palm = np.array(
        [[_world_matrix(stage, palm.GetPath().pathString)[i][j] for j in range(4)] for i in range(4)],
        dtype=float,
    )
    m_root = np.array(
        [[_world_matrix(stage, grip_root.GetPath().pathString)[i][j] for j in range(4)] for i in range(4)],
        dtype=float,
    )
    m_flange = np.array(
        [[_world_matrix(stage, flange.GetPath().pathString)[i][j] for j in range(4)] for i in range(4)],
        dtype=float,
    )
    m_palm_from_root = m_palm @ np.linalg.inv(m_root)
    m_new_root = np.linalg.inv(m_palm_from_root) @ m_flange
    (new_pos, new_quat) = mat_to_pose(m_new_root)
    _set_link_transform(grip_root, new_pos, new_quat)

    # Weld palm to flange, then strip nesting hazards. Note: the referenced
    # gripper root's opinions (including its articulation root) compose onto
    # the mount prim itself, so strip the mount, not an inner path.
    _weld(
        f"{baked_path}/flange_to_robotiq",
        flange.GetPath().pathString,
        palm.GetPath().pathString,
    )
    grip_root.RemoveAPI(UsdPhysics.ArticulationRootAPI)
    world_fixed = stage.GetPrimAtPath(f"{baked_path}/robotiq_2f_140/FixedJoint")
    if world_fixed.IsValid():
        stage.RemovePrim(world_fixed.GetPath())
    for prim in list(stage.Traverse()):
        if prim.IsValid() and prim.GetTypeName() == "PhysicsScene":
            stage.RemovePrim(prim.GetPath())
    # De-instance gripper payload holders (mirrors UUC make_instanceable=False).
    # Nucleus authors instanceable=true on the finger part holders, which hides
    # their meshes from traversal and from the Newton viewer.
    n_deinstanced = 0
    for prim in stage.Traverse():
        if (
            prim.IsValid()
            and prim.IsInstanceable()
            and str(prim.GetPath()).startswith(f"{baked_path}/robotiq_2f_140")
        ):
            prim.SetInstanceable(False)
            n_deinstanced += 1
    print(f"[BAKE] de-instanced {n_deinstanced} gripper prims")
    _freeze_gripper_loops(stage, grip_root)
    # Anchor the trunk like Spot's UUC `root_joint`: weld the root link to the
    # asset top Xform (NOT to world) with identity frames. This moves with the
    # asset through IsaacLab placement/cloning, exactly like the Spot bench.
    root_joint = UsdPhysics.FixedJoint.Define(stage, f"{baked_path}/root_joint").GetPrim()
    root_joint.CreateRelationship("physics:body0").SetTargets([Sdf.Path(baked_path)])
    root_joint.CreateRelationship("physics:body1").SetTargets([Sdf.Path(base.GetPath().pathString)])
    for attr, value in (
        ("physics:localPos0", Gf.Vec3f(0.0, 0.0, 0.0)),
        ("physics:localRot0", Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0))),
        ("physics:localPos1", Gf.Vec3f(0.0, 0.0, 0.0)),
        ("physics:localRot1", Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0))),
    ):
        root_joint.CreateAttribute(attr, Sdf.ValueTypeNames.Point3f if "Pos" in attr else Sdf.ValueTypeNames.Quatf).Set(
            value
        )
    print("[BAKE] added root_joint (trunk welded to asset top, Spot pattern)")
    stage.GetRootLayer().Save()
    print(f"[BAKE] wrote {ASSEMBLY_USDA} ({ASSEMBLY_USDA.stat().st_size // 1024} KiB)")
    return ASSEMBLY_USDA


def _freeze_gripper_loops(stage, grip_root) -> None:
    """Replace gripper loop revolutes with fixed joints at file pose.

    Copies each joint's bodies + local frames onto a PhysicsFixedJoint, then
    deletes the revolute. Only :data:`MOBILE_GRIPPER_JOINTS` stay articulated.
    """
    from pxr import Sdf, UsdPhysics

    targets = [
        p
        for p in stage.Traverse()
        if p.GetTypeName() == "PhysicsRevoluteJoint"
        and str(p.GetPath()).startswith(grip_root.GetPath().pathString)
        and p.GetName() not in MOBILE_GRIPPER_JOINTS
    ]
    for prim in targets:
        joint = UsdPhysics.Joint(prim)
        body0 = [t.pathString for t in joint.CreateBody0Rel().GetTargets()]
        body1 = [t.pathString for t in joint.CreateBody1Rel().GetTargets()]
        frames = {}
        for attr in ("physics:localPos0", "physics:localRot0", "physics:localPos1", "physics:localRot1"):
            owned = prim.GetAttribute(attr)
            frames[attr] = owned.Get() if owned.HasValue() else None
        path, parent = prim.GetPath(), prim.GetParent()
        stage.RemovePrim(path)
        frozen = UsdPhysics.FixedJoint.Define(stage, path).GetPrim()
        frozen.CreateRelationship("physics:body0").SetTargets([Sdf.Path(b) for b in body0])
        frozen.CreateRelationship("physics:body1").SetTargets([Sdf.Path(b) for b in body1])
        for attr, value in frames.items():
            if value is not None:
                frozen.CreateAttribute(attr, prim.GetAttribute(attr).GetTypeName()).Set(value)
        print(f"[BAKE] froze gripper joint {path.name}")


def report_tcp_offset() -> None:
    """Print the measured flange→TCP offset for ``assets/anymal/bench.py``.

    Composes the baked flange→palm weld with the ANYMAL_EE palm→TCP profile
    using pxr row-vector convention (relative = Mp @ inv(Mf)).
    """
    from pxr import Usd, UsdGeom

    from isaaclab_hiveboard.assets.end_effector import ANYMAL_EE

    stage = Usd.Stage.Open(str(ASSEMBLY_USDA))
    baked = stage.GetPrimAtPath("/anymal_dynaarm_robotiq")
    cache = UsdGeom.XformCache()

    def _mat(p) -> np.ndarray:
        w = cache.GetLocalToWorldTransform(p)
        return np.array([[w[i][j] for j in range(4)] for i in range(4)], dtype=float)

    mf = _mat(_find_named_prim(baked, "dynaarm_flange"))
    mp = _mat(_find_named_prim(baked, "robotiq_base_link"))
    rel = mp @ np.linalg.inv(mf)
    q_rel = mat_to_quat_xyzw(rel[:3, :3])
    off_p = np.asarray(ANYMAL_EE.tcp_offset.pos, dtype=float)
    off_q = tuple(ANYMAL_EE.tcp_offset.rot)
    tcp_p = rel[:3, :3].T @ off_p + rel[3, :3]
    tcp_q = quat_mul(q_rel, off_q)
    print(f"[TCP] FLANGE_TO_TCP_POS = ({', '.join(f'{v:.6f}' for v in tcp_p)})")
    print(f"[TCP] FLANGE_TO_TCP_QUAT_XYZW = ({', '.join(f'{v:.6f}' for v in tcp_q)})")


def verify() -> list[str]:
    """Open the baked stage and check the assembly. Empty list = ok."""
    from pxr import Usd, UsdPhysics

    problems = []
    for path, label in ((DYNAARM_USD, "dynaarm"), (ASSEMBLY_USDA, "assembly")):
        if not path.exists():
            problems.append(f"missing {label}: {path}")
    if problems:
        return problems
    stage = Usd.Stage.Open(str(ASSEMBLY_USDA))
    if not stage:
        return [f"pxr could not open {ASSEMBLY_USDA}"]
    baked = "/anymal_dynaarm_robotiq"
    for joint in (
        "dynaarm_shoulder_rotation",
        "dynaarm_shoulder_flexion",
        "dynaarm_elbow_flexion",
        "dynaarm_forearm_rotation",
        "dynaarm_wrist_flexion",
        "dynaarm_wrist_rotation",
        "finger_joint",
        "right_outer_knuckle_joint",
    ):
        found = [p for p in stage.Traverse() if p.GetName() == joint]
        if not found:
            problems.append(f"baked stage missing joint '{joint}'")
    for body in ("dynaarm_flange", "robotiq_base_link"):
        prim = _find_named_prim(stage.GetPrimAtPath(baked), body)
        if prim is None or not prim.IsValid():
            problems.append(f"baked stage missing body '{body}'")
    inner_roots = [
        p.GetPath()
        for p in stage.Traverse()
        if p.HasAPI(UsdPhysics.ArticulationRootAPI) and p.GetPath() != f"{baked}/anymal/base"
    ]
    for path in inner_roots:
        problems.append(f"unexpected nested articulation root: {path}")
    if stage.GetPrimAtPath(f"{baked}/robotiq_2f_140/FixedJoint").IsValid():
        problems.append("gripper world FixedJoint not stripped")
    return problems


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verify-only", action="store_true", help="Do not write anything; check baked USD with pxr.")
    parser.add_argument("--skip-download", action="store_true", help="Reuse vendored nucleus files; skip S3 fetch.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.verify_only:
        problems = verify()
        if problems:
            print("[FAIL]")
            for problem in problems:
                print(f"  - {problem}")
            return 1
        print("[OK] ANYmal baked USD resolves")
        return 0
    if not args.skip_download:
        fetch_nucleus()
    build_dynaarm_usd()
    bake_assembly()
    report_tcp_offset()
    problems = verify()
    for problem in problems:
        print(f"[WARN] {problem}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
