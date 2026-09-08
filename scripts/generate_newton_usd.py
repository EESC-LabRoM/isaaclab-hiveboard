# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regenerate kitless Newton USD assets via urdf-usd-converter (UUC).

The Newton tasks load preconverted USD (gitignored ``**/*.usd*``). This script
is the record of how those files are produced.

Spot: UUC conversion of ``spot_with_arm.urdf`` into ``spot/usd/uuc/``.
Valve: UUC conversion into ``ball_valve/usd/uuc/`` plus a CoACD overlay
(``Ball_Valve_uuc_newton.usda``). A single UUC convexHull wraps the lever in
~3x phantom volume; Newton hardcodes CoACD threshold=0.5, so parts are baked
here (``coacd`` + ``trimesh``, generation-time only).

UUC needs a Python that can ``import urdf_usd_converter`` (isolated venv;
the project venv does not ship it). Overlay rewrite is kitless.

Usage::

    uv run python scripts/generate_newton_usd.py --verify-only
    uv run python scripts/generate_newton_usd.py --skip-conversion
    uv run python scripts/generate_newton_usd.py --uuc-python /path/to/uuc-venv/bin/python
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXT_ASSETS = REPO_ROOT / "source/isaaclab_hiveboard/isaaclab_hiveboard/assets"

MESH_COLLISION_APIS = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]

SPOT_URDF = EXT_ASSETS / "spot/spot_with_arm.urdf"
SPOT_UUC_DIR = EXT_ASSETS / "spot/usd/uuc"
VALVE_URDF = EXT_ASSETS / "hiveboard/ball_valve/Ball_Valve.urdf"
VALVE_USD_DIR = EXT_ASSETS / "hiveboard/ball_valve/usd"
VALVE_UUC_DIR = VALVE_USD_DIR / "uuc"
VALVE_OVERLAY = VALVE_USD_DIR / "Ball_Valve_uuc_newton.usda"

# (link path under /Ball_Valve, collision mesh name, coacd threshold)
VALVE_TARGETS: tuple[tuple[str, str, float], ...] = (
    ("Geometry/valvula_esfera", "Mesh_1", 0.1),
    ("Geometry/valvula_esfera/alavanca_pivot", "Mesh_1_1", 0.05),
)

# OBJs whose unused vertices make usdex definePolyMesh reject the mesh.
SPOT_OBJ_CLEAN = (
    EXT_ASSETS / "spot/meshes/arm/visual/arm_link_sh0.obj",
    EXT_ASSETS / "spot/meshes/arm/visual/arm_link_wr0.obj",
    EXT_ASSETS / "spot/meshes/base/collision/body_collision.obj",
    EXT_ASSETS / "spot/meshes/base/collision/upper_leg_collision.obj",
    EXT_ASSETS / "spot/meshes/base/collision/lower_leg_collision.obj",
)


def urdf_link_collision_meshes(urdf: Path) -> dict[str, list[str]]:
    """Map URDF link name -> authored collision mesh filenames."""
    import xml.etree.ElementTree as ET

    out: dict[str, list[str]] = {}
    for link in ET.parse(str(urdf)).getroot().iter("link"):
        files = [
            mesh.get("filename")
            for col in link.iter("collision")
            for mesh in col.findall("geometry/mesh")
            if mesh.get("filename")
        ]
        if files:
            out[link.get("name")] = files
    return out


def resolve_mesh_file(filename: str, urdf: Path) -> Path:
    """Resolve a URDF mesh filename (relative / package:// / absolute)."""
    if "://" in filename:
        filename = filename.split("://", 1)[1].split("/", 1)[1]
    path = Path(filename)
    if path.is_absolute():
        return path
    return urdf.parent / path


def decompose_mesh(mesh_file: Path, threshold: float) -> list[tuple[object, object]]:
    """Split a mesh into convex parts with CoACD. Returns (verts, faces) pairs."""
    try:
        import coacd  # noqa: PLC0415
        import trimesh  # noqa: PLC0415
    except ImportError as err:
        raise RuntimeError(
            "Baked convex decomposition needs the `coacd` and `trimesh` packages "
            "(generation-time only, e.g. `uv pip install coacd trimesh`)."
        ) from err
    import numpy as np  # noqa: PLC0415

    loaded = trimesh.load(str(mesh_file), process=True, force="mesh")
    verts = np.asarray(loaded.vertices, dtype=np.float64)
    faces = np.asarray(loaded.faces, dtype=np.int32)
    return coacd.run_coacd(coacd.Mesh(verts, faces), threshold=threshold)


def _fmt_floats(values: object) -> str:
    import numpy as np  # noqa: PLC0415

    flat = np.asarray(values, dtype=np.float64).ravel()
    return ", ".join(f"{v:.9g}" for v in flat)


def _fmt_points(verts: object) -> str:
    import numpy as np  # noqa: PLC0415

    return ", ".join(f"({_fmt_floats(v)})" for v in np.asarray(verts, dtype=np.float64))


def _fmt_ints(values: object) -> str:
    import numpy as np  # noqa: PLC0415

    return ", ".join(str(int(v)) for v in np.asarray(values).ravel())


def _parse_obj_index(token: str, count: int) -> int:
    index = int(token)
    if index < 0:
        index = count + index + 1
    return index


def strip_unused_obj_topology(path: Path) -> bool:
    """Drop face-unreferenced v/vt/vn and OBJ polylines (`l`).

    usdex ``definePolyMesh`` rejects meshes with points no face references.
    Some Spot OBJs keep CAD leftover verts only used by ``l`` polylines.
    Returns True if the file changed.
    """
    raw = path.read_text()
    newline = "\r\n" if "\r\n" in raw else "\n"
    lines = raw.splitlines()
    positions, uvs, normals = [], [], []
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("v "):
            positions.append(line)
        elif stripped.startswith("vt "):
            uvs.append(line)
        elif stripped.startswith("vn "):
            normals.append(line)
    counts = [len(positions), len(uvs), len(normals)]
    used: list[set[int]] = [set(), set(), set()]
    for line in lines:
        stripped = line.lstrip()
        if not stripped.startswith("f "):
            continue
        for token in stripped.split()[1:]:
            parts = token.split("/")
            for kind, part in enumerate(parts):
                if part:
                    used[kind].add(_parse_obj_index(part, counts[kind]))
    maps: list[dict[int, int]] = []
    kept: list[list[str]] = []
    for kind, src in enumerate((positions, uvs, normals)):
        mapping: dict[int, int] = {}
        keep: list[str] = []
        next_index = 1
        for old, record in enumerate(src, start=1):
            if old in used[kind]:
                mapping[old] = next_index
                keep.append(record)
                next_index += 1
        maps.append(mapping)
        kept.append(keep)

    def remap_corner(corner: str) -> str:
        parts = corner.split("/")
        out = []
        for kind, part in enumerate(parts):
            if part == "":
                out.append("")
            else:
                out.append(str(maps[kind][_parse_obj_index(part, counts[kind])]))
        return "/".join(out)

    out: list[str] = []
    emitted = [False, False, False]
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("l "):
            continue
        if stripped.startswith("v "):
            if not emitted[0]:
                out.extend(kept[0])
                emitted[0] = True
            continue
        if stripped.startswith("vt "):
            if not emitted[1]:
                out.extend(kept[1])
                emitted[1] = True
            continue
        if stripped.startswith("vn "):
            if not emitted[2]:
                out.extend(kept[2])
                emitted[2] = True
            continue
        if stripped.startswith("f "):
            prefix = line[: len(line) - len(line.lstrip())]
            corners = stripped.split()[1:]
            out.append(f"{prefix}f {' '.join(remap_corner(c) for c in corners)}")
            continue
        out.append(line)
    text = newline.join(out) + newline
    if text == raw:
        return False
    path.write_text(text)
    return True


def run_uuc_conversion(urdf: Path, out_dir: Path, uuc_python: str) -> None:
    """Run urdf-usd-converter into ``out_dir`` (wipes it first)."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    cmd = [uuc_python, "-m", "urdf_usd_converter", str(urdf), str(out_dir)]
    print(f"[UUC] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _op_lines(prim) -> list[str]:
    """Render a prim's ordered xformOps as USDA attribute lines."""
    from pxr import UsdGeom  # noqa: PLC0415

    lines = []
    order = []
    for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
        name = op.GetOpName()
        order.append(name)
        attr = op.GetAttr()
        value = op.Get()
        type_name = str(attr.GetTypeName())
        if "quat" in type_name:
            value = (value.GetReal(), *value.GetImaginary())
        lines.append(f"                {type_name} {name} = ({_fmt_floats(value)})")
    lines.append(
        f"                uniform token[] xformOpOrder = [{', '.join(f'\"{n}\"' for n in order)}]"
    )
    return lines


def render_valve_overlay() -> str:
    """Disable UUC single hulls and add baked CoACD parts in the mesh xform."""
    from pxr import Usd  # noqa: PLC0415

    base = VALVE_UUC_DIR / "Ball_Valve.usda"
    stage = Usd.Stage.Open(str(base))
    if not stage:
        raise RuntimeError(f"pxr could not open UUC base: {base}")
    link_meshes = urdf_link_collision_meshes(VALVE_URDF)
    api_list = ", ".join(f'"{api}"' for api in MESH_COLLISION_APIS)
    lines = [
        "#usda 1.0",
        "(",
        '    defaultPrim = "Ball_Valve"',
        "    metersPerUnit = 1",
        '    upAxis = "Z"',
        ")",
        "",
        "# Overlay over the urdf-usd-converter base (usd/uuc/). UUC already places",
        "# collision schemas on the meshes; this swaps the two single-hull colliders",
        "# for baked CoACD parts (a lone convexHull wraps the lever in ~3x phantom",
        "# volume).",
        'def Xform "Ball_Valve" (',
        "    prepend references = @./uuc/Ball_Valve.usda@</Ball_Valve>",
        ")",
        "{",
    ]
    tree: dict = {}
    contents: dict[tuple[str, ...], list[str]] = {}
    for link_path, mesh_name, threshold in VALVE_TARGETS:
        link = link_path.split("/")[-1]
        mesh_path = f"/Ball_Valve/{link_path}/{mesh_name}"
        prim = stage.GetPrimAtPath(mesh_path)
        if not prim or not prim.IsValid():
            raise RuntimeError(f"UUC base target missing: {mesh_path}")
        mesh_files = link_meshes.get(link, [])
        if len(mesh_files) != 1:
            raise RuntimeError(f"Expected 1 URDF collision mesh for link '{link}', got {mesh_files}")
        parts = decompose_mesh(resolve_mesh_file(mesh_files[0], VALVE_URDF), threshold)
        print(f"[DECOMP] {link}/{mesh_name}: {len(parts)} parts (threshold={threshold})")
        elems = tuple(link_path.split("/"))
        node = tree
        for elem in elems:
            node = node.setdefault(elem, {})
        body: list[str] = [
            f'over "{mesh_name}"',
            "{",
            "    bool physics:collisionEnabled = false",
            "}",
            f'def Xform "decomp_{link}"',
            "{",
            *_op_lines(prim),
        ]
        for part_index, (verts, faces) in enumerate(parts):
            body += [
                f'    def Mesh "part_{part_index}" (',
                f"        prepend apiSchemas = [{api_list}]",
                "    )",
                "    {",
                f"        point3f[] points = [{_fmt_points(verts)}]",
                f"        int[] faceVertexCounts = [{_fmt_ints([3] * len(faces))}]",
                f"        int[] faceVertexIndices = [{_fmt_ints(faces)}]",
                '        uniform token physics:approximation = "convexHull"',
                "    }",
            ]
        body.append("}")
        contents[elems] = body

    def emit(node: dict, path: tuple[str, ...], indent: str) -> None:
        for elem, children in node.items():
            lines.append(f'{indent}over "{elem}"')
            lines.append(f"{indent}{{")
            for content_line in contents.get((*path, elem), []):
                lines.append(f"{indent}    {content_line}")
            emit(children, (*path, elem), indent + "    ")
            lines.append(f"{indent}}}")

    emit(tree, (), "    ")
    lines.append("}")
    return "\n".join(lines) + "\n"


def verify() -> list[str]:
    """Check generated UUC USD resolves. Empty list = ok."""
    from pxr import Usd, UsdGeom  # noqa: PLC0415

    problems = []
    spot = SPOT_UUC_DIR / "spot_with_arm.usda"
    if not spot.exists():
        problems.append(f"missing Spot UUC: {spot}")
    else:
        stage = Usd.Stage.Open(str(spot))
        if not stage:
            problems.append(f"pxr could not open {spot}")
        else:
            for name, path in (
                ("sh0", "/spot_with_arm/Geometry/body/arm_link_sh0/arm_link_sh0"),
                (
                    "wr0",
                    "/spot_with_arm/Geometry/body/arm_link_sh0/arm_link_sh1/"
                    "arm_link_el0/arm_link_el1/arm_link_wr0/arm_link_wr0",
                ),
            ):
                prim = stage.GetPrimAtPath(path)
                if not prim or not prim.IsValid() or not prim.IsA(UsdGeom.Mesh):
                    problems.append(f"Spot UUC missing {name} visual mesh: {path}")
    if not VALVE_OVERLAY.exists():
        problems.append(f"missing valve overlay: {VALVE_OVERLAY}")
    else:
        stage = Usd.Stage.Open(str(VALVE_OVERLAY))
        if not stage:
            problems.append(f"pxr could not open {VALVE_OVERLAY}")
        else:
            lever = stage.GetPrimAtPath(
                "/Ball_Valve/Geometry/valvula_esfera/alavanca_pivot/decomp_alavanca_pivot/part_0"
            )
            if not lever or not lever.IsValid():
                problems.append("valve overlay missing CoACD lever part_0")
    return problems


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--assets",
        choices=["all", "spot", "ball_valve"],
        default="all",
        help="Which asset to process (default: all).",
    )
    parser.add_argument(
        "--uuc-python",
        default="/tmp/opencode/uuc-venv/bin/python",
        help="Python with urdf_usd_converter importable.",
    )
    parser.add_argument(
        "--skip-conversion",
        action="store_true",
        help="Skip UUC; only rewrite the valve CoACD overlay / verify.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Do not write anything; check generated USD with pxr.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    want_spot = args.assets in ("all", "spot")
    want_valve = args.assets in ("all", "ball_valve")

    if args.verify_only:
        problems = verify()
        if problems:
            print("[FAIL]")
            for problem in problems:
                print(f"  - {problem}")
            return 1
        print("[OK] Spot UUC + valve overlay resolve")
        return 0

    if not args.skip_conversion:
        if want_spot:
            for obj in SPOT_OBJ_CLEAN:
                if obj.exists() and strip_unused_obj_topology(obj):
                    print(f"[OBJ] stripped unused topology: {obj.name}")
            run_uuc_conversion(SPOT_URDF, SPOT_UUC_DIR, args.uuc_python)
            print(f"[UUC] spot: {SPOT_UUC_DIR / 'spot_with_arm.usda'}")
        if want_valve:
            run_uuc_conversion(VALVE_URDF, VALVE_UUC_DIR, args.uuc_python)
            print(f"[UUC] ball_valve: {VALVE_UUC_DIR / 'Ball_Valve.usda'}")

    if want_valve:
        text = render_valve_overlay()
        VALVE_OVERLAY.write_text(text, encoding="utf-8")
        print(f"[OVERLAY] wrote {VALVE_OVERLAY} ({len(text) // 1024} KiB)")

    problems = verify()
    for problem in problems:
        print(f"[WARN] {problem}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
