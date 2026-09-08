# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regenerate the kitless Newton USD assets (Spot arm + ball valve).

Tutorial — why this script exists
---------------------------------
The Newton tasks do not load URDFs at runtime. They load a preconverted USD
base plus a small Newton overlay (``*_newton.usda``). Both are gitignored
(``**/*.usd*`` in ``.gitignore``), so a fresh clone cannot run the Newton
tasks until someone generates them locally. This script is the committable
record of *exactly* how those files are produced, so anyone can regenerate
them with one command.

Two layers per asset:

1. Base USD — IsaacLab ``UrdfConverter`` (Isaac Sim URDF importer 3.0) from
   the committed URDF, with the pinned settings below (``fix_base=True``,
   ``make_instanceable=True``, ``convex_hull`` colliders, zero joint-drive
   gains because Newton tasks drive joints with explicit Lab actuators).
   Requires Isaac Sim / Kit (``omni.kit``, ``isaacsim.asset.importer.urdf``).
2. Newton overlay — a tiny text USDA that references the base file and adds
   opinions Newton needs but the importer does not emit: Isaac Sim 5 places
   mesh-collision schemas on the OBJ importer's ``World`` Xform, while Newton
   requires ``PhysicsCollisionAPI`` + ``PhysicsMeshCollisionAPI`` on the
   actual ``UsdGeomMesh``. The overlay also de-instances the patched
   collision branches and supplies fallback mass/inertia for links whose
   URDF inertia is invalid (Spot feet/jaw). Overlay-only mode is kitless
   (needs only ``pxr``).

What is committed vs generated::

    committed:  <asset>.urdf (+ meshes), this script
    generated:  usd/<Base>.usd, usd/configuration/..., usd/.asset_hash,
                usd/config.yaml, usd/<Base>_newton.usda

Usage::

    # check a clone has everything the Newton task needs (kitless)
    uv run python scripts/generate_newton_usd.py --verify-only

    # (re)write just the Newton overlays, e.g. after editing the specs
    # below (kitless)
    uv run python scripts/generate_newton_usd.py --skip-conversion

    # full regeneration incl. URDF import (needs Isaac Sim, run with Kit)
    uv run python scripts/generate_newton_usd.py --force

    # single asset / preview without writing
    uv run python scripts/generate_newton_usd.py --assets spot --dry-run
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXT_ASSETS = REPO_ROOT / "source/isaaclab_hiveboard/isaaclab_hiveboard/assets"

MESH_COLLISION_APIS = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]


@dataclass
class OverlayBlock:
    """One top-level ``over`` in the Newton overlay: optional inertia fix plus
    optional mesh-collision branches under ``<link>/collisions``."""

    link: str
    branches: tuple[str, ...] = ()
    inertia_fix: bool = False


@dataclass
class NewtonAssetSpec:
    """Everything needed to (re)generate one Newton asset."""

    key: str
    urdf: Path
    usd_dir: Path
    base_usd: str
    overlay_usd: str
    default_prim: str
    header_comment: tuple[str, ...]
    blocks: tuple[OverlayBlock, ...]
    mass: str = "0.001"
    inertia_diag: str = "1e-6, 1e-6, 1e-6"


SPOT_SPEC = NewtonAssetSpec(
    key="spot",
    urdf=EXT_ASSETS / "spot/spot_with_arm.urdf",
    usd_dir=EXT_ASSETS / "spot/usd",
    base_usd="spot_with_arm.usd",
    overlay_usd="spot_with_arm_newton.usda",
    default_prim="spot_with_arm",
    header_comment=(
        "# Isaac Sim 5 placed collision schemas on the OBJ importer's World Xform.",
        "# Newton requires them on the actual UsdGeomMesh. Only the two gripper links",
        "# need mesh contacts for this fixed-base valve task; primitive colliders from",
        "# the source asset are preserved unchanged.",
    ),
    blocks=(
        OverlayBlock("arm_link_fngr", ("left_finger", "left_tooth", "right_finger", "right_tooth")),
        OverlayBlock("arm_link_jaw", ("front_jaw", "middle_jaw", "jaw_tooth"), inertia_fix=True),
        OverlayBlock("fl_foot", inertia_fix=True),
        OverlayBlock("fr_foot", inertia_fix=True),
        OverlayBlock("hl_foot", inertia_fix=True),
        OverlayBlock("hr_foot", inertia_fix=True),
    ),
)

BALL_VALVE_SPEC = NewtonAssetSpec(
    key="ball_valve",
    urdf=EXT_ASSETS / "hiveboard/ball_valve/Ball_Valve.urdf",
    usd_dir=EXT_ASSETS / "hiveboard/ball_valve/usd",
    base_usd="Ball_Valve.usd",
    overlay_usd="Ball_Valve_newton.usda",
    default_prim="Ball_Valve",
    header_comment=(
        "# Isaac Sim 5 placed the mesh collision schemas on the OBJ importer's World",
        "# Xform.  Newton requires them on the actual UsdGeomMesh, so this portable",
        "# layer de-instances the two collision branches and supplies the corrected",
        "# schema opinions without needing a converter at runtime.",
    ),
    blocks=(
        OverlayBlock("valvula_esfera", ("Mesh",)),
        OverlayBlock("alavanca_pivot", ("Mesh_1",)),
    ),
)

ASSET_SPECS: dict[str, NewtonAssetSpec] = {spec.key: spec for spec in (SPOT_SPEC, BALL_VALVE_SPEC)}

# Pinned URDF-importer settings. Must match the committed
# ``usd/config.yaml`` files; change them here, not by hand-editing outputs.
CONVERTER_SETTINGS: dict[str, object] = {
    "usd_file_name": None,
    "force_usd_conversion": False,
    "make_instanceable": True,
    "fix_base": True,
    "root_link_name": None,
    "link_density": 0.0,
    "merge_fixed_joints": False,
    "convert_mimic_joints_to_normal_joints": False,
    "joint_drive_type": "force",
    "joint_target_type": "position",
    "joint_stiffness": 0.0,
    "joint_damping": 0.0,
    "collision_from_visuals": False,
    "collision_type": "Convex Hull",
    "self_collision": False,
    "replace_cylinders_with_capsules": False,
}


def render_overlay(spec: NewtonAssetSpec) -> str:
    """Render the Newton overlay USDA text deterministically from ``spec``."""
    lines = [
        "#usda 1.0",
        "(",
        f'    defaultPrim = "{spec.default_prim}"',
        "    metersPerUnit = 1",
        '    upAxis = "Z"',
        ")",
        "",
        *spec.header_comment,
        f'def Xform "{spec.default_prim}" (',
        f"    prepend references = @{spec.base_usd}@</{spec.default_prim}>",
        ")",
        "{",
    ]
    api_list = ", ".join(f'"{api}"' for api in MESH_COLLISION_APIS)
    previous: OverlayBlock | None = None
    for block in spec.blocks:
        # Blank line between top-level overs, except between consecutive
        # inertia-only foot blocks (matches the committed overlays).
        if previous is not None and (previous.branches or block.branches):
            lines.append("")
        previous = block
        lines.append(f'    over "{block.link}"')
        lines.append("    {")
        if block.inertia_fix:
            lines.append(f"        float3 physics:diagonalInertia = ({spec.inertia_diag})")
            lines.append(f"        float physics:mass = {spec.mass}")
            if block.branches:
                lines.append("")
        if block.branches:
            lines.append('        over "collisions" (instanceable = false)')
            lines.append("        {")
            for branch in block.branches:
                lines.append(f'            over "{branch}"')
                lines.append("            {")
                lines.append('                over "World"')
                lines.append("                {")
                lines.append("                    over \"mesh\" (")
                lines.append(f"                        prepend apiSchemas = [{api_list}]")
                lines.append("                    )")
                lines.append("                    {")
                lines.append('                        uniform token physics:approximation = "convexHull"')
                lines.append("                    }")
                lines.append("                }")
                lines.append("            }")
            lines.append("        }")
        lines.append("    }")
    lines.append("}")
    return "\n".join(lines) + "\n"


def overlay_targets(spec: NewtonAssetSpec) -> list[str]:
    """Prim paths in the *base* stage every ``over`` in the overlay addresses."""
    targets = []
    for block in spec.blocks:
        if block.inertia_fix and not block.branches:
            targets.append(f"/{spec.default_prim}/{block.link}")
        for branch in block.branches:
            targets.append(f"/{spec.default_prim}/{block.link}/collisions/{branch}/World/mesh")
    return targets


def write_overlay(spec: NewtonAssetSpec, *, dry_run: bool = False) -> tuple[bool, str]:
    """Write (or preview) the overlay file. Returns (changed, text)."""
    text = render_overlay(spec)
    path = spec.usd_dir / spec.overlay_usd
    if dry_run:
        return True, text
    previous = path.read_text(encoding="utf-8") if path.exists() else None
    if previous != text:
        path.write_text(text, encoding="utf-8")
        return True, text
    return False, text


def verify_asset(spec: NewtonAssetSpec) -> list[str]:
    """Check base + overlay resolve with pxr. Returns a list of problems (empty = ok)."""
    from pxr import Usd  # noqa: PLC0415  (kitless; usd-core ships with the venv)

    problems = []
    base_path = spec.usd_dir / spec.base_usd
    overlay_path = spec.usd_dir / spec.overlay_usd
    if not base_path.exists():
        return [f"missing base USD: {base_path} (run without --skip-conversion on a Kit machine)"]
    stage = Usd.Stage.Open(str(base_path))
    if not stage:
        return [f"pxr could not open base USD: {base_path}"]
    for target in overlay_targets(spec):
        prim = stage.GetPrimAtPath(target)
        if not prim or not prim.IsValid():
            problems.append(f"overlay target missing from {spec.base_usd}: {target}")
    if overlay_path.exists():
        overlay_stage = Usd.Stage.Open(str(overlay_path))
        if not overlay_stage:
            problems.append(f"pxr could not open overlay: {overlay_path}")
    else:
        problems.append(f"missing overlay: {overlay_path} (run with --skip-conversion)")
    return problems


def run_conversion(spec: NewtonAssetSpec, *, force: bool) -> str:
    """Run the Isaac Sim URDF importer. Requires Kit (omni.kit + isaacsim)."""
    try:
        from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: PLC0415
    except ImportError as err:
        raise RuntimeError(
            "URDF conversion needs Isaac Sim / Kit (omni.kit, isaacsim.asset.importer.urdf), "
            "which is not importable here. Re-run this script on a machine with Isaac Sim, "
            "or use --skip-conversion to (re)write just the Newton overlays."
        ) from err
    if not spec.urdf.exists():
        raise FileNotFoundError(f"URDF not found: {spec.urdf}")
    cfg = UrdfConverterCfg(
        asset_path=str(spec.urdf),
        usd_dir=str(spec.usd_dir),
        usd_file_name=None,
        force_usd_conversion=force,
        make_instanceable=True,
        fix_base=True,
        link_density=0.0,
        merge_fixed_joints=False,
        convert_mimic_joints_to_normal_joints=False,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="position",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
        ),
        collision_from_visuals=False,
        collision_type="Convex Hull",
        self_collision=False,
        replace_cylinders_with_capsules=False,
    )
    return UrdfConverter(cfg).usd_path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--assets",
        choices=["all", *ASSET_SPECS],
        default="all",
        help="Which asset to process (default: all).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force URDF re-import even if the converter hash matches.",
    )
    parser.add_argument(
        "--skip-conversion",
        action="store_true",
        help="Skip the URDF importer (kitless); only write overlays / verify.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Do not write anything; check base + overlay resolve with pxr.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the overlay text that would be written, without writing.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    specs = list(ASSET_SPECS.values()) if args.assets == "all" else [ASSET_SPECS[args.assets]]
    failures = 0

    if args.verify_only:
        for spec in specs:
            problems = verify_asset(spec)
            if problems:
                failures += 1
                print(f"[FAIL] {spec.key}:")
                for problem in problems:
                    print(f"  - {problem}")
            else:
                print(f"[OK] {spec.key}: {spec.base_usd} + {spec.overlay_usd} resolve")
        return 1 if failures else 0

    simulation_app = None
    if not args.skip_conversion and not args.dry_run:
        # Kit entry point: the URDF importer needs omni.kit / isaacsim.
        from isaaclab.app import AppLauncher  # noqa: PLC0415

        launcher_parser = argparse.ArgumentParser()
        AppLauncher.add_app_launcher_args(launcher_parser)
        app_args, _ = launcher_parser.parse_known_args([])
        try:
            app_launcher = AppLauncher(app_args)
        except Exception as err:
            raise RuntimeError(
                "Could not start the Kit app (Isaac Sim is required for URDF "
                "conversion). Re-run on a machine with Isaac Sim, or use "
                "--skip-conversion to (re)write just the Newton overlays."
            ) from err
        simulation_app = app_launcher.app
        for spec in specs:
            try:
                usd_path = run_conversion(spec, force=args.force)
            except RuntimeError as err:
                print(f"[FAIL] {spec.key}: {err}", file=sys.stderr)
                failures += 1
                continue
            print(f"[CONVERT] {spec.key}: {usd_path}")
            expected_base = spec.usd_dir / spec.base_usd
            if not expected_base.exists():
                print(
                    f"[WARN] {spec.key}: importer output layout changed; overlay expects "
                    f"{expected_base} but it is missing. Update {spec.overlay_usd}'s "
                    f"reference or the spec instead of hand-editing outputs.",
                    file=sys.stderr,
                )
    try:
        for spec in specs:
            if args.dry_run:
                print(render_overlay(spec))
                continue
            changed, _ = write_overlay(spec)
            print(f"[OVERLAY] {spec.key}: {'wrote' if changed else 'up to date'} {spec.usd_dir / spec.overlay_usd}")
            for problem in verify_asset(spec):
                print(f"[WARN] {spec.key}: {problem}", file=sys.stderr)
    finally:
        if simulation_app is not None:
            simulation_app.close()
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
