# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Recover the drawer's guide primitives from the upstream collision meshes.

The source guide OBJs are boxes and Y-axis cylinders. Keeping those as
separate primitives preserves the slot; hulling the housing would close it.
"""

from __future__ import annotations

import copy
import xml.etree.ElementTree as ET

from drawer_path import DRAWER_URDF, REPO_ROOT

SOURCE_URDF = REPO_ROOT / "dependencies/HiveBoard/Simulation/Drawer/Drawer_Assembly.urdf"
_GUIDES = {
    "left_Guide_Cube": "cut_left_ceiling",
    "left_Guide_Cube_01": "cut_left_upper",
    "left_Guide_Cube_02": "cut_left_floor",
    "left_Guide_Cube_03": "cut_left_lip",
    "left_Guide_Cube_04": "cut_left_rear",
    "Cylinder": "cut_left_round",
    "right_Guide_Cube": "cut_right_ceiling",
    "right_Guide_Cube_01": "cut_right_upper",
    "right_Guide_Cube_02": "cut_right_floor",
    "right_Guide_Cube_03": "cut_right_lip",
    "right_Guide_Cube_04": "cut_right_rear",
    "right_Guide_Cylinder_01": "cut_right_round",
    "tn__Gaveta1_aE_Corpo1_Cylinder_01": "shaft_left",
    "Cylinder_02": "shaft_right",
}


def source_primitives() -> list[ET.Element]:
    """Replace source box/cylinder meshes with equivalent URDF primitives."""
    shapes = []
    for item in ET.parse(SOURCE_URDF).getroot().iter("collision"):
        mesh = item.find("geometry/mesh")
        path = SOURCE_URDF.parent / mesh.attrib["filename"]
        name = _GUIDES.get(path.stem)
        if name is None:
            continue
        points = [
            tuple(float(value) * 0.001 for value in line.split()[1:4])
            for line in path.read_text().splitlines() if line.startswith("v ")
        ]
        low = tuple(min(point[i] for point in points) for i in range(3))
        high = tuple(max(point[i] for point in points) for i in range(3))
        center = tuple((a + b) / 2 for a, b in zip(low, high))
        if any(abs(value) > 1e-7 for value in center):
            raise ValueError(f"Source primitive {path} is not centered")
        col = ET.Element("collision", name=name)
        ET.SubElement(col, "origin", dict(item.find("origin").attrib))
        geom = ET.SubElement(col, "geometry")
        if "Cylinder" in path.stem:
            ET.SubElement(
                geom, "cylinder", radius=f"{(high[0] - low[0]) / 2:.10f}", length=f"{high[2] - low[2]:.10f}"
            )
        else:
            ET.SubElement(geom, "box", size=" ".join(f"{b - a:.10f}" for a, b in zip(low, high)))
        shapes.append(col)
    if len(shapes) != len(_GUIDES):
        raise ValueError("Source drawer is missing guide or shaft collision meshes")
    return shapes


def sync_drawer_collision() -> None:
    """Update guide and shaft collision without changing visuals or the free joint."""
    tree = ET.parse(DRAWER_URDF, parser=ET.XMLParser(target=ET.TreeBuilder(insert_comments=True)))
    root = tree.getroot()
    links = {link.attrib["name"]: link for link in root.findall("link")}
    for link in links.values():
        for col in list(link.findall("collision")):
            if col.attrib.get("name", "").startswith(("cut_", "shaft_")):
                link.remove(col)
    for shape in source_primitives():
        link = links["base" if shape.attrib["name"].startswith("cut_") else "drawer"]
        link.append(copy.deepcopy(shape))
    # Place the diagnostic force application point on the actual shaft axis.
    left = next(
        shape for shape in links["drawer"].findall("collision") if shape.attrib["name"] == "shaft_left"
    )
    x, _, z = left.find("origin").attrib["xyz"].split()
    links["drawer"].find("inertial/origin").set("xyz", f"{x} 0 {z}")
    ET.indent(tree, space="  ")
    tree.write(DRAWER_URDF, encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    sync_drawer_collision()
