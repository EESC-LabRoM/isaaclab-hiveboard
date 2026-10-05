# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Polygonal, smooth-normal meshes in Newton's USD and RTX viewers.

Newton 1.6 ``ViewerUSD.log_mesh`` defines each mesh prim without a
``subdivisionScheme``, so USD's default ``catmullClark`` applies. The RTX
viewer (built on ViewerUSD) then refines the triangles into a smooth limit
surface: the 2F-140 palm loses its screws, cut-outs and split lines and
renders as a blob, and sharp edges look melted. Turning subdivision off alone
shades every triangle flat, because Newton 1.6 meshes usually arrive without
normals. The GL viewer averages adjacent triangle normals instead.

The patch does what Newton main does (PR #4177): author
``subdivisionScheme = none`` and generate smooth vertex normals when none are
given. Drop it once Isaac Lab moves past Newton 1.6.
"""

from __future__ import annotations


def patch_usd_viewer_subdivision() -> None:
    """Log Newton's USD/RTX viewer meshes as polygons with smooth vertex normals.

    Idempotent. Call before the visualizer logs the model's meshes.
    """
    from newton._src.utils.mesh import compute_vertex_normals
    from newton._src.viewer.viewer_usd import ViewerUSD
    from pxr import UsdGeom

    if getattr(ViewerUSD, "_hiveboard_no_subdivision", False):
        return
    log_mesh = ViewerUSD.log_mesh

    def _log_unsubdivided_mesh(self, name, points, indices, normals=None, *args, **kwargs):
        if normals is None:
            normals = compute_vertex_normals(points, indices, device=points.device)
        result = log_mesh(self, name, points, indices, normals, *args, **kwargs)
        mesh_prim = self._meshes.get(self._qualify(name))
        if mesh_prim is not None:
            attr = mesh_prim.GetSubdivisionSchemeAttr()
            if attr.Get() != UsdGeom.Tokens.none:
                attr.Set(UsdGeom.Tokens.none)
        return result

    ViewerUSD.log_mesh = _log_unsubdivided_mesh
    ViewerUSD._hiveboard_no_subdivision = True
