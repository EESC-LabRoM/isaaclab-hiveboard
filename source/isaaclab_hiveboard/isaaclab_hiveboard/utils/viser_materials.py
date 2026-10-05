# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Matte material for textured meshes in Newton's viser viewer.

Newton 1.6 sends a textured mesh to viser as a trimesh ``TextureVisuals``
built from the bare image. trimesh exports that as a glTF material with
``baseColorFactor`` 0.4 (its default diffuse) and no ``metallicFactor``,
which glTF reads as 1.0. viser draws it as a dark metal: Newton removes the
environment map, so a metal surface has nothing to reflect. The ANYmal-D
shell, whose USD materials are all texture maps, came out near black.

The patch keeps the texture and its UVs, and swaps in an untinted,
non-metallic PBR material, as the USD materials intend.
"""

from __future__ import annotations

# Matte, close to UsdPreviewSurface's default roughness (0.5) and the
# untextured shapes viser draws.
TEXTURED_ROUGHNESS = 0.6


def patch_viser_textured_materials() -> None:
    """Make Newton's viser viewer draw textured meshes untinted and non-metallic.

    Idempotent. Call before the visualizer logs the model's meshes.
    """
    from newton.viewer import ViewerViser

    if getattr(ViewerViser, "_hiveboard_matte_textures", False):
        return
    build = ViewerViser._build_trimesh_mesh

    def _build_matte_trimesh_mesh(points, indices, uvs, texture):
        mesh = build(points, indices, uvs, texture)
        image = getattr(getattr(getattr(mesh, "visual", None), "material", None), "image", None)
        if image is None:
            return mesh
        from trimesh.visual.material import PBRMaterial

        mesh.visual.material = PBRMaterial(
            baseColorTexture=image,
            baseColorFactor=[255, 255, 255, 255],
            metallicFactor=0.0,
            roughnessFactor=TEXTURED_ROUGHNESS,
        )
        return mesh

    ViewerViser._build_trimesh_mesh = staticmethod(_build_matte_trimesh_mesh)
    ViewerViser._hiveboard_matte_textures = True
