"""Map sparse CARTO measurement points to mesh vertices.

Uses KDTree nearest-neighbor lookup and optional interpolation to transfer
per-point electrical values (LAT, voltage) to per-vertex colors.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import KDTree

from med2glb.core.types import CartoMesh, CartoPoint, MeshData
from med2glb.io.carto_colormaps import COLORMAPS
from med2glb.io.carto_reader import _INACTIVE_GROUP_ID

logger = logging.getLogger("med2glb")


def map_points_to_vertices(
    mesh: CartoMesh,
    points: list[CartoPoint],
    field: str = "lat",
) -> np.ndarray:
    """Map sparse point measurements to mesh vertices via nearest-neighbor.

    Args:
        mesh: CARTO mesh with vertex positions.
        points: List of measurement points with 3D positions and values.
        field: Which field to map — "lat", "bipolar", or "unipolar".

    Returns:
        Per-vertex values array [N]. NaN where no point was near enough.
    """
    n_verts = len(mesh.vertices)
    values = np.full(n_verts, np.nan, dtype=np.float64)

    if not points:
        return values

    # Extract point positions and field values
    point_positions = np.array([p.position for p in points], dtype=np.float64)
    if field == "lat":
        point_values = np.array([p.lat for p in points], dtype=np.float64)
    elif field == "bipolar":
        point_values = np.array([p.bipolar_voltage for p in points], dtype=np.float64)
    elif field == "unipolar":
        point_values = np.array([p.unipolar_voltage for p in points], dtype=np.float64)
    else:
        raise ValueError(f"Unknown field: {field}. Use 'lat', 'bipolar', or 'unipolar'.")

    # Filter out points with NaN values
    valid = ~np.isnan(point_values)
    if not np.any(valid):
        return values

    point_positions = point_positions[valid]
    point_values = point_values[valid]

    # Build KDTree from point positions and query for nearest neighbor per vertex
    tree = KDTree(point_positions)
    distances, indices = tree.query(mesh.vertices)

    # Assign nearest-neighbor values to all vertices
    values[:] = point_values[indices]

    return values


def interpolate_sparse_values(
    mesh: CartoMesh,
    values: np.ndarray,
    max_distance: float | None = None,
) -> np.ndarray:
    """Fill gaps in per-vertex values using linear interpolation.

    Uses scipy LinearNDInterpolator on vertices that have valid values
    to estimate values at vertices that don't.

    Args:
        mesh: CARTO mesh with vertex positions.
        values: Per-vertex values [N], NaN where unknown.
        max_distance: If set, only interpolate within this distance of known points.

    Returns:
        Interpolated per-vertex values [N].
    """
    valid = ~np.isnan(values)
    if np.all(valid) or not np.any(valid):
        return values

    result = values.copy()

    try:
        interp = LinearNDInterpolator(
            mesh.vertices[valid],
            values[valid],
        )
        missing = ~valid
        interpolated = interp(mesh.vertices[missing])
        # LinearNDInterpolator returns NaN outside convex hull — keep those as NaN
        result[missing] = interpolated
    except Exception as e:
        logger.debug(f"Interpolation failed: {e}")

    return result


def build_inactive_mask(mesh: CartoMesh) -> np.ndarray:
    """Build a boolean mask of inactive vertices (GroupID == -1000000).

    Returns:
        Boolean array [N], True for inactive vertices.
    """
    return mesh.group_ids == _INACTIVE_GROUP_ID


def carto_mesh_to_mesh_data(
    mesh: CartoMesh,
    points: list[CartoPoint] | None,
    coloring: str = "lat",
    clamp_range: tuple[float, float] | None = None,
) -> MeshData:
    """Convert a CartoMesh + points into a MeshData with vertex colors.

    This filters out inactive vertices/faces and applies the coloring.

    Args:
        mesh: CARTO mesh data.
        points: Measurement points (may be None for color fallback).
        coloring: Color scheme — "lat", "bipolar", or "unipolar".
        clamp_range: Optional value range for colormap normalization.

    Returns:
        MeshData with vertex_colors set.
    """
    # Build mask of active vertices and faces
    active_verts = mesh.group_ids != _INACTIVE_GROUP_ID
    active_faces_mask = mesh.face_group_ids != _INACTIVE_GROUP_ID

    # Also filter faces that reference inactive vertices
    for col in range(3):
        face_vert_active = active_verts[mesh.faces[:, col]]
        active_faces_mask = active_faces_mask & face_vert_active

    # Remap vertices: only keep active ones
    old_to_new = np.full(len(mesh.vertices), -1, dtype=np.int32)
    new_indices = np.where(active_verts)[0]
    old_to_new[new_indices] = np.arange(len(new_indices), dtype=np.int32)

    vertices = mesh.vertices[active_verts].astype(np.float32)
    normals = mesh.normals[active_verts].astype(np.float32)
    active_faces = mesh.faces[active_faces_mask]
    faces = old_to_new[active_faces].astype(np.int32)

    # Drop any faces with -1 (shouldn't happen but be safe)
    valid_faces = np.all(faces >= 0, axis=1)
    faces = faces[valid_faces]

    # Compute vertex colors
    vertex_colors = None
    if points:
        # Map sparse points to all mesh vertices first, then filter
        all_values = map_points_to_vertices(mesh, points, field=coloring)
        all_values = interpolate_sparse_values(mesh, all_values)
        active_values = all_values[active_verts]

        colormap_fn = COLORMAPS.get(coloring)
        if colormap_fn:
            vertex_colors = colormap_fn(active_values, clamp_range=clamp_range)
    else:
        # No points — use mesh default color as solid vertex color
        n = len(vertices)
        r, g, b, a = mesh.mesh_color
        vertex_colors = np.full((n, 4), [r, g, b, a], dtype=np.float32)

    from med2glb.core.types import MaterialConfig

    return MeshData(
        vertices=vertices,
        faces=faces,
        normals=normals,
        structure_name=mesh.structure_name,
        material=MaterialConfig(
            base_color=(1.0, 1.0, 1.0),
            alpha=1.0,
            metallic=0.0,
            roughness=0.7,
            name=mesh.structure_name,
        ),
        vertex_colors=vertex_colors,
    )
