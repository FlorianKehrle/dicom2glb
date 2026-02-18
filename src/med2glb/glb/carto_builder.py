"""CARTO animated GLB: LAT wavefront sweep using frame-based visibility.

Creates N mesh copies with different vertex colors representing the
activation wavefront at each time step. Animation switches visible
frame via scale [1,1,1] / [0,0,0] — universally supported in glTF
viewers including HoloLens 2 (MRTK/glTFast).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pygltflib

from med2glb.core.types import MeshData
from med2glb.glb.builder import _pad_to_4, write_accessor
from med2glb.io.carto_colormaps import _LAT_STOPS

logger = logging.getLogger("med2glb")

# Colors for wavefront animation
_INACTIVE_COLOR = np.array([0.3, 0.3, 0.4, 1.0], dtype=np.float32)  # cool gray-blue
_WAVEFRONT_WIDTH = 0.08  # fraction of LAT range for sigmoid transition


def build_carto_animated_glb(
    mesh_data: MeshData,
    lat_values: np.ndarray,
    output_path: Path,
    n_frames: int = 30,
    loop_duration_s: float = 2.0,
    target_faces: int = 20000,
) -> None:
    """Build an animated GLB with LAT wavefront sweep.

    Each frame shows the activation wavefront at a different time step.
    Activated vertices get warm LAT colors, inactive ones get cool gray.

    Args:
        mesh_data: MeshData with vertices, faces, normals (from carto_mesh_to_mesh_data).
        lat_values: Per-vertex LAT values (ms), NaN for unknown.
        output_path: Where to write the .glb file.
        n_frames: Number of animation frames.
        loop_duration_s: Total animation loop time in seconds.
        target_faces: Decimate mesh to this count before creating frames.
    """
    # Decimate if mesh is large (animation duplicates mesh N times)
    if len(mesh_data.faces) > target_faces:
        from med2glb.mesh.processing import decimate, compute_normals

        decimated = decimate(mesh_data, target_faces=target_faces)
        decimated = compute_normals(decimated)
        # Resample LAT values to decimated mesh via nearest neighbor
        from scipy.spatial import KDTree
        tree = KDTree(mesh_data.vertices)
        _, idx = tree.query(decimated.vertices)
        lat_values = lat_values[idx]
        mesh_data = decimated

    valid_lat = ~np.isnan(lat_values)
    if not np.any(valid_lat):
        # No valid LAT — fall back to static export
        from med2glb.glb.builder import build_glb
        build_glb([mesh_data], output_path)
        return

    lat_min = float(np.nanmin(lat_values))
    lat_max = float(np.nanmax(lat_values))
    lat_range = lat_max - lat_min
    if lat_range < 1e-6:
        from med2glb.glb.builder import build_glb
        build_glb([mesh_data], output_path)
        return

    # Normalize LAT to [0, 1]
    lat_norm = (lat_values - lat_min) / lat_range
    lat_norm[~valid_lat] = np.nan

    # Build color stop arrays for LAT colormap interpolation
    positions = np.array([s[0] for s in _LAT_STOPS])
    r_stops = np.array([s[1] for s in _LAT_STOPS])
    g_stops = np.array([s[2] for s in _LAT_STOPS])
    b_stops = np.array([s[3] for s in _LAT_STOPS])

    # Generate per-frame vertex colors
    frame_colors = []
    for fi in range(n_frames):
        t = fi / (n_frames - 1)  # wavefront position in normalized LAT space
        colors = np.zeros((len(mesh_data.vertices), 4), dtype=np.float32)

        # Sigmoid activation: smooth transition at wavefront edge
        activation = _sigmoid((t - lat_norm) / max(_WAVEFRONT_WIDTH, 1e-6))
        activation[~valid_lat] = 0.0

        # Activated color from LAT colormap
        active_r = np.interp(lat_norm, positions, r_stops)
        active_g = np.interp(lat_norm, positions, g_stops)
        active_b = np.interp(lat_norm, positions, b_stops)

        # Blend between inactive and active based on activation
        colors[:, 0] = activation * active_r + (1 - activation) * _INACTIVE_COLOR[0]
        colors[:, 1] = activation * active_g + (1 - activation) * _INACTIVE_COLOR[1]
        colors[:, 2] = activation * active_b + (1 - activation) * _INACTIVE_COLOR[2]
        colors[:, 3] = 1.0

        # NaN vertices stay inactive
        colors[~valid_lat] = _INACTIVE_COLOR

        frame_colors.append(colors)

    # Build glTF with N frame nodes
    gltf = pygltflib.GLTF2(
        scene=0,
        scenes=[pygltflib.Scene(nodes=list(range(n_frames)))],
        nodes=[],
        meshes=[],
        accessors=[],
        bufferViews=[],
        buffers=[],
        materials=[],
        animations=[],
    )
    binary_data = bytearray()

    # Shared geometry: positions, normals, indices (written once)
    vertices = mesh_data.vertices.astype(np.float32)
    pos_acc = write_accessor(
        gltf, binary_data, vertices, pygltflib.ARRAY_BUFFER,
        pygltflib.FLOAT, pygltflib.VEC3, with_minmax=True,
    )

    norm_acc = None
    if mesh_data.normals is not None:
        normals = mesh_data.normals.astype(np.float32)
        norm_acc = write_accessor(
            gltf, binary_data, normals, pygltflib.ARRAY_BUFFER,
            pygltflib.FLOAT, pygltflib.VEC3,
        )

    faces = mesh_data.faces.astype(np.uint32)
    idx_acc = write_accessor(
        gltf, binary_data, faces.ravel(), pygltflib.ELEMENT_ARRAY_BUFFER,
        pygltflib.UNSIGNED_INT, pygltflib.SCALAR, with_minmax=True,
    )

    # Per-frame: material + COLOR_0 + mesh + node
    for fi in range(n_frames):
        # Material: white base color, vertex colors drive appearance
        mat_idx = len(gltf.materials)
        gltf.materials.append(pygltflib.Material(
            name=f"wavefront_{fi}",
            pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
                baseColorFactor=[1.0, 1.0, 1.0, 1.0],
                metallicFactor=0.0,
                roughnessFactor=0.7,
            ),
            doubleSided=True,
        ))

        # COLOR_0 accessor for this frame
        color_acc = write_accessor(
            gltf, binary_data, frame_colors[fi], pygltflib.ARRAY_BUFFER,
            pygltflib.FLOAT, pygltflib.VEC4,
        )

        # Primitive with shared geometry + per-frame color
        attrs = pygltflib.Attributes(POSITION=pos_acc)
        if norm_acc is not None:
            attrs.NORMAL = norm_acc
        attrs.COLOR_0 = color_acc

        mesh_idx = len(gltf.meshes)
        gltf.meshes.append(pygltflib.Mesh(
            name=f"wavefront_{fi}",
            primitives=[pygltflib.Primitive(
                attributes=attrs,
                indices=idx_acc,
                material=mat_idx,
            )],
        ))

        # Node: first frame visible, rest hidden
        scale = [1.0, 1.0, 1.0] if fi == 0 else [0.0, 0.0, 0.0]
        gltf.nodes.append(pygltflib.Node(
            name=f"wavefront_{fi}", mesh=mesh_idx, scale=scale,
        ))

    # Animation: switch visible frame via scale keyframes
    dt = loop_duration_s / n_frames
    keyframe_times = np.array([i * dt for i in range(n_frames)], dtype=np.float32)
    time_acc = write_accessor(
        gltf, binary_data, keyframe_times, None,
        pygltflib.FLOAT, pygltflib.SCALAR, with_minmax=True,
    )

    channels = []
    samplers = []
    for fi in range(n_frames):
        # Scale output: [1,1,1] at this frame's keyframe, [0,0,0] at others
        scales = np.zeros((n_frames, 3), dtype=np.float32)
        scales[fi] = [1.0, 1.0, 1.0]

        scale_acc = write_accessor(
            gltf, binary_data, scales, None,
            pygltflib.FLOAT, pygltflib.VEC3,
        )

        sampler_idx = len(samplers)
        samplers.append(pygltflib.AnimationSampler(
            input=time_acc,
            output=scale_acc,
            interpolation=pygltflib.ANIM_STEP,
        ))
        channels.append(pygltflib.AnimationChannel(
            sampler=sampler_idx,
            target=pygltflib.AnimationChannelTarget(node=fi, path="scale"),
        ))

    gltf.animations.append(pygltflib.Animation(
        name="lat_wavefront",
        channels=channels,
        samplers=samplers,
    ))

    # Finalize
    gltf.buffers.append(pygltflib.Buffer(byteLength=len(binary_data)))
    gltf.set_binary_blob(bytes(binary_data))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gltf.save(str(output_path))

    logger.info(
        f"CARTO animated GLB: {n_frames} frames, "
        f"{len(mesh_data.vertices)} verts, "
        f"{output_path.stat().st_size / 1024:.0f} KB"
    )


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid for wavefront transition."""
    return np.where(
        x >= 0,
        1 / (1 + np.exp(-np.clip(x, -30, 30))),
        np.exp(np.clip(x, -30, 30)) / (1 + np.exp(np.clip(x, -30, 30))),
    )
