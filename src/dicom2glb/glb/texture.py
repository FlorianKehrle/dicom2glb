"""Textured plane GLB: embed a DICOM image as a textured quad via pygltflib."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pygltflib

from dicom2glb.core.volume import DicomVolume, TemporalSequence
from dicom2glb.glb.builder import _pad_to_4


def build_textured_plane_glb(volume: DicomVolume, output_path: Path) -> None:
    """Build a GLB containing a textured quad from a single DICOM slice."""
    gltf = pygltflib.GLTF2(
        scene=0,
        scenes=[pygltflib.Scene(nodes=[0])],
        nodes=[pygltflib.Node(name="image_plane", mesh=0)],
        meshes=[],
        accessors=[],
        bufferViews=[],
        buffers=[],
        materials=[],
        textures=[],
        images=[],
        samplers=[],
    )

    binary_data = bytearray()

    # Get 2D pixel data (first slice of the volume)
    pixel_data = volume.voxels[0]  # [Y, X]
    rows, cols = pixel_data.shape

    # Compute physical dimensions from pixel spacing
    row_spacing, col_spacing = volume.pixel_spacing
    width = cols * col_spacing / 1000.0  # Convert mm to meters for glTF
    height = rows * row_spacing / 1000.0

    # Create quad vertices (two triangles forming a rectangle)
    half_w = width / 2
    half_h = height / 2
    vertices = np.array(
        [
            [-half_w, -half_h, 0.0],
            [half_w, -half_h, 0.0],
            [half_w, half_h, 0.0],
            [-half_w, half_h, 0.0],
        ],
        dtype=np.float32,
    )

    # UV coordinates
    texcoords = np.array(
        [
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, 0.0],
            [0.0, 0.0],
        ],
        dtype=np.float32,
    )

    # Two triangles
    indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint16)

    # Normals (all pointing +Z)
    normals = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    # --- Encode image as PNG ---
    png_bytes = _pixel_data_to_png(pixel_data)

    # Write image data first
    img_offset = len(binary_data)
    binary_data.extend(png_bytes)
    _pad_to_4(binary_data)

    img_bv_idx = len(gltf.bufferViews)
    gltf.bufferViews.append(
        pygltflib.BufferView(
            buffer=0,
            byteOffset=img_offset,
            byteLength=len(png_bytes),
        )
    )

    gltf.images.append(
        pygltflib.Image(bufferView=img_bv_idx, mimeType="image/png")
    )

    gltf.samplers.append(
        pygltflib.Sampler(
            magFilter=pygltflib.LINEAR,
            minFilter=pygltflib.LINEAR,
            wrapS=pygltflib.CLAMP_TO_EDGE,
            wrapT=pygltflib.CLAMP_TO_EDGE,
        )
    )

    gltf.textures.append(pygltflib.Texture(sampler=0, source=0))

    # Material with base color texture
    gltf.materials.append(
        pygltflib.Material(
            name="dicom_image",
            pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
                baseColorTexture=pygltflib.TextureInfo(index=0),
                metallicFactor=0.0,
                roughnessFactor=1.0,
            ),
            doubleSided=True,
        )
    )

    # --- Write vertex data ---
    # Positions
    pos_data = vertices.tobytes()
    pos_offset = len(binary_data)
    binary_data.extend(pos_data)
    _pad_to_4(binary_data)

    pos_bv_idx = len(gltf.bufferViews)
    gltf.bufferViews.append(
        pygltflib.BufferView(
            buffer=0,
            byteOffset=pos_offset,
            byteLength=len(pos_data),
            target=pygltflib.ARRAY_BUFFER,
        )
    )
    pos_acc_idx = len(gltf.accessors)
    gltf.accessors.append(
        pygltflib.Accessor(
            bufferView=pos_bv_idx,
            componentType=pygltflib.FLOAT,
            count=4,
            type=pygltflib.VEC3,
            max=vertices.max(axis=0).tolist(),
            min=vertices.min(axis=0).tolist(),
        )
    )

    # Normals
    norm_data = normals.tobytes()
    norm_offset = len(binary_data)
    binary_data.extend(norm_data)
    _pad_to_4(binary_data)

    norm_bv_idx = len(gltf.bufferViews)
    gltf.bufferViews.append(
        pygltflib.BufferView(
            buffer=0,
            byteOffset=norm_offset,
            byteLength=len(norm_data),
            target=pygltflib.ARRAY_BUFFER,
        )
    )
    norm_acc_idx = len(gltf.accessors)
    gltf.accessors.append(
        pygltflib.Accessor(
            bufferView=norm_bv_idx,
            componentType=pygltflib.FLOAT,
            count=4,
            type=pygltflib.VEC3,
        )
    )

    # Texcoords
    tc_data = texcoords.tobytes()
    tc_offset = len(binary_data)
    binary_data.extend(tc_data)
    _pad_to_4(binary_data)

    tc_bv_idx = len(gltf.bufferViews)
    gltf.bufferViews.append(
        pygltflib.BufferView(
            buffer=0,
            byteOffset=tc_offset,
            byteLength=len(tc_data),
            target=pygltflib.ARRAY_BUFFER,
        )
    )
    tc_acc_idx = len(gltf.accessors)
    gltf.accessors.append(
        pygltflib.Accessor(
            bufferView=tc_bv_idx,
            componentType=pygltflib.FLOAT,
            count=4,
            type=pygltflib.VEC2,
        )
    )

    # Indices
    idx_data = indices.tobytes()
    idx_offset = len(binary_data)
    binary_data.extend(idx_data)
    _pad_to_4(binary_data)

    idx_bv_idx = len(gltf.bufferViews)
    gltf.bufferViews.append(
        pygltflib.BufferView(
            buffer=0,
            byteOffset=idx_offset,
            byteLength=len(idx_data),
            target=pygltflib.ELEMENT_ARRAY_BUFFER,
        )
    )
    idx_acc_idx = len(gltf.accessors)
    gltf.accessors.append(
        pygltflib.Accessor(
            bufferView=idx_bv_idx,
            componentType=pygltflib.UNSIGNED_SHORT,
            count=6,
            type=pygltflib.SCALAR,
            max=[3],
            min=[0],
        )
    )

    # Mesh primitive
    gltf.meshes.append(
        pygltflib.Mesh(
            name="image_plane",
            primitives=[
                pygltflib.Primitive(
                    attributes=pygltflib.Attributes(
                        POSITION=pos_acc_idx,
                        NORMAL=norm_acc_idx,
                        TEXCOORD_0=tc_acc_idx,
                    ),
                    indices=idx_acc_idx,
                    material=0,
                )
            ],
        )
    )

    # Buffer
    gltf.buffers.append(pygltflib.Buffer(byteLength=len(binary_data)))
    gltf.set_binary_blob(bytes(binary_data))

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gltf.save(str(output_path))


def build_animated_textured_plane_glb(
    sequence: TemporalSequence,
    output_path: Path,
    max_grid: int = 128,
    max_frames: int = 40,
    height_scale: float = 0.15,
) -> None:
    """Build an animated GLB from 2D cine using height-map morph targets.

    Each frame's pixel intensities drive vertex Z-displacement on a mesh grid.
    The first frame is the base mesh with its echo image as texture; subsequent
    frames become morph targets that animate the surface shape.
    """
    from scipy.ndimage import zoom

    n_frames = sequence.frame_count
    # Subsample frames if too many
    if n_frames > max_frames:
        step = n_frames / max_frames
        indices = [int(i * step) for i in range(max_frames)]
    else:
        indices = list(range(n_frames))

    # Get base frame pixel data
    base_pixels = sequence.frames[indices[0]].voxels[0].astype(np.float64)
    rows, cols = base_pixels.shape

    # Downsample grid if image is large
    grid_rows, grid_cols = rows, cols
    if max(rows, cols) > max_grid:
        scale = max_grid / max(rows, cols)
        grid_rows = max(2, int(rows * scale))
        grid_cols = max(2, int(cols * scale))

    # Normalize all frames to 0-1 range (global min/max across all frames)
    all_pixels = []
    for idx in indices:
        px = sequence.frames[idx].voxels[0].astype(np.float64)
        all_pixels.append(px)
    global_min = min(p.min() for p in all_pixels)
    global_max = max(p.max() for p in all_pixels)
    if global_max > global_min:
        all_pixels = [(p - global_min) / (global_max - global_min) for p in all_pixels]
    else:
        all_pixels = [np.zeros_like(p) for p in all_pixels]

    # Resize to grid dimensions
    def resize_frame(frame_data):
        if frame_data.shape == (grid_rows, grid_cols):
            return frame_data
        zoom_y = grid_rows / frame_data.shape[0]
        zoom_x = grid_cols / frame_data.shape[1]
        return zoom(frame_data, (zoom_y, zoom_x), order=1)

    grid_frames = [resize_frame(p) for p in all_pixels]

    # Compute physical dimensions
    vol = sequence.frames[0]
    row_spacing, col_spacing = vol.pixel_spacing
    width = cols * col_spacing / 1000.0  # mm → meters
    height = rows * row_spacing / 1000.0

    # Build grid mesh
    ys = np.linspace(-height / 2, height / 2, grid_rows)
    xs = np.linspace(-width / 2, width / 2, grid_cols)
    xx, yy = np.meshgrid(xs, ys)

    base_z = grid_frames[0] * height_scale
    vertices = np.stack([xx, yy, base_z], axis=-1).reshape(-1, 3).astype(np.float32)

    # UV coordinates
    us = np.linspace(0.0, 1.0, grid_cols)
    vs = np.linspace(1.0, 0.0, grid_rows)  # flip V for image orientation
    uu, vv = np.meshgrid(us, vs)
    texcoords = np.stack([uu, vv], axis=-1).reshape(-1, 2).astype(np.float32)

    # Triangle indices for the grid
    faces = []
    for r in range(grid_rows - 1):
        for c in range(grid_cols - 1):
            i = r * grid_cols + c
            faces.append([i, i + grid_cols, i + 1])
            faces.append([i + 1, i + grid_cols, i + grid_cols + 1])
    faces = np.array(faces, dtype=np.uint32)

    # Build morph targets (Z displacements relative to base)
    morph_targets = []
    for frame_idx in range(1, len(grid_frames)):
        frame_z = grid_frames[frame_idx] * height_scale
        delta = np.zeros((len(vertices), 3), dtype=np.float32)
        delta[:, 2] = (frame_z - grid_frames[0] * height_scale).ravel()
        morph_targets.append(delta)

    # Encode base frame image as PNG texture (full resolution)
    texture_pixels = all_pixels[0]
    png_bytes = _pixel_data_to_png(
        (texture_pixels * 255).astype(np.float64)
    )

    # Frame timing
    frame_time_ms = sequence.temporal_resolution  # ms per frame
    dt = frame_time_ms / 1000.0  # seconds
    n_mt = len(morph_targets)
    n_keyframes = n_mt + 1

    # --- Build glTF ---
    gltf = pygltflib.GLTF2(
        scene=0,
        scenes=[pygltflib.Scene(nodes=[0])],
        nodes=[pygltflib.Node(name="echo_surface", mesh=0)],
        meshes=[],
        accessors=[],
        bufferViews=[],
        buffers=[],
        materials=[],
        textures=[],
        images=[],
        samplers=[],
        animations=[],
    )

    binary_data = bytearray()

    # --- Image texture ---
    img_offset = len(binary_data)
    binary_data.extend(png_bytes)
    _pad_to_4(binary_data)

    img_bv_idx = len(gltf.bufferViews)
    gltf.bufferViews.append(
        pygltflib.BufferView(buffer=0, byteOffset=img_offset, byteLength=len(png_bytes))
    )
    gltf.images.append(pygltflib.Image(bufferView=img_bv_idx, mimeType="image/png"))
    gltf.samplers.append(
        pygltflib.Sampler(
            magFilter=pygltflib.LINEAR,
            minFilter=pygltflib.LINEAR,
            wrapS=pygltflib.CLAMP_TO_EDGE,
            wrapT=pygltflib.CLAMP_TO_EDGE,
        )
    )
    gltf.textures.append(pygltflib.Texture(sampler=0, source=0))
    gltf.materials.append(
        pygltflib.Material(
            name="echo_image",
            pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
                baseColorTexture=pygltflib.TextureInfo(index=0),
                metallicFactor=0.0,
                roughnessFactor=1.0,
            ),
            doubleSided=True,
        )
    )

    # --- Vertex positions ---
    pos_data = vertices.tobytes()
    pos_offset = len(binary_data)
    binary_data.extend(pos_data)
    _pad_to_4(binary_data)

    pos_bv_idx = len(gltf.bufferViews)
    gltf.bufferViews.append(
        pygltflib.BufferView(
            buffer=0, byteOffset=pos_offset, byteLength=len(pos_data),
            target=pygltflib.ARRAY_BUFFER,
        )
    )
    pos_acc_idx = len(gltf.accessors)
    gltf.accessors.append(
        pygltflib.Accessor(
            bufferView=pos_bv_idx, componentType=pygltflib.FLOAT,
            count=len(vertices), type=pygltflib.VEC3,
            max=vertices.max(axis=0).tolist(),
            min=vertices.min(axis=0).tolist(),
        )
    )

    # --- Texcoords ---
    tc_data = texcoords.tobytes()
    tc_offset = len(binary_data)
    binary_data.extend(tc_data)
    _pad_to_4(binary_data)

    tc_bv_idx = len(gltf.bufferViews)
    gltf.bufferViews.append(
        pygltflib.BufferView(
            buffer=0, byteOffset=tc_offset, byteLength=len(tc_data),
            target=pygltflib.ARRAY_BUFFER,
        )
    )
    tc_acc_idx = len(gltf.accessors)
    gltf.accessors.append(
        pygltflib.Accessor(
            bufferView=tc_bv_idx, componentType=pygltflib.FLOAT,
            count=len(texcoords), type=pygltflib.VEC2,
        )
    )

    # --- Face indices ---
    idx_data = faces.tobytes()
    idx_offset = len(binary_data)
    binary_data.extend(idx_data)
    _pad_to_4(binary_data)

    idx_bv_idx = len(gltf.bufferViews)
    gltf.bufferViews.append(
        pygltflib.BufferView(
            buffer=0, byteOffset=idx_offset, byteLength=len(idx_data),
            target=pygltflib.ELEMENT_ARRAY_BUFFER,
        )
    )
    idx_acc_idx = len(gltf.accessors)
    gltf.accessors.append(
        pygltflib.Accessor(
            bufferView=idx_bv_idx, componentType=pygltflib.UNSIGNED_INT,
            count=faces.size, type=pygltflib.SCALAR,
            max=[int(faces.max())], min=[int(faces.min())],
        )
    )

    # --- Morph targets ---
    mt_acc_indices = []
    for mt in morph_targets:
        mt_data = mt.tobytes()
        mt_offset = len(binary_data)
        binary_data.extend(mt_data)
        _pad_to_4(binary_data)

        mt_bv_idx = len(gltf.bufferViews)
        gltf.bufferViews.append(
            pygltflib.BufferView(buffer=0, byteOffset=mt_offset, byteLength=len(mt_data))
        )
        mt_acc_idx = len(gltf.accessors)
        gltf.accessors.append(
            pygltflib.Accessor(
                bufferView=mt_bv_idx, componentType=pygltflib.FLOAT,
                count=len(mt), type=pygltflib.VEC3,
                max=mt.max(axis=0).tolist(),
                min=mt.min(axis=0).tolist(),
            )
        )
        mt_acc_indices.append(mt_acc_idx)

    targets = [pygltflib.Attributes(POSITION=acc) for acc in mt_acc_indices]

    # --- Mesh ---
    gltf.meshes.append(
        pygltflib.Mesh(
            name="echo_surface",
            primitives=[
                pygltflib.Primitive(
                    attributes=pygltflib.Attributes(
                        POSITION=pos_acc_idx,
                        TEXCOORD_0=tc_acc_idx,
                    ),
                    indices=idx_acc_idx,
                    material=0,
                    targets=targets if targets else None,
                )
            ],
            weights=[0.0] * n_mt if n_mt else None,
        )
    )

    # --- Animation ---
    if n_mt > 0:
        keyframe_times = np.array([i * dt for i in range(n_keyframes)], dtype=np.float32)

        time_data = keyframe_times.tobytes()
        time_offset = len(binary_data)
        binary_data.extend(time_data)
        _pad_to_4(binary_data)

        time_bv_idx = len(gltf.bufferViews)
        gltf.bufferViews.append(
            pygltflib.BufferView(buffer=0, byteOffset=time_offset, byteLength=len(time_data))
        )
        time_acc_idx = len(gltf.accessors)
        gltf.accessors.append(
            pygltflib.Accessor(
                bufferView=time_bv_idx, componentType=pygltflib.FLOAT,
                count=len(keyframe_times), type=pygltflib.SCALAR,
                max=[float(keyframe_times.max())],
                min=[float(keyframe_times.min())],
            )
        )

        # Morph weights: at each keyframe, activate one target
        weights = np.zeros((n_keyframes, n_mt), dtype=np.float32)
        for i in range(n_mt):
            weights[i + 1, i] = 1.0

        weight_data = weights.tobytes()
        weight_offset = len(binary_data)
        binary_data.extend(weight_data)
        _pad_to_4(binary_data)

        weight_bv_idx = len(gltf.bufferViews)
        gltf.bufferViews.append(
            pygltflib.BufferView(buffer=0, byteOffset=weight_offset, byteLength=len(weight_data))
        )
        weight_acc_idx = len(gltf.accessors)
        gltf.accessors.append(
            pygltflib.Accessor(
                bufferView=weight_bv_idx, componentType=pygltflib.FLOAT,
                count=n_keyframes * n_mt, type=pygltflib.SCALAR,
            )
        )

        gltf.animations.append(
            pygltflib.Animation(
                name="cardiac_cycle",
                channels=[
                    pygltflib.AnimationChannel(
                        sampler=0,
                        target=pygltflib.AnimationChannelTarget(node=0, path="weights"),
                    )
                ],
                samplers=[
                    pygltflib.AnimationSampler(
                        input=time_acc_idx,
                        output=weight_acc_idx,
                        interpolation=pygltflib.ANIM_STEP,
                    )
                ],
            )
        )

    # --- Finalize ---
    gltf.buffers.append(pygltflib.Buffer(byteLength=len(binary_data)))
    gltf.set_binary_blob(bytes(binary_data))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gltf.save(str(output_path))


def _pixel_data_to_png(pixel_data: np.ndarray) -> bytes:
    """Convert 2D pixel array to PNG bytes."""
    from PIL import Image

    # Normalize to 0-255 range
    data = pixel_data.astype(np.float64)
    dmin, dmax = data.min(), data.max()
    if dmax > dmin:
        data = ((data - dmin) / (dmax - dmin) * 255).astype(np.uint8)
    else:
        data = np.zeros_like(data, dtype=np.uint8)

    img = Image.fromarray(data, mode="L")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
