"""Shared test fixtures: synthetic DICOM and CARTO data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pydicom
import pytest
from pydicom.dataset import FileDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from med2glb.core.types import CartoMesh, CartoPoint, CartoStudy, MaterialConfig, MeshData
from med2glb.core.volume import DicomVolume, TemporalSequence

# Path to real CARTO example data (may not exist on CI)
CARTO_EXAMPLE_DATA = Path(__file__).parent.parent / "CARTO_Example_Data"


@pytest.fixture
def synthetic_volume() -> DicomVolume:
    """Create a synthetic 3D volume with a sphere."""
    shape = (30, 30, 30)
    voxels = np.zeros(shape, dtype=np.float32)

    center = np.array([15, 15, 15])
    zz, yy, xx = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    dist = np.sqrt((zz - center[0])**2 + (yy - center[1])**2 + (xx - center[2])**2)
    voxels[dist < 10] = 1000.0

    return DicomVolume(
        voxels=voxels,
        pixel_spacing=(1.0, 1.0),
        slice_thickness=1.0,
        series_uid="1.2.3.4.5",
        modality="CT",
    )


@pytest.fixture
def synthetic_temporal_sequence() -> TemporalSequence:
    """Create a synthetic temporal sequence (3 frames with varying sphere size)."""
    frames = []
    for i in range(3):
        shape = (30, 30, 30)
        voxels = np.zeros(shape, dtype=np.float32)
        radius = 8 + i * 2
        center = np.array([15, 15, 15])
        zz, yy, xx = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
        dist = np.sqrt((zz - center[0])**2 + (yy - center[1])**2 + (xx - center[2])**2)
        voxels[dist < radius] = 1000.0

        frames.append(
            DicomVolume(
                voxels=voxels,
                pixel_spacing=(1.0, 1.0),
                slice_thickness=1.0,
                series_uid="1.2.3.4.5",
                modality="US",
            )
        )

    return TemporalSequence(frames=frames, temporal_resolution=33.3)


@pytest.fixture
def synthetic_mesh() -> MeshData:
    """Create a simple triangulated cube mesh."""
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ], dtype=np.float32)

    faces = np.array([
        [0, 1, 2], [0, 2, 3],
        [4, 6, 5], [4, 7, 6],
        [0, 4, 5], [0, 5, 1],
        [2, 6, 7], [2, 7, 3],
        [0, 3, 7], [0, 7, 4],
        [1, 5, 6], [1, 6, 2],
    ], dtype=np.int32)

    return MeshData(
        vertices=vertices,
        faces=faces,
        structure_name="test_cube",
        material=MaterialConfig(base_color=(0.8, 0.2, 0.2), alpha=1.0),
    )


@pytest.fixture
def dicom_directory(tmp_path) -> Path:
    """Create a temporary directory with synthetic DICOM files (3D volume)."""
    series_uid = generate_uid()

    for i in range(10):
        _write_synthetic_dicom(
            tmp_path / f"slice_{i:03d}.dcm",
            series_uid=series_uid,
            instance_number=i + 1,
            slice_location=float(i),
        )

    return tmp_path


@pytest.fixture
def dicom_temporal_directory(tmp_path) -> Path:
    """Create a temporary directory with synthetic temporal DICOM files."""
    series_uid = generate_uid()

    for frame in range(3):
        for slice_idx in range(10):
            _write_synthetic_dicom(
                tmp_path / f"frame_{frame:02d}_slice_{slice_idx:03d}.dcm",
                series_uid=series_uid,
                instance_number=frame * 10 + slice_idx + 1,
                slice_location=float(slice_idx),
                temporal_position=frame + 1,
            )

    return tmp_path


@pytest.fixture
def dicom_mixed_shapes_directory(tmp_path) -> Path:
    """Create a directory with DICOM files of mixed dimensions (simulates scouts + volume)."""
    series_uid = generate_uid()

    # 8 normal slices at 32x32
    for i in range(8):
        _write_synthetic_dicom(
            tmp_path / f"slice_{i:03d}.dcm",
            series_uid=series_uid,
            instance_number=i + 1,
            slice_location=float(i),
            rows=32,
            cols=32,
        )

    # 2 scout/localizer slices at 64x64 (same series)
    for i in range(2):
        _write_synthetic_dicom(
            tmp_path / f"scout_{i:03d}.dcm",
            series_uid=series_uid,
            instance_number=100 + i,
            slice_location=float(100 + i),
            rows=64,
            cols=64,
        )

    return tmp_path


@pytest.fixture
def dicom_mixed_channels_directory(tmp_path) -> Path:
    """Create a directory with mixed grayscale and RGB DICOM files."""
    series_uid = generate_uid()

    # 6 grayscale slices
    for i in range(6):
        _write_synthetic_dicom(
            tmp_path / f"gray_{i:03d}.dcm",
            series_uid=series_uid,
            instance_number=i + 1,
            slice_location=float(i),
            rows=32,
            cols=32,
        )

    # 3 RGB slices (same series, same Rows/Cols)
    for i in range(3):
        _write_synthetic_dicom(
            tmp_path / f"rgb_{i:03d}.dcm",
            series_uid=series_uid,
            instance_number=50 + i,
            slice_location=float(50 + i),
            rows=32,
            cols=32,
            rgb=True,
        )

    return tmp_path


@pytest.fixture
def dicom_multi_series_directory(tmp_path) -> Path:
    """Create a directory with multiple DICOM series."""
    series_a = generate_uid()
    series_b = generate_uid()

    # Series A: 8 slices at 32x32
    for i in range(8):
        _write_synthetic_dicom(
            tmp_path / f"a_slice_{i:03d}.dcm",
            series_uid=series_a,
            instance_number=i + 1,
            slice_location=float(i),
        )

    # Series B: 4 slices at 48x48
    for i in range(4):
        _write_synthetic_dicom(
            tmp_path / f"b_slice_{i:03d}.dcm",
            series_uid=series_b,
            instance_number=i + 1,
            slice_location=float(i),
            rows=48,
            cols=48,
        )

    return tmp_path


@pytest.fixture
def dicom_multiframe_directory(tmp_path) -> Path:
    """Create a directory with a multi-frame DICOM file (e.g. ultrasound cine clip)."""
    series_uid = generate_uid()

    # Write a multi-frame DICOM file
    _write_synthetic_multiframe_dicom(
        tmp_path / "cine.dcm",
        series_uid=series_uid,
        number_of_frames=20,
        rows=64,
        cols=64,
        modality="US",
    )

    return tmp_path


@pytest.fixture
def dicom_multi_series_with_multiframe_directory(tmp_path) -> Path:
    """Create a directory with mixed series: a 3D volume and a multi-frame cine."""
    volume_uid = generate_uid()
    cine_uid = generate_uid()

    # Series A: 8-slice volume (CT)
    for i in range(8):
        _write_synthetic_dicom(
            tmp_path / f"vol_slice_{i:03d}.dcm",
            series_uid=volume_uid,
            instance_number=i + 1,
            slice_location=float(i),
        )

    # Series B: multi-frame cine (US)
    _write_synthetic_multiframe_dicom(
        tmp_path / "cine.dcm",
        series_uid=cine_uid,
        number_of_frames=30,
        rows=64,
        cols=64,
        modality="US",
    )

    return tmp_path


@pytest.fixture
def dicom_gallery_mixed_directory(tmp_path) -> Path:
    """Create a directory with mixed-dimension DICOM files for gallery mode testing.

    4 slices at 32x32 + 2 at 64x64, all in the same series.
    """
    series_uid = generate_uid()

    # 4 slices at 32x32
    for i in range(4):
        _write_synthetic_dicom(
            tmp_path / f"small_{i:03d}.dcm",
            series_uid=series_uid,
            instance_number=i + 1,
            slice_location=float(i),
            rows=32,
            cols=32,
        )

    # 2 slices at 64x64
    for i in range(2):
        _write_synthetic_dicom(
            tmp_path / f"large_{i:03d}.dcm",
            series_uid=series_uid,
            instance_number=10 + i,
            slice_location=float(10 + i),
            rows=64,
            cols=64,
        )

    return tmp_path


@pytest.fixture
def dicom_temporal_gallery_directory(tmp_path) -> Path:
    """Create a directory with temporal DICOM files for gallery animation testing.

    3 spatial positions x 2 temporal frames = 6 files.
    """
    series_uid = generate_uid()

    for pos_idx in range(3):
        for t_idx in range(2):
            _write_synthetic_dicom(
                tmp_path / f"pos{pos_idx}_t{t_idx}.dcm",
                series_uid=series_uid,
                instance_number=pos_idx * 10 + t_idx + 1,
                slice_location=float(pos_idx),
                temporal_position=t_idx + 1,
                rows=32,
                cols=32,
            )

    return tmp_path


def _write_synthetic_multiframe_dicom(
    path: Path,
    series_uid: str,
    number_of_frames: int = 20,
    rows: int = 64,
    cols: int = 64,
    modality: str = "US",
) -> None:
    """Write a synthetic multi-frame DICOM file."""
    file_meta = pydicom.Dataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.6.1"  # US Image
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\x00" * 128)

    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.6.1"
    ds.SOPInstanceUID = generate_uid()
    ds.SeriesInstanceUID = series_uid
    ds.StudyInstanceUID = generate_uid()
    ds.Modality = modality
    ds.InstanceNumber = 1
    ds.Rows = rows
    ds.Columns = cols
    ds.NumberOfFrames = number_of_frames
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.FrameTime = 33.3
    ds.PixelSpacing = [0.3, 0.3]

    # Generate pixel data for all frames
    pixel_data = np.zeros((number_of_frames, rows, cols), dtype=np.uint16)
    center = (rows // 2, cols // 2)
    yy, xx = np.mgrid[0:rows, 0:cols]
    for f in range(number_of_frames):
        radius = (min(rows, cols) // 4) + f % 5
        dist = (yy - center[0]) ** 2 + (xx - center[1]) ** 2
        pixel_data[f][dist < radius**2] = 500

    ds.PixelData = pixel_data.tobytes()
    ds.save_as(str(path))


def _write_synthetic_dicom(
    path: Path,
    series_uid: str,
    instance_number: int,
    slice_location: float,
    rows: int = 32,
    cols: int = 32,
    temporal_position: int | None = None,
    rgb: bool = False,
) -> None:
    """Write a single synthetic DICOM file."""
    file_meta = pydicom.Dataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\x00" * 128)

    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    ds.SOPInstanceUID = generate_uid()
    ds.SeriesInstanceUID = series_uid
    ds.StudyInstanceUID = generate_uid()
    ds.Modality = "CT"
    ds.InstanceNumber = instance_number
    ds.ImagePositionPatient = [0.0, 0.0, slice_location]
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.PixelSpacing = [1.0, 1.0]
    ds.SliceThickness = 1.0
    ds.SliceLocation = slice_location
    ds.Rows = rows
    ds.Columns = cols
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    if rgb:
        ds.SamplesPerPixel = 3
        ds.PhotometricInterpretation = "RGB"
        ds.PlanarConfiguration = 0
    else:
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
    ds.RescaleSlope = 1.0
    ds.RescaleIntercept = 0.0

    if temporal_position is not None:
        ds.TemporalPositionIdentifier = temporal_position

    # Generate pixel data with a sphere pattern
    if rgb:
        pixel_data = np.zeros((rows, cols, 3), dtype=np.uint16)
        center = (rows // 2, cols // 2)
        radius = min(rows, cols) // 4
        yy, xx = np.mgrid[0:rows, 0:cols]
        dist = (yy - center[0])**2 + (xx - center[1])**2
        pixel_data[dist < radius**2] = [500, 250, 100]
    else:
        pixel_data = np.zeros((rows, cols), dtype=np.uint16)
        center = (rows // 2, cols // 2)
        radius = min(rows, cols) // 4
        yy, xx = np.mgrid[0:rows, 0:cols]
        dist = (yy - center[0])**2 + (xx - center[1])**2
        pixel_data[dist < radius**2] = 500

    ds.PixelData = pixel_data.tobytes()
    ds.save_as(str(path))


# ---- CARTO Fixtures ----


@pytest.fixture
def synthetic_carto_mesh() -> CartoMesh:
    """Create a synthetic CARTO mesh (small icosphere-like shape)."""
    vertices = np.array([
        [0, 0, 1], [1, 0, 0], [0, 1, 0], [-1, 0, 0],
        [0, -1, 0], [0, 0, -1], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5],
        [0.5, -0.5, 0.5], [-0.5, -0.5, 0.5], [0.5, 0.5, -0.5],
        [-0.5, 0.5, -0.5], [0.5, -0.5, -0.5], [-0.5, -0.5, -0.5],
    ], dtype=np.float64) * 30

    faces = np.array([
        [0, 6, 7], [0, 8, 6], [0, 7, 9], [0, 9, 8],
        [5, 10, 11], [5, 12, 10], [5, 11, 13], [5, 13, 12],
        [6, 1, 10], [7, 11, 3], [8, 4, 12], [9, 13, 4],
    ], dtype=np.int32)

    normals = np.zeros_like(vertices)
    norms = np.linalg.norm(vertices, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normals = vertices / norms

    group_ids = np.zeros(len(vertices), dtype=np.int32)
    face_group_ids = np.zeros(len(faces), dtype=np.int32)

    return CartoMesh(
        mesh_id=1,
        vertices=vertices,
        faces=faces,
        normals=normals,
        group_ids=group_ids,
        face_group_ids=face_group_ids,
        mesh_color=(0.0, 1.0, 0.0, 1.0),
        color_names=["Unipolar", "Bipolar", "LAT"],
        structure_name="test_mesh",
    )


@pytest.fixture
def synthetic_carto_points() -> list[CartoPoint]:
    """Create synthetic CARTO measurement points near the mesh vertices."""
    points = []
    rng = np.random.RandomState(42)
    for i in range(20):
        pos = rng.randn(3) * 20
        points.append(CartoPoint(
            point_id=i,
            position=pos,
            orientation=rng.randn(3),
            bipolar_voltage=rng.uniform(0.1, 5.0),
            unipolar_voltage=rng.uniform(1.0, 15.0),
            lat=rng.uniform(-200, 100),
        ))
    return points


@pytest.fixture
def synthetic_carto_study(
    synthetic_carto_mesh, synthetic_carto_points,
) -> CartoStudy:
    """Create a synthetic CARTO study with one mesh and points."""
    return CartoStudy(
        meshes=[synthetic_carto_mesh],
        points={"test_mesh": synthetic_carto_points},
        version="6.0",
        study_name="test_study",
    )


@pytest.fixture
def carto_mesh_dir(tmp_path) -> Path:
    """Create a temporary directory with synthetic CARTO .mesh and _car.txt files."""
    mesh_text = """\
#TriangulatedMeshVersion2.0
; Synthetic test mesh

[GeneralAttributes]
MeshID                 = 1
MeshName               =
NumVertex              = 4
NumTriangle            = 2
TopologyStatus         = 0
MeshColor              = 0.00000   1.00000   0.00000   1.00000
Matrix                 = 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1
NumVertexColors        = 3
ColorsIDs              = 0  1  2
ColorsNames            = Unipolar  Bipolar  LAT

[VerticesSection]
;                   X             Y             Z        NormalX   NormalY   NormalZ  GroupID

       0 =         0.000         0.000         0.000     0.00000   0.00000   1.00000        0
       1 =        10.000         0.000         0.000     0.00000   0.00000   1.00000        0
       2 =         5.000        10.000         0.000     0.00000   0.00000   1.00000        0
       3 =         5.000         5.000         5.000     0.00000   0.00000   1.00000  -1000000

[TrianglesSection]
;           Vertex0  Vertex1  Vertex2     NormalX   NormalY   NormalZ  GroupID

       0 =        0        1        2     0.00000   0.00000   1.00000        0
       1 =        0        1        3     0.00000   0.00000   1.00000  -1000000
"""
    (tmp_path / "1-TestMap.mesh").write_text(mesh_text, encoding="utf-8")

    car_text = """\
VERSION_6_0 1-TestMap
P\t0\t1\t0\t1.0\t1.0\t0.0\t0.1\t0.2\t0.3\t2.5\t8.0\t-50\t-10000\t0\t4\t1\tA\t0\t100\t0
P\t0\t2\t0\t9.0\t1.0\t0.0\t0.1\t0.2\t0.3\t1.0\t5.0\t100\t-10000\t0\t4\t1\tA\t0\t100\t0
P\t0\t3\t0\t5.0\t9.0\t0.0\t0.1\t0.2\t0.3\t3.0\t12.0\t-10000\t-10000\t0\t4\t1\tA\t0\t100\t0
"""
    (tmp_path / "1-TestMap_car.txt").write_text(car_text, encoding="utf-8")

    return tmp_path
