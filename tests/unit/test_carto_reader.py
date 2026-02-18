"""Unit tests for CARTO .mesh and _car.txt parsers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from med2glb.io.carto_reader import (
    detect_carto_directory,
    load_carto_study,
    parse_car_file,
    parse_mesh_file,
)

# Real CARTO data (skipped if not present)
CARTO_DATA = Path(__file__).parent.parent.parent / "CARTO_Example_Data"
CARTO_OLD = CARTO_DATA / "older CARTO versions" / "Study 1" / "Export_Study-1-01_16_2015-15-31-41"
CARTO_V71 = CARTO_DATA / "Version_7.1.80.33" / "Study 1" / "Export_Study"
CARTO_V72 = CARTO_DATA / "Version_7.2.10.423" / "Export_Study-1-01_09_2023-20-30-09"


class TestDetectCartoDirectory:
    def test_detects_synthetic(self, carto_mesh_dir):
        assert detect_carto_directory(carto_mesh_dir) is True

    def test_rejects_empty_dir(self, tmp_path):
        assert detect_carto_directory(tmp_path) is False

    def test_rejects_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello")
        assert detect_carto_directory(f) is False

    @pytest.mark.skipif(not CARTO_OLD.exists(), reason="CARTO old data not available")
    def test_detects_old_version(self):
        assert detect_carto_directory(CARTO_OLD) is True

    @pytest.mark.skipif(not CARTO_V72.exists(), reason="CARTO v7.2 data not available")
    def test_detects_v72(self):
        assert detect_carto_directory(CARTO_V72) is True


class TestParseMeshFile:
    def test_synthetic_mesh(self, carto_mesh_dir):
        mesh_file = carto_mesh_dir / "1-TestMap.mesh"
        mesh = parse_mesh_file(mesh_file)

        assert mesh.mesh_id == 1
        assert len(mesh.vertices) == 4
        assert len(mesh.faces) == 2
        assert len(mesh.normals) == 4
        assert len(mesh.group_ids) == 4
        assert mesh.structure_name == "1-TestMap"
        assert "LAT" in mesh.color_names

    def test_inactive_vertex(self, carto_mesh_dir):
        mesh_file = carto_mesh_dir / "1-TestMap.mesh"
        mesh = parse_mesh_file(mesh_file)

        # Vertex 3 has GroupID -1000000
        assert mesh.group_ids[3] == -1000000
        # Face 1 has GroupID -1000000
        assert mesh.face_group_ids[1] == -1000000

    def test_mesh_color(self, carto_mesh_dir):
        mesh = parse_mesh_file(carto_mesh_dir / "1-TestMap.mesh")
        assert mesh.mesh_color == pytest.approx((0.0, 1.0, 0.0, 1.0))

    @pytest.mark.skipif(not CARTO_OLD.exists(), reason="CARTO old data not available")
    def test_old_version_mesh(self):
        mesh = parse_mesh_file(CARTO_OLD / "1-Map.mesh")
        assert mesh.mesh_id == 231
        assert len(mesh.vertices) == 810
        assert len(mesh.faces) == 1616
        assert "LAT" in mesh.color_names

    @pytest.mark.skipif(not CARTO_V71.exists(), reason="CARTO v7.1 data not available")
    def test_v71_mesh(self):
        mesh = parse_mesh_file(CARTO_V71 / "1-Map.mesh")
        assert mesh.mesh_id == 65
        assert len(mesh.vertices) == 8913
        assert len(mesh.faces) == 17968
        # v7.1 uses TransparentGroupsIDs (e.g. 2) rather than -1000000 for vertices.
        # Inactive faces do use -1000000.
        inactive_faces = np.sum(mesh.face_group_ids == -1000000)
        assert inactive_faces > 0

    @pytest.mark.skipif(not CARTO_V72.exists(), reason="CARTO v7.2 data not available")
    def test_v72_mesh(self):
        mesh = parse_mesh_file(CARTO_V72 / "1-LA.mesh")
        assert mesh.mesh_id == 183
        assert len(mesh.vertices) == 60212
        assert "LAT" in mesh.color_names


class TestParseCarFile:
    def test_synthetic_car(self, carto_mesh_dir):
        version, points = parse_car_file(carto_mesh_dir / "1-TestMap_car.txt")

        assert version == "6.0"
        assert len(points) == 3
        assert points[0].point_id == 1
        assert points[0].bipolar_voltage == pytest.approx(2.5)
        assert points[0].unipolar_voltage == pytest.approx(8.0)
        assert points[0].lat == pytest.approx(-50.0)

    def test_lat_sentinel_becomes_nan(self, carto_mesh_dir):
        _, points = parse_car_file(carto_mesh_dir / "1-TestMap_car.txt")
        # Third point has LAT -10000 (sentinel)
        assert np.isnan(points[2].lat)

    def test_point_positions(self, carto_mesh_dir):
        _, points = parse_car_file(carto_mesh_dir / "1-TestMap_car.txt")
        np.testing.assert_array_almost_equal(points[0].position, [1.0, 1.0, 0.0])
        np.testing.assert_array_almost_equal(points[1].position, [9.0, 1.0, 0.0])

    @pytest.mark.skipif(not CARTO_OLD.exists(), reason="CARTO old data not available")
    def test_v4_car(self):
        version, points = parse_car_file(CARTO_OLD / "1-Map_car.txt")
        assert version == "4.0"
        assert len(points) > 0
        # Check that positions are reasonable
        for p in points[:5]:
            assert not np.any(np.isnan(p.position))

    @pytest.mark.skipif(not CARTO_V71.exists(), reason="CARTO v7.1 data not available")
    def test_v5_car(self):
        version, points = parse_car_file(CARTO_V71 / "1-Map_car.txt")
        assert version == "5.0"
        assert len(points) > 0

    @pytest.mark.skipif(not CARTO_V72.exists(), reason="CARTO v7.2 data not available")
    def test_v6_car(self):
        version, points = parse_car_file(CARTO_V72 / "1-LA_car.txt")
        assert version == "6.0"
        assert len(points) > 0


class TestLoadCartoStudy:
    def test_synthetic_study(self, carto_mesh_dir):
        study = load_carto_study(carto_mesh_dir)

        assert len(study.meshes) == 1
        assert study.version == "6.0"
        assert "1-TestMap" in study.points
        assert len(study.points["1-TestMap"]) == 3

    @pytest.mark.skipif(not CARTO_V72.exists(), reason="CARTO v7.2 data not available")
    def test_v72_study(self):
        study = load_carto_study(CARTO_V72)
        assert len(study.meshes) >= 2  # LA and RA at minimum
        assert study.version == "6.0"
