"""Unit tests for CARTO point-to-vertex mapping and colormaps."""

from __future__ import annotations

import numpy as np
import pytest

from med2glb.io.carto_colormaps import bipolar_colormap, lat_colormap, unipolar_colormap
from med2glb.io.carto_mapper import (
    build_inactive_mask,
    carto_mesh_to_mesh_data,
    interpolate_sparse_values,
    map_points_to_vertices,
)


class TestMapPointsToVertices:
    def test_basic_mapping(self, synthetic_carto_mesh, synthetic_carto_points):
        values = map_points_to_vertices(
            synthetic_carto_mesh, synthetic_carto_points, field="lat"
        )
        assert len(values) == len(synthetic_carto_mesh.vertices)
        # At least some values should be non-NaN (nearest-neighbor assigns to all)
        assert not np.all(np.isnan(values))

    def test_bipolar_field(self, synthetic_carto_mesh, synthetic_carto_points):
        values = map_points_to_vertices(
            synthetic_carto_mesh, synthetic_carto_points, field="bipolar"
        )
        assert not np.all(np.isnan(values))
        # Bipolar voltages should be positive
        assert np.all(values[~np.isnan(values)] >= 0)

    def test_unipolar_field(self, synthetic_carto_mesh, synthetic_carto_points):
        values = map_points_to_vertices(
            synthetic_carto_mesh, synthetic_carto_points, field="unipolar"
        )
        assert not np.all(np.isnan(values))

    def test_empty_points(self, synthetic_carto_mesh):
        values = map_points_to_vertices(synthetic_carto_mesh, [], field="lat")
        assert np.all(np.isnan(values))

    def test_invalid_field(self, synthetic_carto_mesh, synthetic_carto_points):
        with pytest.raises(ValueError, match="Unknown field"):
            map_points_to_vertices(
                synthetic_carto_mesh, synthetic_carto_points, field="invalid"
            )


class TestInterpolateSparseValues:
    def test_fills_gaps(self, synthetic_carto_mesh):
        values = np.full(len(synthetic_carto_mesh.vertices), np.nan)
        # Set a few known values
        values[0] = 10.0
        values[5] = 20.0
        values[10] = 30.0

        result = interpolate_sparse_values(synthetic_carto_mesh, values)
        # Original values should be preserved
        assert result[0] == pytest.approx(10.0)
        assert result[5] == pytest.approx(20.0)
        assert result[10] == pytest.approx(30.0)

    def test_all_valid_unchanged(self, synthetic_carto_mesh):
        values = np.arange(len(synthetic_carto_mesh.vertices), dtype=np.float64)
        result = interpolate_sparse_values(synthetic_carto_mesh, values)
        np.testing.assert_array_equal(result, values)


class TestBuildInactiveMask:
    def test_all_active(self, synthetic_carto_mesh):
        mask = build_inactive_mask(synthetic_carto_mesh)
        assert not np.any(mask)

    def test_with_inactive(self, synthetic_carto_mesh):
        synthetic_carto_mesh.group_ids[0] = -1000000
        synthetic_carto_mesh.group_ids[3] = -1000000
        mask = build_inactive_mask(synthetic_carto_mesh)
        assert mask[0] is np.True_
        assert mask[3] is np.True_
        assert np.sum(mask) == 2


class TestCartoMeshToMeshData:
    def test_basic_conversion(self, synthetic_carto_mesh, synthetic_carto_points):
        mesh_data = carto_mesh_to_mesh_data(
            synthetic_carto_mesh, synthetic_carto_points, coloring="lat"
        )
        assert mesh_data.vertices is not None
        assert mesh_data.faces is not None
        assert mesh_data.vertex_colors is not None
        assert mesh_data.vertex_colors.shape[1] == 4  # RGBA

    def test_filters_inactive(self, carto_mesh_dir):
        from med2glb.io.carto_reader import parse_mesh_file, parse_car_file

        mesh = parse_mesh_file(carto_mesh_dir / "1-TestMap.mesh")
        _, points = parse_car_file(carto_mesh_dir / "1-TestMap_car.txt")

        mesh_data = carto_mesh_to_mesh_data(mesh, points, coloring="lat")
        # Original has 4 verts (1 inactive), so should have 3 active
        assert len(mesh_data.vertices) == 3
        # Original has 2 faces (1 inactive), so should have 1
        assert len(mesh_data.faces) == 1

    def test_no_points_fallback(self, synthetic_carto_mesh):
        mesh_data = carto_mesh_to_mesh_data(
            synthetic_carto_mesh, None, coloring="lat"
        )
        assert mesh_data.vertex_colors is not None
        # Should be mesh default color (green)
        assert mesh_data.vertex_colors[0, 1] == pytest.approx(1.0)  # green channel

    def test_vertex_colors_are_float32(self, synthetic_carto_mesh, synthetic_carto_points):
        mesh_data = carto_mesh_to_mesh_data(
            synthetic_carto_mesh, synthetic_carto_points
        )
        assert mesh_data.vertex_colors.dtype == np.float32


class TestColormaps:
    def test_lat_colormap_range(self):
        values = np.array([-200, -100, 0, 50, 100], dtype=np.float64)
        colors = lat_colormap(values)
        assert colors.shape == (5, 4)
        assert np.all(colors[:, :3] >= 0)
        assert np.all(colors[:, :3] <= 1)
        assert np.all(colors[:, 3] == 1.0)

    def test_lat_nan_transparent(self):
        values = np.array([0, np.nan, 100], dtype=np.float64)
        colors = lat_colormap(values)
        assert colors[1, 3] < 1.0  # NaN vertex is semi-transparent

    def test_bipolar_default_range(self):
        values = np.array([0.05, 0.5, 1.0, 1.5], dtype=np.float64)
        colors = bipolar_colormap(values)
        # Low voltage (scar) should be red
        assert colors[0, 0] > 0.8  # high red
        assert colors.shape == (4, 4)

    def test_unipolar_colormap(self):
        values = np.array([3.0, 6.5, 10.0], dtype=np.float64)
        colors = unipolar_colormap(values)
        assert colors.shape == (3, 4)
        assert np.all(colors[:, 3] == 1.0)

    def test_all_same_value(self):
        values = np.array([5.0, 5.0, 5.0], dtype=np.float64)
        colors = lat_colormap(values)
        # Should not crash — all get mid-range color
        assert colors.shape == (3, 4)

    def test_custom_clamp_range(self):
        values = np.array([-300, -100, 100, 300], dtype=np.float64)
        colors = lat_colormap(values, clamp_range=(-200, 200))
        # Values outside range should be clamped (first red, last purple)
        assert colors[0, 0] > 0.8  # red
