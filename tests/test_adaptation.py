import pytest

from meshcfd.core.adaptation import adapt_mesh, cell_areas
from meshcfd.io.mesh_io import mesh_from_dict, mesh_to_dict


def two_cell_mesh():
    return mesh_from_dict(
        {
            "format": "meshcad-mesh",
            "version": 1,
            "dimension": 2,
            "nodes": {
                "coordinates": [
                    [0, 0],
                    [1, 0],
                    [2, 0],
                    [0, 1],
                    [1, 1],
                    [2, 1],
                ]
            },
            "faces": {
                "nodes": [
                    [0, 1],
                    [1, 4],
                    [4, 3],
                    [3, 0],
                    [1, 2],
                    [2, 5],
                    [5, 4],
                ],
                "owner": [0, 0, 0, 0, 1, 1, 1],
                "patch": [0, -1, 2, 3, 0, 1, 2],
                "neighbour": [-1, 1, -1, -1, -1, -1, -1],
            },
            "cells": {
                "faces": [[0, 1, 2, 3], [4, 5, 6, 1]],
                "face_signs": [[1, 1, 1, 1], [1, 1, 1, -1]],
            },
            "patches": [
                {"id": 0, "name": "bottom", "faces": [0, 4]},
                {"id": 1, "name": "right", "faces": [5]},
                {"id": 2, "name": "top", "faces": [2, 6]},
                {"id": 3, "name": "left", "faces": [3]},
            ],
        }
    )


def test_refinement_creates_hanging_node_subfaces_and_preserves_area():
    original = two_cell_mesh()

    refined, source_groups = adapt_mesh(original, {0: "xy"}, set())

    assert len(refined.cells) == 5
    assert len(refined.cells[4]) == 5
    assert sum(cell_areas(refined)) == pytest.approx(sum(cell_areas(original)))
    assert source_groups == ((0,), (0,), (0,), (0,), (1,))
    assert sum(neighbour >= 0 for neighbour in refined.face_neighbours) == 6
    assert mesh_from_dict(mesh_to_dict(refined)) == refined


@pytest.mark.parametrize(
    ("axis", "varying_coordinate"),
    (("x", 0), ("y", 1)),
)
def test_directional_refinement_splits_along_marked_axis(axis, varying_coordinate):
    original = two_cell_mesh()

    refined, source_groups = adapt_mesh(original, {0: axis}, set())

    children = [
        cell
        for cell, sources in zip(refined.cells, source_groups)
        if sources == (0,)
    ]
    assert len(children) == 2
    centers = [
        tuple(
            sum(refined.points[node][coordinate] for node in cell) / len(cell)
            for coordinate in range(2)
        )
        for cell in children
    ]
    other_coordinate = 1 - varying_coordinate
    assert abs(
        centers[0][varying_coordinate] - centers[1][varying_coordinate]
    ) == pytest.approx(0.5)
    assert centers[0][other_coordinate] == pytest.approx(
        centers[1][other_coordinate]
    )


def test_refined_siblings_can_be_coarsened_back_to_parent():
    original = two_cell_mesh()
    refined, _ = adapt_mesh(original, {0: "x"}, set())

    restored, source_groups = adapt_mesh(
        refined, {}, {"r0"}
    )

    assert len(restored.cells) == len(original.cells)
    assert restored.cell_levels == (0, 0)
    assert restored.refinement_records == ()
    assert source_groups == ((0, 1), (2,))
    assert sum(cell_areas(restored)) == pytest.approx(sum(cell_areas(original)))
    assert mesh_from_dict(mesh_to_dict(restored)) == restored


def test_nested_refinement_can_be_coarsened_to_the_original_cell():
    original = two_cell_mesh()
    refined, _ = adapt_mesh(original, {0: "xy"}, set())
    refined_again, _ = adapt_mesh(refined, {0: "x"}, set())

    partially_coarsened, _ = adapt_mesh(
        refined_again, {}, {"r0.0"}
    )
    restored, _ = adapt_mesh(partially_coarsened, {}, {"r0"})

    assert len(refined_again.cells) == 6
    assert len(partially_coarsened.cells) == 5
    assert len(restored.cells) == len(original.cells)
    assert max(restored.cell_levels) == 0
    assert restored.refinement_records == ()
    assert sum(cell_areas(restored)) == pytest.approx(sum(cell_areas(original)))
