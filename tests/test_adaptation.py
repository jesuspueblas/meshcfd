import pytest

from meshcfd.core.adaptation import (
    _balance_adaptation_plan,
    _cell_directional_levels,
    _split_polygon,
    adapt_mesh,
    cell_areas,
)
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


def test_four_way_refinement_uses_bilinear_center_for_skewed_quad():
    polygon = ((0.0, 0.0), (2.0, 0.0), (1.6, 1.0), (0.2, 1.4))

    children = _split_polygon(polygon, "xy")

    shared_vertices = set(children[0])
    for child in children[1:]:
        shared_vertices.intersection_update(child)
    expected_center = (
        sum(point[0] for point in polygon) / 4.0,
        sum(point[1] for point in polygon) / 4.0,
    )
    assert shared_vertices == {expected_center}


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


def test_refinement_propagates_to_restore_two_to_one_balance():
    original = two_cell_mesh()
    once_refined, _ = adapt_mesh(original, {0: "xy"}, set())

    refined, _ = adapt_mesh(once_refined, {0: "xy"}, set())

    levels = refined.cell_levels
    assert max(levels) == 2
    assert any(level == 1 for level in levels)
    assert all(
        abs(levels[owner] - levels[neighbour]) <= 1
        for owner, neighbour in zip(
            refined.face_owners, refined.face_neighbours
        )
        if neighbour >= 0
    )


def test_refinement_does_not_coarsen_a_cell_into_a_two_to_one_violation():
    original = two_cell_mesh()
    refined, _ = adapt_mesh(original, {0: "xy", 1: "xy"}, set())
    cell_index = {cell_id: index for index, cell_id in enumerate(refined.cell_ids)}
    target = next(
        first
        for face_id, neighbour in enumerate(refined.face_neighbours)
        if neighbour >= 0
        for first, second in (
            (refined.face_owners[face_id], neighbour),
            (neighbour, refined.face_owners[face_id]),
        )
        if refined.cell_ids[first].startswith("r1.")
        and refined.cell_ids[second].startswith("r0.")
    )
    target = cell_index[refined.cell_ids[target]]

    adapted, _ = adapt_mesh(refined, {target: "xy"}, {"r0"})

    assert any(cell_id.startswith("r0.") for cell_id in adapted.cell_ids)
    assert max(adapted.cell_levels) == 2
    assert all(
        abs(adapted.cell_levels[owner] - adapted.cell_levels[neighbour]) <= 1
        for owner, neighbour in zip(
            adapted.face_owners, adapted.face_neighbours
        )
        if neighbour >= 0
    )


def test_anisotropic_refinement_limits_directional_transition_between_neighbours():
    original = two_cell_mesh()
    differently_refined, _ = adapt_mesh(original, {0: "x", 1: "y"}, set())
    target, neighbour = next(
        (left, right)
        for face_id, right in enumerate(differently_refined.face_neighbours)
        if right >= 0
        for left in (differently_refined.face_owners[face_id],)
        if differently_refined.cell_ids[left].startswith("r0.")
        and differently_refined.cell_ids[right].startswith("r1.")
    )
    neighbour_id = differently_refined.cell_ids[neighbour]

    adapted, _ = adapt_mesh(differently_refined, {target: "x"}, set())

    assert any(cell_id.startswith(neighbour_id + ".") for cell_id in adapted.cell_ids)
    directional_levels = _cell_directional_levels(
        adapted, adapted.cell_ids, adapted.refinement_records
    )
    assert all(
        abs(directional_levels[owner][axis] - directional_levels[neighbour][axis]) <= 1
        for face_id, neighbour in enumerate(adapted.face_neighbours)
        if neighbour >= 0
        for owner in (adapted.face_owners[face_id],)
        for axis in (0, 1)
    )


def test_anisotropic_transition_uses_each_original_cells_shape_as_its_baseline():
    original = two_cell_mesh()
    original.points = tuple(
        (2.0 * x if x <= 1.0 else x + 1.0, y)
        for x, y in original.points
    )

    refined, _ = adapt_mesh(original, {0: "y"}, set())

    assert len(refined.cells) == 3
    assert any(cell_id.startswith("r1") for cell_id in refined.cell_ids)


def test_two_to_one_balance_chooses_axis_that_avoids_anisotropic_plan_cycle():
    refine_marks = {1: "x"}

    _balance_adaptation_plan(
        refine_marks,
        set(),
        [{1}, {0}],
        ("r0", "r1"),
        (2, 3),
        ((2, 1), (0, 0)),
        {},
        4,
        (),
        [],
    )

    assert refine_marks == {0: "y", 1: "xy"}


def test_quality_gate_rejects_repeated_split_of_stretched_cell():
    from meshcfd.core.adaptation import _split_passes_quality

    stretched = ((0.0, 0.0), (12.0, 0.0), (12.0, 1.0), (0.0, 1.0))

    assert not _split_passes_quality(stretched, "y")


def test_quality_gate_accepts_regular_quadrilateral_split():
    from meshcfd.core.adaptation import _split_passes_quality

    regular = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))

    assert _split_passes_quality(regular, "x")
    assert _split_passes_quality(regular, "y")
