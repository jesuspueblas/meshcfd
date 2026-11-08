import math

import numpy as np
import pytest

from meshcfd.core.adaptation import _split_polygon, adapt_mesh, cell_areas
from meshcfd.core.curves import evaluate_boundary_curve
from meshcfd.core.model import (
    BoundaryCurve,
    Case,
    FiniteVolumeMesh,
    InitialState,
    MeshPatch,
    PeriodicBoundaryPair,
)
from meshcfd.core.periodic import match_periodic_faces
from meshcfd.core.solver import _transfer_cell_values, solve_case


def _curved_periodic_mesh():
    nx = ny = 2
    left_geometry = {
        "type": "spline",
        "points": [[0.0, 0.0], [-0.4, 0.33], [0.4, 0.67], [0.0, 1.0]],
    }
    right_geometry = {
        "type": "spline",
        "points": [[x + 2.0, y] for x, y in left_geometry["points"]],
    }
    points = []
    for j in range(ny + 1):
        parameter = j / ny
        left = evaluate_boundary_curve(left_geometry, parameter)
        right = evaluate_boundary_curve(right_geometry, parameter)
        points.extend(
            (
                left[0] + (right[0] - left[0]) * i / nx,
                left[1] + (right[1] - left[1]) * i / nx,
            )
            for i in range(nx + 1)
        )

    faces = []
    owners = []
    neighbours = []
    patch_ids = []
    face_curves = []
    cells = []
    cell_faces = []
    cell_signs = []
    edge_to_face = {}
    patch_faces = {patch_id: [] for patch_id in range(4)}

    def boundary_patch(first, second):
        first_i, first_j = first % (nx + 1), first // (nx + 1)
        second_i, second_j = second % (nx + 1), second // (nx + 1)
        if first_j == second_j == 0:
            return 2
        if first_i == second_i == nx:
            return 1
        if first_j == second_j == ny:
            return 3
        if first_i == second_i == 0:
            return 0
        return -1

    for j in range(ny):
        for i in range(nx):
            bottom_left = j * (nx + 1) + i
            cell = (
                bottom_left,
                bottom_left + 1,
                bottom_left + nx + 2,
                bottom_left + nx + 1,
            )
            cell_id = len(cells)
            cells.append(cell)
            local_faces = []
            local_signs = []
            for first, second in zip(cell, cell[1:] + cell[:1]):
                key = tuple(sorted((first, second)))
                if key not in edge_to_face:
                    face_id = len(faces)
                    edge_to_face[key] = face_id
                    faces.append((first, second))
                    owners.append(cell_id)
                    neighbours.append(-1)
                    patch_id = boundary_patch(first, second)
                    patch_ids.append(patch_id)
                    curve = None
                    if patch_id in (0, 1):
                        geometry = left_geometry if patch_id == 0 else right_geometry
                        start_t = (first // (nx + 1)) / ny
                        end_t = (second // (nx + 1)) / ny
                        curve = BoundaryCurve(
                            geometry, start_t, end_t, patch_id
                        )
                    face_curves.append(curve)
                    if patch_id >= 0:
                        patch_faces[patch_id].append(face_id)
                else:
                    face_id = edge_to_face[key]
                    neighbours[face_id] = cell_id
                    patch_ids[face_id] = -1
                    face_curves[face_id] = None
                local_faces.append(face_id)
                local_signs.append(
                    1 if faces[face_id] == (first, second) else -1
                )
            cell_faces.append(tuple(local_faces))
            cell_signs.append(tuple(local_signs))

    patches = tuple(
        MeshPatch(patch_id, name, tuple(patch_faces[patch_id]))
        for patch_id, name in enumerate(("left", "right", "bottom", "top"))
    )
    return FiniteVolumeMesh(
        points=tuple(points),
        cells=tuple(cells),
        faces=tuple(faces),
        face_owners=tuple(owners),
        face_neighbours=tuple(neighbours),
        face_patch_ids=tuple(patch_ids),
        patches=patches,
        cell_face_ids=tuple(cell_faces),
        cell_face_signs=tuple(cell_signs),
        face_curves=tuple(face_curves),
    )


@pytest.mark.parametrize(
    ("axis", "face_count"),
    (("x", 2), ("y", 4), ("xy", 4)),
)
def test_refinement_and_coarsening_preserve_curved_periodic_patches(
    axis, face_count
):
    mesh = _curved_periodic_mesh()

    refined, source_groups = adapt_mesh(
        mesh, {cell: axis for cell in range(4)}, set()
    )

    face_pairs, shift = match_periodic_faces(refined, 0, 1)
    assert shift == pytest.approx((2.0, 0.0))
    assert len(face_pairs) == face_count
    assert len(refined.patches[0].face_ids) == face_count
    assert len(refined.patches[1].face_ids) == face_count
    old_areas = np.asarray(cell_areas(mesh))
    new_areas = np.asarray(cell_areas(refined))
    transferred = _transfer_cell_values(
        np.ones((len(mesh.cells), 1)),
        old_areas,
        new_areas,
        source_groups,
    )
    assert np.dot(new_areas, transferred[:, 0]) == pytest.approx(np.sum(old_areas))
    assert any(
        evaluate_boundary_curve(
            curve.geometry,
            0.5 * (curve.start_parameter + curve.end_parameter),
        )
        != tuple(
            (refined.points[refined.faces[face_id][0]][axis]
             + refined.points[refined.faces[face_id][1]][axis])
            / 2.0
            for axis in (0, 1)
        )
        for face_id, curve in enumerate(refined.face_curves)
        if curve is not None
    )

    refined_again, _ = adapt_mesh(
        refined, {cell: axis for cell in range(len(refined.cells))}, set()
    )
    expected_again = face_count if axis == "x" else 2 * face_count
    assert len(match_periodic_faces(refined_again, 0, 1)[0]) == expected_again

    restored, _ = adapt_mesh(refined, {}, {f"r{cell}" for cell in range(4)})
    assert len(restored.cells) == len(mesh.cells)
    assert len(match_periodic_faces(restored, 0, 1)[0]) == 2


def test_arc_boundary_split_uses_its_parametric_midpoint():
    geometry = {
        "type": "arc",
        "center": [0.5, 0.5],
        "radius": 2**0.5 / 2,
        "start_angle": -3 * 3.141592653589793 / 4,
        "end_angle": -3.141592653589793 / 4,
    }
    curve = BoundaryCurve(geometry, 0.0, 1.0, 0)
    polygon = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))

    children = _split_polygon(polygon, "xy", (curve, None, None, None))

    assert children[0][1] == evaluate_boundary_curve(geometry, 0.5)


def test_periodic_adaptation_marking_keeps_spline_side_cells_paired():
    from meshcfd.core.adaptation import select_adaptation

    mesh = _curved_periodic_mesh()
    case = Case(mesh=mesh)
    case.periodic_pairs = [PeriodicBoundaryPair(0, 1, "x", 2.0)]
    for condition in case.boundary_conditions:
        condition.kind = "Periodic" if condition.patch_id in (0, 1) else "Wall"
    centers = np.asarray(
        [
            [
                sum(mesh.points[node][axis] for node in cell) / len(cell)
                for axis in (0, 1)
            ]
            for cell in mesh.cells
        ]
    )
    marks, _ = select_adaptation(
        mesh,
        case,
        centers,
        np.asarray(cell_areas(mesh)),
        np.asarray((1.0, 2.0, 1.5, 4.0)),
        1e-8,
        1,
    )

    paired_faces, _ = match_periodic_faces(mesh, 0, 1)
    assert marks
    for face_a, face_b in paired_faces:
        cell_a = mesh.face_owners[face_a]
        cell_b = mesh.face_owners[face_b]
        assert marks.get(cell_a) == marks.get(cell_b)
    assert any(mesh.face_owners[face_a] in marks for face_a, _ in paired_faces)

    # The quality gate must not silently change only one side after periodic
    # marks have been synchronized.  The adapted patches must remain matchable.
    refined, _ = adapt_mesh(mesh, marks, set())
    refined_pairs, _ = match_periodic_faces(refined, 0, 1)
    assert refined_pairs
    assert len(refined.patches[0].face_ids) == len(refined.patches[1].face_ids)


def test_inviscid_vortex_runs_after_adapting_curved_periodic_mesh():
    mesh = _curved_periodic_mesh()
    case = Case(mesh=mesh)
    case.periodic_pairs = [
        PeriodicBoundaryPair(0, 1, "x", 2.0),
        PeriodicBoundaryPair(2, 3, "y", 1.0),
    ]
    for condition in case.boundary_conditions:
        condition.kind = "Periodic"
    case.numerics.reconstruction = "First order"
    case.numerics.time_integration = "Explicit Euler"
    case.numerics.cfl = 0.1
    case.numerics.end_time = 0.01
    case.numerics.max_iterations = 1
    case.numerics.adaptive_mesh = True
    case.numerics.adaptation_interval = 1
    case.numerics.adaptation_max_level = 1
    case.numerics.adaptation_refine_threshold = 1e-12
    case.numerics.adaptation_variable = "Velocity X"

    states = []
    for cell in mesh.cells:
        x = sum(mesh.points[node][0] for node in cell) / 4
        y = sum(mesh.points[node][1] for node in cell) / 4
        radius_squared = (x - 1.0) ** 2 + (y - 0.5) ** 2
        exponential = math.exp(0.5 * (1.0 - radius_squared))
        temperature = 1.0 - (
            0.4 / (8.0 * 1.4 * math.pi**2)
        ) * math.exp(1.0 - radius_squared)
        density = temperature ** (1.0 / 0.4)
        states.append(
            InitialState(
                density,
                1.0 - exponential * (y - 0.5) / (2.0 * math.pi),
                1.0 + exponential * (x - 1.0) / (2.0 * math.pi),
                density**1.4,
            )
        )

    result = solve_case(case, tuple(states))

    assert result.steps == 1
    assert len(case.mesh.cells) > len(mesh.cells)
    assert len(match_periodic_faces(case.mesh, 0, 1)[0]) > 0
    assert len(match_periodic_faces(case.mesh, 2, 3)[0]) > 0
