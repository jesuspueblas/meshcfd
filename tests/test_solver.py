import math

import numpy as np
import pytest

from meshcfd.core.euler import (
    numerical_flux,
    physical_flux,
    primitive_to_conservative,
)
from meshcfd.core.model import (
    Case,
    FiniteVolumeMesh,
    InitialState,
    MeshPatch,
    PeriodicBoundaryPair,
)
from meshcfd.core.solver import (
    SolverError,
    SolverProgress,
    _build_geometry,
    _far_field_boundary_states,
    _parabolic_reconstruction,
    solve_case,
)


def make_periodic_mesh(nx=1, ny=1, length_x=1.0, length_y=1.0):
    points = tuple(
        (length_x * i / nx, length_y * j / ny)
        for j in range(ny + 1)
        for i in range(nx + 1)
    )
    cells = []
    faces = []
    owners = []
    neighbours = []
    patch_ids = []
    cell_face_ids = []
    cell_face_signs = []
    edge_to_face = {}
    patch_face_ids = {patch_id: [] for patch_id in range(4)}

    def boundary_patch(start, end):
        x0, y0 = points[start]
        x1, y1 = points[end]
        if y0 == y1 == 0:
            return 0
        if x0 == x1 == length_x:
            return 1
        if y0 == y1 == length_y:
            return 2
        if x0 == x1 == 0:
            return 3
        return -1

    for j in range(ny):
        for i in range(nx):
            bottom_left = j * (nx + 1) + i
            bottom_right = bottom_left + 1
            top_right = bottom_right + nx + 1
            top_left = bottom_left + nx + 1
            cell_id = len(cells)
            cell = (bottom_left, bottom_right, top_right, top_left)
            cells.append(cell)
            local_faces = []
            local_signs = []
            for start, end in zip(cell, cell[1:] + cell[:1]):
                key = tuple(sorted((start, end)))
                if key not in edge_to_face:
                    face_id = len(faces)
                    edge_to_face[key] = face_id
                    faces.append((start, end))
                    owners.append(cell_id)
                    neighbours.append(-1)
                    patch_id = boundary_patch(start, end)
                    patch_ids.append(patch_id)
                    if patch_id >= 0:
                        patch_face_ids[patch_id].append(face_id)
                else:
                    face_id = edge_to_face[key]
                    neighbours[face_id] = cell_id
                    patch_ids[face_id] = -1
                local_faces.append(face_id)
                local_signs.append(1 if faces[face_id] == (start, end) else -1)
            cell_face_ids.append(tuple(local_faces))
            cell_face_signs.append(tuple(local_signs))
    patches = tuple(
        MeshPatch(patch_id, name, tuple(patch_face_ids[patch_id]))
        for patch_id, name in enumerate(("bottom", "right", "top", "left"))
    )
    return FiniteVolumeMesh(
        points=points,
        cells=tuple(cells),
        faces=tuple(faces),
        face_owners=tuple(owners),
        face_neighbours=tuple(neighbours),
        face_patch_ids=tuple(patch_ids),
        patches=patches,
        cell_face_ids=tuple(cell_face_ids),
        cell_face_signs=tuple(cell_face_signs),
    )


def make_periodic_case(nx=1, ny=1, length_x=1.0, length_y=1.0):
    mesh = make_periodic_mesh(nx, ny, length_x, length_y)
    case = Case(mesh=mesh)
    for condition in case.boundary_conditions:
        condition.kind = "Periodic"
    case.periodic_pairs = [
        PeriodicBoundaryPair(1, 3, "x", length_x),
        PeriodicBoundaryPair(0, 2, "y", length_y),
    ]
    return case


@pytest.mark.parametrize("scheme", ("Rusanov", "HLLC", "Roe"))
def test_numerical_fluxes_return_physical_flux_for_equal_states(scheme):
    states = np.array(
        ((1.2, 0.4, -0.3, 1.1), (0.9, -0.2, 0.5, 0.8), (1.0, 0.0, 0.0, 1.0))
    )
    normals = np.array(((1.0, 0.0), (0.0, 1.0), (0.6, 0.8)))

    expected, _ = physical_flux(states, normals, 1.4)
    flux = numerical_flux(states, states, normals, 1.4, scheme)

    np.testing.assert_allclose(flux, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("flux", ("Rusanov", "HLLC", "Roe"))
@pytest.mark.parametrize(
    "integrator", ("Explicit Euler", "Explicit RK2", "Explicit RK3")
)
def test_uniform_periodic_flow_is_preserved(flux, integrator):
    case = make_periodic_case()
    case.numerics.flux = flux
    case.numerics.time_integration = integrator
    case.numerics.end_time = 0.9
    case.numerics.cfl = 0.5
    initial = (InitialState(1.0, 0.2, -0.1, 1.0),)

    result = solve_case(case, initial)

    assert result.completed
    assert result.time == case.numerics.end_time
    assert result.steps > 1
    assert result.completed
    np.testing.assert_allclose(
        (
            result.states[0].density,
            result.states[0].velocity_x,
            result.states[0].velocity_y,
            result.states[0].pressure,
        ),
        (1.0, 0.2, -0.1, 1.0),
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize("flux", ("Rusanov", "HLLC", "Roe"))
@pytest.mark.parametrize(
    "integrator", ("Explicit Euler", "Explicit RK2", "Explicit RK3")
)
@pytest.mark.parametrize(
    "reconstruction", ("First order", "MUSCL linear", "MUSCL parabolic")
)
def test_periodic_solver_conserves_integrated_euler_state(
    flux, integrator, reconstruction
):
    case = make_periodic_case(4, 3)
    case.numerics.flux = flux
    case.numerics.time_integration = integrator
    case.numerics.reconstruction = reconstruction
    case.numerics.end_time = 0.01
    initial = tuple(
        InitialState(
            density=1.0 + 0.03 * math.sin(2 * math.pi * (i + 0.5) / 4),
            velocity_x=0.2,
            velocity_y=-0.1,
            pressure=1.0 + 0.02 * math.cos(2 * math.pi * (j + 0.5) / 3),
        )
        for j in range(3)
        for i in range(4)
    )
    conservative_before = primitive_to_conservative(
        np.array(
            [
                (state.density, state.velocity_x, state.velocity_y, state.pressure)
                for state in initial
            ]
        ),
        case.numerics.gamma,
    )

    result = solve_case(case, initial)
    conservative_after = primitive_to_conservative(
        np.array(
            [
                (
                    state.density,
                    state.velocity_x,
                    state.velocity_y,
                    state.pressure,
                )
                for state in result.states
            ]
        ),
        case.numerics.gamma,
    )

    np.testing.assert_allclose(
        conservative_after.sum(axis=0),
        conservative_before.sum(axis=0),
        rtol=1e-11,
        atol=1e-11,
    )


def test_parabolic_reconstruction_recovers_quadratic_field():
    case = make_periodic_case(6, 6)
    geometry = _build_geometry(case.mesh, case)
    primitive = np.array(
        [
            (
                10.0 + x * x + 2.0 * x * y + 3.0 * y * y,
                0.2,
                -0.1,
                1.0,
            )
            for x, y in geometry.centers
        ]
    )

    gradients, curvature, factors = _parabolic_reconstruction(
        primitive, geometry, "Barth-Jespersen"
    )

    cell_id = 3 * 6 + 3
    x, y = geometry.centers[cell_id]
    np.testing.assert_allclose(
        gradients[cell_id, 0],
        (2.0 * x + 2.0 * y, 2.0 * x + 6.0 * y),
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        curvature[cell_id, 0], (2.0, 2.0, 6.0), rtol=1e-10, atol=1e-10
    )
    assert factors[cell_id, 0] == pytest.approx(1.0)


@pytest.mark.parametrize("flux", ("Rusanov", "HLLC", "Roe"))
def test_velocity_inlet_and_pressure_outlet_preserve_matching_uniform_flow(flux):
    case = make_periodic_case()
    case.periodic_pairs.clear()
    case.numerics.flux = flux
    case.numerics.gas_constant = 1.0
    case.numerics.end_time = 0.01
    for condition in case.boundary_conditions:
        condition.kind = "Wall"
        condition.values.clear()
    case.boundary_conditions[1].kind = "Pressure Outlet"
    case.boundary_conditions[1].values = {"pressure": 1.0}
    case.boundary_conditions[3].kind = "Velocity Inlet"
    case.boundary_conditions[3].values = {
        "density": 1.0,
        "velocity_x": 1.0,
        "velocity_y": 0.0,
        "temperature": 1.0,
    }

    result = solve_case(case, (InitialState(1.0, 1.0, 0.0, 1.0),))

    assert result.completed
    np.testing.assert_allclose(
        [
            result.states[0].density,
            result.states[0].velocity_x,
            result.states[0].velocity_y,
            result.states[0].pressure,
        ],
        (1.0, 1.0, 0.0, 1.0),
        rtol=1e-12,
        atol=1e-12,
    )


def test_pressure_outlet_applies_configured_static_pressure():
    case = make_periodic_case()
    case.periodic_pairs.clear()
    case.numerics.gas_constant = 1.0
    case.numerics.end_time = 0.01
    case.numerics.max_iterations = 1
    for condition in case.boundary_conditions:
        condition.kind = "Wall"
        condition.values.clear()
    case.boundary_conditions[1].kind = "Pressure Outlet"
    case.boundary_conditions[1].values = {"pressure": 0.8}
    case.boundary_conditions[3].kind = "Velocity Inlet"
    case.boundary_conditions[3].values = {
        "density": 1.0,
        "velocity_x": 0.0,
        "velocity_y": 0.0,
        "temperature": 1.0,
    }

    result = solve_case(case, (InitialState(1.0, 0.0, 0.0, 1.0),))

    assert result.states[0].velocity_x > 0.0


def test_velocity_inlet_rejects_non_positive_density():
    case = make_periodic_case()
    case.periodic_pairs.clear()
    for condition in case.boundary_conditions:
        condition.kind = "Wall"
    case.boundary_conditions[3].kind = "Velocity Inlet"
    case.boundary_conditions[3].values = {
        "density": 0.0,
        "velocity_x": 1.0,
        "velocity_y": 0.0,
        "temperature": 1.0,
    }

    with pytest.raises(SolverError, match="Density.*must be positive"):
        solve_case(case, (InitialState(),))


@pytest.mark.parametrize("flux", ("Rusanov", "HLLC", "Roe"))
def test_far_field_preserves_matching_uniform_flow(flux):
    case = make_periodic_case()
    case.periodic_pairs.clear()
    case.numerics.flux = flux
    case.numerics.end_time = 0.01
    for condition in case.boundary_conditions:
        condition.kind = "Far Field"
        condition.values = {
            "density": 1.0,
            "velocity_x": 0.2,
            "velocity_y": -0.1,
            "pressure": 1.0,
        }

    result = solve_case(case, (InitialState(1.0, 0.2, -0.1, 1.0),))

    assert result.completed
    np.testing.assert_allclose(
        [
            result.states[0].density,
            result.states[0].velocity_x,
            result.states[0].velocity_y,
            result.states[0].pressure,
        ],
        (1.0, 0.2, -0.1, 1.0),
        rtol=1e-12,
        atol=1e-12,
    )


def test_far_field_characteristics_select_incoming_and_outgoing_information():
    interior = np.array(
        ((1.0, 3.0, 0.0, 1.0), (1.0, 0.2, 0.0, 1.0), (1.0, -3.0, 0.0, 1.0))
    )
    far_field = np.array(
        ((2.0, 0.0, 0.0, 2.0), (2.0, 0.4, 0.0, 2.0), (2.0, 0.0, 0.0, 2.0))
    )
    normals = np.array(((1.0, 0.0), (1.0, 0.0), (1.0, 0.0)))

    exterior = _far_field_boundary_states(interior, far_field, normals, 1.4)

    np.testing.assert_allclose(exterior[0], interior[0])
    np.testing.assert_allclose(exterior[2], far_field[2])
    assert exterior[1, 1] == pytest.approx(0.3)
    assert exterior[1, 2] == pytest.approx(interior[1, 2])
    assert exterior[1, 3] / exterior[1, 0] ** 1.4 == pytest.approx(
        interior[1, 3] / interior[1, 0] ** 1.4
    )


def test_far_field_rejects_non_positive_pressure():
    case = make_periodic_case()
    case.periodic_pairs.clear()
    for condition in case.boundary_conditions:
        condition.kind = "Wall"
    case.boundary_conditions[1].kind = "Far Field"
    case.boundary_conditions[1].values = {
        "density": 1.0,
        "velocity_x": 1.0,
        "velocity_y": 0.0,
        "pressure": 0.0,
    }

    with pytest.raises(SolverError, match="Pressure.*must be positive"):
        solve_case(case, (InitialState(),))


def test_solver_reports_iteration_limit_and_rejects_unsupported_choices():
    case = make_periodic_case()
    case.numerics.end_time = 10.0
    case.numerics.max_iterations = 1
    result = solve_case(case, (InitialState(1.0, 0.1, 0.0, 1.0),))

    assert not result.completed
    assert result.steps == 1
    assert result.time < case.numerics.end_time

    case.numerics.reconstruction = "MUSCL parabolic"
    parabolic_result = solve_case(case, (InitialState(1.0, 0.1, 0.0, 1.0),))
    assert not parabolic_result.completed
    assert parabolic_result.steps == 1


def test_solver_continues_from_previous_time_and_step_count():
    case = make_periodic_case()
    case.numerics.end_time = 10.0
    case.numerics.max_iterations = 1
    initial = (InitialState(1.0, 0.1, 0.0, 1.0),)

    first = solve_case(case, initial)
    progress = []
    second = solve_case(
        case,
        first.states,
        progress.append,
        start_time=first.time,
        step_offset=first.steps,
    )

    assert not first.completed
    assert not second.completed
    assert second.time > first.time
    assert second.steps == first.steps + 1
    assert progress[0].step == second.steps
    assert progress[0].time > first.time


def test_solver_rejects_end_time_not_after_current_time():
    case = make_periodic_case()
    case.numerics.end_time = 0.1

    with pytest.raises(SolverError, match="greater than the current solution time"):
        solve_case(case, (InitialState(),), start_time=0.1)


def test_solver_reports_residual_history_and_optional_live_states():
    case = make_periodic_case()
    case.numerics.end_time = 0.1
    progress = []

    result = solve_case(
        case,
        (InitialState(1.0, 0.1, 0.0, 1.0),),
        progress.append,
        live_solution_updates=True,
    )

    assert result.completed
    assert progress
    assert all(isinstance(item, SolverProgress) for item in progress)
    assert progress[-1].time == result.time
    assert all(item.time_step > 0.0 for item in progress)
    assert all(len(item.residuals) == 4 for item in progress)
    assert progress[0].states is not None
    assert progress[-1].states is not None
    assert progress[-1].states == result.states


def test_solver_progress_omits_states_when_live_updates_are_disabled():
    case = make_periodic_case()
    case.numerics.end_time = 0.1
    progress = []

    solve_case(
        case,
        (InitialState(1.0, 0.1, 0.0, 1.0),),
        progress.append,
    )

    assert progress
    assert all(item.states is None for item in progress)


@pytest.mark.parametrize("reconstruction", ("MUSCL linear", "MUSCL parabolic"))
def test_isentropic_euler_vortex_runs_for_one_period(reconstruction):
    gamma = 1.4
    beta = 5.0
    freestream_x = 1.0
    freestream_y = 1.0
    center_x = 5.0
    center_y = 5.0
    case = make_periodic_case(10, 10, 10.0, 10.0)
    case.numerics.flux = "HLLC"
    case.numerics.time_integration = "Explicit RK3"
    case.numerics.reconstruction = reconstruction
    case.numerics.limiter = "Barth-Jespersen"
    case.numerics.cfl = 0.4
    case.numerics.end_time = 10.0
    case.numerics.max_iterations = 5000

    states = []
    for cell in case.mesh.cells:
        x = sum(case.mesh.points[node][0] for node in cell) / 4
        y = sum(case.mesh.points[node][1] for node in cell) / 4
        radius_squared = (x - center_x) ** 2 + (y - center_y) ** 2
        exponential = math.exp(0.5 * (1.0 - radius_squared))
        temperature = 1.0 - (
            (gamma - 1.0) * beta**2 / (8.0 * gamma * math.pi**2)
        ) * math.exp(1.0 - radius_squared)
        density = temperature ** (1.0 / (gamma - 1.0))
        states.append(
            InitialState(
                density=density,
                velocity_x=freestream_x
                - beta / (2.0 * math.pi) * exponential * (y - center_y),
                velocity_y=freestream_y
                + beta / (2.0 * math.pi) * exponential * (x - center_x),
                pressure=density**gamma,
            )
        )

    result = solve_case(case, tuple(states))

    assert result.completed
    assert result.steps < case.numerics.max_iterations
    assert all(state.density > 0.0 and state.pressure > 0.0 for state in result.states)
    initial_density = np.array([state.density for state in states])
    final_density = np.array([state.density for state in result.states])
    relative_density_error = np.linalg.norm(final_density - initial_density) / np.linalg.norm(
        initial_density
    )
    assert relative_density_error < 0.2
