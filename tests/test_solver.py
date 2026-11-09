import math
from threading import Event, Thread

import numpy as np
import pytest

from meshcfd.core._kernels import (
    limited_gradient_factors,
    linear_gradients_and_factors,
    point_to_segments_minimum_distance,
    reconstruct_hllc_flux,
    weighted_least_squares_gradients,
)

from meshcfd.core.adaptation import AdaptationBalanceError
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
from meshcfd.core.post import sample_solution_cut
from meshcfd.core.solver import (
    SolverError,
    SolverProgress,
    _build_geometry,
    _dynamic_viscosity,
    _far_field_boundary_states,
    _implicit_euler_step,
    _mass_flow_inlet_boundary_states,
    _limited_gradients,
    _mass_flow_outlet_boundary_states,
    _parabolic_reconstruction,
    _pressure_inlet_boundary_states,
    _reconstruct,
    _primitive_from_conservative,
    _residual,
    _sst_fields,
    _sst_omega_wall_scale,
    _sst_regularized_rate,
    _sst_regularized_turbulence,
    _sst_regularized_variable,
    _sst_wall_values,
    _turbulence_boundary_values,
    _viscous_flux,
    derive_solution_fields,
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


def test_sutherland_viscosity_uses_reference_air_values():
    settings = Case().numerics
    settings.viscosity_model = "Sutherland"

    viscosity = _dynamic_viscosity(
        np.array((273.15, 300.0)), settings
    )

    assert viscosity[0] == pytest.approx(1.716e-5)
    assert viscosity[1] > viscosity[0]


def test_constant_body_acceleration_adds_momentum_and_energy_sources():
    case = make_periodic_case()
    case.numerics.gas_constant = 1.0
    case.numerics.reconstruction = "First order"
    case.numerics.acceleration_x = 2.0
    case.numerics.acceleration_y = -3.0
    geometry = _build_geometry(case.mesh, case)
    conservative = primitive_to_conservative(
        np.array(((1.0, 0.4, -0.2, 1.0),)), case.numerics.gamma
    )

    derivative = _residual(conservative, geometry, case.numerics)

    np.testing.assert_allclose(derivative[0], (0.0, 2.0, -3.0, 1.4), atol=1e-14)


def test_viscous_flux_includes_heat_conduction_and_stokes_stress():
    gradients = np.array((((2.0, 1.0), (2.0, 3.0), (4.0, -1.0)),))
    velocity = np.array(((5.0, 7.0),))

    flux = _viscous_flux(
        gradients,
        velocity,
        np.array((2.0,)),
        np.array((3.0,)),
        np.array(((1.0, 0.0),)),
    )

    assert flux[0, 0] == 0.0
    assert flux[0, 1] == pytest.approx(2.0 * (4.0 - 10.0 / 3.0))
    assert flux[0, 2] == pytest.approx(2.0 * (1.0 + 2.0))
    assert flux[0, 3] == pytest.approx(5.0 * flux[0, 1] + 7.0 * flux[0, 2] + 12.0)


def test_sst_closure_returns_finite_turbulent_viscosity():
    case = make_periodic_case()
    case.numerics.viscosity_model = "Constant"
    case.numerics.dynamic_viscosity = 1e-3
    case.numerics.turbulence_model = "Menter k-omega SST"
    geometry = _build_geometry(case.mesh, case)
    primitive = np.array(((1.0, 0.0, 0.0, 1.0),))
    turbulence = np.array(((0.01, 10.0),))

    fields = _sst_fields(primitive, turbulence, geometry, case.numerics)

    assert fields[0][0] == pytest.approx(0.001)
    assert np.isfinite(fields[0]).all()
    assert np.all(fields[0] >= 0.0)


def test_regularized_sst_variable_uses_article_wall_asymptote():
    case = make_periodic_case()
    case.numerics.viscosity_model = "Constant"
    case.numerics.dynamic_viscosity = 1e-3
    case.numerics.turbulence_model = "Menter k-omega SST"
    case.numerics.turbulence_omega_formulation = "Regularized k-log-omega"
    geometry = _build_geometry(case.mesh, case)
    primitive = np.array(((1.0, 0.0, 0.0, 1.0),))
    omega = np.array((12.0,))
    variable = _sst_regularized_variable(
        primitive, omega, geometry, case.numerics
    )

    restored = _sst_regularized_turbulence(
        primitive, np.array((0.025,)), variable, geometry, case.numerics
    )

    assert variable[0] == pytest.approx(
        math.log(12.0 / (6.0e-3 / 0.09))
    )
    assert restored[0, 0] == pytest.approx(0.025)
    assert restored[0, 1] == pytest.approx(12.0)


def test_regularized_sst_rate_tracks_sutherland_wall_scale():
    settings = Case().numerics
    settings.viscosity_model = "Sutherland"
    settings.turbulence_model = "Menter k-omega SST"
    settings.turbulence_omega_formulation = "Regularized k-log-omega"
    settings.gas_constant = 287.05
    primitive = np.array(((1.2, 10.0, 2.0, 101325.0),))
    turbulence = np.array(((0.03, 12.0),))
    conservative = primitive_to_conservative(primitive, settings.gamma)
    conservative[:, 3] += turbulence[:, 0]
    flow_rate = np.array(((0.2, 0.4, -0.3, 10000.0),))
    turbulence_rate = np.array(((0.005, -0.1),))
    rate = _sst_regularized_rate(
        primitive,
        flow_rate,
        turbulence,
        turbulence_rate,
        settings,
    )

    time_step = 1e-6
    advanced_conservative = conservative + time_step * flow_rate
    advanced_turbulence = turbulence + time_step * turbulence_rate
    advanced_primitive = _primitive_from_conservative(
        advanced_conservative, advanced_turbulence, settings.gamma
    )
    initial_variable = (
        np.log(turbulence[0, 1] / primitive[0, 0])
        - np.log(
            _sst_omega_wall_scale(
                primitive, np.array((0.1,)), settings
            )[0]
        )
    )
    advanced_variable = (
        np.log(advanced_turbulence[0, 1] / advanced_primitive[0, 0])
        - np.log(
            _sst_omega_wall_scale(
                advanced_primitive, np.array((0.1,)), settings
            )[0]
        )
    )

    assert rate[0] == pytest.approx(
        (advanced_variable - initial_variable) / time_step,
        rel=2e-5,
    )


def test_regularized_sst_wall_condition_matches_zero_log_variable():
    case = make_periodic_case()
    case.periodic_pairs.clear()
    for boundary in case.boundary_conditions:
        boundary.kind = "Wall"
        boundary.mechanical_condition = "No-slip"
    case.numerics.viscosity_model = "Constant"
    case.numerics.dynamic_viscosity = 1e-3
    case.numerics.turbulence_model = "Menter k-omega SST"
    case.numerics.turbulence_omega_formulation = "Regularized k-log-omega"
    geometry = _build_geometry(case.mesh, case)
    primitive = np.array(((1.0, 0.0, 0.0, 1.0),))
    turbulence = np.array(((0.01, 10.0),))

    wall_values = _sst_wall_values(
        primitive, turbulence, geometry, case.numerics
    )
    fields = _sst_fields(primitive, turbulence, geometry, case.numerics)

    assert wall_values
    assert all(value[0] == 0.0 for value in wall_values.values())
    assert all(
        value[1] == pytest.approx(6.0e-3 / (0.09 * 0.25))
        for value in wall_values.values()
    )
    assert all(np.isfinite(field).all() for field in fields)


def test_sst_inlet_values_come_from_intensity_and_length_scale():
    case = make_periodic_case()
    case.periodic_pairs.clear()
    case.numerics.gas_constant = 1.0
    case.numerics.turbulence_model = "Menter k-omega SST"
    for boundary in case.boundary_conditions:
        boundary.kind = "Wall"
    inlet = case.boundary_conditions[1]
    inlet.kind = "Velocity Inlet"
    inlet.values = {
        "density": 1.0,
        "temperature": 1.0,
        "velocity_x": 3.0,
        "velocity_y": 4.0,
        "turbulence_intensity": 2.0,
        "turbulence_length_scale": 0.5,
    }
    geometry = _build_geometry(case.mesh, case)
    primitive = np.array(((1.0, 0.0, 0.0, 1.0),))
    turbulence = np.array(((0.01, 10.0),))

    boundary_values = _turbulence_boundary_values(
        primitive, turbulence, geometry, case.numerics
    )

    expected_k = 1.5 * (0.02 * 5.0) ** 2
    assert boundary_values[1][0] == pytest.approx(expected_k)
    assert boundary_values[1][1] == pytest.approx(
        math.sqrt(expected_k) / (0.09**0.25 * 0.5)
    )


def test_derived_stagnation_fields_and_mach_number():
    case = make_periodic_case()
    case.numerics.gas_constant = 1.0
    case.numerics.gamma = 1.4
    velocity = math.sqrt(1.4)

    fields = derive_solution_fields(
        case, (InitialState(1.0, velocity, 0.0, 1.0),)
    )

    total_factor = 1.2
    assert fields["mach_number"][0] == pytest.approx(1.0)
    assert fields["total_temperature"][0] == pytest.approx(total_factor)
    assert fields["total_pressure"][0] == pytest.approx(total_factor**3.5)
    assert fields["total_density"][0] == pytest.approx(total_factor**2.5)


def test_solver_adapts_mesh_at_configured_gradient_interval():
    case = make_periodic_case(2, 1)
    case.periodic_pairs.clear()
    for boundary in case.boundary_conditions:
        boundary.kind = "Wall"
    case.numerics.reconstruction = "First order"
    case.numerics.time_integration = "Explicit Euler"
    case.numerics.max_iterations = 2
    case.numerics.end_time = 1.0
    case.numerics.adaptive_mesh = True
    case.numerics.adaptation_interval = 1
    case.numerics.adaptation_max_level = 1
    case.numerics.adaptation_refine_threshold = 0.01
    states = (
        InitialState(density=1.0, pressure=1.0),
        InitialState(density=2.0, pressure=1.0),
    )

    result = solve_case(case, states)

    assert len(case.mesh.cells) == len(result.states) == 4
    assert case.mesh.cell_levels == (1, 1, 1, 1)
    assert sum(
        state.density * area
        for state, area in zip(result.states, _build_geometry(case.mesh, case).areas)
    ) == pytest.approx(1.5)


def test_solver_continues_when_adaptation_cannot_balance_mesh(monkeypatch):
    case = make_periodic_case(2, 1)
    case.periodic_pairs.clear()
    for boundary in case.boundary_conditions:
        boundary.kind = "Wall"
    case.numerics.reconstruction = "First order"
    case.numerics.time_integration = "Explicit Euler"
    case.numerics.max_iterations = 2
    case.numerics.end_time = 1.0
    case.numerics.adaptive_mesh = True
    case.numerics.adaptation_interval = 1
    case.numerics.adaptation_refine_threshold = 0.01

    attempts = []

    def reject_adaptation(*_args, **_kwargs):
        attempts.append(None)
        raise AdaptationBalanceError(
            "The current mesh cannot satisfy anisotropic transition balance."
        )

    monkeypatch.setattr("meshcfd.core.solver.select_adaptation", reject_adaptation)
    progress = []

    result = solve_case(
        case,
        (
            InitialState(density=1.0, pressure=1.0),
            InitialState(density=2.0, pressure=1.0),
        ),
        progress_callback=progress.append,
    )

    assert result.steps == 2
    assert len(attempts) == 1
    assert len(case.mesh.cells) == 2
    assert any(
        item.warning is not None
        and "Adaptive mesh disabled" in item.warning
        for item in progress
    )


def test_adaptation_keeps_periodic_face_subdivisions_matching():
    case = make_periodic_case(2, 2)
    case.numerics.reconstruction = "First order"
    case.numerics.time_integration = "Explicit Euler"
    case.numerics.max_iterations = 1
    case.numerics.adaptive_mesh = True
    case.numerics.adaptation_interval = 1
    case.numerics.adaptation_max_level = 1
    case.numerics.adaptation_refine_threshold = 0.01
    states = tuple(
        InitialState(density=1.0 if cell < 2 else 2.0, pressure=1.0)
        for cell in range(4)
    )

    result = solve_case(case, states)

    assert len(result.states) == len(case.mesh.cells) == 8
    _build_geometry(case.mesh, case)


def test_sst_derived_fields_include_wall_distance_and_regularized_log_omega():
    case = make_periodic_case()
    case.periodic_pairs.clear()
    for boundary in case.boundary_conditions:
        boundary.kind = "Wall"
        boundary.mechanical_condition = "No-slip"
    case.numerics.viscosity_model = "Constant"
    case.numerics.dynamic_viscosity = 1e-3
    case.numerics.turbulence_model = "Menter k-omega SST"
    case.numerics.turbulence_omega_formulation = "Regularized k-log-omega"
    state = InitialState(1.0, 2.0, 0.0, 1.0, turbulence_omega=3.0)

    fields = derive_solution_fields(case, (state,))

    wall_scale = 6.0e-3 / (0.09 * 0.5**2)
    assert fields["wall_distance"][0] == pytest.approx(0.5)
    assert fields["log_regularized_omega"][0] == pytest.approx(
        math.log(state.turbulence_omega / wall_scale)
    )


def test_y_plus_uses_nearest_no_slip_wall_and_molecular_viscosity():
    case = make_periodic_case()
    case.periodic_pairs.clear()
    for boundary in case.boundary_conditions:
        boundary.kind = "Wall"
        boundary.mechanical_condition = "No-slip"
    case.numerics.viscosity_model = "Constant"
    case.numerics.dynamic_viscosity = 1e-3

    fields = derive_solution_fields(
        case, (InitialState(1.0, 2.0, 0.0, 1.0),)
    )

    assert fields["y_plus"][0] == pytest.approx(math.sqrt(1000.0))


def test_y_plus_is_reported_only_for_cells_touching_a_no_slip_wall():
    case = make_periodic_case(3, 3)
    case.periodic_pairs.clear()
    for boundary in case.boundary_conditions:
        boundary.kind = "Wall"
        boundary.mechanical_condition = "No-slip"
    case.numerics.viscosity_model = "Constant"
    case.numerics.dynamic_viscosity = 1e-3
    states = tuple(
        InitialState(1.0, 1.0, 0.0, 1.0)
        for _cell in case.mesh.cells
    )

    fields = derive_solution_fields(case, states)

    assert fields["y_plus"][4] == 0.0
    assert fields["y_plus"][1] > 0.0


def test_wall_heat_flux_positive_into_fluid_adds_energy():
    case = make_periodic_case()
    case.periodic_pairs.clear()
    case.numerics.gas_constant = 1.0
    case.numerics.reconstruction = "First order"
    case.numerics.viscosity_model = "Constant"
    case.numerics.dynamic_viscosity = 1.0
    for condition in case.boundary_conditions:
        condition.kind = "Wall"
        condition.thermal_condition = "Temperature"
        condition.values = {"temperature": 1.0}
    wall = case.boundary_conditions[1]
    wall.thermal_condition = "Heat Flux"
    wall.values = {"heat_flux": 10.0}
    geometry = _build_geometry(case.mesh, case)
    conservative = primitive_to_conservative(
        np.array(((1.0, 0.0, 0.0, 1.0),)), case.numerics.gamma
    )

    derivative = _residual(conservative, geometry, case.numerics)

    assert derivative[0, 3] == pytest.approx(10.0)
    np.testing.assert_allclose(derivative[0, :3], 0.0, atol=1e-12)


def test_wall_heat_flux_is_exact_with_sutherland_viscosity():
    case = make_periodic_case()
    case.periodic_pairs.clear()
    case.numerics.gas_constant = 1.0
    case.numerics.reconstruction = "First order"
    case.numerics.viscosity_model = "Sutherland"
    case.numerics.sutherland_reference_viscosity = 1.0
    case.numerics.sutherland_reference_temperature = 1.0
    case.numerics.sutherland_temperature = 0.1
    for condition in case.boundary_conditions:
        condition.kind = "Wall"
        condition.thermal_condition = "Heat Flux"
        condition.values = {"heat_flux": 0.0}
    wall = case.boundary_conditions[1]
    wall.values["heat_flux"] = 10.0
    geometry = _build_geometry(case.mesh, case)
    conservative = primitive_to_conservative(
        np.array(((1.0, 0.0, 0.0, 1.0),)), case.numerics.gamma
    )

    derivative = _residual(conservative, geometry, case.numerics)

    assert derivative[0, 3] == pytest.approx(10.0)
    np.testing.assert_allclose(derivative[0, :3], 0.0, atol=1e-12)


def test_viscous_wall_slip_and_no_slip_are_distinct():
    case = make_periodic_case()
    case.numerics.reconstruction = "First order"
    case.numerics.viscosity_model = "Constant"
    case.numerics.dynamic_viscosity = 1.0
    case.numerics.gas_constant = 1.0
    case.periodic_pairs = [case.periodic_pairs[0]]
    for boundary in case.boundary_conditions:
        boundary.kind = "Wall" if boundary.patch_id in (0, 2) else "Periodic"
    for boundary in case.boundary_conditions:
        boundary.thermal_condition = "Temperature"
        boundary.values = {"temperature": 1.0}
    initial = np.array(((1.0, 1.0, 0.0, 1.0),))
    conservative = primitive_to_conservative(initial, case.numerics.gamma)

    case.boundary_conditions[0].mechanical_condition = "Slip"
    case.boundary_conditions[2].mechanical_condition = "Slip"
    geometry = _build_geometry(case.mesh, case)
    slip_derivative = _residual(conservative, geometry, case.numerics)
    case.boundary_conditions[0].mechanical_condition = "No-slip"
    case.boundary_conditions[2].mechanical_condition = "No-slip"
    geometry = _build_geometry(case.mesh, case)
    no_slip_derivative = _residual(conservative, geometry, case.numerics)

    np.testing.assert_allclose(slip_derivative[0, 1:3], 0.0, atol=1e-12)
    assert no_slip_derivative[0, 1] < 0.0
    assert no_slip_derivative[0, 2] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("flux", ("Rusanov", "HLLC", "Roe"))
@pytest.mark.parametrize(
    "integrator",
    ("Explicit Euler", "Explicit RK2", "Explicit RK3", "Explicit RK4"),
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
    "integrator",
    ("Explicit Euler", "Explicit RK2", "Explicit RK3", "Explicit RK4"),
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


@pytest.mark.parametrize("preconditioner", ("ILU", "LU-SGS"))
def test_implicit_euler_gmres_advances_above_explicit_cfl_and_conserves_flow(
    preconditioner,
):
    case = make_periodic_case(3, 2)
    case.numerics.flux = "Rusanov"
    case.numerics.time_integration = "Implicit Euler"
    case.numerics.cfl = 100.0
    case.numerics.implicit_preconditioner = preconditioner
    case.numerics.end_time = 0.005
    initial = tuple(
        InitialState(
            density=1.0 + 0.03 * math.sin(2 * math.pi * (i + 0.5) / 3),
            velocity_x=0.2,
            velocity_y=-0.1,
            pressure=1.0 + 0.02 * math.cos(2 * math.pi * (j + 0.5) / 2),
        )
        for j in range(2)
        for i in range(3)
    )
    initial_conservative = primitive_to_conservative(
        np.array(
            [
                (state.density, state.velocity_x, state.velocity_y, state.pressure)
                for state in initial
            ]
        ),
        case.numerics.gamma,
    )

    result = solve_case(case, initial)

    final_conservative = primitive_to_conservative(
        np.array(
            [
                (state.density, state.velocity_x, state.velocity_y, state.pressure)
                for state in result.states
            ]
        ),
        case.numerics.gamma,
    )
    assert result.completed
    assert result.time == case.numerics.end_time
    assert all(state.density > 0.0 and state.pressure > 0.0 for state in result.states)
    np.testing.assert_allclose(
        final_conservative.sum(axis=0),
        initial_conservative.sum(axis=0),
        rtol=1e-10,
        atol=1e-10,
    )


def test_implicit_euler_halves_time_step_until_nonlinear_solve_converges(
    monkeypatch,
):
    case = make_periodic_case()
    geometry = _build_geometry(case.mesh, case)
    state = primitive_to_conservative(
        np.array(((1.0, 0.2, -0.1, 1.0),)), case.numerics.gamma
    )
    attempted_time_steps = []

    def solve_once(initial, time_step, _geometry, _settings):
        attempted_time_steps.append(time_step)
        if len(attempted_time_steps) < 3:
            raise SolverError("nonlinear solve did not converge")
        return initial

    monkeypatch.setattr(
        "meshcfd.core.solver._implicit_euler_solve_once", solve_once
    )
    solution, accepted_time_step = _implicit_euler_step(
        state, 0.8, geometry, case.numerics
    )

    assert attempted_time_steps == pytest.approx((0.8, 0.4, 0.2))
    assert accepted_time_step == pytest.approx(0.2)
    np.testing.assert_array_equal(solution, state)


def test_implicit_euler_keeps_a_fixed_time_step_on_nonlinear_failure(
    monkeypatch,
):
    case = make_periodic_case()
    geometry = _build_geometry(case.mesh, case)
    state = primitive_to_conservative(
        np.array(((1.0, 0.2, -0.1, 1.0),)), case.numerics.gamma
    )
    attempts = []

    def fail_solve_once(_initial, time_step, _geometry, _settings):
        attempts.append(time_step)
        raise SolverError("nonlinear solve did not converge")

    monkeypatch.setattr(
        "meshcfd.core.solver._implicit_euler_solve_once", fail_solve_once
    )
    with pytest.raises(SolverError, match="configured fixed time step"):
        _implicit_euler_step(
            state,
            0.8,
            geometry,
            case.numerics,
            allow_time_step_reduction=False,
        )

    assert attempts == [0.8]


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


def test_pressure_inlet_reconstructs_matching_subsonic_flow():
    case = make_periodic_case()
    case.periodic_pairs.clear()
    case.numerics.gas_constant = 1.0
    for condition in case.boundary_conditions:
        condition.kind = "Wall"
    total_temperature = 1.0 + 0.2**2 / (2.0 * 3.5)
    total_pressure = total_temperature ** (1.4 / 0.4)
    case.boundary_conditions[3].kind = "Pressure Inlet"
    case.boundary_conditions[3].values = {
        "pressure": total_pressure,
        "temperature": total_temperature,
        "velocity_x": 1.0,
        "velocity_y": 0.0,
    }
    geometry = _build_geometry(case.mesh, case)
    interior = np.array(((1.0, 0.2, 0.0, 1.0),))
    boundary = geometry.pressure_inlets[0]

    states = _pressure_inlet_boundary_states(
        interior,
        geometry.face_normals[boundary.faces],
        boundary,
        case.numerics.gamma,
        case.numerics.gas_constant,
    )

    np.testing.assert_allclose(states, interior, rtol=1e-12, atol=1e-12)
    with pytest.raises(SolverError, match="supersonic inflow"):
        _pressure_inlet_boundary_states(
            np.array(((1.0, 2.0, 0.0, 1.0),)),
            geometry.face_normals[boundary.faces],
            boundary,
            case.numerics.gamma,
            case.numerics.gas_constant,
        )


def test_mass_flow_inlet_satisfies_integrated_flow_rate():
    case = make_periodic_case()
    case.periodic_pairs.clear()
    case.numerics.gas_constant = 1.0
    for condition in case.boundary_conditions:
        condition.kind = "Wall"
    case.boundary_conditions[3].kind = "Mass Flow Inlet"
    case.boundary_conditions[3].values = {
        "mass_flow": 0.25,
        "temperature": 2.0,
        "velocity_x": 1.0,
        "velocity_y": 0.0,
    }
    geometry = _build_geometry(case.mesh, case)
    boundary = geometry.mass_flow_inlets[0]
    normals = geometry.face_normals[boundary.faces]
    lengths = geometry.face_lengths[boundary.faces]

    states = _mass_flow_inlet_boundary_states(
        np.array(((1.0, 0.0, 0.0, 1.0),)),
        normals,
        lengths,
        boundary,
        case.numerics.gamma,
        case.numerics.gas_constant,
    )
    integrated_flow = np.sum(
        states[:, 0] * -(states[:, 1:3] * normals).sum(axis=1) * lengths
    )

    assert integrated_flow == pytest.approx(boundary.mass_flow, rel=1e-12)
    np.testing.assert_allclose(states[:, 3], 1.0)


def test_mass_flow_outlet_preserves_pressure_density_and_tangential_velocity():
    case = make_periodic_case()
    case.periodic_pairs.clear()
    for condition in case.boundary_conditions:
        condition.kind = "Wall"
    case.boundary_conditions[1].kind = "Mass Flow Outlet"
    case.boundary_conditions[1].values = {"mass_flow": 0.6}
    geometry = _build_geometry(case.mesh, case)
    boundary = geometry.mass_flow_outlets[0]
    normals = geometry.face_normals[boundary.faces]
    lengths = geometry.face_lengths[boundary.faces]
    interior = np.array(((2.0, 0.4, 0.3, 1.2),))

    states = _mass_flow_outlet_boundary_states(
        interior, normals, lengths, boundary
    )

    assert np.sum(states[:, 0] * (states[:, 1:3] * normals).sum(axis=1) * lengths) == pytest.approx(
        boundary.mass_flow
    )
    np.testing.assert_allclose(states[:, (0, 3)], interior[:, (0, 3)])
    np.testing.assert_allclose(states[:, 2], interior[:, 2])


@pytest.mark.parametrize("kind", ("Mass Flow Inlet", "Mass Flow Outlet"))
def test_mass_flow_boundaries_reject_non_positive_flow_rate(kind):
    case = make_periodic_case()
    case.periodic_pairs.clear()
    for condition in case.boundary_conditions:
        condition.kind = "Wall"
    condition = case.boundary_conditions[3 if kind == "Mass Flow Inlet" else 1]
    condition.kind = kind
    condition.values = {"mass_flow": 0.0}

    with pytest.raises(SolverError, match="Mass flow.*must be positive"):
        solve_case(case, (InitialState(),))


@pytest.mark.parametrize("kind", ("Pressure Inlet", "Mass Flow Inlet"))
def test_inlet_boundaries_reject_direction_pointing_out_of_domain(kind):
    case = make_periodic_case()
    case.periodic_pairs.clear()
    for condition in case.boundary_conditions:
        condition.kind = "Wall"
    inlet = case.boundary_conditions[3]
    inlet.kind = kind
    inlet.values = (
        {"pressure": 101325.0, "temperature": 288.15}
        if kind == "Pressure Inlet"
        else {"mass_flow": 0.2, "temperature": 288.15}
    )
    inlet.values.update({"velocity_x": -1.0, "velocity_y": 0.0})

    with pytest.raises(SolverError, match="direction must point into the domain"):
        solve_case(case, (InitialState(),))


@pytest.mark.parametrize("inlet_kind", ("Pressure Inlet", "Mass Flow Inlet"))
def test_inlet_boundary_types_run_with_mass_flow_outlet(inlet_kind):
    case = make_periodic_case()
    case.periodic_pairs.clear()
    case.numerics.gas_constant = 1.0
    case.numerics.end_time = 0.01
    case.numerics.max_iterations = 1
    for condition in case.boundary_conditions:
        condition.kind = "Wall"
        condition.values.clear()
    inlet = case.boundary_conditions[3]
    inlet.kind = inlet_kind
    inlet.values = (
        {
            "pressure": (1.0 + 0.2**2 / 7.0) ** 3.5,
            "temperature": 1.0 + 0.2**2 / 7.0,
            "velocity_x": 1.0,
            "velocity_y": 0.0,
        }
        if inlet_kind == "Pressure Inlet"
        else {
            "mass_flow": 0.2,
            "temperature": 1.0 + 0.2**2 / 7.0,
            "velocity_x": 1.0,
            "velocity_y": 0.0,
        }
    )
    outlet = case.boundary_conditions[1]
    outlet.kind = "Mass Flow Outlet"
    outlet.values = {"mass_flow": 0.2}

    result = solve_case(case, (InitialState(1.0, 0.2, 0.0, 1.0),))

    assert result.steps == 1


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


def test_solver_emits_solution_snapshots_at_configured_global_steps():
    case = make_periodic_case()
    case.numerics.end_time = 10.0
    case.numerics.max_iterations = 5
    snapshots = []

    solve_case(
        case,
        (InitialState(1.0, 0.1, 0.0, 1.0),),
        solution_callback=lambda step, time, states: snapshots.append((step, time, states)),
        solution_interval=2,
    )

    assert [snapshot[0] for snapshot in snapshots] == [2, 4]
    assert all(len(snapshot[2]) == len(case.mesh.cells) for snapshot in snapshots)


def test_solver_samples_convergence_history_at_configured_interval():
    case = make_periodic_case()
    case.numerics.end_time = 10.0
    case.numerics.max_iterations = 5
    case.numerics.convergence_history_interval = 2
    progress = []

    result = solve_case(
        case,
        (InitialState(1.0, 0.1, 0.0, 1.0),),
        progress.append,
        cpu_time_offset=3.0,
    )

    assert [item.step for item in progress] == [2, 4, 5]
    assert [item.time for item in progress] == sorted(item.time for item in progress)
    assert all(item.residuals for item in progress)
    assert all(item.cpu_time >= 3.0 for item in progress)
    assert [item.cpu_time for item in progress] == sorted(
        item.cpu_time for item in progress
    )
    assert result.cpu_time >= progress[-1].cpu_time


def test_global_fixed_time_step_is_shared_and_clipped_at_end_time():
    case = make_periodic_case()
    case.numerics.time_step_mode = "Global fixed"
    case.numerics.fixed_time_step = 0.002
    case.numerics.end_time = 0.005
    case.numerics.max_iterations = 5
    case.numerics.convergence_history_interval = 1
    progress = []

    result = solve_case(case, (InitialState(),), progress.append)

    assert result.completed
    assert [entry.time_step for entry in progress] == pytest.approx(
        (0.002, 0.002, 0.001)
    )
    assert result.time == pytest.approx(0.005)


@pytest.mark.parametrize(
    "time_integration",
    ("Explicit Euler", "Explicit RK2", "Explicit RK3", "Explicit RK4"),
)
def test_local_time_step_advances_each_cell_with_its_own_cfl_step(
    monkeypatch, time_integration
):
    case = make_periodic_case(nx=2, length_x=2.0)
    case.numerics.time_step_mode = "Local (steady)"
    case.numerics.time_integration = time_integration
    case.numerics.max_iterations = 1
    case.numerics.residual_tolerance = 1e-14
    local_steps = np.array((0.01, 0.02))
    rates = np.array(((0.0, 1.0, 0.0, 0.0), (0.0, 2.0, 0.0, 0.0)))
    monkeypatch.setattr(
        "meshcfd.core.solver._stable_time_steps",
        lambda *_args: local_steps.copy(),
    )
    monkeypatch.setattr(
        "meshcfd.core.solver._residual", lambda *_args: rates.copy()
    )
    progress = []

    result = solve_case(
        case,
        (
            InitialState(density=1.0, pressure=1.0),
            InitialState(density=1.0, pressure=1.0),
        ),
        progress.append,
    )

    assert not result.completed
    assert result.steps == 1
    assert result.time == pytest.approx(np.mean(local_steps))
    assert progress[0].time_step == pytest.approx(np.mean(local_steps))
    assert [state.velocity_x for state in result.states] == pytest.approx(
        (0.01, 0.04)
    )


def test_local_time_stepping_stops_when_residual_tolerance_is_reached():
    case = make_periodic_case()
    case.numerics.time_step_mode = "Local (steady)"
    case.numerics.end_time = 1e-12

    result = solve_case(case, (InitialState(),))

    assert result.completed
    assert result.steps == 1
    assert result.time > case.numerics.end_time


def test_local_time_stepping_supports_implicit_euler():
    case = make_periodic_case(3, 2)
    case.numerics.time_step_mode = "Local (steady)"
    case.numerics.time_integration = "Implicit Euler"
    case.numerics.cfl = 0.1
    case.numerics.max_iterations = 1
    case.numerics.residual_tolerance = 1e-14
    initial = tuple(
        InitialState(
            density=1.0 + 0.03 * math.sin(2 * math.pi * (i + 0.5) / 3),
            velocity_x=0.2,
            velocity_y=-0.1,
            pressure=1.0 + 0.02 * math.cos(2 * math.pi * (j + 0.5) / 2),
        )
        for j in range(2)
        for i in range(3)
    )

    result = solve_case(case, initial)

    assert result.steps == 1
    assert not result.completed
    assert math.isfinite(result.time)
    assert all(state.density > 0.0 and state.pressure > 0.0 for state in result.states)


def test_solution_cut_samples_cell_values_in_line_order():
    case = make_periodic_case(nx=2, length_x=2.0)
    states = (
        InitialState(density=1.0),
        InitialState(density=2.0),
    )

    samples = sample_solution_cut(case, states, "density", 0.0, 0.0, 0.5)

    assert samples == ((0.5, 1.0), (1.5, 2.0))


def test_solver_pause_callback_waits_at_step_boundary_and_resumes():
    case = make_periodic_case()
    case.numerics.end_time = 10.0
    case.numerics.max_iterations = 2
    pause_reached = Event()
    resume = Event()
    result_box = {}
    errors = []
    callback_count = 0

    def pause_callback():
        nonlocal callback_count
        callback_count += 1
        if callback_count == 2:
            pause_reached.set()
            if not resume.wait(timeout=5.0):
                raise TimeoutError("Test did not resume the solver.")

    def run_solver():
        try:
            result_box["result"] = solve_case(
                case,
                (InitialState(1.0, 0.1, 0.0, 1.0),),
                pause_callback=pause_callback,
            )
        except Exception as error:
            errors.append(error)

    thread = Thread(target=run_solver)
    thread.start()

    assert pause_reached.wait(timeout=5.0)
    assert "result" not in result_box
    resume.set()
    thread.join(timeout=5.0)

    assert not thread.is_alive()
    assert not errors
    assert result_box["result"].steps == 2


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


@pytest.mark.parametrize(
    (
        "wall_treatment",
        "omega_formulation",
        "time_integration",
        "viscosity_model",
    ),
    (
        ("Wall-resolved", "Standard SST", "Explicit RK2", "Constant"),
        ("Wall-resolved", "Standard SST", "Implicit Euler", "Constant"),
        ("Wall functions", "Standard SST", "Explicit RK2", "Constant"),
        ("Wall-resolved", "Regularized k-log-omega", "Explicit Euler", "Constant"),
        ("Wall-resolved", "Regularized k-log-omega", "Explicit RK2", "Constant"),
        ("Wall-resolved", "Regularized k-log-omega", "Explicit RK3", "Constant"),
        ("Wall-resolved", "Regularized k-log-omega", "Explicit RK4", "Constant"),
        ("Wall-resolved", "Regularized k-log-omega", "Implicit Euler", "Constant"),
        ("Wall-resolved", "Regularized k-log-omega", "Explicit RK2", "Sutherland"),
    ),
)
def test_sst_solver_advances_with_supported_wall_formulations(
    wall_treatment, omega_formulation, time_integration, viscosity_model
):
    case = make_periodic_case(2, 2)
    case.periodic_pairs.clear()
    for boundary in case.boundary_conditions:
        boundary.kind = "Wall"
        boundary.mechanical_condition = "No-slip"
        boundary.values = {"temperature": 1.0}
    inlet = case.boundary_conditions[3]
    inlet.kind = "Velocity Inlet"
    inlet.values = {
        "density": 1.0,
        "velocity_x": 0.1,
        "velocity_y": 0.0,
        "temperature": 1.0,
        "turbulence_intensity": 2.0,
        "turbulence_length_scale": 0.1,
    }
    outlet = case.boundary_conditions[1]
    outlet.kind = "Pressure Outlet"
    outlet.values = {"pressure": 1.0}
    case.numerics.viscosity_model = viscosity_model
    case.numerics.dynamic_viscosity = 1e-3
    case.numerics.gas_constant = 1.0
    case.numerics.reconstruction = "First order"
    case.numerics.turbulence_model = "Menter k-omega SST"
    case.numerics.turbulence_wall_treatment = wall_treatment
    case.numerics.turbulence_omega_formulation = omega_formulation
    case.numerics.time_integration = time_integration
    case.numerics.end_time = 1e-5
    case.numerics.max_iterations = 2

    progress = []
    result = solve_case(
        case,
        tuple(
            InitialState(
                1.0,
                0.0,
                0.0,
                1.0,
                turbulence_k=0.01,
                turbulence_omega=10.0,
            )
            for _cell in case.mesh.cells
        ),
        progress_callback=progress.append,
    )

    assert result.completed
    assert result.steps == 1
    assert progress
    assert all(len(item.residuals) == 6 for item in progress)
    assert all(
        math.isfinite(value) and value >= 0.0
        for item in progress
        for value in item.residuals[4:]
    )
    assert all(
        state.turbulence_k >= 0.0
        and state.turbulence_omega > 0.0
        and all(
            math.isfinite(value)
            for value in (
                state.density,
                state.pressure,
                state.turbulence_k,
                state.turbulence_omega,
            )
        )
        for state in result.states
    )


def test_geometry_builds_parabolic_stencil_only_when_requested(monkeypatch):
    case = make_periodic_case(3, 2)
    original = __import__(
        "meshcfd.core.solver", fromlist=["_parabolic_stencil"]
    )._parabolic_stencil
    calls = []

    def track_parabolic_stencil(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(
        "meshcfd.core.solver._parabolic_stencil", track_parabolic_stencil
    )

    case.numerics.reconstruction = "MUSCL linear"
    linear_geometry = _build_geometry(case.mesh, case)
    assert not calls
    assert linear_geometry.parabolic_sources.size == 0
    assert linear_geometry.parabolic_neighbours.size == 0
    assert linear_geometry.parabolic_dx.size == 0
    assert linear_geometry.parabolic_dy.size == 0

    case.numerics.reconstruction = "MUSCL parabolic"
    parabolic_geometry = _build_geometry(case.mesh, case)
    assert len(calls) == 1
    assert parabolic_geometry.parabolic_sources.size > 0
    assert parabolic_geometry.parabolic_neighbours.size > 0


def test_geometry_precomputes_interior_face_data():
    case = make_periodic_case(3, 2)
    case.numerics.reconstruction = "MUSCL linear"

    geometry = _build_geometry(case.mesh, case)
    faces = geometry.interior_faces
    owners = geometry.owners[faces]
    neighbours = geometry.neighbours[faces]

    np.testing.assert_array_equal(geometry.interior_owners, owners)
    np.testing.assert_array_equal(geometry.interior_neighbours, neighbours)
    np.testing.assert_allclose(geometry.inverse_areas, 1.0 / geometry.areas)
    np.testing.assert_allclose(geometry.interior_normals, geometry.face_normals[faces])
    np.testing.assert_allclose(geometry.interior_lengths, geometry.face_lengths[faces])
    np.testing.assert_allclose(
        geometry.interior_owner_offsets,
        geometry.face_centers[faces] - geometry.centers[owners],
    )
    np.testing.assert_allclose(
        geometry.interior_neighbour_offsets,
        geometry.face_centers[faces] - geometry.centers[neighbours],
    )


def test_interior_divergence_operator_matches_add_at_assembly():
    case = make_periodic_case(4, 3)
    case.numerics.reconstruction = "MUSCL linear"
    geometry = _build_geometry(case.mesh, case)
    face_count = len(geometry.interior_faces)
    flux = np.arange(face_count * 4, dtype=float).reshape(face_count, 4) / 17.0

    expected = np.zeros((len(case.mesh.cells), 4))
    integrated = flux * geometry.interior_lengths[:, None]
    np.add.at(
        expected,
        geometry.interior_owners,
        -integrated * geometry.inverse_areas[geometry.interior_owners, None],
    )
    np.add.at(
        expected,
        geometry.interior_neighbours,
        integrated * geometry.inverse_areas[geometry.interior_neighbours, None],
    )

    actual = geometry.interior_divergence_operator @ flux

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-13)


def test_compiled_limiter_matches_numpy_reference_for_all_schemes():
    case = make_periodic_case(5, 4)
    case.numerics.reconstruction = "MUSCL linear"
    geometry = _build_geometry(case.mesh, case)
    primitive = np.column_stack(
        (
            1.0 + 0.1 * np.sin(np.arange(20)),
            0.2 + 0.03 * np.cos(np.arange(20)),
            -0.1 + 0.02 * np.sin(0.7 * np.arange(20)),
            1.0 + 0.05 * np.cos(0.4 * np.arange(20)),
        )
    )
    gradients = np.empty((20, 4, 2))
    gradients[:, :, 0] = geometry.gradient_operator_x @ primitive
    gradients[:, :, 1] = geometry.gradient_operator_y @ primitive

    sources = geometry.gradient_sources
    neighbours = geometry.gradient_neighbours
    cells = geometry.limiter_cells
    local_min = primitive.copy()
    local_max = primitive.copy()
    np.minimum.at(local_min, sources, primitive[neighbours])
    np.maximum.at(local_max, sources, primitive[neighbours])
    increments = (
        gradients[cells, :, 0] * geometry.limiter_dx[:, None]
        + gradients[cells, :, 1] * geometry.limiter_dy[:, None]
    )
    cell_values = primitive[cells]
    tolerance = 1e-14 * np.maximum(np.abs(cell_values), 1.0)
    positive = increments > tolerance
    negative = increments < -tolerance
    ratio = np.ones_like(increments)
    np.divide(local_max[cells] - cell_values, increments, out=ratio, where=positive)
    np.divide(local_min[cells] - cell_values, increments, out=ratio, where=negative)
    ratio = np.maximum(ratio, 0.0)

    for limiter_code in range(4):
        if limiter_code <= 1:
            face_limiter = np.minimum(ratio, 1.0)
        elif limiter_code == 2:
            face_limiter = np.where(
                ratio > 0.0, np.minimum(1.0, 2.0 * ratio / (1.0 + ratio)), 0.0
            )
        else:
            epsilon = 1e-12
            face_limiter = (ratio**2 + 2.0 * ratio + epsilon) / (
                ratio**2 + ratio + 2.0 + epsilon
            )
            face_limiter = np.minimum(face_limiter, 1.0)
        expected = np.ones((20, 4))
        np.minimum.at(expected, cells, face_limiter)

        actual = limited_gradient_factors(
            primitive,
            gradients,
            sources,
            neighbours,
            cells,
            geometry.limiter_dx,
            geometry.limiter_dy,
            limiter_code,
        )

        np.testing.assert_array_equal(actual, expected)


def test_compiled_reconstruction_hllc_matches_vectorized_reference():
    case = make_periodic_case(6, 5)
    case.numerics.reconstruction = "MUSCL linear"
    geometry = _build_geometry(case.mesh, case)
    primitive = np.column_stack(
        (
            1.0 + 0.03 * np.sin(np.arange(30)),
            0.25 + 0.05 * np.cos(0.3 * np.arange(30)),
            -0.04 + 0.02 * np.sin(0.7 * np.arange(30)),
            1.0 + 0.04 * np.cos(0.5 * np.arange(30)),
        )
    )
    gradients, factors = _limited_gradients(
        primitive, geometry, case.numerics.limiter
    )
    owners = geometry.interior_owners
    neighbours = geometry.interior_neighbours
    left = _reconstruct(
        primitive, gradients, factors, owners, geometry.interior_owner_offsets
    )
    right = _reconstruct(
        primitive, gradients, factors, neighbours, geometry.interior_neighbour_offsets
    )
    expected = numerical_flux(
        left, right, geometry.interior_normals, case.numerics.gamma, "HLLC"
    )

    actual = reconstruct_hllc_flux(
        primitive,
        gradients,
        factors,
        owners,
        neighbours,
        geometry.interior_owner_offsets,
        geometry.interior_neighbour_offsets,
        geometry.interior_normals,
        case.numerics.gamma,
    )

    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-14)


def test_compiled_weighted_least_squares_matches_numpy_reference():
    source_ids = np.array((0, 0, 1, 1, 1, 2, 2, 2), dtype=int)
    dx = np.array((1.0, 0.0, -1.0, 1.0, 0.0, -1.0, 0.0, 1.0))
    dy = np.array((0.0, 1.0, 0.0, 0.0, 1.0, 0.0, -1.0, 1.0))
    differences = np.column_stack(
        (
            np.linspace(-0.4, 0.7, len(source_ids)),
            np.linspace(0.8, -0.2, len(source_ids)),
            np.sin(np.arange(len(source_ids))),
        )
    )
    weights = 1.0 / np.maximum(dx * dx + dy * dy, 1e-300)
    xx = np.zeros(3)
    xy = np.zeros(3)
    yy = np.zeros(3)
    rhs_x = np.zeros((3, 3))
    rhs_y = np.zeros((3, 3))
    np.add.at(xx, source_ids, weights * dx * dx)
    np.add.at(xy, source_ids, weights * dx * dy)
    np.add.at(yy, source_ids, weights * dy * dy)
    np.add.at(rhs_x, source_ids, (weights * dx)[:, None] * differences)
    np.add.at(rhs_y, source_ids, (weights * dy)[:, None] * differences)
    expected = np.zeros((3, 3, 2))
    determinant = xx * yy - xy * xy
    valid = determinant > 1e-14 * np.maximum(xx * yy, 1e-300)
    expected[valid, :, 0] = (
        rhs_x[valid] * yy[valid, None] - rhs_y[valid] * xy[valid, None]
    ) / determinant[valid, None]
    expected[valid, :, 1] = (
        rhs_y[valid] * xx[valid, None] - rhs_x[valid] * xy[valid, None]
    ) / determinant[valid, None]

    actual = weighted_least_squares_gradients(
        3, source_ids, dx, dy, differences
    )

    np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=2e-15)


def test_interior_divergence_operator_matches_diffusive_add_at_sign():
    case = make_periodic_case(4, 3)
    geometry = _build_geometry(case.mesh, case)
    face_count = len(geometry.interior_faces)
    flux = np.arange(face_count * 2, dtype=float).reshape(face_count, 2) / 19.0
    integrated = flux * geometry.interior_lengths[:, None]
    expected = np.zeros((len(case.mesh.cells), 2))
    np.add.at(
        expected,
        geometry.interior_owners,
        integrated * geometry.inverse_areas[geometry.interior_owners, None],
    )
    np.add.at(
        expected,
        geometry.interior_neighbours,
        -integrated * geometry.inverse_areas[geometry.interior_neighbours, None],
    )

    actual = -(geometry.interior_divergence_operator @ flux)

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-13)


def test_compiled_point_to_segment_distances_match_reference():
    points = np.array(((0.5, 0.5), (2.0, 0.0), (-1.0, 0.5), (0.25, 2.0)))
    starts = np.array(((0.0, 0.0), (0.0, 1.0), (1.0, 1.0)))
    ends = np.array(((1.0, 0.0), (0.0, 2.0), (2.0, 1.0)))
    expected = np.full(len(points), np.inf)
    for start, end in zip(starts, ends):
        edge = end - start
        offsets = points - start
        projection = np.clip(np.sum(offsets * edge, axis=1) / np.dot(edge, edge), 0.0, 1.0)
        expected = np.minimum(expected, np.linalg.norm(offsets - projection[:, None] * edge, axis=1))
    actual = point_to_segments_minimum_distance(points, starts, ends)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-15)


def test_fused_linear_gradients_and_limiter_match_reference():
    case = make_periodic_case(6, 5)
    case.numerics.reconstruction = "MUSCL linear"
    geometry = _build_geometry(case.mesh, case)
    primitive = np.column_stack(
        (
            1.0 + 0.03 * np.sin(np.arange(30)),
            0.25 + 0.05 * np.cos(0.3 * np.arange(30)),
            -0.04 + 0.02 * np.sin(0.7 * np.arange(30)),
            1.0 + 0.04 * np.cos(0.5 * np.arange(30)),
        )
    )
    expected_gradients = np.zeros((30, 4, 2))
    expected_gradients[:, :, 0] = geometry.gradient_operator_x @ primitive
    expected_gradients[:, :, 1] = geometry.gradient_operator_y @ primitive
    expected_factors = limited_gradient_factors(
        primitive,
        expected_gradients,
        geometry.gradient_sources,
        geometry.gradient_neighbours,
        geometry.limiter_cells,
        geometry.limiter_dx,
        geometry.limiter_dy,
        0,
    )

    actual_gradients, actual_factors = linear_gradients_and_factors(
        primitive,
        geometry.gradient_sources,
        geometry.gradient_neighbours,
        geometry.gradient_coefficients_x,
        geometry.gradient_coefficients_y,
        geometry.limiter_cells,
        geometry.limiter_dx,
        geometry.limiter_dy,
        0,
    )

    np.testing.assert_allclose(
        actual_gradients, expected_gradients, rtol=2e-14, atol=2e-14
    )
    np.testing.assert_allclose(
        actual_factors, expected_factors, rtol=2e-13, atol=2e-13
    )
