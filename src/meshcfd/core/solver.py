"""Explicit finite-volume solver for compressible ideal-gas flow."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from meshcfd.core.euler import (
    conservative_to_primitive,
    numerical_flux,
    primitive_to_conservative,
)
from meshcfd.core.initial_conditions import evaluate_initial_conditions
from meshcfd.core.model import (
    Case,
    FiniteVolumeMesh,
    InitialState,
    NumericalSettings,
    PeriodicBoundaryPair,
)
from meshcfd.core.periodic import detect_periodic_pair, match_periodic_faces


class SolverError(ValueError):
    """Raised when a case cannot be advanced by the Euler solver."""


@dataclass(frozen=True)
class SolverProgress:
    step: int
    time: float
    time_step: float
    residuals: tuple[float, float, float, float]
    states: tuple[InitialState, ...] | None = None


@dataclass(frozen=True)
class SolverResult:
    states: tuple[InitialState, ...]
    time: float
    steps: int
    residual: float
    completed: bool


@dataclass(frozen=True)
class _PressureInlet:
    faces: np.ndarray
    total_pressure: float
    total_temperature: float
    direction: np.ndarray
    patch_name: str


@dataclass(frozen=True)
class _MassFlowInlet:
    faces: np.ndarray
    mass_flow: float
    total_temperature: float
    direction: np.ndarray
    patch_name: str


@dataclass(frozen=True)
class _MassFlowOutlet:
    faces: np.ndarray
    mass_flow: float
    patch_name: str


@dataclass(frozen=True)
class _WallThermalBoundary:
    faces: np.ndarray
    condition: str
    value: float
    patch_name: str
    mechanical_condition: str


@dataclass(frozen=True)
class _Geometry:
    centers: np.ndarray
    areas: np.ndarray
    face_centers: np.ndarray
    face_normals: np.ndarray
    face_lengths: np.ndarray
    owners: np.ndarray
    neighbours: np.ndarray
    patch_ids: np.ndarray
    patch_kinds: dict[int, str]
    interior_faces: np.ndarray
    wall_faces: np.ndarray
    symmetry_faces: np.ndarray
    diffusive_boundary_faces: np.ndarray
    periodic_faces_a: np.ndarray
    periodic_faces_b: np.ndarray
    periodic_pairs: tuple[PeriodicBoundaryPair, ...]
    velocity_inlet_faces: np.ndarray
    velocity_inlet_states: np.ndarray
    pressure_outlet_faces: np.ndarray
    pressure_outlet_pressures: np.ndarray
    far_field_faces: np.ndarray
    far_field_states: np.ndarray
    pressure_inlets: tuple[_PressureInlet, ...]
    mass_flow_inlets: tuple[_MassFlowInlet, ...]
    mass_flow_outlets: tuple[_MassFlowOutlet, ...]
    wall_thermal_boundaries: tuple[_WallThermalBoundary, ...]
    gradient_sources: np.ndarray
    gradient_neighbours: np.ndarray
    gradient_dx: np.ndarray
    gradient_dy: np.ndarray
    parabolic_sources: np.ndarray
    parabolic_neighbours: np.ndarray
    parabolic_dx: np.ndarray
    parabolic_dy: np.ndarray
    limiter_cells: np.ndarray
    limiter_dx: np.ndarray
    limiter_dy: np.ndarray


def solve_case(
    case: Case,
    initial_states: tuple[InitialState, ...] | None = None,
    progress_callback: Callable[[SolverProgress], None] | None = None,
    cancel_callback: Callable[[], bool] | None = None,
    live_solution_updates: bool = False,
    start_time: float = 0.0,
    step_offset: int = 0,
) -> SolverResult:
    """Advance a case from its current time to its end time or iteration limit."""
    if case.mesh is None:
        raise SolverError("Import a mesh before running the solver.")
    settings = case.numerics
    if settings.reconstruction not in (
        "First order",
        "MUSCL linear",
        "MUSCL parabolic",
    ):
        raise SolverError(f"Unsupported reconstruction '{settings.reconstruction}'.")
    if settings.limiter not in (
        "Barth-Jespersen",
        "Venkatakrishnan",
        "Minmod",
        "Van Leer",
    ):
        raise SolverError(f"Unsupported limiter '{settings.limiter}'.")
    if settings.flux not in ("HLLC", "Roe", "Rusanov"):
        raise SolverError(f"Unsupported convective flux '{settings.flux}'.")
    if settings.time_integration not in (
        "Explicit Euler",
        "Explicit RK2",
        "Explicit RK3",
    ):
        raise SolverError(
            f"Unsupported time integration '{settings.time_integration}'."
        )
    if not math.isfinite(settings.cfl) or not 0.0 < settings.cfl <= 1.0:
        raise SolverError("The explicit Euler solver requires a CFL number in (0, 1].")
    if (
        isinstance(settings.max_iterations, bool)
        or not isinstance(settings.max_iterations, int)
        or settings.max_iterations <= 0
    ):
        raise SolverError("Maximum iterations must be a positive integer.")
    if not math.isfinite(settings.end_time) or settings.end_time <= 0.0:
        raise SolverError("End time must be positive and finite.")
    if not math.isfinite(start_time) or start_time < 0.0:
        raise SolverError("Current solution time must be finite and non-negative.")
    if start_time >= settings.end_time:
        raise SolverError(
            "End time must be greater than the current solution time to continue."
        )
    if (
        isinstance(step_offset, bool)
        or not isinstance(step_offset, int)
        or step_offset < 0
    ):
        raise SolverError("Current solution step must be a non-negative integer.")
    if not math.isfinite(settings.gamma) or settings.gamma <= 1.0:
        raise SolverError("Specific heat ratio must be finite and greater than 1.")
    if not math.isfinite(settings.gas_constant) or settings.gas_constant <= 0.0:
        raise SolverError("Gas constant must be positive and finite.")
    if settings.viscosity_model not in ("Inviscid", "Constant", "Sutherland"):
        raise SolverError("Unsupported viscosity model.")
    for value, description in (
        (settings.dynamic_viscosity, "Dynamic viscosity"),
        (settings.sutherland_reference_viscosity, "Sutherland reference viscosity"),
        (
            settings.sutherland_reference_temperature,
            "Sutherland reference temperature",
        ),
        (settings.sutherland_temperature, "Sutherland temperature"),
        (settings.prandtl_number, "Prandtl number"),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise SolverError(f"{description} must be positive and finite.")
    if not math.isfinite(settings.acceleration_x) or not math.isfinite(
        settings.acceleration_y
    ):
        raise SolverError("Body-force acceleration must be finite.")

    mesh = case.mesh
    geometry = _build_geometry(mesh, case)
    if initial_states is None:
        try:
            initial_states = evaluate_initial_conditions(mesh, case.initial_conditions)
        except ValueError as error:
            raise SolverError(f"Could not evaluate initial conditions: {error}") from error
    if len(initial_states) != len(mesh.cells):
        raise SolverError("Initial state count must match the mesh cell count.")
    primitive = np.array(
        [
            (state.density, state.velocity_x, state.velocity_y, state.pressure)
            for state in initial_states
        ],
        dtype=float,
    )
    try:
        conservative = primitive_to_conservative(primitive, settings.gamma)
        conservative_to_primitive(conservative, settings.gamma)
    except ValueError as error:
        raise SolverError(f"Invalid initial state: {error}") from error

    time = start_time
    residual = 0.0
    completed = False
    steps = step_offset
    for local_step in range(1, settings.max_iterations + 1):
        step = step_offset + local_step
        if cancel_callback is not None and cancel_callback():
            break
        try:
            primitive = conservative_to_primitive(conservative, settings.gamma)
            time_step = min(
                _stable_time_step(primitive, geometry, settings),
                settings.end_time - time,
            )
            if not math.isfinite(time_step) or time_step <= 0.0:
                raise SolverError("The stable time step is non-positive or non-finite.")
            start = conservative
            derivative = _residual(start, geometry, settings)
            if settings.time_integration == "Explicit Euler":
                conservative = start + time_step * derivative
            else:
                stage_one = start + time_step * derivative
                derivative_one = _residual(stage_one, geometry, settings)
                if settings.time_integration == "Explicit RK2":
                    conservative = 0.5 * start + 0.5 * (
                        stage_one + time_step * derivative_one
                    )
                else:
                    stage_two = 0.75 * start + 0.25 * (
                        stage_one + time_step * derivative_one
                    )
                    derivative_two = _residual(stage_two, geometry, settings)
                    conservative = (1.0 / 3.0) * start + (2.0 / 3.0) * (
                        stage_two + time_step * derivative_two
                    )
            conservative_to_primitive(conservative, settings.gamma)
        except (FloatingPointError, ValueError, ZeroDivisionError) as error:
            raise SolverError(f"Solver failed at step {step}: {error}") from error
        time += time_step
        steps = step
        residual = float(np.sqrt(np.mean(derivative**2)))
        if progress_callback is not None:
            progress_states = None
            if live_solution_updates and (
                local_step == 1
                or step % 10 == 0
                or time >= settings.end_time * (1.0 - 1e-13)
                or local_step == settings.max_iterations
            ):
                progress_primitive = conservative_to_primitive(
                    conservative, settings.gamma
                )
                progress_states = tuple(
                    InitialState(*row) for row in progress_primitive.tolist()
                )
            component_residuals = tuple(
                float(value)
                for value in np.sqrt(np.mean(derivative**2, axis=0))
            )
            progress_callback(
                SolverProgress(
                    step,
                    time,
                    time_step,
                    component_residuals,
                    progress_states,
                )
            )
        if time >= settings.end_time * (1.0 - 1e-13):
            time = settings.end_time
            completed = True
            break

    final_primitive = conservative_to_primitive(conservative, settings.gamma)
    try:
        final_residual = _residual(conservative, geometry, settings)
    except ValueError as error:
        raise SolverError(f"Could not evaluate final residual: {error}") from error
    residual = float(np.sqrt(np.mean(final_residual**2)))
    states = tuple(InitialState(*row) for row in final_primitive.tolist())
    return SolverResult(states, time, steps, residual, completed)


def _build_geometry(mesh: FiniteVolumeMesh, case: Case) -> _Geometry:
    cell_centers = []
    cell_areas = []
    for cell_id, node_ids in enumerate(mesh.cells):
        points = [mesh.points[node_id] for node_id in node_ids]
        twice_area = sum(
            x0 * y1 - x1 * y0
            for (x0, y0), (x1, y1) in zip(points, points[1:] + points[:1])
        )
        if not math.isfinite(twice_area) or abs(twice_area) <= 1e-15:
            raise SolverError(f"Cell {cell_id} has zero or invalid area.")
        x_sum = 0.0
        y_sum = 0.0
        for (x0, y0), (x1, y1) in zip(points, points[1:] + points[:1]):
            cross = x0 * y1 - x1 * y0
            x_sum += (x0 + x1) * cross
            y_sum += (y0 + y1) * cross
        cell_centers.append(
            (x_sum / (3.0 * twice_area), y_sum / (3.0 * twice_area))
        )
        cell_areas.append(0.5 * abs(twice_area))
    centers = np.asarray(cell_centers, dtype=float)
    areas = np.asarray(cell_areas, dtype=float)

    face_centers = []
    face_normals = []
    face_lengths = []
    for face_id, (node_a, node_b) in enumerate(mesh.faces):
        a = np.asarray(mesh.points[node_a], dtype=float)
        b = np.asarray(mesh.points[node_b], dtype=float)
        edge = b - a
        length = float(np.linalg.norm(edge))
        if not math.isfinite(length) or length <= 1e-15:
            raise SolverError(f"Face {face_id} has zero or invalid length.")
        normal = np.array((edge[1], -edge[0]), dtype=float) / length
        center = 0.5 * (a + b)
        owner = mesh.face_owners[face_id]
        if float(np.dot(normal, center - centers[owner])) < 0.0:
            normal = -normal
        face_centers.append(center)
        face_normals.append(normal)
        face_lengths.append(length)

    owners = np.asarray(mesh.face_owners, dtype=int)
    neighbours = np.asarray(mesh.face_neighbours, dtype=int)
    patch_ids = np.asarray(mesh.face_patch_ids, dtype=int)
    interior_faces = np.flatnonzero(neighbours >= 0)
    boundary_faces = np.flatnonzero(neighbours < 0)
    conditions = {condition.patch_id: condition for condition in case.boundary_conditions}
    pairs_by_patch: dict[int, PeriodicBoundaryPair] = {}
    periodic_mappings = []
    for pair in case.periodic_pairs:
        try:
            axis, pitch = detect_periodic_pair(mesh, pair.patch_a_id, pair.patch_b_id)
            matched_faces, shift = match_periodic_faces(
                mesh, pair.patch_a_id, pair.patch_b_id
            )
        except ValueError as error:
            raise SolverError(f"Invalid periodic boundary pair: {error}") from error
        if pair.axis != axis or not math.isclose(
            pair.pitch, pitch, rel_tol=1e-9, abs_tol=max(pitch, 1.0) * 1e-9
        ):
            raise SolverError("A periodic pair has a stale axis or pitch.")
        if (
            pair.patch_a_id not in conditions
            or pair.patch_b_id not in conditions
            or conditions[pair.patch_a_id].kind != "Periodic"
            or conditions[pair.patch_b_id].kind != "Periodic"
        ):
            raise SolverError("Both patches in every periodic pair must be Periodic.")
        if pair.patch_a_id in pairs_by_patch or pair.patch_b_id in pairs_by_patch:
            raise SolverError("A patch cannot belong to multiple periodic pairs.")
        pairs_by_patch[pair.patch_a_id] = pair
        pairs_by_patch[pair.patch_b_id] = pair
        periodic_mappings.extend((face_a, face_b, shift) for face_a, face_b in matched_faces)

    periodic_patch_ids = {
        patch_id for patch_id, condition in conditions.items()
        if condition.kind == "Periodic"
    }
    if periodic_patch_ids != set(pairs_by_patch):
        raise SolverError("Each Periodic boundary must belong to exactly one pair.")
    if set(conditions) != {patch.id for patch in mesh.patches}:
        raise SolverError("Boundary conditions must be defined for every mesh patch.")
    for patch_id, condition in conditions.items():
        if condition.kind not in (
            "Periodic",
            "Wall",
            "Symmetry",
            "Velocity Inlet",
            "Pressure Inlet",
            "Pressure Outlet",
            "Mass Flow Inlet",
            "Mass Flow Outlet",
            "Far Field",
        ):
            raise SolverError(
                f"Boundary type '{condition.kind}' is not implemented by the solver."
            )
        if condition.kind == "Periodic" and patch_id not in pairs_by_patch:
            raise SolverError(f"Periodic patch '{condition.patch_name}' has no pair.")
    periodic_face_set = {
        face_id for face_a, face_b, _shift in periodic_mappings
        for face_id in (face_a, face_b)
    }
    boundary_face_set = set(boundary_faces.tolist())
    if not periodic_face_set <= boundary_face_set:
        raise SolverError("Periodic pairs must reference boundary faces.")
    if any(
        int(patch_ids[face_id]) in periodic_patch_ids
        and face_id not in periodic_face_set
        for face_id in boundary_faces
    ):
        raise SolverError("Periodic patches contain unmatched faces.")
    wall_faces = np.array(
        [
            face_id
            for face_id in boundary_faces
            if int(face_id) not in periodic_face_set
            and conditions[int(patch_ids[face_id])].kind in ("Wall", "Symmetry")
        ],
        dtype=int,
    )
    symmetry_faces = np.array(
        [
            face_id
            for face_id in boundary_faces
            if int(face_id) not in periodic_face_set
            and conditions[int(patch_ids[face_id])].kind == "Symmetry"
        ],
        dtype=int,
    )
    diffusive_boundary_faces = np.asarray(
        [face_id for face_id in boundary_faces if face_id not in periodic_face_set],
        dtype=int,
    )
    velocity_inlet_faces = []
    velocity_inlet_states = []
    pressure_outlet_faces = []
    pressure_outlet_pressures = []
    far_field_faces = []
    far_field_states = []
    pressure_inlets = []
    mass_flow_inlets = []
    mass_flow_outlets = []
    wall_thermal_boundaries = []
    for patch_id, condition in conditions.items():
        faces = np.asarray(
            [
                face_id
                for face_id in boundary_faces
                if int(patch_ids[face_id]) == patch_id
            ],
            dtype=int,
        )
        if condition.kind == "Velocity Inlet":
            density = _positive_boundary_value(
                condition.values.get("density", 1.225),
                f"Density for velocity inlet '{condition.patch_name}'",
            )
            temperature = _positive_boundary_value(
                condition.values.get("temperature", 288.15),
                f"Temperature for velocity inlet '{condition.patch_name}'",
            )
            velocity_x = _finite_boundary_value(
                condition.values.get("velocity_x", 0.0),
                f"Velocity X for inlet '{condition.patch_name}'",
            )
            velocity_y = _finite_boundary_value(
                condition.values.get("velocity_y", 0.0),
                f"Velocity Y for inlet '{condition.patch_name}'",
            )
            pressure = density * case.numerics.gas_constant * temperature
            if not math.isfinite(pressure) or pressure <= 0.0:
                raise SolverError(
                    f"Velocity inlet '{condition.patch_name}' produces invalid pressure."
                )
            velocity_inlet_faces.extend(faces.tolist())
            velocity_inlet_states.extend(
                [(density, velocity_x, velocity_y, pressure)] * len(faces)
            )
        elif condition.kind == "Wall" and case.numerics.viscosity_model != "Inviscid":
            mechanical_condition = condition.mechanical_condition
            if mechanical_condition not in ("Slip", "No-slip"):
                raise SolverError(
                    f"Unsupported mechanical condition on wall "
                    f"'{condition.patch_name}'."
                )
            thermal_condition = condition.thermal_condition
            if thermal_condition == "Temperature":
                value = _positive_boundary_value(
                    condition.values.get("temperature", 300.0),
                    f"Wall temperature for '{condition.patch_name}'",
                )
            elif thermal_condition == "Heat Flux":
                value = _finite_boundary_value(
                    condition.values.get("heat_flux", 0.0),
                    f"Wall heat flux for '{condition.patch_name}'",
                )
            else:
                raise SolverError(
                    f"Unsupported thermal condition on wall '{condition.patch_name}'."
                )
            wall_thermal_boundaries.append(
                _WallThermalBoundary(
                    faces,
                    thermal_condition,
                    value,
                    condition.patch_name,
                    mechanical_condition,
                )
            )
        elif condition.kind == "Pressure Outlet":
            pressure = _positive_boundary_value(
                condition.values.get("pressure", 101325.0),
                f"Static pressure for outlet '{condition.patch_name}'",
            )
            pressure_outlet_faces.extend(faces.tolist())
            pressure_outlet_pressures.extend([pressure] * len(faces))
        elif condition.kind == "Pressure Inlet":
            total_pressure = _positive_boundary_value(
                condition.values.get("pressure", 101325.0),
                f"Total pressure for inlet '{condition.patch_name}'",
            )
            total_temperature = _positive_boundary_value(
                condition.values.get("temperature", 288.15),
                f"Total temperature for inlet '{condition.patch_name}'",
            )
            direction = _boundary_flow_direction(
                condition.values.get("velocity_x", 1.0),
                condition.values.get("velocity_y", 0.0),
                f"Pressure inlet '{condition.patch_name}'",
            )
            pressure_inlets.append(
                _PressureInlet(
                    faces,
                    total_pressure,
                    total_temperature,
                    direction,
                    condition.patch_name,
                )
            )
        elif condition.kind == "Mass Flow Inlet":
            mass_flow = _positive_boundary_value(
                condition.values.get("mass_flow", 1.0),
                f"Mass flow for inlet '{condition.patch_name}'",
            )
            total_temperature = _positive_boundary_value(
                condition.values.get("temperature", 288.15),
                f"Total temperature for inlet '{condition.patch_name}'",
            )
            direction = _boundary_flow_direction(
                condition.values.get("velocity_x", 1.0),
                condition.values.get("velocity_y", 0.0),
                f"Mass-flow inlet '{condition.patch_name}'",
            )
            mass_flow_inlets.append(
                _MassFlowInlet(
                    faces,
                    mass_flow,
                    total_temperature,
                    direction,
                    condition.patch_name,
                )
            )
        elif condition.kind == "Mass Flow Outlet":
            mass_flow = _positive_boundary_value(
                condition.values.get("mass_flow", 1.0),
                f"Mass flow for outlet '{condition.patch_name}'",
            )
            mass_flow_outlets.append(_MassFlowOutlet(faces, mass_flow, condition.patch_name))
        elif condition.kind == "Far Field":
            density = _positive_boundary_value(
                condition.values.get("density", 1.225),
                f"Density for far field '{condition.patch_name}'",
            )
            velocity_x = _finite_boundary_value(
                condition.values.get("velocity_x", 0.0),
                f"Velocity X for far field '{condition.patch_name}'",
            )
            velocity_y = _finite_boundary_value(
                condition.values.get("velocity_y", 0.0),
                f"Velocity Y for far field '{condition.patch_name}'",
            )
            pressure = _positive_boundary_value(
                condition.values.get("pressure", 101325.0),
                f"Pressure for far field '{condition.patch_name}'",
            )
            far_field_faces.extend(faces.tolist())
            far_field_states.extend(
                [(density, velocity_x, velocity_y, pressure)] * len(faces)
            )

    gradient_sources = []
    gradient_neighbours = []
    gradient_dx = []
    gradient_dy = []
    for face_id in interior_faces:
        owner, neighbour = int(owners[face_id]), int(neighbours[face_id])
        delta = centers[neighbour] - centers[owner]
        gradient_sources.extend((owner, neighbour))
        gradient_neighbours.extend((neighbour, owner))
        gradient_dx.extend((delta[0], -delta[0]))
        gradient_dy.extend((delta[1], -delta[1]))
    for face_a, face_b, shift in periodic_mappings:
        owner_a, owner_b = int(owners[face_a]), int(owners[face_b])
        delta = centers[owner_b] - np.asarray(shift) - centers[owner_a]
        gradient_sources.extend((owner_a, owner_b))
        gradient_neighbours.extend((owner_b, owner_a))
        gradient_dx.extend((delta[0], -delta[0]))
        gradient_dy.extend((delta[1], -delta[1]))
    (
        parabolic_sources,
        parabolic_neighbours,
        parabolic_dx,
        parabolic_dy,
    ) = _parabolic_stencil(
        len(mesh.cells),
        gradient_sources,
        gradient_neighbours,
        gradient_dx,
        gradient_dy,
        max(
            max(x for x, _ in mesh.points) - min(x for x, _ in mesh.points),
            max(y for _, y in mesh.points) - min(y for _, y in mesh.points),
            1e-12,
        ),
    )

    limiter_cells = list(owners)
    limiter_dx = list(np.asarray(face_centers)[:, 0] - centers[owners, 0])
    limiter_dy = list(np.asarray(face_centers)[:, 1] - centers[owners, 1])
    for face_id in interior_faces:
        neighbour = int(neighbours[face_id])
        limiter_cells.append(neighbour)
        limiter_dx.append(face_centers[face_id][0] - centers[neighbour, 0])
        limiter_dy.append(face_centers[face_id][1] - centers[neighbour, 1])

    return _Geometry(
        centers=centers,
        areas=areas,
        face_centers=np.asarray(face_centers),
        face_normals=np.asarray(face_normals),
        face_lengths=np.asarray(face_lengths),
        owners=owners,
        neighbours=neighbours,
        patch_ids=patch_ids,
        patch_kinds={patch_id: condition.kind for patch_id, condition in conditions.items()},
        interior_faces=interior_faces,
        wall_faces=wall_faces,
        symmetry_faces=symmetry_faces,
        diffusive_boundary_faces=diffusive_boundary_faces,
        periodic_faces_a=np.asarray([item[0] for item in periodic_mappings], dtype=int),
        periodic_faces_b=np.asarray([item[1] for item in periodic_mappings], dtype=int),
        periodic_pairs=tuple(case.periodic_pairs),
        velocity_inlet_faces=np.asarray(velocity_inlet_faces, dtype=int),
        velocity_inlet_states=np.asarray(velocity_inlet_states, dtype=float).reshape(
            (-1, 4)
        ),
        pressure_outlet_faces=np.asarray(pressure_outlet_faces, dtype=int),
        pressure_outlet_pressures=np.asarray(
            pressure_outlet_pressures, dtype=float
        ),
        far_field_faces=np.asarray(far_field_faces, dtype=int),
        far_field_states=np.asarray(far_field_states, dtype=float).reshape((-1, 4)),
        pressure_inlets=tuple(pressure_inlets),
        mass_flow_inlets=tuple(mass_flow_inlets),
        mass_flow_outlets=tuple(mass_flow_outlets),
        wall_thermal_boundaries=tuple(wall_thermal_boundaries),
        gradient_sources=np.asarray(gradient_sources, dtype=int),
        gradient_neighbours=np.asarray(gradient_neighbours, dtype=int),
        gradient_dx=np.asarray(gradient_dx, dtype=float),
        gradient_dy=np.asarray(gradient_dy, dtype=float),
        parabolic_sources=parabolic_sources,
        parabolic_neighbours=parabolic_neighbours,
        parabolic_dx=parabolic_dx,
        parabolic_dy=parabolic_dy,
        limiter_cells=np.asarray(limiter_cells, dtype=int),
        limiter_dx=np.asarray(limiter_dx, dtype=float),
        limiter_dy=np.asarray(limiter_dy, dtype=float),
    )


def _parabolic_stencil(
    cell_count: int,
    sources: list[int],
    neighbours: list[int],
    dx: list[float],
    dy: list[float],
    domain_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    adjacency: list[list[tuple[int, float, float]]] = [
        [] for _ in range(cell_count)
    ]
    for source, neighbour, delta_x, delta_y in zip(
        sources, neighbours, dx, dy
    ):
        adjacency[source].append((neighbour, delta_x, delta_y))

    stencil_sources = []
    stencil_neighbours = []
    stencil_dx = []
    stencil_dy = []
    for source in range(cell_count):
        frontier = [(source, 0.0, 0.0)]
        seen = {(source, 0, 0)}
        for _depth in range(2):
            next_frontier = []
            for current, offset_x, offset_y in frontier:
                for neighbour, edge_x, edge_y in adjacency[current]:
                    position_x = offset_x + edge_x
                    position_y = offset_y + edge_y
                    key = (
                        neighbour,
                        round(position_x / domain_scale, 10),
                        round(position_y / domain_scale, 10),
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    next_frontier.append((neighbour, position_x, position_y))
                    if math.hypot(position_x, position_y) > domain_scale * 1e-12:
                        stencil_sources.append(source)
                        stencil_neighbours.append(neighbour)
                        stencil_dx.append(position_x)
                        stencil_dy.append(position_y)
            frontier = next_frontier

    return (
        np.asarray(stencil_sources, dtype=int),
        np.asarray(stencil_neighbours, dtype=int),
        np.asarray(stencil_dx, dtype=float),
        np.asarray(stencil_dy, dtype=float),
    )


def _limited_gradients(
    primitive: np.ndarray, geometry: _Geometry, limiter: str
) -> tuple[np.ndarray, np.ndarray]:
    cell_count = len(primitive)
    gradients = np.zeros((cell_count, 4, 2), dtype=float)
    sources = geometry.gradient_sources
    if len(sources) == 0:
        return gradients, np.ones((cell_count, 4), dtype=float)
    neighbours = geometry.gradient_neighbours
    dx, dy = geometry.gradient_dx, geometry.gradient_dy
    distance_squared = dx * dx + dy * dy
    weights = 1.0 / np.maximum(distance_squared, 1e-300)
    matrix_xx = np.zeros(cell_count)
    matrix_xy = np.zeros(cell_count)
    matrix_yy = np.zeros(cell_count)
    np.add.at(matrix_xx, sources, weights * dx * dx)
    np.add.at(matrix_xy, sources, weights * dx * dy)
    np.add.at(matrix_yy, sources, weights * dy * dy)
    difference = primitive[neighbours] - primitive[sources]
    rhs_x = np.zeros((cell_count, 4))
    rhs_y = np.zeros((cell_count, 4))
    np.add.at(rhs_x, sources, (weights * dx)[:, None] * difference)
    np.add.at(rhs_y, sources, (weights * dy)[:, None] * difference)
    determinant = matrix_xx * matrix_yy - matrix_xy * matrix_xy
    valid = determinant > 1e-14 * np.maximum(matrix_xx * matrix_yy, 1e-300)
    gradients[valid, :, 0] = (
        rhs_x[valid] * matrix_yy[valid, None]
        - rhs_y[valid] * matrix_xy[valid, None]
    ) / determinant[valid, None]
    gradients[valid, :, 1] = (
        rhs_y[valid] * matrix_xx[valid, None]
        - rhs_x[valid] * matrix_xy[valid, None]
    ) / determinant[valid, None]

    local_min = primitive.copy()
    local_max = primitive.copy()
    np.minimum.at(local_min, sources, primitive[neighbours])
    np.maximum.at(local_max, sources, primitive[neighbours])
    cells = geometry.limiter_cells
    increments = (
        gradients[cells, :, 0] * geometry.limiter_dx[:, None]
        + gradients[cells, :, 1] * geometry.limiter_dy[:, None]
    )
    cell_values = primitive[cells]
    positive = increments > 1e-14 * np.maximum(np.abs(cell_values), 1.0)
    negative = increments < -1e-14 * np.maximum(np.abs(cell_values), 1.0)
    ratio = np.ones_like(increments)
    upper = np.broadcast_to(local_max[cells] - cell_values, increments.shape)
    lower = np.broadcast_to(local_min[cells] - cell_values, increments.shape)
    np.divide(upper, increments, out=ratio, where=positive)
    np.divide(lower, increments, out=ratio, where=negative)
    ratio = np.maximum(ratio, 0.0)
    if limiter in ("Barth-Jespersen", "Minmod"):
        face_limiter = np.minimum(ratio, 1.0)
    elif limiter == "Van Leer":
        face_limiter = np.where(
            ratio > 0.0, np.minimum(1.0, 2.0 * ratio / (1.0 + ratio)), 0.0
        )
    else:
        epsilon = 1e-12
        face_limiter = (
            ratio**2 + 2.0 * ratio + epsilon
        ) / (ratio**2 + ratio + 2.0 + epsilon)
        face_limiter = np.minimum(face_limiter, 1.0)
    factors = np.ones((cell_count, 4), dtype=float)
    np.minimum.at(factors, cells, face_limiter)
    return gradients, factors


def _parabolic_reconstruction(
    primitive: np.ndarray, geometry: _Geometry, limiter: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gradients, factors = _limited_gradients(primitive, geometry, limiter)
    cell_count = len(primitive)
    curvature = np.zeros((cell_count, 4, 3), dtype=float)
    sources = geometry.parabolic_sources
    if len(sources) == 0:
        return gradients, curvature, factors

    dx, dy = geometry.parabolic_dx, geometry.parabolic_dy
    scales = np.zeros(cell_count, dtype=float)
    np.maximum.at(scales, sources, np.maximum(np.abs(dx), np.abs(dy)))
    scales = np.maximum(scales, 1e-300)
    normalized_x = dx / scales[sources]
    normalized_y = dy / scales[sources]
    design = np.column_stack(
        (
            normalized_x,
            normalized_y,
            0.5 * normalized_x**2,
            normalized_x * normalized_y,
            0.5 * normalized_y**2,
        )
    )
    weights = 1.0 / np.maximum(normalized_x**2 + normalized_y**2, 1e-300)
    normal_matrix = np.zeros((cell_count, 5, 5), dtype=float)
    right_hand_side = np.zeros((cell_count, 5, 4), dtype=float)
    np.add.at(
        normal_matrix,
        sources,
        weights[:, None, None] * design[:, :, None] * design[:, None, :],
    )
    differences = primitive[geometry.parabolic_neighbours] - primitive[sources]
    np.add.at(
        right_hand_side,
        sources,
        weights[:, None, None] * design[:, :, None] * differences[:, None, :],
    )
    full_rank = np.linalg.matrix_rank(normal_matrix) == 5
    if not np.any(full_rank):
        return gradients, curvature, factors
    coefficients = np.linalg.solve(
        normal_matrix[full_rank], right_hand_side[full_rank]
    )
    cell_scales = scales[full_rank, None]
    gradients[full_rank, :, 0] = coefficients[:, 0, :] / cell_scales
    gradients[full_rank, :, 1] = coefficients[:, 1, :] / cell_scales
    curvature[full_rank, :, 0] = coefficients[:, 2, :] / cell_scales**2
    curvature[full_rank, :, 1] = coefficients[:, 3, :] / cell_scales**2
    curvature[full_rank, :, 2] = coefficients[:, 4, :] / cell_scales**2

    local_min = primitive.copy()
    local_max = primitive.copy()
    np.minimum.at(
        local_min, sources, primitive[geometry.parabolic_neighbours]
    )
    np.maximum.at(
        local_max, sources, primitive[geometry.parabolic_neighbours]
    )
    cells = geometry.limiter_cells
    face_dx, face_dy = geometry.limiter_dx, geometry.limiter_dy
    increments = (
        gradients[cells, :, 0] * face_dx[:, None]
        + gradients[cells, :, 1] * face_dy[:, None]
        + 0.5 * curvature[cells, :, 0] * face_dx[:, None] ** 2
        + curvature[cells, :, 1] * (face_dx * face_dy)[:, None]
        + 0.5 * curvature[cells, :, 2] * face_dy[:, None] ** 2
    )
    cell_values = primitive[cells]
    positive = increments > 1e-14 * np.maximum(np.abs(cell_values), 1.0)
    negative = increments < -1e-14 * np.maximum(np.abs(cell_values), 1.0)
    ratio = np.ones_like(increments)
    upper = np.broadcast_to(local_max[cells] - cell_values, increments.shape)
    lower = np.broadcast_to(local_min[cells] - cell_values, increments.shape)
    np.divide(upper, increments, out=ratio, where=positive)
    np.divide(lower, increments, out=ratio, where=negative)
    ratio = np.maximum(ratio, 0.0)
    if limiter in ("Barth-Jespersen", "Minmod"):
        face_limiter = np.minimum(ratio, 1.0)
    elif limiter == "Van Leer":
        face_limiter = np.where(
            ratio > 0.0, np.minimum(1.0, 2.0 * ratio / (1.0 + ratio)), 0.0
        )
    else:
        epsilon = 1e-12
        face_limiter = (
            ratio**2 + 2.0 * ratio + epsilon
        ) / (ratio**2 + ratio + 2.0 + epsilon)
        face_limiter = np.minimum(face_limiter, 1.0)
    parabolic_factors = np.ones((cell_count, 4), dtype=float)
    np.minimum.at(parabolic_factors, cells, face_limiter)
    factors[full_rank] = parabolic_factors[full_rank]
    return gradients, curvature, factors


def _reconstruct(
    primitive: np.ndarray,
    gradients: np.ndarray,
    factors: np.ndarray,
    cell_ids: np.ndarray,
    offsets: np.ndarray,
    curvature: np.ndarray | None = None,
) -> np.ndarray:
    increments = np.sum(gradients[cell_ids] * offsets[:, None, :], axis=2)
    if curvature is not None:
        dx, dy = offsets[:, 0], offsets[:, 1]
        increments += (
            0.5 * curvature[cell_ids, :, 0] * dx[:, None] ** 2
            + curvature[cell_ids, :, 1] * (dx * dy)[:, None]
            + 0.5 * curvature[cell_ids, :, 2] * dy[:, None] ** 2
        )
    return primitive[cell_ids] + factors[cell_ids] * increments


def _residual(
    conservative: np.ndarray,
    geometry: _Geometry,
    settings: NumericalSettings,
) -> np.ndarray:
    gamma = settings.gamma
    curvature = None
    primitive = conservative_to_primitive(conservative, gamma)
    if settings.reconstruction == "MUSCL parabolic":
        gradients, curvature, factors = _parabolic_reconstruction(
            primitive, geometry, settings.limiter
        )
    elif settings.reconstruction == "MUSCL linear":
        gradients, factors = _limited_gradients(
            primitive, geometry, settings.limiter
        )
    else:
        gradients = np.zeros((len(primitive), 4, 2))
        factors = np.ones((len(primitive), 4))

    def reconstruct(cell_ids: np.ndarray, offsets: np.ndarray) -> np.ndarray:
        return _reconstruct(
            primitive, gradients, factors, cell_ids, offsets, curvature
        )

    derivative = np.zeros_like(conservative)
    if len(geometry.interior_faces):
        faces = geometry.interior_faces
        owners = geometry.owners[faces]
        neighbours = geometry.neighbours[faces]
        left = reconstruct(
            owners,
            geometry.face_centers[faces] - geometry.centers[owners],
        )
        right = reconstruct(
            neighbours,
            geometry.face_centers[faces] - geometry.centers[neighbours],
        )
        flux = numerical_flux(
            left, right, geometry.face_normals[faces], gamma, settings.flux
        )
        integrated = flux * geometry.face_lengths[faces, None]
        np.add.at(derivative, owners, -integrated / geometry.areas[owners, None])
        np.add.at(
            derivative, neighbours, integrated / geometry.areas[neighbours, None]
        )
    if len(geometry.periodic_faces_a):
        faces_a, faces_b = geometry.periodic_faces_a, geometry.periodic_faces_b
        owners_a, owners_b = geometry.owners[faces_a], geometry.owners[faces_b]
        left = reconstruct(
            owners_a,
            geometry.face_centers[faces_a] - geometry.centers[owners_a],
        )
        right = reconstruct(
            owners_b,
            geometry.face_centers[faces_b] - geometry.centers[owners_b],
        )
        flux = numerical_flux(
            left, right, geometry.face_normals[faces_a], gamma, settings.flux
        )
        integrated = flux * geometry.face_lengths[faces_a, None]
        np.add.at(
            derivative, owners_a, -integrated / geometry.areas[owners_a, None]
        )
        np.add.at(
            derivative, owners_b, integrated / geometry.areas[owners_b, None]
        )
    if len(geometry.wall_faces):
        faces = geometry.wall_faces
        owners = geometry.owners[faces]
        left = reconstruct(
            owners,
            geometry.face_centers[faces] - geometry.centers[owners],
        )
        right = left.copy()
        normal_velocity = np.sum(
            left[:, 1:3] * geometry.face_normals[faces], axis=1
        )
        right[:, 1:3] -= 2.0 * normal_velocity[:, None] * geometry.face_normals[faces]
        flux = numerical_flux(
            left, right, geometry.face_normals[faces], gamma, settings.flux
        )
        integrated = flux * geometry.face_lengths[faces, None]
        np.add.at(derivative, owners, -integrated / geometry.areas[owners, None])
    if len(geometry.velocity_inlet_faces):
        faces = geometry.velocity_inlet_faces
        owners = geometry.owners[faces]
        interior = reconstruct(
            owners,
            geometry.face_centers[faces] - geometry.centers[owners],
        )
        exterior = geometry.velocity_inlet_states
        flux = numerical_flux(
            interior, exterior, geometry.face_normals[faces], gamma, settings.flux
        )
        integrated = flux * geometry.face_lengths[faces, None]
        np.add.at(derivative, owners, -integrated / geometry.areas[owners, None])
    if len(geometry.pressure_outlet_faces):
        faces = geometry.pressure_outlet_faces
        owners = geometry.owners[faces]
        interior = reconstruct(
            owners,
            geometry.face_centers[faces] - geometry.centers[owners],
        )
        exterior = interior.copy()
        exterior[:, 3] = geometry.pressure_outlet_pressures
        flux = numerical_flux(
            interior, exterior, geometry.face_normals[faces], gamma, settings.flux
        )
        integrated = flux * geometry.face_lengths[faces, None]
        np.add.at(derivative, owners, -integrated / geometry.areas[owners, None])
    if len(geometry.far_field_faces):
        faces = geometry.far_field_faces
        owners = geometry.owners[faces]
        interior = reconstruct(
            owners,
            geometry.face_centers[faces] - geometry.centers[owners],
        )
        exterior = _far_field_boundary_states(
            interior,
            geometry.far_field_states,
            geometry.face_normals[faces],
            gamma,
        )
        flux = numerical_flux(
            interior, exterior, geometry.face_normals[faces], gamma, settings.flux
        )
        integrated = flux * geometry.face_lengths[faces, None]
        np.add.at(derivative, owners, -integrated / geometry.areas[owners, None])
    for boundary in geometry.pressure_inlets:
        faces = boundary.faces
        owners = geometry.owners[faces]
        interior = reconstruct(
            owners,
            geometry.face_centers[faces] - geometry.centers[owners],
        )
        exterior = _pressure_inlet_boundary_states(
            interior,
            geometry.face_normals[faces],
            boundary,
            gamma,
            settings.gas_constant,
        )
        flux = numerical_flux(
            interior, exterior, geometry.face_normals[faces], gamma, settings.flux
        )
        integrated = flux * geometry.face_lengths[faces, None]
        np.add.at(derivative, owners, -integrated / geometry.areas[owners, None])
    for boundary in geometry.mass_flow_inlets:
        faces = boundary.faces
        owners = geometry.owners[faces]
        interior = reconstruct(
            owners,
            geometry.face_centers[faces] - geometry.centers[owners],
        )
        exterior = _mass_flow_inlet_boundary_states(
            interior,
            geometry.face_normals[faces],
            geometry.face_lengths[faces],
            boundary,
            gamma,
            settings.gas_constant,
        )
        flux = numerical_flux(
            interior, exterior, geometry.face_normals[faces], gamma, settings.flux
        )
        integrated = flux * geometry.face_lengths[faces, None]
        np.add.at(derivative, owners, -integrated / geometry.areas[owners, None])
    for boundary in geometry.mass_flow_outlets:
        faces = boundary.faces
        owners = geometry.owners[faces]
        interior = reconstruct(
            owners,
            geometry.face_centers[faces] - geometry.centers[owners],
        )
        exterior = _mass_flow_outlet_boundary_states(
            interior,
            geometry.face_normals[faces],
            geometry.face_lengths[faces],
            boundary,
        )
        flux = numerical_flux(
            interior, exterior, geometry.face_normals[faces], gamma, settings.flux
        )
        integrated = flux * geometry.face_lengths[faces, None]
        np.add.at(derivative, owners, -integrated / geometry.areas[owners, None])
    if settings.viscosity_model != "Inviscid":
        derivative += _viscous_residual(primitive, geometry, settings)
    if settings.acceleration_x != 0.0 or settings.acceleration_y != 0.0:
        derivative[:, 1] += primitive[:, 0] * settings.acceleration_x
        derivative[:, 2] += primitive[:, 0] * settings.acceleration_y
        derivative[:, 3] += primitive[:, 0] * (
            settings.acceleration_x * primitive[:, 1]
            + settings.acceleration_y * primitive[:, 2]
        )
    if not np.isfinite(derivative).all():
        raise SolverError("Spatial residual contains non-finite values.")
    return derivative


def _dynamic_viscosity(
    temperature: np.ndarray, settings: NumericalSettings
) -> np.ndarray:
    if settings.viscosity_model == "Constant":
        return np.full_like(temperature, settings.dynamic_viscosity, dtype=float)
    if settings.viscosity_model == "Sutherland":
        reference_temperature = settings.sutherland_reference_temperature
        viscosity = settings.sutherland_reference_viscosity * (
            temperature / reference_temperature
        ) ** 1.5
        viscosity *= (reference_temperature + settings.sutherland_temperature) / (
            temperature + settings.sutherland_temperature
        )
        if not np.isfinite(viscosity).all() or np.any(viscosity <= 0.0):
            raise SolverError("Sutherland law produced an invalid viscosity.")
        return viscosity
    return np.zeros_like(temperature, dtype=float)


def _viscous_inlet_states(
    primitive: np.ndarray, geometry: _Geometry, settings: NumericalSettings
) -> dict[int, np.ndarray]:
    states_by_face = {
        int(face_id): state
        for face_id, state in zip(
            geometry.velocity_inlet_faces, geometry.velocity_inlet_states
        )
    }
    for boundary in geometry.pressure_inlets:
        faces = boundary.faces
        states = _pressure_inlet_boundary_states(
            primitive[geometry.owners[faces]],
            geometry.face_normals[faces],
            boundary,
            settings.gamma,
            settings.gas_constant,
        )
        states_by_face.update(
            {int(face_id): state for face_id, state in zip(faces, states)}
        )
    for boundary in geometry.mass_flow_inlets:
        faces = boundary.faces
        states = _mass_flow_inlet_boundary_states(
            primitive[geometry.owners[faces]],
            geometry.face_normals[faces],
            geometry.face_lengths[faces],
            boundary,
            settings.gamma,
            settings.gas_constant,
        )
        states_by_face.update(
            {int(face_id): state for face_id, state in zip(faces, states)}
        )
    if len(geometry.far_field_faces):
        faces = geometry.far_field_faces
        states = _far_field_boundary_states(
            primitive[geometry.owners[faces]],
            geometry.far_field_states,
            geometry.face_normals[faces],
            settings.gamma,
        )
        states_by_face.update(
            {int(face_id): state for face_id, state in zip(faces, states)}
        )
    return states_by_face


def _viscous_gradients(
    primitive: np.ndarray, geometry: _Geometry, settings: NumericalSettings
) -> np.ndarray:
    density = primitive[:, 0]
    temperature = primitive[:, 3] / (density * settings.gas_constant)
    if not np.isfinite(temperature).all() or np.any(temperature <= 0.0):
        raise SolverError("Viscous flow requires finite positive temperature.")
    cell_values = np.column_stack((primitive[:, 1:3], temperature))
    sources = [geometry.gradient_sources]
    neighbours = [geometry.gradient_neighbours]
    dx_values = [geometry.gradient_dx]
    dy_values = [geometry.gradient_dy]
    differences = [cell_values[geometry.gradient_neighbours] - cell_values[geometry.gradient_sources]]

    inlet_values = {
        face_id: np.array(
            (state[1], state[2], state[3] / (state[0] * settings.gas_constant))
        )
        for face_id, state in _viscous_inlet_states(
            primitive, geometry, settings
        ).items()
    }

    wall_by_face = {
        int(face_id): boundary
        for boundary in geometry.wall_thermal_boundaries
        for face_id in boundary.faces
    }
    boundary_sources = []
    boundary_neighbours = []
    boundary_dx = []
    boundary_dy = []
    boundary_differences = []
    for face_id in geometry.diffusive_boundary_faces:
        owner = int(geometry.owners[face_id])
        normal = geometry.face_normals[face_id]
        offset = geometry.face_centers[face_id] - geometry.centers[owner]
        kind = geometry.patch_kinds[int(geometry.patch_ids[face_id])]
        normal_distance = float(np.dot(offset, normal))
        ghost_offset = (
            2.0 * normal_distance * normal
            if kind in ("Wall", "Symmetry")
            else 2.0 * offset
        )
        ghost = cell_values[owner].copy()
        if kind == "Wall":
            wall = wall_by_face[int(face_id)]
            if wall.mechanical_condition == "No-slip":
                ghost[:2] *= -1.0
            else:
                ghost[:2] -= (
                    2.0 * float(np.dot(ghost[:2], normal)) * normal
                )
            if wall.condition == "Temperature":
                ghost[2] = 2.0 * wall.value - cell_values[owner, 2]
            else:
                viscosity = float(_dynamic_viscosity(temperature[owner : owner + 1], settings)[0])
                conductivity = (
                    viscosity
                    * settings.gamma
                    * settings.gas_constant
                    / ((settings.gamma - 1.0) * settings.prandtl_number)
                )
                normal_distance = float(np.dot(offset, normal))
                ghost[2] += (
                    2.0 * normal_distance * wall.value / conductivity
                )
        elif kind == "Symmetry":
            velocity = cell_values[owner, :2]
            ghost[:2] -= 2.0 * float(np.dot(velocity, normal)) * normal
        elif int(face_id) in inlet_values:
            ghost = 2.0 * inlet_values[int(face_id)] - ghost
        boundary_sources.append(owner)
        boundary_neighbours.append(owner)
        boundary_dx.append(ghost_offset[0])
        boundary_dy.append(ghost_offset[1])
        boundary_differences.append(ghost - cell_values[owner])

    if boundary_sources:
        sources.append(np.asarray(boundary_sources, dtype=int))
        neighbours.append(np.asarray(boundary_neighbours, dtype=int))
        dx_values.append(np.asarray(boundary_dx, dtype=float))
        dy_values.append(np.asarray(boundary_dy, dtype=float))
        differences = np.vstack((differences[0], np.asarray(boundary_differences)))
        source_ids = np.concatenate(sources)
        dx = np.concatenate(dx_values)
        dy = np.concatenate(dy_values)
    else:
        source_ids = sources[0]
        dx = dx_values[0]
        dy = dy_values[0]
        differences = differences[0]

    gradients = np.zeros((len(primitive), 3, 2), dtype=float)
    if len(source_ids):
        weights = 1.0 / np.maximum(dx * dx + dy * dy, 1e-300)
        xx = np.zeros(len(primitive))
        xy = np.zeros(len(primitive))
        yy = np.zeros(len(primitive))
        rhs_x = np.zeros((len(primitive), 3))
        rhs_y = np.zeros((len(primitive), 3))
        np.add.at(xx, source_ids, weights * dx * dx)
        np.add.at(xy, source_ids, weights * dx * dy)
        np.add.at(yy, source_ids, weights * dy * dy)
        np.add.at(rhs_x, source_ids, (weights * dx)[:, None] * differences)
        np.add.at(rhs_y, source_ids, (weights * dy)[:, None] * differences)
        determinant = xx * yy - xy * xy
        valid = determinant > 1e-14 * np.maximum(xx * yy, 1e-300)
        gradients[valid, :, 0] = (
            rhs_x[valid] * yy[valid, None] - rhs_y[valid] * xy[valid, None]
        ) / determinant[valid, None]
        gradients[valid, :, 1] = (
            rhs_y[valid] * xx[valid, None] - rhs_x[valid] * xy[valid, None]
        ) / determinant[valid, None]
    return gradients


def _viscous_flux(
    gradients: np.ndarray,
    velocities: np.ndarray,
    viscosity: np.ndarray,
    conductivity: np.ndarray,
    normals: np.ndarray,
) -> np.ndarray:
    du_dx, du_dy = gradients[:, 0, 0], gradients[:, 0, 1]
    dv_dx, dv_dy = gradients[:, 1, 0], gradients[:, 1, 1]
    divergence = du_dx + dv_dy
    tau_xx = viscosity * (2.0 * du_dx - (2.0 / 3.0) * divergence)
    tau_yy = viscosity * (2.0 * dv_dy - (2.0 / 3.0) * divergence)
    tau_xy = viscosity * (du_dy + dv_dx)
    traction_x = tau_xx * normals[:, 0] + tau_xy * normals[:, 1]
    traction_y = tau_xy * normals[:, 0] + tau_yy * normals[:, 1]
    heat_flux = conductivity * np.sum(gradients[:, 2, :] * normals, axis=1)
    return np.column_stack(
        (
            np.zeros(len(normals)),
            traction_x,
            traction_y,
            velocities[:, 0] * traction_x
            + velocities[:, 1] * traction_y
            + heat_flux,
        )
    )


def _viscous_residual(
    primitive: np.ndarray,
    geometry: _Geometry,
    settings: NumericalSettings,
) -> np.ndarray:
    temperature = primitive[:, 3] / (
        primitive[:, 0] * settings.gas_constant
    )
    cell_values = np.column_stack((primitive[:, 1:3], temperature))
    viscosity = _dynamic_viscosity(temperature, settings)
    conductivity = (
        viscosity
        * settings.gamma
        * settings.gas_constant
        / ((settings.gamma - 1.0) * settings.prandtl_number)
    )
    gradients = _viscous_gradients(primitive, geometry, settings)
    derivative = np.zeros((len(primitive), 4), dtype=float)

    def add_face_flux(
        faces: np.ndarray,
        owners: np.ndarray,
        neighbours: np.ndarray,
        normal: np.ndarray,
        face_gradients: np.ndarray,
        face_velocity: np.ndarray,
        face_viscosity: np.ndarray,
        face_conductivity: np.ndarray,
    ) -> None:
        flux = _viscous_flux(
            face_gradients,
            face_velocity,
            face_viscosity,
            face_conductivity,
            normal,
        )
        integrated = flux * geometry.face_lengths[faces, None]
        np.add.at(derivative, owners, integrated / geometry.areas[owners, None])
        np.add.at(
            derivative, neighbours, -integrated / geometry.areas[neighbours, None]
        )

    faces = geometry.interior_faces
    if len(faces):
        owners, neighbours = geometry.owners[faces], geometry.neighbours[faces]
        face_gradients = 0.5 * (gradients[owners] + gradients[neighbours])
        face_velocity = 0.5 * (primitive[owners, 1:3] + primitive[neighbours, 1:3])
        face_viscosity = 0.5 * (viscosity[owners] + viscosity[neighbours])
        face_conductivity = 0.5 * (
            conductivity[owners] + conductivity[neighbours]
        )
        add_face_flux(
            faces,
            owners,
            neighbours,
            geometry.face_normals[faces],
            face_gradients,
            face_velocity,
            face_viscosity,
            face_conductivity,
        )
    faces_a, faces_b = geometry.periodic_faces_a, geometry.periodic_faces_b
    if len(faces_a):
        owners_a, owners_b = geometry.owners[faces_a], geometry.owners[faces_b]
        face_gradients = 0.5 * (gradients[owners_a] + gradients[owners_b])
        face_velocity = 0.5 * (
            primitive[owners_a, 1:3] + primitive[owners_b, 1:3]
        )
        face_viscosity = 0.5 * (viscosity[owners_a] + viscosity[owners_b])
        face_conductivity = 0.5 * (
            conductivity[owners_a] + conductivity[owners_b]
        )
        flux = _viscous_flux(
            face_gradients,
            face_velocity,
            face_viscosity,
            face_conductivity,
            geometry.face_normals[faces_a],
        )
        integrated = flux * geometry.face_lengths[faces_a, None]
        np.add.at(
            derivative, owners_a, integrated / geometry.areas[owners_a, None]
        )
        np.add.at(
            derivative, owners_b, -integrated / geometry.areas[owners_b, None]
        )

    wall_by_face = {
        int(face_id): boundary
        for boundary in geometry.wall_thermal_boundaries
        for face_id in boundary.faces
    }
    inlet_states = _viscous_inlet_states(primitive, geometry, settings)
    for face_id in geometry.diffusive_boundary_faces:
        owner = int(geometry.owners[face_id])
        normal = geometry.face_normals[face_id]
        offset = geometry.face_centers[face_id] - geometry.centers[owner]
        normal_distance = float(np.dot(offset, normal))
        if normal_distance <= 1e-14:
            raise SolverError("Boundary face has invalid cell-center distance.")
        kind = geometry.patch_kinds[int(geometry.patch_ids[face_id])]
        face_gradient = gradients[owner].copy()
        boundary_values = np.array(
            (primitive[owner, 1], primitive[owner, 2], temperature[owner])
        )
        if kind == "Wall":
            boundary = wall_by_face[int(face_id)]
            if boundary.mechanical_condition == "No-slip":
                for component in (0, 1):
                    normal_derivative = (
                        -boundary_values[component]
                        - float(
                            np.dot(
                                face_gradient[component],
                                offset - normal_distance * normal,
                            )
                        )
                    ) / normal_distance
                    face_gradient[component] += (
                        normal_derivative
                        - float(np.dot(face_gradient[component], normal))
                    ) * normal
                    boundary_values[component] = 0.0
            else:
                tangential_offset = offset - normal_distance * normal
                normal_velocity = float(np.dot(boundary_values[:2], normal))
                boundary_values[:2] += face_gradient[:2] @ offset
                boundary_values[:2] -= (
                    float(np.dot(boundary_values[:2], normal)) * normal
                )
                gradient_normal_velocity = (
                    normal[0] * face_gradient[0]
                    + normal[1] * face_gradient[1]
                )
                normal_derivative = -(
                    normal_velocity
                    + float(np.dot(gradient_normal_velocity, tangential_offset))
                ) / normal_distance
                desired_gradient = np.outer(
                    normal_derivative * normal, normal
                )
                velocity_gradient = face_gradient[:2]
                velocity_gradient += desired_gradient - (
                    velocity_gradient @ normal
                )[:, None] * normal
            if boundary.condition == "Temperature":
                normal_derivative = (
                    boundary.value - boundary_values[2]
                    - float(np.dot(face_gradient[2], offset - normal_distance * normal))
                ) / normal_distance
                boundary_values[2] = boundary.value
            else:
                normal_derivative = boundary.value / conductivity[owner]
                boundary_values[2] += float(np.dot(face_gradient[2], offset))
            face_gradient[2] += (
                normal_derivative - float(np.dot(face_gradient[2], normal))
            ) * normal
        elif kind == "Symmetry":
            normal_velocity = float(np.dot(boundary_values[:2], normal))
            derivative_velocity = -normal_velocity / normal_distance * normal
            for component in (0, 1):
                face_gradient[component] += (
                    derivative_velocity[component]
                    - float(np.dot(face_gradient[component], normal))
                ) * normal
            face_gradient[2] -= float(np.dot(face_gradient[2], normal)) * normal
            boundary_values[:2] -= normal_velocity * normal
        else:
            state = inlet_states.get(int(face_id))
            if state is not None:
                boundary_values = np.array(
                    (
                        state[1],
                        state[2],
                        state[3] / (state[0] * settings.gas_constant),
                    )
                )
                for component in range(3):
                    normal_derivative = (
                        boundary_values[component]
                        - cell_values[owner, component]
                        - float(
                            np.dot(
                                face_gradient[component],
                                offset - normal_distance * normal,
                            )
                        )
                    ) / normal_distance
                    face_gradient[component] += (
                        normal_derivative
                        - float(np.dot(face_gradient[component], normal))
                    ) * normal
            else:
                face_gradient -= np.outer(face_gradient @ normal, normal)
                boundary_values += face_gradient @ offset

        face_temperature = boundary_values[2]
        face_mu = float(_dynamic_viscosity(np.array([face_temperature]), settings)[0])
        face_k = (
            face_mu
            * settings.gamma
            * settings.gas_constant
            / ((settings.gamma - 1.0) * settings.prandtl_number)
        )
        flux = _viscous_flux(
            face_gradient[None, :, :],
            boundary_values[None, :2],
            np.array([face_mu]),
            np.array([face_k]),
            normal[None, :],
        )[0]
        if kind == "Wall" and wall_by_face[int(face_id)].condition == "Heat Flux":
            flux[3] = wall_by_face[int(face_id)].value
        derivative[owner] += (
            flux * geometry.face_lengths[face_id] / geometry.areas[owner]
        )
    return derivative


def _stable_time_step(
    primitive: np.ndarray,
    geometry: _Geometry,
    settings: NumericalSettings,
) -> float:
    gamma = settings.gamma
    normal_velocity = np.sum(
        primitive[geometry.owners, 1:3] * geometry.face_normals, axis=1
    )
    sound_speed = _sound_speed(primitive, gamma)
    spectral_radius = np.abs(normal_velocity) + sound_speed[geometry.owners]
    boundary_faces = np.concatenate(
        (
            geometry.velocity_inlet_faces,
            geometry.pressure_outlet_faces,
            geometry.far_field_faces,
        )
    )
    existing_boundary_count = len(boundary_faces)
    additional_faces = []
    additional_states = []
    for boundary in geometry.pressure_inlets:
        faces = boundary.faces
        owners = geometry.owners[faces]
        additional_faces.append(faces)
        additional_states.append(
            _pressure_inlet_boundary_states(
                primitive[owners],
                geometry.face_normals[faces],
                boundary,
                gamma,
                settings.gas_constant,
            )
        )
    for boundary in geometry.mass_flow_inlets:
        faces = boundary.faces
        owners = geometry.owners[faces]
        additional_faces.append(faces)
        additional_states.append(
            _mass_flow_inlet_boundary_states(
                primitive[owners],
                geometry.face_normals[faces],
                geometry.face_lengths[faces],
                boundary,
                gamma,
                settings.gas_constant,
            )
        )
    for boundary in geometry.mass_flow_outlets:
        faces = boundary.faces
        owners = geometry.owners[faces]
        additional_faces.append(faces)
        additional_states.append(
            _mass_flow_outlet_boundary_states(
                primitive[owners],
                geometry.face_normals[faces],
                geometry.face_lengths[faces],
                boundary,
            )
        )
    if additional_faces:
        boundary_faces = np.concatenate((boundary_faces, *additional_faces))
    if len(boundary_faces):
        owners = geometry.owners[boundary_faces]
        exterior = primitive[owners].copy()
        inlet_count = len(geometry.velocity_inlet_faces)
        outlet_end = inlet_count + len(geometry.pressure_outlet_faces)
        if inlet_count:
            exterior[:inlet_count] = geometry.velocity_inlet_states
        if len(geometry.pressure_outlet_faces):
            exterior[inlet_count:outlet_end, 3] = (
                geometry.pressure_outlet_pressures
            )
        far_field_start = outlet_end
        if len(geometry.far_field_faces):
            far_field_end = far_field_start + len(geometry.far_field_faces)
            exterior[far_field_start:far_field_end] = _far_field_boundary_states(
                primitive[owners[far_field_start:far_field_end]],
                geometry.far_field_states,
                geometry.face_normals[boundary_faces[far_field_start:far_field_end]],
                gamma,
            )
        if additional_states:
            exterior[existing_boundary_count:] = np.vstack(additional_states)
        exterior_normal_velocity = np.sum(
            exterior[:, 1:3] * geometry.face_normals[boundary_faces], axis=1
        )
        exterior_radius = np.abs(exterior_normal_velocity) + _sound_speed(
            exterior, gamma
        )
        spectral_radius[boundary_faces] = np.maximum(
            spectral_radius[boundary_faces], exterior_radius
        )
    spectral_sum = np.zeros(len(primitive), dtype=float)
    np.add.at(
        spectral_sum,
        geometry.owners,
        spectral_radius * geometry.face_lengths,
    )
    internal = geometry.interior_faces
    if len(internal):
        neighbours = geometry.neighbours[internal]
        neighbour_normal_velocity = np.sum(
            primitive[neighbours, 1:3] * geometry.face_normals[internal], axis=1
        )
        neighbour_radius = (
            np.abs(neighbour_normal_velocity) + sound_speed[neighbours]
        )
        np.add.at(
            spectral_sum,
            neighbours,
            neighbour_radius * geometry.face_lengths[internal],
        )
    if settings.viscosity_model != "Inviscid":
        temperature = primitive[:, 3] / (
            primitive[:, 0] * settings.gas_constant
        )
        viscosity = _dynamic_viscosity(temperature, settings)
        effective_diffusivity = np.maximum(
            (4.0 / 3.0) * viscosity / primitive[:, 0],
            viscosity / (primitive[:, 0] * settings.prandtl_number),
        )
        diffusion_sum = np.zeros(len(primitive), dtype=float)
        if len(internal):
            owners = geometry.owners[internal]
            neighbours = geometry.neighbours[internal]
            distances = np.abs(
                np.sum(
                    (geometry.centers[neighbours] - geometry.centers[owners])
                    * geometry.face_normals[internal],
                    axis=1,
                )
            )
            face_diffusivity = np.maximum(
                effective_diffusivity[owners], effective_diffusivity[neighbours]
            )
            contribution = (
                face_diffusivity
                * geometry.face_lengths[internal]
                / np.maximum(distances, 1e-14)
            )
            np.add.at(diffusion_sum, owners, contribution)
            np.add.at(diffusion_sum, neighbours, contribution)
        periodic_a, periodic_b = (
            geometry.periodic_faces_a,
            geometry.periodic_faces_b,
        )
        if len(periodic_a):
            owners_a = geometry.owners[periodic_a]
            owners_b = geometry.owners[periodic_b]
            normal_a = geometry.face_normals[periodic_a]
            distance = np.abs(
                np.sum(
                    (geometry.face_centers[periodic_a] - geometry.centers[owners_a])
                    * normal_a,
                    axis=1,
                )
            ) + np.abs(
                np.sum(
                    (
                        geometry.face_centers[periodic_b]
                        - geometry.centers[owners_b]
                    )
                    * geometry.face_normals[periodic_b],
                    axis=1,
                )
            )
            face_diffusivity = np.maximum(
                effective_diffusivity[owners_a],
                effective_diffusivity[owners_b],
            )
            contribution = (
                face_diffusivity
                * geometry.face_lengths[periodic_a]
                / np.maximum(distance, 1e-14)
            )
            np.add.at(diffusion_sum, owners_a, contribution)
            np.add.at(diffusion_sum, owners_b, contribution)
        for face_id in geometry.diffusive_boundary_faces:
            owner = geometry.owners[face_id]
            distance = abs(
                float(
                    np.dot(
                        geometry.face_centers[face_id] - geometry.centers[owner],
                        geometry.face_normals[face_id],
                    )
                )
            )
            diffusion_sum[owner] += (
                effective_diffusivity[owner]
                * geometry.face_lengths[face_id]
                / max(distance, 1e-14)
            )
        spectral_sum += 2.0 * diffusion_sum
    return float(np.min(settings.cfl * geometry.areas / spectral_sum))


def _sound_speed(primitive: np.ndarray, gamma: float) -> np.ndarray:
    return np.sqrt(gamma * primitive[:, 3] / primitive[:, 0])


def _far_field_boundary_states(
    interior: np.ndarray,
    far_field: np.ndarray,
    normals: np.ndarray,
    gamma: float,
) -> np.ndarray:
    normal_interior = np.sum(interior[:, 1:3] * normals, axis=1)
    normal_far_field = np.sum(far_field[:, 1:3] * normals, axis=1)
    sound_interior = _sound_speed(interior, gamma)
    sound_far_field = _sound_speed(far_field, gamma)
    exterior = interior.copy()

    supersonic_inflow = normal_interior <= -sound_interior
    supersonic_outflow = normal_interior >= sound_interior
    exterior[supersonic_inflow] = far_field[supersonic_inflow]
    subsonic = ~(supersonic_inflow | supersonic_outflow)
    if not np.any(subsonic):
        return exterior

    normal = normals[subsonic]
    inside = interior[subsonic]
    outside = far_field[subsonic]
    interior_velocity = normal_interior[subsonic]
    far_field_velocity = normal_far_field[subsonic]
    interior_sound = sound_interior[subsonic]
    far_field_sound = sound_far_field[subsonic]
    outgoing_invariant = interior_velocity + 2.0 * interior_sound / (gamma - 1.0)
    incoming_invariant = far_field_velocity - 2.0 * far_field_sound / (gamma - 1.0)
    boundary_velocity = 0.5 * (outgoing_invariant + incoming_invariant)
    boundary_sound = 0.25 * (gamma - 1.0) * (
        outgoing_invariant - incoming_invariant
    )
    if not np.isfinite(boundary_sound).all() or np.any(boundary_sound <= 0.0):
        raise SolverError("Far-field characteristics produced invalid sound speed.")

    inflow = boundary_velocity < 0.0
    thermodynamic_source = np.where(inflow[:, None], outside, inside)
    entropy = thermodynamic_source[:, 3] / thermodynamic_source[:, 0] ** gamma
    density = (
        boundary_sound**2 / (gamma * entropy)
    ) ** (1.0 / (gamma - 1.0))
    pressure = entropy * density**gamma
    tangent = np.column_stack((-normal[:, 1], normal[:, 0]))
    tangential_source = np.where(inflow[:, None], outside[:, 1:3], inside[:, 1:3])
    tangential_velocity = np.sum(tangential_source * tangent, axis=1)
    velocity = (
        boundary_velocity[:, None] * normal
        + tangential_velocity[:, None] * tangent
    )
    subsonic_states = np.column_stack(
        (density, velocity[:, 0], velocity[:, 1], pressure)
    )
    if (
        not np.isfinite(subsonic_states).all()
        or np.any(subsonic_states[:, 0] <= 0.0)
        or np.any(subsonic_states[:, 3] <= 0.0)
    ):
        raise SolverError("Far-field characteristics produced an invalid state.")
    exterior[subsonic] = subsonic_states
    return exterior


def _pressure_inlet_boundary_states(
    interior: np.ndarray,
    normals: np.ndarray,
    boundary: _PressureInlet,
    gamma: float,
    gas_constant: float,
) -> np.ndarray:
    if not math.isclose(float(np.linalg.norm(boundary.direction)), 1.0):
        raise SolverError(
            f"Pressure inlet '{boundary.patch_name}' has an invalid flow direction."
        )
    direction_normal = normals @ boundary.direction
    if np.any(direction_normal >= -1e-12):
        raise SolverError(
            f"Pressure inlet '{boundary.patch_name}' flow direction must point "
            "into the domain across every face."
        )

    sound_interior = _sound_speed(interior, gamma)
    normal_velocity = np.sum(interior[:, 1:3] * normals, axis=1)
    mach_interior = normal_velocity / sound_interior
    if np.any(mach_interior >= 1.0):
        raise SolverError(
            f"Pressure inlet '{boundary.patch_name}' cannot prescribe a "
            "supersonic outflow."
        )
    total_sound_speed = math.sqrt(
        gamma * gas_constant * boundary.total_temperature
    )
    mach = np.empty(len(interior), dtype=float)
    for face_id, (mach_inside, sound_inside, normal_direction) in enumerate(
        zip(mach_interior, sound_interior, direction_normal)
    ):
        if mach_inside <= -1.0:
            raise SolverError(
                f"Pressure inlet '{boundary.patch_name}' needs Mach number "
                "to define a supersonic inflow."
            )
        outgoing_invariant = (
            normal_velocity[face_id] + 2.0 * sound_inside / (gamma - 1.0)
        )

        def invariant_at(
            candidate: float, face_normal_direction: float
        ) -> float:
            temperature_ratio = 1.0 + 0.5 * (gamma - 1.0) * candidate**2
            sound = total_sound_speed / math.sqrt(temperature_ratio)
            return sound * (
                face_normal_direction * candidate + 2.0 / (gamma - 1.0)
            )

        low_value = invariant_at(0.0, normal_direction)
        high_value = invariant_at(1.0, normal_direction)
        if not high_value <= outgoing_invariant <= low_value:
            raise SolverError(
                f"Pressure inlet '{boundary.patch_name}' total conditions are "
                "incompatible with the interior characteristic state."
            )
        low, high = 0.0, 1.0
        for _ in range(64):
            middle = 0.5 * (low + high)
            if invariant_at(middle, normal_direction) > outgoing_invariant:
                low = middle
            else:
                high = middle
        mach[face_id] = 0.5 * (low + high)

    temperature = boundary.total_temperature / (
        1.0 + 0.5 * (gamma - 1.0) * mach**2
    )
    pressure = boundary.total_pressure * (
        temperature / boundary.total_temperature
    ) ** (gamma / (gamma - 1.0))
    density = pressure / (gas_constant * temperature)
    velocity = (
        mach[:, None]
        * np.sqrt(gamma * gas_constant * temperature)[:, None]
        * boundary.direction[None, :]
    )
    states = np.column_stack((density, velocity, pressure))
    if not np.isfinite(states).all() or np.any(states[:, (0, 3)] <= 0.0):
        raise SolverError(
            f"Pressure inlet '{boundary.patch_name}' produced an invalid state."
        )
    return states


def _mass_flow_inlet_boundary_states(
    interior: np.ndarray,
    normals: np.ndarray,
    face_lengths: np.ndarray,
    boundary: _MassFlowInlet,
    gamma: float,
    gas_constant: float,
) -> np.ndarray:
    inward_projection = -(normals @ boundary.direction)
    if np.any(inward_projection <= 1e-12):
        raise SolverError(
            f"Mass-flow inlet '{boundary.patch_name}' direction must point "
            "into the domain across every face."
        )
    if np.any(interior[:, 3] <= 0.0) or not np.isfinite(interior[:, 3]).all():
        raise SolverError(
            f"Mass-flow inlet '{boundary.patch_name}' has invalid interior pressure."
        )
    heat_capacity = gamma * gas_constant / (gamma - 1.0)
    maximum_speed = math.sqrt(
        2.0 * heat_capacity * boundary.total_temperature
    ) * (1.0 - 1e-12)
    pressure = interior[:, 3]
    def flow_at(speed: float) -> float:
        temperature = boundary.total_temperature - speed**2 / (
            2.0 * heat_capacity
        )
        density = pressure / (gas_constant * temperature)
        return float(
            np.sum(
                density
                * speed
                * inward_projection
                * face_lengths
            )
        )

    low, high = 0.0, maximum_speed
    if flow_at(high) < boundary.mass_flow:
        raise SolverError(
            f"Mass-flow inlet '{boundary.patch_name}' cannot satisfy the "
            "configured flow rate."
        )
    for _ in range(80):
        middle = 0.5 * (low + high)
        if flow_at(middle) < boundary.mass_flow:
            low = middle
        else:
            high = middle
    speed = 0.5 * (low + high)
    temperature = boundary.total_temperature - speed**2 / (
        2.0 * heat_capacity
    )
    density = pressure / (gas_constant * temperature)
    velocity = speed * boundary.direction[None, :]
    return np.column_stack((density, velocity, pressure))


def _mass_flow_outlet_boundary_states(
    interior: np.ndarray,
    normals: np.ndarray,
    face_lengths: np.ndarray,
    boundary: _MassFlowOutlet,
) -> np.ndarray:
    if np.any(interior[:, 0] <= 0.0) or not np.isfinite(interior[:, 0]).all():
        raise SolverError(
            f"Mass-flow outlet '{boundary.patch_name}' has invalid interior density."
        )
    states = interior.copy()
    normal_velocity = boundary.mass_flow / (
        float(np.sum(face_lengths)) * interior[:, 0]
    )
    tangent = np.column_stack((-normals[:, 1], normals[:, 0]))
    tangential_velocity = np.sum(interior[:, 1:3] * tangent, axis=1)
    states[:, 1:3] = (
        normal_velocity[:, None] * normals
        + tangential_velocity[:, None] * tangent
    )
    return states


def _finite_boundary_value(value: float, description: str) -> float:
    if isinstance(value, bool):
        raise SolverError(f"{description} must be finite.")
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise SolverError(f"{description} must be finite.") from error
    if not math.isfinite(number):
        raise SolverError(f"{description} must be finite.")
    return number


def _positive_boundary_value(value: float, description: str) -> float:
    number = _finite_boundary_value(value, description)
    if number <= 0.0:
        raise SolverError(f"{description} must be positive.")
    return number


def _boundary_flow_direction(
    x: float, y: float, description: str
) -> np.ndarray:
    direction = np.array(
        (
            _finite_boundary_value(x, f"{description} flow direction X"),
            _finite_boundary_value(y, f"{description} flow direction Y"),
        ),
        dtype=float,
    )
    magnitude = float(np.linalg.norm(direction))
    if not math.isfinite(magnitude) or magnitude <= 1e-12:
        raise SolverError(f"{description} needs a non-zero flow direction.")
    return direction / magnitude
