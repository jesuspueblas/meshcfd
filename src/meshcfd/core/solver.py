"""Explicit finite-volume solver for the two-dimensional ideal-gas Euler equations."""

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
class _Geometry:
    centers: np.ndarray
    areas: np.ndarray
    face_centers: np.ndarray
    face_normals: np.ndarray
    face_lengths: np.ndarray
    owners: np.ndarray
    neighbours: np.ndarray
    patch_ids: np.ndarray
    interior_faces: np.ndarray
    wall_faces: np.ndarray
    periodic_faces_a: np.ndarray
    periodic_faces_b: np.ndarray
    periodic_pairs: tuple[PeriodicBoundaryPair, ...]
    velocity_inlet_faces: np.ndarray
    velocity_inlet_states: np.ndarray
    pressure_outlet_faces: np.ndarray
    pressure_outlet_pressures: np.ndarray
    far_field_faces: np.ndarray
    far_field_states: np.ndarray
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
                _stable_time_step(primitive, geometry, settings.gamma, settings.cfl),
                settings.end_time - time,
            )
            if not math.isfinite(time_step) or time_step <= 0.0:
                raise SolverError("The stable time step is non-positive or non-finite.")
            start = conservative
            derivative = _residual(
                start, geometry, settings.gamma, settings.flux,
                settings.reconstruction, settings.limiter,
            )
            if settings.time_integration == "Explicit Euler":
                conservative = start + time_step * derivative
            else:
                stage_one = start + time_step * derivative
                derivative_one = _residual(
                    stage_one, geometry, settings.gamma, settings.flux,
                    settings.reconstruction, settings.limiter,
                )
                if settings.time_integration == "Explicit RK2":
                    conservative = 0.5 * start + 0.5 * (
                        stage_one + time_step * derivative_one
                    )
                else:
                    stage_two = 0.75 * start + 0.25 * (
                        stage_one + time_step * derivative_one
                    )
                    derivative_two = _residual(
                        stage_two, geometry, settings.gamma, settings.flux,
                        settings.reconstruction, settings.limiter,
                    )
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
        final_residual = _residual(
            conservative,
            geometry,
            settings.gamma,
            settings.flux,
            settings.reconstruction,
            settings.limiter,
        )
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
            "Pressure Outlet",
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
    velocity_inlet_faces = []
    velocity_inlet_states = []
    pressure_outlet_faces = []
    pressure_outlet_pressures = []
    far_field_faces = []
    far_field_states = []
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
        elif condition.kind == "Pressure Outlet":
            pressure = _positive_boundary_value(
                condition.values.get("pressure", 101325.0),
                f"Static pressure for outlet '{condition.patch_name}'",
            )
            pressure_outlet_faces.extend(faces.tolist())
            pressure_outlet_pressures.extend([pressure] * len(faces))
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
        interior_faces=interior_faces,
        wall_faces=wall_faces,
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
    gamma: float,
    flux_scheme: str,
    reconstruction: str,
    limiter: str,
) -> np.ndarray:
    primitive = conservative_to_primitive(conservative, gamma)
    curvature = None
    if reconstruction == "MUSCL parabolic":
        gradients, curvature, factors = _parabolic_reconstruction(
            primitive, geometry, limiter
        )
    elif reconstruction == "MUSCL linear":
        gradients, factors = _limited_gradients(primitive, geometry, limiter)
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
            left, right, geometry.face_normals[faces], gamma, flux_scheme
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
            left, right, geometry.face_normals[faces_a], gamma, flux_scheme
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
            left, right, geometry.face_normals[faces], gamma, flux_scheme
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
            interior, exterior, geometry.face_normals[faces], gamma, flux_scheme
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
            interior, exterior, geometry.face_normals[faces], gamma, flux_scheme
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
            interior, exterior, geometry.face_normals[faces], gamma, flux_scheme
        )
        integrated = flux * geometry.face_lengths[faces, None]
        np.add.at(derivative, owners, -integrated / geometry.areas[owners, None])
    if not np.isfinite(derivative).all():
        raise SolverError("Spatial residual contains non-finite values.")
    return derivative


def _stable_time_step(
    primitive: np.ndarray, geometry: _Geometry, gamma: float, cfl: float
) -> float:
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
            exterior[far_field_start:] = _far_field_boundary_states(
                primitive[owners[far_field_start:]],
                geometry.far_field_states,
                geometry.face_normals[boundary_faces[far_field_start:]],
                gamma,
            )
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
    return float(np.min(cfl * geometry.areas / spectral_sum))


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
