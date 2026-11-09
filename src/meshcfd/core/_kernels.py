"""Compiled numerical kernels used by the finite-volume solver."""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def limited_gradient_factors(
    primitive: np.ndarray,
    gradients: np.ndarray,
    gradient_sources: np.ndarray,
    gradient_neighbours: np.ndarray,
    limiter_cells: np.ndarray,
    limiter_dx: np.ndarray,
    limiter_dy: np.ndarray,
    limiter_code: int,
) -> np.ndarray:
    """Return MUSCL limiter factors using deterministic cell-wise reductions."""
    cell_count, variable_count = primitive.shape
    local_min = primitive.copy()
    local_max = primitive.copy()
    factors = np.ones((cell_count, variable_count), dtype=np.float64)

    for entry in range(gradient_sources.size):
        source = gradient_sources[entry]
        neighbour = gradient_neighbours[entry]
        for variable in range(variable_count):
            value = primitive[neighbour, variable]
            if value < local_min[source, variable]:
                local_min[source, variable] = value
            if value > local_max[source, variable]:
                local_max[source, variable] = value

    for entry in range(limiter_cells.size):
        cell = limiter_cells[entry]
        dx = limiter_dx[entry]
        dy = limiter_dy[entry]
        for variable in range(variable_count):
            cell_value = primitive[cell, variable]
            increment = (
                gradients[cell, variable, 0] * dx
                + gradients[cell, variable, 1] * dy
            )
            tolerance = 1e-14 * max(abs(cell_value), 1.0)
            ratio = 1.0
            if increment > tolerance:
                ratio = (local_max[cell, variable] - cell_value) / increment
            elif increment < -tolerance:
                ratio = (local_min[cell, variable] - cell_value) / increment
            ratio = max(ratio, 0.0)

            if limiter_code <= 1:
                face_factor = min(ratio, 1.0)
            elif limiter_code == 2:
                face_factor = min(1.0, 2.0 * ratio / (1.0 + ratio)) if ratio > 0.0 else 0.0
            else:
                epsilon = 1e-12
                face_factor = (ratio * ratio + 2.0 * ratio + epsilon) / (
                    ratio * ratio + ratio + 2.0 + epsilon
                )
                face_factor = min(face_factor, 1.0)
            if face_factor < factors[cell, variable]:
                factors[cell, variable] = face_factor

    return factors


@njit(cache=True)
def reconstruct_hllc_flux(
    primitive: np.ndarray,
    gradients: np.ndarray,
    factors: np.ndarray,
    owners: np.ndarray,
    neighbours: np.ndarray,
    owner_offsets: np.ndarray,
    neighbour_offsets: np.ndarray,
    normals: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """Fuse linear reconstruction and HLLC flux evaluation for interior faces."""
    face_count = owners.size
    fluxes = np.empty((face_count, 4), dtype=np.float64)
    for face in range(face_count):
        owner = owners[face]
        neighbour = neighbours[face]
        left = np.empty(4, dtype=np.float64)
        right = np.empty(4, dtype=np.float64)
        for variable in range(4):
            left[variable] = primitive[owner, variable] + factors[owner, variable] * (
                gradients[owner, variable, 0] * owner_offsets[face, 0]
                + gradients[owner, variable, 1] * owner_offsets[face, 1]
            )
            right[variable] = primitive[neighbour, variable] + factors[neighbour, variable] * (
                gradients[neighbour, variable, 0] * neighbour_offsets[face, 0]
                + gradients[neighbour, variable, 1] * neighbour_offsets[face, 1]
            )

        nx = normals[face, 0]
        ny = normals[face, 1]
        rho_l, ux_l, uy_l, p_l = left
        rho_r, ux_r, uy_r, p_r = right
        un_l = ux_l * nx + uy_l * ny
        un_r = ux_r * nx + uy_r * ny
        sound_l = np.sqrt(gamma * p_l / rho_l)
        sound_r = np.sqrt(gamma * p_r / rho_r)
        wave_l = min(un_l - sound_l, un_r - sound_r)
        wave_r = max(un_l + sound_l, un_r + sound_r)

        energy_l = p_l / (gamma - 1.0) + 0.5 * rho_l * (ux_l * ux_l + uy_l * uy_l)
        energy_r = p_r / (gamma - 1.0) + 0.5 * rho_r * (ux_r * ux_r + uy_r * uy_r)
        state_l = np.array((rho_l, rho_l * ux_l, rho_l * uy_l, energy_l))
        state_r = np.array((rho_r, rho_r * ux_r, rho_r * uy_r, energy_r))
        flux_l = np.array((
            rho_l * un_l,
            rho_l * ux_l * un_l + p_l * nx,
            rho_l * uy_l * un_l + p_l * ny,
            (energy_l + p_l) * un_l,
        ))
        flux_r = np.array((
            rho_r * un_r,
            rho_r * ux_r * un_r + p_r * nx,
            rho_r * uy_r * un_r + p_r * ny,
            (energy_r + p_r) * un_r,
        ))
        denominator = rho_l * (wave_l - un_l) - rho_r * (wave_r - un_r)
        if abs(denominator) <= 1e-14:
            denominator = np.copysign(1e-14, denominator)
        contact = (
            p_r - p_l
            + rho_l * un_l * (wave_l - un_l)
            - rho_r * un_r * (wave_r - un_r)
        ) / denominator

        if wave_l >= 0.0:
            fluxes[face] = flux_l
            continue
        if wave_r <= 0.0:
            fluxes[face] = flux_r
            continue

        if contact >= 0.0:
            rho, ux, uy, pressure = rho_l, ux_l, uy_l, p_l
            un, wave, energy = un_l, wave_l, energy_l
            state, physical = state_l, flux_l
        else:
            rho, ux, uy, pressure = rho_r, ux_r, uy_r, p_r
            un, wave, energy = un_r, wave_r, energy_r
            state, physical = state_r, flux_r
        factor = rho * (wave - un) / (wave - contact)
        star_ux = ux + (contact - un) * nx
        star_uy = uy + (contact - un) * ny
        star_energy = factor * (
            energy / rho
            + (contact - un) * (contact + pressure / (rho * (wave - un)))
        )
        star = np.array((factor, factor * star_ux, factor * star_uy, star_energy))
        fluxes[face] = physical + wave * (star - state)
    return fluxes


@njit(cache=True)
def weighted_least_squares_gradients(
    cell_count: int,
    source_ids: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    differences: np.ndarray,
) -> np.ndarray:
    """Compute multi-component weighted least-squares gradients in one pass."""
    component_count = differences.shape[1]
    xx = np.zeros(cell_count, dtype=np.float64)
    xy = np.zeros(cell_count, dtype=np.float64)
    yy = np.zeros(cell_count, dtype=np.float64)
    rhs_x = np.zeros((cell_count, component_count), dtype=np.float64)
    rhs_y = np.zeros((cell_count, component_count), dtype=np.float64)
    for entry in range(source_ids.size):
        cell = source_ids[entry]
        delta_x = dx[entry]
        delta_y = dy[entry]
        weight = 1.0 / max(delta_x * delta_x + delta_y * delta_y, 1e-300)
        weighted_x = weight * delta_x
        weighted_y = weight * delta_y
        xx[cell] += weighted_x * delta_x
        xy[cell] += weighted_x * delta_y
        yy[cell] += weighted_y * delta_y
        for component in range(component_count):
            difference = differences[entry, component]
            rhs_x[cell, component] += weighted_x * difference
            rhs_y[cell, component] += weighted_y * difference

    gradients = np.zeros((cell_count, component_count, 2), dtype=np.float64)
    for cell in range(cell_count):
        determinant = xx[cell] * yy[cell] - xy[cell] * xy[cell]
        if determinant > 1e-14 * max(xx[cell] * yy[cell], 1e-300):
            inverse = 1.0 / determinant
            for component in range(component_count):
                gradients[cell, component, 0] = (
                    rhs_x[cell, component] * yy[cell]
                    - rhs_y[cell, component] * xy[cell]
                ) * inverse
                gradients[cell, component, 1] = (
                    rhs_y[cell, component] * xx[cell]
                    - rhs_x[cell, component] * xy[cell]
                ) * inverse
    return gradients


@njit(cache=True)
def point_to_segments_minimum_distance(
    points: np.ndarray,
    segment_starts: np.ndarray,
    segment_ends: np.ndarray,
) -> np.ndarray:
    """Compute exact minimum point-to-segment distances."""
    distances = np.full(points.shape[0], np.inf, dtype=np.float64)
    for segment in range(segment_starts.shape[0]):
        ax = segment_starts[segment, 0]
        ay = segment_starts[segment, 1]
        ex = segment_ends[segment, 0] - ax
        ey = segment_ends[segment, 1] - ay
        length_squared = ex * ex + ey * ey
        for point in range(points.shape[0]):
            ox = points[point, 0] - ax
            oy = points[point, 1] - ay
            projection = 0.0
            if length_squared > 0.0:
                projection = (ox * ex + oy * ey) / length_squared
                projection = min(max(projection, 0.0), 1.0)
            dx = ox - projection * ex
            dy = oy - projection * ey
            distance = np.sqrt(dx * dx + dy * dy)
            if distance < distances[point]:
                distances[point] = distance
    return distances


@njit(cache=True)
def linear_gradients_and_factors(
    primitive: np.ndarray,
    gradient_sources: np.ndarray,
    gradient_neighbours: np.ndarray,
    gradient_coefficients_x: np.ndarray,
    gradient_coefficients_y: np.ndarray,
    limiter_cells: np.ndarray,
    limiter_dx: np.ndarray,
    limiter_dy: np.ndarray,
    limiter_code: int,
):
    """Compute linear least-squares gradients and limiter factors in one pass."""
    cell_count, variable_count = primitive.shape
    gradients = np.zeros((cell_count, variable_count, 2), dtype=np.float64)
    for entry in range(gradient_sources.size):
        source = gradient_sources[entry]
        neighbour = gradient_neighbours[entry]
        coefficient_x = gradient_coefficients_x[entry]
        coefficient_y = gradient_coefficients_y[entry]
        for variable in range(variable_count):
            difference = primitive[neighbour, variable] - primitive[source, variable]
            gradients[source, variable, 0] += coefficient_x * difference
            gradients[source, variable, 1] += coefficient_y * difference
    factors = limited_gradient_factors(
        primitive,
        gradients,
        gradient_sources,
        gradient_neighbours,
        limiter_cells,
        limiter_dx,
        limiter_dy,
        limiter_code,
    )
    return gradients, factors
