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
