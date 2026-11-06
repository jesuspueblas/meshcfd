"""Interpolation and lightweight flow-overlay geometry for the Qt canvas."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from contourpy import contour_generator
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

from meshcfd.core.model import FiniteVolumeMesh


@dataclass(frozen=True)
class RegularFlowGrid:
    x: np.ndarray
    y: np.ndarray
    scalar: np.ndarray
    velocity_x: np.ndarray
    velocity_y: np.ndarray
    valid: np.ndarray


def cell_centers(mesh: FiniteVolumeMesh) -> np.ndarray:
    result = []
    for cell in mesh.cells:
        points = np.asarray([mesh.points[node] for node in cell], dtype=float)
        result.append(np.mean(points, axis=0))
    return np.asarray(result)


def interpolate_to_regular_grid(
    mesh: FiniteVolumeMesh,
    scalar: tuple[float, ...],
    velocity_x: tuple[float, ...],
    velocity_y: tuple[float, ...],
    resolution: tuple[int, int] = (90, 60),
) -> RegularFlowGrid:
    """Interpolate cell values to a regular grid, masking points outside cells."""
    count = len(mesh.cells)
    if not (len(scalar) == len(velocity_x) == len(velocity_y) == count):
        raise ValueError("Overlay fields must match the mesh cell count.")
    centers = cell_centers(mesh)
    minimum = np.min(np.asarray(mesh.points), axis=0)
    maximum = np.max(np.asarray(mesh.points), axis=0)
    x = np.linspace(minimum[0], maximum[0], max(8, resolution[0]))
    y = np.linspace(minimum[1], maximum[1], max(8, resolution[1]))
    xx, yy = np.meshgrid(x, y)
    samples = np.column_stack((xx.ravel(), yy.ravel()))
    # Vectorised cell containment is several orders faster than testing every
    # grid point against every cell in Python.  The mask also keeps streamlines
    # out of holes and solid bodies.
    valid_flat = np.zeros(len(samples), dtype=bool)
    for cell in mesh.cells:
        polygon = np.asarray([mesh.points[node] for node in cell], dtype=float)
        low, high = polygon.min(axis=0), polygon.max(axis=0)
        candidates = np.flatnonzero(
            (~valid_flat)
            & (samples[:, 0] >= low[0]) & (samples[:, 0] <= high[0])
            & (samples[:, 1] >= low[1]) & (samples[:, 1] <= high[1])
        )
        if candidates.size:
            valid_flat[candidates] = _points_in_polygon(samples[candidates], polygon)
    valid = valid_flat.reshape(xx.shape)
    nearest = NearestNDInterpolator(centers, np.arange(count, dtype=float))

    def interpolate(values):
        values = np.asarray(values, dtype=float)
        linear = LinearNDInterpolator(centers, values, fill_value=np.nan)(samples)
        missing = ~np.isfinite(linear)
        if np.any(missing):
            nearest_indices = nearest(samples[missing]).astype(int)
            linear[missing] = values[nearest_indices]
        return linear.reshape(xx.shape)

    scalar_grid = interpolate(scalar)
    u_grid = interpolate(velocity_x)
    v_grid = interpolate(velocity_y)
    finite = np.isfinite(scalar_grid) & np.isfinite(u_grid) & np.isfinite(v_grid)
    valid &= finite
    return RegularFlowGrid(x, y, scalar_grid, u_grid, v_grid, valid)


def contour_segments(grid: RegularFlowGrid, levels: int = 15):
    """Return contours using contourpy's compiled marching-squares engine."""
    values = grid.scalar[grid.valid]
    if values.size == 0 or float(np.ptp(values)) <= 1e-15:
        return ()
    z = np.ma.array(grid.scalar, mask=~grid.valid)
    generator = contour_generator(x=grid.x, y=grid.y, z=z, corner_mask=True)
    contour_levels = np.linspace(float(values.min()), float(values.max()), levels + 2)[1:-1]
    segments = []
    for level in contour_levels:
        for line in generator.lines(float(level)):
            segments.extend(
                ((tuple(first), tuple(second), float(level))
                 for first, second in zip(line[:-1], line[1:]))
            )
    return tuple(segments)

def vector_samples(grid: RegularFlowGrid, density: int = 2):
    """Return uniformly distributed velocity vectors, independent of mesh refinement."""
    stride = max(1, 5 - int(density))
    samples = []
    for row in range(0, len(grid.y), stride):
        for column in range(0, len(grid.x), stride):
            if grid.valid[row, column]:
                samples.append((grid.x[column], grid.y[row], grid.velocity_x[row, column], grid.velocity_y[row, column]))
    return tuple(samples)


def streamline_polylines(grid: RegularFlowGrid, density: int = 2):
    """Integrate a fast, approximately uniform set of streamlines.

    Candidate seeds cover the domain, but an occupancy grid suppresses seeds
    and trajectories that would run too close to an existing streamline.  This
    is both faster and visually more uniform than integrating every seed.
    """
    density = max(1, min(4, density))
    # Target separation in regular-grid cells. Higher density means a smaller
    # exclusion radius, but the growth is deliberately sub-linear.
    separation = {1: 4, 2: 3, 3: 2, 4: 1}[density]
    occupied = np.zeros_like(grid.valid, dtype=bool)
    lines = []

    row_count = max(3, int(math.ceil((len(grid.y) - 2) / max(2, separation - 1))) + 1)
    column_count = max(3, int(math.ceil((len(grid.x) - 2) / max(3, separation))) + 1)
    row_candidates = np.unique(np.linspace(1, len(grid.y) - 2, row_count, dtype=int))
    column_candidates = np.unique(np.linspace(1, len(grid.x) - 2, column_count, dtype=int))
    candidates = [(row, column) for column in column_candidates for row in row_candidates]
    # Offset alternate columns to avoid a visually rigid rectangular pattern.
    candidates = [
        (min(len(grid.y) - 2, row + (column // max(1, separation)) % 2 * separation // 2), column)
        for row, column in candidates
    ]
    step = 0.85 * min(float(np.diff(grid.x).mean()), float(np.diff(grid.y).mean()))
    max_steps = {1: 120, 2: 150, 3: 175, 4: 200}[density]

    for row, column in candidates:
        if not grid.valid[row, column] or occupied[row, column]:
            continue
        seed = np.asarray((grid.x[column], grid.y[row]), dtype=float)
        backward = _integrate(grid, seed, -step, max_steps)
        forward = _integrate(grid, seed, step, max_steps)
        line = tuple(reversed(backward[1:])) + forward
        if len(line) < 5:
            continue
        indices = _polyline_grid_indices(grid, line)
        if not indices:
            continue
        # Reject trajectories largely duplicating an existing line.
        overlap = sum(occupied[r, c] for r, c in indices) / len(indices)
        if overlap > 0.20:
            continue
        lines.append(line)
        for r, c in indices:
            r0, r1 = max(0, r - separation), min(len(grid.y), r + separation + 1)
            c0, c1 = max(0, c - separation), min(len(grid.x), c + separation + 1)
            occupied[r0:r1, c0:c1] = True
    return tuple(lines)


def _polyline_grid_indices(grid: RegularFlowGrid, line):
    """Map a polyline to unique regular-grid indices."""
    result = []
    seen = set()
    for x, y in line:
        column = int(np.clip(np.searchsorted(grid.x, x), 0, len(grid.x) - 1))
        row = int(np.clip(np.searchsorted(grid.y, y), 0, len(grid.y) - 1))
        key = (row, column)
        if key not in seen:
            seen.add(key)
            result.append(key)
    return result

def _integrate(grid, seed, step, max_steps=220):
    points = [seed]
    point = np.asarray(seed, dtype=float)
    for _ in range(max_steps):
        velocity = _bilinear_velocity(grid, point)
        if velocity is None:
            break
        speed = float(np.linalg.norm(velocity))
        if speed <= 1e-14:
            break
        direction = velocity / speed
        midpoint = point + 0.5 * step * direction
        middle_velocity = _bilinear_velocity(grid, midpoint)
        if middle_velocity is None or np.linalg.norm(middle_velocity) <= 1e-14:
            break
        candidate = point + step * middle_velocity / np.linalg.norm(middle_velocity)
        if _bilinear_velocity(grid, candidate) is None:
            break
        if any(np.linalg.norm(candidate - np.asarray(old)) < abs(step) * 0.25 for old in points[-12:-2]):
            break
        point = candidate
        points.append((float(point[0]), float(point[1])))
    return tuple(points)


def _bilinear_velocity(grid, point):
    x, y = point
    if x < grid.x[0] or x > grid.x[-1] or y < grid.y[0] or y > grid.y[-1]:
        return None
    column = min(len(grid.x) - 2, max(0, int(np.searchsorted(grid.x, x) - 1)))
    row = min(len(grid.y) - 2, max(0, int(np.searchsorted(grid.y, y) - 1)))
    corners = ((row, column), (row, column + 1), (row + 1, column), (row + 1, column + 1))
    if not all(grid.valid[index] for index in corners):
        return None
    tx = (x - grid.x[column]) / (grid.x[column + 1] - grid.x[column])
    ty = (y - grid.y[row]) / (grid.y[row + 1] - grid.y[row])
    weights = ((1 - tx) * (1 - ty), tx * (1 - ty), (1 - tx) * ty, tx * ty)
    u = sum(weight * grid.velocity_x[index] for weight, index in zip(weights, corners))
    v = sum(weight * grid.velocity_y[index] for weight, index in zip(weights, corners))
    return np.asarray((u, v), dtype=float)


def _points_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Vectorised ray casting for many points and one polygon."""
    inside = np.zeros(len(points), dtype=bool)
    previous = polygon[-1]
    x, y = points[:, 0], points[:, 1]
    for current in polygon:
        crosses = (current[1] > y) != (previous[1] > y)
        denominator = previous[1] - current[1]
        if abs(denominator) > 1e-30:
            boundary_x = (previous[0] - current[0]) * (y - current[1]) / denominator + current[0]
            inside ^= crosses & (x < boundary_x)
        previous = current
    return inside
