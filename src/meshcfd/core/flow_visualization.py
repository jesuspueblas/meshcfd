"""Interpolation and lightweight flow-overlay geometry for the Qt canvas."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
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
    valid = np.asarray([_point_in_mesh(point, mesh) for point in samples]).reshape(xx.shape)

    def interpolate(values):
        values = np.asarray(values, dtype=float)
        linear = LinearNDInterpolator(centers, values, fill_value=np.nan)(samples)
        missing = ~np.isfinite(linear)
        if np.any(missing):
            linear[missing] = NearestNDInterpolator(centers, values)(samples[missing])
        return linear.reshape(xx.shape)

    scalar_grid = interpolate(scalar)
    u_grid = interpolate(velocity_x)
    v_grid = interpolate(velocity_y)
    finite = np.isfinite(scalar_grid) & np.isfinite(u_grid) & np.isfinite(v_grid)
    valid &= finite
    return RegularFlowGrid(x, y, scalar_grid, u_grid, v_grid, valid)


def contour_segments(grid: RegularFlowGrid, levels: int = 15):
    """Return marching-squares contour line segments."""
    values = grid.scalar[grid.valid]
    if values.size == 0 or float(np.ptp(values)) <= 1e-15:
        return ()
    contour_levels = np.linspace(float(np.min(values)), float(np.max(values)), levels + 2)[1:-1]
    segments = []
    for level in contour_levels:
        for row in range(len(grid.y) - 1):
            for column in range(len(grid.x) - 1):
                corners = ((row, column), (row, column + 1), (row + 1, column + 1), (row + 1, column))
                if not all(grid.valid[index] for index in corners):
                    continue
                points = []
                for first, second in zip(corners, corners[1:] + corners[:1]):
                    a, b = grid.scalar[first], grid.scalar[second]
                    if (a <= level < b) or (b <= level < a):
                        fraction = (level - a) / (b - a)
                        ax, ay = grid.x[first[1]], grid.y[first[0]]
                        bx, by = grid.x[second[1]], grid.y[second[0]]
                        points.append((ax + fraction * (bx - ax), ay + fraction * (by - ay)))
                if len(points) == 2:
                    segments.append((points[0], points[1], float(level)))
                elif len(points) == 4:
                    segments.extend(((points[0], points[1], float(level)), (points[2], points[3], float(level))))
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
    """Integrate streamlines in both directions with midpoint RK2 steps."""
    spacing = max(3, 11 - 2 * int(density))
    seeds = [(grid.x[1], grid.y[row]) for row in range(1, len(grid.y) - 1, spacing)]
    seeds += [(grid.x[column], grid.y[1]) for column in range(spacing, len(grid.x) - 1, spacing)]
    step = 0.45 * min(float(np.diff(grid.x).mean()), float(np.diff(grid.y).mean()))
    lines = []
    for seed in seeds:
        backward = _integrate(grid, seed, -step)
        forward = _integrate(grid, seed, step)
        line = tuple(reversed(backward[1:])) + forward
        if len(line) >= 3:
            lines.append(line)
    return tuple(lines)


def _integrate(grid, seed, step):
    points = [seed]
    point = np.asarray(seed, dtype=float)
    for _ in range(450):
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


def _point_in_mesh(point, mesh):
    x, y = point
    for cell in mesh.cells:
        vertices = [mesh.points[node] for node in cell]
        inside = False
        previous = vertices[-1]
        for current in vertices:
            if ((current[1] > y) != (previous[1] > y)) and x < (previous[0] - current[0]) * (y - current[1]) / (previous[1] - current[1]) + current[0]:
                inside = not inside
            previous = current
        if inside:
            return True
    return False
