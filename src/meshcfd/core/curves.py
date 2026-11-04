"""Evaluation of serialized MeshCAD boundary curves."""

from __future__ import annotations

import math
from typing import Any


def _spline_point(points: list[list[float]], parameter: float) -> tuple[float, float]:
    if len(points) == 2:
        return tuple(
            points[0][axis] + (points[1][axis] - points[0][axis]) * parameter
            for axis in (0, 1)
        )

    segments = len(points) - 1
    scaled = parameter * segments
    index = min(int(scaled), segments - 1)
    local = scaled - index
    local_squared = local * local
    local_cubed = local_squared * local
    controls = (
        points[max(0, index - 1)],
        points[index],
        points[index + 1],
        points[min(index + 2, len(points) - 1)],
    )
    coordinates = []
    for axis in (0, 1):
        first, second, third, fourth = (point[axis] for point in controls)
        coordinates.append(
            0.5
            * (
                2.0 * second
                + (-first + third) * local
                + (2.0 * first - 5.0 * second + 4.0 * third - fourth)
                * local_squared
                + (-first + 3.0 * second - 3.0 * third + fourth) * local_cubed
            )
        )
    return coordinates[0], coordinates[1]


def evaluate_boundary_curve(
    geometry: dict[str, Any], parameter: float
) -> tuple[float, float]:
    """Evaluate one MeshCAD geometry descriptor at its normalized parameter."""
    parameter = max(0.0, min(1.0, parameter))
    kind = geometry.get("type")
    if kind == "line":
        start, end = geometry["start"], geometry["end"]
        return tuple(
            start[axis] + (end[axis] - start[axis]) * parameter
            for axis in (0, 1)
        )
    if kind == "arc":
        center = geometry["center"]
        span = geometry["end_angle"] - geometry["start_angle"]
        while span <= 0.0:
            span += 2.0 * math.pi
        angle = geometry["start_angle"] + span * parameter
        return (
            center[0] + geometry["radius"] * math.cos(angle),
            center[1] + geometry["radius"] * math.sin(angle),
        )
    if kind in ("spline", "spline_segment"):
        points = geometry["points"]
        if kind == "spline":
            return _spline_point(points, parameter)
        start_t, end_t = geometry["start_t"], geometry["end_t"]
        point = _spline_point(points, start_t + (end_t - start_t) * parameter)
        start_offset = geometry.get("start_offset", (0.0, 0.0))
        end_offset = geometry.get("end_offset", (0.0, 0.0))
        return tuple(
            point[axis]
            + start_offset[axis] * (1.0 - parameter)
            + end_offset[axis] * parameter
            for axis in (0, 1)
        )
    raise ValueError(f"Unsupported boundary curve type: {kind!r}.")
