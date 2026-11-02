"""Post-processing utilities for cell-centered solution data."""

from __future__ import annotations

import math

from meshcfd.core.model import Case, InitialState
from meshcfd.core.solver import derive_solution_fields


def sample_solution_cut(
    case: Case,
    states: tuple[InitialState, ...],
    field: str,
    angle_degrees: float,
    point_x: float,
    point_y: float,
) -> tuple[tuple[float, float], ...]:
    """Return cell-centered field samples along a specified infinite line."""
    mesh = case.mesh
    if mesh is None:
        raise ValueError("Import a mesh before plotting a solution cut.")
    if len(states) != len(mesh.cells):
        raise ValueError("Solution state count must match the mesh cell count.")
    if not all(math.isfinite(value) for value in (angle_degrees, point_x, point_y)):
        raise ValueError("Cut angle and coordinates must be finite.")

    derived_fields = derive_solution_fields(case, states)
    if field in derived_fields:
        values = derived_fields[field]
    elif field in {
        "density",
        "velocity_x",
        "velocity_y",
        "pressure",
        "turbulence_k",
        "turbulence_omega",
    }:
        values = tuple(float(getattr(state, field)) for state in states)
    else:
        raise ValueError(f"Unsupported cut field '{field}'.")

    radians = math.radians(angle_degrees)
    direction = (math.cos(radians), math.sin(radians))
    domain_scale = max(
        max(x for x, _ in mesh.points) - min(x for x, _ in mesh.points),
        max(y for _, y in mesh.points) - min(y for _, y in mesh.points),
        1.0,
    )
    tolerance = domain_scale * 1e-10
    samples = []

    for cell_id, node_ids in enumerate(mesh.cells):
        vertices = [mesh.points[node_id] for node_id in node_ids]
        signed_distances = [
            direction[0] * (y - point_y) - direction[1] * (x - point_x) for x, y in vertices
        ]
        intersections = []
        for index, first in enumerate(vertices):
            second = vertices[(index + 1) % len(vertices)]
            first_distance = signed_distances[index]
            second_distance = signed_distances[(index + 1) % len(vertices)]
            if abs(first_distance) <= tolerance:
                intersections.append(
                    (first[0] - point_x) * direction[0] + (first[1] - point_y) * direction[1]
                )
            if first_distance * second_distance < 0.0:
                fraction = first_distance / (first_distance - second_distance)
                intersection_x = first[0] + fraction * (second[0] - first[0])
                intersection_y = first[1] + fraction * (second[1] - first[1])
                intersections.append(
                    (intersection_x - point_x) * direction[0]
                    + (intersection_y - point_y) * direction[1]
                )
        intersections.sort()
        unique_intersections = []
        for value in intersections:
            if not unique_intersections or value - unique_intersections[-1] > tolerance:
                unique_intersections.append(value)
        for start, end in zip(unique_intersections, unique_intersections[1:]):
            if end - start <= tolerance:
                continue
            samples.append(((start + end) / 2.0, values[cell_id]))

    samples.sort(key=lambda item: item[0])
    return tuple(samples)
