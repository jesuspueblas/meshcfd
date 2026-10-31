"""Geometric checks for translational periodic boundary patches."""

from __future__ import annotations

import math

from meshcfd.core.model import FiniteVolumeMesh, MeshPatch


def detect_periodic_pair(
    mesh: FiniteVolumeMesh, patch_a_id: int, patch_b_id: int
) -> tuple[str, float]:
    if patch_a_id == patch_b_id:
        raise ValueError("Select two different patches.")
    patches = {patch.id: patch for patch in mesh.patches}
    try:
        patch_a = patches[patch_a_id]
        patch_b = patches[patch_b_id]
    except KeyError as error:
        raise ValueError("A selected patch is not present in the mesh.") from error

    edges_a = _patch_edges(mesh, patch_a)
    edges_b = _patch_edges(mesh, patch_b)
    if len(edges_a) != len(edges_b):
        raise ValueError("Patches have different numbers of faces.")
    if not edges_a:
        raise ValueError("Periodic patches must contain at least one face.")

    all_points = tuple(point for edge in edges_a + edges_b for point in edge)
    bounds = _bounds(edges_a + edges_b)
    scale = max(
        1.0,
        max(abs(value) for point in all_points for value in point),
        *(upper - lower for lower, upper in bounds),
    )
    tolerance = scale * 1e-9
    bounds_a = _bounds(edges_a)
    bounds_b = _bounds(edges_b)
    matches = []
    for axis, orthogonal_axis, name in ((0, 1, "x"), (1, 0, "y")):
        if (
            abs(bounds_a[orthogonal_axis][0] - bounds_b[orthogonal_axis][0])
            > tolerance
            or abs(bounds_a[orthogonal_axis][1] - bounds_b[orthogonal_axis][1])
            > tolerance
        ):
            continue
        translation = (
            (bounds_b[axis][0] + bounds_b[axis][1])
            - (bounds_a[axis][0] + bounds_a[axis][1])
        ) / 2
        if abs(translation) <= tolerance:
            continue
        translated_edges = _translate_edges(
            edges_a, translation if axis == 0 else 0.0, translation if axis == 1 else 0.0
        )
        if _same_edges(translated_edges, edges_b, tolerance):
            matches.append((name, abs(translation)))

    if len(matches) != 1:
        raise ValueError(
            "Patches are not a matching translation in exactly one axis (x or y)."
        )
    axis, pitch = matches[0]
    if not math.isfinite(pitch) or pitch <= 0.0:
        raise ValueError("Periodic pitch must be positive and finite.")
    return axis, pitch


def _patch_edges(
    mesh: FiniteVolumeMesh, patch: MeshPatch
) -> tuple[tuple[tuple[float, float], tuple[float, float]], ...]:
    return tuple(
        tuple(sorted((mesh.points[mesh.faces[face_id][0]], mesh.points[mesh.faces[face_id][1]])))
        for face_id in patch.face_ids
    )


def _bounds(
    edges: tuple[tuple[tuple[float, float], tuple[float, float]], ...]
) -> tuple[tuple[float, float], tuple[float, float]]:
    points = tuple(point for edge in edges for point in edge)
    return tuple(
        (min(point[axis] for point in points), max(point[axis] for point in points))
        for axis in (0, 1)
    )


def _translate_edges(
    edges: tuple[tuple[tuple[float, float], tuple[float, float]], ...],
    dx: float,
    dy: float,
) -> tuple[tuple[tuple[float, float], tuple[float, float]], ...]:
    return tuple(
        tuple(sorted(((a[0] + dx, a[1] + dy), (b[0] + dx, b[1] + dy))))
        for a, b in edges
    )


def _same_edges(
    edges_a: tuple[tuple[tuple[float, float], tuple[float, float]], ...],
    edges_b: tuple[tuple[tuple[float, float], tuple[float, float]], ...],
    tolerance: float,
) -> bool:
    ordered_a = sorted(edges_a)
    ordered_b = sorted(edges_b)
    return all(
        abs(point_a[axis] - point_b[axis]) <= tolerance
        for edge_a, edge_b in zip(ordered_a, ordered_b)
        for point_a, point_b in zip(edge_a, edge_b)
        for axis in (0, 1)
    )
