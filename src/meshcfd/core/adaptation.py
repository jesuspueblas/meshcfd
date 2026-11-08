"""Local cell refinement, coarsening, and hanging-node connectivity."""

from __future__ import annotations

import json
import math
from bisect import bisect_left, bisect_right
from dataclasses import dataclass

import numpy as np

from meshcfd.core.curves import evaluate_boundary_curve
from meshcfd.core.model import (
    BoundaryCurve,
    Case,
    FiniteVolumeMesh,
    MeshPatch,
    RefinementRecord,
)
from meshcfd.core.periodic import match_periodic_faces

_GEOMETRY_TOLERANCE = 1e-11


class AdaptationBalanceError(ValueError):
    """Raised when a refinement plan cannot satisfy mesh balance constraints."""


@dataclass(frozen=True)
class _Leaf:
    cell_id: str
    level: int
    vertices: tuple[tuple[float, float], ...]
    source_cells: tuple[int, ...]
    boundary_curves: tuple[BoundaryCurve | None, ...] = ()


def select_adaptation(
    mesh: FiniteVolumeMesh,
    case: Case,
    centers: np.ndarray,
    areas: np.ndarray,
    values: np.ndarray,
    threshold: float,
    maximum_level: int,
) -> tuple[dict[int, str], set[str]]:
    """Mark cells with a scale-stable, distance-weighted gradient indicator.

    Directional refinement is selected in the local directions of each
    quadrilateral.  A candidate direction is accepted only when the predicted
    children pass the geometric quality gate; otherwise isotropic refinement
    is used.  This prevents repeated one-direction splits from creating the
    very high aspect-ratio cells that destabilise the reconstruction.
    """
    cell_ids = mesh.cell_ids or tuple(f"r{index}" for index in range(len(mesh.cells)))
    levels = mesh.cell_levels or (0,) * len(mesh.cells)
    if not (len(centers) == len(areas) == len(values) == len(mesh.cells)):
        raise ValueError("Adaptation indicators must match the mesh cell count.")
    if not np.isfinite(values).all():
        raise ValueError("Adaptation indicator values must be finite.")

    links: list[tuple[int, int, float, float]] = []
    neighbors: list[set[int]] = [set() for _ in mesh.cells]
    for face_id, neighbour in enumerate(mesh.face_neighbours):
        if neighbour < 0:
            continue
        owner = mesh.face_owners[face_id]
        links.append((owner, neighbour, 0.0, 0.0))
        neighbors[owner].add(neighbour)
        neighbors[neighbour].add(owner)

    periodic_cell_pairs = []
    conditions = {condition.patch_id: condition for condition in case.boundary_conditions}
    for pair in case.periodic_pairs:
        if (pair.patch_a_id not in conditions or pair.patch_b_id not in conditions
                or conditions[pair.patch_a_id].kind != "Periodic"
                or conditions[pair.patch_b_id].kind != "Periodic"):
            continue
        face_pairs, shift = match_periodic_faces(mesh, pair.patch_a_id, pair.patch_b_id)
        for face_a, face_b in face_pairs:
            cell_a = mesh.face_owners[face_a]
            cell_b = mesh.face_owners[face_b]
            periodic_cell_pairs.append((cell_a, cell_b, shift))
            neighbors[cell_a].add(cell_b)
            neighbors[cell_b].add(cell_a)
            links.append((cell_a, cell_b, -shift[0], -shift[1]))
            links.append((cell_b, cell_a, shift[0], shift[1]))

    equations: list[list[tuple[float, float, float]]] = [[] for _ in mesh.cells]
    for first, second, shift_x, shift_y in links:
        dx = centers[second, 0] + shift_x - centers[first, 0]
        dy = centers[second, 1] + shift_y - centers[first, 1]
        difference = values[second] - values[first]
        equations[first].append((dx, dy, difference))
        equations[second].append((-dx, -dy, -difference))

    gradients = np.zeros((len(mesh.cells), 2), dtype=float)
    for cell_id, samples in enumerate(equations):
        if len(samples) < 2:
            continue
        offsets = np.asarray([(sample[0], sample[1]) for sample in samples], dtype=float)
        differences = np.asarray([sample[2] for sample in samples], dtype=float)
        distances2 = np.sum(offsets * offsets, axis=1)
        weights = 1.0 / np.maximum(distances2, _GEOMETRY_TOLERANCE**2)
        weighted_offsets = offsets * np.sqrt(weights)[:, None]
        weighted_differences = differences * np.sqrt(weights)
        normal = weighted_offsets.T @ weighted_offsets
        if np.linalg.cond(normal) > 1e10:
            normal += np.eye(2) * max(float(np.trace(normal)), 1.0) * 1e-10
        gradients[cell_id] = np.linalg.lstsq(
            weighted_offsets, weighted_differences, rcond=1e-12
        )[0]

    value_range = float(np.ptp(values))
    value_scale = max(value_range, float(np.max(np.abs(values))) * 1e-8, 1e-30)
    gradient_indicator = np.linalg.norm(gradients, axis=1) * np.sqrt(areas) / value_scale
    jump_indicator = np.zeros(len(mesh.cells), dtype=float)
    for first, second, _shift_x, _shift_y in links:
        jump = abs(float(values[second] - values[first])) / value_scale
        jump_indicator[first] = max(jump_indicator[first], jump)
    indicators = np.maximum(gradient_indicator, jump_indicator)

    refine_marks: dict[int, str] = {}
    for cell_id, indicator in enumerate(indicators):
        if indicator <= threshold or levels[cell_id] >= maximum_level:
            continue
        vertices = tuple(mesh.points[node] for node in mesh.cells[cell_id])
        corners = _remove_collinear_vertices(vertices)
        axis = "xy"
        if len(corners) == 4 and _polygon_aspect_ratio(corners) < 3.0:
            local_x, local_y = _local_cell_directions(corners)
            variation_x = abs(float(np.dot(gradients[cell_id], local_x))) * _direction_length(corners, local_x)
            variation_y = abs(float(np.dot(gradients[cell_id], local_y))) * _direction_length(corners, local_y)
            if variation_x > 2.0 * max(variation_y, 1e-30):
                axis = "x"
            elif variation_y > 2.0 * max(variation_x, 1e-30):
                axis = "y"
            if axis != "xy" and not _split_passes_quality(vertices, axis):
                axis = "xy"
        refine_marks[cell_id] = axis

    parent_by_child = {child_id: record.parent_id for record in mesh.refinement_records for child_id in record.child_ids}
    index_by_id = {cell_id: index for index, cell_id in enumerate(cell_ids)}
    coarsen_parent_ids = {
        record.parent_id for record in mesh.refinement_records
        if all(child_id in index_by_id and indicators[index_by_id[child_id]] < threshold * 0.3
               for child_id in record.child_ids)
    }
    for cell_a, cell_b, _shift in periodic_cell_pairs:
        parent_a = parent_by_child.get(cell_ids[cell_a])
        parent_b = parent_by_child.get(cell_ids[cell_b])
        if parent_a in coarsen_parent_ids and parent_b not in coarsen_parent_ids:
            coarsen_parent_ids.discard(parent_a)
        if parent_b in coarsen_parent_ids and parent_a not in coarsen_parent_ids:
            coarsen_parent_ids.discard(parent_b)

    directional_levels = _cell_directional_levels(mesh, cell_ids, mesh.refinement_records)
    parent_directional_levels = _parent_directional_levels(mesh.refinement_records)
    # Balance can introduce new directional marks.  Resolve their geometric
    # quality before returning, then rebalance so periodic partners always use
    # the same final split topology.  Without this second phase adapt_mesh()
    # could independently promote only one side of a periodic pair to ``xy``.
    while True:
        _balance_adaptation_plan(
            refine_marks, coarsen_parent_ids, neighbors, cell_ids, levels,
            directional_levels, parent_directional_levels, maximum_level,
            mesh.refinement_records, periodic_cell_pairs,
        )
        adjusted = False
        for cell_index, axis in tuple(refine_marks.items()):
            vertices = tuple(mesh.points[node] for node in mesh.cells[cell_index])
            final_axis = _quality_adjusted_split_axis(vertices, axis)
            if final_axis != axis:
                refine_marks[cell_index] = final_axis
                adjusted = True
        if not adjusted:
            break
    return refine_marks, coarsen_parent_ids


def _local_cell_directions(vertices: tuple[tuple[float, float], ...]) -> tuple[np.ndarray, np.ndarray]:
    """Return approximately orthogonal local directions for a quadrilateral."""
    points = np.asarray(vertices, dtype=float)
    first = 0.5 * ((points[1] - points[0]) + (points[2] - points[3]))
    second = 0.5 * ((points[3] - points[0]) + (points[2] - points[1]))
    if np.linalg.norm(first) <= _GEOMETRY_TOLERANCE or np.linalg.norm(second) <= _GEOMETRY_TOLERANCE:
        return np.asarray((1.0, 0.0)), np.asarray((0.0, 1.0))
    return first / np.linalg.norm(first), second / np.linalg.norm(second)


def _direction_length(vertices: tuple[tuple[float, float], ...], direction: np.ndarray) -> float:
    projections = np.asarray(vertices, dtype=float) @ direction
    return float(np.ptp(projections))


def _polygon_aspect_ratio(vertices: tuple[tuple[float, float], ...]) -> float:
    points = np.asarray(_remove_collinear_vertices(vertices), dtype=float)
    if len(points) < 3:
        return math.inf
    centered = points - np.mean(points, axis=0)
    eigenvalues = np.linalg.eigvalsh(centered.T @ centered / len(points))
    if eigenvalues[0] <= _GEOMETRY_TOLERANCE**2:
        return math.inf
    return math.sqrt(float(eigenvalues[-1] / eigenvalues[0]))


def _split_passes_quality(vertices: tuple[tuple[float, float], ...], axis: str, max_aspect: float = 6.0) -> bool:
    """Predict a split and reject degenerate or excessively stretched children."""
    try:
        children = _split_polygon(vertices, axis)
    except ValueError:
        return False
    parent_area = abs(_signed_area(list(vertices)))
    if parent_area <= _GEOMETRY_TOLERANCE:
        return False
    for child in children:
        area = abs(_signed_area(list(child)))
        if area / parent_area < 0.10 or _polygon_aspect_ratio(child) > max_aspect:
            return False
    return True


def _quality_adjusted_split_axis(
    vertices: tuple[tuple[float, float], ...], axis: str
) -> str:
    """Return the actual split axis after applying the geometric quality gate."""
    if axis == "xy" or _split_passes_quality(vertices, axis):
        return axis
    return "xy"


def _least_disruptive_refinement_axis(
    cell_index: int,
    neighbors: list[set[int]],
    projected_directional_levels: list[tuple[int, int]],
    directional_levels: tuple[tuple[int, int], ...],
) -> int:
    projected = projected_directional_levels[cell_index]

    def directional_cost(axis_index: int) -> tuple[int, int, int]:
        jumps = [
            abs(
                projected[coordinate]
                + int(coordinate == axis_index)
                - projected_directional_levels[adjacent][coordinate]
            )
            for adjacent in neighbors[cell_index]
            for coordinate in (0, 1)
        ]
        return (
            max(jumps),
            sum(jumps),
            directional_levels[cell_index][axis_index],
        )

    return min((0, 1), key=directional_cost)


def _balance_adaptation_plan(
    refine_marks: dict[int, str],
    coarsen_parent_ids: set[str],
    neighbors: list[set[int]],
    cell_ids: tuple[str, ...],
    levels: tuple[int, ...],
    directional_levels: tuple[tuple[int, int], ...],
    parent_directional_levels: dict[str, tuple[int, int]],
    maximum_level: int,
    refinement_records: tuple[RefinementRecord, ...],
    periodic_cell_pairs: list[tuple[int, int, tuple[float, float]]],
) -> None:
    """Limit level and directional-size jumps in the projected mesh."""
    records_by_parent = {
        record.parent_id: record for record in refinement_records
    }
    parent_by_child = {
        child_id: record.parent_id
        for record in refinement_records
        for child_id in record.child_ids
    }

    seen_plan_signatures: set[int] = set()
    changed = True
    while changed:
        plan_signature = hash(
            (frozenset(refine_marks.items()), frozenset(coarsen_parent_ids))
        )
        if plan_signature in seen_plan_signatures:
            raise AdaptationBalanceError(
                "The adaptive refinement plan cycles while balancing mesh transitions."
            )
        seen_plan_signatures.add(plan_signature)
        changed = False
        for cell_a, cell_b, _shift in periodic_cell_pairs:
            if cell_a in refine_marks and cell_b not in refine_marks:
                if levels[cell_b] >= maximum_level:
                    del refine_marks[cell_a]
                else:
                    refine_marks[cell_b] = refine_marks[cell_a]
                changed = True
            elif cell_b in refine_marks and cell_a not in refine_marks:
                if levels[cell_a] >= maximum_level:
                    del refine_marks[cell_b]
                else:
                    refine_marks[cell_a] = refine_marks[cell_b]
                changed = True
            elif (
                cell_a in refine_marks
                and cell_b in refine_marks
                and refine_marks[cell_a] != refine_marks[cell_b]
            ):
                if refine_marks[cell_a] != "xy" or refine_marks[cell_b] != "xy":
                    refine_marks[cell_a] = refine_marks[cell_b] = "xy"
                    changed = True

        projected_levels = []
        projected_directional_levels = []
        for cell_id_index, (cell_id, level) in enumerate(zip(cell_ids, levels)):
            parent_id = parent_by_child.get(cell_id)
            if parent_id in coarsen_parent_ids:
                projected_levels.append(records_by_parent[parent_id].level)
                projected_directional_levels.append(
                    parent_directional_levels[parent_id]
                )
            else:
                projected_levels.append(level + int(cell_id_index in refine_marks))
                mark = refine_marks.get(cell_id_index, "")
                projected_directional_levels.append(
                    (
                        directional_levels[cell_id_index][0] + int("x" in mark),
                        directional_levels[cell_id_index][1] + int("y" in mark),
                    )
                )

        for cell_id_index, _cell_id in enumerate(cell_ids):
            for neighbor in neighbors[cell_id_index]:
                if neighbor <= cell_id_index:
                    continue
                difference = (
                    projected_levels[cell_id_index] - projected_levels[neighbor]
                )
                if abs(difference) > 1:
                    lower = cell_id_index if difference < 0 else neighbor
                    higher = neighbor if difference < 0 else cell_id_index
                    parent_id = parent_by_child.get(cell_ids[lower])
                    if parent_id in coarsen_parent_ids:
                        coarsen_parent_ids.remove(parent_id)
                    elif lower in refine_marks:
                        if higher in refine_marks:
                            del refine_marks[higher]
                        else:
                            raise AdaptationBalanceError(
                                "The current mesh cannot satisfy 2:1 refinement "
                                "balance in one adaptation step."
                            )
                    elif levels[lower] < maximum_level:
                        axis_index = _least_disruptive_refinement_axis(
                            lower,
                            neighbors,
                            projected_directional_levels,
                            directional_levels,
                        )
                        refine_marks[lower] = ("x", "y")[axis_index]
                    elif higher in refine_marks:
                        del refine_marks[higher]
                    else:
                        raise AdaptationBalanceError(
                            "The current mesh cannot satisfy 2:1 refinement balance "
                            "within the configured maximum level."
                        )
                    changed = True
                    break

                for axis_index, axis in enumerate(("x", "y")):
                    difference = (
                        projected_directional_levels[cell_id_index][axis_index]
                        - projected_directional_levels[neighbor][axis_index]
                    )
                    if abs(difference) <= 1:
                        continue
                    lower = cell_id_index if difference < 0 else neighbor
                    higher = neighbor if difference < 0 else cell_id_index
                    parent_id = parent_by_child.get(cell_ids[lower])
                    if parent_id in coarsen_parent_ids:
                        coarsen_parent_ids.remove(parent_id)
                    elif lower in refine_marks:
                        current_axes = set(refine_marks[lower])
                        if axis in current_axes:
                            if higher in refine_marks:
                                del refine_marks[higher]
                            else:
                                raise AdaptationBalanceError(
                                    "The current mesh cannot satisfy anisotropic "
                                    "transition balance in one adaptation step."
                                )
                        else:
                            current_axes.add(axis)
                            refine_marks[lower] = (
                                "xy"
                                if len(current_axes) == 2
                                else next(iter(current_axes))
                            )
                    elif levels[lower] < maximum_level:
                        refine_marks[lower] = axis
                    elif higher in refine_marks:
                        del refine_marks[higher]
                    else:
                        raise AdaptationBalanceError(
                            "The current mesh cannot satisfy anisotropic transition "
                            "balance within the configured maximum level."
                        )
                    changed = True
                    break
                if changed:
                    break
            if changed:
                break

    for cell_id, adjacent in enumerate(neighbors):
        for neighbor in adjacent:
            if neighbor <= cell_id:
                continue
            parent_id = parent_by_child.get(cell_ids[cell_id])
            level = (
                records_by_parent[parent_id].level
                if parent_id in coarsen_parent_ids
                else levels[cell_id] + int(cell_id in refine_marks)
            )
            parent_id = parent_by_child.get(cell_ids[neighbor])
            neighbour_level = (
                records_by_parent[parent_id].level
                if parent_id in coarsen_parent_ids
                else levels[neighbor] + int(neighbor in refine_marks)
            )
            if abs(level - neighbour_level) > 1:
                raise AdaptationBalanceError(
                    "The projected adaptive mesh violates 2:1 refinement balance."
                )
            if any(
                abs(first - second) > 1
                for first, second in zip(
                    projected_directional_levels[cell_id],
                    projected_directional_levels[neighbor],
                )
            ):
                raise AdaptationBalanceError(
                    "The projected adaptive mesh violates anisotropic transition "
                    "balance."
                )


def _cell_directional_levels(
    mesh: FiniteVolumeMesh,
    cell_ids: tuple[str, ...],
    refinement_records: tuple[RefinementRecord, ...],
) -> tuple[tuple[int, int], ...]:
    root_vertices = {
        record.parent_id: record.vertices
        for record in refinement_records
        if "." not in record.parent_id
    }
    result = []
    for cell_id, cell in zip(cell_ids, mesh.cells):
        root_id = cell_id.split(".", 1)[0]
        base_vertices = root_vertices.get(
            root_id, tuple(mesh.points[node] for node in cell)
        )
        vertices = tuple(mesh.points[node] for node in cell)
        result.append(_directional_refinement_levels(vertices, base_vertices))
    return tuple(result)


def _parent_directional_levels(
    refinement_records: tuple[RefinementRecord, ...],
) -> dict[str, tuple[int, int]]:
    root_vertices = {
        record.parent_id: record.vertices
        for record in refinement_records
        if "." not in record.parent_id
    }
    result = {}
    for record in refinement_records:
        root_id = record.parent_id.split(".", 1)[0]
        base_vertices = root_vertices.get(root_id, record.vertices)
        result[record.parent_id] = _directional_refinement_levels(
            record.vertices, base_vertices
        )
    return result


def _directional_refinement_levels(
    vertices: tuple[tuple[float, float], ...],
    base_vertices: tuple[tuple[float, float], ...],
) -> tuple[int, int]:
    levels = []
    for axis in (0, 1):
        current_span = max(point[axis] for point in vertices) - min(
            point[axis] for point in vertices
        )
        base_span = max(point[axis] for point in base_vertices) - min(
            point[axis] for point in base_vertices
        )
        if current_span <= _GEOMETRY_TOLERANCE or base_span <= _GEOMETRY_TOLERANCE:
            levels.append(0)
        else:
            levels.append(max(0, round(math.log2(base_span / current_span))))
    return levels[0], levels[1]


def cell_areas(mesh: FiniteVolumeMesh) -> tuple[float, ...]:
    return tuple(abs(_signed_area([mesh.points[node] for node in cell])) for cell in mesh.cells)


def _cell_boundary_curves(
    mesh: FiniteVolumeMesh, cell_id: int
) -> tuple[BoundaryCurve | None, ...]:
    if not mesh.face_curves:
        return ()
    curve_by_face = {
        tuple(sorted(mesh.faces[face_id])): (face_id, curve)
        for face_id, curve in enumerate(mesh.face_curves)
        if curve is not None
    }
    cell = mesh.cells[cell_id]
    curves = []
    for start, end in zip(cell, cell[1:] + cell[:1]):
        match = curve_by_face.get(tuple(sorted((start, end))))
        if match is None:
            curves.append(None)
            continue
        face_id, curve = match
        if mesh.faces[face_id] == (start, end):
            curves.append(curve)
        else:
            curves.append(
                BoundaryCurve(
                    curve.geometry,
                    curve.end_parameter,
                    curve.start_parameter,
                    curve.patch_id,
                )
            )
    return tuple(curves)


def _merge_boundary_curves(
    vertices: tuple[tuple[float, float], ...], leaves: list[_Leaf]
) -> tuple[BoundaryCurve | None, ...]:
    grouped: dict[
        tuple[str, int],
        tuple[dict[str, object], dict[tuple[float, float], float]],
    ] = {}
    for leaf in leaves:
        if len(leaf.boundary_curves) != len(leaf.vertices):
            continue
        for start, end, curve in zip(
            leaf.vertices,
            leaf.vertices[1:] + leaf.vertices[:1],
            leaf.boundary_curves,
        ):
            if curve is None:
                continue
            key = (
                json.dumps(curve.geometry, sort_keys=True, separators=(",", ":")),
                curve.patch_id,
            )
            geometry, parameters = grouped.setdefault(
                key, (curve.geometry, {})
            )
            parameters.setdefault(_point_key(start), curve.start_parameter)
            parameters.setdefault(_point_key(end), curve.end_parameter)

    merged = []
    for start, end in zip(vertices, vertices[1:] + vertices[:1]):
        start_key, end_key = _point_key(start), _point_key(end)
        matches = [
            (geometry, key[1], parameters)
            for key, (geometry, parameters) in grouped.items()
            if start_key in parameters and end_key in parameters
        ]
        if len(matches) != 1:
            merged.append(None)
            continue
        geometry, patch_id, parameters = matches[0]
        merged.append(
            BoundaryCurve(
                geometry,
                parameters[start_key],
                parameters[end_key],
                patch_id,
            )
        )
    return tuple(merged)


def _child_boundary_curves(
    vertices: tuple[tuple[float, float], ...],
    curves: tuple[BoundaryCurve | None, ...],
    children: tuple[tuple[tuple[float, float], ...], ...],
) -> tuple[tuple[BoundaryCurve | None, ...], ...]:
    if len(curves) != len(vertices):
        return tuple(() for _ in children)
    segments: dict[
        tuple[tuple[float, float], tuple[float, float]],
        tuple[dict[str, object], int, dict[tuple[float, float], float]],
    ] = {}
    for start, end, curve in zip(
        vertices, vertices[1:] + vertices[:1], curves
    ):
        if curve is None:
            continue
        midpoint = _curve_midpoint(curve)
        point_parameters = {
            _point_key(start): curve.start_parameter,
            _point_key(midpoint): 0.5
            * (curve.start_parameter + curve.end_parameter),
            _point_key(end): curve.end_parameter,
        }
        segments[_edge_key(start, end)] = (
            curve.geometry,
            curve.patch_id,
            point_parameters,
        )

    child_curves = []
    for child in children:
        child_edges = []
        for start, end in zip(child, child[1:] + child[:1]):
            found = None
            start_key, end_key = _point_key(start), _point_key(end)
            for geometry, patch_id, parameters in segments.values():
                if start_key in parameters and end_key in parameters:
                    found = BoundaryCurve(
                        geometry,
                        parameters[start_key],
                        parameters[end_key],
                        patch_id,
                    )
                    break
            child_edges.append(found)
        child_curves.append(tuple(child_edges))
    return tuple(child_curves)


def _curve_midpoint(curve: BoundaryCurve) -> tuple[float, float]:
    return evaluate_boundary_curve(
        curve.geometry, 0.5 * (curve.start_parameter + curve.end_parameter)
    )


def _has_curved_boundary(
    vertices: tuple[tuple[float, float], ...],
    curves: dict[
        tuple[tuple[float, float], tuple[float, float]], BoundaryCurve
    ],
) -> bool:
    for start, end in zip(vertices, vertices[1:] + vertices[:1]):
        curve = curves.get(_edge_key(start, end))
        if curve is None:
            continue
        curve_midpoint = _curve_midpoint(curve)
        chord_midpoint = (
            0.5 * (start[0] + end[0]),
            0.5 * (start[1] + end[1]),
        )
        scale = max(1.0, *(abs(value) for point in (start, end) for value in point))
        if _distance_squared(curve_midpoint, chord_midpoint) > (scale * 1e-12) ** 2:
            return True
    return False


def _edge_key(
    start: tuple[float, float], end: tuple[float, float]
) -> tuple[tuple[float, float], tuple[float, float]]:
    return tuple(sorted((_point_key(start), _point_key(end))))


def adapt_mesh(
    mesh: FiniteVolumeMesh,
    refine_marks: dict[int, str],
    coarsen_parent_ids: set[str],
) -> tuple[FiniteVolumeMesh, tuple[tuple[int, ...], ...]]:
    """Adapt selected cells and return source-cell groups for conservative transfer."""
    if not mesh.cells:
        raise ValueError("Cannot adapt a mesh without cells.")
    cell_ids = mesh.cell_ids or tuple(f"r{index}" for index in range(len(mesh.cells)))
    cell_levels = mesh.cell_levels or (0,) * len(mesh.cells)
    if len(cell_ids) != len(mesh.cells) or len(cell_levels) != len(mesh.cells):
        raise ValueError("Mesh refinement metadata does not match its cells.")

    refine_marks = dict(refine_marks)
    coarsen_parent_ids = set(coarsen_parent_ids)
    neighbours: list[set[int]] = [set() for _ in mesh.cells]
    for face_id, neighbour in enumerate(mesh.face_neighbours):
        if neighbour < 0:
            continue
        owner = mesh.face_owners[face_id]
        neighbours[owner].add(neighbour)
        neighbours[neighbour].add(owner)
    maximum_level = max(
        (*cell_levels, *(cell_levels[cell_id] + 1 for cell_id in refine_marks)),
        default=0,
    )
    _balance_adaptation_plan(
        refine_marks,
        coarsen_parent_ids,
        neighbours,
        cell_ids,
        cell_levels,
        _cell_directional_levels(mesh, cell_ids, mesh.refinement_records),
        _parent_directional_levels(mesh.refinement_records),
        maximum_level,
        mesh.refinement_records,
        [],
    )

    records = {record.parent_id: record for record in mesh.refinement_records}
    old_leaves = [
        _Leaf(
            cell_ids[index],
            cell_levels[index],
            tuple(mesh.points[node] for node in cell),
            (index,),
            _cell_boundary_curves(mesh, index),
        )
        for index, cell in enumerate(mesh.cells)
    ]
    current_by_id = {leaf.cell_id: index for index, leaf in enumerate(old_leaves)}
    coarsened_children: set[str] = set()
    replacement_by_first_child: dict[str, _Leaf] = {}
    for parent_id in sorted(coarsen_parent_ids):
        record = records.get(parent_id)
        if record is None or any(child not in current_by_id for child in record.child_ids):
            continue
        child_indices = tuple(current_by_id[child] for child in record.child_ids)
        child_leaves = [old_leaves[index] for index in child_indices]
        if any(leaf.level != record.level + 1 for leaf in child_leaves):
            continue
        coarsened_children.update(record.child_ids)
        replacement_by_first_child[record.child_ids[0]] = _Leaf(
            parent_id,
            record.level,
            record.vertices,
            tuple(index for leaf in child_leaves for index in leaf.source_cells),
            _merge_boundary_curves(record.vertices, child_leaves),
        )
        for key in tuple(records):
            if key == parent_id or key.startswith(parent_id + "."):
                del records[key]

    leaves: list[_Leaf] = []
    new_records: list[RefinementRecord] = []
    for index, leaf in enumerate(old_leaves):
        if leaf.cell_id in replacement_by_first_child:
            leaves.append(replacement_by_first_child[leaf.cell_id])
            continue
        if leaf.cell_id in coarsened_children:
            continue
        axis = refine_marks.get(index)
        if axis is None:
            leaves.append(leaf)
            continue
        # select_adaptation() has already stabilized these marks across every
        # periodic pair.  Keep this guard for direct adapt_mesh() callers, but
        # make the chosen topology explicit and deterministic.
        axis = _quality_adjusted_split_axis(leaf.vertices, axis)
        children = _split_polygon(leaf.vertices, axis, leaf.boundary_curves)
        if len(children) < 2:
            leaves.append(leaf)
            continue
        child_curves = _child_boundary_curves(
            leaf.vertices, leaf.boundary_curves, children
        )
        child_ids = tuple(f"{leaf.cell_id}.{child}" for child in range(len(children)))
        new_records.append(
            RefinementRecord(leaf.cell_id, child_ids, leaf.vertices, leaf.level)
        )
        leaves.extend(
            _Leaf(child_id, leaf.level + 1, polygon, leaf.source_cells, curves)
            for child_id, polygon, curves in zip(
                child_ids, children, child_curves
            )
        )

    if not coarsened_children and not new_records:
        return mesh, tuple((index,) for index in range(len(mesh.cells)))

    rebuilt = _rebuild_mesh(mesh, leaves, tuple(records.values()) + tuple(new_records))
    _validate_two_to_one(rebuilt)
    return rebuilt, tuple(leaf.source_cells for leaf in leaves)


def _validate_two_to_one(mesh: FiniteVolumeMesh) -> None:
    levels = mesh.cell_levels or (0,) * len(mesh.cells)
    for face_id, neighbour in enumerate(mesh.face_neighbours):
        if neighbour < 0:
            continue
        owner = mesh.face_owners[face_id]
        if abs(levels[owner] - levels[neighbour]) > 1:
            raise ValueError(
                f"Adapted face {face_id} violates the 2:1 refinement balance."
            )


def _split_polygon(
    polygon: tuple[tuple[float, float], ...],
    axis: str,
    boundary_curves: tuple[BoundaryCurve | None, ...] = (),
) -> tuple[tuple[tuple[float, float], ...], ...]:
    corners = _remove_collinear_vertices(polygon)
    if len(corners) == 4:
        curve_by_edge = {
            _edge_key(first, second): curve
            for first, second, curve in zip(
                polygon,
                polygon[1:] + polygon[:1],
                boundary_curves
                if len(boundary_curves) == len(polygon)
                else (None,) * len(polygon),
            )
            if curve is not None
        }
        edge_midpoints = tuple(
            _curve_midpoint(curve_by_edge.get(_edge_key(first, second)))
            if _edge_key(first, second) in curve_by_edge
            else (
                0.5 * (first[0] + second[0]),
                0.5 * (first[1] + second[1]),
            )
            for first, second in zip(corners, corners[1:] + corners[:1])
        )
        has_curved_boundary = _has_curved_boundary(corners, curve_by_edge)
        if axis in ("x", "y"):
            opposite_pairs = ((0, 2), (1, 3))
            coordinate = 1 if axis == "x" else 0
            edge_pair = max(
                opposite_pairs,
                key=lambda pair: abs(
                    edge_midpoints[pair[0]][coordinate]
                    - edge_midpoints[pair[1]][coordinate]
                ),
            )
            first_edge, opposite_edge = edge_pair
            first_half = [edge_midpoints[first_edge]]
            index = (first_edge + 1) % 4
            while index != (opposite_edge + 1) % 4:
                first_half.append(corners[index])
                index = (index + 1) % 4
            first_half.append(edge_midpoints[opposite_edge])
            second_half = [edge_midpoints[opposite_edge]]
            while index != (first_edge + 1) % 4:
                second_half.append(corners[index])
                index = (index + 1) % 4
            second_half.append(edge_midpoints[first_edge])
            pieces = (tuple(first_half), tuple(second_half))
            return _validate_split_area(
                corners, pieces, allow_curved_boundary=has_curved_boundary
            )
        if axis == "xy":
            center = (
                sum(point[0] for point in corners) / 4.0,
                sum(point[1] for point in corners) / 4.0,
            )
            pieces = tuple(
                (
                    corners[index],
                    edge_midpoints[index],
                    center,
                    edge_midpoints[(index - 1) % 4],
                )
                for index in range(4)
            )
            return _validate_split_area(
                corners, pieces, allow_curved_boundary=has_curved_boundary
            )

    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    middle_x = 0.5 * (min(xs) + max(xs))
    middle_y = 0.5 * (min(ys) + max(ys))
    if axis == "x":
        pieces = (
            _clip_polygon(polygon, 0, middle_x, keep_less=True),
            _clip_polygon(polygon, 0, middle_x, keep_less=False),
        )
    elif axis == "y":
        pieces = (
            _clip_polygon(polygon, 1, middle_y, keep_less=True),
            _clip_polygon(polygon, 1, middle_y, keep_less=False),
        )
    elif axis == "xy":
        pieces = tuple(
            _clip_polygon(
                _clip_polygon(polygon, 0, middle_x, keep_less=keep_left),
                1,
                middle_y,
                keep_less=keep_bottom,
            )
            for keep_left, keep_bottom in (
                (True, True),
                (False, True),
                (True, False),
                (False, False),
            )
        )
    else:
        raise ValueError(f"Unsupported refinement split axis '{axis}'.")
    valid_pieces = []
    for piece in pieces:
        cleaned = _clean_polygon(piece)
        if len(cleaned) >= 3 and abs(_signed_area(cleaned)) > 1e-14:
            valid_pieces.append(cleaned)
    return _validate_split_area(polygon, tuple(valid_pieces))


def _validate_split_area(
    parent: tuple[tuple[float, float], ...],
    pieces: tuple[tuple[tuple[float, float], ...], ...],
    *,
    allow_curved_boundary: bool = False,
) -> tuple[tuple[tuple[float, float], ...], ...]:
    parent_area = abs(_signed_area(list(parent)))
    child_area = sum(abs(_signed_area(list(piece))) for piece in pieces)
    if allow_curved_boundary:
        orientation = math.copysign(1.0, _signed_area(list(parent)))
        if any(
            _signed_area(list(piece)) * orientation <= 1e-14
            for piece in pieces
        ):
            raise ValueError("Curved boundary splitting produced an invalid cell.")
        return pieces
    if not math.isclose(
        parent_area, child_area, rel_tol=1e-10, abs_tol=max(parent_area, 1.0) * 1e-12
    ):
        raise ValueError(
            "Adaptive splitting requires a convex cell that can be partitioned "
            "by axis-aligned cuts."
        )
    return pieces


def _remove_collinear_vertices(
    polygon: tuple[tuple[float, float], ...]
) -> tuple[tuple[float, float], ...]:
    vertices = list(_clean_polygon(polygon))
    changed = True
    while changed and len(vertices) > 3:
        changed = False
        for index, current in enumerate(vertices):
            previous = vertices[index - 1]
            following = vertices[(index + 1) % len(vertices)]
            first = (current[0] - previous[0], current[1] - previous[1])
            second = (following[0] - current[0], following[1] - current[1])
            cross = first[0] * second[1] - first[1] * second[0]
            scale = max(math.hypot(*first) * math.hypot(*second), 1e-30)
            if abs(cross) <= _GEOMETRY_TOLERANCE * scale:
                vertices.pop(index)
                changed = True
                break
    return tuple(vertices)


def _clip_polygon(
    polygon: tuple[tuple[float, float], ...],
    axis: int,
    coordinate: float,
    *,
    keep_less: bool,
) -> tuple[tuple[float, float], ...]:
    if not polygon:
        return ()
    result: list[tuple[float, float]] = []

    def inside(point: tuple[float, float]) -> bool:
        value = point[axis] - coordinate
        return value <= _GEOMETRY_TOLERANCE if keep_less else value >= -_GEOMETRY_TOLERANCE

    for first, second in zip(polygon, polygon[1:] + polygon[:1]):
        first_inside = inside(first)
        second_inside = inside(second)
        if first_inside:
            result.append(first)
        if first_inside != second_inside:
            denominator = second[axis] - first[axis]
            fraction = (coordinate - first[axis]) / denominator
            result.append(
                (
                    first[0] + fraction * (second[0] - first[0]),
                    first[1] + fraction * (second[1] - first[1]),
                )
            )
    return tuple(result)


def _clean_polygon(
    polygon: tuple[tuple[float, float], ...]
) -> tuple[tuple[float, float], ...]:
    cleaned = []
    for point in polygon:
        if not cleaned or _distance_squared(cleaned[-1], point) > 1e-26:
            cleaned.append(point)
    if len(cleaned) > 1 and _distance_squared(cleaned[0], cleaned[-1]) <= 1e-26:
        cleaned.pop()
    return tuple(cleaned)


def _rebuild_mesh(
    old_mesh: FiniteVolumeMesh,
    leaves: list[_Leaf],
    records: tuple[RefinementRecord, ...],
) -> FiniteVolumeMesh:
    all_vertices = tuple(point for leaf in leaves for point in leaf.vertices)
    points_by_key: dict[tuple[float, float], tuple[float, float]] = {}
    for point in all_vertices:
        points_by_key.setdefault(_point_key(point), point)
    known_points = tuple(points_by_key.values())
    sorted_x = sorted(known_points, key=lambda point: point[0])
    sorted_y = sorted(known_points, key=lambda point: point[1])
    x_coordinates = [point[0] for point in sorted_x]
    y_coordinates = [point[1] for point in sorted_y]

    boundary_edges = []
    for face_id, neighbour in enumerate(old_mesh.face_neighbours):
        if neighbour >= 0:
            continue
        start, end = old_mesh.faces[face_id]
        boundary_edges.append(
            (
                old_mesh.points[start],
                old_mesh.points[end],
                old_mesh.face_patch_ids[face_id],
            )
        )

    points: list[tuple[float, float]] = []
    point_ids: dict[tuple[float, float], int] = {}
    faces: list[tuple[int, int]] = []
    owners: list[int] = []
    neighbours: list[int] = []
    patch_ids: list[int] = []
    cell_faces: list[tuple[int, ...]] = []
    cell_signs: list[tuple[int, ...]] = []
    cell_vertices: list[tuple[int, ...]] = []
    face_curves: list[BoundaryCurve | None] = []
    edge_to_face: dict[tuple[int, int], int] = {}

    def point_id(point: tuple[float, float]) -> int:
        key = _point_key(point)
        if key not in point_ids:
            point_ids[key] = len(points)
            points.append(point)
        return point_ids[key]

    for cell_id, leaf in enumerate(leaves):
        face_ids = []
        signs = []
        loop = []
        leaf_curves = (
            leaf.boundary_curves
            if len(leaf.boundary_curves) == len(leaf.vertices)
            else (None,) * len(leaf.vertices)
        )
        for edge_index, (start, end) in enumerate(
            zip(leaf.vertices, leaf.vertices[1:] + leaf.vertices[:1])
        ):
            boundary_curve = leaf_curves[edge_index]
            axis_span = 0 if abs(end[0] - start[0]) >= abs(end[1] - start[1]) else 1
            ordered_points = sorted_x if axis_span == 0 else sorted_y
            coordinates = x_coordinates if axis_span == 0 else y_coordinates
            lower = min(start[axis_span], end[axis_span]) - _GEOMETRY_TOLERANCE
            upper = max(start[axis_span], end[axis_span]) + _GEOMETRY_TOLERANCE
            candidates = ordered_points[
                bisect_left(coordinates, lower) : bisect_right(coordinates, upper)
            ]
            edge_points = [
                point
                for point in candidates
                if _point_on_segment(point, start, end)
            ]
            edge_points.sort(key=lambda point: _segment_parameter(point, start, end))
            for first, second in zip(edge_points, edge_points[1:]):
                if _distance_squared(first, second) <= 1e-26:
                    continue
                first_id, second_id = point_id(first), point_id(second)
                loop.append(first_id)
                key = (min(first_id, second_id), max(first_id, second_id))
                face_id = edge_to_face.get(key)
                if face_id is None:
                    face_id = len(faces)
                    edge_to_face[key] = face_id
                    faces.append((first_id, second_id))
                    owners.append(cell_id)
                    neighbours.append(-1)
                    if boundary_curve is None:
                        patch_ids.append(
                            _boundary_patch(first, second, boundary_edges)
                        )
                        face_curves.append(None)
                    else:
                        parameter_start = _segment_parameter(first, start, end)
                        parameter_end = _segment_parameter(second, start, end)
                        patch_ids.append(boundary_curve.patch_id)
                        face_curves.append(
                            BoundaryCurve(
                                boundary_curve.geometry,
                                boundary_curve.start_parameter
                                + parameter_start
                                * (
                                    boundary_curve.end_parameter
                                    - boundary_curve.start_parameter
                                ),
                                boundary_curve.start_parameter
                                + parameter_end
                                * (
                                    boundary_curve.end_parameter
                                    - boundary_curve.start_parameter
                                ),
                                boundary_curve.patch_id,
                            )
                        )
                    sign = 1
                else:
                    if neighbours[face_id] >= 0:
                        raise ValueError("Adaptation produced a non-manifold face.")
                    neighbours[face_id] = cell_id
                    patch_ids[face_id] = -1
                    face_curves[face_id] = None
                    sign = 1 if faces[face_id] == (first_id, second_id) else -1
                face_ids.append(face_id)
                signs.append(sign)
        cell_faces.append(tuple(face_ids))
        cell_signs.append(tuple(signs))
        cell_vertices.append(tuple(loop))

    patches = tuple(
        MeshPatch(
            patch.id,
            patch.name,
            tuple(
                face_id
                for face_id, patch_id in enumerate(patch_ids)
                if patch_id == patch.id
            ),
        )
        for patch in old_mesh.patches
    )
    return FiniteVolumeMesh(
        points=tuple(points),
        cells=tuple(cell_vertices),
        faces=tuple(faces),
        face_owners=tuple(owners),
        face_neighbours=tuple(neighbours),
        face_patch_ids=tuple(patch_ids),
        patches=patches,
        cell_face_ids=tuple(cell_faces),
        cell_face_signs=tuple(cell_signs),
        cell_ids=tuple(leaf.cell_id for leaf in leaves),
        cell_levels=tuple(leaf.level for leaf in leaves),
        refinement_records=records,
        face_curves=(
            tuple(face_curves)
            if any(curve is not None for curve in face_curves)
            else ()
        ),
    )


def _boundary_patch(
    start: tuple[float, float],
    end: tuple[float, float],
    boundary_edges: list[
        tuple[tuple[float, float], tuple[float, float], int]
    ],
) -> int:
    midpoint = (0.5 * (start[0] + end[0]), 0.5 * (start[1] + end[1]))
    for first, second, patch_id in boundary_edges:
        if _point_on_segment(midpoint, first, second):
            return patch_id
    return -1


def _point_on_segment(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> bool:
    edge_x, edge_y = end[0] - start[0], end[1] - start[1]
    offset_x, offset_y = point[0] - start[0], point[1] - start[1]
    length = math.hypot(edge_x, edge_y)
    if length <= 1e-15:
        return _distance_squared(point, start) <= _GEOMETRY_TOLERANCE**2
    cross = abs(edge_x * offset_y - edge_y * offset_x) / length
    projection = (offset_x * edge_x + offset_y * edge_y) / (length * length)
    return (
        cross <= _GEOMETRY_TOLERANCE
        and -_GEOMETRY_TOLERANCE <= projection <= 1 + _GEOMETRY_TOLERANCE
    )


def _segment_parameter(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    dx, dy = end[0] - start[0], end[1] - start[1]
    denominator = dx * dx + dy * dy
    return ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / denominator


def _point_key(point: tuple[float, float]) -> tuple[float, float]:
    return round(point[0], 14), round(point[1], 14)


def _distance_squared(
    first: tuple[float, float], second: tuple[float, float]
) -> float:
    return (first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2


def _signed_area(vertices: list[tuple[float, float]]) -> float:
    return 0.5 * sum(
        first[0] * second[1] - second[0] * first[1]
        for first, second in zip(vertices, vertices[1:] + vertices[:1])
    )
