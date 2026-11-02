"""Local cell refinement, coarsening, and hanging-node connectivity."""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from dataclasses import dataclass

import numpy as np

from meshcfd.core.model import Case, FiniteVolumeMesh, MeshPatch, RefinementRecord
from meshcfd.core.periodic import match_periodic_faces

_GEOMETRY_TOLERANCE = 1e-11


@dataclass(frozen=True)
class _Leaf:
    cell_id: str
    level: int
    vertices: tuple[tuple[float, float], ...]
    source_cells: tuple[int, ...]


def select_adaptation(
    mesh: FiniteVolumeMesh,
    case: Case,
    centers: np.ndarray,
    areas: np.ndarray,
    values: np.ndarray,
    threshold: float,
    maximum_level: int,
) -> tuple[dict[int, str], set[str]]:
    """Mark cells using a normalized least-squares gradient indicator."""
    cell_ids = mesh.cell_ids or tuple(f"r{index}" for index in range(len(mesh.cells)))
    levels = mesh.cell_levels or (0,) * len(mesh.cells)
    if not (
        len(centers) == len(areas) == len(values) == len(mesh.cells)
    ):
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
        if (
            pair.patch_a_id not in conditions
            or pair.patch_b_id not in conditions
            or conditions[pair.patch_a_id].kind != "Periodic"
            or conditions[pair.patch_b_id].kind != "Periodic"
        ):
            continue
        face_pairs, shift = match_periodic_faces(
            mesh, pair.patch_a_id, pair.patch_b_id
        )
        for face_a, face_b in face_pairs:
            cell_a = mesh.face_owners[face_a]
            cell_b = mesh.face_owners[face_b]
            periodic_cell_pairs.append((cell_a, cell_b, shift))
            neighbors[cell_a].add(cell_b)
            neighbors[cell_b].add(cell_a)
            links.append((cell_a, cell_b, -shift[0], -shift[1]))
            links.append((cell_b, cell_a, shift[0], shift[1]))

    equations: list[list[tuple[float, float, float]]] = [
        [] for _ in mesh.cells
    ]
    for first, second, shift_x, shift_y in links:
        dx = centers[second, 0] + shift_x - centers[first, 0]
        dy = centers[second, 1] + shift_y - centers[first, 1]
        difference = values[second] - values[first]
        equations[first].append((dx, dy, difference))
        equations[second].append((-dx, -dy, -difference))

    gradients = np.zeros((len(mesh.cells), 2), dtype=float)
    for cell_id, samples in enumerate(equations):
        if not samples:
            continue
        offsets = np.asarray([(sample[0], sample[1]) for sample in samples])
        differences = np.asarray([sample[2] for sample in samples])
        gradients[cell_id] = np.linalg.lstsq(offsets, differences, rcond=None)[0]

    value_scale = max(float(np.ptp(values)), float(np.max(np.abs(values))) * 1e-12, 1e-30)
    indicators = np.linalg.norm(gradients, axis=1) * np.sqrt(areas) / np.maximum(
        np.abs(values), value_scale * 1e-8
    )
    refine_marks: dict[int, str] = {}
    for cell_id, indicator in enumerate(indicators):
        if indicator <= threshold or levels[cell_id] >= maximum_level:
            continue
        vertices = [mesh.points[node] for node in mesh.cells[cell_id]]
        width = max(point[0] for point in vertices) - min(
            point[0] for point in vertices
        )
        height = max(point[1] for point in vertices) - min(
            point[1] for point in vertices
        )
        variation_x = abs(gradients[cell_id, 0]) * width
        variation_y = abs(gradients[cell_id, 1]) * height
        if variation_x > 2.0 * variation_y:
            refine_marks[cell_id] = "x"
        elif variation_y > 2.0 * variation_x:
            refine_marks[cell_id] = "y"
        else:
            refine_marks[cell_id] = "xy"

    parent_by_child = {
        child_id: record.parent_id
        for record in mesh.refinement_records
        for child_id in record.child_ids
    }
    index_by_id = {cell_id: index for index, cell_id in enumerate(cell_ids)}
    coarsen_parent_ids = {
        record.parent_id
        for record in mesh.refinement_records
        if all(
            child_id in index_by_id
            and indicators[index_by_id[child_id]] < threshold * 0.1
            for child_id in record.child_ids
        )
    }
    for cell_a, cell_b, _shift in periodic_cell_pairs:
        parent_a = parent_by_child.get(cell_ids[cell_a])
        parent_b = parent_by_child.get(cell_ids[cell_b])
        if parent_a in coarsen_parent_ids and parent_b not in coarsen_parent_ids:
            coarsen_parent_ids.discard(parent_a)
        if parent_b in coarsen_parent_ids and parent_a not in coarsen_parent_ids:
            coarsen_parent_ids.discard(parent_b)

    children_by_parent = {
        record.parent_id: set(record.child_ids)
        for record in mesh.refinement_records
    }
    for parent_id in tuple(coarsen_parent_ids):
        children = children_by_parent[parent_id]
        neighbor_levels = [
            levels[neighbor]
            for child_id in children
            if child_id in index_by_id
            for neighbor in neighbors[index_by_id[child_id]]
            if cell_ids[neighbor] not in children
        ]
        parent_level = next(
            record.level
            for record in mesh.refinement_records
            if record.parent_id == parent_id
        )
        if any(level > parent_level + 1 for level in neighbor_levels):
            coarsen_parent_ids.discard(parent_id)

    changed = True
    while changed:
        changed = False
        for cell_id in tuple(refine_marks):
            for neighbor in neighbors[cell_id]:
                if levels[neighbor] < levels[cell_id] and levels[neighbor] < maximum_level:
                    if neighbor not in refine_marks:
                        refine_marks[neighbor] = "xy"
                        changed = True
        for cell_a, cell_b, _shift in periodic_cell_pairs:
            if cell_a in refine_marks and cell_b not in refine_marks:
                if levels[cell_b] >= maximum_level:
                    del refine_marks[cell_a]
                    changed = True
                    continue
                refine_marks[cell_b] = refine_marks[cell_a]
                changed = True
            elif cell_b in refine_marks and cell_a not in refine_marks:
                if levels[cell_a] >= maximum_level:
                    del refine_marks[cell_b]
                    changed = True
                    continue
                refine_marks[cell_a] = refine_marks[cell_b]
                changed = True
            elif (
                cell_a in refine_marks
                and cell_b in refine_marks
                and refine_marks[cell_a] != refine_marks[cell_b]
            ):
                if (
                    refine_marks[cell_a] != "xy"
                    or refine_marks[cell_b] != "xy"
                ):
                    refine_marks[cell_a] = refine_marks[cell_b] = "xy"
                    changed = True

    return refine_marks, coarsen_parent_ids


def cell_areas(mesh: FiniteVolumeMesh) -> tuple[float, ...]:
    return tuple(abs(_signed_area([mesh.points[node] for node in cell])) for cell in mesh.cells)


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

    records = {record.parent_id: record for record in mesh.refinement_records}
    old_leaves = [
        _Leaf(
            cell_ids[index],
            cell_levels[index],
            tuple(mesh.points[node] for node in cell),
            (index,),
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
        children = _split_polygon(leaf.vertices, axis)
        if len(children) < 2:
            leaves.append(leaf)
            continue
        child_ids = tuple(f"{leaf.cell_id}.{child}" for child in range(len(children)))
        new_records.append(
            RefinementRecord(leaf.cell_id, child_ids, leaf.vertices, leaf.level)
        )
        leaves.extend(
            _Leaf(child_id, leaf.level + 1, polygon, leaf.source_cells)
            for child_id, polygon in zip(child_ids, children)
        )

    if not coarsened_children and not new_records:
        return mesh, tuple((index,) for index in range(len(mesh.cells)))

    rebuilt = _rebuild_mesh(mesh, leaves, tuple(records.values()) + tuple(new_records))
    return rebuilt, tuple(leaf.source_cells for leaf in leaves)


def _split_polygon(
    polygon: tuple[tuple[float, float], ...], axis: str
) -> tuple[tuple[tuple[float, float], ...], ...]:
    corners = _remove_collinear_vertices(polygon)
    if len(corners) == 4:
        edge_midpoints = tuple(
            (
                0.5 * (first[0] + second[0]),
                0.5 * (first[1] + second[1]),
            )
            for first, second in zip(corners, corners[1:] + corners[:1])
        )
        if axis in ("x", "y"):
            opposite_pairs = ((0, 2), (1, 3))
            coordinate = 0 if axis == "x" else 1
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
            return _validate_split_area(corners, pieces)
        if axis == "xy":
            center = _diagonal_intersection(corners)
            pieces = tuple(
                (
                    corners[index],
                    edge_midpoints[index],
                    center,
                    edge_midpoints[(index - 1) % 4],
                )
                for index in range(4)
            )
            return _validate_split_area(corners, pieces)

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
) -> tuple[tuple[tuple[float, float], ...], ...]:
    parent_area = abs(_signed_area(list(parent)))
    child_area = sum(abs(_signed_area(list(piece))) for piece in pieces)
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


def _diagonal_intersection(
    polygon: tuple[tuple[float, float], ...]
) -> tuple[float, float]:
    first, opposite = polygon[0], polygon[2]
    second, other = polygon[1], polygon[3]
    first_direction = (opposite[0] - first[0], opposite[1] - first[1])
    second_direction = (other[0] - second[0], other[1] - second[1])
    denominator = (
        first_direction[0] * second_direction[1]
        - first_direction[1] * second_direction[0]
    )
    if abs(denominator) <= 1e-14:
        raise ValueError("Cannot four-way split a degenerate quadrilateral.")
    offset = (second[0] - first[0], second[1] - first[1])
    parameter = (
        offset[0] * second_direction[1] - offset[1] * second_direction[0]
    ) / denominator
    return (
        first[0] + parameter * first_direction[0],
        first[1] + parameter * first_direction[1],
    )


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
        for start, end in zip(leaf.vertices, leaf.vertices[1:] + leaf.vertices[:1]):
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
                    patch_ids.append(_boundary_patch(first, second, boundary_edges))
                    sign = 1
                else:
                    if neighbours[face_id] >= 0:
                        raise ValueError("Adaptation produced a non-manifold face.")
                    neighbours[face_id] = cell_id
                    patch_ids[face_id] = -1
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
