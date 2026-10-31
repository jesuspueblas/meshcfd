"""Reader for MeshCAD's versioned finite-volume mesh JSON format."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from meshcfd.core.model import FiniteVolumeMesh, MeshPatch

FORMAT_NAME = "meshcad-mesh"
FORMAT_VERSION = 1
NO_NEIGHBOUR = -1


class MeshImportError(ValueError):
    """Raised when a mesh file is unsupported or invalid."""


def _integer(value: Any, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise MeshImportError(f"{description} must be an integer.")
    return value


def _integer_list(value: Any, description: str) -> list[int]:
    if not isinstance(value, list):
        raise MeshImportError(f"{description} must be a list.")
    return [_integer(item, description) for item in value]


def mesh_from_dict(data: dict[str, Any]) -> FiniteVolumeMesh:
    if data.get("format") != FORMAT_NAME or data.get("version") != FORMAT_VERSION:
        raise MeshImportError(
            "Unsupported mesh format. Expected MeshCAD finite-volume mesh JSON "
            f"version {FORMAT_VERSION}."
        )
    if data.get("dimension") != 2:
        raise MeshImportError("Only two-dimensional meshes are supported.")

    try:
        coordinates = data["nodes"]["coordinates"]
        face_nodes = data["faces"]["nodes"]
        face_owners = data["faces"]["owner"]
        face_neighbours = data["faces"]["neighbour"]
        face_patch_ids = data["faces"]["patch"]
        cell_faces = data["cells"]["faces"]
        cell_signs = data["cells"]["face_signs"]
        raw_patches = data["patches"]
    except (KeyError, TypeError) as error:
        raise MeshImportError(f"Mesh file is missing required data: {error}.") from error

    if not isinstance(coordinates, list) or not coordinates:
        raise MeshImportError("Mesh must contain at least one node.")
    points = []
    for index, coordinate in enumerate(coordinates):
        if not isinstance(coordinate, list) or len(coordinate) != 2:
            raise MeshImportError(f"Node {index} must have two coordinates.")
        try:
            point = (float(coordinate[0]), float(coordinate[1]))
        except (TypeError, ValueError) as error:
            raise MeshImportError(f"Node {index} has invalid coordinates.") from error
        if not all(math.isfinite(value) for value in point):
            raise MeshImportError(f"Node {index} coordinates must be finite.")
        points.append(point)

    if not isinstance(face_nodes, list) or not face_nodes:
        raise MeshImportError("Mesh must contain at least one face.")
    faces = []
    for face_id, nodes in enumerate(face_nodes):
        if not isinstance(nodes, list) or len(nodes) != 2:
            raise MeshImportError(f"Face {face_id} must have exactly two nodes.")
        node_ids = tuple(_integer(node, f"Face {face_id} node") for node in nodes)
        if (
            node_ids[0] == node_ids[1]
            or any(node < 0 or node >= len(points) for node in node_ids)
        ):
            raise MeshImportError(f"Face {face_id} references invalid nodes.")
        faces.append(node_ids)

    face_owners = _integer_list(face_owners, "Face owners")
    face_neighbours = _integer_list(face_neighbours, "Face neighbours")
    face_patch_ids = _integer_list(face_patch_ids, "Face patch identifiers")
    if not (
        len(faces) == len(face_owners) == len(face_neighbours) == len(face_patch_ids)
    ):
        raise MeshImportError("Face connectivity arrays have inconsistent lengths.")

    if not isinstance(raw_patches, list):
        raise MeshImportError("Patches must be a list.")
    patches = []
    patch_ids = set()
    patch_names = set()
    for item in raw_patches:
        if not isinstance(item, dict):
            raise MeshImportError("Each patch must be a JSON object.")
        patch_id = _integer(item.get("id"), "Patch identifier")
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            raise MeshImportError(f"Patch {patch_id} must have a non-empty name.")
        if patch_id in patch_ids or name in patch_names:
            raise MeshImportError("Patch identifiers and names must be unique.")
        raw_face_ids = _integer_list(item.get("faces"), f"Patch {name} faces")
        if any(face < 0 or face >= len(faces) for face in raw_face_ids):
            raise MeshImportError(f"Patch {name} references invalid faces.")
        patches.append(MeshPatch(patch_id, name, tuple(raw_face_ids)))
        patch_ids.add(patch_id)
        patch_names.add(name)

    if not isinstance(cell_faces, list) or not isinstance(cell_signs, list):
        raise MeshImportError("Cell connectivity must be represented by lists.")
    if not cell_faces or len(cell_faces) != len(cell_signs):
        raise MeshImportError("Mesh must have cells with consistent face connectivity.")
    if any(
        patch_id != NO_NEIGHBOUR and patch_id not in patch_ids
        for patch_id in face_patch_ids
    ):
        raise MeshImportError("A face references an unknown boundary patch.")
    for face_id, (owner, neighbour, patch_id) in enumerate(
        zip(face_owners, face_neighbours, face_patch_ids)
    ):
        if owner < 0 or owner >= len(cell_faces):
            raise MeshImportError(f"Face {face_id} references an invalid owner cell.")
        if neighbour != NO_NEIGHBOUR and (
            neighbour < 0 or neighbour >= len(cell_faces) or neighbour == owner
        ):
            raise MeshImportError(f"Face {face_id} references an invalid neighbour cell.")
        if (neighbour == NO_NEIGHBOUR) != (patch_id != NO_NEIGHBOUR):
            raise MeshImportError(f"Face {face_id} has inconsistent boundary data.")

    cells = []
    stored_cell_face_ids = []
    stored_cell_face_signs = []
    for cell_id, (raw_face_ids, raw_signs) in enumerate(zip(cell_faces, cell_signs)):
        cell_face_ids = _integer_list(raw_face_ids, f"Cell {cell_id} faces")
        signs = _integer_list(raw_signs, f"Cell {cell_id} face signs")
        if len(cell_face_ids) < 3 or len(cell_face_ids) != len(signs):
            raise MeshImportError(f"Cell {cell_id} must have at least three faces.")
        if any(face < 0 or face >= len(faces) for face in cell_face_ids):
            raise MeshImportError(f"Cell {cell_id} references invalid faces.")
        if any(sign not in (-1, 1) for sign in signs):
            raise MeshImportError(f"Cell {cell_id} face signs must be -1 or 1.")
        stored_cell_face_ids.append(tuple(cell_face_ids))
        stored_cell_face_signs.append(tuple(signs))

        directed_edges = [
            faces[face_id] if sign == 1 else tuple(reversed(faces[face_id]))
            for face_id, sign in zip(cell_face_ids, signs)
        ]
        ordered_edges = [directed_edges.pop(0)]
        while directed_edges:
            next_start = ordered_edges[-1][1]
            next_index = next(
                (
                    index
                    for index, edge in enumerate(directed_edges)
                    if edge[0] == next_start
                ),
                None,
            )
            if next_index is None:
                break
            ordered_edges.append(directed_edges.pop(next_index))
        if (
            directed_edges
            or ordered_edges[-1][1] != ordered_edges[0][0]
        ):
            raise MeshImportError(f"Cell {cell_id} faces do not form a closed node loop.")
        cells.append(tuple(edge[0] for edge in ordered_edges))

    for patch in patches:
        for face_id in patch.face_ids:
            if face_patch_ids[face_id] != patch.id:
                raise MeshImportError(f"Patch {patch.name} contains an unrelated face.")
    listed_faces = [
        face_id for patch in patches for face_id in patch.face_ids
    ]
    if len(listed_faces) != len(set(listed_faces)):
        raise MeshImportError("A boundary face cannot belong to multiple patches.")
    listed_face_ids = set(listed_faces)
    for face_id, patch_id in enumerate(face_patch_ids):
        if (patch_id != NO_NEIGHBOUR) != (face_id in listed_face_ids):
            raise MeshImportError(f"Face {face_id} has inconsistent patch membership.")

    return FiniteVolumeMesh(
        points=tuple(points),
        cells=tuple(cells),
        faces=tuple(faces),
        face_owners=tuple(face_owners),
        face_neighbours=tuple(face_neighbours),
        face_patch_ids=tuple(face_patch_ids),
        patches=tuple(patches),
        cell_face_ids=tuple(stored_cell_face_ids),
        cell_face_signs=tuple(stored_cell_face_signs),
    )


def load_mesh(path: str | Path) -> FiniteVolumeMesh:
    source = Path(path)
    try:
        with source.open("r", encoding="utf-8") as stream:
            data = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise MeshImportError(f"Could not read mesh file '{source}': {error}") from error
    if not isinstance(data, dict):
        raise MeshImportError("Mesh file must contain a JSON object.")
    return mesh_from_dict(data)


def mesh_to_dict(mesh: FiniteVolumeMesh) -> dict[str, Any]:
    """Serialize a mesh for embedding in a portable case file."""
    return {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "dimension": 2,
        "nodes": {"coordinates": [list(point) for point in mesh.points]},
        "faces": {
            "nodes": [list(face) for face in mesh.faces],
            "owner": list(mesh.face_owners),
            "patch": list(mesh.face_patch_ids),
            "neighbour": list(mesh.face_neighbours),
        },
        "cells": {
            "faces": [list(face_ids) for face_ids in mesh.cell_face_ids],
            "face_signs": [list(signs) for signs in mesh.cell_face_signs],
        },
        "patches": [
            {"id": patch.id, "name": patch.name, "faces": list(patch.face_ids)}
            for patch in mesh.patches
        ],
    }
