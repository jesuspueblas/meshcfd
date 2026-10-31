import pytest

from meshcfd.io.mesh_io import MeshImportError, mesh_from_dict, mesh_to_dict


@pytest.fixture
def mesh_data():
    return {
        "format": "meshcad-mesh",
        "version": 1,
        "dimension": 2,
        "nodes": {
            "coordinates": [[0, 0], [1, 0], [1, 1], [0, 1]],
        },
        "faces": {
            "nodes": [[0, 1], [1, 2], [2, 3], [3, 0]],
            "owner": [0, 0, 0, 0],
            "patch": [0, 1, 2, 3],
            "neighbour": [-1, -1, -1, -1],
        },
        "cells": {"faces": [[0, 1, 2, 3]], "face_signs": [[1, 1, 1, 1]]},
        "patches": [
            {"id": 0, "name": "bottom", "faces": [0]},
            {"id": 1, "name": "right", "faces": [1]},
            {"id": 2, "name": "top", "faces": [2]},
            {"id": 3, "name": "left", "faces": [3]},
        ],
    }


def test_meshcad_mesh_import_and_export_round_trip(mesh_data):
    mesh = mesh_from_dict(mesh_data)

    assert mesh.points[2] == (1.0, 1.0)
    assert mesh.cells == ((0, 1, 2, 3),)
    assert [patch.name for patch in mesh.patches] == [
        "bottom",
        "right",
        "top",
        "left",
    ]
    assert mesh_to_dict(mesh) == mesh_data


def test_mesh_import_rejects_open_cell(mesh_data):
    mesh_data["cells"]["face_signs"][0][-1] = -1

    with pytest.raises(MeshImportError, match="closed node loop"):
        mesh_from_dict(mesh_data)


def test_mesh_import_rejects_non_finite_coordinates(mesh_data):
    mesh_data["nodes"]["coordinates"][0][0] = float("nan")

    with pytest.raises(MeshImportError, match="finite"):
        mesh_from_dict(mesh_data)
