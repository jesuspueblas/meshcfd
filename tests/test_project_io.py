import json

import pytest

from meshcfd.core.model import (
    Case,
    FiniteVolumeMesh,
    InitialConditionBlock,
    MeshPatch,
    PeriodicBoundaryPair,
)
from meshcfd.core.periodic import detect_periodic_pair
from meshcfd.io.mesh_io import mesh_from_dict, mesh_to_dict
from meshcfd.io.project_io import case_from_dict, case_to_dict, load_case, save_case


def make_mesh():
    return mesh_from_dict(
        {
            "format": "meshcad-mesh",
            "version": 1,
            "dimension": 2,
            "nodes": {"coordinates": [[0, 0], [1, 0], [1, 1], [0, 1]]},
            "faces": {
                "nodes": [[0, 1], [1, 2], [2, 3], [3, 0]],
                "owner": [0, 0, 0, 0],
                "patch": [0, 1, 2, 3],
                "neighbour": [-1, -1, -1, -1],
            },
            "cells": {
                "faces": [[0, 1, 2, 3]],
                "face_signs": [[1, 1, 1, 1]],
            },
            "patches": [
                {"id": 0, "name": "bottom", "faces": [0]},
                {"id": 1, "name": "right", "faces": [1]},
                {"id": 2, "name": "top", "faces": [2]},
                {"id": 3, "name": "left", "faces": [3]},
            ],
        }
    )


def test_case_round_trip_keeps_mesh_and_setup(tmp_path):
    case = Case(name="channel")
    case.set_mesh(make_mesh())
    case.boundary_conditions[1].kind = "Velocity Inlet"
    case.boundary_conditions[1].values = {
        "density": 1.2,
        "velocity_x": 80.0,
        "velocity_y": 0.0,
        "temperature": 290.0,
    }
    case.numerics.reconstruction = "MUSCL parabolic"
    case.numerics.cfl = 0.8
    path = tmp_path / "channel.mcf.json"

    save_case(path, case)
    restored = load_case(path)

    assert case_to_dict(restored) == case_to_dict(case)
    assert len(restored.boundary_conditions) == 4
    assert restored.boundary_conditions[1].values["velocity_x"] == 80.0
    assert restored.numerics.reconstruction == "MUSCL parabolic"


def test_case_saves_relative_mesh_reference_and_reloads_mesh(tmp_path):
    mesh_path = tmp_path / "mesh.json"
    mesh_path.write_text(json.dumps(mesh_to_dict(make_mesh())), encoding="utf-8")
    case_path = tmp_path / "cases" / "channel.mcf.json"
    case_path.parent.mkdir()
    case = Case(name="channel")
    case.set_mesh(make_mesh(), mesh_path)

    save_case(case_path, case)

    saved_data = json.loads(case_path.read_text(encoding="utf-8"))
    assert saved_data["version"] == 4
    assert saved_data["mesh"] == "../mesh.json"
    assert isinstance(saved_data["mesh"], str)

    restored = load_case(case_path)
    assert restored.mesh == case.mesh
    assert restored.mesh_path == mesh_path.resolve()


def test_case_round_trip_preserves_initial_condition_blocks(tmp_path):
    case = Case(name="channel")
    case.initial_conditions.base_state.density = 0.95
    case.initial_conditions.blocks = [
        InitialConditionBlock(
            name="hot core",
            domain="x*x + y*y <= 1",
            density="1 + 0.1*cos(x)",
            velocity_x="10*tanh(y)",
            velocity_y="0",
            pressure="101325 + 1000*sin(x)",
        )
    ]
    path = tmp_path / "channel.mcf.json"

    save_case(path, case)
    restored = load_case(path)

    assert restored.initial_conditions == case.initial_conditions


def test_version_two_cases_with_relative_mesh_reference_are_supported(tmp_path):
    mesh_path = tmp_path / "mesh.json"
    mesh_path.write_text(json.dumps(mesh_to_dict(make_mesh())), encoding="utf-8")
    case_path = tmp_path / "saved.mcf.json"
    data = case_to_dict(Case(name="legacy"))
    data.update(version=2, mesh="mesh.json")
    case_path.write_text(json.dumps(data), encoding="utf-8")

    restored = load_case(case_path)

    assert restored.mesh == make_mesh()
    assert restored.mesh_path == mesh_path.resolve()


def test_case_loads_version_one_embedded_mesh():
    case = Case(name="legacy")
    case.set_mesh(make_mesh())
    data = case_to_dict(case)
    data["version"] = 1

    restored = case_from_dict(data)

    assert restored.mesh == case.mesh
    assert restored.mesh_path is None


def test_periodic_pair_detection_and_case_round_trip(tmp_path):
    mesh = make_mesh()
    assert detect_periodic_pair(mesh, 1, 3) == ("x", 1.0)
    assert detect_periodic_pair(mesh, 0, 2) == ("y", 1.0)

    case = Case(name="periodic", mesh=mesh)
    case.boundary_conditions[1].kind = "Periodic"
    case.boundary_conditions[3].kind = "Periodic"
    case.periodic_pairs.append(PeriodicBoundaryPair(1, 3, "x", 1.0))
    path = tmp_path / "periodic.mcf.json"
    save_case(path, case)

    restored = load_case(path)

    assert restored.periodic_pairs == case.periodic_pairs
    assert restored.boundary_conditions[1].kind == "Periodic"
    assert restored.boundary_conditions[3].kind == "Periodic"
    assert json.loads(path.read_text(encoding="utf-8"))["version"] == 4


def test_periodic_pair_detection_rejects_nonmatching_patches():
    with pytest.raises(ValueError, match="matching translation"):
        detect_periodic_pair(make_mesh(), 1, 0)


def test_periodic_pair_detection_rejects_different_face_geometry():
    mesh = FiniteVolumeMesh(
        points=((0, 0), (0, 0.5), (0, 1), (1, 0), (1, 0.4), (1, 1)),
        cells=(),
        faces=((0, 1), (1, 2), (3, 4), (4, 5)),
        face_owners=(),
        face_neighbours=(),
        face_patch_ids=(),
        patches=(MeshPatch(0, "left", (0, 1)), MeshPatch(1, "right", (2, 3))),
        cell_face_ids=(),
        cell_face_signs=(),
    )

    with pytest.raises(ValueError, match="matching translation"):
        detect_periodic_pair(mesh, 0, 1)


def test_case_save_rejects_inconsistent_periodic_pitch():
    case = Case(mesh=make_mesh())
    case.boundary_conditions[1].kind = "Periodic"
    case.boundary_conditions[3].kind = "Periodic"
    case.periodic_pairs.append(PeriodicBoundaryPair(1, 3, "x", 2.0))

    with pytest.raises(ValueError, match="axis or pitch"):
        case_to_dict(case)


def test_case_save_rejects_invalid_initial_condition_expression():
    case = Case()
    case.initial_conditions.blocks.append(
        InitialConditionBlock(domain="__import__('os').system('whoami')")
    )

    with pytest.raises(ValueError, match="Unsupported expression syntax"):
        case_to_dict(case)


def test_case_load_fills_default_conditions_for_unconfigured_patches():
    case = Case(name="defaults", mesh=make_mesh())
    data = case_to_dict(case)
    data["boundary_conditions"] = []

    restored = case_from_dict(data)

    assert [item.patch_name for item in restored.boundary_conditions] == [
        "bottom",
        "right",
        "top",
        "left",
    ]
