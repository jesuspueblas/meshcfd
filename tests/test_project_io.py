from meshcfd.core.model import Case
from meshcfd.io.mesh_io import mesh_from_dict
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
