import json

import pytest

from meshcfd.core.adaptation import adapt_mesh
from meshcfd.core.model import (
    Case,
    FiniteVolumeMesh,
    InitialConditionBlock,
    InitialState,
    MeshPatch,
    NumericalSettings,
    PeriodicBoundaryPair,
)
from meshcfd.core.periodic import detect_periodic_pair, match_periodic_faces
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
    case.numerics.end_time = 3.5
    case.numerics.viscosity_model = "Sutherland"
    case.numerics.sutherland_reference_viscosity = 1.8e-5
    case.numerics.prandtl_number = 0.74
    case.numerics.acceleration_y = -9.81
    case.numerics.turbulence_model = "Menter k-omega SST"
    case.numerics.turbulence_wall_treatment = "Wall functions"
    case.numerics.turbulence_omega_formulation = "Standard SST"
    case.initial_conditions.base_state.turbulence_k = 0.025
    case.initial_conditions.base_state.turbulence_omega = 12.0
    case.boundary_conditions[1].values.update(
        turbulence_intensity=2.5, turbulence_length_scale=0.15
    )
    case.boundary_conditions[0].thermal_condition = "Heat Flux"
    case.boundary_conditions[0].mechanical_condition = "Slip"
    case.boundary_conditions[0].values["heat_flux"] = 250.0
    path = tmp_path / "channel.mcf.json"

    save_case(path, case)
    restored = load_case(path)

    assert case_to_dict(restored) == case_to_dict(case)
    assert len(restored.boundary_conditions) == 4
    assert restored.boundary_conditions[1].values["velocity_x"] == 80.0
    assert restored.numerics.reconstruction == "MUSCL parabolic"
    assert restored.numerics.end_time == 3.5
    assert restored.numerics.viscosity_model == "Sutherland"
    assert restored.numerics.sutherland_reference_viscosity == 1.8e-5
    assert restored.numerics.prandtl_number == 0.74
    assert restored.numerics.acceleration_y == -9.81
    assert restored.numerics.turbulence_model == "Menter k-omega SST"
    assert restored.numerics.turbulence_wall_treatment == "Wall functions"
    assert restored.numerics.turbulence_omega_formulation == "Standard SST"
    assert restored.initial_conditions.base_state.turbulence_k == 0.025
    assert restored.initial_conditions.base_state.turbulence_omega == 12.0
    assert restored.boundary_conditions[1].values["turbulence_intensity"] == 2.5
    assert restored.boundary_conditions[1].values["turbulence_length_scale"] == 0.15
    assert restored.boundary_conditions[0].thermal_condition == "Heat Flux"
    assert restored.boundary_conditions[0].mechanical_condition == "Slip"
    assert restored.boundary_conditions[0].values["heat_flux"] == 250.0


def test_version_six_case_loads_with_inviscid_defaults():
    data = case_to_dict(Case(name="legacy"))
    data["version"] = 6
    for boundary in data["boundary_conditions"]:
        boundary.pop("mechanical_condition")
        boundary.pop("thermal_condition")
    for key in (
        "viscosity_model",
        "dynamic_viscosity",
        "sutherland_reference_viscosity",
        "sutherland_reference_temperature",
        "sutherland_temperature",
        "prandtl_number",
        "acceleration_x",
        "acceleration_y",
    ):
        data["numerics"].pop(key)

    restored = case_from_dict(data)

    assert restored.numerics.viscosity_model == "Inviscid"
    assert restored.numerics.prandtl_number == pytest.approx(0.72)
    assert restored.numerics.acceleration_x == 0.0
    assert restored.boundary_conditions == []


def test_version_nine_case_defaults_to_standard_sst_omega():
    data = case_to_dict(Case(name="legacy"))
    data["version"] = 9
    data["numerics"].pop("turbulence_omega_formulation")

    restored = case_from_dict(data)

    assert restored.numerics.turbulence_omega_formulation == "Standard SST"


def test_version_seven_wall_defaults_to_no_slip():
    case = Case(name="legacy")
    case.set_mesh(make_mesh())
    data = case_to_dict(case)
    data["version"] = 7
    for boundary in data["boundary_conditions"]:
        boundary.pop("mechanical_condition")

    restored = case_from_dict(data)

    assert all(
        boundary.mechanical_condition == "No-slip"
        for boundary in restored.boundary_conditions
    )


def test_version_eight_case_loads_with_laminar_defaults():
    data = case_to_dict(Case(name="legacy"))
    data["version"] = 8
    data["numerics"].pop("turbulence_model")
    data["numerics"].pop("turbulence_wall_treatment")
    data["initial_conditions"]["base_state"].pop("turbulence_k")
    data["initial_conditions"]["base_state"].pop("turbulence_omega")

    restored = case_from_dict(data)

    assert restored.numerics.turbulence_model == "None"
    assert restored.numerics.turbulence_wall_treatment == "Wall-resolved"
    assert restored.initial_conditions.base_state.turbulence_k == 1e-6
    assert restored.initial_conditions.base_state.turbulence_omega == 1.0


def test_case_round_trip_preserves_new_boundary_condition_values(tmp_path):
    case = Case(name="mass-flow")
    case.set_mesh(make_mesh())
    case.boundary_conditions[0].kind = "Pressure Inlet"
    case.boundary_conditions[0].values = {
        "pressure": 120000.0,
        "temperature": 310.0,
        "velocity_x": 1.0,
        "velocity_y": 0.0,
    }
    case.boundary_conditions[1].kind = "Mass Flow Outlet"
    case.boundary_conditions[1].values = {"mass_flow": 0.45}
    case.boundary_conditions[3].kind = "Mass Flow Inlet"
    case.boundary_conditions[3].values = {
        "mass_flow": 0.45,
        "temperature": 300.0,
        "velocity_x": 1.0,
        "velocity_y": 0.0,
    }
    path = tmp_path / "mass-flow.mcf.json"

    save_case(path, case)
    restored = load_case(path)

    assert case_to_dict(restored) == case_to_dict(case)


def test_replacing_mesh_preserves_wall_thermal_condition():
    case = Case()
    case.set_mesh(make_mesh())
    case.boundary_conditions[0].thermal_condition = "Heat Flux"
    case.boundary_conditions[0].values["heat_flux"] = 200.0

    case.set_mesh(make_mesh())

    assert case.boundary_conditions[0].thermal_condition == "Heat Flux"
    assert case.boundary_conditions[0].values["heat_flux"] == 200.0


def test_replacing_mesh_preserves_wall_mechanical_condition():
    case = Case()
    case.set_mesh(make_mesh())
    case.boundary_conditions[0].mechanical_condition = "Slip"

    case.set_mesh(make_mesh())

    assert case.boundary_conditions[0].mechanical_condition == "Slip"


def test_case_saves_relative_mesh_reference_and_reloads_mesh(tmp_path):
    mesh_path = tmp_path / "mesh.json"
    mesh_path.write_text(json.dumps(mesh_to_dict(make_mesh())), encoding="utf-8")
    case_path = tmp_path / "cases" / "channel.mcf.json"
    case_path.parent.mkdir()
    case = Case(name="channel")
    case.set_mesh(make_mesh(), mesh_path)

    save_case(case_path, case)

    saved_data = json.loads(case_path.read_text(encoding="utf-8"))
    assert saved_data["version"] == 12
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


def test_case_round_trip_preserves_final_solution(tmp_path):
    case = Case(name="solved", mesh=make_mesh())
    case.final_solution = (InitialState(0.9, 12.0, -3.0, 98000.0),)
    case.solution_time = 0.25
    case.solution_steps = 17
    case.solution_cpu_time = 4.5
    case.convergence_history = [
        (10, 0.15, 2.5, (0.4, 0.3, 0.2, 0.1)),
        (17, 0.25, 4.0, (0.2, 0.15, 0.1, 0.05)),
    ]
    path = tmp_path / "solved.mcf.json"

    save_case(path, case)
    saved_data = json.loads(path.read_text(encoding="utf-8"))
    restored = load_case(path)

    assert saved_data["version"] == 12
    assert saved_data["final_solution"] == [
        {
            "density": 0.9,
            "velocity_x": 12.0,
            "velocity_y": -3.0,
            "pressure": 98000.0,
            "turbulence_k": 1e-6,
            "turbulence_omega": 1.0,
        }
    ]
    assert restored.final_solution == case.final_solution
    assert restored.solution_time == case.solution_time
    assert restored.solution_steps == case.solution_steps
    assert restored.solution_cpu_time == case.solution_cpu_time
    assert restored.convergence_history == case.convergence_history
    assert case_to_dict(restored) == case_to_dict(case)


def test_case_loads_without_convergence_history_or_cpu_time():
    data = case_to_dict(Case(name="legacy", mesh=make_mesh()))
    data["version"] = 11
    data["final_solution"] = [
        {
            "density": 1.0,
            "velocity_x": 0.0,
            "velocity_y": 0.0,
            "pressure": 101325.0,
            "turbulence_k": 1e-6,
            "turbulence_omega": 1.0,
        }
    ]
    data["solution_time"] = 0.1
    data["solution_steps"] = 3
    data.pop("solution_cpu_time")
    data.pop("convergence_history")

    restored = case_from_dict(data)

    assert restored.solution_steps == 3
    assert restored.solution_cpu_time == 0.0
    assert restored.convergence_history == []


def test_case_load_rejects_convergence_history_beyond_saved_solution():
    data = case_to_dict(Case(name="solved", mesh=make_mesh()))
    data["final_solution"] = [
        {
            "density": 1.0,
            "velocity_x": 0.0,
            "velocity_y": 0.0,
            "pressure": 101325.0,
            "turbulence_k": 1e-6,
            "turbulence_omega": 1.0,
        }
    ]
    data["solution_steps"] = 2
    data["solution_cpu_time"] = 1.0
    data["convergence_history"] = [
        {"step": 3, "time": 0.1, "cpu_time": 0.5, "residuals": [1, 1, 1, 1]}
    ]

    with pytest.raises(ValueError, match="exceeds solution steps"):
        case_from_dict(data)


def test_legacy_case_without_final_solution_loads_without_one():
    data = case_to_dict(Case(name="legacy"))
    data["version"] = 5
    data.pop("final_solution")

    restored = case_from_dict(data)

    assert restored.final_solution is None
    assert restored.solution_time == 0.0
    assert restored.solution_steps == 0


def test_case_load_rejects_final_solution_with_wrong_cell_count():
    data = case_to_dict(Case(name="solved", mesh=make_mesh()))
    data["final_solution"] = []

    with pytest.raises(ValueError, match="state count"):
        case_from_dict(data)


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
    assert json.loads(path.read_text(encoding="utf-8"))["version"] == 12


def test_regularized_sst_formulation_round_trips(tmp_path):
    case = Case(name="regularized")
    case.numerics.turbulence_model = "Menter k-omega SST"
    case.numerics.turbulence_omega_formulation = "Regularized k-log-omega"
    path = tmp_path / "regularized.mcf.json"

    save_case(path, case)
    restored = load_case(path)

    assert restored.numerics.turbulence_omega_formulation == (
        "Regularized k-log-omega"
    )
    assert json.loads(path.read_text(encoding="utf-8"))["version"] == 12


def test_adaptive_mesh_settings_and_hierarchy_round_trip(tmp_path):
    case = Case(name="adaptive")
    source_path = tmp_path / "source-mesh.json"
    case.set_mesh(make_mesh(), source_path)
    case.mesh, _ = adapt_mesh(case.mesh, {0: "xy"}, set())
    case.numerics.adaptive_mesh = True
    case.numerics.adaptation_interval = 7
    case.numerics.adaptation_max_level = 4
    case.numerics.adaptation_refine_threshold = 0.15
    case.numerics.adaptation_variable = "Pressure"
    path = tmp_path / "adaptive.mcf.json"

    save_case(path, case)
    restored = load_case(path)
    saved_data = json.loads(path.read_text(encoding="utf-8"))

    assert isinstance(saved_data["mesh"], dict)
    assert restored.mesh_path is None
    assert restored.mesh == case.mesh
    assert restored.numerics.adaptive_mesh
    assert restored.numerics.adaptation_interval == 7
    assert restored.numerics.convergence_history_interval == 10
    assert restored.numerics.adaptation_max_level == 4
    assert restored.numerics.adaptation_refine_threshold == pytest.approx(0.15)
    assert restored.numerics.adaptation_variable == "Pressure"


def test_implicit_euler_and_large_cfl_round_trip(tmp_path):
    case = Case(name="implicit")
    case.numerics.time_integration = "Implicit Euler"
    case.numerics.cfl = 25.0
    case.numerics.gmres_tolerance = 1e-8
    case.numerics.gmres_max_iterations = 500
    case.numerics.gmres_krylov_dimension = 40
    case.numerics.implicit_preconditioner = "LU-SGS"
    path = tmp_path / "implicit.mcf.json"

    save_case(path, case)
    restored = load_case(path)

    assert restored.numerics.time_integration == "Implicit Euler"
    assert restored.numerics.cfl == pytest.approx(25.0)
    assert restored.numerics.gmres_tolerance == pytest.approx(1e-8)
    assert restored.numerics.gmres_max_iterations == 500
    assert restored.numerics.gmres_krylov_dimension == 40
    assert restored.numerics.implicit_preconditioner == "LU-SGS"


def test_regularized_sst_formulation_requires_wall_resolved_treatment():
    case = Case()
    case.numerics.turbulence_model = "Menter k-omega SST"
    case.numerics.turbulence_wall_treatment = "Wall functions"
    case.numerics.turbulence_omega_formulation = "Regularized k-log-omega"

    with pytest.raises(ValueError, match="requires wall-resolved treatment"):
        case_to_dict(case)


def test_version_four_periodic_case_defaults_end_time():
    case = Case(name="periodic", mesh=make_mesh())
    case.boundary_conditions[1].kind = "Periodic"
    case.boundary_conditions[3].kind = "Periodic"
    case.periodic_pairs.append(PeriodicBoundaryPair(1, 3, "x", 1.0))
    data = case_to_dict(case)
    data["version"] = 4
    data["numerics"].pop("end_time")
    data["numerics"].pop("convergence_history_interval")

    restored = case_from_dict(data)

    assert restored.periodic_pairs == case.periodic_pairs
    assert restored.numerics.end_time == NumericalSettings().end_time
    assert (
        restored.numerics.convergence_history_interval
        == NumericalSettings().convergence_history_interval
    )


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


def test_periodic_pair_detection_tolerates_coordinate_noise_and_face_order():
    y = (0.0, 0.25, 0.5, 0.75, 1.0)
    left_x = (0.0, 3e-14, -2e-14, 4e-14, -1e-14)
    right_x = (1.0 + 2e-14, 1.0 - 3e-14, 1.0 + 1e-14, 1.0 - 1e-14, 1.0 + 3e-14)
    points = tuple(zip(left_x, y)) + tuple(zip(right_x, y))
    left_faces = tuple((index, index + 1) for index in range(4))
    right_faces = tuple(
        (6 + index, 5 + index) for index in range(3, -1, -1)
    )
    mesh = FiniteVolumeMesh(
        points=points,
        cells=(),
        faces=left_faces + right_faces,
        face_owners=(),
        face_neighbours=(),
        face_patch_ids=(),
        patches=(
            MeshPatch(0, "inlet", (0, 1, 2, 3)),
            MeshPatch(1, "outlet", (4, 5, 6, 7)),
        ),
        cell_face_ids=(),
        cell_face_signs=(),
    )

    assert detect_periodic_pair(mesh, 0, 1) == ("x", pytest.approx(1.0))
    pairs, shift = match_periodic_faces(mesh, 0, 1)
    assert set(pairs) == {(0, 7), (1, 6), (2, 5), (3, 4)}
    assert shift == pytest.approx((1.0, 0.0))


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
