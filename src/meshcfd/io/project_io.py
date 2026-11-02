"""Persistence for MeshCFD case setup files."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

from meshcfd.core.expressions import ExpressionError, compile_expression
from meshcfd.core.model import (
    BOUNDARY_TYPES,
    FLUX_SCHEMES,
    LIMITER_SCHEMES,
    RECONSTRUCTION_SCHEMES,
    TIME_INTEGRATION_SCHEMES,
    BoundaryCondition,
    Case,
    FiniteVolumeMesh,
    InitialConditionBlock,
    InitialConditionSettings,
    InitialState,
    NumericalSettings,
    PeriodicBoundaryPair,
)
from meshcfd.core.periodic import detect_periodic_pair
from meshcfd.io.mesh_io import load_mesh, mesh_from_dict, mesh_to_dict

PROJECT_FORMAT = "meshcfd-case"
PROJECT_VERSION = 10


def case_to_dict(
    case: Case, case_path: str | Path | None = None
) -> dict[str, Any]:
    end_time = _positive_float(case.numerics.end_time, "End time")
    for condition in case.boundary_conditions:
        _mechanical_condition(condition.mechanical_condition)
        _thermal_condition(condition.thermal_condition)
    if case.numerics.viscosity_model not in ("Inviscid", "Constant", "Sutherland"):
        raise ValueError("Unsupported viscosity model.")
    if case.numerics.turbulence_model not in ("None", "Menter k-omega SST"):
        raise ValueError("Unsupported turbulence model.")
    if case.numerics.turbulence_wall_treatment not in (
        "Wall-resolved",
        "Wall functions",
    ):
        raise ValueError("Unsupported turbulence wall treatment.")
    if case.numerics.turbulence_omega_formulation not in (
        "Standard SST",
        "Regularized k-log-omega",
    ):
        raise ValueError("Unsupported SST omega formulation.")
    if (
        case.numerics.turbulence_model == "Menter k-omega SST"
        and case.numerics.turbulence_omega_formulation
        == "Regularized k-log-omega"
        and case.numerics.turbulence_wall_treatment != "Wall-resolved"
    ):
        raise ValueError(
            "The regularized k-log-omega formulation requires wall-resolved treatment."
        )
    _positive_float(case.numerics.prandtl_number, "Prandtl number")
    _positive_float(case.numerics.dynamic_viscosity, "Dynamic viscosity")
    _positive_float(
        case.numerics.sutherland_reference_viscosity,
        "Sutherland reference viscosity",
    )
    _positive_float(
        case.numerics.sutherland_reference_temperature,
        "Sutherland reference temperature",
    )
    _positive_float(
        case.numerics.sutherland_temperature, "Sutherland temperature"
    )
    _finite_float(case.numerics.acceleration_x, "Acceleration X")
    _finite_float(case.numerics.acceleration_y, "Acceleration Y")
    _validate_periodic_pairs(
        case.mesh, case.boundary_conditions, case.periodic_pairs
    )
    if case.mesh is None:
        mesh_data = None
    elif case.mesh_path is None:
        mesh_data = mesh_to_dict(case.mesh)
    else:
        if case_path is None:
            raise ValueError("A case file path is required to save the mesh reference.")
        mesh_path = Path(case.mesh_path).resolve()
        case_directory = Path(case_path).resolve().parent
        try:
            mesh_data = Path(os.path.relpath(mesh_path, case_directory)).as_posix()
        except ValueError as error:
            raise ValueError(
                "The mesh file and case file must be on the same drive "
                "to save a relative mesh reference."
            ) from error
    initial_conditions = _initial_conditions_to_dict(case.initial_conditions)
    final_solution = _final_solution_to_dict(case.final_solution, case.mesh)
    if case.final_solution is None:
        if case.solution_time != 0.0 or case.solution_steps != 0:
            raise ValueError("Solution time and steps require a final solution.")
        solution_time = 0.0
        solution_steps = 0
    else:
        solution_time = _finite_non_negative_float(
            case.solution_time, "Solution time"
        )
        solution_steps = _non_negative_integer(case.solution_steps, "Solution steps")
    return {
        "format": PROJECT_FORMAT,
        "version": PROJECT_VERSION,
        "name": case.name,
        "mesh": mesh_data,
        "boundary_conditions": [
            {
                "patch_id": condition.patch_id,
                "patch_name": condition.patch_name,
                "kind": condition.kind,
                "values": dict(condition.values),
                "mechanical_condition": condition.mechanical_condition,
                "thermal_condition": condition.thermal_condition,
            }
            for condition in case.boundary_conditions
        ],
        "periodic_pairs": [
            {
                "patch_a_id": pair.patch_a_id,
                "patch_b_id": pair.patch_b_id,
                "axis": pair.axis,
                "pitch": pair.pitch,
            }
            for pair in case.periodic_pairs
        ],
        "numerics": {
            "reconstruction": case.numerics.reconstruction,
            "limiter": case.numerics.limiter,
            "flux": case.numerics.flux,
            "time_integration": case.numerics.time_integration,
            "cfl": case.numerics.cfl,
            "max_iterations": case.numerics.max_iterations,
            "end_time": end_time,
            "residual_tolerance": case.numerics.residual_tolerance,
            "gamma": case.numerics.gamma,
            "gas_constant": case.numerics.gas_constant,
            "viscosity_model": case.numerics.viscosity_model,
            "dynamic_viscosity": case.numerics.dynamic_viscosity,
            "sutherland_reference_viscosity": (
                case.numerics.sutherland_reference_viscosity
            ),
            "sutherland_reference_temperature": (
                case.numerics.sutherland_reference_temperature
            ),
            "sutherland_temperature": case.numerics.sutherland_temperature,
            "prandtl_number": case.numerics.prandtl_number,
            "acceleration_x": case.numerics.acceleration_x,
            "acceleration_y": case.numerics.acceleration_y,
            "turbulence_model": case.numerics.turbulence_model,
            "turbulence_wall_treatment": (
                case.numerics.turbulence_wall_treatment
            ),
            "turbulence_omega_formulation": (
                case.numerics.turbulence_omega_formulation
            ),
        },
        "initial_conditions": initial_conditions,
        "final_solution": final_solution,
        "solution_time": solution_time,
        "solution_steps": solution_steps,
    }


def case_from_dict(
    data: dict[str, Any], case_path: str | Path | None = None
) -> Case:
    version = data.get("version")
    if data.get("format") != PROJECT_FORMAT or version not in (
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        PROJECT_VERSION,
    ):
        raise ValueError(
            f"Unsupported case format. Expected MeshCFD case version 1 through "
            f"{PROJECT_VERSION}."
        )
    name = data.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Case name must not be empty.")

    mesh_data = data.get("mesh")
    mesh = None
    mesh_path: Path | None = None
    if mesh_data is not None:
        if isinstance(mesh_data, dict):
            mesh = mesh_from_dict(mesh_data)
        elif version >= 2:
            if not isinstance(mesh_data, str) or not mesh_data.strip():
                raise ValueError("Case mesh reference must be a non-empty path or null.")
            reference = Path(mesh_data)
            if reference.is_absolute():
                raise ValueError("Case mesh reference must be relative.")
            if case_path is None:
                raise ValueError("A case file path is required to resolve the mesh reference.")
            mesh_path = (Path(case_path).resolve().parent / reference).resolve()
            mesh = load_mesh(mesh_path)
        else:
            raise ValueError("Case mesh data must be an object or null.")

    raw_conditions = data.get("boundary_conditions", [])
    if not isinstance(raw_conditions, list):
        raise ValueError("Boundary conditions must be a list.")
    conditions = []
    seen_patch_ids = set()
    seen_patch_names = set()
    for raw_condition in raw_conditions:
        if not isinstance(raw_condition, dict):
            raise ValueError("Each boundary condition must be an object.")
        patch_id = raw_condition.get("patch_id")
        patch_name = raw_condition.get("patch_name")
        kind = raw_condition.get("kind")
        values = raw_condition.get("values", {})
        if isinstance(patch_id, bool) or not isinstance(patch_id, int):
            raise ValueError("Boundary patch identifiers must be integers.")
        if not isinstance(patch_name, str) or not patch_name.strip():
            raise ValueError("Boundary patch names must not be empty.")
        if not isinstance(kind, str) or not kind:
            raise ValueError(f"Boundary condition for {patch_name} needs a type.")
        if kind not in BOUNDARY_TYPES:
            raise ValueError(f"Unsupported boundary type '{kind}'.")
        if patch_id in seen_patch_ids or patch_name in seen_patch_names:
            raise ValueError("Boundary patch identifiers and names must be unique.")
        if not isinstance(values, dict):
            raise ValueError(f"Boundary values for {patch_name} must be an object.")
        numeric_values = {}
        for key, value in values.items():
            if not isinstance(key, str):
                raise ValueError("Boundary value names must be strings.")
            if isinstance(value, bool):
                raise ValueError(f"Boundary value '{key}' must be numeric.")
            try:
                number = float(value)
            except (OverflowError, TypeError, ValueError) as error:
                raise ValueError(f"Boundary value '{key}' must be numeric.") from error
            if not math.isfinite(number):
                raise ValueError(f"Boundary value '{key}' must be finite.")
            numeric_values[key] = number
        conditions.append(
            BoundaryCondition(
                patch_id,
                patch_name,
                kind,
                numeric_values,
                thermal_condition=_thermal_condition(
                    raw_condition.get("thermal_condition", "Temperature")
                ),
                mechanical_condition=_mechanical_condition(
                    raw_condition.get("mechanical_condition", "No-slip")
                ),
            )
        )
        seen_patch_ids.add(patch_id)
        seen_patch_names.add(patch_name)

    raw_numerics = data.get("numerics", {})
    if not isinstance(raw_numerics, dict):
        raise ValueError("Numerical settings must be an object.")
    defaults = NumericalSettings()
    settings = NumericalSettings(
        reconstruction=raw_numerics.get("reconstruction", defaults.reconstruction),
        limiter=raw_numerics.get("limiter", defaults.limiter),
        flux=raw_numerics.get("flux", defaults.flux),
        time_integration=raw_numerics.get(
            "time_integration", defaults.time_integration
        ),
        cfl=_positive_float(raw_numerics.get("cfl", defaults.cfl), "CFL"),
        max_iterations=_positive_integer(
            raw_numerics.get("max_iterations", defaults.max_iterations),
            "Maximum iterations",
        ),
        end_time=_positive_float(
            raw_numerics.get("end_time", defaults.end_time), "End time"
        ),
        residual_tolerance=_positive_float(
            raw_numerics.get(
                "residual_tolerance", defaults.residual_tolerance
            ),
            "Residual tolerance",
        ),
        gamma=_positive_float(raw_numerics.get("gamma", defaults.gamma), "Gamma"),
        gas_constant=_positive_float(
            raw_numerics.get("gas_constant", defaults.gas_constant),
            "Gas constant",
        ),
        viscosity_model=raw_numerics.get(
            "viscosity_model", defaults.viscosity_model
        ),
        dynamic_viscosity=_positive_float(
            raw_numerics.get("dynamic_viscosity", defaults.dynamic_viscosity),
            "Dynamic viscosity",
        ),
        sutherland_reference_viscosity=_positive_float(
            raw_numerics.get(
                "sutherland_reference_viscosity",
                defaults.sutherland_reference_viscosity,
            ),
            "Sutherland reference viscosity",
        ),
        sutherland_reference_temperature=_positive_float(
            raw_numerics.get(
                "sutherland_reference_temperature",
                defaults.sutherland_reference_temperature,
            ),
            "Sutherland reference temperature",
        ),
        sutherland_temperature=_positive_float(
            raw_numerics.get(
                "sutherland_temperature", defaults.sutherland_temperature
            ),
            "Sutherland temperature",
        ),
        prandtl_number=_positive_float(
            raw_numerics.get("prandtl_number", defaults.prandtl_number),
            "Prandtl number",
        ),
        acceleration_x=_finite_float(
            raw_numerics.get("acceleration_x", defaults.acceleration_x),
            "Acceleration X",
        ),
        acceleration_y=_finite_float(
            raw_numerics.get("acceleration_y", defaults.acceleration_y),
            "Acceleration Y",
        ),
        turbulence_model=raw_numerics.get(
            "turbulence_model", defaults.turbulence_model
        ),
        turbulence_wall_treatment=raw_numerics.get(
            "turbulence_wall_treatment", defaults.turbulence_wall_treatment
        ),
        turbulence_omega_formulation=raw_numerics.get(
            "turbulence_omega_formulation",
            defaults.turbulence_omega_formulation,
        ),
    )
    for key, choices in (
        ("reconstruction", RECONSTRUCTION_SCHEMES),
        ("limiter", LIMITER_SCHEMES),
        ("flux", FLUX_SCHEMES),
        ("time_integration", TIME_INTEGRATION_SCHEMES),
        ("viscosity_model", ("Inviscid", "Constant", "Sutherland")),
        ("turbulence_model", ("None", "Menter k-omega SST")),
        ("turbulence_wall_treatment", ("Wall-resolved", "Wall functions")),
        (
            "turbulence_omega_formulation",
            ("Standard SST", "Regularized k-log-omega"),
        ),
    ):
        if getattr(settings, key) not in choices:
            raise ValueError(f"Unsupported numerical setting for {key}.")
    if settings.gamma <= 1.0:
        raise ValueError("Gamma must be greater than 1.")
    if (
        settings.turbulence_model == "Menter k-omega SST"
        and settings.turbulence_omega_formulation
        == "Regularized k-log-omega"
        and settings.turbulence_wall_treatment != "Wall-resolved"
    ):
        raise ValueError(
            "The regularized k-log-omega formulation requires wall-resolved treatment."
        )

    initial_conditions = _initial_conditions_from_dict(
        data.get("initial_conditions", {})
    )
    final_solution = _final_solution_from_dict(data.get("final_solution"), mesh)
    solution_time = _finite_non_negative_float(
        data.get("solution_time", 0.0), "Solution time"
    )
    solution_steps = _non_negative_integer(
        data.get("solution_steps", 0), "Solution steps"
    )
    if final_solution is None and (solution_time != 0.0 or solution_steps != 0):
        raise ValueError("Solution time and steps require a final solution.")

    if mesh is not None:
        valid_patches = {patch.id: patch.name for patch in mesh.patches}
        if any(
            condition.patch_id not in valid_patches
            or valid_patches[condition.patch_id] != condition.patch_name
            for condition in conditions
        ):
            raise ValueError("A boundary condition refers to a patch not in the mesh.")
        configured_ids = {condition.patch_id for condition in conditions}
        conditions.extend(
            BoundaryCondition(patch.id, patch.name)
            for patch in mesh.patches
            if patch.id not in configured_ids
        )
    periodic_pairs = _periodic_pairs_from_dict(
        data.get("periodic_pairs", []), mesh
    )
    _validate_periodic_pairs(mesh, conditions, periodic_pairs)
    return Case(
        name=name,
        mesh=mesh,
        boundary_conditions=conditions,
        numerics=settings,
        mesh_path=mesh_path,
        initial_conditions=initial_conditions,
        periodic_pairs=periodic_pairs,
        final_solution=final_solution,
        solution_time=solution_time,
        solution_steps=solution_steps,
    )


def _final_solution_to_dict(
    states: tuple[InitialState, ...] | None,
    mesh: FiniteVolumeMesh | None,
) -> list[dict[str, float]] | None:
    if states is None:
        return None
    if mesh is None:
        raise ValueError("A final solution requires a mesh.")
    if len(states) != len(mesh.cells):
        raise ValueError("Final solution state count must match the mesh cell count.")
    result = []
    for state in states:
        density = _positive_float(state.density, "Final solution density")
        pressure = _positive_float(state.pressure, "Final solution pressure")
        velocity_x = _finite_float(state.velocity_x, "Final solution velocity X")
        velocity_y = _finite_float(state.velocity_y, "Final solution velocity Y")
        result.append(
            {
                "density": density,
                "velocity_x": velocity_x,
                "velocity_y": velocity_y,
                "pressure": pressure,
                "turbulence_k": _finite_non_negative_float(
                    state.turbulence_k, "Final solution turbulence k"
                ),
                "turbulence_omega": _positive_float(
                    state.turbulence_omega, "Final solution turbulence omega"
                ),
            }
        )
    return result


def _final_solution_from_dict(
    raw_states: Any, mesh: FiniteVolumeMesh | None
) -> tuple[InitialState, ...] | None:
    if raw_states is None:
        return None
    if mesh is None:
        raise ValueError("A final solution requires a mesh.")
    if not isinstance(raw_states, list):
        raise ValueError("Final solution must be a list of cell states or null.")
    if len(raw_states) != len(mesh.cells):
        raise ValueError("Final solution state count must match the mesh cell count.")

    states = []
    for index, raw_state in enumerate(raw_states):
        if not isinstance(raw_state, dict):
            raise ValueError(f"Final solution state {index} must be an object.")
        states.append(
            InitialState(
                density=_positive_float(
                    raw_state.get("density"), f"Final solution density at cell {index}"
                ),
                velocity_x=_finite_float(
                    raw_state.get("velocity_x"),
                    f"Final solution velocity X at cell {index}",
                ),
                velocity_y=_finite_float(
                    raw_state.get("velocity_y"),
                    f"Final solution velocity Y at cell {index}",
                ),
                pressure=_positive_float(
                    raw_state.get("pressure"),
                    f"Final solution pressure at cell {index}",
                ),
                turbulence_k=_finite_non_negative_float(
                    raw_state.get("turbulence_k", 1e-6),
                    f"Final solution turbulence k at cell {index}",
                ),
                turbulence_omega=_positive_float(
                    raw_state.get("turbulence_omega", 1.0),
                    f"Final solution turbulence omega at cell {index}",
                ),
            )
        )
    return tuple(states)


def _periodic_pairs_from_dict(
    raw_pairs: Any, mesh: FiniteVolumeMesh | None
) -> list[PeriodicBoundaryPair]:
    if not isinstance(raw_pairs, list):
        raise ValueError("Periodic boundary pairs must be a list.")
    pairs = []
    for raw_pair in raw_pairs:
        if not isinstance(raw_pair, dict):
            raise ValueError("Each periodic boundary pair must be an object.")
        patch_a_id = raw_pair.get("patch_a_id")
        patch_b_id = raw_pair.get("patch_b_id")
        if any(
            isinstance(patch_id, bool) or not isinstance(patch_id, int)
            for patch_id in (patch_a_id, patch_b_id)
        ):
            raise ValueError("Periodic patch identifiers must be integers.")
        axis = raw_pair.get("axis")
        if axis not in ("x", "y"):
            raise ValueError("Periodic axis must be 'x' or 'y'.")
        pitch = _positive_float(raw_pair.get("pitch"), "Periodic pitch")
        pairs.append(PeriodicBoundaryPair(patch_a_id, patch_b_id, axis, pitch))
    return pairs


def _validate_periodic_pairs(
    mesh: FiniteVolumeMesh | None,
    conditions: list[BoundaryCondition],
    pairs: list[PeriodicBoundaryPair],
) -> None:
    if not pairs and not any(condition.kind == "Periodic" for condition in conditions):
        return
    if mesh is None:
        raise ValueError("Periodic boundary conditions require a mesh.")

    condition_kinds = {condition.patch_id: condition.kind for condition in conditions}
    paired_patch_ids = set()
    for pair in pairs:
        if pair.patch_a_id == pair.patch_b_id:
            raise ValueError("A periodic pair must contain two different patches.")
        if pair.patch_a_id in paired_patch_ids or pair.patch_b_id in paired_patch_ids:
            raise ValueError("A patch can belong to only one periodic pair.")
        paired_patch_ids.update((pair.patch_a_id, pair.patch_b_id))
        if any(
            condition_kinds.get(patch_id) != "Periodic"
            for patch_id in (pair.patch_a_id, pair.patch_b_id)
        ):
            raise ValueError("Both patches in a periodic pair must use the Periodic type.")
        try:
            axis, pitch = detect_periodic_pair(
                mesh, pair.patch_a_id, pair.patch_b_id
            )
        except ValueError as error:
            raise ValueError(f"Invalid periodic patch pair: {error}") from error
        if pair.axis != axis or not math.isclose(
            pair.pitch, pitch, rel_tol=1e-9, abs_tol=max(pitch, 1.0) * 1e-9
        ):
            raise ValueError("Stored periodic axis or pitch does not match the mesh.")
    if paired_patch_ids != {
        condition.patch_id
        for condition in conditions
        if condition.kind == "Periodic"
    }:
        raise ValueError("Each Periodic boundary must belong to exactly one pair.")


def _initial_conditions_from_dict(data: Any) -> InitialConditionSettings:
    if not isinstance(data, dict):
        raise ValueError("Initial conditions must be an object.")
    raw_base = data.get("base_state", {})
    if not isinstance(raw_base, dict):
        raise ValueError("Initial-condition base state must be an object.")
    defaults = InitialState()
    base_state = InitialState(
        density=_positive_float(raw_base.get("density", defaults.density), "Density"),
        velocity_x=_finite_float(
            raw_base.get("velocity_x", defaults.velocity_x), "Velocity X"
        ),
        velocity_y=_finite_float(
            raw_base.get("velocity_y", defaults.velocity_y), "Velocity Y"
        ),
        pressure=_positive_float(
            raw_base.get("pressure", defaults.pressure), "Pressure"
        ),
        turbulence_k=_finite_non_negative_float(
            raw_base.get("turbulence_k", defaults.turbulence_k),
            "Initial turbulence k",
        ),
        turbulence_omega=_positive_float(
            raw_base.get("turbulence_omega", defaults.turbulence_omega),
            "Initial turbulence omega",
        ),
    )
    raw_blocks = data.get("blocks", [])
    if not isinstance(raw_blocks, list):
        raise ValueError("Initial-condition blocks must be a list.")
    blocks = []
    for index, raw_block in enumerate(raw_blocks):
        if not isinstance(raw_block, dict):
            raise ValueError("Each initial-condition block must be an object.")
        name = raw_block.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Initial-condition block {index + 1} needs a name.")
        expressions = {
            key: raw_block.get(key)
            for key in ("domain", "density", "velocity_x", "velocity_y", "pressure")
        }
        if not all(isinstance(value, str) for value in expressions.values()):
            raise ValueError(
                f"Initial-condition block '{name}' has an invalid expression."
            )
        try:
            compile_expression(expressions["domain"], boolean=True)
            for key in ("density", "velocity_x", "velocity_y", "pressure"):
                compile_expression(expressions[key])
        except ExpressionError as error:
            raise ValueError(
                f"Invalid expression in initial-condition block '{name}': {error}"
            ) from error
        blocks.append(InitialConditionBlock(name=name, **expressions))
    return InitialConditionSettings(base_state=base_state, blocks=blocks)


def _initial_conditions_to_dict(
    settings: InitialConditionSettings,
) -> dict[str, Any]:
    state = settings.base_state
    _positive_float(state.density, "Density")
    _finite_float(state.velocity_x, "Velocity X")
    _finite_float(state.velocity_y, "Velocity Y")
    _positive_float(state.pressure, "Pressure")
    _finite_non_negative_float(state.turbulence_k, "Initial turbulence k")
    _positive_float(state.turbulence_omega, "Initial turbulence omega")
    blocks = []
    for block in settings.blocks:
        if not isinstance(block.name, str) or not block.name.strip():
            raise ValueError("Initial-condition block name must not be empty.")
        try:
            compile_expression(block.domain, boolean=True)
            for key in ("density", "velocity_x", "velocity_y", "pressure"):
                compile_expression(getattr(block, key))
        except ExpressionError as error:
            raise ValueError(
                f"Invalid expression in initial-condition block '{block.name}': {error}"
            ) from error
        blocks.append(
            {
                "name": block.name,
                "domain": block.domain,
                "density": block.density,
                "velocity_x": block.velocity_x,
                "velocity_y": block.velocity_y,
                "pressure": block.pressure,
            }
        )
    return {
        "base_state": {
            "density": state.density,
            "velocity_x": state.velocity_x,
            "velocity_y": state.velocity_y,
            "pressure": state.pressure,
            "turbulence_k": state.turbulence_k,
            "turbulence_omega": state.turbulence_omega,
        },
        "blocks": blocks,
    }


def _finite_float(value: Any, description: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{description} must be numeric.")
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{description} must be numeric.") from error
    if not math.isfinite(number):
        raise ValueError(f"{description} must be finite.")
    return number


def _finite_non_negative_float(value: Any, description: str) -> float:
    number = _finite_float(value, description)
    if number < 0.0:
        raise ValueError(f"{description} must be non-negative.")
    return number


def _non_negative_integer(value: Any, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{description} must be a non-negative integer.")
    return value


def _positive_float(value: Any, description: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{description} must be a positive number.")
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{description} must be a positive number.") from error
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{description} must be a positive finite number.")
    return number


def _thermal_condition(value: Any) -> str:
    if value not in ("Temperature", "Heat Flux"):
        raise ValueError("Wall thermal condition must be Temperature or Heat Flux.")
    return value


def _mechanical_condition(value: Any) -> str:
    if value not in ("Slip", "No-slip"):
        raise ValueError("Wall mechanical condition must be Slip or No-slip.")
    return value


def _positive_integer(value: Any, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{description} must be a positive integer.")
    return value


def load_case(path: str | Path) -> Case:
    source = Path(path)
    try:
        with source.open("r", encoding="utf-8") as stream:
            data = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read case file '{source}': {error}") from error
    if not isinstance(data, dict):
        raise ValueError("Case file must contain a JSON object.")
    return case_from_dict(data, source)


def save_case(path: str | Path, case: Case) -> None:
    output = Path(path)
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(
            case_to_dict(case, output),
            stream,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        stream.write("\n")
