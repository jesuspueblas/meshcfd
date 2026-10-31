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
    InitialConditionBlock,
    InitialConditionSettings,
    InitialState,
    NumericalSettings,
)
from meshcfd.io.mesh_io import load_mesh, mesh_from_dict, mesh_to_dict

PROJECT_FORMAT = "meshcfd-case"
PROJECT_VERSION = 3


def case_to_dict(
    case: Case, case_path: str | Path | None = None
) -> dict[str, Any]:
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
            }
            for condition in case.boundary_conditions
        ],
        "numerics": {
            "reconstruction": case.numerics.reconstruction,
            "limiter": case.numerics.limiter,
            "flux": case.numerics.flux,
            "time_integration": case.numerics.time_integration,
            "cfl": case.numerics.cfl,
            "max_iterations": case.numerics.max_iterations,
            "residual_tolerance": case.numerics.residual_tolerance,
            "gamma": case.numerics.gamma,
            "gas_constant": case.numerics.gas_constant,
        },
        "initial_conditions": initial_conditions,
    }


def case_from_dict(
    data: dict[str, Any], case_path: str | Path | None = None
) -> Case:
    version = data.get("version")
    if data.get("format") != PROJECT_FORMAT or version not in (1, 2, PROJECT_VERSION):
        raise ValueError(
            f"Unsupported case format. Expected MeshCFD case version 1, 2 or "
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
            BoundaryCondition(patch_id, patch_name, kind, numeric_values)
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
    )
    for key, choices in (
        ("reconstruction", RECONSTRUCTION_SCHEMES),
        ("limiter", LIMITER_SCHEMES),
        ("flux", FLUX_SCHEMES),
        ("time_integration", TIME_INTEGRATION_SCHEMES),
    ):
        if getattr(settings, key) not in choices:
            raise ValueError(f"Unsupported numerical setting for {key}.")
    if settings.gamma <= 1.0:
        raise ValueError("Gamma must be greater than 1.")

    initial_conditions = _initial_conditions_from_dict(
        data.get("initial_conditions", {})
    )

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
    return Case(
        name=name,
        mesh=mesh,
        boundary_conditions=conditions,
        numerics=settings,
        mesh_path=mesh_path,
        initial_conditions=initial_conditions,
    )


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
