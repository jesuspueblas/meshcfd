"""Explicit finite-volume solver for compressible ideal-gas flow."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, replace
from time import thread_time
from typing import Callable

import numpy as np
from scipy.sparse import coo_matrix, eye, kron, tril, triu
from scipy.sparse.linalg import (
    LinearOperator,
    gmres,
    spilu,
    spsolve_triangular,
)

from meshcfd.core.adaptation import (
    AdaptationBalanceError,
    adapt_mesh,
    cell_areas,
    select_adaptation,
)
from meshcfd.core.euler import (
    conservative_to_primitive,
    numerical_flux,
    primitive_to_conservative,
)
from meshcfd.core.initial_conditions import evaluate_initial_conditions
from meshcfd.core.model import (
    Case,
    FiniteVolumeMesh,
    InitialState,
    NumericalSettings,
    PeriodicBoundaryPair,
)
from meshcfd.core.periodic import detect_periodic_pair, match_periodic_faces


class SolverError(ValueError):
    """Raised when a case cannot be advanced by the Euler solver."""


@dataclass(frozen=True)
class SolverProgress:
    step: int
    time: float
    time_step: float
    residuals: tuple[float, ...]
    states: tuple[InitialState, ...] | None = None
    mesh: FiniteVolumeMesh | None = None
    cpu_time: float = 0.0
    warning: str | None = None


@dataclass(frozen=True)
class SolverResult:
    states: tuple[InitialState, ...]
    time: float
    steps: int
    residual: float
    completed: bool
    cpu_time: float = 0.0


@dataclass(frozen=True)
class _PressureInlet:
    faces: np.ndarray
    total_pressure: float
    total_temperature: float
    direction: np.ndarray
    patch_name: str


@dataclass(frozen=True)
class _MassFlowInlet:
    faces: np.ndarray
    mass_flow: float
    total_temperature: float
    direction: np.ndarray
    patch_name: str


@dataclass(frozen=True)
class _MassFlowOutlet:
    faces: np.ndarray
    mass_flow: float
    patch_name: str


@dataclass(frozen=True)
class _WallThermalBoundary:
    faces: np.ndarray
    condition: str
    value: float
    patch_name: str
    mechanical_condition: str


@dataclass(frozen=True)
class _TurbulenceInlet:
    faces: np.ndarray
    intensity: float
    length_scale: float
    patch_name: str


@dataclass(frozen=True)
class _Geometry:
    centers: np.ndarray
    areas: np.ndarray
    face_centers: np.ndarray
    face_normals: np.ndarray
    face_lengths: np.ndarray
    owners: np.ndarray
    neighbours: np.ndarray
    patch_ids: np.ndarray
    patch_kinds: dict[int, str]
    patch_mechanical_conditions: dict[int, str]
    wall_distances: np.ndarray
    interior_faces: np.ndarray
    wall_faces: np.ndarray
    symmetry_faces: np.ndarray
    diffusive_boundary_faces: np.ndarray
    periodic_faces_a: np.ndarray
    periodic_faces_b: np.ndarray
    periodic_pairs: tuple[PeriodicBoundaryPair, ...]
    velocity_inlet_faces: np.ndarray
    velocity_inlet_states: np.ndarray
    pressure_outlet_faces: np.ndarray
    pressure_outlet_pressures: np.ndarray
    far_field_faces: np.ndarray
    far_field_states: np.ndarray
    pressure_inlets: tuple[_PressureInlet, ...]
    mass_flow_inlets: tuple[_MassFlowInlet, ...]
    mass_flow_outlets: tuple[_MassFlowOutlet, ...]
    wall_thermal_boundaries: tuple[_WallThermalBoundary, ...]
    turbulence_inlets: tuple[_TurbulenceInlet, ...]
    gradient_sources: np.ndarray
    gradient_neighbours: np.ndarray
    gradient_dx: np.ndarray
    gradient_dy: np.ndarray
    parabolic_sources: np.ndarray
    parabolic_neighbours: np.ndarray
    parabolic_dx: np.ndarray
    parabolic_dy: np.ndarray
    limiter_cells: np.ndarray
    limiter_dx: np.ndarray
    limiter_dy: np.ndarray


def solve_case(
    case: Case,
    initial_states: tuple[InitialState, ...] | None = None,
    progress_callback: Callable[[SolverProgress], None] | None = None,
    cancel_callback: Callable[[], bool] | None = None,
    live_solution_updates: bool = False,
    start_time: float = 0.0,
    step_offset: int = 0,
    pause_callback: Callable[[], None] | None = None,
    solution_callback: Callable[[int, float, tuple[InitialState, ...]], None] | None = None,
    solution_interval: int = 1,
    cpu_time_offset: float = 0.0,
) -> SolverResult:
    """Advance a transient case or iterate a steady pseudo-time solution."""
    cpu_start = thread_time()
    if case.mesh is None:
        raise SolverError("Import a mesh before running the solver.")
    settings = case.numerics
    if settings.reconstruction not in (
        "First order",
        "MUSCL linear",
        "MUSCL parabolic",
    ):
        raise SolverError(f"Unsupported reconstruction '{settings.reconstruction}'.")
    if settings.limiter not in (
        "Barth-Jespersen",
        "Venkatakrishnan",
        "Minmod",
        "Van Leer",
    ):
        raise SolverError(f"Unsupported limiter '{settings.limiter}'.")
    if settings.flux not in ("HLLC", "Roe", "Rusanov"):
        raise SolverError(f"Unsupported convective flux '{settings.flux}'.")
    if settings.time_integration not in (
        "Explicit Euler",
        "Explicit RK2",
        "Explicit RK3",
        "Explicit RK4",
        "Implicit Euler",
    ):
        raise SolverError(
            f"Unsupported time integration '{settings.time_integration}'."
        )
    if settings.time_step_mode not in (
        "Global adaptive",
        "Global fixed",
        "Local (steady)",
    ):
        raise SolverError(f"Unsupported time-stepping mode '{settings.time_step_mode}'.")
    local_steady = settings.time_step_mode == "Local (steady)"
    if not math.isfinite(settings.fixed_time_step) or settings.fixed_time_step <= 0.0:
        raise SolverError("Fixed time step must be positive and finite.")
    if (
        not math.isfinite(settings.residual_tolerance)
        or settings.residual_tolerance <= 0.0
    ):
        raise SolverError("Residual tolerance must be positive and finite.")
    maximum_cfl = 1000.0 if settings.time_integration == "Implicit Euler" else 1.0
    if not math.isfinite(settings.cfl) or not 0.0 < settings.cfl <= maximum_cfl:
        raise SolverError(
            f"{settings.time_integration} requires a CFL number in (0, "
            f"{maximum_cfl:g}]."
        )
    if (
        isinstance(settings.max_iterations, bool)
        or not isinstance(settings.max_iterations, int)
        or settings.max_iterations <= 0
    ):
        raise SolverError("Maximum iterations must be a positive integer.")
    if (
        not math.isfinite(settings.gmres_tolerance)
        or not 0.0 < settings.gmres_tolerance < 1.0
    ):
        raise SolverError("GMRES tolerance must be finite and in (0, 1).")
    for value, label in (
        (settings.gmres_max_iterations, "Maximum GMRES iterations"),
        (settings.gmres_krylov_dimension, "GMRES Krylov dimension"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise SolverError(f"{label} must be a positive integer.")
    if settings.implicit_preconditioner not in ("ILU", "LU-SGS"):
        raise SolverError("Unsupported implicit preconditioner.")
    if (
        not local_steady
        and (not math.isfinite(settings.end_time) or settings.end_time <= 0.0)
    ):
        raise SolverError("End time must be positive and finite.")
    if not math.isfinite(start_time) or start_time < 0.0:
        raise SolverError("Current solution time must be finite and non-negative.")
    if not local_steady and start_time >= settings.end_time:
        raise SolverError(
            "End time must be greater than the current solution time to continue."
        )
    if (
        isinstance(step_offset, bool)
        or not isinstance(step_offset, int)
        or step_offset < 0
    ):
        raise SolverError("Current solution step must be a non-negative integer.")
    if (
        isinstance(solution_interval, bool)
        or not isinstance(solution_interval, int)
        or solution_interval <= 0
    ):
        raise SolverError("Solution output interval must be a positive integer.")
    if (
        isinstance(settings.convergence_history_interval, bool)
        or not isinstance(settings.convergence_history_interval, int)
        or settings.convergence_history_interval <= 0
    ):
        raise SolverError("Convergence history interval must be a positive integer.")
    if not math.isfinite(cpu_time_offset) or cpu_time_offset < 0.0:
        raise SolverError("CPU time offset must be finite and non-negative.")
    if not isinstance(settings.adaptive_mesh, bool):
        raise SolverError("Adaptive mesh setting must be boolean.")
    if (
        isinstance(settings.adaptation_interval, bool)
        or not isinstance(settings.adaptation_interval, int)
        or settings.adaptation_interval <= 0
    ):
        raise SolverError("Adaptation interval must be a positive integer.")
    if (
        isinstance(settings.adaptation_max_level, bool)
        or not isinstance(settings.adaptation_max_level, int)
        or not 1 <= settings.adaptation_max_level <= 12
    ):
        raise SolverError("Maximum adaptation level must be an integer from 1 to 12.")
    if (
        not math.isfinite(settings.adaptation_refine_threshold)
        or settings.adaptation_refine_threshold <= 0.0
    ):
        raise SolverError("Adaptation threshold must be positive and finite.")
    if settings.adaptation_variable not in (
        "Density",
        "Velocity X",
        "Velocity Y",
        "Pressure",
        "Turbulent k",
        "Specific dissipation rate",
    ):
        raise SolverError("Unsupported adaptive mesh indicator variable.")
    if settings.adaptive_mesh and settings.adaptation_variable in (
        "Turbulent k",
        "Specific dissipation rate",
    ) and settings.turbulence_model != "Menter k-omega SST":
        raise SolverError(
            "Turbulence-based adaptation indicators require the SST model."
        )
    if not math.isfinite(settings.gamma) or settings.gamma <= 1.0:
        raise SolverError("Specific heat ratio must be finite and greater than 1.")
    if not math.isfinite(settings.gas_constant) or settings.gas_constant <= 0.0:
        raise SolverError("Gas constant must be positive and finite.")
    if settings.viscosity_model not in ("Inviscid", "Constant", "Sutherland"):
        raise SolverError("Unsupported viscosity model.")
    if settings.turbulence_model not in ("None", "Menter k-omega SST"):
        raise SolverError("Unsupported turbulence model.")
    if settings.turbulence_wall_treatment not in (
        "Wall-resolved",
        "Wall functions",
    ):
        raise SolverError("Unsupported turbulence wall treatment.")
    if settings.turbulence_omega_formulation not in (
        "Standard SST",
        "Regularized k-log-omega",
    ):
        raise SolverError("Unsupported SST omega formulation.")
    if (
        settings.turbulence_model == "Menter k-omega SST"
        and settings.turbulence_omega_formulation
        == "Regularized k-log-omega"
        and settings.turbulence_wall_treatment != "Wall-resolved"
    ):
        raise SolverError(
            "The regularized k-log-omega formulation requires wall-resolved treatment."
        )
    if (
        settings.turbulence_model == "Menter k-omega SST"
        and settings.viscosity_model == "Inviscid"
    ):
        raise SolverError(
            "Menter k-omega SST requires Constant or Sutherland molecular viscosity."
        )
    for value, description in (
        (settings.dynamic_viscosity, "Dynamic viscosity"),
        (settings.sutherland_reference_viscosity, "Sutherland reference viscosity"),
        (
            settings.sutherland_reference_temperature,
            "Sutherland reference temperature",
        ),
        (settings.sutherland_temperature, "Sutherland temperature"),
        (settings.prandtl_number, "Prandtl number"),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise SolverError(f"{description} must be positive and finite.")
    if not math.isfinite(settings.acceleration_x) or not math.isfinite(
        settings.acceleration_y
    ):
        raise SolverError("Body-force acceleration must be finite.")

    mesh = case.mesh
    geometry = _build_geometry(mesh, case)
    if initial_states is None:
        try:
            initial_states = evaluate_initial_conditions(mesh, case.initial_conditions)
        except ValueError as error:
            raise SolverError(f"Could not evaluate initial conditions: {error}") from error
    if len(initial_states) != len(mesh.cells):
        raise SolverError("Initial state count must match the mesh cell count.")
    primitive = np.array(
        [
            (state.density, state.velocity_x, state.velocity_y, state.pressure)
            for state in initial_states
        ],
        dtype=float,
    )
    try:
        conservative = primitive_to_conservative(primitive, settings.gamma)
        turbulence_conservative = None
        regularized_omega = None
        if settings.turbulence_model == "Menter k-omega SST":
            turbulence_primitive = np.asarray(
                [
                    (state.turbulence_k, state.turbulence_omega)
                    for state in initial_states
                ],
                dtype=float,
            )
            if (
                not np.isfinite(turbulence_primitive).all()
                or np.any(turbulence_primitive[:, 0] < 0.0)
                or np.any(turbulence_primitive[:, 1] <= 0.0)
            ):
                raise SolverError(
                    "Initial turbulence k must be non-negative and omega positive."
                )
            turbulence_conservative = primitive[:, :1] * turbulence_primitive
            conservative[:, 3] += turbulence_conservative[:, 0]
            if settings.turbulence_omega_formulation == "Regularized k-log-omega":
                regularized_omega = _sst_regularized_variable(
                    primitive, turbulence_primitive[:, 1], geometry, settings
                )
                turbulence_conservative = _sst_regularized_turbulence(
                    primitive,
                    turbulence_conservative[:, 0],
                    regularized_omega,
                    geometry,
                    settings,
                )
        _primitive_from_conservative(
            conservative, turbulence_conservative, settings.gamma
        )
    except ValueError as error:
        raise SolverError(f"Invalid initial state: {error}") from error

    time = start_time
    completed = False
    steps = step_offset
    adaptation_disabled = False
    for local_step in range(1, settings.max_iterations + 1):
        step = step_offset + local_step
        if pause_callback is not None:
            pause_callback()
        if cancel_callback is not None and cancel_callback():
            break
        try:
            primitive = _primitive_from_conservative(
                conservative, turbulence_conservative, settings.gamma
            )
            if local_steady:
                step_sizes = _stable_time_steps(
                    primitive, geometry, settings, turbulence_conservative
                )
                time_step = float(np.mean(step_sizes))
            elif settings.time_step_mode == "Global fixed":
                step_sizes = settings.fixed_time_step
                time_step = min(step_sizes, settings.end_time - time)
                step_sizes = time_step
            else:
                time_step = min(
                    _stable_time_step(
                        primitive, geometry, settings, turbulence_conservative
                    ),
                    settings.end_time - time,
                )
                step_sizes = time_step
            if not math.isfinite(time_step) or time_step <= 0.0:
                raise SolverError("The stable time step is non-positive or non-finite.")
            start = conservative
            start_turbulence = turbulence_conservative
            derivative = _residual(
                start, geometry, settings, start_turbulence
            )
            turbulence_derivative = None
            omega_variable_derivative = None
            if start_turbulence is not None:
                turbulence_derivative = _sst_residual(
                    primitive,
                    start_turbulence / primitive[:, :1],
                    geometry,
                    settings,
                )
                if regularized_omega is not None:
                    omega_variable_derivative = _sst_regularized_rate(
                        primitive,
                        derivative,
                        start_turbulence,
                        turbulence_derivative,
                        settings,
                    )
            if settings.time_integration == "Implicit Euler":
                packed_state = _pack_time_state(
                    start, start_turbulence, regularized_omega
                )
                updated_state, time_step = _implicit_euler_step(
                    packed_state,
                    step_sizes,
                    geometry,
                    settings,
                    allow_time_step_reduction=(
                        settings.time_step_mode != "Global fixed"
                    ),
                )
                step_sizes = time_step
                time_step = float(np.mean(np.asarray(step_sizes)))
                (
                    conservative,
                    turbulence_conservative,
                    regularized_omega,
                ) = _unpack_time_state(
                    updated_state,
                    geometry,
                    settings,
                    start_turbulence is not None,
                    regularized_omega is not None,
                )
            elif settings.time_integration == "Explicit RK4":
                packed_state = _pack_time_state(
                    start, start_turbulence, regularized_omega
                )
                updated_state = _explicit_rk4_step(
                    packed_state, step_sizes, geometry, settings
                )
                (
                    conservative,
                    turbulence_conservative,
                    regularized_omega,
                ) = _unpack_time_state(
                    updated_state,
                    geometry,
                    settings,
                    start_turbulence is not None,
                    regularized_omega is not None,
                )
            elif settings.time_integration == "Explicit Euler":
                conservative = start + _scale_cell_rate(derivative, step_sizes)
                if start_turbulence is not None:
                    updated_k = np.maximum(
                        start_turbulence[:, 0]
                        + _scale_cell_rate(
                            turbulence_derivative[:, 0], step_sizes
                        ),
                        0.0,
                    )
                    if regularized_omega is not None:
                        regularized_omega = (
                            regularized_omega
                            + _scale_cell_rate(
                                omega_variable_derivative, step_sizes
                            )
                        )
                        updated_primitive = _primitive_from_conservative(
                            conservative,
                            np.column_stack((updated_k, start_turbulence[:, 1])),
                            settings.gamma,
                        )
                        turbulence_conservative = _sst_regularized_turbulence(
                            updated_primitive,
                            updated_k,
                            regularized_omega,
                            geometry,
                            settings,
                        )
                    else:
                        turbulence_conservative = _positive_turbulence_state(
                            start_turbulence
                            + _scale_cell_rate(turbulence_derivative, step_sizes)
                        )
            else:
                stage_one = start + _scale_cell_rate(derivative, step_sizes)
                stage_one_turbulence = None
                stage_one_omega_variable = None
                if start_turbulence is not None:
                    stage_one_k = np.maximum(
                        start_turbulence[:, 0]
                        + _scale_cell_rate(
                            turbulence_derivative[:, 0], step_sizes
                        ),
                        0.0,
                    )
                    if regularized_omega is not None:
                        stage_one_omega_variable = (
                            regularized_omega
                            + _scale_cell_rate(
                                omega_variable_derivative, step_sizes
                            )
                        )
                        stage_one_primitive = _primitive_from_conservative(
                            stage_one,
                            np.column_stack(
                                (stage_one_k, start_turbulence[:, 1])
                            ),
                            settings.gamma,
                        )
                        stage_one_turbulence = _sst_regularized_turbulence(
                            stage_one_primitive,
                            stage_one_k,
                            stage_one_omega_variable,
                            geometry,
                            settings,
                        )
                    else:
                        stage_one_turbulence = _positive_turbulence_state(
                            start_turbulence
                            + _scale_cell_rate(turbulence_derivative, step_sizes)
                        )
                stage_one_primitive = _primitive_from_conservative(
                    stage_one, stage_one_turbulence, settings.gamma
                )
                derivative_one = _residual(
                    stage_one, geometry, settings, stage_one_turbulence
                )
                if stage_one_turbulence is not None:
                    turbulence_derivative_one = _sst_residual(
                        stage_one_primitive,
                        stage_one_turbulence / stage_one_primitive[:, :1],
                        geometry,
                        settings,
                    )
                    if stage_one_omega_variable is not None:
                        omega_variable_derivative_one = _sst_regularized_rate(
                            stage_one_primitive,
                            derivative_one,
                            stage_one_turbulence,
                            turbulence_derivative_one,
                            settings,
                        )
                if settings.time_integration == "Explicit RK2":
                    conservative = 0.5 * start + 0.5 * (
                        stage_one + _scale_cell_rate(derivative_one, step_sizes)
                    )
                    if start_turbulence is not None:
                        updated_k = np.maximum(
                            0.5 * start_turbulence[:, 0]
                            + 0.5
                            * (
                                stage_one_turbulence[:, 0]
                                + _scale_cell_rate(
                                    turbulence_derivative_one[:, 0], step_sizes
                                )
                            ),
                            0.0,
                        )
                        if regularized_omega is not None:
                            regularized_omega = 0.5 * regularized_omega + 0.5 * (
                                stage_one_omega_variable
                                + _scale_cell_rate(
                                    omega_variable_derivative_one, step_sizes
                                )
                            )
                            updated_primitive = _primitive_from_conservative(
                                conservative,
                                np.column_stack(
                                    (updated_k, stage_one_turbulence[:, 1])
                                ),
                                settings.gamma,
                            )
                            turbulence_conservative = (
                                _sst_regularized_turbulence(
                                    updated_primitive,
                                    updated_k,
                                    regularized_omega,
                                    geometry,
                                    settings,
                                )
                            )
                        else:
                            turbulence_conservative = _positive_turbulence_state(
                                0.5 * start_turbulence
                                + 0.5
                                * (
                                    stage_one_turbulence
                                    + _scale_cell_rate(
                                        turbulence_derivative_one, step_sizes
                                    )
                                )
                            )
                else:
                    stage_two = 0.75 * start + 0.25 * (
                        stage_one + _scale_cell_rate(derivative_one, step_sizes)
                    )
                    stage_two_turbulence = None
                    stage_two_omega_variable = None
                    if start_turbulence is not None:
                        stage_two_k = np.maximum(
                            0.75 * start_turbulence[:, 0]
                            + 0.25
                            * (
                                stage_one_turbulence[:, 0]
                                + _scale_cell_rate(
                                    turbulence_derivative_one[:, 0], step_sizes
                                )
                            ),
                            0.0,
                        )
                        if regularized_omega is not None:
                            stage_two_omega_variable = (
                                0.75 * regularized_omega
                                + 0.25
                                * (
                                    stage_one_omega_variable
                                    + _scale_cell_rate(
                                        omega_variable_derivative_one, step_sizes
                                    )
                                )
                            )
                            stage_two_primitive = _primitive_from_conservative(
                                stage_two,
                                np.column_stack(
                                    (stage_two_k, stage_one_turbulence[:, 1])
                                ),
                                settings.gamma,
                            )
                            stage_two_turbulence = (
                                _sst_regularized_turbulence(
                                    stage_two_primitive,
                                    stage_two_k,
                                    stage_two_omega_variable,
                                    geometry,
                                    settings,
                                )
                            )
                        else:
                            stage_two_turbulence = _positive_turbulence_state(
                                0.75 * start_turbulence
                                + 0.25
                                * (
                                    stage_one_turbulence
                                    + _scale_cell_rate(
                                        turbulence_derivative_one, step_sizes
                                    )
                                )
                            )
                    stage_two_primitive = _primitive_from_conservative(
                        stage_two, stage_two_turbulence, settings.gamma
                    )
                    derivative_two = _residual(
                        stage_two, geometry, settings, stage_two_turbulence
                    )
                    if stage_two_turbulence is not None:
                        turbulence_derivative_two = _sst_residual(
                            stage_two_primitive,
                            stage_two_turbulence / stage_two_primitive[:, :1],
                            geometry,
                            settings,
                        )
                        if stage_two_omega_variable is not None:
                            omega_variable_derivative_two = (
                                _sst_regularized_rate(
                                    stage_two_primitive,
                                    derivative_two,
                                    stage_two_turbulence,
                                    turbulence_derivative_two,
                                    settings,
                                )
                            )
                    conservative = (1.0 / 3.0) * start + (2.0 / 3.0) * (
                        stage_two + _scale_cell_rate(derivative_two, step_sizes)
                    )
                    if start_turbulence is not None:
                        updated_k = np.maximum(
                            (1.0 / 3.0) * start_turbulence[:, 0]
                            + (2.0 / 3.0)
                            * (
                                stage_two_turbulence[:, 0]
                                + _scale_cell_rate(
                                    turbulence_derivative_two[:, 0], step_sizes
                                )
                            ),
                            0.0,
                        )
                        if regularized_omega is not None:
                            regularized_omega = (
                                (1.0 / 3.0) * regularized_omega
                                + (2.0 / 3.0)
                                * (
                                    stage_two_omega_variable
                                    + _scale_cell_rate(
                                        omega_variable_derivative_two, step_sizes
                                    )
                                )
                            )
                            updated_primitive = _primitive_from_conservative(
                                conservative,
                                np.column_stack(
                                    (updated_k, stage_two_turbulence[:, 1])
                                ),
                                settings.gamma,
                            )
                            turbulence_conservative = (
                                _sst_regularized_turbulence(
                                    updated_primitive,
                                    updated_k,
                                    regularized_omega,
                                    geometry,
                                    settings,
                                )
                            )
                        else:
                            turbulence_conservative = _positive_turbulence_state(
                                (1.0 / 3.0) * start_turbulence
                                + (2.0 / 3.0)
                                * (
                                    stage_two_turbulence
                                    + _scale_cell_rate(
                                        turbulence_derivative_two, step_sizes
                                    )
                                )
                            )
            _primitive_from_conservative(
                conservative, turbulence_conservative, settings.gamma
            )
        except (FloatingPointError, ValueError, ZeroDivisionError) as error:
            raise SolverError(f"Solver failed at step {step}: {error}") from error
        time += time_step
        steps = step
        adapted_mesh = None
        adaptation_warning = None
        if (
            settings.adaptive_mesh
            and not adaptation_disabled
            and step % settings.adaptation_interval == 0
        ):
            current_primitive = _primitive_from_conservative(
                conservative, turbulence_conservative, settings.gamma
            )
            indicator_values = _adaptation_indicator_values(
                current_primitive,
                turbulence_conservative,
                settings.adaptation_variable,
            )
            try:
                refine_marks, coarsen_parent_ids = select_adaptation(
                    mesh,
                    case,
                    geometry.centers,
                    geometry.areas,
                    indicator_values,
                    settings.adaptation_refine_threshold,
                    settings.adaptation_max_level,
                )
                if refine_marks or coarsen_parent_ids:
                    old_areas = np.asarray(cell_areas(mesh), dtype=float)
                    adapted, source_groups = adapt_mesh(
                        mesh,
                        refine_marks,
                        coarsen_parent_ids,
                        periodic_pairs=case.periodic_pairs,
                    )
                    if adapted is not mesh:
                        new_areas = np.asarray(cell_areas(adapted), dtype=float)
                        conservative = _transfer_cell_values(
                            conservative,
                            old_areas,
                            new_areas,
                            source_groups,
                        )
                        if turbulence_conservative is not None:
                            turbulence_conservative = _transfer_cell_values(
                                turbulence_conservative,
                                old_areas,
                                new_areas,
                                source_groups,
                            )
                        mesh = adapted
                        case.mesh = adapted
                        geometry = _build_geometry(mesh, case)
                        if turbulence_conservative is not None:
                            current_primitive = _primitive_from_conservative(
                                conservative,
                                turbulence_conservative,
                                settings.gamma,
                            )
                            if regularized_omega is not None:
                                regularized_omega = _sst_regularized_variable(
                                    current_primitive,
                                    turbulence_conservative[:, 1]
                                    / current_primitive[:, :1].ravel(),
                                    geometry,
                                    settings,
                                )
                        adapted_mesh = mesh
            except AdaptationBalanceError as error:
                adaptation_disabled = True
                adaptation_warning = f"Adaptive mesh disabled at step {step}: {error}"
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
                raise SolverError(f"Adaptive mesh update failed at step {step}: {error}") from error
        steady_converged = False
        if local_steady:
            current_primitive = _primitive_from_conservative(
                conservative, turbulence_conservative, settings.gamma
            )
            derivative = _residual(
                conservative, geometry, settings, turbulence_conservative
            )
            residual_components = [derivative.ravel()]
            if turbulence_conservative is not None:
                turbulence_derivative = _sst_residual(
                    current_primitive,
                    turbulence_conservative / current_primitive[:, :1],
                    geometry,
                    settings,
                )
                residual_components.append(turbulence_derivative.ravel())
            steady_residual = float(
                np.sqrt(np.mean(np.concatenate(residual_components) ** 2))
            )
            steady_converged = steady_residual <= settings.residual_tolerance
        end_time_reached = (
            not local_steady
            and time >= settings.end_time * (1.0 - 1e-13)
        )
        final_step = (
            end_time_reached
            or steady_converged
            or local_step == settings.max_iterations
            or (
                cancel_callback is not None and cancel_callback()
            )
        )
        report_residual = (
            progress_callback is not None
            and (
                step % settings.convergence_history_interval == 0
                or adapted_mesh is not None
                or final_step
            )
        )
        if progress_callback is not None:
            progress_states = None
            if (
                live_solution_updates
                and (
                    local_step == 1
                    or step % 10 == 0
                    or end_time_reached
                    or steady_converged
                    or local_step == settings.max_iterations
                )
            ) or adapted_mesh is not None:
                progress_primitive = _primitive_from_conservative(
                    conservative, turbulence_conservative, settings.gamma
                )
                progress_states = _states_from_solution(
                    progress_primitive, turbulence_conservative
                )
            if report_residual or progress_states is not None or adaptation_warning:
                component_residuals = ()
                if report_residual:
                    component_residuals = tuple(
                        float(value)
                        for value in np.sqrt(np.mean(derivative**2, axis=0))
                    )
                    if turbulence_derivative is not None:
                        component_residuals += tuple(
                            float(value)
                            for value in np.sqrt(
                                np.mean(turbulence_derivative**2, axis=0)
                            )
                        )
                progress_callback(
                    SolverProgress(
                        step,
                        time,
                        time_step,
                        component_residuals,
                        progress_states,
                        adapted_mesh,
                        cpu_time_offset + thread_time() - cpu_start,
                        adaptation_warning,
                    )
                )
            if adaptation_warning and progress_callback is None:
                warnings.warn(adaptation_warning, RuntimeWarning, stacklevel=2)
        if solution_callback is not None and step % solution_interval == 0:
            snapshot_states = progress_states if progress_callback is not None else None
            if snapshot_states is None:
                snapshot_primitive = _primitive_from_conservative(
                    conservative, turbulence_conservative, settings.gamma
                )
                snapshot_states = _states_from_solution(
                    snapshot_primitive, turbulence_conservative
                )
            solution_callback(step, time, snapshot_states)
        if end_time_reached:
            time = settings.end_time
            completed = True
            break
        if steady_converged:
            completed = True
            break

    final_primitive = _primitive_from_conservative(
        conservative, turbulence_conservative, settings.gamma
    )
    try:
        final_residual = _residual(
            conservative, geometry, settings, turbulence_conservative
        )
    except ValueError as error:
        raise SolverError(f"Could not evaluate final residual: {error}") from error
    residual = float(np.sqrt(np.mean(final_residual**2)))
    states = _states_from_solution(final_primitive, turbulence_conservative)
    return SolverResult(
        states,
        time,
        steps,
        residual,
        completed,
        cpu_time_offset + thread_time() - cpu_start,
    )


def _pack_time_state(
    conservative: np.ndarray,
    turbulence_conservative: np.ndarray | None,
    regularized_omega: np.ndarray | None,
) -> np.ndarray:
    if turbulence_conservative is None:
        return conservative.copy()
    if regularized_omega is not None:
        return np.column_stack(
            (conservative, turbulence_conservative[:, 0], regularized_omega)
        )
    return np.column_stack((conservative, turbulence_conservative))


def _unpack_time_state(
    state: np.ndarray,
    geometry: _Geometry,
    settings: NumericalSettings,
    has_turbulence: bool,
    is_regularized: bool,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    conservative = state[:, :4].copy()
    if not has_turbulence:
        return conservative, None, None
    if is_regularized:
        k_conservative = np.maximum(state[:, 4], 0.0)
        regularized_omega = state[:, 5].copy()
        primitive = _primitive_from_conservative(
            conservative,
            np.column_stack((k_conservative, np.ones_like(k_conservative))),
            settings.gamma,
        )
        turbulence = _sst_regularized_turbulence(
            primitive, k_conservative, regularized_omega, geometry, settings
        )
        return conservative, turbulence, regularized_omega
    turbulence = _positive_turbulence_state(state[:, 4:6])
    return conservative, turbulence, None


def _time_state_rhs(
    state: np.ndarray,
    geometry: _Geometry,
    settings: NumericalSettings,
) -> np.ndarray:
    conservative = state[:, :4]
    has_turbulence = state.shape[1] > 4
    is_regularized = (
        has_turbulence
        and settings.turbulence_omega_formulation == "Regularized k-log-omega"
    )
    turbulence_conservative = None
    regularized_omega = None
    if has_turbulence:
        if is_regularized:
            k_conservative = np.maximum(state[:, 4], 0.0)
            regularized_omega = state[:, 5]
            primitive = _primitive_from_conservative(
                conservative,
                np.column_stack((k_conservative, np.ones_like(k_conservative))),
                settings.gamma,
            )
            turbulence_conservative = _sst_regularized_turbulence(
                primitive,
                k_conservative,
                regularized_omega,
                geometry,
                settings,
            )
        else:
            turbulence_conservative = _positive_turbulence_state(state[:, 4:6])
    primitive = _primitive_from_conservative(
        conservative, turbulence_conservative, settings.gamma
    )
    flow_rate = _residual(
        conservative, geometry, settings, turbulence_conservative
    )
    if turbulence_conservative is None:
        return flow_rate
    turbulence_rate = _sst_residual(
        primitive,
        turbulence_conservative / primitive[:, :1],
        geometry,
        settings,
    )
    if is_regularized:
        omega_rate = _sst_regularized_rate(
            primitive,
            flow_rate,
            turbulence_conservative,
            turbulence_rate,
            settings,
        )
        return np.column_stack((flow_rate, turbulence_rate[:, 0], omega_rate))
    return np.column_stack((flow_rate, turbulence_rate))


def _limit_time_state(
    state: np.ndarray, settings: NumericalSettings
) -> np.ndarray:
    result = state.copy()
    if result.shape[1] > 4:
        result[:, 4] = np.maximum(result[:, 4], 0.0)
        if settings.turbulence_omega_formulation != "Regularized k-log-omega":
            result[:, 5] = np.maximum(result[:, 5], 1e-12)
    return result


def _scale_cell_rate(rate: np.ndarray, time_step: float | np.ndarray) -> np.ndarray:
    if np.ndim(time_step) == 0:
        return time_step * rate
    step_sizes = np.asarray(time_step, dtype=float)
    if rate.ndim == 1:
        return step_sizes * rate
    return step_sizes[:, None] * rate


def _explicit_rk4_step(
    state: np.ndarray,
    time_step: float | np.ndarray,
    geometry: _Geometry,
    settings: NumericalSettings,
) -> np.ndarray:
    initial = _limit_time_state(state, settings)
    derivative_one = _time_state_rhs(initial, geometry, settings)
    stage_two = _limit_time_state(
        initial + _scale_cell_rate(derivative_one, 0.5 * time_step), settings
    )
    derivative_two = _time_state_rhs(stage_two, geometry, settings)
    stage_three = _limit_time_state(
        initial + _scale_cell_rate(derivative_two, 0.5 * time_step), settings
    )
    derivative_three = _time_state_rhs(stage_three, geometry, settings)
    stage_four = _limit_time_state(
        initial + _scale_cell_rate(derivative_three, time_step), settings
    )
    derivative_four = _time_state_rhs(stage_four, geometry, settings)
    return _limit_time_state(
        initial
        + _scale_cell_rate(
            derivative_one
            + 2.0 * derivative_two
            + 2.0 * derivative_three
            + derivative_four,
            time_step / 6.0,
        ),
        settings,
    )


def _implicit_preconditioner(
    primitive: np.ndarray,
    geometry: _Geometry,
    settings: NumericalSettings,
    time_step: float | np.ndarray,
    variable_count: int,
) -> LinearOperator:
    cell_count = len(primitive)
    diagonal = np.ones(cell_count)
    rows = list(range(cell_count))
    columns = list(range(cell_count))
    entries = [1.0] * cell_count
    sound_speed = _sound_speed(primitive, settings.gamma)
    step_sizes = (
        np.full(cell_count, time_step, dtype=float)
        if np.ndim(time_step) == 0
        else np.asarray(time_step, dtype=float)
    )

    def add_interior_faces(faces: np.ndarray) -> None:
        if not len(faces):
            return
        owners = geometry.owners[faces]
        neighbours = geometry.neighbours[faces]
        normal_velocity_owner = np.sum(
            primitive[owners, 1:3] * geometry.face_normals[faces], axis=1
        )
        normal_velocity_neighbour = np.sum(
            primitive[neighbours, 1:3] * geometry.face_normals[faces], axis=1
        )
        radius = np.maximum(
            np.abs(normal_velocity_owner) + sound_speed[owners],
            np.abs(normal_velocity_neighbour) + sound_speed[neighbours],
        )
        coefficients = 0.5 * radius * geometry.face_lengths[faces]
        owner_coefficients = step_sizes[owners] * coefficients
        neighbour_coefficients = step_sizes[neighbours] * coefficients
        np.add.at(
            diagonal,
            owners,
            owner_coefficients / geometry.areas[owners],
        )
        np.add.at(
            diagonal,
            neighbours,
            neighbour_coefficients / geometry.areas[neighbours],
        )
        rows.extend(owners.tolist())
        columns.extend(neighbours.tolist())
        entries.extend(
            (-owner_coefficients / geometry.areas[owners]).tolist()
        )
        rows.extend(neighbours.tolist())
        columns.extend(owners.tolist())
        entries.extend(
            (-neighbour_coefficients / geometry.areas[neighbours]).tolist()
        )

    add_interior_faces(geometry.interior_faces)
    if len(geometry.periodic_faces_a):
        faces_a = geometry.periodic_faces_a
        faces_b = geometry.periodic_faces_b
        owners_a = geometry.owners[faces_a]
        owners_b = geometry.owners[faces_b]
        normal_velocity_a = np.sum(
            primitive[owners_a, 1:3] * geometry.face_normals[faces_a], axis=1
        )
        normal_velocity_b = np.sum(
            primitive[owners_b, 1:3] * geometry.face_normals[faces_b], axis=1
        )
        radius = np.maximum(
            np.abs(normal_velocity_a) + sound_speed[owners_a],
            np.abs(normal_velocity_b) + sound_speed[owners_b],
        )
        coefficients = 0.5 * radius * geometry.face_lengths[faces_a]
        coefficients_a = step_sizes[owners_a] * coefficients
        coefficients_b = step_sizes[owners_b] * coefficients
        np.add.at(
            diagonal, owners_a, coefficients_a / geometry.areas[owners_a]
        )
        np.add.at(
            diagonal, owners_b, coefficients_b / geometry.areas[owners_b]
        )
        rows.extend(owners_a.tolist())
        columns.extend(owners_b.tolist())
        entries.extend(
            (-coefficients_a / geometry.areas[owners_a]).tolist()
        )
        rows.extend(owners_b.tolist())
        columns.extend(owners_a.tolist())
        entries.extend(
            (-coefficients_b / geometry.areas[owners_b]).tolist()
        )

    boundary_faces = np.flatnonzero(geometry.neighbours < 0)
    if len(boundary_faces):
        owners = geometry.owners[boundary_faces]
        radius = (
            np.abs(
                np.sum(
                    primitive[owners, 1:3]
                    * geometry.face_normals[boundary_faces],
                    axis=1,
                )
            )
            + sound_speed[owners]
        )
        np.add.at(
            diagonal,
            owners,
            step_sizes[owners]
            * radius
            * geometry.face_lengths[boundary_faces]
            / geometry.areas[owners],
        )
    for cell_id, value in enumerate(diagonal):
        entries[cell_id] = value
    scalar_matrix = coo_matrix(
        (entries, (rows, columns)), shape=(cell_count, cell_count)
    ).tocsc()
    block_matrix = kron(
        scalar_matrix, eye(variable_count, format="csc"), format="csc"
    )
    dimension = cell_count * variable_count
    if settings.implicit_preconditioner == "LU-SGS":
        diagonal_blocks = np.repeat(scalar_matrix.diagonal(), variable_count)
        lower = tril(block_matrix, format="csr")
        upper = triu(block_matrix, format="csr")
        diagonal_blocks = np.maximum(diagonal_blocks, 1e-14)

        def apply_lu_sgs(vector: np.ndarray) -> np.ndarray:
            forward = spsolve_triangular(lower, vector, lower=True)
            return spsolve_triangular(
                upper, diagonal_blocks * forward, lower=False
            )

        return LinearOperator(
            (dimension, dimension), matvec=apply_lu_sgs, dtype=float
        )
    try:
        factor = spilu(block_matrix, drop_tol=1e-3, fill_factor=10.0)
    except RuntimeError as error:
        raise SolverError(
            f"Could not factor the first-order ILU preconditioner: {error}"
        ) from error
    return LinearOperator(
        (dimension, dimension),
        matvec=factor.solve,
        dtype=float,
    )


def _implicit_euler_step(
    initial: np.ndarray,
    time_step: float | np.ndarray,
    geometry: _Geometry,
    settings: NumericalSettings,
    allow_time_step_reduction: bool = True,
) -> tuple[np.ndarray, float | np.ndarray]:
    attempted_time_step = (
        float(time_step)
        if np.ndim(time_step) == 0
        else np.asarray(time_step, dtype=float).copy()
    )
    last_error: SolverError | None = None
    attempts = 12 if allow_time_step_reduction else 1
    for _ in range(attempts):
        try:
            solution = _implicit_euler_solve_once(
                initial, attempted_time_step, geometry, settings
            )
            return solution, attempted_time_step
        except SolverError as error:
            last_error = error
            attempted_time_step *= 0.5
    if not allow_time_step_reduction:
        raise SolverError(
            "Implicit Euler could not converge at the configured fixed time step: "
            f"{last_error}"
        ) from last_error
    raise SolverError(
        "Implicit Euler could not converge after reducing the time step "
        f"12 times; last failure: {last_error}"
    ) from last_error


def _implicit_euler_solve_once(
    initial: np.ndarray,
    time_step: float | np.ndarray,
    geometry: _Geometry,
    settings: NumericalSettings,
) -> np.ndarray:
    state = _limit_time_state(initial, settings)
    jacobian_settings = replace(settings, reconstruction="First order")
    is_regularized = (
        state.shape[1] == 6
        and settings.turbulence_omega_formulation == "Regularized k-log-omega"
    )
    flow_state, turbulence_state, _ = _unpack_time_state(
        state,
        geometry,
        settings,
        state.shape[1] > 4,
        is_regularized,
    )
    primitive = _primitive_from_conservative(
        flow_state, turbulence_state, settings.gamma
    )
    rho_scale = max(float(np.max(primitive[:, 0])), 1e-12)
    speed_scale = max(
        float(np.max(np.abs(primitive[:, 1:3]))),
        float(np.max(_sound_speed(primitive, settings.gamma))),
        1e-12,
    )
    scales = np.maximum(np.max(np.abs(state), axis=0), 1e-12)
    scales[0] = max(scales[0], rho_scale)
    scales[1:3] = np.maximum(scales[1:3], rho_scale * speed_scale)
    scales[3] = max(
        scales[3],
        rho_scale * speed_scale**2,
        float(np.max(primitive[:, 3])) / (settings.gamma - 1.0),
    )
    if state.shape[1] > 4:
        scales[4] = max(scales[4], rho_scale * speed_scale**2 * 1e-8)
        if is_regularized:
            scales[5] = max(scales[5], 1.0)
        else:
            scales[5] = max(scales[5], rho_scale * 1e-8)
    scale_grid = scales[None, :]

    def evaluate(candidate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rate = _time_state_rhs(candidate, geometry, settings)
        equation = candidate - initial - _scale_cell_rate(rate, time_step)
        return rate, equation

    for _ in range(12):
        rate, equation = evaluate(state)
        scaled_equation = equation / scale_grid
        equation_norm = float(np.max(np.abs(scaled_equation)))
        update_norm = float(np.max(np.abs((state - initial) / scale_grid)))
        if equation_norm <= 1e-9 * max(1.0, update_norm):
            return _limit_time_state(state, settings)

        flow_state, turbulence_state, _ = _unpack_time_state(
            state,
            geometry,
            settings,
            state.shape[1] > 4,
            is_regularized,
        )
        primitive = _primitive_from_conservative(
            flow_state, turbulence_state, settings.gamma
        )
        preconditioner = _implicit_preconditioner(
            primitive,
            geometry,
            jacobian_settings,
            time_step,
            state.shape[1],
        )
        jacobian_rate = _time_state_rhs(state, geometry, jacobian_settings)
        operator = _matrix_free_implicit_operator(
            state,
            jacobian_rate,
            scale_grid,
            time_step,
            geometry,
            jacobian_settings,
        )
        correction = _solve_gmres(
            operator,
            preconditioner,
            -scaled_equation.ravel(),
            settings.gmres_tolerance,
            settings.gmres_max_iterations,
            settings.gmres_krylov_dimension,
        )

        accepted = False
        damping = 1.0
        for _ in range(8):
            candidate = _limit_time_state(
                state + damping * correction.reshape(state.shape) * scale_grid,
                settings,
            )
            try:
                _, candidate_equation = evaluate(candidate)
                candidate_norm = float(
                    np.max(np.abs(candidate_equation / scale_grid))
                )
            except (SolverError, ValueError, FloatingPointError):
                candidate_norm = math.inf
            if candidate_norm < equation_norm:
                state = candidate
                accepted = True
                break
            damping *= 0.5
        if not accepted:
            raise SolverError(
                "Implicit Euler Newton iteration could not reduce its residual."
            )
    raise SolverError("Implicit Euler Newton iteration did not converge.")


def _solve_gmres(
    operator: LinearOperator,
    preconditioner: LinearOperator,
    right_hand_side: np.ndarray,
    tolerance: float,
    maximum_iterations: int,
    krylov_dimension: int,
) -> np.ndarray:
    solution = np.zeros_like(right_hand_side)
    iterations_used = 0
    while iterations_used < maximum_iterations:
        cycle_iterations = 0

        def count_iteration(_residual: float) -> None:
            nonlocal cycle_iterations
            cycle_iterations += 1

        restart = min(krylov_dimension, maximum_iterations - iterations_used)
        solution, info = gmres(
            operator,
            right_hand_side,
            x0=solution,
            M=preconditioner,
            restart=restart,
            maxiter=1,
            rtol=tolerance,
            atol=0.0,
            callback=count_iteration,
            callback_type="pr_norm",
        )
        iterations_used += cycle_iterations
        if info == 0:
            return solution
        if info < 0:
            raise SolverError(f"Implicit Euler GMRES broke down (status {info}).")
        if cycle_iterations == 0:
            break
    raise SolverError(
        "Implicit Euler GMRES did not converge within "
        f"{maximum_iterations} iterations (relative tolerance {tolerance:g})."
    )


def _matrix_free_implicit_operator(
    state: np.ndarray,
    rate: np.ndarray,
    scale_grid: np.ndarray,
    time_step: float | np.ndarray,
    geometry: _Geometry,
    settings: NumericalSettings,
) -> LinearOperator:
    state_shape = state.shape
    state_dimensionless_norm = float(np.linalg.norm(state / scale_grid))
    finite_difference_scale = math.sqrt(np.finfo(float).eps)

    def matrix_vector(vector: np.ndarray) -> np.ndarray:
        direction = vector.reshape(state_shape)
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm == 0.0:
            return vector.copy()
        epsilon = (
            finite_difference_scale
            * max(1.0, state_dimensionless_norm)
            / direction_norm
        )
        perturbed_rate = _time_state_rhs(
            state + epsilon * scale_grid * direction,
            geometry,
            settings,
        )
        jacobian_vector = (perturbed_rate - rate) / epsilon
        return (
            direction
            - _scale_cell_rate(jacobian_vector, time_step) / scale_grid
        ).ravel()

    return LinearOperator(
        (state.size, state.size), matvec=matrix_vector, dtype=float
    )


def _adaptation_indicator_values(
    primitive: np.ndarray,
    turbulence_conservative: np.ndarray | None,
    variable: str,
) -> np.ndarray:
    if variable == "Density":
        return primitive[:, 0]
    if variable == "Velocity X":
        return primitive[:, 1]
    if variable == "Velocity Y":
        return primitive[:, 2]
    if variable == "Pressure":
        return primitive[:, 3]
    if variable == "Turbulent k":
        if turbulence_conservative is None:
            raise SolverError("Turbulent k adaptation requires the SST model.")
        return turbulence_conservative[:, 0] / primitive[:, 0]
    if variable == "Specific dissipation rate":
        if turbulence_conservative is None:
            raise SolverError("Omega adaptation requires the SST model.")
        return turbulence_conservative[:, 1] / primitive[:, 0]
    raise SolverError(f"Unsupported adaptive mesh indicator '{variable}'.")


def _transfer_cell_values(
    values: np.ndarray,
    old_areas: np.ndarray,
    new_areas: np.ndarray,
    source_groups: tuple[tuple[int, ...], ...],
) -> np.ndarray:
    transferred = []
    for source_group in source_groups:
        indices = np.asarray(source_group, dtype=int)
        weights = old_areas[indices]
        transferred.append(
            np.average(values[indices], axis=0, weights=weights)
        )
    transferred_values = np.asarray(transferred, dtype=float)
    outputs_by_sources: dict[tuple[int, ...], list[int]] = {}
    for output_id, source_group in enumerate(source_groups):
        outputs_by_sources.setdefault(source_group, []).append(output_id)
    for source_group, output_ids in outputs_by_sources.items():
        old_area = float(np.sum(old_areas[np.asarray(source_group, dtype=int)]))
        new_area = float(np.sum(new_areas[np.asarray(output_ids, dtype=int)]))
        if not math.isfinite(new_area) or new_area <= 0.0:
            raise ValueError("Adapted cells must have positive finite areas.")
        transferred_values[output_ids] *= old_area / new_area
    return transferred_values


def _primitive_from_conservative(
    conservative: np.ndarray,
    turbulence_conservative: np.ndarray | None,
    gamma: float,
) -> np.ndarray:
    if turbulence_conservative is None:
        return conservative_to_primitive(conservative, gamma)
    thermodynamic_conservative = conservative.copy()
    thermodynamic_conservative[:, 3] -= turbulence_conservative[:, 0]
    return conservative_to_primitive(thermodynamic_conservative, gamma)


def _positive_turbulence_state(conservative: np.ndarray) -> np.ndarray:
    state = conservative.copy()
    state[:, 0] = np.maximum(state[:, 0], 0.0)
    state[:, 1] = np.maximum(state[:, 1], 1e-12)
    if not np.isfinite(state).all():
        raise SolverError("SST update produced non-finite turbulence variables.")
    return state


def _states_from_solution(
    primitive: np.ndarray,
    turbulence_conservative: np.ndarray | None,
) -> tuple[InitialState, ...]:
    if turbulence_conservative is None:
        return tuple(InitialState(*row) for row in primitive.tolist())
    turbulence = turbulence_conservative / primitive[:, :1]
    return tuple(
        InitialState(*row, turbulence_k=float(k), turbulence_omega=float(omega))
        for row, (k, omega) in zip(primitive.tolist(), turbulence.tolist())
    )


def _build_geometry(mesh: FiniteVolumeMesh, case: Case) -> _Geometry:
    cell_centers = []
    cell_areas = []
    for cell_id, node_ids in enumerate(mesh.cells):
        points = [mesh.points[node_id] for node_id in node_ids]
        twice_area = sum(
            x0 * y1 - x1 * y0
            for (x0, y0), (x1, y1) in zip(points, points[1:] + points[:1])
        )
        if not math.isfinite(twice_area) or abs(twice_area) <= 1e-15:
            raise SolverError(f"Cell {cell_id} has zero or invalid area.")
        x_sum = 0.0
        y_sum = 0.0
        for (x0, y0), (x1, y1) in zip(points, points[1:] + points[:1]):
            cross = x0 * y1 - x1 * y0
            x_sum += (x0 + x1) * cross
            y_sum += (y0 + y1) * cross
        cell_centers.append(
            (x_sum / (3.0 * twice_area), y_sum / (3.0 * twice_area))
        )
        cell_areas.append(0.5 * abs(twice_area))
    centers = np.asarray(cell_centers, dtype=float)
    areas = np.asarray(cell_areas, dtype=float)

    face_centers = []
    face_normals = []
    face_lengths = []
    for face_id, (node_a, node_b) in enumerate(mesh.faces):
        a = np.asarray(mesh.points[node_a], dtype=float)
        b = np.asarray(mesh.points[node_b], dtype=float)
        edge = b - a
        length = float(np.linalg.norm(edge))
        if not math.isfinite(length) or length <= 1e-15:
            raise SolverError(f"Face {face_id} has zero or invalid length.")
        normal = np.array((edge[1], -edge[0]), dtype=float) / length
        center = 0.5 * (a + b)
        owner = mesh.face_owners[face_id]
        if float(np.dot(normal, center - centers[owner])) < 0.0:
            normal = -normal
        face_centers.append(center)
        face_normals.append(normal)
        face_lengths.append(length)

    owners = np.asarray(mesh.face_owners, dtype=int)
    neighbours = np.asarray(mesh.face_neighbours, dtype=int)
    patch_ids = np.asarray(mesh.face_patch_ids, dtype=int)
    interior_faces = np.flatnonzero(neighbours >= 0)
    boundary_faces = np.flatnonzero(neighbours < 0)
    conditions = {condition.patch_id: condition for condition in case.boundary_conditions}
    pairs_by_patch: dict[int, PeriodicBoundaryPair] = {}
    periodic_mappings = []
    for pair in case.periodic_pairs:
        try:
            axis, pitch = detect_periodic_pair(mesh, pair.patch_a_id, pair.patch_b_id)
            matched_faces, shift = match_periodic_faces(
                mesh, pair.patch_a_id, pair.patch_b_id
            )
        except ValueError as error:
            raise SolverError(f"Invalid periodic boundary pair: {error}") from error
        if pair.axis != axis or not math.isclose(
            pair.pitch, pitch, rel_tol=1e-9, abs_tol=max(pitch, 1.0) * 1e-9
        ):
            raise SolverError("A periodic pair has a stale axis or pitch.")
        if (
            pair.patch_a_id not in conditions
            or pair.patch_b_id not in conditions
            or conditions[pair.patch_a_id].kind != "Periodic"
            or conditions[pair.patch_b_id].kind != "Periodic"
        ):
            raise SolverError("Both patches in every periodic pair must be Periodic.")
        if pair.patch_a_id in pairs_by_patch or pair.patch_b_id in pairs_by_patch:
            raise SolverError("A patch cannot belong to multiple periodic pairs.")
        pairs_by_patch[pair.patch_a_id] = pair
        pairs_by_patch[pair.patch_b_id] = pair
        periodic_mappings.extend((face_a, face_b, shift) for face_a, face_b in matched_faces)

    periodic_patch_ids = {
        patch_id for patch_id, condition in conditions.items()
        if condition.kind == "Periodic"
    }
    if periodic_patch_ids != set(pairs_by_patch):
        raise SolverError("Each Periodic boundary must belong to exactly one pair.")
    if set(conditions) != {patch.id for patch in mesh.patches}:
        raise SolverError("Boundary conditions must be defined for every mesh patch.")
    for patch_id, condition in conditions.items():
        if condition.kind not in (
            "Periodic",
            "Wall",
            "Symmetry",
            "Velocity Inlet",
            "Pressure Inlet",
            "Pressure Outlet",
            "Mass Flow Inlet",
            "Mass Flow Outlet",
            "Far Field",
        ):
            raise SolverError(
                f"Boundary type '{condition.kind}' is not implemented by the solver."
            )
        if condition.kind == "Periodic" and patch_id not in pairs_by_patch:
            raise SolverError(f"Periodic patch '{condition.patch_name}' has no pair.")
    periodic_face_set = {
        face_id for face_a, face_b, _shift in periodic_mappings
        for face_id in (face_a, face_b)
    }
    boundary_face_set = set(boundary_faces.tolist())
    if not periodic_face_set <= boundary_face_set:
        raise SolverError("Periodic pairs must reference boundary faces.")
    if any(
        int(patch_ids[face_id]) in periodic_patch_ids
        and face_id not in periodic_face_set
        for face_id in boundary_faces
    ):
        raise SolverError("Periodic patches contain unmatched faces.")
    wall_faces = np.array(
        [
            face_id
            for face_id in boundary_faces
            if int(face_id) not in periodic_face_set
            and conditions[int(patch_ids[face_id])].kind in ("Wall", "Symmetry")
        ],
        dtype=int,
    )
    symmetry_faces = np.array(
        [
            face_id
            for face_id in boundary_faces
            if int(face_id) not in periodic_face_set
            and conditions[int(patch_ids[face_id])].kind == "Symmetry"
        ],
        dtype=int,
    )
    patch_mechanical_conditions = {
        patch_id: condition.mechanical_condition
        for patch_id, condition in conditions.items()
    }
    turbulent_wall_faces = np.asarray(
        [
            face_id
            for face_id in boundary_faces
            if int(face_id) not in periodic_face_set
            and conditions[int(patch_ids[face_id])].kind == "Wall"
            and conditions[int(patch_ids[face_id])].mechanical_condition == "No-slip"
        ],
        dtype=int,
    )
    if len(turbulent_wall_faces):
        wall_distances = np.full(len(mesh.cells), np.inf)
        for face_id in turbulent_wall_faces:
            node_a, node_b = mesh.faces[face_id]
            point_a = np.asarray(mesh.points[node_a], dtype=float)
            point_b = np.asarray(mesh.points[node_b], dtype=float)
            edge = point_b - point_a
            edge_length_squared = float(np.dot(edge, edge))
            offsets = centers - point_a
            projections = np.clip(
                np.sum(offsets * edge, axis=1) / edge_length_squared, 0.0, 1.0
            )
            distances = np.linalg.norm(
                offsets - projections[:, None] * edge, axis=1
            )
            wall_distances = np.minimum(wall_distances, distances)
    else:
        domain_length = max(
            max(x for x, _ in mesh.points) - min(x for x, _ in mesh.points),
            max(y for _, y in mesh.points) - min(y for _, y in mesh.points),
            1e-12,
        )
        wall_distances = np.full(len(mesh.cells), domain_length)
    diffusive_boundary_faces = np.asarray(
        [face_id for face_id in boundary_faces if face_id not in periodic_face_set],
        dtype=int,
    )
    velocity_inlet_faces = []
    velocity_inlet_states = []
    pressure_outlet_faces = []
    pressure_outlet_pressures = []
    far_field_faces = []
    far_field_states = []
    pressure_inlets = []
    mass_flow_inlets = []
    mass_flow_outlets = []
    wall_thermal_boundaries = []
    turbulence_inlets = []
    for patch_id, condition in conditions.items():
        faces = np.asarray(
            [
                face_id
                for face_id in boundary_faces
                if int(patch_ids[face_id]) == patch_id
            ],
            dtype=int,
        )
        if (
            case.numerics.turbulence_model == "Menter k-omega SST"
            and condition.kind
            in ("Velocity Inlet", "Pressure Inlet", "Mass Flow Inlet", "Far Field")
        ):
            intensity = _finite_boundary_value(
                condition.values.get("turbulence_intensity", 1.0),
                f"Turbulence intensity for '{condition.patch_name}'",
            )
            if intensity < 0.0 or intensity > 100.0:
                raise SolverError(
                    f"Turbulence intensity for '{condition.patch_name}' "
                    "must be between 0 and 100 percent."
                )
            length_scale = _positive_boundary_value(
                condition.values.get("turbulence_length_scale", 0.01),
                f"Turbulence length scale for '{condition.patch_name}'",
            )
            turbulence_inlets.append(
                _TurbulenceInlet(faces, intensity, length_scale, condition.patch_name)
            )
        if condition.kind == "Velocity Inlet":
            density = _positive_boundary_value(
                condition.values.get("density", 1.225),
                f"Density for velocity inlet '{condition.patch_name}'",
            )
            temperature = _positive_boundary_value(
                condition.values.get("temperature", 288.15),
                f"Temperature for velocity inlet '{condition.patch_name}'",
            )
            velocity_x = _finite_boundary_value(
                condition.values.get("velocity_x", 0.0),
                f"Velocity X for inlet '{condition.patch_name}'",
            )
            velocity_y = _finite_boundary_value(
                condition.values.get("velocity_y", 0.0),
                f"Velocity Y for inlet '{condition.patch_name}'",
            )
            pressure = density * case.numerics.gas_constant * temperature
            if not math.isfinite(pressure) or pressure <= 0.0:
                raise SolverError(
                    f"Velocity inlet '{condition.patch_name}' produces invalid pressure."
                )
            velocity_inlet_faces.extend(faces.tolist())
            velocity_inlet_states.extend(
                [(density, velocity_x, velocity_y, pressure)] * len(faces)
            )
        elif condition.kind == "Wall" and case.numerics.viscosity_model != "Inviscid":
            mechanical_condition = condition.mechanical_condition
            if mechanical_condition not in ("Slip", "No-slip"):
                raise SolverError(
                    f"Unsupported mechanical condition on wall "
                    f"'{condition.patch_name}'."
                )
            thermal_condition = condition.thermal_condition
            if thermal_condition == "Temperature":
                value = _positive_boundary_value(
                    condition.values.get("temperature", 300.0),
                    f"Wall temperature for '{condition.patch_name}'",
                )
            elif thermal_condition == "Heat Flux":
                value = _finite_boundary_value(
                    condition.values.get("heat_flux", 0.0),
                    f"Wall heat flux for '{condition.patch_name}'",
                )
            else:
                raise SolverError(
                    f"Unsupported thermal condition on wall '{condition.patch_name}'."
                )
            wall_thermal_boundaries.append(
                _WallThermalBoundary(
                    faces,
                    thermal_condition,
                    value,
                    condition.patch_name,
                    mechanical_condition,
                )
            )
        elif condition.kind == "Pressure Outlet":
            pressure = _positive_boundary_value(
                condition.values.get("pressure", 101325.0),
                f"Static pressure for outlet '{condition.patch_name}'",
            )
            pressure_outlet_faces.extend(faces.tolist())
            pressure_outlet_pressures.extend([pressure] * len(faces))
        elif condition.kind == "Pressure Inlet":
            total_pressure = _positive_boundary_value(
                condition.values.get("pressure", 101325.0),
                f"Total pressure for inlet '{condition.patch_name}'",
            )
            total_temperature = _positive_boundary_value(
                condition.values.get("temperature", 288.15),
                f"Total temperature for inlet '{condition.patch_name}'",
            )
            direction = _boundary_flow_direction(
                condition.values.get("velocity_x", 1.0),
                condition.values.get("velocity_y", 0.0),
                f"Pressure inlet '{condition.patch_name}'",
            )
            pressure_inlets.append(
                _PressureInlet(
                    faces,
                    total_pressure,
                    total_temperature,
                    direction,
                    condition.patch_name,
                )
            )
        elif condition.kind == "Mass Flow Inlet":
            mass_flow = _positive_boundary_value(
                condition.values.get("mass_flow", 1.0),
                f"Mass flow for inlet '{condition.patch_name}'",
            )
            total_temperature = _positive_boundary_value(
                condition.values.get("temperature", 288.15),
                f"Total temperature for inlet '{condition.patch_name}'",
            )
            direction = _boundary_flow_direction(
                condition.values.get("velocity_x", 1.0),
                condition.values.get("velocity_y", 0.0),
                f"Mass-flow inlet '{condition.patch_name}'",
            )
            mass_flow_inlets.append(
                _MassFlowInlet(
                    faces,
                    mass_flow,
                    total_temperature,
                    direction,
                    condition.patch_name,
                )
            )
        elif condition.kind == "Mass Flow Outlet":
            mass_flow = _positive_boundary_value(
                condition.values.get("mass_flow", 1.0),
                f"Mass flow for outlet '{condition.patch_name}'",
            )
            mass_flow_outlets.append(_MassFlowOutlet(faces, mass_flow, condition.patch_name))
        elif condition.kind == "Far Field":
            density = _positive_boundary_value(
                condition.values.get("density", 1.225),
                f"Density for far field '{condition.patch_name}'",
            )
            velocity_x = _finite_boundary_value(
                condition.values.get("velocity_x", 0.0),
                f"Velocity X for far field '{condition.patch_name}'",
            )
            velocity_y = _finite_boundary_value(
                condition.values.get("velocity_y", 0.0),
                f"Velocity Y for far field '{condition.patch_name}'",
            )
            pressure = _positive_boundary_value(
                condition.values.get("pressure", 101325.0),
                f"Pressure for far field '{condition.patch_name}'",
            )
            far_field_faces.extend(faces.tolist())
            far_field_states.extend(
                [(density, velocity_x, velocity_y, pressure)] * len(faces)
            )

    gradient_sources = []
    gradient_neighbours = []
    gradient_dx = []
    gradient_dy = []
    for face_id in interior_faces:
        owner, neighbour = int(owners[face_id]), int(neighbours[face_id])
        delta = centers[neighbour] - centers[owner]
        gradient_sources.extend((owner, neighbour))
        gradient_neighbours.extend((neighbour, owner))
        gradient_dx.extend((delta[0], -delta[0]))
        gradient_dy.extend((delta[1], -delta[1]))
    for face_a, face_b, shift in periodic_mappings:
        owner_a, owner_b = int(owners[face_a]), int(owners[face_b])
        delta = centers[owner_b] - np.asarray(shift) - centers[owner_a]
        gradient_sources.extend((owner_a, owner_b))
        gradient_neighbours.extend((owner_b, owner_a))
        gradient_dx.extend((delta[0], -delta[0]))
        gradient_dy.extend((delta[1], -delta[1]))
    if case.numerics.reconstruction == "MUSCL parabolic":
        (
            parabolic_sources,
            parabolic_neighbours,
            parabolic_dx,
            parabolic_dy,
        ) = _parabolic_stencil(
            len(mesh.cells),
            gradient_sources,
            gradient_neighbours,
            gradient_dx,
            gradient_dy,
            max(
                max(x for x, _ in mesh.points) - min(x for x, _ in mesh.points),
                max(y for _, y in mesh.points) - min(y for _, y in mesh.points),
                1e-12,
            ),
        )
    else:
        parabolic_sources = np.empty(0, dtype=int)
        parabolic_neighbours = np.empty(0, dtype=int)
        parabolic_dx = np.empty(0, dtype=float)
        parabolic_dy = np.empty(0, dtype=float)

    limiter_cells = list(owners)
    limiter_dx = list(np.asarray(face_centers)[:, 0] - centers[owners, 0])
    limiter_dy = list(np.asarray(face_centers)[:, 1] - centers[owners, 1])
    for face_id in interior_faces:
        neighbour = int(neighbours[face_id])
        limiter_cells.append(neighbour)
        limiter_dx.append(face_centers[face_id][0] - centers[neighbour, 0])
        limiter_dy.append(face_centers[face_id][1] - centers[neighbour, 1])

    return _Geometry(
        centers=centers,
        areas=areas,
        face_centers=np.asarray(face_centers),
        face_normals=np.asarray(face_normals),
        face_lengths=np.asarray(face_lengths),
        owners=owners,
        neighbours=neighbours,
        patch_ids=patch_ids,
        patch_kinds={patch_id: condition.kind for patch_id, condition in conditions.items()},
        patch_mechanical_conditions=patch_mechanical_conditions,
        wall_distances=wall_distances,
        interior_faces=interior_faces,
        wall_faces=wall_faces,
        symmetry_faces=symmetry_faces,
        diffusive_boundary_faces=diffusive_boundary_faces,
        periodic_faces_a=np.asarray([item[0] for item in periodic_mappings], dtype=int),
        periodic_faces_b=np.asarray([item[1] for item in periodic_mappings], dtype=int),
        periodic_pairs=tuple(case.periodic_pairs),
        velocity_inlet_faces=np.asarray(velocity_inlet_faces, dtype=int),
        velocity_inlet_states=np.asarray(velocity_inlet_states, dtype=float).reshape(
            (-1, 4)
        ),
        pressure_outlet_faces=np.asarray(pressure_outlet_faces, dtype=int),
        pressure_outlet_pressures=np.asarray(
            pressure_outlet_pressures, dtype=float
        ),
        far_field_faces=np.asarray(far_field_faces, dtype=int),
        far_field_states=np.asarray(far_field_states, dtype=float).reshape((-1, 4)),
        pressure_inlets=tuple(pressure_inlets),
        mass_flow_inlets=tuple(mass_flow_inlets),
        mass_flow_outlets=tuple(mass_flow_outlets),
        wall_thermal_boundaries=tuple(wall_thermal_boundaries),
        turbulence_inlets=tuple(turbulence_inlets),
        gradient_sources=np.asarray(gradient_sources, dtype=int),
        gradient_neighbours=np.asarray(gradient_neighbours, dtype=int),
        gradient_dx=np.asarray(gradient_dx, dtype=float),
        gradient_dy=np.asarray(gradient_dy, dtype=float),
        parabolic_sources=parabolic_sources,
        parabolic_neighbours=parabolic_neighbours,
        parabolic_dx=parabolic_dx,
        parabolic_dy=parabolic_dy,
        limiter_cells=np.asarray(limiter_cells, dtype=int),
        limiter_dx=np.asarray(limiter_dx, dtype=float),
        limiter_dy=np.asarray(limiter_dy, dtype=float),
    )


def _parabolic_stencil(
    cell_count: int,
    sources: list[int],
    neighbours: list[int],
    dx: list[float],
    dy: list[float],
    domain_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    adjacency: list[list[tuple[int, float, float]]] = [
        [] for _ in range(cell_count)
    ]
    for source, neighbour, delta_x, delta_y in zip(
        sources, neighbours, dx, dy
    ):
        adjacency[source].append((neighbour, delta_x, delta_y))

    stencil_sources = []
    stencil_neighbours = []
    stencil_dx = []
    stencil_dy = []
    for source in range(cell_count):
        frontier = [(source, 0.0, 0.0)]
        seen = {(source, 0, 0)}
        for _depth in range(2):
            next_frontier = []
            for current, offset_x, offset_y in frontier:
                for neighbour, edge_x, edge_y in adjacency[current]:
                    position_x = offset_x + edge_x
                    position_y = offset_y + edge_y
                    key = (
                        neighbour,
                        round(position_x / domain_scale, 10),
                        round(position_y / domain_scale, 10),
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    next_frontier.append((neighbour, position_x, position_y))
                    if math.hypot(position_x, position_y) > domain_scale * 1e-12:
                        stencil_sources.append(source)
                        stencil_neighbours.append(neighbour)
                        stencil_dx.append(position_x)
                        stencil_dy.append(position_y)
            frontier = next_frontier

    return (
        np.asarray(stencil_sources, dtype=int),
        np.asarray(stencil_neighbours, dtype=int),
        np.asarray(stencil_dx, dtype=float),
        np.asarray(stencil_dy, dtype=float),
    )


def _limited_gradients(
    primitive: np.ndarray, geometry: _Geometry, limiter: str
) -> tuple[np.ndarray, np.ndarray]:
    cell_count = len(primitive)
    gradients = np.zeros((cell_count, 4, 2), dtype=float)
    sources = geometry.gradient_sources
    if len(sources) == 0:
        return gradients, np.ones((cell_count, 4), dtype=float)
    neighbours = geometry.gradient_neighbours
    dx, dy = geometry.gradient_dx, geometry.gradient_dy
    distance_squared = dx * dx + dy * dy
    weights = 1.0 / np.maximum(distance_squared, 1e-300)
    matrix_xx = np.zeros(cell_count)
    matrix_xy = np.zeros(cell_count)
    matrix_yy = np.zeros(cell_count)
    np.add.at(matrix_xx, sources, weights * dx * dx)
    np.add.at(matrix_xy, sources, weights * dx * dy)
    np.add.at(matrix_yy, sources, weights * dy * dy)
    difference = primitive[neighbours] - primitive[sources]
    rhs_x = np.zeros((cell_count, 4))
    rhs_y = np.zeros((cell_count, 4))
    np.add.at(rhs_x, sources, (weights * dx)[:, None] * difference)
    np.add.at(rhs_y, sources, (weights * dy)[:, None] * difference)
    determinant = matrix_xx * matrix_yy - matrix_xy * matrix_xy
    valid = determinant > 1e-14 * np.maximum(matrix_xx * matrix_yy, 1e-300)
    gradients[valid, :, 0] = (
        rhs_x[valid] * matrix_yy[valid, None]
        - rhs_y[valid] * matrix_xy[valid, None]
    ) / determinant[valid, None]
    gradients[valid, :, 1] = (
        rhs_y[valid] * matrix_xx[valid, None]
        - rhs_x[valid] * matrix_xy[valid, None]
    ) / determinant[valid, None]

    local_min = primitive.copy()
    local_max = primitive.copy()
    np.minimum.at(local_min, sources, primitive[neighbours])
    np.maximum.at(local_max, sources, primitive[neighbours])
    cells = geometry.limiter_cells
    increments = (
        gradients[cells, :, 0] * geometry.limiter_dx[:, None]
        + gradients[cells, :, 1] * geometry.limiter_dy[:, None]
    )
    cell_values = primitive[cells]
    positive = increments > 1e-14 * np.maximum(np.abs(cell_values), 1.0)
    negative = increments < -1e-14 * np.maximum(np.abs(cell_values), 1.0)
    ratio = np.ones_like(increments)
    upper = np.broadcast_to(local_max[cells] - cell_values, increments.shape)
    lower = np.broadcast_to(local_min[cells] - cell_values, increments.shape)
    np.divide(upper, increments, out=ratio, where=positive)
    np.divide(lower, increments, out=ratio, where=negative)
    ratio = np.maximum(ratio, 0.0)
    if limiter in ("Barth-Jespersen", "Minmod"):
        face_limiter = np.minimum(ratio, 1.0)
    elif limiter == "Van Leer":
        face_limiter = np.where(
            ratio > 0.0, np.minimum(1.0, 2.0 * ratio / (1.0 + ratio)), 0.0
        )
    else:
        epsilon = 1e-12
        face_limiter = (
            ratio**2 + 2.0 * ratio + epsilon
        ) / (ratio**2 + ratio + 2.0 + epsilon)
        face_limiter = np.minimum(face_limiter, 1.0)
    factors = np.ones((cell_count, 4), dtype=float)
    np.minimum.at(factors, cells, face_limiter)
    return gradients, factors


def _parabolic_reconstruction(
    primitive: np.ndarray, geometry: _Geometry, limiter: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gradients, factors = _limited_gradients(primitive, geometry, limiter)
    cell_count = len(primitive)
    curvature = np.zeros((cell_count, 4, 3), dtype=float)
    sources = geometry.parabolic_sources
    if len(sources) == 0:
        return gradients, curvature, factors

    dx, dy = geometry.parabolic_dx, geometry.parabolic_dy
    scales = np.zeros(cell_count, dtype=float)
    np.maximum.at(scales, sources, np.maximum(np.abs(dx), np.abs(dy)))
    scales = np.maximum(scales, 1e-300)
    normalized_x = dx / scales[sources]
    normalized_y = dy / scales[sources]
    design = np.column_stack(
        (
            normalized_x,
            normalized_y,
            0.5 * normalized_x**2,
            normalized_x * normalized_y,
            0.5 * normalized_y**2,
        )
    )
    weights = 1.0 / np.maximum(normalized_x**2 + normalized_y**2, 1e-300)
    normal_matrix = np.zeros((cell_count, 5, 5), dtype=float)
    right_hand_side = np.zeros((cell_count, 5, 4), dtype=float)
    np.add.at(
        normal_matrix,
        sources,
        weights[:, None, None] * design[:, :, None] * design[:, None, :],
    )
    differences = primitive[geometry.parabolic_neighbours] - primitive[sources]
    np.add.at(
        right_hand_side,
        sources,
        weights[:, None, None] * design[:, :, None] * differences[:, None, :],
    )
    full_rank = np.linalg.matrix_rank(normal_matrix) == 5
    if not np.any(full_rank):
        return gradients, curvature, factors
    coefficients = np.linalg.solve(
        normal_matrix[full_rank], right_hand_side[full_rank]
    )
    cell_scales = scales[full_rank, None]
    gradients[full_rank, :, 0] = coefficients[:, 0, :] / cell_scales
    gradients[full_rank, :, 1] = coefficients[:, 1, :] / cell_scales
    curvature[full_rank, :, 0] = coefficients[:, 2, :] / cell_scales**2
    curvature[full_rank, :, 1] = coefficients[:, 3, :] / cell_scales**2
    curvature[full_rank, :, 2] = coefficients[:, 4, :] / cell_scales**2

    local_min = primitive.copy()
    local_max = primitive.copy()
    np.minimum.at(
        local_min, sources, primitive[geometry.parabolic_neighbours]
    )
    np.maximum.at(
        local_max, sources, primitive[geometry.parabolic_neighbours]
    )
    cells = geometry.limiter_cells
    face_dx, face_dy = geometry.limiter_dx, geometry.limiter_dy
    increments = (
        gradients[cells, :, 0] * face_dx[:, None]
        + gradients[cells, :, 1] * face_dy[:, None]
        + 0.5 * curvature[cells, :, 0] * face_dx[:, None] ** 2
        + curvature[cells, :, 1] * (face_dx * face_dy)[:, None]
        + 0.5 * curvature[cells, :, 2] * face_dy[:, None] ** 2
    )
    cell_values = primitive[cells]
    positive = increments > 1e-14 * np.maximum(np.abs(cell_values), 1.0)
    negative = increments < -1e-14 * np.maximum(np.abs(cell_values), 1.0)
    ratio = np.ones_like(increments)
    upper = np.broadcast_to(local_max[cells] - cell_values, increments.shape)
    lower = np.broadcast_to(local_min[cells] - cell_values, increments.shape)
    np.divide(upper, increments, out=ratio, where=positive)
    np.divide(lower, increments, out=ratio, where=negative)
    ratio = np.maximum(ratio, 0.0)
    if limiter in ("Barth-Jespersen", "Minmod"):
        face_limiter = np.minimum(ratio, 1.0)
    elif limiter == "Van Leer":
        face_limiter = np.where(
            ratio > 0.0, np.minimum(1.0, 2.0 * ratio / (1.0 + ratio)), 0.0
        )
    else:
        epsilon = 1e-12
        face_limiter = (
            ratio**2 + 2.0 * ratio + epsilon
        ) / (ratio**2 + ratio + 2.0 + epsilon)
        face_limiter = np.minimum(face_limiter, 1.0)
    parabolic_factors = np.ones((cell_count, 4), dtype=float)
    np.minimum.at(parabolic_factors, cells, face_limiter)
    factors[full_rank] = parabolic_factors[full_rank]
    return gradients, curvature, factors


def _reconstruct(
    primitive: np.ndarray,
    gradients: np.ndarray,
    factors: np.ndarray,
    cell_ids: np.ndarray,
    offsets: np.ndarray,
    curvature: np.ndarray | None = None,
) -> np.ndarray:
    increments = np.sum(gradients[cell_ids] * offsets[:, None, :], axis=2)
    if curvature is not None:
        dx, dy = offsets[:, 0], offsets[:, 1]
        increments += (
            0.5 * curvature[cell_ids, :, 0] * dx[:, None] ** 2
            + curvature[cell_ids, :, 1] * (dx * dy)[:, None]
            + 0.5 * curvature[cell_ids, :, 2] * dy[:, None] ** 2
        )
    return primitive[cell_ids] + factors[cell_ids] * increments


def _residual(
    conservative: np.ndarray,
    geometry: _Geometry,
    settings: NumericalSettings,
    turbulence_conservative: np.ndarray | None = None,
) -> np.ndarray:
    gamma = settings.gamma
    curvature = None
    primitive = _primitive_from_conservative(
        conservative, turbulence_conservative, gamma
    )
    turbulence = (
        None
        if turbulence_conservative is None
        else turbulence_conservative / primitive[:, :1]
    )
    boundary_turbulence = (
        {}
        if turbulence is None
        else _turbulence_boundary_values(primitive, turbulence, geometry, settings)
    )
    if settings.reconstruction == "MUSCL parabolic":
        gradients, curvature, factors = _parabolic_reconstruction(
            primitive, geometry, settings.limiter
        )
    elif settings.reconstruction == "MUSCL linear":
        gradients, factors = _limited_gradients(
            primitive, geometry, settings.limiter
        )
    else:
        gradients = np.zeros((len(primitive), 4, 2))
        factors = np.ones((len(primitive), 4))

    def reconstruct(cell_ids: np.ndarray, offsets: np.ndarray) -> np.ndarray:
        return _reconstruct(
            primitive, gradients, factors, cell_ids, offsets, curvature
        )

    derivative = np.zeros_like(conservative)
    if len(geometry.interior_faces):
        faces = geometry.interior_faces
        owners = geometry.owners[faces]
        neighbours = geometry.neighbours[faces]
        left = reconstruct(
            owners,
            geometry.face_centers[faces] - geometry.centers[owners],
        )
        right = reconstruct(
            neighbours,
            geometry.face_centers[faces] - geometry.centers[neighbours],
        )
        flux = numerical_flux(
            left, right, geometry.face_normals[faces], gamma, settings.flux
        )
        if turbulence is not None:
            flux[:, 3] += flux[:, 0] * np.where(
                flux[:, 0] >= 0.0, turbulence[owners, 0], turbulence[neighbours, 0]
            )
        integrated = flux * geometry.face_lengths[faces, None]
        np.add.at(derivative, owners, -integrated / geometry.areas[owners, None])
        np.add.at(
            derivative, neighbours, integrated / geometry.areas[neighbours, None]
        )
    if len(geometry.periodic_faces_a):
        faces_a, faces_b = geometry.periodic_faces_a, geometry.periodic_faces_b
        owners_a, owners_b = geometry.owners[faces_a], geometry.owners[faces_b]
        left = reconstruct(
            owners_a,
            geometry.face_centers[faces_a] - geometry.centers[owners_a],
        )
        right = reconstruct(
            owners_b,
            geometry.face_centers[faces_b] - geometry.centers[owners_b],
        )
        flux = numerical_flux(
            left, right, geometry.face_normals[faces_a], gamma, settings.flux
        )
        if turbulence is not None:
            flux[:, 3] += flux[:, 0] * np.where(
                flux[:, 0] >= 0.0,
                turbulence[owners_a, 0],
                turbulence[owners_b, 0],
            )
        integrated = flux * geometry.face_lengths[faces_a, None]
        np.add.at(
            derivative, owners_a, -integrated / geometry.areas[owners_a, None]
        )
        np.add.at(
            derivative, owners_b, integrated / geometry.areas[owners_b, None]
        )
    if len(geometry.wall_faces):
        faces = geometry.wall_faces
        owners = geometry.owners[faces]
        left = reconstruct(
            owners,
            geometry.face_centers[faces] - geometry.centers[owners],
        )
        right = left.copy()
        normal_velocity = np.sum(
            left[:, 1:3] * geometry.face_normals[faces], axis=1
        )
        right[:, 1:3] -= 2.0 * normal_velocity[:, None] * geometry.face_normals[faces]
        flux = numerical_flux(
            left, right, geometry.face_normals[faces], gamma, settings.flux
        )
        if turbulence is not None:
            flux[:, 3] += flux[:, 0] * turbulence[owners, 0]
        integrated = flux * geometry.face_lengths[faces, None]
        np.add.at(derivative, owners, -integrated / geometry.areas[owners, None])
    if len(geometry.velocity_inlet_faces):
        faces = geometry.velocity_inlet_faces
        owners = geometry.owners[faces]
        interior = reconstruct(
            owners,
            geometry.face_centers[faces] - geometry.centers[owners],
        )
        exterior = geometry.velocity_inlet_states
        flux = numerical_flux(
            interior, exterior, geometry.face_normals[faces], gamma, settings.flux
        )
        if turbulence is not None:
            exterior_k = np.array(
                [
                    boundary_turbulence.get(int(face_id), turbulence[owner])[0]
                    for face_id, owner in zip(faces, owners)
                ]
            )
            flux[:, 3] += flux[:, 0] * np.where(
                flux[:, 0] >= 0.0, turbulence[owners, 0], exterior_k
            )
        integrated = flux * geometry.face_lengths[faces, None]
        np.add.at(derivative, owners, -integrated / geometry.areas[owners, None])
    if len(geometry.pressure_outlet_faces):
        faces = geometry.pressure_outlet_faces
        owners = geometry.owners[faces]
        interior = reconstruct(
            owners,
            geometry.face_centers[faces] - geometry.centers[owners],
        )
        exterior = interior.copy()
        exterior[:, 3] = geometry.pressure_outlet_pressures
        flux = numerical_flux(
            interior, exterior, geometry.face_normals[faces], gamma, settings.flux
        )
        if turbulence is not None:
            exterior_k = np.array(
                [
                    boundary_turbulence.get(int(face_id), turbulence[owner])[0]
                    for face_id, owner in zip(faces, owners)
                ]
            )
            flux[:, 3] += flux[:, 0] * np.where(
                flux[:, 0] >= 0.0, turbulence[owners, 0], exterior_k
            )
        integrated = flux * geometry.face_lengths[faces, None]
        np.add.at(derivative, owners, -integrated / geometry.areas[owners, None])
    if len(geometry.far_field_faces):
        faces = geometry.far_field_faces
        owners = geometry.owners[faces]
        interior = reconstruct(
            owners,
            geometry.face_centers[faces] - geometry.centers[owners],
        )
        exterior = _far_field_boundary_states(
            interior,
            geometry.far_field_states,
            geometry.face_normals[faces],
            gamma,
        )
        flux = numerical_flux(
            interior, exterior, geometry.face_normals[faces], gamma, settings.flux
        )
        if turbulence is not None:
            exterior_k = np.array(
                [
                    boundary_turbulence.get(int(face_id), turbulence[owner])[0]
                    for face_id, owner in zip(faces, owners)
                ]
            )
            flux[:, 3] += flux[:, 0] * np.where(
                flux[:, 0] >= 0.0, turbulence[owners, 0], exterior_k
            )
        integrated = flux * geometry.face_lengths[faces, None]
        np.add.at(derivative, owners, -integrated / geometry.areas[owners, None])
    for boundary in geometry.pressure_inlets:
        faces = boundary.faces
        owners = geometry.owners[faces]
        interior = reconstruct(
            owners,
            geometry.face_centers[faces] - geometry.centers[owners],
        )
        exterior = _pressure_inlet_boundary_states(
            interior,
            geometry.face_normals[faces],
            boundary,
            gamma,
            settings.gas_constant,
        )
        flux = numerical_flux(
            interior, exterior, geometry.face_normals[faces], gamma, settings.flux
        )
        if turbulence is not None:
            exterior_k = np.array(
                [
                    boundary_turbulence.get(int(face_id), turbulence[owner])[0]
                    for face_id, owner in zip(faces, owners)
                ]
            )
            flux[:, 3] += flux[:, 0] * np.where(
                flux[:, 0] >= 0.0, turbulence[owners, 0], exterior_k
            )
        integrated = flux * geometry.face_lengths[faces, None]
        np.add.at(derivative, owners, -integrated / geometry.areas[owners, None])
    for boundary in geometry.mass_flow_inlets:
        faces = boundary.faces
        owners = geometry.owners[faces]
        interior = reconstruct(
            owners,
            geometry.face_centers[faces] - geometry.centers[owners],
        )
        exterior = _mass_flow_inlet_boundary_states(
            interior,
            geometry.face_normals[faces],
            geometry.face_lengths[faces],
            boundary,
            gamma,
            settings.gas_constant,
        )
        flux = numerical_flux(
            interior, exterior, geometry.face_normals[faces], gamma, settings.flux
        )
        if turbulence is not None:
            exterior_k = np.array(
                [
                    boundary_turbulence.get(int(face_id), turbulence[owner])[0]
                    for face_id, owner in zip(faces, owners)
                ]
            )
            flux[:, 3] += flux[:, 0] * np.where(
                flux[:, 0] >= 0.0, turbulence[owners, 0], exterior_k
            )
        integrated = flux * geometry.face_lengths[faces, None]
        np.add.at(derivative, owners, -integrated / geometry.areas[owners, None])
    for boundary in geometry.mass_flow_outlets:
        faces = boundary.faces
        owners = geometry.owners[faces]
        interior = reconstruct(
            owners,
            geometry.face_centers[faces] - geometry.centers[owners],
        )
        exterior = _mass_flow_outlet_boundary_states(
            interior,
            geometry.face_normals[faces],
            geometry.face_lengths[faces],
            boundary,
        )
        flux = numerical_flux(
            interior, exterior, geometry.face_normals[faces], gamma, settings.flux
        )
        if turbulence is not None:
            flux[:, 3] += flux[:, 0] * turbulence[owners, 0]
        integrated = flux * geometry.face_lengths[faces, None]
        np.add.at(derivative, owners, -integrated / geometry.areas[owners, None])
    if settings.viscosity_model != "Inviscid":
        derivative += _viscous_residual(
            primitive, geometry, settings, turbulence
        )
    if settings.acceleration_x != 0.0 or settings.acceleration_y != 0.0:
        derivative[:, 1] += primitive[:, 0] * settings.acceleration_x
        derivative[:, 2] += primitive[:, 0] * settings.acceleration_y
        derivative[:, 3] += primitive[:, 0] * (
            settings.acceleration_x * primitive[:, 1]
            + settings.acceleration_y * primitive[:, 2]
        )
    if not np.isfinite(derivative).all():
        raise SolverError("Spatial residual contains non-finite values.")
    return derivative


def _dynamic_viscosity(
    temperature: np.ndarray, settings: NumericalSettings
) -> np.ndarray:
    if settings.viscosity_model == "Constant":
        return np.full_like(temperature, settings.dynamic_viscosity, dtype=float)
    if settings.viscosity_model == "Sutherland":
        reference_temperature = settings.sutherland_reference_temperature
        viscosity = settings.sutherland_reference_viscosity * (
            temperature / reference_temperature
        ) ** 1.5
        viscosity *= (reference_temperature + settings.sutherland_temperature) / (
            temperature + settings.sutherland_temperature
        )
        if not np.isfinite(viscosity).all() or np.any(viscosity <= 0.0):
            raise SolverError("Sutherland law produced an invalid viscosity.")
        return viscosity
    return np.zeros_like(temperature, dtype=float)


def _viscous_inlet_states(
    primitive: np.ndarray, geometry: _Geometry, settings: NumericalSettings
) -> dict[int, np.ndarray]:
    states_by_face = {
        int(face_id): state
        for face_id, state in zip(
            geometry.velocity_inlet_faces, geometry.velocity_inlet_states
        )
    }
    for boundary in geometry.pressure_inlets:
        faces = boundary.faces
        states = _pressure_inlet_boundary_states(
            primitive[geometry.owners[faces]],
            geometry.face_normals[faces],
            boundary,
            settings.gamma,
            settings.gas_constant,
        )
        states_by_face.update(
            {int(face_id): state for face_id, state in zip(faces, states)}
        )
    for boundary in geometry.mass_flow_inlets:
        faces = boundary.faces
        states = _mass_flow_inlet_boundary_states(
            primitive[geometry.owners[faces]],
            geometry.face_normals[faces],
            geometry.face_lengths[faces],
            boundary,
            settings.gamma,
            settings.gas_constant,
        )
        states_by_face.update(
            {int(face_id): state for face_id, state in zip(faces, states)}
        )
    if len(geometry.far_field_faces):
        faces = geometry.far_field_faces
        states = _far_field_boundary_states(
            primitive[geometry.owners[faces]],
            geometry.far_field_states,
            geometry.face_normals[faces],
            settings.gamma,
        )
        states_by_face.update(
            {int(face_id): state for face_id, state in zip(faces, states)}
        )
    return states_by_face


def _wall_friction_velocity(
    speed: float, density: float, viscosity: float, distance: float
) -> float:
    if speed <= 0.0:
        return 0.0
    kappa = 0.41
    kinematic_viscosity = viscosity / density

    def law_value(y_plus: float) -> float:
        if y_plus <= 11.06:
            return y_plus
        return math.log(y_plus) / kappa + 5.2

    lower = 1e-14
    upper = max(speed, math.sqrt(speed * kinematic_viscosity / distance))
    for _ in range(80):
        friction_velocity = 0.5 * (lower + upper)
        y_plus = friction_velocity * distance / kinematic_viscosity
        residual = speed / friction_velocity - law_value(y_plus)
        if residual > 0.0:
            lower = friction_velocity
        else:
            upper = friction_velocity
    return 0.5 * (lower + upper)


def derive_solution_fields(
    case: Case, states: tuple[InitialState, ...]
) -> dict[str, tuple[float, ...]]:
    """Compute cell-centered thermodynamic and wall-resolution fields."""
    if case.mesh is None:
        return {}
    if len(states) != len(case.mesh.cells):
        raise SolverError("Solution state count must match the mesh cell count.")
    settings = case.numerics
    if settings.gamma <= 1.0 or settings.gas_constant <= 0.0:
        raise SolverError("Derived fields require gamma > 1 and a positive gas constant.")
    primitive = np.asarray(
        [
            (state.density, state.velocity_x, state.velocity_y, state.pressure)
            for state in states
        ],
        dtype=float,
    )
    if not np.isfinite(primitive).all() or np.any(primitive[:, (0, 3)] <= 0.0):
        raise SolverError(
            "Derived fields require finite states with positive density and pressure."
        )

    density, velocity_x, velocity_y, pressure = primitive.T
    speed_squared = velocity_x**2 + velocity_y**2
    mach_squared = speed_squared * density / (settings.gamma * pressure)
    total_factor = 1.0 + 0.5 * (settings.gamma - 1.0) * mach_squared
    temperature = pressure / (density * settings.gas_constant)
    total_temperature = temperature * total_factor
    total_pressure = pressure * total_factor ** (
        settings.gamma / (settings.gamma - 1.0)
    )
    total_density = total_pressure / (settings.gas_constant * total_temperature)
    y_plus = np.zeros(len(states), dtype=float)

    if settings.viscosity_model != "Inviscid":
        temperature = np.maximum(temperature, 1e-12)
        viscosity = _dynamic_viscosity(temperature, settings)
        if not np.isfinite(viscosity).all() or np.any(viscosity <= 0.0):
            raise SolverError("y+ requires positive finite molecular viscosity.")
        cell_centers = []
        for node_ids in case.mesh.cells:
            points = [case.mesh.points[node_id] for node_id in node_ids]
            twice_area = sum(
                x0 * y1 - x1 * y0
                for (x0, y0), (x1, y1) in zip(
                    points, points[1:] + points[:1]
                )
            )
            x_sum = 0.0
            y_sum = 0.0
            for (x0, y0), (x1, y1) in zip(
                points, points[1:] + points[:1]
            ):
                cross = x0 * y1 - x1 * y0
                x_sum += (x0 + x1) * cross
                y_sum += (y0 + y1) * cross
            cell_centers.append(
                (x_sum / (3.0 * twice_area), y_sum / (3.0 * twice_area))
            )
        conditions = {
            boundary.patch_id: boundary for boundary in case.boundary_conditions
        }
        closest_wall: dict[int, tuple[float, np.ndarray]] = {}
        no_slip_faces = [
            face_id
            for face_id, neighbour in enumerate(case.mesh.face_neighbours)
            if neighbour < 0
            and int(case.mesh.face_patch_ids[face_id]) in conditions
            and conditions[int(case.mesh.face_patch_ids[face_id])].kind == "Wall"
            and conditions[
                int(case.mesh.face_patch_ids[face_id])
            ].mechanical_condition
            == "No-slip"
        ]
        for face_id in no_slip_faces:
            owner = case.mesh.face_owners[face_id]
            node_a, node_b = case.mesh.faces[face_id]
            point_a = np.asarray(case.mesh.points[node_a], dtype=float)
            point_b = np.asarray(case.mesh.points[node_b], dtype=float)
            edge = point_b - point_a
            edge_length = float(np.linalg.norm(edge))
            if edge_length <= 1e-14:
                raise SolverError("Wall face has invalid length.")
            normal = np.array((edge[1], -edge[0])) / edge_length
            distance = abs(
                float(np.dot(0.5 * (point_a + point_b) - cell_centers[owner], normal))
            )
            if distance <= 1e-14:
                raise SolverError("Wall face has invalid cell-center distance.")
            if owner not in closest_wall or distance < closest_wall[owner][0]:
                closest_wall[owner] = (distance, normal)

        for owner, (distance, normal) in closest_wall.items():
            velocity = primitive[owner, 1:3]
            tangential_velocity = velocity - float(np.dot(velocity, normal)) * normal
            speed = float(np.linalg.norm(tangential_velocity))
            if (
                settings.turbulence_model == "Menter k-omega SST"
                and settings.turbulence_wall_treatment == "Wall functions"
            ):
                friction_velocity = _wall_friction_velocity(
                    speed,
                    float(density[owner]),
                    float(viscosity[owner]),
                    distance,
                )
            else:
                wall_shear = viscosity[owner] * speed / distance
                friction_velocity = math.sqrt(wall_shear / density[owner])
            y_plus[owner] = (
                friction_velocity * distance * density[owner] / viscosity[owner]
            )

    fields = {
        "mach_number": np.sqrt(mach_squared),
        "total_pressure": total_pressure,
        "total_temperature": total_temperature,
        "total_density": total_density,
        "y_plus": y_plus,
    }
    if settings.turbulence_model == "Menter k-omega SST":
        geometry = _build_geometry(case.mesh, case)
        fields["wall_distance"] = geometry.wall_distances
        if settings.turbulence_omega_formulation == "Regularized k-log-omega":
            turbulence_omega = np.asarray(
                [state.turbulence_omega for state in states], dtype=float
            )
            wall_scale = _sst_omega_wall_scale(
                primitive, geometry.wall_distances, settings
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                fields["log_regularized_omega"] = (
                    np.log(turbulence_omega) - np.log(wall_scale)
                )
    if any(not np.isfinite(values).all() for values in fields.values()):
        raise SolverError("Derived solution fields contain non-finite values.")
    return {
        key: tuple(float(value) for value in values)
        for key, values in fields.items()
    }


def _sst_wall_values(
    primitive: np.ndarray,
    turbulence: np.ndarray,
    geometry: _Geometry,
    settings: NumericalSettings,
) -> dict[int, tuple[float, float]]:
    temperature = primitive[:, 3] / (
        primitive[:, 0] * settings.gas_constant
    )
    viscosity = _dynamic_viscosity(temperature, settings)
    wall_values = {}
    for boundary in geometry.wall_thermal_boundaries:
        if boundary.mechanical_condition != "No-slip":
            continue
        for face_id in boundary.faces:
            owner = int(geometry.owners[face_id])
            normal = geometry.face_normals[face_id]
            offset = geometry.face_centers[face_id] - geometry.centers[owner]
            distance = abs(float(np.dot(offset, normal)))
            if distance <= 1e-14:
                raise SolverError("Wall face has invalid cell-center distance.")
            if settings.turbulence_wall_treatment == "Wall-resolved":
                k_wall = 0.0
                if (
                    settings.turbulence_omega_formulation
                    == "Regularized k-log-omega"
                ):
                    omega_wall = (
                        6.0
                        * viscosity[owner]
                        / (0.09 * primitive[owner, 0] * distance**2)
                    )
                else:
                    omega_wall = (
                        60.0
                        * viscosity[owner]
                        / (0.075 * primitive[owner, 0] * distance**2)
                    )
            else:
                tangential_velocity = primitive[owner, 1:3] - (
                    float(np.dot(primitive[owner, 1:3], normal)) * normal
                )
                friction_velocity = _wall_friction_velocity(
                    float(np.linalg.norm(tangential_velocity)),
                    float(primitive[owner, 0]),
                    float(viscosity[owner]),
                    distance,
                )
                k_wall = friction_velocity**2 / math.sqrt(0.09)
                omega_wall = friction_velocity / (
                    math.sqrt(0.09) * 0.41 * distance
                )
            wall_values[int(face_id)] = (
                k_wall,
                max(omega_wall, 1e-12),
            )
    return wall_values


def _turbulence_boundary_values(
    primitive: np.ndarray,
    turbulence: np.ndarray,
    geometry: _Geometry,
    settings: NumericalSettings,
) -> dict[int, tuple[float, float]]:
    flow_states = _viscous_inlet_states(primitive, geometry, settings)
    boundary_values = {}
    for boundary in geometry.turbulence_inlets:
        for face_id in boundary.faces:
            state = flow_states.get(int(face_id))
            if state is None:
                continue
            speed = math.hypot(float(state[1]), float(state[2]))
            kinetic_energy = 1.5 * (
                boundary.intensity / 100.0 * speed
            ) ** 2
            omega = math.sqrt(kinetic_energy) / (
                0.09**0.25 * boundary.length_scale
            )
            boundary_values[int(face_id)] = (
                kinetic_energy,
                max(omega, 1e-12),
            )
    return boundary_values


def _scalar_gradients(
    values: np.ndarray,
    geometry: _Geometry,
    boundary_values: dict[int, tuple[float, float]],
) -> np.ndarray:
    cell_count, component_count = values.shape
    sources = [geometry.gradient_sources]
    dx_parts = [geometry.gradient_dx]
    dy_parts = [geometry.gradient_dy]
    differences = [
        values[geometry.gradient_neighbours]
        - values[geometry.gradient_sources]
    ]
    boundary_sources = []
    boundary_dx = []
    boundary_dy = []
    boundary_differences = []
    for face_id in geometry.diffusive_boundary_faces:
        owner = int(geometry.owners[face_id])
        normal = geometry.face_normals[face_id]
        offset = geometry.face_centers[face_id] - geometry.centers[owner]
        normal_distance = float(np.dot(offset, normal))
        ghost_offset = 2.0 * normal_distance * normal
        boundary_value = boundary_values.get(int(face_id))
        ghost_difference = (
            2.0 * (np.asarray(boundary_value) - values[owner])
            if boundary_value is not None
            else np.zeros(component_count)
        )
        boundary_sources.append(owner)
        boundary_dx.append(ghost_offset[0])
        boundary_dy.append(ghost_offset[1])
        boundary_differences.append(ghost_difference)
    if boundary_sources:
        sources.append(np.asarray(boundary_sources, dtype=int))
        dx_parts.append(np.asarray(boundary_dx, dtype=float))
        dy_parts.append(np.asarray(boundary_dy, dtype=float))
        differences = [
            differences[0],
            np.asarray(boundary_differences, dtype=float),
        ]
    source_ids = np.concatenate(sources)
    dx = np.concatenate(dx_parts)
    dy = np.concatenate(dy_parts)
    delta = np.vstack(differences)
    gradients = np.zeros((cell_count, component_count, 2), dtype=float)
    if not len(source_ids):
        return gradients
    weights = 1.0 / np.maximum(dx * dx + dy * dy, 1e-300)
    xx = np.zeros(cell_count)
    xy = np.zeros(cell_count)
    yy = np.zeros(cell_count)
    rhs_x = np.zeros((cell_count, component_count))
    rhs_y = np.zeros((cell_count, component_count))
    np.add.at(xx, source_ids, weights * dx * dx)
    np.add.at(xy, source_ids, weights * dx * dy)
    np.add.at(yy, source_ids, weights * dy * dy)
    np.add.at(rhs_x, source_ids, (weights * dx)[:, None] * delta)
    np.add.at(rhs_y, source_ids, (weights * dy)[:, None] * delta)
    determinant = xx * yy - xy * xy
    valid = determinant > 1e-14 * np.maximum(xx * yy, 1e-300)
    gradients[valid, :, 0] = (
        rhs_x[valid] * yy[valid, None] - rhs_y[valid] * xy[valid, None]
    ) / determinant[valid, None]
    gradients[valid, :, 1] = (
        rhs_y[valid] * xx[valid, None] - rhs_x[valid] * xy[valid, None]
    ) / determinant[valid, None]
    return gradients


def _sst_omega_wall_scale(
    primitive: np.ndarray,
    distances: np.ndarray,
    settings: NumericalSettings,
) -> np.ndarray:
    density = primitive[:, 0]
    temperature = primitive[:, 3] / (density * settings.gas_constant)
    viscosity = _dynamic_viscosity(temperature, settings)
    distance = np.maximum(distances, 1e-14)
    return 6.0 * viscosity / (density * 0.09 * distance**2)


def _sst_regularized_variable(
    primitive: np.ndarray,
    omega: np.ndarray,
    geometry: _Geometry,
    settings: NumericalSettings,
) -> np.ndarray:
    wall_scale = _sst_omega_wall_scale(
        primitive, geometry.wall_distances, settings
    )
    variable = np.log(omega) - np.log(wall_scale)
    if not np.isfinite(variable).all():
        raise SolverError("Regularized SST omega initialization is non-finite.")
    return variable


def _sst_regularized_turbulence(
    primitive: np.ndarray,
    k_conservative: np.ndarray,
    omega_variable: np.ndarray,
    geometry: _Geometry,
    settings: NumericalSettings,
) -> np.ndarray:
    wall_scale = _sst_omega_wall_scale(
        primitive, geometry.wall_distances, settings
    )
    log_omega = np.log(wall_scale) + omega_variable
    maximum_log = math.log(np.finfo(float).max) - 2.0
    if not np.isfinite(log_omega).all() or np.any(log_omega > maximum_log):
        raise SolverError("Regularized SST omega update exceeded the finite range.")
    omega = np.maximum(np.exp(log_omega), 1e-12)
    turbulence = np.column_stack(
        (np.maximum(k_conservative, 0.0), primitive[:, 0] * omega)
    )
    if not np.isfinite(turbulence).all():
        raise SolverError("Regularized SST update produced non-finite values.")
    return turbulence


def _sst_regularized_rate(
    primitive: np.ndarray,
    flow_derivative: np.ndarray,
    turbulence_conservative: np.ndarray,
    turbulence_derivative: np.ndarray,
    settings: NumericalSettings,
) -> np.ndarray:
    density = primitive[:, 0]
    omega = turbulence_conservative[:, 1] / density
    rate = turbulence_derivative[:, 1] / (
        density * np.maximum(omega, 1e-12)
    )
    if settings.viscosity_model == "Sutherland":
        velocity = primitive[:, 1:3]
        density_rate = flow_derivative[:, 0]
        internal_energy_rate = (
            flow_derivative[:, 3]
            - np.sum(velocity * flow_derivative[:, 1:3], axis=1)
            + 0.5 * np.sum(velocity**2, axis=1) * density_rate
            - turbulence_derivative[:, 0]
        )
        pressure_rate = (settings.gamma - 1.0) * internal_energy_rate
        temperature = primitive[:, 3] / (density * settings.gas_constant)
        temperature_rate = (
            pressure_rate / (density * settings.gas_constant)
            - temperature * density_rate / density
        )
        viscosity_log_rate = (
            1.5 / temperature
            - 1.0 / (temperature + settings.sutherland_temperature)
        ) * temperature_rate
        rate -= viscosity_log_rate
    if not np.isfinite(rate).all():
        raise SolverError("Regularized SST omega rate is non-finite.")
    return rate


def _sst_fields(
    primitive: np.ndarray,
    turbulence: np.ndarray,
    geometry: _Geometry,
    settings: NumericalSettings,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    velocity_gradients = _viscous_gradients(primitive, geometry, settings)
    wall_values = _sst_wall_values(
        primitive, turbulence, geometry, settings
    )
    boundary_values = _turbulence_boundary_values(
        primitive, turbulence, geometry, settings
    )
    boundary_values.update(wall_values)
    turbulence_gradients = _scalar_gradients(
        turbulence, geometry, boundary_values
    )
    density = primitive[:, 0]
    kinetic_energy = np.maximum(turbulence[:, 0], 0.0)
    omega = np.maximum(turbulence[:, 1], 1e-12)
    if settings.turbulence_omega_formulation == "Regularized k-log-omega":
        wall_scale = _sst_omega_wall_scale(
            primitive, geometry.wall_distances, settings
        )
        regularized_values = np.column_stack(
            (turbulence[:, 0], np.log(omega) - np.log(wall_scale))
        )
        regularized_boundaries = {}
        wall_log_boundaries = {}
        for face_id, (k_value, omega_value) in boundary_values.items():
            owner = int(geometry.owners[face_id])
            if face_id in wall_values:
                regularized_boundaries[face_id] = (k_value, 0.0)
                normal_distance = abs(
                    float(
                        np.dot(
                            geometry.face_centers[face_id]
                            - geometry.centers[owner],
                            geometry.face_normals[face_id],
                        )
                    )
                )
                face_scale = _sst_omega_wall_scale(
                    primitive[owner : owner + 1],
                    np.array([normal_distance]),
                    settings,
                )[0]
                log_face_scale = math.log(float(face_scale))
                wall_log_boundaries[face_id] = (
                    log_face_scale,
                    log_face_scale,
                )
            else:
                regularized_boundaries[face_id] = (
                    k_value,
                    math.log(max(omega_value, 1e-12))
                    - math.log(wall_scale[owner]),
                )
        regularized_gradients = _scalar_gradients(
            regularized_values, geometry, regularized_boundaries
        )
        log_wall_values = np.log(wall_scale)
        wall_log_gradients = _scalar_gradients(
            np.column_stack((log_wall_values, log_wall_values)),
            geometry,
            wall_log_boundaries,
        )
        turbulence_gradients = regularized_gradients
        turbulence_gradients[:, 1, :] = omega[:, None] * (
            regularized_gradients[:, 1, :]
            + wall_log_gradients[:, 0, :]
        )
    temperature = primitive[:, 3] / (density * settings.gas_constant)
    molecular_viscosity = _dynamic_viscosity(temperature, settings)
    k_gradient = turbulence_gradients[:, 0, :]
    omega_gradient = turbulence_gradients[:, 1, :]
    cross_diffusion = np.maximum(
        2.0
        * density
        * 0.856
        / omega
        * np.sum(k_gradient * omega_gradient, axis=1),
        1e-20,
    )
    distance = np.maximum(geometry.wall_distances, 1e-12)
    k_term = np.sqrt(kinetic_energy) / (0.09 * omega * distance)
    viscous_term = 500.0 * molecular_viscosity / (
        density * distance**2 * omega
    )
    arg1 = np.minimum(
        np.maximum(k_term, viscous_term),
        4.0 * density * 0.856 * kinetic_energy / (cross_diffusion * distance**2),
    )
    f1 = np.tanh(np.maximum(arg1, 0.0) ** 4)
    arg2 = np.maximum(2.0 * k_term, viscous_term)
    f2 = np.tanh(np.maximum(arg2, 0.0) ** 2)
    sigma_k = 0.85 * f1 + 1.0 * (1.0 - f1)
    sigma_omega = 0.5 * f1 + 0.856 * (1.0 - f1)
    beta = 0.075 * f1 + 0.0828 * (1.0 - f1)
    gamma_1 = 0.075 / 0.09 - 0.5 * 0.41**2 / math.sqrt(0.09)
    gamma_2 = 0.0828 / 0.09 - 0.856 * 0.41**2 / math.sqrt(0.09)
    gamma = gamma_1 * f1 + gamma_2 * (1.0 - f1)
    du_dx = velocity_gradients[:, 0, 0]
    du_dy = velocity_gradients[:, 0, 1]
    dv_dx = velocity_gradients[:, 1, 0]
    dv_dy = velocity_gradients[:, 1, 1]
    divergence = du_dx + dv_dy
    strain_squared = (
        du_dx**2 + dv_dy**2 + 0.5 * (du_dy + dv_dx) ** 2
    )
    production = (
        2.0
        * 0.31
        * density
        * kinetic_energy
        / np.maximum(
            0.31 * omega,
            np.abs(du_dy - dv_dx) * f2,
        )
        * (strain_squared - divergence**2 / 3.0)
        - (2.0 / 3.0) * density * kinetic_energy * divergence
    )
    production = np.minimum(production, 20.0 * 0.09 * density * omega * kinetic_energy)
    turbulent_viscosity = (
        density
        * 0.31
        * kinetic_energy
        / np.maximum(0.31 * omega, np.abs(du_dy - dv_dx) * f2)
    )
    return (
        turbulent_viscosity,
        turbulence_gradients,
        sigma_k,
        sigma_omega,
        np.column_stack((production, beta, gamma, cross_diffusion)),
    )


def _sst_mass_fluxes(
    primitive: np.ndarray,
    geometry: _Geometry,
    settings: NumericalSettings,
) -> np.ndarray:
    mass_fluxes = np.zeros(len(geometry.owners))
    faces = geometry.interior_faces
    if len(faces):
        owners, neighbours = geometry.owners[faces], geometry.neighbours[faces]
        mass_fluxes[faces] = numerical_flux(
            primitive[owners],
            primitive[neighbours],
            geometry.face_normals[faces],
            settings.gamma,
            settings.flux,
        )[:, 0]
    faces_a, faces_b = geometry.periodic_faces_a, geometry.periodic_faces_b
    if len(faces_a):
        owners_a, owners_b = geometry.owners[faces_a], geometry.owners[faces_b]
        mass_fluxes[faces_a] = numerical_flux(
            primitive[owners_a],
            primitive[owners_b],
            geometry.face_normals[faces_a],
            settings.gamma,
            settings.flux,
        )[:, 0]

    faces = geometry.diffusive_boundary_faces
    owners = geometry.owners[faces]
    exterior = primitive[owners].copy()
    patch_kinds = np.array(
        [geometry.patch_kinds[int(geometry.patch_ids[face])] for face in faces]
    )
    walls = np.isin(patch_kinds, ("Wall", "Symmetry"))
    if np.any(walls):
        normal_velocity = np.sum(
            exterior[walls, 1:3] * geometry.face_normals[faces[walls]], axis=1
        )
        exterior[walls, 1:3] -= (
            2.0
            * normal_velocity[:, None]
            * geometry.face_normals[faces[walls]]
        )
    for boundary_faces, states in (
        (geometry.velocity_inlet_faces, geometry.velocity_inlet_states),
        (
            geometry.pressure_outlet_faces,
            np.column_stack(
                (
                    primitive[geometry.owners[geometry.pressure_outlet_faces], 0],
                    primitive[geometry.owners[geometry.pressure_outlet_faces], 1:3],
                    geometry.pressure_outlet_pressures,
                )
            ),
        ),
    ):
        for face_id, state in zip(boundary_faces, states):
            position = int(np.flatnonzero(faces == face_id)[0])
            exterior[position] = state
    if len(geometry.far_field_faces):
        owners_far = geometry.owners[geometry.far_field_faces]
        far_states = _far_field_boundary_states(
            primitive[owners_far],
            geometry.far_field_states,
            geometry.face_normals[geometry.far_field_faces],
            settings.gamma,
        )
        for face_id, state in zip(geometry.far_field_faces, far_states):
            position = int(np.flatnonzero(faces == face_id)[0])
            exterior[position] = state
    for boundary in geometry.pressure_inlets:
        boundary_states = _pressure_inlet_boundary_states(
            primitive[geometry.owners[boundary.faces]],
            geometry.face_normals[boundary.faces],
            boundary,
            settings.gamma,
            settings.gas_constant,
        )
        for face_id, state in zip(boundary.faces, boundary_states):
            position = int(np.flatnonzero(faces == face_id)[0])
            exterior[position] = state
    for boundary in geometry.mass_flow_inlets:
        boundary_states = _mass_flow_inlet_boundary_states(
            primitive[geometry.owners[boundary.faces]],
            geometry.face_normals[boundary.faces],
            geometry.face_lengths[boundary.faces],
            boundary,
            settings.gamma,
            settings.gas_constant,
        )
        for face_id, state in zip(boundary.faces, boundary_states):
            position = int(np.flatnonzero(faces == face_id)[0])
            exterior[position] = state
    for boundary in geometry.mass_flow_outlets:
        boundary_states = _mass_flow_outlet_boundary_states(
            primitive[geometry.owners[boundary.faces]],
            geometry.face_normals[boundary.faces],
            geometry.face_lengths[boundary.faces],
            boundary,
        )
        for face_id, state in zip(boundary.faces, boundary_states):
            position = int(np.flatnonzero(faces == face_id)[0])
            exterior[position] = state
    if len(faces):
        mass_fluxes[faces] = numerical_flux(
            primitive[owners],
            exterior,
            geometry.face_normals[faces],
            settings.gamma,
            settings.flux,
        )[:, 0]
    return mass_fluxes


def _sst_residual(
    primitive: np.ndarray,
    turbulence: np.ndarray,
    geometry: _Geometry,
    settings: NumericalSettings,
) -> np.ndarray:
    (
        turbulent_viscosity,
        turbulence_gradients,
        sigma_k,
        sigma_omega,
        sources,
    ) = _sst_fields(primitive, turbulence, geometry, settings)
    density = primitive[:, 0]
    kinetic_energy = np.maximum(turbulence[:, 0], 0.0)
    omega = np.maximum(turbulence[:, 1], 1e-12)
    temperature = primitive[:, 3] / (density * settings.gas_constant)
    molecular_viscosity = _dynamic_viscosity(temperature, settings)
    diffusion_k = molecular_viscosity + sigma_k * turbulent_viscosity
    diffusion_omega = molecular_viscosity + sigma_omega * turbulent_viscosity
    derivative = np.zeros((len(primitive), 2), dtype=float)
    mass_flux = _sst_mass_fluxes(primitive, geometry, settings)
    faces = geometry.interior_faces
    if len(faces):
        owners, neighbours = geometry.owners[faces], geometry.neighbours[faces]
        flux = np.zeros((len(faces), 2))
        for variable in range(2):
            values = turbulence[:, variable]
            upwind = np.where(
                mass_flux[faces] >= 0.0, values[owners], values[neighbours]
            )
            flux[:, variable] = mass_flux[faces] * upwind
        integrated = flux * geometry.face_lengths[faces, None]
        np.add.at(derivative, owners, -integrated / geometry.areas[owners, None])
        np.add.at(
            derivative, neighbours, integrated / geometry.areas[neighbours, None]
        )
    faces_a, faces_b = geometry.periodic_faces_a, geometry.periodic_faces_b
    if len(faces_a):
        owners_a, owners_b = geometry.owners[faces_a], geometry.owners[faces_b]
        flux = np.zeros((len(faces_a), 2))
        for variable in range(2):
            values = turbulence[:, variable]
            upwind = np.where(
                mass_flux[faces_a] >= 0.0,
                values[owners_a],
                values[owners_b],
            )
            flux[:, variable] = mass_flux[faces_a] * upwind
        integrated = flux * geometry.face_lengths[faces_a, None]
        np.add.at(
            derivative, owners_a, -integrated / geometry.areas[owners_a, None]
        )
        np.add.at(
            derivative, owners_b, integrated / geometry.areas[owners_b, None]
        )

    boundary_values = _turbulence_boundary_values(
        primitive, turbulence, geometry, settings
    )
    wall_values = _sst_wall_values(
        primitive, turbulence, geometry, settings
    )
    boundary_values.update(wall_values)
    faces = geometry.diffusive_boundary_faces
    owners = geometry.owners[faces]
    if len(faces):
        exterior_values = np.asarray(
            [
                boundary_values.get(int(face_id), turbulence[owner])
                for face_id, owner in zip(faces, owners)
            ]
        )
        upwind = np.where(
            mass_flux[faces, None] >= 0.0,
            turbulence[owners],
            exterior_values,
        )
        flux = mass_flux[faces, None] * upwind
        integrated = flux * geometry.face_lengths[faces, None]
        np.add.at(derivative, owners, -integrated / geometry.areas[owners, None])

    gradients = turbulence_gradients
    faces = geometry.interior_faces
    if len(faces):
        owners, neighbours = geometry.owners[faces], geometry.neighbours[faces]
        face_gradient = 0.5 * (gradients[owners] + gradients[neighbours])
        coefficients = np.column_stack(
            (
                0.5 * (diffusion_k[owners] + diffusion_k[neighbours]),
                0.5 * (diffusion_omega[owners] + diffusion_omega[neighbours]),
            )
        )
        flux = coefficients * np.sum(
            face_gradient * geometry.face_normals[faces, None, :], axis=2
        )
        integrated = flux * geometry.face_lengths[faces, None]
        np.add.at(derivative, owners, integrated / geometry.areas[owners, None])
        np.add.at(
            derivative, neighbours, -integrated / geometry.areas[neighbours, None]
        )
    faces_a, faces_b = geometry.periodic_faces_a, geometry.periodic_faces_b
    if len(faces_a):
        owners_a, owners_b = geometry.owners[faces_a], geometry.owners[faces_b]
        face_gradient = 0.5 * (gradients[owners_a] + gradients[owners_b])
        coefficients = np.column_stack(
            (
                0.5 * (diffusion_k[owners_a] + diffusion_k[owners_b]),
                0.5 * (diffusion_omega[owners_a] + diffusion_omega[owners_b]),
            )
        )
        flux = coefficients * np.sum(
            face_gradient * geometry.face_normals[faces_a, None, :], axis=2
        )
        integrated = flux * geometry.face_lengths[faces_a, None]
        np.add.at(
            derivative, owners_a, integrated / geometry.areas[owners_a, None]
        )
        np.add.at(
            derivative, owners_b, -integrated / geometry.areas[owners_b, None]
        )
    for face_id in geometry.diffusive_boundary_faces:
        boundary_value = boundary_values.get(int(face_id))
        if boundary_value is None:
            continue
        owner = int(geometry.owners[face_id])
        normal = geometry.face_normals[face_id]
        distance = abs(
            float(
                np.dot(
                    geometry.face_centers[face_id] - geometry.centers[owner],
                    normal,
                )
            )
        )
        if distance <= 1e-14:
            raise SolverError("Turbulence boundary has invalid cell-center distance.")
        coefficient = np.array(
            (diffusion_k[owner], diffusion_omega[owner])
        )
        flux = coefficient * (
            np.asarray(boundary_value) - turbulence[owner]
        ) / distance
        derivative[owner] += (
            flux * geometry.face_lengths[face_id] / geometry.areas[owner]
        )

    production, beta, gamma, cross_diffusion = sources.T
    beta_star = 0.09
    derivative[:, 0] += (
        production - beta_star * density * omega * kinetic_energy
    )
    omega_production = np.divide(
        gamma * production,
        turbulent_viscosity / density,
        out=np.zeros_like(production),
        where=turbulent_viscosity > 1e-30,
    )
    derivative[:, 1] += (
        omega_production
        - beta * density * omega**2
        + cross_diffusion
    )
    if not np.isfinite(derivative).all():
        raise SolverError("SST residual contains non-finite values.")
    return derivative


def _viscous_gradients(
    primitive: np.ndarray, geometry: _Geometry, settings: NumericalSettings
) -> np.ndarray:
    density = primitive[:, 0]
    temperature = primitive[:, 3] / (density * settings.gas_constant)
    if not np.isfinite(temperature).all() or np.any(temperature <= 0.0):
        raise SolverError("Viscous flow requires finite positive temperature.")
    cell_values = np.column_stack((primitive[:, 1:3], temperature))
    sources = [geometry.gradient_sources]
    neighbours = [geometry.gradient_neighbours]
    dx_values = [geometry.gradient_dx]
    dy_values = [geometry.gradient_dy]
    differences = [cell_values[geometry.gradient_neighbours] - cell_values[geometry.gradient_sources]]

    inlet_values = {
        face_id: np.array(
            (state[1], state[2], state[3] / (state[0] * settings.gas_constant))
        )
        for face_id, state in _viscous_inlet_states(
            primitive, geometry, settings
        ).items()
    }

    wall_by_face = {
        int(face_id): boundary
        for boundary in geometry.wall_thermal_boundaries
        for face_id in boundary.faces
    }
    boundary_sources = []
    boundary_neighbours = []
    boundary_dx = []
    boundary_dy = []
    boundary_differences = []
    for face_id in geometry.diffusive_boundary_faces:
        owner = int(geometry.owners[face_id])
        normal = geometry.face_normals[face_id]
        offset = geometry.face_centers[face_id] - geometry.centers[owner]
        kind = geometry.patch_kinds[int(geometry.patch_ids[face_id])]
        normal_distance = float(np.dot(offset, normal))
        ghost_offset = (
            2.0 * normal_distance * normal
            if kind in ("Wall", "Symmetry")
            else 2.0 * offset
        )
        ghost = cell_values[owner].copy()
        if kind == "Wall":
            wall = wall_by_face[int(face_id)]
            if wall.mechanical_condition == "No-slip":
                ghost[:2] *= -1.0
            else:
                ghost[:2] -= (
                    2.0 * float(np.dot(ghost[:2], normal)) * normal
                )
            if wall.condition == "Temperature":
                ghost[2] = 2.0 * wall.value - cell_values[owner, 2]
            else:
                viscosity = float(_dynamic_viscosity(temperature[owner : owner + 1], settings)[0])
                conductivity = (
                    viscosity
                    * settings.gamma
                    * settings.gas_constant
                    / ((settings.gamma - 1.0) * settings.prandtl_number)
                )
                normal_distance = float(np.dot(offset, normal))
                ghost[2] += (
                    2.0 * normal_distance * wall.value / conductivity
                )
        elif kind == "Symmetry":
            velocity = cell_values[owner, :2]
            ghost[:2] -= 2.0 * float(np.dot(velocity, normal)) * normal
        elif int(face_id) in inlet_values:
            ghost = 2.0 * inlet_values[int(face_id)] - ghost
        boundary_sources.append(owner)
        boundary_neighbours.append(owner)
        boundary_dx.append(ghost_offset[0])
        boundary_dy.append(ghost_offset[1])
        boundary_differences.append(ghost - cell_values[owner])

    if boundary_sources:
        sources.append(np.asarray(boundary_sources, dtype=int))
        neighbours.append(np.asarray(boundary_neighbours, dtype=int))
        dx_values.append(np.asarray(boundary_dx, dtype=float))
        dy_values.append(np.asarray(boundary_dy, dtype=float))
        differences = np.vstack((differences[0], np.asarray(boundary_differences)))
        source_ids = np.concatenate(sources)
        dx = np.concatenate(dx_values)
        dy = np.concatenate(dy_values)
    else:
        source_ids = sources[0]
        dx = dx_values[0]
        dy = dy_values[0]
        differences = differences[0]

    gradients = np.zeros((len(primitive), 3, 2), dtype=float)
    if len(source_ids):
        weights = 1.0 / np.maximum(dx * dx + dy * dy, 1e-300)
        xx = np.zeros(len(primitive))
        xy = np.zeros(len(primitive))
        yy = np.zeros(len(primitive))
        rhs_x = np.zeros((len(primitive), 3))
        rhs_y = np.zeros((len(primitive), 3))
        np.add.at(xx, source_ids, weights * dx * dx)
        np.add.at(xy, source_ids, weights * dx * dy)
        np.add.at(yy, source_ids, weights * dy * dy)
        np.add.at(rhs_x, source_ids, (weights * dx)[:, None] * differences)
        np.add.at(rhs_y, source_ids, (weights * dy)[:, None] * differences)
        determinant = xx * yy - xy * xy
        valid = determinant > 1e-14 * np.maximum(xx * yy, 1e-300)
        gradients[valid, :, 0] = (
            rhs_x[valid] * yy[valid, None] - rhs_y[valid] * xy[valid, None]
        ) / determinant[valid, None]
        gradients[valid, :, 1] = (
            rhs_y[valid] * xx[valid, None] - rhs_x[valid] * xy[valid, None]
        ) / determinant[valid, None]
    return gradients


def _viscous_flux(
    gradients: np.ndarray,
    velocities: np.ndarray,
    viscosity: np.ndarray,
    conductivity: np.ndarray,
    normals: np.ndarray,
    turbulent_rho_k: np.ndarray | None = None,
) -> np.ndarray:
    du_dx, du_dy = gradients[:, 0, 0], gradients[:, 0, 1]
    dv_dx, dv_dy = gradients[:, 1, 0], gradients[:, 1, 1]
    divergence = du_dx + dv_dy
    tau_xx = viscosity * (2.0 * du_dx - (2.0 / 3.0) * divergence)
    tau_yy = viscosity * (2.0 * dv_dy - (2.0 / 3.0) * divergence)
    tau_xy = viscosity * (du_dy + dv_dx)
    if turbulent_rho_k is not None:
        isotropic_stress = (2.0 / 3.0) * turbulent_rho_k
        tau_xx -= isotropic_stress
        tau_yy -= isotropic_stress
    traction_x = tau_xx * normals[:, 0] + tau_xy * normals[:, 1]
    traction_y = tau_xy * normals[:, 0] + tau_yy * normals[:, 1]
    heat_flux = conductivity * np.sum(gradients[:, 2, :] * normals, axis=1)
    return np.column_stack(
        (
            np.zeros(len(normals)),
            traction_x,
            traction_y,
            velocities[:, 0] * traction_x
            + velocities[:, 1] * traction_y
            + heat_flux,
        )
    )


def _viscous_residual(
    primitive: np.ndarray,
    geometry: _Geometry,
    settings: NumericalSettings,
    turbulence: np.ndarray | None = None,
) -> np.ndarray:
    temperature = primitive[:, 3] / (
        primitive[:, 0] * settings.gas_constant
    )
    cell_values = np.column_stack((primitive[:, 1:3], temperature))
    viscosity = _dynamic_viscosity(temperature, settings)
    conductivity = (
        viscosity
        * settings.gamma
        * settings.gas_constant
        / ((settings.gamma - 1.0) * settings.prandtl_number)
    )
    gradients = _viscous_gradients(primitive, geometry, settings)
    turbulent_viscosity = np.zeros(len(primitive))
    if turbulence is not None:
        (
            turbulent_viscosity,
            turbulence_gradients,
            sigma_k,
            _sigma_omega,
            _sst_sources,
        ) = _sst_fields(primitive, turbulence, geometry, settings)
        conductivity += (
            turbulent_viscosity
            * settings.gamma
            * settings.gas_constant
            / ((settings.gamma - 1.0) * 0.9)
        )
    derivative = np.zeros((len(primitive), 4), dtype=float)

    def add_face_flux(
        faces: np.ndarray,
        owners: np.ndarray,
        neighbours: np.ndarray,
        normal: np.ndarray,
        face_gradients: np.ndarray,
        face_velocity: np.ndarray,
        face_viscosity: np.ndarray,
        face_conductivity: np.ndarray,
        face_rho_k: np.ndarray | None = None,
        turbulent_energy_flux: np.ndarray | None = None,
    ) -> None:
        flux = _viscous_flux(
            face_gradients,
            face_velocity,
            face_viscosity,
            face_conductivity,
            normal,
            face_rho_k,
        )
        if turbulent_energy_flux is not None:
            flux[:, 3] += turbulent_energy_flux
        integrated = flux * geometry.face_lengths[faces, None]
        np.add.at(derivative, owners, integrated / geometry.areas[owners, None])
        np.add.at(
            derivative, neighbours, -integrated / geometry.areas[neighbours, None]
        )

    faces = geometry.interior_faces
    if len(faces):
        owners, neighbours = geometry.owners[faces], geometry.neighbours[faces]
        face_gradients = 0.5 * (gradients[owners] + gradients[neighbours])
        face_velocity = 0.5 * (primitive[owners, 1:3] + primitive[neighbours, 1:3])
        face_viscosity = 0.5 * (viscosity[owners] + viscosity[neighbours])
        face_rho_k = None
        turbulent_energy_flux = None
        if turbulence is not None:
            face_viscosity += 0.5 * (
                turbulent_viscosity[owners] + turbulent_viscosity[neighbours]
            )
            face_rho_k = 0.5 * (
                primitive[owners, 0] * turbulence[owners, 0]
                + primitive[neighbours, 0] * turbulence[neighbours, 0]
            )
            face_k_gradient = 0.5 * (
                turbulence_gradients[owners, 0]
                + turbulence_gradients[neighbours, 0]
            )
            face_k_diffusion = 0.5 * (
                viscosity[owners]
                + turbulent_viscosity[owners] / sigma_k[owners]
                + viscosity[neighbours]
                + turbulent_viscosity[neighbours] / sigma_k[neighbours]
            )
            turbulent_energy_flux = face_k_diffusion * np.sum(
                face_k_gradient * geometry.face_normals[faces], axis=1
            )
        face_conductivity = 0.5 * (
            conductivity[owners] + conductivity[neighbours]
        )
        add_face_flux(
            faces,
            owners,
            neighbours,
            geometry.face_normals[faces],
            face_gradients,
            face_velocity,
            face_viscosity,
            face_conductivity,
            face_rho_k,
            turbulent_energy_flux,
        )
    faces_a, faces_b = geometry.periodic_faces_a, geometry.periodic_faces_b
    if len(faces_a):
        owners_a, owners_b = geometry.owners[faces_a], geometry.owners[faces_b]
        face_gradients = 0.5 * (gradients[owners_a] + gradients[owners_b])
        face_velocity = 0.5 * (
            primitive[owners_a, 1:3] + primitive[owners_b, 1:3]
        )
        face_viscosity = 0.5 * (viscosity[owners_a] + viscosity[owners_b])
        face_rho_k = None
        turbulent_energy_flux = None
        if turbulence is not None:
            face_viscosity += 0.5 * (
                turbulent_viscosity[owners_a] + turbulent_viscosity[owners_b]
            )
            face_rho_k = 0.5 * (
                primitive[owners_a, 0] * turbulence[owners_a, 0]
                + primitive[owners_b, 0] * turbulence[owners_b, 0]
            )
            face_k_gradient = 0.5 * (
                turbulence_gradients[owners_a, 0]
                + turbulence_gradients[owners_b, 0]
            )
            face_k_diffusion = 0.5 * (
                viscosity[owners_a]
                + turbulent_viscosity[owners_a] / sigma_k[owners_a]
                + viscosity[owners_b]
                + turbulent_viscosity[owners_b] / sigma_k[owners_b]
            )
            turbulent_energy_flux = face_k_diffusion * np.sum(
                face_k_gradient * geometry.face_normals[faces_a], axis=1
            )
        face_conductivity = 0.5 * (
            conductivity[owners_a] + conductivity[owners_b]
        )
        flux = _viscous_flux(
            face_gradients,
            face_velocity,
            face_viscosity,
            face_conductivity,
            geometry.face_normals[faces_a],
            face_rho_k,
        )
        if turbulent_energy_flux is not None:
            flux[:, 3] += turbulent_energy_flux
        integrated = flux * geometry.face_lengths[faces_a, None]
        np.add.at(
            derivative, owners_a, integrated / geometry.areas[owners_a, None]
        )
        np.add.at(
            derivative, owners_b, -integrated / geometry.areas[owners_b, None]
        )

    wall_by_face = {
        int(face_id): boundary
        for boundary in geometry.wall_thermal_boundaries
        for face_id in boundary.faces
    }
    inlet_states = _viscous_inlet_states(primitive, geometry, settings)
    wall_sst_values = (
        _sst_wall_values(primitive, turbulence, geometry, settings)
        if turbulence is not None
        else {}
    )
    turbulence_boundary_values = (
        _turbulence_boundary_values(
            primitive, turbulence, geometry, settings
        )
        if turbulence is not None
        else {}
    )
    turbulence_boundary_values.update(wall_sst_values)
    for face_id in geometry.diffusive_boundary_faces:
        owner = int(geometry.owners[face_id])
        normal = geometry.face_normals[face_id]
        offset = geometry.face_centers[face_id] - geometry.centers[owner]
        normal_distance = float(np.dot(offset, normal))
        if normal_distance <= 1e-14:
            raise SolverError("Boundary face has invalid cell-center distance.")
        kind = geometry.patch_kinds[int(geometry.patch_ids[face_id])]
        face_gradient = gradients[owner].copy()
        boundary_values = np.array(
            (primitive[owner, 1], primitive[owner, 2], temperature[owner])
        )
        if kind == "Wall":
            boundary = wall_by_face[int(face_id)]
            if boundary.mechanical_condition == "No-slip":
                for component in (0, 1):
                    normal_derivative = (
                        -boundary_values[component]
                        - float(
                            np.dot(
                                face_gradient[component],
                                offset - normal_distance * normal,
                            )
                        )
                    ) / normal_distance
                    face_gradient[component] += (
                        normal_derivative
                        - float(np.dot(face_gradient[component], normal))
                    ) * normal
                    boundary_values[component] = 0.0
            else:
                tangential_offset = offset - normal_distance * normal
                normal_velocity = float(np.dot(boundary_values[:2], normal))
                boundary_values[:2] += face_gradient[:2] @ offset
                boundary_values[:2] -= (
                    float(np.dot(boundary_values[:2], normal)) * normal
                )
                gradient_normal_velocity = (
                    normal[0] * face_gradient[0]
                    + normal[1] * face_gradient[1]
                )
                normal_derivative = -(
                    normal_velocity
                    + float(np.dot(gradient_normal_velocity, tangential_offset))
                ) / normal_distance
                desired_gradient = np.outer(
                    normal_derivative * normal, normal
                )
                velocity_gradient = face_gradient[:2]
                velocity_gradient += desired_gradient - (
                    velocity_gradient @ normal
                )[:, None] * normal
            if boundary.condition == "Temperature":
                normal_derivative = (
                    boundary.value - boundary_values[2]
                    - float(np.dot(face_gradient[2], offset - normal_distance * normal))
                ) / normal_distance
                boundary_values[2] = boundary.value
            else:
                normal_derivative = boundary.value / conductivity[owner]
                boundary_values[2] += float(np.dot(face_gradient[2], offset))
            face_gradient[2] += (
                normal_derivative - float(np.dot(face_gradient[2], normal))
            ) * normal
        elif kind == "Symmetry":
            normal_velocity = float(np.dot(boundary_values[:2], normal))
            derivative_velocity = -normal_velocity / normal_distance * normal
            for component in (0, 1):
                face_gradient[component] += (
                    derivative_velocity[component]
                    - float(np.dot(face_gradient[component], normal))
                ) * normal
            face_gradient[2] -= float(np.dot(face_gradient[2], normal)) * normal
            boundary_values[:2] -= normal_velocity * normal
        else:
            state = inlet_states.get(int(face_id))
            if state is not None:
                boundary_values = np.array(
                    (
                        state[1],
                        state[2],
                        state[3] / (state[0] * settings.gas_constant),
                    )
                )
                for component in range(3):
                    normal_derivative = (
                        boundary_values[component]
                        - cell_values[owner, component]
                        - float(
                            np.dot(
                                face_gradient[component],
                                offset - normal_distance * normal,
                            )
                        )
                    ) / normal_distance
                    face_gradient[component] += (
                        normal_derivative
                        - float(np.dot(face_gradient[component], normal))
                    ) * normal
            else:
                face_gradient -= np.outer(face_gradient @ normal, normal)
                boundary_values += face_gradient @ offset

        face_temperature = boundary_values[2]
        face_mu = float(_dynamic_viscosity(np.array([face_temperature]), settings)[0])
        face_k = (
            face_mu
            * settings.gamma
            * settings.gas_constant
            / ((settings.gamma - 1.0) * settings.prandtl_number)
        )
        face_rho_k = None
        if turbulence is not None:
            is_wall = kind == "Wall"
            is_wall_function = (
                is_wall
                and settings.turbulence_wall_treatment == "Wall functions"
                and wall_by_face[int(face_id)].mechanical_condition == "No-slip"
            )
            if not is_wall:
                face_mu += turbulent_viscosity[owner]
                face_k += (
                    turbulent_viscosity[owner]
                    * settings.gamma
                    * settings.gas_constant
                    / ((settings.gamma - 1.0) * 0.9)
                )
                face_rho_k = primitive[owner, 0] * turbulence[owner, 0]
            elif is_wall_function:
                face_rho_k = (
                    primitive[owner, 0]
                    * wall_sst_values[int(face_id)][0]
                )
            elif wall_by_face[int(face_id)].mechanical_condition == "Slip":
                face_mu += turbulent_viscosity[owner]
                face_rho_k = primitive[owner, 0] * turbulence[owner, 0]
        flux = _viscous_flux(
            face_gradient[None, :, :],
            boundary_values[None, :2],
            np.array([face_mu]),
            np.array([face_k]),
            normal[None, :],
            None if face_rho_k is None else np.array([face_rho_k]),
        )[0]
        if turbulence is not None:
            wall_value = turbulence_boundary_values.get(int(face_id))
            if wall_value is not None:
                energy_k_diffusion = (
                    viscosity[owner]
                    + turbulent_viscosity[owner] / sigma_k[owner]
                ) * (wall_value[0] - turbulence[owner, 0]) / normal_distance
                flux[3] += energy_k_diffusion
        if turbulence is not None and kind == "Wall":
            wall = wall_by_face[int(face_id)]
            if (
                settings.turbulence_wall_treatment == "Wall functions"
                and wall.mechanical_condition == "No-slip"
            ):
                tangential_velocity = primitive[owner, 1:3] - (
                    float(np.dot(primitive[owner, 1:3], normal)) * normal
                )
                friction_velocity = _wall_friction_velocity(
                    float(np.linalg.norm(tangential_velocity)),
                    float(primitive[owner, 0]),
                    float(viscosity[owner]),
                    normal_distance,
                )
                tangent_speed = float(np.linalg.norm(tangential_velocity))
                if tangent_speed > 0.0:
                    flux[1:3] = (
                        -primitive[owner, 0]
                        * friction_velocity**2
                        * tangential_velocity
                        / tangent_speed
                    )
        if kind == "Wall" and wall_by_face[int(face_id)].condition == "Heat Flux":
            flux[3] = wall_by_face[int(face_id)].value
        derivative[owner] += (
            flux * geometry.face_lengths[face_id] / geometry.areas[owner]
        )
    return derivative


def _stable_time_step(
    primitive: np.ndarray,
    geometry: _Geometry,
    settings: NumericalSettings,
    turbulence_conservative: np.ndarray | None = None,
) -> float:
    return float(
        np.min(
            _stable_time_steps(
                primitive, geometry, settings, turbulence_conservative
            )
        )
    )


def _stable_time_steps(
    primitive: np.ndarray,
    geometry: _Geometry,
    settings: NumericalSettings,
    turbulence_conservative: np.ndarray | None = None,
) -> np.ndarray:
    gamma = settings.gamma
    normal_velocity = np.sum(
        primitive[geometry.owners, 1:3] * geometry.face_normals, axis=1
    )
    sound_speed = _sound_speed(primitive, gamma)
    spectral_radius = np.abs(normal_velocity) + sound_speed[geometry.owners]
    boundary_faces = np.concatenate(
        (
            geometry.velocity_inlet_faces,
            geometry.pressure_outlet_faces,
            geometry.far_field_faces,
        )
    )
    existing_boundary_count = len(boundary_faces)
    additional_faces = []
    additional_states = []
    for boundary in geometry.pressure_inlets:
        faces = boundary.faces
        owners = geometry.owners[faces]
        additional_faces.append(faces)
        additional_states.append(
            _pressure_inlet_boundary_states(
                primitive[owners],
                geometry.face_normals[faces],
                boundary,
                gamma,
                settings.gas_constant,
            )
        )
    for boundary in geometry.mass_flow_inlets:
        faces = boundary.faces
        owners = geometry.owners[faces]
        additional_faces.append(faces)
        additional_states.append(
            _mass_flow_inlet_boundary_states(
                primitive[owners],
                geometry.face_normals[faces],
                geometry.face_lengths[faces],
                boundary,
                gamma,
                settings.gas_constant,
            )
        )
    for boundary in geometry.mass_flow_outlets:
        faces = boundary.faces
        owners = geometry.owners[faces]
        additional_faces.append(faces)
        additional_states.append(
            _mass_flow_outlet_boundary_states(
                primitive[owners],
                geometry.face_normals[faces],
                geometry.face_lengths[faces],
                boundary,
            )
        )
    if additional_faces:
        boundary_faces = np.concatenate((boundary_faces, *additional_faces))
    if len(boundary_faces):
        owners = geometry.owners[boundary_faces]
        exterior = primitive[owners].copy()
        inlet_count = len(geometry.velocity_inlet_faces)
        outlet_end = inlet_count + len(geometry.pressure_outlet_faces)
        if inlet_count:
            exterior[:inlet_count] = geometry.velocity_inlet_states
        if len(geometry.pressure_outlet_faces):
            exterior[inlet_count:outlet_end, 3] = (
                geometry.pressure_outlet_pressures
            )
        far_field_start = outlet_end
        if len(geometry.far_field_faces):
            far_field_end = far_field_start + len(geometry.far_field_faces)
            exterior[far_field_start:far_field_end] = _far_field_boundary_states(
                primitive[owners[far_field_start:far_field_end]],
                geometry.far_field_states,
                geometry.face_normals[boundary_faces[far_field_start:far_field_end]],
                gamma,
            )
        if additional_states:
            exterior[existing_boundary_count:] = np.vstack(additional_states)
        exterior_normal_velocity = np.sum(
            exterior[:, 1:3] * geometry.face_normals[boundary_faces], axis=1
        )
        exterior_radius = np.abs(exterior_normal_velocity) + _sound_speed(
            exterior, gamma
        )
        spectral_radius[boundary_faces] = np.maximum(
            spectral_radius[boundary_faces], exterior_radius
        )
    spectral_sum = np.zeros(len(primitive), dtype=float)
    np.add.at(
        spectral_sum,
        geometry.owners,
        spectral_radius * geometry.face_lengths,
    )
    internal = geometry.interior_faces
    if len(internal):
        neighbours = geometry.neighbours[internal]
        neighbour_normal_velocity = np.sum(
            primitive[neighbours, 1:3] * geometry.face_normals[internal], axis=1
        )
        neighbour_radius = (
            np.abs(neighbour_normal_velocity) + sound_speed[neighbours]
        )
        np.add.at(
            spectral_sum,
            neighbours,
            neighbour_radius * geometry.face_lengths[internal],
        )
    if settings.viscosity_model != "Inviscid":
        temperature = primitive[:, 3] / (
            primitive[:, 0] * settings.gas_constant
        )
        viscosity = _dynamic_viscosity(temperature, settings)
        if turbulence_conservative is not None:
            turbulence = turbulence_conservative / primitive[:, :1]
            turbulent_viscosity = _sst_fields(
                primitive, turbulence, geometry, settings
            )[0]
            viscosity = viscosity + turbulent_viscosity
        effective_diffusivity = np.maximum(
            (4.0 / 3.0) * viscosity / primitive[:, 0],
            viscosity / (primitive[:, 0] * min(settings.prandtl_number, 0.9)),
        )
        diffusion_sum = np.zeros(len(primitive), dtype=float)
        if len(internal):
            owners = geometry.owners[internal]
            neighbours = geometry.neighbours[internal]
            distances = np.abs(
                np.sum(
                    (geometry.centers[neighbours] - geometry.centers[owners])
                    * geometry.face_normals[internal],
                    axis=1,
                )
            )
            face_diffusivity = np.maximum(
                effective_diffusivity[owners], effective_diffusivity[neighbours]
            )
            contribution = (
                face_diffusivity
                * geometry.face_lengths[internal]
                / np.maximum(distances, 1e-14)
            )
            np.add.at(diffusion_sum, owners, contribution)
            np.add.at(diffusion_sum, neighbours, contribution)
        periodic_a, periodic_b = (
            geometry.periodic_faces_a,
            geometry.periodic_faces_b,
        )
        if len(periodic_a):
            owners_a = geometry.owners[periodic_a]
            owners_b = geometry.owners[periodic_b]
            normal_a = geometry.face_normals[periodic_a]
            distance = np.abs(
                np.sum(
                    (geometry.face_centers[periodic_a] - geometry.centers[owners_a])
                    * normal_a,
                    axis=1,
                )
            ) + np.abs(
                np.sum(
                    (
                        geometry.face_centers[periodic_b]
                        - geometry.centers[owners_b]
                    )
                    * geometry.face_normals[periodic_b],
                    axis=1,
                )
            )
            face_diffusivity = np.maximum(
                effective_diffusivity[owners_a],
                effective_diffusivity[owners_b],
            )
            contribution = (
                face_diffusivity
                * geometry.face_lengths[periodic_a]
                / np.maximum(distance, 1e-14)
            )
            np.add.at(diffusion_sum, owners_a, contribution)
            np.add.at(diffusion_sum, owners_b, contribution)
        for face_id in geometry.diffusive_boundary_faces:
            owner = geometry.owners[face_id]
            distance = abs(
                float(
                    np.dot(
                        geometry.face_centers[face_id] - geometry.centers[owner],
                        geometry.face_normals[face_id],
                    )
                )
            )
            diffusion_sum[owner] += (
                effective_diffusivity[owner]
                * geometry.face_lengths[face_id]
                / max(distance, 1e-14)
            )
        spectral_sum += 2.0 * diffusion_sum
        if turbulence_conservative is not None:
            turbulence = turbulence_conservative / primitive[:, :1]
            spectral_sum += (
                2.0 * geometry.areas
                * 0.075
                * turbulence[:, 1]
            )
    time_steps = settings.cfl * geometry.areas / spectral_sum
    if not np.isfinite(time_steps).all() or np.any(time_steps <= 0.0):
        raise SolverError("Local stable time steps must be positive and finite.")
    return time_steps


def _sound_speed(primitive: np.ndarray, gamma: float) -> np.ndarray:
    return np.sqrt(gamma * primitive[:, 3] / primitive[:, 0])


def _far_field_boundary_states(
    interior: np.ndarray,
    far_field: np.ndarray,
    normals: np.ndarray,
    gamma: float,
) -> np.ndarray:
    normal_interior = np.sum(interior[:, 1:3] * normals, axis=1)
    normal_far_field = np.sum(far_field[:, 1:3] * normals, axis=1)
    sound_interior = _sound_speed(interior, gamma)
    sound_far_field = _sound_speed(far_field, gamma)
    exterior = interior.copy()

    supersonic_inflow = normal_interior <= -sound_interior
    supersonic_outflow = normal_interior >= sound_interior
    exterior[supersonic_inflow] = far_field[supersonic_inflow]
    subsonic = ~(supersonic_inflow | supersonic_outflow)
    if not np.any(subsonic):
        return exterior

    normal = normals[subsonic]
    inside = interior[subsonic]
    outside = far_field[subsonic]
    interior_velocity = normal_interior[subsonic]
    far_field_velocity = normal_far_field[subsonic]
    interior_sound = sound_interior[subsonic]
    far_field_sound = sound_far_field[subsonic]
    outgoing_invariant = interior_velocity + 2.0 * interior_sound / (gamma - 1.0)
    incoming_invariant = far_field_velocity - 2.0 * far_field_sound / (gamma - 1.0)
    boundary_velocity = 0.5 * (outgoing_invariant + incoming_invariant)
    boundary_sound = 0.25 * (gamma - 1.0) * (
        outgoing_invariant - incoming_invariant
    )
    if not np.isfinite(boundary_sound).all() or np.any(boundary_sound <= 0.0):
        raise SolverError("Far-field characteristics produced invalid sound speed.")

    inflow = boundary_velocity < 0.0
    thermodynamic_source = np.where(inflow[:, None], outside, inside)
    entropy = thermodynamic_source[:, 3] / thermodynamic_source[:, 0] ** gamma
    density = (
        boundary_sound**2 / (gamma * entropy)
    ) ** (1.0 / (gamma - 1.0))
    pressure = entropy * density**gamma
    tangent = np.column_stack((-normal[:, 1], normal[:, 0]))
    tangential_source = np.where(inflow[:, None], outside[:, 1:3], inside[:, 1:3])
    tangential_velocity = np.sum(tangential_source * tangent, axis=1)
    velocity = (
        boundary_velocity[:, None] * normal
        + tangential_velocity[:, None] * tangent
    )
    subsonic_states = np.column_stack(
        (density, velocity[:, 0], velocity[:, 1], pressure)
    )
    if (
        not np.isfinite(subsonic_states).all()
        or np.any(subsonic_states[:, 0] <= 0.0)
        or np.any(subsonic_states[:, 3] <= 0.0)
    ):
        raise SolverError("Far-field characteristics produced an invalid state.")
    exterior[subsonic] = subsonic_states
    return exterior


def _pressure_inlet_boundary_states(
    interior: np.ndarray,
    normals: np.ndarray,
    boundary: _PressureInlet,
    gamma: float,
    gas_constant: float,
) -> np.ndarray:
    if not math.isclose(float(np.linalg.norm(boundary.direction)), 1.0):
        raise SolverError(
            f"Pressure inlet '{boundary.patch_name}' has an invalid flow direction."
        )
    direction_normal = normals @ boundary.direction
    if np.any(direction_normal >= -1e-12):
        raise SolverError(
            f"Pressure inlet '{boundary.patch_name}' flow direction must point "
            "into the domain across every face."
        )

    sound_interior = _sound_speed(interior, gamma)
    normal_velocity = np.sum(interior[:, 1:3] * normals, axis=1)
    mach_interior = normal_velocity / sound_interior
    if np.any(mach_interior >= 1.0):
        raise SolverError(
            f"Pressure inlet '{boundary.patch_name}' cannot prescribe a "
            "supersonic outflow."
        )
    total_sound_speed = math.sqrt(
        gamma * gas_constant * boundary.total_temperature
    )
    mach = np.empty(len(interior), dtype=float)
    for face_id, (mach_inside, sound_inside, normal_direction) in enumerate(
        zip(mach_interior, sound_interior, direction_normal)
    ):
        if mach_inside <= -1.0:
            raise SolverError(
                f"Pressure inlet '{boundary.patch_name}' needs Mach number "
                "to define a supersonic inflow."
            )
        outgoing_invariant = (
            normal_velocity[face_id] + 2.0 * sound_inside / (gamma - 1.0)
        )

        def invariant_at(
            candidate: float, face_normal_direction: float
        ) -> float:
            temperature_ratio = 1.0 + 0.5 * (gamma - 1.0) * candidate**2
            sound = total_sound_speed / math.sqrt(temperature_ratio)
            return sound * (
                face_normal_direction * candidate + 2.0 / (gamma - 1.0)
            )

        low_value = invariant_at(0.0, normal_direction)
        high_value = invariant_at(1.0, normal_direction)
        if not high_value <= outgoing_invariant <= low_value:
            raise SolverError(
                f"Pressure inlet '{boundary.patch_name}' total conditions are "
                "incompatible with the interior characteristic state."
            )
        low, high = 0.0, 1.0
        for _ in range(64):
            middle = 0.5 * (low + high)
            if invariant_at(middle, normal_direction) > outgoing_invariant:
                low = middle
            else:
                high = middle
        mach[face_id] = 0.5 * (low + high)

    temperature = boundary.total_temperature / (
        1.0 + 0.5 * (gamma - 1.0) * mach**2
    )
    pressure = boundary.total_pressure * (
        temperature / boundary.total_temperature
    ) ** (gamma / (gamma - 1.0))
    density = pressure / (gas_constant * temperature)
    velocity = (
        mach[:, None]
        * np.sqrt(gamma * gas_constant * temperature)[:, None]
        * boundary.direction[None, :]
    )
    states = np.column_stack((density, velocity, pressure))
    if not np.isfinite(states).all() or np.any(states[:, (0, 3)] <= 0.0):
        raise SolverError(
            f"Pressure inlet '{boundary.patch_name}' produced an invalid state."
        )
    return states


def _mass_flow_inlet_boundary_states(
    interior: np.ndarray,
    normals: np.ndarray,
    face_lengths: np.ndarray,
    boundary: _MassFlowInlet,
    gamma: float,
    gas_constant: float,
) -> np.ndarray:
    inward_projection = -(normals @ boundary.direction)
    if np.any(inward_projection <= 1e-12):
        raise SolverError(
            f"Mass-flow inlet '{boundary.patch_name}' direction must point "
            "into the domain across every face."
        )
    if np.any(interior[:, 3] <= 0.0) or not np.isfinite(interior[:, 3]).all():
        raise SolverError(
            f"Mass-flow inlet '{boundary.patch_name}' has invalid interior pressure."
        )
    heat_capacity = gamma * gas_constant / (gamma - 1.0)
    maximum_speed = math.sqrt(
        2.0 * heat_capacity * boundary.total_temperature
    ) * (1.0 - 1e-12)
    pressure = interior[:, 3]
    def flow_at(speed: float) -> float:
        temperature = boundary.total_temperature - speed**2 / (
            2.0 * heat_capacity
        )
        density = pressure / (gas_constant * temperature)
        return float(
            np.sum(
                density
                * speed
                * inward_projection
                * face_lengths
            )
        )

    low, high = 0.0, maximum_speed
    if flow_at(high) < boundary.mass_flow:
        raise SolverError(
            f"Mass-flow inlet '{boundary.patch_name}' cannot satisfy the "
            "configured flow rate."
        )
    for _ in range(80):
        middle = 0.5 * (low + high)
        if flow_at(middle) < boundary.mass_flow:
            low = middle
        else:
            high = middle
    speed = 0.5 * (low + high)
    temperature = boundary.total_temperature - speed**2 / (
        2.0 * heat_capacity
    )
    density = pressure / (gas_constant * temperature)
    velocity = speed * boundary.direction[None, :]
    return np.column_stack((density, velocity, pressure))


def _mass_flow_outlet_boundary_states(
    interior: np.ndarray,
    normals: np.ndarray,
    face_lengths: np.ndarray,
    boundary: _MassFlowOutlet,
) -> np.ndarray:
    if np.any(interior[:, 0] <= 0.0) or not np.isfinite(interior[:, 0]).all():
        raise SolverError(
            f"Mass-flow outlet '{boundary.patch_name}' has invalid interior density."
        )
    states = interior.copy()
    normal_velocity = boundary.mass_flow / (
        float(np.sum(face_lengths)) * interior[:, 0]
    )
    tangent = np.column_stack((-normals[:, 1], normals[:, 0]))
    tangential_velocity = np.sum(interior[:, 1:3] * tangent, axis=1)
    states[:, 1:3] = (
        normal_velocity[:, None] * normals
        + tangential_velocity[:, None] * tangent
    )
    return states


def _finite_boundary_value(value: float, description: str) -> float:
    if isinstance(value, bool):
        raise SolverError(f"{description} must be finite.")
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise SolverError(f"{description} must be finite.") from error
    if not math.isfinite(number):
        raise SolverError(f"{description} must be finite.")
    return number


def _positive_boundary_value(value: float, description: str) -> float:
    number = _finite_boundary_value(value, description)
    if number <= 0.0:
        raise SolverError(f"{description} must be positive.")
    return number


def _boundary_flow_direction(
    x: float, y: float, description: str
) -> np.ndarray:
    direction = np.array(
        (
            _finite_boundary_value(x, f"{description} flow direction X"),
            _finite_boundary_value(y, f"{description} flow direction Y"),
        ),
        dtype=float,
    )
    magnitude = float(np.linalg.norm(direction))
    if not math.isfinite(magnitude) or magnitude <= 1e-12:
        raise SolverError(f"{description} needs a non-zero flow direction.")
    return direction / magnitude
