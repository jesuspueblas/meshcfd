"""Pure-Python case and mesh data models."""

from meshcfd.core.model import (
    BOUNDARY_TYPES,
    FLUX_SCHEMES,
    LIMITER_SCHEMES,
    RECONSTRUCTION_SCHEMES,
    TIME_INTEGRATION_SCHEMES,
    BoundaryCondition,
    Case,
    FiniteVolumeMesh,
    MeshPatch,
    NumericalSettings,
    PeriodicBoundaryPair,
)
from meshcfd.core.periodic import detect_periodic_pair
from meshcfd.core.solver import SolverError, SolverProgress, SolverResult, solve_case

__all__ = [
    "BOUNDARY_TYPES",
    "BoundaryCondition",
    "Case",
    "FLUX_SCHEMES",
    "FiniteVolumeMesh",
    "LIMITER_SCHEMES",
    "MeshPatch",
    "NumericalSettings",
    "PeriodicBoundaryPair",
    "RECONSTRUCTION_SCHEMES",
    "SolverError",
    "SolverProgress",
    "SolverResult",
    "TIME_INTEGRATION_SCHEMES",
    "detect_periodic_pair",
    "solve_case",
]
