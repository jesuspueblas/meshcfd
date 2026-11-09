"""Helpers shared by the MeshCFD performance benchmarks."""

from __future__ import annotations

import math
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from meshcfd.core.euler import primitive_to_conservative
from meshcfd.core.model import (
    Case,
    FiniteVolumeMesh,
    InitialState,
    MeshPatch,
    PeriodicBoundaryPair,
)


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    nx: int
    ny: int
    reconstruction: str
    time_integration: str
    viscosity_model: str = "Inviscid"
    turbulence_model: str = "None"

    @property
    def cells(self) -> int:
        return self.nx * self.ny


def _boundary_patch(points: tuple[tuple[float, float], ...], start: int, end: int) -> int:
    x0, y0 = points[start]
    x1, y1 = points[end]
    if y0 == y1 == 0.0:
        return 0
    if x0 == x1 == 1.0:
        return 1
    if y0 == y1 == 1.0:
        return 2
    if x0 == x1 == 0.0:
        return 3
    return -1


def make_periodic_mesh(nx: int, ny: int) -> FiniteVolumeMesh:
    """Build a deterministic unit-square quadrilateral mesh without GUI dependencies."""
    points = tuple((i / nx, j / ny) for j in range(ny + 1) for i in range(nx + 1))
    cells: list[tuple[int, int, int, int]] = []
    faces: list[tuple[int, int]] = []
    owners: list[int] = []
    neighbours: list[int] = []
    patch_ids: list[int] = []
    cell_face_ids: list[tuple[int, ...]] = []
    cell_face_signs: list[tuple[int, ...]] = []
    edge_to_face: dict[tuple[int, int], int] = {}
    patch_face_ids: dict[int, list[int]] = {patch_id: [] for patch_id in range(4)}

    for j in range(ny):
        for i in range(nx):
            bottom_left = j * (nx + 1) + i
            bottom_right = bottom_left + 1
            top_right = bottom_right + nx + 1
            top_left = bottom_left + nx + 1
            vertices = (bottom_left, bottom_right, top_right, top_left)
            cell_id = len(cells)
            cells.append(vertices)
            local_faces: list[int] = []
            local_signs: list[int] = []
            for start, end in zip(vertices, vertices[1:] + vertices[:1]):
                edge = (min(start, end), max(start, end))
                if edge in edge_to_face:
                    face_id = edge_to_face[edge]
                    neighbours[face_id] = cell_id
                    local_faces.append(face_id)
                    local_signs.append(-1)
                else:
                    face_id = len(faces)
                    edge_to_face[edge] = face_id
                    faces.append((start, end))
                    owners.append(cell_id)
                    neighbours.append(-1)
                    patch_id = _boundary_patch(points, start, end)
                    patch_ids.append(patch_id)
                    if patch_id >= 0:
                        patch_face_ids[patch_id].append(face_id)
                    local_faces.append(face_id)
                    local_signs.append(1)
            cell_face_ids.append(tuple(local_faces))
            cell_face_signs.append(tuple(local_signs))

    patches = tuple(
        MeshPatch(patch_id, name, tuple(patch_face_ids[patch_id]))
        for patch_id, name in enumerate(("bottom", "right", "top", "left"))
    )
    return FiniteVolumeMesh(
        points=points,
        cells=tuple(cells),
        faces=tuple(faces),
        face_owners=tuple(owners),
        face_neighbours=tuple(neighbours),
        face_patch_ids=tuple(patch_ids),
        patches=patches,
        cell_face_ids=tuple(cell_face_ids),
        cell_face_signs=tuple(cell_face_signs),
    )


def make_case(spec: BenchmarkSpec, iterations: int) -> tuple[Case, tuple[InitialState, ...]]:
    mesh = make_periodic_mesh(spec.nx, spec.ny)
    case = Case(name=spec.name, mesh=mesh)
    for condition in case.boundary_conditions:
        condition.kind = "Periodic"
    case.periodic_pairs = [
        PeriodicBoundaryPair(1, 3, "x", 1.0),
        PeriodicBoundaryPair(0, 2, "y", 1.0),
    ]
    settings = case.numerics
    settings.reconstruction = spec.reconstruction
    settings.time_integration = spec.time_integration
    settings.viscosity_model = spec.viscosity_model
    settings.turbulence_model = spec.turbulence_model
    settings.max_iterations = iterations
    settings.end_time = 1.0e30
    settings.residual_tolerance = 1.0e-300
    settings.convergence_history_interval = max(iterations + 1, 10)
    if spec.turbulence_model != "None" and settings.viscosity_model == "Inviscid":
        settings.viscosity_model = "Constant"

    states = tuple(
        InitialState(
            density=1.0 + 0.02 * math.sin(0.017 * cell_id),
            velocity_x=0.20 + 0.03 * math.sin(0.011 * cell_id),
            velocity_y=0.03 * math.cos(0.013 * cell_id),
            pressure=1.0 + 0.02 * math.cos(0.019 * cell_id),
            turbulence_k=1.0e-4,
            turbulence_omega=1.0,
        )
        for cell_id in range(spec.cells)
    )
    return case, states


def conservative_states(states: tuple[InitialState, ...], gamma: float) -> np.ndarray:
    primitive = np.array(
        [(s.density, s.velocity_x, s.velocity_y, s.pressure) for s in states],
        dtype=float,
    )
    return primitive_to_conservative(primitive, gamma)


def system_metadata() -> dict[str, Any]:
    try:
        import scipy
    except ImportError:
        scipy_version = "not-installed"
    else:
        scipy_version = scipy.__version__
    return {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "numpy": np.__version__,
        "scipy": scipy_version,
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "executable": str(Path(sys.executable).resolve()),
    }


def spec_as_dict(spec: BenchmarkSpec) -> dict[str, Any]:
    result = asdict(spec)
    result["cells"] = spec.cells
    return result
