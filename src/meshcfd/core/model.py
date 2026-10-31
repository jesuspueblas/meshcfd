"""Core data models used by the MeshCFD case editor."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

BOUNDARY_TYPES = (
    "Wall",
    "Velocity Inlet",
    "Pressure Inlet",
    "Pressure Outlet",
    "Symmetry",
    "Far Field",
    "Periodic",
)

RECONSTRUCTION_SCHEMES = ("First order", "MUSCL linear", "MUSCL parabolic")
LIMITER_SCHEMES = ("Barth-Jespersen", "Venkatakrishnan", "Minmod", "Van Leer")
FLUX_SCHEMES = ("HLLC", "Roe", "Rusanov")
TIME_INTEGRATION_SCHEMES = ("Explicit Euler", "Explicit RK2", "Explicit RK3")


@dataclass
class MeshPatch:
    id: int
    name: str
    face_ids: tuple[int, ...]


@dataclass
class FiniteVolumeMesh:
    points: tuple[tuple[float, float], ...]
    cells: tuple[tuple[int, ...], ...]
    faces: tuple[tuple[int, int], ...]
    face_owners: tuple[int, ...]
    face_neighbours: tuple[int, ...]
    face_patch_ids: tuple[int, ...]
    patches: tuple[MeshPatch, ...]
    cell_face_ids: tuple[tuple[int, ...], ...]
    cell_face_signs: tuple[tuple[int, ...], ...]


@dataclass
class BoundaryCondition:
    patch_id: int
    patch_name: str
    kind: str = "Wall"
    values: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class PeriodicBoundaryPair:
    patch_a_id: int
    patch_b_id: int
    axis: str
    pitch: float


@dataclass
class NumericalSettings:
    reconstruction: str = "MUSCL linear"
    limiter: str = "Barth-Jespersen"
    flux: str = "HLLC"
    time_integration: str = "Explicit RK2"
    cfl: float = 0.5
    max_iterations: int = 1000
    residual_tolerance: float = 1e-6
    gamma: float = 1.4
    gas_constant: float = 287.05


@dataclass
class InitialState:
    density: float = 1.225
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    pressure: float = 101325.0


@dataclass
class InitialConditionBlock:
    name: str = "Block 1"
    domain: str = "True"
    density: str = "1.225"
    velocity_x: str = "0"
    velocity_y: str = "0"
    pressure: str = "101325"


@dataclass
class InitialConditionSettings:
    base_state: InitialState = field(default_factory=InitialState)
    blocks: list[InitialConditionBlock] = field(default_factory=list)


@dataclass
class Case:
    name: str = "Untitled"
    mesh: FiniteVolumeMesh | None = None
    boundary_conditions: list[BoundaryCondition] = field(default_factory=list)
    numerics: NumericalSettings = field(default_factory=NumericalSettings)
    mesh_path: Path | None = None
    initial_conditions: InitialConditionSettings = field(
        default_factory=InitialConditionSettings
    )
    periodic_pairs: list[PeriodicBoundaryPair] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.mesh is not None and not self.boundary_conditions:
            self.boundary_conditions = [
                BoundaryCondition(patch.id, patch.name)
                for patch in self.mesh.patches
            ]

    def set_mesh(
        self, mesh: FiniteVolumeMesh, mesh_path: str | Path | None = None
    ) -> None:
        previous = {
            condition.patch_name: condition
            for condition in self.boundary_conditions
        }
        self.mesh = mesh
        self.mesh_path = Path(mesh_path).resolve() if mesh_path is not None else None
        self.boundary_conditions = [
            BoundaryCondition(
                patch_id=patch.id,
                patch_name=patch.name,
                kind=(
                    previous[patch.name].kind
                    if patch.name in previous
                    and previous[patch.name].kind != "Periodic"
                    else "Wall"
                ),
                values=(
                    dict(previous[patch.name].values)
                    if patch.name in previous
                    else {}
                ),
            )
            for patch in mesh.patches
        ]
        self.periodic_pairs.clear()
