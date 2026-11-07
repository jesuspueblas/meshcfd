"""VTU/PVD writers for cell-centered MeshCFD solutions."""

from __future__ import annotations

import csv
import math
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from pathlib import Path

from meshcfd.core.model import Case, InitialState
from meshcfd.core.solver import derive_solution_fields

_VTK_CELL_POLYGON = 7


_CONVERGENCE_COLUMNS = (
    "Iteration",
    "PhysicalTime",
    "CPUTime",
    "MassResidual",
    "MomentumXResidual",
    "MomentumYResidual",
    "EnergyResidual",
    "TurbulentKResidual",
    "TurbulentOmegaResidual",
)


def export_convergence_csv(
    path: str | Path,
    history: Iterable[tuple[int, float, float, tuple[float, ...]]],
) -> Path:
    """Write convergence history as a ParaView-readable CSV table.

    ParaView opens CSV files directly as tables, allowing any residual to be
    plotted against iteration, physical/pseudo-time, or CPU time. Missing
    turbulence residuals are written as empty fields.
    """
    destination = Path(path)
    if destination.suffix.lower() != ".csv":
        raise ValueError("A convergence-history filename must end in '.csv'.")

    rows = list(history)
    if not rows:
        raise ValueError("There is no convergence history to export.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(_CONVERGENCE_COLUMNS)
        for iteration, physical_time, cpu_time, residuals in rows:
            if len(residuals) not in (4, 6):
                raise ValueError(
                    "Each convergence sample must contain four or six residuals."
                )
            values = [iteration, physical_time, cpu_time, *residuals]
            values.extend("" for _ in range(len(_CONVERGENCE_COLUMNS) - len(values)))
            writer.writerow(_format_csv_value(value) for value in values)
    return destination


def _format_csv_value(value: int | float | str) -> int | str:
    if isinstance(value, str):
        return value
    if isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError("Convergence history contains a non-finite numeric value.")
    if isinstance(value, int):
        return value
    return format(value, ".17g")


def export_vtu(
    path: str | Path,
    case: Case,
    states: tuple[InitialState, ...],
    time: float,
    step: int,
) -> Path:
    """Write one cell-centered solution as an ASCII VTU file."""
    destination = Path(path)
    if destination.suffix.lower() != ".vtu":
        raise ValueError("A VTU export filename must end in '.vtu'.")
    if case.mesh is None:
        raise ValueError("Import a mesh before exporting a solution.")
    if len(states) != len(case.mesh.cells):
        raise ValueError("Solution state count must match the mesh cell count.")
    if not math.isfinite(time) or time < 0:
        raise ValueError("Solution time must be finite and non-negative.")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("Solution step must be a non-negative integer.")

    mesh = case.mesh
    fields = derive_solution_fields(case, states)
    root = ET.Element(
        "VTKFile",
        {"type": "UnstructuredGrid", "version": "0.1", "byte_order": "LittleEndian"},
    )
    grid = ET.SubElement(root, "UnstructuredGrid")
    piece = ET.SubElement(
        grid,
        "Piece",
        {"NumberOfPoints": str(len(mesh.points)), "NumberOfCells": str(len(mesh.cells))},
    )
    field_data = ET.SubElement(piece, "FieldData")
    _add_data_array(field_data, "Float64", "TimeValue", (time,))
    _add_data_array(field_data, "Int64", "Iteration", (step,))

    points = ET.SubElement(piece, "Points")
    point_data = []
    for x, y in mesh.points:
        point_data.extend((x, y, 0.0))
    _add_data_array(points, "Float64", "Points", point_data, 3)

    cells = ET.SubElement(piece, "Cells")
    connectivity = [node for cell in mesh.cells for node in cell]
    offsets = []
    offset = 0
    for cell in mesh.cells:
        offset += len(cell)
        offsets.append(offset)
    _add_data_array(cells, "Int32", "connectivity", connectivity)
    _add_data_array(cells, "Int32", "offsets", offsets)
    _add_data_array(cells, "UInt8", "types", (_VTK_CELL_POLYGON for _ in mesh.cells))

    cell_data = ET.SubElement(piece, "CellData", {"Scalars": "Pressure"})
    _add_data_array(cell_data, "Float64", "Density", (state.density for state in states))
    _add_data_array(cell_data, "Float64", "Velocity", _velocity_data(states), 3)
    _add_data_array(cell_data, "Float64", "Pressure", (state.pressure for state in states))
    _add_data_array(cell_data, "Float64", "TurbulentK", (state.turbulence_k for state in states))
    _add_data_array(
        cell_data,
        "Float64",
        "TurbulenceOmega",
        (state.turbulence_omega for state in states),
    )
    for name, key in (
        ("MachNumber", "mach_number"),
        ("TotalPressure", "total_pressure"),
        ("TotalTemperature", "total_temperature"),
        ("TotalDensity", "total_density"),
        ("YPlus", "y_plus"),
    ):
        _add_data_array(cell_data, "Float64", name, fields[key])

    destination.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(destination, encoding="utf-8", xml_declaration=True)
    return destination


class VTKSeriesWriter:
    """Incrementally write VTU snapshots and update their PVD collection."""

    def __init__(self, path: str | Path, append: bool = True) -> None:
        self.path = Path(path)
        if self.path.suffix.lower() != ".pvd":
            raise ValueError("A VTK time-series filename must end in '.pvd'.")
        self._entries: list[tuple[float, str]] = []
        if append and self.path.exists():
            self._read_existing_collection()

    def add_snapshot(
        self,
        case: Case,
        states: tuple[InitialState, ...],
        time: float,
        step: int,
    ) -> Path:
        snapshot_name = f"{self.path.stem}_{step:08d}.vtu"
        snapshot_path = self.path.with_name(snapshot_name)
        export_vtu(snapshot_path, case, states, time, step)
        self._entries = [entry for entry in self._entries if entry[1] != snapshot_name]
        self._entries.append((time, snapshot_name))
        self._entries.sort(key=lambda entry: (entry[0], entry[1]))
        self._write_collection()
        return snapshot_path

    def _read_existing_collection(self) -> None:
        try:
            root = ET.parse(self.path).getroot()
        except (ET.ParseError, OSError) as error:
            raise ValueError(f"Could not read VTK collection '{self.path}': {error}") from error
        if root.attrib.get("type") != "Collection":
            raise ValueError(f"'{self.path}' is not a VTK collection file.")
        collection = root.find("Collection")
        if collection is None:
            raise ValueError(f"'{self.path}' does not contain a VTK collection.")
        for dataset in collection.findall("DataSet"):
            filename = dataset.attrib.get("file")
            timestep = dataset.attrib.get("timestep")
            if filename is None or timestep is None:
                raise ValueError(f"'{self.path}' contains an invalid dataset entry.")
            try:
                time = float(timestep)
            except ValueError as error:
                raise ValueError(f"'{self.path}' contains an invalid dataset time.") from error
            if not math.isfinite(time):
                raise ValueError(f"'{self.path}' contains a non-finite dataset time.")
            self._entries.append((time, filename))

    def _write_collection(self) -> None:
        root = ET.Element(
            "VTKFile",
            {"type": "Collection", "version": "0.1", "byte_order": "LittleEndian"},
        )
        collection = ET.SubElement(root, "Collection")
        for time, filename in self._entries:
            ET.SubElement(
                collection,
                "DataSet",
                {
                    "timestep": f"{time:.17g}",
                    "group": "",
                    "part": "0",
                    "file": filename,
                },
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.path.with_name(self.path.name + ".tmp")
        ET.ElementTree(root).write(temporary_path, encoding="utf-8", xml_declaration=True)
        temporary_path.replace(self.path)


def _velocity_data(states: tuple[InitialState, ...]) -> tuple[float, ...]:
    values = []
    for state in states:
        values.extend((state.velocity_x, state.velocity_y, 0.0))
    return tuple(values)


def _add_data_array(
    parent: ET.Element,
    vtk_type: str,
    name: str,
    values: Iterable[int | float],
    components: int = 1,
) -> None:
    attributes = {"type": vtk_type, "Name": name, "format": "ascii"}
    if components != 1:
        attributes["NumberOfComponents"] = str(components)
    array = ET.SubElement(parent, "DataArray", attributes)
    array.text = " ".join(
        str(value) if isinstance(value, int) else f"{value:.17g}" for value in values
    )
