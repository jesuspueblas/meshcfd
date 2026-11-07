import xml.etree.ElementTree as ET

from meshcfd.core.model import Case, FiniteVolumeMesh, InitialState, MeshPatch
from meshcfd.io.vtk_io import VTKSeriesWriter, export_convergence_csv, export_vtu


def make_case():
    mesh = FiniteVolumeMesh(
        points=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
        cells=((0, 1, 2, 3),),
        faces=((0, 1), (1, 2), (2, 3), (3, 0)),
        face_owners=(0, 0, 0, 0),
        face_neighbours=(-1, -1, -1, -1),
        face_patch_ids=(0, 1, 2, 3),
        patches=(
            MeshPatch(0, "bottom", (0,)),
            MeshPatch(1, "right", (1,)),
            MeshPatch(2, "top", (2,)),
            MeshPatch(3, "left", (3,)),
        ),
        cell_face_ids=((0, 1, 2, 3),),
        cell_face_signs=((1, 1, 1, 1),),
    )
    return Case(mesh=mesh)


def test_export_vtu_writes_polygon_mesh_and_cell_fields(tmp_path):
    case = make_case()
    path = export_vtu(
        tmp_path / "solution.vtu",
        case,
        (InitialState(density=1.2, velocity_x=3.0, pressure=101000.0),),
        0.25,
        7,
    )

    root = ET.parse(path).getroot()
    piece = root.find("./UnstructuredGrid/Piece")
    assert piece is not None
    assert piece.attrib == {"NumberOfPoints": "4", "NumberOfCells": "1"}
    arrays = {array.attrib["Name"]: array.text for array in piece.findall("./CellData/DataArray")}
    assert arrays["Density"] == "1.2"
    assert arrays["Velocity"] == "3 0 0"
    assert "MachNumber" in arrays
    assert "YPlus" in arrays
    assert piece.find("./FieldData/DataArray[@Name='Iteration']").text == "7"
    assert piece.find("./Cells/DataArray[@Name='types']").text == "7"


def test_vtk_series_is_incremental_and_can_resume_after_reopening(tmp_path):
    case = make_case()
    states = (InitialState(),)
    series_path = tmp_path / "run.pvd"
    writer = VTKSeriesWriter(series_path)
    writer.add_snapshot(case, states, 0.0, 0)
    writer.add_snapshot(case, states, 0.1, 2)

    reopened = VTKSeriesWriter(series_path)
    reopened.add_snapshot(case, states, 0.2, 4)

    root = ET.parse(series_path).getroot()
    datasets = root.findall("./Collection/DataSet")
    assert [entry.attrib["timestep"] for entry in datasets] == [
        "0",
        "0.10000000000000001",
        "0.20000000000000001",
    ]
    assert [entry.attrib["file"] for entry in datasets] == [
        "run_00000000.vtu",
        "run_00000002.vtu",
        "run_00000004.vtu",
    ]
    assert all((tmp_path / entry.attrib["file"]).exists() for entry in datasets)

    restarted = VTKSeriesWriter(series_path, append=False)
    restarted.add_snapshot(case, states, 0.0, 0)
    assert len(ET.parse(series_path).findall("./Collection/DataSet")) == 1


def test_export_convergence_csv_writes_paraview_table(tmp_path):
    path = export_convergence_csv(
        tmp_path / "convergence.csv",
        [
            (10, 0.05, 0.2, (1e-2, 2e-3, 3e-3, 4e-4)),
            (20, 0.10, 0.5, (5e-5, 6e-6, 7e-6, 8e-7, 9e-8, 1e-8)),
        ],
    )

    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == (
        "Iteration,PhysicalTime,CPUTime,MassResidual,MomentumXResidual,"
        "MomentumYResidual,EnergyResidual,TurbulentKResidual,"
        "TurbulentOmegaResidual"
    )
    import csv

    rows = list(csv.DictReader(lines))
    assert int(rows[0]["Iteration"]) == 10
    assert float(rows[0]["MassResidual"]) == 1e-2
    assert rows[0]["TurbulentKResidual"] == ""
    assert rows[0]["TurbulentOmegaResidual"] == ""
    assert int(rows[1]["Iteration"]) == 20
    assert float(rows[1]["TurbulentOmegaResidual"]) == 1e-8


def test_export_convergence_csv_rejects_empty_history(tmp_path):
    import pytest

    with pytest.raises(ValueError, match="no convergence history"):
        export_convergence_csv(tmp_path / "empty.csv", [])
