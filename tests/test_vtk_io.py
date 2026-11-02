import xml.etree.ElementTree as ET

from meshcfd.core.model import Case, FiniteVolumeMesh, InitialState, MeshPatch
from meshcfd.io.vtk_io import VTKSeriesWriter, export_vtu


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
