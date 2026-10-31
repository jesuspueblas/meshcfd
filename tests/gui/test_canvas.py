import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor

from meshcfd.core.model import FiniteVolumeMesh, InitialState, MeshPatch
from meshcfd.gui.canvas import MeshCanvas


def make_mesh():
    return FiniteVolumeMesh(
        points=((0, 0), (1, 0), (1, 1), (0, 1)),
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


def test_solution_is_colored_per_cell_and_mesh_edges_can_be_hidden(qtbot):
    canvas = MeshCanvas()
    qtbot.addWidget(canvas)
    canvas.set_mesh(make_mesh())
    canvas.set_solution((InitialState(density=2.0),))

    values = canvas.set_display_options(
        show_grid=True,
        show_mesh=False,
        solution_variable="Density",
        color_map="Plasma",
        color_scale="Linear",
    )

    assert values == (2.0,)
    assert canvas.show_grid
    assert not canvas.show_mesh
    assert len(canvas.scene().items()) == 1
    item = canvas.scene().items()[0]
    assert item.brush().color() != QColor("#F1F5F9")
    assert item.pen().style() == Qt.NoPen


def test_logarithmic_color_scale_rejects_non_positive_values(qtbot):
    canvas = MeshCanvas()
    qtbot.addWidget(canvas)
    canvas.set_mesh(make_mesh())
    canvas.set_solution((InitialState(velocity_x=-1.0),))

    with pytest.raises(ValueError, match="positive"):
        canvas.set_display_options(
            show_grid=True,
            show_mesh=True,
            solution_variable="Velocity X",
            color_map="Viridis",
            color_scale="Logarithmic",
        )


def test_grid_and_mesh_visibility_can_be_toggled_from_main_window(qtbot):
    from meshcfd.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window.case.set_mesh(make_mesh())
    window._set_case(window.case, None)

    window.action_grid.trigger()
    window.visualization_panel.show_mesh_checkbox.setChecked(False)

    assert not window.canvas.show_grid
    assert not window.canvas.show_mesh


def test_visualization_panel_selects_solution_field_and_reports_invalid_log_scale(
    qtbot,
):
    from meshcfd.core.model import Case, InitialConditionBlock
    from meshcfd.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    case = Case()
    case.set_mesh(make_mesh())
    case.initial_conditions.blocks.append(
        InitialConditionBlock(domain="True", density="2", pressure="200000")
    )
    window._set_case(case, None)

    panel = window.visualization_panel
    panel.variable_combo.setCurrentText("Pressure")
    panel.palette_combo.setCurrentText("Inferno")
    assert window.canvas.solution_variable == "Pressure"
    assert window.canvas.color_map == "Inferno"
    assert panel.legend.minimum == 200000.0
    assert panel.legend.maximum == 200000.0

    panel.variable_combo.setCurrentText("Velocity X")
    panel.scale_combo.setCurrentText("Logarithmic")
    assert "positive" in panel.status_label.text()
    assert window.canvas.solution is None

    panel.variable_combo.setCurrentText("Pressure")
    assert window.canvas.solution is not None
    assert window.canvas.color_scale == "Logarithmic"
