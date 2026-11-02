import pytest
from PyQt5.QtCore import QEvent, QPoint, QPointF, Qt
from PyQt5.QtGui import QColor, QMouseEvent
from PyQt5.QtWidgets import QApplication, QGraphicsPixmapItem, QGraphicsPolygonItem

from meshcfd.core.model import FiniteVolumeMesh, InitialState, MeshPatch
from meshcfd.gui.canvas import COLOR_MAPS, ColorLegend, MeshCanvas, color_values


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


def make_two_cell_mesh():
    return FiniteVolumeMesh(
        points=((0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)),
        cells=((0, 1, 4, 3), (1, 2, 5, 4)),
        faces=((0, 1), (1, 4), (4, 3), (3, 0), (1, 2), (2, 5), (5, 4)),
        face_owners=(0, 0, 0, 0, 1, 1, 1),
        face_neighbours=(-1, 1, -1, -1, -1, -1, -1),
        face_patch_ids=(0, -1, 2, 3, 0, 1, 2),
        patches=(
            MeshPatch(0, "bottom", (0, 4)),
            MeshPatch(1, "right", (5,)),
            MeshPatch(2, "top", (2, 6)),
            MeshPatch(3, "left", (3,)),
        ),
        cell_face_ids=((0, 1, 2, 3), (4, 5, 6, 1)),
        cell_face_signs=((1, 1, 1, 1), (1, 1, 1, -1)),
    )


def send_canvas_mouse_event(canvas, event_type, position, button, buttons):
    global_position = canvas.viewport().mapToGlobal(position)
    QApplication.sendEvent(
        canvas.viewport(),
        QMouseEvent(
            event_type,
            QPointF(position),
            QPointF(global_position),
            QPointF(global_position),
            button,
            buttons,
            Qt.NoModifier,
        ),
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


@pytest.mark.parametrize(
    ("variable", "expected"),
    (
        ("Turbulent k", 0.02),
        ("Specific dissipation rate", 3.0),
    ),
)
def test_sst_fields_can_be_visualized(qtbot, variable, expected):
    canvas = MeshCanvas()
    qtbot.addWidget(canvas)
    canvas.set_mesh(make_mesh())
    canvas.set_solution(
        (InitialState(turbulence_k=0.02, turbulence_omega=3.0),)
    )

    values = canvas.set_display_options(
        show_grid=True,
        show_mesh=True,
        solution_variable=variable,
        color_map="Viridis",
        color_scale="Linear",
    )

    assert values == (expected,)


@pytest.mark.parametrize(
    ("variable", "field", "expected"),
    (
        ("Mach number", "mach_number", 0.8),
        ("Total pressure", "total_pressure", 120000.0),
        ("Total temperature", "total_temperature", 320.0),
        ("Total density", "total_density", 1.25),
        ("y+ (wall distance)", "y_plus", 25.0),
    ),
)
def test_derived_flow_fields_can_be_visualized(qtbot, variable, field, expected):
    canvas = MeshCanvas()
    qtbot.addWidget(canvas)
    canvas.set_mesh(make_mesh())
    canvas.set_solution(
        (InitialState(),),
        {field: (expected,)},
    )

    values = canvas.set_display_options(
        show_grid=True,
        show_mesh=True,
        solution_variable=variable,
        color_map="Viridis",
        color_scale="Linear",
    )

    assert values == (expected,)


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


def test_color_palettes_and_manual_range_clamp_values():
    assert {"Rainbow", "Turbo", "Cividis"} <= COLOR_MAPS.keys()
    for palette in COLOR_MAPS:
        assert len(color_values((0.0, 0.5, 1.0), palette, "Linear")) == 3

    colors = color_values((0.0, 5.0, 10.0), "Turbo", "Linear", (2.0, 8.0))
    limits = color_values((2.0, 8.0), "Turbo", "Linear", (2.0, 8.0))
    assert colors[0] == limits[0]
    assert colors[2] == limits[1]


def test_background_grid_draws_coordinate_labels(qtbot, monkeypatch):
    canvas = MeshCanvas()
    qtbot.addWidget(canvas)
    canvas.resize(500, 400)
    labels = []
    draw_label = canvas._draw_coordinate_label

    def capture_label(*args, **kwargs):
        labels.append(args[1])
        draw_label(*args, **kwargs)

    monkeypatch.setattr(canvas, "_draw_coordinate_label", capture_label)
    canvas.show()
    qtbot.waitExposed(canvas)
    qtbot.waitUntil(lambda: len(labels) >= 2)

    assert labels


def test_manual_color_range_validation_and_logarithmic_mapping():
    with pytest.raises(ValueError, match="less than"):
        color_values((1.0,), "Viridis", "Linear", (2.0, 2.0))
    with pytest.raises(ValueError, match="positive"):
        color_values((1.0,), "Viridis", "Logarithmic", (-1.0, 2.0))

    colors = color_values((1.0, 10.0, 100.0), "Viridis", "Logarithmic", (1.0, 100.0))
    assert colors[0] == color_values((1.0,), "Viridis", "Logarithmic", (1.0, 100.0))[0]
    assert colors[-1] == color_values(
        (100.0,), "Viridis", "Logarithmic", (1.0, 100.0)
    )[0]


def test_horizontal_color_legend_draws_variable_above_bar(qtbot):
    legend = ColorLegend()
    qtbot.addWidget(legend)
    legend.resize(310, 62)
    legend.set_values((1.0, 2.0), "Viridis", variable="Density")

    image = legend.grab().toImage()
    title_band_has_ink = any(
        image.pixelColor(x, y).value() < 200
        for y in range(legend.height() - 20)
        for x in range(legend.width())
    )

    assert title_band_has_ink


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
    assert not window.action_grid.icon().isNull()


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
    assert window.canvas.color_bar_overlay.legend.minimum == 200000.0
    assert window.canvas.color_bar_overlay.legend.maximum == 200000.0

    panel.variable_combo.setCurrentText("Velocity X")
    panel.scale_combo.setCurrentText("Logarithmic")
    assert "positive" in panel.status_label.text()
    assert window.canvas.solution is None

    panel.variable_combo.setCurrentText("Pressure")
    assert window.canvas.solution is not None
    assert window.canvas.color_scale == "Logarithmic"


def test_manual_range_controls_update_canvas_and_legend(qtbot):
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
    panel.auto_range_checkbox.setChecked(False)
    panel.minimum_spin.setValue(100000.0)
    panel.maximum_spin.setValue(300000.0)

    assert window.canvas.color_range == (100000.0, 300000.0)
    assert window.canvas.color_bar_overlay.legend.minimum == 100000.0
    assert window.canvas.color_bar_overlay.legend.maximum == 300000.0

    panel.minimum_spin.setValue(400000.0)
    assert "less than" in panel.status_label.text()


def test_smooth_coloring_interpolates_between_cells(qtbot):
    canvas = MeshCanvas()
    qtbot.addWidget(canvas)
    canvas.set_mesh(make_two_cell_mesh())
    canvas.set_solution(
        (InitialState(density=1.0), InitialState(density=3.0))
    )

    canvas.set_display_options(
        show_grid=False,
        show_mesh=False,
        solution_variable="Density",
        color_map="Viridis",
        color_scale="Linear",
    )
    flat_items = canvas.scene().items()
    assert any(isinstance(item, QGraphicsPolygonItem) for item in flat_items)

    canvas.set_display_options(
        show_grid=False,
        show_mesh=False,
        solution_variable="Density",
        color_map="Viridis",
        color_scale="Linear",
        smooth_colors=True,
    )
    smooth_items = canvas.scene().items()
    image_item = next(
        item for item in smooth_items if isinstance(item, QGraphicsPixmapItem)
    )
    image = image_item.pixmap().toImage()
    assert image.pixelColor(40, image.height() // 2) != image.pixelColor(
        image.width() - 40, image.height() // 2
    )
    expected_shared_node = color_values((2.0,), "Viridis", "Linear")[0]
    assert image.pixelColor(image.width() // 2, image.height() // 2) == (
        expected_shared_node
    )
    cached = canvas._smooth_solution_image((1.0, 3.0))
    assert canvas._smooth_solution_image((1.0, 3.0)) is cached


def test_box_zoom_fits_the_dragged_rectangle(qtbot):
    canvas = MeshCanvas()
    qtbot.addWidget(canvas)
    canvas.resize(500, 400)
    canvas.set_mesh(make_two_cell_mesh())
    canvas.show()
    qtbot.waitExposed(canvas)
    canvas.reset_zoom()
    initial_scale = canvas.transform().m11()
    canvas.set_interaction_mode("Box Zoom")
    start = canvas.mapFromScene(QPointF(0.1, 0.1))
    end = canvas.mapFromScene(QPointF(1.9, 0.9))

    send_canvas_mouse_event(
        canvas, QEvent.MouseButtonPress, start, Qt.LeftButton, Qt.LeftButton
    )
    send_canvas_mouse_event(
        canvas, QEvent.MouseMove, end, Qt.NoButton, Qt.LeftButton
    )
    send_canvas_mouse_event(
        canvas, QEvent.MouseButtonRelease, end, Qt.LeftButton, Qt.NoButton
    )

    assert canvas.transform().m11() > initial_scale
    assert not canvas._zoom_rubber_band.isVisible()


def test_cell_info_reports_zero_based_index_centroid_and_active_field(qtbot):
    canvas = MeshCanvas()
    qtbot.addWidget(canvas)
    canvas.resize(500, 400)
    canvas.set_mesh(make_two_cell_mesh())
    canvas.set_solution(
        (InitialState(density=1.0), InitialState(density=3.0))
    )
    canvas.set_display_options(
        show_grid=False,
        show_mesh=False,
        solution_variable="Density",
        color_map="Viridis",
        color_scale="Linear",
        smooth_colors=True,
    )
    canvas.show()
    qtbot.waitExposed(canvas)
    canvas.reset_zoom()
    selected = []
    canvas.cell_selected.connect(
        lambda *details: selected.append(details)
    )
    canvas.set_interaction_mode("Cell Info")
    position = canvas.mapFromScene(QPointF(0.25, 0.5))

    send_canvas_mouse_event(
        canvas, QEvent.MouseButtonPress, position, Qt.LeftButton, Qt.LeftButton
    )

    assert selected == [(0, 0.5, 0.5, "Density", 1.0)]


def test_cell_info_reports_derived_solution_field(qtbot):
    canvas = MeshCanvas()
    qtbot.addWidget(canvas)
    canvas.resize(500, 400)
    canvas.set_mesh(make_two_cell_mesh())
    canvas.set_solution(
        (InitialState(), InitialState()),
        {"mach_number": (0.25, 0.75)},
    )
    canvas.set_display_options(
        show_grid=False,
        show_mesh=True,
        solution_variable="Mach number",
        color_map="Viridis",
        color_scale="Linear",
    )
    canvas.show()
    qtbot.waitExposed(canvas)
    canvas.reset_zoom()
    selected = []
    canvas.cell_selected.connect(lambda *details: selected.append(details))
    canvas.set_interaction_mode("Cell Info")
    position = canvas.mapFromScene(QPointF(0.25, 0.5))
    send_canvas_mouse_event(
        canvas, QEvent.MouseButtonPress, position, Qt.LeftButton, Qt.LeftButton
    )

    assert selected == [(0, 0.5, 0.5, "Mach number", 0.25)]


def test_solution_toolbar_layout_and_interactive_color_bar(qtbot):
    from meshcfd.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(1800, 1000)
    window.show()
    qtbot.waitExposed(window)

    assert window.dockWidgetArea(window.boundary_dock) == Qt.LeftDockWidgetArea
    assert (
        window.dockWidgetArea(window.initial_conditions_dock)
        == Qt.LeftDockWidgetArea
    )
    assert window.dockWidgetArea(window.numerics_dock) == Qt.RightDockWidgetArea
    assert (
        window.toolBarArea(window.visualization_toolbar) == Qt.TopToolBarArea
    )

    panel = window.visualization_panel
    assert panel.height() == 42
    assert panel.show_mesh_checkbox.height() == panel.smooth_colors_checkbox.height()
    assert panel.variable_combo.height() == panel.minimum_spin.height()
    assert not panel.show_mesh_checkbox.icon().isNull()
    assert not panel.smooth_colors_checkbox.icon().isNull()
    assert not panel.show_colorbar_checkbox.icon().isNull()
    assert set(panel.interaction_mode_buttons) == {
        "Navigate",
        "Box Zoom",
        "Cell Info",
    }
    assert all(
        not button.icon().isNull()
        for button in panel.interaction_mode_buttons.values()
    )
    assert panel.interaction_mode_buttons["Navigate"].isChecked()
    window.case.set_mesh(make_mesh())
    window._set_case(window.case, None)
    panel.interaction_mode_buttons["Cell Info"].click()
    assert window.canvas.interaction_mode == "Cell Info"
    position = window.canvas.mapFromScene(QPointF(0.5, 0.5))
    send_canvas_mouse_event(
        window.canvas,
        QEvent.MouseButtonPress,
        position,
        Qt.LeftButton,
        Qt.LeftButton,
    )
    assert "Cell 0" in window.status_label.text()
    assert "x = 0.5, y = 0.5" in window.status_label.text()
    assert "Density = 1.225" in window.status_label.text()
    assert not hasattr(panel, "legend")
    panel.show_colorbar_checkbox.setChecked(True)
    overlay = window.canvas.color_bar_overlay
    assert overlay.isVisible()
    panel.orientation_combo.setCurrentText("Vertical")
    assert overlay.legend.orientation == "Vertical"
    assert overlay.legend.variable == "Density"
    assert overlay.height() >= overlay.width()
    vertical_width = overlay.width()
    vertical_bar_right = overlay.legend.rect().right() - 62
    assert vertical_bar_right < vertical_width - 40
    panel.orientation_combo.setCurrentText("Horizontal")
    assert overlay.legend.orientation == "Horizontal"
    panel.variable_combo.setCurrentText("Velocity X")
    assert overlay.legend.variable == "Velocity X"
    overlay.move_to(QPoint(80, 60))
    assert overlay.pos() == QPoint(80, 60)

    def send_mouse_event(widget, event_type, local, global_position, button, buttons):
        QApplication.sendEvent(
            widget,
            QMouseEvent(
                event_type,
                QPointF(local),
                QPointF(global_position),
                QPointF(global_position),
                button,
                buttons,
                Qt.NoModifier,
            ),
        )

    legend = overlay.legend
    legend_local = QPoint(12, 10)
    legend_global = legend.mapToGlobal(legend_local)
    start_position = overlay.pos()
    send_mouse_event(
        legend,
        QEvent.MouseButtonPress,
        legend_local,
        legend_global,
        Qt.LeftButton,
        Qt.LeftButton,
    )
    send_mouse_event(
        legend,
        QEvent.MouseMove,
        legend_local,
        legend_global + QPoint(30, 20),
        Qt.NoButton,
        Qt.LeftButton,
    )
    send_mouse_event(
        legend,
        QEvent.MouseButtonRelease,
        legend_local,
        legend_global + QPoint(30, 20),
        Qt.LeftButton,
        Qt.NoButton,
    )
    assert overlay.pos() == start_position + QPoint(30, 20)

    handle = overlay.resize_handle
    handle_local = QPoint(8, 8)
    handle_global = handle.mapToGlobal(handle_local)
    start_size = overlay.size()
    send_mouse_event(
        handle,
        QEvent.MouseButtonPress,
        handle_local,
        handle_global,
        Qt.LeftButton,
        Qt.LeftButton,
    )
    send_mouse_event(
        handle,
        QEvent.MouseMove,
        handle_local,
        handle_global + QPoint(30, 20),
        Qt.NoButton,
        Qt.LeftButton,
    )
    send_mouse_event(
        handle,
        QEvent.MouseButtonRelease,
        handle_local,
        handle_global + QPoint(30, 20),
        Qt.LeftButton,
        Qt.NoButton,
    )
    assert overlay.size().width() == start_size.width() + 30
    assert overlay.size().height() == start_size.height() + 20

    panel.smooth_colors_checkbox.setChecked(True)
    assert window.canvas.smooth_colors
