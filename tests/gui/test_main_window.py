import json
import xml.etree.ElementTree as ET
from threading import Event

import pytest
from PyQt5.QtCore import QEvent, QPointF, Qt
from PyQt5.QtGui import QMouseEvent
from PyQt5.QtWidgets import QApplication, QMessageBox

from meshcfd.core.model import Case, InitialState
from meshcfd.core.solver import SolverResult
from meshcfd.gui.main_window import MainWindow, _SolverWorker
from meshcfd.gui.results import ConvergenceHistoryWidget, CutProfileWidget, LinePlotWidget
from meshcfd.io.mesh_io import mesh_from_dict, mesh_to_dict
from meshcfd.io.project_io import load_case, save_case


def make_mesh():
    return mesh_from_dict(
        {
            "format": "meshcad-mesh",
            "version": 1,
            "dimension": 2,
            "nodes": {"coordinates": [[0, 0], [1, 0], [1, 1], [0, 1]]},
            "faces": {
                "nodes": [[0, 1], [1, 2], [2, 3], [3, 0]],
                "owner": [0, 0, 0, 0],
                "patch": [0, 1, 2, 3],
                "neighbour": [-1, -1, -1, -1],
            },
            "cells": {
                "faces": [[0, 1, 2, 3]],
                "face_signs": [[1, 1, 1, 1]],
            },
            "patches": [
                {"id": 0, "name": "bottom", "faces": [0]},
                {"id": 1, "name": "right", "faces": [1]},
                {"id": 2, "name": "top", "faces": [2]},
                {"id": 3, "name": "left", "faces": [3]},
            ],
        }
    )


def test_main_window_initial_state(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)

    assert window.case.mesh is None
    assert window.boundary_dock.windowTitle() == "Boundary Conditions"
    assert window.numerics_dock.windowTitle() == "Numerical Setup"
    assert window.initial_conditions_dock.windowTitle() == "Initial Conditions"
    assert window.output_dock.windowTitle() == "Output"
    assert window.dockWidgetArea(window.output_dock) == Qt.BottomDockWidgetArea
    assert not window.output_dock.isHidden()
    assert window.output_terminal.isReadOnly()
    assert "MeshCFD ready" in window.output_terminal.toPlainText()
    assert window.action_run.text() == "Run Solver"
    assert window.action_stop.text() == "Stop Solver"
    assert not window.action_stop.isEnabled()
    panel = window.visualization_panel
    assert panel.variable_combo.currentText() == "Density"
    assert panel.palette_combo.currentText() == "Rainbow"
    assert panel.scale_combo.currentText() == "Linear"
    assert panel.auto_range_checkbox.isChecked()
    assert panel.show_colorbar_checkbox.isChecked()
    assert panel.orientation_combo.currentText() == "Vertical"
    assert panel.orientation_combo.isEnabled()
    assert window.canvas.color_map == "Rainbow"
    assert window.canvas.color_scale == "Linear"
    assert window.canvas.show_colorbar
    assert window.canvas.color_bar_overlay.orientation == "Vertical"
    assert not window.action_export_vtu.isEnabled()
    assert not window.action_solution_cut.isEnabled()
    assert window.minimumHeight() == 480
    assert window.boundary_dock.widget() is window.boundary_scroll
    assert window.boundary_scroll.widget() is window.boundary_panel
    assert window.convergence_history_dock.widget() is window.convergence_history_scroll
    assert window.convergence_history_scroll.widget() is window.convergence_history_widget
    assert window.cut_profile_dock.widget() is window.cut_profile_scroll
    assert window.cut_profile_scroll.widget() is window.cut_profile_widget
    for dock in (window.convergence_history_dock, window.cut_profile_dock):
        assert dock.features() & dock.DockWidgetMovable
        assert dock.features() & dock.DockWidgetFloatable
        assert dock.allowedAreas() == Qt.AllDockWidgetAreas


def test_loading_case_resets_canvas_zoom_and_centers_mesh(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(1400, 900)
    window.show()
    qtbot.waitExposed(window)
    window.canvas.scale(2.0, 2.0)
    window.canvas.horizontalScrollBar().setValue(
        window.canvas.horizontalScrollBar().value() + 30
    )
    window.canvas.verticalScrollBar().setValue(
        window.canvas.verticalScrollBar().value() + 20
    )

    case = Case()
    case.set_mesh(make_mesh())
    window._set_case(case, None)

    mesh_center = QPointF(0.5, 0.5)
    viewport_center = window.canvas.viewport().rect().center()
    mapped_center = window.canvas.mapFromScene(mesh_center)
    assert abs(mapped_center.x() - viewport_center.x()) <= 2
    assert abs(mapped_center.y() - viewport_center.y()) <= 2


def test_implicit_integrator_enables_large_cfl_only_for_implicit(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    panel = window.numerics_panel

    panel.time_integration_combo.setCurrentText("Implicit Euler")
    panel.cfl_spin.setValue(25.0)
    panel.gmres_tolerance_spin.setValue(1e-8)
    panel.gmres_iterations_spin.setValue(500)
    panel.gmres_krylov_dimension_spin.setValue(40)
    panel.implicit_preconditioner_combo.setCurrentText("LU-SGS")
    assert panel.cfl_spin.maximum() == 1000.0
    assert window.case.numerics.cfl == pytest.approx(25.0)
    assert window.case.numerics.gmres_tolerance == pytest.approx(1e-8)
    assert window.case.numerics.gmres_max_iterations == 500
    assert window.case.numerics.gmres_krylov_dimension == 40
    assert window.case.numerics.implicit_preconditioner == "LU-SGS"

    panel.time_integration_combo.setCurrentText("Explicit RK4")
    assert panel.cfl_spin.maximum() == 1.0
    assert panel.cfl_spin.value() == 1.0
    assert window.case.numerics.cfl == pytest.approx(1.0)


def test_adaptive_mesh_controls_update_numerical_settings(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    panel = window.numerics_panel

    assert not panel.adaptive_mesh_checkbox.isChecked()
    assert not panel.adaptation_interval_spin.isVisible()
    panel.adaptive_mesh_checkbox.setChecked(True)
    panel.adaptation_interval_spin.setValue(5)
    panel.adaptation_max_level_spin.setValue(6)
    panel.adaptation_threshold_spin.setValue(0.1)
    panel.adaptation_variable_combo.setCurrentText("Pressure")

    assert window.case.numerics.adaptive_mesh
    assert window.case.numerics.adaptation_interval == 5
    assert window.case.numerics.adaptation_max_level == 6
    assert window.case.numerics.adaptation_refine_threshold == pytest.approx(0.1)
    assert window.case.numerics.adaptation_variable == "Pressure"


def test_convergence_history_interval_control_updates_settings(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)

    panel = window.numerics_panel
    assert panel.convergence_history_interval_spin.value() == 10
    panel.convergence_history_interval_spin.setValue(25)

    assert window.case.numerics.convergence_history_interval == 25


def test_result_actions_export_vtu_and_configure_series(qtbot, monkeypatch, tmp_path):
    window = MainWindow()
    qtbot.addWidget(window)
    case = Case()
    case.set_mesh(make_mesh())
    window._set_case(case, None)
    window._simulation_solution = (InitialState(),)
    window.case.solution_time = 0.25
    window.case.solution_steps = 5
    window._update_results_actions()
    assert window.action_export_vtu.isEnabled()
    assert window.action_solution_cut.isEnabled()

    export_path = tmp_path / "final.vtu"
    monkeypatch.setattr(
        "meshcfd.gui.main_window.QFileDialog.getSaveFileName",
        lambda *_args: (str(export_path), "VTU files (*.vtu)"),
    )
    window._on_export_vtu()
    assert export_path.exists()

    series_path = tmp_path / "history.pvd"
    monkeypatch.setattr(
        "meshcfd.gui.main_window.QFileDialog.getSaveFileName",
        lambda *_args: (str(series_path), "PVD files (*.pvd)"),
    )
    monkeypatch.setattr(
        "meshcfd.gui.main_window.QInputDialog.getInt",
        lambda *_args: (25, True),
    )
    window._on_configure_vtk_series()
    assert window._vtk_series_path == series_path
    assert window._vtk_interval == 25
    assert window.action_disable_vtk_series.isEnabled()
    window._on_disable_vtk_series()
    assert window._vtk_series_path is None
    assert not window.action_disable_vtk_series.isEnabled()


def test_result_widgets_prepare_convergence_and_cut_plot_data(qtbot):
    history = [(1, 0.1, 0.025, (0.4, 0.3, 0.2, 0.1, 0.05, 0.01))]
    convergence = ConvergenceHistoryWidget(lambda: history)
    qtbot.addWidget(convergence)
    assert convergence.plot.logarithmic_y
    assert convergence.plot.x_values == (1.0,)
    assert convergence.plot.canvas.x_label == "Iteration"
    assert convergence.plot.series["Energy"] == (0.1,)
    assert convergence._turbulence_group.isHidden()
    assert "Turbulent k" not in convergence.plot.series
    convergence.set_turbulence_enabled(True)
    assert not convergence._turbulence_group.isHidden()
    assert convergence.plot.series["Turbulent k"] == (0.05,)
    convergence.variable_items["Mass"].setCheckState(0, Qt.Unchecked)
    assert "Mass" not in convergence.plot.series
    assert "Energy" in convergence.plot.series
    convergence.variable_items["Mass"].setCheckState(0, Qt.Checked)
    assert "Mass" in convergence.plot.series
    convergence.scale_combo.setCurrentText("Linear")
    assert not convergence.plot.logarithmic_y
    convergence.scale_combo.setCurrentText("Logarithmic")
    assert convergence.plot.logarithmic_y
    convergence.x_axis_combo.setCurrentText("Physical time")
    assert convergence.plot.x_values == (0.1,)
    assert convergence.plot.canvas.x_label == "Physical time (s)"
    convergence.x_axis_combo.setCurrentText("CPU time")
    assert convergence.plot.x_values == (0.025,)
    assert convergence.plot.canvas.x_label == "CPU time (s)"

    history[:] = [(2, 0.2, 0.05, (0.4, 0.3, 0.2, 0.1))]
    convergence.refresh_plot()
    assert "Turbulent k" not in convergence.plot.series

    case = Case()
    case.set_mesh(make_mesh())
    cut = CutProfileWidget("Density")
    qtbot.addWidget(cut)
    cut.set_solution(case, (InitialState(density=1.5),), "Density")
    assert cut.plot.x_values == (0.0,)
    assert cut.plot.series["Density"] == (1.5,)
    assert not hasattr(cut, "plot_button")
    assert not hasattr(cut.plot, "pan_button")
    assert not hasattr(cut.plot, "zoom_button")
    assert not cut.reset_zoom_button.icon().isNull()
    assert not convergence.reset_zoom_button.icon().isNull()


def test_plot_zoom_pan_and_reset_controls(qtbot):
    plot = LinePlotWidget()
    qtbot.addWidget(plot)
    plot.resize(640, 420)
    plot.set_data((0.0, 1.0, 2.0), {"Mass": (1.0, 2.0, 3.0)}, "Iteration", "Residual")
    plot.show()
    qtbot.waitExposed(plot)
    canvas = plot.canvas
    initial_bounds = canvas._view_bounds

    start = QPointF(110.0, 60.0)
    end = QPointF(300.0, 220.0)
    QApplication.sendEvent(
        canvas,
        QMouseEvent(
            QEvent.MouseButtonPress,
            start,
            Qt.LeftButton,
            Qt.LeftButton,
            Qt.NoModifier,
        ),
    )
    QApplication.sendEvent(
        canvas,
        QMouseEvent(
            QEvent.MouseButtonRelease,
            end,
            Qt.LeftButton,
            Qt.NoButton,
            Qt.NoModifier,
        ),
    )
    zoomed_bounds = canvas._view_bounds
    assert zoomed_bounds != initial_bounds

    QApplication.sendEvent(
        canvas,
        QMouseEvent(
            QEvent.MouseButtonPress,
            QPointF(200.0, 140.0),
            Qt.MiddleButton,
            Qt.MiddleButton,
            Qt.NoModifier,
        ),
    )
    QApplication.sendEvent(
        canvas,
        QMouseEvent(
            QEvent.MouseMove,
            QPointF(240.0, 160.0),
            Qt.NoButton,
            Qt.MiddleButton,
            Qt.NoModifier,
        ),
    )
    QApplication.sendEvent(
        canvas,
        QMouseEvent(
            QEvent.MouseButtonRelease,
            QPointF(240.0, 160.0),
            Qt.MiddleButton,
            Qt.NoButton,
            Qt.NoModifier,
        ),
    )
    assert canvas._view_bounds != zoomed_bounds
    plot.reset_zoom()
    assert canvas._view_bounds == initial_bounds


def test_view_toolbar_and_all_toolbar_visibility_actions(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)

    assert window.view_toolbar.objectName() == "view_toolbar"
    assert window.toolBarArea(window.case_toolbar) == Qt.TopToolBarArea
    assert window.toolBarArea(window.view_toolbar) == Qt.TopToolBarArea
    assert window.toolBarArea(window.visualization_toolbar) == Qt.TopToolBarArea

    view_menu = next(
        action.menu()
        for action in window.menuBar().actions()
        if action.text() == "&View"
    )
    menu_actions = {action.text(): action for action in view_menu.actions()}
    assert {"Case", "View", "Solution View"} <= menu_actions.keys()
    assert all(
        not action.icon().isNull()
        for action in (
            window.action_new,
            window.action_open,
            window.action_save,
            window.action_import_mesh,
            window.action_run,
            window.action_stop,
            window.action_fit_mesh,
            window.action_grid,
            window.action_clear_output,
        )
    )

    menu_actions["View"].trigger()
    assert window.view_toolbar.isHidden()
    menu_actions["View"].trigger()
    assert not window.view_toolbar.isHidden()


def test_plot_series_are_clipped_to_axes(qtbot):
    plot = LinePlotWidget()
    qtbot.addWidget(plot)
    plot.resize(480, 320)
    plot.set_data(
        (0.0, 1.0, 2.0, 3.0),
        {"Mass": (0.5, 10.0, -10.0, 0.5)},
        "Iteration",
        "Residual",
    )
    plot.show()
    qtbot.waitExposed(plot)
    canvas = plot.canvas
    canvas._view_bounds = (0.0, 3.0, 0.0, 1.0)
    canvas.update()
    qtbot.wait(20)

    image = canvas.grab().toImage()
    axes = canvas._plot_rect()
    for y in (*range(0, int(axes.top())), *range(int(axes.bottom()) + 2, image.height())):
        assert not any(
            (color := image.pixelColor(x, y)).red() < 100
            and color.green() < 150
            and color.blue() > 170
            for x in range(int(axes.left()) + 1, int(axes.right()))
        )


def test_cut_dock_draws_and_updates_reference_line(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)
    case = Case()
    case.set_mesh(make_mesh())
    window._set_case(case, None)
    window._simulation_solution = (InitialState(density=1.5),)
    window.cut_profile_dock.show()

    line = next(item for item in window.canvas.scene().items() if item.data(0) == "solution_cut")
    assert line.line().x1() == pytest.approx(0.0)
    assert line.line().x2() == pytest.approx(1.0)
    assert line.line().y1() == pytest.approx(0.5)
    assert line.line().y2() == pytest.approx(0.5)

    window.cut_profile_widget.angle_spin.setValue(90.0)
    line = next(item for item in window.canvas.scene().items() if item.data(0) == "solution_cut")
    assert line.line().x1() == pytest.approx(0.5)
    assert line.line().x2() == pytest.approx(0.5)

    window.cut_profile_dock.hide()
    assert not any(
        item.data(0) == "solution_cut" for item in window.canvas.scene().items()
    )


def test_solver_worker_writes_interval_and_final_vtk_snapshots(tmp_path):
    case = Case()
    case.set_mesh(make_mesh())
    case.numerics.end_time = 100.0
    case.numerics.max_iterations = 3
    worker = _SolverWorker(
        case,
        (InitialState(),),
        False,
        tmp_path / "run.pvd",
        2,
    )

    worker.run()

    assert worker.error is None
    assert worker.output_error is None
    assert worker.result is not None
    assert not worker.result.completed
    entries = [
        entry.attrib for entry in ET.parse(tmp_path / "run.pvd").findall("./Collection/DataSet")
    ]
    assert [entry["file"] for entry in entries] == [
        "run_00000000.vtu",
        "run_00000002.vtu",
        "run_00000003.vtu",
    ]


def test_run_pauses_and_stop_preserves_state_for_updated_settings(qtbot, monkeypatch):
    window = MainWindow()
    qtbot.addWidget(window)
    monkeypatch.setattr(
        "meshcfd.gui.main_window.QMessageBox.critical", lambda *_args: None
    )
    case = Case()
    case.set_mesh(make_mesh())
    case.numerics.end_time = 1.0
    window._set_case(case, None)
    started = Event()
    calls = []
    partial_state = InitialState(1.05, 0.1, 0.0, 1.0)

    def fake_solve_case(
        run_case,
        initial_states,
        _progress_callback,
        cancel_callback,
        _live_solution_updates,
        start_time,
        step_offset,
        pause_callback,
        **_kwargs,
    ):
        calls.append((run_case, tuple(initial_states), start_time, step_offset))
        if len(calls) == 1:
            started.set()
            while not cancel_callback():
                pause_callback()
                Event().wait(0.001)
            return SolverResult((partial_state,), 0.25, 4, 0.5, False)
        return SolverResult(
            tuple(initial_states), start_time + 0.1, step_offset + 1, 0.0, False
        )

    monkeypatch.setattr("meshcfd.gui.main_window.solve_case", fake_solve_case)
    window._on_run()
    try:
        assert started.wait(timeout=5.0)
        assert window.action_run.text() == "Pause Solver"
        assert window.action_stop.isEnabled()
        window.action_run.trigger()
        qtbot.waitUntil(
            lambda: window.status_label.text() == "Solver paused.", timeout=5000
        )
        assert window.action_run.text() == "Resume Solver"
        window.numerics_panel.end_time_spin.setValue(2.0)
        window.action_stop.trigger()
        qtbot.waitUntil(lambda: window._solver_worker is None, timeout=5000)
    finally:
        if window._solver_worker is not None:
            window._solver_worker.stop()

    assert window.case.final_solution == (partial_state,)
    assert window.case.solution_time == 0.25
    assert window.case.solution_steps == 4
    assert window.action_run.text() == "Run Solver"
    assert not window.action_stop.isEnabled()

    window.action_run.trigger()
    qtbot.waitUntil(lambda: window._solver_worker is None, timeout=5000)
    resumed_case, resumed_states, start_time, step_offset = calls[1]
    assert resumed_case.numerics.end_time == 2.0
    assert resumed_states == (partial_state,)
    assert start_time == 0.25
    assert step_offset == 4


def test_output_dock_layout_preserves_canvas_width(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(1400, 850)
    window.show()
    qtbot.waitUntil(lambda: window.canvas.width() > 300)

    assert window.initial_conditions_scroll.horizontalScrollBar().maximum() == 0
    assert window.numerics_scroll.horizontalScrollBar().maximum() == 0
    assert (
        window.visualization_scroll.horizontalScrollBarPolicy()
        == Qt.ScrollBarAsNeeded
    )
    assert window.minimumSizeHint().height() < 600


def test_boundary_and_numerical_controls_update_case(qtbot, monkeypatch):
    window = MainWindow()
    qtbot.addWidget(window)
    monkeypatch.setattr(window, "_confirm_discard_changes", lambda: True)
    case = Case()
    case.set_mesh(make_mesh())
    window._set_case(case, None)

    window.boundary_panel.patch_list.setCurrentRow(1)
    window.boundary_panel.kind_combo.setCurrentText("Velocity Inlet")
    velocity = window.boundary_panel.value_controls["velocity_x"]
    velocity.setValue(75.0)
    velocity.editingFinished.emit()
    window.numerics_panel.reconstruction_combo.setCurrentText("First order")
    window.numerics_panel.end_time_spin.setValue(2.5)

    condition = window.case.boundary_conditions[1]
    assert condition.kind == "Velocity Inlet"
    assert condition.values["velocity_x"] == 75.0
    assert window.case.numerics.reconstruction == "First order"
    assert window.case.numerics.end_time == 2.5
    assert window._dirty
    assert window.canvas.selected_patch_id == 1
    before_select = window.canvas.transform()
    window.boundary_panel.patch_list.setCurrentRow(2)
    assert window.canvas.transform() == before_select
    assert window.canvas.selected_patch_id == 2


def test_viscous_and_acceleration_controls_update_case(qtbot, monkeypatch):
    window = MainWindow()
    qtbot.addWidget(window)
    monkeypatch.setattr(window, "_confirm_discard_changes", lambda: True)
    case = Case()
    case.numerics.viscosity_model = "Sutherland"
    case.numerics.sutherland_reference_viscosity = 1.7e-5
    case.numerics.prandtl_number = 0.73
    window._set_case(case, None)

    panel = window.numerics_panel
    assert panel.sutherland_reference_viscosity_spin.value() == 1.7e-5
    assert panel.prandtl_number_spin.value() == 0.73
    panel.sutherland_reference_viscosity_spin.setValue(1.8e-5)
    panel.prandtl_number_spin.setValue(0.74)
    panel.acceleration_y_spin.setValue(-9.81)

    assert window.case.numerics.viscosity_model == "Sutherland"
    assert window.case.numerics.sutherland_reference_viscosity == 1.8e-5
    assert window.case.numerics.prandtl_number == 0.74
    assert window.case.numerics.acceleration_y == -9.81
    assert not panel.sutherland_temperature_spin.isHidden()
    assert panel.dynamic_viscosity_spin.isHidden()
    window._dirty = False


def test_sst_controls_update_model_and_boundary_inputs(qtbot, monkeypatch):
    window = MainWindow()
    qtbot.addWidget(window)
    monkeypatch.setattr(window, "_confirm_discard_changes", lambda: True)
    case = Case()
    case.set_mesh(make_mesh())
    window._set_case(case, None)

    panel = window.numerics_panel
    panel.turbulence_model_combo.setCurrentText("Menter k-omega SST")
    assert not window.convergence_history_widget._turbulence_group.isHidden()
    assert panel.turbulence_omega_formulation_combo.currentText() == "Standard SST"
    panel.turbulence_omega_formulation_combo.setCurrentText(
        "Regularized k-log-omega"
    )
    assert window.case.numerics.turbulence_omega_formulation == (
        "Regularized k-log-omega"
    )
    assert not panel.turbulence_wall_treatment_combo.isEnabled()
    panel.turbulence_omega_formulation_combo.setCurrentText("Standard SST")
    assert panel.turbulence_wall_treatment_combo.isEnabled()
    panel.turbulence_wall_treatment_combo.setCurrentText("Wall functions")
    assert window.case.numerics.turbulence_model == "Menter k-omega SST"
    assert window.case.numerics.turbulence_wall_treatment == "Wall functions"
    initial_panel = window.initial_conditions_panel
    initial_panel.base_controls["turbulence_k"].setValue(0.025)
    initial_panel.base_controls["turbulence_omega"].setValue(12.0)
    assert window.case.initial_conditions.base_state.turbulence_k == 0.025
    assert window.case.initial_conditions.base_state.turbulence_omega == 12.0

    boundary_panel = window.boundary_panel
    boundary_panel.patch_list.setCurrentRow(1)
    boundary_panel.kind_combo.setCurrentText("Velocity Inlet")
    assert "turbulence_intensity" in boundary_panel.value_controls
    assert "turbulence_length_scale" in boundary_panel.value_controls
    intensity = boundary_panel.value_controls["turbulence_intensity"]
    assert intensity.minimum() == 0.0
    assert intensity.maximum() == 100.0
    intensity.setValue(3.0)
    intensity.editingFinished.emit()
    assert window.case.boundary_conditions[1].values["turbulence_intensity"] == 3.0

    panel.turbulence_model_combo.setCurrentText("None")
    assert window.convergence_history_widget._turbulence_group.isHidden()
    assert "turbulence_intensity" not in boundary_panel.value_controls
    window._dirty = False


def test_wall_thermal_condition_selects_temperature_or_heat_flux(
    qtbot, monkeypatch
):
    window = MainWindow()
    qtbot.addWidget(window)
    monkeypatch.setattr(window, "_confirm_discard_changes", lambda: True)
    case = Case()
    case.set_mesh(make_mesh())
    window._set_case(case, None)

    panel = window.boundary_panel
    panel.patch_list.setCurrentRow(0)
    assert panel.wall_mechanical_combo.currentText() == "No-slip"
    panel.wall_mechanical_combo.setCurrentText("Slip")
    assert window.case.boundary_conditions[0].mechanical_condition == "Slip"
    assert panel.wall_thermal_combo.currentText() == "Temperature"
    panel.wall_thermal_combo.setCurrentText("Heat Flux")

    assert window.case.boundary_conditions[0].mechanical_condition == "Slip"
    assert window.case.boundary_conditions[0].thermal_condition == "Heat Flux"
    assert set(panel.value_controls) == {"heat_flux"}
    assert "positive from the wall into" in panel.value_controls["heat_flux"].toolTip()
    panel.value_controls["heat_flux"].setValue(120.0)
    panel.value_controls["heat_flux"].editingFinished.emit()
    assert window.case.boundary_conditions[0].values["heat_flux"] == 120.0
    panel.wall_thermal_combo.setCurrentText("Temperature")
    assert set(panel.value_controls) == {"temperature"}
    assert panel.wall_mechanical_combo.currentText() == "Slip"
    window._dirty = False


def test_muscl_parabolic_reconstruction_is_available(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)

    combo = window.numerics_panel.reconstruction_combo
    combo.setCurrentText("MUSCL parabolic")

    assert combo.currentText() == "MUSCL parabolic"
    assert window.case.numerics.reconstruction == "MUSCL parabolic"


def test_far_field_boundary_uses_density_velocity_and_pressure_controls(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    case = Case()
    case.set_mesh(make_mesh())
    window._set_case(case, None)

    window.boundary_panel.patch_list.setCurrentRow(1)
    window.boundary_panel.kind_combo.setCurrentText("Far Field")

    assert set(window.boundary_panel.value_controls) == {
        "density",
        "velocity_x",
        "velocity_y",
        "pressure",
    }


def test_mass_flow_boundary_controls_use_positive_per_depth_rate(
    qtbot, monkeypatch
):
    monkeypatch.setattr(
        "meshcfd.gui.main_window.QMessageBox.question",
        lambda *_: QMessageBox.Yes,
    )
    window = MainWindow()
    qtbot.addWidget(window)
    case = Case()
    case.set_mesh(make_mesh())
    window._set_case(case, None)

    panel = window.boundary_panel
    panel.patch_list.setCurrentRow(3)
    panel.kind_combo.setCurrentText("Mass Flow Inlet")
    rate = panel.value_controls["mass_flow"]
    assert rate.minimum() > 0.0
    assert "per unit depth" in rate.toolTip()
    rate.setValue(2.5)
    rate.editingFinished.emit()
    assert window.case.boundary_conditions[3].values["mass_flow"] == 2.5
    assert set(panel.value_controls) == {
        "mass_flow",
        "temperature",
        "velocity_x",
        "velocity_y",
    }

    panel.kind_combo.setCurrentText("Mass Flow Outlet")
    assert set(panel.value_controls) == {"mass_flow"}
    assert panel.value_controls["mass_flow"].minimum() > 0.0
    window._dirty = False


def test_boundary_panel_creates_detected_periodic_pair(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    case = Case()
    case.set_mesh(make_mesh())
    window._set_case(case, None)

    panel = window.boundary_panel
    panel.periodic_patch_a_combo.setCurrentIndex(
        panel.periodic_patch_a_combo.findData(1)
    )
    panel.periodic_patch_b_combo.setCurrentIndex(
        panel.periodic_patch_b_combo.findData(3)
    )

    assert panel.add_periodic_pair_button.isEnabled()
    assert "X" in panel.periodic_status.text()
    assert "pitch = 1" in panel.periodic_status.text()

    panel.add_periodic_pair_button.click()

    assert window.case.periodic_pairs[0].axis == "x"
    assert window.case.periodic_pairs[0].pitch == 1.0
    assert window.case.boundary_conditions[1].kind == "Periodic"
    assert window.case.boundary_conditions[3].kind == "Periodic"
    assert window._dirty
    assert panel.periodic_pairs_list.count() == 1

    panel.periodic_pairs_list.setCurrentRow(0)
    panel.remove_periodic_pair_button.click()
    assert not window.case.periodic_pairs
    assert window.case.boundary_conditions[1].kind == "Wall"


def test_initial_condition_panel_adds_and_edits_ordered_blocks(
    qtbot, monkeypatch
):
    monkeypatch.setattr(
        "meshcfd.gui.main_window.QMessageBox.question",
        lambda *_: QMessageBox.Yes,
    )
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()

    panel = window.initial_conditions_panel
    qtbot.waitUntil(panel.add_button.isVisible)
    panel.add_button.click()
    assert len(window.case.initial_conditions.blocks) == 1
    panel.block_controls["domain"].setText("x < 0")
    panel.add_button.click()
    panel.up_button.click()

    assert window.case.initial_conditions.blocks[1].domain == "x < 0"
    assert window._dirty


def test_import_mesh_action_loads_patches_and_displays_mesh(
    qtbot, monkeypatch, tmp_path
):
    window = MainWindow()
    qtbot.addWidget(window)
    mesh_path = tmp_path / "mesh.json"
    mesh_path.write_text(json.dumps(mesh_to_dict(make_mesh())), encoding="utf-8")
    monkeypatch.setattr(
        "meshcfd.gui.main_window.QFileDialog.getOpenFileName",
        lambda *args, **kwargs: (str(mesh_path), ""),
    )

    window._on_import_mesh()

    assert window.case.mesh is not None
    assert window.case.mesh_path == mesh_path.resolve()
    assert window.canvas.solution is not None
    assert len(window.boundary_panel.case.boundary_conditions) == 4
    assert window.canvas.mesh is window.case.mesh
    assert window._dirty
    assert window.mesh_info_label.text() == "1 cells | 4 faces | 4 patches"


def test_run_action_runs_solver_and_displays_solution(qtbot, tmp_path):
    window = MainWindow()
    qtbot.addWidget(window)
    case = Case()
    case.set_mesh(make_mesh())
    case.numerics.end_time = 1e-3
    window._set_case(case, None)

    window._on_run()

    qtbot.waitUntil(lambda: window._solver_worker is None, timeout=5000)
    assert "Completed at t" in window.status_label.text()
    assert window.canvas.solution is not None
    assert window.action_run.isEnabled()
    output = window.output_terminal.toPlainText()
    assert "Starting HLLC solver" in output
    assert "Iteration" in output
    assert "ITER" in output
    assert "Completed at t" in output
    assert not window.output_dock.isHidden()
    assert window.case.final_solution == window.canvas.solution

    case_path = tmp_path / "solved.mcf.json"
    save_case(case_path, window.case)
    restored = load_case(case_path)
    assert restored.final_solution == window.case.final_solution
    assert restored.solution_time == window.case.solution_time
    assert restored.solution_steps == window.case.solution_steps
    assert restored.convergence_history == window.case.convergence_history
    assert restored.solution_cpu_time == window.case.solution_cpu_time

    window._set_case(restored, case_path)
    assert window.canvas.solution == restored.final_solution
    assert window._convergence_history == restored.convergence_history


def test_running_again_continues_from_current_solution(qtbot, tmp_path):
    window = MainWindow()
    qtbot.addWidget(window)
    case = Case()
    case.set_mesh(make_mesh())
    case.numerics.end_time = 10.0
    case.numerics.max_iterations = 1
    window._set_case(case, None)

    window._on_run()
    qtbot.waitUntil(lambda: window._solver_worker is None, timeout=5000)
    first_solution = window.case.final_solution
    first_time = window.case.solution_time
    first_history = list(window.case.convergence_history)
    first_cpu_time = window.case.solution_cpu_time
    assert first_solution is not None
    assert window.case.solution_steps == 1
    assert first_history
    assert first_cpu_time >= 0.0

    case_path = tmp_path / "restart.mcf.json"
    save_case(case_path, window.case)
    restored = load_case(case_path)
    window._set_case(restored, case_path)
    assert window.canvas.solution == first_solution
    assert window._convergence_history == first_history
    assert window._cpu_time_offset == first_cpu_time
    window.convergence_history_widget.refresh_plot()
    assert window.convergence_history_widget.plot.x_values == (
        float(first_history[0][0]),
    )
    window.numerics_panel.end_time_spin.setValue(20.0)
    assert window.canvas.solution == first_solution
    assert window.case.solution_time == first_time
    assert window._convergence_history == first_history

    window._on_run()
    worker = window._solver_worker
    assert worker is not None
    assert worker.initial_states == restored.final_solution
    assert worker.case.solution_time == first_time
    assert worker.case.solution_steps == 1
    qtbot.waitUntil(lambda: window._solver_worker is None, timeout=5000)

    assert window.case.solution_time > first_time
    assert window.case.solution_steps == 2
    assert len(window.case.convergence_history) == len(first_history) + 1
    assert window.case.convergence_history[: len(first_history)] == first_history
    assert window.case.solution_cpu_time >= first_cpu_time


def test_live_solution_toggle_is_logged_and_used_for_new_runs(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)

    window.numerics_panel.live_solution_checkbox.setChecked(True)

    assert window._live_solution_updates
    assert "Live solution updates enabled" in window.output_terminal.toPlainText()


def test_case_save_and_open(qtbot, monkeypatch, tmp_path):
    window = MainWindow()
    qtbot.addWidget(window)
    mesh_path = tmp_path / "mesh.json"
    mesh_path.write_text(json.dumps(mesh_to_dict(make_mesh())), encoding="utf-8")
    window.case.set_mesh(make_mesh(), mesh_path)
    window.boundary_panel.set_case(window.case)
    case_path = tmp_path / "saved.mcf.json"
    monkeypatch.setattr(
        "meshcfd.gui.main_window.QFileDialog.getSaveFileName",
        lambda *args, **kwargs: (str(case_path), ""),
    )

    window._on_save_as()

    assert case_path.exists()
    assert json.loads(case_path.read_text(encoding="utf-8"))["mesh"] == "mesh.json"
    assert not window._dirty
    assert window.current_path == case_path

    monkeypatch.setattr(
        "meshcfd.gui.main_window.QFileDialog.getOpenFileName",
        lambda *args, **kwargs: (str(case_path), ""),
    )
    window._set_case(Case(), None)
    window._on_open()

    assert window.case.mesh is not None
    assert window.case.mesh_path == mesh_path.resolve()
    assert len(window.case.boundary_conditions) == 4
    assert window.current_path == case_path
