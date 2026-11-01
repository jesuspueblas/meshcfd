import json

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMessageBox

from meshcfd.core.model import Case
from meshcfd.gui.main_window import MainWindow
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


def test_output_dock_layout_preserves_canvas_width(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(1400, 850)
    window.show()
    qtbot.waitUntil(lambda: window.canvas.width() > 300)

    assert window.initial_conditions_scroll.horizontalScrollBar().maximum() == 0
    assert window.numerics_scroll.horizontalScrollBar().maximum() == 0


def test_boundary_and_numerical_controls_update_case(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
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

    window._set_case(restored, case_path)
    assert window.canvas.solution == restored.final_solution


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
    assert first_solution is not None
    assert window.case.solution_steps == 1

    case_path = tmp_path / "restart.mcf.json"
    save_case(case_path, window.case)
    restored = load_case(case_path)
    window._set_case(restored, case_path)
    assert window.canvas.solution == first_solution
    window.numerics_panel.end_time_spin.setValue(20.0)
    assert window.canvas.solution == first_solution
    assert window.case.solution_time == first_time

    window._on_run()
    worker = window._solver_worker
    assert worker is not None
    assert worker.initial_states == restored.final_solution
    assert worker.case.solution_time == first_time
    assert worker.case.solution_steps == 1
    qtbot.waitUntil(lambda: window._solver_worker is None, timeout=5000)

    assert window.case.solution_time > first_time
    assert window.case.solution_steps == 2


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
