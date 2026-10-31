import json

from PyQt5.QtWidgets import QMessageBox

from meshcfd.core.model import Case
from meshcfd.gui.main_window import MainWindow
from meshcfd.io.mesh_io import mesh_from_dict, mesh_to_dict


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
    assert window.action_run.text() == "Run Solver"


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
    window.numerics_panel.reconstruction_combo.setCurrentText("MUSCL parabolic")

    condition = window.case.boundary_conditions[1]
    assert condition.kind == "Velocity Inlet"
    assert condition.values["velocity_x"] == 75.0
    assert window.case.numerics.reconstruction == "MUSCL parabolic"
    assert window._dirty
    assert window.canvas.selected_patch_id == 1
    before_select = window.canvas.transform()
    window.boundary_panel.patch_list.setCurrentRow(2)
    assert window.canvas.transform() == before_select
    assert window.canvas.selected_patch_id == 2


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
    assert len(window.boundary_panel.case.boundary_conditions) == 4
    assert window.canvas.mesh is window.case.mesh
    assert window._dirty
    assert window.mesh_info_label.text() == "1 cells | 4 faces | 4 patches"


def test_run_action_reports_solver_placeholder(qtbot, monkeypatch):
    window = MainWindow()
    qtbot.addWidget(window)
    case = Case()
    case.set_mesh(make_mesh())
    window._set_case(case, None)
    messages = []
    monkeypatch.setattr(
        "meshcfd.gui.main_window.QMessageBox.information",
        lambda *args: messages.append(args[2]),
    )

    window._on_run()

    assert len(messages) == 1
    assert "not been implemented" in messages[0]
    assert "no simulation was run" in window.status_label.text()


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
