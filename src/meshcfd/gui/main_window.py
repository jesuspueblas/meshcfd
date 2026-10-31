"""MeshCFD case setup application window."""

from __future__ import annotations

import sys
from pathlib import Path

from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QDockWidget,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QStyle,
    QToolBar,
)

from meshcfd.core.model import Case
from meshcfd.gui.boundary_panel import BoundaryPanel
from meshcfd.gui.canvas import MeshCanvas
from meshcfd.gui.numerics_panel import NumericsPanel
from meshcfd.io.mesh_io import MeshImportError, load_mesh
from meshcfd.io.project_io import load_case, save_case


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.case = Case()
        self.current_path: Path | None = None
        self._dirty = False
        self.resize(1400, 850)

        self.canvas = MeshCanvas(self)
        self.setCentralWidget(self.canvas)

        self.boundary_panel = BoundaryPanel(self)
        self.boundary_dock = self._add_dock(
            "Boundary Conditions", Qt.LeftDockWidgetArea, self.boundary_panel
        )
        self.numerics_panel = NumericsPanel(self)
        self.numerics_dock = self._add_dock(
            "Numerical Setup", Qt.RightDockWidgetArea, self.numerics_panel
        )

        self.status_label = QLabel("Ready")
        self.statusBar().addWidget(self.status_label)
        self.mesh_info_label = QLabel("No mesh loaded")
        self.statusBar().addPermanentWidget(self.mesh_info_label)

        self.boundary_panel.changed.connect(self._mark_dirty)
        self.boundary_panel.patch_selected.connect(self.canvas.select_patch)
        self.numerics_panel.changed.connect(self._mark_dirty)

        self._create_actions()
        self._create_menus()
        self._create_toolbar()
        self._update_title()

    def _add_dock(self, title: str, area, widget) -> QDockWidget:
        dock = QDockWidget(title, self)
        dock.setObjectName(title.lower().replace(" ", "_"))
        dock.setWidget(widget)
        dock.setAllowedAreas(
            Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea
        )
        self.addDockWidget(area, dock)
        return dock

    def _create_actions(self) -> None:
        self.action_new = QAction("New Case", self)
        self.action_new.setShortcut(QKeySequence.New)
        self.action_new.triggered.connect(self._on_new)

        self.action_open = QAction("Open Case...", self)
        self.action_open.setShortcut(QKeySequence.Open)
        self.action_open.triggered.connect(self._on_open)

        self.action_save = QAction("Save Case", self)
        self.action_save.setShortcut(QKeySequence.Save)
        self.action_save.triggered.connect(self._on_save)

        self.action_save_as = QAction("Save Case As...", self)
        self.action_save_as.setShortcut(QKeySequence.SaveAs)
        self.action_save_as.triggered.connect(self._on_save_as)

        self.action_import_mesh = QAction("Import Mesh...", self)
        self.action_import_mesh.triggered.connect(self._on_import_mesh)

        self.action_fit_mesh = QAction("Fit Mesh", self)
        self.action_fit_mesh.triggered.connect(self.canvas.reset_zoom)

        self.action_run = QAction("Run Solver", self)
        self.action_run.triggered.connect(self._on_run)

    def _create_menus(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        for action in (self.action_new, self.action_open, self.action_save, self.action_save_as):
            file_menu.addAction(action)
        file_menu.addSeparator()
        file_menu.addAction(self.action_import_mesh)
        file_menu.addSeparator()
        action_exit = QAction("Exit", self)
        action_exit.triggered.connect(self.close)
        file_menu.addAction(action_exit)

        simulation_menu = self.menuBar().addMenu("&Simulation")
        simulation_menu.addAction(self.action_run)

        view_menu = self.menuBar().addMenu("&View")
        view_menu.addAction(self.action_fit_mesh)
        view_menu.addAction(self.boundary_dock.toggleViewAction())
        view_menu.addAction(self.numerics_dock.toggleViewAction())

    def _create_toolbar(self) -> None:
        toolbar = QToolBar("Case", self)
        toolbar.setObjectName("case_toolbar")
        toolbar.setMovable(True)
        toolbar.setFloatable(False)
        icon_size = self.style().pixelMetric(QStyle.PM_SmallIconSize)
        toolbar.setIconSize(QSize(icon_size, icon_size))
        toolbar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        toolbar.setStyleSheet(
            "QToolBar { spacing: 4px; padding: 4px; background: #F8FAFC; "
            "border-bottom: 1px solid #CBD5E1; }"
            "QToolButton { padding: 4px 7px; border-radius: 5px; }"
            "QToolButton:hover { background: #DBEAFE; }"
        )
        self.addToolBar(Qt.TopToolBarArea, toolbar)
        self.case_toolbar = toolbar
        for action, icon in (
            (self.action_new, QStyle.SP_FileIcon),
            (self.action_open, QStyle.SP_DialogOpenButton),
            (self.action_save, QStyle.SP_DialogSaveButton),
            (self.action_import_mesh, QStyle.SP_DirOpenIcon),
            (self.action_fit_mesh, QStyle.SP_DesktopIcon),
            (self.action_run, QStyle.SP_MediaPlay),
        ):
            action.setIcon(self.style().standardIcon(icon))
            action.setToolTip(action.text())
            action.setStatusTip(action.text())
            toolbar.addAction(action)
        toolbar.addSeparator()

    def _update_title(self) -> None:
        filename = self.current_path.name if self.current_path else self.case.name
        marker = "*" if self._dirty else ""
        self.setWindowTitle(f"MeshCFD - {filename}{marker}")

    def _mark_dirty(self) -> None:
        self._dirty = True
        self._update_title()

    def _set_case(self, case: Case, path: Path | None) -> None:
        self.case = case
        self.current_path = path
        self._dirty = False
        self.canvas.set_mesh(case.mesh)
        self.boundary_panel.set_case(case)
        self.numerics_panel.set_case(case)
        self._update_mesh_info()
        self._update_title()

    def _update_mesh_info(self) -> None:
        if self.case.mesh is None:
            self.mesh_info_label.setText("No mesh loaded")
            return
        self.mesh_info_label.setText(
            f"{len(self.case.mesh.cells)} cells | "
            f"{len(self.case.mesh.faces)} faces | "
            f"{len(self.case.mesh.patches)} patches"
        )

    def _confirm_discard_changes(self) -> bool:
        if not self._dirty:
            return True
        answer = QMessageBox.question(
            self,
            "Unsaved Changes",
            "Discard unsaved case changes?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        return answer == QMessageBox.Yes

    def _on_new(self) -> None:
        if self._confirm_discard_changes():
            self._set_case(Case(), None)
            self.status_label.setText("New case.")

    def _on_open(self) -> None:
        if not self._confirm_discard_changes():
            return
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open MeshCFD Case",
            "",
            "MeshCFD Case (*.mcf.json);;JSON Files (*.json)",
        )
        if not filename:
            return
        path = Path(filename)
        try:
            case = load_case(path)
        except (OSError, UnicodeError, ValueError) as error:
            QMessageBox.critical(self, "Open Case Error", str(error))
            return
        self._set_case(case, path)
        self.status_label.setText(f"Opened case: {path.name}")

    def _on_save(self) -> None:
        if self.current_path is None:
            self._on_save_as()
            return
        try:
            save_case(self.current_path, self.case)
        except (OSError, ValueError) as error:
            QMessageBox.critical(self, "Save Case Error", str(error))
            return
        self._dirty = False
        self._update_title()
        self.status_label.setText(f"Saved case: {self.current_path.name}")

    def _on_save_as(self) -> None:
        default_name = f"{self.case.name}.mcf.json"
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save MeshCFD Case",
            default_name,
            "MeshCFD Case (*.mcf.json);;JSON Files (*.json)",
        )
        if not filename:
            return
        path = Path(filename)
        if not path.name.lower().endswith(".json"):
            path = path.with_name(path.name + ".mcf.json")
        previous_name = self.case.name
        self.case.name = path.name.replace(".mcf.json", "")
        try:
            save_case(path, self.case)
        except (OSError, ValueError) as error:
            self.case.name = previous_name
            QMessageBox.critical(self, "Save Case Error", str(error))
            return
        self.current_path = path
        self._dirty = False
        self._update_title()
        self.status_label.setText(f"Saved case: {path.name}")

    def _on_import_mesh(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Import Finite-Volume Mesh",
            "",
            "MeshCAD CFD Mesh (*.json);;JSON Files (*.json)",
        )
        if not filename:
            return
        try:
            mesh = load_mesh(filename)
        except (OSError, UnicodeError, MeshImportError) as error:
            QMessageBox.critical(self, "Mesh Import Error", str(error))
            return
        self.case.set_mesh(mesh)
        self.canvas.set_mesh(mesh)
        self.boundary_panel.set_case(self.case)
        self._update_mesh_info()
        self._mark_dirty()
        self.status_label.setText(f"Imported mesh: {Path(filename).name}")

    def _on_run(self) -> None:
        if self.case.mesh is None:
            QMessageBox.information(
                self, "No Mesh", "Import a finite-volume mesh before setting up a run."
            )
            return
        QMessageBox.information(
            self,
            "Solver Not Available",
            "The case setup interface is ready, but the CFD solver has not "
            "been implemented yet.",
        )
        self.status_label.setText("Solver not implemented yet; no simulation was run.")

    def closeEvent(self, event) -> None:
        if self._confirm_discard_changes():
            event.accept()
        else:
            event.ignore()


def main() -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec_()


if __name__ == "__main__":
    main()
