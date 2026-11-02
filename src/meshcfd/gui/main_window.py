"""MeshCFD case setup application window."""

from __future__ import annotations

import copy
import sys
from datetime import datetime
from pathlib import Path
from threading import Event

from PyQt5.QtCore import QRect, QSize, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QCursor, QIcon, QKeySequence, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QDockWidget,
    QFileDialog,
    QFrame,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QScrollArea,
    QStyle,
    QToolBar,
    QToolTip,
)

from meshcfd.core.initial_conditions import evaluate_initial_conditions
from meshcfd.core.model import Case, InitialState
from meshcfd.core.solver import (
    SolverProgress,
    SolverResult,
    derive_solution_fields,
    solve_case,
)
from meshcfd.gui.boundary_panel import BoundaryPanel
from meshcfd.gui.canvas import MeshCanvas
from meshcfd.gui.initial_conditions_panel import InitialConditionsPanel
from meshcfd.gui.numerics_panel import NumericsPanel
from meshcfd.gui.visualization_panel import VisualizationPanel
from meshcfd.io.mesh_io import MeshImportError, load_mesh
from meshcfd.io.project_io import load_case, save_case


class _SolverWorker(QThread):
    progress = pyqtSignal(object)
    pause_state_changed = pyqtSignal(bool)

    def __init__(
        self, case: Case, initial_states, live_solution_updates: bool
    ) -> None:
        super().__init__()
        self.case = case
        self.initial_states = initial_states
        self.live_solution_updates = live_solution_updates
        self.result: SolverResult | None = None
        self.error: str | None = None
        self._resume_event = Event()
        self._resume_event.set()
        self._stop_requested = Event()

    def run(self) -> None:
        try:
            self.result = solve_case(
                self.case,
                self.initial_states,
                self._report_progress,
                self.isInterruptionRequested,
                self.live_solution_updates,
                self.case.solution_time,
                self.case.solution_steps,
                self._pause_at_checkpoint,
            )
        except Exception as error:
            self.error = str(error)

    def pause(self) -> None:
        self._resume_event.clear()

    def resume(self) -> None:
        self._resume_event.set()

    def stop(self) -> None:
        self._stop_requested.set()
        self.requestInterruption()
        self.resume()

    @property
    def stop_requested(self) -> bool:
        return self._stop_requested.is_set()

    def _pause_at_checkpoint(self) -> None:
        if self._resume_event.is_set():
            return
        self.pause_state_changed.emit(True)
        self._resume_event.wait()
        self.pause_state_changed.emit(False)

    def _report_progress(self, progress: SolverProgress) -> None:
        self.progress.emit(progress)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.case = Case()
        self.current_path: Path | None = None
        self._initial_solution: tuple[InitialState, ...] | None = None
        self._simulation_solution: tuple[InitialState, ...] | None = None
        self._case_revision = 0
        self._run_revision: int | None = None
        self._run_case_ref: Case | None = None
        self._solver_worker: _SolverWorker | None = None
        self._pause_requested = False
        self._live_solution_updates = False
        self._dirty = False
        self.resize(1400, 850)

        self.canvas = MeshCanvas(self)
        self.setCentralWidget(self.canvas)
        self.setDockNestingEnabled(True)

        self.boundary_panel = BoundaryPanel(self)
        self.boundary_dock = self._add_dock(
            "Boundary Conditions", Qt.LeftDockWidgetArea, self.boundary_panel
        )
        self.initial_conditions_panel = InitialConditionsPanel(self)
        self.initial_conditions_dock = self._add_dock(
            "Initial Conditions",
            Qt.LeftDockWidgetArea,
            self.initial_conditions_panel,
        )
        self.splitDockWidget(
            self.boundary_dock, self.initial_conditions_dock, Qt.Vertical
        )
        self.numerics_panel = NumericsPanel(self)
        self.numerics_scroll = QScrollArea(self)
        self.numerics_scroll.setFrameShape(QFrame.NoFrame)
        self.numerics_scroll.setWidgetResizable(True)
        self.numerics_scroll.setWidget(self.numerics_panel)
        self.numerics_dock = self._add_dock(
            "Numerical Setup", Qt.RightDockWidgetArea, self.numerics_scroll
        )
        self.initial_conditions_scroll = QScrollArea(self)
        self.initial_conditions_scroll.setFrameShape(QFrame.NoFrame)
        self.initial_conditions_scroll.setWidgetResizable(True)
        self.initial_conditions_scroll.setWidget(self.initial_conditions_panel)
        self.initial_conditions_dock.setWidget(self.initial_conditions_scroll)
        self.output_terminal = QPlainTextEdit(self)
        self.output_terminal.setObjectName("output_terminal")
        self.output_terminal.setReadOnly(True)
        self.output_terminal.setMaximumBlockCount(10000)
        self.output_terminal.setPlaceholderText("MeshCFD messages and solver residual history")
        self.output_terminal.setStyleSheet(
            "QPlainTextEdit { background: #0F172A; color: #E2E8F0; "
            "font-family: Consolas, monospace; font-size: 9pt; }"
        )
        self.output_dock = self._add_dock(
            "Output", Qt.BottomDockWidgetArea, self.output_terminal
        )
        self.output_dock.setMinimumHeight(120)
        self.resizeDocks(
            [self.boundary_dock, self.initial_conditions_dock],
            [420, 450],
            Qt.Vertical,
        )
        self.resizeDocks(
            [self.boundary_dock, self.numerics_dock], [420, 400], Qt.Horizontal
        )
        self.resizeDocks([self.output_dock], [180], Qt.Vertical)
        self.visualization_panel = VisualizationPanel(self)
        self.boundary_panel.set_case(self.case)
        self.numerics_panel.set_case(self.case)
        self.initial_conditions_panel.set_case(self.case)

        self.status_label = QLabel("Ready")
        self.statusBar().addWidget(self.status_label)
        self.mesh_info_label = QLabel("No mesh loaded")
        self.statusBar().addPermanentWidget(self.mesh_info_label)

        self.boundary_panel.changed.connect(self._mark_dirty)
        self.boundary_panel.changed.connect(self._clear_simulation_solution)
        self.boundary_panel.patch_selected.connect(self.canvas.select_patch)
        self.numerics_panel.changed.connect(self._mark_dirty)
        self.numerics_panel.changed.connect(self._update_visualization)
        self.numerics_panel.turbulence_model_combo.currentTextChanged.connect(
            self.boundary_panel.refresh_value_fields
        )
        self.numerics_panel.live_solution_changed.connect(
            self._set_live_solution_updates
        )
        self.initial_conditions_panel.changed.connect(self._mark_dirty)
        self.initial_conditions_panel.changed.connect(self._refresh_solution)
        self.visualization_panel.changed.connect(self._update_visualization)
        self.canvas.cell_selected.connect(self._show_cell_info)

        self._create_actions()
        self._create_toolbar()
        self._create_menus()
        self._update_title()
        self._log_output("MeshCFD ready.")

    def _add_dock(
        self, title: str, area, widget, allowed_areas=None
    ) -> QDockWidget:
        dock = QDockWidget(title, self)
        dock.setObjectName(title.lower().replace(" ", "_"))
        dock.setWidget(widget)
        dock.setAllowedAreas(
            allowed_areas
            if allowed_areas is not None
            else Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea
        )
        self.addDockWidget(area, dock)
        return dock

    def _log_output(self, message: str, level: str = "INFO") -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.output_terminal.appendPlainText(
            f"{timestamp} | {level:<5} | {message}"
        )

    def _set_live_solution_updates(self, enabled: bool) -> None:
        self._live_solution_updates = enabled
        state = "enabled" if enabled else "disabled"
        self._log_output(f"Live solution updates {state} for future runs.")

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

        self.action_stop = QAction("Stop Solver", self)
        self.action_stop.setEnabled(False)
        self.action_stop.triggered.connect(self._on_stop)

        self.action_grid = QAction("Show Grid", self)
        self.action_grid.setCheckable(True)
        self.action_grid.setChecked(True)
        self.action_grid.triggered.connect(self._on_toggle_grid)

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
        simulation_menu.addAction(self.action_stop)

        view_menu = self.menuBar().addMenu("&View")
        view_menu.addAction(self.action_grid)
        view_menu.addSeparator()
        view_menu.addAction(self.action_fit_mesh)
        view_menu.addAction(self.boundary_dock.toggleViewAction())
        view_menu.addAction(self.numerics_dock.toggleViewAction())
        view_menu.addAction(self.initial_conditions_dock.toggleViewAction())
        view_menu.addAction(self.visualization_toolbar.toggleViewAction())
        view_menu.addAction(self.output_dock.toggleViewAction())
        self.action_clear_output = QAction("Clear Output", self)
        self.action_clear_output.triggered.connect(self.output_terminal.clear)
        view_menu.addAction(self.action_clear_output)

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
        ):
            action.setIcon(self.style().standardIcon(icon))
            action.setToolTip(action.text())
            action.setStatusTip(action.text())
            toolbar.addAction(action)
        self.action_grid.setIcon(self._create_grid_icon(icon_size))
        self.action_grid.setToolTip(self.action_grid.text())
        self.action_grid.setStatusTip(self.action_grid.text())
        toolbar.addAction(self.action_grid)
        self.action_run.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.action_run.setToolTip(self.action_run.text())
        self.action_run.setStatusTip(self.action_run.text())
        toolbar.addAction(self.action_run)
        self.action_stop.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))
        self.action_stop.setToolTip(self.action_stop.text())
        self.action_stop.setStatusTip(self.action_stop.text())
        toolbar.addAction(self.action_stop)
        toolbar.addSeparator()

        self.visualization_toolbar = QToolBar("Solution View", self)
        self.visualization_toolbar.setObjectName("solution_view_toolbar")
        self.visualization_toolbar.setMovable(True)
        self.visualization_toolbar.setFloatable(True)
        self.visualization_toolbar.setStyleSheet(
            "QToolBar { spacing: 4px; padding: 2px; background: #F8FAFC; "
            "border-bottom: 1px solid #CBD5E1; }"
        )
        self.addToolBarBreak(Qt.TopToolBarArea)
        self.addToolBar(Qt.TopToolBarArea, self.visualization_toolbar)
        self.visualization_toolbar.addWidget(self.visualization_panel)

    @staticmethod
    def _create_grid_icon(size: int) -> QIcon:
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        margin = max(2, size // 8)
        grid = pixmap.rect().adjusted(margin, margin, -margin, -margin)
        painter.setPen(QPen(QColor("#475569"), max(1, size // 12)))
        painter.drawRoundedRect(grid, 2, 2)
        step = grid.width() / 3
        for index in (1, 2):
            position = round(grid.left() + step * index)
            painter.setPen(QPen(QColor("#2563EB"), max(1, size // 14)))
            painter.drawLine(position, grid.top() + 1, position, grid.bottom() - 1)
            painter.drawLine(grid.left() + 1, position, grid.right() - 1, position)
        painter.end()
        return QIcon(pixmap)

    def _update_title(self) -> None:
        filename = self.current_path.name if self.current_path else self.case.name
        marker = "*" if self._dirty else ""
        self.setWindowTitle(f"MeshCFD - {filename}{marker}")

    def _mark_dirty(self) -> None:
        self._case_revision += 1
        self._dirty = True
        self._update_title()

    def _set_case(self, case: Case, path: Path | None) -> None:
        self._case_revision += 1
        self.case = case
        self.current_path = path
        self._simulation_solution = None
        self._dirty = False
        self.canvas.set_mesh(case.mesh)
        self.boundary_panel.set_case(case)
        self.numerics_panel.set_case(case)
        self.initial_conditions_panel.set_case(case)
        self._update_mesh_info()
        self._refresh_solution(preserve_saved_solution=True)
        self._update_title()
        self._log_output(f"Loaded case settings: {case.name}.")

    def _on_toggle_grid(self, _checked: bool) -> None:
        self.canvas.show_grid = _checked
        self.canvas.viewport().update()
        self._update_visualization()

    def _refresh_solution(self, preserve_saved_solution: bool = False) -> None:
        self._simulation_solution = (
            self.case.final_solution if preserve_saved_solution else None
        )
        if not preserve_saved_solution:
            self.case.final_solution = None
            self.case.solution_time = 0.0
            self.case.solution_steps = 0
        if self.case.mesh is None:
            self._initial_solution = None
            self.canvas.set_solution(None)
            self._update_visualization()
            return
        try:
            self._initial_solution = evaluate_initial_conditions(
                self.case.mesh, self.case.initial_conditions
            )
        except ValueError as error:
            self._initial_solution = None
            self.canvas.set_solution(None)
            self._update_visualization()
            self._log_output(
                f"Initial solution preview could not be updated: {error}",
                "WARNING",
            )
            self.visualization_panel.set_result(
                None,
                self.visualization_panel.palette_combo.currentText(),
                scale=self.visualization_panel.scale_combo.currentText(),
                error=str(error),
            )
            return
        self._update_visualization()

    def _clear_simulation_solution(self) -> None:
        self._simulation_solution = None
        self.case.final_solution = None
        self.case.solution_time = 0.0
        self.case.solution_steps = 0
        self._update_visualization()

    def _update_visualization(self, *_) -> None:
        variable = self.visualization_panel.variable_combo.currentText()
        palette = self.visualization_panel.palette_combo.currentText()
        solution = self._simulation_solution or self._initial_solution
        self.canvas.set_interaction_mode(
            self.visualization_panel.interaction_mode
        )
        try:
            derived_fields = (
                derive_solution_fields(self.case, solution)
                if solution is not None
                else {}
            )
            self.canvas.set_solution(solution, derived_fields)
            values = self.canvas.set_display_options(
                show_grid=self.action_grid.isChecked(),
                show_mesh=self.visualization_panel.show_mesh_checkbox.isChecked(),
                solution_variable=variable,
                color_map=palette,
                color_scale=self.visualization_panel.scale_combo.currentText(),
                color_range=self.visualization_panel.color_range(),
                smooth_colors=self.visualization_panel.smooth_colors_checkbox.isChecked(),
                show_colorbar=self.visualization_panel.show_colorbar_checkbox.isChecked(),
                colorbar_orientation=self.visualization_panel.colorbar_orientation(),
                colorbar_variable=variable,
            )
        except ValueError as error:
            self.canvas.set_solution(None)
            self.canvas.set_display_options(
                show_grid=self.action_grid.isChecked(),
                show_mesh=self.visualization_panel.show_mesh_checkbox.isChecked(),
                solution_variable=variable,
                color_map=palette,
                color_scale=self.visualization_panel.scale_combo.currentText(),
                color_range=None,
                smooth_colors=self.visualization_panel.smooth_colors_checkbox.isChecked(),
                show_colorbar=self.visualization_panel.show_colorbar_checkbox.isChecked(),
                colorbar_orientation=self.visualization_panel.colorbar_orientation(),
                colorbar_variable=variable,
            )
            self.visualization_panel.set_result(
                None,
                palette,
                error=str(error),
                scale=self.visualization_panel.scale_combo.currentText(),
            )
            return
        self.visualization_panel.set_result(
            values,
            palette,
            scale=self.visualization_panel.scale_combo.currentText(),
        )

    def _show_cell_info(
        self,
        cell_id: int,
        center_x: float,
        center_y: float,
        variable: str,
        value: float | None,
    ) -> None:
        value_text = "unavailable" if value is None else f"{value:.6g}"
        message = (
            f"Cell {cell_id}\n"
            f"x = {center_x:.6g}, y = {center_y:.6g}\n"
            f"{variable} = {value_text}"
        )
        self.status_label.setText(
            message.replace("\n", " | ")
        )
        QToolTip.showText(
            QCursor.pos(), message, self.canvas.viewport(), QRect(), 5000
        )

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
            self._log_output("Created a new case.")

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
            self._log_output(f"Could not open case: {error}", "ERROR")
            QMessageBox.critical(self, "Open Case Error", str(error))
            return
        self._set_case(case, path)
        self.status_label.setText(f"Opened case: {path.name}")
        self._log_output(f"Opened case: {path}.")

    def _on_save(self) -> None:
        if self.current_path is None:
            self._on_save_as()
            return
        try:
            save_case(self.current_path, self.case)
        except (OSError, ValueError) as error:
            self._log_output(f"Could not save case: {error}", "ERROR")
            QMessageBox.critical(self, "Save Case Error", str(error))
            return
        self._dirty = False
        self._update_title()
        self.status_label.setText(f"Saved case: {self.current_path.name}")
        self._log_output(f"Saved case: {self.current_path}.")

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
            self._log_output(f"Could not save case: {error}", "ERROR")
            QMessageBox.critical(self, "Save Case Error", str(error))
            return
        self.current_path = path
        self._dirty = False
        self._update_title()
        self.status_label.setText(f"Saved case: {path.name}")
        self._log_output(f"Saved case: {path}.")

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
            self._log_output(f"Could not import mesh: {error}", "ERROR")
            QMessageBox.critical(self, "Mesh Import Error", str(error))
            return
        self.case.set_mesh(mesh, Path(filename))
        self.canvas.set_mesh(mesh)
        self.boundary_panel.set_case(self.case)
        self._update_mesh_info()
        self._refresh_solution()
        self._mark_dirty()
        self.status_label.setText(f"Imported mesh: {Path(filename).name}")
        self._log_output(
            f"Imported mesh {Path(filename)}: {len(mesh.cells)} cells, "
            f"{len(mesh.faces)} faces, {len(mesh.patches)} patches."
        )

    def _on_run(self) -> None:
        if self._solver_worker is not None:
            if self._pause_requested:
                self._pause_requested = False
                self._solver_worker.resume()
                self._set_run_action("Pause Solver", QStyle.SP_MediaPause)
                self.status_label.setText("Solver resuming...")
                self._log_output("Resuming solver.")
            else:
                self._pause_requested = True
                self._solver_worker.pause()
                self._set_run_action("Resume Solver", QStyle.SP_MediaPlay)
                self.status_label.setText("Pausing after the current time step...")
                self._log_output(
                    "Solver pause requested; waiting for the current time step."
                )
            return
        if self.case.mesh is None:
            self._log_output("Cannot start solver: no mesh is loaded.", "WARNING")
            QMessageBox.information(
                self, "No Mesh", "Import a finite-volume mesh before setting up a run."
            )
            return
        snapshot = copy.deepcopy(self.case)
        initial_states = copy.deepcopy(
            self._simulation_solution or self._initial_solution
        )
        self._run_revision = self._case_revision
        self._run_case_ref = self.case
        worker = _SolverWorker(
            snapshot, initial_states, self._live_solution_updates
        )
        self._solver_worker = worker
        worker.progress.connect(self._on_solver_progress)
        worker.pause_state_changed.connect(self._on_solver_pause_state_changed)
        worker.finished.connect(self._on_solver_finished)
        self._pause_requested = False
        self._set_run_action("Pause Solver", QStyle.SP_MediaPause)
        self.action_stop.setEnabled(True)
        self.status_label.setText("Solver running...")
        self.output_dock.show()
        self._log_output(
            f"Starting {snapshot.numerics.flux} solver with "
            f"{snapshot.numerics.time_integration}, "
            f"{snapshot.numerics.reconstruction}, "
            f"starting at t={snapshot.solution_time:g} "
            f"(step {snapshot.solution_steps}), "
            f"CFL={snapshot.numerics.cfl:g}, "
            f"end time={snapshot.numerics.end_time:g}, "
            f"maximum steps={snapshot.numerics.max_iterations}."
        )
        self._log_output(
            " Iteration          Time            dt          rho         rhoU"
            "         rhoV         rhoE"
        )
        worker.start()

    def _set_run_action(self, text: str, icon: QStyle.StandardPixmap) -> None:
        self.action_run.setText(text)
        self.action_run.setToolTip(text)
        self.action_run.setStatusTip(text)
        self.action_run.setIcon(self.style().standardIcon(icon))

    def _on_stop(self) -> None:
        worker = self._solver_worker
        if worker is None:
            return
        self.action_stop.setEnabled(False)
        self.action_run.setEnabled(False)
        self.status_label.setText("Stopping solver...")
        self._log_output("Stopping solver after the current time step.")
        worker.stop()

    def _on_solver_pause_state_changed(self, paused: bool) -> None:
        worker = self._solver_worker
        if paused and worker is not None and not worker.stop_requested:
            self.status_label.setText("Solver paused.")
            self._log_output("Solver paused at a time-step boundary.")
        elif (
            worker is not None and not worker.stop_requested
        ):
            self.status_label.setText("Solver running...")
            self._log_output("Solver resumed.")

    def _on_solver_progress(self, progress: SolverProgress) -> None:
        density, momentum_x, momentum_y, energy = progress.residuals
        worker = self._solver_worker
        self._log_output(
            f"{progress.step:10d} {progress.time:12.5e} {progress.time_step:12.5e}"
            f" {density:12.5e} {momentum_x:12.5e} {momentum_y:12.5e}"
            f" {energy:12.5e}",
            "ITER",
        )
        if worker is not None and not worker.stop_requested:
            self.status_label.setText(
                f"Solver running: step {progress.step}, t = {progress.time:.6g}"
            )
        if (
            progress.states is not None
            and self._run_revision == self._case_revision
        ):
            self._simulation_solution = progress.states
            self._update_visualization()

    def _on_solver_finished(self) -> None:
        worker = self._solver_worker
        if worker is None:
            return
        stopped_with_current_mesh = (
            worker.stop_requested
            and self.case is self._run_case_ref
            and self.case.mesh == worker.case.mesh
        )
        self._solver_worker = None
        self._pause_requested = False
        self._run_case_ref = None
        self._set_run_action("Run Solver", QStyle.SP_MediaPlay)
        self.action_run.setEnabled(True)
        self.action_stop.setEnabled(False)
        worker.deleteLater()
        if worker.error is not None:
            self.status_label.setText("Solver failed.")
            self._log_output(f"Solver failed: {worker.error}", "ERROR")
            QMessageBox.critical(self, "Solver Error", worker.error)
            return
        if (
            self._run_revision != self._case_revision
            and not stopped_with_current_mesh
        ):
            self.status_label.setText(
                "Run finished for older case settings; results were discarded."
            )
            self._log_output(
                "Run finished for older case settings; results were discarded.",
                "WARNING",
            )
            return
        result = worker.result
        if result is None:
            self.status_label.setText("Solver failed without a result.")
            self._log_output("Solver finished without producing a result.", "ERROR")
            QMessageBox.critical(
                self, "Solver Error", "The solver finished without producing a result."
            )
            return
        self._simulation_solution = result.states
        self.case.final_solution = result.states
        self.case.solution_time = result.time
        self.case.solution_steps = result.steps
        self._update_visualization()
        self._mark_dirty()
        if result.completed:
            message = (
                f"Completed at t = {result.time:.6g} in {result.steps} steps "
                f"(residual {result.residual:.3e})."
            )
            self.status_label.setText(message)
            self._log_output(message)
        else:
            message = (
                f"Stopped at t = {result.time:.6g} after {result.steps} steps "
                f"before end time {self.case.numerics.end_time:.6g}."
            )
            self.status_label.setText(message)
            self._log_output(message, "WARNING")

    def closeEvent(self, event) -> None:
        if self._confirm_discard_changes():
            if self._solver_worker is not None:
                self._solver_worker.stop()
                self._solver_worker.wait()
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
