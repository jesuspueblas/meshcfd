"""MeshCFD case setup application window."""

from __future__ import annotations

import copy
import sys
from datetime import datetime
from pathlib import Path
from threading import Event
from typing import Callable

from PyQt5.QtCore import QRect, QSize, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QCursor, QIcon, QKeySequence, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QDockWidget,
    QFileDialog,
    QFrame,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QScrollArea,
    QSizePolicy,
    QStyle,
    QToolBar,
    QToolTip,
    QWidget,
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
from meshcfd.gui.canvas import MeshCanvas, available_solution_variables
from meshcfd.gui.initial_conditions_panel import InitialConditionsPanel
from meshcfd.gui.numerics_panel import NumericsPanel
from meshcfd.gui.results import ConvergenceHistoryWidget, CutProfileWidget
from meshcfd.gui.visualization_panel import VisualizationPanel
from meshcfd.io.mesh_io import MeshImportError, load_mesh
from meshcfd.io.project_io import load_case, save_case
from meshcfd.io.vtk_io import VTKSeriesWriter, export_vtu


class _SolverWorker(QThread):
    progress = pyqtSignal(object)
    pause_state_changed = pyqtSignal(bool)

    def __init__(
        self,
        case: Case,
        initial_states: tuple[InitialState, ...] | None,
        live_solution_updates: bool,
        vtk_series_path: Path | None = None,
        vtk_interval: int = 100,
        cpu_time_offset: float = 0.0,
    ) -> None:
        super().__init__()
        self.case = case
        self.initial_states = initial_states
        self.live_solution_updates = live_solution_updates
        self.vtk_series_path = vtk_series_path
        self.vtk_interval = vtk_interval
        self.cpu_time_offset = cpu_time_offset
        self.result: SolverResult | None = None
        self.error: str | None = None
        self.output_error: str | None = None
        self._resume_event = Event()
        self._resume_event.set()
        self._stop_requested = Event()

    def run(self) -> None:
        try:
            series_writer: VTKSeriesWriter | None = None
            solution_callback: Callable[[int, float, tuple[InitialState, ...]], None] | None = None
            if self.vtk_series_path is not None:
                try:
                    series_writer = VTKSeriesWriter(
                        self.vtk_series_path, append=self.case.solution_steps > 0
                    )
                    if self.initial_states is not None:
                        series_writer.add_snapshot(
                            self.case,
                            self.initial_states,
                            self.case.solution_time,
                            self.case.solution_steps,
                        )
                except (OSError, ValueError) as error:
                    self.output_error = str(error)
                    series_writer = None

                if series_writer is not None:
                    writer = series_writer

                    def write_snapshot(
                        step: int, time: float, states: tuple[InitialState, ...]
                    ) -> None:
                        try:
                            writer.add_snapshot(self.case, states, time, step)
                        except (OSError, ValueError) as error:
                            self.output_error = str(error)
                            self._stop_requested.set()
                            self.requestInterruption()
                            self.resume()

                    solution_callback = write_snapshot
            self.result = solve_case(
                self.case,
                self.initial_states,
                self._report_progress,
                self.isInterruptionRequested,
                self.live_solution_updates,
                self.case.solution_time,
                self.case.solution_steps,
                self._pause_at_checkpoint,
                **(
                    {
                        "solution_callback": solution_callback,
                        "solution_interval": self.vtk_interval,
                    }
                    if solution_callback is not None
                    else {}
                ),
                cpu_time_offset=self.cpu_time_offset,
            )
            if series_writer is not None and self.result is not None:
                if self.output_error is None:
                    try:
                        series_writer.add_snapshot(
                            self.case,
                            self.result.states,
                            self.result.time,
                            self.result.steps,
                        )
                    except (OSError, ValueError) as error:
                        self.output_error = str(error)
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
        self._latest_solution_progress: SolverProgress | None = None
        self._pause_requested = False
        self._live_solution_updates = False
        self._vtk_series_path: Path | None = None
        self._vtk_interval = 100
        self._convergence_history: list[
            tuple[int, float, float, tuple[float, ...]]
        ] = []
        self._cpu_time_offset = 0.0
        self._convergence_uses_pseudo_time = False
        self._dirty = False
        self.setMinimumSize(640, 480)
        self.resize(1400, 850)

        self.canvas = MeshCanvas(self)
        self.setCentralWidget(self.canvas)
        self.setDockNestingEnabled(True)

        self.boundary_panel = BoundaryPanel(self)
        self.boundary_scroll = self._create_scroll_area(self.boundary_panel)
        self.boundary_dock = self._add_dock(
            "Boundary Conditions", Qt.LeftDockWidgetArea, self.boundary_scroll
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
        self.convergence_history_widget = ConvergenceHistoryWidget(
            lambda: tuple(self._convergence_history), self
        )
        self.numerics_panel.changed.connect(self._update_convergence_time_axis)
        self.convergence_history_scroll = self._create_scroll_area(
            self.convergence_history_widget
        )
        self.convergence_history_dock = self._add_dock(
            "Convergence History",
            Qt.BottomDockWidgetArea,
            self.convergence_history_scroll,
            Qt.AllDockWidgetAreas,
        )
        self.cut_profile_widget = CutProfileWidget(
            self.visualization_panel.variable_combo.currentText(), self
        )
        self.cut_profile_scroll = self._create_scroll_area(self.cut_profile_widget)
        self.cut_profile_dock = self._add_dock(
            "Solution Cut",
            Qt.RightDockWidgetArea,
            self.cut_profile_scroll,
            Qt.AllDockWidgetAreas,
        )
        self.convergence_history_dock.hide()
        self.cut_profile_dock.hide()
        self._cut_profile_active = False
        self.cut_profile_widget.cut_line_changed.connect(
            self._on_cut_profile_line_changed
        )
        self.cut_profile_dock.visibilityChanged.connect(
            self._on_cut_profile_visibility_changed
        )
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
        self._update_visualization()
        self._update_results_actions()
        self._update_title()
        self._log_output("MeshCFD ready.")

    def _add_dock(
        self, title: str, area, widget, allowed_areas=None
    ) -> QDockWidget:
        dock = QDockWidget(title, self)
        dock.setObjectName(title.lower().replace(" ", "_"))
        dock.setWidget(widget)
        dock.setMinimumSize(0, 0)
        dock.setAllowedAreas(
            allowed_areas
            if allowed_areas is not None
            else Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea
        )
        self.addDockWidget(area, dock)
        return dock

    def _create_scroll_area(self, widget: QWidget) -> QScrollArea:
        scroll = QScrollArea(self)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidgetResizable(True)
        scroll.setMinimumSize(0, 0)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setWidget(widget)
        return scroll

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

        self.action_export_vtu = QAction("Export Current Solution to VTU...", self)
        self.action_export_vtu.triggered.connect(self._on_export_vtu)
        self.action_configure_vtk_series = QAction(
            "Configure VTU/PVD Time-Series Export...", self
        )
        self.action_configure_vtk_series.triggered.connect(
            self._on_configure_vtk_series
        )
        self.action_disable_vtk_series = QAction(
            "Disable Time-Series Export", self
        )
        self.action_disable_vtk_series.triggered.connect(
            self._on_disable_vtk_series
        )
        self.action_clear_output = QAction("Clear Output", self)
        self.action_clear_output.triggered.connect(self.output_terminal.clear)
        self.action_exit = QAction("Exit", self)
        self.action_exit.triggered.connect(self.close)
        self.action_convergence_history = (
            self.convergence_history_dock.toggleViewAction()
        )
        self.action_solution_cut = self.cut_profile_dock.toggleViewAction()
        self.action_boundary_dock = self.boundary_dock.toggleViewAction()
        self.action_numerics_dock = self.numerics_dock.toggleViewAction()
        self.action_initial_conditions_dock = (
            self.initial_conditions_dock.toggleViewAction()
        )
        self.action_output_dock = self.output_dock.toggleViewAction()

        for action, icon in (
            (self.action_new, QStyle.SP_FileIcon),
            (self.action_open, QStyle.SP_DialogOpenButton),
            (self.action_save, QStyle.SP_DialogSaveButton),
            (self.action_save_as, QStyle.SP_DialogSaveButton),
            (self.action_import_mesh, QStyle.SP_DirOpenIcon),
            (self.action_fit_mesh, QStyle.SP_DesktopIcon),
            (self.action_run, QStyle.SP_MediaPlay),
            (self.action_stop, QStyle.SP_MediaStop),
            (self.action_export_vtu, QStyle.SP_ArrowDown),
            (self.action_configure_vtk_series, QStyle.SP_FileDialogDetailedView),
            (self.action_disable_vtk_series, QStyle.SP_DialogCancelButton),
            (self.action_clear_output, QStyle.SP_TrashIcon),
            (self.action_exit, QStyle.SP_DialogCloseButton),
            (self.action_boundary_dock, QStyle.SP_DirIcon),
            (self.action_numerics_dock, QStyle.SP_ComputerIcon),
            (self.action_initial_conditions_dock, QStyle.SP_FileDialogInfoView),
            (self.action_output_dock, QStyle.SP_MessageBoxInformation),
            (self.action_convergence_history, QStyle.SP_ComputerIcon),
            (self.action_solution_cut, QStyle.SP_ArrowForward),
        ):
            action.setIcon(self.style().standardIcon(icon))

    def _create_menus(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        for action in (self.action_new, self.action_open, self.action_save, self.action_save_as):
            file_menu.addAction(action)
        file_menu.addSeparator()
        file_menu.addAction(self.action_import_mesh)
        file_menu.addSeparator()
        file_menu.addAction(self.action_exit)

        simulation_menu = self.menuBar().addMenu("&Simulation")
        simulation_menu.addAction(self.action_run)
        simulation_menu.addAction(self.action_stop)

        results_menu = self.menuBar().addMenu("&Results")
        results_menu.addAction(self.action_export_vtu)
        results_menu.addAction(self.action_configure_vtk_series)
        results_menu.addAction(self.action_disable_vtk_series)
        results_menu.addSeparator()
        results_menu.addAction(self.action_convergence_history)
        results_menu.addAction(self.action_solution_cut)

        view_menu = self.menuBar().addMenu("&View")
        view_menu.addAction(self.action_grid)
        view_menu.addSeparator()
        view_menu.addAction(self.action_fit_mesh)
        view_menu.addAction(self.action_boundary_dock)
        view_menu.addAction(self.action_numerics_dock)
        view_menu.addAction(self.action_initial_conditions_dock)
        view_menu.addAction(self.action_output_dock)
        view_menu.addAction(self.action_convergence_history)
        view_menu.addAction(self.action_solution_cut)
        view_menu.addSeparator()
        for toolbar in (
            self.case_toolbar,
            self.view_toolbar,
            self.visualization_toolbar,
        ):
            toggle_action = toolbar.toggleViewAction()
            toggle_action.setIcon(self.style().standardIcon(QStyle.SP_TitleBarShadeButton))
            view_menu.addAction(toggle_action)
        view_menu.addSeparator()
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
        for action in (
            self.action_new,
            self.action_open,
            self.action_save,
            self.action_import_mesh,
        ):
            action.setToolTip(action.text())
            action.setStatusTip(action.text())
            toolbar.addAction(action)
        self.action_run.setToolTip(self.action_run.text())
        self.action_run.setStatusTip(self.action_run.text())
        toolbar.addAction(self.action_run)
        self.action_stop.setToolTip(self.action_stop.text())
        self.action_stop.setStatusTip(self.action_stop.text())
        toolbar.addAction(self.action_stop)
        toolbar.addSeparator()

        self.view_toolbar = QToolBar("View", self)
        self.view_toolbar.setObjectName("view_toolbar")
        self.view_toolbar.setMovable(True)
        self.view_toolbar.setFloatable(False)
        self.view_toolbar.setIconSize(QSize(icon_size, icon_size))
        self.view_toolbar.setToolButtonStyle(Qt.ToolButtonIconOnly)
        self.view_toolbar.setStyleSheet(toolbar.styleSheet())
        self.addToolBar(Qt.TopToolBarArea, self.view_toolbar)
        self.action_grid.setIcon(self._create_grid_icon(icon_size))
        self.action_grid.setToolTip(self.action_grid.text())
        self.action_grid.setStatusTip(self.action_grid.text())
        for action in (self.action_grid, self.action_fit_mesh):
            self.view_toolbar.addAction(action)
        self.view_toolbar.addSeparator()
        for action in (
            self.action_boundary_dock,
            self.action_numerics_dock,
            self.action_initial_conditions_dock,
            self.action_output_dock,
            self.action_convergence_history,
            self.action_solution_cut,
        ):
            self.view_toolbar.addAction(action)
        self.view_toolbar.addSeparator()
        self.view_toolbar.addAction(self.action_clear_output)

        for action in (
            self.action_boundary_dock,
            self.action_numerics_dock,
            self.action_initial_conditions_dock,
            self.action_output_dock,
            self.action_convergence_history,
            self.action_solution_cut,
            self.action_clear_output,
        ):
            action.setToolTip(action.text())
            action.setStatusTip(action.text())

        self.addToolBarBreak(Qt.TopToolBarArea)
        self.visualization_toolbar = QToolBar("Solution View", self)
        self.visualization_toolbar.setObjectName("solution_view_toolbar")
        self.visualization_toolbar.setMovable(True)
        self.visualization_toolbar.setFloatable(True)
        self.visualization_toolbar.setStyleSheet(
            "QToolBar { spacing: 4px; padding: 2px; background: #F8FAFC; "
            "border-bottom: 1px solid #CBD5E1; }"
        )
        self.addToolBar(Qt.TopToolBarArea, self.visualization_toolbar)
        self.visualization_scroll = QScrollArea(self)
        self.visualization_scroll.setFrameShape(QFrame.NoFrame)
        self.visualization_scroll.setWidgetResizable(False)
        self.visualization_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.visualization_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.visualization_scroll.setMinimumSize(0, 0)
        self.visualization_scroll.setMaximumHeight(60)
        self.visualization_scroll.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Fixed
        )
        self.visualization_scroll.setWidget(self.visualization_panel)
        self.visualization_toolbar.addWidget(self.visualization_scroll)

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
        self._convergence_uses_pseudo_time = (
            case.numerics.time_step_mode == "Local (steady)"
        )
        self._convergence_history = list(case.convergence_history)
        self._cpu_time_offset = case.solution_cpu_time
        self._simulation_solution = None
        self._latest_solution_progress = None
        self._dirty = False
        self.canvas.set_mesh(case.mesh)
        self.boundary_panel.set_case(case)
        self.numerics_panel.set_case(case)
        self.initial_conditions_panel.set_case(case)
        self._update_mesh_info()
        self._refresh_solution(preserve_saved_solution=True)
        self._update_convergence_time_axis()
        self._update_results_actions()
        self._update_title()
        self._log_output(f"Loaded case settings: {case.name}.")

    def _update_convergence_time_axis(self, *_args) -> None:
        uses_pseudo_time = self.case.numerics.time_step_mode == "Local (steady)"
        if (
            uses_pseudo_time != self._convergence_uses_pseudo_time
            and self.case.final_solution is not None
        ):
            self.case.solution_time = 0.0
            self.case.solution_steps = 0
            self.case.convergence_history.clear()
            self._convergence_history.clear()
            self._cpu_time_offset = self.case.solution_cpu_time
            self._log_output(
                "Time-stepping mode changed between physical and pseudo-time; "
                "the existing solution is retained as the new initial state and "
                "time/iteration history has been reset."
            )
        self._convergence_uses_pseudo_time = uses_pseudo_time
        self.convergence_history_widget.set_pseudo_time_mode(uses_pseudo_time)

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
            self.case.solution_cpu_time = 0.0
            self.case.convergence_history.clear()
            self._convergence_history.clear()
            self._cpu_time_offset = 0.0
            self.convergence_history_widget.refresh_plot()
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
        self.case.solution_cpu_time = 0.0
        self.case.convergence_history.clear()
        self._convergence_history.clear()
        self._cpu_time_offset = 0.0
        self.convergence_history_widget.refresh_plot()
        self._update_visualization()
        self._update_results_actions()

    def _update_visualization(self, *_) -> None:
        available_variables = available_solution_variables(
            self.case.numerics.turbulence_model,
            self.case.numerics.turbulence_omega_formulation,
            self.case.numerics.viscosity_model,
        )
        self.visualization_panel.set_available_variables(available_variables)
        self.cut_profile_widget.set_available_variables(available_variables)
        variable = self.visualization_panel.variable_combo.currentText()
        palette = self.visualization_panel.palette_combo.currentText()
        solution = self._simulation_solution or self._initial_solution
        self.convergence_history_widget.set_turbulence_enabled(
            self.case.numerics.turbulence_model == "Menter k-omega SST"
        )
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
            if self.cut_profile_dock.isVisible():
                self.cut_profile_widget.set_solution(
                    self.case, solution, variable
                )
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
        self._update_results_actions()

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

    def _update_results_actions(self) -> None:
        has_solution = (
            self._simulation_solution is not None
            or self.case.final_solution is not None
        )
        self.action_export_vtu.setEnabled(has_solution)
        self.action_solution_cut.setEnabled(self.case.mesh is not None)
        self.action_convergence_history.setEnabled(True)
        self.action_disable_vtk_series.setEnabled(self._vtk_series_path is not None)

    def _on_export_vtu(self) -> None:
        states = self._simulation_solution or self.case.final_solution
        if states is None:
            QMessageBox.information(
                self, "No Solution", "Run the solver before exporting a solution."
            )
            return
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Current Solution", "", "VTU files (*.vtu)"
        )
        if not filename:
            return
        path = Path(filename)
        if path.suffix.lower() != ".vtu":
            path = path.with_suffix(".vtu")
        try:
            solution_time = self.case.solution_time
            solution_step = self.case.solution_steps
            if (
                self._solver_worker is not None
                and self._latest_solution_progress is not None
            ):
                solution_time = self._latest_solution_progress.time
                solution_step = self._latest_solution_progress.step
            export_vtu(
                path, self.case, states, solution_time, solution_step
            )
        except (OSError, ValueError) as error:
            self._log_output(f"Could not export VTU solution: {error}", "ERROR")
            QMessageBox.critical(self, "VTU Export Error", str(error))
            return
        self._log_output(f"Exported solution to {path}.")
        self.status_label.setText(f"Exported solution: {path.name}")

    def _on_configure_vtk_series(self) -> None:
        filename, _ = QFileDialog.getSaveFileName(
            self, "Configure VTU/PVD Time-Series Export", "", "PVD files (*.pvd)"
        )
        if not filename:
            return
        interval, accepted = QInputDialog.getInt(
            self,
            "VTK Output Interval",
            "Write a snapshot every N solver iterations:",
            self._vtk_interval,
            1,
            10_000_000,
        )
        if not accepted:
            return
        path = Path(filename)
        if path.suffix.lower() != ".pvd":
            path = path.with_suffix(".pvd")
        self._vtk_series_path = path
        self._vtk_interval = interval
        self._update_results_actions()
        self._log_output(
            f"VTK time-series export configured: {path}, every {interval} steps. "
            "It will be used by the next solver run."
        )
        self.status_label.setText(f"VTK series configured: {path.name}")

    def _on_disable_vtk_series(self) -> None:
        if self._vtk_series_path is None:
            return
        self._log_output(f"Disabled VTK time-series export to {self._vtk_series_path}.")
        self._vtk_series_path = None
        self._update_results_actions()

    def _on_cut_profile_line_changed(
        self, point_x: float, point_y: float, angle_degrees: float
    ) -> None:
        if self._cut_profile_active:
            self.canvas.set_solution_cut(point_x, point_y, angle_degrees)

    def _on_cut_profile_visibility_changed(self, visible: bool) -> None:
        self._cut_profile_active = visible
        if not visible:
            self.canvas.clear_solution_cut()
            return
        states = self._simulation_solution or self.case.final_solution
        if states is None:
            states = self._initial_solution
        self.cut_profile_widget.set_solution(
            self.case,
            states,
            self.visualization_panel.variable_combo.currentText(),
        )
        self.canvas.set_solution_cut(
            self.cut_profile_widget.x_spin.value(),
            self.cut_profile_widget.y_spin.value(),
            self.cut_profile_widget.angle_spin.value(),
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
        self._latest_solution_progress = None
        worker = _SolverWorker(
            snapshot,
            initial_states,
            self._live_solution_updates,
            self._vtk_series_path,
            self._vtk_interval,
            self._cpu_time_offset,
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
        if snapshot.numerics.time_step_mode == "Local (steady)":
            run_limits = (
                f"starting pseudo-time={snapshot.solution_time:g} "
                f"(iteration {snapshot.solution_steps}), "
                f"CFL={snapshot.numerics.cfl:g}, "
                f"residual tolerance={snapshot.numerics.residual_tolerance:g}, "
                f"maximum iterations={snapshot.numerics.max_iterations}."
            )
        else:
            run_limits = (
                f"starting at t={snapshot.solution_time:g} "
                f"(step {snapshot.solution_steps}), "
                f"CFL={snapshot.numerics.cfl:g}, "
                f"end time={snapshot.numerics.end_time:g}, "
                f"maximum steps={snapshot.numerics.max_iterations}."
            )
        self._log_output(
            f"Starting {snapshot.numerics.flux} solver with "
            f"{snapshot.numerics.time_integration}, "
            f"{snapshot.numerics.time_step_mode.lower()} time stepping, "
            f"{snapshot.numerics.reconstruction}, {run_limits}"
        )
        time_label = (
            "Pseudo-time"
            if snapshot.numerics.time_step_mode == "Local (steady)"
            else "Time"
        )
        self._log_output(
            f" Iteration {time_label:>15}            dt          rho         rhoU"
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
        if progress.warning is not None:
            self._log_output(progress.warning, "WARNING")
        if progress.residuals:
            density, momentum_x, momentum_y, energy = progress.residuals[:4]
            self._convergence_history.append(
                (progress.step, progress.time, progress.cpu_time, progress.residuals)
            )
            if len(self._convergence_history) > 5000:
                self._convergence_history = self._convergence_history[::2]
            residual_message = (
                f"{progress.step:10d} {progress.time:12.5e} {progress.time_step:12.5e}"
                f" {density:12.5e} {momentum_x:12.5e} {momentum_y:12.5e}"
                f" {energy:12.5e}"
            )
            if len(progress.residuals) >= 6:
                residual_message += (
                    f" {progress.residuals[4]:12.5e}"
                    f" {progress.residuals[5]:12.5e}"
                )
            self._log_output(residual_message, "ITER")
        worker = self._solver_worker
        if worker is not None and not worker.stop_requested:
            time_label = (
                "pseudo-time"
                if self.case.numerics.time_step_mode == "Local (steady)"
                else "t"
            )
            self.status_label.setText(
                f"Solver running: step {progress.step}, "
                f"{time_label} = {progress.time:.6g}"
            )
        if (
            progress.states is not None
            and self._run_revision == self._case_revision
        ):
            if progress.mesh is not None:
                self.case.mesh = progress.mesh
                self.case.mesh_path = None
                self.canvas.set_mesh(progress.mesh)
                self._update_mesh_info()
                self._log_output(
                    f"Adaptive mesh updated at step {progress.step}: "
                    f"{len(progress.mesh.cells)} cells."
                )
            self._simulation_solution = progress.states
            self._latest_solution_progress = progress
            self._update_visualization()
        else:
            self._update_results_actions()

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
        self._cpu_time_offset = result.cpu_time
        if worker.case.mesh != self.case.mesh:
            self.case.mesh = worker.case.mesh
            self.case.mesh_path = None
            self.canvas.set_mesh(worker.case.mesh)
            self._update_mesh_info()
            self._log_output(
                f"Adaptive mesh finished with {len(worker.case.mesh.cells)} cells."
            )
        self._simulation_solution = result.states
        self._latest_solution_progress = None
        self.case.final_solution = result.states
        self.case.solution_time = result.time
        self.case.solution_steps = result.steps
        self.case.solution_cpu_time = result.cpu_time
        self.case.convergence_history = list(self._convergence_history)
        self._update_visualization()
        self._update_results_actions()
        self._mark_dirty()
        if result.completed:
            if self.case.numerics.time_step_mode == "Local (steady)":
                message = (
                    f"Steady solution converged in {result.steps} iterations "
                    f"(residual {result.residual:.3e})."
                )
            else:
                message = (
                    f"Completed at t = {result.time:.6g} in {result.steps} steps "
                    f"(residual {result.residual:.3e})."
                )
            self.status_label.setText(message)
            self._log_output(message)
        else:
            if self.case.numerics.time_step_mode == "Local (steady)":
                message = (
                    f"Steady solve reached the {result.steps}-iteration limit "
                    f"(residual {result.residual:.3e}) without reaching tolerance."
                )
            else:
                message = (
                    f"Stopped at t = {result.time:.6g} after {result.steps} steps "
                    f"before end time {self.case.numerics.end_time:.6g}."
                )
            self.status_label.setText(message)
            self._log_output(message, "WARNING")
        if worker.output_error is not None:
            warning = (
                "VTK time-series export failed; the available solver result was "
                f"retained. {worker.output_error}"
            )
            self._log_output(warning, "WARNING")
            QMessageBox.warning(self, "VTK Export Warning", warning)

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
