"""Dialogs and lightweight Qt plotting widgets for simulation results."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence

from PyQt5.QtCore import QPoint, QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QPainter, QPen, QPolygonF
from PyQt5.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QStyle,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from meshcfd.core.model import Case, InitialState
from meshcfd.core.post import sample_solution_cut
from meshcfd.gui.canvas import SOLUTION_VARIABLES

_SERIES_COLORS = ("#2563EB", "#DC2626", "#16A34A", "#9333EA")


class _PlotCanvas(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(0, 0)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.x_values: tuple[float, ...] = ()
        self.series: dict[str, tuple[float, ...]] = {}
        self.x_label = ""
        self.y_label = ""
        self.logarithmic_y = False
        self._view_bounds: tuple[float, float, float, float] | None = None
        self._full_bounds = (0.0, 1.0, 0.0, 1.0)
        self._drag_start: QPoint | None = None
        self._drag_last: QPoint | None = None
        self._drag_position: QPoint | None = None
        self._drag_mode: str | None = None
        self.setMouseTracking(True)

    def set_data(
        self,
        x_values: Sequence[float],
        series: Mapping[str, Sequence[float]],
        x_label: str,
        y_label: str,
        logarithmic_y: bool = False,
    ) -> None:
        if self.logarithmic_y != logarithmic_y:
            self._view_bounds = None
        self.x_values = tuple(float(value) for value in x_values)
        self.series = {
            name: tuple(float(value) for value in values) for name, values in series.items()
        }
        self.x_label = x_label
        self.y_label = y_label
        self.logarithmic_y = logarithmic_y
        self._update_full_bounds()
        self.update()

    def _update_full_bounds(self) -> None:
        points = [
            (x, y)
            for values in self.series.values()
            for x, y in zip(self.x_values, values)
            if math.isfinite(x) and math.isfinite(y) and (not self.logarithmic_y or y > 0.0)
        ]
        if not points:
            self._full_bounds = (0.0, 1.0, 0.0, 1.0)
            return
        x_min = min(x for x, _ in points)
        x_max = max(x for x, _ in points)
        y_values = [math.log10(y) if self.logarithmic_y else y for _, y in points]
        y_min, y_max = min(y_values), max(y_values)
        if x_max == x_min:
            x_padding = max(abs(x_min) * 0.05, 0.5)
        else:
            x_padding = (x_max - x_min) * 0.05
        if y_max == y_min:
            y_padding = max(abs(y_min) * 0.05, 0.5)
        else:
            y_padding = (y_max - y_min) * 0.05
        self._full_bounds = (
            x_min - x_padding,
            x_max + x_padding,
            y_min - y_padding,
            y_max + y_padding,
        )
        if self._view_bounds is None:
            self._view_bounds = self._full_bounds

    def reset_zoom(self) -> None:
        self._view_bounds = self._full_bounds
        self.update()

    def _plot_rect(self) -> QRectF:
        return QRectF(78, 22, max(1, self.width() - 102), max(1, self.height() - 84))

    def _active_bounds(self) -> tuple[float, float, float, float]:
        if self._view_bounds is not None:
            return self._view_bounds
        return self._full_bounds

    def _point_to_data(self, point: QPoint) -> tuple[float, float]:
        rect = self._plot_rect()
        x_min, x_max, y_min, y_max = self._active_bounds()
        x = x_min + (point.x() - rect.left()) / rect.width() * (x_max - x_min)
        y = y_max - (point.y() - rect.top()) / rect.height() * (y_max - y_min)
        return x, y

    def wheelEvent(self, event) -> None:
        rect = self._plot_rect()
        if not rect.contains(event.pos()):
            event.ignore()
            return
        x_min, x_max, y_min, y_max = self._active_bounds()
        anchor_x, anchor_y = self._point_to_data(event.pos())
        factor = 0.8 if event.angleDelta().y() > 0 else 1.25
        x_min = anchor_x + (x_min - anchor_x) * factor
        x_max = anchor_x + (x_max - anchor_x) * factor
        y_min = anchor_y + (y_min - anchor_y) * factor
        y_max = anchor_y + (y_max - anchor_y) * factor
        if x_max > x_min and y_max > y_min:
            self._view_bounds = (x_min, x_max, y_min, y_max)
            self.update()
        event.accept()

    def mousePressEvent(self, event) -> None:
        if (
            event.button() in (Qt.LeftButton, Qt.MiddleButton)
            and self._plot_rect().contains(event.pos())
        ):
            self._drag_start = event.pos()
            self._drag_last = event.pos()
            self._drag_position = event.pos()
            self._drag_mode = "Zoom" if event.button() == Qt.LeftButton else "Pan"
            if self._drag_mode == "Pan":
                self.setCursor(Qt.ClosedHandCursor)
            else:
                self.setCursor(Qt.CrossCursor)
            self.update()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._drag_start is None:
            super().mouseMoveEvent(event)
            return
        self._drag_position = event.pos()
        if self._drag_mode == "Pan" and self._drag_last is not None:
            rect = self._plot_rect()
            x_min, x_max, y_min, y_max = self._active_bounds()
            delta_x = (event.x() - self._drag_last.x()) / rect.width() * (x_max - x_min)
            delta_y = (event.y() - self._drag_last.y()) / rect.height() * (y_max - y_min)
            self._view_bounds = (
                x_min - delta_x,
                x_max - delta_x,
                y_min + delta_y,
                y_max + delta_y,
            )
            self._drag_last = event.pos()
        self.update()
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        if self._drag_start is not None and (
            (self._drag_mode == "Zoom" and event.button() == Qt.LeftButton)
            or (self._drag_mode == "Pan" and event.button() == Qt.MiddleButton)
        ):
            if self._drag_mode == "Zoom":
                selection = QRectF(QPointF(self._drag_start), QPointF(event.pos())).normalized()
                rect = self._plot_rect()
                selection = selection.intersected(rect)
                if selection.width() >= 4 and selection.height() >= 4:
                    first = self._point_to_data(
                        QPoint(round(selection.left()), round(selection.top()))
                    )
                    second = self._point_to_data(
                        QPoint(round(selection.right()), round(selection.bottom()))
                    )
                    self._view_bounds = (
                        min(first[0], second[0]),
                        max(first[0], second[0]),
                        min(first[1], second[1]),
                        max(first[1], second[1]),
                    )
            self._drag_start = None
            self._drag_last = None
            self._drag_position = None
            self._drag_mode = None
            self.setCursor(Qt.ArrowCursor)
            self.update()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#FFFFFF"))
        plot = self._plot_rect()
        points = [
            (x, y)
            for values in self.series.values()
            for x, y in zip(self.x_values, values)
            if math.isfinite(x) and math.isfinite(y) and (not self.logarithmic_y or y > 0.0)
        ]
        if not points:
            painter.setPen(QColor("#64748B"))
            painter.drawText(self.rect(), Qt.AlignCenter, "No data available")
            painter.end()
            return

        x_min, x_max, y_min, y_max = self._active_bounds()

        painter.setFont(self.font())
        painter.setPen(QPen(QColor("#CBD5E1"), 1))
        for index in range(6):
            fraction = index / 5
            x = plot.left() + fraction * plot.width()
            y = plot.top() + fraction * plot.height()
            painter.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()))
            painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
            painter.setPen(QColor("#475569"))
            painter.drawText(
                QRectF(x - 35, plot.bottom() + 5, 70, 18),
                Qt.AlignHCenter | Qt.AlignTop,
                f"{x_min + fraction * (x_max - x_min):.4g}",
            )
            displayed_y = y_max - fraction * (y_max - y_min)
            label = f"{10**displayed_y:.2g}" if self.logarithmic_y else f"{displayed_y:.3g}"
            painter.drawText(
                QRectF(0, y - 9, plot.left() - 9, 18),
                Qt.AlignRight | Qt.AlignVCenter,
                label,
            )
            painter.setPen(QPen(QColor("#CBD5E1"), 1))

        painter.setPen(QPen(QColor("#334155"), 1.2))
        painter.drawRect(plot)
        for series_index, (name, values) in enumerate(self.series.items()):
            color = QColor(_SERIES_COLORS[series_index % len(_SERIES_COLORS)])
            painter.setPen(QPen(color, 1.8))
            polygon = QPolygonF()
            for x, value in zip(self.x_values, values):
                if (
                    not math.isfinite(x)
                    or not math.isfinite(value)
                    or (self.logarithmic_y and value <= 0.0)
                ):
                    continue
                y_value = math.log10(value) if self.logarithmic_y else value
                polygon.append(
                    QPointF(
                        plot.left() + (x - x_min) / (x_max - x_min) * plot.width(),
                        plot.bottom() - (y_value - y_min) / (y_max - y_min) * plot.height(),
                    )
                )
            if len(polygon) > 1:
                painter.save()
                painter.setClipRect(plot, Qt.IntersectClip)
                painter.drawPolyline(polygon)
                painter.restore()
            painter.drawLine(
                QPointF(plot.right() - 150, plot.top() + 14 + 19 * series_index),
                QPointF(plot.right() - 130, plot.top() + 14 + 19 * series_index),
            )
            painter.drawText(
                QPointF(plot.right() - 124, plot.top() + 18 + 19 * series_index),
                name,
            )

        painter.setPen(QColor("#1E293B"))
        painter.drawText(
            QRectF(plot.left(), self.height() - 24, plot.width(), 20),
            Qt.AlignCenter,
            self.x_label,
        )
        painter.save()
        painter.translate(18, plot.center().y())
        painter.rotate(-90)
        painter.drawText(
            QRectF(-plot.height() / 2, -10, plot.height(), 20), Qt.AlignCenter, self.y_label
        )
        painter.restore()
        if self._drag_start is not None and self._drag_mode == "Zoom":
            painter.save()
            painter.setClipRect(plot, Qt.IntersectClip)
            painter.setPen(QPen(QColor("#2563EB"), 1.0, Qt.DashLine))
            painter.drawRect(
                QRectF(
                    QPointF(self._drag_start), QPointF(self._drag_position or self._drag_start)
                ).normalized()
            )
            painter.restore()
        painter.end()


class LinePlotWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(0, 0)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        self.canvas = _PlotCanvas(self)
        layout.addWidget(self.canvas, 1)

    @property
    def x_values(self) -> tuple[float, ...]:
        return self.canvas.x_values

    @property
    def series(self) -> dict[str, tuple[float, ...]]:
        return self.canvas.series

    @property
    def logarithmic_y(self) -> bool:
        return self.canvas.logarithmic_y

    def set_data(
        self,
        x_values: Sequence[float],
        series: Mapping[str, Sequence[float]],
        x_label: str,
        y_label: str,
        logarithmic_y: bool = False,
    ) -> None:
        self.canvas.set_data(x_values, series, x_label, y_label, logarithmic_y)

    def reset_zoom(self) -> None:
        self.canvas.reset_zoom()


class ConvergenceHistoryWidget(QWidget):
    def __init__(
        self,
        history_provider: Callable[
            [], Sequence[tuple[int, float, tuple[float, ...]]]
        ],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._history_provider = history_provider
        layout = QVBoxLayout(self)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Residual scale"))
        self.scale_combo = QComboBox(self)
        self.scale_combo.addItems(("Linear", "Logarithmic"))
        self.scale_combo.setCurrentText("Logarithmic")
        self.scale_combo.currentTextChanged.connect(self.refresh_plot)
        controls.addWidget(self.scale_combo)
        self.reset_zoom_button = QPushButton(
            self.style().standardIcon(QStyle.SP_BrowserReload), "Reset zoom", self
        )
        self.reset_zoom_button.clicked.connect(lambda: self.plot.reset_zoom())
        controls.addWidget(self.reset_zoom_button)
        controls.addStretch(1)
        layout.addLayout(controls)
        self.variable_tree = QTreeWidget(self)
        self.variable_tree.setHeaderHidden(True)
        self.variable_tree.setMaximumHeight(150)
        self.variable_items: dict[str, QTreeWidgetItem] = {}
        self._variable_indices = {
            "Mass": 0,
            "Momentum X": 1,
            "Momentum Y": 2,
            "Energy": 3,
            "Turbulent k": 4,
            "Turbulent omega": 5,
        }
        flow_group = QTreeWidgetItem(self.variable_tree, ["Flow"])
        turbulence_group = QTreeWidgetItem(
            self.variable_tree, ["Turbulence (SST)"]
        )
        for label in ("Mass", "Momentum X", "Momentum Y", "Energy"):
            self._add_variable_item(flow_group, label)
        for label in ("Turbulent k", "Turbulent omega"):
            self._add_variable_item(turbulence_group, label)
        self._turbulence_group = turbulence_group
        self._turbulence_enabled = False
        self._turbulence_group.setHidden(True)
        self.variable_tree.expandAll()
        self.variable_tree.itemChanged.connect(self.refresh_plot)
        layout.addWidget(self.variable_tree)
        self.plot = LinePlotWidget(self)
        layout.addWidget(self.plot)
        self._timer = QTimer(self)
        self._timer.setInterval(500)
        self._timer.timeout.connect(self.refresh_plot)
        self.refresh_plot()

    def _add_variable_item(
        self, parent: QTreeWidgetItem, label: str
    ) -> None:
        item = QTreeWidgetItem(parent, [label])
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
        item.setCheckState(0, Qt.Checked)
        self.variable_items[label] = item

    def set_turbulence_enabled(self, enabled: bool) -> None:
        self._turbulence_enabled = enabled
        self._turbulence_group.setHidden(not enabled)
        self.refresh_plot()

    def showEvent(self, event) -> None:
        self._timer.start()
        self.refresh_plot()
        super().showEvent(event)

    def hideEvent(self, event) -> None:
        self._timer.stop()
        super().hideEvent(event)

    def refresh_plot(self) -> None:
        history = self._history_provider()
        selected = [
            (index, label)
            for label, index in self._variable_indices.items()
            if self.variable_items[label].checkState(0) == Qt.Checked
            and (index < 4 or self._turbulence_enabled)
            and any(index < len(entry[2]) for entry in history)
        ]
        self.plot.set_data(
            [entry[0] for entry in history],
            {
                name: [
                    entry[2][index] if index < len(entry[2]) else math.nan
                    for entry in history
                ]
                for index, name in selected
            },
            "Iteration",
            "Residual",
            logarithmic_y=self.scale_combo.currentText() == "Logarithmic",
        )


class CutProfileWidget(QWidget):
    cut_line_changed = pyqtSignal(float, float, float)

    def __init__(
        self,
        selected_field: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._case: Case | None = None
        self._states: tuple[InitialState, ...] | None = None
        self._mesh_ref = None
        layout = QVBoxLayout(self)
        controls = QHBoxLayout()
        form = QFormLayout()
        self.field_combo = QComboBox(self)
        self.field_combo.addItems(SOLUTION_VARIABLES)
        if selected_field in SOLUTION_VARIABLES:
            self.field_combo.setCurrentText(selected_field)
        field_controls = QHBoxLayout()
        field_controls.addWidget(self.field_combo)
        self.reset_zoom_button = QPushButton(
            self.style().standardIcon(QStyle.SP_BrowserReload), "Reset zoom", self
        )
        self.reset_zoom_button.clicked.connect(lambda: self.plot.reset_zoom())
        field_controls.addWidget(self.reset_zoom_button)
        self.angle_spin = self._coordinate_spin()
        self.x_spin = self._coordinate_spin()
        self.y_spin = self._coordinate_spin()
        form.addRow("Field", field_controls)
        form.addRow("Angle (degrees)", self.angle_spin)
        form.addRow("Point X", self.x_spin)
        form.addRow("Point Y", self.y_spin)
        controls.addLayout(form)
        controls.addStretch(1)
        layout.addLayout(controls)
        self.status_label = QLabel("Load a solution to plot a cut.")
        layout.addWidget(self.status_label)
        self.plot = LinePlotWidget(self)
        layout.addWidget(self.plot)
        self.field_combo.currentTextChanged.connect(self.plot_cut)
        self.angle_spin.valueChanged.connect(self._on_cut_parameters_changed)
        self.x_spin.valueChanged.connect(self._on_cut_parameters_changed)
        self.y_spin.valueChanged.connect(self._on_cut_parameters_changed)

    def set_solution(
        self, case: Case, states: tuple[InitialState, ...] | None, selected_field: str
    ) -> None:
        mesh_changed = case.mesh is not self._mesh_ref
        self._case = case
        self._states = states
        if mesh_changed and selected_field in SOLUTION_VARIABLES:
            self.field_combo.setCurrentText(selected_field)
        mesh = case.mesh
        if mesh is not None and mesh_changed and mesh.points:
            self._mesh_ref = mesh
            self.x_spin.setValue(
                (min(x for x, _ in mesh.points) + max(x for x, _ in mesh.points)) / 2.0
            )
            self.y_spin.setValue(
                (min(y for _, y in mesh.points) + max(y for _, y in mesh.points)) / 2.0
            )
        self.plot_cut()

    def set_available_variables(self, variables: Sequence[str]) -> None:
        if tuple(
            self.field_combo.itemText(index)
            for index in range(self.field_combo.count())
        ) == tuple(variables):
            return
        selected = self.field_combo.currentText()
        signals_were_blocked = self.field_combo.blockSignals(True)
        self.field_combo.clear()
        self.field_combo.addItems(variables)
        self.field_combo.setCurrentText(
            selected if selected in variables else "Density"
        )
        self.field_combo.blockSignals(signals_were_blocked)

    @staticmethod
    def _coordinate_spin() -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(-1e12, 1e12)
        spin.setDecimals(8)
        return spin

    def plot_cut(self) -> None:
        if self._case is None or self._states is None:
            self.plot.set_data((), {}, "Distance along cut from reference point", "")
            self.status_label.setText("Load or run a solution to plot a cut.")
            return
        try:
            field_name = self.field_combo.currentText()
            samples = sample_solution_cut(
                self._case,
                self._states,
                SOLUTION_VARIABLES[field_name],
                self.angle_spin.value(),
                self.x_spin.value(),
                self.y_spin.value(),
            )
        except ValueError as error:
            self.plot.set_data((), {}, "Distance along cut from reference point", "")
            self.status_label.setText(str(error))
            return
        self.status_label.setText(
            f"{len(samples)} cell samples | angle {self.angle_spin.value():g} degrees"
        )
        self.plot.set_data(
            [sample[0] for sample in samples],
            {field_name: [sample[1] for sample in samples]},
            "Distance along cut from reference point",
            field_name,
        )

    def _on_cut_parameters_changed(self, _value: float) -> None:
        self._emit_cut_line()
        self.plot_cut()

    def _emit_cut_line(self) -> None:
        self.cut_line_changed.emit(
            self.x_spin.value(), self.y_spin.value(), self.angle_spin.value()
        )
