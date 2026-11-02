"""Compact controls for the mesh and initial-solution display."""

from __future__ import annotations

import math

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap
from PyQt5.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QToolButton,
    QWidget,
)

from meshcfd.gui.canvas import COLOR_MAPS, SOLUTION_VARIABLES


def _option_icon(kind: str) -> QIcon:
    size = 20
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    outline = QColor("#475569")
    accent = QColor("#2563EB")
    if kind == "mesh":
        painter.setPen(QPen(outline, 1.5))
        painter.drawRoundedRect(2, 2, 16, 16, 2, 2)
        painter.setPen(QPen(accent, 1.2))
        painter.drawLine(7, 3, 7, 17)
        painter.drawLine(13, 3, 13, 17)
        painter.drawLine(3, 7, 17, 7)
        painter.drawLine(3, 13, 17, 13)
    elif kind == "smooth":
        painter.setPen(QPen(outline, 1.2))
        painter.drawRoundedRect(2, 3, 16, 14, 2, 2)
        path = QPainterPath()
        path.moveTo(4, 14)
        path.cubicTo(7, 12, 8, 5, 11, 6)
        path.cubicTo(14, 7, 14, 11, 17, 5)
        painter.setPen(QPen(accent, 2.2))
        painter.drawPath(path)
    elif kind == "colorbar":
        colors = ("#6E40AA", "#417DE0", "#1AC7C2", "#7AD151", "#F9E721", "#D7191C")
        for index, color in enumerate(colors):
            painter.fillRect(3 + index * 2, 3, 3, 14, QColor(color))
        painter.setPen(QPen(outline, 1))
        painter.drawRect(2, 2, 15, 16)
    elif kind == "navigate":
        painter.setPen(QPen(outline, 1.5, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        cursor = QPainterPath()
        cursor.moveTo(4, 2)
        cursor.lineTo(5, 17)
        cursor.lineTo(9, 13)
        cursor.lineTo(12, 19)
        cursor.lineTo(15, 17)
        cursor.lineTo(12, 11)
        cursor.lineTo(18, 11)
        cursor.closeSubpath()
        painter.setBrush(QColor("#FFFFFF"))
        painter.drawPath(cursor)
    elif kind == "box_zoom":
        painter.setPen(QPen(outline, 1.5, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.drawRect(2, 3, 13, 13)
        painter.setPen(QPen(accent, 2))
        painter.drawLine(6, 8, 11, 8)
        painter.drawLine(9, 6, 9, 11)
        painter.setPen(QPen(outline, 2, Qt.SolidLine, Qt.RoundCap))
        painter.drawLine(14, 14, 19, 19)
    elif kind == "cell_info":
        painter.setPen(QPen(outline, 1.3))
        painter.drawRoundedRect(2, 2, 16, 16, 2, 2)
        painter.drawLine(7, 3, 7, 17)
        painter.drawLine(13, 3, 13, 17)
        painter.drawLine(3, 7, 17, 7)
        painter.drawLine(3, 13, 17, 13)
        painter.setPen(QPen(accent, 2, Qt.SolidLine, Qt.RoundCap))
        painter.drawPoint(16, 4)
        painter.drawLine(16, 8, 16, 14)
    painter.end()
    return QIcon(pixmap)


class VisualizationPanel(QWidget):
    changed = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 3, 8, 3)
        layout.setSpacing(7)
        self.interaction_mode_group = QButtonGroup(self)
        self.interaction_mode_group.setExclusive(True)
        self.interaction_mode_buttons: dict[str, QToolButton] = {}
        self.interaction_mode = "Navigate"
        for mode, icon, tooltip in (
            ("Navigate", "navigate", "Navigate: wheel zoom, middle-button pan"),
            ("Box Zoom", "box_zoom", "Box zoom: drag a rectangle with left button"),
            ("Cell Info", "cell_info", "Cell info: click a cell to inspect its values"),
        ):
            button = self._icon_toggle(icon, tooltip, checked=mode == "Navigate")
            button.setAccessibleName(mode)
            button.setToolTip(tooltip)
            self.interaction_mode_group.addButton(button)
            self.interaction_mode_buttons[mode] = button
            layout.addWidget(button)
            button.toggled.connect(
                lambda checked, selected_mode=mode: self._set_interaction_mode(
                    selected_mode, checked
                )
            )
        self._add_separator(layout)
        self.show_mesh_checkbox = self._icon_toggle(
            "mesh", "Show mesh edges", checked=True
        )
        self.smooth_colors_checkbox = self._icon_toggle(
            "smooth", "Smooth solution colors"
        )
        layout.addWidget(self.show_mesh_checkbox)
        layout.addWidget(self.smooth_colors_checkbox)
        self._add_separator(layout)

        self.variable_combo = self._combo(SOLUTION_VARIABLES, 140)
        self.variable_combo.setToolTip(
            "Includes Mach number, stagnation properties, and y+ at the first "
            "cell centers adjacent to no-slip walls."
        )
        self.palette_combo = self._combo(COLOR_MAPS, 105)
        self.palette_combo.setCurrentText("Rainbow")
        self.scale_combo = self._combo(("Linear", "Logarithmic"), 94)
        self._add_labeled_widget(layout, "Field", self.variable_combo)
        self._add_labeled_widget(layout, "Palette", self.palette_combo)
        self._add_labeled_widget(layout, "Scale", self.scale_combo)
        self._add_separator(layout)

        self.auto_range_checkbox = QCheckBox("Auto range", self)
        self.auto_range_checkbox.setChecked(True)
        self.minimum_spin = self._range_spin_box()
        self.maximum_spin = self._range_spin_box()
        self.maximum_spin.setValue(1.0)
        self.minimum_spin.setEnabled(False)
        self.maximum_spin.setEnabled(False)
        range_widget = QWidget(self)
        range_layout = QHBoxLayout(range_widget)
        range_layout.setContentsMargins(0, 0, 0, 0)
        range_layout.setSpacing(3)
        range_layout.addWidget(self.auto_range_checkbox)
        range_layout.addWidget(QLabel("Min", self))
        range_layout.addWidget(self.minimum_spin)
        range_layout.addWidget(QLabel("Max", self))
        range_layout.addWidget(self.maximum_spin)
        layout.addWidget(range_widget)
        self._add_separator(layout)

        self.show_colorbar_checkbox = self._icon_toggle(
            "colorbar", "Show color bar in plot", checked=True
        )
        layout.addWidget(self.show_colorbar_checkbox)
        self.orientation_combo = self._combo(("Horizontal", "Vertical"), 94)
        self.orientation_combo.setCurrentText("Vertical")
        self.orientation_combo.setToolTip("Color bar orientation")
        self.orientation_combo.setEnabled(self.show_colorbar_checkbox.isChecked())
        layout.addWidget(self.orientation_combo)
        layout.addStretch(1)

        self.status_label = QLabel(self)
        self.status_label.setStyleSheet("color: #B91C1C;")
        self.status_label.setMaximumWidth(220)
        self.status_label.setWordWrap(False)
        self.status_label.setVisible(False)
        layout.addWidget(self.status_label)
        self.setFixedHeight(42)

        self.show_mesh_checkbox.toggled.connect(self.changed.emit)
        self.smooth_colors_checkbox.toggled.connect(self.changed.emit)
        self.variable_combo.currentTextChanged.connect(self.changed.emit)
        self.palette_combo.currentTextChanged.connect(self.changed.emit)
        self.scale_combo.currentTextChanged.connect(self.changed.emit)
        self.auto_range_checkbox.toggled.connect(self._set_range_inputs_enabled)
        self.auto_range_checkbox.toggled.connect(self.changed.emit)
        self.show_colorbar_checkbox.toggled.connect(self.changed.emit)
        self.show_colorbar_checkbox.toggled.connect(
            self.orientation_combo.setEnabled
        )
        self.orientation_combo.currentTextChanged.connect(self.changed.emit)
        self.minimum_spin.valueChanged.connect(self.changed.emit)
        self.maximum_spin.valueChanged.connect(self.changed.emit)

    def _set_interaction_mode(self, mode: str, checked: bool) -> None:
        if checked and self.interaction_mode != mode:
            self.interaction_mode = mode
            self.changed.emit()

    def _icon_toggle(
        self, icon_name: str, tooltip: str, *, checked: bool = False
    ) -> QToolButton:
        button = QToolButton(self)
        button.setIcon(_option_icon(icon_name))
        button.setToolTip(tooltip)
        button.setAccessibleName(tooltip)
        button.setCheckable(True)
        button.setChecked(checked)
        button.setFixedSize(30, 30)
        button.setStyleSheet(
            "QToolButton { border: 1px solid #CBD5E1; border-radius: 5px; "
            "background: #FFFFFF; }"
            "QToolButton:hover { background: #EFF6FF; border-color: #93C5FD; }"
            "QToolButton:checked { background: #DBEAFE; border-color: #3B82F6; }"
        )
        return button

    def _combo(self, values, width: int) -> QComboBox:
        combo = QComboBox(self)
        combo.addItems(values)
        combo.setFixedWidth(width)
        combo.setFixedHeight(28)
        return combo

    def _add_labeled_widget(self, layout, label: str, widget: QWidget) -> None:
        group = QWidget(self)
        row = QHBoxLayout(group)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        row.addWidget(QLabel(label, group))
        row.addWidget(widget)
        layout.addWidget(group)

    @staticmethod
    def _add_separator(layout) -> None:
        separator = QWidget()
        separator.setFixedSize(1, 24)
        separator.setStyleSheet("background: #CBD5E1;")
        layout.addWidget(separator)

    @staticmethod
    def _range_spin_box() -> QDoubleSpinBox:
        spin_box = QDoubleSpinBox()
        spin_box.setRange(-1e300, 1e300)
        spin_box.setDecimals(6)
        spin_box.setKeyboardTracking(False)
        spin_box.setFixedSize(88, 28)
        return spin_box

    def _set_range_inputs_enabled(self, automatic: bool) -> None:
        self.minimum_spin.setEnabled(not automatic)
        self.maximum_spin.setEnabled(not automatic)

    def color_range(self) -> tuple[float, float] | None:
        if self.auto_range_checkbox.isChecked():
            return None
        return self.minimum_spin.value(), self.maximum_spin.value()

    def colorbar_orientation(self) -> str:
        return self.orientation_combo.currentText()

    def set_result(
        self,
        values: tuple[float, ...] | None,
        palette: str,
        error: str | None = None,
        scale: str = "Linear",
    ) -> None:
        if values and self.auto_range_checkbox.isChecked():
            minimum, maximum = min(values), max(values)
            if minimum == maximum:
                if scale == "Logarithmic":
                    minimum, maximum = minimum * 0.9, maximum * 1.1
                else:
                    radius = max(abs(minimum) * 0.05, 0.5)
                    minimum, maximum = minimum - radius, maximum + radius
                if not math.isfinite(minimum) or not math.isfinite(maximum):
                    minimum, maximum = min(values), max(values)
            self.minimum_spin.blockSignals(True)
            self.maximum_spin.blockSignals(True)
            self.minimum_spin.setValue(minimum)
            self.maximum_spin.setValue(maximum)
            self.minimum_spin.blockSignals(False)
            self.maximum_spin.blockSignals(False)

        if error:
            self.status_label.setText(error)
            self.status_label.setToolTip(error)
            self.status_label.setVisible(True)
        else:
            self.status_label.clear()
            self.status_label.setToolTip("")
            self.status_label.setVisible(False)
