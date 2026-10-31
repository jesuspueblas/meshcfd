"""Controls for the mesh and initial-solution display."""

from __future__ import annotations

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from meshcfd.gui.canvas import COLOR_MAPS, SOLUTION_VARIABLES, ColorLegend


class VisualizationPanel(QWidget):
    changed = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.show_mesh_checkbox = QCheckBox("Show mesh edges", self)
        self.show_mesh_checkbox.setChecked(True)
        layout.addWidget(self.show_mesh_checkbox)
        form = QFormLayout()
        self.variable_combo = QComboBox(self)
        self.variable_combo.addItems(SOLUTION_VARIABLES)
        self.palette_combo = QComboBox(self)
        self.palette_combo.addItems(COLOR_MAPS)
        self.scale_combo = QComboBox(self)
        self.scale_combo.addItems(("Linear", "Logarithmic"))
        form.addRow("Variable:", self.variable_combo)
        form.addRow("Color palette:", self.palette_combo)
        form.addRow("Color scale:", self.scale_combo)
        layout.addLayout(form)

        self.legend = ColorLegend(self)
        layout.addWidget(self.legend)
        self.status_label = QLabel("Import a mesh to preview its initial solution.", self)
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        layout.addStretch(1)

        self.show_mesh_checkbox.toggled.connect(self.changed.emit)
        self.variable_combo.currentTextChanged.connect(self.changed.emit)
        self.palette_combo.currentTextChanged.connect(self.changed.emit)
        self.scale_combo.currentTextChanged.connect(self.changed.emit)

    def set_result(
        self,
        values: tuple[float, ...] | None,
        palette: str,
        error: str | None = None,
    ) -> None:
        self.legend.set_values(values, palette)
        if error:
            self.status_label.setText(error)
        elif not values:
            self.status_label.setText("Import a mesh to preview its initial solution.")
        else:
            self.status_label.setText(
                f"Range: {min(values):.4g} to {max(values):.4g}"
            )
