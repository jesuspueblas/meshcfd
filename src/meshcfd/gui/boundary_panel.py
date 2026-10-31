"""Patch boundary-condition editor."""

from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)

from meshcfd.core.model import BOUNDARY_TYPES, BoundaryCondition, Case

BOUNDARY_FIELDS = {
    "Wall": (("temperature", "Wall temperature", 300.0),),
    "Velocity Inlet": (
        ("density", "Density", 1.225),
        ("velocity_x", "Velocity X", 0.0),
        ("velocity_y", "Velocity Y", 0.0),
        ("temperature", "Temperature", 288.15),
    ),
    "Pressure Inlet": (
        ("pressure", "Total pressure", 101325.0),
        ("velocity_x", "Velocity X", 0.0),
        ("velocity_y", "Velocity Y", 0.0),
        ("temperature", "Total temperature", 288.15),
    ),
    "Pressure Outlet": (("pressure", "Static pressure", 101325.0),),
    "Symmetry": (),
    "Far Field": (
        ("density", "Density", 1.225),
        ("velocity_x", "Velocity X", 0.0),
        ("velocity_y", "Velocity Y", 0.0),
        ("pressure", "Pressure", 101325.0),
        ("temperature", "Temperature", 288.15),
    ),
}


class BoundaryPanel(QWidget):
    changed = pyqtSignal()
    patch_selected = pyqtSignal(object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.patch_list = QListWidget(self)
        self.patch_list.currentRowChanged.connect(self._select_row)
        layout.addWidget(self.patch_list, 1)

        self.empty_label = QLabel("Import a mesh to configure its boundary patches.")
        self.empty_label.setWordWrap(True)
        layout.addWidget(self.empty_label)

        self.editor = QWidget(self)
        self.form = QFormLayout(self.editor)
        self.kind_combo = QComboBox(self.editor)
        self.kind_combo.addItems(BOUNDARY_TYPES)
        self.form.addRow("Boundary type:", self.kind_combo)
        self.value_controls: dict[str, QDoubleSpinBox] = {}
        self.kind_combo.currentTextChanged.connect(self._change_kind)
        layout.addWidget(self.editor)

        self.case = Case()
        self._active_condition: BoundaryCondition | None = None
        self.refresh()

    def set_case(self, case: Case) -> None:
        self.case = case
        self.refresh()

    def refresh(self) -> None:
        self.patch_list.blockSignals(True)
        self.patch_list.clear()
        if self.case.mesh is not None:
            for condition in self.case.boundary_conditions:
                item = QListWidgetItem(condition.patch_name)
                item.setData(Qt.UserRole, condition.patch_id)
                self.patch_list.addItem(item)
        self.patch_list.blockSignals(False)
        has_patches = self.patch_list.count() > 0
        self.empty_label.setVisible(not has_patches)
        self.editor.setVisible(has_patches)
        if has_patches:
            self.patch_list.setCurrentRow(0)
        else:
            self._active_condition = None
            self.patch_selected.emit(None)

    def _select_row(self, row: int) -> None:
        if row < 0 or row >= len(self.case.boundary_conditions):
            self._active_condition = None
            self.patch_selected.emit(None)
            return
        self._active_condition = self.case.boundary_conditions[row]
        self.patch_selected.emit(self._active_condition.patch_id)
        self._load_condition()

    def _load_condition(self) -> None:
        condition = self._active_condition
        if condition is None:
            return
        self.kind_combo.blockSignals(True)
        self.kind_combo.setCurrentText(condition.kind)
        self.kind_combo.blockSignals(False)
        self._rebuild_value_fields()

    def _change_kind(self, kind: str) -> None:
        if self._active_condition is None:
            return
        self._active_condition.kind = kind
        self._rebuild_value_fields()
        self.changed.emit()

    def _rebuild_value_fields(self) -> None:
        for control in self.value_controls.values():
            self.form.removeRow(control)
        self.value_controls.clear()
        condition = self._active_condition
        if condition is None:
            return
        for key, label, default in BOUNDARY_FIELDS[condition.kind]:
            control = QDoubleSpinBox(self.editor)
            minimum = 1e-12 if key in ("density", "pressure", "temperature") else -1e12
            control.setRange(minimum, 1e12)
            control.setDecimals(6)
            control.setSingleStep(1.0)
            control.setValue(condition.values.get(key, default))
            control.setToolTip(f"{label} prescribed on this patch.")
            control.editingFinished.connect(
                lambda field=key, spin=control: self._set_value(field, spin.value())
            )
            self.form.addRow(f"{label}:", control)
            self.value_controls[key] = control

    def _set_value(self, key: str, value: float) -> None:
        if self._active_condition is None:
            return
        self._active_condition.values[key] = value
        self.changed.emit()
