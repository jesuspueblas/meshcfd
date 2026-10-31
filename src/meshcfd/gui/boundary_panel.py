"""Patch boundary-condition editor."""

from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from meshcfd.core.model import (
    BOUNDARY_TYPES,
    BoundaryCondition,
    Case,
    PeriodicBoundaryPair,
)
from meshcfd.core.periodic import detect_periodic_pair

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
    "Periodic": (),
    "Far Field": (
        ("density", "Density", 1.225),
        ("velocity_x", "Velocity X", 0.0),
        ("velocity_y", "Velocity Y", 0.0),
        ("pressure", "Pressure", 101325.0),
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

        self.periodic_group = QGroupBox("Periodic pairs", self)
        periodic_layout = QVBoxLayout(self.periodic_group)
        pair_form = QFormLayout()
        self.periodic_patch_a_combo = QComboBox(self.periodic_group)
        self.periodic_patch_b_combo = QComboBox(self.periodic_group)
        self.periodic_patch_a_combo.currentIndexChanged.connect(
            self._check_periodic_candidate
        )
        self.periodic_patch_b_combo.currentIndexChanged.connect(
            self._check_periodic_candidate
        )
        pair_form.addRow("Patch A:", self.periodic_patch_a_combo)
        pair_form.addRow("Patch B:", self.periodic_patch_b_combo)
        periodic_layout.addLayout(pair_form)
        self.periodic_status = QLabel("Select two patches to check compatibility.")
        self.periodic_status.setWordWrap(True)
        periodic_layout.addWidget(self.periodic_status)
        self.periodic_pairs_list = QListWidget(self.periodic_group)
        self.periodic_pairs_list.setMaximumHeight(90)
        self.periodic_pairs_list.currentRowChanged.connect(
            self._update_remove_pair_button
        )
        periodic_layout.addWidget(self.periodic_pairs_list)
        pair_buttons = QHBoxLayout()
        self.add_periodic_pair_button = QPushButton("Add pair", self.periodic_group)
        self.add_periodic_pair_button.setEnabled(False)
        self.add_periodic_pair_button.clicked.connect(self._add_periodic_pair)
        self.remove_periodic_pair_button = QPushButton(
            "Remove pair", self.periodic_group
        )
        self.remove_periodic_pair_button.setEnabled(False)
        self.remove_periodic_pair_button.clicked.connect(self._remove_periodic_pair)
        pair_buttons.addWidget(self.add_periodic_pair_button)
        pair_buttons.addWidget(self.remove_periodic_pair_button)
        periodic_layout.addLayout(pair_buttons)
        layout.addWidget(self.periodic_group)

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
        self.periodic_group.setVisible(has_patches)
        self._refresh_periodic_controls()

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
        condition = self._active_condition
        paired = any(
            condition.patch_id in (pair.patch_a_id, pair.patch_b_id)
            for pair in self.case.periodic_pairs
        )
        if kind == "Periodic" and not paired:
            self.periodic_status.setText(
                "Create a compatible pair below to set a patch as periodic."
            )
            self.kind_combo.blockSignals(True)
            self.kind_combo.setCurrentText(condition.kind)
            self.kind_combo.blockSignals(False)
            return
        if paired and kind != "Periodic":
            self.periodic_status.setText(
                "Remove this periodic pair before changing its boundary type."
            )
            self.kind_combo.blockSignals(True)
            self.kind_combo.setCurrentText("Periodic")
            self.kind_combo.blockSignals(False)
            return
        condition.kind = kind
        self._rebuild_value_fields()
        self.changed.emit()

    def _refresh_periodic_controls(self) -> None:
        combos = (self.periodic_patch_a_combo, self.periodic_patch_b_combo)
        for combo in combos:
            combo.blockSignals(True)
            selected = combo.currentData()
            combo.clear()
            for patch in self.case.mesh.patches if self.case.mesh else ():
                combo.addItem(patch.name, patch.id)
            selected_index = combo.findData(selected)
            if selected_index >= 0:
                combo.setCurrentIndex(selected_index)
            combo.blockSignals(False)
        if self.periodic_patch_a_combo.count() > 0:
            self.periodic_patch_a_combo.setCurrentIndex(0)
        if self.periodic_patch_b_combo.count() > 1:
            self.periodic_patch_b_combo.setCurrentIndex(1)

        self.periodic_pairs_list.clear()
        patch_names = {
            patch.id: patch.name for patch in self.case.mesh.patches
        } if self.case.mesh else {}
        for pair in self.case.periodic_pairs:
            self.periodic_pairs_list.addItem(
                f"{patch_names.get(pair.patch_a_id, pair.patch_a_id)} <-> "
                f"{patch_names.get(pair.patch_b_id, pair.patch_b_id)} | "
                f"{pair.axis.upper()} | pitch {pair.pitch:.6g}"
            )
        self._check_periodic_candidate()
        self._update_remove_pair_button()

    def _check_periodic_candidate(self, *_args) -> None:
        patch_a_id = self.periodic_patch_a_combo.currentData()
        patch_b_id = self.periodic_patch_b_combo.currentData()
        compatible = False
        if patch_a_id is None or patch_b_id is None:
            message = "Select two patches to check compatibility."
        elif patch_a_id == patch_b_id:
            message = "Select two different patches."
        elif any(
            patch_id in (pair.patch_a_id, pair.patch_b_id)
            for pair in self.case.periodic_pairs
            for patch_id in (patch_a_id, patch_b_id)
        ):
            message = "A patch can belong to only one periodic pair."
        elif self.case.mesh is None:
            message = "Import a mesh to check periodic compatibility."
        else:
            try:
                axis, pitch = detect_periodic_pair(
                    self.case.mesh, patch_a_id, patch_b_id
                )
            except ValueError as error:
                message = f"Incompatible: {error}"
            else:
                message = f"Compatible in {axis.upper()} | pitch = {pitch:.8g}"
                compatible = True
        self.periodic_status.setText(message)
        self.add_periodic_pair_button.setEnabled(compatible)

    def _add_periodic_pair(self) -> None:
        if self.case.mesh is None:
            return
        patch_a_id = self.periodic_patch_a_combo.currentData()
        patch_b_id = self.periodic_patch_b_combo.currentData()
        if patch_a_id is None or patch_b_id is None:
            return
        try:
            axis, pitch = detect_periodic_pair(
                self.case.mesh, patch_a_id, patch_b_id
            )
        except ValueError as error:
            self.periodic_status.setText(f"Incompatible: {error}")
            self.add_periodic_pair_button.setEnabled(False)
            return
        self.case.periodic_pairs.append(
            PeriodicBoundaryPair(patch_a_id, patch_b_id, axis, pitch)
        )
        for condition in self.case.boundary_conditions:
            if condition.patch_id in (patch_a_id, patch_b_id):
                condition.kind = "Periodic"
                condition.values.clear()
        self._refresh_periodic_controls()
        self._load_condition()
        self.changed.emit()

    def _remove_periodic_pair(self) -> None:
        row = self.periodic_pairs_list.currentRow()
        if row < 0 or row >= len(self.case.periodic_pairs):
            return
        pair = self.case.periodic_pairs.pop(row)
        for condition in self.case.boundary_conditions:
            if condition.patch_id in (pair.patch_a_id, pair.patch_b_id):
                condition.kind = "Wall"
                condition.values.clear()
        self._refresh_periodic_controls()
        self._load_condition()
        self.changed.emit()

    def _update_remove_pair_button(self, row: int = -1) -> None:
        self.remove_periodic_pair_button.setEnabled(row >= 0)

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
