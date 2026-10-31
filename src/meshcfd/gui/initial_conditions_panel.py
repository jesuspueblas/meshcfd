"""Editor for spatially varying initial-condition blocks."""

from __future__ import annotations

from functools import partial

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from meshcfd.core.model import (
    Case,
    InitialConditionBlock,
)

_BLOCK_FIELDS = (
    ("name", "Block name"),
    ("domain", "Domain predicate"),
    ("density", "Density expression"),
    ("velocity_x", "Velocity X expression"),
    ("velocity_y", "Velocity Y expression"),
    ("pressure", "Pressure expression"),
)
_EXAMPLES = {
    "domain": "x^2 + y^2 <= 1",
    "density": "1.225",
    "velocity_x": "10*cos(y)",
    "velocity_y": "0",
    "pressure": "101325",
}
_BASE_FIELDS = (
    ("density", "Base density", 1e-12, 1e12, 1.225),
    ("velocity_x", "Base velocity X", -1e12, 1e12, 0.0),
    ("velocity_y", "Base velocity Y", -1e12, 1e12, 0.0),
    ("pressure", "Base pressure", 1e-12, 1e12, 101325.0),
)


class InitialConditionsPanel(QWidget):
    changed = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Base state used outside all blocks:", self))
        base_form = QFormLayout()
        self.base_controls: dict[str, QDoubleSpinBox] = {}
        for key, label, minimum, maximum, default in _BASE_FIELDS:
            control = QDoubleSpinBox(self)
            control.setRange(minimum, maximum)
            control.setDecimals(8)
            control.setValue(default)
            control.valueChanged.connect(
                lambda value, field=key: self._set_base_value(field, value)
            )
            base_form.addRow(f"{label}:", control)
            self.base_controls[key] = control
        layout.addLayout(base_form)

        self.block_list = QListWidget(self)
        self.block_list.currentRowChanged.connect(self._select_block)
        layout.addWidget(self.block_list, 1)

        button_layout = QHBoxLayout()
        self.add_button = QPushButton("Add", self)
        self.remove_button = QPushButton("Remove", self)
        self.up_button = QPushButton("Up", self)
        self.down_button = QPushButton("Down", self)
        for button in (
            self.add_button,
            self.remove_button,
            self.up_button,
            self.down_button,
        ):
            button_layout.addWidget(button)
        layout.addLayout(button_layout)
        self.add_button.clicked.connect(self._add_block)
        self.remove_button.clicked.connect(self._remove_block)
        self.up_button.clicked.connect(lambda: self._move_block(-1))
        self.down_button.clicked.connect(lambda: self._move_block(1))

        self.empty_label = QLabel("Add a block to define a spatial region.", self)
        self.empty_label.setWordWrap(True)
        layout.addWidget(self.empty_label)
        self.editor = QWidget(self)
        self.form = QFormLayout(self.editor)
        self.block_controls: dict[str, QLineEdit] = {}
        for key, label in _BLOCK_FIELDS:
            control = QLineEdit(self.editor)
            control.setPlaceholderText(_EXAMPLES.get(key, ""))
            if key != "name":
                control.setToolTip(
                    "Use x and y, arithmetic operators, comparisons and functions "
                    "such as sin, cos, tanh, exp, log and sqrt. Use ^ or ** "
                    "for powers."
                )
            control.textChanged.connect(partial(self._set_block_value, key))
            self.form.addRow(f"{label}:", control)
            self.block_controls[key] = control
        layout.addWidget(self.editor)

        self._case = Case()
        self._active_block: InitialConditionBlock | None = None
        self.set_case(self._case)

    def set_case(self, case: Case) -> None:
        self._case = case
        state = case.initial_conditions.base_state
        for key, control in self.base_controls.items():
            control.blockSignals(True)
            control.setValue(getattr(state, key))
            control.blockSignals(False)
        self._refresh_blocks()

    def _refresh_blocks(self, row: int | None = None) -> None:
        self.block_list.blockSignals(True)
        self.block_list.clear()
        for block in self._case.initial_conditions.blocks:
            self.block_list.addItem(QListWidgetItem(block.name))
        self.block_list.blockSignals(False)
        has_blocks = self.block_list.count() > 0
        self.empty_label.setVisible(not has_blocks)
        self.editor.setVisible(has_blocks)
        if has_blocks:
            selected_row = 0 if row is None else min(row, self.block_list.count() - 1)
            self.block_list.setCurrentRow(selected_row)
        else:
            self._active_block = None
        self._update_buttons()

    def _update_buttons(self) -> None:
        row = self.block_list.currentRow()
        count = self.block_list.count()
        self.remove_button.setEnabled(row >= 0)
        self.up_button.setEnabled(row > 0)
        self.down_button.setEnabled(0 <= row < count - 1)

    def _select_block(self, row: int) -> None:
        blocks = self._case.initial_conditions.blocks
        self._active_block = blocks[row] if 0 <= row < len(blocks) else None
        if self._active_block is not None:
            for key, control in self.block_controls.items():
                control.blockSignals(True)
                control.setText(getattr(self._active_block, key))
                control.blockSignals(False)
        self._update_buttons()

    def _set_base_value(self, key: str, value: float) -> None:
        setattr(self._case.initial_conditions.base_state, key, value)
        self.changed.emit()

    def _set_block_value(self, key: str, value: str) -> None:
        block = self._active_block
        if block is None or getattr(block, key) == value:
            return
        setattr(block, key, value)
        if key == "name":
            item = self.block_list.currentItem()
            if item is not None:
                item.setText(value)
        self.changed.emit()

    def _add_block(self) -> None:
        blocks = self._case.initial_conditions.blocks
        block = InitialConditionBlock(name=f"Block {len(blocks) + 1}")
        blocks.append(block)
        self._refresh_blocks(len(blocks) - 1)
        self.changed.emit()

    def _remove_block(self) -> None:
        row = self.block_list.currentRow()
        blocks = self._case.initial_conditions.blocks
        if not 0 <= row < len(blocks):
            return
        del blocks[row]
        self._refresh_blocks(max(0, row - 1))
        self.changed.emit()

    def _move_block(self, offset: int) -> None:
        row = self.block_list.currentRow()
        blocks = self._case.initial_conditions.blocks
        target = row + offset
        if not 0 <= row < len(blocks) or not 0 <= target < len(blocks):
            return
        blocks[row], blocks[target] = blocks[target], blocks[row]
        self._refresh_blocks(target)
        self.changed.emit()
