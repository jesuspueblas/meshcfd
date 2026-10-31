"""Initial numerical setup controls."""

from __future__ import annotations

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QSpinBox,
    QWidget,
)

from meshcfd.core.model import (
    FLUX_SCHEMES,
    LIMITER_SCHEMES,
    RECONSTRUCTION_SCHEMES,
    TIME_INTEGRATION_SCHEMES,
    Case,
)


class NumericsPanel(QWidget):
    changed = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        form = QFormLayout(self)

        self.reconstruction_combo = self._combo(RECONSTRUCTION_SCHEMES)
        parabolic_index = self.reconstruction_combo.findText("MUSCL parabolic")
        self.reconstruction_combo.model().item(parabolic_index).setEnabled(False)
        self.reconstruction_combo.setToolTip(
            "MUSCL parabolic is reserved for a future solver version."
        )
        self.limiter_combo = self._combo(LIMITER_SCHEMES)
        self.flux_combo = self._combo(FLUX_SCHEMES)
        self.time_integration_combo = self._combo(TIME_INTEGRATION_SCHEMES)
        form.addRow("Reconstruction:", self.reconstruction_combo)
        form.addRow("Limiter:", self.limiter_combo)
        form.addRow("Convective flux:", self.flux_combo)
        form.addRow("Time integration:", self.time_integration_combo)

        self.cfl_spin = self._double_spin(0.001, 1.0, 0.5, 3)
        self.iterations_spin = QSpinBox(self)
        self.iterations_spin.setRange(1, 10000000)
        self.end_time_spin = self._double_spin(1e-12, 1e12, 1.0, 8)
        self.residual_spin = self._double_spin(1e-14, 1.0, 1e-6, 12)
        self.residual_spin.setToolTip(
            "Residual is reported for monitoring; transient runs stop at End time "
            "or Maximum time steps."
        )
        self.gamma_spin = self._double_spin(1.0001, 2.0, 1.4, 4)
        self.gas_constant_spin = self._double_spin(1e-6, 1e9, 287.05, 4)
        form.addRow("CFL number:", self.cfl_spin)
        form.addRow("Maximum time steps:", self.iterations_spin)
        form.addRow("End time:", self.end_time_spin)
        form.addRow("Residual tolerance:", self.residual_spin)
        form.addRow("Specific heat ratio:", self.gamma_spin)
        form.addRow("Gas constant:", self.gas_constant_spin)

        self._case = Case()
        self._controls = (
            self.reconstruction_combo,
            self.limiter_combo,
            self.flux_combo,
            self.time_integration_combo,
            self.cfl_spin,
            self.iterations_spin,
            self.end_time_spin,
            self.residual_spin,
            self.gamma_spin,
            self.gas_constant_spin,
        )
        for combo in (
            self.reconstruction_combo,
            self.limiter_combo,
            self.flux_combo,
            self.time_integration_combo,
        ):
            combo.currentTextChanged.connect(self._sync_case)
        for control in (
            self.cfl_spin,
            self.iterations_spin,
            self.end_time_spin,
            self.residual_spin,
            self.gamma_spin,
            self.gas_constant_spin,
        ):
            control.valueChanged.connect(self._sync_case)
        self.set_case(self._case)

    def _combo(self, values: tuple[str, ...]) -> QComboBox:
        combo = QComboBox(self)
        combo.addItems(values)
        return combo

    def _double_spin(
        self, minimum: float, maximum: float, value: float, decimals: int
    ) -> QDoubleSpinBox:
        control = QDoubleSpinBox(self)
        control.setRange(minimum, maximum)
        control.setDecimals(decimals)
        control.setValue(value)
        return control

    def set_case(self, case: Case) -> None:
        self._case = case
        for control in self._controls:
            control.blockSignals(True)
        settings = case.numerics
        self.reconstruction_combo.setCurrentText(settings.reconstruction)
        self.limiter_combo.setCurrentText(settings.limiter)
        self.flux_combo.setCurrentText(settings.flux)
        self.time_integration_combo.setCurrentText(settings.time_integration)
        self.cfl_spin.setValue(settings.cfl)
        self.iterations_spin.setValue(settings.max_iterations)
        self.end_time_spin.setValue(settings.end_time)
        self.residual_spin.setValue(settings.residual_tolerance)
        self.gamma_spin.setValue(settings.gamma)
        self.gas_constant_spin.setValue(settings.gas_constant)
        for control in self._controls:
            control.blockSignals(False)

    def _sync_case(self, *_args) -> None:
        settings = self._case.numerics
        settings.reconstruction = self.reconstruction_combo.currentText()
        settings.limiter = self.limiter_combo.currentText()
        settings.flux = self.flux_combo.currentText()
        settings.time_integration = self.time_integration_combo.currentText()
        settings.cfl = self.cfl_spin.value()
        settings.max_iterations = self.iterations_spin.value()
        settings.end_time = self.end_time_spin.value()
        settings.residual_tolerance = self.residual_spin.value()
        settings.gamma = self.gamma_spin.value()
        settings.gas_constant = self.gas_constant_spin.value()
        self.changed.emit()
