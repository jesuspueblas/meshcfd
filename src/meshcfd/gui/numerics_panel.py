"""Initial numerical setup controls."""

from __future__ import annotations

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
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
    live_solution_changed = pyqtSignal(bool)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        form = QFormLayout(self)
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)

        self.reconstruction_combo = self._combo(RECONSTRUCTION_SCHEMES)
        self.reconstruction_combo.setToolTip(
            "MUSCL parabolic uses a limited quadratic fit over a two-cell stencil."
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
        self.end_time_spin.setToolTip(
            "Absolute simulation end time. Increase it to continue a run that "
            "has already reached its current end time."
        )
        self.residual_spin = self._double_spin(1e-14, 1.0, 1e-6, 12)
        self.residual_spin.setToolTip(
            "Residual is reported for monitoring; transient runs stop at End time "
            "or Maximum time steps."
        )
        self.gamma_spin = self._double_spin(1.0001, 2.0, 1.4, 4)
        self.gas_constant_spin = self._double_spin(1e-6, 1e9, 287.05, 4)
        self.viscosity_model_combo = self._combo(
            ("Inviscid", "Constant", "Sutherland")
        )
        self.dynamic_viscosity_spin = self._double_spin(1e-12, 1e6, 1.8e-5, 10)
        self.sutherland_reference_viscosity_spin = self._double_spin(
            1e-12, 1e6, 1.716e-5, 10
        )
        self.sutherland_reference_temperature_spin = self._double_spin(
            1e-6, 1e9, 273.15, 4
        )
        self.sutherland_temperature_spin = self._double_spin(1e-6, 1e9, 110.4, 4)
        self.prandtl_number_spin = self._double_spin(1e-6, 1e6, 0.72, 5)
        self.acceleration_x_spin = self._double_spin(-1e12, 1e12, 0.0, 6)
        self.acceleration_y_spin = self._double_spin(-1e12, 1e12, 0.0, 6)
        self.turbulence_model_combo = self._combo(
            ("None", "Menter k-omega SST")
        )
        self.turbulence_wall_treatment_combo = self._combo(
            ("Wall-resolved", "Wall functions")
        )
        self.turbulence_omega_formulation_combo = self._combo(
            ("Standard SST", "Regularized k-log-omega")
        )
        self.viscosity_model_combo.setToolTip(
            "Inviscid disables molecular diffusion. Sutherland uses air reference "
            "values and temperature in kelvin."
        )
        self.dynamic_viscosity_spin.setToolTip("Constant dynamic viscosity (Pa·s).")
        self.sutherland_reference_viscosity_spin.setToolTip(
            "Dynamic viscosity at the reference temperature (Pa·s)."
        )
        self.sutherland_reference_temperature_spin.setToolTip(
            "Reference temperature (K)."
        )
        self.sutherland_temperature_spin.setToolTip("Sutherland temperature (K).")
        self.prandtl_number_spin.setToolTip(
            "Prandtl number used to compute thermal conductivity k = μ cp / Pr."
        )
        self.acceleration_x_spin.setToolTip("Constant volumetric acceleration X (m/s²).")
        self.acceleration_y_spin.setToolTip("Constant volumetric acceleration Y (m/s²).")
        self.turbulence_model_combo.setToolTip(
            "Compressible Menter SST two-equation model. Requires Constant or "
            "Sutherland molecular viscosity."
        )
        self.turbulence_wall_treatment_combo.setToolTip(
            "Wall-resolved uses k=0 and the SST near-wall omega condition. "
            "Wall functions use an equilibrium logarithmic law."
        )
        self.turbulence_omega_formulation_combo.setToolTip(
            "The regularized formulation evolves log(omega/omega_wall) and "
            "requires wall-resolved treatment."
        )
        form.addRow("CFL number:", self.cfl_spin)
        form.addRow("Maximum time steps:", self.iterations_spin)
        form.addRow("End time:", self.end_time_spin)
        form.addRow("Residual tolerance:", self.residual_spin)
        form.addRow("Specific heat ratio:", self.gamma_spin)
        form.addRow("Gas constant:", self.gas_constant_spin)
        form.addRow("Viscosity model:", self.viscosity_model_combo)
        form.addRow("Mu (Pa s):", self.dynamic_viscosity_spin)
        form.addRow("Mu ref (Pa s):", self.sutherland_reference_viscosity_spin)
        form.addRow("T ref (K):", self.sutherland_reference_temperature_spin)
        form.addRow("Sutherland S (K):", self.sutherland_temperature_spin)
        form.addRow("Prandtl:", self.prandtl_number_spin)
        form.addRow("Accel. X (m/s^2):", self.acceleration_x_spin)
        form.addRow("Accel. Y (m/s^2):", self.acceleration_y_spin)
        form.addRow("Turbulence model:", self.turbulence_model_combo)
        form.addRow("SST wall treatment:", self.turbulence_wall_treatment_combo)
        form.addRow(
            "SST omega formulation:", self.turbulence_omega_formulation_combo
        )

        self.live_solution_checkbox = QCheckBox("Update during run", self)
        self.live_solution_checkbox.setToolTip(
            "Refresh the mesh display every 10 solver steps. This adds rendering "
            "overhead and applies to runs started after changing the option."
        )
        form.addRow("Live solution:", self.live_solution_checkbox)

        self.live_solution_checkbox.toggled.connect(
            self.live_solution_changed.emit
        )
        self.viscosity_model_combo.currentTextChanged.connect(
            self._update_viscosity_controls
        )
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
            self.viscosity_model_combo,
            self.dynamic_viscosity_spin,
            self.sutherland_reference_viscosity_spin,
            self.sutherland_reference_temperature_spin,
            self.sutherland_temperature_spin,
            self.prandtl_number_spin,
            self.acceleration_x_spin,
            self.acceleration_y_spin,
            self.turbulence_model_combo,
            self.turbulence_wall_treatment_combo,
            self.turbulence_omega_formulation_combo,
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
            self.dynamic_viscosity_spin,
            self.sutherland_reference_viscosity_spin,
            self.sutherland_reference_temperature_spin,
            self.sutherland_temperature_spin,
            self.prandtl_number_spin,
            self.acceleration_x_spin,
            self.acceleration_y_spin,
        ):
            control.valueChanged.connect(self._sync_case)
        self.viscosity_model_combo.currentTextChanged.connect(self._sync_case)
        self.turbulence_model_combo.currentTextChanged.connect(
            self._update_turbulence_controls
        )
        self.turbulence_model_combo.currentTextChanged.connect(self._sync_case)
        self.turbulence_wall_treatment_combo.currentTextChanged.connect(
            self._update_turbulence_controls
        )
        self.turbulence_wall_treatment_combo.currentTextChanged.connect(self._sync_case)
        self.turbulence_omega_formulation_combo.currentTextChanged.connect(
            self._update_turbulence_controls
        )
        self.turbulence_omega_formulation_combo.currentTextChanged.connect(
            self._sync_case
        )
        self.set_case(self._case)
        self._update_viscosity_controls()
        self._update_turbulence_controls()

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
        self.viscosity_model_combo.setCurrentText(settings.viscosity_model)
        self.dynamic_viscosity_spin.setValue(settings.dynamic_viscosity)
        self.sutherland_reference_viscosity_spin.setValue(
            settings.sutherland_reference_viscosity
        )
        self.sutherland_reference_temperature_spin.setValue(
            settings.sutherland_reference_temperature
        )
        self.sutherland_temperature_spin.setValue(settings.sutherland_temperature)
        self.prandtl_number_spin.setValue(settings.prandtl_number)
        self.acceleration_x_spin.setValue(settings.acceleration_x)
        self.acceleration_y_spin.setValue(settings.acceleration_y)
        self.turbulence_model_combo.setCurrentText(settings.turbulence_model)
        self.turbulence_wall_treatment_combo.setCurrentText(
            settings.turbulence_wall_treatment
        )
        self.turbulence_omega_formulation_combo.setCurrentText(
            settings.turbulence_omega_formulation
        )
        for control in self._controls:
            control.blockSignals(False)
        self._update_viscosity_controls()
        self._update_turbulence_controls()

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
        settings.viscosity_model = self.viscosity_model_combo.currentText()
        settings.dynamic_viscosity = self.dynamic_viscosity_spin.value()
        settings.sutherland_reference_viscosity = (
            self.sutherland_reference_viscosity_spin.value()
        )
        settings.sutherland_reference_temperature = (
            self.sutherland_reference_temperature_spin.value()
        )
        settings.sutherland_temperature = self.sutherland_temperature_spin.value()
        settings.prandtl_number = self.prandtl_number_spin.value()
        settings.acceleration_x = self.acceleration_x_spin.value()
        settings.acceleration_y = self.acceleration_y_spin.value()
        settings.turbulence_model = self.turbulence_model_combo.currentText()
        settings.turbulence_wall_treatment = (
            self.turbulence_wall_treatment_combo.currentText()
        )
        settings.turbulence_omega_formulation = (
            self.turbulence_omega_formulation_combo.currentText()
        )
        self.changed.emit()

    def _update_viscosity_controls(self, *_args) -> None:
        model = self.viscosity_model_combo.currentText()
        for control, visible in (
            (self.dynamic_viscosity_spin, model == "Constant"),
            (self.sutherland_reference_viscosity_spin, model == "Sutherland"),
            (self.sutherland_reference_temperature_spin, model == "Sutherland"),
            (self.sutherland_temperature_spin, model == "Sutherland"),
        ):
            control.setVisible(visible)
            label = self.layout().labelForField(control)
            if label is not None:
                label.setVisible(visible)

    def _update_turbulence_controls(self, *_args) -> None:
        visible = self.turbulence_model_combo.currentText() != "None"
        self.turbulence_wall_treatment_combo.setVisible(visible)
        self.turbulence_omega_formulation_combo.setVisible(visible)
        regularized = (
            self.turbulence_omega_formulation_combo.currentText()
            == "Regularized k-log-omega"
        )
        resolved_wall = (
            self.turbulence_wall_treatment_combo.currentText() == "Wall-resolved"
        )
        self.turbulence_omega_formulation_combo.setEnabled(
            not visible or resolved_wall or regularized
        )
        self.turbulence_wall_treatment_combo.setEnabled(
            not visible or not regularized
        )
        for control in (
            self.turbulence_wall_treatment_combo,
            self.turbulence_omega_formulation_combo,
        ):
            label = self.layout().labelForField(control)
            if label is not None:
                label.setVisible(visible)
