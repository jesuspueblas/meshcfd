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
        self.time_integration_combo.setToolTip(
            "Implicit Euler uses a first-order spatial residual for robust "
            "high-CFL convergence."
        )
        form.addRow("Reconstruction:", self.reconstruction_combo)
        form.addRow("Limiter:", self.limiter_combo)
        form.addRow("Convective flux:", self.flux_combo)
        form.addRow("Time integration:", self.time_integration_combo)

        self.time_step_mode_combo = self._combo(
            ("Global adaptive", "Global fixed", "Local (steady)")
        )
        self.time_step_mode_combo.setToolTip(
            "Global adaptive uses one CFL-limited step for all cells. Global fixed "
            "uses the configured dt. Local (steady) uses a CFL step per cell as "
            "pseudo-time and stops at the residual tolerance or iteration limit."
        )
        form.addRow("Time stepping:", self.time_step_mode_combo)
        self.cfl_spin = self._double_spin(0.001, 1000.0, 0.5, 3)
        self.cfl_spin.setToolTip(
            "Explicit methods require CFL <= 1. Implicit Euler supports larger "
            "CFL values and solves each step with matrix-free GMRES."
        )
        self.fixed_time_step_spin = self._double_spin(
            1e-14, 1e6, 0.001, 12
        )
        self.fixed_time_step_spin.setToolTip(
            "Fixed global time step in seconds. Stability is the user's "
            "responsibility; reduce dt if the solution becomes unstable."
        )
        self.implicit_preconditioner_combo = self._combo(("LU-SGS", "ILU"))
        self.gmres_tolerance_spin = self._double_spin(
            1e-12, 0.999999999999, 1e-6, 12
        )
        self.gmres_tolerance_spin.setToolTip(
            "Relative linear-system convergence tolerance for GMRES."
        )
        self.gmres_iterations_spin = QSpinBox(self)
        self.gmres_iterations_spin.setRange(1, 100000)
        self.gmres_krylov_dimension_spin = QSpinBox(self)
        self.gmres_krylov_dimension_spin.setRange(1, 1000)
        form.addRow("Implicit preconditioner:", self.implicit_preconditioner_combo)
        form.addRow("GMRES relative tolerance:", self.gmres_tolerance_spin)
        form.addRow("GMRES maximum iterations:", self.gmres_iterations_spin)
        form.addRow("GMRES Krylov dimension:", self.gmres_krylov_dimension_spin)
        self.implicit_controls = (
            self.implicit_preconditioner_combo,
            self.gmres_tolerance_spin,
            self.gmres_iterations_spin,
            self.gmres_krylov_dimension_spin,
        )
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
        self.convergence_history_interval_spin = QSpinBox(self)
        self.convergence_history_interval_spin.setRange(1, 10000000)
        self.convergence_history_interval_spin.setToolTip(
            "Calculate and record convergence residual norms every N solver steps."
        )
        form.addRow(
            "History interval:",
            self.convergence_history_interval_spin,
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
        form.addRow("Fixed dt (s):", self.fixed_time_step_spin)
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

        self.adaptive_mesh_checkbox = QCheckBox("Enabled", self)
        self.adaptive_mesh_checkbox.setToolTip(
            "Refine and coarsen cells from the selected solution gradient while "
            "the solver runs. Direction is selected automatically."
        )
        self.adaptation_interval_spin = QSpinBox(self)
        self.adaptation_interval_spin.setRange(1, 10000000)
        self.adaptation_max_level_spin = QSpinBox(self)
        self.adaptation_max_level_spin.setRange(1, 12)
        self.adaptation_threshold_spin = self._double_spin(
            1e-4, 10.0, 0.25, 4
        )
        self.adaptation_variable_combo = self._combo(
            (
                "Density",
                "Velocity X",
                "Velocity Y",
                "Pressure",
                "Turbulent k",
                "Specific dissipation rate",
            )
        )
        self.adaptation_variable_combo.setToolTip(
            "Turbulent k and omega indicators require Menter k-omega SST."
        )
        form.addRow("Adaptive mesh:", self.adaptive_mesh_checkbox)
        form.addRow("Adapt every N steps:", self.adaptation_interval_spin)
        form.addRow("Maximum refinement level:", self.adaptation_max_level_spin)
        form.addRow("Refinement threshold:", self.adaptation_threshold_spin)
        form.addRow("Gradient variable:", self.adaptation_variable_combo)
        self.adaptive_controls = (
            self.adaptation_interval_spin,
            self.adaptation_max_level_spin,
            self.adaptation_threshold_spin,
            self.adaptation_variable_combo,
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
            self.time_step_mode_combo,
            self.cfl_spin,
            self.fixed_time_step_spin,
            self.implicit_preconditioner_combo,
            self.gmres_tolerance_spin,
            self.gmres_iterations_spin,
            self.gmres_krylov_dimension_spin,
            self.iterations_spin,
            self.end_time_spin,
            self.residual_spin,
            self.convergence_history_interval_spin,
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
            self.adaptive_mesh_checkbox,
            self.adaptation_interval_spin,
            self.adaptation_max_level_spin,
            self.adaptation_threshold_spin,
            self.adaptation_variable_combo,
        )
        self.time_integration_combo.currentTextChanged.connect(
            self._update_cfl_limit
        )
        self.time_integration_combo.currentTextChanged.connect(
            self._update_implicit_controls
        )
        self.time_step_mode_combo.currentTextChanged.connect(
            self._update_time_step_controls
        )
        for combo in (
            self.reconstruction_combo,
            self.limiter_combo,
            self.flux_combo,
            self.time_integration_combo,
            self.time_step_mode_combo,
        ):
            combo.currentTextChanged.connect(self._sync_case)
        for control in (
            self.cfl_spin,
            self.fixed_time_step_spin,
            self.gmres_tolerance_spin,
            self.iterations_spin,
            self.end_time_spin,
            self.residual_spin,
            self.convergence_history_interval_spin,
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
        for control in (
            self.gmres_iterations_spin,
            self.gmres_krylov_dimension_spin,
        ):
            control.valueChanged.connect(self._sync_case)
        self.implicit_preconditioner_combo.currentTextChanged.connect(
            self._sync_case
        )
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
        self.adaptive_mesh_checkbox.toggled.connect(self._update_adaptive_controls)
        self.adaptive_mesh_checkbox.toggled.connect(self._sync_case)
        for control in (
            self.adaptation_interval_spin,
            self.adaptation_max_level_spin,
            self.adaptation_threshold_spin,
        ):
            control.valueChanged.connect(self._sync_case)
        self.adaptation_variable_combo.currentTextChanged.connect(self._sync_case)
        self.set_case(self._case)
        self._update_viscosity_controls()
        self._update_turbulence_controls()
        self._update_adaptive_controls()
        self._update_cfl_limit()
        self._update_implicit_controls()
        self._update_time_step_controls()

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
        self.time_step_mode_combo.setCurrentText(settings.time_step_mode)
        self._update_cfl_limit()
        self.cfl_spin.setValue(settings.cfl)
        self.fixed_time_step_spin.setValue(settings.fixed_time_step)
        self.implicit_preconditioner_combo.setCurrentText(
            settings.implicit_preconditioner
        )
        self.gmres_tolerance_spin.setValue(settings.gmres_tolerance)
        self.gmres_iterations_spin.setValue(settings.gmres_max_iterations)
        self.gmres_krylov_dimension_spin.setValue(
            settings.gmres_krylov_dimension
        )
        self.iterations_spin.setValue(settings.max_iterations)
        self.end_time_spin.setValue(settings.end_time)
        self.residual_spin.setValue(settings.residual_tolerance)
        self.convergence_history_interval_spin.setValue(
            settings.convergence_history_interval
        )
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
        self.adaptive_mesh_checkbox.setChecked(settings.adaptive_mesh)
        self.adaptation_interval_spin.setValue(settings.adaptation_interval)
        self.adaptation_max_level_spin.setValue(settings.adaptation_max_level)
        self.adaptation_threshold_spin.setValue(
            settings.adaptation_refine_threshold
        )
        self.adaptation_variable_combo.setCurrentText(
            settings.adaptation_variable
        )
        for control in self._controls:
            control.blockSignals(False)
        self._update_viscosity_controls()
        self._update_turbulence_controls()
        self._update_adaptive_controls()
        self._update_implicit_controls()
        self._update_time_step_controls()

    def _sync_case(self, *_args) -> None:
        settings = self._case.numerics
        settings.reconstruction = self.reconstruction_combo.currentText()
        settings.limiter = self.limiter_combo.currentText()
        settings.flux = self.flux_combo.currentText()
        settings.time_integration = self.time_integration_combo.currentText()
        settings.time_step_mode = self.time_step_mode_combo.currentText()
        settings.cfl = self.cfl_spin.value()
        settings.fixed_time_step = self.fixed_time_step_spin.value()
        settings.implicit_preconditioner = (
            self.implicit_preconditioner_combo.currentText()
        )
        settings.gmres_tolerance = self.gmres_tolerance_spin.value()
        settings.gmres_max_iterations = self.gmres_iterations_spin.value()
        settings.gmres_krylov_dimension = (
            self.gmres_krylov_dimension_spin.value()
        )
        settings.max_iterations = self.iterations_spin.value()
        settings.end_time = self.end_time_spin.value()
        settings.residual_tolerance = self.residual_spin.value()
        settings.convergence_history_interval = (
            self.convergence_history_interval_spin.value()
        )
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
        settings.adaptive_mesh = self.adaptive_mesh_checkbox.isChecked()
        settings.adaptation_interval = self.adaptation_interval_spin.value()
        settings.adaptation_max_level = self.adaptation_max_level_spin.value()
        settings.adaptation_refine_threshold = (
            self.adaptation_threshold_spin.value()
        )
        settings.adaptation_variable = self.adaptation_variable_combo.currentText()
        self.changed.emit()

    def _update_adaptive_controls(self, *_args) -> None:
        enabled = self.adaptive_mesh_checkbox.isChecked()
        for control in self.adaptive_controls:
            control.setVisible(enabled)
            label = self.layout().labelForField(control)
            if label is not None:
                label.setVisible(enabled)

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

    def _update_cfl_limit(self, *_args) -> None:
        maximum = (
            1000.0
            if self.time_integration_combo.currentText() == "Implicit Euler"
            else 1.0
        )
        self.cfl_spin.setMaximum(maximum)

    def _update_implicit_controls(self, *_args) -> None:
        visible = self.time_integration_combo.currentText() == "Implicit Euler"
        for control in self.implicit_controls:
            control.setVisible(visible)
            label = self.layout().labelForField(control)
            if label is not None:
                label.setVisible(visible)

    def _update_time_step_controls(self, *_args) -> None:
        mode = self.time_step_mode_combo.currentText()
        local = mode == "Local (steady)"
        fixed = mode == "Global fixed"
        for control, visible in (
            (self.cfl_spin, not fixed),
            (self.fixed_time_step_spin, fixed),
            (self.end_time_spin, not local),
            (self.residual_spin, local),
        ):
            control.setVisible(visible)
            label = self.layout().labelForField(control)
            if label is not None:
                label.setVisible(visible)
        self.end_time_spin.setToolTip(
            "Not used for local steady pseudo-time stepping."
            if local
            else "Absolute simulation end time. Increase it to continue a run "
            "that has already reached its current end time."
        )
        self.residual_spin.setToolTip(
            "The local steady solve stops when the residual reaches this tolerance."
            if local
            else "Residual is reported for monitoring; transient runs stop at End time "
            "or Maximum time steps."
        )
