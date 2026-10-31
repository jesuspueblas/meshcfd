# MeshCFD

MeshCFD is the initial case-setup application for a cell-based compressible
CFD solver. It is a separate application from MeshCAD and uses PyQt5 with a
similar dock-panel and toolbar layout.

The current version provides:

- Import and visualization of finite-volume mesh JSON files exported by
  MeshCAD.
- Per-patch boundary-condition setup.
- Translational periodic boundaries, with automatic patch compatibility checks
  and pitch detection along the x or y axis.
- Transient finite-volume integration of the 2D ideal-gas Euler equations,
  with Rusanov, HLLC, and Roe fluxes; explicit Euler, RK2, and RK3; and
  first-order or limited MUSCL-linear reconstruction.
- A bottom Output dock that records application messages and per-step time-step
  and conservative residual history. Live mesh-solution updates can be enabled
  in Numerical Setup; they are off by default and refresh every ten steps.
- Spatial initial-condition blocks with safe expressions in `x` and `y`.
- Initial-solution preview by cell, with selectable variable, palette, and
  linear or logarithmic color scale, automatic or manually specified limits,
  clamping outside the selected range, and optional smooth coloring.
- Initial numerical controls for MUSCL reconstruction, limiters, fluxes, and
  time integration.
- New, open, and save case configuration.
- Persistence of the final cell-centered solution in the case file, so a saved
  run can be reopened and visualized without running the solver again.
- Repeated runs continue from the current solution and accumulated simulation
  time. Increase the configured end time to continue after reaching it.

Cases reference imported mesh files using a path relative to the case file.
Keep the mesh file alongside the case when moving or sharing it.
Initial-condition domains are boolean predicates such as `x^2 + y^2 <= 1`.
Field expressions support arithmetic, comparisons, and common trigonometric,
hyperbolic, exponential, logarithmic, and root functions. Blocks are applied in
order, with later matching blocks taking precedence; cells outside all blocks
use the configurable base state.
The View menu and toolbar toggle the labeled background grid. Boundary
Conditions and Initial Conditions are initially stacked on the left,
Numerical Setup is on the right, and Solution View is available in its own
top toolbar. The Solution View toolbar controls the displayed field,
mesh-edge visibility, smooth coloring, and color scale.
The Output dock is at the bottom by default and can be shown or hidden from the
View menu. Its residual history is reported per time step; use Clear Output in
the View menu to clear the terminal.
Palettes include Viridis, Plasma, Inferno, Coolwarm, Rainbow, Turbo, and
Cividis. Automatic color limits are enabled by default; disable Automatic to
enter manual minimum and maximum values. Logarithmic scales require positive
field values and positive limits. Enable the color-bar icon to display a
floating legend over the mesh; drag the legend itself to move it, use its
bottom-right grip to resize it, and select horizontal or vertical orientation.

The transient solver supports periodic, inviscid slip-wall, symmetry,
velocity-inlet, pressure-outlet, and far-field boundaries. A velocity inlet
prescribes density, velocity components, and temperature; inlet pressure is
calculated with the ideal-gas law and configured gas constant. A pressure
outlet extrapolates density and velocity from the interior and imposes the
configured static pressure. Far-field boundaries use density, velocity
components, and pressure with characteristic treatment for subsonic flow and
the appropriate interior or freestream state for supersonic outflow or inflow.
Pressure inlet and the selectable MUSCL-parabolic reconstruction are not
implemented and are reported as solver errors. Runs stop at the configured
End time or maximum time-step count; the residual is reported for monitoring
and does not terminate an unsteady run.

### Isentropic Euler vortex test

For a nondimensional vortex on a `[0, 10] x [0, 10]` domain, pair the opposing
x and y boundary patches periodically. Use `gamma = 1.4`, `CFL = 0.4`,
`End time = 10`, and increase the maximum time steps as needed. Add an initial
condition block over `True` with `beta = 5`, center `(5, 5)`, and freestream
`rho = p = u = v = 1`. Enter the following field expressions (`r²` is expanded
because block fields accept expressions in `x` and `y` only):

- Density:
  `(1 - 0.4*25/(8*1.4*pi^2)*exp(1-((x-5)^2+(y-5)^2)))^(1/0.4)`
- Velocity X:
  `1 - 5/(2*pi)*exp(0.5*(1-((x-5)^2+(y-5)^2)))*(y-5)`
- Velocity Y:
  `1 + 5/(2*pi)*exp(0.5*(1-((x-5)^2+(y-5)^2)))*(x-5)`
- Pressure:
  `(1 - 0.4*25/(8*1.4*pi^2)*exp(1-((x-5)^2+(y-5)^2)))^(1.4/0.4)`

The vortex should return to its initial position after one period. Results are
cell-centered; use the solution visualization controls to inspect the evolved
field.

## Installation

From this directory:

    pip install -e ".[dev]"

## Usage

    python -m meshcfd

## Tests

    python -m pytest

## Package boundaries

- `core` contains application data models and has no PyQt5 dependency.
- `io` reads MeshCAD finite-volume mesh JSON and MeshCFD case JSON.
- `gui` contains the PyQt5 application and visualization.
