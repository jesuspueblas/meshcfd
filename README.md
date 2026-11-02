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
- Transient finite-volume integration of the 2D compressible ideal-gas
  equations,
  with Rusanov, HLLC, and Roe fluxes; explicit Euler, RK2, and RK3; and
  first-order, limited MUSCL-linear, and limited quadratic MUSCL-parabolic
  reconstruction.
- Optional gradient-driven adaptive refinement during solver runs, including
  directional or four-way splits, hanging-node subfaces, and coarsening through
  the saved refinement hierarchy.
- A bottom Output dock that records application messages and per-step time-step
  and conservative residual history. Live mesh-solution updates can be enabled
  in Numerical Setup; they are off by default and refresh every ten steps.
- Spatial initial-condition blocks with safe expressions in `x` and `y`.
- Initial-solution preview by cell, with selectable variable, palette, and
  linear or logarithmic color scale, automatic or manually specified limits,
  clamping outside the selected range, and optional smooth coloring.
- Visualization of Mach number, total (stagnation) pressure, temperature and
  density, plus y+ at the first cell centers adjacent to no-slip walls.
- SST visualization also includes wall distance; the regularized SST omega
  formulation additionally exposes log(omega/omega_wall).
- Default solution-view setup: Density, Rainbow palette, linear scale,
  automatic range, and a visible vertical color legend.
- Initial numerical controls for MUSCL reconstruction, limiters, fluxes, and
  time integration.
- Optional laminar viscous stresses and Fourier heat conduction, using
  constant dynamic viscosity or Sutherland's law and conductivity derived from
  the Prandtl number. Constant body acceleration is configurable in both
  coordinate directions.
- New, open, and save case configuration.
- Persistence of the final cell-centered solution in the case file, so a saved
  run can be reopened and visualized without running the solver again.
- Repeated runs continue from the current solution and accumulated simulation
  time. Increase the configured end time to continue after reaching it.
- While a simulation is running, Run Solver toggles between Pause Solver and
  Resume Solver. Stop Solver ends the run at the current time-step boundary and
  keeps its partial solution. After stopping, setup changes apply to the next
  run, which continues from that saved solution (provided the mesh is unchanged).

Cases reference imported mesh files using a path relative to the case file.
Keep the mesh file alongside the case when moving or sharing it.
Initial-condition domains are boolean predicates such as `x^2 + y^2 <= 1`.
Field expressions support arithmetic, comparisons, and common trigonometric,
hyperbolic, exponential, logarithmic, and root functions. Blocks are applied in
order, with later matching blocks taking precedence; cells outside all blocks
use the configurable base state.
The View toolbar toggles the labeled background grid and provides shortcuts
for the main panels. The View menu can show or hide the Case, View, and
Solution View toolbars, as well as the application panels. Boundary
Conditions and Initial Conditions are initially stacked on the left,
Numerical Setup is on the right, and Solution View is available in its own
top toolbar. The Solution View toolbar controls the displayed field,
mesh-edge visibility, smooth coloring, and color scale.
Choose Box Zoom in Solution View and drag with the left mouse button to zoom
to a region. Choose Cell Info and click a cell to show its zero-based index,
geometric centroid coordinates, and the value of the currently displayed
field in the status bar. Navigate mode preserves wheel zoom and middle-button
panning.
The Output dock is at the bottom by default and can be shown or hidden from the
View menu. Its residual history is reported per time step; use Clear Output in
the View menu to clear the terminal.
Palettes include Viridis, Plasma, Inferno, Coolwarm, Rainbow, Turbo, and
Cividis. Automatic color limits are enabled by default; disable Automatic to
enter manual minimum and maximum values. Logarithmic scales require positive
field values and positive limits. Enable the color-bar icon to display a
floating legend over the mesh; drag the legend itself to move it, use its
bottom-right grip to resize it, and select horizontal or vertical orientation.

The transient solver supports periodic, wall, symmetry,
velocity-inlet, pressure-inlet, pressure-outlet, mass-flow-inlet,
mass-flow-outlet, and far-field boundaries. A velocity inlet prescribes
density, velocity components, and temperature; inlet pressure is calculated
with the ideal-gas law and configured gas constant. A pressure inlet prescribes
total pressure and temperature plus a normalized flow-direction vector, and
uses an isentropic characteristic closure for compatible subsonic inflow;
supersonic inflow/outflow and incompatible states are rejected. A pressure
outlet extrapolates density and velocity from the interior and imposes the
configured static pressure. Mass-flow rates are specified in kg/s/m, positive
into the domain for inlets and out of the domain for outlets. A mass-flow inlet
also prescribes total temperature and flow direction; it extrapolates static
pressure and solves for a common inlet speed that satisfies the integrated
rate. A mass-flow outlet extrapolates density, pressure, and tangential
velocity, then imposes the requested outward mass flux uniformly per face
length. Far-field boundaries use density, velocity components, and pressure
with characteristic treatment for subsonic flow and the appropriate interior
or freestream state for supersonic outflow or inflow. MUSCL parabolic fits a
quadratic polynomial over the two-ring cell stencil and limits face values to
the local stencil range; cells without a full-rank stencil use the linear
reconstruction. Runs stop at the configured End time or maximum time-step
count; the residual is reported for monitoring and does not terminate an
unsteady run.

Viscous terms are disabled by default to preserve inviscid Euler behavior.
Select Constant or Sutherland viscosity in Numerical Setup to add the
Newtonian stress tensor (Stokes hypothesis) and Fourier heat conduction.
Sutherland defaults are for air: μref = 1.716e-5 Pa·s at Tref = 273.15 K and
S = 110.4 K. Thermal conductivity is k = μ cp / Pr, with Pr = 0.72 by
default. Each wall patch independently selects Slip or No-slip and either a
fixed wall temperature (K) or prescribed heat flux (W/m²); positive heat flux
enters the fluid. No-slip applies when viscosity is enabled; inviscid walls
remain slip. Symmetry boundaries remain slip and adiabatic. Acceleration
components are constant throughout the domain in m/s² and contribute both
ρf to momentum and ρf·u to total energy. Diffusive explicit stability limits
are included in the solver time step.

Adaptive mesh refinement is enabled in Numerical Setup. Choose an indicator
field, the adaptation interval, a refinement threshold, and the maximum
refinement level. The solver estimates normalized least-squares gradients,
bisects along the dominant direction, and splits into four when both
directions are comparable. Existing sibling cells are coarsened when their
indicator falls below one tenth of the refinement threshold. A one-level
neighbor constraint limits hanging-node depth; periodic faces are refined
together. Adapted solutions are transferred in conservative form, and the
mesh hierarchy is embedded in the case file so refinement can be undone after
reopening it.

The optional Menter k-omega SST model requires Constant or Sutherland
molecular viscosity. Set the base turbulent kinetic energy `k` in m²/s² and
specific dissipation rate `omega` in 1/s under Initial Conditions. For
Velocity Inlet, Pressure Inlet, Mass Flow Inlet, and Far Field boundaries,
specify turbulence intensity in percent and a turbulence length scale in
metres; the solver derives `k` and `omega` from those values and the boundary
flow speed. No-slip walls support either wall-resolved SST (use a mesh that
resolves the viscous near-wall region) or equilibrium logarithmic wall
functions. The SST omega formulation defaults to Standard SST. The optional
Regularized k-log-omega formulation is available with wall-resolved treatment:
it evolves `log(omega/omega_wall)`, sets that variable to zero at the wall, and
uses `omega_wall = 6 mu / (rho beta* y_w^2)` with `beta* = 0.09`, following the
attached regularization. It is not available with wall functions. The model
includes turbulent stress and heat transport in the momentum and energy
equations; turbulent Prandtl number is fixed at 0.9.

The solution viewer also offers Mach number and isentropic stagnation
properties (total pressure, temperature, and density), computed from the local
static state and velocity. The y+ field is evaluated at the first cell centers
adjacent to no-slip walls using molecular viscosity; other cells are set to
zero. These first-cell values are useful for checking near-wall resolution.
For SST with wall functions, the friction velocity uses the solver's
equilibrium log-law estimate; otherwise it uses the local linear wall shear
estimate. y+ is not meaningful for inviscid or slip-wall cases.
For SST, the viewer also provides the solver's cell-centered distance to the
nearest no-slip wall. When the regularized k-log-omega formulation is active
with a viscous model, it exposes the evolved regularized variable
log(omega/omega_wall).

The Results menu exports the latest completed or partial solution as an ASCII
VTU file, configures incremental VTU snapshots collected in a PVD time series,
plots the conservative residual histories (plus k and omega for SST), and
plots a cell-centered solution cut. Configure the PVD filename and iteration
interval before a run; the series records the starting state, each interval,
and the final state.
Only one snapshot is held at a time by the exporter. A cut is an infinite line
defined by an angle in degrees and a point `(x, y)`; the selected solution
field is sampled at cell centers along the intersected mesh cells. The
convergence graph uses a logarithmic residual axis and retains a bounded,
decimated history for long runs. Convergence History offers linear and
logarithmic residual scales and lets you select which residuals to monitor.
Residuals are organized in a tree, with the turbulence group available only
when the SST model is selected. Curves are clipped to the plot axes, and the
color-bar overlay stays fixed in its chosen viewport position while panning.
Both graph panels zoom by dragging with the left mouse button or using the
mouse wheel, and pan by dragging with the middle mouse button. Each panel has
a Reset zoom button; Solution Cut updates automatically when its field or cut
parameters change. Convergence History and Solution Cut are movable,
floatable dock panels whose positions can be arranged alongside the other
application panels. While the Solution Cut panel is visible, its reference
point and angle are shown as a highlighted line over the mesh and update as
the cut parameters change. Panels with potentially constrained content provide
scroll bars as needed.

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
