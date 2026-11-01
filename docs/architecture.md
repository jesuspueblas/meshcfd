# MeshCFD Architecture

```text
core (case and mesh models, no Qt)
  ^                    ^
  |                    |
gui (PyQt5)         io (JSON)
```

## Current modules

- `meshcfd.core.model` defines the case, finite-volume mesh, boundary
  conditions, initial-condition blocks, and numerical settings.
- `meshcfd.core.periodic` checks that two boundary patches match after a
  translation along x or y, then returns the axis, pitch, and matching face
  pairs.
- `meshcfd.core.euler` converts ideal-gas states and evaluates Rusanov, HLLC,
  and Roe numerical fluxes.
- `meshcfd.core.solver` assembles conservative finite-volume residuals on
  polygonal cells and advances them with explicit Euler, SSP-RK2, or SSP-RK3.
  Periodic partners share one conservative interface flux; walls and symmetry
  use reflected states. MUSCL-linear reconstruction supports the configured
  limiters. Optional viscous residuals add Stokes-hypothesis Newtonian stresses,
  stress work, and Fourier heat conduction; explicit time steps account for
  diffusive stability. The optional compressible Menter k-omega SST model
  transports density-weighted k and omega, blends coefficients with F1/F2,
  limits eddy viscosity and production, and couples turbulent stresses,
  turbulent heat conduction, and k transport to the flow energy. No-slip walls
  support resolved near-wall values or equilibrium logarithmic wall functions.
  Uniform acceleration contributes momentum and energy source terms.
- `meshcfd.core.expressions` safely parses mathematical expressions in `x` and
  `y`; `meshcfd.core.initial_conditions` evaluates blocks at cell centroids,
  using the last matching block and a configurable base state elsewhere.
  Powers accept `^` or `**`.
- `meshcfd.io.mesh_io` reads MeshCAD's versioned 2D finite-volume mesh JSON
  and preserves face, cell, and patch connectivity.
- `meshcfd.io.project_io` saves mesh references relative to the case file and
  loads them when opening a case. Cases without a source mesh file can still
  embed their mesh; versions 1 through 9 remain supported. Version 9 adds SST
  model and wall-treatment settings, initial and saved k/omega fields, and
  turbulence intensity/length-scale boundary values.
- `meshcfd.gui.canvas` displays cells and colored patch faces.
- `meshcfd.gui.visualization_panel` selects the initial-solution field,
  palette, linear or logarithmic scale, automatic or manual color limits, and
  mesh-edge visibility and optional smooth coloring; it is hosted in a top
  toolbar. Its interaction-mode selector chooses navigation, drag-to-zoom, or
  cell inspection; inspection reports the zero-based cell index, geometric
  centroid, and active field value in the main-window status bar. The canvas
  draws an adaptive coordinate-labeled grid and cell-centered or nodally
  interpolated solution colors. Manual limits clamp values to the palette
  endpoints, and logarithmic scales require positive limits and field values.
  An optional horizontal or vertical color legend overlays the plot and can be
  moved by dragging the legend and resized with its corner grip. Vertical
  legends show their numeric labels to the right of the color ramp and identify
  the displayed variable. Smooth colors are rendered using native Qt gradients
  to avoid per-pixel Python loops.
- `meshcfd.gui.boundary_panel` configures one boundary condition per patch and
  creates compatible x- or y-periodic pairs with their detected pitch. Each
  viscous wall patch independently selects Slip or No-slip, plus fixed
  temperature or prescribed heat flux, positive into the fluid. SST inlet and
  far-field patches accept turbulence intensity (percent) and length scale
  (metres).
- `meshcfd.gui.numerics_panel` edits reconstruction, limiter, flux,
  time-integration, run duration, gas parameters, viscosity model, Prandtl
  number, uniform acceleration, and SST model/wall treatment.
- `meshcfd.gui.initial_conditions_panel` edits the base state, including
  initial k (m²/s²) and omega (1/s), and ordered spatial initial-condition
  blocks.
- `meshcfd.gui.main_window` coordinates case actions, panels, and runs the
  solver on a background thread, displaying completed cell-centered fields
  and streaming iteration/residual logs to a bottom Output dock. Optional
  live field updates are throttled to every ten steps. The latest solver
  result and cumulative time-step count are stored in the case file and
  restored when reopening a solved case. Subsequent runs resume from that
  field and time; the configured end time remains an absolute target.

The solver handles transient ideal-gas flow with periodic, wall, symmetry,
velocity-inlet, pressure-inlet, pressure-outlet,
mass-flow-inlet, mass-flow-outlet, and characteristic far-field boundaries.
Pressure inlet prescribes total pressure and temperature and uses a
characteristic isentropic closure for compatible subsonic inflow; unsupported
supersonic or incompatible states produce explicit errors. Mass-flow rates are
specified per unit depth (kg/s/m), positive into the domain at inlets and out
at outlets. Mass-flow inlet additionally specifies total temperature and a
normalized direction, while mass-flow outlet preserves extrapolated density,
pressure, and tangential velocity and distributes its required outward rate
uniformly per face length. MUSCL parabolic uses a limited quadratic
reconstruction over a two-ring stencil, falling back to linear reconstruction
when the local stencil is rank-deficient.

The default Inviscid model preserves Euler behavior. Constant viscosity or
Sutherland's law enables laminar viscous stresses and heat conduction, with
thermal conductivity computed as `k = μ cp / Pr`. Sutherland defaults to air:
`μref = 1.716e-5 Pa·s`, `Tref = 273.15 K`, and `S = 110.4 K`; the default
Prandtl number is 0.72. Body acceleration is entered in `m/s²`; dynamic
viscosity in `Pa·s`; wall temperature in `K`; and wall heat flux in `W/m²`.
The SST boundary conversion uses `k = 1.5 (I U)²` and
`omega = sqrt(k) / (0.09^(1/4) L)` with intensity `I` entered as a fraction
and length scale `L` in metres. Turbulent Prandtl number is fixed at 0.9.
