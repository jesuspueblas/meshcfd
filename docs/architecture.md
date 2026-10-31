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
  translation along x or y, then returns the axis and pitch.
- `meshcfd.core.expressions` safely parses mathematical expressions in `x` and
  `y`; `meshcfd.core.initial_conditions` evaluates blocks at cell centroids,
  using the last matching block and a configurable base state elsewhere.
  Powers accept `^` or `**`.
- `meshcfd.io.mesh_io` reads MeshCAD's versioned 2D finite-volume mesh JSON
  and preserves face, cell, and patch connectivity.
- `meshcfd.io.project_io` saves mesh references relative to the case file and
  loads them when opening a case. Cases without a source mesh file can still
  embed their mesh; versions 1 through 3 remain supported, and version 4
  stores periodic patch pairs.
- `meshcfd.gui.canvas` displays cells and colored patch faces.
- `meshcfd.gui.visualization_panel` selects the initial-solution field,
  palette, linear or logarithmic scale, automatic or manual color limits, and
  mesh-edge visibility and optional smooth coloring; it is hosted in a top
  toolbar. The canvas draws an adaptive coordinate-labeled grid and cell-
  centered or nodally interpolated solution colors. Manual limits clamp values
  to the palette endpoints, and logarithmic scales require positive limits and
  field values. An optional horizontal or vertical color legend overlays the
  plot and can be moved by dragging the legend and resized with its corner
  grip. Vertical legends show their numeric labels to the right of the color
  ramp and identify the displayed variable. Smooth colors are rendered using
  native Qt gradients to avoid per-pixel Python loops.
- `meshcfd.gui.boundary_panel` configures one boundary condition per patch and
  creates compatible x- or y-periodic pairs with their detected pitch.
- `meshcfd.gui.numerics_panel` edits reconstruction, limiter, flux,
  time-integration, and gas parameters.
- `meshcfd.gui.initial_conditions_panel` edits the base state and ordered
  spatial initial-condition blocks.
- `meshcfd.gui.main_window` coordinates case actions, panels, and the
  not-yet-implemented solver action.

The GUI and file formats are initial scaffolding. No flow equations,
reconstruction, flux evaluation, or time integration are implemented yet.
