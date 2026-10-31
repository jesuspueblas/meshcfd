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
- Spatial initial-condition blocks with safe expressions in `x` and `y`.
- Initial-solution preview by cell, with selectable variable, palette, and
  linear or logarithmic color scale, automatic or manually specified limits,
  clamping outside the selected range, and optional smooth coloring.
- Initial numerical controls for MUSCL reconstruction, limiters, fluxes, and
  time integration.
- New, open, and save case configuration.

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
Palettes include Viridis, Plasma, Inferno, Coolwarm, Rainbow, Turbo, and
Cividis. Automatic color limits are enabled by default; disable Automatic to
enter manual minimum and maximum values. Logarithmic scales require positive
field values and positive limits. Enable the color-bar icon to display a
floating legend over the mesh; drag the legend itself to move it, use its
bottom-right grip to resize it, and select horizontal or vertical orientation.

The CFD solver is not implemented yet. The Run action is an explicit
placeholder and does not start a simulation.

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
