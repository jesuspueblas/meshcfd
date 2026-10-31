# MeshCFD Architecture

```text
core (case and mesh models, no Qt)
  ^                    ^
  |                    |
gui (PyQt5)         io (JSON)
```

## Current modules

- `meshcfd.core.model` defines the case, finite-volume mesh, boundary
  conditions, and initial numerical settings.
- `meshcfd.io.mesh_io` reads MeshCAD's versioned 2D finite-volume mesh JSON
  and preserves face, cell, and patch connectivity.
- `meshcfd.io.project_io` saves mesh references relative to the case file and
  loads them when opening a case. Cases without a source mesh file can still
  embed their mesh; version 1 cases with embedded meshes remain supported.
- `meshcfd.gui.canvas` displays cells and colored patch faces.
- `meshcfd.gui.boundary_panel` configures one boundary condition per patch.
- `meshcfd.gui.numerics_panel` edits reconstruction, limiter, flux,
  time-integration, and gas parameters.
- `meshcfd.gui.main_window` coordinates case actions, panels, and the
  not-yet-implemented solver action.

The GUI and file formats are initial scaffolding. No flow equations,
reconstruction, flux evaluation, or time integration are implemented yet.
