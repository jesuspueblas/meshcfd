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
- `meshcfd.io.project_io` saves and loads self-contained MeshCFD case files.
- `meshcfd.gui.canvas` displays cells and colored patch faces.
- `meshcfd.gui.boundary_panel` configures one boundary condition per patch.
- `meshcfd.gui.numerics_panel` edits reconstruction, limiter, flux,
  time-integration, and gas parameters.
- `meshcfd.gui.main_window` coordinates case actions, panels, and the
  not-yet-implemented solver action.

The GUI and file formats are initial scaffolding. No flow equations,
reconstruction, flux evaluation, or time integration are implemented yet.
