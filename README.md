# MeshCFD

MeshCFD is the initial case-setup application for a cell-based compressible
CFD solver. It is a separate application from MeshCAD and uses PyQt5 with a
similar dock-panel and toolbar layout.

The current version provides:

- Import and visualization of finite-volume mesh JSON files exported by
  MeshCAD.
- Per-patch boundary-condition setup.
- Initial numerical controls for MUSCL reconstruction, limiters, fluxes, and
  time integration.
- New, open, and save case configuration.

Cases reference imported mesh files using a path relative to the case file.
Keep the mesh file alongside the case when moving or sharing it.

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
