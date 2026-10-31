"""Evaluate initial-condition blocks at finite-volume cell centroids."""

from __future__ import annotations

import math

from meshcfd.core.expressions import ExpressionError, compile_expression
from meshcfd.core.model import (
    FiniteVolumeMesh,
    InitialConditionSettings,
    InitialState,
)


def evaluate_initial_conditions(
    mesh: FiniteVolumeMesh, settings: InitialConditionSettings
) -> tuple[InitialState, ...]:
    """Return one initial state per cell, applying matching blocks in list order."""
    blocks = [
        (
            block,
            compile_expression(block.domain, boolean=True),
            {
                field: compile_expression(getattr(block, field))
                for field in ("density", "velocity_x", "velocity_y", "pressure")
            },
        )
        for block in settings.blocks
    ]
    states = []
    for cell_id, node_ids in enumerate(mesh.cells):
        x, y = _cell_centroid(mesh, node_ids)
        state = settings.base_state
        for block, domain, fields in reversed(blocks):
            try:
                if not domain.evaluate(x, y):
                    continue
                values = {
                    field: expression.evaluate(x, y)
                    for field, expression in fields.items()
                }
                state = InitialState(**values)
                _validate_state(state)
            except ValueError as error:
                raise ExpressionError(
                    f"Initial-condition block '{block.name}', cell {cell_id}: {error}"
                ) from error
            break
        states.append(state)
    _validate_state(settings.base_state)
    return tuple(states)


def _cell_centroid(
    mesh: FiniteVolumeMesh, node_ids: tuple[int, ...]
) -> tuple[float, float]:
    points = [mesh.points[node_id] for node_id in node_ids]
    twice_area = sum(
        x0 * y1 - x1 * y0
        for (x0, y0), (x1, y1) in zip(points, points[1:] + points[:1])
    )
    if not math.isfinite(twice_area) or abs(twice_area) <= 1e-15:
        raise ValueError("Cannot evaluate initial conditions for a zero-area cell.")
    x_sum = 0.0
    y_sum = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:] + points[:1]):
        cross = x0 * y1 - x1 * y0
        x_sum += (x0 + x1) * cross
        y_sum += (y0 + y1) * cross
    centroid = (x_sum / (3.0 * twice_area), y_sum / (3.0 * twice_area))
    if not all(math.isfinite(value) for value in centroid):
        raise ValueError("Cannot evaluate initial conditions at a non-finite cell centroid.")
    return centroid


def _validate_state(state: InitialState) -> None:
    values = (state.density, state.velocity_x, state.velocity_y, state.pressure)
    try:
        finite = all(math.isfinite(value) for value in values)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError("Initial-state values must be finite.")
    if state.density <= 0.0 or state.pressure <= 0.0:
        raise ValueError("Initial density and pressure must be positive.")
