import pytest

from meshcfd.core.expressions import ExpressionError, compile_expression
from meshcfd.core.initial_conditions import evaluate_initial_conditions
from meshcfd.core.model import (
    FiniteVolumeMesh,
    InitialConditionBlock,
    InitialConditionSettings,
    InitialState,
)


def make_mesh() -> FiniteVolumeMesh:
    return FiniteVolumeMesh(
        points=((0, 0), (1, 0), (1, 1), (0, 1)),
        cells=((0, 1, 2, 3),),
        faces=((0, 1), (1, 2), (2, 3), (3, 0)),
        face_owners=(0, 0, 0, 0),
        face_neighbours=(-1, -1, -1, -1),
        face_patch_ids=(0, 1, 2, 3),
        patches=(),
        cell_face_ids=((0, 1, 2, 3),),
        cell_face_signs=((1, 1, 1, 1),),
    )


def test_expression_parser_evaluates_arithmetic_and_math_functions():
    expression = compile_expression("sin(pi / 2) + cosh(0) + sqrt(4) + log(e)")

    assert expression.evaluate(0.0, 0.0) == pytest.approx(5.0)


def test_domain_expression_supports_comparisons_and_short_circuit():
    expression = compile_expression("x < 0 and 1 / 0 > 0", boolean=True)

    assert expression.evaluate(1.0, 0.0) is False


def test_expression_parser_accepts_caret_for_powers():
    expression = compile_expression("x^2 + y^2 <= 1", boolean=True)

    assert expression.evaluate(0.5, 0.5) is True
    assert expression.evaluate(1.0, 1.0) is False


@pytest.mark.parametrize(
    ("source", "boolean"),
    [
        ("__import__('os').system('whoami')", False),
        ("x + 1", True),
        ("x > 0", False),
    ],
)
def test_expression_parser_rejects_unsupported_or_wrong_type(source, boolean):
    with pytest.raises(ExpressionError):
        compile_expression(source, boolean=boolean)


def test_initial_conditions_use_cell_centroids_and_later_blocks_win():
    settings = InitialConditionSettings(
        blocks=[
            InitialConditionBlock(
                name="first",
                domain="x*x + y*y < 2",
                density="1",
                velocity_x="x + 2*y",
                pressure="100000",
            ),
            InitialConditionBlock(
                name="last",
                domain="x >= 0 and y >= 0",
                density="2",
                velocity_x="0",
                pressure="200000",
            ),
        ]
    )

    (state,) = evaluate_initial_conditions(make_mesh(), settings)

    assert state.density == 2.0
    assert state.velocity_x == 0.0
    assert state.pressure == 200000.0


def test_initial_conditions_use_base_state_outside_all_blocks():
    settings = InitialConditionSettings(
        base_state=InitialState(density=0.9, pressure=90000.0),
        blocks=[InitialConditionBlock(domain="x < 0", density="2")],
    )

    (state,) = evaluate_initial_conditions(make_mesh(), settings)

    assert state.density == 0.9
    assert state.pressure == 90000.0


def test_initial_conditions_reject_non_positive_density():
    settings = InitialConditionSettings(
        blocks=[InitialConditionBlock(density="-1")]
    )

    with pytest.raises(ExpressionError, match="must be positive"):
        evaluate_initial_conditions(make_mesh(), settings)
