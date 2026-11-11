from fractions import Fraction

import pytest

from src.validate import validate_range as check_range


@pytest.fixture
def validate_range():
    return check_range


@pytest.mark.parametrize("include_min", [False, True])
@pytest.mark.parametrize("include_max", [False, True])
@pytest.mark.parametrize(
    ("minimum", "maximum", "value"),
    [(None, 1.0, -100.0), (0.0, None, 100.0), (None, None, -100.0)],
)
def test_optional_bounds(validate_range, minimum, maximum, value, include_min, include_max):
    validate_range(
        "value", value, minimum, maximum,
        include_min=include_min, include_max=include_max,
    )


@pytest.mark.parametrize(
    ("minimum", "maximum", "value", "options", "interval"),
    [
        (None, 1.0, 2.0, {}, "(-inf, 1.0]"),
        (0.0, None, -1.0, {}, "[0.0, inf)"),
        (None, 1.0, 1.0, {"include_max": False}, "(-inf, 1.0)"),
        (0.0, None, 0.0, {"include_min": False}, "(0.0, inf)"),
    ],
)
def test_remaining_bound_is_enforced(
    validate_range, minimum, maximum, value, options, interval
):
    with pytest.raises(ValueError) as error:
        validate_range("value", value, minimum, maximum, **options)
    assert str(error.value) == f"value must be in {interval}"


@pytest.mark.parametrize(
    ("minimum", "maximum"), [(None, None), (None, 1.0), (0.0, None), (0.0, 1.0)]
)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_non_finite_is_rejected(validate_range, minimum, maximum, value):
    with pytest.raises(ValueError):
        validate_range("value", value, minimum, maximum)


def test_default_range_is_unchanged(validate_range):
    validate_range("value", 0.0)
    validate_range("value", 1.0)
    for value in (-0.1, 1.1, float("inf"), -float("inf")):
        with pytest.raises(ValueError):
            validate_range("value", value)


@pytest.mark.parametrize("value", ["0.5", None, True, False, 1j])
def test_invalid_types_are_rejected(validate_range, value):
    with pytest.raises(TypeError):
        validate_range("value", value, None, None)


def test_fraction_and_large_integer_keep_exact_values():
    assert check_range("value", Fraction(1, 3), None, Fraction(1, 2)) is None
    check_range("value", 10 ** 400, 0, None)
    exact = Fraction(10 ** 400 + 1, 10 ** 400)
    check_range("value", exact, 1, None, include_min=False)
    with pytest.raises(ValueError):
        check_range("value", exact, None, 1)


@pytest.mark.parametrize("include_min", [False, True])
@pytest.mark.parametrize("include_max", [False, True])
@pytest.mark.parametrize("value", [0, 1])
def test_endpoint_inclusion(validate_range, include_min, include_max, value):
    options = dict(include_min=include_min, include_max=include_max)
    allowed = include_min if value == 0 else include_max
    if allowed:
        validate_range("value", value, **options)
    else:
        with pytest.raises(ValueError):
            validate_range("value", value, **options)
