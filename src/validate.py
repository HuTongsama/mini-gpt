from numbers import Real

def validate_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")

def validate_non_negative_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")

def validate_range(
    name: str,
    value: Real,
    min_value: Real | None = 0.0,
    max_value: Real | None = 1.0,
    *,
    include_min: bool = True,
    include_max: bool = True,
) -> None:
    """Validate a finite real without conversion, rejecting bool.

    None removes a bound and ignores its inclusion flag. Defaults to [0, 1].
    """
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    if not (-float("inf") < value < float("inf")):
        raise ValueError(f"{name} must be finite")

    lower_ok = min_value is None or (
        value >= min_value if include_min else value > min_value
    )
    upper_ok = max_value is None or (
        value <= max_value if include_max else value < max_value
    )

    if not (lower_ok and upper_ok):
        left = "[" if min_value is not None and include_min else "("
        right = "]" if max_value is not None and include_max else ")"
        lower = min_value if min_value is not None else "-inf"
        upper = max_value if max_value is not None else "inf"
        raise ValueError(
            f"{name} must be in {left}{lower}, {upper}{right}"
        )
    
def validate_boolean(name: str, value: bool) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")