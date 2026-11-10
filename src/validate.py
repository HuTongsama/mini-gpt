def validate_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")

def validate_float_range(
    name: str,
    value: float,
    min_value: float = 0.0,
    max_value: float = 1.0,
    *,
    include_min: bool = True,
    include_max: bool = True,
) -> None:
    if not isinstance(value, float):
        raise TypeError(f"{name} must be a float")

    lower_ok = value >= min_value if include_min else value > min_value
    upper_ok = value <= max_value if include_max else value < max_value

    if not (lower_ok and upper_ok):
        left = "[" if include_min else "("
        right = "]" if include_max else ")"
        raise ValueError(
            f"{name} must be in {left}{min_value}, {max_value}{right}"
        )

def validate_boolean(name: str, value: bool) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")