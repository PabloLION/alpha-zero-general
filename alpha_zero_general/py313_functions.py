import warnings
from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def deprecated(func: F) -> F:
    """
    See https://docs.python.org/3.13/library/warnings.html#warnings.deprecated
    """

    def wrapper(*args, **kwargs):  # type: ignore
        warnings.warn(
            f"{func.__name__} is deprecated and will be removed in future versions.",
            DeprecationWarning,
            stacklevel=2,
        )
        return func(*args, **kwargs)

    return wrapper  # type: ignore
