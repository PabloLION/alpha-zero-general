from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

from numpy import ndarray

Player = Callable[[ndarray[Any, Any] | list[list[int]]], int]
Display = Callable[[ndarray[Any, Any] | list[list[int]]], None]
TrainExample = tuple[ndarray[Any, Any] | list[list[int]], int, list[float], float]
TrainExamplesHistory = list[list[TrainExample]]
CheckpointFile = Path
TrainExamplesFile = Path


class WinState(NamedTuple):
    is_ended: bool
    winner: int | None
