from pathlib import Path
from typing import Any, Callable, NamedTuple, Union

from numpy import ndarray

Player = Callable[[Union[ndarray[Any, Any], list[list[int]]]], int]
Display = Callable[[Union[ndarray[Any, Any], list[list[int]]]], None]
TrainExample = tuple[Union[ndarray[Any, Any], list[list[int]]], int, list[float], float]
TrainExamplesHistory = list[list[TrainExample]]
CheckpointFile = Path
TrainExamplesFile = Path


class WinState(NamedTuple):
    is_ended: bool
    winner: int | None
