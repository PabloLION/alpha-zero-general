from pathlib import Path
from typing import Any, Callable, NamedTuple, TypedDict, Union

import numpy as np

Player = Callable[[Union[np.ndarray[Any, Any], list[list[int]]]], int]
Display = Callable[[Union[np.ndarray[Any, Any], list[list[int]]]], None]
TrainExample = tuple[
    Union[np.ndarray[Any, Any], list[list[int]]], int, list[float], float
]
TrainExamplesHistory = list[list[TrainExample]]
CheckpointFile = Path
TrainExamplesFile = Path


class BoardState(TypedDict):
    height: int
    width: int
    win_length: int
    np_pieces: np.ndarray[Any, Any]


class WinState(NamedTuple):
    is_ended: bool
    winner: int | None
