from typing import Union, Callable, TypedDict
from pathlib import Path
import numpy as np

Player = Callable[[Union[np.ndarray, list[list[int]]]], int]
Game = Any
Display = Callable[[Union[np.ndarray, list[list[int]]]], None]
TrainExample = tuple[Union[np.ndarray, list[list[int]]], int, list[float], float]
TrainExamplesHistory = list[list[TrainExample]]
CheckpointFile = Path
TrainExamplesFile = Path

class BoardState(TypedDict):
    height: int
    width: int
    win_length: int
    np_pieces: np.ndarray

class WinState(TypedDict):
    is_ended: bool
    winner: int
