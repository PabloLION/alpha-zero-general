from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, NamedTuple, TypeAlias, TypedDict, TypeVar, Union

from numpy import dtype, int8, ndarray

Player = Callable[[Union[ndarray[Any, Any], list[list[int]]]], int]
Display = Callable[[Union[ndarray[Any, Any], list[list[int]]]], None]
TrainExample = tuple[Union[ndarray[Any, Any], list[list[int]]], int, list[float], float]
TrainExamplesHistory = list[list[TrainExample]]
CheckpointFile = Path
TrainExamplesFile = Path


BoardDataType: TypeAlias = int8
BoardShapeType: TypeAlias = Any
BoardMatrix: TypeAlias = ndarray[BoardShapeType, dtype[BoardDataType]]


class BoardState(TypedDict):
    height: int
    width: int
    win_length: int
    np_pieces: BoardMatrix


class WinState(NamedTuple):
    is_ended: bool
    winner: int | None


@dataclass
class MctsArgs:
    numMCTSSims: int
    cpuct: float
