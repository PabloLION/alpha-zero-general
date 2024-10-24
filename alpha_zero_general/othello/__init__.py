from dataclasses import dataclass
from typing import Any, TypeAlias

from numpy import bool_, dtype, float32, int8, ndarray

from alpha_zero_general import TrainingExample

OthelloBoardShapeType: TypeAlias = Any
OthelloBoardDataType = int8
OthelloBoardTensor: TypeAlias = ndarray[
    OthelloBoardShapeType, dtype[OthelloBoardDataType]
]
OthelloBooleanBoardTensor: TypeAlias = ndarray[OthelloBoardDataType, dtype[bool_]]

OthelloPolicyShape: TypeAlias = Any
OthelloPolicyType: TypeAlias = float32
OthelloPolicyTensor: TypeAlias = ndarray[OthelloPolicyShape, dtype[OthelloPolicyType]]
OthelloValueTensor: TypeAlias = ndarray[OthelloPolicyShape, dtype[OthelloPolicyType]]

OthelloTrainingExample = TrainingExample[OthelloBoardTensor, OthelloPolicyTensor]


@dataclass(frozen=True)
class OthelloNNArg:
    lr: float
    dropout: float
    epochs: int
    batch_size: int
    cuda: bool
    num_channels: int
