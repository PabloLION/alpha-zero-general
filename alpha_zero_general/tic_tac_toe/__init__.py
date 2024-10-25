from dataclasses import dataclass
from typing import Any, TypeAlias

from numpy import bool_, dtype, float32, int8, ndarray

from alpha_zero_general import TrainingExample

TicTacToeBoardShapeType: TypeAlias = Any
TicTacToeBoardDataType = int8
TicTacToeBoardTensor: TypeAlias = ndarray[
    TicTacToeBoardShapeType, dtype[TicTacToeBoardDataType]
]
TicTacToeBooleanBoardTensor: TypeAlias = ndarray[TicTacToeBoardDataType, dtype[bool_]]

TicTacToePolicyShape: TypeAlias = Any
TicTacToePolicyType = float32
TicTacToePolicyTensor: TypeAlias = ndarray[
    TicTacToePolicyShape, dtype[TicTacToePolicyType]
]
TicTacToeValueTensor: TypeAlias = ndarray[
    TicTacToePolicyShape, dtype[TicTacToePolicyType]
]

TicTacToeTrainingExample = TrainingExample[TicTacToeBoardTensor, TicTacToePolicyTensor]


@dataclass(frozen=True)
class TicTacToeNNArg:
    lr: float
    dropout: float
    epochs: int
    batch_size: int
    cuda: bool
    num_channels: int
