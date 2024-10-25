from dataclasses import dataclass
from typing import Any, TypeAlias

from numpy import bool_, dtype, float32, int8, ndarray

from alpha_zero_general import TrainingExample

TicTacToe3DBoardShapeType: TypeAlias = Any
TicTacToe3DBoardDataType = int8
TicTacToe3DBoardTensor: TypeAlias = ndarray[
    TicTacToe3DBoardShapeType, dtype[TicTacToe3DBoardDataType]
]
TicTacToe3DBooleanBoardTensor: TypeAlias = ndarray[
    TicTacToe3DBoardDataType, dtype[bool_]
]

TicTacToe3DPolicyShape: TypeAlias = Any
TicTacToe3DPolicyType: TypeAlias = float32
TicTacToe3DPolicyTensor: TypeAlias = ndarray[
    TicTacToe3DPolicyShape, dtype[TicTacToe3DPolicyType]
]
TicTacToe3DValueTensor: TypeAlias = ndarray[
    TicTacToe3DPolicyShape, dtype[TicTacToe3DPolicyType]
]

TicTacToe3DTrainingExample = TrainingExample[
    TicTacToe3DBoardTensor, TicTacToe3DPolicyTensor
]


@dataclass(frozen=True)
class TicTacToe3DNNArg:
    lr: float
    dropout: float
    epochs: int
    batch_size: int
    cuda: bool
    num_channels: int
