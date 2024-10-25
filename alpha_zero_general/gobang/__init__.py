from dataclasses import dataclass
from typing import Any, TypeAlias

from numpy import bool_, dtype, float32, int8, ndarray

from alpha_zero_general import TrainingExample

GobangBoardShapeType: TypeAlias = Any
GobangBoardDataType = int8
GobangBoardTensor: TypeAlias = ndarray[GobangBoardShapeType, dtype[GobangBoardDataType]]
GobangBooleanBoardTensor: TypeAlias = ndarray[GobangBoardDataType, dtype[bool_]]

GobangPolicyShape: TypeAlias = Any
GobangPolicyType = float32
GobangPolicyTensor: TypeAlias = ndarray[GobangPolicyShape, dtype[GobangPolicyType]]
GobangValueTensor: TypeAlias = ndarray[GobangPolicyShape, dtype[GobangPolicyType]]


GobangTrainingExample = TrainingExample[GobangBoardTensor, GobangPolicyTensor]


@dataclass(frozen=True)
class GobangNNArg:
    lr: float
    dropout: float
    epochs: int
    batch_size: int
    cuda: bool
    num_channels: int


CHAR_B = "●"  # U+25CF
CHAR_W = "○"  # U+25CB
