from typing import Any, TypeAlias

from numpy import bool_, dtype, float32, int8, ndarray

GobangBoardShapeType: TypeAlias = Any
GobangBoardDataType: TypeAlias = int8
GobangBoardTensor: TypeAlias = ndarray[GobangBoardShapeType, dtype[GobangBoardDataType]]
GobangBooleanBoardTensor: TypeAlias = ndarray[GobangBoardDataType, dtype[bool_]]

GobangPolicyShape: TypeAlias = Any
GobangPolicyType: TypeAlias = float32
GobangPolicyTensor: TypeAlias = ndarray[GobangPolicyShape, dtype[GobangPolicyType]]
GobangValueTensor: TypeAlias = ndarray[GobangPolicyShape, dtype[GobangPolicyType]]


CHAR_B = "●"  # U+25CF
CHAR_W = "○"  # U+25CB
