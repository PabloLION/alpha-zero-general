from typing import Any, TypeAlias

from numpy import bool_, dtype, float32, int8, ndarray

TaflBoardShapeType: TypeAlias = Any
TaflBoardDataType = int8
TaflBoardTensor: TypeAlias = ndarray[TaflBoardShapeType, dtype[TaflBoardDataType]]
TaflBooleanBoardTensor: TypeAlias = ndarray[TaflBoardDataType, dtype[bool_]]

TaflPolicyShape: TypeAlias = Any
TaflPolicyType: TypeAlias = float32
TaflPolicyTensor: TypeAlias = ndarray[TaflPolicyShape, dtype[TaflPolicyType]]
TaflValueTensor: TypeAlias = ndarray[TaflPolicyShape, dtype[TaflPolicyType]]
