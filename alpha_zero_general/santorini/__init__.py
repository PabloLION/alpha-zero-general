from typing import Any, TypeAlias

from numpy import bool_, dtype, float32, int8, ndarray

SantoriniBoardShapeType: TypeAlias = Any
SantoriniBoardDataType = int8
SantoriniBoardTensor: TypeAlias = ndarray[
    SantoriniBoardShapeType, dtype[SantoriniBoardDataType]
]
SantoriniBooleanBoardTensor: TypeAlias = ndarray[SantoriniBoardDataType, dtype[bool_]]

SantoriniPolicyShape: TypeAlias = Any
SantoriniPolicyType = float32
SantoriniPolicyTensor: TypeAlias = ndarray[
    SantoriniPolicyShape, dtype[SantoriniPolicyType]
]
SantoriniValueTensor: TypeAlias = ndarray[
    SantoriniPolicyShape, dtype[SantoriniPolicyType]
]
