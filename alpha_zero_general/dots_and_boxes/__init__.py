from typing import Any, TypeAlias

from numpy import bool_, dtype, float32, int8, ndarray

DotsAndBoxesBoardShapeType: TypeAlias = Any
DotsAndBoxesBoardDataType: TypeAlias = int8
DotsAndBoxesBoardTensor: TypeAlias = ndarray[
    DotsAndBoxesBoardShapeType, dtype[DotsAndBoxesBoardDataType]
]
DotsAndBoxesBooleanBoardTensor: TypeAlias = ndarray[
    DotsAndBoxesBoardDataType, dtype[bool_]
]

DotsAndBoxesPolicyShape: TypeAlias = Any
DotsAndBoxesPolicyType: TypeAlias = float32
DotsAndBoxesPolicyTensor: TypeAlias = ndarray[
    DotsAndBoxesPolicyShape, dtype[DotsAndBoxesPolicyType]
]
DotsAndBoxesValueTensor: TypeAlias = ndarray[
    DotsAndBoxesPolicyShape, dtype[DotsAndBoxesPolicyType]
]
