from typing import Any, TypeAlias

from numpy import bool_, dtype, float32, int8, ndarray

Connect4BoardShapeType: TypeAlias = Any
Connect4BoardDataType = int8
Connect4BoardTensor: TypeAlias = ndarray[
    Connect4BoardShapeType, dtype[Connect4BoardDataType]
]
Connect4BooleanBoardTensor: TypeAlias = ndarray[Connect4BoardDataType, dtype[bool_]]

Connect4PolicyShape: TypeAlias = Any
Connect4PolicyType: TypeAlias = float32
Connect4PolicyTensor: TypeAlias = ndarray[
    Connect4PolicyShape, dtype[Connect4PolicyType]
]
Connect4ValueTensor: TypeAlias = ndarray[Connect4PolicyShape, dtype[Connect4PolicyType]]
