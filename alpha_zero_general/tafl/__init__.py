from typing import Any, Literal, NamedTuple, TypeAlias

from numpy import bool_, dtype, float32, int8, ndarray

TaflBoardShapeType: TypeAlias = Any
TaflBoardDataType = int8
TaflBoardTensor: TypeAlias = ndarray[TaflBoardShapeType, dtype[TaflBoardDataType]]
TaflBooleanBoardTensor: TypeAlias = ndarray[TaflBoardDataType, dtype[bool_]]

TaflPolicyShape: TypeAlias = Any
TaflPolicyType = float32
TaflPolicyTensor: TypeAlias = ndarray[TaflPolicyShape, dtype[TaflPolicyType]]
TaflValueTensor: TypeAlias = ndarray[TaflPolicyShape, dtype[TaflPolicyType]]


class TaflPiece(NamedTuple):
    x: int
    y: int
    type: Literal[-99, -1, 0, 1, 2]
    # type: -99: removed, -1: black, 0: empty, 1: white, 2: king
