from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeAlias

from numpy import bool_, dtype, ndarray

GenericBoardDataType: TypeAlias = Any
GenericBoardShapeType: TypeAlias = Any
GenericBoardTensor: TypeAlias = ndarray[
    GenericBoardShapeType, dtype[GenericBoardDataType]
]
GenericBooleanBoardTensor: TypeAlias = ndarray[GenericBoardDataType, dtype[bool_]]

GenericPolicyDataType: TypeAlias = Any
GenericPolicyShapeType: TypeAlias = GenericBoardShapeType  # same as board shape
GenericPolicyTensor: TypeAlias = ndarray[
    GenericPolicyShapeType, dtype[GenericPolicyDataType]
]
# GenericPolicyTensor: TypeAlias = list[float]
# #TODO: GenericPolicyTensor should be tensor or list[float]? consider perf too.

PolicyMakerAsPlayer: TypeAlias = Callable[[GenericPolicyTensor], int]


@dataclass(frozen=True)  # freeze to check for immutability in refactor
class MctsArgs:
    numMCTSSims: int
    cpuct: float
