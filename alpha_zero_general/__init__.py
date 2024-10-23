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


@dataclass(frozen=True)  # freeze to check for immutability in refactor
class MctsArgs:
    numMCTSSims: int
    cpuct: float
