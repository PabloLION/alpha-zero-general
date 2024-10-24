from typing import Any, TypeAlias

from numpy import dtype, float32, ndarray

Connect4PolicyShape: TypeAlias = Any
Connect4PolicyType: TypeAlias = float32
Connect4PolicyTensor: TypeAlias = ndarray[
    Connect4PolicyShape, dtype[Connect4PolicyType]
]
Connect4ValueTensor: TypeAlias = ndarray[Connect4PolicyShape, dtype[Connect4PolicyType]]
