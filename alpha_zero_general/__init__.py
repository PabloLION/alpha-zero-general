from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple, TypeAlias, TypeVar

from numpy import bool_, dtype, ndarray, random

# Board Tensors
GenericBoardDataType: TypeAlias = Any  # #TODO: TBD
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
BoardTensorType = TypeVar("BoardTensorType", bound=GenericBoardTensor)
BooleanBoardType = TypeVar("BooleanBoardType", bound=GenericBooleanBoardTensor)
PolicyTensorType = TypeVar("PolicyTensorType", bound=GenericPolicyTensor)


PolicyMakerAsPlayer: TypeAlias = Callable[[GenericPolicyTensor], int]


@dataclass(frozen=True)  # freeze to check for immutability in refactor
class MctsArgs:
    num_mcts_sims: int
    c_puct: float


BoardEvaluation: TypeAlias = float
PlayerId: TypeAlias = int


## coach.py
class RawTrainExample(NamedTuple):
    """
    (canonical_board, current_player, pi, v)
    pi is the MCTS informed policy vector, v is +1 if
    the player eventually won the game, else -1.
    """

    board: GenericBoardTensor
    current_player: PlayerId
    policy: GenericPolicyTensor
    neutral_evaluation: BoardEvaluation  # from neutral perspective


class TrainExample(NamedTuple):
    board: GenericBoardTensor
    policy: GenericPolicyTensor
    evaluation: BoardEvaluation  # from player's perspective


TrainExampleHistory = list[list[TrainExample]]
CheckpointFile = str
TrainExamplesFile = str

# not categorized
Player = Callable[[ndarray[Any, Any] | list[list[int]]], int]
Display = Callable[[Any], None]
# maybe Display = Callable[[ndarray[Any, Any] | list[list[int]]], None]
# if possible, we should use GenericBoardTensor
CheckpointFile = Path
TrainExamplesFile = Path


class WinState(NamedTuple):
    is_ended: bool
    winner: int | None


## Constants

EPS = 1e-8
RANDOM_SEED = 32342
RNG = random.default_rng(RANDOM_SEED)
