from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple, TypeAlias

from numpy import bool_, dtype, ndarray

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

PolicyMakerAsPlayer: TypeAlias = Callable[[GenericPolicyTensor], int]


@dataclass(frozen=True)  # freeze to check for immutability in refactor
class MctsArgs:
    num_mcts_sims: int
    c_puct: float


Display = Callable[[Any], None]
Board = Any
BoardEvaluation: TypeAlias = float
PlayerId: TypeAlias = int
# LengthyTrainExample = tuple[
#     GenericBoardTensor, PlayerID, GenericPolicyTensor, ExampleValue
# ]


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
