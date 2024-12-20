import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from alpha_zero_general import TrainingExample
from alpha_zero_general.dots_and_boxes.dots_and_boxes_game import (
    DotsAndBoxesBoardTensor,
    DotsAndBoxesBooleanBoardTensor,
    DotsAndBoxesPolicyTensor,
)
from alpha_zero_general.dots_and_boxes.keras.dots_and_boxes_n_net import (
    DotsAndBoxesNNet,
)
from alpha_zero_general.neural_net import NeuralNetInterface


@dataclass(frozen=True)
class DotsAndBoxesBoardKerasArg:
    lr: float
    dropout: float
    epochs: int
    batch_size: int
    cuda: bool
    num_channels: int


args = DotsAndBoxesBoardKerasArg(
    lr=0.001,
    dropout=0.3,
    epochs=10,
    batch_size=64,
    cuda=True,
    num_channels=512,
)


def normalize_score(board: DotsAndBoxesBoardTensor) -> None:
    p1_score = board[:, 0, -1]
    p2_score = board[:, 1, -1]
    score = p1_score - p2_score

    n = board.shape[-1] - 1

    max_score = n**2
    min_score = -max_score

    min_normalized, max_normalized = 0, 1
    normalized_score = ((score - max_score) / (min_score - max_score)) * (
        min_normalized - max_normalized
    ) + max_normalized

    board[:, 0, -1] = normalized_score
    board[:, 1, -1] = 0


class DotsAndBoxesNNInterface(
    NeuralNetInterface[
        DotsAndBoxesBoardTensor,
        DotsAndBoxesBooleanBoardTensor,
        DotsAndBoxesPolicyTensor,
    ]
):
    def __init__(self, game: Any):
        self.nn = DotsAndBoxesNNet(game, args)
        self.board_x, self.board_y = game.get_board_size()
        self.action_size = game.get_action_size()

    def train(
        self,
        examples: list[
            TrainingExample[DotsAndBoxesBoardTensor, DotsAndBoxesPolicyTensor]
        ],
    ) -> None:
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        input_boards, target_pis, target_vs = list(zip(*examples))
        input_boards = np.asarray(input_boards)

        normalize_score(input_boards)

        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)
        self.nn.model.fit(
            x=input_boards,
            y=[target_pis, target_vs],
            batch_size=args.batch_size,
            epochs=args.epochs,
        )

    def predict(
        self, board: DotsAndBoxesBoardTensor
    ) -> tuple[DotsAndBoxesPolicyTensor, float]:
        """
        board: np array with board
        """

        board = np.copy(board)
        board = board[np.newaxis, :, :]
        normalize_score(board)

        pi, v = self.nn.model.predict(board, verbose=0)

        return pi[0], v[0]

    def save_checkpoint(
        self, folder: str = "checkpoint", filename: str = "checkpoint.pth.tar"
    ) -> None:
        # change extension
        filename = filename.split(".")[0] + ".weights.h5"

        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print(
                "Checkpoint Directory does not exist! Making directory {}".format(
                    folder
                )
            )
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        self.nn.model.save_weights(filepath)

    def load_checkpoint(
        self, folder: str = "checkpoint", filename: str = "checkpoint.pth.tar"
    ) -> None:
        # change extension
        filename = filename.split(".")[0] + ".weights.h5"

        filepath = os.path.join(folder, filename)
        self.nn.model.load_weights(filepath)
