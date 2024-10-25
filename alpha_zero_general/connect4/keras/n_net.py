import logging
import os
import time
from dataclasses import dataclass

import numpy as np

from alpha_zero_general import TrainingExample
from alpha_zero_general.connect4.connect4_game import (
    Connect4BoardTensor,
    Connect4BooleanBoardTensor,
    Connect4Game,
    Connect4PolicyTensor,
)
from alpha_zero_general.neural_net import NeuralNetInterface

log = logging.getLogger(__name__)


from alpha_zero_general.connect4.keras.connect4_n_net import Connect4NNet


@dataclass(frozen=True)
class Connect4NNArgs:
    lr: float
    dropout: float
    epochs: int
    batch_size: int
    cuda: bool
    num_channels: int
    num_residual_layers: int


args: Connect4NNArgs = Connect4NNArgs(
    lr=0.001,
    dropout=0.3,
    epochs=10,
    batch_size=64,
    cuda=True,
    num_channels=128,
    num_residual_layers=20,
)


class Connect4NNInterface(
    NeuralNetInterface[
        Connect4BoardTensor, Connect4BooleanBoardTensor, Connect4PolicyTensor
    ]
):

    nn: Connect4NNet
    board_x: int
    board_y: int
    action_size: int

    def __init__(self, game: Connect4Game):
        self.nn = Connect4NNet(game, args)
        # self.nn.model.summary()
        self.board_x, self.board_y = game.get_board_size()
        self.action_size = game.get_action_size()

    def train(
        self, examples: list[TrainingExample[Connect4BoardTensor, Connect4PolicyTensor]]
    ) -> None:
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        input_boards, target_pis, target_vs = list(zip(*examples))
        input_boards = np.asarray(input_boards)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)
        self.nn.model.fit(
            x=input_boards,
            y=[target_pis, target_vs],
            batch_size=args.batch_size,
            epochs=args.epochs,
        )

    def predict(self, board: Connect4BoardTensor) -> tuple[Connect4PolicyTensor, float]:
        """
        board: np array with board
        """
        # timing
        time.time()

        # preparing input
        expanded_board = board[np.newaxis, :, :]

        # run
        pi, v = self.nn.model.predict(expanded_board, verbose=0)

        # print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
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

        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        # if not os.path.exists(filepath):
        # raise("No model in path {}".format(filepath))
        self.nn.model.load_weights(filepath)
        log.info("Loading Weights...")
