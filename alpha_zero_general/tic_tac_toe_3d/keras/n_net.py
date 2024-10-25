import os
import time

import numpy as np

from alpha_zero_general.neural_net import NeuralNetInterface
from alpha_zero_general.tic_tac_toe_3d import (
    TicTacToe3DBoardTensor,
    TicTacToe3DBooleanBoardTensor,
    TicTacToe3DNNArg,
    TicTacToe3DPolicyTensor,
    TicTacToe3DTrainingExample,
)
from alpha_zero_general.tic_tac_toe_3d.keras.tic_tac_toe_3d_n_net import (
    TicTacToeNNet as onnet,
)
from alpha_zero_general.tic_tac_toe_3d.tic_tac_toe_3d_game import TicTacToe3DGame

"""
NeuralNet wrapper class for the TicTacToeNNet.

Author: Evgeny Tyurin, github.com/evg-tyurin
Date: Jan 5, 2018.

Based on (copy-pasted from) the NNet by SourKream and Surag Nair.
"""

args = TicTacToe3DNNArg(
    lr=0.001, dropout=0.3, epochs=10, batch_size=64, cuda=False, num_channels=512
)


class NNetWrapper(
    NeuralNetInterface[
        TicTacToe3DBoardTensor, TicTacToe3DBooleanBoardTensor, TicTacToe3DPolicyTensor
    ]
):
    def __init__(self, game: TicTacToe3DGame):
        self.nn = onnet(game, args)
        self.board_z, self.board_x, self.board_y = game.get_board_size()
        self.action_size = game.get_action_size()

    def train(self, examples: list[TicTacToe3DTrainingExample]) -> None:
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

    def predict(
        self, board: TicTacToe3DBoardTensor
    ) -> tuple[TicTacToe3DPolicyTensor, float]:
        """
        board: np array with board
        """
        # timing
        time.time()

        # preparing input
        board = board[np.newaxis, :, :]

        # run
        pi, v = self.nn.model.predict(board, verbose=0)

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
        if not os.path.exists(filepath):
            raise FileNotFoundError("No model in path '{}'".format(filepath))
        self.nn.model.load_weights(filepath)
