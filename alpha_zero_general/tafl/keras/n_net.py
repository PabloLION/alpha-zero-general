import os
import sys
import time

import numpy as np

sys.path.append("../..")

from alpha_zero_general.neural_net import NeuralNetInterface
from alpha_zero_general.tafl import (
    TaflBoardTensor,
    TaflBooleanBoardTensor,
    TaflNNArg,
    TaflPolicyTensor,
    TaflTrainingExample,
)
from alpha_zero_general.tafl.keras.tafl_n_net import TaflNNet
from alpha_zero_general.tafl.tafl_game import TaflGame

args = TaflNNArg(
    lr=0.001, dropout=0.3, epochs=10, batch_size=64, cuda=False, num_channels=512
)


class NNetWrapper(
    NeuralNetInterface[TaflBoardTensor, TaflBooleanBoardTensor, TaflPolicyTensor]
):

    nn: TaflNNet

    def __init__(self, game: TaflGame):
        self.nn = TaflNNet(game, args)
        self.board_x, self.board_y = game.get_board_size()
        self.action_size = game.get_action_size()

    def train(self, examples: list[TaflTrainingExample]):
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

    def predict(self, board: TaflBoardTensor) -> tuple[TaflPolicyTensor, float]:
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
            raise FileNotFoundError("No model in path {}".format(filepath))
        self.nn.model.load_weights(filepath)
