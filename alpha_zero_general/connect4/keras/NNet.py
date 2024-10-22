import logging
import os
import time

import numpy as np

from alpha_zero_general.connect4.connect4_game import Connect4Game
from alpha_zero_general.NeuralNet import NeuralNet
from alpha_zero_general.type import BoardMatrix
from alpha_zero_general.utils import dotdict

log = logging.getLogger(__name__)


from alpha_zero_general.connect4.keras.connect4_nnet import Connect4NNet as onnet

args = dotdict(
    {
        "lr": 0.001,
        "dropout": 0.3,
        "epochs": 10,
        "batch_size": 64,
        "cuda": True,
        "num_channels": 128,
        "num_residual_layers": 20,
    }
)


class NNetWrapper(NeuralNet):

    def __init__(self, game: Connect4Game):
        self.nnet = onnet(game, args)
        self.nnet.model.summary()
        self.board_x, self.board_y = game.get_board_size()
        self.action_size = game.get_action_size()

    def train(self, examples: list[tuple[BoardMatrix, list[float], float]]) -> None:
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        input_boards, target_pis, target_vs = list(zip(*examples))
        input_boards = np.asarray(input_boards)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)
        self.nnet.model.fit(
            x=input_boards,
            y=[target_pis, target_vs],
            batch_size=args.batch_size,
            epochs=args.epochs,
        )

    def predict(self, board: BoardMatrix) -> tuple[np.ndarray, float]:
        """
        board: np array with board
        """
        # timing
        time.time()

        # preparing input
        board = board[np.newaxis, :, :]

        # run
        pi, v = self.nnet.model.predict(board, verbose=False)

        # print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return pi[0], v[0]

    def save_checkpoint(
        self, folder: str = "checkpoint", filename: str = "checkpoint.pth.tar"
    ) -> None:
        # change extension
        filename = filename.split(".")[0] + ".h5"

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
        self.nnet.model.save_weights(filepath)

    def load_checkpoint(
        self, folder: str = "checkpoint", filename: str = "checkpoint.pth.tar"
    ) -> None:
        # change extension
        filename = filename.split(".")[0] + ".h5"

        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        # if not os.path.exists(filepath):
        # raise("No model in path {}".format(filepath))
        self.nnet.model.load_weights(filepath)
        log.info("Loading Weights...")
