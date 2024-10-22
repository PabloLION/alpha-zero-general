import argparse
import math
import os
import random
import shutil
import time

import numpy as np
import tensorflow as tf
from NeuralNet import NeuralNet
from utils import *

from alpha_zero_general.type import Any, list, tuple

from .GobangNNet import GobangNNet as onnet

args = dotdict(
    {
        "lr": 0.001,
        "dropout": 0.3,
        "epochs": 10,
        "batch_size": 64,
        "cuda": True,
        "num_channels": 512,
    }
)


class NNetWrapper(NeuralNet):
    def __init__(self, game: Any):
        self.nnet = onnet(game, args)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

    def train(self, examples: list[tuple[np.ndarray, list[float], float]]) -> None:
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

    def predict(self, board: np.ndarray) -> tuple[np.ndarray, float]:
        """
        board: np array with board
        """
        # timing
        # start = time.time()

        # preparing input
        board = board[np.newaxis, :, :]

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
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        self.nnet.model.load_weights(filepath)
