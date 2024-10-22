import pytest
import numpy as np
from alpha_zero_general.connect4.Connect4Game import Connect4Game
from alpha_zero_general.connect4.keras.NNet import NNetWrapper as nn
from alpha_zero_general.utils import dotdict
from pytest_mock import mocker


class TestNeuralNet:
    @pytest.fixture(autouse=True)
    def setup_method(self):
        self.game = Connect4Game()
        self.nnet = nn(self.game)
        self.args = dotdict(
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

    def test_train(self):
        examples = [(self.game.getInitBoard(), [1] * self.game.getActionSize(), 1)]
        self.nnet.train(examples)
        assert True  # If no exception is raised, the test passes

    def test_predict(self):
        board = self.game.getInitBoard()
        pi, v = self.nnet.predict(board)
        assert len(pi) == self.game.getActionSize()
        assert -1 <= v <= 1

    def test_save_checkpoint(self):
        self.nnet.save_checkpoint(folder="./temp", filename="test.pth.tar")
        assert True  # If no exception is raised, the test passes

    def test_load_checkpoint(self):
        self.nnet.save_checkpoint(folder="./temp", filename="test.pth.tar")
        self.nnet.load_checkpoint(folder="./temp", filename="test.pth.tar")
        assert True  # If no exception is raised, the test passes
