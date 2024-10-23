import pytest

from alpha_zero_general.connect4.connect4_game import Connect4Game
from alpha_zero_general.connect4.keras.n_net import NNetWrapper as nn
from alpha_zero_general.utils import dotdict


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
        examples = [(self.game.get_init_board(), [1] * self.game.get_action_size(), 1)]
        self.nnet.train(examples)
        assert True  # If no exception is raised, the test passes

    def test_predict(self):
        board = self.game.get_init_board()
        pi, v = self.nnet.predict(board)
        assert len(pi) == self.game.get_action_size()
        assert -1 <= v <= 1

    def test_save_checkpoint(self):
        self.nnet.save_checkpoint(folder="./temp", filename="test.pth.tar")
        assert True  # If no exception is raised, the test passes

    def test_load_checkpoint(self):
        self.nnet.save_checkpoint(folder="./temp", filename="test.pth.tar")
        self.nnet.load_checkpoint(folder="./temp", filename="test.pth.tar")
        assert True  # If no exception is raised, the test passes
