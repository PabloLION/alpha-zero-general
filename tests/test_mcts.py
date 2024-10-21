import numpy as np
import pytest
from alpha_zero_general.connect4.Connect4Game import Connect4Game
from alpha_zero_general.connect4.keras.NNet import NNetWrapper as nn
from alpha_zero_general.MCTS import MCTS
from alpha_zero_general.utils import dotdict


class TestMCTS:

    @pytest.fixture(autouse=True)
    def setup_method(self):
        self.game = Connect4Game()
        self.nnet = nn(self.game)
        self.args = dotdict({"numMCTSSims": 10, "cpuct": 1})
        self.mcts = MCTS(self.game, self.nnet, self.args)

    def test_getActionProb(self):
        board = self.game.getInitBoard()
        canonicalBoard = self.game.getCanonicalForm(board, 1)
        probs = self.mcts.getActionProb(canonicalBoard, temp=1)
        assert len(probs) == self.game.getActionSize()
        assert np.isclose(sum(probs), 1)

    def test_search(self, mocker):
        mocker.patch(
            "alpha_zero_general.connect4.keras.NNetWrapper.predict",
            return_value=(np.array([1 / 7] * 7), 0),
        )
        board = self.game.getInitBoard()
        canonicalBoard = self.game.getCanonicalForm(board, 1)
        v = self.mcts.search(canonicalBoard)
        assert -1 <= v <= 1
