import numpy as np
import pytest
from pytest_mock import MockerFixture

from alpha_zero_general import MctsArgs
from alpha_zero_general.connect4.connect4_game import Connect4Game
from alpha_zero_general.connect4.keras.n_net import NNetWrapper as nn
from alpha_zero_general.mcts import MCTS


class TestMCTS:
    @pytest.fixture(autouse=True)
    def setup_method(self):
        self.game = Connect4Game()
        self.nnet = nn(self.game)
        self.args = MctsArgs(num_mcts_sims=10, c_puct=1)
        self.mcts = MCTS(self.game, self.nnet, self.args)

    def test_get_action_prob(self):
        board = self.game.get_init_board()
        canonicalBoard = self.game.get_canonical_form(board, 1)
        probs = self.mcts.get_action_probabilities(canonicalBoard, temperature=1)
        assert len(probs) == self.game.get_action_size()
        assert np.isclose(sum(probs), 1)

    def test_search(self, mocker: MockerFixture):
        mocker.patch(
            "alpha_zero_general.connect4.keras.n_net.NNetWrapper.predict",
            return_value=(np.array([1 / 7] * 7), 0),
        )
        board = self.game.get_init_board()
        canonicalBoard = self.game.get_canonical_form(board, 1)
        v = self.mcts.search(canonicalBoard)
        assert -1 <= v <= 1
