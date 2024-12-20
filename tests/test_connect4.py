"""
Refactor note:
#TODO: merge with another test_connect4.py
"""

import numpy as np
import pytest

from alpha_zero_general.connect4.connect4_game import Connect4Game


class TestConnect4Game:
    @pytest.fixture(autouse=True)
    def setup_method(self):
        self.game = Connect4Game()

    def test_get_init_board(self):
        board = self.game.get_init_board()
        assert board.shape == (6, 7)
        assert np.all(board == 0)

    def test_get_board_size(self):
        assert self.game.get_board_size() == (6, 7)

    def test_get_action_size(self):
        assert self.game.get_action_size() == 7

    def test_get_next_state(self):
        board = self.game.get_init_board()
        next_board, next_player = self.game.get_next_state(board, 1, 0)
        assert next_board[5, 0] == 1
        assert next_player == -1

    def test_get_valid_moves(self):
        board = self.game.get_init_board()
        valid_moves = self.game.get_valid_moves(board, 1)
        assert np.all(valid_moves == 1)

    def test_get_game_ended(self):
        board = self.game.get_init_board()
        assert self.game.get_game_ended(board, 1) == 0

    def test_get_canonical_form(self):
        board = self.game.get_init_board()
        canonical_board = self.game.get_canonical_form(board, 1)
        assert np.array_equal(board, canonical_board)

    def test_get_symmetries(self):
        board = self.game.get_init_board()
        pi = np.array([1 / 7] * 7)
        symmetries = self.game.get_symmetries(board, pi)
        assert len(symmetries) == 2

    def test_get_board_str(self):
        # TODO: need a `get hash`
        board = self.game.get_init_board()
        board_str = self.game.get_board_str(board)
        assert isinstance(board_str, str)
