import numpy as np
import pytest

from alpha_zero_general.othello.othello_game import OthelloGame


class TestOthelloGame:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.game = OthelloGame(8)

    def test_get_init_board(self):
        board = self.game.get_init_board()
        assert board.shape == (8, 8)
        assert board[3, 4] == 1
        assert board[4, 3] == 1
        assert board[3, 3] == -1
        assert board[4, 4] == -1

    def test_get_board_size(self):
        assert self.game.get_board_size() == (8, 8)

    def test_get_action_size(self):
        assert self.game.get_action_size() == 65

    def test_get_next_state(self):
        board = self.game.get_init_board()
        next_board, next_player = self.game.get_next_state(board, 1, 19)
        assert next_board[2, 3] == 1
        assert next_board[3, 3] == 1
        assert next_board[4, 3] == 1
        assert next_player == -1

    def test_get_valid_moves(self):
        board = self.game.get_init_board()
        valid_moves = self.game.get_valid_moves(board, 1)
        assert valid_moves[19] == 1
        assert valid_moves[26] == 1
        assert valid_moves[37] == 1
        assert valid_moves[44] == 1

    def test_get_game_ended(self):
        board = self.game.get_init_board()
        assert self.game.get_game_ended(board, 1) == 0

    def test_getCanonicalForm(self):
        board = self.game.get_init_board()
        canonical_board = self.game.get_canonical_form(board, 1)
        assert np.array_equal(board, canonical_board)

    def test_get_symmetries(self):
        board = self.game.get_init_board()
        pi = [1 / 65] * 65
        symmetries = self.game.get_symmetries(board, pi)
        assert len(symmetries) == 8

    def test_string_representation(self):
        board = self.game.get_init_board()
        board_str = self.game.string_representation(board)
        assert isinstance(board_str, bytes)
