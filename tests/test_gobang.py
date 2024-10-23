import numpy as np
import pytest

from alpha_zero_general.gobang.gobang_game import GobangGame


class TestGobangGame:
    @pytest.fixture(autouse=True)
    def setup_method(self):
        self.game = GobangGame()

    def test_get_init_board(self):
        board = self.game.get_init_board()
        assert board.shape == (15, 15)
        assert np.all(board == 0)

    def test_get_board_size(self):
        assert self.game.get_board_size() == (15, 15)

    def test_get_action_size(self):
        assert self.game.get_action_size() == 226

    def test_get_next_state(self):
        board = self.game.get_init_board()
        next_board, next_player = self.game.get_next_state(board, 1, 0)
        assert next_board[0, 0] == 1
        assert next_player == -1

    def test_get_valid_moves(self):
        board = self.game.get_init_board()
        valid_moves = self.game.get_valid_moves(board, 1)
        assert np.sum(valid_moves) == 225
        assert valid_moves[-1] == 0

    def test_get_game_ended(self):
        board = self.game.get_init_board()
        assert self.game.get_game_ended(board, 1) == 0
        board[0, :5] = 1
        assert self.game.get_game_ended(board, 1) == 1
        board[0, :5] = -1
        assert self.game.get_game_ended(board, -1) == -1

    def test_getCanonicalForm(self):
        board = self.game.get_init_board()
        canonical_board = self.game.get_canonical_form(board, 1)
        assert np.array_equal(board, canonical_board)
        canonical_board = self.game.get_canonical_form(board, -1)
        assert np.array_equal(board, -canonical_board)

    def test_get_symmetries(self):
        board = self.game.get_init_board()
        pi = [0] * 225 + [1]
        symmetries = self.game.get_symmetries(board, pi)
        assert len(symmetries) == 8

    def test_string_representation(self):
        board = self.game.get_init_board()
        board_str = self.game.string_representation(board)
        assert isinstance(board_str, bytes)
