import numpy as np
import pytest

from alpha_zero_general.dots_and_boxes.dots_and_boxes_game import DotsAndBoxesGame


class TestDotsAndBoxesGame:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.game = DotsAndBoxesGame(n=3)

    def test_get_init_board(self):
        board = self.game.get_init_board()
        assert board.shape == (7, 4)
        assert np.all(board == 0)

    def test_get_board_size(self):
        assert self.game.get_board_size() == (7, 4)

    def test_get_action_size(self):
        assert self.game.get_action_size() == 25

    def test_get_next_state(self):
        board = self.game.get_init_board()
        next_board, next_player = self.game.get_next_state(board, 1, 0)
        assert next_player == -1
        assert next_board[0, 0] == 1

    def test_get_valid_moves(self):
        board = self.game.get_init_board()
        valid_moves = self.game.get_valid_moves(board, 1)
        assert len(valid_moves) == 25
        assert np.all(valid_moves[:-1] == 1)
        assert valid_moves[-1] == 0

    def test_get_game_ended(self):
        board = self.game.get_init_board()
        # no player should win on an empty board
        assert self.game.get_game_ended(board, 1) == 0
        return
        # #TODO: logic not checked, need cases where a player wins
        board[0, -1] = 5
        board[1, -1] = 4
        assert self.game.get_game_ended(board, 1) == 1
        assert self.game.get_game_ended(board, -1) == -1

    def test_get_canonical_form(self):
        board = self.game.get_init_board()
        canonical_board = self.game.get_canonical_form(board, 1)
        assert np.array_equal(board, canonical_board)
        board[0, -1] = 5
        board[1, -1] = 4
        canonical_board = self.game.get_canonical_form(board, -1)
        assert canonical_board[0, -1] == 4
        assert canonical_board[1, -1] == 5

    def test_get_symmetries(self):
        board = self.game.get_init_board()
        pi = [1] * 25
        symmetries = self.game.get_symmetries(board, pi)
        assert len(symmetries) == 8

    def test_get_board_str(self):
        board = self.game.get_init_board()
        board_str = self.game.get_board_str(board)
        assert board_str == np.array2string(board)
