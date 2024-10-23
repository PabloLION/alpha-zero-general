import numpy as np
import pytest

from alpha_zero_general.dotsandboxes.dots_and_boxes_game import DotsAndBoxesGame


class TestDotsAndBoxesGame:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.game = DotsAndBoxesGame(n=3)

    def test_getInitBoard(self):
        board = self.game.get_init_board()
        assert board.shape == (7, 4)
        assert np.all(board == 0)

    def test_get_board_size(self):
        assert self.game.get_board_size() == (7, 4)

    def test_get_action_size(self):
        assert self.game.get_action_size() == 25

    def test_getNextState(self):
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

    def test_getGameEnded(self):
        board = self.game.get_init_board()
        assert self.game.get_game_ended(board, 1) == 0
        board[0, -1] = 5
        board[1, -1] = 4
        assert self.game.get_game_ended(board, 1) == 1
        assert self.game.get_game_ended(board, -1) == -1

    def test_getCanonicalForm(self):
        board = self.game.get_init_board()
        canonical_board = self.game.get_canonical_form(board, 1)
        assert np.array_equal(board, canonical_board)
        board[0, -1] = 5
        board[1, -1] = 4
        canonical_board = self.game.get_canonical_form(board, -1)
        assert canonical_board[0, -1] == 4
        assert canonical_board[1, -1] == 5

    def test_getSymmetries(self):
        board = self.game.get_init_board()
        pi = [1] * 25
        symmetries = self.game.get_symmetries(board, pi)
        assert len(symmetries) == 8

    def test_stringRepresentation(self):
        board = self.game.get_init_board()
        board_str = self.game.string_representation(board)
        assert board_str == np.array2string(board)
