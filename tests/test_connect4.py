import numpy as np
import pytest

from alpha_zero_general.connect4.connect4_game import Connect4Game


class TestConnect4Game:
    @pytest.fixture(autouse=True)
    def setup_method(self):
        self.game = Connect4Game()

    def test_getInitBoard(self):
        board = self.game.getInitBoard()
        assert board.shape == (6, 7)
        assert np.all(board == 0)

    def test_getBoardSize(self):
        assert self.game.getBoardSize() == (6, 7)

    def test_getActionSize(self):
        assert self.game.getActionSize() == 7

    def test_getNextState(self):
        board = self.game.getInitBoard()
        next_board, next_player = self.game.getNextState(board, 1, 0)
        assert next_board[5, 0] == 1
        assert next_player == -1

    def test_getValidMoves(self):
        board = self.game.getInitBoard()
        valid_moves = self.game.getValidMoves(board, 1)
        assert np.all(valid_moves == 1)

    def test_getGameEnded(self):
        board = self.game.getInitBoard()
        assert self.game.getGameEnded(board, 1) == 0

    def test_getCanonicalForm(self):
        board = self.game.getInitBoard()
        canonical_board = self.game.getCanonicalForm(board, 1)
        assert np.array_equal(board, canonical_board)

    def test_getSymmetries(self):
        board = self.game.getInitBoard()
        pi = [1 / 7] * 7
        symmetries = self.game.getSymmetries(board, pi)
        assert len(symmetries) == 2

    def test_stringRepresentation(self):
        board = self.game.getInitBoard()
        board_str = self.game.stringRepresentation(board)
        assert isinstance(board_str, bytes)
