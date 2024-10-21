import pytest
import numpy as np
from alpha_zero_general.othello.OthelloGame import OthelloGame
from pytest_mock import mocker

class TestOthelloGame:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.game = OthelloGame(8)

    def test_getInitBoard(self):
        board = self.game.getInitBoard()
        assert board.shape == (8, 8)
        assert board[3, 4] == 1
        assert board[4, 3] == 1
        assert board[3, 3] == -1
        assert board[4, 4] == -1

    def test_getBoardSize(self):
        assert self.game.getBoardSize() == (8, 8)

    def test_getActionSize(self):
        assert self.game.getActionSize() == 65

    def test_getNextState(self):
        board = self.game.getInitBoard()
        next_board, next_player = self.game.getNextState(board, 1, 19)
        assert next_board[2, 3] == 1
        assert next_board[3, 3] == 1
        assert next_board[4, 3] == 1
        assert next_player == -1

    def test_getValidMoves(self):
        board = self.game.getInitBoard()
        valid_moves = self.game.getValidMoves(board, 1)
        assert valid_moves[19] == 1
        assert valid_moves[26] == 1
        assert valid_moves[37] == 1
        assert valid_moves[44] == 1

    def test_getGameEnded(self):
        board = self.game.getInitBoard()
        assert self.game.getGameEnded(board, 1) == 0

    def test_getCanonicalForm(self):
        board = self.game.getInitBoard()
        canonical_board = self.game.getCanonicalForm(board, 1)
        assert np.array_equal(board, canonical_board)

    def test_getSymmetries(self):
        board = self.game.getInitBoard()
        pi = [1/65] * 65
        symmetries = self.game.getSymmetries(board, pi)
        assert len(symmetries) == 8

    def test_stringRepresentation(self):
        board = self.game.getInitBoard()
        board_str = self.game.stringRepresentation(board)
        assert isinstance(board_str, bytes)
