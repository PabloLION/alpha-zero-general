import pytest
import numpy as np
from gobang.GobangGame import GobangGame

class TestGobangGame:

    @pytest.fixture(autouse=True)
    def setup_method(self):
        self.game = GobangGame()

    def test_getInitBoard(self):
        board = self.game.getInitBoard()
        assert board.shape == (15, 15)
        assert np.all(board == 0)

    def test_getBoardSize(self):
        assert self.game.getBoardSize() == (15, 15)

    def test_getActionSize(self):
        assert self.game.getActionSize() == 226

    def test_getNextState(self):
        board = self.game.getInitBoard()
        next_board, next_player = self.game.getNextState(board, 1, 0)
        assert next_board[0, 0] == 1
        assert next_player == -1

    def test_getValidMoves(self):
        board = self.game.getInitBoard()
        valid_moves = self.game.getValidMoves(board, 1)
        assert np.sum(valid_moves) == 225
        assert valid_moves[-1] == 0

    def test_getGameEnded(self):
        board = self.game.getInitBoard()
        assert self.game.getGameEnded(board, 1) == 0
        board[0, :5] = 1
        assert self.game.getGameEnded(board, 1) == 1
        board[0, :5] = -1
        assert self.game.getGameEnded(board, -1) == -1

    def test_getCanonicalForm(self):
        board = self.game.getInitBoard()
        canonical_board = self.game.getCanonicalForm(board, 1)
        assert np.array_equal(board, canonical_board)
        canonical_board = self.game.getCanonicalForm(board, -1)
        assert np.array_equal(board, -canonical_board)

    def test_getSymmetries(self):
        board = self.game.getInitBoard()
        pi = [0] * 225 + [1]
        symmetries = self.game.getSymmetries(board, pi)
        assert len(symmetries) == 8

    def test_stringRepresentation(self):
        board = self.game.getInitBoard()
        board_str = self.game.stringRepresentation(board)
        assert isinstance(board_str, bytes)
