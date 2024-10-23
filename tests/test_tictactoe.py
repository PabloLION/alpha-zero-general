import numpy as np
import pytest

from alpha_zero_general.tictactoe.tic_tac_toe_game import TicTacToeGame


class TestTicTacToeGame:
    @pytest.fixture(autouse=True)
    def setup_method(self):
        self.game = TicTacToeGame()

    def test_getInitBoard(self):
        board = self.game.get_init_board()
        expected_board = np.zeros((3, 3))
        assert np.array_equal(board, expected_board)

    def test_getBoardSize(self):
        assert self.game.get_board_size() == (3, 3)

    def test_getActionSize(self):
        assert self.game.get_action_size() == 10

    def test_getNextState(self):
        board = self.game.get_init_board()
        next_board, next_player = self.game.get_next_state(board, 1, 0)
        expected_board = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
        assert np.array_equal(next_board, expected_board)
        assert next_player == -1

    def test_getValidMoves(self):
        board = self.game.get_init_board()
        valid_moves = self.game.get_valid_moves(board, 1)
        expected_valid_moves = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 0])
        assert np.array_equal(valid_moves, expected_valid_moves)

    def test_getGameEnded(self):
        board = np.array([[1, 1, 1], [0, -1, -1], [0, 0, 0]])
        assert self.game.get_game_ended(board, 1) == 1
        board = np.array([[-1, -1, -1], [0, 1, 1], [0, 0, 0]])
        assert self.game.get_game_ended(board, -1) == -1
        board = np.array([[1, -1, 1], [-1, 1, -1], [1, -1, 1]])
        assert self.game.get_game_ended(board, 1) == 1e-4

    def test_getCanonicalForm(self):
        board = np.array([[1, -1, 0], [0, 1, -1], [0, 0, 1]])
        canonical_board = self.game.get_canonical_form(board, 1)
        expected_board = np.array([[1, -1, 0], [0, 1, -1], [0, 0, 1]])
        assert np.array_equal(canonical_board, expected_board)
        canonical_board = self.game.get_canonical_form(board, -1)
        expected_board = np.array([[-1, 1, 0], [0, -1, 1], [0, 0, -1]])
        assert np.array_equal(canonical_board, expected_board)

    def test_getSymmetries(self):
        board = np.array([[1, 0, -1], [0, 1, 0], [-1, 0, 1]])
        pi = [0.1] * 9 + [0]
        symmetries = self.game.get_symmetries(board, pi)
        assert len(symmetries) == 8

    def test_stringRepresentation(self):
        board = np.array([[1, 0, -1], [0, 1, 0], [-1, 0, 1]])
        board_string = self.game.string_representation(board)
        expected_string = board.tostring()
        assert board_string == expected_string
