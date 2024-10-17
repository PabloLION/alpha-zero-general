import sys
import numpy as np
from types import Board, Game

sys.path.append('..')
from Game import Game
from .Connect4Logic import Board


class Connect4Game(Game):
    """
    Connect4 Game class implementing the alpha-zero-general Game interface.
    """

    def __init__(self, height: int = None, width: int = None, win_length: int = None, np_pieces: np.ndarray = None):
        Game.__init__(self)
        self._base_board = Board(height, width, win_length, np_pieces)

    def getInitBoard(self) -> np.ndarray:
        return self._base_board.np_pieces

    def getBoardSize(self) -> tuple[int, int]:
        return (self._base_board.height, self._base_board.width)

    def getActionSize(self) -> int:
        return self._base_board.width

    def getNextState(self, board: np.ndarray, player: int, action: int) -> tuple[np.ndarray, int]:
        """Returns a copy of the board with updated move, original board is unmodified."""
        b = self._base_board.with_np_pieces(np_pieces=np.copy(board))
        b.add_stone(action, player)
        return b.np_pieces, -player

    def getValidMoves(self, board: np.ndarray, player: int) -> np.ndarray:
        "Any zero value in top row in a valid move"
        return self._base_board.with_np_pieces(np_pieces=board).get_valid_moves()

    def getGameEnded(self, board: np.ndarray, player: int) -> float:
        b = self._base_board.with_np_pieces(np_pieces=board)
        winstate = b.get_win_state()
        if winstate.is_ended:
            if winstate.winner is None:
                # draw has very little value.
                return 1e-4
            elif winstate.winner == player:
                return +1
            elif winstate.winner == -player:
                return -1
            else:
                raise ValueError('Unexpected winstate found: ', winstate)
        else:
            # 0 used to represent unfinished game.
            return 0

    def getCanonicalForm(self, board: np.ndarray, player: int) -> np.ndarray:
        # Flip player from 1 to -1
        return board * player

    def getSymmetries(self, board: np.ndarray, pi: list[float]) -> list[tuple[np.ndarray, list[float]]]:
        """Board is left/right board symmetric"""
        return [(board, pi), (board[:, ::-1], pi[::-1])]

    def stringRepresentation(self, board: np.ndarray) -> str:
        return board.tostring()

    @staticmethod
    def display(board: np.ndarray) -> None:
        print(" -----------------------")
        print(' '.join(map(str, range(len(board[0])))))
        print(board)
        print(" -----------------------")
