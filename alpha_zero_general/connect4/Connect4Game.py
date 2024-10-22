from numpy import array2string, copy

from alpha_zero_general.connect4.Connect4Logic import Connect4Board
from alpha_zero_general.Game import Game
from alpha_zero_general.type import BoardMatrix

DEFAULT_CONNECT4_BOARD_HEIGHT = 6
DEFAULT_CONNECT4_BOARD_WIDTH = 7
DEFAULT_CONNECT4_BOARD_WIN_LENGTH = 4


class Connect4Game(Game):
    """
    Connect4 Game class implementing the alpha-zero-general Game interface.
    """

    def __init__(
        self,
        height: int = DEFAULT_CONNECT4_BOARD_HEIGHT,
        width: int = DEFAULT_CONNECT4_BOARD_WIDTH,
        win_length: int = DEFAULT_CONNECT4_BOARD_WIN_LENGTH,
        np_pieces: BoardMatrix | None = None,
    ):
        Game.__init__(self)
        self._base_board = Connect4Board(height, width, win_length, np_pieces)

    def getInitBoard(self) -> BoardMatrix:
        return self._base_board.np_pieces

    def getBoardSize(self) -> tuple[int, int]:
        return (self._base_board.height, self._base_board.width)

    def getActionSize(self) -> int:
        return self._base_board.width

    def getNextState(
        self, board: BoardMatrix, player: int, action: int
    ) -> tuple[BoardMatrix, int]:
        """Returns a copy of the board with updated move, original board is unmodified."""
        b = self._base_board.with_np_pieces(np_pieces=copy(board))
        b.add_stone(action, player)
        return b.np_pieces, -player

    def getValidMoves(self, board: BoardMatrix, player: int) -> BoardMatrix:
        "Any zero value in top row in a valid move"
        return self._base_board.with_np_pieces(np_pieces=board).get_valid_moves()

    def getGameEnded(self, board: BoardMatrix, player: int) -> float:
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
                raise ValueError("Unexpected winstate found: ", winstate)
        else:
            # 0 used to represent unfinished game.
            return 0

    def getCanonicalForm(self, board: BoardMatrix, player: int) -> BoardMatrix:
        # Flip player from 1 to -1
        return board * player

    def getSymmetries(
        self, board: BoardMatrix, pi: list[float]
    ) -> list[tuple[BoardMatrix, list[float]]]:
        """Board is left/right board symmetric"""
        return [(board, pi), (board[:, ::-1], pi[::-1])]

    def stringRepresentation(self, board: BoardMatrix) -> str:
        return array2string(board)

    @staticmethod
    def display(board: BoardMatrix) -> None:
        print(" -----------------------")
        print(" ".join(map(str, range(len(board[0])))))
        print(board)
        print(" -----------------------")
