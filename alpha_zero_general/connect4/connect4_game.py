from numpy import array2string, copy

from alpha_zero_general.connect4 import (
    Connect4BoardTensor,
    Connect4BooleanBoardTensor,
    Connect4PolicyTensor,
)
from alpha_zero_general.connect4.connect4_logic import Connect4Board
from alpha_zero_general.game import GenericGame

DEFAULT_CONNECT4_BOARD_HEIGHT = 6
DEFAULT_CONNECT4_BOARD_WIDTH = 7
DEFAULT_CONNECT4_BOARD_WIN_LENGTH = 4


class Connect4Game(
    GenericGame[Connect4BoardTensor, Connect4BooleanBoardTensor, Connect4PolicyTensor]
):
    """
    Connect4 Game class implementing the alpha-zero-general Game interface.
    """

    def __init__(
        self,
        height: int = DEFAULT_CONNECT4_BOARD_HEIGHT,
        width: int = DEFAULT_CONNECT4_BOARD_WIDTH,
        win_length: int = DEFAULT_CONNECT4_BOARD_WIN_LENGTH,
        np_pieces: Connect4BoardTensor | None = None,
    ):
        # GenericGame.__init__(self)
        self._base_board = Connect4Board(height, width, win_length, np_pieces)

    def get_init_board(self) -> Connect4BoardTensor:
        return self._base_board.chip_tensor

    def get_board_size(self) -> tuple[int, int]:
        return (self._base_board.height, self._base_board.width)

    def get_action_size(self) -> int:
        return self._base_board.width

    def get_next_state(
        self, board: Connect4BoardTensor, player: int, action: int
    ) -> tuple[Connect4BoardTensor, int]:
        """Returns a copy of the board with updated move, original board is unmodified."""
        b = self._base_board.with_np_pieces(np_pieces=copy(board))
        b.add_chip(action, player)
        return b.chip_tensor, -player

    def get_valid_moves(
        self, board: Connect4BoardTensor, player: int
    ) -> Connect4BooleanBoardTensor:
        "Any zero value in top row in a valid move"
        return self._base_board.with_np_pieces(np_pieces=board).get_valid_moves()

    def get_game_ended(self, board: Connect4BoardTensor, player: int) -> float:
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

    def get_canonical_form(
        self, board: Connect4BoardTensor, player: int
    ) -> Connect4BoardTensor:
        # Flip player from 1 to -1
        return board * player

    def get_symmetries(
        self, board: Connect4BoardTensor, pi: Connect4PolicyTensor
    ) -> list[tuple[Connect4BoardTensor, Connect4PolicyTensor]]:
        """Board is left/right board symmetric"""
        return [(board, pi), (board[:, ::-1], pi[::-1])]

    def string_representation(self, board: Connect4BoardTensor) -> str:
        return array2string(board)

    def get_board_hash(self, board: Connect4BoardTensor) -> int:
        return hash(board.tobytes())

    @staticmethod
    def display(board: Connect4BoardTensor) -> None:
        print(" -----------------------")
        print(" ".join(map(str, range(len(board[0])))))
        print(board)
        print(" -----------------------")
