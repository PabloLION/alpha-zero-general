from numpy import array2string, copy

from alpha_zero_general import GenericBoardTensor
from alpha_zero_general.connect4.connect4_logic import Connect4Board
from alpha_zero_general.game import GenericGame

DEFAULT_CONNECT4_BOARD_HEIGHT = 6
DEFAULT_CONNECT4_BOARD_WIDTH = 7
DEFAULT_CONNECT4_BOARD_WIN_LENGTH = 4


class Connect4Game(GenericGame):
    """
    Connect4 Game class implementing the alpha-zero-general Game interface.
    """

    def __init__(
        self,
        height: int = DEFAULT_CONNECT4_BOARD_HEIGHT,
        width: int = DEFAULT_CONNECT4_BOARD_WIDTH,
        win_length: int = DEFAULT_CONNECT4_BOARD_WIN_LENGTH,
        np_pieces: GenericBoardTensor | None = None,
    ):
        # GenericGame.__init__(self)
        self._base_board = Connect4Board(height, width, win_length, np_pieces)

    def get_init_board(self) -> GenericBoardTensor:
        return self._base_board.np_pieces

    def get_board_size(self) -> tuple[int, int]:
        return (self._base_board.height, self._base_board.width)

    def get_action_size(self) -> int:
        return self._base_board.width

    def get_next_state(
        self, board: GenericBoardTensor, player: int, action: int
    ) -> tuple[GenericBoardTensor, int]:
        """Returns a copy of the board with updated move, original board is unmodified."""
        b = self._base_board.with_np_pieces(np_pieces=copy(board))
        b.add_stone(action, player)
        return b.np_pieces, -player

    def get_valid_moves(
        self, board: GenericBoardTensor, player: int
    ) -> GenericBoardTensor:
        "Any zero value in top row in a valid move"
        return self._base_board.with_np_pieces(np_pieces=board).get_valid_moves()

    def get_game_ended(self, board: GenericBoardTensor, player: int) -> float:
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
        self, board: GenericBoardTensor, player: int
    ) -> GenericBoardTensor:
        # Flip player from 1 to -1
        return board * player

    def get_symmetries(
        self, board: GenericBoardTensor, pi: list[float]
    ) -> list[tuple[GenericBoardTensor, list[float]]]:
        """Board is left/right board symmetric"""
        return [(board, pi), (board[:, ::-1], pi[::-1])]

    def string_representation(self, board: GenericBoardTensor) -> str:
        return array2string(board)

    @staticmethod
    def display(board: GenericBoardTensor) -> None:
        print(" -----------------------")
        print(" ".join(map(str, range(len(board[0])))))
        print(board)
        print(" -----------------------")
