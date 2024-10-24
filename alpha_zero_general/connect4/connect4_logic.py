import numpy as np

from alpha_zero_general import GenericBoardTensor, WinState
from alpha_zero_general.connect4 import Connect4BoardDataType, Connect4BoardTensor
from alpha_zero_general.connect4.connect4_game import Connect4BooleanBoardTensor


class Connect4Board:
    """
    Connect4 Board.
    The pieces of this game are called "chips" (or "discs").
    """

    chip_tensor: Connect4BoardTensor

    def __init__(
        self,
        height: int,
        width: int,
        win_length: int,
        chip_tensor: Connect4BoardTensor | None = None,
    ):
        "Set up initial board configuration."
        self.height = height
        self.width = width
        self.win_length = win_length

        if chip_tensor is None:
            self.chip_tensor = np.zeros(
                [self.height, self.width], dtype=Connect4BoardDataType
            )
        else:
            self.chip_tensor = chip_tensor
            assert self.chip_tensor.shape == (self.height, self.width)

    def add_chip(self, column: int, player: int) -> None:
        "Create copy of board containing new stone."
        (available_idx,) = np.nonzero(self.chip_tensor[:, column] == 0)
        if len(available_idx) == 0:
            raise ValueError("Can't play column %s on board %s" % (column, self))

        self.chip_tensor[available_idx[-1]][column] = player

    def get_valid_moves(self) -> Connect4BooleanBoardTensor:
        # here we should have new types for Connect4GenericBoardTensor / Tensor and
        # Connect4BoardFirstRowTensor #TODO
        "Any zero value in top row in a valid move"
        return self.chip_tensor[0] == 0

    def get_win_state(self) -> WinState:
        for player in [-1, 1]:
            player_pieces = self.chip_tensor == -player
            # Check rows & columns for win
            if (
                self._is_straight_winner(player_pieces)
                or self._is_straight_winner(player_pieces.transpose())
                or self._is_diagonal_winner(player_pieces)
            ):
                return WinState(True, -player)

        # draw has very little value.
        if not self.get_valid_moves().any():
            return WinState(True, None)

        # Game is not ended yet.
        return WinState(False, None)

    def with_np_pieces(self, np_pieces: GenericBoardTensor) -> "Connect4Board":
        """Create copy of board with specified pieces."""
        if np_pieces is None:
            np_pieces = self.chip_tensor
        return Connect4Board(self.height, self.width, self.win_length, np_pieces)

    def _is_diagonal_winner(self, player_pieces: GenericBoardTensor) -> bool:
        """Checks if player_pieces contains a diagonal win."""
        win_length = self.win_length
        for i in range(len(player_pieces) - win_length + 1):
            for j in range(len(player_pieces[0]) - win_length + 1):
                if all(player_pieces[i + x][j + x] for x in range(win_length)):
                    return True
            for j in range(win_length - 1, len(player_pieces[0])):
                if all(player_pieces[i + x][j - x] for x in range(win_length)):
                    return True
        return False

    def _is_straight_winner(self, player_pieces: GenericBoardTensor) -> bool:
        """Checks if player_pieces contains a vertical or horizontal win."""
        run_lengths = [
            player_pieces[:, i : i + self.win_length].sum(axis=1)
            for i in range(len(player_pieces) - self.win_length + 2)
        ]
        return max([x.max() for x in run_lengths]) >= self.win_length

    def __str__(self) -> str:
        return str(self.chip_tensor)
