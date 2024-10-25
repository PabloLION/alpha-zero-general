"""
Refactor note:

The `mirror_across_board`, (old `expand eighth`) is not very convenient to use.
string and return a board/pieces. Both adding new board /piece (asymmetric) and
writing test benefit from this. (we need the string board in test anyway)

The output is better to be an numpy array.
"""

# https://en.wikipedia.org/wiki/Tafl_games
from alpha_zero_general.tafl import TaflPiece


class TaflGameVariant:
    size: int = 0
    board: list[TaflPiece]
    pieces: list[TaflPiece]

    @staticmethod
    def mirror_across_board(
        board_size: int, positions_to_mirror: list[TaflPiece]
    ) -> list[TaflPiece]:
        """
        This was called `expand_eighth` in the original code.
        It mirrors the board across the center of the board 8 times.

        """
        hs = board_size // 2
        a_quarter = positions_to_mirror.copy()
        for b in positions_to_mirror:
            if b[0] != b[1]:
                a_quarter.append(TaflPiece(b[1], b[0], b[2]))
        whole = a_quarter.copy()
        for b in a_quarter:
            if b[0] != hs:
                whole.append(TaflPiece(board_size - b[0] - 1, b[1], b[2]))
            if b[1] != hs:
                whole.append(TaflPiece(b[0], board_size - b[1] - 1, b[2]))
            if b[0] != hs and b[1] != hs:
                whole.append(
                    TaflPiece(board_size - b[0] - 1, board_size - b[1] - 1, b[2])
                )
        return whole


class Brandubh(TaflGameVariant):
    def __init__(self):
        self.size = 7
        self.board = self.mirror_across_board(
            self.size,
            [
                TaflPiece(0, 0, 1),
                TaflPiece(3, 3, 2),
            ],
        )
        self.pieces = self.mirror_across_board(
            self.size,
            [
                TaflPiece(3, 0, -1),
                TaflPiece(3, 1, -1),
                TaflPiece(3, 2, 1),
                TaflPiece(3, 3, 2),
            ],
        )


class ArdRi(TaflGameVariant):
    def __init__(self):
        self.size = 7
        self.board = self.mirror_across_board(
            self.size,
            [
                TaflPiece(0, 0, 1),
                TaflPiece(3, 3, 2),
            ],
        )
        self.pieces = self.mirror_across_board(
            self.size,
            [
                TaflPiece(2, 0, -1),
                TaflPiece(3, 0, -1),
                TaflPiece(3, 1, -1),
                TaflPiece(3, 2, 1),
                TaflPiece(2, 2, 1),
                TaflPiece(3, 3, 2),
            ],
        )


class Tablut(TaflGameVariant):
    def __init__(self):
        self.size = 9
        self.board = self.mirror_across_board(
            self.size,
            [
                TaflPiece(0, 0, 1),
                TaflPiece(4, 4, 2),
            ],
        )
        self.pieces = self.mirror_across_board(
            self.size,
            [
                TaflPiece(3, 0, -1),
                TaflPiece(4, 0, -1),
                TaflPiece(4, 1, -1),
                TaflPiece(4, 2, 1),
                TaflPiece(4, 3, 1),
                TaflPiece(4, 4, 2),
            ],
        )


class Tawlbwrdd(TaflGameVariant):
    def __init__(self):
        self.size = 11
        self.board = self.mirror_across_board(
            self.size,
            [
                TaflPiece(0, 0, 1),
                TaflPiece(5, 5, 2),
            ],
        )
        self.pieces = self.mirror_across_board(
            self.size,
            # [4, 0, -1],[5, 0, -1],[4, 1, -1],[5, 2, -1],[5, 3, 1],[5, 4, 1],[4, 4, 1],[5, 5, 2],
            [
                TaflPiece(4, 0, -1),
                TaflPiece(5, 0, -1),
                TaflPiece(4, 1, -1),
                TaflPiece(5, 2, -1),
                TaflPiece(5, 3, 1),
                TaflPiece(5, 4, 1),
                TaflPiece(4, 4, 1),
                TaflPiece(5, 5, 2),
            ],
        )


class Hnefatafl(TaflGameVariant):
    def __init__(self):
        self.size = 11
        self.board = self.mirror_across_board(
            self.size,
            [
                TaflPiece(0, 0, 1),
                TaflPiece(5, 5, 2),
            ],
        )
        self.pieces = self.mirror_across_board(
            self.size,
            [
                # [3, 0, -1],[4, 0, -1],[5, 0, -1],[5, 1, -1],[5, 3, 1],[5, 4, 1],[4, 4, 1],[5, 5, 2],
                TaflPiece(3, 0, -1),
                TaflPiece(4, 0, -1),
                TaflPiece(5, 0, -1),
                TaflPiece(5, 1, -1),
                TaflPiece(5, 3, 1),
                TaflPiece(5, 4, 1),
                TaflPiece(4, 4, 1),
                TaflPiece(5, 5, 2),
            ],
        )


class AleaEvangelii(TaflGameVariant):
    def __init__(self):
        self.size = 19
        self.board = self.mirror_across_board(
            self.size,
            [
                TaflPiece(0, 0, 1),
                TaflPiece(9, 9, 2),
            ],
        )
        self.pieces = self.mirror_across_board(
            self.size,
            [
                # [2, 0, -1],[5, 0, -1],[5, 2, -1],[7, 3, -1],[9, 3, -1],[6, 4, -1],[5, 5, -1],[8, 4, 1],[9, 6, 1],[8, 7, 1],[9, 8, 1],[9, 9, 2],
                TaflPiece(2, 0, -1),
                TaflPiece(5, 0, -1),
                TaflPiece(5, 2, -1),
                TaflPiece(7, 3, -1),
                TaflPiece(9, 3, -1),
            ],
        )
