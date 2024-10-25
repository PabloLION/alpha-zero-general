# https://en.wikipedia.org/wiki/Tafl_games


class TaflGameVariant:
    size: int = 0
    board = []
    pieces = []

    def mirror_across_board(
        self, board_size: int, initial_positions: list[list[int]]
    ) -> list[list[int]]:
        hs = board_size // 2
        a_quarter = initial_positions.copy()
        for b in initial_positions:
            if b[0] != b[1]:
                a_quarter.extend([[b[1], b[0], b[2]]])
        whole = a_quarter.copy()
        for b in a_quarter:
            if b[0] != hs:
                whole.extend([[board_size - b[0] - 1, b[1], b[2]]])
            if b[1] != hs:
                whole.extend([[b[0], board_size - b[1] - 1, b[2]]])
            if b[0] != hs and b[1] != hs:
                whole.extend([[board_size - b[0] - 1, board_size - b[1] - 1, b[2]]])
        return whole


class Brandubh(TaflGameVariant):
    def __init__(self):
        self.size = 7
        self.board = self.mirror_across_board(self.size, [[0, 0, 1], [3, 3, 2]])
        self.pieces = self.mirror_across_board(
            self.size, [[3, 0, -1], [3, 1, -1], [3, 2, 1], [3, 3, 2]]
        )


class ArdRi(TaflGameVariant):
    def __init__(self):
        self.size = 7
        self.board = self.mirror_across_board(self.size, [[0, 0, 1], [3, 3, 2]])
        self.pieces = self.mirror_across_board(
            self.size,
            [[2, 0, -1], [3, 0, -1], [3, 1, -1], [3, 2, 1], [2, 2, 1], [3, 3, 2]],
        )


class Tablut(TaflGameVariant):
    def __init__(self):
        self.size = 9
        self.board = self.mirror_across_board(self.size, [[0, 0, 1], [4, 4, 2]])
        self.pieces = self.mirror_across_board(
            self.size,
            [[3, 0, -1], [4, 0, -1], [4, 1, -1], [4, 2, 1], [4, 3, 1], [4, 4, 2]],
        )


class Tawlbwrdd(TaflGameVariant):
    def __init__(self):
        self.size = 11
        self.board = self.mirror_across_board(self.size, [[0, 0, 1], [5, 5, 2]])
        self.pieces = self.mirror_across_board(
            self.size,
            [
                [4, 0, -1],
                [5, 0, -1],
                [4, 1, -1],
                [5, 2, -1],
                [5, 3, 1],
                [5, 4, 1],
                [4, 4, 1],
                [5, 5, 2],
            ],
        )


class Hnefatafl(TaflGameVariant):
    def __init__(self):
        self.size = 11
        self.board = self.mirror_across_board(self.size, [[0, 0, 1], [5, 5, 2]])
        self.pieces = self.mirror_across_board(
            self.size,
            [
                [3, 0, -1],
                [4, 0, -1],
                [5, 0, -1],
                [5, 1, -1],
                [5, 3, 1],
                [5, 4, 1],
                [4, 4, 1],
                [5, 5, 2],
            ],
        )


class AleaEvangelii(TaflGameVariant):
    def __init__(self):
        self.size = 19
        self.board = self.mirror_across_board(self.size, [[0, 0, 1], [9, 9, 2]])
        self.pieces = self.mirror_across_board(
            self.size,
            [
                [2, 0, -1],
                [5, 0, -1],
                [5, 2, -1],
                [7, 3, -1],
                [9, 3, -1],
                [6, 4, -1],
                [5, 5, -1],
                [8, 4, 1],
                [9, 6, 1],
                [8, 7, 1],
                [9, 8, 1],
                [9, 9, 2],
            ],
        )
