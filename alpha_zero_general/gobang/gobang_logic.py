from numpy import zeros

from alpha_zero_general.gobang import GobangBoardDataType, GobangBoardTensor


class Board:
    """
    Gobang Board.

    0 for empty, 1 for white, -1 for black
    """

    pieces: GobangBoardTensor

    def __init__(self, n: int):
        "Set up initial board configuration."
        self.n = n
        # Create the empty board array.
        self.pieces = zeros((n, n), dtype=GobangBoardDataType)
        for i in range(self.n):
            self.pieces[i] = [0] * self.n

    # add [][] indexer syntax to the Board
    def __getitem__(self, index: int) -> list[int]:
        return self.pieces[index]

    def get_legal_moves(self, color: int) -> list[tuple[int, int]]:
        """
        Returns all the legal moves for the given color.
        (1 for white, -1 for black)

        REFACTOR: #FIX/FATAL
        This is against my knowledge of the go game.
        In some cases, e.g. 6x2 board with (0,O,_) for (black,white,empty)
        00_00o | This board on the left has two empty spaces, but only one
        ooooo_ | legal move for both players.

        #TODO/PERF: use numpy array
        """
        # moves = set()  # stores the legal moves.
        moves = list[tuple[int, int]]()

        # Get all empty locations.
        for y in range(self.n):
            for x in range(self.n):
                if self[x][y] == 0:
                    moves.append((x, y))
        return moves

    def has_legal_moves(self) -> bool:
        # TODO/PERF: use numpy array
        """Returns True if has legal move else False"""
        # Get all empty locations.
        for y in range(self.n):
            for x in range(self.n):
                if self[x][y] == 0:
                    return True
        return False

    def execute_move(self, move: tuple[int, int], color: int) -> None:
        """Perform the given move on the board; flips pieces as necessary.
        color gives the color pf the piece to play (1=white,-1=black)
        """
        (x, y) = move
        assert self[x][y] == 0
        self[x][y] = color
