"""
Author: Eric P. Nichols
Date: Feb 8, 2008.
Board class.
Board data:
  1=white, -1=black, 0=empty
  first dim is column , 2nd is row:
     pieces[1][7] is the square in column 2,
     at the opposite end of the board in row 8.
Squares are stored and manipulated as (x,y) tuples.
x is the column, y is the row.
"""


class OthelloBoard:
    # list of all 8 directions on the board, as (x,y) offsets
    __directions = [
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1),
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, 1),
    ]

    pieces: list[list[int]]

    def __init__(self, n: int) -> None:
        "Set up initial board configuration."

        self.n = n
        # Create the empty board array.
        self.pieces = [[0] * n for _ in range(n)]

        # Set up the initial 4 pieces.
        self.pieces[int(self.n / 2) - 1][int(self.n / 2)] = 1
        self.pieces[int(self.n / 2)][int(self.n / 2) - 1] = 1
        self.pieces[int(self.n / 2) - 1][int(self.n / 2) - 1] = -1
        self.pieces[int(self.n / 2)][int(self.n / 2)] = -1

    # #TODO/PERF: use numpy array
    # add [][] indexer syntax to the Board
    def __getitem__(self, index: int) -> list[int]:
        return self.pieces[index]

    def count_diff(self, color: int) -> int:
        """Counts the # pieces of the given color
        (1 for white, -1 for black, 0 for empty spaces)"""
        count = 0
        for y in range(self.n):
            for x in range(self.n):
                if self[x][y] == color:
                    count += 1
                if self[x][y] == -color:
                    count -= 1
        return count

    def get_legal_moves(self, color: int) -> list[tuple[int, int]]:
        """Returns all the legal moves for the given color.
        (1 for white, -1 for black
        """
        # #TODO/PERF: use numpy array
        moves = set[tuple[int, int]]()  # stores the legal moves.
        # Get all the squares with pieces of the given color.
        for y in range(self.n):
            for x in range(self.n):
                if self[x][y] == color:
                    new_moves = self.get_moves_for_square((x, y))
                    moves.update(new_moves)
        return list(moves)

    def has_legal_moves(self, color: int) -> bool:
        for y in range(self.n):
            for x in range(self.n):
                if self[x][y] == color:
                    new_moves = self.get_moves_for_square((x, y))
                    if len(new_moves) > 0:
                        return True
        return False

    def get_moves_for_square(self, square: tuple[int, int]) -> list[tuple[int, int]]:
        """Returns all the legal moves that use the given square as a base.
        That is, if the given square is (3,4) and it contains a black piece,
        and (3,5) and (3,6) contain white pieces, and (3,7) is empty, one
        of the returned moves is (3,7) because everything from there to (3,4)
        is flipped.
        """
        (x, y) = square

        # determine the color of the piece.
        color = self[x][y]

        # skip empty source squares.
        if color == 0:
            return []

        # search all possible directions.
        moves = list[tuple[int, int]]()
        for direction in self.__directions:
            move = self._discover_move(square, direction)
            if move:
                # print(square,move,direction)
                moves.append(move)

        # return the generated move list
        return moves

    def execute_move(self, move: tuple[int, int], color: int) -> None:
        """Perform the given move on the board; flips pieces as necessary.
        color gives the color pf the piece to play (1=white,-1=black)
        """

        # Much like move generation, start at the new piece's square and
        # follow it on all 8 directions to look for a piece allowing flipping.

        # Add the piece to the empty square.
        # print(move)
        flips = [
            flip
            for direction in self.__directions
            for flip in self._get_flips(move, direction, color)
        ]
        assert len(list(flips)) > 0
        for x, y in flips:
            # print(self[x][y],color)
            self[x][y] = color

    def _discover_move(self, origin: tuple[int, int], direction: tuple[int, int]):
        """Returns the endpoint for a legal move, starting at the given origin,
        moving by the given increment."""
        x, y = origin
        color = self[x][y]
        flips: list[tuple[int, int]] = [origin]

        for x, y in OthelloBoard._increment_move(origin, direction, self.n):
            if self[x][y] == 0:
                if flips:
                    # print("Found", x,y)
                    return (x, y)
                else:
                    return None
            elif self[x][y] == color:
                return None
            elif self[x][y] == -color:
                # print("Flip",x,y)
                flips.append((x, y))

    def _get_flips(
        self, origin: tuple[int, int], direction: tuple[int, int], color: int
    ):
        """Gets the list of flips for a vertex and direction to use with the
        execute_move function"""
        # initialize variables
        flips: list[tuple[int, int]] = [origin]

        for x, y in OthelloBoard._increment_move(origin, direction, self.n):
            # print(x,y)
            if self[x][y] == 0:
                return list[tuple[int, int]]()
            if self[x][y] == -color:
                flips.append((x, y))
            elif self[x][y] == color and len(flips) > 0:
                # print(flips)
                return flips

        return list[tuple[int, int]]()

    @staticmethod
    def _increment_move(move: tuple[int, int], direction: tuple[int, int], n: int):
        # print(move)
        """Generator expression for incrementing moves"""

        def move_one_step(
            move: tuple[int, int], direction: tuple[int, int]
        ) -> tuple[int, int]:
            # returning (move[0]+direction[0], move[1]+direction[1])
            return tuple[int, int](map(sum, zip(move, direction)))

        move = move_one_step(move, direction)
        while all(map(lambda x: 0 <= x < n, move)):
            # while 0<=move[0] and move[0]<n and 0<=move[1] and move[1]<n:
            yield move
            move = move_one_step(move, direction)
