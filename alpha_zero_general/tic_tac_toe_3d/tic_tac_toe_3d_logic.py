from typing import no_type_check

from numpy import argwhere, zeros

from alpha_zero_general.py313_functions import deprecated
from alpha_zero_general.tic_tac_toe_3d import (
    TicTacToe3DBoardDataType,
    TicTacToe3DBoardTensor,
)

"""
Board class for the game of TicTacToe.
Default board size is 3x3.
Board data:
  1=white(O), -1=black(X), 0=empty
  first dim is column , 2nd is row:
     pieces[0][0] is the top left square,
     pieces[2][0] is the bottom left square,
Squares are stored and manipulated as (x,y) tuples.

Author: Evgeny Tyurin, github.com/evg-tyurin
Date: Jan 5, 2018.

Based on the board for the game of Othello by Eric P. Nichols.

"""


# from bkcharts.attributes import color
class Board:
    pieces: TicTacToe3DBoardTensor

    def __init__(self, n: int = 4):
        "Set up initial board configuration."

        self.n = n
        # Create the empty board array.
        self.pieces = zeros((n, n, n), dtype=TicTacToe3DBoardDataType)

    # add [][] indexer syntax to the Board
    def __getitem__(self, index: tuple[int, int, int]):
        """
        Convert natural index to 0-based index and return the piece on the board.
        """
        return self.pieces[index[0] - 1][index[1] - 1][index[2] - 1]

    @no_type_check
    @deprecated
    def __old__getitem__(self, index: tuple[int, int, int]):
        index1 = [None, None, None]
        for i in range(3):
            index1[i] = str(index[i])
        for i in range(len(index1)):
            x = index1[i]
            index1[i] = str(int(x) - 1)
        return self.pieces[list(map(int, index1))]

    def get_legal_moves(self, color: int):
        """Returns all the legal moves for the given color.
        (1 for white, -1 for black)
        @param color not used and came from previous version.
        """
        return argwhere(self.pieces == 0)

    @no_type_check
    @deprecated
    def old_get_legal_moves(self, color):
        """Returns all the legal moves for the given color.
        (1 for white, -1 for black)
        @param color not used and came from previous version.
        """
        moves = set()  # stores the legal moves.

        # Get all the empty squares (color==0)
        for z in range(self.n):
            for y in range(self.n):
                for x in range(self.n):
                    if self.pieces[z][x][y] == 0:
                        new_move = (z, x, y)
                        moves.add(new_move)
        return list(moves)

    def has_legal_moves(self):
        for z in range(self.n):
            for y in range(self.n):
                for x in range(self.n):
                    if self.pieces[z][x][y] == 0:
                        return True
        return False

    def is_win(self, color: int) -> bool:
        """Check whether the given player has collected a triplet in any direction;
        @param color (1=white,-1=black)
        """
        win = self.n
        # check z-dimension
        count = 0
        for z in range(self.n):
            count = 0
            for y in range(self.n):
                count = 0
                for x in range(self.n):
                    if self.pieces[z, x, y] == color:
                        count += 1
                if count == win:
                    return True

        count = 0
        for z in range(self.n):
            count = 0
            for x in range(self.n):
                count = 0
                for y in range(self.n):
                    if self.pieces[z, x, y] == color:
                        count += 1
                if count == win:
                    return True

        # check x dimension
        count = 0
        for x in range(self.n):
            count = 0
            for z in range(self.n):
                count = 0
                for y in range(self.n):
                    if self.pieces[z, x, y] == color:
                        count += 1
                if count == win:
                    return True

        count = 0
        for x in range(self.n):
            count = 0
            for y in range(self.n):
                count = 0
                for z in range(self.n):
                    if self.pieces[z, x, y] == color:
                        count += 1
                if count == win:
                    return True

        # check y dimension
        count = 0
        for y in range(self.n):
            count = 0
            for x in range(self.n):
                count = 0
                for z in range(self.n):
                    if self.pieces[z, x, y] == color:
                        count += 1
                if count == win:
                    return True

        count = 0
        for y in range(self.n):
            count = 0
            for z in range(self.n):
                count = 0
                for x in range(self.n):
                    if self.pieces[z, x, y] == color:
                        count += 1
                if count == win:
                    return True

        # check flat diagonals
        # check z dimension
        count = 0
        for z in range(self.n):
            count = 0
            for d in range(self.n):
                if self.pieces[z, d, d] == color:
                    count += 1
            if count == win:
                return True

        count = 0
        for z in range(self.n):
            count = 0
            for d in range(self.n):
                if self.pieces[z, d, self.n - d - 1] == color:
                    count += 1
            if count == win:
                return True

        # check x dimension
        count = 0
        for x in range(self.n):
            count = 0
            for d in range(self.n):
                if self.pieces[d, x, d] == color:
                    count += 1
            if count == win:
                return True

        count = 0
        for x in range(self.n):
            count = 0
            for d in range(self.n):
                if self.pieces[d, x, self.n - d - 1] == color:
                    count += 1
            if count == win:
                return True

        # check y dimension
        count = 0
        for y in range(self.n):
            count = 0
            for d in range(self.n):
                if self.pieces[d, d, y] == color:
                    count += 1
            if count == win:
                return True

        count = 0
        for y in range(self.n):
            count = 0
            for d in range(self.n):
                if self.pieces[self.n - d - 1, d, y] == color:
                    count += 1
            if count == win:
                return True

        # check 4 true diagonals
        count = 0
        if self.pieces[0, 0, 0] == color:
            count += 1
            if self.pieces[1, 1, 1] == color:
                count += 1
                if self.pieces[2, 2, 2] == color:
                    count += 1
                    if count == win:
                        return True

        count = 0
        if self.pieces[2, 0, 0] == color:
            count += 1
            if self.pieces[1, 1, 1] == color:
                count += 1
                if self.pieces[0, 2, 2] == color:
                    count += 1
                    if count == win:
                        return True

        count = 0
        if self.pieces[2, 2, 0] == color:
            count += 1
            if self.pieces[1, 1, 1] == color:
                count += 1
                if self.pieces[0, 0, 2] == color:
                    count += 1
                    if count == win:
                        return True

        count = 0
        if self.pieces[0, 2, 0] == color:
            count += 1
            if self.pieces[1, 1, 1] == color:
                count += 1
                if self.pieces[2, 0, 2] == color:
                    count += 1
                    if count == win:
                        return True

        # return false if no 3 is reached
        return False

    def execute_move(self, move: tuple[int, int, int], color: int) -> None:
        """Perform the given move on the board;
        color gives the color pf the piece to play (1=white,-1=black)
        """

        (z, x, y) = move

        # Add the piece to the empty square.
        assert self.pieces[z][x][y] == 0
        self.pieces[z][x][y] = color
