from __future__ import print_function

import numpy as np
from numpy import fliplr, reshape, rot90

from alpha_zero_general.game import GenericGame
from alpha_zero_general.tic_tac_toe import (
    TicTacToeBoardTensor,
    TicTacToeBooleanBoardTensor,
    TicTacToePolicyTensor,
)
from alpha_zero_general.tic_tac_toe.tic_tac_toe_logic import TicTacToeBoard

"""
Game class implementation for the game of TicTacToe.
Based on the OthelloGame then get_game_ended() was adapted to new rules.

Author: Evgeny Tyurin, github.com/evg-tyurin
Date: Jan 5, 2018.

Based on the OthelloGame by Surag Nair.
"""


class TicTacToeGame(
    GenericGame[
        TicTacToeBoardTensor, TicTacToeBooleanBoardTensor, TicTacToePolicyTensor
    ]
):
    def __init__(self, n: int = 3) -> None:
        self.n = n

    def get_init_board(self):
        # return initial board (numpy board)
        b = TicTacToeBoard(self.n)
        return np.array(b.pieces)

    def get_board_size(self):
        # (a,b) tuple
        return (self.n, self.n)

    def get_action_size(self):
        # return number of actions
        return self.n * self.n + 1

    def get_next_state(
        self, board: TicTacToeBoardTensor, player: int, action: int
    ) -> tuple[TicTacToeBoardTensor, int]:
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        if action == self.n * self.n:
            return (board, -player)
        b = TicTacToeBoard(self.n)
        b.pieces = np.copy(board)
        move = (int(action / self.n), action % self.n)
        b.execute_move(move, player)
        return (b.pieces, -player)

    def get_valid_moves(self, board: TicTacToeBoardTensor, player: int):
        # return a fixed size binary vector
        valids = [0] * self.get_action_size()
        b = TicTacToeBoard(self.n)
        b.pieces = np.copy(board)
        legalMoves = b.get_legal_moves(player)
        if len(legalMoves) == 0:
            valids[-1] = 1
            return np.array(valids)
        for x, y in legalMoves:
            valids[self.n * x + y] = 1
        return np.array(valids)

    def get_game_ended(self, board: TicTacToeBoardTensor, player: int):
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player = 1
        b = TicTacToeBoard(self.n)
        b.pieces = np.copy(board)

        if b.is_win(player):
            return 1
        if b.is_win(-player):
            return -1
        if b.has_legal_moves():
            return 0
        # draw has a very little value
        return 1e-4

    def get_canonical_form(self, board: TicTacToeBoardTensor, player: int):
        # return state if player==1, else return -state if player==-1
        return player * board

    def get_symmetries(self, board: TicTacToeBoardTensor, pi: TicTacToePolicyTensor):
        # mirror and rotational
        assert len(pi) == self.n**2 + 1  # #TODO/FIX: there's no pass in tic tac toe
        pi_board = reshape(pi[:-1], (self.n, self.n))
        ans = list[tuple[TicTacToeBoardTensor, TicTacToePolicyTensor]]()

        for i in range(1, 5):
            rot_board = rot90(board, i)
            rot_pi_b = rot90(pi_board, i)

            ans.append(
                (
                    rot_board,
                    rot_pi_b.ravel(),
                )
            )
            ans.append((fliplr(rot_board), fliplr(rot_pi_b).ravel()))

        return ans

    def get_board_str(self, board: TicTacToeBoardTensor):
        # 8x8 numpy array (canonical board)
        return np.array2string(board)

    def get_board_hash(self, board: TicTacToeBoardTensor) -> int:
        return hash(board.tobytes())

    @staticmethod
    def display(board: TicTacToeBoardTensor):
        n = board.shape[0]

        print("   ", end="")
        for y in range(n):
            print(y, "", end="")
        print("")
        print("  ", end="")
        for _ in range(n):
            print("-", end="-")
        print("--")
        for y in range(n):
            print(y, "|", end="")  # print the row #
            for x in range(n):
                piece = board[y][x]  # get the piece to print
                if piece == -1:
                    print("X ", end="")
                elif piece == 1:
                    print("O ", end="")
                else:
                    if x == n:
                        print("-", end="")
                    else:
                        print("- ", end="")
            print("|")

        print("  ", end="")
        for _ in range(n):
            print("-", end="-")
        print("--")
