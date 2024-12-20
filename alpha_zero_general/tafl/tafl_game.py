"""
REFACTOR
Logic here is confusing, we need to change it in the future.
"""

from __future__ import print_function

import sys

sys.path.append("..")
import numpy as np

from alpha_zero_general.game import GenericGame
from alpha_zero_general.tafl import (
    TaflBoardTensor,
    TaflBooleanBoardTensor,
    TaflPolicyTensor,
)
from alpha_zero_general.tafl.digits import int2base
from alpha_zero_general.tafl.game_variants import (
    AleaEvangelii,
    ArdRi,
    Brandubh,
    Hnefatafl,
    Tablut,
    Tawlbwrdd,
)
from alpha_zero_general.tafl.tafl_logic import TaflBoard


class TaflGame(GenericGame[TaflBoardTensor, TaflBooleanBoardTensor, TaflPolicyTensor]):
    name: str
    n: int  # board size
    board: TaflBoard

    def __init__(self, name: str = "Brandubh") -> None:
        self.name = name
        self.get_init_board()

    def get_init_board(self) -> TaflBoard:
        # #TODO: add enum for game variants
        if self.name == "Brandubh":
            board = TaflBoard(Brandubh())
        elif self.name == "ArdRi":
            board = TaflBoard(ArdRi())
        elif self.name == "Tablut":
            board = TaflBoard(Tablut())
        elif self.name == "Tawlbwrdd":
            board = TaflBoard(Tawlbwrdd())
        elif self.name == "Hnefatafl":
            board = TaflBoard(Hnefatafl())
        elif self.name == "AleaEvangelii":
            board = TaflBoard(AleaEvangelii())
        else:
            raise ValueError("Unknown variant")
        self.n = board.size
        return board

    def get_board_size(self) -> tuple[int, int]:
        # (a,b) tuple
        return (self.n, self.n)

    def get_action_size(self) -> int:
        # return number of actions
        return self.n**4

    def get_next_state(
        self, board: TaflBoard, player: int, action: int
    ) -> tuple[TaflBoard, int]:
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        b = board.get_copy()
        move = int2base(action, self.n, 4)
        b.execute_move(move, player)
        return (b, -player)

    def get_valid_moves(self, board: TaflBoard, player: int) -> TaflBoardTensor:
        # return a fixed size binary vector
        # Note: Ignoreing the passed in player variable since we are not inverting colors for get_canonical_form and Arena calls with constant 1.
        valid_moves = [0] * self.get_action_size()
        b = board.get_copy()
        legal_moves = b.get_legal_moves(board.get_player_to_move())
        if len(legal_moves) == 0:
            valid_moves[-1] = 1
            return np.array(valid_moves)
        for x1, y1, x2, y2 in legal_moves:
            valid_moves[x1 + y1 * self.n + x2 * self.n**2 + y2 * self.n**3] = 1
        return np.array(valid_moves)

    def get_game_ended(self, board: TaflBoard, player: int) -> int:
        # return 0 if not ended, if player 1 won, -1 if player 1 lost
        return board.done * player

    def get_canonical_form(self, board: TaflBoard, player: int) -> TaflBoard:
        b = board.get_copy()
        # rules and objectives are different for the different players, so inverting board results in an invalid state.
        return b

    def get_symmetries(
        self, board: TaflBoard, pi: list[float]
    ) -> list[tuple[TaflBoard, list[float]]]:
        raise NotImplementedError("Symmetries not implemented")
        # return [(board, pi)]
        # mirror, rotational
        # assert(len(pi) == self.n**4)
        # pi_board = np.reshape(pi[:-1], (self.n, self.n))
        # l = []

        # for i in range(1, 5):
        #    for j in [True, False]:
        #        newB = np.rot90(board, i)
        #        newPi = np.rot90(pi_board, i)
        #        if j:
        #            newB = np.fliplr(newB)
        #            newPi = np.fliplr(newPi)
        #        l += [(newB, list(newPi.ravel()) + [pi[-1]])]
        # return l

    def get_board_str(self, board: TaflBoard) -> str:
        # #TODO: check the type of board here!
        # print("->",str(board))
        return str(board)

    def get_board_hash(self, board: TaflBoard) -> int:
        # #TODO: check the type of board here!
        # return hash(board.tobytes())
        return hash(board)

    def get_score(self, board: TaflBoard, player: int) -> int:
        if board.done:
            return 1000 * board.done * player
        return board.count_diff(player)


def display(board: TaflBoard) -> None:
    render_chars = {
        "-1": "b",
        "0": " ",
        "1": "W",
        "2": "K",
        "10": "#",
        "12": "E",
        "20": "_",
        "22": "x",
    }
    print("---------------------")
    image = board.get_image()

    print("  ", " ".join(str(i) for i in range(len(image))))
    for i in range(len(image) - 1, -1, -1):
        print("{:2}".format(i), end=" ")

        row = image[i]
        for col in row:
            c = render_chars[str(col)]
            sys.stdout.write(c)
        print(" ")
    # if (board.done!=0): print("***** Done: ",board.done)
    print("---------------------")
