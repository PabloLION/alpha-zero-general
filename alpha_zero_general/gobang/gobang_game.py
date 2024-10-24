from __future__ import print_function

from numpy import append, array, array2string, copy, fliplr, reshape, rot90

from alpha_zero_general.game import GenericGame
from alpha_zero_general.gobang import (
    CHAR_B,
    CHAR_W,
    GobangBoardTensor,
    GobangBooleanBoardTensor,
    GobangPolicyTensor,
)
from alpha_zero_general.gobang.gobang_logic import Board


class GobangGame(
    GenericGame[GobangBoardTensor, GobangBooleanBoardTensor, GobangPolicyTensor]
):
    def __init__(self, n: int = 15, nir: int = 5):
        self.n = n
        self.n_in_row = nir

    def get_init_board(self) -> GobangBoardTensor:
        # return initial board (numpy board)
        b = Board(self.n)
        return array(b.pieces)

    def get_board_size(self) -> tuple[int, int]:
        # (a,b) tuple
        return (self.n, self.n)

    def get_action_size(self) -> int:
        # return number of actions
        return self.n * self.n + 1

    def get_next_state(
        self, board: GobangBoardTensor, player: int, action: int
    ) -> tuple[GobangBoardTensor, int]:
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        if action == self.n * self.n:
            return (board, -player)
        b = Board(self.n)
        b.pieces = copy(board)
        move = (int(action / self.n), action % self.n)
        b.execute_move(move, player)
        return (b.pieces, -player)

    def get_valid_moves(
        self, board: GobangBoardTensor, player: int
    ) -> GobangBooleanBoardTensor:
        # return a fixed size binary vector
        # #TODO/PERF: use numpy array
        valid_moves = [0] * self.get_action_size()
        b = Board(self.n)
        b.pieces = copy(board)
        legal_moves = b.get_legal_moves(player)
        if len(legal_moves) == 0:
            valid_moves[-1] = 1
            return array(valid_moves)
        for x, y in legal_moves:
            valid_moves[self.n * x + y] = 1
        return array(valid_moves)

    def get_game_ended(self, board: GobangBoardTensor, player: int) -> float:
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player = 1
        b = Board(self.n)
        b.pieces = copy(board)
        n = self.n_in_row

        for w in range(self.n):
            for h in range(self.n):
                if (
                    w in range(self.n - n + 1)
                    and board[w][h] != 0
                    and len(set(board[i][h] for i in range(w, w + n))) == 1
                ):
                    return board[w][h]
                if (
                    h in range(self.n - n + 1)
                    and board[w][h] != 0
                    and len(set(board[w][j] for j in range(h, h + n))) == 1
                ):
                    return board[w][h]
                if (
                    w in range(self.n - n + 1)
                    and h in range(self.n - n + 1)
                    and board[w][h] != 0
                    and len(set(board[w + k][h + k] for k in range(n))) == 1
                ):
                    return board[w][h]
                if (
                    w in range(self.n - n + 1)
                    and h in range(n - 1, self.n)
                    and board[w][h] != 0
                    and len(set(board[w + l][h - l] for l in range(n))) == 1
                ):
                    return board[w][h]
        if b.has_legal_moves():
            return 0
        return 1e-4

    def get_canonical_form(
        self, board: GobangBoardTensor, player: int
    ) -> GobangBoardTensor:
        # return state if player==1, else return -state if player==-1
        return player * board

    # modified
    def get_symmetries(
        self, board: GobangBoardTensor, pi: GobangPolicyTensor
    ) -> list[tuple[GobangBoardTensor, GobangPolicyTensor]]:
        # mirror and rotation
        assert len(pi) == self.n**2 + 1  # 1 for pass
        pi_board = reshape(pi[:-1], (self.n, self.n))
        ans = list[tuple[GobangBoardTensor, GobangPolicyTensor]]()

        for i in range(1, 5):
            rot_board = rot90(board, i)
            rot_pi_b = rot90(pi_board, i)

            ans.append((rot_board, append(rot_pi_b.ravel(), pi[-1])))
            ans.append((fliplr(rot_board), append(fliplr(rot_pi_b).ravel(), [pi[-1]])))

        return ans

    def get_board_str(self, board: GobangBoardTensor) -> str:
        # 8x8 numpy array (canonical board)
        return array2string(board)

    def get_board_hash(self, board: GobangBoardTensor) -> int:
        return hash(board.tobytes())

    @staticmethod
    def display(board: GobangBoardTensor) -> None:
        n = board.shape[0]

        for y in range(n):
            print(y, "|", end="")
        print("")
        print(" -----------------------")
        for y in range(n):
            print(y, "|", end="")  # print the row #
            for x in range(n):
                piece = board[y][x]  # get the piece to print
                if piece == -1:
                    print(CHAR_B, end=" ")
                elif piece == 1:
                    print(CHAR_W, end=" ")
                else:
                    if x == n:
                        print("-", end="")
                    else:
                        print("- ", end="")
            print("|")
        print("   -----------------------")
