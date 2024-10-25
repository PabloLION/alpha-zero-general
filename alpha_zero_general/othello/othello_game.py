from __future__ import print_function

from numpy import append, array, array2string, fliplr, reshape, rot90

from alpha_zero_general.game import GenericGame
from alpha_zero_general.othello import (
    OthelloBoardTensor,
    OthelloBooleanBoardTensor,
    OthelloPolicyTensor,
)
from alpha_zero_general.othello.othello_logic import OthelloBoard


class OthelloGame(
    GenericGame[OthelloBoardTensor, OthelloBooleanBoardTensor, OthelloPolicyTensor]
):
    # #TODO/REF: make this a enum or a typed dict
    square_content = {-1: "X", +0: "-", +1: "O"}

    @staticmethod
    def get_square_piece(piece: int) -> str:
        return OthelloGame.square_content[piece]

    def __init__(self, n: int):
        self.n = n

    def get_init_board(self):
        # return initial board (numpy board)
        b = OthelloBoard(self.n)
        return array(b.pieces)

    def get_board_size(self):
        # (a,b) tuple
        return (self.n, self.n)

    def get_action_size(self):
        # return number of actions
        return self.n * self.n + 1

    def get_next_state(
        self, board: OthelloBoardTensor, player: int, action: int
    ) -> tuple[OthelloBoardTensor, int]:

        # if player takes action on board, return next (board,player)
        # action must be a valid move
        if action == self.n * self.n:
            return (board, -player)
        b = OthelloBoard(self.n)
        b.pieces = [list(row) for row in board]
        # b.pieces = np.copy(board)  # #TODO/REF: this is not only typing...
        move = (int(action / self.n), action % self.n)
        b.execute_move(move, player)
        return (array(b.pieces), -player)

    def get_valid_moves(self, board: OthelloBoardTensor, player: int):
        # return a fixed size binary vector
        valid_moves = [0] * self.get_action_size()
        b = OthelloBoard(self.n)
        b.pieces = [list(row) for row in board]
        legal_moves = b.get_legal_moves(player)
        if len(legal_moves) == 0:
            valid_moves[-1] = 1
            return array(valid_moves)
        for x, y in legal_moves:
            valid_moves[self.n * x + y] = 1
        return array(valid_moves)

    def get_game_ended(self, board: OthelloBoardTensor, player: int) -> float:
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player = 1
        b = OthelloBoard(self.n)
        b.pieces = [list(row) for row in board]
        if b.has_legal_moves(player):
            return 0
        if b.has_legal_moves(-player):
            return 0
        if b.count_diff(player) > 0:
            return 1
        return -1

    def get_canonical_form(
        self, board: OthelloBoardTensor, player: int
    ) -> OthelloBoardTensor:
        # return state if player==1, else return -state if player==-1
        return player * board

    def get_symmetries(
        self, board: OthelloBoardTensor, pi: OthelloPolicyTensor
    ) -> list[tuple[OthelloBoardTensor, OthelloPolicyTensor]]:
        # mirror, rotational
        assert len(pi) == self.n**2 + 1  # 1 for pass
        pi_board = reshape(pi[:-1], (self.n, self.n))
        ans = list[tuple[OthelloBoardTensor, OthelloPolicyTensor]]()

        for i in range(1, 5):
            rot_board = rot90(board, i)
            rot_pi_b = rot90(pi_board, i)

            ans.append((rot_board, append(rot_pi_b.ravel(), pi[-1])))
            ans.append((fliplr(rot_board), append(fliplr(rot_pi_b).ravel(), [pi[-1]])))

        return ans

    def get_board_str(self, board: OthelloBoardTensor) -> str:
        return array2string(board)

    def get_board_hash(self, board: OthelloBoardTensor) -> int:
        return hash(board.tobytes())

    def string_representation_readable(self, board: OthelloBoardTensor) -> str:
        board_s = "".join(
            self.square_content[square] for row in board for square in row
        )
        return board_s

    def get_score(self, board: OthelloBoardTensor, player: int) -> float:
        b = OthelloBoard(self.n)
        b.pieces = [list(row) for row in board]
        return b.count_diff(player)

    @staticmethod
    def display(board: OthelloBoardTensor) -> None:
        n = board.shape[0]
        print("   ", end="")
        for y in range(n):
            print(y, end=" ")
        print("")
        print("-----------------------")
        for y in range(n):
            print(y, "|", end="")  # print the row #
            for x in range(n):
                piece = board[y][x]  # get the piece to print
                print(OthelloGame.square_content[piece], end=" ")
            print("|")

        print("-----------------------")
