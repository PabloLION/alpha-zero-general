import numpy as np

from alpha_zero_general import GenericBoardTensor
from alpha_zero_general.game import GenericGame


class RandomPlayer:
    def __init__(self, game: GenericGame):
        self.game = game

    def play(self, board: GenericBoardTensor) -> int:
        a = np.random.randint(self.game.get_action_size())
        valids = self.game.get_valid_moves(board, 1)
        while valids[a] != 1:
            a = np.random.randint(self.game.get_action_size())
        return a


class HumanGobangPlayer:
    def __init__(self, game: GenericGame):
        self.game = game

    def play(self, board: GenericBoardTensor) -> int:
        # display(board)
        valid = self.game.get_valid_moves(board, 1)
        for i in range(len(valid)):
            if valid[i]:
                print(int(i / self.game.n), int(i % self.game.n))
        while True:
            a = input()

            x, y = [int(x) for x in a.split(" ")]
            a = self.game.n * x + y if x != -1 else self.game.n**2
            if valid[a]:
                break
            else:
                print("Invalid")

        return a


class GreedyGobangPlayer:
    def __init__(self, game: GenericGame):
        self.game = game

    def play(self, board: GenericBoardTensor) -> int:
        valids = self.game.get_valid_moves(board, 1)
        candidates = []
        for a in range(self.game.get_action_size()):
            if valids[a] == 0:
                continue
            nextBoard, _ = self.game.get_next_state(board, 1, a)
            score = self.game.get_score(nextBoard, 1)
            candidates += [(-score, a)]
        candidates.sort()
        return candidates[0][1]
