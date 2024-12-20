import numpy as np

from alpha_zero_general import GenericBoardTensor
from alpha_zero_general.tafl.digits import int2base


class RandomTaflPlayer:
    def __init__(self, game):
        self.game = game

    def play(self, board: GenericBoardTensor):
        a = np.random.randint(self.game.get_action_size())
        valids = self.game.get_valid_moves(board, board.getPlayerToMove())
        while valids[a] != 1:
            a = np.random.randint(self.game.get_action_size())
        return a


class HumanTaflPlayer:
    def __init__(self, game):
        self.game = game

    def play(self, board: GenericBoardTensor):
        # display(board)
        valid = self.game.get_valid_moves(board, board.getPlayerToMove())
        m = []
        for i in range(len(valid)):
            if valid[i]:
                m.extend([int2base(i, self.game.n, 4)])
        print(m)
        while True:
            a = input()

            x1, y1, x2, y2 = [int(x) for x in a.strip().split(" ")]
            a = x1 + y1 * self.game.n + x2 * self.game.n**2 + y2 * self.game.n**3
            if valid[a]:
                break
            else:
                print("Invalid")

        return a


class GreedyTaflPlayer:
    def __init__(self, game):
        self.game = game

    def play(self, board: GenericBoardTensor):
        valids = self.game.get_valid_moves(board, board.getPlayerToMove())
        candidates = []
        for a in range(self.game.get_action_size()):
            if valids[a] == 0:
                continue
            next_board, _ = self.game.get_next_state(board, board.getPlayerToMove(), a)
            score = self.game.getScore(next_board, board.getPlayerToMove())
            candidates += [(-score, a)]
        candidates.sort()
        return candidates[0][1]
