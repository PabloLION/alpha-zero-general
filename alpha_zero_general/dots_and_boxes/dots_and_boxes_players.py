import numpy as np

from alpha_zero_general import GenericBoardTensor
from alpha_zero_general.dots_and_boxes.dots_and_boxes_game import DotsAndBoxesGame


class RandomPlayer:
    def __init__(self, game: DotsAndBoxesGame):
        self.game = game

    def play(self, board: GenericBoardTensor) -> int:
        a = np.random.randint(self.game.get_action_size())
        valids = self.game.get_valid_moves(board, 1)
        while valids[a] != 1:
            a = np.random.randint(self.game.get_action_size())
        return a


# Will play at random, unless there's a chance to score a square
class GreedyRandomPlayer:
    def __init__(self, game: DotsAndBoxesGame):
        self.game = game

    def play(self, board: GenericBoardTensor) -> int:
        valids = self.game.get_valid_moves(board, 1)
        previous_score = board[0, -1]
        for action in np.nonzero(valids)[0]:
            new_board, _ = self.game.get_next_state(board, 1, action)
            new_score = new_board[0, -1]
            if new_score > previous_score:
                return action
        a = np.random.randint(self.game.get_action_size())
        while valids[a] != 1:
            a = np.random.randint(self.game.get_action_size())
        return a


class HumanDotsAndBoxesPlayer:
    def __init__(self, game: DotsAndBoxesGame):
        self.game = game

    def play(self, board: GenericBoardTensor) -> int:
        if board[2][-1] == 1:
            # We have to pass
            return self.game.get_action_size() - 1
        valids = self.game.get_valid_moves(board, 1)
        while True:
            print("Valid moves: {}".format(np.where(valids == True)[0]))
            a = int(input())
            if valids[a]:
                return a
            print("Invalid move")
