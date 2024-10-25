import numpy as np
from alpha_zero_general import GenericBoardTensor
from alpha_zero_general.santorini.santorini_game import SantoriniGame


class RandomPlayer:
    def __init__(self, game: SantoriniGame):
        self.game = game

    def play(self, board: GenericBoardTensor) -> int:
        a = np.random.randint(self.game.get_action_size())
        valids = self.game.get_valid_moves(board, 1)
        while valids[a] != 1:
            a = np.random.randint(self.game.get_action_size())
        return a


class HumanSantoriniPlayer:
    def __init__(self, game: SantoriniGame):
        self.game = game

    def play(self, board: GenericBoardTensor) -> int:
        valids, all_moves, all_moves_binary = self.game.get_valid_moves_human(board, 1)

        for i in range(len(all_moves)):
            if all_moves_binary[i]:
                print(
                    "|{}: {}, {}, {}|".format(
                        i, all_moves[i][0], all_moves[i][1], all_moves[i][2]
                    )
                )
        valid_move = False
        while not valid_move:
            input_move = int(input("\nPlease enter a move number: "))
            if all_moves_binary[input_move]:
                valid_move = True
            else:
                print("Sorry, that move is not valid. Please enter another.")
        return input_move


class GreedySantoriniPlayer:
    def __init__(self, game: SantoriniGame):
        self.game = game

    def play(self, board: GenericBoardTensor) -> int:
        valids = self.game.get_valid_moves(board, 1)
        candidates = []
        for a in range(self.game.get_action_size()):
            if valids[a] == 0:
                continue
            next_board, _ = self.game.get_next_state(board, 1, a)
            score = self.game.get_score(next_board, 1)
            candidates += [(-score, a)]
        candidates.sort()

        return candidates[0][1]
