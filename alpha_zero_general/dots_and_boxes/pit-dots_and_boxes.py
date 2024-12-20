import os

import numpy as np

from alpha_zero_general import MctsArgs
from alpha_zero_general.arena import Arena
from alpha_zero_general.dots_and_boxes.dots_and_boxes_game import (
    DotsAndBoxesBoardTensor,
    DotsAndBoxesGame,
)
from alpha_zero_general.dots_and_boxes.dots_and_boxes_players import (
    GreedyRandomPlayer,
    HumanDotsAndBoxesPlayer,
    RandomPlayer,
)
from alpha_zero_general.dots_and_boxes.keras.n_net import DotsAndBoxesNNInterface
from alpha_zero_general.mcts import MCTS

NUM_MCTS_SIMS = 50

g = DotsAndBoxesGame(n=3)

hp1 = HumanDotsAndBoxesPlayer(g).play
hp2 = HumanDotsAndBoxesPlayer(g).play

rp1 = RandomPlayer(g).play
rp2 = RandomPlayer(g).play

grp1 = GreedyRandomPlayer(g).play
grp2 = GreedyRandomPlayer(g).play

n1 = DotsAndBoxesNNInterface(g)
n1.load_checkpoint(
    os.path.join("../", "pretrained_models", "dotsandboxes", "keras", "3x3"),
    "best.pth.tar",
)
args1 = MctsArgs(num_mcts_sims=NUM_MCTS_SIMS, c_puct=1.0)
mcts1 = MCTS(g, n1, args1)


def n1p(x: DotsAndBoxesBoardTensor) -> int:
    return int(np.argmax(mcts1.get_action_probabilities(x, temperature=0)))


n2 = DotsAndBoxesNNInterface(g)
n2.load_checkpoint(
    os.path.join("../", "pretrained_models", "dotsandboxes", "keras", "3x3"),
    "best.pth.tar",
)
args2 = MctsArgs(num_mcts_sims=NUM_MCTS_SIMS, c_puct=1.0)
mcts2 = MCTS(g, n2, args2)


def n2p(x: DotsAndBoxesBoardTensor) -> int:
    return int(np.argmax(mcts2.get_action_probabilities(x, temperature=0)))


# Play AlphaZero versus Human
p1 = n1p
p2 = hp1
arena = Arena(p1, p2, g, DotsAndBoxesGame.display)
oneWon, twoWon, draws = arena.play_games(2, verbose=True)
print("oneWon: {}, twoWon: {}, draws: {}".format(oneWon, twoWon, draws))

# # Play Greedy vs Greedy
# p1 = grp1
# p2 = grp2
# arena = Arena(p1, p2, g, display=DotsAndBoxesGame.display)
# oneWon, twoWon, draws = arena.play_games(100, verbose=False)
# print("oneWon: {}, twoWon: {}, draws: {}".format(oneWon, twoWon, draws))

# # Play AlphaZero vs Greedy
# p1 = n1p
# p2 = grp2
# arena = Arena(p1, p2, g, display=DotsAndBoxesGame.display)
# oneWon, twoWon, draws = arena.play_games(2, verbose=False)
# print("oneWon: {}, twoWon: {}, draws: {}".format(oneWon, twoWon, draws))
