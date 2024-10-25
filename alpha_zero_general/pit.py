"""
Using Othello as an example, this script pits two agents against each other.
"""

import numpy as np

from alpha_zero_general import MctsArgs
from alpha_zero_general.arena import Arena
from alpha_zero_general.mcts import MCTS
from alpha_zero_general.othello.othello_game import OthelloGame
from alpha_zero_general.othello.othello_players import (
    GreedyOthelloPlayer,
    HumanOthelloPlayer,
    RandomPlayer,
)
from alpha_zero_general.othello.pytorch.n_net import OthelloTorchNNInterface as NNet

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

MINI_OTHELLO = False  # Play in 6x6 instead of the normal 8x8.
HUMAN_VS_CPU = True

if MINI_OTHELLO:
    g = OthelloGame(6)
else:
    g = OthelloGame(8)

# all players
rp = RandomPlayer(g).play
gp = GreedyOthelloPlayer(g).play
hp = HumanOthelloPlayer(g).play


# nnet players
n1 = NNet(g)
if MINI_OTHELLO:
    n1.load_checkpoint("./pretrained_models/othello/pytorch/", "6x100x25_best.pth.tar")
else:
    n1.load_checkpoint(
        "./pretrained_models/othello/pytorch/", "8x8_100checkpoints_best.pth.tar"
    )
args1 = MctsArgs(num_mcts_sims=50, c_puct=1.0)
mcts1 = MCTS(g, n1, args1)


def n1p(x: np.ndarray) -> int:
    return np.argmax(mcts1.get_action_probabilities(x, temperature=0))


if HUMAN_VS_CPU:
    player2 = hp
else:
    n2 = NNet(g)
    n2.load_checkpoint(
        "./pretrained_models/othello/pytorch/", "8x8_100checkpoints_best.pth.tar"
    )
    args2 = MctsArgs(num_mcts_sims=50, c_puct=1.0)
    mcts2 = MCTS(g, n2, args2)
    n2p = lambda x: np.argmax(mcts2.get_action_probabilities(x, temperature=0))

    player2 = n2p  # Player 2 is neural network if it's cpu vs cpu.

arena = Arena(n1p, player2, g, display=OthelloGame.display)

print(arena.play_games(2, verbose=True))
