"""
new file by copilot workspace, not working.
There's no SantoriniKerasNNet


import numpy as np
from alpha_zero_general import MctsArgs
from alpha_zero_general.arena import Arena
from alpha_zero_general.mcts import MCTS
from alpha_zero_general.santorini.santorini_game import SantoriniGame
from alpha_zero_general.santorini.santorini_players import RandomPlayer
from alpha_zero_general.santorini.keras.n_net import SantoriniKerasNNet

def test_santorini_keras():
    game = SantoriniGame()
    neural_net = SantoriniKerasNNet(game)
    random_play = RandomPlayer(game).play

    args = MctsArgs(num_mcts_sims=25, c_puct=1.0)
    mcts = MCTS(game, neural_net, args)
    n1p = lambda x: np.argmax(mcts.get_action_probabilities(x, temperature=0))

    arena = Arena(n1p, random_play, game)
    print(arena.play_games(2, verbose=False))

if __name__ == "__main__":
    test_santorini_keras()
"""
