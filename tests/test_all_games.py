""""

    This is a Regression Test Suite to automatically test all combinations of games and ML frameworks. Each test
    plays two quick games using an untrained neural network (randomly initialized) against a random player.

    In order for the entire test suite to run successfully, all the required libraries must be installed.  They are:
    Pytorch, Keras.

     [ Games ]      Pytorch      Keras
      -----------   -------      -----
    - Othello        [Yes]       [Yes]
    - TicTacToe                  [Yes]
    - TicTacToe3D                [Yes]
    - Connect4                   [Yes]
    - Gobang                     [Yes]
    - Tafl           [Yes]       [Yes]
    - Rts                        [Yes]
    - DotsAndBoxes               [Yes]
"""

import numpy as np

from alpha_zero_general.arena import Arena
from alpha_zero_general.connect4.connect4_game import Connect4Game
from alpha_zero_general.connect4.keras.n_net import NNetWrapper as Connect4KerasNNet
from alpha_zero_general.dotsandboxes.dots_and_boxes_game import DotsAndBoxesGame
from alpha_zero_general.dotsandboxes.keras.n_net import (
    NNetWrapper as DotsAndBoxesKerasNNet,
)
from alpha_zero_general.game import Game
from alpha_zero_general.gobang.gobang_game import GobangGame
from alpha_zero_general.gobang.keras.n_net import NNetWrapper as GobangKerasNNet
from alpha_zero_general.mcts import MCTS
from alpha_zero_general.neural_net import NeuralNet
from alpha_zero_general.othello.keras.n_net import NNetWrapper as OthelloKerasNNet
from alpha_zero_general.othello.othello_game import OthelloGame
from alpha_zero_general.othello.othello_players import RandomPlayer
from alpha_zero_general.othello.pytorch.n_net import NNetWrapper as OthelloPytorchNNet
from alpha_zero_general.rts.keras.n_net import NNetWrapper as RTSKerasNNet
from alpha_zero_general.rts.rts_game import RTSGame
from alpha_zero_general.tafl.keras.n_net import NNetWrapper as TaflKerasNNet
from alpha_zero_general.tafl.pytorch.n_net import NNetWrapper as TaflPytorchNNet
from alpha_zero_general.tafl.tafl_game import TaflGame
from alpha_zero_general.tictactoe.keras.n_net import NNetWrapper as TicTacToeKerasNNet
from alpha_zero_general.tictactoe.tic_tac_toe_game import TicTacToeGame
from alpha_zero_general.tictactoe_3d.keras.n_net import (
    NNetWrapper as TicTacToe3DKerasNNet,
)
from alpha_zero_general.tictactoe_3d.tic_tac_toe_game import (
    TicTacToeGame as TicTacToe3DGame,
)
from alpha_zero_general.utils import dotdict


def execute_game_test(game: Game, neural_net: type[NeuralNet]):
    random_play = RandomPlayer(game).play

    args = dotdict({"numMCTSSims": 25, "cpuct": 1.0})
    mcts = MCTS(game, neural_net(game), args)
    n1p = lambda x: np.argmax(mcts.getActionProb(x, temp=0))

    arena = Arena(n1p, random_play, game)
    print(arena.playGames(2, verbose=False))


def test_othello_pytorch():
    execute_game_test(OthelloGame(6), OthelloPytorchNNet)


def test_othello_keras():
    execute_game_test(OthelloGame(6), OthelloKerasNNet)


def test_tictactoe_keras():
    execute_game_test(TicTacToeGame(), TicTacToeKerasNNet)


def test_tictactoe3d_keras():
    execute_game_test(TicTacToe3DGame(3), TicTacToe3DKerasNNet)


def test_gobang_keras():
    execute_game_test(GobangGame(), GobangKerasNNet)


def test_tafl_pytorch():
    execute_game_test(TaflGame(5), TaflPytorchNNet)


def test_tafl_keras():
    execute_game_test(TaflGame(5), TaflKerasNNet)


def test_connect4_keras():
    execute_game_test(Connect4Game(5), Connect4KerasNNet)


def test_rts_keras():
    execute_game_test(RTSGame(), RTSKerasNNet)


def test_dotsandboxes_keras():
    execute_game_test(DotsAndBoxesGame(3), DotsAndBoxesKerasNNet)
