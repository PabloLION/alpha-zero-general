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

from alpha_zero_general import BoardTensor, BooleanBoard, MctsArgs, PolicyTensor
from alpha_zero_general.arena import Arena
from alpha_zero_general.connect4.connect4_game import Connect4Game
from alpha_zero_general.connect4.keras.n_net import (
    Connect4NNInterface as Connect4KerasNNet,
)
from alpha_zero_general.dots_and_boxes.dots_and_boxes_game import DotsAndBoxesGame
from alpha_zero_general.dots_and_boxes.keras.n_net import (
    DotsAndBoxesNNInterface as DotsAndBoxesKerasNNet,
)
from alpha_zero_general.game import GenericGame
from alpha_zero_general.gobang.gobang_game import GobangGame
from alpha_zero_general.gobang.keras.n_net import NNetWrapper as GobangKerasNNet
from alpha_zero_general.mcts import MCTS
from alpha_zero_general.neural_net import NeuralNetInterface
from alpha_zero_general.othello.keras.n_net import NNetWrapper as OthelloKerasNNet
from alpha_zero_general.othello.othello_game import OthelloGame
from alpha_zero_general.othello.othello_players import RandomPlayer
from alpha_zero_general.othello.pytorch.n_net import (
    OthelloTorchNNInterface as OthelloPytorchNNet,
)
from alpha_zero_general.rts.keras.n_net import NNetWrapper as RTSKerasNNet
from alpha_zero_general.rts.rts_game import RTSGame
from alpha_zero_general.tafl.keras.n_net import NNetWrapper as TaflKerasNNet
from alpha_zero_general.tafl.pytorch.n_net import NNetWrapper as TaflPytorchNNet
from alpha_zero_general.tafl.tafl_game import TaflGame
from alpha_zero_general.tic_tac_toe.keras.n_net import NNetWrapper as TicTacToeKerasNNet
from alpha_zero_general.tic_tac_toe.tic_tac_toe_game import TicTacToeGame
from alpha_zero_general.tic_tac_toe_3d.keras.n_net import (
    NNetWrapper as TicTacToe3DKerasNNet,
)
from alpha_zero_general.tic_tac_toe_3d.tic_tac_toe_3d_game import TicTacToe3DGame


def execute_game_test(
    game: GenericGame[BoardTensor, BooleanBoard, PolicyTensor],
    neural_net: type[NeuralNetInterface[BoardTensor, BooleanBoard, PolicyTensor]],
):
    # #TODO/REF: to type this, we need a generic Player class
    random_play = RandomPlayer(game).play
    # need RandomPlayer ABC, Player ABC

    args = MctsArgs(num_mcts_sims=10, c_puct=1.0)
    mcts = MCTS(game, neural_net(game), args)

    def n1policy(x: BoardTensor) -> int:
        return int(np.argmax(mcts.get_action_probabilities(x, temperature=0)))

    arena = Arena(n1policy, random_play, game)
    print(arena.play_games(2, verbose=False))


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
    execute_game_test(TaflGame("Brandubh"), TaflPytorchNNet)


def test_tafl_keras():
    execute_game_test(TaflGame("Brandubh"), TaflKerasNNet)


def test_connect4_keras():
    execute_game_test(Connect4Game(5), Connect4KerasNNet)


def test_rts_keras():
    execute_game_test(RTSGame(), RTSKerasNNet)


def test_dots_and_boxes_keras():
    execute_game_test(DotsAndBoxesGame(3), DotsAndBoxesKerasNNet)
