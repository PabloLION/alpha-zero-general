# noinspection PyUnresolvedReferences
import os

import numpy as np
import tensorflow as tf

# noinspection PyUnresolvedReferences
# noinspection PyUnresolvedReferences
from TFPluginAPI import TFPluginAPI  # this is the API for UE4 plugin, not on PyPI

from alpha_zero_general import MctsArgs
from alpha_zero_general.mcts import MCTS
from alpha_zero_general.rts.keras.n_net import NNetWrapper as NNet
from alpha_zero_general.rts.rts_game import RTSGame
from alpha_zero_general.rts.src.config import ACTS_REV, NUM_ACTS
from alpha_zero_general.rts.src.encoders import OneHotEncoder
from alpha_zero_general.utils import DotDict

"""
rts_ue4.py

This classes intended use is connecting to ue4 TensorFlow plugin as client and execute predict on given board.
Connect to UE4 using https://github.com/getnamo/tensorflow-ue4
See this release https://github.com/getnamo/tensorflow-ue4/releases/tag/0.8.0
More info in readme.md
"""


# noinspection PyPep8Naming
class TD2020LearnAPI(TFPluginAPI):
    def __init__(self):
        self.owning_player = None
        self.initial_board_config = None
        self.setup = False
        self.g = None
        self.graph_var = None
        self.session_var = None
        self.mcts = None

    def on_setup(self):
        """
        Sets up nnet configs and mcts. It loads model in ram. Session variable is saved, so it can be then used async in 'onJsonInput'
        """
        graph = tf.Graph()
        with graph.as_default():
            session = tf.Session()
            with session.as_default():
                current_directory = os.path.join(os.path.dirname(__file__), "temp/")
                self.g = RTSGame()
                n1 = NNet(self.g, OneHotEncoder())
                n1.load_checkpoint(current_directory, "best.pth.tar")
                args = MctsArgs()(num_mcts_sims=2, c_puct=1.0)
                self.mcts = MCTS(self.g, n1, args)

                self.graph_var = graph
                self.session_var = session

                self.setup = True

    def on_json_input(self, json_input):
        """
        Request for action for specific game state of specific player.
        Json input is recieved from UE4, providing game state in ue4. This game state must reflect same configuration as Python one.
        Keep in mind coordinate system orientation
        :param json_input: initial board config and player, requesting action
        :return: recommended action using our nnet
        """
        if not self.setup:
            return
        encoded_actors = json_input["data"]
        initial_board_config = []
        for encoded_actor in encoded_actors:
            initial_board_config.append(
                DotDict(
                    {
                        "x": encoded_actor["x"],
                        "y": encoded_actor["y"],
                        "player": encoded_actor["player"],
                        "a_type": encoded_actor["actorType"],
                        "health": encoded_actor["health"],
                        "carry": encoded_actor["carry"],
                        "gold": encoded_actor["money"],
                        "timeout": encoded_actor["remaining"],
                    }
                )
            )

        self.initial_board_config = initial_board_config
        self.owning_player = json_input["player"]
        ######
        with self.graph_var.as_default():
            with self.session_var.as_default():
                self.g.setInitBoard(self.initial_board_config)
                b = self.g.get_init_board()

                def n1p(board):
                    return np.argmax(
                        self.mcts.get_action_probabilities(board, temperature=0)
                    )

                canonical_board = self.g.get_canonical_form(b, self.owning_player)

                recommended_act = n1p(canonical_board)
                y, x, action_index = np.unravel_index(
                    recommended_act, [b.shape[0], b.shape[0], NUM_ACTS]
                )

                # gc.collect()
                act = {"x": str(x), "y": str(y), "action": ACTS_REV[action_index]}
                print("Printing recommended action >>>>>>>>>>>>>>>>>>>>>>>>" + str(act))
        return act

    def on_begin_training(self):
        pass

    def run(self, args):
        pass

    # noinspection PyUnusedLocal
    def close(self, args):
        """
        Just clear everything, so it's not memory leaking
        :param args: /
        """
        print("Closing Get Action")
        if self.session_var:
            self.session_var.close()
        self.owning_player = None
        self.initial_board_config = None
        self.setup = False
        self.g = None
        self.graph_var = None
        self.session_var = None
        self.mcts = None


# required function to get our api
# noinspection PyPep8Naming
def get_api():
    return TD2020LearnAPI.getInstance()
