from typing import TYPE_CHECKING

from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv3D,
    Dense,
    Dropout,
    Flatten,
    Input,
    Reshape,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from alpha_zero_general.tic_tac_toe_3d import (
    TicTacToe3DBoardTensor,
    TicTacToe3DNNArg,
    TicTacToe3DPolicyTensor,
)
from alpha_zero_general.tic_tac_toe_3d.tic_tac_toe_3d_game import TicTacToe3DGame

"""
NeuralNet for the game of TicTacToe.

Author: Evgeny Tyurin, github.com/evg-tyurin
Date: Jan 5, 2018.

Based on the OthelloNNet by SourKream and Surag Nair.
"""


class TicTacToeNNet:
    if TYPE_CHECKING:
        model: Model[
            TicTacToe3DBoardTensor, tuple[list[TicTacToe3DPolicyTensor], list[float]]
        ]
    else:
        model: Model

    def __init__(self, game: TicTacToe3DGame, args: TicTacToe3DNNArg):
        # game params
        self.board_z, self.board_x, self.board_y = game.get_board_size()
        self.action_size = game.get_action_size()
        self.args = args

        # Neural Net
        self.input_boards = Input(
            shape=(self.board_z, self.board_x, self.board_y)
        )  # s: batch_size x board_x x board_y

        x_image = Reshape((self.board_z, self.board_x, self.board_y, 1))(
            self.input_boards
        )  # batch_size  x board_x x board_y x 1
        h_conv1 = Activation("relu")(
            BatchNormalization(axis=3)(
                Conv3D(args.num_channels, 3, padding="same")(x_image)
            )
        )  # batch_size  x board_x x board_y x num_channels
        h_conv2 = Activation("relu")(
            BatchNormalization(axis=3)(
                Conv3D(args.num_channels, 3, padding="same")(h_conv1)
            )
        )  # batch_size  x board_x x board_y x num_channels
        h_conv3 = Activation("relu")(
            BatchNormalization(axis=3)(
                Conv3D(args.num_channels, 3, padding="same")(h_conv2)
            )
        )  # batch_size  x (board_x) x (board_y) x num_channels
        h_conv4 = Activation("relu")(
            BatchNormalization(axis=3)(
                Conv3D(args.num_channels, 3, padding="valid")(h_conv3)
            )
        )  # batch_size  x (board_x-2) x (board_y-2) x num_channels
        h_conv4_flat = Flatten()(h_conv4)
        s_fc1 = Dropout(args.dropout)(
            Activation("relu")(BatchNormalization(axis=1)(Dense(1024)(h_conv4_flat)))
        )  # batch_size x 1024
        s_fc2 = Dropout(args.dropout)(
            Activation("relu")(BatchNormalization(axis=1)(Dense(512)(s_fc1)))
        )  # batch_size x 1024
        self.pi = Dense(self.action_size, activation="softmax", name="pi")(
            s_fc2
        )  # batch_size x self.action_size
        self.v = Dense(1, activation="tanh", name="v")(s_fc2)  # batch_size x 1

        self.model = Model(inputs=self.input_boards, outputs=[self.pi, self.v])
        self.model.compile(
            loss=["categorical_crossentropy", "mean_squared_error"],
            optimizer=Adam(args.lr),
        )
