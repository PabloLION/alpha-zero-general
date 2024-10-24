from typing import Any

from tensorflow import Tensor
from tensorflow.keras.activations import relu
from tensorflow.keras.layers import (
    Add,
    BatchNormalization,
    Conv2D,
    Dense,
    Flatten,
    Input,
    Reshape,
)
from tensorflow.keras.losses import CategoricalCrossentropy, MeanSquaredError
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from alpha_zero_general.connect4.connect4_game import Connect4Game

# from tensorflow.python.ops.array_ops import Reshape


class Connect4NNet:
    model: Model  # type: ignore # Model is not subscriptable

    def __init__(self, game: Connect4Game, args: Any):
        # game params
        self.board_x, self.board_y = game.get_board_size()
        self.action_size = game.get_action_size()
        self.args = args

        # Inputs
        self.input_boards = Input(shape=game.get_board_size())
        # inputs: Tensor = cast(Tensor, expand_dims(self.input_boards, axis=-1))

        inputs: Tensor = Reshape((self.board_x, self.board_y, 1))(self.input_boards)

        # Network architecture
        input_bn: Tensor = BatchNormalization()(inputs)
        input_conv = Conv2D(
            self.args.num_channels, kernel_size=3, strides=1, padding="same"
        )(input_bn)
        t: Tensor = relu(input_conv)
        t = BatchNormalization()(t)

        # Residual layers (inlined)
        for _ in range(self.args.num_residual_layers):
            y = Conv2D(
                filters=self.args.num_channels, kernel_size=3, strides=1, padding="same"
            )(t)
            y = relu(y)
            y = BatchNormalization()(y)
            y = Conv2D(
                filters=self.args.num_channels, kernel_size=3, strides=1, padding="same"
            )(y)
            y = BatchNormalization()(y)
            t = Add()([t, y])
            t = relu(t)

        # Policy head
        policy_conv = Conv2D(filters=1, kernel_size=2, strides=1, padding="same")(t)
        policy_bn = BatchNormalization()(policy_conv)
        policy_relu = relu(policy_bn)
        flat_policy = Flatten()(policy_relu)
        self.pi = Dense(self.action_size, activation="softmax", name="pi")(flat_policy)

        # Value head
        value_conv = Conv2D(filters=1, kernel_size=1, strides=1, padding="same")(t)
        value_bn = BatchNormalization()(value_conv)
        value_relu = relu(value_bn)
        flat_value = Flatten()(value_relu)
        dense_value_1 = Dense(256)(flat_value)
        dense_value_1_relu = relu(dense_value_1)
        self.v = Dense(1, activation="tanh", name="v")(dense_value_1_relu)

        # Model definition
        self.model = Model(inputs=self.input_boards, outputs=[self.pi, self.v])

        # Compile the model with losses and optimizer
        self.compile_model()

    def compile_model(self) -> None:
        # Loss functions for policy and value heads
        loss_pi = CategoricalCrossentropy(from_logits=False)
        loss_v = MeanSquaredError()

        # Compile the model with Adam optimizer
        self.model.compile(optimizer=Adam(self.args.lr), loss=[loss_pi, loss_v])  # type: ignore # Model not subscriptable
