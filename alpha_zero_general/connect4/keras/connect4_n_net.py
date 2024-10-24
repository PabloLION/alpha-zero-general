from typing import Any

import tensorflow as tf
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
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


# Helper function for a ReLU followed by Batch Normalization
def relu_bn(inputs: Any) -> Any:
    relu_input = relu(inputs)
    bn = BatchNormalization()(relu_input)
    return bn


# Residual Block implementation
def residual_block(x: Any, filters: int, kernel_size: int = 3) -> Any:
    y = Conv2D(filters=filters, kernel_size=kernel_size, strides=1, padding="same")(x)
    y = relu_bn(y)
    y = Conv2D(filters=filters, kernel_size=kernel_size, strides=1, padding="same")(y)
    y = BatchNormalization()(y)
    out = Add()([x, y])
    out = relu(out)
    return out


# Value head for the network (producing the value of the board)
def value_head(input: Any) -> Any:
    conv1 = Conv2D(filters=1, kernel_size=1, strides=1, padding="same")(input)
    bn1 = BatchNormalization()(conv1)
    bn1_relu = relu(bn1)
    flat = Flatten()(bn1_relu)
    dense1 = Dense(256)(flat)
    dn_relu = relu(dense1)
    dense2 = Dense(256)(dn_relu)
    return dense2


# Policy head for the network (producing the policy vector)
def policy_head(input: Any) -> Any:
    conv1 = Conv2D(filters=1, kernel_size=2, strides=1, padding="same")(input)
    bn1 = BatchNormalization()(conv1)
    bn1_relu = relu(bn1)
    flat = Flatten()(bn1_relu)
    return flat


# Connect4 Neural Network class
class Connect4NNet:
    def __init__(self, game: Any, args: Any):
        # Game parameters
        self.board_x, self.board_y = game.get_board_size()
        self.action_size = game.get_action_size()
        self.args = args

        # Inputs
        self.input_boards = Input(shape=(self.board_x, self.board_y))
        inputs = Reshape((self.board_x, self.board_y, 1))(self.input_boards)

        # Network architecture
        bn1 = BatchNormalization()(inputs)
        conv1 = Conv2D(
            self.args.num_channels, kernel_size=3, strides=1, padding="same"
        )(bn1)
        t = relu_bn(conv1)

        # Residual layers
        for _ in range(self.args.num_residual_layers):
            t = residual_block(t, filters=self.args.num_channels)

        # Output layers
        self.pi = Dense(self.action_size, activation="softmax", name="pi")(
            policy_head(t)
        )
        self.v = Dense(1, activation="tanh", name="v")(value_head(t))

        # Model definition
        self.model = Model(inputs=self.input_boards, outputs=[self.pi, self.v])

        # Compile the model with losses and optimizer
        self.compile_model()

    def compile_model(self) -> None:
        # Loss functions for policy and value heads
        loss_pi = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        loss_v = tf.keras.losses.MeanSquaredError()

        # Compile the model with Adam optimizer
        self.model.compile(optimizer=Adam(self.args.lr), loss=[loss_pi, loss_v])
