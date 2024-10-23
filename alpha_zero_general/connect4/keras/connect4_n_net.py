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


def relu_bn(inputs: Any) -> Any:
    relu1 = relu(inputs)
    bn = BatchNormalization()(relu1)
    return bn


def residual_block(x: Any, filters: int, kernel_size: int = 3) -> Any:
    y = Conv2D(kernel_size=kernel_size, strides=(1), filters=filters, padding="same")(x)

    y = relu_bn(y)
    y = Conv2D(kernel_size=kernel_size, strides=1, filters=filters, padding="same")(y)

    y = BatchNormalization()(y)
    out = Add()([x, y])
    out = relu(out)

    return out


def value_head(input: Any) -> Any:
    conv1 = Conv2D(kernel_size=1, strides=1, filters=1, padding="same")(input)

    bn1 = BatchNormalization()(conv1)
    bn1_relu = relu(bn1)

    flat = Flatten()(bn1_relu)

    dense1 = Dense(256)(flat)
    dn_relu = relu(dense1)

    dense2 = Dense(256)(dn_relu)

    return dense2


def policy_head(input: Any) -> Any:
    conv1 = Conv2D(kernel_size=2, strides=1, filters=1, padding="same")(input)
    bn1 = BatchNormalization()(conv1)
    bn1_relu = relu(bn1)
    flat = Flatten()(bn1_relu)
    return flat


class Connect4NNet:
    def __init__(self, game: Any, args: Any):
        # game params
        self.board_x, self.board_y = game.get_board_size()
        self.action_size = game.get_action_size()
        self.args = args

        # Neural Net
        # Inputs
        self.input_boards = Input(shape=(self.board_x, self.board_y))
        inputs = Reshape((self.board_x, self.board_y, 1))(self.input_boards)

        bn1 = BatchNormalization()(inputs)
        conv1 = Conv2D(args.num_channels, kernel_size=3, strides=1, padding="same")(bn1)
        t = relu_bn(conv1)

        for _ in range(self.args.num_residual_layers):
            t = residual_block(t, filters=self.args.num_channels)

        self.pi = Dense(self.action_size, activation="softmax", name="pi")(
            policy_head(t)
        )
        self.v = Dense(1, activation="tanh", name="v")(value_head(t))

        self.calculate_loss(self.pi, self.v)

        self.model = Model(inputs=self.input_boards, outputs=[self.pi, self.v])
        self.model.compile(loss=[self.loss_pi, self.loss_v], optimizer=Adam(args.lr))

    def calculate_loss(self, target_pis, target_vs) -> None:
        # outdated APIs
        # self.target_pis = tf.placeholder(tf.float32, shape=[None, self.action_size])
        # self.target_vs = tf.placeholder(tf.float32, shape=[None])
        # self.loss_pi = tf.losses.softmax_cross_entropy(self.target_pis, self.pi)
        # self.loss_v = tf.losses.mean_squared_error(
        #     self.target_vs, tf.reshape(self.v, shape=[-1])
        # )

        self.loss_pi = tf.keras.losses.CategoricalCrossentropy()(target_pis, self.pi)
        self.loss_v = tf.keras.losses.MeanSquaredError()(
            target_vs, tf.reshape(self.v, shape=[-1])
        )

        self.total_loss = self.loss_pi + self.loss_v
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_step = tf.train.AdamOptimizer(self.args.lr).minimize(
                self.total_loss
            )
