from typing import Any

from alpha_zero_general import GenericPolicyTensor
from alpha_zero_general.game import GenericGame
from alpha_zero_general.mcts import GenericPolicyTensor


class NeuralNet:
    """
    This class specifies the base NeuralNet class. To define your own neural
    network, subclass this class and implement the functions below. The neural
    network does not consider the current player, and instead only deals with
    the canonical form of the board.

    See othello/NNet.py for an example implementation.
    """

    def __init__(self, game: GenericGame):
        raise NotImplementedError(
            "The __init__ method must be implemented by the subclass"
        )

    def train(self, examples: list[tuple]):
        """
        This function trains the neural network with examples obtained from
        self-play.

        Input:
            examples: a list of training examples, where each example is of form
                      (board, pi, v). pi is the MCTS informed policy vector for
                      the given board, and v is its value. The examples has
                      board in its canonical form.
        """
        raise NotImplementedError("train method must be implemented by the subclass")

    def predict(self, board: Any) -> tuple[GenericPolicyTensor, float]:
        """
        Input:
            board: current board in its canonical form.

        Returns:
            pi: a policy vector for the current board- a numpy array of length
                game.getActionSize
            v: a float in [-1,1] that gives the value of the current board
        """
        raise NotImplementedError("predict method must be implemented by the subclass")

    def save_checkpoint(self, folder: str, filename: str) -> None:
        """
        Saves the current neural network (with its parameters) in
        folder/filename
        """
        raise NotImplementedError(
            "save_checkpoint method must be implemented by the subclass"
        )

    def load_checkpoint(self, folder: str, filename: str) -> None:
        """
        Loads parameters of the neural network from folder/filename
        """
        raise NotImplementedError(
            "load_checkpoint method must be implemented by the subclass"
        )
