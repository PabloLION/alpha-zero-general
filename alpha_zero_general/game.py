from abc import ABC, abstractmethod
from typing import Generic

from alpha_zero_general import BoardTensor, BooleanBoard, PolicyTensor


class GenericGame(ABC, Generic[BoardTensor, BooleanBoard, PolicyTensor]):
    """
    This class specifies the base Game class. To define your own game, subclass
    this class and implement the functions below. This works when the game is
    two-player, adversarial, and turn-based.

    Use 1 for player1 and -1 for player2.

    See othello/OthelloGame.py for an example implementation.
    """

    @abstractmethod
    def __init__(self) -> None:
        raise NotImplementedError(
            "The __init__ method must be implemented by the subclass"
        )

    @abstractmethod
    def get_init_board(self) -> BoardTensor:
        """
        Returns:
            start_board: a representation of the board (ideally this is the form
                         that will be the input to your neural network)
        """
        raise NotImplementedError("get_init_board must be implemented by the subclass")

    @abstractmethod
    def get_board_size(self) -> tuple[int, ...]:
        """
        Returns:
            (x, y, ...): a tuple of board dimensions
        """
        raise NotImplementedError("get_board_size must be implemented by the subclass")

    @abstractmethod
    def get_action_size(self) -> int:
        """
        Returns:
            action_size: number of all possible actions
        """
        raise NotImplementedError("get_action_size must be implemented by the subclass")

    @abstractmethod
    def get_next_state(
        self, board: BoardTensor, player: int, action: int
    ) -> tuple[BoardTensor, int]:
        """
        Input:
            board: current board
            player: current player (1 or -1)
            action: action taken by current player

        Returns:
            next_board: board after applying action
            next_player: player who plays in the next turn (should be -player)
        """
        raise NotImplementedError("get_next_state must be implemented by the subclass")

    @abstractmethod
    def get_valid_moves(self, board: BoardTensor, player: int) -> BooleanBoard:
        """
        Input:
            board: current board
            player: current player

        Returns:
            valid_moves: a binary vector of length self.get_action_size(), 1 for
                         moves that are valid from the current board and player,
                         0 for invalid moves
        """
        raise NotImplementedError("get_valid_moves must be implemented by the subclass")

    @abstractmethod
    def get_game_ended(self, board: BoardTensor, player: int) -> float:
        """
        #TODO:
            Function name not corresponding to return type.
            Here it gets `value`, not "if ended"

        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost,
                small non-zero value for draw.
        """
        raise NotImplementedError("get_game_ended must be implemented by the subclass")

    @abstractmethod
    def get_canonical_form(self, board: BoardTensor, player: int) -> BoardTensor:
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            canonical_board: returns canonical form of board. The canonical form
                             should be independent of player. For e.g. in chess,
                             the canonical form can be chosen to be from the pov
                             of white. When the player is white, we can return
                             board as is. When the player is black, we can invert
                             the colors and return the board.
        """
        raise NotImplementedError(
            "get_canonical_form must be implemented by the subclass"
        )

    @abstractmethod
    def get_symmetries(
        self, board: BoardTensor, pi: PolicyTensor
    ) -> list[tuple[BoardTensor, PolicyTensor]]:
        """
        #TODO:
            - Offer a new function to flip and rotate the board as the base
                element of the symmetry group, and we can generate the rest of
                the group by applying the base element multiple times.

        Input:
            board: current board
            pi: policy vector of size self.get_action_size()

        Returns:
            symm_forms: a list of [(board, pi)] where each tuple is a symmetrical
                        form of the board and the corresponding pi vector. This
                        is used when training the neural network from examples.
        """
        raise NotImplementedError("get_symmetries must be implemented by the subclass")

    @abstractmethod
    def get_board_str(self, board: BoardTensor) -> str:
        """
        Input:
            board: current board

        Returns:
            board_string: a quick conversion of board to a string format.
                          Required by MCTS for hashing.
        """
        raise NotImplementedError("get_board_str must be implemented by the subclass")

    @abstractmethod
    def get_board_hash(self, board: BoardTensor) -> int:
        """
        Input:
            board: current board

        Returns:
            hash: a quick conversion of board to a hashable format.
                    Required by MCTS for hashing.
        """
        raise NotImplementedError("get_board_hash must be implemented by the subclass")

    # #TODO: planned
    # @abstractmethod
    # def move_is_valid(self, board: BoardTensor, player: int, action: int) -> bool:
    #     """
    #     Input:
    #         board: current board
    #         player: current player
    #         action: action to be taken

    #     Returns:
    #         is_valid: whether the action is valid
    #     """
    #     raise NotImplementedError("move_is_valid must be implemented by the subclass")

    # @abstractmethod
    # def __hash__(self) -> int:
    #     """
    #     Use hash to store game in dictionary, instead of string representation.
    #     This hash should include these parameters:
    #         - board size
    #         - current player
    #         - board state
    #         - game specific parameters

    #     Returns:
    #         hash: hash of the game
    #     """
    #     raise NotImplementedError("__hash__ must be implemented by the subclass")
