class Game:
    """
    This class specifies the base Game class. To define your own game, subclass
    this class and implement the functions below. This works when the game is
    two-player, adversarial, and turn-based.

    Use 1 for player1 and -1 for player2.

    See othello/OthelloGame.py for an example implementation.
    """

    def __init__(self):
        raise NotImplementedError(
            "The __init__ method must be implemented by the subclass"
        )

    def get_init_board(self):
        """
        Returns:
            start_board: a representation of the board (ideally this is the form
                         that will be the input to your neural network)
        """
        raise NotImplementedError("get_init_board must be implemented by the subclass")

    def get_board_size(self):
        """
        Returns:
            (x, y): a tuple of board dimensions
        """
        raise NotImplementedError("get_board_size must be implemented by the subclass")

    def get_action_size(self):
        """
        Returns:
            action_size: number of all possible actions
        """
        raise NotImplementedError("get_action_size must be implemented by the subclass")

    def get_next_state(self, board, player, action):
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

    def get_valid_moves(self, board, player):
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

    def get_game_ended(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw.
        """
        raise NotImplementedError("get_game_ended must be implemented by the subclass")

    def get_canonical_form(self, board, player):
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

    def get_symmetries(self, board, pi):
        """
        Input:
            board: current board
            pi: policy vector of size self.get_action_size()

        Returns:
            symm_forms: a list of [(board, pi)] where each tuple is a symmetrical
                        form of the board and the corresponding pi vector. This
                        is used when training the neural network from examples.
        """
        raise NotImplementedError("get_symmetries must be implemented by the subclass")

    def string_representation(self, board):
        """
        Input:
            board: current board

        Returns:
            board_string: a quick conversion of board to a string format.
                          Required by MCTS for hashing.
        """
        raise NotImplementedError(
            "string_representation must be implemented by the subclass"
        )
