import numpy as np

from alpha_zero_general import GenericBoardTensor
from alpha_zero_general.game import GenericGame
from alpha_zero_general.rts.src.board import Board
from alpha_zero_general.rts.src.config import (
    A_TYPE_IDX,
    FPS,
    NUM_ACTS,
    NUM_ENCODERS,
    P_NAME_IDX,
    TIME_IDX,
)
from alpha_zero_general.rts.src.config_class import CONFIG

""" USE_TIMEOUT, MAX_TIME, d_a_type, a_max_health, INITIAL_GOLD, TIMEOUT, visibility"""

"""
RTSGame.pyefined rules for RTS game TD2020
Includes: 
- init - contains board configuration
- get_game_ended - contains end game checking
"""


# noinspection PyPep8Naming,PyMethodMayBeStatic
class RTSGame(GenericGame):
    def __init__(self, n: int = CONFIG.grid_size) -> None:
        self.n = n

        self.initial_board_config = CONFIG.initial_board_config

    def set_init_board(self, board_config) -> None:
        """
        Sets initial_board_config. This function can be used dynamically to change board configuration. It is currently being used by rts_ue4.py, to set board configuration from ue4 game state
        :param board_config: new initial board configuration
        """
        self.initial_board_config = board_config

    def get_init_board(self) -> GenericBoardTensor:
        """
        :return: Returns new board from initial_board_config. That config can be dynamically changed as game progresses.
        """
        b = Board(self.n)
        remaining_time = (
            None  # when setting initial board, remaining time might be different
        )
        for e in self.initial_board_config:
            b.pieces[e.x, e.y] = [
                e.player,
                e.a_type,
                e.health,
                e.carry,
                e.gold,
                e.timeout,
            ]
            remaining_time = e.timeout
        # remaining time is stored in all squares
        b.pieces[:, :, TIME_IDX] = remaining_time
        return np.array(b.pieces)

    def get_board_size(self) -> tuple[int, int, int]:
        # (a,b) tuple
        return self.n, self.n, NUM_ENCODERS

    def get_action_size(self) -> int:
        return self.n * self.n * NUM_ACTS + 1

    def get_next_state(
        self, board: GenericBoardTensor, player: int, action: int
    ) -> tuple[GenericBoardTensor, int]:
        """
        Gets next state for board. It also updates tick for board as game tick iterations are transfered within board as 6. parameter
        :param board: current board
        :param player: player executing action
        :param action: action to apply to new board
        :return: new board with applied action
        """
        b = Board(self.n)
        b.pieces = np.copy(board)

        y, x, action_index = np.unravel_index(action, [self.n, self.n, NUM_ACTS])
        move = (x, y, action_index)

        # first execute move, then run time function to destroy any actors if needed
        b.execute_move(move, player)

        # get config for timeout
        if player == 1:
            USE_TIMEOUT = CONFIG.player1_config.USE_TIMEOUT
        else:
            USE_TIMEOUT = CONFIG.player2_config.USE_TIMEOUT

        # update timer on every tile:
        if USE_TIMEOUT:
            b.pieces[:, :, TIME_IDX] -= 1
        else:
            b.pieces[:, :, TIME_IDX] += 1
            b.time_killer(player)

        return b.pieces, -player

    def get_valid_moves(self, board: GenericBoardTensor, player: int):
        valids = []
        b = Board(self.n)
        b.pieces = np.copy(board)

        if player == 1:
            config = CONFIG.player1_config
        else:
            config = CONFIG.player2_config

        for y in range(self.n):
            for x in range(self.n):
                if (
                    b[x][y][P_NAME_IDX] == player and b[x][y][A_TYPE_IDX] != 1
                ):  # for this player and not Gold
                    valids.extend(b.get_moves_for_square(x, y, config=config))
                else:
                    valids.extend([0] * NUM_ACTS)
        valids.append(0)  # because of that +1 in action Size

        return np.array(valids)

    # noinspection PyUnusedLocal
    def get_game_ended(self, board: GenericBoardTensor, player) -> float:
        """
        Ok, this function is where it gets complicated...
        See, its  hard to decide when to finish rts game, as players might not have enough time to execute wanted actions, but in the other hand, if players are left to play for too long, games become very long, or even 'infinitely' long
        Few different approaches have been used - one is with killer_function that is starting to gradually reduce health of players as the game progresses, so players that produce more units could live longer or players that attack enemy actors, could pull themselves in winning position, as enemy now has less health
        And the other is using timeout. Timeout just cuts game and evaluates winner using one of 3 elo functions. We've found this one to be more useful, as it can be applied in 3d rts games easier and more sensibly.
        :param board: current game state
        :param player: current player
        :return: real number on interval [-1,1] - return 0 if not ended, 1 if player 1 won, -1 if player 1 lost, 0.001 if tie
        """

        n = board.shape[0]

        # detect timeout
        if player == 1:
            USE_TIMEOUT = CONFIG.player1_config.USE_TIMEOUT
        else:
            USE_TIMEOUT = CONFIG.player2_config.USE_TIMEOUT

        if USE_TIMEOUT:
            if board[0, 0, TIME_IDX] < 1:
                score_player1 = self.get_score(board, player)
                score_player2 = self.get_score(board, -player)

                if score_player1 == score_player2:
                    return 0.001
                better_player = 1 if score_player1 > score_player2 else -1
                return better_player
        else:
            if player == 1:
                MAX_TIME = CONFIG.player1_config.MAX_TIME
            else:
                MAX_TIME = CONFIG.player2_config.MAX_TIME

            if board[0, 0, TIME_IDX] >= MAX_TIME:
                return 0.001

        # detect win condition
        sum_p1 = 0
        sum_p2 = 0
        for y in range(n):
            for x in range(n):
                if board[x][y][P_NAME_IDX] == 1:
                    sum_p1 += 1
                if board[x][y][P_NAME_IDX] == -1:
                    sum_p2 += 1

        if sum_p1 < 2:  # SUM IS 1 WHEN PLAYER ONLY HAS MINERALS LEFT
            return -1
        if sum_p2 < 2:  # SUM IS 1 WHEN PLAYER ONLY HAS MINERALS LEFT
            return +1

        # detect no valid actions - possible tie by overpopulating on non-attacking units and buildings - all fields are full or one player is surrounded:
        if sum(self.get_valid_moves(board, 1)) == 0:
            return -1

        if sum(self.get_valid_moves(board, -1)) == 0:
            return 1
        # continue game
        return 0

    def get_canonical_form(self, board: GenericBoardTensor, player: int):
        b = np.copy(board)
        b[:, :, P_NAME_IDX] = b[:, :, P_NAME_IDX] * player
        return b

    def get_symmetries(self, board: GenericBoardTensor, pi):
        # mirror, rotational
        assert len(pi) == self.n * self.n * NUM_ACTS + 1  # 1 for pass
        pi_board = np.reshape(pi[:-1], (self.n, self.n, NUM_ACTS))
        return_list = []
        for i in range(1, 5):
            for j in [True, False]:
                newB = np.rot90(board, i)
                newPi = np.rot90(pi_board, i)
                if j:
                    newB = np.fliplr(newB)
                    newPi = np.fliplr(newPi)
                return_list += [(newB, list(newPi.ravel()) + [pi[-1]])]
        return return_list

    def get_board_str(self, board: GenericBoardTensor):
        return np.array2string(board)

    def get_board_hash(self, board: GenericBoardTensor) -> int:
        return hash(board.tobytes())

    def get_score(self, board: GenericBoardTensor, player: int):
        """
        Uses one of 3 elo functions that determine better player
        :param board: game state
        :param player: current player
        :return: elo for current player on this board
        """
        b = Board(self.n)
        b.pieces = np.copy(board)

        # can use different score functions for each player
        if player == 1:
            score_function = CONFIG.player1_config.score_function
        else:
            score_function = CONFIG.player2_config.score_function

        if score_function == 1:
            return b.get_health_score(player)
        elif score_function == 2:
            return b.get_money_score(player)
        else:
            return b.get_combined_score(player)


def display(board):
    """
    Console presentation of board
    :param board: game state
    :return: /
    """
    from alpha_zero_general.rts.visualization.rts_pygame import (
        init_visuals,
        update_graphics,
    )

    if not CONFIG.visibility:
        return

    n = board.shape[0]
    if CONFIG.visibility > 3:
        game_display, clock = init_visuals(n, n, CONFIG.visibility)
        update_graphics(board, game_display, clock, FPS)
    else:
        for y in range(n):
            print("-" * (n * 8 + 1))
            for x in range(n):
                a_player = board[x][y][P_NAME_IDX]
                if a_player == 1:
                    a_player = "+1"
                if a_player == -1:
                    a_player = "-1"
                if a_player == 0:
                    a_player = " 0"
                print("|" + a_player + " " + str(board[x][y][A_TYPE_IDX]) + " ", end="")
            print("|")
        print("-" * (n * 8 + 1))
