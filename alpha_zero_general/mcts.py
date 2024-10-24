import logging
import math
from typing import Generic

from numpy import argwhere, array, zeros

from alpha_zero_general import (
    EPS,
    RNG,
    GenericBoardTensor,
    GenericBooleanBoardTensor,
    GenericPolicyTensor,
    MctsArgs,
)
from alpha_zero_general.game import (
    BoardTensorType,
    BooleanBoardType,
    GenericGame,
    PolicyTensorType,
)
from alpha_zero_general.neural_net import NeuralNetInterface

log = logging.getLogger(__name__)


class MCTS(Generic[BoardTensorType, BooleanBoardType, PolicyTensorType]):
    """
    This class handles the MCTS tree.
    """

    game: GenericGame[BoardTensorType, BooleanBoardType, PolicyTensorType]
    nn: NeuralNetInterface[BoardTensorType, BooleanBoardType, PolicyTensorType]
    args: MctsArgs

    q_values_cache: dict[tuple[int, int], float]  # Q_sa, Q value of board_hash,action
    n_edge_visit: dict[tuple[int, int], int]
    # n_edge_visit: N(s,a), #_times edge board_hash, action was visited
    # this variable is explicitly used to in both search and get_action_prob
    # which adds the coupling between the two functions. #TODO: try decouple

    n_node_visit: dict[int, int]  # N(s,*) #_times board s was visited
    policy_cache: dict[int, GenericPolicyTensor]  # Pi(s) policy_tensor of s from nn
    game_value_cache: dict[int, float]  # ref-note: old Es
    # cache the game value returned by game.get_game_ended of key board_hash
    valid_moves_cache: dict[int, GenericBooleanBoardTensor]  # ref-note: old Vs
    # cache the valid moves returned by game.get_valid_moves of key board_hash
    board_cache: dict[int, GenericBoardTensor]  # restore the board from hash

    def __init__(
        self,
        game: GenericGame[BoardTensorType, BooleanBoardType, PolicyTensorType],
        nn: NeuralNetInterface[BoardTensorType, BooleanBoardType, PolicyTensorType],
        args: MctsArgs,
    ) -> None:
        self.game = game
        self.nn = nn
        self.args = args
        self.q_values_cache = {}
        self.n_edge_visit = {}
        self.n_node_visit = {}
        self.policy_cache = {}
        self.game_value_cache = {}
        self.valid_moves_cache = {}
        self.board_cache = {}

    def _cached_hash(self, canonical_board: BoardTensorType) -> int:
        """
        Cache the board and return the hash of the board.
        """
        h = self.game.get_board_hash(canonical_board)
        if h not in self.board_cache:
            self.board_cache[h] = canonical_board
        # we can also check collision here
        return h

    def get_action_probabilities(
        self, canonical_board: BoardTensorType, temperature: int = 1
    ) -> GenericPolicyTensor:
        """
        This function performs num_mcts_sims simulations of MCTS starting from
        canonicalBoard.

        Args:
            canonical_board: a board that is a canonical form of the current
                                board state.
            temp: temperature parameter in (0, 1] that controls the level of
                    exploration of the MCTS. A higher value will encourage the
                    AI to explore new actions while a lower value will make it
                    greedy.

        Returns:
            prob: a policy vector where the probability of the ith action is
                   proportional to n_edge_visit[(board,action)]**(1./temp)
        """

        for _ in range(self.args.num_mcts_sims):
            self.search(canonical_board)

        h = self._cached_hash(canonical_board)
        counts = array(
            [
                self.n_edge_visit[(h, a)] if (h, a) in self.n_edge_visit else 0
                for a in range(self.game.get_action_size())
            ]
        )

        if temperature == 0:
            all_best_action = argwhere(a=counts == max(counts)).flatten()
            random_best_action = RNG.choice(all_best_action)
            # prob: GenericPolicyTensor = zeros(len(counts))
            prob: GenericPolicyTensor = zeros(len(counts))
            prob[random_best_action] = 1
            return prob

        counts = [x ** (1.0 / temperature) for x in counts]
        counts_sum = float(sum(counts))
        prob = array([x / counts_sum for x in counts])
        return prob

    def search(self, canonical_board: BoardTensorType) -> float:
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path.
        Update attr n_edge_visit, n_node_visit and q_values_cache.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonical_board

        REF-NOTE: the return is more like an evaluation of the current board.
        We should improve the documentation when we have a better understanding
        on how players work. #TODO
        """

        h = self._cached_hash(canonical_board)

        if h not in self.game_value_cache:
            self.game_value_cache[h] = self.game.get_game_ended(canonical_board, 1)
        if self.game_value_cache[h] != 0:  # terminal node
            return -self.game_value_cache[h]

        if h not in self.policy_cache:  # leaf node
            self.policy_cache[h], v = self.nn.predict(canonical_board)
            valid_move = self.game.get_valid_moves(canonical_board, 1)
            self.policy_cache[h] = self.policy_cache[h] * valid_move  # mask invalid
            sum_policy_cache_h: float = sum(self.policy_cache[h])
            if sum_policy_cache_h > 0:
                self.policy_cache[h] /= sum_policy_cache_h
            else:
                # if all valid moves were masked make all valid moves equally probable

                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.
                # #TODO: better error message
                log.error("All valid moves were masked, doing a workaround.")
                self.policy_cache[h] = self.policy_cache[h] + valid_move
                self.policy_cache[h] /= sum(self.policy_cache[h])

            self.valid_moves_cache[h] = valid_move
            self.n_node_visit[h] = 0
            return -v

        valid_move = self.valid_moves_cache[h]
        best_u = float("-inf")
        best_action = -1

        # pick the action with the highest upper confidence bound
        for action in range(self.game.get_action_size()):
            if not valid_move[action]:
                continue
            if (h, action) in self.q_values_cache:
                u: float = self.q_values_cache[
                    (h, action)
                ] + self.args.c_puct * self.policy_cache[h][action] * math.sqrt(
                    self.n_node_visit[h]
                ) / (
                    1 + self.n_edge_visit[(h, action)]
                )
            else:
                u: float = (
                    self.args.c_puct
                    * self.policy_cache[h][action]
                    * math.sqrt(self.n_node_visit[h] + EPS)
                )  # Q = 0 ?

            if u > best_u:
                best_u = u
                best_action = action

        action = best_action
        next_board, next_player = self.game.get_next_state(canonical_board, 1, action)
        next_board = self.game.get_canonical_form(next_board, next_player)

        v = self.search(next_board)

        if (h, action) in self.q_values_cache:
            self.q_values_cache[(h, action)] = (
                self.n_edge_visit[(h, action)] * self.q_values_cache[(h, action)] + v
            ) / (self.n_edge_visit[(h, action)] + 1)
            self.n_edge_visit[(h, action)] += 1
        else:
            self.q_values_cache[(h, action)] = v
            self.n_edge_visit[(h, action)] = 1

        self.n_node_visit[h] += 1

        return -v
