import logging
import math

from numpy import argwhere, array, random, zeros

from alpha_zero_general import (
    GenericBoardTensor,
    GenericBooleanBoardTensor,
    GenericPolicyTensor,
    MctsArgs,
)
from alpha_zero_general.game import GenericGame
from alpha_zero_general.neural_net import NeuralNet

EPS = 1e-8
rng = random.default_rng()

log = logging.getLogger(__name__)


class MCTS:
    """
    This class handles the MCTS tree.
    """

    game: GenericGame
    nnet: NeuralNet
    args: MctsArgs

    ### retiring
    Qsa: dict[tuple[str, int], float]  # Q values for s,a (as in the paper)
    Nsa: dict[tuple[str, int], int]  # #times edge s,a was visited
    Ns: dict[str, int]  # #_times board s was visited
    Ps: dict[str, GenericPolicyTensor]  # policy tensor (returned by neural net)
    Es: dict[str, float]  # cache game.get_game_ended for board s
    Vs: dict[str, GenericBooleanBoardTensor]  # game.get_valid_moves for board s
    ### hiring
    q_values_cache: dict[tuple[int, int], float]  # Q values for board_hash, action
    n_edge_visit: dict[tuple[int, int], int]
    # n_edge_visit: #_times edge board_hash, action was visited
    n_node_visit: dict[int, int]  # #times board s was visited
    policy_cache: dict[int, GenericPolicyTensor]  # policy tensor by neural net
    game_value_cache: dict[int, float]
    # cache the game value returned by game.get_game_ended for key as board_hash
    valid_moves_cache: dict[int, GenericBooleanBoardTensor]
    # cache the valid moves returned by game.get_valid_moves for key as board_hash
    board_cache: dict[int, GenericBoardTensor]  # restore the board from board_hash

    def __init__(self, game: GenericGame, nnet: NeuralNet, args: MctsArgs) -> None:
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {}
        self.Nsa = {}
        self.Ns = {}
        self.Ps = {}
        self.Es = {}
        self.Vs = {}
        self.q_values_cache = {}
        self.n_edge_visit = {}
        self.n_node_visit = {}
        self.policy_cache = {}
        self.game_value_cache = {}
        self.valid_moves_cache = {}

    def get_action_prob(
        self, canonical_board: GenericBoardTensor, temp: int = 1
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
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        for _ in range(self.args.num_mcts_sims):
            self.search(canonical_board)

        s = self.game.string_representation(canonical_board)
        h = self.game.get_board_hash(canonical_board)
        counts = array(
            [
                self.Nsa[(s, a)] if (s, a) in self.Nsa else 0
                for a in range(self.game.get_action_size())
            ]
        )
        new_counts = array(
            [
                self.n_edge_visit[(h, a)] if (h, a) in self.n_edge_visit else 0
                for a in range(self.game.get_action_size())
            ]
        )

        if temp == 0:
            all_best_action = argwhere(a=counts == max(counts)).flatten()
            random_best_action = rng.choice(all_best_action)
            prob: GenericPolicyTensor = zeros(len(counts))
            prob[random_best_action] = 1
            return prob

        counts = [x ** (1.0 / temp) for x in counts]
        counts_sum = float(sum(counts))
        prob = array([x / counts_sum for x in counts])

        new_counts = [x ** (1.0 / temp) for x in new_counts]
        new_counts_sum = float(sum(new_counts))
        new_prob = array([x / new_counts_sum for x in new_counts])
        return new_prob

    def search(self, canonical_board: GenericBoardTensor) -> float:
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """

        # retiring
        s = self.game.string_representation(canonical_board)
        if s not in self.Es:
            self.Es[s] = self.game.get_game_ended(canonical_board, 1)
        if self.Es[s] != 0:
            # terminal node
            return -self.Es[s]

        # hiring
        h = self.game.get_board_hash(canonical_board)

        if h not in self.game_value_cache:
            self.game_value_cache[h] = self.game.get_game_ended(canonical_board, 1)
        if self.game_value_cache[h] != 0:  # terminal node
            return -self.game_value_cache[h]

        # retiring (and False)
        if s not in self.Ps and False:
            # leaf node
            self.Ps[s], v = self.nnet.predict(canonical_board)
            valid = self.game.get_valid_moves(canonical_board, 1)
            self.Ps[s] = self.Ps[s] * valid  # masking invalid moves
            sum_Ps_s: float = sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s  # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable

                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.
                log.error("All valid moves were masked, doing a workaround.")
                self.Ps[s] = self.Ps[s] + valid
                self.Ps[s] /= sum(self.Ps[s])

            self.Vs[s] = valid
            self.Ns[s] = 0
            return -v

        # hiring
        if h not in self.policy_cache:
            self.policy_cache[h], v = self.nnet.predict(canonical_board)
            new_valid = self.game.get_valid_moves(canonical_board, 1)
            self.policy_cache[h] = self.policy_cache[h] * new_valid
            sum_policy_cache_h: float = sum(self.policy_cache[h])
            if sum_policy_cache_h > 0:
                self.policy_cache[h] /= sum_policy_cache_h
            else:
                # if all valid moves were masked make all valid moves equally probable

                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.
                # #TODO: better error messagej
                log.error("All valid moves were masked, doing a workaround.")
                self.policy_cache[h] = self.policy_cache[h] + new_valid
                self.policy_cache[h] /= sum(self.policy_cache[h])

            self.valid_moves_cache[h] = new_valid
            self.n_node_visit[h] = 0
            return -v

        # retiring
        valid = self.Vs[s]
        cur_best = -float("inf")
        best_act = -1

        # hiring
        new_valid = self.valid_moves_cache[h]
        new_current_best = float("-inf")
        new_best_action = -1

        # retiring
        # pick the action with the highest upper confidence bound
        for a in range(self.game.get_action_size()):
            if valid[a]:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self.args.c_puct * self.Ps[s][a] * math.sqrt(
                        self.Ns[s]
                    ) / (1 + self.Nsa[(s, a)])
                else:
                    u = (
                        self.args.c_puct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)
                    )  # Q = 0 ?

                if u > cur_best:
                    cur_best = u
                    best_act = a

        # hiring
        for action in range(self.game.get_action_size()):
            if new_valid[action]:
                if (h, action) in self.q_values_cache:
                    u = self.q_values_cache[
                        (h, action)
                    ] + self.args.c_puct * self.policy_cache[h][action] * math.sqrt(
                        self.n_node_visit[h]
                    ) / (
                        1 + self.n_edge_visit[(h, action)]
                    )
                else:
                    u = (
                        self.args.c_puct
                        * self.policy_cache[h][action]
                        * math.sqrt(self.n_node_visit[h] + EPS)
                    )

                if u > new_current_best:
                    new_current_best = u
                    new_best_action = action

        # retiring
        a = best_act
        next_s, next_player = self.game.get_next_state(canonical_board, 1, a)
        next_s = self.game.get_canonical_form(next_s, next_player)

        v = self.search(next_s)

        # hiring
        action = new_best_action
        next_board, next_player = self.game.get_next_state(canonical_board, 1, action)
        next_board = self.game.get_canonical_form(next_board, next_player)

        new_v = self.search(next_board)

        # retiring
        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (
                self.Nsa[(s, a)] + 1
            )
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1

        # hiring

        if (h, action) in self.q_values_cache:
            self.q_values_cache[(h, action)] = (
                self.n_edge_visit[(h, action)] * self.q_values_cache[(h, action)]
                + new_v
            ) / (self.n_edge_visit[(h, action)] + 1)
            self.n_edge_visit[(h, action)] += 1
        else:
            self.q_values_cache[(h, action)] = new_v
            self.n_edge_visit[(h, action)] = 1

        self.n_node_visit[h] += 1
        return -new_v  # instead of return -v
