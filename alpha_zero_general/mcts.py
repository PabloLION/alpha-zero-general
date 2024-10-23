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

log = logging.getLogger(__name__)


class MCTS:
    """
    This class handles the MCTS tree.
    """

    game: GenericGame
    nnet: NeuralNet
    args: MctsArgs
    Qsa: dict[tuple[str, int], float]  # Q values for s,a (as in the paper)
    Nsa: dict[tuple[str, int], int]  # #times edge s,a was visited
    Ns: dict[str, int]  # #_times board s was visited
    Ps: dict[str, GenericPolicyTensor]  # policy tensor (returned by neural net)
    Es: dict[str, float]  # game.get_game_ended ended for board s
    Vs: dict[str, GenericBooleanBoardTensor]  # game.get_valid_moves for board s

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
        counts = array(
            [
                self.Nsa[(s, a)] if (s, a) in self.Nsa else 0
                for a in range(self.game.get_action_size())
            ]
        )

        if temp == 0:
            best_as = argwhere(counts == max(counts)).flatten()
            best_a = random.choice(best_as)
            prob: GenericPolicyTensor = zeros(len(counts))
            prob[best_a] = 1
            return prob

        counts = [x ** (1.0 / temp) for x in counts]
        counts_sum = float(sum(counts))
        prob = array([x / counts_sum for x in counts])
        return prob

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

        s = self.game.string_representation(canonical_board)

        if s not in self.Es:
            self.Es[s] = self.game.get_game_ended(canonical_board, 1)
        if self.Es[s] != 0:
            # terminal node
            return -self.Es[s]

        if s not in self.Ps:
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

        valid = self.Vs[s]
        cur_best = -float("inf")
        best_act = -1

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

        a = best_act
        next_s, next_player = self.game.get_next_state(canonical_board, 1, a)
        next_s = self.game.get_canonical_form(next_s, next_player)

        v = self.search(next_s)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (
                self.Nsa[(s, a)] + 1
            )
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v
