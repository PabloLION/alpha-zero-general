import logging
import os
from collections import deque
from dataclasses import dataclass
from pickle import Pickler, Unpickler
from random import shuffle

import numpy as np
from numpy import random
from tqdm import tqdm

from alpha_zero_general import (
    CheckpointFile,
    GenericBoardTensor,
    MctsArgs,
    RawTrainingExample,
    TrainExampleHistory,
    TrainingExample,
)
from alpha_zero_general.arena import Arena
from alpha_zero_general.game import GenericGame
from alpha_zero_general.mcts import MCTS
from alpha_zero_general.neural_net import NeuralNetInterface

log = logging.getLogger(__name__)


@dataclass(frozen=True)  # freeze to check for immutability in refactor
class CoachArgs:
    num_iters: int
    num_eps: int
    temp_threshold: int
    update_threshold: float
    max_len_of_queue: int
    num_mcts_sims: int
    arena_compare: int
    c_puct: float
    checkpoint: str
    load_model: bool
    load_folder_file: tuple[str, str]
    num_iters_for_train_examples_history: int
    max_len_of_queue: int
    load_folder_file: tuple[str, str]

    def to_mcts_args(self) -> MctsArgs:
        return MctsArgs(
            num_mcts_sims=self.num_mcts_sims,
            c_puct=self.c_puct,
        )


class Coach:
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game: GenericGame, nn: NeuralNetInterface, args: CoachArgs):
        self.game = game
        self.nn = nn
        self.pnet = self.nn.__class__(self.game)  # the competitor network
        self.args: CoachArgs = args
        self.mcts = MCTS(self.game, self.nn, self.args.to_mcts_args())
        self.train_examples_history: TrainExampleHistory = (
            []
        )  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skip_first_self_play = False  # can be overriden in loadTrainExamples()

    def execute_episode(self) -> list[TrainingExample]:
        """
        Executes one episode of self-play, starting with player 1.

        As the game is played, each turn is added as a training example to
        train_examples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in train_examples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            train_examples: a list of examples of the form
                        (canonical_board, current_player, pi, v)
                        pi is the MCTS informed policy vector, v is +1 if
                        the player eventually won the game, else -1.
        """
        raw_train_examples: list[RawTrainingExample] = []
        # train_examples: board, player, policy, value
        board = self.game.get_init_board()
        self.current_player = 1
        episode_step = 0

        while True:
            episode_step += 1
            canonical_board = self.game.get_canonical_form(board, self.current_player)
            temp = int(episode_step < self.args.temp_threshold)

            pi = self.mcts.get_action_probabilities(canonical_board, temperature=temp)
            sym = self.game.get_symmetries(canonical_board, pi)
            for b, p in sym:
                raw_train_examples.append(
                    RawTrainingExample(b, self.current_player, p, 0)
                )

            action = random.choice(len(pi), p=pi)
            board, self.current_player = self.game.get_next_state(
                board, self.current_player, action
            )

            r = self.game.get_game_ended(board, self.current_player)

            if r != 0:  # game ended
                return [  # filter out draws #TODO: use filter()
                    TrainingExample(
                        rte.board,
                        rte.policy,
                        r * ((-1) ** (rte.current_player != self.current_player)),
                    )
                    for rte in raw_train_examples
                ]

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in train_examples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        for i in range(1, self.args.num_iters + 1):
            # bookkeeping
            log.info(f"Starting Iter #{i} ...")
            # examples of the iteration
            if not self.skip_first_self_play or i > 1:
                iteration_train_examples: deque[TrainingExample] = deque(
                    [], maxlen=self.args.max_len_of_queue
                )

                for _ in tqdm(range(self.args.num_eps), desc="Self Play"):
                    self.mcts = MCTS(
                        self.game, self.nn, self.args.to_mcts_args()
                    )  # reset search tree
                    iteration_train_examples += self.execute_episode()

                # save the iteration examples to the history
                self.train_examples_history.append(list(iteration_train_examples))

            if (
                len(self.train_examples_history)
                > self.args.num_iters_for_train_examples_history
            ):
                log.warning(
                    f"Removing the oldest entry in train_examples. len(train_examplesHistory) = {len(self.train_examples_history)}"
                )
                self.train_examples_history.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)
            self.save_train_examples(i - 1)

            # shuffle examples before training
            train_examples: list[TrainingExample] = []
            for e in self.train_examples_history:
                train_examples.extend(e)
            shuffle(train_examples)

            # training new network, keeping a copy of the old one
            self.nn.save_checkpoint(
                folder=self.args.checkpoint, filename="temp.pth.tar"
            )

            self.pnet.load_checkpoint(
                folder=self.args.checkpoint, filename="temp.pth.tar"
            )  # #TODO: using file system to pass the model is not the best way
            # we should add a method to pass the model directly.
            pmcts = MCTS(self.game, self.pnet, self.args.to_mcts_args())

            def get_p1_policy(board: GenericBoardTensor) -> int:
                return int(
                    np.argmax(a=pmcts.get_action_probabilities(board, temperature=0))
                )

            self.nn.train(train_examples)
            nmcts = MCTS(self.game, self.nn, self.args.to_mcts_args())

            def get_p2_policy(board: GenericBoardTensor) -> int:
                return int(
                    np.argmax(nmcts.get_action_probabilities(board, temperature=0))
                )

            log.info("PITTING AGAINST PREVIOUS VERSION")

            arena = Arena(
                get_p1_policy,
                get_p2_policy,
                self.game,
            )
            pwins, nwins, draws = arena.play_games(self.args.arena_compare)

            log.info("NEW/PREV WINS : %d / %d ; DRAWS : %d" % (nwins, pwins, draws))
            if (
                pwins + nwins == 0
                or float(nwins) / (pwins + nwins) < self.args.update_threshold
            ):
                log.info("REJECTING NEW MODEL")
                self.nn.load_checkpoint(
                    folder=self.args.checkpoint, filename="temp.pth.tar"
                )
            else:
                log.info("ACCEPTING NEW MODEL")
                self.nn.save_checkpoint(
                    folder=self.args.checkpoint, filename=self.get_checkpoint_file(i)
                )
                self.nn.save_checkpoint(
                    folder=self.args.checkpoint, filename="best.pth.tar"
                )

    def get_checkpoint_file(self, iteration: int) -> CheckpointFile:
        return "checkpoint_" + str(iteration) + ".pth.tar"

    def save_train_examples(self, iteration: int):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(
            folder, self.get_checkpoint_file(iteration) + ".examples"
        )
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.train_examples_history)
        f.closed

    def load_train_examples(self):
        model_file = os.path.join(
            self.args.load_folder_file[0], self.args.load_folder_file[1]
        )
        examples_file = model_file + ".examples"
        if not os.path.isfile(examples_file):
            log.warning(f'File "{examples_file}" with train_examples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                return
        else:
            log.info("File with train_examples found. Loading it...")
            with open(examples_file, "rb") as f:
                self.train_examples_history = Unpickler(f).load()
            log.info("Loading done!")

            # examples based on the model were already collected (loaded)
            self.skip_first_self_play = True
