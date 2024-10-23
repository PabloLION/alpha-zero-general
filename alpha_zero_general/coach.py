import logging
import os
from collections import deque
from dataclasses import dataclass
from pickle import Pickler, Unpickler
from random import shuffle
from typing import Any, Callable, NamedTuple, TypeAlias

import numpy as np
from numpy import random
from tqdm import tqdm

from alpha_zero_general import MctsArgs
from alpha_zero_general.arena import Arena
from alpha_zero_general.game import GenericBoardTensor, GenericGame
from alpha_zero_general.mcts import MCTS, GenericPolicyTensor

log = logging.getLogger(__name__)

Player = Callable[[Any], int]
Display = Callable[[Any], None]
Board = Any
ExampleValue: TypeAlias = float
PlayerID: TypeAlias = int
# LengthyTrainExample = tuple[
#     GenericBoardTensor, PlayerID, GenericPolicyTensor, ExampleValue
# ]


class RawTrainExample(NamedTuple):
    """
    (canonical_board, current_player, pi, v)
    pi is the MCTS informed policy vector, v is +1 if
    the player eventually won the game, else -1.
    """

    board: GenericBoardTensor
    current_player: PlayerID
    policy: GenericPolicyTensor
    neutral_value: ExampleValue  # from neutral perspective


class TrainExample(NamedTuple):
    board: GenericBoardTensor
    policy: GenericPolicyTensor
    value: ExampleValue  # from player's perspective


TrainExampleHistory = list[list[TrainExample]]
CheckpointFile = str
TrainExamplesFile = str


@dataclass(frozen=True)  # freeze to check for immutability in refactor
class CoachArgs:
    numIters: int
    numEps: int
    tempThreshold: int
    updateThreshold: float
    maxLenOfQueue: int
    numMCTSSims: int
    arenaCompare: int
    cpuct: float
    checkpoint: str
    loadModel: bool
    loadFolderFile: tuple[str, str]
    numItersForTrainExamplesHistory: int
    maxlenOfQueue: int

    def to_mcts_args(self) -> MctsArgs:
        return MctsArgs(
            numMCTSSims=self.numMCTSSims,
            cpuct=self.cpuct,
        )


class Coach:
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game: GenericGame, nnet: Any, args: CoachArgs):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)  # the competitor network
        self.args: CoachArgs = args
        self.mcts = MCTS(self.game, self.nnet, self.args.to_mcts_args())
        self.train_examples_history: TrainExampleHistory = (
            []
        )  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()

    def execute_episode(self) -> list[TrainExample]:
        """
        Executes one episode of self-play, starting with player 1.

        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form
                        (canonical_board, current_player, pi, v)
                        pi is the MCTS informed policy vector, v is +1 if
                        the player eventually won the game, else -1.
        """
        raw_train_examples: list[RawTrainExample] = []
        # train_examples: board, player, policy, value
        board = self.game.get_init_board()
        self.current_player = 1
        episode_step = 0

        while True:
            episode_step += 1
            canonical_board = self.game.get_canonical_form(board, self.current_player)
            temp = int(episode_step < self.args.tempThreshold)

            pi = self.mcts.get_action_prob(canonical_board, temp=temp)
            sym = self.game.get_symmetries(canonical_board, pi)
            for b, p in sym:
                raw_train_examples.append(RawTrainExample(b, self.current_player, p, 0))

            action = random.choice(len(pi), p=pi)
            board, self.current_player = self.game.get_next_state(
                board, self.current_player, action
            )

            r = self.game.get_game_ended(board, self.current_player)

            if r != 0:
                # filter out draws
                return [
                    TrainExample(
                        te.board,
                        te.policy,
                        r * ((-1) ** (te.current_player != self.current_player)),
                    )
                    for te in raw_train_examples
                ]

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        for i in range(1, self.args.numIters + 1):
            # bookkeeping
            log.info(f"Starting Iter #{i} ...")
            # examples of the iteration
            if not self.skipFirstSelfPlay or i > 1:
                iteration_train_examples: deque[TrainExample] = deque(
                    [], maxlen=self.args.maxlenOfQueue
                )

                for _ in tqdm(range(self.args.numEps), desc="Self Play"):
                    self.mcts = MCTS(
                        self.game, self.nnet, self.args
                    )  # reset search tree
                    iteration_train_examples += self.execute_episode()

                # save the iteration examples to the history
                self.train_examples_history.append(iteration_train_examples)

            if (
                len(self.train_examples_history)
                > self.args.numItersForTrainExamplesHistory
            ):
                log.warning(
                    f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.train_examples_history)}"
                )
                self.train_examples_history.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)
            self.save_train_examples(i - 1)

            # shuffle examples before training
            train_examples: list[TrainExample] = []
            for e in self.train_examples_history:
                train_examples.extend(e)
            shuffle(train_examples)

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(
                folder=self.args.checkpoint, filename="temp.pth.tar"
            )
            self.pnet.load_checkpoint(
                folder=self.args.checkpoint, filename="temp.pth.tar"
            )
            pmcts = MCTS(self.game, self.pnet, self.args)

            self.nnet.train(train_examples)
            nmcts = MCTS(self.game, self.nnet, self.args)

            log.info("PITTING AGAINST PREVIOUS VERSION")
            arena = Arena(
                lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
                lambda x: np.argmax(nmcts.getActionProb(x, temp=0)),
                self.game,
            )
            pwins, nwins, draws = arena.playGames(self.args.arenaCompare)

            log.info("NEW/PREV WINS : %d / %d ; DRAWS : %d" % (nwins, pwins, draws))
            if (
                pwins + nwins == 0
                or float(nwins) / (pwins + nwins) < self.args.updateThreshold
            ):
                log.info("REJECTING NEW MODEL")
                self.nnet.load_checkpoint(
                    folder=self.args.checkpoint, filename="temp.pth.tar"
                )
            else:
                log.info("ACCEPTING NEW MODEL")
                self.nnet.save_checkpoint(
                    folder=self.args.checkpoint, filename=self.get_checkpoint_file(i)
                )
                self.nnet.save_checkpoint(
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
        examplesFile = model_file + ".examples"
        if not os.path.isfile(examplesFile):
            log.warning(f'File "{examplesFile}" with trainExamples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                return
        else:
            log.info("File with trainExamples found. Loading it...")
            with open(examplesFile, "rb") as f:
                self.train_examples_history = Unpickler(f).load()
            log.info("Loading done!")

            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True
