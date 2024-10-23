import logging
from dataclasses import dataclass

import coloredlogs  # type: ignore

from alpha_zero_general.coach import Coach, CoachArgs
from alpha_zero_general.othello.othello_game import OthelloGame as Game
from alpha_zero_general.othello.pytorch.n_net import NNetWrapper as nn

log = logging.getLogger(__name__)

coloredlogs.install(  # type: ignore
    level="INFO"
)  # Change this to DEBUG to see more info.


@dataclass(frozen=True)  # freeze to check for immutability in refactor
class MainArgs:
    n_iter: int = 1000
    n_eps: int = 100
    temp_threshold: int = 15
    update_threshold: float = 0.6
    maxlen_of_queue: int = 200000
    n_mcts_sims: int = 25
    arena_compare: int = 40
    cpuct: int = 1
    checkpoint: str = "./temp/"
    load_model: bool = False
    load_folder_file: tuple[str, str] = ("/dev/models/8x100x50", "best.pth.tar")
    n_iters_for_train_examples_history: int = 20

    def to_coach_args(self) -> CoachArgs:
        return CoachArgs(
            numIters=self.n_iter,
            numEps=self.n_eps,
            tempThreshold=self.temp_threshold,
            updateThreshold=self.update_threshold,
            maxLenOfQueue=self.maxlen_of_queue,
            numMCTSSims=self.n_mcts_sims,
            arenaCompare=self.arena_compare,
            cpuct=self.cpuct,
            checkpoint=self.checkpoint,
            loadModel=self.load_model,
            loadFolderFile=self.load_folder_file,
            numItersForTrainExamplesHistory=self.n_iters_for_train_examples_history,
            maxlenOfQueue=self.maxlen_of_queue,
            load_folder_file=self.load_folder_file,
        )


args = MainArgs()


def main() -> None:
    log.info("Loading %s...", Game.__name__)
    g = Game(6)

    log.info("Loading %s...", nn.__name__)
    nnet = nn(g)

    if args.load_model:
        log.info(
            'Loading checkpoint "%s/%s"...',
            args.load_folder_file[0],
            args.load_folder_file[1],
        )
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning("Not loading a checkpoint!")

    log.info("Loading the Coach...")
    c = Coach(g, nnet, args.to_coach_args())

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.load_train_examples()

    log.info("Starting the learning process 🎉")
    c.learn()


if __name__ == "__main__":
    main()
