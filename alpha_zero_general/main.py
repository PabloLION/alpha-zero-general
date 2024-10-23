import logging
from dataclasses import dataclass

import coloredlogs  # type: ignore

from alpha_zero_general.coach import Coach
from alpha_zero_general.othello.othello_game import OthelloGame as Game
from alpha_zero_general.othello.pytorch.n_net import NNetWrapper as nn

log = logging.getLogger(__name__)

coloredlogs.install(level="INFO")  # Change this to DEBUG to see more info.


@dataclass
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
    c = Coach(g, nnet, args)

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info("Starting the learning process 🎉")
    c.learn()


if __name__ == "__main__":
    main()
