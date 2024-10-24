import logging
from dataclasses import dataclass

import coloredlogs  # type: ignore

from alpha_zero_general.coach import Coach, CoachArgs
from alpha_zero_general.othello.othello_game import OthelloGame
from alpha_zero_general.othello.pytorch.n_net import NNetWrapper

log = logging.getLogger(__name__)

coloredlogs.install(  # type: ignore
    level="INFO"
)  # Change this to DEBUG to see more info.


@dataclass(frozen=True)  # freeze to check for immutability in refactor
class MainArgs:
    num_iter: int = 1000
    num_eps: int = 100
    temp_threshold: int = 15
    update_threshold: float = 0.6
    max_len_of_queue: int = 200000
    n_mcts_sims: int = 25
    arena_compare: int = 40
    c_puct: int = 1
    checkpoint: str = "./temp/"
    load_model: bool = False
    load_folder_file: tuple[str, str] = ("/dev/models/8x100x50", "best.pth.tar")
    num_iters_for_train_examples_history: int = 20

    def to_coach_args(self) -> CoachArgs:
        return CoachArgs(
            num_iters=self.num_iter,
            num_eps=self.num_eps,
            temp_threshold=self.temp_threshold,
            update_threshold=self.update_threshold,
            max_len_of_queue=self.max_len_of_queue,
            num_mcts_sims=self.n_mcts_sims,
            arena_compare=self.arena_compare,
            c_puct=self.c_puct,
            checkpoint=self.checkpoint,
            load_model=self.load_model,
            load_folder_file=self.load_folder_file,
            num_iters_for_train_examples_history=self.num_iters_for_train_examples_history,
        )


args = MainArgs()


def main() -> None:
    log.info("Loading %s...", OthelloGame.__name__)
    g = OthelloGame(6)

    log.info("Loading %s...", NNetWrapper.__name__)
    nnet = NNetWrapper(g)

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
        log.info("Loading 'train_examples' from file...")
        c.load_train_examples()

    log.info("Starting the learning process 🎉")
    c.learn()


if __name__ == "__main__":
    main()
