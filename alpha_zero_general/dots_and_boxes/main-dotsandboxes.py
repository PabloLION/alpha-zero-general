import logging

import coloredlogs

from alpha_zero_general.coach import Coach
from alpha_zero_general.dots_and_boxes.dots_and_boxes_game import DotsAndBoxesGame
from alpha_zero_general.dots_and_boxes.keras.n_net import NNetWrapper as nn
from alpha_zero_general.main import MainArgs

log = logging.getLogger(__name__)

coloredlogs.install(level="INFO")  # Change this to DEBUG to see more info.
args = MainArgs(
    num_iter=100,
    num_eps=25,  # Number of complete self-play games to simulate during a new iteration.
    temp_threshold=15,  #
    update_threshold=0.6,  # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    max_len_of_queue=200000,  # Number of game examples to train the neural networks.
    n_mcts_sims=25,  # Number of games moves for MCTS to simulate.
    arena_compare=40,  # Number of games to play during arena play to determine if new net will be accepted.
    c_puct=1,
    checkpoint="./temp/",
    load_model=False,
    load_folder_file=("/dev/models/8x100x50", "best.pth.tar"),
    num_iters_for_train_examples_history=20,
)

prod_args = MainArgs(
    num_iter=1000,
    num_eps=100,  # Number of complete self-play games to simulate during a new iteration.
    temp_threshold=15,  #
    update_threshold=0.6,  # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    max_len_of_queue=200000,  # Number of game examples to train the neural networks.
    n_mcts_sims=25,  # Number of games moves for MCTS to simulate.
    arena_compare=40,  # Number of games to play during arena play to determine if new net will be accepted.
    c_puct=1,
    checkpoint="./temp/",
    load_model=False,
    load_folder_file=("/dev/models/8x100x50", "best.pth.tar"),
    num_iters_for_train_examples_history=20,
)


def main():
    log.info("Loading %s...", DotsAndBoxesGame.__name__)
    g = DotsAndBoxesGame(n=3)
    log.info("Loading %s...", nn.__name__)
    nnet = nn(g)
    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning("Not loading a checkpoint!")
    log.info("Loading the Coach...")
    c = Coach(g, nnet, args)
    log.info("Starting the learning process 🎉")
    c.learn()


if __name__ == "__main__":
    main()
