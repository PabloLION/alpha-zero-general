from alpha_zero_general.coach import Coach
from alpha_zero_general.main import MainArgs
from alpha_zero_general.tafl.pytorch.n_net import NNetWrapper as nn
from alpha_zero_general.tafl.tafl_game import TaflGame as Game

args = MainArgs(
    num_iter=1000,
    num_eps=100,
    temp_threshold=15,
    update_threshold=0.6,
    max_len_of_queue=200000,
    n_mcts_sims=25,
    arena_compare=40,
    c_puct=1,
    checkpoint="./temp/",
    load_model=False,
    load_folder_file=("~/dev/models/8x100x50", "best.pth.tar"),
    num_iters_for_train_examples_history=20,
)

if __name__ == "__main__":
    g = Game(6)
    nnet = nn(g)

    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = Coach(g, nnet, args)
    if args.load_model:
        print("Load train_examples from file")
        c.loadTrainExamples()
    c.learn()
