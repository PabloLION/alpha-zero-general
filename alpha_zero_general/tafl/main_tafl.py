from alpha_zero_general.coach import Coach
from alpha_zero_general.tafl.pytorch.n_net import NNetWrapper as nn
from alpha_zero_general.tafl.tafl_game import TaflGame as Game
from alpha_zero_general.utils import dotdict

args = dotdict(
    {
        "numIters": 1000,
        "numEps": 100,
        "tempThreshold": 15,
        "updateThreshold": 0.6,
        "maxlenOfQueue": 200000,
        "numMCTSSims": 25,
        "arenaCompare": 40,
        "cpuct": 1,
        "checkpoint": "./temp/",
        "load_model": False,
        "load_folder_file": ("~/dev/models/8x100x50", "best.pth.tar"),
        "numItersForTrainExamplesHistory": 20,
    }
)

if __name__ == "__main__":
    g = Game(6)
    nnet = nn(g)

    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = Coach(g, nnet, args)
    if args.load_model:
        print("Load trainExamples from file")
        c.loadTrainExamples()
    c.learn()
