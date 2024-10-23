from alpha_zero_general.Coach import Coach
from alpha_zero_general.rts.keras.n_net import NNetWrapper as nn
from alpha_zero_general.rts.rts_game import RTSGame as Game
from alpha_zero_general.rts.src.config_class import CONFIG

"""
rts/learn.py

Teaches neural network playing of specified game configuration using self play
This configuration needs to be kept seperate, as different nnet and game configs are set
"""

if __name__ == "__main__":
    CONFIG.set_runner("learn")  # set visibility as learn

    # create nnet for this game
    g = Game()
    nnet = nn(g, CONFIG.nnet_args.encoder)

    # If training examples should be loaded from file
    if CONFIG.learn_args.load_model:
        nnet.load_checkpoint(
            CONFIG.learn_args.load_folder_file[0], CONFIG.learn_args.load_folder_file[1]
        )

    # Create coach instance that starts teaching nnet on newly created game using self-play
    c = Coach(g, nnet, CONFIG.learn_args)
    if CONFIG.learn_args.load_model:
        print("Load trainExamples from file")
        c.loadTrainExamples()
    c.learn()
