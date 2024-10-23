import os

import numpy as np
from flask import Flask, Response, request

from alpha_zero_general import MctsArgs
from alpha_zero_general.dots_and_boxes.dots_and_boxes_game import DotsAndBoxesGame
from alpha_zero_general.dots_and_boxes.dots_and_boxes_players import GreedyRandomPlayer
from alpha_zero_general.dots_and_boxes.keras.n_net import NNetWrapper
from alpha_zero_general.mcts import MCTS

app = Flask(__name__)

# #TODO: is it one mcts per game? and one game per server?
mcts = None
game = None


# curl -d "board=0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0" -X POST http://localhost:8888/predict
@app.route("/predict", methods=["POST"])
def predict():
    board = np.fromstring(request.form["board"], sep=",").reshape(game.get_board_size())

    use_alpha_zero = True
    if use_alpha_zero:
        action = np.argmax(mcts.get_action_prob(board, temp=0))
    else:
        action = GreedyRandomPlayer(game).play(board)

    resp = Response(str(action))
    # https://stackoverflow.com/questions/5584923/a-cors-post-request-works-from-plain-javascript-but-why-not-with-jquery
    # https://stackoverflow.com/questions/25860304/how-do-i-set-response-headers-in-flask
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp


if __name__ == "__main__":
    game = DotsAndBoxesGame(n=3)
    n1 = NNetWrapper(game)
    mcts = MCTS(game, n1, MctsArgs(num_mcts_sims=50, c_puct=1.0))
    n1.load_checkpoint(
        os.path.join("..", "pretrained_models", "dotsandboxes", "keras", "3x3"),
        "best.pth.tar",
    )
    app.run(debug=False, host="0.0.0.0", port=8888)
