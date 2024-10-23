import os

import numpy as np
from flask import Flask, Response, request
from alpha_zero_general.dotsandboxes.dots_and_boxes_game import DotsAndBoxesGame
from alpha_zero_general.dotsandboxes.dots_and_boxes_players import GreedyRandomPlayer
from alpha_zero_general.dotsandboxes.keras.n_net import NNetWrapper as nn
from alpha_zero_general.MCTS import MCTS
from alpha_zero_general.utils import dotdict

app = Flask(__name__)

mcts = None
g = None


# curl -d "board=0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0" -X POST http://localhost:8888/predict
@app.route("/predict", methods=["POST"])
def predict():
    board = np.fromstring(request.form["board"], sep=",").reshape(g.getBoardSize())

    use_alpha_zero = True
    if use_alpha_zero:
        action = np.argmax(mcts.getActionProb(board, temp=0))
    else:
        action = GreedyRandomPlayer(g).play(board)

    resp = Response(str(action))
    # https://stackoverflow.com/questions/5584923/a-cors-post-request-works-from-plain-javascript-but-why-not-with-jquery
    # https://stackoverflow.com/questions/25860304/how-do-i-set-response-headers-in-flask
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp


if __name__ == "__main__":
    g = DotsAndBoxesGame(n=3)
    n1 = NNetWrapper(g)
    mcts = MCTS(g, n1, dotdict({"numMCTSSims": 50, "cpuct": 1.0}))
    n1.load_checkpoint(
        os.path.join("..", "pretrained_models", "dotsandboxes", "keras", "3x3"),
        "best.pth.tar",
    )
    app.run(debug=False, host="0.0.0.0", port=8888)
