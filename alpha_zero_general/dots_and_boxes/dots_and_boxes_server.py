import os
from typing import cast

import numpy as np
from flask import Flask, Response, request

from alpha_zero_general import MctsArgs
from alpha_zero_general.dots_and_boxes.dots_and_boxes_game import (
    DotsAndBoxesBoardTensor,
    DotsAndBoxesBooleanBoardTensor,
    DotsAndBoxesGame,
    DotsAndBoxesPolicyTensor,
)
from alpha_zero_general.dots_and_boxes.dots_and_boxes_players import GreedyRandomPlayer
from alpha_zero_general.dots_and_boxes.keras.n_net import DotsAndBoxesNNInterface
from alpha_zero_general.mcts import MCTS

USE_ALPHA_ZERO = True

app = Flask(__name__)

# #TODO: is it one mcts per game? and one game per server?
mcts: MCTS[
    DotsAndBoxesBoardTensor, DotsAndBoxesBooleanBoardTensor, DotsAndBoxesPolicyTensor
]
game: DotsAndBoxesGame


# curl -d "board=0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0" -X POST http://localhost:8888/predict
@app.route("/predict", methods=["POST"])
def predict():
    board = cast(
        DotsAndBoxesBoardTensor,
        np.fromstring(request.form["board"], sep=",").reshape(game.get_board_size()),
    )

    if USE_ALPHA_ZERO:
        action = int(np.argmax(mcts.get_action_probabilities(board, temperature=0)))
    else:
        action = GreedyRandomPlayer(game).play(board)

    resp = Response(str(action))
    # https://stackoverflow.com/questions/5584923/a-cors-post-request-works-from-plain-javascript-but-why-not-with-jquery
    # https://stackoverflow.com/questions/25860304/how-do-i-set-response-headers-in-flask
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp


if __name__ == "__main__":
    game = DotsAndBoxesGame(n=3)
    nn1 = DotsAndBoxesNNInterface(game)
    mcts = MCTS(game, nn1, MctsArgs(num_mcts_sims=50, c_puct=1.0))
    nn1.load_checkpoint(
        os.path.join("..", "pretrained_models", "dotsandboxes", "keras", "3x3"),
        "best.pth.tar",
    )
    app.run(debug=False, host="0.0.0.0", port=8888)
