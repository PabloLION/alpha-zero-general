import pytest
from pytest_mock import mocker
from alpha_zero_general.Coach import Coach
from alpha_zero_general.connect4.Connect4Game import Connect4Game
from alpha_zero_general.connect4.keras.NNet import NNetWrapper as nn
from alpha_zero_general.utils import dotdict


class TestCoach:
    @pytest.fixture(autouse=True)
    def setup_method(self):
        self.game = Connect4Game()
        self.nnet = nn(self.game)
        self.args = dotdict(
            {
                "numIters": 1,
                "numEps": 1,
                "tempThreshold": 1,
                "updateThreshold": 0.6,
                "maxlenOfQueue": 200,
                "numMCTSSims": 1,
                "arenaCompare": 1,
                "cpuct": 1,
                "checkpoint": "./temp/",
                "load_model": False,
                "load_folder_file": ("/dev/models/8x100x50", "best.pth.tar"),
                "numItersForTrainExamplesHistory": 1,
            }
        )
        self.coach = Coach(self.game, self.nnet, self.args)

    def test_executeEpisode(self, mocker):
        self.coach.mcts.getActionProb = mocker.MagicMock(return_value=[1])
        self.coach.game.getNextState = mocker.MagicMock(
            return_value=(self.game.getInitBoard(), 1)
        )
        self.coach.game.getGameEnded = mocker.MagicMock(return_value=1)
        trainExamples = self.coach.executeEpisode()
        assert len(trainExamples) == 1
        assert trainExamples[0][2] == 1

    def test_learn(self, mocker):
        self.coach.executeEpisode = mocker.MagicMock(
            return_value=[(self.game.getInitBoard(), 1, [1], 1)]
        )
        self.coach.nnet.train = mocker.MagicMock()
        self.coach.nnet.save_checkpoint = mocker.MagicMock()
        self.coach.nnet.load_checkpoint = mocker.MagicMock()
        self.coach.learn()
        self.coach.nnet.train.assert_called()
        self.coach.nnet.save_checkpoint.assert_called()
        self.coach.nnet.load_checkpoint.assert_called()
