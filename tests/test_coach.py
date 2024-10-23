import pytest
from pytest_mock import MockerFixture

from alpha_zero_general.coach import Coach, CoachArgs
from alpha_zero_general.connect4.connect4_game import Connect4Game
from alpha_zero_general.connect4.keras.n_net import NNetWrapper as nn


class TestCoach:
    @pytest.fixture(autouse=True)
    def setup_method(self):
        self.game = Connect4Game()
        self.nnet = nn(self.game)
        self.args = CoachArgs(
            num_iters=1,
            num_eps=1,
            temp_threshold=1,
            update_threshold=0.6,
            max_len_of_queue=200,
            num_mcts_sims=1,
            arena_compare=1,
            c_puct=1,
            checkpoint="./temp/",
            load_model=False,
            load_folder_file=("/dev/models/8x100x50", "best.pth.tar"),
            num_iters_for_train_examples_history=1,
        )
        self.coach = Coach(self.game, self.nnet, self.args)

    def test_execute_episode(self, mocker: MockerFixture):
        self.coach.mcts.get_action_prob = mocker.MagicMock(return_value=[1])
        self.coach.game.get_next_state = mocker.MagicMock(
            return_value=(self.game.get_init_board(), 1)
        )
        self.coach.game.get_game_ended = mocker.MagicMock(return_value=1)
        train_examples = self.coach.execute_episode()
        assert len(train_examples) == 1
        assert train_examples[0][2] == 1

    def test_learn(self, mocker: MockerFixture):
        self.coach.execute_episode = mocker.MagicMock(
            return_value=[(self.game.get_init_board(), 1, [1], 1)]
        )
        self.coach.nnet.train = mocker.MagicMock()
        self.coach.nnet.save_checkpoint = mocker.MagicMock()
        self.coach.nnet.load_checkpoint = mocker.MagicMock()
        self.coach.learn()
        self.coach.nnet.train.assert_called()
        self.coach.nnet.save_checkpoint.assert_called()
        self.coach.nnet.load_checkpoint.assert_called()
