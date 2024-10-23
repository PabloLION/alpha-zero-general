import pytest

from alpha_zero_general.arena import Arena
from alpha_zero_general.connect4.connect4_game import Connect4Game
from alpha_zero_general.connect4.connect4_players import RandomPlayer


class TestArena:
    @pytest.fixture(autouse=True)
    def setup_method(self):
        self.game = Connect4Game()
        self.player1 = RandomPlayer(self.game).play
        self.player2 = RandomPlayer(self.game).play
        self.arena = Arena(self.player1, self.player2, self.game)

    def test_play_game(self):
        result = self.arena.play_game(verbose=False)
        assert result in [1, -1, 1e-4]

    def test_play_games(self):
        num_games = 10
        oneWon, twoWon, draws = self.arena.play_games(num_games, verbose=False)
        assert oneWon + twoWon + draws == num_games
        assert oneWon >= 0
        assert twoWon >= 0
        assert draws >= 0
