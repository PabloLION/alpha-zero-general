import logging
from typing import Generic

from tqdm import tqdm

from alpha_zero_general import Display, PolicyMakerAsPlayer
from alpha_zero_general.coach import BoardTensor, BooleanBoard, PolicyTensor
from alpha_zero_general.game import GenericGame

log = logging.getLogger(__name__)


class Arena(Generic[BoardTensor, BooleanBoard, PolicyTensor]):
    """
    An Arena class where any 2 agents can be pit against each other.
    """

    def __init__(
        self,
        player1: PolicyMakerAsPlayer[BoardTensor],
        player2: PolicyMakerAsPlayer[BoardTensor],
        game: GenericGame[BoardTensor, BooleanBoard, PolicyTensor],
        display: Display | None = None,
    ):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                      display in othello/OthelloGame). Is necessary for verbose
                      mode.

        see othello/OthelloPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """
        self.player1 = player1
        self.player2 = player2
        self.game: GenericGame[BoardTensor, BooleanBoard, PolicyTensor] = game
        self.display = display

    def play_game(self, verbose: bool = False) -> float:
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        players: list[PolicyMakerAsPlayer[BoardTensor]] = [
            self.player1,
            None,
            self.player2,
        ]
        current_player = 1
        board = self.game.get_init_board()
        it = 0

        for player in players[0], players[2]:
            if hasattr(player, "start_game"):
                player.start_game()

        while self.game.get_game_ended(board, current_player) == 0:
            it += 1
            if verbose:
                assert self.display
                print("Turn ", str(it), "Player ", str(current_player))
                self.display(board)
            action: int = players[current_player + 1](
                self.game.get_canonical_form(board, current_player)
            )

            valid_moves = self.game.get_valid_moves(
                self.game.get_canonical_form(board, current_player), 1
            )

            if valid_moves[action] == 0:
                log.error(f"Action {action} is not valid!")
                log.debug(msg=f"valids = {valid_moves}")
                assert (
                    valid_moves[action] > 0
                ), f"Action {action} is not valid in {valid_moves}"

            # Notifying the opponent for the move
            opponent = players[-current_player + 1]
            if hasattr(opponent, "notify"):
                opponent.notify(board, action)

            board, current_player = self.game.get_next_state(
                board, current_player, action
            )

        for player in players[0], players[2]:
            if hasattr(player, "end_game"):
                player.end_game()

        if verbose:
            assert self.display
            print(
                "Game over: Turn ",
                str(it),
                "Result ",
                str(self.game.get_game_ended(board, 1)),
            )
            self.display(board)
        return current_player * self.game.get_game_ended(board, current_player)

    def play_games(self, num: int, verbose: bool = False) -> tuple[int, int, int]:
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            one_won: games won by player1
            two_won: games won by player2
            draws:  games won by nobody
        """

        num = int(num / 2)
        one_won = 0
        two_won = 0
        draws = 0
        for _ in tqdm(range(num), desc="Arena.play_games (1)"):
            game_result = self.play_game(verbose=verbose)
            if game_result == 1:
                one_won += 1
            elif game_result == -1:
                two_won += 1
            else:
                draws += 1

        self.player1, self.player2 = self.player2, self.player1

        for _ in tqdm(range(num), desc="Arena.play_games (2)"):
            game_result = self.play_game(verbose=verbose)
            if game_result == -1:
                one_won += 1
            elif game_result == 1:
                two_won += 1
            else:
                draws += 1

        return one_won, two_won, draws
