import subprocess
from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar, cast

from alpha_zero_general import BoardTensor, BooleanBoard, PolicyTensor
from alpha_zero_general.game import GenericGame
from alpha_zero_general.othello import OthelloBoardTensor
from alpha_zero_general.othello.othello_game import OthelloGame


class Player(ABC, Generic[BoardTensor, BooleanBoard, PolicyTensor]):
    game: GenericGame[BoardTensor, BooleanBoard, PolicyTensor]

    @abstractmethod
    def __init__(
        self, game: GenericGame[BoardTensor, BooleanBoard, PolicyTensor]
    ) -> None:
        """
        Input:
            game: the game instance
        """
        raise NotImplementedError(
            "The __init__ method must be implemented by the subclass"
        )

    @abstractmethod
    def play(self, board: BoardTensor) -> int:
        """
        Input:
            board: current board
        Returns:
            action: the action to be taken
        """
        raise NotImplementedError("play must be implemented by the subclass")

    def check_and_play(self, board: BoardTensor) -> int:
        """
        Input:
            board: current board
        Returns:
            action: the action to be taken
        """
        action = self.play(board)
        # self.game.check_action(board, action)
        return action


class InteractivePlayer(Protocol): ...


class HumanOthelloPlayer:
    game: OthelloGame

    def __init__(self, game: OthelloGame) -> None:
        self.game = game

    def play(self, board: OthelloBoardTensor) -> int:
        # display(board)
        valid = self.game.get_valid_moves(board, 1)
        for i in range(len(valid)):
            if valid[i]:
                print("[", int(i / self.game.n), int(i % self.game.n), end="] ")
        while True:
            input_move = input()
            input_a = input_move.split(" ")
            if len(input_a) == 2:
                try:
                    x, y = [int(i) for i in input_a]
                    if (
                        (0 <= x)
                        and (x < self.game.n)
                        and (0 <= y)
                        and (y < self.game.n)
                    ) or ((x == self.game.n) and (y == 0)):
                        a = self.game.n * x + y if x != -1 else self.game.n**2
                        if valid[a]:
                            break
                except ValueError:
                    # Input needs to be an integer
                    "Invalid integer"
            print("Invalid move")
        return a


class GreedyOthelloPlayer:
    game: OthelloGame

    def __init__(self, game: OthelloGame) -> None:
        self.game = game

    def play(self, board: OthelloBoardTensor) -> int:
        valid_moves = self.game.get_valid_moves(board, 1)
        candidates = list[tuple[int, int]]()
        for a in range(self.game.get_action_size()):
            if valid_moves[a] == 0:
                continue
            next_board, _ = self.game.get_next_state(board, 1, a)
            score = self.game.get_score(next_board, 1)
            candidates += [(-score, a)]
        candidates.sort()
        return candidates[0][1]


F = TypeVar("F", bound=Callable[..., Any])


class GTPOthelloPlayer:
    """
    Player that plays with Othello programs using the Go Text Protocol.
    """

    _process: None | subprocess.Popen[bytes]

    # The colors are reversed as the Othello programs seems to have the board setup with the opposite colours
    player_colors = {
        -1: "white",
        1: "black",
    }

    def __init__(self, game: OthelloGame, gtp_client: list[str]):
        """
        Input:
            game: the game instance
            gtp_client: list with the command line arguments to start the GTP client with.
                        The first argument should be the absolute path to the executable.
        """
        self.game = game
        self.gtp_client = gtp_client
        self._process = None

    @staticmethod
    def _require_subprocess(func: F) -> F:
        """
        A decorator to ensure that the subprocess is running before calling the decorated method.
        """

        @wraps(func)
        def wrapper(self: "GTPOthelloPlayer", *args: Any, **kwargs: Any) -> Any:
            if self._process is None:
                raise RuntimeError("The subprocess is not running.")
            return func(*args, **kwargs)

        return cast(F, wrapper)

    def start_game(self):
        """
        Should be called before the game starts in order to setup the board.
        """
        self._current_player = 1  # Arena does not notify players about their colour so we need to keep track here
        self._process = subprocess.Popen(
            self.gtp_client, bufsize=0, stdin=subprocess.PIPE, stdout=subprocess.PIPE
        )
        self._send_command("boardsize " + str(self.game.n))
        self._send_command("clear_board")

    @_require_subprocess
    def end_game(self):
        """
        Should be called after the game ends in order to clean-up the used resources.
        """
        if TYPE_CHECKING:
            assert self._process  # checked by @_require_subprocess

        self._send_command("quit")
        # Waits for the client to terminate gracefully for 10 seconds. If it does not - kills it.
        try:
            self._process.wait(10)
        except subprocess.TimeoutExpired:
            self._process.kill()
        self._process = None  # type: ignore

    def notify(self, board: OthelloBoardTensor, action: int) -> None:
        """
        Should be called after the opponent turn. This way we can update the GTP client with the opponent move.
        """
        color = GTPOthelloPlayer.player_colors[self._current_player]
        move = self._convert_action_to_move(action)
        self._send_command("play {} {}".format(color, move))
        self._switch_players()

    def play(self, board: OthelloBoardTensor) -> int:
        color = GTPOthelloPlayer.player_colors[self._current_player]
        move = self._send_command("genmove {}".format(color))
        action = self._convert_move_to_action(move)
        self._switch_players()
        return action

    def _switch_players(self):
        self._current_player = -self._current_player

    def _convert_action_to_move(self, action: int) -> str:
        if action < self.game.n**2:
            row, col = int(action / self.game.n), int(action % self.game.n)
            return "{}{}".format(chr(ord("A") + col), row + 1)
        else:
            return "PASS"

    def _convert_move_to_action(self, move: str) -> int:
        if move != "PASS":
            col, row = ord(move[0]) - ord("A"), int(move[1:])
            return (row - 1) * self.game.n + col
        else:
            return self.game.n**2

    @_require_subprocess
    def _send_command(self, cmd: str) -> str:
        if TYPE_CHECKING:
            assert self._process  # checked by @_require_subprocess

        if not self._process.stdin:
            raise RuntimeError("The subprocess has no stdin.")
        if not self._process.stdout:
            raise RuntimeError("The subprocess has no stdout.")

        self._process.stdin.write(cmd.encode() + b"\n")

        response = ""
        while True:
            line = self._process.stdout.readline().decode()
            if line == "\n":
                if response:
                    break  # Empty line means end of the response is reached
                else:
                    continue  # Ignoring leading empty lines
            response += line

        # If the first character of the response is '=', then is success. '?' is error.
        if response.startswith("="):
            # Some clients return uppercase other lower case.
            # Normalizing to uppercase in order to simplify handling.
            return response[1:].strip().upper()
        else:
            raise subprocess.SubprocessError(
                "Error calling GTP client: {}".format(response[1:].strip())
            )

    def __call__(self, board: OthelloBoardTensor) -> int:
        return self.play(board)
