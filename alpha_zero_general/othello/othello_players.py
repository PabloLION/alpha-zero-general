import subprocess

import numpy as np

from alpha_zero_general.othello import OthelloBoardTensor
from alpha_zero_general.othello.othello_game import OthelloGame


class RandomPlayer:
    def __init__(self, game: OthelloGame):
        self.game = game

    def play(self, board: OthelloBoardTensor):
        a = np.random.randint(self.game.get_action_size())
        valid_moves = self.game.get_valid_moves(board, 1)
        while valid_moves[a] != 1:
            a = np.random.randint(self.game.get_action_size())
        return a


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


class GTPOthelloPlayer:
    """
    Player that plays with Othello programs using the Go Text Protocol.
    """

    _process: None | subprocess.Popen[bytes]

    # The colours are reversed as the Othello programs seems to have the board setup with the opposite colours
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

    def end_game(self):
        """
        Should be called after the game ends in order to clean-up the used resources.
        """
        if self._process is not None:
            self._send_command("quit")
            # Waits for the client to terminate gracefully for 10 seconds. If it does not - kills it.
            try:
                self._process.wait(10)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None

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

    def _send_command(self, cmd: str) -> str:
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
