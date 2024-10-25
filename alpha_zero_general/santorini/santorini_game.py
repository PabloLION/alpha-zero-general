from __future__ import print_function

import sys
from typing import Any, List, Tuple

import numpy as np

from alpha_zero_general import GenericBoardTensor, GenericBooleanBoardTensor, GenericPolicyTensor
from alpha_zero_general.game import GenericGame
from alpha_zero_general.santorini.santorini_logic import Board

sys.path.append("..")


class SantoriniGame(GenericGame[GenericBoardTensor, GenericBooleanBoardTensor, GenericPolicyTensor]):
    """
    Many of these functions are based on those from OthelloGame.py:
        https://github.com/suragnair/alpha-zero-general/blob/master/othello/OthelloGame.py
    """

    square_content = {-2: "Y", -1: "X", +0: "-", +1: "O", +2: "U"}

    # NOTE THESE ARE NEITHER CCW NOR CW!
    __directions = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]

    @staticmethod
    def get_square_piece(piece: int) -> str:
        return SantoriniGame.square_content[piece]

    def __init__(self, board_length: int = 5, true_random_placement: bool = False) -> None:
        self.n = board_length

    def get_init_board(self) -> GenericBoardTensor:
        # return initial board (numpy board)
        b = Board(self.n)
        return np.array(b.pieces)

    def get_board_size(self) -> Tuple[int, int, int]:
        # (dimension,a,b) tuple
        return (2, self.n, self.n)

    def get_action_size(self) -> int:
        # return number of actions
        return 128

    def get_next_state(self, board: GenericBoardTensor, player: int, action: int) -> Tuple[GenericBoardTensor, int]:
        # if player takes action on board, return next (board,player)
        # action must be a valid move

        piece_locations = self.get_character_locations(board, player)

        b = Board(self.n)
        b.pieces = np.copy(board)

        # color = player
        if action > 63:
            char_idx = 1  # character is 2
            action = action % 64
        else:
            char_idx = 0  # character is 1
        action_move = action // 8
        action_build = action % 8

        try:
            action_move = self.__directions[action_move]
        except IndexError as e:
            print(e)
            print("index error on action_move from directions")
            self.display(board)
            print("player: ", player)
            print("action: ", action)
            print("char_idx: ", char_idx)
            print("action_move: ", action_move)
            print("action_build: ", action_build)
        action_build = self.__directions[action_build]

        char = piece_locations[char_idx]

        action_move = (action_move[0] + char[0], action_move[1] + char[1])
        action_build = (
            action_move[0] + action_build[0],
            action_build[1] + action_build[1],
        )
        action = [char, action_move, action_build]

        try:
            b.execute_move(action, player)
        except IndexError as e:
            print(e)
            self.display(board)
            #            print('l')
            #            print(l)
            print("player: ", player)
            print("action: ", action)
        return (b.pieces, -player)

    def get_valid_moves(self, board: GenericBoardTensor, player: int) -> GenericBooleanBoardTensor:
        # return a fixed size binary vector
        # _, _, valids = board.get_all_moves
        b = Board(self.n)
        b.pieces = np.copy(board)
        color = player
        # valids = []
        return np.array(b.get_legal_moves_binary(color))
        # Get all the squares with pieces of the given color.

    def get_valid_moves_human(self, board: GenericBoardTensor, player: int) -> Tuple[List[Any], List[Any], List[int]]:
        b = Board(self.n)
        b.pieces = np.copy(board)
        color = player

        return b.get_all_moves(color)

    def get_character_locations(self, board: GenericBoardTensor, player: int) -> List[Tuple[int, int]]:
        """
        Returns a list of both character's locations as tuples for the player
        """
        b = Board(self.n)
        b.pieces = np.copy(board)

        color = player

        # Get all the squares with pieces of the given color.
        char1_location = np.where(b.pieces[0] == 1 * color)
        char1_location = (char1_location[0][0], char1_location[1][0])

        char2_location = np.where(b.pieces[0] == 2 * color)
        char2_location = (char2_location[0][0], char2_location[1][0])

        return [char1_location, char2_location]

    def get_game_ended(self, board: GenericBoardTensor, player: int) -> float:
        """
        Assumes player is about to move. THIS IS NOT COMPATIBLE with the prior implementation of Arena.py
        which returned self.game.get_game_ended(board, 1).
        Input:
            board: current board
            player: current player (1 or -1)
        Returns:
            r: 0 if game has not ended. 1 if THIS player has won, -1 if player THIS lost,
               small non-zero value for draw.
        """

        b = Board(self.n)
        b.pieces = np.copy(board)
        player_pieces = self.get_character_locations(b.pieces, player)
        opponent_pieces = self.get_character_locations(b.pieces, -1 * player)

        for piece in player_pieces:
            if b.pieces[1][piece] == 3:
                return 1

        for piece in opponent_pieces:
            if b.pieces[1][piece] == 3:
                return -1
        if not b.has_legal_moves(player):
            return -1
        return 0

    def get_canonical_form(self, board: GenericBoardTensor, player: int) -> GenericBoardTensor:
        # return state if player==1, else return -state if player==-1
        board = board * np.append(
            np.ones((1, self.n, self.n), dtype="int") * player,
            np.ones((1, self.n, self.n), dtype="int"),
            axis=0,
        )

        return board

    def get_random_board_symmetry(self, board: GenericBoardTensor) -> GenericBoardTensor:
        """
        Returns a random board symmetry.
        """
        b = Board(self.n)
        b.pieces = np.copy(board)
        i = np.random.randint(0, 4)
        k = np.random.choice([True, False])
        newB0 = np.rot90(b.pieces[0], i)
        newB1 = np.rot90(b.pieces[1], i)
        if k:
            newB0 = np.fliplr(newB0)
            newB1 = np.fliplr(newB1)

        return np.array([newB0, newB1])

    def get_symmetries(self, board: GenericBoardTensor, pi: GenericPolicyTensor) -> List[Tuple[GenericBoardTensor, GenericPolicyTensor]]:
        # mirror, rotational

        assert len(pi) == 128  # each player has two pieces which can move in

        b = Board(self.n)
        b.pieces = np.copy(board)

        syms = []

        Pi0 = pi[:64]
        Pi1 = pi[64:]

        for i in range(1, 5):
            for k in [True, False]:
                # rotate board:
                newB0 = np.rot90(b.pieces[0], 1)
                newB1 = np.rot90(b.pieces[1], 1)

                # rotate pi:
                newPi0 = self.rotate(Pi0)
                newPi1 = self.rotate(Pi1)

                # We will record  var_, which may be modified by a flip, but
                # reinitialize the next rotation/flip with var to get all syms:
                # rotate, rotate+flip, rotate^2, rotate^2+flip,
                # rotate^3, rotate^3+flip, rotate^4=Identity, rotate^4+flip
                newB0_ = newB0
                newB1_ = newB1
                newPi0_ = newPi0
                newPi1_ = newPi1
                if k:
                    # flip board:
                    newB0_ = np.fliplr(newB0)
                    newB1_ = np.fliplr(newB1)

                    # flip pi:
                    newPi0_ = self.flip(Pi0)
                    newPi1_ = self.flip(Pi1)

                newPi = np.ravel([newPi0_, newPi1_])

                # record the symmetry
                syms += [(np.array([newB0_, newB1_]), list(newPi))]

                # reset the board as the rotated one values with the new values
                b.pieces[0] = np.copy(newB0)
                b.pieces[1] = np.copy(newB1)
                Pi0 = newPi0
                Pi1 = newPi1

        return syms

    def rotate(self, pi_64: GenericPolicyTensor) -> GenericPolicyTensor:
        """
        Input: first XOR second half of Pi
        Returns: the half of pie in a reordered list that corresponds
                 to a counterclockwise rotation of the board
        """
        assert len(pi_64) == 64

        rotation_indices = [
            18,
            20,
            23,
            17,
            22,
            16,
            19,
            21,
            34,
            36,
            39,
            33,
            38,
            32,
            35,
            37,
            58,
            60,
            63,
            57,
            62,
            56,
            59,
            61,
            10,
            12,
            15,
            9,
            14,
            8,
            11,
            13,
            50,
            52,
            55,
            49,
            54,
            48,
            51,
            53,
            2,
            4,
            7,
            1,
            6,
            0,
            3,
            5,
            26,
            28,
            31,
            25,
            30,
            24,
            27,
            29,
            42,
            44,
            47,
            41,
            46,
            40,
            43,
            45,
        ]

        pi_new = [pi_64[i] for i in rotation_indices]

        return pi_new

    def flip(self, pi_64: GenericPolicyTensor) -> GenericPolicyTensor:
        """
        Input: first XOR second half of Pi
        Returns: the half of pie in a reordered list that corresponds
                 to a left<--->right flip of the board
        """
        assert len(pi_64) == 64

        flip_indices = [
            18,
            17,
            16,
            20,
            19,
            23,
            22,
            21,
            10,
            9,
            8,
            12,
            11,
            15,
            14,
            13,
            2,
            1,
            0,
            4,
            3,
            7,
            6,
            5,
            34,
            33,
            32,
            36,
            35,
            39,
            38,
            37,
            26,
            25,
            24,
            28,
            27,
            31,
            30,
            29,
            58,
            57,
            56,
            60,
            59,
            63,
            62,
            61,
            50,
            49,
            48,
            52,
            51,
            55,
            54,
            53,
            42,
            41,
            40,
            44,
            43,
            47,
            46,
            45,
        ]

        pi_new = [pi_64[i] for i in flip_indices]

        return pi_new

    def get_board_str(self, board: GenericBoardTensor) -> str:
        return np.array2string(board)

    def get_board_hash(self, board: GenericBoardTensor) -> int:
        return hash(board.tobytes())

    def string_representation_readable(self, board: GenericBoardTensor) -> str:
        # Do not think this works.
        board_s = "".join(
            self.square_content[square] for row in board for square in row
        )
        return board_s

    def get_score(self, board: GenericBoardTensor, player: int) -> int:
        """
        Only used by 'Greedy player'
        """

        b = Board(self.n)
        b.pieces = np.copy(board)

        piece_locations = self.get_character_locations(board, player)
        char0 = piece_locations[0]
        char1 = piece_locations[1]

        opponent_piece_locations = self.get_character_locations(board, -player)
        opp_char0 = opponent_piece_locations[0]
        opp_char1 = opponent_piece_locations[1]

        player_score = max(b.pieces[1][char0], b.pieces[1][char1])
        opponent_score = max(b.pieces[1][opp_char0], b.pieces[1][opp_char1])
        if player_score == 3:
            # this is a winning move, set score very high
            player_score = 100
        if opponent_score == 3:
            # this is a winning move, set score very high
            opponent_score = 100
        score = player_score - opponent_score
        # height of highest piece for player:
        return score

    @staticmethod
    def display(board: GenericBoardTensor) -> None:
        n = board.shape[1]
        print("   ", end="")
        for y in range(n):
            print(y, end=" ")
        print("")
        print("-----------------------")
        for y in range(n):
            print(y, "|", end="")  # print the row #
            for x in range(n):
                piece = board[0][y][x]  # get the piece to print
                print(SantoriniGame.square_content[piece], end=" ")
            print("|")

        print("-----------------------")
        print("   ", end="")
        for y in range(n):
            print(y, end=" ")
        print("")
        print("-----------------------")
        for y in range(n):
            print(y, "|", end="")  # print the row #
            for x in range(n):
                piece = board[1][y][x]  # get the piece to print
                print(piece, end=" ")
            print("|")

        print("-----------------------")
