from __future__ import print_function

import sys
from typing import Any

import numpy as np

from alpha_zero_general.game import GenericGame
from alpha_zero_general.santorini import (
    SantoriniBoardTensor,
    SantoriniBooleanBoardTensor,
    SantoriniPolicyTensor,
)
from alpha_zero_general.santorini.santorini_logic import Board

sys.path.append("..")


class SantoriniGame(
    GenericGame[
        SantoriniBoardTensor, SantoriniBooleanBoardTensor, SantoriniPolicyTensor
    ]
):
    """
    Many of these functions are based on those from OthelloGame.py:
        https://github.com/suragnair/alpha-zero-general/blob/master/othello/OthelloGame.py
    """

    # #TODO/REF: think where to put this constant
    square_content = {-2: "Y", -1: "X", +0: "-", +1: "O", +2: "U"}

    # #TODO/REF: think where to put this constant
    # #TODO/REF: duped constant
    # fmt: off
    # NOTE THESE ARE NEITHER CCW NOR CW!
    __directions = list[tuple[int, int]](
        [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)])
    #         NW,      N,     NE,      W,     E,     SW,     S,    SE
    # fmt: on

    @staticmethod
    def get_square_piece(piece: int) -> str:
        return SantoriniGame.square_content[piece]

    def __init__(
        self, board_length: int = 5, true_random_placement: bool = False
    ) -> None:
        self.n = board_length

    def get_init_board(self) -> SantoriniBoardTensor:
        # return initial board (numpy board)
        b = Board(self.n)
        return np.array(b.pieces)

    def get_board_size(self) -> tuple[int, int, int]:
        # (dimension,a,b) tuple
        return (2, self.n, self.n)

    def get_action_size(self) -> int:
        # return number of actions
        return 128

    def get_next_state(
        self, board: SantoriniBoardTensor, player: int, action: int
    ) -> tuple[SantoriniBoardTensor, int]:
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
        action_move_idx = action // 8
        action_build_idx = action % 8

        assert 0 <= action_move_idx < 8, f"{action_move_idx=} should be less than 8"
        action_move: tuple[int, int] = self.__directions[action_move_idx]

        # #TODO/REF: maybe it's easier to just check action < 64.
        if action_build_idx >= 8:
            print("index error on action_move from directions")
            self.display(board)
            print("player: ", player)
            print("action: ", action)
            print("char_idx: ", char_idx)
            print("action_int: ", action_move_idx)
            print("action_build: ", action_build_idx)
            raise ValueError(
                f"{action_build_idx=} and {action_move_idx=} should be less than 8"
            )

        action_build = self.__directions[action_build_idx]

        char: tuple[int, int] = piece_locations[char_idx]

        action_move: tuple[int, int] = (
            action_move[0] + char[0],
            action_move[1] + char[1],
        )
        action_build: tuple[int, int] = (
            action_move[0] + action_build[0],
            action_build[1] + action_build[1],
        )
        new_action = [char, action_move, action_build]

        try:
            b.execute_move(new_action, player)
        except IndexError as e:
            print(e)
            self.display(board)
            #            print('l')
            #            print(l)
            print("player: ", player)
            print("action: ", action)
        return (b.pieces, -player)

    def get_valid_moves(
        self, board: SantoriniBoardTensor, player: int
    ) -> SantoriniBooleanBoardTensor:
        # return a fixed size binary vector
        # _, _, valids = board.get_all_moves
        b = Board(self.n)
        b.pieces = np.copy(board)
        color = player
        # valids = []
        return np.array(b.get_legal_moves_binary(color))
        # Get all the squares with pieces of the given color.

    def get_valid_moves_human(
        self, board: SantoriniBoardTensor, player: int
    ) -> tuple[list[Any], list[Any], list[int]]:
        b = Board(self.n)
        b.pieces = np.copy(board)
        color = player

        return b.get_all_moves(color)

    def get_character_locations(
        self, board: SantoriniBoardTensor, player: int
    ) -> tuple[tuple[int, int], tuple[int, int]]:
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

        return (char1_location, char2_location)

    def get_game_ended(self, board: SantoriniBoardTensor, player: int) -> float:
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

    def get_canonical_form(
        self, board: SantoriniBoardTensor, player: int
    ) -> SantoriniBoardTensor:
        # return state if player==1, else return -state if player==-1
        board = board * np.append(
            np.ones((1, self.n, self.n), dtype="int") * player,
            np.ones((1, self.n, self.n), dtype="int"),
            axis=0,
        )

        return board

    def get_random_board_symmetry(
        self, board: SantoriniBoardTensor
    ) -> SantoriniBoardTensor:
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

    def get_symmetries(
        self, board: SantoriniBoardTensor, pi: SantoriniPolicyTensor
    ) -> list[tuple[SantoriniBoardTensor, SantoriniPolicyTensor]]:
        # mirror, rotational

        assert len(pi) == 128  # each player has two pieces which can move in

        b = Board(self.n)
        b.pieces = np.copy(board)

        symmetries = list[tuple[SantoriniBoardTensor, SantoriniPolicyTensor]]()

        policy0 = pi[:64]
        policy1 = pi[64:]

        for n_rot in range(1, 5):
            for k in [True, False]:
                # rotate board:
                new_board0 = np.rot90(b.pieces[0], n_rot)
                new_board1 = np.rot90(b.pieces[1], n_rot)

                # rotate pi:
                new_policy0 = self.rotate(policy0)
                new_policy1 = self.rotate(policy1)

                # We will record var_, which may be modified by a flip, but
                # reinitialize the next rotation/flip with var to get all syms:
                # rotate, rotate+flip, rotate^2, rotate^2+flip,
                # rotate^3, rotate^3+flip, rotate^4=Identity, rotate^4+flip
                newB0_ = new_board0
                newB1_ = new_board1
                newPi0_ = new_policy0
                newPi1_ = new_policy1
                if k:
                    # flip board:
                    newB0_ = np.fliplr(new_board0)
                    newB1_ = np.fliplr(new_board1)

                    # flip pi:
                    newPi0_ = self.flip(policy0)
                    newPi1_ = self.flip(policy1)

                newPi = np.ravel([newPi0_, newPi1_])

                # record the symmetry
                symmetries.append((np.array([newB0_, newB1_]), list(newPi)))
                symmetries += [(np.array([newB0_, newB1_]), list(newPi))]

                # reset the board as the rotated one values with the new values
                b.pieces[0] = np.copy(new_board0)
                b.pieces[1] = np.copy(new_board1)
                policy0 = new_policy0
                policy1 = new_policy1

        return symmetries

    def rotate(self, pi_64: SantoriniPolicyTensor) -> SantoriniPolicyTensor:
        """
        Input: first XOR second half of Pi
        Returns: the half of pie in a reordered list that corresponds
                 to a counterclockwise rotation of the board
        """
        assert len(pi_64) == 64
        # fmt: off
        rotation_indices = [18,20,23,17,22,16,19,21,34,36,39,33,38,32,35,37,58,60,63,57,62,56,59,61,10,12,15,9,14,8,11,13,50,52,55,49,54,48,51,53,2,4,7,1,6,0,3,5,26,28,31,25,30,24,27,29,42,44,47,41,46,40,43,45]
        # fmt: on

        pi_new = [pi_64[i] for i in rotation_indices]

        return pi_new

    def flip(self, pi_64: SantoriniPolicyTensor) -> SantoriniPolicyTensor:
        """
        Input: first XOR second half of Pi
        Returns: the half of pie in a reordered list that corresponds
                 to a left<--->right flip of the board
        """
        assert len(pi_64) == 64

        # fmt: off
        flip_indices = [18,17,16,20,19,23,22,21,10,9,8,12,11,15,14,13,2,1,0,4,3,7,6,5,34,33,32,36,35,39,38,37,26,25,24,28,27,31,30,29,58,57,56,60,59,63,62,61,50,49,48,52,51,55,54,53,42,41,40,44,43,47,46,45]
        # fmt: on

        pi_new = [pi_64[i] for i in flip_indices]

        return pi_new

        #        # One counter clockwise rotation:
        #        l = []
        #        for i in range(8):
        #            l2 = []
        #            for k in range(8):
        #                l2.append(pi[i*8 + k])
        #            l3 = []
        #            l3 = [l2[i] for i in [2, 4, 7, 1, 6, 0, 3, 5]]
        #            l.append(l3)
        #        l_pi = [l[i] for i in [2, 4, 7, 1, 6, 0, 3, 5]]
        #
        #
        #        # One flip (mirror) left <-->> right
        #        l_flip
        #        for i in range(8):
        #            l2_flip = []
        #            for k in range(8):
        #                l2_flip.append(pi[i*8 + k])
        #            l3_flip = []
        #            l3_flip = [l2_flip[i] for i in [2, 1, 0, 4, 3, 7, 6, 5]]
        #            l_flip.append(l3_flip)
        #        l_pi_flip = [l_flip[i] for i in [2, 1, 0, 4, 3, 7, 6, 5]]
        #
        """
        split into 
        0-63, and 64-127. Here the letters a,...,h denote move locations
        
        These are the actions the first 64 values of pi correspond to doing
        
        0  1  2    8  9  10    16 17 18 
        3  a  4    11 b  12    19 c  20
        5  6  7    13 14 15    21 22 23
        
        24 25 26   original    32 33 34
        27 d  28    piece      35 e  36
        29 30 31   location    37 38 39
        
        40 41 42   48 49 50    56 57 58
        43 f  44   51 g  52    59 h  60
        45 46 47   53 54 55    61 62 63
        
        
        
        Initially we have: 
            
            a  b  c
            d     e
            f  g  h
        
        after CCW rotation we have: 
            
            c  e  h
            b     g
            a  d  f
        
        where for each move location a,..,h:
        
                                0  1  2    
        initially a is:         3  a  4    
                                5  6  7 
        
                                2  4  7
        after CCW rotation:     1  a  6
                                0  3  5
                                
        

                              
        initial values at indices: [0, 1, 2, 3, 4, 5, 6, 7]
                        --->[2, 4, 7, 1, 6, 0, 3, 5] under 1 CCW rotation
        
        
        
        For flips left <---> right:
               
        [0, 1, 2, 3, 4, 5, 6, 7]
    --->[2, 1, 0, 4, 3, 7, 6, 5] under 1 flip
        
        
        """

    def get_board_str(self, board: SantoriniBoardTensor) -> str:
        return np.array2string(board)

    def get_board_hash(self, board: SantoriniBoardTensor) -> int:
        return hash(board.tobytes())

    def string_representation_readable(self, board: SantoriniBoardTensor) -> str:
        # Do not think this works.
        board_s = "".join(
            self.square_content[square] for row in board for square in row
        )
        return board_s

    def get_score(self, board: SantoriniBoardTensor, player: int) -> int:
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
    def display(board: SantoriniBoardTensor) -> None:
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
