from typing import Any

import numpy as np

from alpha_zero_general.tafl.game_variants import TaflGameVariant


class TaflBoard:
    size: int
    width: int
    height: int
    board: list[list[int]]  # TODO: ndarray might be better? #TODO: ren
    pieces: list[list[int]]  # TODO: ndarray might be better?
    time: int
    done: int

    def __init__(self, gv: TaflGameVariant):
        """
        gv is a game variant object
        """
        self.size = gv.size
        self.width = gv.size
        self.height = gv.size
        self.board = gv.board  # [x,y,type] #TODO/TYPE: type better
        self.pieces = gv.pieces  # [x,y,type] #TODO/TYPE: type better
        self.time = 0
        self.done = 0

    def __str__(self):
        return str(self.get_player_to_move()) + "".join(
            str(r) for v in self.get_image() for r in v
        )

    # add [][] indexer syntax to the Board
    def __getitem__(self, index: int):
        return np.array(self.get_image())[index]

    def astype(self, t: np.dtype[Any]):
        return np.array(self.get_image()).astype(t)

    def get_copy(self):
        gv = TaflGameVariant()
        gv.size = self.size
        gv.board = np.copy(np.array(self.board)).tolist()
        gv.pieces = np.copy(np.array(self.pieces)).tolist()
        b = TaflBoard(gv)
        b.time = self.time
        b.done = self.done
        return b

    def count_diff(self, color: int) -> int:
        """Counts the # pieces of the given color
        (1 for white, -1 for black, 0 for empty spaces)"""
        count = 0
        for p in self.pieces:
            if p[0] >= 0:
                if p[2] * color > 0:
                    count += 1
                else:
                    count -= 1
        return count

    def get_legal_moves(self, color: int):
        """Returns all the legal moves for the given color.
        (1 for white, -1 for black
        """
        return self._get_valid_moves(color)

    def has_legal_moves(self, color: int) -> bool:
        vm = self._get_valid_moves(color)
        if len(vm) > 0:
            return True
        return False

    def execute_move(self, move: list[int], color: int):
        """Perform the given move on the board.
        color gives the color pf the piece to play (1=white,-1=black)
        """
        x1, y1, x2, y2 = move
        piece_num = self._get_piece_no(x1, y1)
        legal = self._is_legal_move(piece_num, x2, y2)
        if legal >= 0:
            # print("Accepted move: ",move)
            self._move_by_piece_no(piece_num, x2, y2)
        # else:
        # print("Illegal move:",move,legal)

    def get_image(self):
        image = [[0 for col in range(self.width)] for row in range(self.height)]
        for item in self.board:
            image[item[1]][item[0]] = item[2] * 10
        for piece in self.pieces:
            if piece[0] >= 0:
                image[piece[1]][piece[0]] = piece[2] + image[piece[1]][piece[0]]
        return image

    def get_player_to_move(self):
        return -(self.time % 2 * 2 - 1)

    ################## Internal methods ##################

    def _is_legal_move(self, piece_num: int, x2: int, y2: int):
        try:
            if x2 < 0 or y2 < 0 or x2 >= self.width or y2 > self.height:
                return -1

            piece = self.pieces[piece_num]
            x1 = piece[0]
            y1 = piece[1]
            if x1 < 0:
                return -2  # piece was captured
            if x1 != x2 and y1 != y2:
                return -3  # must move in straight line
            if x1 == x2 and y1 == y2:
                return -4  # no move

            piece_type = piece[2]
            if (piece_type == -1 and self.time % 2 == 0) or (
                piece_type != -1 and self.time % 2 == 1
            ):
                return -5  # wrong player

            for item in self.board:
                if item[0] == x2 and item[1] == y2 and item[2] > 0:
                    if piece_type != 2:
                        return -10  # forbidden space
            for a_piece in self.pieces:
                if (
                    y1 == y2
                    and y1 == a_piece[1]
                    and (
                        (x1 < a_piece[0] and x2 >= a_piece[0])
                        or (x1 > a_piece[0] and x2 <= a_piece[0])
                    )
                ):
                    return -20  # interposing piece
                if (
                    x1 == x2
                    and x1 == a_piece[0]
                    and (
                        (y1 < a_piece[1] and y2 >= a_piece[1])
                        or (y1 > a_piece[1] and y2 <= a_piece[1])
                    )
                ):
                    return -20  # interposing piece

            return 0  # legal move
        except Exception as ex:
            print("error in islegalmove ", ex, piece_num, x2, y2)
            raise

    def _get_captures(self, piece_num: int, x2: int, y2: int):
        # Assumes was already checked for legal move
        captures = []
        piece = self.pieces[piece_num]
        piece_type = piece[2]
        for a_piece in self.pieces:
            if piece_type * a_piece[2] < 0:
                d1 = a_piece[0] - x2
                d2 = a_piece[1] - y2
                if (abs(d1) == 1 and d2 == 0) or (abs(d2) == 1 and d1 == 0):
                    for b_piece in self.pieces:
                        if piece_type * b_piece[2] > 0 and not (
                            piece[0] == b_piece[0] and piece[1] == b_piece[1]
                        ):
                            e1 = b_piece[0] - a_piece[0]
                            e2 = b_piece[1] - a_piece[1]
                            if d1 == e1 and d2 == e2:
                                captures.append(a_piece)
        return captures

    # returns code for invalid mode (<0) or number of pieces captured
    def _move_by_piece_no(self, piece_num: int, x2: int, y2: int):
        legal = self._is_legal_move(piece_num, x2, y2)
        if legal != 0:
            return legal

        self.time = self.time + 1

        piece = self.pieces[piece_num]
        piece[0] = x2
        piece[1] = y2
        caps = self._get_captures(piece_num, x2, y2)
        # print("Captures = ",caps)
        for c in caps:
            c[0] = -99

        self.done = self._get_win_lose()

        return len(caps)

    def _get_win_lose(self):
        if self.time > 50:
            return -1
        for a_piece in self.pieces:
            if a_piece[2] == 2 and a_piece[0] > -1:
                for item in self.board:
                    if item[0] == a_piece[0] and item[1] == a_piece[1] and item[2] == 1:
                        return 1  # white won
                return 0  # no winner
        return -1  # white lost

    def _get_piece_no(self, x: int, y: int) -> int:
        for piece_num in range(len(self.pieces)):
            piece = self.pieces[piece_num]
            if piece[0] == x and piece[1] == y:
                return piece_num
        return -1

    def _get_valid_moves(self, player: int) -> list[tuple[int, int, int, int]]:
        moves = list[tuple[int, int, int, int]]()
        for piece_num in range(len(self.pieces)):
            piece = self.pieces[piece_num]
            if piece[2] * player <= 0:
                continue
            # print("checking piece_num ",piece_num,piece)
            for x in range(0, self.width):
                if self._is_legal_move(piece_num, x, piece[1]) >= 0:
                    moves.append((piece[0], piece[1], x, piece[1]))
            for y in range(0, self.height):
                if self._is_legal_move(piece_num, piece[0], y) >= 0:
                    moves.append((piece[0], piece[1], piece[0], y))
        # print("moves ",moves)
        return moves
