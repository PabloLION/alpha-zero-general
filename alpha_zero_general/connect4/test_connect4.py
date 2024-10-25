"""
To run tests:
pytest-3 connect4
"""

import textwrap
from typing import NamedTuple, TypeAlias

import numpy as np

from alpha_zero_general.connect4.connect4_game import Connect4BoardTensor, Connect4Game

Move: TypeAlias = int
Moves: TypeAlias = list[Move]

# Tuple of (Board, Player, Game) to simplify testing.
BPGTuple = NamedTuple(
    "BPGTuple",
    [("board", Connect4BoardTensor), ("player", int), ("game", Connect4Game)],
)


def init_board_from_moves(
    moves: Moves, height: int | None = None, width: int | None = None
) -> BPGTuple:
    """
    Arg:
        moves: List of moves to make.
        height: Height of board. If None, default height is used.
        width: Width of board. If None, default width is used.

    Return:
        BPGTuple based on series of specified moved.
    """
    if height is None:
        game = Connect4Game()
    elif width is None:
        game = Connect4Game(height=height)
    else:
        game = Connect4Game(height=height, width=width)

    board, player = game.get_init_board(), 1
    for move in moves:
        board, player = game.get_next_state(board, player, move)
    return BPGTuple(board, player, game)


def init_board_from_array(board: Connect4BoardTensor, player: int) -> BPGTuple:
    """Returns a BPGTuple based on series of specified moved."""
    game = Connect4Game(height=len(board), width=len(board[0]))
    return BPGTuple(board, player, game)


def test_simple_moves():
    board, _player, game = init_board_from_moves([4, 5, 4, 3, 0, 6])
    expected = textwrap.dedent(
        """\
        [[ 0.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  1.  0.  0.]
         [ 1.  0.  0. -1.  1. -1. -1.]]"""
    )
    assert expected == game.get_board_str(board)


def test_overfull_column():
    for height in range(1, 10):
        # Fill to max height is ok
        init_board_from_moves([4] * height, height=height)

        # Check overfilling causes an error.
        try:
            init_board_from_moves([4] * (height + 1), height=height)
            assert False, "Expected error when overfilling column"
        except ValueError:
            pass  # Expected.


def test_get_valid_moves():
    """Tests vector of valid moved is correct."""
    move_valid_pairs: list[tuple[Moves, list[bool]]] = [
        ([], [True] * 7),
        ([0, 1, 2, 3, 4, 5, 6], [True] * 7),
        ([0, 1, 2, 3, 4, 5, 6] * 5, [True] * 7),
        ([0, 1, 2, 3, 4, 5, 6] * 6, [False] * 7),
        ([0, 1, 2] * 3 + [3, 4, 5, 6] * 6, [True] * 3 + [False] * 4),
    ]

    for moves, expected_valid in move_valid_pairs:
        board, player, game = init_board_from_moves(moves)
        assert (np.array(expected_valid) == game.get_valid_moves(board, player)).all()


def test_symmetries():
    """Tests symetric board are produced."""
    board, _player, game = init_board_from_moves([0, 0, 1, 0, 6])
    pi = [0.1, 0.2, 0.3]
    (board1, pi1), (board2, pi2) = game.get_symmetries(board, np.array(pi))
    assert [0.1, 0.2, 0.3] == pi1 and [0.3, 0.2, 0.1] == pi2

    expected_board1 = textwrap.dedent(
        """\
        [[ 0.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.]
         [-1.  0.  0.  0.  0.  0.  0.]
         [-1.  0.  0.  0.  0.  0.  0.]
         [ 1.  1.  0.  0.  0.  0.  1.]]"""
    )
    assert expected_board1 == game.get_board_str(board1)

    expected_board2 = textwrap.dedent(
        """\
        [[ 0.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0. -1.]
         [ 0.  0.  0.  0.  0.  0. -1.]
         [ 1.  0.  0.  0.  0.  1.  1.]]"""
    )
    assert expected_board2 == game.get_board_str(board2)


def test_game_ended():
    """Tests game end detection logic based on fixed boards."""
    array_end_state_pairs = [
        (
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                ]
            ),
            1,
            0,
        ),
        (
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                ]
            ),
            1,
            1,
        ),
        (
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                ]
            ),
            -1,
            -1,
        ),
        (
            np.array(
                [
                    [0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0, 0],
                ]
            ),
            -1,
            -1,
        ),
        (np.array([[0, 0, 0, -1], [0, 0, -1, 0], [0, -1, 0, 0], [-1, 0, 0, 0]]), 1, -1),
        (
            np.array(
                [[0, 0, 0, 0, 1], [0, 0, 0, 1, 0], [0, 0, 1, 0, 0], [0, 1, 0, 0, 0]]
            ),
            -1,
            -1,
        ),
        (
            np.array(
                [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0]]
            ),
            -1,
            -1,
        ),
        (
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, -1, 0, 0, 0],
                    [0, 0, 0, -1, 0, 0, 1],
                    [0, 0, 0, 1, 1, -1, -1],
                    [0, 0, 0, -1, 1, 1, 1],
                    [0, -1, 0, -1, 1, -1, 1],
                ]
            ),
            -1,
            0,
        ),
        (
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0],
                    [-1.0, -1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0, -1.0, 0.0, 0.0, 0.0],
                    [1.0, -1.0, 1.0, -1.0, 0.0, -1.0, 0.0],
                ]
            ),
            -1,
            -1,
        ),
        (
            np.array(
                [
                    [
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        -1.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    [
                        0.0,
                        0.0,
                        1.0,
                        1.0,
                        -1.0,
                        0.0,
                        -1.0,
                    ],
                    [
                        0.0,
                        0.0,
                        -1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                    ],
                    [
                        -1.0,
                        0.0,
                        -1.0,
                        1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                    ],
                ]
            ),
            1,
            1,
        ),
    ]

    for np_pieces, player, expected_end_state in array_end_state_pairs:
        board, player, game = init_board_from_array(np_pieces, player)
        end_state = game.get_game_ended(board, player)
        assert expected_end_state == end_state, "expected=%s, actual=%s, board=\n%s" % (
            expected_end_state,
            end_state,
            board,
        )


def test_immutable_move():
    """Test original board is not mutated when get_next_state() called."""
    board, _player, game = init_board_from_moves([1, 2, 3, 3, 4])
    original_board_string = game.get_board_str(board)

    new_np_pieces, _new_player = game.get_next_state(board, 3, -1)

    assert original_board_string == game.get_board_str(board)
    assert original_board_string != game.get_board_str(new_np_pieces)
