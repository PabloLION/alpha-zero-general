# https://stackoverflow.com/questions/2267362/how-to-convert-an-integer-in-any-base-to-a-string

import string
from typing import no_type_check

from alpha_zero_general.py313_backport import deprecated

digs = string.digits + string.ascii_letters


@deprecated
@no_type_check
def old_int2base(x: int, base: int, length: int) -> list[int]:
    if x < 0:
        sign = -1
    elif x == 0:
        return digs[0]
    else:
        sign = 1

    x *= sign
    digits = list[int]()

    while x:
        digits.append(digs[int(x % base)])
        x = int(x / base)

    if sign < 0:
        digits.append("-")

    while len(digits) < length:
        digits.extend(["0"])

    return list(map(lambda x: int(x), digits))


def int2base(x: int, base: int, length: int) -> list[int]:
    assert x >= 0
    digits = list[int]()
    while x:
        digits.append(x % base)
        x = x // base
    while len(digits) < length:
        digits.append(0)
    return digits


def test():
    size = 7
    valid_moves = [
        [3, 0, 1, 0],
        [3, 0, 2, 0],
        [3, 0, 4, 0],
        [3, 0, 5, 0],
        [3, 1, 0, 1],
        [3, 1, 1, 1],
        [3, 1, 2, 1],
        [3, 1, 4, 1],
        [3, 1, 5, 1],
        [3, 1, 6, 1],
        [0, 3, 0, 1],
        [0, 3, 0, 2],
        [0, 3, 0, 4],
        [0, 3, 0, 5],
        [1, 3, 1, 0],
        [1, 3, 1, 1],
        [1, 3, 1, 2],
        [1, 3, 1, 4],
        [1, 3, 1, 5],
        [1, 3, 1, 6],
        [3, 6, 1, 6],
        [3, 6, 2, 6],
        [3, 6, 4, 6],
        [3, 6, 5, 6],
        [3, 5, 0, 5],
        [3, 5, 1, 5],
        [3, 5, 2, 5],
        [3, 5, 4, 5],
        [3, 5, 5, 5],
        [3, 5, 6, 5],
        [6, 3, 6, 1],
        [6, 3, 6, 2],
        [6, 3, 6, 4],
        [6, 3, 6, 5],
        [5, 3, 5, 0],
        [5, 3, 5, 1],
        [5, 3, 5, 2],
        [5, 3, 5, 4],
        [5, 3, 5, 5],
        [5, 3, 5, 6],
    ]
    print(valid_moves)
    for m in valid_moves:
        i = m[0] + m[1] * size + m[2] * size**2 + m[3] * size**3
        print(i, ":", int2base(i, size, 4))


if __name__ == "__main__":
    test()
