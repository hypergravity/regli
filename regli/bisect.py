__all__ = ["find_neighboring_indices"]

import bisect
from typing import Iterable


def find_neighboring_indices(seq: Iterable, a: int) -> tuple[int, int]:
    r"""Find neighboring indices in a sequence for value ``a`` .

    Prefer left interval if on grid points.

    Parameters
    ----------
    seq: Iterable
        Sequence data.
    a: int
        Target value.

    Returns
    -------
    tuple[int, int]
        The left and right indices.
    """
    l = len(seq)
    if seq[0] < a < seq[-1]:
        i_l = bisect.bisect_left(seq, a)
        i_r = bisect.bisect_right(seq, a)
        if i_l < i_r:
            # exactly on grid points
            return i_l - 1, i_r - 1
        else:
            # between grid points
            return i_l - 1, i_r
    # left most
    if a == seq[0]:
        return 0, 1
    # right most
    if a == seq[-1]:
        return l - 2, l - 1
    # out of bounds
    return -1, -1


def _f(a):
    seq = [1, 2, 3, 4]
    print(f"{a} \t", find_neighboring_indices(seq, a))


def _f1(a):
    seq = [1, 2, 3, 4]
    print(f"{a} \t", bisect.bisect_left(seq, a), bisect.bisect_right(seq, a))


if __name__ == "__main__":
    _f(0.999)
    _f(1)
    _f(1.5)
    _f(2)
    _f(2.5)
    _f(3)
    _f(3.5)
    _f(4)
    _f(4.5)
    f1(5)
