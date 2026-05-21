""" General purpose utility/helper functions """
from typing import Iterable  as _Iterable
from itertools import zip_longest as _zip_longest, chain as _chain, combinations as _combinations
import re as _re

_to_file_safe_sub_pattern = _re.compile(r"[^\w\d._-]", _re.IGNORECASE)

def to_file_safe_str(text: str, replacement: str="-", lower: bool=True) -> str:
    """
    Parse the text and replace any potentially troublesome characters when used as a file name.
    Do no pass in a full path as / and \\ are among the characters which will be replaced.

    :text: the original text
    :replacement: the character to substitute for any troublesome characters
    :lower: whether or not to force the revised text to lower case [True]
    :returns: the revised text
    """
    retval = _to_file_safe_sub_pattern.sub(replacement, text)
    return retval.lower() if lower else retval


def grouper(iterable: _Iterable, size: int, fillvalue=None):
    """
    Iterates over iterable, yielding the contents in groups (tuples) of the requested size.

    From https://docs.python.org/3/library/itertools.html#itertools-recipes

    :iterable: the iterable to iterate in groups
    :size: the size of each group
    :fillvalue: used to fill out the final group when there are insufficient items in the iterable
    :returns: tuples, of the requested size, of values from iterator 
    """
    batchable = [iter(iterable)] * size
    return _zip_longest(*batchable, fillvalue=fillvalue)


def powerset(iterable: _Iterable, min_len: int=0):
    """
    Yields every possible subset of the source sequence. 

    i.e.: powerset([1,2,3]) -> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    
    From https://docs.python.org/3/library/itertools.html#itertools-recipes
    with a modification to specify the minimum length of the subsets.

    :iterable: the iterable to yield from
    :min_size: the minimum length of the subsets to yield
    """
    seq = list(iterable)
    min_len = max(0, min_len)
    return _chain.from_iterable(_combinations(seq, r) for r in range(min_len, len(seq)+1))


def partitions_slices(sequence_len: int, min_slice_len: int=1, max_slice_len: int=None):
    """
    Yields all possible order-preserving lists of slices onto a sequence of the given length.

    i.e.: partitions_slices(3) -> [[0:3]] [[0:1],[1:3]] [[0:2],[2:3]] [[0:1],[1:2],[2:3]]

    Based on
    https://more-itertools.readthedocs.io/en/stable/_modules/more_itertools/more.html#partitions
    with a modifications to yield slices (so it doesn't need to see the sequence, with only the
    length required) and to restrict the minimum & maximum length of any slices.
    """
    max_slice_len = min(sequence_len, max_slice_len or sequence_len)
    for ix in powerset(range(1, sequence_len)):
        slices = [slice(i, j) for i, j in zip((0,) + ix, ix + (sequence_len,))]
        if all(min_slice_len <= sl.stop - sl.start <= max_slice_len for sl in slices):
            yield slices
