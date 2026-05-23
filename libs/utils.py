""" General purpose utility/helper functions """
from typing import Iterable  as _Iterable
from numbers import Number as _Number
from itertools import zip_longest as _zip_longest, chain as _chain
from itertools import combinations as _combinations, product as _product
import re as _re

import numpy as _np

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


def partitions_slices(sequence_len: int, min_slice_len: int=1, max_slice_len: int=None,
                      exclude_ixs: _Iterable[int]=None, offset: int=0):
    """
    Yields all possible order-preserving lists of slices onto a sequence of the given length.

    i.e.: partitions_slices(3) -> [[0:3]] [[0:1],[1:3]] [[0:2],[2:3]] [[0:1],[1:2],[2:3]]

    The exclusions break up the sequence, yielding the Cartesian product of the sub-sequences

    i.e.: partitions_slices(4, exclude_ixs=[1]) -> [[0:1],[2:4]] [[0:1],[2:3],[3:4]]

    The offset is used to shift all slices up or down

    i.e.: partitions_slices(3, offset=1) -> [[1:4]] [[1:2],[2:4]] [[1:3],[3:4]] [[1:2],[2:3],[3:4]]
    
    Based on
    https://more-itertools.readthedocs.io/en/stable/_modules/more_itertools/more.html#partitions
    with modifications to yield slices (so it doesn't need to see the sequence, only the length),
    to restrict the minimum & maximum length of any slices, and to handle exclusions and offsets.

    :sequence_len: the length of the sequence to produce slices over
    :min_slice_len: the minimum length of slices to produce
    :max_slice_len: the maximimum length of slices to produce
    :exclude_ixs: optional list of indices to "work around", effectively breaking up the sequence
    into sub-sequences. These are in the range of the output slices, so should include any offset
    :offset: optional offset to shift all of the slices up or down
    :returns: Generator of tuples of slices; one per unique combination; order is not guaranteed
    """
    max_slice_len = min(sequence_len, max_slice_len or sequence_len)
    if exclude_ixs is None:
        exclude_ixs = []
    elif isinstance(exclude_ixs, _Number):
        exclude_ixs = [exclude_ixs]

    if len(exclude_ixs) == 0:
        # Simple, just produce partitions across the entire sequence
        for ix in powerset(range(1+offset, sequence_len+offset)):
            slices = [slice(i, j) for i, j in zip((offset,) + ix, ix + (sequence_len+offset,))]
            if all(min_slice_len <= sl.stop - sl.start <= max_slice_len for sl in slices):
                yield slices
    else:
        # We have to handle exclusions, so there are sub-sequences separated by the exclusions.
        # Need to produce the product of the partitions of each of the sub-sequences.

        # Exclusions sorted, de-duped & de-offset (into generated range before offset is re-applied)
        exclude_ixs = sorted(ix-offset for ix in set(exclude_ixs) if 0 <= ix-offset <= sequence_len)

        # With exclusions we want the size of each sub sequence (diff-1) and offsets excl+1
        sub_seq_zipper = zip(
            _np.diff([-1] + exclude_ixs + [sequence_len]) - 1,      # ss_len
            _np.array([-1] + exclude_ixs) + 1 + offset              # ss_offset
        )

        for tuple_of_slices in _product(*tuple(
            partitions_slices(ss_len, min_slice_len, max_slice_len, offset=ss_offset)
                                        for ss_len, ss_offset in sub_seq_zipper if ss_len > 0
        )):
            # Yield each as a unified list so it matches the form when there are no exclusions
            yield [sl for sls in tuple_of_slices for sl in sls]
