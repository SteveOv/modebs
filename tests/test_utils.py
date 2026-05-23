""" Unit tests for the utils module. """
import unittest
import numpy as np

from libs.utils import to_file_safe_str
from libs.utils import grouper
from libs.utils import partitions_slices

# pylint: disable=too-many-public-methods, line-too-long
class Testutils(unittest.TestCase):
    """ Unit tests for the utils module. """

    #
    #   to_file_safe_str(text: str, replacement: str="-", lower: bool=True) -> str
    #
    def test_to_file_safe_str_happy_path(self):
        """ Simple happy path tests of to_file_safe_str() """
        for (input_str,         replacement,    lower,      exp_output) in [
            ("Hello World",     "-",            False,      "Hello-World"),
            ("Hello World",     "_",            False,      "Hello_World"),
            ("Hello World",     "-",            True,       "hello-world"),
            ("Hello-World",     "_",            True,       "hello-world"),
            ("no replacement",  "",             False,      "noreplacement"),
            ("./^with:colon$",  "-",            True,       ".--with-colon-"),
        ]:
            with self.subTest(f"to_file_safe_str('{input_str}', '{replacement}', {lower}) -> '{exp_output}'"):
                self.assertEqual(exp_output, to_file_safe_str(input_str, replacement, lower))


    #
    #   grouper(iterable: Iterable, size: int, fillvalue: any=None) -> zip_longest
    #
    def test_grouper_happy_path(self):
        """ Simple happy path tests of grouper() """
        for (iterable,      size,       fillvalue,      exp_first) in [
            (range(100),    5,          None,           (0, 1, 2, 3, 4)),
            (range(5),      5,          None,           (0, 1, 2, 3, 4)),
            ([0,1,2,3,4],   5,          None,           (0, 1, 2, 3, 4)),
            (range(3),      5,          None,           (0, 1, 2, None, None)),
            (range(3),      5,          -1,             (0, 1, 2, -1, -1)),
        ]:
            with self.subTest(f"next(grouper({iterable}, size={size}, fillvalue={fillvalue})) -> {exp_first}"):
                self.assertEqual(exp_first, next(grouper(iterable, size, fillvalue)))

        for (iterable,      size,       exp_sum) in [
            (range(100),    10,         100),
            (range(105),    10,         110),
            (range(0),      10,         0),
            ([],            10,         0),
        ]:
            with self.subTest(f"grouper({iterable}, size={size}) -> {exp_sum} items"):
                self.assertEqual(exp_sum, np.sum([len(g) for g in grouper(iterable, size)]))

    #
    #   partitions_slices(sequence_len: int, min_slice_len: int=1, max_slice_len: int=None,
    #                     exclude_ixs: _Iterable[int]=None, offset: int=0) -> Generator[Tuple[slice]]
    #
    def test_partitions_slices_happy_path(self):
        """ Simple happy path tests of grouper() """
        for (sequence_len,  min_sl_len, max_sl_len, offset, exp_slices_lists) in [
            (3,             1,          None,       0,      [[slice(0,3)], [slice(0,1),slice(1,3)], [slice(0,2),slice(2,3)], [slice(0,1),slice(1,2),slice(2,3)]]),
            # Min slice lengths
            (4,             2,          None,       0,      [[slice(0,4)], [slice(0,2),slice(2,4)]]),
            (3,             4,          None,       0,      []),
            # Max slice lengths
            (3,             1,          1,          0,      [[slice(0,1),slice(1,2),slice(2,3)]]),
            (3,             1,          4,          0,      [[slice(0,3)], [slice(0,1),slice(1,3)], [slice(0,2),slice(2,3)], [slice(0,1),slice(1,2),slice(2,3)]]),
            # Offsets
            (3,             1,          None,       1,      [[slice(1,4)], [slice(1,2),slice(2,4)], [slice(1,3),slice(3,4)], [slice(1,2),slice(2,3),slice(3,4)]]),
            # Not sure why you would want to do this, but it behaves sensibly/as expected
            (3,             1,          None,      -1,      [[slice(-1,2)], [slice(-1,0),slice(0,2)], [slice(-1,1),slice(1,2)], [slice(-1,0),slice(0,1),slice(1,2)]]),
        ]:
            with self.subTest(f"partitions_slices({sequence_len}, {min_sl_len}, {max_sl_len}, offset={offset})"):
                slices_lists = list(partitions_slices(sequence_len, min_sl_len, max_sl_len, offset=offset))
                # print("\nslices_lists=", slices_lists)

                self.assertEqual(len(exp_slices_lists), len(slices_lists))
                for slices in slices_lists:
                    self.assertIn(slices, exp_slices_lists)

    def test_partitions_slices_with_exclusions(self):
        """ Simple happy path tests of grouper() """
        for (seq_len,   min_sl_len, max_sl_len, excl_ixs,   offset, exp_slices_lists) in [
            (3,         1,          None,       [],         0,      [[slice(0,3)], [slice(0,1),slice(1,3)], [slice(0,2),slice(2,3)], [slice(0,1),slice(1,2),slice(2,3)]]),
            (3,         1,          None,       None,       0,      [[slice(0,3)], [slice(0,1),slice(1,3)], [slice(0,2),slice(2,3)], [slice(0,1),slice(1,2),slice(2,3)]]),
            # Simple exclusions
            (4,         1,          None,       [3],        0,      [[slice(0,3)], [slice(0,1),slice(1,3)], [slice(0,2),slice(2,3)], [slice(0,1),slice(1,2),slice(2,3)]]),
            (4,         1,          None,       [2],        0,      [[slice(0,2),slice(3,4)], [slice(0,1),slice(1,2),slice(3,4)]]),
            (4,         1,          None,       2,          0,      [[slice(0,2),slice(3,4)], [slice(0,1),slice(1,2),slice(3,4)]]),
            (5,         1,          None,       [1,3],      0,      [[slice(0,1),slice(2,3),slice(4,5)]]),
            (5,         1,          None,       [2,3],      0,      [[slice(0,2),slice(4,5)], [slice(0,1),slice(1,2),slice(4,5)]]),
            # Handles exclusion duplication, out of order or out of range
            (5,         1,          None,       [2,3,3],    0,      [[slice(0,2),slice(4,5)], [slice(0,1),slice(1,2),slice(4,5)]]),
            (5,         1,          None,       [3,2],      0,      [[slice(0,2),slice(4,5)], [slice(0,1),slice(1,2),slice(4,5)]]),
            (3,         1,          None,       [-1,4],     0,      [[slice(0,3)], [slice(0,1),slice(1,3)], [slice(0,2),slice(2,3)], [slice(0,1),slice(1,2),slice(2,3)]]),
            # Exclusions with min len
            (5,         2,          None,       [2],        0,      [[slice(0,2),slice(3,5)]]),
            # Exclusions with max len
            (5,         1,          1,          [2],        0,      [[slice(0,1),slice(1,2),slice(3,4),slice(4,5)]]),
            # Exclusion with offset (exclude_ixs are wrt to the output range so are expected to have the offset built in)
            (4,         1,          None,       [3],        1,      [[slice(1,3),slice(4,5)], [slice(1,2),slice(2,3),slice(4,5)]]),
        ]:
            with self.subTest(f"partitions_slices({seq_len}, {min_sl_len}, {max_sl_len}, exclude_ixs={excl_ixs}, offset={offset})"):
                slices_lists = list(partitions_slices(seq_len, min_sl_len, max_sl_len, excl_ixs, offset))
                # print("\nslices_lists=", slices_lists)

                self.assertEqual(len(exp_slices_lists), len(slices_lists))
                for slices in slices_lists:
                    self.assertIn(slices, exp_slices_lists)
