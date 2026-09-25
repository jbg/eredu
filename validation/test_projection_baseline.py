"""Behavioral checks for isolation and completeness of native trace attribution."""
import unittest
from projection_baseline import kernel_selections


class KernelSelections(unittest.TestCase):
    def test_only_synchronized_phase_selections_are_attributed(self):
        trace = """[mlx-metal-kernel] unrelated_setup
[projection-phase-begin] case-1 8 native_mixed 0
[mlx-metal-kernel] copy_bfloat_float
[mlx-metal-kernel] steel_splitk_float
[mlx-metal-kernel] splitk_accumulate
[projection-phase-end] case-1 8 native_mixed 0
[mlx-metal-kernel] unrelated_cleanup
[projection-phase-begin] case-1 8 native_mixed 1
[mlx-metal-kernel] steel_splitk_float
[projection-phase-end] case-1 8 native_mixed 1
"""
        self.assertEqual(dict(kernel_selections(trace)[("case-1", "8", "native_mixed")]),
                         {"copy_bfloat_float": 1, "steel_splitk_float": 2, "splitk_accumulate": 1})

    def test_missing_native_evidence_or_broken_phase_boundaries_fail(self):
        for trace in ["", "[projection-phase-begin] case-0 2 cast 0\n",
                      "[projection-phase-end] case-0 2 cast 0\n",
                      "[projection-phase-begin] case-0 2 cast 0\n[projection-phase-end] case-0 2 cast 0\n",
                      "[projection-phase-begin] case-0 2 cast 0\n[projection-phase-begin] case-1 2 cast 0\n"]:
            with self.subTest(trace=trace), self.assertRaises(ValueError):
                kernel_selections(trace)


if __name__ == "__main__":
    unittest.main()
