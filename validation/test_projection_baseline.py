"""Behavioral checks for isolation and completeness of native trace attribution."""
import unittest
from projection_baseline import kernel_selections, validate_prototype_phases, compare_reference_case


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


class PrototypeEvidence(unittest.TestCase):
    def phases(self):
        return [
            {"phase": "preconverted_gemm", "peak_growth_bytes": 4096,
             "kernel_selections": {"steel_gemm_float32_bm16": 3, "splitk_accum": 3}},
            {"phase": "mixed_storage_prototype", "peak_growth_bytes": 4096,
             "kernel_selections": {"steel_gemm_float32_storage_bfloat16_bm16": 3, "splitk_accum": 3}},
        ]

    def test_storage_only_specialization_is_accepted(self):
        validate_prototype_phases(self.phases())

    def test_extra_copy_changed_geometry_missing_reduction_or_extra_workspace_fail(self):
        for mutation in ["copy", "geometry", "reduction", "allocation"]:
            phases = self.phases()
            candidate = phases[1]
            if mutation == "copy":
                candidate["kernel_selections"]["vn_copybfloat16float32"] = 3
            elif mutation == "geometry":
                del candidate["kernel_selections"]["steel_gemm_float32_storage_bfloat16_bm16"]
                candidate["kernel_selections"]["steel_gemm_float32_storage_bfloat16_bm32"] = 3
            elif mutation == "reduction":
                del candidate["kernel_selections"]["splitk_accum"]
            else:
                candidate["peak_growth_bytes"] += 64 * 1024 * 1024
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_prototype_phases(phases)


class OriginalReference(unittest.TestCase):
    def test_changed_fingerprint_or_generation_is_rejected(self):
        from copy import deepcopy
        reference = {"classes": [{"rows": 128}], "ordinary": [{"tokens": [1, 2]}],
                     "replays": [{"label": "case-0", "rows": 128, "output_f32_bits_sha256": "abc"}]}
        compare_reference_case(reference, reference)
        for mutation in ["classes", "fingerprint", "tokens"]:
            case = deepcopy(reference)
            if mutation == "classes":
                case["classes"] = []
            elif mutation == "fingerprint":
                case["replays"][0]["output_f32_bits_sha256"] = "def"
            else:
                case["ordinary"][0]["tokens"] = [1, 3]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                compare_reference_case(case, reference)


if __name__ == "__main__":
    unittest.main()
