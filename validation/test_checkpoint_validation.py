import argparse
import json
import pathlib
import struct
import tempfile
import unittest
from unittest import mock

from validation import compare_checkpoints as comparator
from validation import reference_runner


def write_safetensors(path, tensors):
    header = {"__metadata__": {"test": "true"}}
    chunks = []
    offset = 0
    for name, (shape, values) in tensors.items():
        raw = b"".join(struct.pack("<f", value) for value in values)
        header[name] = {
            "dtype": "F32",
            "shape": list(shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        chunks.append(raw)
        offset += len(raw)
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(chunks))


def write_artifact(root, name, input_ids, fed_ids, prefill, decode):
    tensor_path = root / (name + ".safetensors")
    write_safetensors(
        tensor_path,
        {
            "prefill.logits": ((1, len(prefill)), prefill),
            "decode.logits": (
                (len(decode), len(prefill)),
                [value for row in decode for value in row],
            ),
        },
    )
    report_path = root / (name + ".json")
    report_path.write_text(
        json.dumps(
            {
                "input": {"token_ids": input_ids},
                "output": {
                    "fed_token_ids": fed_ids,
                    "tensor_file": str(tensor_path),
                },
            }
        )
    )
    return report_path


def thresholds(**overrides):
    values = {
        "relative_l2_max": 0.02,
        "cosine_similarity_min": 0.999,
        "top_k": 2,
        "top_k_overlap_min": 2,
        "require_unambiguous_argmax_match": True,
        "argmax_margin_min": 0.0,
    }
    values.update(overrides)
    return comparator.Thresholds(**values)


class ComparatorTests(unittest.TestCase):
    def test_identical_artifacts_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            actual = write_artifact(root, "actual", [1, 2], [3], [0.1, 0.8, 0.2], [[0.7, 0.2, 0.1]])
            reference = write_artifact(
                root, "reference", [1, 2], [3], [0.1, 0.8, 0.2], [[0.7, 0.2, 0.1]]
            )
            report = comparator.compare_artifacts(actual, reference, thresholds())
            self.assertTrue(report["passed"])
            self.assertEqual(report["tensors"]["prefill.logits"]["summary"]["argmax_matches"], 1)

    def test_numeric_and_token_identity_failures_are_reported(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            actual = write_artifact(root, "actual", [1, 9], [3], [0.9, 0.1, 0.0], [[0.7, 0.2, 0.1]])
            reference = write_artifact(
                root, "reference", [1, 2], [3], [0.1, 0.8, 0.2], [[0.7, 0.2, 0.1]]
            )
            report = comparator.compare_artifacts(actual, reference, thresholds())
            self.assertFalse(report["passed"])
            self.assertFalse(report["identity_checks"]["input.token_ids"])
            self.assertFalse(report["tensors"]["prefill.logits"]["passed"])
            self.assertTrue(report["tensors"]["decode.logits"]["passed"])

    def test_ambiguous_reference_argmax_can_differ(self):
        metrics = comparator.compare_row(
            [0.8, 0.9, 0.0],
            [0.9, 0.9, 0.0],
            thresholds(
                relative_l2_max=1.0,
                cosine_similarity_min=0.0,
                top_k_overlap_min=1,
            ),
        )
        self.assertFalse(metrics["argmax_required"])
        self.assertTrue(metrics["passed"])

    def test_top_one_still_uses_runner_up_for_argmax_margin(self):
        metrics = comparator.compare_row(
            [0.1, 0.8, 0.2],
            [0.1, 0.8, 0.2],
            thresholds(top_k=1, top_k_overlap_min=1),
        )
        self.assertAlmostEqual(metrics["reference_argmax_margin"], 0.6)
        self.assertTrue(metrics["argmax_required"])

    def test_rejects_invalid_thresholds(self):
        args = argparse.Namespace(
            manifest=None,
            case=None,
            relative_l2_max=-1.0,
            cosine_similarity_min=None,
            top_k=None,
            top_k_overlap_min=None,
            require_unambiguous_argmax_match=None,
            argmax_margin_min=0.0,
        )
        with self.assertRaisesRegex(ValueError, "relative_l2_max"):
            comparator.thresholds_from_args(args)

    def test_manifest_profile_supplies_thresholds(self):
        args = argparse.Namespace(
            manifest=pathlib.Path("models.yaml"),
            case="model_case",
            relative_l2_max=None,
            cosine_similarity_min=None,
            top_k=None,
            top_k_overlap_min=None,
            require_unambiguous_argmax_match=None,
            argmax_margin_min=0.0,
        )
        profile = {
            "relative_l2_max": 0.08,
            "cosine_similarity_min": 0.99,
            "top_k": 5,
            "top_k_overlap_min": 3,
            "require_unambiguous_argmax_match": True,
        }
        with mock.patch.object(
            comparator, "load_manifest_profile", return_value=profile
        ):
            selected = comparator.thresholds_from_args(args)
        self.assertEqual(selected.relative_l2_max, 0.08)
        self.assertEqual(selected.top_k_overlap_min, 3)

    def test_cli_writes_report_and_returns_failure_status(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            actual = write_artifact(
                root, "actual", [1], [2], [0.9, 0.1], [[0.8, 0.2]]
            )
            reference = write_artifact(
                root, "reference", [1], [2], [0.1, 0.9], [[0.8, 0.2]]
            )
            output = root / "comparison.json"
            status = comparator.main(
                [
                    "--actual",
                    str(actual),
                    "--reference",
                    str(reference),
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(status, 1)
            self.assertFalse(json.loads(output.read_text())["passed"])


class ReferenceRunnerTests(unittest.TestCase):
    def test_probe_validation_does_not_import_gpu_dependencies(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "probe.json"
            path.write_text(
                json.dumps(
                    {
                        "input": {"token_ids": [1, 2]},
                        "output": {"fed_token_ids": [3]},
                    }
                )
            )
            self.assertEqual(reference_runner.read_probe(path)["input"]["token_ids"], [1, 2])

    def test_output_extensions_replace_existing_suffix(self):
        self.assertEqual(
            reference_runner.output_paths(pathlib.Path("result.probe")),
            (pathlib.Path("result.json"), pathlib.Path("result.safetensors")),
        )


if __name__ == "__main__":
    unittest.main()
