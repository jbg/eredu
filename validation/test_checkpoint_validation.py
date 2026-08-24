import json
import pathlib
import tempfile
import unittest
from unittest import mock

from validation import reference_runner
from validation import run_pilot_case
from validation import run_prompt_matrix


class ReferenceRunnerTests(unittest.TestCase):
    def test_prefill_mode_defaults_to_full_and_accepts_tokenwise(self):
        common = ["--probe", "probe.json", "--output", "result"]
        self.assertEqual(reference_runner.parse_args(common).prefill_mode, "full")
        self.assertEqual(
            reference_runner.parse_args(
                common + ["--prefill-mode", "tokenwise"]
            ).prefill_mode,
            "tokenwise",
        )

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

    def test_qwen3_vl_uses_image_text_auto_model(self):
        fake_transformers = mock.Mock()
        selected = reference_runner.auto_model_class(fake_transformers, "qwen3_vl")
        self.assertIs(selected, fake_transformers.AutoModelForImageTextToText)
        selected = reference_runner.auto_model_class(fake_transformers, "qwen3")
        self.assertIs(selected, fake_transformers.AutoModelForCausalLM)

    def test_nemotron_h_registers_mlp_cache_placeholder(self):
        placeholder = object()
        registry = {"linear_attention": object()}

        patches = reference_runner.patch_nemotron_h_cache_registry(
            "nemotron_h", registry, placeholder
        )

        self.assertIs(registry["mlp"], placeholder)
        self.assertEqual(patches, ["nemotron_h_mlp_cache_placeholder"])

    def test_nemotron_h_preserves_upstream_mlp_cache_mapping(self):
        upstream = object()
        registry = {"mlp": upstream}

        patches = reference_runner.patch_nemotron_h_cache_registry(
            "nemotron_h", registry, object()
        )

        self.assertIs(registry["mlp"], upstream)
        self.assertEqual(patches, [])

    def test_other_models_do_not_patch_cache_registry(self):
        registry = {}

        patches = reference_runner.patch_nemotron_h_cache_registry(
            "qwen3", registry, object()
        )

        self.assertEqual(registry, {})
        self.assertEqual(patches, [])


class PilotPromptTests(unittest.TestCase):
    def test_select_prompt_expands_repeat(self):
        manifest = {
            "input_sets": {
                "text_correctness": {
                    "prompts": [{"id": "pattern", "text": "ab", "repeat": 3}]
                }
            }
        }

        self.assertEqual(
            run_pilot_case.select_prompt(manifest, "pattern"),
            {"id": "pattern", "text": "ababab", "repeat": 3},
        )

    def test_select_prompt_rejects_unknown_or_invalid_repeat(self):
        manifest = {
            "input_sets": {
                "text_correctness": {
                    "prompts": [{"id": "pattern", "text": "ab", "repeat": 0}]
                }
            }
        }

        with self.assertRaisesRegex(ValueError, "resolve exactly once"):
            run_pilot_case.select_prompt(manifest, "missing")
        with self.assertRaisesRegex(ValueError, "repeat must be positive"):
            run_pilot_case.select_prompt(manifest, "pattern")

    def test_prompt_matrix_defaults_to_all_prompts(self):
        manifest = {
            "input_sets": {
                "text_correctness": {
                    "prompts": [{"id": "first"}, {"id": "second"}]
                }
            }
        }

        self.assertEqual(
            run_prompt_matrix.prompt_ids(manifest, None), ["first", "second"]
        )
        self.assertEqual(
            run_prompt_matrix.prompt_ids(manifest, ["second"]), ["second"]
        )

    def test_prompt_matrix_rejects_unknown_or_duplicate_prompts(self):
        manifest = {
            "input_sets": {
                "text_correctness": {"prompts": [{"id": "first"}]}
            }
        }

        with self.assertRaisesRegex(ValueError, "unknown prompt ids: second"):
            run_prompt_matrix.prompt_ids(manifest, ["second"])
        with self.assertRaisesRegex(ValueError, "must be unique"):
            run_prompt_matrix.prompt_ids(manifest, ["first", "first"])

if __name__ == "__main__":
    unittest.main()
