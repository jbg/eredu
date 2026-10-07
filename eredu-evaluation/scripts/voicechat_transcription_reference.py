#!/usr/bin/env python3
"""Run the pinned NVIDIA transcript method on explicit vocabulary-piece cases.

The AST-extracted method executes unchanged without importing NeMo or loading
weights. Its outputs verify presentation semantics, not neural RNNT accuracy.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace


REVISION = "097dfe9e2f55baf653b83035868bdc89849f1b47"
SOURCE = "nemo/collections/speechlm2/inference/model_wrappers/nemotron_voicechat_inference_wrapper.py"
SHA256 = "b50aeab4b36285cf041330b315e35a01788e1edb639380bb6105446787cc6a1a"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wrapper", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.wrapper.read_bytes()
    if hashlib.sha256(source).hexdigest() != SHA256:
        raise SystemExit("wrapper bytes differ from the pinned NVIDIA reference")
    method = next(node for node in ast.walk(ast.parse(source))
                  if isinstance(node, ast.FunctionDef)
                  and node.name == "_rnnt_decode_text")
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    namespace = {}
    exec(compile(module, SOURCE, "exec"), namespace)
    vocabulary = ["▁hello", "▁", "world", "!", "\t\n", "\u001c\u001d\u001e\u001f",
                  "\u00a0\u2003", "世", "界", "<unk>", "<0xE2>", "<0x82>", "<0xAC>", "", "▁▁hi"]
    owner = SimpleNamespace(model=SimpleNamespace(
        stt_model=SimpleNamespace(rnnt_joint=SimpleNamespace(vocabulary=vocabulary))))
    decode = lambda ids: namespace["_rnnt_decode_text"](owner, ids)
    cases = []
    for name, tokens in [
        ("words_and_punctuation", [0, 1, 2, 3]),
        ("leading_and_trailing_whitespace", [4, 1, 0, 4, 1]),
        ("python_information_separators", [5, 0, 5, 2, 5]),
        ("unicode_whitespace", [6, 0, 6, 2, 6]),
        ("unicode_word_pieces", [1, 7, 8, 1]),
        ("literal_unknown_and_byte_pieces", [9, 10, 11, 12]),
        ("empty_and_repeated_markers", [13, 14, 13, 1, 1, 2]),
        ("whitespace_only", [1, 4, 5, 6]),
        ("empty", []),
    ]:
        cases.append(dict(name=name, tokens=tokens, text=decode(tokens),
                          prefixes=[decode(tokens[:end]) for end in range(1, len(tokens) + 1)]))
    payload = dict(reference_revision=REVISION, reference_source=SOURCE,
                   reference_sha256=SHA256, method="_rnnt_decode_text",
                   vocabulary=vocabulary, cases=cases)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
