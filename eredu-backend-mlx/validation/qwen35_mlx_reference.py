"""Compare a chat_probe report with MLX-LM on the exact same cached token prefix.

Validated with mlx-lm==0.31.3, mlx==0.32.2, transformers==5.17.0.
The checkpoint must be pinned and verified before running; this script uses no
network. Example: python qwen35_mlx_reference.py /tmp/qwen35-ordinary.json
"""

import argparse
import importlib.metadata
import json
from pathlib import Path

import mlx.core as mx
from mlx_lm import load


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    if report["temperature"] != 0:
        raise ValueError("Reference comparison requires greedy decoding")
    model, tokenizer = load(report["checkpoint"])
    cache = model.make_cache()
    inputs = mx.array([report["prompt_ids"]])
    generated = []
    for _ in report["token_ids"]:
        logits = model(inputs, cache=cache)[0, -1].astype(mx.float32)
        token = int(mx.argmax(logits).item())
        generated.append(token)
        inputs = mx.array([[token]])
    result = {
        "checkpoint": report["checkpoint"],
        "versions": {name: importlib.metadata.version(name)
                     for name in ("mlx", "mlx-lm", "transformers")},
        "prompt_ids": report["prompt_ids"],
        "eredu_token_ids": report["token_ids"],
        "reference_token_ids": generated,
        "exact_match": generated == report["token_ids"],
        "reference_text": tokenizer.decode(generated),
    }
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded)
    print(encoded, end="")
    if not result["exact_match"]:
        raise AssertionError("Eredu and MLX-LM generated different token sequences")


if __name__ == "__main__":
    main()
