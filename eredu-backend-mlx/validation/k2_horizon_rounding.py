"""Inspect divergent BF16 projections using independently recomputed FP64 dots.

Inputs are a pinned local checkpoint and publisher/native component captures.
This diagnostic neither changes the reference nor relaxes parity thresholds.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("native", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--layers", type=int, default=3)
    args = parser.parse_args()
    torch.set_num_threads(4)
    index = json.loads((args.artifact / "model.safetensors.index.json").read_text())["weight_map"]
    rows = []
    with safe_open(args.reference, framework="pt") as reference, safe_open(args.native, framework="pt") as native:
        for layer in range(args.layers):
            inputs = reference.get_tensor(f"layers.0.{layer}.input_norm")
            inputs = inputs.reshape(-1, inputs.shape[-1]).double()
            for projection in ["q_proj", "k_proj", "v_proj"]:
                key = f"model.layers.{layer}.self_attn.{projection}.weight"
                with safe_open(args.artifact / index[key], framework="pt") as source:
                    weights = source.get_tensor(key)
                name = f"layers.0.{layer}.{projection}"
                expected = reference.get_tensor(name).reshape(inputs.shape[0], -1)
                actual = native.get_tensor(name).reshape_as(expected)
                for row, column in torch.nonzero(expected != actual):
                    value = (inputs[row] * weights[column].double()).sum()
                    rows.append({
                        "layer": layer, "projection": projection,
                        "row": int(row), "column": int(column),
                        "publisher": expected[row, column].item(),
                        "native": actual[row, column].item(),
                        "fp64_dot": value.item(),
                        "rounded_bf16": value.bfloat16().float().item(),
                    })
    report = {
        "torch": torch.__version__,
        "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        "native_sha256": hashlib.sha256(args.native.read_bytes()).hexdigest(),
        "divergent_projections": rows,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
