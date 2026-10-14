"""Compare independently decoded quantized logits with official-weight reference logits.

This measures weight-representation drift, separately from backend execution error.
Usage: python nemotron_quantized_quality.py ARTIFACT.safetensors OFFICIAL.safetensors
"""
import argparse
import json
from safetensors import safe_open
import torch


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("artifact")
    parser.add_argument("official")
    parser.add_argument("--output")
    args = parser.parse_args()
    torch.set_num_threads(4)
    results = {}
    with safe_open(args.artifact, framework="pt", device="cpu") as artifact, safe_open(
        args.official, framework="pt", device="cpu"
    ) as official:
        keys = sorted(key for key in artifact.keys() if key.endswith(".logits"))
        if not keys or keys != sorted(key for key in official.keys() if key.endswith(".logits")):
            raise ValueError("reference logits do not describe the same invocation cases")
        for key in keys:
            actual, expected = artifact.get_tensor(key), official.get_tensor(key)
            if actual.shape != expected.shape or actual.ndim != 3:
                raise ValueError(f"reference logit geometry differs: {key}")
            difference = actual.double() - expected.double()
            if not torch.isfinite(difference).all():
                raise ValueError(f"nonfinite reference logits: {key}")
            results[key] = {
                "rms_vs_official": difference.square().mean().sqrt().item(),
                "max_abs_vs_official": difference.abs().max().item(),
                "greedy_matching_rows": (actual.argmax(-1) == expected.argmax(-1)).sum().item(),
                "rows": actual.shape[0] * actual.shape[1],
            }
    report = json.dumps(results, indent=2) + "\n"
    if args.output:
        with open(args.output, "w") as destination:
            destination.write(report)
    else:
        print(report, end="")


if __name__ == "__main__":
    main()
