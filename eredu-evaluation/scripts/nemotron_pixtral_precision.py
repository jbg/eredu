"""Check pinned Pixtral patch embedding against independently bounded FP64 equations.

Usage: python nemotron_pixtral_precision.py CHECKPOINT VISION_REFERENCE.safetensors
The native normal-loader test emits its prepared pixels and normalized embedding.
The bound includes FP32 product accumulation and the published BF16 rounding
before normalization gain; it does not relax the final-logit comparison.
"""
import argparse
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import load_file
from nemotron_reference_provenance import verify_release


def spacing(value):
    rounded = value.to(torch.bfloat16)
    return (torch.nextafter(rounded, torch.full_like(rounded, float("inf"))).double()
            - rounded.double()).abs()


def check(patches, actual, weight, gain, label):
    x = patches.to(torch.bfloat16).double()
    projected = (x @ weight.T).unsqueeze(0)
    eps = torch.finfo(torch.float32).eps
    n = x.shape[-1]
    gamma = eps * n / (1 - eps * n)
    error = (gamma * (x.abs() @ weight.abs().T)).unsqueeze(0)
    error += spacing(projected) * 0.50001
    lower = projected - error
    upper = projected + error
    magnitude = projected.abs()
    # RMS denominator bounds over the entire uncertain projected row.
    norm_lower = (((magnitude - error).clamp_min(0).square().mean(-1, keepdim=True)
                   + 1e-5).sqrt())
    norm_upper = (((magnitude + error).square().mean(-1, keepdim=True)
                   + 1e-5).sqrt())
    reduction_error = eps * (projected.shape[-1] + 4)
    norm_lower *= 1 - reduction_error
    norm_upper *= 1 + reduction_error
    ratios = torch.stack([lower / norm_lower, lower / norm_upper,
                          upper / norm_lower, upper / norm_upper])
    lo, hi = ratios.min(0).values, ratios.max(0).values
    rounding = torch.maximum(spacing(lo), spacing(hi)) * 0.50001
    products = torch.stack([(lo - rounding) * gain, (hi + rounding) * gain])
    lo, hi = products.min(0).values, products.max(0).values
    # One output rounding follows the learned gain multiplication.
    rounding = torch.maximum(spacing(lo), spacing(hi)) * 0.50001
    lo -= rounding
    hi += rounding
    actual = actual.double()
    valid = (actual >= lo) & (actual <= hi)
    ideal = (projected / (projected.square().mean(-1, keepdim=True) + 1e-5).sqrt())
    ideal = (ideal.to(torch.bfloat16).double() * gain).to(torch.bfloat16).double()
    print(label, "max_ideal_difference", (actual - ideal).abs().max().item(),
          "different_ideal_entries", int((actual != ideal).sum()),
          "outside_interval", int((~valid).sum()), flush=True)
    assert actual.shape == ideal.shape and valid.all(), label


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("reference", type=Path)
    args = parser.parse_args()
    release = verify_release(args.checkpoint)
    torch.set_num_threads(8)
    parameters = {}
    names = ["encoder.vision_tower.patch_conv.weight", "encoder.vision_tower.ln_pre.weight"]
    for member in release["weight_files"]:
        with safe_open(args.checkpoint / member["rfilename"], framework="pt", device="cpu") as f:
            for name in names:
                if name in f.keys():
                    parameters[name] = f.get_tensor(name).double()
    weight = parameters[names[0]].flatten(1)
    gain = parameters[names[1]]
    reference = load_file(str(args.reference))
    native = load_file(str(args.checkpoint / "normal-loader-bf16-context-components.safetensors"))
    check(reference["vision.patches"], reference["vision.embedding"], weight, gain, "published")
    check(native["vision.patches"], native["vision.embedding"], weight, gain, "native")


if __name__ == "__main__":
    main()
