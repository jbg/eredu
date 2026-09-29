#!/usr/bin/env python3
"""Small independent CPU oracle for the pinned Flash-Next inherited vision equations.

Reference: Transformers 27166ea03f12c940f23176a904ab1d2ff1a3dcbb,
models/qwen3_5_moe/modeling_qwen3_5_moe.py. No Eredu code is imported.
"""
import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    shapes = {
        "pos_embed.weight": [16, 8],
        "patch_embed.proj.weight": [8, 3, 2, 2, 2],
        "patch_embed.proj.bias": [8],
        "merger.norm.weight": [8], "merger.norm.bias": [8],
        "merger.linear_fc1.weight": [32, 32], "merger.linear_fc1.bias": [32],
        "merger.linear_fc2.weight": [32, 32], "merger.linear_fc2.bias": [32],
    }
    for layer in range(2):
        for name, shape in {
            "norm1.weight": [8], "norm1.bias": [8],
            "norm2.weight": [8], "norm2.bias": [8],
            "attn.qkv.weight": [24, 8], "attn.qkv.bias": [24],
            "attn.proj.weight": [8, 8], "attn.proj.bias": [8],
            "mlp.linear_fc1.weight": [12, 8], "mlp.linear_fc1.bias": [12],
            "mlp.linear_fc2.weight": [8, 12], "mlp.linear_fc2.bias": [8],
        }.items():
            shapes[f"blocks.{layer}.{name}"] = shape
    weights = {}
    for name, shape in shapes.items():
        seed = sum(f"model.visual.{name}".encode()) % 17
        values = ((torch.arange(math.prod(shape)) * 7 + seed) % 37 - 18).float() / 256
        if "norm" in name and name.endswith("weight"):
            values += 1
        weights[name] = values.reshape(shape)

    def linear(x, name):
        return F.linear(x, weights[name + ".weight"], weights[name + ".bias"])

    def norm(x, name):
        return F.layer_norm(x, [8], weights[name + ".weight"], weights[name + ".bias"], 1e-6)

    cases = []
    for grids in [[[1, 2, 6]], [[2, 4, 2]], [[1, 2, 6], [2, 4, 2]]]:
        count = sum(t * h * w for t, h, w in grids)
        pixels = ((torch.arange(count * 24) * 11 % 53) - 26).float().reshape(count, 24) / 64
        hidden = F.conv3d(pixels.reshape(-1, 3, 2, 2, 2), weights["patch_embed.proj.weight"],
                          weights["patch_embed.proj.bias"], stride=[2, 2, 2]).reshape(count, 8)
        positions, coords, chunks = [], [], []
        table = weights["pos_embed.weight"].reshape(4, 4, 8).permute(2, 0, 1).unsqueeze(0)
        for t, h, w in grids:
            resized = F.interpolate(table, size=[h, w], mode="bilinear", align_corners=True)[0].permute(1, 2, 0)
            order = [(y, x) for by in range(0, h, 2) for bx in range(0, w, 2)
                     for y in range(by, by + 2) for x in range(bx, bx + 2)]
            for _ in range(t):
                positions.extend(resized[y, x] for y, x in order)
                coords.extend(order)
                chunks.append(h * w)
        hidden += torch.stack(positions)
        angles = torch.tensor(coords, dtype=torch.float32).repeat(1, 2).unsqueeze(1)
        cos, sin = angles.cos(), angles.sin()
        for layer in range(2):
            root = f"blocks.{layer}"
            q, k, v = linear(norm(hidden, root + ".norm1"), root + ".attn.qkv").reshape(count, 3, 2, 4).unbind(1)
            def rotary(x):
                a, b = x.chunk(2, dim=-1)
                return x * cos + torch.cat([-b, a], dim=-1) * sin
            q, k = rotary(q), rotary(k)
            attended, start = [], 0
            for length in chunks:
                sl = slice(start, start + length)
                query, key, value = (x[sl].transpose(0, 1) for x in [q, k, v])
                scores = (query @ key.transpose(-1, -2)) * 0.5
                attended.append((scores.softmax(-1) @ value).transpose(0, 1).reshape(length, 8))
                start += length
            hidden += linear(torch.cat(attended), root + ".attn.proj")
            hidden += linear(F.gelu(linear(norm(hidden, root + ".norm2"), root + ".mlp.linear_fc1"),
                                    approximate="tanh"), root + ".mlp.linear_fc2")
        merged = linear(F.gelu(linear(norm(hidden, "merger.norm").reshape(-1, 32), "merger.linear_fc1")), "merger.linear_fc2")
        cases.append({"grid": grids, "shape": [1, count // 4, 32], "values": merged.flatten().tolist()})
    args.output.write_text(json.dumps({"weights": shapes, "cases": cases}, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
