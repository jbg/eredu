#!/usr/bin/env python3
"""Build a reproducible processed-image sensor request for the pinned Qwen fixture.

Requires Transformers and tokenizers for independent local template/token IDs.
The image is uniform, so patch ordering cannot change its values. This exercises
real encoder input and source authentication without claiming visual recognition.
All processing here is caller-owned, outside the framework allocation domain.
"""

import argparse
import importlib.metadata
import json
from pathlib import Path

from prepared_chat_tools import request, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "image.request.json"
    if path.exists():
        parser.error("output already contains image.request.json")

    from transformers import AutoTokenizer

    processor = json.loads((args.checkpoint / "preprocessor_config.json").read_text())
    vision = json.loads((args.checkpoint / "config.json").read_text())["vision_config"]
    patch = processor["patch_size"]
    temporal = processor["temporal_patch_size"]
    merge = processor["merge_size"]
    assert (patch, temporal, merge) == (
        vision["patch_size"], vision["temporal_patch_size"], vision["spatial_merge_size"],
    )
    width = height = 256
    assert width % (patch * merge) == height % (patch * merge) == 0
    assert processor["size"]["shortest_edge"] <= width * height <= processor["size"]["longest_edge"]
    grid = [1, height // patch, width // patch]
    rows = grid[1] * grid[2]
    row = [
        (153 / 255 - mean) / std
        for mean, std in zip(processor["image_mean"], processor["image_std"], strict=True)
        for _ in range(temporal * patch * patch)
    ]
    assert len(row) == 3 * temporal * patch * patch

    document = request("required", 48, 128)
    document["messages"][1]["content"] = (
        "Reference image: <|vision_start|><|image_pad|><|vision_end|> "
        "The sensor value is 17. Please record this reading."
    )
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    prompt = tokenizer.apply_chat_template(
        document["messages"], tools=document["tools"], add_generation_prompt=True,
        enable_thinking=False, tokenize=False,
    )
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    marker = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    assert ids.count(marker) == 1
    position = ids.index(marker)

    def text_part(values):
        return {"modality": "text", "kind": "token_ids",
                "payload": {"shape": [1, len(values)], "values": {"u32": values}}}

    document["prepared_parts"] = [
        text_part(ids[:position]),
        {"modality": "image", "kind": "tensor",
         "payload": {"shape": [rows, len(row)], "values": {"f32": row * rows}},
         "metadata": [["patch_grid", {"shape": [1, 3], "values": {"i32": grid}}]],
         "extents": [{"patch_grid": {"time": grid[0], "height": grid[1], "width": grid[2]}}]},
        text_part(ids[position + 1:]),
    ]
    write_json(path, document)
    write_json(args.output / "image.preparation.json", {
        "checkpoint": str(args.checkpoint.resolve()),
        "request_sha256": sha256(path),
        "generator_sha256": sha256(Path(__file__)),
        "versions": {name: importlib.metadata.version(name) for name in ("transformers", "tokenizers")},
        "image": {"width": width, "height": height, "rgb": [153, 153, 153], "patch_grid": grid},
        "text_positions": len(ids), "marker_position": position,
        "decoder_positions": len(ids) - 1 + rows // (merge * merge),
        "purpose": "Authenticated image plus ordinary tools; no visual recognition or numerical-reference claim.",
    })
    print(path)


if __name__ == "__main__":
    main()
