#!/usr/bin/env python3
"""Inspect pinned Qwen4Exp catalogs with bounded HTTP ranges; never fetch weights.

The output is metadata coverage, not inference parity or payload verification.
Use --download to populate an external artifact directory, then --fixture to
emit compact catalog fixtures. No Python packages beyond the standard library.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import re
import struct
import urllib.request

VARIANTS = {
    "bf16": ("Qwen3.8-Flash-Next", "de4b8e4d43b917e7706784d8bb445c9af86a3540",
             "99e815241ef03325536b0aaa4441deea45174c17fae31e10f0bb456410c590de"),
    "fp8": ("Qwen3.8-Flash-Next-FP8", "236dfdf285828023ca3bcd3f37366c58a3469b13",
            "0419e2c2dfbb925257d7409405433a793cf7ff7d96f3eba882a815ec6d9fe7a6"),
}
CONSTANTS = ("layer_multipliers", "ngram_heads_offsets", "ngram_heads_vocab_sizes", "weight_scale")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def fetch(url, limit, start=None):
    headers = {}
    if start is not None:
        end = start + limit - 1
        url += f"?eredu_range={start}-{end}"
        headers["Range"] = f"bytes={start}-{end}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as response:
        if start is not None and (response.status != 206 or not response.headers.get(
                "Content-Range", "").startswith(f"bytes {start}-{end}/")):
            raise ValueError("server did not honor bounded range")
        data = response.read(limit + 1)
    if len(data) > limit or start is not None and len(data) != limit:
        raise ValueError("response exceeded bound or was truncated")
    return data


def download(root):
    for variant, (repo, revision, expected) in VARIANTS.items():
        base = f"https://huggingface.co/Qwen/{repo}/resolve/{revision}"
        index = fetch(base + "/model.safetensors.index.json", 64 * 1024 * 1024)
        if digest(index) != expected:
            raise ValueError("pinned index hash changed")
        (root / f"{variant}.index.json").write_bytes(index)
        (root / f"{variant}.config.json").write_bytes(fetch(base + "/config.json", 1024 * 1024))
        for filename in sorted(set(json.loads(index)["weight_map"].values())):
            url = f"{base}/{filename}"
            length = struct.unpack("<Q", fetch(url, 8, 0))[0]
            if not 2 <= length <= 2 * 1024 * 1024:
                raise ValueError("header exceeds inspection budget")
            data = fetch(url, length, 8)
            tensors = json.loads(data)
            record = dict(url=url, header_bytes=length, header_sha256=digest(data), tensors=tensors)
            (root / f"{variant}-{filename}.header.json").write_text(json.dumps(record, indent=2) + "\n")
            print(variant, filename, length, flush=True)
            for name, metadata in tensors.items():
                suffix = name.rsplit(".", 1)[-1]
                if ".ple.ple_embedding." not in name or suffix not in CONSTANTS:
                    continue
                start, end = metadata["data_offsets"]
                if not 0 < end - start <= 4096:
                    raise ValueError("constant exceeds inspection budget")
                data = fetch(url, end - start, 8 + length + start)
                record = dict(source=url, tensor=name, **metadata, sha256=digest(data), bytes=data.hex())
                (root / f"{variant}-{suffix}.json").write_text(json.dumps(record, indent=2) + "\n")


def compact_catalog(root, variant):
    repo, revision, expected = VARIANTS[variant]
    index_bytes = (root / f"{variant}.index.json").read_bytes()
    if digest(index_bytes) != expected:
        raise ValueError("index hash differs from the pinned source")
    index = json.loads(index_bytes)["weight_map"]
    catalog = {}
    headers = []
    for filename in sorted(set(index.values())):
        record = json.loads((root / f"{variant}-{filename}.header.json").read_text())
        if record["url"] != f"https://huggingface.co/Qwen/{repo}/resolve/{revision}/{filename}":
            raise ValueError("header provenance differs from the pinned artifact")
        headers.append(dict(file=filename, sha256=record["header_sha256"], bytes=record["header_bytes"]))
        for name, meta in record["tensors"].items():
            if name == "__metadata__":
                continue
            if index.get(name) != filename or name in catalog:
                raise ValueError("header/index identity mismatch")
            count = 1
            for dim in meta["shape"]:
                count *= dim
            size = {"BF16": 2, "F16": 2, "F32": 4, "I64": 8, "F8_E4M3": 1}[meta["dtype"]]
            start, end = meta["data_offsets"]
            if start < 0 or end - start != count * size:
                raise ValueError("physical tensor range disagrees with geometry")
            catalog[name] = meta
    if catalog.keys() != index.keys():
        raise ValueError("incomplete header inspection")
    groups = {}
    for name, meta in sorted(catalog.items()):
        coordinates = []

        def replace(match):
            coordinates.append(int(match[2]))
            return match[1] + "{" + str(len(coordinates) - 1) + "}"

        template = re.sub(r"((?:layers\.|blocks\.|experts\.|shard_))(\d+)", replace, name)
        key = (template, meta["dtype"], tuple(meta["shape"]))
        groups.setdefault(key, set()).add(tuple(coordinates))
    records = []
    for (template, dtype, shape), coordinates in sorted(groups.items()):
        axes = [sorted(set(c[i] for c in coordinates)) for i in range(len(next(iter(coordinates))))]
        if set(itertools.product(*axes)) != coordinates:
            raise ValueError("catalog is not rectangular; retain literal coordinate sets")
        records.append(dict(name=template, axes=axes, dtype=dtype, shape=shape))
    constants = []
    for suffix in CONSTANTS:
        path = root / f"{variant}-{suffix}.json"
        if not path.exists() and variant == "bf16" and suffix == "weight_scale":
            continue
        constant = json.loads(path.read_text())
        if digest(bytes.fromhex(constant["bytes"])) != constant["sha256"]:
            raise ValueError("constant payload hash mismatch")
        constants.append({k: constant[k] for k in ("tensor", "dtype", "shape", "bytes", "sha256")})
    return dict(repository=f"Qwen/{repo}", revision=revision, index_sha256=expected,
                tensor_count=len(catalog), headers=headers, tensors=records, constants=constants)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path, help="external directory for pinned metadata")
    parser.add_argument("--download", action="store_true", help="fetch bounded headers and tiny constants")
    parser.add_argument("--fixture", type=Path, help="write compact metadata fixtures to this directory")
    args = parser.parse_args()
    args.artifacts.mkdir(parents=True, exist_ok=True)
    if args.download:
        download(args.artifacts)
    for variant in VARIANTS:
        record = compact_catalog(args.artifacts, variant)
        print(variant, record["tensor_count"], "tensors;", len(record["headers"]), "headers")
        if args.fixture:
            args.fixture.mkdir(parents=True, exist_ok=True)
            (args.fixture / f"{variant}.json").write_text(json.dumps(record, indent=2) + "\n")
    if args.fixture:
        config = json.loads((args.artifacts / "fp8.config.json").read_text())
        (args.fixture / "fp8-policy.json").write_text(json.dumps(config["quantization_config"], indent=2) + "\n")


if __name__ == "__main__":
    main()
