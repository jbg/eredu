"""Retrieve bounded, pinned GGUF headers without downloading tensor payloads.

The original prefixes stay outside tracked source. Full file lengths and hashes
come from the pinned publisher inventory; header hashes are measured locally.
The portable Rust GGUF reader independently validates the retained prefixes.
"""

import argparse
import concurrent.futures
import hashlib
import json
import re
import struct
import urllib.parse
from pathlib import Path

from muse_spark_inventory import MAX_HEADER_BYTES, credential, read_bounded


class NeedMore(Exception):
    pass


def parse_header(data):
    cursor = 0
    endian = "<"

    def raw(count):
        nonlocal cursor
        if count < 0 or count > MAX_HEADER_BYTES or cursor + count > len(data):
            raise NeedMore()
        value = data[cursor:cursor + count]
        cursor += count
        return value

    def scalar(code):
        return struct.unpack(endian + code, raw(struct.calcsize(code)))[0]

    def count():
        value = scalar("Q")
        if value > 1_000_000:
            raise ValueError("GGUF entry count exceeds the inspection bound")
        return value

    def string():
        return raw(scalar("Q")).decode("utf-8")

    def value(kind):
        codes = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i",
                 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
        if kind == 8:
            return string()
        if kind == 9:
            element, length = scalar("I"), count()
            if element == 9:
                raise ValueError("nested GGUF metadata arrays are invalid")
            return [value(element) for _ in range(length)]
        if kind not in codes:
            raise ValueError("unknown GGUF metadata value type")
        return scalar(codes[kind])

    magic = raw(4)
    if magic == b"FUGG":
        endian = ">"
    elif magic != b"GGUF":
        raise ValueError("invalid GGUF magic")
    version = scalar("I")
    if version != 3:
        raise ValueError("Spark GGUF header has an unverified container version")
    tensor_count, metadata_count = count(), count()
    metadata = {}
    for _ in range(metadata_count):
        key = string()
        if key in metadata:
            raise ValueError("duplicate GGUF metadata key")
        metadata[key] = value(scalar("I"))
    tensors = {}
    for _ in range(tensor_count):
        name = string()
        rank = scalar("I")
        if not 1 <= rank <= 4 or name in tensors:
            raise ValueError("invalid GGUF tensor rank or duplicate tensor")
        dimensions = [scalar("Q") for _ in range(rank)]
        tensors[name] = {"gguf_dimensions": dimensions,
                         "shape": list(reversed(dimensions)),
                         "ggml_type": scalar("I"), "offset": scalar("Q")}
    alignment = metadata.get("general.alignment", 32)
    if type(alignment) is not int or alignment <= 0:
        raise ValueError("invalid GGUF alignment")
    return {"version": version, "endian": "little" if endian == "<" else "big",
            "header_bytes": cursor, "data_start": (cursor + alignment - 1) // alignment * alignment,
            "metadata": metadata, "tensors": tensors}


def retrieve(base, token, output, name, size):
    path = output / (name + ".header.json")
    binary = output / (name + ".header.bin")
    if path.is_file() and binary.is_file():
        evidence = json.loads(path.read_text())
        raw = binary.read_bytes()
        if evidence["bytes"] == size and hashlib.sha256(raw).hexdigest() == evidence["header_sha256"]:
            return evidence
        raise ValueError("retained GGUF header evidence differs from its pinned source")
    length = min(65536, size)
    while True:
        prefix = read_bounded(base + urllib.parse.quote(name, safe="/"), token, length, 0)
        try:
            evidence = parse_header(prefix)
            break
        except NeedMore:
            if length >= min(MAX_HEADER_BYTES, size):
                raise ValueError("GGUF header exceeds the inspection bound") from None
            length = min(length * 2, MAX_HEADER_BYTES, size)
    prefix = prefix[:evidence["header_bytes"]]
    evidence.update({"file": name, "bytes": size,
                     "header_sha256": hashlib.sha256(prefix).hexdigest(),
                     "payload_sha256_verified": False})
    if evidence["tensors"] and evidence["data_start"] > size:
        raise ValueError("GGUF data start exceeds the pinned file length")
    path.parent.mkdir(parents=True, exist_ok=True)
    binary.write_bytes(prefix)
    path.write_text(json.dumps(evidence, separators=(",", ":")) + "\n")
    print(name, "header", len(prefix), "tensors", len(evidence["tensors"]), flush=True)
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inventory", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--repository", action="append", required=True)
    args = parser.parse_args()
    inventory = json.loads(args.inventory.read_text())
    token = credential()
    for repository in args.repository:
        artifact = next(a for a in inventory["artifacts"] if a["repository"] == repository)
        output = args.output / repository / artifact["revision"]
        output.mkdir(parents=True, exist_ok=True)
        base = f"https://huggingface.co/{repository}/resolve/{artifact['revision']}/"
        files = [(name, record["bytes"]) for name, record in artifact["files"].items() if name.endswith(".gguf")]
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            headers = list(pool.map(lambda item: retrieve(base, token, output, *item), files))
        groups = {}
        for header in headers:
            variant = re.sub(r"-\d{5}-of-\d{5}\.gguf$", "", header["file"])
            groups.setdefault(variant, []).append(header)
        evidence = {"repository": repository, "revision": artifact["revision"],
                    "payload_sha256_verified": False, "variants": {}}
        for variant, members in groups.items():
            tensors = {}
            for member in members:
                for name, tensor in member["tensors"].items():
                    if name in tensors:
                        raise ValueError("duplicate tensor across GGUF shards")
                    tensors[name] = dict(tensor, shard=member["file"])
            evidence["variants"][variant] = {
                "headers": [{key: member[key] for key in ["file", "bytes", "header_bytes", "header_sha256", "data_start"]} for member in members],
                "tensors": tensors,
            }
        (output / "evidence.json").write_text(json.dumps(evidence, separators=(",", ":")) + "\n")
        print(repository, "variants", len(groups), "shards", len(headers), flush=True)


if __name__ == "__main__":
    import urllib.error
    try:
        main()
    except urllib.error.HTTPError as error:
        raise SystemExit(f"Hugging Face returned HTTP {error.code}; credentials and redirects are not recorded") from None
    except urllib.error.URLError:
        raise SystemExit("Hugging Face network request failed; credentials and redirects are not recorded") from None
