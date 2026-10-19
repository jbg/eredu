"""Retrieve pinned Spark metadata and bounded SafeTensors headers.

Payload shards remain remote. Authentication uses the normal Hugging Face
environment or token file; credentials and signed URLs are never recorded.
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import struct
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


MAX_HEADER_BYTES = 16 * 1024 * 1024


class CredentialRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, url):
        redirected = super().redirect_request(request, response, code, message, headers, url)
        if redirected is not None:
            origin = urllib.parse.urlsplit(url)
            if origin.scheme != "https" or origin.hostname != "huggingface.co":
                redirected.remove_header("Authorization")
        return redirected


def credential():
    value = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if value:
        return value.strip()
    home = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface")))
    path = Path(os.environ.get("HF_TOKEN_PATH", str(home / "token")))
    return path.read_text().strip() if path.is_file() else None


def read_bounded(url, token, length, start=None):
    request = urllib.request.Request(url)
    if token:
        request.add_header("Authorization", "Bearer " + token)
    if start is not None:
        request.add_header("Range", f"bytes={start}-{start + length - 1}")
    opener = urllib.request.build_opener(CredentialRedirect())
    with opener.open(request, timeout=60) as response:
        if start is not None:
            if response.status == 206:
                expected = f"bytes {start}-{start + length - 1}/"
                if not response.headers.get("Content-Range", "").startswith(expected):
                    raise ValueError("server returned a different byte range")
            elif response.status == 200:
                # Some hosts ignore Range. Read only the bounded prefix and
                # close the response without downloading the remaining shard.
                if start + length > MAX_HEADER_BYTES:
                    raise ValueError("server ignored a payload range outside the bounded metadata prefix")
                if len(response.read(start)) != start:
                    raise ValueError("short prefix before the requested header")
            else:
                raise ValueError("unexpected range response status")
        value = response.read(length + (start is None))
        if start is None and len(value) > length:
            raise ValueError("metadata exceeds the download bound")
        if start is not None and len(value) != length:
            raise ValueError("short SafeTensors header range")
        return value


def json_file(url, token, path):
    value = read_bounded(url, token, MAX_HEADER_BYTES)
    decoded = json.loads(value)
    path.write_bytes(value)
    return decoded, hashlib.sha256(value).hexdigest()


def header(base, token, name, size):
    url = base + urllib.parse.quote(name, safe="/")
    prefix = read_bounded(url, token, 8, 0)
    length = struct.unpack("<Q", prefix)[0]
    if length == 0 or length > MAX_HEADER_BYTES or length + 8 > size:
        raise ValueError("SafeTensors header length is outside its shard bounds")
    raw = read_bounded(url, token, length, 8)
    decoded = json.loads(raw)
    tensors = {}
    intervals = []
    widths = {"BOOL": 1, "U8": 1, "I8": 1, "I16": 2, "U16": 2,
              "F16": 2, "BF16": 2, "I32": 4, "U32": 4, "F32": 4,
              "I64": 8, "U64": 8, "F64": 8, "F8_E4M3": 1,
              "F8_E4M3FN": 1, "F8_E5M2": 1, "F8_E8M0": 1}
    for key, tensor in decoded.items():
        if key == "__metadata__":
            continue
        dtype, shape, offsets = tensor["dtype"], tensor["shape"], tensor["data_offsets"]
        if not isinstance(shape, list) or any(type(d) is not int or d < 0 for d in shape):
            raise ValueError("invalid tensor shape")
        if len(offsets) != 2 or any(type(o) is not int for o in offsets):
            raise ValueError("invalid tensor offsets")
        first, end = offsets
        if first < 0 or end < first or end > size - 8 - length:
            raise ValueError("tensor payload lies outside the pinned shard size")
        if dtype not in widths:
            raise ValueError("unknown SafeTensors scalar encoding: " + dtype)
        elements = 1
        for dimension in shape:
            elements *= dimension
        if end - first != elements * widths[dtype]:
            raise ValueError("tensor byte length differs from its scalar geometry")
        tensors[key] = tensor
        intervals.append((first, end))
    cursor = 0
    for first, end in sorted(intervals):
        if first != cursor:
            raise ValueError("SafeTensors payload has a gap or overlapping tensors")
        cursor = end
    if cursor != size - 8 - length:
        raise ValueError("SafeTensors payload length differs from the pinned shard size")
    return {"file": name, "bytes": size, "header_sha256": hashlib.sha256(prefix + raw).hexdigest(),
            "header_bytes": 8 + length, "tensors": tensors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inventory", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--repository", action="append", required=True)
    parser.add_argument("--headers", action="store_true")
    parser.add_argument("--snapshot-prefix", type=Path,
                        help="write compact config/catalog/header-evidence snapshots for one selected repository")
    args = parser.parse_args()
    if args.snapshot_prefix and (len(args.repository) != 1 or not args.headers):
        parser.error("--snapshot-prefix requires one repository and --headers")
    inventory = json.loads(args.inventory.read_text())
    token = credential()
    for repository in args.repository:
        artifact = next(a for a in inventory["artifacts"] if a["repository"] == repository)
        revision = artifact["revision"]
        output = args.output / repository / revision
        output.mkdir(parents=True, exist_ok=True)
        base = f"https://huggingface.co/{repository}/resolve/{revision}/"
        metadata = {}
        for name in ["config.json", "model.safetensors.index.json", "processor_config.json",
                     "tokenizer_config.json", "generation_config.json"]:
            if name in artifact["files"]:
                value, digest = json_file(base + name, token, output / name)
                metadata[name] = {"sha256": digest}
        report = {"repository": repository, "revision": revision,
                  "metadata": metadata, "payload_sha256_verified": False}
        if args.headers:
            shards = [(name, record["bytes"]) for name, record in artifact["files"].items()
                      if name.endswith(".safetensors")]
            def retrieve(item):
                name, size = item
                path = output / (name + ".header.json")
                result = header(base, token, name, size)
                path.write_text(json.dumps(result, separators=(",", ":")) + "\n")
                return result
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                results = list(pool.map(retrieve, shards))
            seen = {}
            for result in results:
                for name in result["tensors"]:
                    if name in seen:
                        raise ValueError("duplicate tensor across shards: " + name)
                    seen[name] = result["file"]
            index_path = output / "model.safetensors.index.json"
            if index_path.exists() and json.loads(index_path.read_text())["weight_map"] != seen:
                raise ValueError("tensor index and actual shard headers differ")
            report["shard_count"] = len(results)
            report["tensor_count"] = len(seen)
            report["headers"] = [{k: v for k, v in r.items() if k != "tensors"} for r in results]
        (output / "evidence.json").write_text(json.dumps(report, indent=2) + "\n")
        if args.snapshot_prefix:
            snapshot = str(args.snapshot_prefix)
            catalog = {"repository": repository, "revision": revision,
                       "payload_sha256_verified": False,
                       "tensors": {name: {**tensor, "shard": result["file"]}
                                   for result in results for name, tensor in result["tensors"].items()}}
            Path(snapshot + "-catalog.json").write_text(json.dumps(catalog, separators=(",", ":")) + "\n")
            Path(snapshot + "-header-evidence.json").write_text(json.dumps(report, indent=2) + "\n")
            Path(snapshot + "-config.json").write_bytes((output / "config.json").read_bytes())
        print(repository, revision, "metadata retrieved", "headers", report.get("shard_count", 0), flush=True)


if __name__ == "__main__":
    try:
        main()
    except urllib.error.HTTPError as error:
        raise SystemExit(f"Hugging Face returned HTTP {error.code}; no credentials or redirect URLs are recorded") from None
    except urllib.error.URLError:
        raise SystemExit("Hugging Face network request failed; no credentials or redirect URLs are recorded") from None
