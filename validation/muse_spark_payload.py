"""Download pinned payload shards outside source, with resumption and SHA-256 checks.

Credentials and signed redirects are never persisted. The destination retains
the selected disk headroom (64 GiB by default) after outstanding payload bytes.
"""

import argparse
import concurrent.futures
import hashlib
import json
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from muse_spark_inventory import CredentialRedirect, credential

CHUNK = 8 * 1024 * 1024


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(CHUNK):
            result.update(block)
    return result.hexdigest()


def download(base, token, output, name, record):
    if not name.endswith(".safetensors") or Path(name).name != name:
        raise ValueError("payload name must be one direct SafeTensors shard")
    destination = output / name
    partial = output / (name + ".partial")
    size, expected = record["bytes"], record["sha256"]
    if destination.exists():
        if destination.stat().st_size != size or digest(destination) != expected:
            raise ValueError("existing destination differs from the pinned payload; no file is removed")
        print(name, "existing SHA-256 verified", flush=True)
        return {"file": name, "bytes": size, "sha256": expected, "verified": True}
    for attempt in range(5):
        start = partial.stat().st_size if partial.exists() else 0
        if start > size:
            raise ValueError("partial payload exceeds its pinned length; no file is removed")
        if start == size:
            break
        request = urllib.request.Request(base + urllib.parse.quote(name))
        if token:
            request.add_header("Authorization", "Bearer " + token)
        if start:
            request.add_header("Range", f"bytes={start}-{size - 1}")
        opener = urllib.request.build_opener(CredentialRedirect())
        try:
            with opener.open(request, timeout=60) as response:
                if start and (response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{size - 1}/{size}"):
                    raise ValueError("server did not honor the exact pinned payload resume range")
                if not start and response.status != 200:
                    raise ValueError("unexpected fresh payload response")
                remaining = size - start
                with partial.open("ab") as target:
                    while remaining:
                        block = response.read(min(CHUNK, remaining))
                        if not block:
                            raise OSError("payload transfer ended before its pinned length")
                        target.write(block)
                        remaining -= len(block)
                    if response.read(1):
                        raise ValueError("payload transfer exceeds its pinned length")
            break
        except (urllib.error.URLError, TimeoutError, OSError):
            if attempt == 4:
                raise RuntimeError("payload transfer failed after bounded retries; partial bytes remain resumable") from None
            print(name, "transfer interrupted; resuming pinned bytes", flush=True)
            time.sleep(min(2 ** attempt, 16))
    if partial.stat().st_size != size or digest(partial) != expected:
        raise ValueError("downloaded payload SHA-256 differs; partial file is retained and not admitted")
    partial.replace(destination)
    print(name, size, "SHA-256 verified", flush=True)
    return {"file": name, "bytes": size, "sha256": expected, "verified": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inventory", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--headroom-gib", type=int, default=64,
                        help="local free-space reserve after all selected outstanding shards (default: 64)")
    args = parser.parse_args()
    if args.headroom_gib < 0:
        raise ValueError("disk headroom must be nonnegative")
    artifact = next(item for item in json.loads(args.inventory.read_text())["artifacts"] if item["repository"] == args.repository)
    output = args.output / artifact["repository"] / artifact["revision"]
    output.mkdir(parents=True, exist_ok=True)
    if output.resolve().is_relative_to(Path(__file__).resolve().parent.parent):
        raise ValueError("released payloads must remain outside the source checkout")
    shards = [(name, record) for name, record in artifact["files"].items() if name.endswith(".safetensors")]
    remaining = 0
    for name, record in shards:
        if not (output / name).exists():
            partial = output / (name + ".partial")
            remaining += max(0, record["bytes"] - (partial.stat().st_size if partial.exists() else 0))
    free = shutil.disk_usage(output).free
    if remaining + args.headroom_gib * 1024 ** 3 > free:
        raise ValueError(f"pinned payloads do not fit while retaining {args.headroom_gib} GiB free; no download starts")
    print(artifact["repository"], artifact["revision"], "shards", len(shards), "remaining bytes", remaining, "free bytes", free, flush=True)
    token = credential()
    base = f"https://huggingface.co/{artifact['repository']}/resolve/{artifact['revision']}/"
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        verified = list(pool.map(lambda item: download(base, token, output, *item), shards))
    report = {"repository": artifact["repository"], "revision": artifact["revision"],
              "all_weight_payload_sha256_verified": True, "files": verified,
              "total_bytes": sum(item["bytes"] for item in verified), "released_model_execution_verified": False}
    (output / "payload-evidence.json").write_text(json.dumps(report, indent=2) + "\n")
    print("all pinned payload digests verified", report["total_bytes"], flush=True)


if __name__ == "__main__":
    try:
        main()
    except urllib.error.HTTPError as error:
        raise SystemExit(f"Hugging Face returned HTTP {error.code}; credentials and redirect URLs are not recorded") from None
