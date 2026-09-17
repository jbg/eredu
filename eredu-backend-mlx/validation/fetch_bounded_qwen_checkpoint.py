"""Download and verify a pinned official checkpoint outside the tracked tree.

Usage: python fetch_bounded_qwen_checkpoint.py /private/tmp/qwen-reference
"""
import argparse
import hashlib
import json
import shutil
import urllib.parse
import urllib.request
from pathlib import Path

REPOSITORY = "Qwen/Qwen3.5-0.8B"
REVISION = "2fc06364715b967f1860aea9cf38778875588b17"


def digest_file(path, lfs):
    digest = hashlib.sha256() if lfs else hashlib.sha1()
    if not lfs:
        digest.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    url = f"https://huggingface.co/api/models/{REPOSITORY}/revision/{REVISION}?blobs=true"
    with urllib.request.urlopen(url, timeout=120) as response:
        metadata = json.load(response)
    if metadata["id"] != REPOSITORY or metadata["sha"] != REVISION:
        raise ValueError("Publisher metadata does not match the pinned revision")
    root = args.destination
    root.mkdir(parents=True, exist_ok=True)
    manifest = {"repository": REPOSITORY, "revision": REVISION, "files": []}
    for entry in metadata["siblings"]:
        name = entry["rfilename"]
        if Path(name).name != name:
            raise ValueError("Unexpected nested checkpoint filename")
        path = root / name
        lfs = entry.get("lfs", {}).get("sha256")
        expected = lfs or entry["blobId"]
        if not path.exists() or path.stat().st_size != entry["size"] or digest_file(path, lfs) != expected:
            url = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{urllib.parse.quote(name)}?download=true"
            print("Downloading", name, entry["size"], flush=True)
            partial = path.with_suffix(path.suffix + ".partial")
            with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as output:
                shutil.copyfileobj(response, output, 4 << 20)
            if partial.stat().st_size != entry["size"] or digest_file(partial, lfs) != expected:
                raise ValueError(f"Downloaded checkpoint digest mismatch: {name}")
            partial.replace(path)
        observed = digest_file(path, lfs)
        manifest["files"].append({"name": name, "size": entry["size"],
                                  "digest_algorithm": "sha256" if lfs else "git-blob-sha1",
                                  "digest": observed})
        print("Verified", name, observed, flush=True)
    (root / "eredu-provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Verified complete checkpoint:", root, flush=True)


if __name__ == "__main__":
    main()
