"""Strict pinned source verification shared by independent reference generators."""
import hashlib
import json
from pathlib import Path


def verify_release(directory, require_weights=True, adapter=False):
    repository = Path(__file__).resolve().parents[2]
    releases = json.loads((repository / "eredu-architectures/tests/fixtures/nemotron_labs_diffusion/releases.json").read_text())
    manifest = json.loads((directory / "manifest.json").read_text())
    release = next(item for item in releases if item["repo"] == manifest["id"])
    if manifest["sha"] != release["revision"]:
        raise ValueError("reference source revision differs from the pinned release")
    records = list(release["assets"])
    if require_weights:
        for item in release["weight_files"]:
            if item["rfilename"].startswith("linear_spec_lora/") and not adapter:
                continue
            records.append({"path":item["rfilename"],"size":item["size"],"algorithm":"sha256","digest":item["lfs"]["sha256"]})
    for item in records:
        file = directory / item["path"]
        if file.stat().st_size != item["size"]:
            raise ValueError(f"reference artifact size differs: {file}")
        digest = hashlib.sha256() if item["algorithm"] == "sha256" else hashlib.sha1()
        if item["algorithm"] == "git_blob_sha1":
            digest.update(f"blob {item['size']}\0".encode())
        with file.open("rb") as source:
            while block := source.read(8*1024*1024):
                digest.update(block)
        if digest.hexdigest() != item["digest"]:
            raise ValueError(f"reference artifact hash differs: {file}")
    print(f"verified pinned reference {release['repo']}@{release['revision']}",flush=True)
    return release


def verify_compressed(directory):
    """Verify a pinned conversion and every retained metadata/weight artifact."""
    repository = Path(__file__).resolve().parents[2]
    records = json.loads((repository / "eredu-architectures/tests/fixtures/nemotron_labs_diffusion/compressed-conversions.json").read_text())
    manifest = json.loads((directory / "huggingface-manifest.json").read_text())
    record = next(item for item in records if item["repo"] == manifest["id"])
    if record["revision"] != manifest["sha"]:
        raise ValueError("compressed source revision differs from its pin")
    assets = record["assets"] + [{"path":item["rfilename"],"size":item["size"],"algorithm":"sha256","digest":item["lfs"]["sha256"]} for item in record["weight_files"]]
    for item in assets:
        path = directory / item["path"]
        if path.stat().st_size != item["size"]:
            raise ValueError(f"compressed artifact size differs: {path}")
        with path.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        if digest != item["digest"]:
            raise ValueError(f"compressed artifact hash differs: {path}")
    print(f"verified compressed reference {record['repo']}@{record['revision']}", flush=True)
    return record
