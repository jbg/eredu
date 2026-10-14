"""Download pinned public reference artifacts outside tracked source and verify hashes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import urllib.request


def download(url, destination, size, digest, lfs):
    def checksum(path):
        value = hashlib.sha256() if lfs else hashlib.sha1()
        if not lfs:
            value.update(f"blob {size}\0".encode())
        with path.open("rb") as source:
            while block := source.read(8*1024*1024):
                value.update(block)
        return value.hexdigest()
    if destination.exists() and destination.stat().st_size == size and checksum(destination) == digest:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    for attempt in range(5):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            request = urllib.request.Request(url,headers={"Range":f"bytes={offset}-"} if offset else {})
            with urllib.request.urlopen(request,timeout=120) as response:
                mode = "ab" if offset and response.status == 206 else "wb"
                with partial.open(mode) as output:
                    while block := response.read(8*1024*1024):
                        output.write(block)
            if partial.stat().st_size != size or checksum(partial) != digest:
                raise ValueError(f"artifact hash or size differs: {destination}")
            partial.replace(destination)
            print(f"verified {destination.name}: {size} bytes {digest}",flush=True)
            return
        except Exception:
            if attempt == 4:
                raise
            time.sleep(min(2**attempt,8))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("output",type=Path)
    parser.add_argument("--weights",nargs="*",default=[])
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[2]
    if args.output.resolve().is_relative_to(repository):
        parser.error("released weights must be outside tracked source")
    releases = json.loads((repository / "eredu-architectures/tests/fixtures/nemotron_labs_diffusion/releases.json").read_text())
    assets = {"config.json","tokenizer.json","tokenizer_config.json","special_tokens_map.json",
              "chat_template.jinja","processor_config.json","model.safetensors.index.json",
              "modeling_ministral.py","modeling_nemotron_labs_diffusion.py",
              "configuration_nemotron_labs_diffusion.py","modeling_nemotron_labs_diffusion_vlm.py",
              "configuration_nemotron_labs_diffusion_vlm.py","tokenization_nemotron_labs_diffusion_vlm.py",
              "image_processing.py","chat_utils.py","linear_spec_lora/adapter_config.json"}
    jobs = []
    records = []
    for release in releases:
        name = release["repo"].removeprefix("nvidia/Nemotron-Labs-Diffusion-")
        directory = args.output / name
        directory.mkdir(parents=True,exist_ok=True)
        manifest_path = directory / "manifest.json"
        with urllib.request.urlopen(f"https://huggingface.co/api/models/{release['repo']}/revision/{release['revision']}?blobs=true",timeout=60) as response:
            manifest = json.load(response)
        assert manifest["sha"] == release["revision"]
        manifest_path.write_text(json.dumps(manifest,indent=2)+"\n")
        selected = {entry["rfilename"] for entry in release["weight_files"]} if name in args.weights else set()
        record = {"repo":release["repo"],"revision":release["revision"],"assets":[]}
        for entry in manifest["siblings"]:
            filename = entry["rfilename"]
            if filename not in assets and filename not in selected:
                continue
            expected = next((item for item in release["weight_files"] if item["rfilename"] == filename),None)
            if expected is not None:
                assert expected["lfs"]["sha256"] == entry["lfs"]["sha256"] and expected["size"] == entry["size"]
            digest = entry["lfs"]["sha256"] if "lfs" in entry else entry["blobId"]
            asset = next((item for item in release["assets"] if item["path"] == filename), None)
            if asset is not None:
                assert asset["size"] == entry["size"] and asset["digest"] == digest
            url = f"https://huggingface.co/{release['repo']}/resolve/{release['revision']}/{filename}"
            jobs.append((url,directory / filename,entry["size"],digest,"lfs" in entry))
            record["assets"].append({"path":filename,"size":entry["size"],"digest":digest,"algorithm":"sha256" if "lfs" in entry else "git_blob_sha1"})
        records.append(record)
    with ThreadPoolExecutor(max_workers=3) as executor:
        for result in executor.map(lambda job: download(*job),jobs):
            pass
    (args.output / "verified-assets.json").write_text(json.dumps(records,indent=2)+"\n")


if __name__ == "__main__":
    main()
