#!/usr/bin/env python3
"""Bound disposable compiler outputs and retain one snapshot per configuration."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess


def prune(root: Path, limit: int) -> int:
    # Do this only after all Cargo commands finish. Removing outputs lets Cargo
    # rebuild missing units normally; no source, native cache, or lockfile is touched.
    for directory in list(root.rglob("incremental")) + list(root.rglob("package")) + list(root.rglob("examples")):
        if directory.is_dir():
            shutil.rmtree(directory)
    # Test executables are large and are not reused as dependencies. Prefer
    # compiler libraries and metadata over binaries that Cargo can relink.
    for path in root.rglob("*"):
        if path.is_file() and path.parent.name == "deps" and path.suffix in ("", ".exe", ".pdb"):
            path.unlink()
    files = [path for path in root.rglob("*") if path.is_file() and not path.is_symlink()]
    sizes = {path: path.stat().st_size for path in files}
    total = sum(sizes.values())
    for path in sorted(files, key=lambda path: path.stat().st_mtime):
        if total <= limit:
            break
        total -= sizes[path]
        path.unlink()
    return total


def expired(caches: list[dict], limit: int) -> list[int]:
    seen = set()
    total = 0
    remove = []
    for cache in sorted(caches, key=lambda c: c["created_at"], reverse=True):
        key = cache["key"]
        if not key.startswith(("rust-v2-", "rust-v3-")):
            continue
        configuration = key.split("--", 1)[0]
        size = cache["size_in_bytes"]
        if key.startswith("rust-v2-") or configuration in seen or total + size > limit:
            remove.append(cache["id"])
        else:
            seen.add(configuration)
            total += size
    return remove


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prune", "retain"))
    parser.add_argument("target", nargs="?", type=Path)
    parser.add_argument("--max-mib", type=int, default=4096)
    args = parser.parse_args()
    if args.command == "prune":
        if args.target is None or args.target.resolve() == Path.cwd():
            parser.error("prune requires a dedicated compiler-output directory")
        print(f"Retained {prune(args.target, args.max_mib * 1024**2)} bytes")
    else:
        pages = json.loads(subprocess.check_output([
            "gh", "api", "--paginate", "--slurp", "repos/{owner}/{repo}/actions/caches?per_page=100",
        ]))
        caches = [cache for page in pages for cache in page["actions_caches"]]
        for cache_id in expired(caches, args.max_mib * 1024**2):
            subprocess.run(["gh", "api", "--method", "DELETE",
                            f"repos/{{owner}}/{{repo}}/actions/caches/{cache_id}"], check=True)
            print(f"Removed superseded Rust cache {cache_id}")


if __name__ == "__main__":
    main()
