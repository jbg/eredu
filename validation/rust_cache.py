#!/usr/bin/env python3
"""Bound disposable compiler outputs and retain one snapshot per configuration."""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess


def prune(root: Path, limit: int) -> int:
    # Do this only after all Cargo commands finish. Removing outputs lets Cargo
    # rebuild missing units normally; no source, native cache, or lockfile is touched.
    if limit < 0:
        raise ValueError("cache limit must be nonnegative")
    for directory in list(root.rglob("incremental")) + list(root.rglob("package")):
        if directory.is_dir():
            shutil.rmtree(directory)
    files = [path for path in root.rglob("*") if path.is_file() and not path.is_symlink()]
    sizes = {path: path.stat().st_size for path in files}
    total = sum(sizes.values())
    # Keep complete Cargo units: metadata, dependency files, fingerprints and
    # executables belong together. File-by-file eviction can strand a costly
    # binary by removing its tiny fingerprint, forcing a full recompilation.
    units: dict[tuple, list[Path]] = {}
    for path in files:
        relative = path.relative_to(root)
        key = (str(relative),)
        for position, part in enumerate(relative.parts):
            match = re.search(r"-([0-9a-f]{16})(?:\.|$)", part)
            if match:
                # deps, .fingerprint, build and examples share their profile
                # directory but use different names for the same unit hash.
                key = (*relative.parts[:max(0, position - 1)], match[1])
                break
        units.setdefault(key, []).append(path)

    def priority(paths):
        # Dependency libraries first; then complete test/example executables;
        # then other outputs. Within each class retain the most recent units.
        libraries = any(p.suffix in (".rlib", ".rmeta", ".so", ".dylib", ".dll") for p in paths)
        binary = any(p.suffix in ("", ".exe") and p.parent.name in ("deps", "examples") for p in paths)
        support = any("build" in p.relative_to(root).parts or p.name == "mlx.metallib" for p in paths)
        return (3 if support else 2 if libraries else 1 if binary else 0,
                max(p.stat().st_mtime_ns for p in paths))

    removed = 0
    for paths in sorted(units.values(), key=priority):
        if total <= limit:
            break
        for path in paths:
            total -= sizes[path]
            path.unlink()
        removed += 1
    print(f"Rust snapshot: {sum(sizes.values())} -> {total} bytes; evicted {removed} complete units")
    return total


def expired(caches: list[dict], limit: int) -> list[int]:
    seen = set()
    total = 0
    remove = []
    for cache in sorted(caches, key=lambda c: c["created_at"], reverse=True):
        key = cache["key"]
        if not key.startswith(("rust-v2-", "rust-v3-", "rust-v4-")):
            continue
        configuration = key.split("--", 1)[0].split("-", 2)[2]
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
