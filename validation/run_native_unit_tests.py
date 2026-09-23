#!/usr/bin/env python3
"""Run every backend unit test in bounded, fresh libtest processes.

MLX retains stream workers for the process lifetime. The full backend suite
creates enough independent streams to exceed macOS's thread limit even when
tests run sequentially. Batching bounds that accumulation without skipping tests
or changing production stream ownership. Explicitly ignored tests remain ignored.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys


def test_names(listing: str) -> list[str]:
    return [line.removesuffix(": test") for line in listing.splitlines()
            if line.endswith(": test")]


def run_batches(executable: str, batch_size: int) -> int:
    listing = subprocess.check_output(
        [executable, "--list", "--format=terse"], text=True,
    )
    names = test_names(listing)
    if not names:
        raise RuntimeError("backend unit test executable listed no tests")
    failed = False
    for start in range(0, len(names), batch_size):
        batch = names[start:start + batch_size]
        print(f"Backend tests {start + 1}–{start + len(batch)} / {len(names)}", flush=True)
        result = subprocess.run([executable, "--exact", *batch])
        failed |= result.returncode != 0
    return int(failed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default="metal,accelerate")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--executable", help="reuse an already built test executable")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    executable = args.executable
    if not executable:
        command = ["cargo", "test", "--locked", "-p", "eredu-backend-mlx", "--lib",
                   "--no-run", "--timings", "--message-format=json"]
        if args.features:
            command += ["--features", args.features]
        result = subprocess.run(command, cwd=Path(__file__).resolve().parents[1],
                                stdout=subprocess.PIPE, text=True)
        for line in result.stdout.splitlines():
            record = json.loads(line)
            if record.get("reason") == "compiler-message":
                print(record["message"].get("rendered", ""), end="", file=sys.stderr)
            if record.get("reason") == "compiler-artifact" and record.get("executable"):
                executable = record["executable"]
        if result.returncode:
            return result.returncode
        if not executable:
            raise RuntimeError("Cargo produced no backend unit test executable")
    return run_batches(executable, args.batch_size)


if __name__ == "__main__":
    sys.exit(main())
