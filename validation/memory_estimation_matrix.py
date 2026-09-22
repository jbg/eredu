#!/usr/bin/env python3
"""Run current-source CLI memory forecasts against fresh-process observations.

Build first: cargo build -p eredu-cli --locked
Run outside the sandbox on Metal. The checkpoint and outputs stay outside source.
macOS time RSS is a process high-water mark; MLX peak is allocator active bytes.
Neither counter measures total unified physical-memory pressure.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--binary", type=Path, default=Path("target/debug/eredu"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-lengths", type=int, nargs="+", default=[32, 2000])
    parser.add_argument("--chunks", type=int, nargs="+", default=[0, 512])
    parser.add_argument("--quantizations", nargs="+", default=["original", "4"])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for quant in args.quantizations:
        for length in args.prompt_lengths:
            for chunk in args.chunks:
                name = f"{quant}-prompt-{length}-chunk-{chunk}"
                telemetry = args.output / f"{name}.telemetry.json"
                estimate = args.output / f"{name}.estimate.json"
                command = [str(args.binary.resolve()), "--model", str(args.model.resolve()),
                           "--no-auto", "--raw", "--max-tokens", "8", "--temperature", "0",
                           "--prefill-chunk-size", str(chunk), "--mlx-cache-limit-bytes", "0",
                           "--telemetry-json", str(telemetry.resolve()),
                           "--memory-report", str(estimate.resolve())]
                if quant != "original":
                    command += ["--quantize", quant]
                started = time.monotonic()
                result = subprocess.run(["/usr/bin/time", "-l", *command], input=" hello" * length,
                                        text=True, capture_output=True)
                (args.output / f"{name}.stdout").write_text(result.stdout)
                (args.output / f"{name}.stderr").write_text(result.stderr)
                rss = re.search(r"(\d+)\s+maximum resident set size", result.stderr)
                footprint = re.search(r"(\d+)\s+peak memory footprint", result.stderr)
                row = {"case": name, "command": command, "prompt": {"repeat": " hello", "count": length},
                       "returncode": result.returncode, "elapsed_seconds": time.monotonic() - started,
                       "process_peak_rss_bytes": int(rss[1]) if rss else None,
                       "process_peak_footprint_bytes": int(footprint[1]) if footprint else None}
                if result.returncode == 0:
                    row["telemetry"] = json.loads(telemetry.read_text())
                    row["estimate"] = json.loads(estimate.read_text())
                rows.append(row)
                (args.output / "matrix.json").write_text(json.dumps(rows, indent=2) + "\n")
                print(f"{name}: exit={result.returncode} RSS={row['process_peak_rss_bytes']}", flush=True)
                if result.returncode:
                    raise SystemExit(f"Failed {name}; see its stderr artifact")


if __name__ == "__main__":
    main()
