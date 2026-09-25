#!/usr/bin/env python3
"""Measure the pinned GGUF's real projection workloads in isolated processes.

Build projection_baseline in release mode first. Ordinary inference, diagnostic
capture/replay, and native kernel selection tracing use separate processes.
Tracing is never enabled in timing runs. Large logs and raw data stay in --output.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time

CHECKPOINT_REVISION = "6767265158422fb8a19c62ceb45f16f05363615b"
CHECKPOINT_SHA256 = "3d80914b903cd6f3cc041208cf20ec46a3224f840c732e5fd7698832b4743d1b"
BUILD = "CARGO_INCREMENTAL=0 cargo build -p eredu --release --example projection_baseline --no-default-features --features mlx,metal,projection-profiling --locked --offline"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def command_output(command):
    return subprocess.check_output(command, text=True).strip()


def kernel_selections(log):
    """Only count selections bracketed by a synchronized replay measurement."""
    selections = {}
    active = None
    for line in log.splitlines():
        if line.startswith("[projection-phase-begin] "):
            if active is not None:
                raise ValueError("overlapping projection trace markers")
            active = tuple(line.split()[1:])
            selections.setdefault(active[:3], Counter())
        elif line.startswith("[projection-phase-end] "):
            if active != tuple(line.split()[1:]):
                raise ValueError("unmatched projection trace marker")
            active = None
        elif line.startswith("[mlx-metal-kernel] ") and active:
            selections[active[:3]][line.split(" ", 1)[1]] += 1
    if active is not None:
        raise ValueError("unterminated projection trace")
    if not selections or any(not kernels for kernels in selections.values()):
        raise ValueError("missing native selection evidence in a replay phase")
    return selections


def summarize(capture, baseline, trace):
    selections = kernel_selections(trace)
    if baseline["ordinary"][0]["tokens"] != capture["ordinary"][0]["tokens"]:
        raise ValueError("capture changed ordinary tokens")
    cases = []
    for case in capture["replays"]:
        if not case["bitwise_equal"]:
            raise ValueError("projection arithmetic mismatch")
        phases = []
        for phase in case["phases"]:
            warm = [sample for sample in phase["samples"] if not sample["first_sample"]]
            keys = (case["label"], str(case["rows"]), phase["phase"])
            phases.append({
                "phase": phase["phase"], "warm_samples": len(warm),
                "preconverted_weight_bytes": phase["preconverted_weight_bytes"],
                "first_sample": phase["samples"][0],
                "median_build_wall_ms": statistics.median(s["build_wall_ms"] for s in warm),
                "median_total_wall_ms": statistics.median(s["total_wall_ms"] for s in warm),
                "median_gpu_evaluation_ms": statistics.median(s["gpu_evaluation_ms"] for s in warm),
                "min_total_wall_ms": min(s["total_wall_ms"] for s in warm),
                "max_total_wall_ms": max(s["total_wall_ms"] for s in warm),
                "peak_growth_bytes": max(s["peak_growth_bytes"] for s in warm),
                "kernel_selections": dict(selections[keys]),
            })
        cases.append({**{key: value for key, value in case.items() if key != "phases"}, "phases": phases})
    return {"positions": capture["positions"], "retention_policy": capture["retention_policy"],
            "ordinary": baseline["ordinary"], "classes": capture["classes"], "replays": cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policies", nargs="+", default=["disabled", "268435456", "unlimited"])
    parser.add_argument("--positions", nargs="+", type=int, default=[128, 2000])
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--build-command", default=BUILD)
    args = parser.parse_args()
    if args.samples < 1 or any(n < 1 for n in args.positions):
        parser.error("samples/positions must be positive")
    if sha256(args.model) != CHECKPOINT_SHA256:
        parser.error("checkpoint SHA-256 differs from pinned official BF16 GGUF")
    args.output.mkdir(parents=True, exist_ok=True)
    patch = Path("safemlx-sys/src/mlx-c/patches/mlx-metal-kernel-selection-trace.patch")
    metadata = {
        "schema_version": 1, "checkpoint_revision": CHECKPOINT_REVISION,
        "checkpoint_sha256": CHECKPOINT_SHA256, "checkpoint_bytes": args.model.stat().st_size,
        "binary_sha256": sha256(args.binary), "runner_sha256": sha256(__file__),
        "native_trace_patch_sha256": sha256(patch),
        "mlx_archive_sha256": sha256("safemlx-sys/vendor/mlx-v0.32.0.tar.gz"),
        "git_revision": command_output(["git", "rev-parse", "HEAD"]),
        "tracked_diff_sha256": hashlib.sha256(subprocess.check_output(["git", "diff", "HEAD", "--"])).hexdigest(),
        "untracked_file_sha256": {name: sha256(name) for name in subprocess.check_output(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"]).decode().split("\0")
            if name and Path(name).is_file()},
        "build_command": args.build_command, "rustc": command_output(["rustc", "--version", "--verbose"]),
        "platform": platform.platform(), "hardware": command_output(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "memory_bytes": int(command_output(["sysctl", "-n", "hw.memsize"])),
        "notes": ["Release build; sequential isolated processes, allocator cache disabled.",
                  "Native selection trace logs pipeline requests during graph evaluation, not GPU event counts or durations.",
                  "Kernel tracing runs separately from timing. All serialized outputs use exact F32 bit comparisons.",
                  "Projection representatives use real model arrays; short verification-shaped workloads slice leading prefill rows.",
                  "Ordinary baseline is eight predictions: cold request followed by reset/warm requests. Loading is excluded.",
                  "Captured workload counts are per projection class; representative latency times count is not an end-to-end attribution."],
        "runs": [], "cases": [],
    }
    for policy in args.policies:
        for positions in args.positions:
            base = f"retention-{policy}-positions-{positions}"
            payloads = {}
            trace_text = None
            for mode in ["baseline", "capture", "trace"]:
                name = f"{base}-{mode}"
                output = args.output / f"{name}.json"
                log = args.output / f"{name}.log"
                env = os.environ.copy()
                # Make selection and timing controls reproducible, independent of shell.
                for key in list(env):
                    if key.startswith(("EREDU_LFM2_", "MLX_")):
                        del env[key]
                if mode == "trace":
                    env["MLX_METAL_LOG_KERNEL_SELECTION"] = "1"
                command = [str(args.binary.resolve()), str(args.model.absolute()), str(output.resolve()),
                           "capture" if mode == "trace" else mode, policy, str(positions),
                           str(1 if mode == "trace" else args.samples)]
                print(name, flush=True)
                started = time.monotonic()
                with log.open("w") as stream:
                    result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
                metadata["runs"].append({"name": name, "command": command, "exit_code": result.returncode,
                    "elapsed_seconds": time.monotonic() - started, "log_sha256": sha256(log),
                    "kernel_tracing": mode == "trace"})
                (args.output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
                if result.returncode:
                    raise SystemExit(f"failed: {name}; see {log}")
                payload = json.loads(output.read_text())
                if mode == "trace":
                    trace_text = log.read_text()
                    if payload["classes"] != payloads["capture"]["classes"]:
                        raise ValueError("trace and timing projection classes differ")
                    if payload["ordinary"][0]["tokens"] != payloads["baseline"]["ordinary"][0]["tokens"]:
                        raise ValueError("kernel selection tracing changed tokens")
                else:
                    payloads[mode] = payload
            metadata["cases"].append(summarize(payloads["capture"], payloads["baseline"], trace_text))
            (args.output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"All {len(metadata['cases'])} policy/position cases passed; {args.output / 'manifest.json'}")


if __name__ == "__main__":
    main()
