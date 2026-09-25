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


def validate_prototype_phases(phases):
    """Require the same native geometry/reduction and output/partial allocations."""
    by_name = {phase["phase"]: phase for phase in phases}
    if "mixed_storage_prototype" in by_name:
        prototype = by_name["mixed_storage_prototype"]
        reference = by_name["preconverted_gemm"]
        # Storage is the only specialization change; split-K accumulator
        # selections and all geometry/tail keys must remain identical.
        normalized = {key.replace("_storage_bfloat16", "").replace("_storage_float16", "")
                      for key in prototype["kernel_selections"]}
        if normalized != set(reference["kernel_selections"]):
            raise ValueError("prototype changed native dispatch beyond weight storage")
        if prototype["peak_growth_bytes"] != reference["peak_growth_bytes"]:
            raise ValueError("prototype allocated beyond the native output/partial workspace")


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
        validate_prototype_phases(phases)
        cases.append({**{key: value for key, value in case.items() if key != "phases"}, "phases": phases})
    return {"positions": capture["positions"], "retention_policy": capture["retention_policy"],
            "ordinary": baseline["ordinary"], "classes": capture["classes"], "replays": cases}



def compare_reference_case(case, reference):
    """Compare with pre-patch evidence, not only a same-build arithmetic oracle."""
    if case["classes"] != reference["classes"]:
        raise ValueError("projection classes differ from reference manifest")
    fingerprint = lambda item: [(r["label"], r["rows"], r["output_f32_bits_sha256"])
                                for r in item["replays"]]
    if fingerprint(case) != fingerprint(reference):
        raise ValueError("projection output bits differ from reference manifest")
    expected = reference["ordinary"][0]["tokens"]
    if any(request["tokens"] != expected for request in case["ordinary"]):
        raise ValueError("generation tokens differ from reference manifest")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dispatch", action="store_true",
                        help="select the diagnostic pre-integration cast-GEMM path")
    parser.add_argument("--mixed-storage-prototype", action="store_true",
                        help="also require exact replay of the loader-only native GEMM prototype")
    parser.add_argument("--disable-tf32", action="store_true",
                        help="explicitly select native SIMD FP32 dispatch on NAX-capable hardware")
    parser.add_argument("--reference-manifest", type=Path,
                        help="require original projection fingerprints and cached generation tokens")
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
    reference = json.loads(args.reference_manifest.read_text()) if args.reference_manifest else None
    if reference and reference["checkpoint_sha256"] != CHECKPOINT_SHA256:
        parser.error("reference manifest uses a different checkpoint")
    args.output.mkdir(parents=True, exist_ok=True)
    patch = Path("safemlx-sys/src/mlx-c/patches/mlx-metal-kernel-selection-trace.patch")
    metadata = {
        "schema_version": 1, "mixed_storage_prototype": args.mixed_storage_prototype,
        "tf32_disabled": args.disable_tf32, "reference_dispatch": args.reference_dispatch,
        "reference_manifest_sha256": sha256(args.reference_manifest) if reference else None,
        "native_mixed_storage_patch_sha256": sha256(
            "safemlx-sys/src/mlx-c/patches/mlx-metal-mixed-storage-gemm.patch"),
        "checkpoint_revision": CHECKPOINT_REVISION,
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
                if args.disable_tf32:
                    env["MLX_ENABLE_TF32"] = "0"
                if mode == "trace":
                    env["MLX_METAL_LOG_KERNEL_SELECTION"] = "1"
                execution_mode = "baseline" if mode == "baseline" else (
                    "capture-prototype" if args.mixed_storage_prototype else "capture")
                command = [str(args.binary.resolve()), str(args.model.absolute()), str(output.resolve()),
                           execution_mode, policy, str(positions),
                           str(1 if mode == "trace" else args.samples),
                           "reference" if args.reference_dispatch else "mixed"]
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
            case = summarize(payloads["capture"], payloads["baseline"], trace_text)
            if reference:
                reference_case = next(c for c in reference["cases"]
                                      if c["positions"] == positions and c["retention_policy"] == policy)
                compare_reference_case(case, reference_case)
                case["matches_reference_manifest"] = True
            metadata["cases"].append(case)
            (args.output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"All {len(metadata['cases'])} policy/position cases passed; {args.output / 'manifest.json'}")


if __name__ == "__main__":
    main()
