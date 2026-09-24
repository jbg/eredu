#!/usr/bin/env python3
"""Run isolated native retention/cache-policy cases against a pinned checkpoint.

Build native_execution_control first, then pass its executable (not Cargo). Each
case runs in a fresh process to keep allocator caches and first admission isolated.
JSON artifacts and full logs are written outside the source tree by the caller.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import time

CHECKPOINT_SHA256 = "3d80914b903cd6f3cc041208cf20ec46a3224f840c732e5fd7698832b4743d1b"
CHECKPOINT_REVISION = "6767265158422fb8a19c62ceb45f16f05363615b"
MIB = 1024 * 1024


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * MIB), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(command):
    return subprocess.check_output(command, text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True,
                        help="pre-change native parity JSON for positions 128 and 2000")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policies", nargs="+",
                        default=["disabled", str(32 * MIB), str(256 * MIB),
                                 str(1024 * MIB), "unlimited"])
    parser.add_argument("--allocator-caches", nargs="+", type=int,
                        default=[0, 256 * MIB])
    parser.add_argument("--repetitions", type=int, default=1,
                        help="fresh-process timing repetitions; parity runs once per policy/cache")
    parser.add_argument("--build-command", default="CARGO_INCREMENTAL=0 cargo test -p eredu --features mlx,metal --test native_execution_control --no-run --locked --offline --config profile.test.package.eredu-backend-mlx.debug=0 --config profile.test.package.eredu.debug=0",
                        help="exact command used to build the supplied executable")
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_hash = sha256(args.model)
    if checkpoint_hash != CHECKPOINT_SHA256:
        parser.error(f"checkpoint SHA-256 mismatch: {checkpoint_hash}")
    metadata = {
        "checkpoint_revision": CHECKPOINT_REVISION,
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_bytes": args.model.stat().st_size,
        "reference_sha256": sha256(args.reference),
        "binary_sha256": sha256(args.binary),
        "git_revision": command_output(["git", "rev-parse", "HEAD"]),
        "tracked_diff_sha256": hashlib.sha256(subprocess.check_output(["git", "diff", "HEAD", "--"])).hexdigest(),
        "runner_sha256": sha256(Path(__file__)),
        "untracked_file_sha256": {
            name: sha256(Path(name))
            for name in subprocess.check_output(
                ["git", "ls-files", "--others", "--exclude-standard", "-z"]
            ).decode().split("\0")
            if name and Path(name).is_file()
        },
        "rustc": command_output(["rustc", "--version", "--verbose"]),
        "build_command": args.build_command,
        "repetitions": args.repetitions,
        "timing_note": "Each repetition runs in a fresh process, cycling all policies before the next repetition. Each process measures cold/warm requests at both lengths. Decode samples contain four synchronized tokens. Parity runs once per policy/cache. Cached allocator peak is sampled at token boundaries, not native high-water telemetry.",
        "platform": platform.platform(),
        "hardware": command_output(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "memory_bytes": int(command_output(["sysctl", "-n", "hw.memsize"])),
        "cases": [],
    }
    for repetition in range(args.repetitions):
        for cache in args.allocator_caches:
            for policy in args.policies:
                base_name = f"retention-{policy}-allocator-{cache}"
                name = (f"{base_name}-repeat-{repetition + 1}"
                        if args.repetitions > 1 else base_name)
                env = os.environ.copy()
                # Clear all harness overrides so caller shell state cannot contaminate
                # checkpoint identity, dtype, policy, lengths, or baseline writes.
                for key in list(env):
                    if key.startswith("EREDU_LFM2_"):
                        del env[key]
                env.update({
                    "EREDU_LFM2_MEMORY_MODEL": str(args.model.absolute()),
                    "EREDU_LFM2_RETENTION": policy,
                    "EREDU_LFM2_ALLOCATOR_CACHE_BYTES": str(cache),
                    "EREDU_LFM2_MEMORY_TRIM": "1",
                })
                record = {"name": name, "policy": policy, "allocator_cache_bytes": cache,
                          "repetition": repetition + 1}
                for mode, test in [
                    ("memory", "native_lfm2_workspace_forecasts_cover_cold_loaded_and_continued_execution"),
                    ("parity", "native_lfm2_forecast_recalibration_preserves_logits_and_cached_generation"),
                ]:
                    if mode == "parity" and repetition > 0:
                        continue
                    env["EREDU_LFM2_MEMORY_LENGTHS"] = (
                        "128,128,2000,2000,128" if mode == "memory" else "128,2000")
                    if mode == "memory":
                        env["EREDU_LFM2_MEMORY_WRITE"] = str((args.output / f"{name}.json").resolve())
                    else:
                        env.pop("EREDU_LFM2_MEMORY_WRITE", None)
                        env["EREDU_LFM2_PARITY_REFERENCE"] = str(args.reference.resolve())
                    command = [str(args.binary.resolve()), f"lfm2_memory::{test}", "--exact", "--ignored", "--nocapture", "--test-threads=1"]
                    print(f"{name}: {mode}", flush=True)
                    started = time.monotonic()
                    with (args.output / f"{name}-{mode}.log").open("w") as log:
                        result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                                check=False)
                    log_text = (args.output / f"{name}-{mode}.log").read_text()
                    selected_one = re.search(r"test result: ok\. 1 passed; 0 failed;", log_text) is not None
                    record[mode] = {"exit_code": result.returncode, "selected_one_test": selected_one,
                                    "environment": {key: value for key, value in env.items()
                                                    if key.startswith("EREDU_LFM2_")},
                                    "seconds": time.monotonic() - started,
                                    "command": command}
                    if result.returncode or not selected_one:
                        metadata["cases"].append(record)
                        (args.output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
                        raise SystemExit(f"{name} {mode} failed; see {args.output}")
                metadata["cases"].append(record)
                (args.output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"All {len(metadata['cases'])} policy/cache pairs passed; artifacts: {args.output}")


if __name__ == "__main__":
    main()
