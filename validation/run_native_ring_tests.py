#!/usr/bin/env python3
"""Run native Ring integration tests in bounded, fresh libtest processes.

Fresh processes isolate MLX stream workers and allocator state. Process-group
deadlines also bound child Ring workers. Released-checkpoint cases run separately
with the artifacts they require.
"""

import argparse
import json
import os
import signal
import shutil
import time
from pathlib import Path
import subprocess
import sys
import tempfile
import threading


def test_names(listing: str) -> list[str]:
    return [line.removesuffix(": test") for line in listing.splitlines()
            if line.endswith(": test")]


# Shared contracts exercised with synthetic fixtures and real local ranks.
RING_TESTS = [
    "partition::ring_two_process_partition_load",
    "partition::ring_two_process_bounded_consensus_transport",
    "realtime::moshi_ring_tp2_synthetic_model_parity",
    "realtime::moshi_ring_tp2_schedule_disagreement_publishes_nothing",
    "row_parallel::native_row_parallel_agreement_tp2_rejects_failures_and_retries",
]


def select_tests(executable: str, shard: int, shards: int, filters: list[str]) -> list[str]:
    listing = subprocess.check_output(
        [executable, "--list", "--format=terse", "--ignored"], text=True)
    missing = set(RING_TESTS) - set(test_names(listing))
    if missing:
        raise RuntimeError(f"missing ignored Ring integration tests: {sorted(missing)}")
    names = RING_TESTS
    if filters:
        for pattern in filters:
            if not any(pattern in name for name in names):
                raise RuntimeError(f"test filter matched nothing: {pattern}")
        names = [name for name in names if any(p in name for p in filters)]
    selected = names[shard::shards]
    if not selected:
        raise RuntimeError("no Ring integration tests selected for this shard")
    return selected


def run_batches(executable: str, batch_size: int, *,
                shard: int = 0, shards: int = 1, filters: list[str] = (),
                test_timeout: float = 600, suite_timeout: float = 1800,
                results_dir: Path | None = None, fail_fast: bool = False) -> int:
    names = select_tests(executable, shard, shards, filters)
    environment = dict(os.environ)
    environment.setdefault("MLX_ENABLE_TF32", "0")
    # Unoptimized typed construction needs more than libtest's default stack.
    environment.setdefault("RUST_MIN_STACK", str(16 << 20))
    failed = False
    records = []
    deadline = time.monotonic() + suite_timeout
    if results_dir:
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / "selected.json").write_text(json.dumps(names, indent=2))
        (results_dir / "results.json").write_text("[]\n")
    for start in range(0, len(names), batch_size):
        batch = names[start:start + batch_size]
        print(f"Ring tests {start + 1}–{start + len(batch)} / {len(names)}: "
              + ", ".join(batch), flush=True)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            print("Suite time budget exhausted; remaining tests were NOT run", flush=True)
            failed = True
            break
        began = time.monotonic()
        timed_out = False
        # Ring workers inherit the process group, so timeout kills the whole tree.
        command = [executable, "--test-threads=1", "--exact", "--nocapture", "--ignored", *batch]
        log_path = results_dir / f"{start:04d}.log" if results_dir else None
        output = log_path.open("w") if log_path else tempfile.TemporaryFile(mode="w+")
        with output:
            process = subprocess.Popen(command, env=environment, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, encoding="utf-8",
                                       errors="replace", start_new_session=True)

            def relay():
                for line in process.stdout:
                    output.write(line)
                    output.flush()
                    print(line, end="", flush=True)

            reader = threading.Thread(target=relay)
            reader.start()
            try:
                process.wait(timeout=min(test_timeout, remaining))
            except subprocess.TimeoutExpired:
                timed_out = True
            finally:
                # Also reap worker descendants left by a panicking parent.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
                reader.join()
                process.stdout.close()
        elapsed = time.monotonic() - began
        bad = timed_out or process.returncode != 0
        records.append(dict(tests=batch, seconds=round(elapsed, 3),
                            returncode=process.returncode, timed_out=timed_out))
        failed |= bad
        print(f"{'FAIL' if bad else 'PASS'} in {elapsed:.1f}s"
              + (" (test time limit exceeded)" if timed_out else ""), flush=True)
        if results_dir:
            (results_dir / "results.json").write_text(json.dumps(records, indent=2))
        if bad and fail_fast:
            break
    return int(failed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default="mlx,distributed")
    parser.add_argument("--profile", default="test")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--executable", help="reuse an already built test executable")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--filter", action="append", default=[])
    parser.add_argument("--test-timeout", type=float, default=600)
    parser.add_argument("--suite-timeout", type=float, default=1800)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--export-executable", type=Path,
                        help="build only and copy the executable for isolated CI shards")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if not 0 <= args.shard < args.shards:
        parser.error("--shard must be in [0, --shards)")
    if args.test_timeout <= 0 or args.suite_timeout <= 0:
        parser.error("timeouts must be positive")
    executable = args.executable
    if not executable:
        command = ["cargo", "test", "--locked", "-p", "eredu", "--no-default-features", "--test", "native_ring",
                   "--no-run", "--timings", "--message-format=json", "--profile", args.profile]
        if args.features:
            command += ["--features", args.features]
        result = subprocess.run(command, cwd=Path(__file__).resolve().parents[1],
                                stdout=subprocess.PIPE, text=True)
        for line in result.stdout.splitlines():
            record = json.loads(line)
            if record.get("reason") == "compiler-message":
                print(record["message"].get("rendered", ""), end="", file=sys.stderr)
            if record.get("reason") == "compiler-artifact" and record.get("executable") and record["target"]["name"] == "native_ring":
                executable = record["executable"]
        if result.returncode:
            return result.returncode
        if not executable:
            raise RuntimeError("Cargo produced no Ring integration test executable")
    if args.export_executable:
        args.export_executable.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(executable, args.export_executable)
        return 0
    return run_batches(executable, args.batch_size,
                       shard=args.shard, shards=args.shards, filters=args.filter,
                       test_timeout=args.test_timeout, suite_timeout=args.suite_timeout,
                       results_dir=args.results_dir, fail_fast=args.fail_fast)


if __name__ == "__main__":
    sys.exit(main())
