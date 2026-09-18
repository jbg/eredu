#!/usr/bin/env python3
"""Validate released ordinary tool calls through prepared_chat_generate.

Uses an existing binary and pinned local checkpoint. It neither downloads
artifacts nor selects another generation implementation after a refusal.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import pathlib
import platform
import re
import signal
import subprocess
import sys
import time


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def verify_checkpoint(root, evidence_path):
    evidence_bytes = evidence_path.read_bytes()
    evidence = json.loads(evidence_bytes)
    pinned = evidence.get("checkpoint", evidence)
    verified = []
    for entry in pinned["files"]:
        relative = pathlib.Path(entry["name"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("checkpoint manifest contains an escaping path")
        # Standard checkpoint-cache snapshots can contain symlinks to blobs.
        path = root / relative
        size = path.stat().st_size
        if size != entry["size"]:
            raise ValueError(f"checkpoint size mismatch: {entry['name']}")
        algorithm = entry["digest_algorithm"]
        if algorithm == "sha256":
            digest = hashlib.sha256()
        elif algorithm == "git-blob-sha1":
            digest = hashlib.sha1(f"blob {size}\0".encode())
        else:
            raise ValueError(f"unsupported pinned digest: {algorithm}")
        actual_sha256 = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
                actual_sha256.update(chunk)
        if digest.hexdigest() != entry["digest"]:
            raise ValueError(f"checkpoint digest mismatch: {entry['name']}")
        verified.append({**entry, "actual_sha256": actual_sha256.hexdigest()})
    return {
        "repository": pinned["repository"],
        "revision": pinned["revision"],
        "pinned_record": str(evidence_path),
        "pinned_record_sha256": hashlib.sha256(evidence_bytes).hexdigest(),
        "files": verified,
    }


def request(policy, max_tokens, chunk):
    return {
        "messages": [
            {"role": "system", "content": "You are a sensor logging assistant. Use the reading function to save the sensor value the user gives you."},
            {"role": "user", "content": "The sensor value is 17. Please record this reading."},
        ],
        "tools": [{
            "type": "function",
            "function": {
                "name": "reading",
                "description": "Record the requested integer value.",
                "parameters": {
                    "type": "object",
                    "properties": {"value": {"type": "integer", "enum": [17]}},
                    "required": ["value"],
                    "additionalProperties": False,
                },
            },
        }],
        "tool_choice": policy,
        "max_tokens": max_tokens,
        "seed": 0,
        "prefill_chunk_positions": chunk,
        "enable_thinking": False,
    }


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError(f"non-JSON numeric constant: {value}")


def validate_events(events):
    active = None
    calls = []
    visible = []
    reasoning = []
    finished = None
    for event in events:
        if finished is not None:
            raise ValueError("semantic event after Finished")
        if event == "ToolCallEnd":
            if active is None:
                raise ValueError("ToolCallEnd without a matching start")
            active["arguments"] = json.loads(
                "".join(active.pop("fragments")),
                object_pairs_hook=unique_object, parse_constant=reject_constant,
            )
            calls.append(active)
            active = None
            continue
        if not isinstance(event, dict) or len(event) != 1:
            raise ValueError(f"invalid semantic event: {event!r}")
        kind, value = next(iter(event.items()))
        if kind == "ToolCallStart":
            if active is not None:
                raise ValueError("overlapping tool calls in a single-call request")
            if value["index"] != len(calls) or not value["id"]:
                raise ValueError("invalid tool-call index or identifier")
            active = {**value, "fragments": []}
        elif kind == "ToolArgumentsDelta":
            if active is None or value["index"] != active["index"]:
                raise ValueError("tool-argument delta has no matching call")
            active["fragments"].append(value["json_fragment"])
        elif kind == "Finished":
            if active is not None:
                raise ValueError("generation finished inside a tool call")
            finished = value["reason"]
        elif kind == "TextDelta":
            if active is not None:
                raise ValueError("visible text inside an active tool call")
            visible.append(value)
        elif kind == "ReasoningDelta":
            if active is not None:
                raise ValueError("reasoning inside an active tool call")
            reasoning.append(value)
        else:
            raise ValueError(f"unknown semantic event: {kind}")
    if finished not in ("eos", "grammar_complete", "stop_sequence"):
        raise ValueError(f"tool turn did not complete: {finished!r}")
    if len(calls) != 1 or calls[0]["name"] != "reading" or calls[0]["arguments"] != {"value": 17}:
        raise ValueError(f"expected one reading(value=17) call, received {calls!r}")
    return {
        "tool_calls": calls,
        "finish_reason": finished,
        "visible_text": "".join(visible),
        "reasoning_text": "".join(reasoning),
    }


def run_policy(args, policy):
    request_path = args.output / f"{policy}.request.json"
    stdout_path = args.output / f"{policy}.events.jsonl"
    stderr_path = args.output / f"{policy}.stderr.log"
    template = getattr(args, "request_template", None)
    if template is None:
        document = request(policy, args.max_tokens, args.chunk)
    else:
        document = json.loads(template.read_text())
        document.update(tool_choice=policy, max_tokens=args.max_tokens,
                        prefill_chunk_positions=args.chunk)
    write_json(request_path, document)
    command = [
        str(args.binary), str(args.checkpoint), str(request_path),
        str(args.capacity), args.device,
    ]
    # Darwin counters cover the whole subprocess, including loading and I/O.
    measured_command = ["/usr/bin/time", "-l", *command] if sys.platform == "darwin" else command
    result = {"policy": policy, "command": measured_command, "status": "failed"}
    started = time.monotonic()
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        process = subprocess.Popen(
            measured_command, stdout=stdout, stderr=stderr, start_new_session=True,
        )
        try:
            result["returncode"] = process.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            # The timing wrapper and native child form one owned process group.
            # Killing only the wrapper could leave an accelerator job running.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            result["returncode"] = process.wait()
            result["timed_out"] = True
            result["error"] = "process timeout"
    result["wall_seconds"] = time.monotonic() - started
    stderr = stderr_path.read_text()
    result.update({
        "request": json.loads(request_path.read_text()),
        "request_sha256": sha256(request_path),
        "stdout_sha256": sha256(stdout_path),
        "stderr_sha256": sha256(stderr_path),
        "stderr": stderr,
    })
    counters = {}
    for key, label in [("maximum_resident_set_size_bytes", "maximum resident set size"),
                       ("peak_memory_footprint_bytes", "peak memory footprint")]:
        match = re.search(r"(\d+)\s+" + label, stderr)
        if match:
            counters[key] = int(match.group(1))
    result["process_memory"] = {"scope": "whole subprocess; not framework-domain usage", **counters}
    try:
        result["events"] = [json.loads(line) for line in stdout_path.read_text().splitlines() if line.strip()]
        if result.get("timed_out") or result.get("returncode") != 0:
            raise ValueError(f"native process failed: {result.get('returncode')!r}")
        result["semantic_result"] = validate_events(result["events"])
        terminal = re.search(r"committed_tokens=(\d+) finish=(\w+)", stderr)
        debug_reason = {"eos": "Eos", "grammar_complete": "GrammarComplete", "stop_sequence": "StopSequence"}[
            result["semantic_result"]["finish_reason"]
        ]
        if terminal is None or terminal.group(2) != debug_reason:
            raise ValueError("public result and Finished event disagree")
        result["committed_tokens"] = int(terminal.group(1))
        if not 0 < result["committed_tokens"] <= args.max_tokens:
            raise ValueError("committed-token count is outside the configured limit")
        result["status"] = "passed"
    except (ValueError, KeyError, TypeError) as error:
        result["error"] = str(error)
    write_json(args.output / f"{policy}.result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=pathlib.Path, required=True)
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--pinned-record", type=pathlib.Path, required=True)
    parser.add_argument("--source-root", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--request-template", type=pathlib.Path,
                        help="sensor request with optional prepared media; policy and limits come from flags")
    parser.add_argument("--capacity", type=int, required=True)
    parser.add_argument("--device", default="metal:0")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--chunk", type=int, default=128)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--policies", choices=("required", "auto"), nargs="+", default=["required", "auto"])
    args = parser.parse_args()
    for name in ("capacity", "max_tokens", "chunk", "timeout"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("binary", "checkpoint", "pinned_record", "source_root", "output"):
        setattr(args, name, getattr(args, name).resolve())
    if args.request_template is not None:
        args.request_template = args.request_template.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "record.json").exists():
        parser.error("output already contains record.json; choose a fresh evidence directory")
    example = args.source_root / "eredu/examples/prepared_chat_generate.rs"
    record = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "validator_command": [sys.executable, *sys.argv],
        "working_directory": str(pathlib.Path.cwd()),
        "scope": "released public ordinary prepared-chat tools; no drafter or numerical-reference rerun",
        "checkpoint": verify_checkpoint(args.checkpoint, args.pinned_record),
        "binary": {"path": str(args.binary), "size": args.binary.stat().st_size, "sha256": sha256(args.binary)},
        "sources": {"validator_sha256": sha256(pathlib.Path(__file__)), "example_sha256": sha256(example)},
        "environment": {
            "system": platform.system(), "macos": platform.mac_ver()[0], "machine": platform.machine(),
            **{name: os.environ[name] for name in (
                "RUST_MIN_STACK", "DEVELOPER_DIR", "SDKROOT", "CLANG_MODULE_CACHE_PATH",
            ) if name in os.environ},
        },
        "capacity_bytes": args.capacity,
        "capacity_scope": "enforced framework working-memory limit; not an observed memory count",
        "runs": [],
    }
    if args.request_template is not None:
        record["request_template"] = {
            "path": str(args.request_template),
            "sha256": sha256(args.request_template),
        }
    write_json(args.output / "record.json", record)
    for policy in args.policies:
        record["runs"].append(run_policy(args, policy))
        write_json(args.output / "record.json", record)
        print(f"{policy}: {record['runs'][-1]['status']}", flush=True)
    record["status"] = "passed" if all(run["status"] == "passed" for run in record["runs"]) else "failed"
    write_json(args.output / "record.json", record)
    return 0 if record["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
