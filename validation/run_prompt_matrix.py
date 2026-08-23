#!/usr/bin/env python3
"""Run one checkpoint case across deterministic text prompts in one GPU job."""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from typing import Optional, Sequence

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--manifest", type=pathlib.Path, default=SCRIPT_DIR / "models.yaml")
    parser.add_argument("--output-root", type=pathlib.Path, required=True)
    parser.add_argument(
        "--model-root", type=pathlib.Path, default=pathlib.Path("/tmp/eredu-models")
    )
    parser.add_argument("--prompt-id", action="append", dest="prompt_ids")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def prompt_ids(manifest: dict, selected: list[str] | None) -> list[str]:
    available = [
        prompt["id"]
        for prompt in manifest["input_sets"]["text_correctness"]["prompts"]
    ]
    if selected is None:
        return available
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise ValueError("unknown prompt ids: " + ", ".join(unknown))
    if len(selected) != len(set(selected)):
        raise ValueError("prompt ids must be unique")
    return selected


def write_json(path: pathlib.Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    import yaml

    args = parse_args(argv)
    manifest = yaml.safe_load(args.manifest.read_text())
    selected = prompt_ids(manifest, args.prompt_ids)
    summary_path = args.output_root / "prompt-matrix-summary.json"
    if summary_path.exists() and not args.overwrite:
        raise ValueError("refusing to replace {}; pass --overwrite".format(summary_path))
    summary = {
        "schema_version": 1,
        "kind": "cuda_checkpoint_prompt_matrix",
        "status": "running",
        "started_at": utc_now(),
        "case": args.case,
        "prompt_ids": selected,
        "prompts": {},
    }
    write_json(summary_path, summary)
    any_error = False
    any_failure = False
    for prompt_id in selected:
        prompt_root = args.output_root / "prompts" / prompt_id
        command = [
            sys.executable,
            str(SCRIPT_DIR / "run_pilot_case.py"),
            "--case",
            args.case,
            "--prompt-id",
            prompt_id,
            "--manifest",
            str(args.manifest),
            "--output-root",
            str(prompt_root),
            "--model-root",
            str(args.model_root),
            "--skip-performance",
        ]
        if args.overwrite:
            command.append("--overwrite")
        pilot_path = prompt_root / args.case / "pilot-summary.json"
        try:
            completed = subprocess.run(command, check=False)
            pilot = json.loads(pilot_path.read_text())
            comparison = pilot.get("evidence", {}).get("comparison")
            prompt_summary = {
                "return_code": completed.returncode,
                "status": pilot["status"],
                "token_count": pilot.get("evidence", {})
                .get("eredu_correctness", {})
                .get("input", {})
                .get("token_count"),
                "comparison": comparison,
                "pilot_summary": str(pilot_path),
            }
            any_error |= pilot["status"] == "error"
            any_failure |= pilot["status"] == "correctness_failed"
        except Exception as error:
            prompt_summary = {
                "status": "error",
                "error": {
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                },
                "pilot_summary": str(pilot_path),
            }
            any_error = True
        summary["prompts"][prompt_id] = prompt_summary
        write_json(summary_path, summary)
    summary["status"] = (
        "error" if any_error else "correctness_failed" if any_failure else "passed"
    )
    summary["finished_at"] = utc_now()
    write_json(summary_path, summary)
    print("PROMPT_MATRIX_SUMMARY=" + json.dumps(summary, sort_keys=True), flush=True)
    return 2 if any_error else 1 if any_failure else 0


if __name__ == "__main__":
    sys.exit(main())
