#!/usr/bin/env python3
"""Run one pinned CUDA checkpoint pilot and persist phase-by-phase evidence."""

import argparse
import json
import math
import os
import pathlib
import platform
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from typing import Optional, Sequence

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--prompt-id", default="short_ascii")
    parser.add_argument(
        "--manifest", type=pathlib.Path, default=SCRIPT_DIR / "models.yaml"
    )
    parser.add_argument("--output-root", type=pathlib.Path, required=True)
    parser.add_argument(
        "--model-root", type=pathlib.Path, default=pathlib.Path("/tmp/eredu-models")
    )
    parser.add_argument("--performance-prompt-tokens", type=int, default=128)
    parser.add_argument("--performance-decode-steps", type=int, default=128)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--refresh-model", action="store_true")
    parser.add_argument("--skip-performance", action="store_true")
    args = parser.parse_args(argv)
    if args.performance_prompt_tokens <= 0:
        parser.error("--performance-prompt-tokens must be positive")
    if args.performance_decode_steps <= 0:
        parser.error("--performance-decode-steps must be positive")
    return args


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_case(path: pathlib.Path, case_id: str) -> tuple[dict, dict]:
    import yaml

    manifest = yaml.safe_load(path.read_text())
    matches = [case for case in manifest.get("cases", ()) if case.get("id") == case_id]
    if len(matches) != 1:
        raise ValueError("case {!r} must resolve exactly once".format(case_id))
    case = matches[0]
    if case.get("status") != "enabled":
        raise ValueError("case {!r} is not enabled".format(case_id))
    return manifest, case


def repeat_token_ids(token_ids: Sequence[int], count: int) -> list[int]:
    if not token_ids:
        raise ValueError("cannot construct a performance prompt from zero token IDs")
    return (list(token_ids) * math.ceil(count / len(token_ids)))[:count]


def select_prompt(manifest: dict, prompt_id: str) -> dict:
    prompts = manifest["input_sets"]["text_correctness"]["prompts"]
    matches = [prompt for prompt in prompts if prompt.get("id") == prompt_id]
    if len(matches) != 1:
        raise ValueError("prompt {!r} must resolve exactly once".format(prompt_id))
    prompt = matches[0]
    text = prompt.get("text")
    repeat = prompt.get("repeat", 1)
    if not isinstance(text, str) or not text:
        raise ValueError("prompt {!r} must have non-empty text".format(prompt_id))
    if not isinstance(repeat, int) or repeat <= 0:
        raise ValueError("prompt {!r} repeat must be positive".format(prompt_id))
    return {"id": prompt_id, "text": text * repeat, "repeat": repeat}


def write_json(path: pathlib.Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def command_display(command: Sequence[str]) -> str:
    return " ".join(command)


def run_command(
    command: Sequence[str], phase: dict, log_path: pathlib.Path, allowed=(0,)
) -> int:
    phase.update(
        {
            "status": "running",
            "started_at": utc_now(),
            "command": command_display(command),
            "log": str(log_path),
        }
    )
    started = time.perf_counter()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        process = subprocess.Popen(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        return_code = process.wait()
    phase.update(
        {
            "finished_at": utc_now(),
            "wall_seconds": time.perf_counter() - started,
            "return_code": return_code,
            "status": "passed" if return_code in allowed else "failed",
        }
    )
    if return_code not in allowed:
        raise subprocess.CalledProcessError(return_code, command)
    return return_code


def gpu_inventory() -> list[dict]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,uuid,compute_cap,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.check_output(command, text=True)
    fields = ("index", "name", "uuid", "compute_capability", "memory_mib", "driver")
    return [
        dict(zip(fields, (part.strip() for part in line.split(","))))
        for line in output.splitlines()
        if line.strip()
    ]


def download_checkpoint(case: dict, model_dir: pathlib.Path) -> str:
    from huggingface_hub import snapshot_download

    model = case["model"]
    model_dir.mkdir(parents=True, exist_ok=True)
    return snapshot_download(
        repo_id=model["repo"],
        revision=model["revision"],
        local_dir=model_dir,
        allow_patterns=[
            "*.json",
            "*.jinja",
            "*.model",
            "*.py",
            "*.safetensors",
            "*.tiktoken",
            "*.txt",
        ],
        ignore_patterns=[
            "*.bin",
            "*.gguf",
            "*.msgpack",
            "*.onnx",
            "*.pt",
            "*.pth",
            "original/**",
        ],
        token=os.environ.get("HF_TOKEN"),
    )


def artifact_metrics(path: pathlib.Path) -> dict:
    report = json.loads(path.read_text())
    return {
        "timings": report.get("timings"),
        "memory": report.get("memory"),
        "runtime": report.get("runtime"),
        "input": report.get("input"),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    manifest, case = load_case(args.manifest, args.case)
    prompt = select_prompt(manifest, args.prompt_id)
    output_dir = args.output_root / args.case
    summary_path = output_dir / "pilot-summary.json"
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise ValueError("refusing to replace {}; pass --overwrite".format(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.model_root / args.case
    if args.refresh_model and model_dir.exists():
        shutil.rmtree(model_dir)

    summary = {
        "schema_version": 1,
        "kind": "cuda_checkpoint_pilot",
        "status": "running",
        "started_at": utc_now(),
        "case": args.case,
        "family": case["family"],
        "branch": case["branch"],
        "model": case["model"],
        "hardware_profile": case["hardware_profile"],
        "container_image": os.environ.get("EREDU_VALIDATION_IMAGE"),
        "job_id": os.environ.get("JOB_ID") or os.environ.get("HF_JOB_ID"),
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "gpu": gpu_inventory(),
        },
        "pilot_scope": {
            "correctness_inputs": ["text_correctness." + prompt["id"]],
            "performance_prompt_tokens": args.performance_prompt_tokens,
            "performance_decode_steps": args.performance_decode_steps,
            "measured_runs": 1,
            "qwen3_vl_scope": "text_decoder_only" if case["family"] == "qwen3_vl" else None,
        },
        "phases": {},
        "artifacts": {},
    }
    write_json(summary_path, summary)

    try:
        phase = summary["phases"].setdefault("download", {})
        phase.update({"status": "running", "started_at": utc_now()})
        started = time.perf_counter()
        checkpoint_path = download_checkpoint(case, model_dir)
        phase.update(
            {
                "status": "passed",
                "finished_at": utc_now(),
                "wall_seconds": time.perf_counter() - started,
                "checkpoint_path": checkpoint_path,
            }
        )
        write_json(summary_path, summary)

        actual_prefix = output_dir / "correctness-eredu"
        actual_json = actual_prefix.with_suffix(".json")
        run_command(
            [
                "checkpoint_probe",
                "--model",
                str(model_dir),
                "--device",
                "gpu",
                "--prompt",
                prompt["text"],
                "--decode-steps",
                str(
                    manifest["correctness_profiles"][case["correctness_profile"]][
                        "teacher_forced_decode_steps"
                    ]
                ),
                "--warmup-runs",
                "1",
                "--output",
                str(actual_prefix),
                "--overwrite",
            ],
            summary["phases"].setdefault("eredu_correctness", {}),
            output_dir / "eredu-correctness.log",
        )
        summary["artifacts"]["eredu_correctness"] = str(actual_json)
        write_json(summary_path, summary)

        reference_prefix = output_dir / "correctness-transformers"
        reference_json = reference_prefix.with_suffix(".json")
        profile = manifest["correctness_profiles"][case["correctness_profile"]]
        reference_command = [
            "python",
            str(SCRIPT_DIR / "reference_runner.py"),
            "--probe",
            str(actual_json),
            "--model",
            str(model_dir),
            "--output",
            str(reference_prefix),
            "--device",
            "cuda:0",
            "--dtype",
            "auto" if profile["reference_dtype"] == "checkpoint_native" else "bfloat16",
            "--attn-implementation",
            "auto" if case["family"] in ("gpt_oss", "nemotron_h", "qwen3_vl") else "eager",
            "--prefill-mode",
            case["reference"].get("prefill_mode", "full"),
            "--warmup-runs",
            "1",
            "--overwrite",
        ]
        if case["reference"].get("trust_remote_code"):
            reference_command.append("--trust-remote-code")
        run_command(
            reference_command,
            summary["phases"].setdefault("transformers_correctness", {}),
            output_dir / "transformers-correctness.log",
        )
        summary["artifacts"]["transformers_correctness"] = str(reference_json)
        write_json(summary_path, summary)

        comparison_json = output_dir / "correctness-comparison.json"
        comparison_status = run_command(
            [
                "python",
                str(SCRIPT_DIR / "compare_checkpoints.py"),
                "--actual",
                str(actual_json),
                "--reference",
                str(reference_json),
                "--manifest",
                str(args.manifest),
                "--case",
                args.case,
                "--output",
                str(comparison_json),
                "--overwrite",
            ],
            summary["phases"].setdefault("comparison", {}),
            output_dir / "comparison.log",
            allowed=(0, 1),
        )
        summary["artifacts"]["comparison"] = str(comparison_json)
        write_json(summary_path, summary)

        performance_json = None
        if not args.skip_performance:
            correctness_report = json.loads(actual_json.read_text())
            performance_ids = repeat_token_ids(
                correctness_report["input"]["token_ids"], args.performance_prompt_tokens
            )
            performance_prefix = output_dir / "performance-eredu-128x128"
            performance_json = performance_prefix.with_suffix(".json")
            run_command(
                [
                    "checkpoint_probe",
                    "--model",
                    str(model_dir),
                    "--device",
                    "gpu",
                    "--input-ids",
                    ",".join(str(token) for token in performance_ids),
                    "--decode-steps",
                    str(args.performance_decode_steps),
                    "--warmup-runs",
                    "1",
                    "--output",
                    str(performance_prefix),
                    "--overwrite",
                ],
                summary["phases"].setdefault("eredu_performance", {}),
                output_dir / "eredu-performance.log",
            )
            summary["artifacts"]["eredu_performance"] = str(performance_json)
        summary["evidence"] = {
            "comparison": json.loads(comparison_json.read_text()),
            "eredu_correctness": artifact_metrics(actual_json),
            "transformers_correctness": artifact_metrics(reference_json),
        }
        if performance_json is not None:
            summary["evidence"]["eredu_performance"] = artifact_metrics(performance_json)
        summary["status"] = "passed" if comparison_status == 0 else "correctness_failed"
        return_code = 0 if comparison_status == 0 else 1
    except Exception as error:
        summary["status"] = "error"
        summary["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        return_code = 2
    finally:
        summary["finished_at"] = utc_now()
        write_json(summary_path, summary)
        print("PILOT_SUMMARY=" + json.dumps(summary, sort_keys=True), flush=True)
    return return_code


if __name__ == "__main__":
    sys.exit(main())
