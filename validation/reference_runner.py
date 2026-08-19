#!/usr/bin/env python3
"""Run a Transformers reference along an exact Eredu probe token path."""

import argparse
import json
import os
import pathlib
import platform
import resource
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence


@dataclass
class PhaseTiming:
    wall_seconds: float
    device_seconds: Optional[float]


@dataclass
class ReferenceRun:
    prefill_logits: object
    decode_logits: object
    greedy_token_ids: List[int]
    prefill_timing: PhaseTiming
    decode_timings: List[PhaseTiming]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=pathlib.Path, required=True)
    parser.add_argument(
        "--model",
        help="Checkpoint directory or Hub repository; defaults to probe model.checkpoint_path",
    )
    parser.add_argument("--revision", help="Immutable Hub commit when --model is a repository")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument(
        "--attn-implementation",
        choices=("auto", "eager", "sdpa", "flash_attention_2"),
        default="eager",
    )
    parser.add_argument(
        "--prefill-mode",
        choices=("full", "tokenwise"),
        default="full",
        help="Build the reference cache in one full prefill or one token at a time",
    )
    parser.add_argument(
        "--device-map",
        help="Optional Transformers device_map (for example cuda:0 or auto)",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.warmup_runs < 0:
        parser.error("--warmup-runs must be non-negative")
    if args.model and not pathlib.Path(args.model).exists() and not args.revision:
        parser.error("--revision is required when --model is not a local path")
    return args


def import_runtime():
    try:
        import torch
        import transformers
        from safetensors.torch import save_file
    except ImportError as error:
        raise RuntimeError(
            "missing reference dependency {}; install validation/requirements.txt".format(
                error.name
            )
        ) from error
    return torch, transformers, save_file


def read_probe(path: pathlib.Path) -> dict:
    try:
        report = json.loads(path.read_text())
        input_ids = report["input"]["token_ids"]
        fed_ids = report["output"]["fed_token_ids"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError("invalid checkpoint probe {}: {}".format(path, error)) from error
    for label, values in (("input.token_ids", input_ids), ("output.fed_token_ids", fed_ids)):
        if not isinstance(values, list) or not all(
            isinstance(value, int) and 0 <= value <= 0xFFFFFFFF for value in values
        ):
            raise ValueError("probe {} must be a list of u32 values".format(label))
    if not input_ids:
        raise ValueError("probe input.token_ids must not be empty")
    return report


def dtype_from_name(torch, name: str):
    return {
        "auto": "auto",
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def auto_model_class(transformers, model_type: Optional[str]):
    if model_type in ("qwen3_vl", "qwen3_vl_moe"):
        return transformers.AutoModelForImageTextToText
    return transformers.AutoModelForCausalLM


def patch_nemotron_h_cache_registry(
    model_type: Optional[str], cache_registry: dict, linear_attention_layer
) -> List[str]:
    """Backport the Transformers cache placeholder for cache-free MLP blocks."""
    if model_type != "nemotron_h" or "mlp" in cache_registry:
        return []
    cache_registry["mlp"] = linear_attention_layer
    return ["nemotron_h_mlp_cache_placeholder"]


def apply_reference_compatibility_patches(model_type: Optional[str]) -> List[str]:
    if model_type != "nemotron_h":
        return []
    from transformers.cache_utils import (  # pylint: disable=import-outside-toplevel
        DYNAMIC_LAYER_TYPE_MAPPING,
        LinearAttentionLayer,
    )

    return patch_nemotron_h_cache_registry(
        model_type, DYNAMIC_LAYER_TYPE_MAPPING, LinearAttentionLayer
    )


def synchronize(torch, device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class Timer:
    def __init__(self, torch, device):
        self.torch = torch
        self.device = device
        self.wall_started = None
        self.start_event = None
        self.end_event = None

    def start(self) -> None:
        synchronize(self.torch, self.device)
        if self.device.type == "cuda":
            self.start_event = self.torch.cuda.Event(enable_timing=True)
            self.end_event = self.torch.cuda.Event(enable_timing=True)
            self.start_event.record()
        self.wall_started = time.perf_counter()

    def finish(self) -> PhaseTiming:
        if self.device.type == "cuda":
            self.end_event.record()
            self.end_event.synchronize()
            device_seconds = self.start_event.elapsed_time(self.end_event) / 1000.0
        else:
            device_seconds = None
        synchronize(self.torch, self.device)
        return PhaseTiming(time.perf_counter() - self.wall_started, device_seconds)


def run_sequence(
    torch,
    model,
    device,
    input_ids,
    fed_ids,
    capture: bool,
    prefill_mode: str = "full",
) -> ReferenceRun:
    tokens = torch.tensor([input_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(tokens)
    timer = Timer(torch, device)
    timer.start()
    if prefill_mode == "full":
        outputs = model(input_ids=tokens, attention_mask=attention_mask, use_cache=True)
    elif prefill_mode == "tokenwise":
        attention_mask = torch.empty((1, 0), dtype=torch.long, device=device)
        past_key_values = None
        for token_id in input_ids:
            token = torch.tensor([[token_id]], dtype=torch.long, device=device)
            attention_mask = torch.cat(
                (
                    attention_mask,
                    torch.ones((1, 1), dtype=attention_mask.dtype, device=device),
                ),
                dim=1,
            )
            outputs = model(
                input_ids=token,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
    else:
        raise ValueError("unsupported prefill mode {!r}".format(prefill_mode))
    prefill_timing = timer.finish()
    current = outputs.logits[:, -1, :]
    vocabulary_size = current.shape[-1]
    greedy_ids = [int(current.argmax(dim=-1).item())] if capture else []
    prefill_logits = current.float().cpu().contiguous() if capture else None
    past_key_values = outputs.past_key_values
    decode_rows = []
    decode_timings = []

    for token_id in fed_ids:
        token = torch.tensor([[token_id]], dtype=torch.long, device=device)
        attention_mask = torch.cat(
            (attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=device)),
            dim=1,
        )
        timer = Timer(torch, device)
        timer.start()
        outputs = model(
            input_ids=token,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        decode_timings.append(timer.finish())
        current = outputs.logits[:, -1, :]
        if current.shape[-1] != vocabulary_size:
            raise ValueError("reference vocabulary size changed during cached decode")
        if capture:
            greedy_ids.append(int(current.argmax(dim=-1).item()))
            decode_rows.append(current.float().cpu().contiguous())
        past_key_values = outputs.past_key_values

    if capture:
        decode_logits = (
            torch.cat(decode_rows, dim=0)
            if decode_rows
            else torch.empty((0, vocabulary_size), dtype=torch.float32)
        )
    else:
        decode_logits = None
    return ReferenceRun(
        prefill_logits=prefill_logits,
        decode_logits=decode_logits,
        greedy_token_ids=greedy_ids,
        prefill_timing=prefill_timing,
        decode_timings=decode_timings,
    )


def process_peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def memory_snapshot(torch, device) -> dict:
    snapshot = {"process_peak_rss_bytes": process_peak_rss_bytes()}
    if device.type == "cuda":
        snapshot.update(
            {
                "cuda_allocated_bytes": torch.cuda.memory_allocated(device),
                "cuda_reserved_bytes": torch.cuda.memory_reserved(device),
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            }
        )
    return snapshot


def output_paths(prefix: pathlib.Path):
    return prefix.with_suffix(".json"), prefix.with_suffix(".safetensors")


def ensure_outputs(json_path: pathlib.Path, tensor_path: pathlib.Path, overwrite: bool) -> None:
    if json_path == tensor_path:
        raise ValueError("JSON and SafeTensors output paths must differ")
    if not overwrite:
        for path in (json_path, tensor_path):
            if path.exists():
                raise ValueError("refusing to replace {}; pass --overwrite".format(path))
    json_path.parent.mkdir(parents=True, exist_ok=True)
    tensor_path.parent.mkdir(parents=True, exist_ok=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        probe = read_probe(args.probe)
        model_source = args.model or probe["model"]["checkpoint_path"]
        if not pathlib.Path(model_source).exists() and not args.revision:
            raise ValueError(
                "--revision is required when the reference model is not a local path"
            )
        json_path, tensor_path = output_paths(args.output)
        ensure_outputs(json_path, tensor_path, args.overwrite)
        torch, transformers, save_file = import_runtime()
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but torch.cuda.is_available() is false")

        model_type = probe.get("model", {}).get("model_type")
        compatibility_patches = apply_reference_compatibility_patches(model_type)

        load_kwargs = {
            "trust_remote_code": args.trust_remote_code,
            "local_files_only": args.local_files_only,
            "torch_dtype": dtype_from_name(torch, args.dtype),
            "low_cpu_mem_usage": True,
        }
        if args.revision:
            load_kwargs["revision"] = args.revision
        if args.attn_implementation != "auto":
            load_kwargs["attn_implementation"] = args.attn_implementation
        if args.device_map:
            load_kwargs["device_map"] = args.device_map

        load_started = time.perf_counter()
        model_loader = auto_model_class(transformers, model_type)
        model = model_loader.from_pretrained(model_source, **load_kwargs)
        model.eval()
        if not args.device_map:
            model.to(device)
        synchronize(torch, device)
        load_wall_seconds = time.perf_counter() - load_started
        after_load_memory = memory_snapshot(torch, device)
        input_ids = probe["input"]["token_ids"]
        fed_ids = probe["output"]["fed_token_ids"]

        with torch.inference_mode():
            warmup_started = time.perf_counter()
            for _ in range(args.warmup_runs):
                run_sequence(
                    torch,
                    model,
                    device,
                    input_ids,
                    fed_ids,
                    capture=False,
                    prefill_mode=args.prefill_mode,
                )
            warmup_wall_seconds = time.perf_counter() - warmup_started
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            before_run_memory = memory_snapshot(torch, device)
            result = run_sequence(
                torch,
                model,
                device,
                input_ids,
                fed_ids,
                capture=True,
                prefill_mode=args.prefill_mode,
            )
            after_run_memory = memory_snapshot(torch, device)

        tensors = {
            "input.token_ids": torch.tensor(input_ids, dtype=torch.int64),
            "output.fed_token_ids": torch.tensor(fed_ids, dtype=torch.int64),
            "output.greedy_token_ids": torch.tensor(
                result.greedy_token_ids, dtype=torch.int64
            ),
            "prefill.logits": result.prefill_logits,
            "decode.logits": result.decode_logits,
        }
        save_file(
            tensors,
            str(tensor_path),
            metadata={
                "schema_version": "1",
                "implementation": "transformers",
                "source_probe": str(args.probe),
            },
        )
        prefill_device = result.prefill_timing.device_seconds
        decode_device = [timing.device_seconds for timing in result.decode_timings]
        decode_device_total = (
            sum(value for value in decode_device if value is not None)
            if all(value is not None for value in decode_device)
            else None
        )
        report = {
            "schema_version": 1,
            "kind": "transformers_checkpoint_reference",
            "source_probe": str(args.probe),
            "model": {
                "checkpoint_path": str(model_source),
                "revision": args.revision,
                "model_type": getattr(model.config, "model_type", None),
                "class": type(model).__name__,
                "auto_model_class": model_loader.__name__,
            },
            "runtime": {
                "python_version": platform.python_version(),
                "torch_version": torch.__version__,
                "transformers_version": transformers.__version__,
                "device": str(device),
                "dtype": args.dtype,
                "attn_implementation": args.attn_implementation,
                "prefill_mode": args.prefill_mode,
                "cuda_version": torch.version.cuda,
                "cuda_device_name": (
                    torch.cuda.get_device_name(device) if device.type == "cuda" else None
                ),
                "cuda_capability": (
                    list(torch.cuda.get_device_capability(device))
                    if device.type == "cuda"
                    else None
                ),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "trust_remote_code": args.trust_remote_code,
                "compatibility_patches": compatibility_patches,
            },
            "input": {
                "token_ids": input_ids,
                "token_count": len(input_ids),
            },
            "output": {
                "tensor_file": str(tensor_path),
                "fed_token_ids": fed_ids,
                "greedy_token_ids": result.greedy_token_ids,
            },
            "timings": {
                "load_wall_seconds": load_wall_seconds,
                "warmup_runs": args.warmup_runs,
                "warmup_wall_seconds": warmup_wall_seconds,
                "prefill_wall_seconds": result.prefill_timing.wall_seconds,
                "prefill_device_seconds": prefill_device,
                "decode_wall_seconds": [
                    timing.wall_seconds for timing in result.decode_timings
                ],
                "decode_device_seconds": decode_device,
                "decode_total_wall_seconds": sum(
                    timing.wall_seconds for timing in result.decode_timings
                ),
                "decode_total_device_seconds": decode_device_total,
            },
            "memory": {
                "after_load": after_load_memory,
                "before_measured_run": before_run_memory,
                "after_measured_run": after_run_memory,
            },
        }
        json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    except (KeyError, RuntimeError, ValueError, OSError) as error:
        print("error: {}".format(error), file=sys.stderr)
        return 2
    print("json={}".format(json_path))
    print("safetensors={}".format(tensor_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
