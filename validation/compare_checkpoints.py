#!/usr/bin/env python3
"""Compare SafeMLX checkpoint-probe artifacts with a reference run."""

import argparse
import heapq
import json
import math
import pathlib
import struct
import sys
from array import array
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_RELATIVE_L2_MAX = 0.02
DEFAULT_COSINE_SIMILARITY_MIN = 0.999
DEFAULT_TOP_K = 5
DEFAULT_TOP_K_OVERLAP_MIN = 4
LOGIT_TENSORS = ("prefill.logits", "decode.logits")


@dataclass(frozen=True)
class Tensor:
    dtype: str
    shape: Tuple[int, ...]
    values: Sequence[float]


@dataclass(frozen=True)
class Thresholds:
    relative_l2_max: float
    cosine_similarity_min: float
    top_k: int
    top_k_overlap_min: int
    require_unambiguous_argmax_match: bool
    argmax_margin_min: float


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actual", type=pathlib.Path, required=True)
    parser.add_argument("--reference", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--manifest", type=pathlib.Path)
    parser.add_argument("--case", help="Case ID in --manifest")
    parser.add_argument("--relative-l2-max", type=float)
    parser.add_argument("--cosine-similarity-min", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--top-k-overlap-min", type=int)
    parser.add_argument(
        "--require-unambiguous-argmax-match",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--argmax-margin-min",
        type=float,
        default=0.0,
        help="Only enforce argmax equality when the reference top-1 margin exceeds this value",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if bool(args.manifest) != bool(args.case):
        parser.error("--manifest and --case must be supplied together")
    return args


def load_json(path: pathlib.Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("failed to read JSON {}: {}".format(path, error)) from error
    if not isinstance(value, dict):
        raise ValueError("{} must contain a JSON object".format(path))
    return value


def resolve_tensor_path(report_path: pathlib.Path, report: dict) -> pathlib.Path:
    try:
        raw = pathlib.Path(report["output"]["tensor_file"])
    except (KeyError, TypeError) as error:
        raise ValueError("{} has no output.tensor_file".format(report_path)) from error
    if raw.is_absolute() or raw.exists():
        return raw
    sibling = report_path.parent / raw.name
    if sibling.exists():
        return sibling
    return report_path.parent / raw


def read_safetensors(path: pathlib.Path, names: Iterable[str]) -> Dict[str, Tensor]:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ValueError("failed to read SafeTensors {}: {}".format(path, error)) from error
    if len(payload) < 8:
        raise ValueError("{} is shorter than a SafeTensors header".format(path))
    header_length = struct.unpack_from("<Q", payload, 0)[0]
    data_start = 8 + header_length
    if data_start > len(payload):
        raise ValueError("{} has a truncated SafeTensors header".format(path))
    try:
        header = json.loads(payload[8:data_start].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("{} has an invalid SafeTensors header".format(path)) from error

    result = {}
    for name in names:
        if name not in header:
            raise ValueError("{} is missing tensor {!r}".format(path, name))
        descriptor = header[name]
        dtype = descriptor.get("dtype")
        shape = tuple(descriptor.get("shape", ()))
        offsets = descriptor.get("data_offsets")
        if dtype != "F32":
            raise ValueError("{} tensor {!r} must be F32, got {}".format(path, name, dtype))
        if (
            len(offsets or ()) != 2
            or not all(isinstance(value, int) for value in offsets)
            or offsets[0] < 0
            or offsets[1] < offsets[0]
            or data_start + offsets[1] > len(payload)
        ):
            raise ValueError("{} tensor {!r} has invalid offsets".format(path, name))
        if not all(isinstance(value, int) and value >= 0 for value in shape):
            raise ValueError("{} tensor {!r} has invalid shape".format(path, name))
        element_count = math.prod(shape)
        raw = payload[data_start + offsets[0] : data_start + offsets[1]]
        if len(raw) != element_count * 4:
            raise ValueError(
                "{} tensor {!r} has {} bytes for shape {}".format(
                    path, name, len(raw), shape
                )
            )
        values = array("f")
        values.frombytes(raw)
        if sys.byteorder != "little":
            values.byteswap()
        result[name] = Tensor(dtype=dtype, shape=shape, values=values)
    return result


def load_manifest_profile(path: pathlib.Path, case_id: str) -> dict:
    try:
        import yaml
    except ImportError as error:
        raise ValueError(
            "PyYAML is required for --manifest; install validation/requirements.txt"
        ) from error
    try:
        manifest = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as error:
        raise ValueError("failed to read manifest {}: {}".format(path, error)) from error
    cases = [case for case in manifest.get("cases", ()) if case.get("id") == case_id]
    if len(cases) != 1:
        raise ValueError("manifest case {!r} must resolve exactly once".format(case_id))
    profile_name = cases[0].get("correctness_profile")
    try:
        return manifest["correctness_profiles"][profile_name]
    except KeyError as error:
        raise ValueError(
            "case {!r} references missing correctness profile {!r}".format(
                case_id, profile_name
            )
        ) from error


def thresholds_from_args(args: argparse.Namespace) -> Thresholds:
    profile = load_manifest_profile(args.manifest, args.case) if args.manifest else {}

    def selected(argument, key, fallback):
        return argument if argument is not None else profile.get(key, fallback)

    thresholds = Thresholds(
        relative_l2_max=float(
            selected(args.relative_l2_max, "relative_l2_max", DEFAULT_RELATIVE_L2_MAX)
        ),
        cosine_similarity_min=float(
            selected(
                args.cosine_similarity_min,
                "cosine_similarity_min",
                DEFAULT_COSINE_SIMILARITY_MIN,
            )
        ),
        top_k=int(selected(args.top_k, "top_k", DEFAULT_TOP_K)),
        top_k_overlap_min=int(
            selected(args.top_k_overlap_min, "top_k_overlap_min", DEFAULT_TOP_K_OVERLAP_MIN)
        ),
        require_unambiguous_argmax_match=bool(
            selected(
                args.require_unambiguous_argmax_match,
                "require_unambiguous_argmax_match",
                True,
            )
        ),
        argmax_margin_min=float(args.argmax_margin_min),
    )
    if thresholds.relative_l2_max < 0:
        raise ValueError("relative_l2_max must be non-negative")
    if not -1.0 <= thresholds.cosine_similarity_min <= 1.0:
        raise ValueError("cosine_similarity_min must be between -1 and 1")
    if thresholds.top_k <= 0:
        raise ValueError("top_k must be positive")
    if not 0 <= thresholds.top_k_overlap_min <= thresholds.top_k:
        raise ValueError("top_k_overlap_min must be between zero and top_k")
    if thresholds.argmax_margin_min < 0:
        raise ValueError("argmax_margin_min must be non-negative")
    return thresholds


def identity_checks(actual: dict, reference: dict) -> Tuple[dict, List[str]]:
    checks = {}
    failures = []
    fields = (
        ("input.token_ids", ("input", "token_ids")),
        ("output.fed_token_ids", ("output", "fed_token_ids")),
    )
    for label, path in fields:
        try:
            actual_value = actual[path[0]][path[1]]
            reference_value = reference[path[0]][path[1]]
            matches = actual_value == reference_value
        except (KeyError, TypeError):
            matches = False
        checks[label] = matches
        if not matches:
            failures.append("{} differ between actual and reference".format(label))
    return checks, failures


def tensor_rows(tensor: Tensor) -> Tuple[int, int]:
    if len(tensor.shape) != 2:
        raise ValueError("expected rank-2 logits, got shape {}".format(tensor.shape))
    return tensor.shape[0], tensor.shape[1]


def top_indices(values: Sequence[float], count: int) -> List[int]:
    return heapq.nlargest(count, range(len(values)), key=values.__getitem__)


def compare_row(
    actual: Sequence[float], reference: Sequence[float], thresholds: Thresholds
) -> dict:
    finite = all(math.isfinite(value) for value in actual) and all(
        math.isfinite(value) for value in reference
    )
    if not finite:
        return {
            "finite": False,
            "relative_l2": None,
            "cosine_similarity": None,
            "max_absolute_error": None,
            "mean_absolute_error": None,
            "top_k_overlap": 0,
            "actual_argmax": None,
            "reference_argmax": None,
            "reference_argmax_margin": None,
            "argmax_required": False,
            "argmax_match": False,
            "passed": False,
        }

    difference_squared = math.fsum(
        (actual_value - reference_value) ** 2
        for actual_value, reference_value in zip(actual, reference)
    )
    reference_squared = math.fsum(value * value for value in reference)
    actual_squared = math.fsum(value * value for value in actual)
    dot = math.fsum(
        actual_value * reference_value
        for actual_value, reference_value in zip(actual, reference)
    )
    relative_l2 = math.sqrt(difference_squared) / max(math.sqrt(reference_squared), 1e-12)
    denominator = math.sqrt(actual_squared) * math.sqrt(reference_squared)
    cosine = dot / denominator if denominator else (1.0 if actual == reference else 0.0)
    absolute_errors = [
        abs(actual_value - reference_value)
        for actual_value, reference_value in zip(actual, reference)
    ]
    top_count = min(thresholds.top_k, len(reference))
    actual_top = top_indices(actual, top_count)
    reference_order = top_indices(reference, min(max(top_count, 2), len(reference)))
    reference_top = reference_order[:top_count]
    top_overlap = len(set(actual_top).intersection(reference_top))
    actual_argmax = actual_top[0]
    reference_argmax = reference_top[0]
    reference_runner_up = (
        reference_order[1] if len(reference_order) > 1 else reference_argmax
    )
    reference_margin = reference[reference_argmax] - reference[reference_runner_up]
    argmax_required = (
        thresholds.require_unambiguous_argmax_match
        and reference_margin > thresholds.argmax_margin_min
    )
    argmax_match = actual_argmax == reference_argmax
    passed = (
        relative_l2 <= thresholds.relative_l2_max
        and cosine >= thresholds.cosine_similarity_min
        and top_overlap >= min(thresholds.top_k_overlap_min, top_count)
        and (not argmax_required or argmax_match)
    )
    return {
        "finite": True,
        "relative_l2": relative_l2,
        "cosine_similarity": cosine,
        "max_absolute_error": max(absolute_errors, default=0.0),
        "mean_absolute_error": math.fsum(absolute_errors) / max(len(absolute_errors), 1),
        "top_k_overlap": top_overlap,
        "actual_argmax": actual_argmax,
        "reference_argmax": reference_argmax,
        "reference_argmax_margin": reference_margin,
        "argmax_required": argmax_required,
        "argmax_match": argmax_match,
        "passed": passed,
    }


def compare_tensor(name: str, actual: Tensor, reference: Tensor, thresholds: Thresholds) -> dict:
    if actual.shape != reference.shape:
        return {
            "passed": False,
            "actual_shape": list(actual.shape),
            "reference_shape": list(reference.shape),
            "failure": "shape mismatch",
            "rows": [],
        }
    row_count, vocabulary_size = tensor_rows(actual)
    if vocabulary_size == 0:
        raise ValueError("{} has an empty vocabulary dimension".format(name))
    rows = []
    for row_index in range(row_count):
        start = row_index * vocabulary_size
        end = start + vocabulary_size
        metrics = compare_row(
            actual.values[start:end], reference.values[start:end], thresholds
        )
        metrics["index"] = row_index
        rows.append(metrics)
    return {
        "passed": all(row["passed"] for row in rows),
        "shape": list(actual.shape),
        "rows": rows,
        "summary": {
            "max_relative_l2": max(
                (row["relative_l2"] for row in rows if row["relative_l2"] is not None),
                default=None,
            ),
            "min_cosine_similarity": min(
                (
                    row["cosine_similarity"]
                    for row in rows
                    if row["cosine_similarity"] is not None
                ),
                default=None,
            ),
            "min_top_k_overlap": min(
                (row["top_k_overlap"] for row in rows), default=None
            ),
            "argmax_matches": sum(row["argmax_match"] for row in rows),
            "argmax_required": sum(row["argmax_required"] for row in rows),
        },
    }


def compare_artifacts(
    actual_path: pathlib.Path,
    reference_path: pathlib.Path,
    thresholds: Thresholds,
) -> dict:
    actual_report = load_json(actual_path)
    reference_report = load_json(reference_path)
    checks, failures = identity_checks(actual_report, reference_report)
    actual_tensors = read_safetensors(
        resolve_tensor_path(actual_path, actual_report), LOGIT_TENSORS
    )
    reference_tensors = read_safetensors(
        resolve_tensor_path(reference_path, reference_report), LOGIT_TENSORS
    )
    comparisons = {}
    for name in LOGIT_TENSORS:
        comparison = compare_tensor(
            name, actual_tensors[name], reference_tensors[name], thresholds
        )
        comparisons[name] = comparison
        if not comparison["passed"]:
            failures.append("{} failed correctness thresholds".format(name))
    return {
        "schema_version": 1,
        "kind": "checkpoint_comparison",
        "passed": not failures,
        "actual": str(actual_path),
        "reference": str(reference_path),
        "thresholds": asdict(thresholds),
        "identity_checks": checks,
        "tensors": comparisons,
        "failures": failures,
    }


def write_report(path: pathlib.Path, report: dict, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise ValueError("refusing to replace {}; pass --overwrite".format(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        thresholds = thresholds_from_args(args)
        report = compare_artifacts(args.actual, args.reference, thresholds)
        write_report(args.output, report, args.overwrite)
    except ValueError as error:
        print("error: {}".format(error), file=sys.stderr)
        return 2
    print("comparison={} passed={}".format(args.output, str(report["passed"]).lower()))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
