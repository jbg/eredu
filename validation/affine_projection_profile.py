#!/usr/bin/env python3
"""Rank isolated real-operand replays; optionally attach separate dispatch evidence."""
import argparse
import collections
import json
from pathlib import Path
import re
import statistics

MARKER = re.compile(r"\[affine-phase-(begin|end)\] (\S+) (\S+) rotating=(true|false) iteration=(\d+)$")


def selections(path):
    groups = collections.defaultdict(collections.Counter)
    samples = collections.Counter()
    active = None
    for line in path.read_text().splitlines():
        match = MARKER.fullmatch(line)
        if match:
            action, label, phase, rotating, iteration = match.groups()
            key = (label, phase, rotating == "true", int(iteration))
            if action == "begin":
                if active is not None:
                    raise ValueError("nested replay markers")
                active = key
            else:
                if active != key:
                    raise ValueError("unmatched replay end marker")
                samples[key[:3]] += 1
                active = None
        elif line.startswith("[mlx-metal-kernel] ") and active is not None:
            groups[active[:3]][line.removeprefix("[mlx-metal-kernel] ")] += 1
    if active is not None:
        raise ValueError("incomplete replay trace")
    return {key: {"trace_samples": samples[key], "pipeline_requests_per_replay":
                 {name: count / samples[key] for name, count in counts.items()}}
            for key, counts in groups.items()}


def summarize(profile, kernel_log=None, selection_profile=None):
    if profile.get("schema_version") != 1:
        raise ValueError("unsupported affine profile schema")
    if kernel_log:
        if selection_profile is None:
            raise ValueError("--kernel-log requires --selection-profile to verify replay geometry")
        fields = ("site", "invocations", "distinct_parameter_sets", "input_shape", "input_strides", "input_dtype", "weight_shape", "weight_strides", "scale_strides", "scale_dtype", "bias_strides", "bias_dtype", "output_dtype", "bits", "group_size")
        def inventory(p):
            return {case["label"]: {key: case["captured"][key] for key in fields} for case in p["replays"]}
        if inventory(profile) != inventory(selection_profile):
            raise ValueError("timing and selection replay geometry differ")
    traces = selections(kernel_log) if kernel_log else None
    result = []
    for case in profile["replays"]:
        captured = case["captured"]
        phases = []
        for phase in case["phases"]:
            warm = [sample for sample in phase["samples"] if not sample["first_sample"]]
            if not warm:
                raise ValueError("missing warm replay samples")
            count = warm[0]["projection_count"]
            if any(sample["projection_count"] != count for sample in warm):
                raise ValueError("inconsistent rotation counts")
            gpu_ms = statistics.median(sample["gpu_evaluation_ms"] for sample in warm)
            wall_ms = statistics.median(sample["total_wall_ms"] for sample in warm)
            row = {"phase": phase["phase"], "rotating_weights": phase["rotating_weights"],
                   "warm_samples": len(warm), "projection_count": count,
                   "median_gpu_evaluation_ms": gpu_ms,
                   "median_gpu_ms_per_projection": gpu_ms / count,
                   "median_total_wall_ms": wall_ms,
                   "median_wall_ms_per_projection": wall_ms / count,
                   "median_build_wall_ms": statistics.median(sample["build_wall_ms"] for sample in warm),
                   "gpu_interval_sum_exceeds_wall": gpu_ms > wall_ms,
                   "parameter_payload_bytes": phase["parameter_payload_bytes"],
                   "nominal_parameter_payload_gb_per_gpu_second":
                       phase["parameter_payload_bytes"] / gpu_ms / 1e6 if gpu_ms > 0 else None,
                   "comparison": phase["comparison"]}
            interval_fields = ("native_interval_union_ms", "native_interval_span_ms",
                               "native_interval_count", "missing_timestamps")
            for field in interval_fields:
                if all(field in sample for sample in warm):
                    row["median_" + field] = statistics.median(sample[field] for sample in warm)
            if traces is not None:
                key = (case["label"], phase["phase"], phase["rotating_weights"])
                if key not in traces:
                    raise ValueError(f"missing native selection trace for {key}")
                row["separate_selection_trace"] = traces[key]
            phases.append(row)
        baseline = next(p for p in phases if p["phase"] == "native" and p["rotating_weights"])
        padded = next((p for p in phases if p["phase"] == "native_padded_fast_q8" and p["rotating_weights"]), None)
        result.append({"label": case["label"], "captured": captured, "phases": phases,
                       "isolated_invocation_weighted_wall_ms_estimate":
                           baseline["median_wall_ms_per_projection"] * captured["invocations"],
                       "isolated_invocation_weighted_gpu_ms_estimate":
                           baseline["median_gpu_ms_per_projection"] * captured["invocations"],
                       "padded_rotating_wall_change_percent":
                           (padded["median_total_wall_ms"] / baseline["median_total_wall_ms"] - 1) * 100 if padded else None,
                       "padded_rotating_gpu_change_percent":
                           (padded["median_gpu_evaluation_ms"] / baseline["median_gpu_evaluation_ms"] - 1) * 100 if padded else None})
    operation_classes = []
    for case in profile.get("operation_replays", []):
        warm = [sample for sample in case["samples"] if not sample["first_sample"]]
        if not warm or not case["bitwise_equal"]:
            raise ValueError("operation replay lacks warm samples or exact parity")
        count = case["captured"]["invocations"]
        if any(sample["operation_count"] != count for sample in warm):
            raise ValueError("inconsistent operation replay counts")
        operation_classes.append({"captured": case["captured"], "warm_samples": len(warm),
                                  "bitwise_equal": True,
                                  **{"median_" + key: statistics.median(sample[key] for sample in warm)
                                     for key in ("build_wall_ms", "total_wall_ms", "native_interval_sum_ms",
                                                 "native_interval_union_ms", "native_interval_span_ms",
                                                 "native_interval_count", "missing_timestamps", "peak_growth_bytes")}})
    operation_classes.sort(key=lambda c: c["median_total_wall_ms"], reverse=True)
    result.sort(key=lambda c: c["isolated_invocation_weighted_wall_ms_estimate"], reverse=True)
    return {"schema_version": 1, "advance": profile["advance"],
            "ranking_metric": "isolated_invocation_weighted_wall_ms_estimate", "ranked_classes": result,
            "operation_classes": operation_classes,
            "notes": ["Ranking uses isolated wall time including graph construction and completion; it is not additive end-to-end attribution.",
                      "Metal interval sums can overlap and exceed wall time. Union removes overlap; span includes gaps. All retain buffer waits and are not individual shader compute time. Compare wall time and full conversations before accepting gains.",
                      "Nominal payload/time is not measured DRAM bandwidth. Reuse and concurrent independent dispatches affect replay.",
                      "Selection evidence comes from a separate, instrumented process; counts are pipeline requests, not GPU durations.",
                      "Captured conversations and optional Metal exports are perturbed diagnostic workloads."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    parser.add_argument("--kernel-log", type=Path)
    parser.add_argument("--selection-profile", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selection_profile = json.loads(args.selection_profile.read_text()) if args.selection_profile else None
    result = summarize(json.loads(args.profile.read_text()), args.kernel_log, selection_profile)
    with args.output.open("x") as out:
        json.dump(result, out, indent=2)
        out.write("\n")
    print("label  weight_shape  invocations  distinct_weights  isolated_weighted_wall_ms  padded_wall_change_%")
    for case in result["ranked_classes"]:
        c = case["captured"]
        change = case["padded_rotating_wall_change_percent"]
        print(case["label"], c["weight_shape"], c["invocations"], c["distinct_parameter_sets"],
              f'{case["isolated_invocation_weighted_wall_ms_estimate"]:.3f}',
              f"{change:+.2f}" if change is not None else "n/a")


if __name__ == "__main__":
    main()
