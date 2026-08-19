# Generic Llama runtime validation

Validation date: 2026-08-19

This record compares the generic-runtime candidate following `35d010f` with
the pre-migration implementation at `3514800`. Both binaries were optimized
release builds from isolated source trees and used the same checkpoint and
benchmark arguments.

## Environment

- Apple M3 Ultra Mac Studio, 32 CPU cores, 256 GB unified memory
- macOS 26.5.2 (25F84), Xcode 26.6
- Rust 1.97.1
- `HuggingFaceTB/SmolLM-135M`, SafeTensors, `model_type: llama`
- one mapped checkpoint shard
- prompt: token IDs 1 through 128
- cached greedy decode: 32 tokens
- seven runs per implementation and residency mode, with alternating run order

The reported dispersion is median absolute deviation (MAD). Prefill uses the
second prefill after a cache reset, so shader and graph setup from the first
pass are excluded.

## Throughput

| Mode | Metric | Pre-migration median ± MAD | Generic median ± MAD | Change | Gate |
| --- | --- | ---: | ---: | ---: | ---: |
| Resident | Prefill tokens/s | 14,845.673 ± 146.641 | 15,319.709 ± 73.926 | +3.193% | ≥ -2% |
| Resident | Cached decode tokens/s | 293.740 ± 0.660 | 312.120 ± 0.640 | +6.257% | ≥ -2% |
| Two-layer window | Prefill tokens/s | 6,513.235 ± 303.147 | 6,745.096 ± 80.372 | +3.560% | ≥ -3% |
| Two-layer window | Cached decode tokens/s | 75.480 ± 0.290 | 74.210 ± 0.170 | -1.683% | ≥ -3% |

All measured throughput gates pass.

## Residency and I/O

| Counter | Pre-migration | Generic | Result |
| --- | ---: | ---: | --- |
| Resident charged host bytes | 0 | 0 | unchanged |
| Resident logical device peak | 538,060,032 | 538,060,032 | unchanged |
| Layerwise charged host peak | 425,656,320 | 425,656,320 | unchanged |
| Layerwise logical device peak | 155,729,664 | 141,569,280 | 9.093% lower |
| Layerwise host-to-device transfers | 1,020 / 14,443,591,680 bytes | 1,020 / 14,443,591,680 bytes | unchanged |
| Layerwise disk-to-host transfers | 30 / 424,811,520 bytes | 30 / 424,811,520 bytes | unchanged |
| Maximum mapped shards | 1 | 1 | unchanged |
| Reported physical reads | 0 / 0 bytes | 0 / 0 bytes | unchanged |

The old path submitted two disk-to-device operations for the static resident
weights and the generic path submits one; transferred bytes are identical.

The MLX allocator's transient layerwise peak changed from 571,187,468 to
577,260,296 bytes (+1.063%) for the 128-token prompt. This is not retained or
charged residency growth: final active bytes changed from 547,169,796 to
547,167,492, cached bytes changed from 50,972,716 to 50,677,800, charged host
bytes are identical, and the exact logical device high-water mark is 14,160,384
bytes lower. The transient difference is treated as allocator/graph noise rather
than a residency regression.

Tensor-parallel communication was not measured because the validation host
exposes one MLX device. Existing topology and collective-count regression tests
remain the gate for multi-rank plans.

## Structural and hardware gates

The release-mode Metal tests for exact transfer completion, direct disk-to-device
materialization, two-block GPU-ordered prefetch, and packed-byte residency all
pass. LLVM IR for the concrete reference Llama hot path contains no indirect
call and no tensor-to-bytes, synchronization, or evaluation symbol. Dependency
tests enforce that MLX backend source has no Llama names/imports and that neutral
architecture source has no MLX or integration dependency.

The benchmark exposed and permanently fixed two generic-policy defects before
these measurements were accepted: an overlapping-window self-wait and failure
to evict completed units outside the configured device window. The final
two-layer run retains one last layer and peaks at exactly the static allocation
plus two execution units.

## Reproduction

Build and run the current candidate with:

```sh
cargo build -p eredu --release --example llama_residency
target/release/examples/llama_residency "$MODEL_DIR" \
  --fully-resident --prompt-tokens "$PROMPT_TOKENS" \
  --decode-tokens 32 --mapped-shards 1
target/release/examples/llama_residency "$MODEL_DIR" \
  --prompt-tokens "$PROMPT_TOKENS" --decode-tokens 32 \
  --device-layer-window 2 --mapped-shards 1
```

For the baseline, export `3514800` into an isolated directory, add only the
benchmark's `--fully-resident` selection flag, and build the same example. Run
the two binaries in alternating order seven times and compute the median and
MAD from `repeated-process prefill latency after cache reset` and `decode
throughput`.
