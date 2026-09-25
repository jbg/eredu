# Mixed-width Metal projection profiling

The optional `projection-profiling` feature records real arrays from ordinary
prefill and replays representative projections in isolation. It reports actual
shapes, strides, dtypes, native dispatch, conversion allocations and elapsed time.
The profiling hook does not evaluate inputs or change dispatch. The feature is
disabled in ordinary builds and adds no native dependencies to portable defaults.

## Checkpoint and reproduction

The checkpoint is official `LiquidAI/LFM2.5-1.2B-Instruct-GGUF`, revision
`6767265158422fb8a19c62ceb45f16f05363615b`, file
`LFM2.5-1.2B-Instruct-BF16.gguf`, 2,343,326,528 bytes, SHA-256
`3d80914b903cd6f3cc041208cf20ec46a3224f840c732e5fd7698832b4743d1b`.
The runner hashes the complete file before inference. The source remains outside
the tracked tree. The checkpoint contains mixed F32/BF16 tensors and tokenizer
metadata.

```sh
CARGO_INCREMENTAL=0 cargo build -p eredu --release \
  --example projection_baseline --no-default-features \
  --features mlx,metal,projection-profiling --locked --offline
python3 validation/projection_baseline.py \
  --binary target/release/examples/projection_baseline \
  --model /path/to/pinned/LFM2.5-1.2B-Instruct-BF16.gguf \
  --output /tmp/eredu-projection-profile --samples 5 --disable-tf32
```

Native Metal access is required. The runner sanitizes MLX environment overrides,
checks checkpoint provenance, exact outputs and trace/timing class agreement, and
rejects missing native selection evidence. `--disable-tf32` selects native SIMD
FP32 arithmetic where required. `--reference-dispatch` selects the diagnostic
cast-plus-matmul path for multi-row projections; omitting it profiles the shared
mixed-storage selector. Both modes retain ordinary single-row mixed GEMV.

`--mixed-storage-prototype` adds a direct native-operation replay, named
`mixed_storage_prototype` in the output, with a settled contiguous `[M,K]` input.
That diagnostic replay and the integrated selector call the same native GEMM;
the integrated selector additionally accepts lazy/batched/strided activations and
restores their leading dimensions. Unsupported direct replays fail explicitly.
The optional `--reference-manifest` also checks recorded same-device output
fingerprints, class inventories and tokens.

## Measurement contract

Each policy/length pair uses three fresh processes:

1. **Ordinary inference:** a cold request and five reset/warm requests, each using
   repeated token ID 1 and eight greedy predictions. Loading is excluded. Reset
   preserves admitted conversions. These are the only ordinary latency/peak data.
2. **Capture and replay:** collect real mixed-width dense/readout invocations
   through the first prediction without adding evaluations in the collection hook.
   At an explicit finish boundary, settle arrays and group by dtype, shape, actual
   strides, projection site and observed dispatch path. Retain the first example
   and count for every class. Captured owners change graph retention, so this
   process's full-model latency and peak are explicitly invalid as ordinary metrics.
3. **Selection trace:** replay the same classes with
   `MLX_METAL_LOG_KERNEL_SELECTION=1`. Native MLX logs pipeline requests, including
   cache hits and function-constant specialization keys. Only requests inside
   synchronized replay-operation markers are attributed. Logging is disabled in
   timing runs.

The named replay operations are an explicit F32 cast, GEMM with an evaluated F32
weight, native mixed matmul, and the shared Eredu projection dispatch. The latter
inherits conversion ownership from capture and previous replays; the other three
operations isolate defined conversion states. The preconverted GEMM's resident F32
weight is charged in its baseline, not its incremental peak; its byte count is
recorded separately. Thus its smaller incremental peak is not free conversion
storage.

Every replay reports one first sample, then five measured warm samples after an
additional unreported warmup. Capture/reference creation may already warm kernel
caches; first replay does not mean cold JIT. Wall time includes graph building,
submission and completion. GPU time uses native completion timing and excludes
eager work done while constructing the graph. Neither duration is a precise
allocation-only timer, and independently synchronized operation times are not additive
end-to-end attribution. Peaks are MLX active allocations above each settled sample's
baseline, not RSS, driver memory or allocator-cache high water.

Rows 2, 4, 8 and 16 slice leading real 128-position activations. They are
verification-sized projection workloads, not full speculative decoding. Original
128- and 2,000-row invocations retain their observed input shapes and layouts.
All rows of each non-cast replay output are compared as F32 bits: full-sized cases
use the captured output; sliced cases use explicit F32-weight native matmul. Each
reference also has a SHA-256 fingerprint. These comparisons validate projection
arithmetic against native FP32 matmul; independent full-model validation has a
separate reference contract.

## Pinned projection workload

Both 128- and 2,000-position requests contain 93 mixed-width prefill projections
across six weight shapes. Weights are BF16, inputs/outputs are F32 and the observed
weight rows are contiguous. Eligible invocations use the shared mixed-storage
selector; reference-mode classes can additionally split by conversion admission.

| Weight shape `[N,K]` | Invocations | F32 cast per matrix (MiB) |
|---|---:|---:|
| `[2048,2048]` | 22 | 16 |
| `[512,2048]` | 12 | 4 |
| `[6144,2048]` | 10 | 48 |
| `[8192,2048]` | 32 | 64 |
| `[2048,8192]` | 16 | 64 |
| `[65536,2048]` | 1 | 512 |

The hypothetical full F32 matrix payloads sum to 4,464 MiB. Eligible mixed-storage
projections avoid those copies. The vocabulary/readout projection receives all
prompt rows in this request path: its output alone is 32 MiB at 128 rows and
500 MiB at 2,000 rows. Output allocation is separate from weight promotion.

On M3 Ultra, short and 128-row projections include native split-K and regular
GEMM; all six 2,000-row shapes use regular GEMM. Full specialization keys and
aligned/tail flags are in the
[validation manifest](../validation/results/projection-forecast-2026-09-25.json).
This is a device-specific selection inventory, not a universal Metal selector.
The loader preserves the selected tile geometry, partitioning and accumulator;
FP32 accumulation alone does not establish exact arithmetic parity.

## Interpreting results

The profiler separates cast-only, preconverted GEMM, native mixed matmul and
shared-selector costs. Native mixed matmul can include a full weight-conversion
kernel even when the shared mixed-storage selector avoids it. An isolated replay's
incremental peak excludes resident operands; a full-model active peak includes
all live native allocations. Neither measures process RSS or driver memory.

See [mixed-storage GEMM](mixed-storage-gemm.md) for the supported dispatch contract,
repeated same-build reference/selector measurements, released-checkpoint parity,
loaded forecast evidence and hardware-validation gaps. Its integration runner
also exercises controlled and speculative drivers; short projection replays alone
do not establish those execution-path guarantees.
