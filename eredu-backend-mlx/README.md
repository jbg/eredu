# eredu-backend-mlx

`eredu-backend-mlx` is the concrete MLX implementation of Eredu's
backend-neutral contracts. It owns MLX tensor and neural operations,
checkpoint materialization, execution runtime, model-family binding, media
processing, distributed execution, and optional codec integration. It does not
depend on the `eredu` facade crate.

The crate's `MlxTensor` is a transparent, zero-copy wrapper around the native
MLX array handle. The wrapper lets this crate implement `eredu_nn::Tensor`
without placing accelerator code in `eredu-nn` and without violating Rust's
orphan rules.

Applications that use only loading, chat, and generation should normally use
the `eredu` facade. The implementation crate is a direct dependency only for
backend development and backend-specific low-level tooling.

```rust,no_run
use eredu_backend_mlx::{
    native::{backend, Device, DeviceType, ExecutionContext, ModelLoadOptions},
};
use eredu_core::load_model;

let execution = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
let weights = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
let backend = backend(execution.stream(), weights.stream());
let model = load_model(&backend, "/path/to/model", ModelLoadOptions::default())?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

Reusable low-level mechanics are rooted directly under
`eredu_backend_mlx::backend`, including `backend::nn` and `backend::runtime`.
Because this crate implements only MLX, there is no additional backend-name
module below that root.

The `native` module is a deliberate escape hatch for device assignment,
device-bound parallel topology, streams, allocator state, random state,
low-level arrays, sampling, and platform setup needed by concrete MLX tooling.
Prepared models, load and inspection policy that can carry native topology,
backend errors, model-session outputs, and stream-bound checkpoint conversion
also live in this namespace rather than the flat adapter.
`DeviceAssignment` and `MlxParallelContext` are available from `native` and the
reusable `backend` hierarchy, but not from the flat application-facing root.
Native sampling APIs that exchange raw arrays, streams, or random state are
available only through this namespace. Backend-neutral APIs exchange
`MlxTensor` instead of raw arrays.

## Features

All features are opt-in; the default build contains tensor, checkpoint,
execution, distributed, and model-composition support.

| Feature | Additional support |
| --- | --- |
| `accelerate` | Accelerate-backed operations on Apple platforms |
| `metal` | Metal execution on Apple platforms |
| `image` | Image/video decoding and shared multimodal request preparation |
| `audio` | Audio preprocessing and shared multimodal request preparation |
| `codec` | MLX Mimi integration and codec examples; implies `audio` |
| `cuda` | CUDA execution for native MLX operations |
| `nccl` | NCCL collectives; implies `cuda` |

The `metal` and `cuda` features are mutually exclusive.

Enabling either `image` or `audio` compiles the shared multimodal processor;
there is no standalone `media` feature.

The packaged `mimi_realtime_bench`, `personaplex_full_path_bench`, and
`personaplex_quantization_eval` examples require `codec`.
The quantization example is a thin MLX composition entry point over the
backend-neutral driver in `eredu-evaluation`.

Backend-specific reference-fixture generators used by the Moshi and
PersonaPlex parity examples live in `validation/` and are packaged with this
crate. Reusable evidence schemas, comparison policy, and reports remain in
`eredu-evaluation`.

Completed text-session outputs are materialized through the neutral
`BackendSession::observe_output` contract. Explicit instrumented passes bind
the architecture's named activation points to `InspectableBackendSession`.
Parity policy and numerical comparison remain in `eredu-evaluation`; the
backend owns only execution and host observation.

MLX device enumeration reports only device capabilities. Exact session
capabilities are derived from the inspected architecture, residency policy,
and topology before checkpoint payloads are materialized, then verified again
on the realized session. Activation inspection executes the production forward
path for resident, bounded, tensor-parallel, and pipeline/Cartesian sessions.
Distributed results are rank-local: ranks report the global activation paths
they own, and only the final/logits-owning rank reports `model.logits`; no
implicit host gather is performed.
