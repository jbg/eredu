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

See the [MLX backend documentation](doc/README.md) for implementation
architecture, native platform setup, and low-level `safemlx` guides.

```rust,no_run
use eredu_backend_mlx::{
    backend::config::ModelLoadOptions,
    native::{backend, Device, DeviceType, ExecutionContext},
};
use eredu_core::load_model;

let execution = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
let weights = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
let backend = backend(execution.stream(), weights.stream());
let model = load_model(&backend, "/path/to/model", ModelLoadOptions::default())?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

Reusable low-level mechanics are rooted directly under
`eredu_backend_mlx::backend`, including neural modules, quantization,
checkpoint materialization, mutable execution state, logical distributed
subgroups, axis-aware collectives, and runtime facilities.
`safemlx` beneath this boundary contains only safe wrappers over native MLX.
Backend-generic sampling policies come from `eredu-runtime`, while
`backend::runtime::generation::MlxSamplingBackend` implements the native
sampling primitives those policies require. Runtime `Sampler` and
`SpeculativeSampler` implementations bind to it through `MlxTensor`; the MLX
backend does not define parallel policy traits.

The `native` module is a deliberate escape hatch for devices, device-bound
streams, allocator state, random state, low-level arrays, and platform setup
needed by concrete MLX tooling. It supplies the native types required by raw
operations alongside composition-owned model sessions, inputs, outputs,
inspection policy, and realtime types. Backend configuration, errors, topology,
prepared models, and reusable runtime facilities are organized under `backend`.
Backend-generic sampling APIs exchange `MlxTensor` values.

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
crate. See the general [evaluation architecture](../doc/evaluation.md) for the
portable observation, evidence, comparison, and reporting contracts.
