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
backend development and backend-specific low-level tooling:

```toml
[dependencies]
eredu-core = "0.1"
eredu-backend-mlx = "0.1"
```

```rust,no_run
use eredu_backend_mlx::{
    native::{backend, Device, DeviceType, ExecutionContext},
    ModelLoadOptions,
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

The `native` module is a deliberate escape hatch for device, stream, allocator,
random-state, low-level array, sampling, and platform setup needed by concrete
MLX applications. Native sampling APIs that exchange raw arrays, streams, or
random state are available only through this namespace. Backend-neutral APIs
exchange `MlxTensor` instead of raw arrays.

## Features

All features are opt-in; the default build contains tensor, checkpoint,
execution, distributed, and model-composition support.

| Feature | Additional support |
| --- | --- |
| `media` | Typed multimodal processor and request preparation |
| `image` | Image/video decoding primitives; implies `media` |
| `audio` | Audio preprocessing; implies `media` |
| `codec` | MLX Mimi integration and codec examples; implies `audio` |
| `cuda` | SafeMLX CUDA execution |
| `nccl` | NCCL collectives; implies `cuda` |
| `test-support` | Cross-crate backend fixtures; not application API |

The packaged `mimi_realtime_bench`, `personaplex_full_path_bench`, and
`personaplex_quantization_eval` examples require `codec`.

## Migration notes

This crate is new in the backend extraction release. The concrete tensor type
at Eredu's neutral trait boundary is now `eredu_backend_mlx::MlxTensor`, not
`safemlx::Array`. Use `MlxTensor::from_array`, `MlxTensor::as_array`, and
`MlxTensor::into_array` at native integration boundaries; these conversions do
not evaluate or copy the lazy MLX handle.

The former `eredu-nn/mlx` and `eredu-codec/mlx` features were removed. Backend
code depends on `eredu-backend-mlx` directly for concrete tensor or Mimi
integration and enables its `codec` feature for Mimi. Application code uses
the selected adapter exposed by `eredu::api`.
