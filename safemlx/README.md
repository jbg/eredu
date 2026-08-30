# safemlx

`safemlx` is the low-level MLX implementation used by Eredu's default backend.
It provides arrays and operations, lazy graph evaluation, transforms,
neural-network modules, optimizers, quantization helpers, checkpoint I/O,
streams, and distributed collectives.

Most applications should use [`eredu`](https://github.com/jbg/eredu/tree/main/eredu). Depend on this crate
directly only when building MLX-specific operations or backend components.

## Features

- `accelerate`: Accelerate-backed operations on Apple platforms.
- `metal`: Metal execution on Apple platforms.
- `cuda`: CUDA execution on x86-64 Linux or Windows.
- `nccl`: CUDA plus MLX's optional Linux NCCL backend.
- `safetensors`: conversion between `Array` and `safetensors::TensorView`.

The default feature set enables Accelerate and Metal where those backends are
available.

Operations are lazy and run on an explicit stream:

```rust
use safemlx::{array, Device, DeviceType, Stream};

let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
let values = array!([1.0, 2.0, 3.0]);
let squares = values.square(&stream)?.into_evaluated()?;

assert_eq!(squares.as_slice::<f32>(), &[1.0, 4.0, 9.0]);
# Ok::<(), safemlx::error::Exception>(())
```

Rustdoc contains the API guide and examples for arrays, indexing, lazy
evaluation, graph transforms, and I/O.

## Notable APIs

- Completion events submit selected lazy graphs and support host observation
  or same-device stream ordering without a whole-stream drain.
- Typed host-transfer buffers provide explicit CPU, Metal shared, CUDA pinned,
  or CUDA managed storage policies.
- `GgufCheckpoint` validates a complete sharded GGUF catalog before materializing
  individual tensors.
- Supported packed GGUF weights can execute without a persistent dense copy.
- The `distributed` module wraps MLX groups, collectives, point-to-point
  operations, and variable-count all-to-all.

See the [`safemlx` implementation
guides](https://github.com/jbg/eredu/blob/main/safemlx/doc/README.md) for
completion events, asynchronous device timing, and host-transfer buffers.

## Platforms

Eredu's MLX implementation supports Apple silicon on macOS and selected Apple
device targets, x86-64 Linux with CPU or CUDA, and native x86-64 Windows with
CPU or CUDA. Backend prerequisites and application-bundling requirements are
documented in the MLX backend's [platform setup
guide](https://github.com/jbg/eredu/blob/main/eredu-backend-mlx/doc/platforms.md).

The minimum supported Rust version is 1.89.

## License

Licensed under either MIT or Apache-2.0.
