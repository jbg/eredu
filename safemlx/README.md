# safemlx

`safemlx` provides safe, low-level Rust bindings to MLX: arrays and native
operations, devices and streams, lazy graph evaluation, function transforms,
native serialization, and accelerator/runtime facilities. It deliberately
contains no neural-network layers, models, optimizers, checkpoint policy, or
other framework abstractions. Eredu's MLX-specific abstractions live in
`eredu-backend-mlx`.

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
- The `distributed` module wraps MLX groups, collectives, point-to-point
  operations, and their native execution semantics.

See the [`safemlx` implementation
guides](https://github.com/jbg/eredu/blob/main/safemlx/doc/README.md) for
completion events, asynchronous device timing, and host-transfer buffers.

## Platforms

Eredu's MLX implementation supports Apple silicon on macOS and selected Apple
device targets, x86-64 Linux with CPU or CUDA, and native x86-64 Windows with
CPU or CUDA. The compressed Metal library is embedded automatically. Backend
prerequisites and platform packaging details are documented in the MLX
backend's [platform setup
guide](https://github.com/jbg/eredu/blob/main/eredu-backend-mlx/doc/platforms.md).

The minimum supported Rust version is 1.89.

## License

Licensed under either MIT or Apache-2.0.
