# safemlx

`safemlx` is a safe Rust interface to the MLX array framework. It provides
arrays and operations, lazy graph evaluation, transforms, neural-network
modules, optimizers, quantization helpers, checkpoint I/O, streams, and
distributed collectives.

Use this crate when you want to build directly on MLX. For a model-oriented
runtime, use [`safemlx-lm`](../safemlx-lm/).

## Features

- `accelerate`: Accelerate-backed operations on Apple platforms.
- `metal`: Metal execution on Apple platforms.
- `cuda`: CUDA execution on x86-64 Linux or Windows.
- `nccl`: CUDA plus MLX's optional Linux NCCL backend.
- `safetensors`: conversion between `Array` and `safetensors::TensorView`.

The default feature set enables Accelerate and Metal where those backends are
available.

## Installation

```toml
[dependencies]
safemlx = "0.1.3"
```

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

See the workspace documentation for the [completion-event
contract](../doc/completion-events.md), [host-transfer
buffers](../doc/host-transfer-buffers.md), and [platform
setup](../doc/platforms.md).

## Platforms

SafeMLX supports Apple silicon on macOS and selected Apple device targets,
x86-64 Linux with CPU or CUDA, and native x86-64 Windows with CPU or CUDA.
Backend prerequisites and application-bundling requirements are documented in
[Platform setup](../doc/platforms.md).

The minimum supported Rust version is 1.89.

## License

Licensed under either MIT or Apache-2.0.
