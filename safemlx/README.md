# safemlx

Rust bindings for the MLX machine learning framework.

`safemlx` provides a safe, idiomatic Rust interface over the low-level
`safemlx-sys` bindings. It includes array operations, neural-network building
blocks, transforms, optimizers, quantization helpers, optional SafeTensors
support, and typed streaming GGUF tensor/metadata loading.

This crate targets macOS 14+, iOS/iPadOS 17+, tvOS 17+, and visionOS 1+ on
Apple silicon, as well as CPU-only and NVIDIA CUDA Linux systems and native
Windows x86-64 MSVC. The default feature set enables Accelerate and Metal on
Apple targets; those features are ignored on Linux and Windows, where `cuda`
can be selected explicitly. Cross-compilation, Xcode Metal-resource integration,
and native backend prerequisites are documented in the
[`safemlx-sys` README](../safemlx-sys/README.md).

GGUF checkpoints are opened with `ops::GgufCheckpoint`. Opening validates all
canonical shard headers without reading payloads; `converted_tensors` and
`for_each_converted_tensor` then materialize one physical tensor as either a
dense array or one atomic affine weight/scales/biases group.
`GgufCheckpoint::materializer` provides indexed named access while reusing one
open shard reader, which is useful for bounded multi-tensor model transforms.

## Checkpoint-native quantization

`native_quantization` retains Q4_K, Q5_1, Q8_0, and the canonical GGML IQ
encodings as their original packed blocks. On Metal, decode and prefill use
direct packed kernels; prefill reuses decoded weights across eight activation
rows at a time. IQ routed experts also fuse gate/GELU/up and
down/route-weight/reduction operations. Embedding and grouped linear operations
do not require a persistent dense copy.

CPU execution is bounded as well: it decodes one logical weight row at a time
and accumulates directly into the result. Peak scratch is proportional to one
row, not the full matrix. This preserves packed resident memory, although the
custom kernels remain correctness-first and can be slower than dense F16 for
small matrices. Run the reproducible packed-versus-dense microbenchmark with:

```console
cargo bench -p safemlx --bench native_quantization
```

`SAFEMLX_BENCH_ROWS`, `SAFEMLX_BENCH_COLUMNS`, and
`SAFEMLX_BENCH_ITERS` control its dimensions and sample count.

## Features

- `accelerate`: enables Accelerate-backed MLX operations.
- `cuda`: builds MLX's CUDA backend on Linux or Windows x86-64 MSVC.
- `metal`: enables Metal-backed MLX operations.
- `nccl`: enables CUDA plus MLX's optional Linux-only NCCL distributed backend.
- `safetensors`: enables conversion between `Array` and
  `safetensors::TensorView`.

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
safemlx = "0.1"
```

## Asynchronous completion events

MLX graphs are lazy. `async_eval_with_event` is the point which submits the
requested outputs; graph construction by itself does not record work in an
event. A compatible consumer stream can wait without blocking the host:

```rust
use safemlx::{
    transforms::async_eval_with_event, Array, Device, DeviceType, Stream,
};

let device = Device::new(DeviceType::Cpu, 0);
let producer = Stream::new_with_device(&device);
let consumer = Stream::new_with_device(&device);
let output = Array::ones::<f32>(&[16], &producer)?.square(&producer)?;
let completion = async_eval_with_event([&output])?;

consumer.wait_event(&completion)?;
let consumed = output.add(&Array::from(1.0f32), &consumer)?;
async_eval_with_event([&consumed])?.synchronize()?;
# Ok::<(), safemlx::error::Exception>(())
```

Events are single-shot but support repeated query/host waits and multiple
consumer waits. Producer and consumer devices must match. Dropping the public
handle is safe after a wait is queued because MLX retains the backend event.
Asynchronous failures are retained and returned by host observation or later
consumer synchronization.

## Host transfer buffers

`HostTransferBuffer` is a typed allocation for explicit host/device transfers,
separate from ordinary MLX arrays. `HostTransferPolicy::Transfer` selects
ordinary CPU storage, Metal shared storage, or CUDA page-locked host storage;
`HostTransferPolicy::Managed` is a distinct CUDA-only policy and fails on other
backends. The physical selection is available through `storage_kind()`.

Transfers return pending values which retain their storage and completion
event. Array-to-host bytes are unavailable until `synchronize()` succeeds.
Host-to-array submission consumes the host buffer and returns it with the
completed array, preventing safe Rust from mutating source bytes while DMA is
in flight. A buffer can instead be frozen into immutable shareable storage;
borrowed submissions from that storage support repeated promotions and each
return an independently ordered completion event.

```rust
use safemlx::{
    Array, Device, DeviceType, HostTransferBuffer, HostTransferPolicy, Stream,
};

let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
let source = Array::from_slice(&[1.0f32, 2.0], &[2]);
let host = HostTransferBuffer::copy_from_array(
    &source,
    HostTransferPolicy::Transfer,
    &stream,
)?.synchronize()?;
let (restored, host) = host.copy_to_array(&stream)?.synchronize()?;
# let _ = (restored, host);
# Ok::<(), safemlx::error::Exception>(())
```

Backend verification lives in `tests/host_transfer.rs`. CPU-only tests assert
exact `Cpu` storage, numerical round-trip parity, and allocator high-water
bounds. Explicit Metal tests assert `MetalShared` storage, Metal completion
events, cross-stream ordering, and that device-to-host or host-to-device copies
do not add another payload-sized MLX staging allocation. CUDA-gated tests make
the corresponding assertions for `CudaPinned` storage and separately verify
that the managed policy reports `CudaManaged`; pinned host bytes are outside
the MLX device allocator counter, so any payload-sized device-counter increase
during device-to-host copying is treated as hidden staging.

```sh
cargo test -p safemlx --no-default-features --test host_transfer
cargo test -p safemlx --test host_transfer metal_ -- --ignored
cargo test -p safemlx --features cuda --no-default-features \
  --test host_transfer cuda_ -- --ignored
```

The language-model paged live-cache manager uses frozen buffers for host-tier
key/value and compressed-MLA blocks. Weight residency and independent expert
caching have not yet adopted this storage class.

## Distributed MLX

The `distributed` module wraps MLX groups, collectives, and point-to-point
operations with owned handles and explicit streams. Non-strict initialization
keeps MLX's useful singleton fallback:

```rust
use safemlx::{distributed::{self, Backend}, Array, Device, DeviceType, Stream};

let group = distributed::init(false, Backend::Any)?;
let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
let input = Array::ones::<f32>(&[2], &stream)?;
let sum = distributed::all_sum(&input, &group, &stream)?;
# Ok::<(), safemlx::error::Exception>(())
```

`distributed::all_to_all_v` exchanges compact leading-axis row blocks without
padding. Its input is destination-major, its output is source-major, and the
caller supplies checked row counts for both layouts. Native MPI, Ring, JACCL,
and NCCL groups use the patched MLX primitive lazily. Logical Cartesian
subgroups which Ring or JACCL cannot split use the existing topology-planned
neighbor routes and transfer only addressed payloads; their small routed count
headers are materialized to size intermediate receives. Ring payloads may cross
multiple physical hops, so endpoint bytes and hop bytes are not equivalent.

Choose process-local devices with `distributed::device_for_local_rank`. A
global distributed rank is not a local GPU index because ranks may span
machines. In a one-process-per-visible-GPU launch, the local device index is
often zero: `CUDA_VISIBLE_DEVICES` has already restricted each process to one
GPU.

The real two-process Ring integration test is opt-in because it launches child
processes and opens loopback sockets. Run it on Unix with:

```console
cargo test -p safemlx --test distributed_ring ring_two_process_loopback -- --ignored --exact --nocapture
```

## Versioning

The `safemlx` crates use normal Rust semantic versioning. The initial
crates.io release is `0.1.0`.

## Status

`safemlx` is in active development.

## MSRV

The minimum supported Rust version is 1.89.0.

Each published crate declares its MSRV in `Cargo.toml`.

## License

Licensed under either MIT or Apache-2.0.
