# Host-transfer buffers

SafeMLX carries `mlx-host-transfer-buffer.patch` against pinned MLX 0.32.0 and
exposes the resulting storage through MLX-C and safe Rust. This is a storage and
copy primitive, not a second execution runtime: transfers are MLX primitives,
are submitted through the ordinary lazy evaluator, and return the existing
single-shot completion type.

## Storage contract

`HostTransferPolicy::Transfer` means transfer-ready, host-addressable storage:

- CPU-only MLX uses owned CPU allocation.
- Metal uses `MTL::StorageModeShared`. It is transfer-ready but remains Apple
  unified memory and does not create capacity separate from system RAM.
- CUDA uses explicit page-locked storage from `cudaMallocHost`.

`HostTransferPolicy::Managed` selects `cudaMallocManaged` explicitly. CPU and
Metal reject it rather than silently mapping it to a different allocation. A
buffer reports its actual `HostTransferStorageKind`, shape, dtype, logical byte
length, and allocation capacity.

CUDA copies use `cudaMemcpyAsync` on the MLX command encoder. When CUDA graphs
are enabled, the encoder records a dependency-tracked memcpy node. Metal uses
its existing GPU copy path into a distinct shared allocation. CPU copies are
queued on the selected CPU stream. Noncontiguous inputs are made contiguous in
the same evaluated graph before transfer.

## Ownership and ordering

Native buffers use shared allocation ownership internally, so destroying a
public handle does not invalidate submitted work. MLX-C exposes opaque owning
handles and returns an `mlx_event` with every submitted copy.

Safe Rust strengthens that contract:

- `PendingHostTransfer` does not expose its destination bytes before a
  successful completion wait.
- Host-to-array submission consumes the host buffer. The completed operation
  returns both the array and reusable buffer, preventing mutation during an
  in-flight copy.
- Uninitialized buffers expose mutable bytes only through an exclusive Rust
  borrow.

Events remain available on pending transfers for completion queries. Host byte
access still requires synchronization; ordering a GPU consumer stream alone
does not make CPU observation safe.

## Current adoption boundary

This change deliberately stops before residency integration. Language-model
weight residency and paged cache residency still represent host blocks as MLX
arrays, so device-to-host demotion still performs its existing completion wait
and independent clone. Replacing those payloads with `HostTransferBuffer` is
the next layer of work and does not require another backend API.
