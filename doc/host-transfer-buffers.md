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
- `freeze()` converts an exclusively mutable buffer into
  `ImmutableHostTransferBuffer`. Frozen storage is shareable and supports
  repeated borrowed host-to-array submissions; every submission retains the
  native allocation through its lazy graph and returns its own completion
  event.
- Uninitialized buffers expose mutable bytes only through an exclusive Rust
  borrow.

Events remain available on pending transfers for completion queries. Host byte
access still requires synchronization; ordering a GPU consumer stream alone
does not make CPU observation safe.

## Current adoption boundary

Paged live-cache residency stores sealed host-tier key/value and compressed-MLA
blocks as immutable typed host-transfer buffers. Its physical block state is a
sum type: a block is device arrays, host buffers, or disk backing, with only the
pending operation valid for that state. Disk reads and writes serialize the
typed host bytes directly, and promotion creates device arrays only on demand.
Device demotion is a fourth, explicitly transitional variant: a dedicated
worker creates a task-local stream on the cache's bound execution device and
retains the device arrays and pending typed destinations through both completion
events. Residency accounting charges those bytes to device and host budgets
simultaneously until immutable host ownership is published.

Immutable weight residency and independent expert caching use the same typed
host-transfer storage for every host-planned parameter binding. Host leases
expose immutable buffers rather than executable arrays; device promotion
submits one buffer-to-array copy per binding and retains the host owners and
their events through the manager's aggregate completion. Device-resident
bindings remain ordinary MLX arrays.
