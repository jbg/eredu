# Host-transfer buffers

`HostTransferBuffer` is typed, host-addressable storage for explicit
array-to-host and host-to-array transfers. It is separate from ordinary MLX
arrays and uses the normal lazy evaluator and completion events for ordering.

## Storage policies

`HostTransferPolicy::Transfer` selects the backend's transfer-ready storage:

| Backend | Storage kind |
| --- | --- |
| CPU | owned CPU allocation |
| Metal | shared Metal storage |
| CUDA | page-locked host storage |

`HostTransferPolicy::Managed` explicitly selects CUDA managed memory. CPU and
Metal reject it rather than silently substituting another storage kind.

A buffer reports its actual storage kind, shape, dtype, logical byte length,
and allocation capacity. Capacity can exceed logical length because backend
allocations are page- or granularity-rounded. Copy operations transfer only the
logical bytes.

`host_transfer_capacity_upper_bound` lets a residency manager reserve a safe
capacity before allocation. `host_transfer_memory_stats` reports active and
peak owned bytes and allocation counts separately for CPU, Metal shared, CUDA
pinned, and CUDA managed storage. Those counters do not claim visibility into
opaque driver bookkeeping.

## Ownership and byte access

The safe Rust API prevents access while a transfer is incomplete:

- `PendingHostTransfer` exposes its destination bytes only after a successful
  completion wait.
- Host-to-array submission consumes a mutable buffer and returns it with the
  completed array, so safe code cannot mutate an in-flight source.
- `freeze()` creates immutable, shareable storage. Each borrowed promotion from
  a frozen buffer has an independent completion event and retains the native
  allocation until that submission finishes.
- Uninitialized bytes require an exclusive mutable borrow.

Ordering a GPU consumer stream makes the buffer safe for that consumer, but it
does not make the bytes safe for CPU observation. Host access still requires a
successful synchronization.

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

SafeMLX uses frozen buffers for host-resident model weights, expert-cache
bindings, and sealed live-cache blocks. A residency report charges the buffer's
physical capacity, not only its logical tensor length.
