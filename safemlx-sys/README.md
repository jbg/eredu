# safemlx-sys

`safemlx-sys` provides low-level Rust bindings and native build integration for
the MLX C API used by Eredu's `safemlx` implementation layer. Applications
should use [`eredu`](https://github.com/jbg/eredu/tree/main/eredu) rather than depending on this crate directly.

The crate vendors MLX C, the compatible MLX source archive, and its common CPU
and Metal build dependencies, and exposes checked-in bindings. CMake verifies
each vendored archive's SHA-256 before extracting and patching it in the build
directory. Source provenance is recorded in
[`vendor/SOURCES.md`](vendor/SOURCES.md). The native surface includes the
completion events, typed host-transfer storage, variable-count all-to-all, and
packed-quantization support required by Eredu's MLX backend.
Event, timed-evaluation, and host-transfer C producers allocate output wrappers
before native submission. Successful publication uses nonthrowing moves;
producer errors restore initially empty or previously populated output handles.
Submission scopes and exact CPU/GPU progress records retain native resources
when scheduling fails before an event can be published. Cleanup uses terminal
evidence rather than retrying evaluation; unresolved work remains owned without
process termination or a blocking destructor.
Scope progress publishes terminal evidence separately from primitive-owner
destruction. Ordinary native calls reclaim terminal records on their original
owner thread; unresolved records and records whose owner has exited remain
retained. Patch-content identities select fresh extracted source trees, so
changing a native patch cannot silently reuse an older successful patch stamp.

Metal two-pass attention captures `MLX_SDPA_BLOCKS` once, on first cold inspection
or execution. Positive multiples of 32 select that exact scratch-block count;
unset, non-numeric and nonpositive values use device defaults. Integer overflow
and positive counts incompatible with the native 32-block reduction are rejected.
Later environment edits cannot invalidate a previously inspected allocation bound. The standalone CMake target `mlx-cold-sdpa-settings-test`
checks retained settings and concurrent reads without initializing a device.

`mlx_array_allocation_info` reads the backing identity and full capacity of
completed native allocator-owned or certified host-transfer-backed arrays. It does not evaluate, poll, wait, or
detach completion events. Synchronously evaluated arrays without an event are
complete; unresolved event-backed arrays remain unknown. The query checks the
native allocator deleter before asking for buffer capacity, because a foreign
CPU pointer is not a valid allocator-size argument. The host-transfer patch uses
a named retaining deleter and verifies its exact buffer before reporting that
owner's charged capacity and identity. A host buffer and its completed array/view
aliases report the same identity in a separate host-transfer domain. Other custom
owned/borrowed storage remains unknown unless construction copied it into
native-owned storage. Identities use non-reused 64-bit generations, rather than
buffer addresses. The allocation itself still needs a native owner to remain
alive; copying its identity does not retain its backing.
The safe wrapper translates these facts inside the existing native FFI safety
boundary; portable crates receive only neutral storage identities and byte
capacities. These are existing-allocation facts, not a future or process-memory
bound.

`mlx-allocation-lifetime-generations.patch` assigns generations when physical
`array::Data` or `HostTransferStorage` ownership begins. One out-of-line MLX
function owns the atomic generation counter; callers in other translation units
and the C wrapper do not instantiate their own counters. Native aliases and
donated backing preserve the `Data` generation. Independently wrapped arrays
sharing host storage use that storage's generation. A reused buffer address
therefore cannot collide with an older charge awaiting deferred host cleanup.
These identities are local to one linked MLX runtime instance: they are not
checkpoint identifiers and must not be persisted or compared across processes
or independently loaded runtimes.

The counter issues each nonzero `uint64_t` once and remains at an exhausted zero
sentinel after the last value. Exhaustion does not prevent allocation or change
normal deallocation; it makes backing uncertifiable. Array queries then report
unknown and the safe host-buffer query returns an error. Completed attachment
APIs preserve caller ownership when a generation is unavailable. Allocation-free
empty array descriptors can still report their known zero-byte bound without a
physical generation. The standalone `mlx-allocation-generations-test` target
checks concurrent exhaustion without resetting the production counter,
deterministic reuse of the same backing address while old identities remain
retained, move/alias preservation, and unknown-generation rejection.

`mlx_array_retain_allocation_owner` attaches an opaque owner to completed,
certified physical storage without replacing the data pointer or allocator
deleter. The `mlx-allocation-owner-retention.patch` keeps the owner on MLX's
shared `array::Data`, so preexisting views, lazy dependencies, native submission
pins, and donated backing preserve its lifetime. Certified host-transfer arrays
attach to `HostTransferStorage` instead: separately created array wrappers and
the host buffer can share that physical storage. The matching
`mlx_host_transfer_buffer_retain_allocation_owner` API attaches directly to that
storage without constructing or evaluating an array; the safe entry point is
`ImmutableHostTransferBuffer::retain_allocation_owner`. Independent copied allocations
require independent attachments. Unfinished, foreign and allocation-free
backing cannot accept an attachment.

The C caller serializes attachment and retains payload ownership unless the
function succeeds with `attached == true`. Callback ownership is armed only
after every fallible allocation and insertion, so errors cannot consume the
payload. The callback runs after physical storage retires and can execute on
any native thread; it must not throw, block, allocate or reenter MLX. The safe
`Array::retain_allocation_owner` wrapper accepts `Send + 'static` owners and
returns the original owner with an attachment error. Its callback only publishes
a preallocated node to an atomic retirement queue. `reclaim_allocation_owners`
drops those owners on ordinary host code outside the native runtime lock;
ordinary runtime entry also reclaims before acquiring that lock. Reclamation
is suppressed during unwinding and reentrant cleanup, and a panicking owner
leaves other queued owners available for later reclamation. Explicitly reclaim
after final native cleanup when no further runtime calls are expected.

An attached owner must not itself retain the covered allocation or an ancestor
graph that retains it, because that creates an ownership cycle. Attach a charge
or lifetime lease rather than a physical inventory containing the same array.
This mechanism preserves resource ownership; it does not quote the owner,
sidecar metadata, future copies or execution workspace, and does not establish
a managed-memory budget by itself. Its unsafe implementation remains confined
to the existing `safemlx`/`safemlx-sys` FFI boundary.

`mlx_array_retain_deferred_allocation_owner` is a separate lazy API. The safe
entry point, `Array::retain_deferred_allocation_owner`, accepts payload-free
`Send + Sync + 'static` authority without evaluating, polling, waiting, or
changing completion status. Existing clones share a descriptor-owned bundle.
Both `set_data` overloads and `copy_shared_buffer` publish that bundle to the
physical native or certified host owner before kernels use the backing. Views,
donation, submission recovery pins, and separately wrapped host aliases then
retain it. Later attachments to the descriptor update the same bundle already
retained by aliases. Independently allocated results need their own attachment.
The bundle contains no graph or storage roots; callers must uphold the same
no-cycle contract as the completed API.

An abandoned or failed lazy graph retains its authority until the descriptor
retires. Empty descriptors can retain it without creating physical storage.
Existing uncertified backing rejects attachment and returns the payload;
future uncertified backing, including an exhausted generation, rejects
materialization before installing that backing and preserves descriptor
authority. Staged physical storage is released normally on failure. The C
payload remains disarmed until all fallible bundle allocation and publication
have succeeded. Retirement uses the same deferred host queue as completed
attachment, so arbitrary Rust destructors never run under native locks.

The wrapper serializes attachment and native graph encoding. In the pinned
CPU/Metal implementation, primitive evaluation installs data inline before
dispatch; native workers use already-published pointers and ownership pins.
Internal `unsafe_weak_copy` descriptors deliberately remain non-owning and do
not duplicate the bundle. Their real backing remains pinned by the encoder or
submission recovery. Worker reference retirement cannot destroy an owner list
while attachment retains its descriptor, bundle and physical owner. This
contract does not permit arbitrary concurrent native graph mutation or add a
memory-bound guarantee.

The standalone `mlx-deferred-allocation-owner-test` target checks rejected
borrowed/future backing, exact shared-buffer donation lifetime, failure after
native work is accepted, and attachment while a gated CPU worker is pending.
The safe-wrapper lifecycle tests additionally exercise CPU and Metal lazy
clones/views, unmaterialized retirement, async submission, late attachment,
shared host aliases, and unlocked Rust destruction.

The focused lifecycle checks run CPU and Metal in one default-feature build:

```sh
CARGO_INCREMENTAL=0 cargo test -p safemlx --lib allocation_retention -- --test-threads=1 --nocapture
```

## Backends

| Target | Backend selection |
| --- | --- |
| macOS and supported Apple device targets | Accelerate and Metal through the default features |
| x86-64 Linux | CPU by default; optional CUDA and NCCL |
| x86-64 Windows MSVC | CPU by default; optional CUDA |

Native compilers, CMake, platform libraries, CUDA/cuDNN setup, Apple deployment
targets, and the embedded Metal library are covered in [Platform
setup](https://github.com/jbg/eredu/blob/main/eredu-backend-mlx/doc/platforms.md).

## Features

- `accelerate`: build the Accelerate backend on Apple platforms.
- `metal`: build the Metal backend on Apple platforms.
- `cuda`: build MLX with CUDA support on x86-64 Linux or Windows.
- `nccl`: enable MLX's optional NCCL distributed backend on Linux.

Applications normally select these features through `eredu` rather than
depending on `safemlx-sys` directly.

## License

The Rust crate is MIT licensed. Vendored components retain their stated
licenses and attribution files.
