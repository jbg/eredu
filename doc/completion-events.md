# Completion events

The `safemlx` implementation crate exposes completion events for one submission
of selected lazy MLX graph outputs. Eredu's MLX backend uses these events to
observe a submission from the host or order a compatible consumer stream
without draining unrelated work.

## Submission and observation

`transforms::async_eval_with_event(outputs)` submits the requested outputs and
returns an owning `Event`. Constructing array operations alone does not submit
them.

- `Event::is_complete` is a nonblocking, monotonic query.
- `Event::synchronize` waits on the host.
- `Stream::wait_event` inserts a backend-side dependency for work submitted
  later on the consumer stream.

Producer and consumer devices must match exactly. An empty output set, or one
whose values are already available on the host, returns an already-complete
event.

Events are single-shot: they cannot be re-recorded as timelines. They may be
queried or waited more than once, and more than one compatible stream may wait
on the same event. Dropping a public event handle does not invalidate submitted
producers or queued consumers because the native queues retain the required
ownership.

Asynchronous CPU, Metal, or CUDA failures remain attached to the event.
Repeated queries and waits continue to report the failure, and a failed
dependency poisons the consumer stream rather than allowing it to run with
invalid inputs.

## Ordering boundary

An event covers only the outputs passed to its submission call. It does not
capture unrelated lazy graphs or work submitted later. Applications should
place the wait at the real data dependency and separately evaluate the
consumer graph.

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

Distributed completions are rank-local backend events. They order local MLX
distributed operations and transport payloads; they are not cross-process
event handles. Distributed schedulers perform their own descriptor and success
consensus before publishing a result.

Completion events do not physically cancel kernels or committed command
buffers. Submitted-work cancellation is described in [Cancellation and bounded
execution](cancellation.md).
