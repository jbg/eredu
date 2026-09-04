use crate::{
    error::Result,
    utils::{guard::Guarded, runtime_lock},
    Device, Stream,
};
use std::time::Duration;

/// MLX device backend which owns an [`Event`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventBackend {
    /// No backend identity because the event was complete when created.
    None,
    /// CPU stream work queue.
    Cpu,
    /// Metal shared-event and command-buffer integration.
    Metal,
    /// CUDA event and stream-wait integration.
    Cuda,
}

/// A single-shot completion event independent of the selected MLX device backend.
///
/// Events are produced by [`crate::transforms::async_eval_with_event`], which
/// explicitly submits the requested lazy graphs for evaluation. Merely
/// constructing an MLX operation does not record it in an event.
///
/// An event can be queried or host-waited repeatedly and can order multiple
/// consumer streams on the same logical device. Waiting on a stream inserts a
/// backend dependency; it does not synchronize the host. Dropping this handle
/// is safe while producer work or queued consumer waits remain outstanding,
/// because MLX retains the underlying event implementation for that work.
///
/// Completion queries are monotonic. If asynchronous execution fails, host
/// waits and completed queries return the retained MLX error; a stream wait
/// poisons dependent work so its later synchronization reports the error.
///
/// `Event` is intentionally neither `Send` nor `Sync`: the MLX implementation does not yet
/// expose an upstream MLX guarantee for moving C event handles between host
/// threads. Streams created and used on one host thread remain the supported
/// contract.
pub struct Event {
    pub(crate) c_event: safemlx_sys::mlx_event,
}

/// An asynchronously submitted evaluation with execution-timeline timestamps.
///
/// This token owns the ordinary completion [`Event`] and the backend timestamp
/// resources recorded by [`crate::transforms::async_eval_timed`]. Submission
/// never waits for the measured work. [`try_elapsed`](Self::try_elapsed) is
/// nonblocking; [`elapsed`](Self::elapsed) is the explicit host synchronization
/// point.
///
/// The measured interval begins after earlier work on the selected stream and
/// ends after the requested lazy graph has executed. It therefore excludes
/// unrelated work queued before the start marker. Dependencies on other
/// streams are honored. Metal sums native command-buffer GPU start/end
/// intervals and therefore excludes gaps when no measured command buffer is
/// active; CUDA event and CPU marker intervals include waits and idle gaps
/// between their markers.
///
/// Repeated elapsed-time queries return one cached duration. Backend failures
/// are retained and returned by both completion and timing resolution.
pub struct TimedEvaluation {
    completion: Event,
}

impl TimedEvaluation {
    pub(crate) fn from_completion(completion: Event) -> Self {
        Self { completion }
    }

    /// The completion event for dependency insertion or completion queries.
    pub fn event(&self) -> &Event {
        &self.completion
    }

    /// Query whether the evaluation and both timestamp markers are complete.
    pub fn is_complete(&self) -> Result<bool> {
        Ok(self.try_elapsed()?.is_some())
    }

    /// Block the host until evaluation completes, without resolving duration.
    pub fn synchronize(&self) -> Result<()> {
        self.completion.synchronize()
    }

    /// Return elapsed execution-timeline time, blocking only here if needed.
    ///
    /// Metal timestamps have command-buffer accuracy and exclude command-queue
    /// wait time; CUDA event precision is typically around half a microsecond;
    /// CPU timing is limited by the platform monotonic clock and scheduler
    /// dispatch. An empty or already-completed evaluation is defined to take
    /// zero time.
    pub fn elapsed(&self) -> Result<Duration> {
        let _guard = runtime_lock::enter();
        let seconds = f64::try_from_op(|seconds| unsafe {
            safemlx_sys::mlx_event_elapsed(seconds, self.completion.c_event)
        })?;
        duration_from_seconds(seconds)
    }

    /// Query elapsed time without blocking the host.
    ///
    /// Returns `Ok(None)` while work is outstanding. Once ready, this and
    /// [`elapsed`](Self::elapsed) return the same stable duration.
    pub fn try_elapsed(&self) -> Result<Option<Duration>> {
        let _guard = runtime_lock::enter();
        let mut seconds = 0.0;
        let mut ready = false;
        <() as Guarded>::try_from_op(|_| unsafe {
            safemlx_sys::mlx_event_try_elapsed(&mut seconds, &mut ready, self.completion.c_event)
        })?;
        ready.then(|| duration_from_seconds(seconds)).transpose()
    }
}

fn duration_from_seconds(seconds: f64) -> Result<Duration> {
    Duration::try_from_secs_f64(seconds).map_err(|_| {
        crate::error::Exception::custom(format!(
            "backend returned invalid elapsed time: {seconds} seconds"
        ))
    })
}

impl std::fmt::Debug for TimedEvaluation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TimedEvaluation")
            .field("completion", &self.completion)
            .field("elapsed", &self.try_elapsed())
            .finish()
    }
}

impl Event {
    /// Block the host until this event completes.
    ///
    /// Any asynchronous backend error retained by the event is returned. This
    /// method may be called repeatedly.
    pub fn synchronize(&self) -> Result<()> {
        let _guard = runtime_lock::enter();
        <() as Guarded>::try_from_op(|_| unsafe {
            safemlx_sys::mlx_event_synchronize(self.c_event)
        })
    }

    /// Query completion without blocking the host.
    ///
    /// Once this returns `true`, later successful queries also return `true`.
    /// A completed event with an asynchronous failure returns that error.
    pub fn is_complete(&self) -> Result<bool> {
        let _guard = runtime_lock::enter();
        bool::try_from_op(|complete| unsafe {
            safemlx_sys::mlx_event_query(complete, self.c_event)
        })
    }

    /// Attempts a completion query without waiting for the process-wide MLX
    /// runtime lock. `None` means another host call currently owns that lock;
    /// it does not imply anything about device completion.
    pub fn try_is_complete(&self) -> Result<Option<bool>> {
        let Some(_guard) = runtime_lock::try_enter() else {
            return Ok(None);
        };
        bool::try_from_op(|complete| unsafe {
            safemlx_sys::mlx_event_query(complete, self.c_event)
        })
        .map(Some)
    }

    /// Runs `on_complete` under the same nonblocking runtime-lock acquisition
    /// which observes exact event completion. This prevents another blocking
    /// MLX host call from entering between the completion query and a required
    /// materialized-result read.
    pub fn try_with_complete<T>(
        &self,
        on_complete: impl FnOnce() -> Result<T>,
    ) -> Result<Option<T>> {
        let Some(_guard) = runtime_lock::try_enter() else {
            return Ok(None);
        };
        let complete = bool::try_from_op(|complete| unsafe {
            safemlx_sys::mlx_event_query(complete, self.c_event)
        })?;
        if complete {
            on_complete().map(Some)
        } else {
            Ok(None)
        }
    }

    /// Order subsequently submitted work on `stream` after this event.
    ///
    /// The producer and consumer must have the same logical MLX device. The
    /// dependency is encoded by MLX's CPU, Metal, or CUDA backend; no host wait
    /// is performed. Because MLX operations are lazy, call this before
    /// evaluating the consumer graph, not merely before constructing it.
    pub fn wait_on(&self, stream: impl AsRef<Stream>) -> Result<()> {
        stream.as_ref().wait_event(self)
    }

    /// Return the producer device, or `None` for an event which was already
    /// complete and therefore has no producer identity.
    pub fn device(&self) -> Result<Option<Device>> {
        let _guard = runtime_lock::enter();
        let has_device = bool::try_from_op(|present| unsafe {
            safemlx_sys::mlx_event_has_device(present, self.c_event)
        })?;
        if has_device {
            Device::try_from_op(|device| unsafe {
                safemlx_sys::mlx_event_get_device(device, self.c_event)
            })
            .map(Some)
        } else {
            Ok(None)
        }
    }

    /// Return the backend identity of this event.
    pub fn backend(&self) -> Result<EventBackend> {
        let _guard = runtime_lock::enter();
        let raw = u32::try_from_op(|backend| unsafe {
            safemlx_sys::mlx_event_get_backend(backend.cast(), self.c_event)
        })?;
        match raw {
            safemlx_sys::mlx_event_backend__MLX_EVENT_BACKEND_NONE => Ok(EventBackend::None),
            safemlx_sys::mlx_event_backend__MLX_EVENT_BACKEND_CPU => Ok(EventBackend::Cpu),
            safemlx_sys::mlx_event_backend__MLX_EVENT_BACKEND_METAL => Ok(EventBackend::Metal),
            safemlx_sys::mlx_event_backend__MLX_EVENT_BACKEND_CUDA => Ok(EventBackend::Cuda),
            _ => unreachable!("MLX returned an unknown completion backend"),
        }
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        let _guard = runtime_lock::enter();
        let status = unsafe { safemlx_sys::mlx_event_free(self.c_event) };
        debug_assert_eq!(status, crate::utils::SUCCESS);
    }
}

impl std::fmt::Debug for Event {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Event")
            .field("backend", &self.backend())
            .field("device", &self.device())
            .field("complete", &self.is_complete())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completion_query_does_not_wait_for_busy_runtime_lock() {
        let event =
            crate::transforms::async_eval_with_event(std::iter::empty::<&crate::Array>()).unwrap();
        let (ready_tx, ready_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let holder = std::thread::spawn(move || {
            let _guard = runtime_lock::enter();
            ready_tx.send(()).unwrap();
            release_rx.recv().unwrap();
        });
        ready_rx.recv().unwrap();

        let started = std::time::Instant::now();
        assert_eq!(event.try_is_complete().unwrap(), None);
        assert!(started.elapsed() < std::time::Duration::from_millis(50));

        release_tx.send(()).unwrap();
        holder.join().unwrap();
        event.synchronize().unwrap();
    }
}
