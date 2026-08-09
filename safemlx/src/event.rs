use crate::{
    error::Result,
    utils::{guard::Guarded, runtime_lock},
    Device, Stream,
};

/// Execution backend which owns an [`Event`].
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

/// A backend-independent, single-shot MLX completion event.
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
/// `Event` is intentionally neither `Send` nor `Sync`: SafeMLX does not yet
/// expose an upstream MLX guarantee for moving C event handles between host
/// threads. Streams created and used on one host thread remain the supported
/// contract.
pub struct Event {
    pub(crate) c_event: safemlx_sys::mlx_event,
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
