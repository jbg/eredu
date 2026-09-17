//! Move-only error handoff for the shared ordinary background worker.
use std::{any::Any, error::Error, fmt, sync::Mutex};

/// A concrete worker failure. Implementations preserve any source/accepted
/// custody; conversion and arbitrary destruction occur outside lifecycle locks.
pub trait BackgroundPrefetchFailure: fmt::Debug + fmt::Display + Send + 'static {
    /// Convert the exact caught panic payload without panicking. No native
    /// completion or permission is implied by a worker panic.
    fn from_panic(payload: Box<dyn Any + Send>) -> Self;
    /// Retained typed source when this transport carries an actual error.
    fn error_source(&self) -> Option<&(dyn Error + 'static)> {
        None
    }
}
impl BackgroundPrefetchFailure for String {
    fn from_panic(payload: Box<dyn Any + Send>) -> Self {
        payload
            .downcast_ref::<&str>()
            .map(|message| (*message).to_string())
            .or_else(|| payload.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "background prefetch operation panicked".to_string())
    }
}

/// Owns a panic payload until demand/cancellation/error retirement. The mutex
/// permits a Send-only payload to remain a Sync error source without an unsafe
/// assertion, cloning the payload, or running its destructor on a worker lock.
pub struct BackgroundPrefetchPanic {
    payload: Mutex<Box<dyn Any + Send>>,
}
impl BackgroundPrefetchPanic {
    /// Take the exact caught payload. This adds no heap allocation.
    pub fn new(payload: Box<dyn Any + Send>) -> Self {
        Self {
            payload: Mutex::new(payload),
        }
    }
}
impl fmt::Debug for BackgroundPrefetchPanic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BackgroundPrefetchPanic")
            .finish_non_exhaustive()
    }
}
impl fmt::Display for BackgroundPrefetchPanic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Ok(payload) = self.payload.lock() else {
            return f.write_str("background prefetch operation panicked");
        };
        if let Some(message) = payload.downcast_ref::<&str>() {
            f.write_str(message)
        } else if let Some(message) = payload.downcast_ref::<String>() {
            f.write_str(message)
        } else {
            f.write_str("background prefetch operation panicked")
        }
    }
}
impl Error for BackgroundPrefetchPanic {}
