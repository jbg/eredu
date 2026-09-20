//! Prospective policy injection for lazy header construction.
use std::{any::Any, error::Error, fmt, sync::Arc};

/// Encoded input and first-party buffer size known before header construction.
/// Decoded metadata and dependency overhead require a separate policy estimate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SafetensorsHeaderRequest {
    /// JSON bytes excluding the eight-byte prefix.
    pub json_bytes: usize,
    /// Fresh byte-buffer request, including the prefix.
    pub buffer_bytes: usize,
    /// Encoded path bytes, used when estimating retained per-tensor paths.
    pub path_bytes: usize,
}

/// Opaque reservation retained until the last header or failure alias retires.
/// Construct this only after the caller's admission policy accepts the request.
#[derive(Debug, Clone)]
pub struct SafetensorsHeaderReservation {
    _custody: Arc<dyn HeaderCustody>,
}
trait HeaderCustody: Any + fmt::Debug + Send + Sync {}
impl<T: Any + fmt::Debug + Send + Sync> HeaderCustody for T {}
impl SafetensorsHeaderReservation {
    /// Retains caller-owned reservation custody without exposing it to readers.
    pub fn new<C: Any + fmt::Debug + Send + Sync>(custody: C) -> Self {
        Self {
            _custody: Arc::new(custody),
        }
    }
}

/// Admission policy invoked once when each shared lazy header is initialized.
/// Refusals are retained as header failures and are not automatically retried.
/// This policy covers header construction only, not initial source discovery,
/// index/catalog storage, payload reads, caches or independently cloned metadata.
pub trait SafetensorsHeaderAdmission: fmt::Debug + Send + Sync {
    /// Reserve before allocating/reading the encoded body or decoding metadata.
    fn reserve(
        &self,
        request: SafetensorsHeaderRequest,
    ) -> Result<SafetensorsHeaderReservation, Arc<dyn Error + Send + Sync>>;
}

/// A header error retaining any accepted reservation through all error clones.
#[derive(Debug)]
pub struct SafetensorsHeaderFailure {
    cause: Arc<dyn Error + Send + Sync>,
    _reservation: Option<SafetensorsHeaderReservation>,
}
impl SafetensorsHeaderFailure {
    pub(crate) fn new(
        cause: impl Error + Send + Sync + 'static,
        reservation: Option<SafetensorsHeaderReservation>,
    ) -> Self {
        Self {
            cause: Arc::new(cause),
            _reservation: reservation,
        }
    }
    pub(crate) fn refused(cause: Arc<dyn Error + Send + Sync>) -> Self {
        Self {
            cause,
            _reservation: None,
        }
    }
}
impl fmt::Display for SafetensorsHeaderFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.cause.fmt(f)
    }
}
impl Error for SafetensorsHeaderFailure {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(self.cause.as_ref())
    }
}
