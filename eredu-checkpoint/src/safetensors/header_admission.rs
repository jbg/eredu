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
trait HeaderCustody: Any + fmt::Debug + Send + Sync {
    fn complete(&self) -> Result<(), Arc<dyn Error + Send + Sync>>;
}
#[derive(Debug)]
struct Custody<C> {
    value: C,
    complete: fn(&C) -> Result<(), Arc<dyn Error + Send + Sync>>,
}
impl<C: Any + fmt::Debug + Send + Sync> HeaderCustody for Custody<C> {
    fn complete(&self) -> Result<(), Arc<dyn Error + Send + Sync>> {
        (self.complete)(&self.value)
    }
}
impl SafetensorsHeaderReservation {
    /// Retains caller-owned reservation custody without exposing it to readers.
    pub fn new<C: Any + fmt::Debug + Send + Sync>(custody: C) -> Self {
        Self::with_completion(custody, |_| Ok(()))
    }

    /// Retains custody and calls `complete` once after construction succeeds or
    /// fails, before publishing the shared result. Completion ends an active
    /// construction phase; it must not release the retained byte reservation.
    pub fn with_completion<C: Any + fmt::Debug + Send + Sync>(
        custody: C,
        complete: fn(&C) -> Result<(), Arc<dyn Error + Send + Sync>>,
    ) -> Self {
        Self {
            _custody: Arc::new(Custody {
                value: custody,
                complete,
            }),
        }
    }

    /// Payload layout of this reservation's one shared custody allocation.
    /// The caller must also price its allocator/sharing header and policy data.
    pub const fn custody_layout<C>() -> std::alloc::Layout {
        std::alloc::Layout::new::<Custody<C>>()
    }

    pub(crate) fn complete(&self) -> Result<(), Arc<dyn Error + Send + Sync>> {
        self._custody.complete()
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
    ) -> Result<SafetensorsHeaderReservation, Arc<SafetensorsHeaderFailure>>;
}

/// A header error retaining any accepted reservation through all error clones.
#[derive(Debug)]
pub struct SafetensorsHeaderFailure {
    cause: Arc<dyn Error + Send + Sync>,
    completion: Option<Arc<dyn Error + Send + Sync>>,
    _completed: Option<crate::store::AdmittedHeader>,
    _reservation: Option<SafetensorsHeaderReservation>,
}
impl SafetensorsHeaderFailure {
    pub(crate) fn new(
        cause: impl Error + Send + Sync + 'static,
        reservation: Option<SafetensorsHeaderReservation>,
        completion: Option<Arc<dyn Error + Send + Sync>>,
    ) -> Self {
        Self {
            cause: Arc::new(cause),
            completion,
            _completed: None,
            _reservation: reservation,
        }
    }
    /// Wraps a policy refusal. The policy prices this wrapper before constructing
    /// it; the header reader retains the supplied Arc without another allocation.
    pub fn refused(cause: Arc<dyn Error + Send + Sync>) -> Self {
        Self {
            cause,
            completion: None,
            _completed: None,
            _reservation: None,
        }
    }
    pub(crate) fn completion_failed(
        cause: Arc<dyn Error + Send + Sync>,
        header: crate::store::AdmittedHeader,
        reservation: SafetensorsHeaderReservation,
    ) -> Self {
        Self {
            cause: cause.clone(),
            completion: Some(cause),
            _completed: Some(header),
            _reservation: Some(reservation),
        }
    }

    /// Completion refusal, if construction also failed or could not be settled.
    /// The ordinary error source remains the original construction failure when
    /// both fail. All accepted custody and completed metadata remain retained.
    pub fn completion_failure(&self) -> Option<&(dyn Error + Send + Sync + 'static)> {
        self.completion.as_deref()
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
