//! Custom error types and handler for the c ffi

use crate::Dtype;
use std::convert::Infallible;
use std::ffi::{CStr, NulError};
use std::panic::Location;
use std::sync::Once;
use std::{cell::Cell, cell::RefCell, ffi::c_char};
use thiserror::Error;
mod retained;
use retained::ExceptionSource;

/// Type alias for a `Result` with an `Exception` error type.
pub type Result<T> = std::result::Result<T, Exception>;

/// Error with io operations
#[derive(Error, PartialEq, Debug)]
pub enum IoError {
    /// Path must point to a local file
    #[error("Path must point to a local file")]
    NotFile,

    /// Path contains invalid UTF-8
    #[error("Path contains invalid UTF-8")]
    InvalidUtf8,

    /// Path contains null bytes
    #[error("Path contains null bytes")]
    NullBytes,

    /// No file extension found
    #[error("No file extension found")]
    NoExtension,

    /// Unsupported file format
    #[error("Unsupported file format")]
    UnsupportedFormat,

    /// Invalid or unsupported serialized data
    #[error("invalid serialized data: {0}")]
    InvalidFormat(String),

    /// Unable to open file
    #[error("Unable to open file")]
    UnableToOpenFile,

    /// Unable to allocate memory
    #[error("Unable to allocate memory")]
    AllocationError,

    /// Null error
    #[error(transparent)]
    NulError(#[from] NulError),

    /// Exception
    #[error(transparent)]
    Exception(#[from] Exception),
}

impl From<Infallible> for IoError {
    fn from(_: Infallible) -> Self {
        unreachable!()
    }
}

impl From<RawException> for IoError {
    #[track_caller]
    fn from(e: RawException) -> Self {
        let exception = Exception::from(e);
        Self::Exception(exception)
    }
}

/// A synchronous copy from completed native storage into caller-owned host storage.
#[derive(Debug, PartialEq, Error)]
pub enum CompletedReadbackError {
    /// The cold descriptor query could not establish the retained source facts.
    #[error(transparent)]
    Descriptor(#[from] crate::ArrayDescriptorError),
    /// The caller supplied a different number of logical elements.
    #[error("completed readback destination has {found} elements, expected {expected}")]
    DestinationLength {
        /// Required logical element count.
        expected: usize,
        /// Supplied destination element count.
        found: usize,
    },
    /// The completed backing or its declared layout is unavailable.
    #[error(transparent)]
    Source(#[from] AsSliceError),
    /// Native serialization is held by another thread; no copy started.
    #[error("completed readback runtime is busy")]
    Busy,
    /// The authenticated native backing or extent failed validation.
    #[error("completed readback storage attribution is unavailable")]
    Attribution,
    /// The synchronous CUDA worker failed; its exact runtime status is retained.
    #[error("completed CUDA readback failed with runtime status {0}")]
    CudaStatus(i32),
}

/// A failed copy of evaluated values into caller-owned native-endian bytes.
#[derive(Debug, PartialEq, Error)]
pub enum NativeBytesCopyError {
    /// A synchronous native copy failed after completion-safe return.
    #[error(transparent)]
    Readback(#[from] CompletedReadbackError),
    /// The destination must hold exactly the logical array's bytes.
    #[error("native byte destination has {found} bytes, expected {expected}")]
    DestinationLength {
        /// Required logical byte count.
        expected: usize,
        /// Supplied byte count.
        found: usize,
    },
    /// The evaluated source cannot be read with its declared layout.
    #[error(transparent)]
    Source(#[from] AsSliceError),
}

/// Error associated with `Array::try_as_slice()`
#[derive(Debug, PartialEq, Error)]
pub enum AsSliceError {
    /// A borrowed Rust slice cannot represent a strided or broadcast view.
    #[error("The array is not contiguous in logical row order.")]
    NonContiguous,

    /// Shape and stride metadata do not describe a valid logical array.
    #[error("The array has invalid shape or stride metadata.")]
    InvalidLayout,

    /// The underlying data pointer is null.
    ///
    /// This is likely because the array has not been evaluated yet.
    #[error("The data pointer is null.")]
    Null,

    /// The underlying data pointer does not satisfy the element alignment.
    #[error("The data pointer is not aligned for the requested element type.")]
    Misaligned,

    /// The requested slice would exceed Rust's maximum allocation size.
    #[error("The array is too large to represent as a Rust slice.")]
    TooLarge,

    /// The output dtype does not match the data type of the array.
    #[error("dtype mismatch: expected {expecting:?}, found {found:?}")]
    DtypeMismatch {
        /// The expected data type.
        expecting: Dtype,

        /// The actual data type
        found: Dtype,
    },

    /// Exception
    #[error(transparent)]
    Exception(#[from] Exception),
}

cfg_safetensors! {
    /// Error associated with conversion between `safetensors::tensor::TensorView` and `Array`
    /// when the data type is not supported.
    #[derive(Debug, Error)]
    pub enum ConversionError {
        /// The safetensors data type that is not supported.
        ///
        /// This is the error type for conversions from `safetensors::tensor::TensorView` to `Array`.
        #[error("The safetensors data type {0:?} is not supported.")]
        SafeTensorDtype(safetensors::tensor::Dtype),

        /// The mlx data type that is not supported.
        ///
        /// This is the error type for conversions from `Array` to `safetensors::tensor::TensorView`.
        #[error("The mlx data type {0:?} is not supported.")]
        MlxDtype(crate::Dtype),

        /// Error casting the data buffer to `&[u8]`.
        #[error(transparent)]
        PodCastError(#[from] bytemuck::PodCastError),

        /// Error with creating a `safetensors::tensor::TensorView`.
        #[error(transparent)]
        SafeTensorError(#[from] safetensors::tensor::SafeTensorError),

        /// MLX rejected construction of an externally backed array.
        #[error(transparent)]
        Exception(#[from] Exception),
    }
}

/// Enforced submission tracking ceiling failure. This does not imply completion.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
pub enum SubmissionTrackingFailure {
    /// The same original arena could not reserve the requested physical block.
    #[error("submission tracking capacity exhausted")]
    Exhausted,
    /// Concrete capacity/alignment arithmetic was invalid.
    #[error("invalid submission tracking layout")]
    InvalidLayout,
    /// A nested scope attempted to substitute another arena.
    #[error("submission tracking parent mismatch")]
    ParentMismatch,
    /// An extension attempted an unbound Record allocation inside a quota.
    #[error("record requires its bounded factory")]
    UnboundRecord,
}
impl SubmissionTrackingFailure {
    fn from_native(code: u32) -> Option<Self> {
        match code {
            1 => Some(Self::Exhausted),
            2 => Some(Self::InvalidLayout),
            3 => Some(Self::ParentMismatch),
            4 => Some(Self::UnboundRecord),
            _ => None,
        }
    }
}
/// Enforced graph metadata ceiling failure. This does not imply completion.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
pub enum GraphMetadataFailure {
    /// The same original arena could not reserve the requested physical block.
    #[error("graph metadata capacity exhausted")]
    Exhausted,
    /// Concrete capacity/alignment arithmetic was invalid.
    #[error("invalid graph metadata layout")]
    InvalidLayout,
    /// A nested scope attempted to substitute another arena.
    #[error("graph metadata parent mismatch")]
    ParentMismatch,
}
impl GraphMetadataFailure {
    fn from_native(code: u32) -> Option<Self> {
        match code {
            1 => Some(Self::Exhausted),
            2 => Some(Self::InvalidLayout),
            3 => Some(Self::ParentMismatch),
            _ => None,
        }
    }
}
pub(crate) struct RawException {
    pub(crate) what: String,
    tracking: Option<SubmissionTrackingFailure>,
    graph: Option<GraphMetadataFailure>,
}

/// Fixed result from original scoped synchronous evaluation. This is not
/// completion or recovery authority; native source storage remains retained.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
pub enum ScopedEvaluationCause {
    /// The borrowed native value or transport was invalid.
    #[error("invalid scoped evaluation")]
    Invalid,
    /// A fixed original capacity refused the operation.
    #[error("scoped evaluation capacity exhausted")]
    Capacity,
    /// A native allocation failed before completion.
    #[error("scoped evaluation allocation failed")]
    Allocation,
    /// Scope, event or original quota identity does not match.
    #[error("scoped evaluation domain mismatch")]
    Domain,
    /// The source operation was already consumed.
    #[error("scoped evaluation is spent")]
    Spent,
    /// Work remains pending; this does not establish safe release.
    #[error("scoped evaluation remains pending")]
    Pending,
    /// The actual native cause is retained by the same prepaid carrier.
    #[error("scoped evaluation retained native failure")]
    Failed,
    /// Progress requires a separately funded operation.
    #[error("scoped evaluation requires funded progress")]
    NeedsFundedProgress,
    /// The exact native frontier cannot be observed by this operation.
    #[error("scoped evaluation is unobservable")]
    Unobservable,
    /// Runtime or failure publication is currently busy.
    #[error("scoped evaluation runtime busy")]
    RuntimeBusy,
    /// Preserve an unexpected ABI value without allocating a diagnostic.
    #[error("invalid scoped evaluation status {0}")]
    InvalidStatus(u32),
}
impl ScopedEvaluationCause {
    pub(crate) fn from_status(value: u32) -> Self {
        match value {
            1 => Self::Invalid,
            2 => Self::Capacity,
            3 => Self::Allocation,
            4 => Self::Domain,
            5 => Self::Spent,
            6 => Self::Pending,
            7 => Self::Failed,
            8 => Self::NeedsFundedProgress,
            9 => Self::Unobservable,
            10 => Self::RuntimeBusy,
            other => Self::InvalidStatus(other),
        }
    }
    fn message(self) -> &'static str {
        match self {
            Self::Invalid => "invalid scoped evaluation",
            Self::Capacity => "scoped evaluation capacity exhausted",
            Self::Allocation => "scoped evaluation allocation failed",
            Self::Domain => "scoped evaluation domain mismatch",
            Self::Spent => "scoped evaluation is spent",
            Self::Pending => "scoped evaluation remains pending",
            Self::Failed => "scoped evaluation retained native failure",
            Self::NeedsFundedProgress => "scoped evaluation requires funded progress",
            Self::Unobservable => "scoped evaluation is unobservable",
            Self::RuntimeBusy => "scoped evaluation runtime busy",
            Self::InvalidStatus(_) => "invalid scoped evaluation status",
        }
    }
}
#[derive(Debug)]
pub(crate) struct ScopedEvaluationFailure {
    cause: ScopedEvaluationCause,
    // Source retires before its independent custody alias. An unpublished
    // carrier is custody only and never masquerades as a native error source.
    source: Option<crate::PrefillNativeError>,
    custody: Option<crate::RetainedPrefillFailure>,
}
impl PartialEq for ScopedEvaluationFailure {
    fn eq(&self, other: &Self) -> bool {
        self.cause == other.cause
            && match (&self.custody, &other.custody) {
                (Some(a), Some(b)) => a.raw().ctx == b.raw().ctx,
                (None, None) => true,
                _ => false,
            }
    }
}
impl std::fmt::Display for ScopedEvaluationFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.cause, f)?;
        if let Some(source) = &self.source {
            write!(f, ": {source}")?;
        }
        Ok(())
    }
}
impl std::error::Error for ScopedEvaluationFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source.as_ref().map(|source| source as _)
    }
}

/// Exception. Most will come from the C API.
#[derive(Debug)]
pub struct Exception {
    pub(crate) what: String,
    pub(crate) location: &'static Location<'static>,
    pub(crate) source: Option<ExceptionSource>,
    // Inline fixed native classification adds no new error-source allocation.
    pub(crate) tracking: Option<SubmissionTrackingFailure>,
    pub(crate) graph: Option<GraphMetadataFailure>,
    pub(crate) scoped: Option<ScopedEvaluationFailure>,
}

impl std::fmt::Display for Exception {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(scoped) = &self.scoped {
            return write!(formatter, "{scoped} at {}", self.location);
        }
        if let Some(source) = self.source.as_ref().filter(|source| source.retained()) {
            return write!(formatter, "{} at {}", source.source(), self.location);
        }
        write!(formatter, "{:?} at {}", self.what, self.location)
    }
}

impl std::error::Error for Exception {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source
            .as_ref()
            .map(|source| source.source() as _)
            .or_else(|| self.tracking.as_ref().map(|source| source as _))
            .or_else(|| self.graph.as_ref().map(|source| source as _))
            .or_else(|| self.scoped.as_ref().map(|source| source as _))
    }
}

impl PartialEq for Exception {
    fn eq(&self, other: &Self) -> bool {
        self.what == other.what
            && self.location == other.location
            && self.tracking == other.tracking
            && self.graph == other.graph
            && self.scoped == other.scoped
            && match (&self.source, &other.source) {
                (None, None) => true,
                (Some(a), Some(b)) => a == b,
                _ => false,
            }
    }
}

impl Exception {
    /// The error message.
    pub fn what(&self) -> &str {
        if self.source.as_ref().is_some_and(ExceptionSource::retained) {
            return "retained Rust error source";
        }
        self.scoped
            .as_ref()
            .map_or(self.what.as_str(), |value| value.cause.message())
    }

    /// Fixed original evaluation result, when this exception owns scoped custody.
    pub fn scoped_evaluation_cause(&self) -> Option<ScopedEvaluationCause> {
        self.scoped.as_ref().map(|value| value.cause)
    }

    #[track_caller]
    pub(crate) fn from_scoped_evaluation(
        cause: ScopedEvaluationCause,
        custody: Option<crate::RetainedPrefillFailure>,
    ) -> Self {
        let source = custody
            .as_ref()
            .and_then(crate::RetainedPrefillFailure::error);
        Self {
            what: String::new(),
            location: Location::caller(),
            source: None,
            tracking: None,
            graph: None,
            scoped: Some(ScopedEvaluationFailure {
                cause,
                source,
                custody,
            }),
        }
    }

    /// The location of the error.
    ///
    /// The location is obtained from `std::panic::Location::caller()` and points
    /// to the location in the code where the error was created and not where it was
    /// propagated.
    pub fn location(&self) -> &'static Location<'static> {
        self.location
    }

    /// Creates a new exception with the given message.
    #[track_caller]
    pub fn custom(what: impl Into<String>) -> Self {
        Self {
            what: what.into(),
            location: Location::caller(),
            source: None,
            tracking: None,
            graph: None,
            scoped: None,
        }
    }

    /// Preserves a Rust failure while crossing an API that returns MLX exceptions.
    /// The original error remains available through [`std::error::Error::source`].
    #[track_caller]
    pub fn from_source(error: impl std::error::Error + Send + Sync + 'static) -> Self {
        Self {
            what: error.to_string(),
            location: Location::caller(),
            source: Some(ExceptionSource::Ordinary(std::sync::Arc::new(error))),
            tracking: None,
            graph: None,
            scoped: None,
        }
    }
}

impl From<RawException> for Exception {
    #[track_caller]
    fn from(e: RawException) -> Self {
        if let Some(source) = PHYSICAL_BACKING_ERROR.with(|error| error.borrow_mut().take()) {
            return Self {
                what: e.what,
                location: Location::caller(),
                source: Some(ExceptionSource::Ordinary(source)),
                tracking: e.tracking,
                graph: e.graph,
                scoped: None,
            };
        }
        Self {
            what: e.what,
            location: Location::caller(),
            source: None,
            tracking: e.tracking,
            graph: e.graph,
            scoped: None,
        }
    }
}

impl From<&str> for Exception {
    #[track_caller]
    fn from(what: &str) -> Self {
        Self {
            what: what.to_string(),
            location: Location::caller(),
            source: None,
            tracking: None,
            graph: None,
            scoped: None,
        }
    }
}

impl From<Infallible> for Exception {
    fn from(_: Infallible) -> Self {
        unreachable!()
    }
}

impl From<Exception> for String {
    fn from(e: Exception) -> Self {
        if e.scoped.is_some() || e.source.as_ref().is_some_and(ExceptionSource::retained) {
            // Explicit caller-requested materialization is ordinary. Keep the
            // retained cause and custody alive until formatting completes.
            e.to_string()
        } else {
            e.what
        }
    }
}

thread_local! {
    static PHYSICAL_BACKING_ERROR: RefCell<Option<std::sync::Arc<Exception>>> = const { RefCell::new(None) };
    static CLOSURE_ERROR: Cell<Option<Exception>> = const { Cell::new(None) };
    static LAST_MLX_ERROR: RefCell<Option<RawException>> = const { RefCell::new(None) };
}

static INIT_ERR_HANDLER: Once = Once::new();

pub(crate) fn static_storage_bytes() -> usize {
    std::mem::size_of_val(&INIT_ERR_HANDLER)
}

#[no_mangle]
extern "C" fn default_mlx_error_handler(msg: *const c_char, _data: *mut std::ffi::c_void) {
    let message = unsafe { CStr::from_ptr(msg) }
        .to_string_lossy()
        .into_owned();
    // SAFETY: the native synchronous error handler classified its caught
    // exception before invoking us. No query/progress or textual parsing occurs.
    let tracking = SubmissionTrackingFailure::from_native(unsafe {
        safemlx_sys::mlx_error_submission_tracking_failure()
    });
    // Same synchronous caught-exception classification, independent of Record.
    let graph = GraphMetadataFailure::from_native(unsafe {
        safemlx_sys::mlx_error_graph_metadata_failure()
    });
    // SAFETY: the active native catch retains this exact failed-observer node.
    let backing = unsafe { safemlx_sys::mlx_error_physical_backing_failure() };
    PHYSICAL_BACKING_ERROR.with(|error| {
        error.replace(if backing.is_null() {
            None
        } else {
            Some(unsafe { crate::PhysicalBackingCustody::borrowed_error(backing) })
        })
    });
    LAST_MLX_ERROR.with(|last_error| {
        last_error.replace(Some(RawException {
            what: message,
            tracking,
            graph,
        }));
    });
}

fn take_last_mlx_error() -> Option<RawException> {
    LAST_MLX_ERROR.with(|last_error| last_error.borrow_mut().take())
}

fn setup_mlx_error_handler() {
    let handler = default_mlx_error_handler;
    unsafe {
        safemlx_sys::mlx_set_error_handler(Some(handler), std::ptr::null_mut(), None);
    }

    #[cfg(all(feature = "metal", target_vendor = "apple"))]
    {
        let status = unsafe {
            safemlx_sys::mlx_metal_set_embedded_metallib(
                safemlx_sys::MLX_METALLIB_LZFSE.as_ptr(),
                safemlx_sys::MLX_METALLIB_LZFSE.len(),
                safemlx_sys::MLX_METALLIB_UNCOMPRESSED_SIZE,
            )
        };
        assert_eq!(status, 0, "failed to register the embedded MLX metallib");
    }
}

pub(crate) fn ensure_mlx_error_handler() {
    #[cfg(test)]
    HANDLER_INITIALIZATION_ENTRIES.with(|entries| entries.set(entries.get() + 1));
    INIT_ERR_HANDLER.call_once(setup_mlx_error_handler);
}

#[cfg(test)]
thread_local! {
    static HANDLER_INITIALIZATION_ENTRIES: Cell<usize> = const { Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn mlx_error_handler_state_for_test() -> (bool, usize) {
    (
        INIT_ERR_HANDLER.is_completed(),
        HANDLER_INITIALIZATION_ENTRIES.with(Cell::get),
    )
}

pub(crate) fn set_closure_error(err: Exception) {
    CLOSURE_ERROR.with(|closure_error| closure_error.set(Some(err)));
}

pub(crate) fn get_and_clear_closure_error() -> Option<Exception> {
    CLOSURE_ERROR.with(|closure_error| closure_error.replace(None))
}

#[track_caller]
pub(crate) fn get_and_clear_last_mlx_error() -> Option<RawException> {
    take_last_mlx_error()
}

/// The dtype is not a float-point type
#[derive(Debug, Error)]
#[error("[finfo] dtype {:?} is not inexact", .0)]
pub struct InexactDtypeError(pub Dtype);

impl From<InexactDtypeError> for Exception {
    #[track_caller]
    fn from(value: InexactDtypeError) -> Self {
        Exception::custom(value.to_string())
    }
}

#[cfg(test)]
mod tests {
    use crate::{array, Array};

    #[test]
    fn rust_exception_source_survives_nested_native_error_domains() {
        use std::error::Error;
        let cause = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "source sentinel");
        let inner = super::Exception::from_source(cause);
        assert_eq!(inner.what(), "source sentinel");
        assert_eq!(inner.location().file(), file!());
        let outer = super::Exception::from_source(inner);
        let retained = outer
            .source()
            .unwrap()
            .downcast_ref::<super::Exception>()
            .unwrap();
        let cause = retained
            .source()
            .unwrap()
            .downcast_ref::<std::io::Error>()
            .unwrap();
        assert_eq!(cause.kind(), std::io::ErrorKind::PermissionDenied);
        assert_eq!(cause.to_string(), "source sentinel");
        assert_eq!(outer, outer);
        assert!(super::Exception::custom("source sentinel")
            .source()
            .is_none());
    }

    #[test]
    fn test_exception() {
        let stream = crate::test_stream();
        let a = array!([1.0, 2.0, 3.0]);
        let b = array!([4.0, 5.0]);

        let result = a.add(&b, stream);
        let error = result.expect_err("Expected error");

        // The full error message would also contain the full path to the original c++ file,
        // so we just check for a substring
        assert!(error
            .what()
            .contains("Shapes (3) and (2) cannot be broadcast."))
    }

    #[test]
    fn mlx_errors_are_thread_local() {
        let threads = (0..8)
            .map(|thread_index| {
                std::thread::spawn(move || {
                    let stream = crate::test_stream();
                    for iteration in 0..64 {
                        let lhs_len = 3 + thread_index;
                        let rhs_len = lhs_len + 1 + iteration % 3;
                        let lhs = Array::from_slice(&vec![0.0f32; lhs_len], &[lhs_len as i32]);
                        let rhs = Array::from_slice(&vec![0.0f32; rhs_len], &[rhs_len as i32]);
                        let error = lhs.add(&rhs, stream).expect_err("add should fail");
                        let expected = format!("Shapes ({lhs_len}) and ({rhs_len})");
                        assert!(
                            error.what().contains(&expected),
                            "expected thread-local error containing {expected:?}, got {:?}",
                            error.what()
                        );
                    }
                })
            })
            .collect::<Vec<_>>();

        for thread in threads {
            thread.join().expect("worker thread panicked");
        }
    }
}
