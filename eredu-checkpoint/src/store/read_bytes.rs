//! Shared byte transitions and a destination tied to an actual admitted source.
use super::{
    AdmittedFile, AdmittedFileIdentity, SafetensorsLease, SafetensorsReadTelemetry, StoreError,
};
use crate::artifact::file::FileVersion;
use std::{
    alloc::Layout,
    fs::File,
    io::{self, Read, Seek, SeekFrom},
    ops::Range,
    path::Path,
    sync::atomic::Ordering,
};

/// Private loan: the actual source owns every reference, including telemetry.
#[derive(Clone, Copy, Debug)]
pub(super) struct ReadFileSource<'a> {
    pub path: &'a Path,
    pub admitted: &'a AdmittedFile,
    pub header_payload_start: usize,
    pub telemetry: &'a SafetensorsReadTelemetry,
}

/// A finite byte destination made only by an admitted source's selection plan.
/// This is a payload worker, not a cache acquisition or an admission authority.
///
/// ```compile_fail
/// use eredu_checkpoint::store::SafetensorsBytePlan;
/// fn duplicate(plan: SafetensorsBytePlan<'_, '_>) { let _ = plan.clone(); }
/// ```
#[derive(Debug)]
pub struct SafetensorsBytePlan<'s, 'r> {
    source: ReadFileSource<'s>,
    key: &'s str,
    tensor_payload_start: usize,
    tensor_len: usize,
    ranges: &'r [Range<usize>],
    layout: Layout,
    physically_bounded: bool,
}
impl<'s, 'r> SafetensorsBytePlan<'s, 'r> {
    pub(super) fn new(
        source: ReadFileSource<'s>,
        key: &'s str,
        tensor_payload_start: usize,
        tensor_len: usize,
        ranges: &'r [Range<usize>],
        layout: Layout,
        physically_bounded: bool,
    ) -> Self {
        Self {
            source,
            key,
            tensor_payload_start,
            tensor_len,
            ranges,
            layout,
            physically_bounded,
        }
    }
    /// Exact initialized caller-buffer layout. This excludes OS/cache/error owners.
    pub fn destination_layout(&self) -> Layout {
        self.layout
    }
    /// Read the retained admitted source into exactly this destination.
    /// No output Vec, path clone, header initialization or cache mutation occurs.
    /// OS operations and an actual `io::Error` may own unpriced resources.
    pub fn read_into<'d>(
        self,
        destination: &'d mut [u8],
    ) -> Result<SafetensorsBytes<'s, 'd>, SafetensorsByteFailure<'s, 'r, 'd>> {
        self.read_into_with(destination, || {})
    }
    fn read_into_with<'d>(
        self,
        destination: &'d mut [u8],
        after_read: impl FnOnce(),
    ) -> Result<SafetensorsBytes<'s, 'd>, SafetensorsByteFailure<'s, 'r, 'd>> {
        let errors = FixedErrors {
            path: self.source.path,
            key: self.key,
        };
        let mut output = FixedOutput {
            bytes: destination,
            used: 0,
        };
        let mut progress = Progress::default();
        let mut file = None;
        let result = (|| {
            self.check_destination(output.bytes.len())?;
            check_path(self.source.admitted, self.source.path, &errors)?;
            file = Some(File::open(self.source.path).map_err(|cause| errors.fs(cause))?);
            let opened = file.as_mut().expect("opened above");
            validate_file(self.source.admitted, self.source.path, opened, &errors)?;
            read_pass(
                opened,
                self.tensor_payload_start,
                self.ranges,
                self.source.telemetry,
                &mut output,
                &mut progress,
                &errors,
            )?;
            after_read();
            validate_file(self.source.admitted, self.source.path, opened, &errors)
        })();
        match result {
            Ok(()) => {
                drop(file); // Same final handle closes before the successful bytes escape.
                Ok(SafetensorsBytes {
                    source: self.source,
                    bytes: output.bytes,
                    physically_bounded: self.physically_bounded,
                    physical_reads: progress.completed_ranges,
                })
            }
            Err(cause) => Err(SafetensorsByteFailure {
                cause,
                destination: output.bytes,
                progress,
                file,
                plan: self,
            }),
        }
    }
    /// Copy from a genuine retained full payload of this same tensor and admitted
    /// file. A selected/foreign lease cannot provide backing for this operation.
    /// The existing ordinary cache lookup, eviction and insertion remain separate.
    pub fn copy_from<'d>(
        self,
        lease: &SafetensorsLease,
        destination: &'d mut [u8],
    ) -> Result<SafetensorsBytes<'s, 'd>, SafetensorsByteFailure<'s, 'r, 'd>> {
        let errors = FixedErrors {
            path: self.source.path,
            key: self.key,
        };
        let mut output = FixedOutput {
            bytes: destination,
            used: 0,
        };
        let mut progress = Progress::default();
        let mut file = None;
        let result = (|| {
            self.check_destination(output.bytes.len())?;
            if !std::ptr::eq(self.source.admitted, lease.shard.admitted_file.as_ref())
                || lease.metadata.name != self.key
                || !matches!(lease.selection, super::TensorSelection::Full)
                || lease.bytes.len() != self.tensor_len
                || lease.proof.offset_bytes != 0
                || u64::try_from(self.tensor_len).ok() != Some(lease.proof.length_bytes)
            {
                return Err(SafetensorsByteError::SourceMismatch);
            }
            // Ordinary cached acquisition validates and drops the opened file
            // before copying. Keep that successful lifetime and no second check.
            check_path(self.source.admitted, self.source.path, &errors)?;
            file = Some(File::open(self.source.path).map_err(|cause| errors.fs(cause))?);
            validate_file(
                self.source.admitted,
                self.source.path,
                file.as_ref().unwrap(),
                &errors,
            )?;
            drop(file.take());
            copy_pass(
                &lease.bytes,
                self.ranges,
                &mut output,
                &mut progress,
                &errors,
            )
        })();
        match result {
            Ok(()) => Ok(SafetensorsBytes {
                source: self.source,
                bytes: output.bytes,
                physically_bounded: self.physically_bounded,
                physical_reads: 0,
            }),
            Err(cause) => Err(SafetensorsByteFailure {
                cause,
                destination: output.bytes,
                progress,
                file,
                plan: self,
            }),
        }
    }
    // A cache witness comes only from the real weak-map lookup and its ordinary
    // file validation. Public copy_from still requires an actual Full lease.
    pub(super) fn copy_validated_cache<'d>(
        self,
        cached: &super::cache_policy::ValidatedCachedPayload<'_>,
        destination: &'d mut [u8],
    ) -> Result<SafetensorsBytes<'s, 'd>, SafetensorsByteFailure<'s, 'r, 'd>> {
        let errors = FixedErrors {
            path: self.source.path,
            key: self.key,
        };
        let mut output = FixedOutput {
            bytes: destination,
            used: 0,
        };
        let mut progress = Progress::default();
        let result = (|| {
            self.check_destination(output.bytes.len())?;
            if !cached.matches(self.source.admitted, self.key, self.tensor_len) {
                return Err(SafetensorsByteError::SourceMismatch);
            }
            copy_pass(
                cached.bytes(),
                self.ranges,
                &mut output,
                &mut progress,
                &errors,
            )
        })();
        match result {
            Ok(()) => Ok(SafetensorsBytes {
                source: self.source,
                bytes: output.bytes,
                physically_bounded: self.physically_bounded,
                physical_reads: 0,
            }),
            Err(cause) => Err(SafetensorsByteFailure {
                cause,
                destination: output.bytes,
                progress,
                file: None,
                plan: self,
            }),
        }
    }
    fn check_destination(&self, actual: usize) -> Result<(), SafetensorsByteError<'s>> {
        if actual != self.layout.size() {
            Err(SafetensorsByteError::Destination {
                expected: self.layout.size(),
                actual,
            })
        } else {
            Ok(())
        }
    }
}

/// A successful byte output retaining the genuine source loan and caller storage.
#[derive(Debug)]
pub struct SafetensorsBytes<'s, 'd> {
    source: ReadFileSource<'s>,
    bytes: &'d mut [u8],
    physically_bounded: bool,
    physical_reads: usize,
}
impl SafetensorsBytes<'_, '_> {
    /// Completed bytes, valid only after all required file-version checks passed.
    pub fn as_slice(&self) -> &[u8] {
        self.bytes
    }
    /// Physical selection behavior inherited from the actual range worker.
    pub fn physically_bounded(&self) -> bool {
        self.physically_bounded
    }
    /// Completed payload reads; cache copying reports zero.
    pub fn physical_reads(&self) -> usize {
        self.physical_reads
    }
    /// Actual admitted path; this is a borrow, not a newly owned path.
    pub fn source_path(&self) -> &Path {
        self.source.path
    }
}

/// Fixed rejection or the original owned OS cause. No error is formatted into a
/// String on the prepared path. The OS cause's dynamic storage remains explicit.
#[derive(Debug)]
pub enum SafetensorsByteError<'a> {
    /// Exact caller destination extent differs; no read or write occurred.
    Destination {
        /// Required selected payload length in bytes.
        expected: usize,
        /// Supplied destination length in bytes.
        actual: usize,
    },
    /// Cached full payload does not belong to this admitted tensor source.
    SourceMismatch,
    /// The admitted file version/path changed.
    Changed {
        /// Actual admitted checkpoint path.
        path: &'a Path,
    },
    /// Filesystem validation/open failure, including the real original cause.
    Filesystem {
        /// Actual admitted checkpoint path.
        path: &'a Path,
        /// Original filesystem failure.
        cause: io::Error,
    },
    /// Seek or payload read failure, retaining the real original cause.
    Read {
        /// Actual admitted checkpoint path.
        path: &'a Path,
        /// Original seek or payload read failure.
        cause: io::Error,
    },
    /// Absolute source offset cannot be represented.
    Offset {
        /// Actual admitted checkpoint path.
        path: &'a Path,
    },
    /// Cached payload contradicts the selected range.
    CachedRange {
        /// Actual tensor key associated with the payload.
        key: &'a str,
    },
}
// One formatter for borrowed failures and owned final-destination failures.
// I/O causes are borrowed here; neither adapter clones or formats them eagerly.
enum ByteErrorView<'a> {
    Destination {
        expected: usize,
        actual: usize,
    },
    SourceMismatch,
    Changed {
        path: &'a Path,
    },
    Filesystem {
        path: &'a Path,
        cause: &'a io::Error,
    },
    Read {
        path: &'a Path,
        cause: &'a io::Error,
    },
    Offset {
        path: &'a Path,
    },
    CachedRange {
        key: &'a str,
    },
}
impl std::fmt::Display for ByteErrorView<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Destination { expected, actual } => write!(
                f,
                "byte destination has {actual} bytes; expected {expected}"
            ),
            Self::SourceMismatch => {
                f.write_str("cached payload does not match the admitted tensor")
            }
            Self::Changed { path } => {
                write!(
                    f,
                    "admitted checkpoint file changed after preparation: {}",
                    path.display()
                )
            }
            Self::Filesystem { path, cause } if cause.kind() == io::ErrorKind::NotFound => {
                write!(f, "checkpoint shard does not exist: {}", path.display())
            }
            Self::Filesystem { path, cause } | Self::Read { path, cause } => {
                write!(f, "checkpoint I/O failed for {}: {cause}", path.display())
            }
            Self::Offset { path } => write!(
                f,
                "checkpoint size overflow: selected payload offset for {}",
                path.display()
            ),
            Self::CachedRange { key } => write!(
                f,
                "invalid selection for tensor {key:?}: cached selection exceeds payload"
            ),
        }
    }
}
impl std::fmt::Display for SafetensorsByteError<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let view = match self {
            Self::Destination { expected, actual } => ByteErrorView::Destination {
                expected: *expected,
                actual: *actual,
            },
            Self::SourceMismatch => ByteErrorView::SourceMismatch,
            Self::Changed { path } => ByteErrorView::Changed { path },
            Self::Filesystem { path, cause } => ByteErrorView::Filesystem { path, cause },
            Self::Read { path, cause } => ByteErrorView::Read { path, cause },
            Self::Offset { path } => ByteErrorView::Offset { path },
            Self::CachedRange { key } => ByteErrorView::CachedRange { key },
        };
        view.fmt(f)
    }
}
impl std::error::Error for SafetensorsByteError<'_> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Filesystem { cause, .. } | Self::Read { cause, .. } => Some(cause),
            _ => None,
        }
    }
}

#[derive(Debug, Default)]
struct Progress {
    completed_ranges: usize,
    completed_bytes: usize,
    initialized_bytes: usize,
}
/// A failed read retaining the destination's exclusive loan, plan/source and any
/// opened handle. Drop the failure to release those loans and the actual cause.
/// Completed ranges are progress only: a Changed error invalidates the result.
///
/// ```compile_fail
/// use eredu_checkpoint::store::SafetensorsBytePlan;
/// fn overwrite(plan: SafetensorsBytePlan<'_, '_>, bytes: &mut [u8]) {
///     let failure = plan.read_into(bytes).unwrap_err();
///     bytes[0] = 0;
///     println!("{}", failure.completed_bytes());
/// }
/// ```
#[derive(Debug)]
pub struct SafetensorsByteFailure<'s, 'r, 'd> {
    cause: SafetensorsByteError<'s>,
    destination: &'d mut [u8],
    progress: Progress,
    file: Option<File>,
    plan: SafetensorsBytePlan<'s, 'r>,
}
impl SafetensorsByteFailure<'_, '_, '_> {
    pub(super) fn into_owned(self) -> OwnedByteFailure {
        let cause = match self.cause {
            SafetensorsByteError::Destination { expected, actual } => {
                OwnedByteCause::Destination { expected, actual }
            }
            SafetensorsByteError::SourceMismatch => OwnedByteCause::SourceMismatch,
            SafetensorsByteError::Changed { .. } => OwnedByteCause::Changed,
            SafetensorsByteError::Filesystem { cause, .. } => OwnedByteCause::Filesystem(cause),
            SafetensorsByteError::Read { cause, .. } => OwnedByteCause::Read(cause),
            SafetensorsByteError::Offset { .. } => OwnedByteCause::Offset,
            SafetensorsByteError::CachedRange { .. } => OwnedByteCause::CachedRange,
        };
        OwnedByteFailure {
            cause,
            file: self.file,
            progress: self.progress,
        }
    }

    /// Typed rejection or the original retained I/O failure.
    pub fn cause(&self) -> &SafetensorsByteError<'_> {
        &self.cause
    }
    /// Whole ranges completed before failure; not proof of unchanged source.
    pub fn completed_ranges(&self) -> usize {
        self.progress.completed_ranges
    }
    /// Bytes in those completed ranges. A failing read_exact may write more.
    pub fn completed_bytes(&self) -> usize {
        self.progress.completed_bytes
    }
    /// Prefix initialized before reads/copies. Within a failing read range its
    /// bytes are deliberately unspecified, even though all remain initialized.
    pub fn initialized_bytes(&self) -> usize {
        self.progress.initialized_bytes
    }
    /// Inspect failed storage; this is not a valid completed tensor payload.
    pub fn destination(&self) -> &[u8] {
        self.destination
    }
    /// Whether this failure still owns the opened handle.
    pub fn retains_file(&self) -> bool {
        self.file.is_some()
    }
    /// Actual admitted source path retained by the failed plan.
    pub fn source_path(&self) -> &Path {
        self.plan.source.path
    }
}
impl std::fmt::Display for SafetensorsByteFailure<'_, '_, '_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.cause.fmt(f)
    }
}
impl std::error::Error for SafetensorsByteFailure<'_, '_, '_> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        std::error::Error::source(&self.cause)
    }
}

// The owned lease destination supplies the already-owned path/key context.
// No borrowed plan or destination reference escapes this conversion.
#[derive(Debug)]
pub(super) struct OwnedByteFailure {
    cause: OwnedByteCause,
    file: Option<File>,
    progress: Progress,
}
#[derive(Debug)]
enum OwnedByteCause {
    Destination { expected: usize, actual: usize },
    SourceMismatch,
    Changed,
    Filesystem(io::Error),
    Read(io::Error),
    Offset,
    CachedRange,
}
impl OwnedByteFailure {
    pub(super) fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.cause {
            OwnedByteCause::Filesystem(cause) | OwnedByteCause::Read(cause) => Some(cause),
            _ => None,
        }
    }
    pub(super) fn completed_bytes(&self) -> usize {
        self.progress.completed_bytes
    }
    pub(super) fn initialized_bytes(&self) -> usize {
        self.progress.initialized_bytes
    }
    pub(super) fn retains_file(&self) -> bool {
        self.file.is_some()
    }
    pub(super) fn fmt_at(
        &self,
        path: &Path,
        key: &str,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        let view = match &self.cause {
            OwnedByteCause::Destination { expected, actual } => ByteErrorView::Destination {
                expected: *expected,
                actual: *actual,
            },
            OwnedByteCause::SourceMismatch => ByteErrorView::SourceMismatch,
            OwnedByteCause::Changed => ByteErrorView::Changed { path },
            OwnedByteCause::Filesystem(cause) => ByteErrorView::Filesystem { path, cause },
            OwnedByteCause::Read(cause) => ByteErrorView::Read { path, cause },
            OwnedByteCause::Offset => ByteErrorView::Offset { path },
            OwnedByteCause::CachedRange => ByteErrorView::CachedRange { key },
        };
        std::fmt::Display::fmt(&view, f)
    }
}

trait ByteOutput {
    fn zeroed(&mut self, count: usize) -> &mut [u8];
    fn append(&mut self, bytes: &[u8]);
}
impl ByteOutput for Vec<u8> {
    fn zeroed(&mut self, count: usize) -> &mut [u8] {
        let start = self.len();
        self.resize(start + count, 0);
        &mut self[start..]
    }
    fn append(&mut self, bytes: &[u8]) {
        self.extend_from_slice(bytes);
    }
}
struct FixedOutput<'a> {
    bytes: &'a mut [u8],
    used: usize,
}
impl ByteOutput for FixedOutput<'_> {
    fn zeroed(&mut self, count: usize) -> &mut [u8] {
        let start = self.used;
        self.used += count;
        self.bytes[start..self.used].fill(0);
        &mut self.bytes[start..self.used]
    }
    fn append(&mut self, bytes: &[u8]) {
        let end = self.used + bytes.len();
        self.bytes[self.used..end].copy_from_slice(bytes);
        self.used = end;
    }
}
trait Errors {
    type Error;
    fn offset(&self) -> Self::Error;
    fn io(&self, cause: io::Error) -> Self::Error;
    fn cached_range(&self) -> Self::Error;
}
struct OrdinaryErrors<'a> {
    path: &'a Path,
    key: &'a str,
}
impl Errors for OrdinaryErrors<'_> {
    type Error = StoreError;
    fn offset(&self) -> StoreError {
        StoreError::Overflow {
            context: format!("selected payload offset for {}", self.path.display()),
        }
    }
    fn io(&self, cause: io::Error) -> StoreError {
        super::io_error(self.path, cause)
    }
    fn cached_range(&self) -> StoreError {
        super::invalid_selection(self.key, "cached selection exceeds payload")
    }
}
struct FixedErrors<'a> {
    path: &'a Path,
    key: &'a str,
}
impl<'a> Errors for FixedErrors<'a> {
    type Error = SafetensorsByteError<'a>;
    fn offset(&self) -> Self::Error {
        SafetensorsByteError::Offset { path: self.path }
    }
    fn io(&self, cause: io::Error) -> Self::Error {
        SafetensorsByteError::Read {
            path: self.path,
            cause,
        }
    }
    fn cached_range(&self) -> Self::Error {
        SafetensorsByteError::CachedRange { key: self.key }
    }
}
fn read_pass<R: Read + Seek, B: ByteOutput, E: Errors>(
    file: &mut R,
    tensor_payload_start: usize,
    ranges: &[Range<usize>],
    telemetry: &SafetensorsReadTelemetry,
    output: &mut B,
    progress: &mut Progress,
    errors: &E,
) -> Result<(), E::Error> {
    for range in ranges {
        let absolute = tensor_payload_start
            .checked_add(range.start)
            .ok_or_else(|| errors.offset())?;
        file.seek(SeekFrom::Start(
            u64::try_from(absolute).map_err(|_| errors.offset())?,
        ))
        .map_err(|cause| errors.io(cause))?;
        let bytes = output.zeroed(range.len());
        progress.initialized_bytes = progress.completed_bytes + range.len();
        file.read_exact(bytes).map_err(|cause| errors.io(cause))?;
        telemetry.physical_reads.fetch_add(1, Ordering::Relaxed);
        telemetry
            .physical_read_bytes
            .fetch_add(range.len() as u64, Ordering::Relaxed);
        progress.completed_ranges += 1;
        progress.completed_bytes += range.len();
    }
    Ok(())
}
fn copy_pass<B: ByteOutput, E: Errors>(
    payload: &[u8],
    ranges: &[Range<usize>],
    output: &mut B,
    progress: &mut Progress,
    errors: &E,
) -> Result<(), E::Error> {
    for range in ranges {
        output.append(
            payload
                .get(range.clone())
                .ok_or_else(|| errors.cached_range())?,
        );
        progress.completed_ranges += 1;
        progress.completed_bytes += range.len();
        progress.initialized_bytes = progress.completed_bytes;
    }
    Ok(())
}
pub(super) fn ordinary_read_pass(
    path: &Path,
    file: &mut File,
    tensor_payload_start: usize,
    ranges: &[Range<usize>],
    capacity: usize,
    telemetry: &SafetensorsReadTelemetry,
) -> Result<Vec<u8>, StoreError> {
    let mut output = Vec::with_capacity(capacity);
    read_pass(
        file,
        tensor_payload_start,
        ranges,
        telemetry,
        &mut output,
        &mut Progress::default(),
        &OrdinaryErrors { path, key: "" },
    )?;
    Ok(output)
}
pub(super) fn ordinary_copy(
    key: &str,
    payload: &[u8],
    ranges: &[Range<usize>],
) -> Result<Vec<u8>, StoreError> {
    let capacity = ranges.iter().try_fold(0usize, |total, range| {
        total
            .checked_add(range.len())
            .ok_or_else(|| StoreError::Overflow {
                context: format!("cached selected payload length for {key:?}"),
            })
    })?;
    let mut output = Vec::with_capacity(capacity);
    copy_pass(
        payload,
        ranges,
        &mut output,
        &mut Progress::default(),
        &OrdinaryErrors {
            path: Path::new(""),
            key,
        },
    )?;
    Ok(output)
}

trait FileErrors {
    type Error;
    type Identity;
    fn fs(&self, cause: io::Error) -> Self::Error;
    fn changed(&self, path: &Path) -> Self::Error;
    fn identity(
        &self,
        path: &Path,
        metadata: &std::fs::Metadata,
    ) -> Result<Self::Identity, Self::Error>;
    fn matches(&self, admitted: &AdmittedFile, path: &Path, current: &Self::Identity) -> bool;
}
fn check_path<E: FileErrors>(
    admitted: &AdmittedFile,
    path: &Path,
    errors: &E,
) -> Result<(), E::Error> {
    if path != admitted.identity.canonical_path {
        return Err(errors.changed(path));
    }
    Ok(())
}
fn validate_file<E: FileErrors>(
    admitted: &AdmittedFile,
    path: &Path,
    file: &File,
    errors: &E,
) -> Result<(), E::Error> {
    let current = errors.identity(path, &file.metadata().map_err(|cause| errors.fs(cause))?)?;
    if !errors.matches(admitted, path, &current) {
        return Err(errors.changed(path));
    }
    Ok(())
}
struct OrdinaryFileErrors<'a>(&'a Path);
impl FileErrors for OrdinaryFileErrors<'_> {
    type Error = StoreError;
    type Identity = AdmittedFileIdentity;
    fn fs(&self, cause: io::Error) -> StoreError {
        super::fs_error(self.0, cause)
    }
    fn changed(&self, path: &Path) -> StoreError {
        StoreError::AdmittedFileChanged {
            path: path.to_path_buf(),
        }
    }
    fn identity(
        &self,
        path: &Path,
        metadata: &std::fs::Metadata,
    ) -> Result<Self::Identity, StoreError> {
        AdmittedFileIdentity::from_metadata(path, metadata)
    }
    fn matches(&self, admitted: &AdmittedFile, _: &Path, current: &Self::Identity) -> bool {
        *current == admitted.identity
    }
}
impl<'a> FileErrors for FixedErrors<'a> {
    type Error = SafetensorsByteError<'a>;
    type Identity = FileVersion;
    fn fs(&self, cause: io::Error) -> Self::Error {
        SafetensorsByteError::Filesystem {
            path: self.path,
            cause,
        }
    }
    fn changed(&self, _: &Path) -> Self::Error {
        SafetensorsByteError::Changed { path: self.path }
    }
    fn identity(
        &self,
        _: &Path,
        metadata: &std::fs::Metadata,
    ) -> Result<Self::Identity, Self::Error> {
        FileVersion::from_metadata(metadata).map_err(|cause| self.fs(cause))
    }
    fn matches(&self, admitted: &AdmittedFile, path: &Path, current: &Self::Identity) -> bool {
        path == admitted.identity.canonical_path && *current == admitted.identity.version
    }
}
pub(super) fn fixed_validate_file<'a>(
    admitted: &AdmittedFile,
    path: &'a Path,
    file: &File,
) -> Result<(), SafetensorsByteError<'a>> {
    validate_file(admitted, path, file, &FixedErrors { path, key: "" })
}

pub(super) fn ordinary_validate_file(
    admitted: &AdmittedFile,
    path: &Path,
    file: &File,
) -> Result<(), StoreError> {
    validate_file(admitted, path, file, &OrdinaryFileErrors(path))
}

#[cfg(test)]
mod tests;
