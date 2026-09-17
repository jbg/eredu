//! One native failure publication with the existing closed Rust retirement queue.
use super::{destroy, retire, take_owner, OwnedNode, RetiredOwner};
use std::{alloc::Layout, ffi::c_void, fmt, mem, ptr};

/// Fixed construction refusal. It preserves the supplied owner and allocates no diagnostic.
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum PrefillFailureCause {
    /// Checked concrete layout arithmetic failed.
    #[error("invalid prefill failure layout")]
    InvalidLayout,
    /// The Rust retirement node could not be allocated.
    #[error("prefill failure custody allocation failed")]
    Allocation,
    /// The native carrier could not be allocated; no handoff occurred.
    #[error("native prefill failure carrier allocation failed")]
    NativeAllocation,
}
/// Cause first and unchanged custody last on every failed construction.
pub struct PrefillFailureError<T> {
    cause: PrefillFailureCause,
    owner: T,
}
impl<T> PrefillFailureError<T> {
    /// The fixed construction cause.
    pub fn cause(&self) -> PrefillFailureCause {
        self.cause
    }
    /// The actual unchanged original owner.
    pub fn owner(&self) -> &T {
        &self.owner
    }
    /// Consume this failure without releasing its owner.
    pub fn into_parts(self) -> (PrefillFailureCause, T) {
        (self.cause, self.owner)
    }
}
impl<T> fmt::Debug for PrefillFailureError<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PrefillFailureError")
            .field("cause", &self.cause)
            .finish_non_exhaustive()
    }
}
impl<T> fmt::Display for PrefillFailureError<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.cause.fmt(f)
    }
}
impl<T> std::error::Error for PrefillFailureError<T> {}

/// Measured shell and queue storage. Original thrown objects, exception ABI,
/// event/task producers and opaque platform storage remain separate populations.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PrefillFailureLayout {
    /// Concrete C++ carrier allocation, including its atomic publication.
    pub native_owner_bytes: usize,
    /// Alignment of that carrier.
    pub native_owner_alignment: usize,
    /// Existing Rust retirement node including the actual T.
    pub rust_node_bytes: usize,
    /// Named construction, inspection, failure and retirement values.
    pub control_bytes: usize,
}
impl PrefillFailureLayout {
    /// Checked contribution consumed before either owner is constructed.
    pub fn total_bytes(self) -> Option<usize> {
        self.native_owner_bytes
            .checked_add(self.rust_node_bytes)?
            .checked_add(self.control_bytes)
    }
}
struct Construction {
    raw: safemlx_sys::mlx_prefill_failure,
    status: u32,
}

/// Closed native strong owner of one immutable first failure. Cloning retains
/// the same allocation and custody, with no new header or native source capture.
/// No raw ownership, Weak, reset, replacement or publication API is exported.
pub struct RetainedPrefillFailure {
    raw: safemlx_sys::mlx_prefill_failure,
}
// SAFETY: all native references use an atomic count and immutable release/acquire
// publication. Rust T is never borrowed natively; its sole callback queues the
// existing T: Send node after every native payload/header retires. No Scope,
// Record or root owner is retained by this carrier.
unsafe impl Send for RetainedPrefillFailure {}
unsafe impl Sync for RetainedPrefillFailure {}
impl Clone for RetainedPrefillFailure {
    fn clone(&self) -> Self {
        // SAFETY: this live handle pins the exact atomic native allocation.
        unsafe { safemlx_sys::mlx_prefill_failure_retain(self.raw) };
        Self { raw: self.raw }
    }
}
impl Drop for RetainedPrefillFailure {
    fn drop(&mut self) {
        // SAFETY: native payload and header retire before the callback only
        // enqueues the Rust node. No arbitrary Rust Drop runs in this call.
        unsafe { safemlx_sys::mlx_prefill_failure_free(self.raw) };
    }
}
impl fmt::Debug for RetainedPrefillFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RetainedPrefillFailure")
            .finish_non_exhaustive()
    }
}
impl RetainedPrefillFailure {
    /// Consume exactly one native retained alias supplied by a closed C helper.
    /// A null result carries no invented custody or publication authority.
    pub(crate) unsafe fn from_owned_raw(raw: safemlx_sys::mlx_prefill_failure) -> Option<Self> {
        if raw.ctx.is_null() {
            None
        } else {
            Some(Self { raw })
        }
    }
    pub(crate) fn raw(&self) -> safemlx_sys::mlx_prefill_failure {
        self.raw
    }
    fn view(&self) -> Option<safemlx_sys::mlx_prefill_failure_view> {
        let mut value = safemlx_sys::mlx_prefill_failure_view {
            kind: 0,
            message: ptr::null(),
            message_size: 0,
            source_type: ptr::null(),
            source_type_size: 0,
            native_code: 0,
        };
        // SAFETY: the live owner pins all returned immutable borrowed bytes.
        (unsafe { safemlx_sys::mlx_prefill_failure_view_get(&mut value, self.raw) } == 0)
            .then_some(value)
    }
    /// Clone only a fully published cause. Empty or writing state is not an error
    /// source and cannot be mistaken for completion or an invented diagnostic.
    pub fn error(&self) -> Option<PrefillNativeError> {
        self.view()?;
        Some(PrefillNativeError {
            owner: self.clone(),
        })
    }
}

/// Actual retained source representation, without converting it to a String.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PrefillNativeFailureKind {
    /// Original C++ exception object and dynamic type remain retained.
    Exception,
    /// Original platform error object and domain/code remain retained.
    Native,
    /// The producer could not expose a source; this is an explicit missing fact.
    SourceUnavailable,
}
/// Escapable alias of the actual first native cause and its original custody.
#[derive(Clone)]
pub struct PrefillNativeError {
    owner: RetainedPrefillFailure,
}
impl PrefillNativeError {
    fn view(&self) -> safemlx_sys::mlx_prefill_failure_view {
        self.owner
            .view()
            .expect("immutable published native failure")
    }
    /// The actual native source representation.
    pub fn kind(&self) -> PrefillNativeFailureKind {
        match self.view().kind {
            1 => PrefillNativeFailureKind::Exception,
            2 => PrefillNativeFailureKind::Native,
            _ => PrefillNativeFailureKind::SourceUnavailable,
        }
    }
    /// Borrow exact producer bytes, preserving non-UTF8 C++ diagnostics.
    pub fn message_bytes(&self) -> Option<&[u8]> {
        let value = self.view();
        if value.kind != 1 {
            return None;
        }
        if value.message_size == 0 {
            return Some(&[]);
        }
        // SAFETY: nonempty native view points into the retained source.
        Some(unsafe { std::slice::from_raw_parts(value.message.cast(), value.message_size) })
    }
    /// Borrow the original C++ exception type name. Native domains use
    /// [`Self::copy_native_text`] with [`PrefillNativeText::Domain`].
    pub fn source_type_bytes(&self) -> Option<&[u8]> {
        let value = self.view();
        if value.kind != 1 {
            return None;
        }
        if value.source_type_size == 0 {
            return Some(&[]);
        }
        // SAFETY: the retained exception pins its immutable C++ type-name bytes.
        Some(unsafe {
            std::slice::from_raw_parts(value.source_type.cast(), value.source_type_size)
        })
    }
    /// Copy native UTF-16 into caller storage, returning copied and remaining
    /// units. Native conversion buffers never escape into a Rust borrow.
    pub fn copy_native_text(
        &self,
        field: PrefillNativeText,
        offset: usize,
        output: &mut [u16],
    ) -> Option<(usize, usize)> {
        let (mut used, mut remaining) = (0, 0);
        // SAFETY: output is the supplied valid caller-owned range; the native
        // owner pins its immutable retained NSString throughout this copy.
        let status = unsafe {
            safemlx_sys::mlx_prefill_failure_native_text(
                &mut used,
                &mut remaining,
                self.owner.raw,
                field as u32,
                offset,
                output.as_mut_ptr(),
                output.len(),
            )
        };
        (status == 0).then_some((used, remaining))
    }
    /// Original platform error code when the source is native.
    pub fn native_code(&self) -> Option<i64> {
        (self.kind() == PrefillNativeFailureKind::Native).then(|| self.view().native_code)
    }
}
impl fmt::Debug for PrefillNativeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PrefillNativeError")
            .field("kind", &self.kind())
            .finish_non_exhaustive()
    }
}
impl fmt::Display for PrefillNativeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.kind() == PrefillNativeFailureKind::Native {
            return write_utf16(
                NativeTextReader {
                    error: self,
                    field: PrefillNativeText::Description,
                },
                f,
            );
        }
        let mut bytes = self.message_bytes().unwrap_or_default();
        if bytes.is_empty() {
            return write!(f, "native prefill failure ({:?})", self.kind());
        }
        // Stream valid UTF8 and replacement characters directly to the caller's
        // formatter. No owned lossy conversion or source recapture occurs.
        while !bytes.is_empty() {
            match std::str::from_utf8(bytes) {
                Ok(text) => return f.write_str(text),
                Err(error) => {
                    let valid = error.valid_up_to();
                    f.write_str(
                        std::str::from_utf8(&bytes[..valid]).expect("validated UTF8 prefix"),
                    )?;
                    f.write_str("\u{fffd}")?;
                    bytes = match error.error_len() {
                        Some(length) => &bytes[valid + length..],
                        None => &[],
                    };
                }
            }
        }
        Ok(())
    }
}
impl std::error::Error for PrefillNativeError {}

/// Exact retained native text field; each read copies into caller storage.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum PrefillNativeText {
    /// Actual localized description captured with the native error.
    Description = 0,
    /// Actual native error domain retained independently of conversion buffers.
    Domain = 1,
}
trait ReadUtf16 {
    fn read(&mut self, offset: usize, output: &mut [u16]) -> Option<(usize, usize)>;
}
struct NativeTextReader<'a> {
    error: &'a PrefillNativeError,
    field: PrefillNativeText,
}
impl ReadUtf16 for NativeTextReader<'_> {
    fn read(&mut self, offset: usize, output: &mut [u16]) -> Option<(usize, usize)> {
        self.error.copy_native_text(self.field, offset, output)
    }
}
struct Utf16Chunks<R> {
    reader: R,
    buffer: [u16; 64],
    start: usize,
    used: usize,
    offset: usize,
    remaining: usize,
}
impl<R: ReadUtf16> Iterator for Utf16Chunks<R> {
    type Item = u16;
    fn next(&mut self) -> Option<u16> {
        if self.start == self.used {
            if self.remaining == 0 {
                return None;
            }
            let (used, remaining) = self.reader.read(self.offset, &mut self.buffer)?;
            if used == 0 {
                self.remaining = 0;
                return None;
            }
            self.start = 0;
            self.used = used;
            self.offset += used;
            self.remaining = remaining;
        }
        let value = self.buffer[self.start];
        self.start += 1;
        Some(value)
    }
}
fn write_utf16<R: ReadUtf16>(reader: R, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    let units = Utf16Chunks {
        reader,
        buffer: [0; 64],
        start: 0,
        used: 0,
        offset: 0,
        remaining: usize::MAX,
    };
    // decode_utf16 retains a high surrogate while next() refills the same
    // bounded buffer. Lone surrogates use U+FFFD; embedded NUL is preserved.
    for value in std::char::decode_utf16(units) {
        let mut utf8 = [0; 4];
        f.write_str(
            value
                .unwrap_or(char::REPLACEMENT_CHARACTER)
                .encode_utf8(&mut utf8),
        )?;
    }
    Ok(())
}

/// Move-only preallocated retirement node. T must contain only source/account
/// custody, never a Scope, Record or root that will retain this carrier.
pub struct PreparedPrefillFailure<T: Send + 'static> {
    layout: PrefillFailureLayout,
    node: Option<Box<OwnedNode<T>>>,
}
impl<T: Send + 'static> Drop for PreparedPrefillFailure<T> {
    fn drop(&mut self) {
        if let Some(node) = self.node.take() {
            drop(take_owner(node));
        }
    }
}
impl<T: Send + 'static> fmt::Debug for PreparedPrefillFailure<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreparedPrefillFailure")
            .field("layout", &self.layout)
            .finish_non_exhaustive()
    }
}
impl<T: Send + 'static> PreparedPrefillFailure<T> {
    /// Cold layout; no TLS, allocator, native runtime or source callback.
    pub fn layout() -> Result<PrefillFailureLayout, PrefillFailureCause> {
        let mut facts = safemlx_sys::mlx_prefill_failure_layout {
            owner_bytes: 0,
            owner_alignment: 0,
            native_controls: 0,
            retirement_controls: 0,
        };
        // SAFETY: fixed arithmetic writes only the supplied layout.
        if unsafe { safemlx_sys::mlx_prefill_failure_layout_for(&mut facts) } != 0 {
            return Err(PrefillFailureCause::InvalidLayout);
        }
        let parts = [
            mem::size_of::<Self>(),
            mem::size_of::<T>(),
            mem::size_of::<OwnedNode<T>>(),
            mem::size_of::<Layout>(),
            mem::size_of::<*mut OwnedNode<T>>(),
            mem::size_of::<Construction>(),
            mem::size_of::<*mut c_void>(),
            mem::size_of::<Box<OwnedNode<T>>>(),
            mem::size_of::<RetainedPrefillFailure>(),
            mem::size_of::<Result<Self, PrefillFailureError<T>>>(),
            mem::size_of::<Result<RetainedPrefillFailure, PrefillFailureError<Self>>>(),
            mem::size_of::<Option<Box<OwnedNode<T>>>>(),
            mem::size_of::<T>(),
            mem::size_of::<super::RetirementBatch>(),
            mem::size_of::<*mut RetiredOwner>(),
            mem::size_of::<unsafe fn(*mut RetiredOwner)>(),
            mem::size_of::<PrefillNativeError>(),
            mem::size_of::<Option<PrefillNativeError>>(),
            mem::size_of::<safemlx_sys::mlx_prefill_failure_view>(),
            mem::size_of::<safemlx_sys::mlx_prefill_failure_layout>(),
            mem::size_of::<Utf16Chunks<NativeTextReader<'static>>>(),
            mem::size_of::<std::char::DecodeUtf16<Utf16Chunks<NativeTextReader<'static>>>>(),
            mem::size_of::<[u8; 4]>(),
            facts.native_controls,
            facts.retirement_controls,
        ];
        let control_bytes = parts
            .into_iter()
            .try_fold(mem::size_of_val(&parts), usize::checked_add)
            .ok_or(PrefillFailureCause::InvalidLayout)?;
        Ok(PrefillFailureLayout {
            native_owner_bytes: facts.owner_bytes,
            native_owner_alignment: facts.owner_alignment,
            rust_node_bytes: mem::size_of::<OwnedNode<T>>(),
            control_bytes,
        })
    }
    /// Allocate only the existing Rust queue node after original comparison.
    pub fn try_new(owner: T) -> Result<Self, PrefillFailureError<T>> {
        let layout = match Self::layout() {
            Ok(value) => value,
            Err(cause) => return Err(PrefillFailureError { cause, owner }),
        };
        let allocation = Layout::new::<OwnedNode<T>>();
        // SAFETY: nonzero matching layout, initialized before Box creation.
        let node = unsafe { std::alloc::alloc(allocation) }.cast::<OwnedNode<T>>();
        if node.is_null() {
            return Err(PrefillFailureError {
                cause: PrefillFailureCause::Allocation,
                owner,
            });
        }
        let node = unsafe {
            node.write(OwnedNode {
                retired: RetiredOwner {
                    next: ptr::null_mut(),
                    destroy: destroy::<T>,
                },
                owner,
            });
            Box::from_raw(node)
        };
        Ok(Self {
            layout,
            node: Some(node),
        })
    }
    /// Deallocate the never-handed-off node before returning its owner.
    pub fn into_owner(mut self) -> T {
        take_owner(self.node.take().expect("unconsumed failure preparation"))
    }
    /// Allocate one native carrier. Only success transfers the queue node.
    pub fn try_allocate(mut self) -> Result<RetainedPrefillFailure, PrefillFailureError<Self>> {
        let mut local = Construction {
            raw: safemlx_sys::mlx_prefill_failure {
                ctx: ptr::null_mut(),
            },
            status: 1,
        };
        let payload = (&mut **self.node.as_mut().expect("unconsumed failure preparation")
            as *mut OwnedNode<T>)
            .cast();
        // SAFETY: this no-hooks constructor either leaves custody untouched or
        // accepts exactly this initialized node; nothing can release it here.
        local.status = unsafe {
            safemlx_sys::mlx_prefill_failure_new_retaining(&mut local.raw, payload, Some(retire))
        };
        if local.status == 0 {
            let _ = Box::into_raw(self.node.take().expect("unconsumed failure preparation"));
            Ok(RetainedPrefillFailure { raw: local.raw })
        } else {
            Err(PrefillFailureError {
                cause: PrefillFailureCause::NativeAllocation,
                owner: self,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Barrier,
    };
    struct Units<'a>(&'a [u16]);
    impl ReadUtf16 for Units<'_> {
        fn read(&mut self, offset: usize, output: &mut [u16]) -> Option<(usize, usize)> {
            let rest = self.0.get(offset..)?;
            let count = rest.len().min(output.len());
            output[..count].copy_from_slice(&rest[..count]);
            Some((count, rest.len() - count))
        }
    }
    struct Text<'a>(&'a [u16]);
    impl fmt::Display for Text<'_> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write_utf16(Units(self.0), f)
        }
    }
    #[test]
    fn native_text_surrogates_cross_the_actual_fixed_buffer_without_loss() {
        let mut units = vec![u16::from(b'x'); 63];
        units.extend([0xd83d, 0xde42, 0, u16::from(b'z')]);
        assert_eq!(
            format!("{}", Text(&units)),
            format!("{}🙂\0z", "x".repeat(63))
        );
        assert_eq!(format!("{}", Text(&[])), "");
        assert_eq!(format!("{}", Text(&[0xd83d, 0x61, 0xde42])), "�a�");
        assert_eq!(format!("{}", Text(&[0xd83d])), "�");
    }

    thread_local! { static NATIVE_RELEASE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) }; }
    struct Custody(Arc<AtomicUsize>);
    impl Drop for Custody {
        fn drop(&mut self) {
            NATIVE_RELEASE
                .with(|active| assert!(!active.get(), "native release ran Rust custody Drop"));
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    #[test]
    fn concurrent_final_native_carrier_aliases_queue_one_actual_rust_owner() {
        let count = Arc::new(AtomicUsize::new(0));
        let prepared = PreparedPrefillFailure::try_new(Custody(count.clone())).unwrap();
        let owner = prepared.try_allocate().unwrap();
        assert!(owner.error().is_none());
        let alias = owner.clone();
        let barrier = Arc::new(Barrier::new(2));
        let one = barrier.clone();
        let worker = std::thread::spawn(move || {
            one.wait();
            NATIVE_RELEASE.with(|active| active.set(true));
            drop(alias);
            NATIVE_RELEASE.with(|active| active.set(false));
        });
        assert_eq!(count.load(Ordering::SeqCst), 0);
        barrier.wait();
        NATIVE_RELEASE.with(|active| active.set(true));
        drop(owner);
        NATIVE_RELEASE.with(|active| active.set(false));
        worker.join().unwrap();
        // Native release never runs Drop inline. A concurrent ordinary host
        // reclaimer may already have consumed the queued node after final release.
        crate::reclaim_allocation_owners();
        assert_eq!(count.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn unallocated_native_failure_returns_the_identical_custody() {
        let count = Arc::new(AtomicUsize::new(0));
        let prepared = PreparedPrefillFailure::try_new(Custody(count.clone())).unwrap();
        let owner = prepared.into_owner();
        assert_eq!(count.load(Ordering::SeqCst), 0);
        assert!(Arc::ptr_eq(&owner.0, &count));
        drop(owner);
        assert_eq!(count.load(Ordering::SeqCst), 1);
    }
}

impl RetainedPrefillFailure {
    /// Bind this actual prepaid owner to one original-required empty Scope,
    /// without creating roots or treating a failure as completion evidence.
    pub fn bind_original_scope(
        &self,
        scope: &crate::SubmissionScope,
    ) -> Result<(), crate::OriginalNativeControlError> {
        // SAFETY: both retained handles are live and the Scope remains thread-affine;
        // this fixed operation only binds an unused carrier before native work.
        let status = unsafe {
            safemlx_sys::mlx_prefill_failure_bind_original_scope(self.raw(), scope.raw())
        };
        crate::OriginalNativeControlError::from_status(status)
    }
}
