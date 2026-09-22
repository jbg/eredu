//! Explicit copied immutable host sources; no ordinary pointer-import fallback.
use super::{Array, ArrayElement};
use crate::{
    error::Exception, utils::runtime_lock, Dtype, OriginalScopeObserver, PreparedInputRuntime,
    PreparedSubmissionGraphQuota, SubmissionGraphQuota, SubmissionGraphQuotaCause,
    SubmissionGraphQuotaLayout,
};
use std::{alloc::Layout, fmt, marker::PhantomData, mem::size_of, ptr};

/// Physical mechanism identity. This is not ordinary managed-pointer import.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OwnedHostCopyStrategy {
    /// Copy into one immutable prepared CPU/shared-Metal source allocation.
    CopyToImmutablePreparedBacking,
}

/// Fixed refusal; no status implies completion or a new retry budget.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OwnedHostCopyCause {
    /// Shape, dtype, bytes or caller capacity do not match the prepared source.
    Invalid,
    /// Runtime/allocator ownership is currently unavailable; inspect attempted().
    Busy,
    /// A real backing or fixed-control allocation failed.
    AllocationFailed,
    /// Actual metadata or backing capacity is insufficient.
    Capacity,
    /// Non-repeating native allocation identities are exhausted.
    IdentityExhausted,
    /// The supplied observer does not own the actual current original role.
    Domain,
    /// This slot already attempted physical construction or published its array.
    Spent,
    /// The linked native storage implementation does not provide this mechanism.
    Unsupported,
    /// The actual original role already has a retained native failure.
    NativeFailed,
}
impl fmt::Display for OwnedHostCopyCause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Invalid => "prepared host-copy source mismatch",
            Self::Busy => "prepared host-copy owner busy",
            Self::AllocationFailed => "prepared host-copy allocation failed",
            Self::Capacity => "prepared host-copy capacity exhausted",
            Self::IdentityExhausted => "prepared host-copy identity supply exhausted",
            Self::Domain => "prepared host-copy original role mismatch",
            Self::Spent => "prepared host-copy slot already attempted",
            Self::Unsupported => "prepared host-copy storage unavailable",
            Self::NativeFailed => "prepared host-copy original role failed",
        })
    }
}
impl std::error::Error for OwnedHostCopyCause {}
fn cause(status: u32) -> OwnedHostCopyCause {
    match status {
        2 => OwnedHostCopyCause::Busy,
        3 => OwnedHostCopyCause::AllocationFailed,
        4 => OwnedHostCopyCause::Capacity,
        5 => OwnedHostCopyCause::IdentityExhausted,
        6 => OwnedHostCopyCause::Domain,
        7 => OwnedHostCopyCause::Spent,
        8 => OwnedHostCopyCause::Unsupported,
        9 => OwnedHostCopyCause::NativeFailed,
        _ => OwnedHostCopyCause::Invalid,
    }
}

/// Requested storage/copy facts, excluding allocator/platform overhead and fit.
#[derive(Clone, Copy, Debug)]
pub struct OwnedHostCopyFacts {
    raw: safemlx_sys::mlx_owned_host_copy_layout,
    maximum_input_capacity: usize,
    maximum_input_bytes: usize,
}
impl OwnedHostCopyFacts {
    /// Explicit copy mechanism, never ordinary pointer import.
    pub fn strategy(self) -> OwnedHostCopyStrategy {
        OwnedHostCopyStrategy::CopyToImmutablePreparedBacking
    }
    /// Exact source metadata arena capacity requested before activation.
    pub fn metadata_bytes(self) -> usize {
        self.raw.metadata_bytes
    }
    /// Actual prepared allocator's physical capacity for this source.
    pub fn backing_bytes(self) -> usize {
        self.raw.backing_bytes
    }
    /// Bytes copied once on successful physical construction.
    pub fn copy_bytes(self) -> usize {
        self.raw.copy_bytes
    }
    /// Admitted maximum Vec capacity, checked against the actual incoming Vec.
    pub fn maximum_input_capacity(self) -> usize {
        self.maximum_input_capacity
    }
    /// Requested typed allocation bytes at that capacity, not just initialized len.
    pub fn maximum_input_bytes(self) -> usize {
        self.maximum_input_bytes
    }
    /// Exact cold outer array handle, included in metadata_bytes; Array drop
    /// uses its existing prepared-owner disposition and frees the arena block.
    pub fn native_handle_bytes(self) -> usize {
        self.raw.handle_bytes
    }
    /// Actual source arena/retirement layout retaining the caller's C.
    pub fn source_arena_layout<C: Send + 'static>(
        self,
    ) -> Result<SubmissionGraphQuotaLayout, SubmissionGraphQuotaCause> {
        PreparedSubmissionGraphQuota::<C>::layout(self.metadata_bytes())
    }
}

/// Borrowed exact shape and actual runtime facts; no tensor payload is acquired.
#[derive(Debug)]
pub struct OwnedHostCopyPlan<'a, T: ArrayElement + Send + 'static> {
    runtime: &'a PreparedInputRuntime,
    shape: &'a [i32],
    facts: OwnedHostCopyFacts,
    _element: PhantomData<T>,
}
impl<'a, T: ArrayElement + Send + 'static> OwnedHostCopyPlan<'a, T> {
    /// Describe one actual source and its caller-supplied Vec capacity ceiling.
    /// The ceiling is not admission: the selected source owner must justify it.
    pub fn new(
        runtime: &'a PreparedInputRuntime,
        shape: &'a [i32],
        maximum_input_capacity: usize,
    ) -> Result<Self, OwnedHostCopyCause> {
        if T::DTYPE == Dtype::Bool {
            return Err(OwnedHostCopyCause::Unsupported);
        }
        let input =
            Layout::array::<T>(maximum_input_capacity).map_err(|_| OwnedHostCopyCause::Invalid)?;
        let mut raw = safemlx_sys::mlx_owned_host_copy_layout {
            metadata_bytes: 0,
            backing_bytes: 0,
            copy_bytes: 0,
            handle_bytes: 0,
            handle_alignment: 0,
            controls: 0,
            strategy: 0,
        };
        // SAFETY: checked borrowed shape and private actual prepared runtime;
        // pure native layout producer takes no allocation or runtime lock.
        let status = unsafe {
            safemlx_sys::mlx_owned_host_copy_layout_for(
                &mut raw,
                runtime.raw(),
                shape.as_ptr(),
                shape.len(),
                T::DTYPE.into(),
            )
        };
        if status != 0 {
            return Err(cause(status));
        }
        if raw.strategy != safemlx_sys::MLX_OWNED_HOST_COPY_STRATEGY
            || raw.copy_bytes > input.size()
            || raw.copy_bytes % size_of::<T>() != 0
        {
            return Err(OwnedHostCopyCause::Invalid);
        }
        Ok(Self {
            runtime,
            shape,
            facts: OwnedHostCopyFacts {
                raw,
                maximum_input_capacity,
                maximum_input_bytes: input.size(),
            },
            _element: PhantomData,
        })
    }
    /// Copy/backing/control facts for this exact source shape.
    pub fn facts(&self) -> OwnedHostCopyFacts {
        self.facts
    }
    /// Named control representations plus the actual arena allocation and final
    /// handle. Excludes the incoming Vec and native backing/platform allocation.
    /// This remains a partial fact, not a Complete producer-fit certificate.
    pub fn control_bytes<C: Send + 'static>(&self) -> Option<usize> {
        [
            self.facts.source_arena_layout::<C>().ok()?.total_bytes()?,
            self.facts.raw.controls,
            size_of::<Self>(),
            size_of::<PreparedOwnedHostCopy<T, C>>(),
            size_of::<OwnedHostCopyError<T, C>>(),
            size_of::<OwnedHostCopyPreparationError<C>>(),
            size_of::<OwnedHostCopyPreparationOwner<C>>(),
            size_of::<Result<Array, OwnedHostCopyError<T, C>>>(),
            size_of::<Result<PreparedOwnedHostCopy<T, C>, OwnedHostCopyPreparationError<C>>>(),
            size_of::<runtime_lock::RuntimeLockGuard>(),
            size_of::<Option<runtime_lock::RuntimeLockGuard>>(),
            size_of::<safemlx_sys::mlx_array>(),
            size_of::<u32>(),
            size_of::<&OriginalScopeObserver>(),
            size_of::<Layout>(),
            size_of::<Result<CompletedOwnedHostCopy, (OwnedHostCopyCause, Option<Exception>)>>(),
            size_of::<&[T]>(),
            size_of::<usize>(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
    }
    /// Concrete additional transport for an owning buffer, including the shared
    /// borrowed fill worker. Buffer payload/custody is the caller's contribution.
    pub fn control_bytes_with_buffer<C: Send + 'static, B>(&self) -> Option<usize> {
        self.control_bytes::<C>()?
            .checked_add(size_of::<B>())?
            .checked_add(size_of::<OwnedHostBufferCopyError<T, C, B>>())?
            .checked_add(size_of::<Result<Array, OwnedHostBufferCopyError<T, C, B>>>())?
            .checked_add(size_of::<
                Result<CompletedOwnedHostCopy, OwnedHostBufferCopyError<T, C, B>>,
            >())
    }
    /// Consume a genuine caller-owned arena preparation. No payload backing is
    /// allocated yet. The supplied C must contain source/account custody only,
    /// with no native arena, Array, observer, session or manager backedge.
    pub fn prepare<C: Send + 'static>(
        self,
        prepared: PreparedSubmissionGraphQuota<C>,
    ) -> Result<PreparedOwnedHostCopy<T, C>, OwnedHostCopyPreparationError<C>> {
        if prepared.capacity() != self.facts.metadata_bytes() {
            return Err(OwnedHostCopyPreparationError {
                cause: OwnedHostCopyCause::Invalid,
                owner: OwnedHostCopyPreparationOwner::Preparation(prepared),
            });
        }
        let arena = prepared.try_allocate().map_err(|error| {
            let (why, owner) = error.into_parts();
            OwnedHostCopyPreparationError {
                cause: match why {
                    SubmissionGraphQuotaCause::RuntimeBusy => OwnedHostCopyCause::Busy,
                    SubmissionGraphQuotaCause::InvalidCapacity => OwnedHostCopyCause::Invalid,
                    _ => OwnedHostCopyCause::AllocationFailed,
                },
                owner: OwnedHostCopyPreparationOwner::Preparation(owner),
            }
        })?;
        let mut raw = safemlx_sys::mlx_owned_host_copy_slot {
            ctx: ptr::null_mut(),
        };
        let status = {
            let Some(_guard) = runtime_lock::try_enter_for_recovery() else {
                return Err(OwnedHostCopyPreparationError {
                    cause: OwnedHostCopyCause::Busy,
                    owner: OwnedHostCopyPreparationOwner::Arena(arena),
                });
            };
            // SAFETY: empty output, live source arena/private runtime, and exact
            // borrowed shape. Success owns the closed native slot; no input Vec.
            unsafe {
                safemlx_sys::mlx_owned_host_copy_new(
                    &mut raw,
                    self.runtime.raw(),
                    arena.raw(),
                    self.shape.as_ptr(),
                    self.shape.len(),
                    T::DTYPE.into(),
                )
            }
        };
        if status != 0 {
            return Err(OwnedHostCopyPreparationError {
                cause: cause(status),
                owner: OwnedHostCopyPreparationOwner::Arena(arena),
            });
        }
        Ok(PreparedOwnedHostCopy {
            values: None,
            native: Some(NativeSlot(raw)),
            facts: self.facts,
            _custody: PhantomData,
            arena,
        })
    }
}

/// Actual retained stage of a failed cold preparation.
pub enum OwnedHostCopyPreparationOwner<C: Send + 'static> {
    /// Caller custody remains in the same never-allocated arena preparation.
    Preparation(PreparedSubmissionGraphQuota<C>),
    /// The source arena has been allocated and retains C until actual retirement.
    Arena(SubmissionGraphQuota),
}
impl<C: Send + 'static> fmt::Debug for OwnedHostCopyPreparationOwner<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("OwnedHostCopyPreparationOwner")
    }
}
/// Cold failure retains its actual control/source owner; no input was consumed.
pub struct OwnedHostCopyPreparationError<C: Send + 'static> {
    cause: OwnedHostCopyCause,
    owner: OwnedHostCopyPreparationOwner<C>,
}
impl<C: Send + 'static> fmt::Debug for OwnedHostCopyPreparationError<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OwnedHostCopyPreparationError")
            .field("cause", &self.cause)
            .finish_non_exhaustive()
    }
}
impl<C: Send + 'static> OwnedHostCopyPreparationError<C> {
    /// Fixed actual refusal.
    pub fn cause(&self) -> OwnedHostCopyCause {
        self.cause
    }
    /// Move out the same retained owner and cause.
    pub fn into_parts(self) -> (OwnedHostCopyCause, OwnedHostCopyPreparationOwner<C>) {
        (self.cause, self.owner)
    }
}
impl<C: Send + 'static> fmt::Display for OwnedHostCopyPreparationError<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.cause.fmt(f)
    }
}
impl<C: Send + 'static> std::error::Error for OwnedHostCopyPreparationError<C> {}

struct NativeSlot(safemlx_sys::mlx_owned_host_copy_slot);
impl Drop for NativeSlot {
    fn drop(&mut self) {
        // SAFETY: exclusive closed slot; destruction frees native controls and
        // queues source custody, never invokes arbitrary Rust Drop in the lock.
        unsafe { safemlx_sys::mlx_owned_host_copy_free(self.0) };
    }
}

/// Once-only copied-source slot. Source custody is kept by its actual arena.
/// This type is thread-affine; the successfully returned immutable Array is not.
pub struct PreparedOwnedHostCopy<T: ArrayElement + Send + 'static, C: Send + 'static> {
    values: Option<Vec<T>>,
    native: Option<NativeSlot>,
    facts: OwnedHostCopyFacts,
    _custody: PhantomData<fn() -> C>,
    // Last: caller custody outlives Vec allocation, native slot, and their frees.
    arena: SubmissionGraphQuota,
}
impl<T: ArrayElement + Send + 'static, C: Send + 'static> fmt::Debug
    for PreparedOwnedHostCopy<T, C>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreparedOwnedHostCopy")
            .field("facts", &self.facts)
            .field("attempted", &self.attempted())
            .finish_non_exhaustive()
    }
}
impl<T: ArrayElement + Send + 'static, C: Send + 'static> Drop for PreparedOwnedHostCopy<T, C> {
    fn drop(&mut self) {
        drop(self.values.take());
        drop(self.native.take());
    }
}
impl<T: ArrayElement + Send + 'static, C: Send + 'static> PreparedOwnedHostCopy<T, C> {
    /// Explicit copy/storage facts for this source.
    pub fn facts(&self) -> OwnedHostCopyFacts {
        self.facts
    }
    /// Whether native physical construction was attempted. Busy after allocator
    /// entry is spent; runtime/owner/input preflight Busy/refusal is still Ready.
    pub fn attempted(&self) -> bool {
        self.native.as_ref().is_none_or(|native| {
            // SAFETY: read-only state of the exclusive live slot.
            unsafe { safemlx_sys::mlx_owned_host_copy_state(native.0) != 0 }
        })
    }
    /// Consume the actual current Vec. On preflight refusal its pointer, length,
    /// capacity and slot remain in the returned error. No ordinary fallback,
    /// housekeeping, evaluation or extra Vec/Box allocation occurs here.
    pub fn try_fill(
        mut self,
        values: Vec<T>,
        observer: &OriginalScopeObserver,
    ) -> Result<Array, OwnedHostCopyError<T, C>> {
        self.values = Some(values);
        self.fill_stored(observer)
    }
    fn refused(
        self,
        cause: OwnedHostCopyCause,
        native_source: Option<Exception>,
    ) -> OwnedHostCopyError<T, C> {
        OwnedHostCopyError {
            cause,
            native_source,
            owner: self,
        }
    }
    fn fill_stored(
        mut self,
        observer: &OriginalScopeObserver,
    ) -> Result<Array, OwnedHostCopyError<T, C>> {
        let values = self.values.as_ref().expect("fill owns exact current input");
        let array = match self.fill_borrowed(values, values.capacity(), observer) {
            Ok(array) => array,
            Err((cause, native)) => return Err(self.refused(cause, native)),
        };
        drop(self.values.take());
        drop(self);
        Ok(array.into_array())
    }

    // One C fill path. Callers resolve adapter access before this runtime loan.
    fn fill_borrowed(
        &self,
        values: &[T],
        capacity: usize,
        observer: &OriginalScopeObserver,
    ) -> Result<CompletedOwnedHostCopy, (OwnedHostCopyCause, Option<Exception>)> {
        if capacity > self.facts.maximum_input_capacity
            || values.len().checked_mul(size_of::<T>()) != Some(self.facts.raw.copy_bytes)
        {
            return Err((OwnedHostCopyCause::Invalid, None));
        }
        let mut output = safemlx_sys::mlx_array {
            ctx: ptr::null_mut(),
            prepared_owner: ptr::null_mut(),
        };
        let mut birth = safemlx_sys::mlx_original_buffer_info {
            host_control_bytes: 0,
            known: false,
            identity: 0,
            charged_bytes: 0,
            placement: safemlx_sys::mlx_memory_placement {
                kind: 0,
                device: -1,
                device_count: 0,
            },
        };
        let status = {
            let Some(_guard) = runtime_lock::try_enter_for_recovery() else {
                return Err((OwnedHostCopyCause::Busy, Some(observer.error(10))));
            };
            // SAFETY: exact initialized bytes borrowed while this exclusive
            // owner retains the Vec and source arena. Native checks the actual
            // current observer before the sole physical allocation attempt.
            unsafe {
                safemlx_sys::mlx_owned_host_copy_fill_completed(
                    &mut output,
                    &mut birth,
                    self.native.as_ref().expect("live native slot").0,
                    observer.raw,
                    values.as_ptr().cast(),
                    self.facts.raw.copy_bytes,
                )
            }
        };
        if status != 0 {
            let why = cause(status);
            let scoped = match why {
                OwnedHostCopyCause::Busy => 10,
                OwnedHostCopyCause::Domain => 4,
                OwnedHostCopyCause::Spent => 5,
                OwnedHostCopyCause::Capacity => 2,
                OwnedHostCopyCause::AllocationFailed => 3,
                OwnedHostCopyCause::NativeFailed => 7,
                _ => 1,
            };
            return Err((why, Some(observer.error(scoped))));
        }
        Ok(CompletedOwnedHostCopy {
            array: Array { c_array: output },
            birth,
        })
    }

    /// Consume a caller-owned buffer together with its allocation receipt.
    /// Adapter methods run before the native loan; failure keeps that exact buffer,
    /// slot and attempt state. This performs the same immutable copy as try_fill.
    pub fn try_fill_owned<B: OwnedHostCopyBuffer<T>>(
        self,
        buffer: B,
        observer: &OriginalScopeObserver,
    ) -> Result<Array, OwnedHostBufferCopyError<T, C, B>> {
        self.try_fill_owned_completed(buffer, observer)
            .map(CompletedOwnedHostCopy::into_array)
    }
    /// Same actual copy with a success-only owning completion. This carries no
    /// funding authority; a closed accepted source worker must establish origin.
    pub fn try_fill_owned_completed<B: OwnedHostCopyBuffer<T>>(
        self,
        buffer: B,
        observer: &OriginalScopeObserver,
    ) -> Result<CompletedOwnedHostCopy, OwnedHostBufferCopyError<T, C, B>> {
        let capacity = buffer.capacity();
        let result = self.fill_borrowed(buffer.as_slice(), capacity, observer);
        match result {
            Ok(array) => {
                drop(buffer);
                drop(self);
                Ok(array)
            }
            Err((cause, native_source)) => Err(OwnedHostBufferCopyError {
                cause,
                native_source,
                buffer,
                slot: self,
            }),
        }
    }
}

/// The actual Array returned by one successful immutable copy, plus its native
/// birth facts. No raw facts/Array constructor exists. A separately obtained
/// completion does not establish an accepted source-bank origin.
pub struct CompletedOwnedHostCopy {
    array: Array,
    birth: safemlx_sys::mlx_original_buffer_info,
}
impl fmt::Debug for CompletedOwnedHostCopy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CompletedOwnedHostCopy")
            .field("allocation", &self.allocation())
            .finish_non_exhaustive()
    }
}
impl CompletedOwnedHostCopy {
    /// Actual successful birth; None is the producer's allocation-free zero copy.
    pub fn allocation(&self) -> Option<crate::AllocationInfo> {
        self.birth.known.then(|| {
            crate::AllocationInfo::from_native(
                self.birth.identity,
                self.birth.charged_bytes,
                self.birth.placement,
            )
            .with_host_controls(self.birth.host_control_bytes)
        })
    }
    /// Reobserve this retained Array and require the exact successful generation
    /// and whole capacity. No borrowed external Array can replace this owner.
    pub fn observe(
        &self,
    ) -> Result<crate::ImmutableSourceInspection<'_>, crate::OriginalBufferCause> {
        let actual = self.array.inspect_immutable_source()?;
        let same = match &actual {
            crate::ImmutableSourceInspection::Empty => {
                !self.birth.known && self.birth.identity == 0 && self.birth.charged_bytes == 0
            }
            crate::ImmutableSourceInspection::Allocation(value) => {
                let info = value.allocation();
                self.allocation().is_some_and(|prior| {
                    prior.identity() == info.identity() && prior.bytes() == info.bytes()
                })
            }
            crate::ImmutableSourceInspection::Unknown => false,
        };
        if same {
            Ok(actual)
        } else {
            Err(crate::OriginalBufferCause::BirthChanged)
        }
    }
    /// Borrow the actual immutable completed copy without constructing an alias.
    /// Its source owner and successful birth facts remain retained by this value.
    pub fn array(&self) -> &Array {
        &self.array
    }
    /// Extract the actual Array; this operation supplies no publication authority.
    pub fn into_array(self) -> Array {
        self.array
    }
    /// Concrete successful-copy and observation representations, including
    /// their inline Array handles. Native payload and source/input owners are
    /// separately priced; these conservative transport layouts do not add B.
    pub fn control_bytes() -> Option<usize> {
        let parts = [
            size_of::<Self>(),
            size_of::<safemlx_sys::mlx_original_buffer_info>(),
            size_of::<crate::ImmutableSourceInspection<'static>>(),
            size_of::<Result<crate::ImmutableSourceInspection<'static>, crate::OriginalBufferCause>>(
            ),
            crate::ImmutableSourceWitness::inspection_control_bytes()?,
        ];
        parts
            .into_iter()
            .try_fold(std::mem::size_of_val(&parts), usize::checked_add)
    }
}

/// Owning failure: cause before Vec/native/source custody, with no early refund.
pub struct OwnedHostCopyError<T: ArrayElement + Send + 'static, C: Send + 'static> {
    cause: OwnedHostCopyCause,
    native_source: Option<Exception>,
    owner: PreparedOwnedHostCopy<T, C>,
}

/// Owned initialized input. The implementation retains its allocation custody
/// until after storage destruction; it grants no native operation or byte budget.
pub trait OwnedHostCopyBuffer<T: ArrayElement> {
    /// Borrow the actual initialized values.
    fn as_slice(&self) -> &[T];
    /// Actual exposed typed capacity, checked against the prepared ceiling.
    fn capacity(&self) -> usize;
}

/// Failure retaining the exact caller buffer and unchanged native slot.
pub struct OwnedHostBufferCopyError<T: ArrayElement + Send + 'static, C: Send + 'static, B> {
    cause: OwnedHostCopyCause,
    native_source: Option<Exception>,
    buffer: B,
    slot: PreparedOwnedHostCopy<T, C>,
}
impl<T: ArrayElement + Send + 'static, C: Send + 'static, B> OwnedHostBufferCopyError<T, C, B> {
    /// Fixed actual refusal.
    pub fn cause(&self) -> OwnedHostCopyCause {
        self.cause
    }
    /// Same owned input, including its allocation receipt.
    pub fn buffer(&self) -> &B {
        &self.buffer
    }
    /// Whether the physical allocation was actually attempted.
    pub fn attempted(&self) -> bool {
        self.slot.attempted()
    }
    /// Move only the retained native snapshot; buffer/slot custody stays here.
    pub fn take_native_source(&mut self) -> Option<Exception> {
        self.native_source.take()
    }
}
impl<T: ArrayElement + Send + 'static, C: Send + 'static, B> fmt::Debug
    for OwnedHostBufferCopyError<T, C, B>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OwnedHostBufferCopyError")
            .field("cause", &self.cause)
            .field("attempted", &self.attempted())
            .finish_non_exhaustive()
    }
}
impl<T: ArrayElement + Send + 'static, C: Send + 'static, B> fmt::Display
    for OwnedHostBufferCopyError<T, C, B>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.cause.fmt(f)
    }
}
impl<T: ArrayElement + Send + 'static, C: Send + 'static, B> std::error::Error
    for OwnedHostBufferCopyError<T, C, B>
{
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.native_source.as_ref().map(|e| e as _)
    }
}
impl<T: ArrayElement + Send + 'static, C: Send + 'static> OwnedHostCopyError<T, C> {
    /// Fixed actual refusal, not completion.
    pub fn cause(&self) -> OwnedHostCopyCause {
        self.cause
    }
    /// Actual original native snapshot, when the refusal crossed that boundary.
    pub fn native_source(&self) -> Option<&Exception> {
        self.native_source.as_ref()
    }
    /// Moves the actual native snapshot out once, without cloning or formatting.
    ///
    /// The same Vec, native slot, source custody and attempt state remain here.
    /// Afterwards `native_source()` and `Error::source()` return `None`; a second
    /// extraction also returns `None`. An escaping error owner must retain its
    /// own required source/control custody independently of this pending owner.
    pub fn take_native_source(&mut self) -> Option<Exception> {
        self.native_source.take()
    }
    /// Same owned Vec, including the original pointer and capacity.
    pub fn values(&self) -> &Vec<T> {
        self.owner
            .values
            .as_ref()
            .expect("error retains current input")
    }
    /// Actual slot attempt state; source validation refusal does not consume it.
    pub fn attempted(&self) -> bool {
        self.owner.attempted()
    }
    /// Retry this same retained input/slot only. A spent slot still refuses;
    /// this method grants no new admission or retry budget.
    pub fn retry(self, observer: &OriginalScopeObserver) -> Result<Array, Self> {
        self.owner.fill_stored(observer)
    }
}
impl<T: ArrayElement + Send + 'static, C: Send + 'static> fmt::Debug for OwnedHostCopyError<T, C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OwnedHostCopyError")
            .field("cause", &self.cause)
            .field("attempted", &self.attempted())
            .finish_non_exhaustive()
    }
}
impl<T: ArrayElement + Send + 'static, C: Send + 'static> fmt::Display
    for OwnedHostCopyError<T, C>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.cause.fmt(f)
    }
}
impl<T: ArrayElement + Send + 'static, C: Send + 'static> std::error::Error
    for OwnedHostCopyError<T, C>
{
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.native_source.as_ref().map(|e| e as _)
    }
}
