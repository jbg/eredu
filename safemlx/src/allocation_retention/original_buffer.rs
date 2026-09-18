//! Fixed native budget custody and borrowed birth authentication.
//! This mechanism does not issue a neutral reservation or certify producer fit.
use super::{destroy, retire, take_owner, OwnedNode, PreparedAllocationOwner, RetiredOwner};
use crate::{
    utils::runtime_lock, AllocationInfo, Array, OriginalNativeControlError, PreparedInputRuntime,
};
use std::{alloc::Layout, ffi::c_void, fmt, marker::PhantomData, mem::size_of, ptr, rc::Rc};

/// Fixed refusal. No variant certifies completion or refunds a reservation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum OriginalBufferCause {
    /// The actual role or budget is absent.
    #[error("original native buffer binding is missing")]
    MissingBinding,
    /// The actual birth or parent belongs to a different budget.
    #[error("original native buffer budget differs")]
    ForeignDomain,
    /// The role is not current, empty and configured for this operation.
    #[error("original native buffer scope is invalid")]
    InvalidScope,
    /// The same role already has its fixed budget.
    #[error("original native buffer budget is already bound")]
    AlreadyBound,
    /// The checked representation or input is invalid.
    #[error("original native buffer layout is invalid")]
    InvalidLayout,
    /// Physical or metadata capacity is insufficient.
    #[error("original native buffer capacity is exhausted")]
    Capacity,
    /// No runtime/allocator loan was obtained; no hidden retry is performed.
    #[error("original native buffer runtime is busy")]
    RuntimeBusy,
    /// The actual owner or backing allocator refused construction.
    #[error("original native buffer allocation failed")]
    AllocationFailed,
    /// The actual allocator cannot provide this mechanism.
    #[error("original native buffer mechanism is unavailable")]
    Unsupported,
    /// Nonrepeating native generations are exhausted.
    #[error("original native buffer identity supply is exhausted")]
    IdentityExhausted,
    /// The actual original control contract rejected the operation.
    #[error("{0}")]
    NativeControl(#[source] OriginalNativeControlError),
    /// The current value has no certified settled original mutable birth.
    #[error("original native buffer backing is uncertified")]
    UncertifiedBacking,
    /// The actual birth identity or full capacity differs from the observation.
    #[error("original native buffer birth changed before attachment")]
    BirthChanged,
    /// An unknown ABI status is retained without formatting or reinterpretation.
    #[error("unexpected original native buffer status {0}")]
    InvalidStatus(u32),
}
impl OriginalBufferCause {
    pub(crate) fn check(status: u32) -> Result<(), Self> {
        use safemlx_sys::*;
        Err(match status {
            MLX_ORIGINAL_BUFFER_OK => return Ok(()),
            MLX_ORIGINAL_BUFFER_MISSING => Self::MissingBinding,
            MLX_ORIGINAL_BUFFER_FOREIGN => Self::ForeignDomain,
            MLX_ORIGINAL_BUFFER_SCOPE => Self::InvalidScope,
            MLX_ORIGINAL_BUFFER_BOUND => Self::AlreadyBound,
            MLX_ORIGINAL_BUFFER_LAYOUT => Self::InvalidLayout,
            MLX_ORIGINAL_BUFFER_CAPACITY => Self::Capacity,
            MLX_ORIGINAL_BUFFER_BUSY => Self::RuntimeBusy,
            MLX_ORIGINAL_BUFFER_ALLOCATION => Self::AllocationFailed,
            MLX_ORIGINAL_BUFFER_UNSUPPORTED => Self::Unsupported,
            MLX_ORIGINAL_BUFFER_IDENTITY => Self::IdentityExhausted,
            MLX_ORIGINAL_BUFFER_UNCERTIFIED => Self::UncertifiedBacking,
            MLX_ORIGINAL_BUFFER_CHANGED => Self::BirthChanged,
            value
                if value > MLX_ORIGINAL_BUFFER_NATIVE_CONTROL_BASE
                    && value < MLX_ORIGINAL_BUFFER_UNEXPECTED =>
            {
                match OriginalNativeControlError::from_status(
                    value - MLX_ORIGINAL_BUFFER_NATIVE_CONTROL_BASE,
                ) {
                    Err(error) => Self::NativeControl(error),
                    Ok(()) => Self::InvalidStatus(value),
                }
            }
            value => Self::InvalidStatus(value),
        })
    }
}

/// Unchanged original owner or preparation follows its fixed cause.
pub struct OriginalBufferError<T> {
    cause: OriginalBufferCause,
    owner: T,
}
impl<T> OriginalBufferError<T> {
    pub(super) fn new(cause: OriginalBufferCause, owner: T) -> Self {
        Self { cause, owner }
    }
    /// The actual refusal without cloning or dropping custody.
    pub fn cause(&self) -> OriginalBufferCause {
        self.cause
    }
    /// Borrow the unchanged exclusive owner.
    pub fn owner(&self) -> &T {
        &self.owner
    }
    /// Consume the error while keeping all original custody.
    pub fn into_parts(self) -> (OriginalBufferCause, T) {
        (self.cause, self.owner)
    }
}
impl<T> fmt::Debug for OriginalBufferError<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OriginalBufferError")
            .field("cause", &self.cause)
            .finish_non_exhaustive()
    }
}
impl<T> fmt::Display for OriginalBufferError<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.cause.fmt(f)
    }
}
impl<T> std::error::Error for OriginalBufferError<T> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}

/// Exact owner representations; excludes the separately reserved capacity,
/// owner payload allocations, Graph birth blocks and whole-execution fit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OriginalBufferBudgetLayout {
    /// Fixed future native physical allowance; no payload is allocated at setup.
    pub capacity: usize,
    /// Actual native budget object.
    pub native_owner_bytes: usize,
    /// The existing Rust retirement node, including the concrete owner T.
    pub rust_node_bytes: usize,
    /// Named creation, binding, inspection and retirement representations.
    pub control_bytes: usize,
}
impl OriginalBufferBudgetLayout {
    /// Owner construction contribution only; capacity is independently reserved.
    pub fn total_owner_bytes(self) -> Option<usize> {
        self.native_owner_bytes
            .checked_add(self.rust_node_bytes)?
            .checked_add(self.control_bytes)
    }
}
/// Physical allowance for an original allocation request or finite population.
/// This is a bound from caller-supplied producer facts, never admission authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OriginalBufferPopulationLayout {
    capacity: usize,
    control_bytes: usize,
}
impl OriginalBufferPopulationLayout {
    /// Sum bound after independently rounding every possible positive birth.
    pub fn capacity(self) -> usize {
        self.capacity
    }
    /// Actual fixed native and Rust representations of this pure query.
    pub fn control_bytes(self) -> usize {
        self.control_bytes
    }
}
struct Construction {
    raw: safemlx_sys::mlx_original_buffer_budget,
    status: u32,
}

/// Closed strong owner of one native budget. Native Scope and birth owners
/// independently retain it. No raw owner, Weak, resizing or grant API escapes.
pub struct OriginalBufferBudget {
    raw: safemlx_sys::mlx_original_buffer_budget,
    _thread: PhantomData<Rc<()>>,
}
impl fmt::Debug for OriginalBufferBudget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OriginalBufferBudget")
            .finish_non_exhaustive()
    }
}
impl Clone for OriginalBufferBudget {
    fn clone(&self) -> Self {
        // SAFETY: this strong handle pins the same intrusive atomic native owner.
        unsafe { safemlx_sys::mlx_original_buffer_budget_retain(self.raw) };
        Self {
            raw: self.raw,
            _thread: PhantomData,
        }
    }
}
impl Drop for OriginalBufferBudget {
    fn drop(&mut self) {
        // SAFETY: native final release deletes its object before enqueueing the
        // Send capsule. Arbitrary Rust destruction uses the existing unlocked queue.
        unsafe { safemlx_sys::mlx_original_buffer_budget_release(self.raw) };
    }
}
impl OriginalBufferBudget {
    /// Exact one-birth capacity from the retained allocator's existing physical
    /// worker. CPU's header (also for zero bytes) and Metal's empty/rounded
    /// allocation remain distinct. No allocation or authority is issued.
    pub fn request_layout(runtime:&PreparedInputRuntime,requested_bytes:usize)
        ->Result<OriginalBufferPopulationLayout,OriginalBufferCause> {
        let mut raw=safemlx_sys::mlx_original_buffer_population_layout{capacity:0,control_bytes:0};
        // SAFETY: closed runtime retains its actual immutable allocator facts.
        OriginalBufferCause::check(unsafe{safemlx_sys::mlx_original_buffer_request_layout_for(
            &mut raw,runtime.raw(),requested_bytes)})?;
        Ok(OriginalBufferPopulationLayout{capacity:raw.capacity,
            control_bytes:Self::request_layout_control_bytes().ok_or(OriginalBufferCause::InvalidLayout)?})
    }
    /// Fixed complete query transports, available before request_layout.
    pub fn request_layout_control_bytes()->Option<usize> {
        // SAFETY: only fixed native layouts; no runtime/source query or work.
        let native=unsafe{safemlx_sys::mlx_original_buffer_request_control_bytes()};
        let frames=[size_of::<safemlx_sys::mlx_original_buffer_population_layout>(),
            size_of::<OriginalBufferPopulationLayout>(),
            size_of::<Result<OriginalBufferPopulationLayout,OriginalBufferCause>>(),
            size_of::<&PreparedInputRuntime>(),size_of::<usize>()*3,size_of::<u32>()];
        frames.into_iter().try_fold(native.checked_add(size_of::<[usize;6]>())?,usize::checked_add)
    }
    /// Bound original Metal backing from the sum of requested payload bytes and
    /// the maximum number of positive allocation attempts, retaining every birth.
    /// The actual retained allocator supplies the page quantum. Zero payload has
    /// zero physical backing; CPU header rules are deliberately unsupported.
    /// This pure query does not initialize, allocate, lend a runtime, or issue credit.
    pub fn metal_population_layout(
        runtime: &PreparedInputRuntime,
        requested_bytes: usize,
        maximum_births: usize,
    ) -> Result<OriginalBufferPopulationLayout, OriginalBufferCause> {
        let mut raw = safemlx_sys::mlx_original_buffer_population_layout {
            capacity: 0,
            control_bytes: 0,
        };
        // SAFETY: runtime keeps genuine immutable allocator facts alive. Native
        // reads those scalar facts and the linked physical rounding worker only.
        OriginalBufferCause::check(unsafe {
            safemlx_sys::mlx_original_buffer_metal_population_layout_for(
                &mut raw,
                runtime.raw(),
                requested_bytes,
                maximum_births,
            )
        })?;
        let controls = [
            raw.control_bytes,
            size_of::<safemlx_sys::mlx_original_buffer_population_layout>(),
            size_of::<OriginalBufferPopulationLayout>(),
            size_of::<Result<OriginalBufferPopulationLayout, OriginalBufferCause>>(),
            size_of::<&PreparedInputRuntime>(),
            size_of::<usize>() * 3,
            size_of::<u32>(),
        ];
        let control_bytes = controls
            .into_iter()
            .try_fold(size_of::<[usize; 7]>(), usize::checked_add)
            .ok_or(OriginalBufferCause::InvalidLayout)?;
        Ok(OriginalBufferPopulationLayout {
            capacity: raw.capacity,
            control_bytes,
        })
    }
    pub(crate) fn raw(&self) -> safemlx_sys::mlx_original_buffer_budget {
        self.raw
    }
    pub(crate) fn bind_to_scope(
        &self,
        scope: safemlx_sys::mlx_submission_scope,
    ) -> Result<(), OriginalBufferCause> {
        // SAFETY: only the closed Scope owner supplies this retained handle.
        // Native validates current role/parent before changing fixed fields.
        OriginalBufferCause::check(unsafe {
            safemlx_sys::mlx_original_buffer_budget_bind(scope, self.raw)
        })
    }
    /// Exact owner equality, not equality of advertised capacity.
    pub fn same_budget(&self, other: &Self) -> bool {
        self.raw.ctx == other.raw.ctx
    }
    /// Immutable physical allowance; no completion or admission evidence.
    pub fn capacity(&self) -> usize {
        // SAFETY: this handle retains the native immutable field.
        unsafe { safemlx_sys::mlx_original_buffer_budget_capacity(self.raw) }
    }
    /// Occupied native charge for diagnostics, never pool credit or completion.
    pub fn occupied_bytes(&self) -> usize {
        // SAFETY: the actual retained native counter is atomic.
        unsafe { safemlx_sys::mlx_original_buffer_budget_occupied(self.raw) }
    }
    /// Pure controls for the exact-budget source inspection below. The shared
    /// native query includes its runtime guard and fixed allocation facts.
    pub fn inspection_control_bytes() -> Option<usize> {
        OriginalBufferAliasWitness::inspection_control_bytes()?
            .checked_add(size_of::<OriginalBufferWitness<'_>>())?
            .checked_add(size_of::<Result<Option<OriginalBufferWitness<'_>>, OriginalBufferCause>>())?
            .checked_add(size_of::<&OriginalBufferBudget>())
    }
    /// Authenticate this budget's actual completed mutable birth. Unknown and
    /// ordinary storage stays None; a genuine foreign budget rejects. Does not
    /// evaluate, poll, run housekeeping or attach registration.
    pub fn inspect_array<'a>(
        &'a self,
        array: &'a Array,
    ) -> Result<Option<OriginalBufferWitness<'a>>, OriginalBufferCause> {
        let _loan =
            runtime_lock::try_enter_for_recovery().ok_or(OriginalBufferCause::RuntimeBusy)?;
        let mut facts = safemlx_sys::mlx_original_buffer_info {
            known: false,
            identity: 0,
            charged_bytes: 0,
        };
        // SAFETY: both closed handles remain live through the same immediate
        // runtime loan. The fixed C kernel neither allocates nor mutates Data.
        OriginalBufferCause::check(unsafe {
            safemlx_sys::mlx_original_buffer_array_info(&mut facts, array.as_ptr(), self.raw)
        })?;
        Ok(facts.known.then(|| OriginalBufferWitness {
            facts,
            _array: array,
            _budget: self,
        }))
    }
}

/// Borrowed authenticated facts, not a durable backing owner or registration.
/// Another alias may change a descriptor later. Publication still requires an
/// atomic expected-identity/capacity comparison at its actual attachment boundary.
pub struct OriginalBufferWitness<'a> {
    facts: safemlx_sys::mlx_original_buffer_info,
    _array: &'a Array,
    _budget: &'a OriginalBufferBudget,
}
impl fmt::Debug for OriginalBufferWitness<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OriginalBufferWitness")
            .field("allocation", &self.allocation())
            .finish_non_exhaustive()
    }
}
impl OriginalBufferWitness<'_> {
    /// Existing opaque allocation equality key and full charged capacity.
    pub fn allocation(&self) -> AllocationInfo {
        AllocationInfo::from_native(self.facts.identity, self.facts.charged_bytes, false)
    }
    /// Attach this prepared owner only if the same birth and originating budget
    /// still match in one immediate runtime/native observation. No rollback of
    /// earlier owners is implied; any refusal returns the exact preparation.
    pub fn try_attach<T: Send + 'static>(
        self,
        owner: PreparedAllocationOwner<T>,
    ) -> Result<(), OriginalBufferError<PreparedAllocationOwner<T>>> {
        owner.attach_original(self._array, self.facts, Some(self._budget.raw))
    }
}

/// Borrowed native-birth facts without retaining the originating safe budget.
/// This is not a neutral origin or permission to publish a new registry row.
/// A closed existing-canonical lookup must resolve its already published origin.
/// Attachment still rechecks the exact current birth in its own runtime loan.
pub struct OriginalBufferAliasWitness<'a> {
    facts: safemlx_sys::mlx_original_buffer_info,
    array: &'a Array,
}
impl fmt::Debug for OriginalBufferAliasWitness<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OriginalBufferAliasWitness")
            .field("allocation", &self.allocation())
            .finish_non_exhaustive()
    }
}
impl OriginalBufferAliasWitness<'_> {
    /// Opaque existing allocation key and full charged capacity, no budget handle.
    pub fn allocation(&self) -> AllocationInfo {
        AllocationInfo::from_native(self.facts.identity, self.facts.charged_bytes, false)
    }
    /// Named fixed inspection transports. No owner, backing, allocator charge or
    /// complete request fit is included; attachment uses the actual owner's layout.
    pub fn inspection_control_bytes() -> Option<usize> {
        // SAFETY: pure native sizeof query, with no runtime or pointer access.
        let native = unsafe { safemlx_sys::mlx_original_buffer_inspection_control_bytes() };
        let parts = [
            native,
            size_of::<Self>(),
            size_of::<AllocationInfo>(),
            size_of::<safemlx_sys::mlx_original_buffer_info>(),
            size_of::<Result<Option<Self>, OriginalBufferCause>>(),
            size_of::<runtime_lock::RuntimeLockGuard>(),
            size_of::<&Array>(),
        ];
        parts
            .into_iter()
            .try_fold(std::mem::size_of_val(&parts), usize::checked_add)
    }
    /// Recheck this exact settled mutable birth, then append the prepared owner
    /// to that same Data. Success supplies physical custody, not neutral authority.
    pub fn try_attach<T: Send + 'static>(
        self,
        owner: PreparedAllocationOwner<T>,
    ) -> Result<(), OriginalBufferError<PreparedAllocationOwner<T>>> {
        owner.attach_original(self.array, self.facts, None)
    }
}
impl Array {
    /// Inspect an existing settled original mutable birth without A's safe budget
    /// handle. Ordinary, custom and unfinished storage remains None. No evaluation,
    /// polling, owner attachment, housekeeping or allocation is performed.
    pub fn inspect_original_buffer_alias(
        &self,
    ) -> Result<Option<OriginalBufferAliasWitness<'_>>, OriginalBufferCause> {
        let _loan =
            runtime_lock::try_enter_for_recovery().ok_or(OriginalBufferCause::RuntimeBusy)?;
        let mut facts = safemlx_sys::mlx_original_buffer_info {
            known: false,
            identity: 0,
            charged_bytes: 0,
        };
        // SAFETY: the actual Array stays borrowed under one runtime loan; C
        // returns only checked immutable birth facts and no native owner.
        OriginalBufferCause::check(unsafe {
            safemlx_sys::mlx_original_buffer_array_alias_info(&mut facts, self.as_ptr())
        })?;
        Ok(facts
            .known
            .then_some(OriginalBufferAliasWitness { facts, array: self }))
    }
}

/// Positive ordinary-storage observation from the same completed descriptor
/// kernel used by original births. Unknown is never an ordinary storage proof.
#[derive(Debug)]
pub enum OrdinaryBufferInspection<'a> {
    /// No positive ordinary proof. Source-owned values need their own retained
    /// source contract; unfinished or custom storage remains unknown.
    Unknown,
    /// The descriptor is complete and has no physical backing allocation.
    /// In particular, an original CPU zero-length page is not empty backing.
    Empty,
    /// Actual ordinary default-allocator storage, with a checked attach path.
    Allocation(OrdinaryBufferWitness<'a>),
}

/// Borrowed positive default-allocator storage proof. No raw constructor, native
/// pointer or budget authority escapes. Another alias can change storage later,
/// so attachment must recheck this exact storage kind, generation and capacity.
pub struct OrdinaryBufferWitness<'a> {
    facts: safemlx_sys::mlx_original_buffer_info,
    array: &'a Array,
}
impl fmt::Debug for OrdinaryBufferWitness<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OrdinaryBufferWitness")
            .field("allocation", &self.allocation())
            .finish_non_exhaustive()
    }
}
impl OrdinaryBufferWitness<'_> {
    /// Actual opaque ordinary allocation key and existing full-capacity semantics.
    pub fn allocation(&self) -> AllocationInfo {
        AllocationInfo::from_native(self.facts.identity, self.facts.charged_bytes, false)
    }
    /// Append only if the same positive ordinary kind, generation and capacity
    /// still match. Refusal preserves the exact original preparation and owner.
    pub fn try_attach<T: Send + 'static>(
        self,
        owner: PreparedAllocationOwner<T>,
    ) -> Result<(), OriginalBufferError<PreparedAllocationOwner<T>>> {
        owner.attach_ordinary(self.array, self.facts)
    }
    /// Fixed query representations only; no backing, allocator or request fit.
    pub fn inspection_control_bytes() -> Option<usize> {
        // SAFETY: linked sizeof-only query; no pointer or runtime access.
        let native = unsafe { safemlx_sys::mlx_original_buffer_inspection_control_bytes() };
        let parts = [
            native,
            size_of::<Self>(),
            size_of::<OrdinaryBufferInspection<'static>>(),
            size_of::<Result<OrdinaryBufferInspection<'static>, OriginalBufferCause>>(),
            size_of::<safemlx_sys::mlx_original_buffer_info>(),
            size_of::<runtime_lock::RuntimeLockGuard>(),
            size_of::<u32>(),
            size_of::<&Array>(),
        ];
        parts
            .into_iter()
            .try_fold(std::mem::size_of_val(&parts), usize::checked_add)
    }
}
impl Array {
    /// Observe positive ordinary storage without evaluation, polling, allocation
    /// or housekeeping. Source/immutable-input/custom values cannot become an
    /// ordinary proof merely because an original-budget query returned None.
    pub fn inspect_ordinary_buffer(
        &self,
    ) -> Result<OrdinaryBufferInspection<'_>, OriginalBufferCause> {
        let _loan =
            runtime_lock::try_enter_for_recovery().ok_or(OriginalBufferCause::RuntimeBusy)?;
        let mut kind = safemlx_sys::MLX_ORDINARY_BUFFER_UNKNOWN;
        let mut facts = safemlx_sys::mlx_original_buffer_info {
            known: false,
            identity: 0,
            charged_bytes: 0,
        };
        // SAFETY: the borrowed Array remains live under this same immediate
        // runtime loan; the shared descriptor kernel returns only fixed facts.
        OriginalBufferCause::check(unsafe {
            safemlx_sys::mlx_ordinary_buffer_array_info(&mut kind, &mut facts, self.as_ptr())
        })?;
        match kind {
            safemlx_sys::MLX_ORDINARY_BUFFER_UNKNOWN
                if !facts.known && facts.identity == 0 && facts.charged_bytes == 0 =>
            {
                Ok(OrdinaryBufferInspection::Unknown)
            }
            safemlx_sys::MLX_ORDINARY_BUFFER_EMPTY
                if !facts.known && facts.identity == 0 && facts.charged_bytes == 0 =>
            {
                Ok(OrdinaryBufferInspection::Empty)
            }
            safemlx_sys::MLX_ORDINARY_BUFFER_ALLOCATION if facts.known && facts.identity != 0 => {
                Ok(OrdinaryBufferInspection::Allocation(
                    OrdinaryBufferWitness { facts, array: self },
                ))
            }
            _ => Err(OriginalBufferCause::InvalidStatus(kind)),
        }
    }
}

/// Positive immutable prepared-source observation. It grants no source origin:
/// only an existing prepaid-host canonical row can authorize alias publication.
#[derive(Debug)]
pub enum ImmutableSourceInspection<'a> {
    /// Unfinished, foreign source kind or uncertified Data.
    Unknown,
    /// Actual settled zero-copy Data with no native generation or allocation.
    Empty,
    /// Actual immutable Data generation and full allocator capacity.
    Allocation(ImmutableSourceWitness<'a>),
}

/// Borrowed immutable Data evidence, consumed by one checked attachment. It
/// cannot create a native budget, host receipt or neutral origin.
pub struct ImmutableSourceWitness<'a> {
    facts: safemlx_sys::mlx_original_buffer_info,
    array: &'a Array,
}
impl fmt::Debug for ImmutableSourceWitness<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ImmutableSourceWitness")
            .field("allocation", &self.allocation())
            .finish_non_exhaustive()
    }
}
impl ImmutableSourceWitness<'_> {
    /// Opaque actual allocation identity/capacity, without accounting authority.
    pub fn allocation(&self) -> AllocationInfo {
        AllocationInfo::from_native(self.facts.identity, self.facts.charged_bytes, false)
    }
    /// Compare source kind, generation and whole capacity in the same immediate
    /// native observation as append. Refusal retains both prepared nodes.
    pub fn try_attach<T: Send + 'static>(
        self,
        owner: PreparedAllocationOwner<T>,
    ) -> Result<(), OriginalBufferError<PreparedAllocationOwner<T>>> {
        owner.attach_immutable(self.array, self.facts)
    }
    /// Pure concrete inspection transport layout; no grant or backing bound.
    pub fn inspection_control_bytes() -> Option<usize> {
        // SAFETY: linked sizeof-only query without runtime activity.
        let native = unsafe { safemlx_sys::mlx_original_buffer_inspection_control_bytes() };
        let parts = [
            native,
            size_of::<Self>(),
            size_of::<AllocationInfo>(),
            size_of::<ImmutableSourceInspection<'static>>(),
            size_of::<Result<ImmutableSourceInspection<'static>, OriginalBufferCause>>(),
            size_of::<safemlx_sys::mlx_original_buffer_info>(),
            size_of::<runtime_lock::RuntimeLockGuard>(),
            size_of::<u32>(),
            size_of::<&Array>(),
        ];
        parts
            .into_iter()
            .try_fold(std::mem::size_of_val(&parts), usize::checked_add)
    }
}
impl Array {
    /// Observe settled immutable source Data without evaluation, polling or
    /// housekeeping. This does not certify how the source's backing was funded.
    pub fn inspect_immutable_source(
        &self,
    ) -> Result<ImmutableSourceInspection<'_>, OriginalBufferCause> {
        let _loan =
            runtime_lock::try_enter_for_recovery().ok_or(OriginalBufferCause::RuntimeBusy)?;
        let mut kind = safemlx_sys::MLX_ORDINARY_BUFFER_UNKNOWN;
        let mut facts = safemlx_sys::mlx_original_buffer_info {
            known: false,
            identity: 0,
            charged_bytes: 0,
        };
        // SAFETY: exact borrowed Array under one runtime loan. Only fixed
        // descriptive facts return; no payload/callback is consumed here.
        OriginalBufferCause::check(unsafe {
            safemlx_sys::mlx_immutable_source_array_info(&mut kind, &mut facts, self.as_ptr())
        })?;
        match kind {
            safemlx_sys::MLX_ORDINARY_BUFFER_UNKNOWN
                if !facts.known && facts.identity == 0 && facts.charged_bytes == 0 =>
            {
                Ok(ImmutableSourceInspection::Unknown)
            }
            safemlx_sys::MLX_ORDINARY_BUFFER_EMPTY
                if !facts.known && facts.identity == 0 && facts.charged_bytes == 0 =>
            {
                Ok(ImmutableSourceInspection::Empty)
            }
            safemlx_sys::MLX_ORDINARY_BUFFER_ALLOCATION if facts.known && facts.identity != 0 => {
                Ok(ImmutableSourceInspection::Allocation(
                    ImmutableSourceWitness { facts, array: self },
                ))
            }
            _ => Err(OriginalBufferCause::InvalidStatus(kind)),
        }
    }
}

/// Prepared payload-free custody for the actual allocator selected by runtime.
/// T must not retain the future budget, a native Array, Graph, Scope or Record.
/// The caller must reserve both capacity and actual owner costs before this call.
pub struct PreparedOriginalBufferBudget<'a, T: Send + 'static> {
    runtime: &'a PreparedInputRuntime,
    layout: OriginalBufferBudgetLayout,
    node: Option<Box<OwnedNode<T>>>,
}
impl<T: Send + 'static> fmt::Debug for PreparedOriginalBufferBudget<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreparedOriginalBufferBudget")
            .field("layout", &self.layout)
            .finish_non_exhaustive()
    }
}
impl<T: Send + 'static> Drop for PreparedOriginalBufferBudget<'_, T> {
    fn drop(&mut self) {
        if let Some(node) = self.node.take() {
            drop(take_owner(node));
        }
    }
}
impl<'a, T: Send + 'static> PreparedOriginalBufferBudget<'a, T> {
    /// Cold facts from the actual already prepared allocator; performs no native
    /// initialization, allocator call or runtime loan.
    pub fn layout(
        runtime: &PreparedInputRuntime,
        capacity: usize,
    ) -> Result<OriginalBufferBudgetLayout, OriginalBufferCause> {
        let mut native = safemlx_sys::mlx_original_buffer_layout {
            owner_bytes: 0,
            control_bytes: 0,
        };
        // SAFETY: pure sizeof query initializes valid output or returns fixed refusal.
        OriginalBufferCause::check(unsafe {
            safemlx_sys::mlx_original_buffer_layout_for(&mut native)
        })?;
        let parts = [
            native.control_bytes,
            runtime.raw().controls,
            size_of::<Self>(),
            size_of::<T>(),
            size_of::<OwnedNode<T>>(),
            size_of::<Layout>(),
            size_of::<*mut OwnedNode<T>>(),
            size_of::<Construction>(),
            size_of::<OriginalBufferBudget>(),
            size_of::<OriginalBufferWitness<'a>>(),
            size_of::<OriginalBufferAliasWitness<'a>>(),
            size_of::<safemlx_sys::mlx_original_buffer_info>(),
            size_of::<OriginalBufferCause>(),
            size_of::<Result<Self, OriginalBufferError<T>>>(),
            size_of::<Result<OriginalBufferBudget, OriginalBufferError<Self>>>(),
            size_of::<Result<Option<OriginalBufferWitness<'a>>, OriginalBufferCause>>(),
            size_of::<runtime_lock::RuntimeLockGuard>(),
            size_of::<*mut c_void>(),
            size_of::<Option<Box<OwnedNode<T>>>>(),
            size_of::<super::RetirementBatch>(),
            size_of::<*mut RetiredOwner>(),
            size_of::<unsafe fn(*mut RetiredOwner)>(),
        ];
        let control_bytes = parts
            .into_iter()
            .try_fold(std::mem::size_of_val(&parts), usize::checked_add)
            .ok_or(OriginalBufferCause::InvalidLayout)?;
        Ok(OriginalBufferBudgetLayout {
            capacity,
            native_owner_bytes: native.owner_bytes,
            rust_node_bytes: size_of::<OwnedNode<T>>(),
            control_bytes,
        })
    }
    /// Allocate only the existing retirement node. Failure returns unchanged T.
    pub fn try_new(
        runtime: &'a PreparedInputRuntime,
        capacity: usize,
        owner: T,
    ) -> Result<Self, OriginalBufferError<T>> {
        let layout = match Self::layout(runtime, capacity) {
            Ok(layout) => layout,
            Err(cause) => return Err(OriginalBufferError { cause, owner }),
        };
        // SAFETY: matching nonzero OwnedNode layout; initialized before Box creation.
        let node =
            unsafe { std::alloc::alloc(Layout::new::<OwnedNode<T>>()) }.cast::<OwnedNode<T>>();
        if node.is_null() {
            return Err(OriginalBufferError {
                cause: OriginalBufferCause::AllocationFailed,
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
            runtime,
            layout,
            node: Some(node),
        })
    }
    /// Borrow original custody without exposing its node or native owner.
    pub fn owner(&self) -> &T {
        &self
            .node
            .as_ref()
            .expect("unconsumed budget preparation")
            .owner
    }
    /// Cancel this unattached preparation; free its node before returning T.
    pub fn into_owner(mut self) -> T {
        take_owner(self.node.take().expect("unconsumed budget preparation"))
    }
    /// Construct once using the actual prepared allocator. Busy/refusal returns
    /// this same node. This ordinary setup may lock the prepared native allocator;
    /// it must precede the original role's operation and is not a worker entry.
    pub fn try_allocate(mut self) -> Result<OriginalBufferBudget, OriginalBufferError<Self>> {
        let mut controls = Construction {
            raw: safemlx_sys::mlx_original_buffer_budget {
                ctx: ptr::null_mut(),
            },
            status: 0,
        };
        {
            let Some(_loan) = runtime_lock::try_enter_for_recovery() else {
                return Err(OriginalBufferError {
                    cause: OriginalBufferCause::RuntimeBusy,
                    owner: self,
                });
            };
            let payload = (&mut **self.node.as_mut().expect("unconsumed budget preparation")
                as *mut OwnedNode<T>)
                .cast();
            // SAFETY: native success alone consumes this initialized exclusive
            // node. Refusal leaves output null and never invokes retirement.
            controls.status = unsafe {
                safemlx_sys::mlx_original_buffer_budget_new_retaining(
                    &mut controls.raw,
                    self.runtime.raw(),
                    self.layout.capacity,
                    payload,
                    Some(retire),
                )
            };
            if controls.status == 0 {
                let _ = Box::into_raw(self.node.take().expect("unconsumed budget preparation"));
            }
        }
        match OriginalBufferCause::check(controls.status) {
            Ok(()) => Ok(OriginalBufferBudget {
                raw: controls.raw,
                _thread: PhantomData,
            }),
            Err(cause) => Err(OriginalBufferError { cause, owner: self }),
        }
    }
}

#[cfg(test)]
mod tests;
