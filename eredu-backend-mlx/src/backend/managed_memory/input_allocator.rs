//! Shared allocator birth in the actual managed domain. No Device promotion.
use eredu_runtime::working_memory::{
    InitializedSharedNative, SharedNativeInitializationCustody, SharedNativeInitializationError,
    SharedNativeInitializer, WorkingMemoryError, WorkingMemoryPool,
};
use safemlx::{
    InitializedInputAllocator, InputAllocatorCause, InputAllocatorError, PreparedInputAllocator,
    PreparedInputRuntime,
};
use std::sync::{
    OnceLock,
    atomic::{AtomicBool, Ordering},
};

static INITIALIZED: OnceLock<InitializedSharedNative<InitializedInputAllocator>> = OnceLock::new();
static INITIALIZING: AtomicBool = AtomicBool::new(false);
pub(super) fn static_storage_bytes() -> usize {
    std::mem::size_of_val(&INITIALIZED) + std::mem::size_of_val(&INITIALIZING)
}
struct Winner;
impl Drop for Winner {
    fn drop(&mut self) {
        INITIALIZING.store(false, Ordering::Release);
    }
}

/// Only the allocator's dynamic object and owned constructor controls are
/// covered. The joined original entry also funds the eager Device/default
/// library; JIT, stream and context owners remain separate prerequisites.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputAllocatorCoverage {
    /// The ordinary API initialized/reused an unadmitted predecessor.
    Ordinary,
    /// The actual singleton retains its same once-admitted constructor account.
    Admitted {
        requires_device: bool,
        device_admitted: bool,
    },
}

#[derive(Debug)]
struct Initializer;
#[derive(Debug, thiserror::Error)]
enum ConstructorFailure {
    #[error("input allocator owner preparation: {0}")]
    Preparation(#[source] InputAllocatorError<SharedNativeInitializationCustody>),
    #[error("input allocator native constructor: {0}")]
    Native(#[source] InputAllocatorError<SharedNativeInitializationCustody>),
}
impl SharedNativeInitializer for Initializer {
    type Output = InitializedInputAllocator;
    type Error = ConstructorFailure;
    fn required_storage_bytes(&self) -> Result<usize, WorkingMemoryError> {
        let layout = PreparedInputAllocator::<SharedNativeInitializationCustody>::layout()
            .map_err(|_| WorkingMemoryError::UnknownBound)?;
        layout
            .required_bytes()
            .and_then(|bytes| bytes.checked_add(std::mem::size_of::<Winner>()))
            .and_then(|bytes| {
                bytes.checked_add(std::mem::size_of::<MlxInputAllocatorInitializationError>())
            })
            .ok_or(WorkingMemoryError::Overflow)
    }
    fn initialize(
        self,
        custody: SharedNativeInitializationCustody,
    ) -> Result<Self::Output, Self::Error> {
        PreparedInputAllocator::try_new(custody)
            .map_err(ConstructorFailure::Preparation)?
            .try_initialize()
            .map_err(|error| {
                ConstructorFailure::Native(error.map_owner(PreparedInputAllocator::into_owner))
            })
    }
}

/// Typed refusal retaining the exact native cause and original source custody.
/// Any unadopted preparation shell is freed before this error is exposed.
#[derive(Debug)]
pub struct MlxInputAllocatorInitializationError(Failure);
#[derive(Debug)]
enum Failure {
    Policy(WorkingMemoryError),
    Constructor(SharedNativeInitializationError<Initializer>),
    Native(InputAllocatorCause),
    Device(super::metal_device::MlxMetalDeviceInitializationError),
    Scheduler(super::scheduler::MlxSchedulerInitializationError),
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    Pointwise(super::pointwise_kernel::MlxPointwiseDefinitionError),
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    Row(super::row_kernels::MlxRowKernelError),
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    Recurrent(super::recurrent_kernel::MlxRecurrentKernelError),
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    Bf16Projection(super::bf16_projection_kernel::MlxBf16ProjectionError),
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    Fp8(crate::backend::nn::fp8::kernel::MlxFp8KernelError),
}
impl std::fmt::Display for MlxInputAllocatorInitializationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.0 {
            Failure::Policy(e) => e.fmt(f),
            Failure::Constructor(e) => e.fmt(f),
            Failure::Native(e) => e.fmt(f),
            Failure::Device(e) => e.fmt(f),
            Failure::Scheduler(e) => e.fmt(f),
            #[cfg(all(feature = "metal", not(feature = "cuda")))]
            Failure::Pointwise(e) => e.fmt(f),
            #[cfg(all(feature = "metal", not(feature = "cuda")))]
            Failure::Row(e) => e.fmt(f),
            #[cfg(all(feature = "metal", not(feature = "cuda")))]
            Failure::Recurrent(e) => e.fmt(f),
            #[cfg(all(feature = "metal", not(feature = "cuda")))]
            Failure::Bf16Projection(e) => e.fmt(f),
            #[cfg(all(feature = "metal", not(feature = "cuda")))]
            Failure::Fp8(e) => e.fmt(f),
        }
    }
}
impl std::error::Error for MlxInputAllocatorInitializationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.0 {
            Failure::Policy(e) => Some(e),
            Failure::Constructor(e) => Some(e),
            Failure::Native(e) => Some(e),
            Failure::Device(e) => Some(e),
            Failure::Scheduler(e) => Some(e),
            #[cfg(all(feature = "metal", not(feature = "cuda")))]
            Failure::Pointwise(e) => Some(e),
            #[cfg(all(feature = "metal", not(feature = "cuda")))]
            Failure::Row(e) => Some(e),
            #[cfg(all(feature = "metal", not(feature = "cuda")))]
            Failure::Recurrent(e) => Some(e),
            #[cfg(all(feature = "metal", not(feature = "cuda")))]
            Failure::Bf16Projection(e) => Some(e),
            #[cfg(all(feature = "metal", not(feature = "cuda")))]
            Failure::Fp8(e) => Some(e),
        }
    }
}
fn borrow(
    owner: &InitializedSharedNative<InitializedInputAllocator>,
    pool: &WorkingMemoryPool,
    device_admitted: bool,
) -> Result<(PreparedInputRuntime, InputAllocatorCoverage), MlxInputAllocatorInitializationError> {
    owner
        .validate_pool(pool)
        .map_err(|e| MlxInputAllocatorInitializationError(Failure::Policy(e)))?;
    let runtime = owner
        .output()
        .try_borrow_runtime()
        .map_err(|e| MlxInputAllocatorInitializationError(Failure::Native(e)))?;
    Ok((
        runtime,
        InputAllocatorCoverage::Admitted {
            requires_device: owner.output().requires_device(),
            device_admitted,
        },
    ))
}

/// Explicit original cold entry. The exact managed domain must already cover
/// its fixed baseline. The Device initializer shares this exact pool and runs
/// before the shared Scheduler, and both precede any allocator loan. Their
/// actual successful owners survive a later failure. Active ordinary work
/// still rejects through the existing Pool comparison; it is never bypassed.
pub(crate) fn prepare_admitted(
    pool: &WorkingMemoryPool,
) -> Result<(PreparedInputRuntime, InputAllocatorCoverage), MlxInputAllocatorInitializationError> {
    if !pool.same_domain(&super::domain()) {
        return Err(MlxInputAllocatorInitializationError(Failure::Policy(
            WorkingMemoryError::IdentityMismatch,
        )));
    }
    let device_admitted = super::metal_device::prepare_admitted(pool)
        .map_err(|error| MlxInputAllocatorInitializationError(Failure::Device(error)))?;
    super::scheduler::prepare_admitted(pool)
        .map_err(|error| MlxInputAllocatorInitializationError(Failure::Scheduler(error)))?;
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    super::pointwise_kernel::prepare_admitted(pool)
        .map_err(|error| MlxInputAllocatorInitializationError(Failure::Pointwise(error)))?;
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    super::row_kernels::prepare_admitted(pool)
        .map_err(|error| MlxInputAllocatorInitializationError(Failure::Row(error)))?;
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    super::recurrent_kernel::prepare_admitted(pool)
        .map_err(|error| MlxInputAllocatorInitializationError(Failure::Recurrent(error)))?;
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    super::bf16_projection_kernel::prepare_admitted(pool)
        .map_err(|error| MlxInputAllocatorInitializationError(Failure::Bf16Projection(error)))?;
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    crate::backend::nn::fp8::kernel::prepare_admitted(pool)
        .map_err(|error| MlxInputAllocatorInitializationError(Failure::Fp8(error)))?;
    if let Some(owner) = INITIALIZED.get() {
        return borrow(owner, pool, device_admitted);
    }
    if INITIALIZING
        .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
        .is_err()
    {
        return Err(MlxInputAllocatorInitializationError(Failure::Native(
            InputAllocatorCause::Busy,
        )));
    }
    let _winner = Winner;
    // Another winner may have published between the first read and our CAS.
    if let Some(owner) = INITIALIZED.get() {
        return borrow(owner, pool, device_admitted);
    }
    let owner = pool
        .initialize_shared_native(Initializer)
        .map_err(|e| MlxInputAllocatorInitializationError(Failure::Constructor(e)))?;
    // Exactly one writer can reach set; readers never wait on this OnceLock.
    // The native singleton has already retained its original account alias.
    INITIALIZED
        .set(owner)
        .expect("exclusive shared allocator initializer");
    borrow(
        INITIALIZED.get().expect("published initializer"),
        pool,
        device_admitted,
    )
}

/// Compatibility for callers already holding ordinary construction authority.
/// It retains ordinary initialization/housekeeping and never attempts admission
/// or promotes a predecessor. The immutable completed slot can only document a
/// real earlier admitted birth of this same global allocator.
pub(crate) fn prepare_ordinary()
-> Result<(PreparedInputRuntime, InputAllocatorCoverage), safemlx::error::Exception> {
    let runtime = PreparedInputRuntime::prepare()?;
    let coverage = match INITIALIZED.get() {
        Some(owner) => InputAllocatorCoverage::Admitted {
            requires_device: owner.output().requires_device(),
            device_admitted: super::metal_device::has_admitted_owner(),
        },
        None => InputAllocatorCoverage::Ordinary,
    };
    Ok((runtime, coverage))
}

impl MlxInputAllocatorInitializationError {
    pub(super) fn permits_ordinary_stream(&self) -> bool {
        match &self.0 {
            Failure::Device(error) => error.permits_ordinary_stream(),
            Failure::Scheduler(error) => error.permits_ordinary_stream(),
            Failure::Native(
                InputAllocatorCause::UnknownLayout | InputAllocatorCause::OrdinaryPredecessor,
            ) => true,
            Failure::Constructor(error) if error.accounting_failure().is_none() => {
                let cause = match error.constructor_failure() {
                    Some(ConstructorFailure::Preparation(error)) => Some(error.cause()),
                    Some(ConstructorFailure::Native(error)) => Some(error.cause()),
                    None => None,
                };
                matches!(
                    cause,
                    Some(
                        InputAllocatorCause::UnknownLayout
                            | InputAllocatorCause::OrdinaryPredecessor
                    )
                )
            }
            _ => false,
        }
    }
}

/// Borrow only the actual admitted allocator; never initialize under a live source constructor.
pub(crate) fn borrow_admitted(
    pool: &WorkingMemoryPool,
) -> Result<PreparedInputRuntime, WorkingMemoryError> {
    admitted_initializer(pool)?
        .try_borrow_runtime()
        .map_err(|_| WorkingMemoryError::UnknownBound)
}

/// Immutable admitted identity, after matching its source domain. A deferred
/// source slot can retain this process-owned identity without carrying the
/// current-thread runtime witness across execution boundaries. Actual borrowing
/// still authenticates the native singleton and never initializes a fallback.
pub(crate) fn admitted_initializer(
    pool: &WorkingMemoryPool,
) -> Result<&'static InitializedInputAllocator, WorkingMemoryError> {
    let owner = INITIALIZED.get().ok_or(WorkingMemoryError::UnknownBound)?;
    owner.validate_pool(pool)?;
    Ok(owner.output())
}
