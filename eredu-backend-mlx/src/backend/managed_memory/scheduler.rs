//! One process Scheduler constructor charged to the actual managed domain.
use eredu_runtime::working_memory::{
    InitializedSharedNative, SharedNativeInitializationCustody, SharedNativeInitializationError,
    SharedNativeInitializer, WorkingMemoryError, WorkingMemoryPool,
};
use safemlx::{
    InitializedScheduler, PreparedScheduler, SchedulerCause, SchedulerError, SchedulerLayout,
};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    OnceLock,
};

static INITIALIZED: OnceLock<InitializedSharedNative<InitializedScheduler>> = OnceLock::new();
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
#[derive(Clone, Copy, Debug)]
struct Initializer(SchedulerLayout<SharedNativeInitializationCustody>);
impl Initializer {
    fn prepare() -> Result<Self, SchedulerCause> {
        PreparedScheduler::<SharedNativeInitializationCustody>::layout().map(Self)
    }
}
#[derive(Debug, thiserror::Error)]
enum ConstructorFailure {
    #[error("Scheduler owner preparation: {0}")]
    Preparation(#[source] SchedulerError<SharedNativeInitializationCustody>),
    #[error("Scheduler native constructor: {0}")]
    Native(#[source] SchedulerError<SharedNativeInitializationCustody>),
}
impl SharedNativeInitializer for Initializer {
    type Output = InitializedScheduler;
    type Error = ConstructorFailure;
    fn required_storage_bytes(&self) -> Result<usize, WorkingMemoryError> {
        self.0
            .required_bytes()
            .and_then(|n| n.checked_add(std::mem::size_of::<Winner>()))
            .and_then(|n| n.checked_add(std::mem::size_of::<MlxSchedulerInitializationError>()))
            .and_then(|n| {
                n.checked_add(std::mem::size_of::<
                    super::input_allocator::MlxInputAllocatorInitializationError,
                >())
            })
            .ok_or(WorkingMemoryError::Overflow)
    }
    fn initialize(
        self,
        custody: SharedNativeInitializationCustody,
    ) -> Result<Self::Output, Self::Error> {
        PreparedScheduler::with_layout(self.0, custody)
            .map_err(ConstructorFailure::Preparation)?
            .try_initialize()
            .map_err(|error| {
                ConstructorFailure::Native(error.map_owner(PreparedScheduler::into_owner))
            })
    }
}
/// Typed refusal retaining the exact native cause and original source custody.
/// Any unadopted preparation shell is freed before this error is exposed.
#[derive(Debug)]
pub struct MlxSchedulerInitializationError(Failure);
#[derive(Debug)]
enum Failure {
    Policy(WorkingMemoryError),
    Constructor(SharedNativeInitializationError<Initializer>),
    Native(SchedulerCause),
}
impl std::fmt::Display for MlxSchedulerInitializationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.0 {
            Failure::Policy(e) => e.fmt(f),
            Failure::Constructor(e) => e.fmt(f),
            Failure::Native(e) => e.fmt(f),
        }
    }
}
impl std::error::Error for MlxSchedulerInitializationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.0 {
            Failure::Policy(e) => Some(e),
            Failure::Constructor(e) => Some(e),
            Failure::Native(e) => Some(e),
        }
    }
}
fn borrow(
    owner: &InitializedSharedNative<InitializedScheduler>,
    pool: &WorkingMemoryPool,
) -> Result<(), MlxSchedulerInitializationError> {
    owner
        .validate_pool(pool)
        .map_err(|e| MlxSchedulerInitializationError(Failure::Policy(e)))?;
    owner
        .output()
        .try_borrow()
        .map_err(|e| MlxSchedulerInitializationError(Failure::Native(e)))
}
/// Original cold entry; the exact pool identity is checked before any native
/// inspection. No worker/stream is initialized and no ordinary fallback runs.
/// Native success retains its account permanently, including after a later
/// allocator/source stage fails. Repeated borrowers add no constructor charge.
pub(super) fn prepare_admitted(
    pool: &WorkingMemoryPool,
) -> Result<(), MlxSchedulerInitializationError> {
    if !pool.same_domain(&super::domain()) {
        return Err(MlxSchedulerInitializationError(Failure::Policy(
            WorkingMemoryError::IdentityMismatch,
        )));
    }
    if let Some(owner) = INITIALIZED.get() {
        return borrow(owner, pool);
    }
    if INITIALIZING
        .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
        .is_err()
    {
        return Err(MlxSchedulerInitializationError(Failure::Native(
            SchedulerCause::Busy,
        )));
    }
    let _winner = Winner;
    if let Some(owner) = INITIALIZED.get() {
        return borrow(owner, pool);
    }
    let plan =
        Initializer::prepare().map_err(|e| MlxSchedulerInitializationError(Failure::Native(e)))?;
    let owner = pool
        .initialize_shared_native(plan)
        .map_err(|e| MlxSchedulerInitializationError(Failure::Constructor(e)))?;
    INITIALIZED
        .set(owner)
        .expect("exclusive shared Scheduler initializer");
    borrow(
        INITIALIZED.get().expect("published Scheduler initializer"),
        pool,
    )
}

// Test-only observation of the actual retained account; no raw native identity
// or allocation authority is exposed.
#[cfg(test)]
pub(super) fn initialized_original_bytes(pool: &WorkingMemoryPool) -> Option<u64> {
    let owner = INITIALIZED.get()?;
    owner.validate_pool(pool).ok()?;
    Some(owner.original_bytes())
}
#[cfg(test)]
mod tests;

/// Borrow only an already admitted process Scheduler from the exact source pool.
/// This never creates a singleton or promotes an ordinary predecessor.
pub(crate) fn admitted_owner(
    pool: &WorkingMemoryPool,
) -> Result<&'static InitializedScheduler, MlxSchedulerInitializationError> {
    let owner = INITIALIZED.get().ok_or_else(|| {
        MlxSchedulerInitializationError(Failure::Native(SchedulerCause::IdentityMismatch))
    })?;
    borrow(owner, pool)?;
    Ok(owner.output())
}

impl MlxSchedulerInitializationError {
    // Only absent qualification or an actual ordinary predecessor permits the
    // factory's unchanged ordinary creator. Budget/busy/allocation failures do not.
    pub(super) fn permits_ordinary_stream(&self) -> bool {
        let acceptable = |cause| {
            matches!(
                cause,
                SchedulerCause::UnknownLayout | SchedulerCause::OrdinaryPredecessor
            )
        };
        match &self.0 {
            Failure::Native(cause) => acceptable(*cause),
            Failure::Constructor(error) if error.accounting_failure().is_none() => {
                match error.constructor_failure() {
                    Some(ConstructorFailure::Preparation(error)) => acceptable(error.cause()),
                    Some(ConstructorFailure::Native(error)) => acceptable(error.cause()),
                    None => false,
                }
            }
            _ => false,
        }
    }
}
