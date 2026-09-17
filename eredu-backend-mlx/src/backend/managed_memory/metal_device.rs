//! One real embedded Device/default library owner in the same managed domain.
use eredu_runtime::working_memory::{
    InitializedSharedNative, SharedNativeInitializationCustody, SharedNativeInitializationError,
    SharedNativeInitializer, WorkingMemoryError, WorkingMemoryPool,
};
use safemlx::{
    InitializedMetalDevice, MetalDeviceCause, MetalDeviceError, MetalDeviceLayout,
    PreparedMetalDevice,
};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    OnceLock,
};

static INITIALIZED: OnceLock<InitializedSharedNative<InitializedMetalDevice>> = OnceLock::new();
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
struct Initializer(MetalDeviceLayout<SharedNativeInitializationCustody>);
impl Initializer {
    fn prepare() -> Result<Self, MetalDeviceCause> {
        PreparedMetalDevice::<SharedNativeInitializationCustody>::layout().map(Self)
    }
}
#[derive(Debug, thiserror::Error)]
enum ConstructorFailure {
    #[error("Device owner preparation: {0}")]
    Preparation(#[source] MetalDeviceError<SharedNativeInitializationCustody>),
    #[error("Device native constructor: {0}")]
    Native(#[source] MetalDeviceError<SharedNativeInitializationCustody>),
}
impl SharedNativeInitializer for Initializer {
    type Output = InitializedMetalDevice;
    type Error = ConstructorFailure;
    fn required_storage_bytes(&self) -> Result<usize, WorkingMemoryError> {
        self.0
            .required_bytes()
            .and_then(|n| n.checked_add(std::mem::size_of::<Winner>()))
            .and_then(|n| n.checked_add(std::mem::size_of::<MlxMetalDeviceInitializationError>()))
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
        PreparedMetalDevice::with_layout(self.0, custody)
            .map_err(ConstructorFailure::Preparation)?
            .try_initialize()
            .map_err(|error| {
                ConstructorFailure::Native(error.map_owner(PreparedMetalDevice::into_owner))
            })
    }
}
/// Typed refusal retaining the exact native cause and original source custody.
/// Any unadopted preparation shell is freed before this error is exposed.
#[derive(Debug)]
pub struct MlxMetalDeviceInitializationError(Failure);
#[derive(Debug)]
enum Failure {
    Policy(WorkingMemoryError),
    Constructor(SharedNativeInitializationError<Initializer>),
    Native(MetalDeviceCause),
}
impl std::fmt::Display for MlxMetalDeviceInitializationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.0 {
            Failure::Policy(e) => e.fmt(f),
            Failure::Constructor(e) => e.fmt(f),
            Failure::Native(e) => e.fmt(f),
        }
    }
}
impl std::error::Error for MlxMetalDeviceInitializationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.0 {
            Failure::Policy(e) => Some(e),
            Failure::Constructor(e) => Some(e),
            Failure::Native(e) => Some(e),
        }
    }
}
fn borrow(
    owner: &InitializedSharedNative<InitializedMetalDevice>,
    pool: &WorkingMemoryPool,
) -> Result<(), MlxMetalDeviceInitializationError> {
    owner
        .validate_pool(pool)
        .map_err(|e| MlxMetalDeviceInitializationError(Failure::Policy(e)))?;
    owner
        .output()
        .try_borrow()
        .map_err(|e| MlxMetalDeviceInitializationError(Failure::Native(e)))
}
/// Original cold entry; no native slot is inspected/created before the real pool
/// identity check. CPU requires no Device. Runtime loans end before the caller
/// begins input allocator construction. Later JIT/queue ownership is separate.
pub(super) fn prepare_admitted(
    pool: &WorkingMemoryPool,
) -> Result<bool, MlxMetalDeviceInitializationError> {
    if !pool.same_domain(&super::domain()) {
        return Err(MlxMetalDeviceInitializationError(Failure::Policy(
            WorkingMemoryError::IdentityMismatch,
        )));
    }
    if !safemlx::metal_device_static_layout().required {
        return Ok(false);
    }
    if let Some(owner) = INITIALIZED.get() {
        borrow(owner, pool)?;
        return Ok(true);
    }
    if INITIALIZING
        .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
        .is_err()
    {
        return Err(MlxMetalDeviceInitializationError(Failure::Native(
            MetalDeviceCause::Busy,
        )));
    }
    let _winner = Winner;
    if let Some(owner) = INITIALIZED.get() {
        borrow(owner, pool)?;
        return Ok(true);
    }
    let plan = Initializer::prepare()
        .map_err(|e| MlxMetalDeviceInitializationError(Failure::Native(e)))?;
    let owner = pool
        .initialize_shared_native(plan)
        .map_err(|e| MlxMetalDeviceInitializationError(Failure::Constructor(e)))?;
    INITIALIZED
        .set(owner)
        .expect("exclusive shared Device initializer");
    borrow(
        INITIALIZED.get().expect("published Device initializer"),
        pool,
    )?;
    Ok(true)
}
/// Diagnostic only: ordinary callers acquire no new Device authority here.
pub(super) fn has_admitted_owner() -> bool {
    INITIALIZED.get().is_some()
}

#[cfg(test)]
mod tests;

/// Borrow the actual same-pool Device constructor; never initialize or promote an ordinary slot.
pub(crate) fn admitted_owner(
    pool: &WorkingMemoryPool,
) -> Result<&'static InitializedMetalDevice, MlxMetalDeviceInitializationError> {
    let owner = INITIALIZED.get().ok_or_else(|| {
        MlxMetalDeviceInitializationError(Failure::Native(MetalDeviceCause::IdentityMismatch))
    })?;
    borrow(owner, pool)?;
    Ok(owner.output())
}

impl MlxMetalDeviceInitializationError {
    // Only absent qualification or an actual ordinary predecessor permits the
    // factory's unchanged ordinary creator. Budget/busy/allocation failures do not.
    pub(super) fn permits_ordinary_stream(&self) -> bool {
        let acceptable = |cause| {
            matches!(
                cause,
                MetalDeviceCause::UnknownLayout
                    | MetalDeviceCause::OrdinaryPredecessor
                    | MetalDeviceCause::UnqualifiedSource
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
