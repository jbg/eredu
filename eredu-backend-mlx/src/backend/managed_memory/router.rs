//! One admitted CPU stream used by the shared router's value-only tie branch.
//! Ordinary routing retains its existing per-tie stream path. This cached owner
//! is constructed before native module construction or at an ordinary selector boundary, never by
//! a cold quote or an original graph operation.
use super::{domain, scheduler};
use crate::backend::runtime::checkpoint::store::{
    prepare_one_materialization_stream, MaterializationSourceStreamError,
    MaterializationSourceWorkerError, PreparedMaterializationSourceStream,
    PreparedMaterializationSourceWorker, PreparedMaterializationStreamError,
};
use eredu_runtime::working_memory::{SharedNativeInitializationCustody, WorkingMemoryPool};
use safemlx::{
    error::Exception, OriginalScopeObserver, PreparedStreamCopy, ScopedSubmissionProgress, Stream,
};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    OnceLock,
};

struct Runtime {
    stream: PreparedStreamCopy<SharedNativeInitializationCustody>,
    worker: PreparedMaterializationSourceWorker,
}
static READY: OnceLock<Runtime> = OnceLock::new();
static PREPARING: AtomicBool = AtomicBool::new(false);
static FAILED: AtomicBool = AtomicBool::new(false);
pub(super) fn static_storage_bytes() -> usize {
    std::mem::size_of_val(&READY)
        + std::mem::size_of_val(&PREPARING)
        + std::mem::size_of_val(&FAILED)
}
struct Preparing;
impl Drop for Preparing {
    fn drop(&mut self) {
        PREPARING.store(false, Ordering::Release);
    }
}
#[derive(Debug, thiserror::Error)]
enum PreparationError {
    #[error("router source stream: {0}")]
    Stream(#[from] MaterializationSourceStreamError),
    #[error("router source worker: {0}")]
    Worker(#[from] MaterializationSourceWorkerError),
    #[error("router immutable stream copy: {0}")]
    Copy(#[from] PreparedMaterializationStreamError),
}
fn prepare(pool: &WorkingMemoryPool) -> Result<Runtime, PreparationError> {
    let source = PreparedMaterializationSourceStream::prepare(pool)?;
    // Finish every fallible wrapper allocation before launching the worker.
    let stream = prepare_one_materialization_stream(pool, source.as_stream())?;
    let worker = source.prepare_cpu_worker(pool)?;
    // The registry and worker retain their real process birth after this
    // registration wrapper retires. The copied immutable wrapper stays shared.
    Ok(Runtime { stream, worker })
}

/// Opportunistic at cold factory/runtime or ordinary selector construction. Lack of original
/// authority or a failed extra constructor must not change ordinary behavior;
/// later original quotation sees no ready owner and reports unknown. Failed
/// prefixes retain/release their exact existing initialization custody.
pub(crate) fn prepare_before_native_construction() {
    if READY.get().is_some()
        || FAILED.load(Ordering::Acquire)
        || !matches!(OriginalScopeObserver::try_current(), Ok(None))
    {
        return;
    }
    let pool = domain();
    if scheduler::admitted_owner(&pool).is_err()
        || PREPARING
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
    {
        return;
    }
    let _preparing = Preparing;
    if READY.get().is_none() {
        match prepare(&pool) {
            Ok(runtime) => {
                let _ = READY.set(runtime);
            }
            Err(_) => {
                // Registration is process-lived even if a later constructor
                // refuses. Do not repeatedly manufacture new failed prefixes.
                FAILED.store(true, Ordering::Release);
            }
        }
    }
}

/// Side-effect-free producer readiness: no lock, source, native initialization
/// or allocator observation. The process-owned stream cannot be replaced.
pub(crate) fn is_ready() -> bool {
    READY.get().is_some()
}

/// Actual worker/source authentication precedes graph construction. No new
/// stream or wrapper is allocated on this path; the Scope supplies only its
/// fixed refusal carrier, not retroactive worker funding.
pub(crate) fn stream(observer: &OriginalScopeObserver) -> Result<&'static Stream, Exception> {
    let unavailable = || {
        observer
            .observation_error(ScopedSubmissionProgress::Unobservable)
            .expect("fixed unobservable refusal")
    };
    let runtime = READY.get().ok_or_else(unavailable)?;
    let pool = domain();
    runtime
        .stream
        .owner()
        .validate_pool(&pool)
        .map_err(|_| unavailable())?;
    runtime.worker.validate_pool(&pool).map_err(|error| {
        if error.is_busy() {
            observer
                .observation_error(ScopedSubmissionProgress::Busy)
                .expect("fixed busy refusal")
        } else {
            unavailable()
        }
    })?;
    Ok(runtime.stream.as_stream())
}

/// Actual borrowed runtime/query controls; permanent owner bodies are already
/// retained by their independent shared initialization accounts.
pub(crate) fn control_bytes() -> Option<usize> {
    use std::mem::size_of;
    [
        size_of::<&Runtime>(),
        size_of::<WorkingMemoryPool>(),
        size_of::<&OriginalScopeObserver>(),
        size_of::<Result<&'static Stream, Exception>>(),
        size_of::<Result<(), MaterializationSourceWorkerError>>(),
        size_of::<Result<(), eredu_runtime::working_memory::WorkingMemoryError>>(),
        size_of::<ScopedSubmissionProgress>(),
    ]
    .into_iter()
    .try_fold(Stream::device_type_control_bytes()?, usize::checked_add)?
    .checked_add(OriginalScopeObserver::control_bytes()?)
}
