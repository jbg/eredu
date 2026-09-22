//! One admitted CPU stream used by the shared router's value-only tie branch.
//! Both execution modes borrow this source. Its actual stream and worker are
//! constructed during load readiness, before an admitted numerical invocation.
use super::{ledger, scheduler};
use crate::backend::runtime::checkpoint::store::{
    MaterializationSourceStreamError, MaterializationSourceWorkerError,
    PreparedMaterializationSourceStream, PreparedMaterializationSourceWorker,
    PreparedMaterializationStreamError, prepare_one_materialization_stream,
};
use eredu_runtime::working_memory::{
    MemoryLedger, SharedNativeInitializationCustody, WorkingMemoryError,
};
use safemlx::{
    OriginalScopeObserver, PreparedStreamCopy, ScopedSubmissionProgress, Stream, error::Exception,
};
use std::sync::{
    OnceLock,
    atomic::{AtomicBool, Ordering},
};

struct Runtime {
    stream: PreparedStreamCopy<SharedNativeInitializationCustody>,
    worker: PreparedMaterializationSourceWorker,
    _host: eredu_core::HostPreparationAuthority,
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
pub(crate) enum PreparationError {
    #[error("router source scheduler: {0}")]
    Scheduler(#[from] super::scheduler::MlxSchedulerInitializationError),
    #[error("router source metadata: {0}")]
    Metadata(#[from] eredu_core::HostMetadataFundingError),
    #[error("router source accounting: {0}")]
    Accounting(#[from] WorkingMemoryError),
    #[error("router source initialization is busy")]
    Busy,
    #[error("router source initialization previously failed")]
    Failed,
    #[error("router source stream: {0}")]
    Stream(#[from] MaterializationSourceStreamError),
    #[error("router source worker: {0}")]
    Worker(#[from] MaterializationSourceWorkerError),
    #[error("router immutable stream copy: {0}")]
    Copy(#[from] PreparedMaterializationStreamError),
}
#[derive(Debug, thiserror::Error)]
enum ReadinessError {
    #[error("router source initialization is busy")]
    Busy,
    #[error("router source is unavailable after failed initialization")]
    Failed,
}
impl PreparationError {
    pub(crate) fn into_exception(self) -> Exception {
        use eredu_core::BackendFailure;
        let source = match self {
            Self::Stream(error) => error.into_backend_failure(),
            Self::Worker(error) => error.into_backend_failure(),
            Self::Scheduler(error) => BackendFailure::from_error(error),
            Self::Metadata(error) => BackendFailure::from_error(error),
            Self::Accounting(error) => BackendFailure::from_error(error),
            Self::Copy(error) => BackendFailure::from_error(error),
            Self::Busy => BackendFailure::from_error(ReadinessError::Busy),
            Self::Failed => BackendFailure::from_error(ReadinessError::Failed),
        };
        Exception::from_source(source)
    }
}
fn prepare(pool: &MemoryLedger) -> Result<Runtime, PreparationError> {
    let frames = [
        std::mem::size_of::<Runtime>(),
        std::mem::size_of::<Preparing>(),
        std::mem::size_of::<PreparedMaterializationSourceStream>(),
        std::mem::size_of::<Result<Runtime, PreparationError>>(),
        std::mem::size_of::<Result<(), PreparationError>>(),
        std::mem::size_of::<MemoryLedger>(),
        std::mem::size_of::<&MemoryLedger>(),
    ];
    let bytes = frames
        .into_iter()
        .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
        .ok_or(WorkingMemoryError::Overflow)?;
    let metadata = pool.prepare_storage_metadata()?;
    let host = metadata.prepare_host_owner(bytes)?;
    let source = PreparedMaterializationSourceStream::prepare(pool)?;
    // Finish every fallible wrapper allocation before launching the worker.
    let stream = prepare_one_materialization_stream(pool, source.as_stream())?;
    let worker = source.prepare_cpu_worker(pool)?;
    // The registry and worker retain their real process birth after this
    // registration wrapper retires. The copied immutable wrapper stays shared.
    Ok(Runtime {
        stream,
        worker,
        _host: host,
    })
}

/// The cold prepared factory may attempt this extra source before module
/// construction. A selector requiring it later receives the typed failure.
pub(crate) fn prepare_before_native_construction() {
    if READY.get().is_some()
        || FAILED.load(Ordering::Acquire)
        || !matches!(OriginalScopeObserver::try_current(), Ok(None))
    {
        return;
    }
    let pool = ledger();
    if scheduler::admitted_owner(&pool).is_ok() {
        let _ = prepare_for_routing(&pool);
    }
}

/// Actual load-time preparation of the shared fallback. Existing native
/// predecessors retain their real qualification refusal; no invocation can
/// substitute a byte allowance for scheduler, stream, or worker provenance.
pub(crate) fn prepare_for_routing(pool: &MemoryLedger) -> Result<(), PreparationError> {
    if let Some(runtime) = READY.get() {
        runtime.stream.owner().validate_pool(pool)?;
        runtime.worker.validate_pool(pool)?;
        return Ok(());
    }
    if FAILED.load(Ordering::Acquire) {
        return Err(PreparationError::Failed);
    }
    scheduler::prepare_admitted(pool)?;
    if PREPARING
        .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
        .is_err()
    {
        return Err(PreparationError::Busy);
    }
    let _preparing = Preparing;
    if READY.get().is_none() {
        match prepare(pool) {
            Ok(runtime) => {
                let _ = READY.set(runtime);
            }
            Err(error) => {
                FAILED.store(true, Ordering::Release);
                return Err(error);
            }
        }
    }
    Ok(())
}

/// Borrow the actual paid source during ordinary routing. This grants no
/// original graph authority and creates no per-tie stream or worker.
pub(crate) fn ordinary_stream() -> Result<&'static Stream, Exception> {
    let runtime = READY
        .get()
        .ok_or_else(|| Exception::from_source(ReadinessError::Failed))?;
    let pool = ledger();
    runtime
        .stream
        .owner()
        .validate_pool(&pool)
        .map_err(Exception::from_source)?;
    runtime
        .worker
        .validate_pool(&pool)
        .map_err(|error| Exception::from_source(error.into_backend_failure()))?;
    Ok(runtime.stream.as_stream())
}

/// Fixed calls made when the ordinary completed predicate borrows the retained
/// CPU stream. The stream, worker and their original account remain process-owned.
pub(crate) fn ordinary_control_bytes() -> Option<usize> {
    use std::mem::{size_of, size_of_val};
    let frames = [
        size_of::<&Runtime>(),
        size_of::<MemoryLedger>(),
        size_of::<Result<&'static Stream, Exception>>(),
        size_of::<Result<(), WorkingMemoryError>>(),
        size_of::<Result<(), MaterializationSourceWorkerError>>(),
        size_of::<ReadinessError>(),
    ];
    frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)
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
    let pool = ledger();
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
        size_of::<MemoryLedger>(),
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

#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
mod tests {
    use super::*;
    #[test]
    fn ordinary_scheduler_predecessor_cannot_be_promoted_into_paid_router_custody() {
        if !crate::tests::support::native_process::enter("router-ordinary-predecessor") {
            return;
        }
        let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let input = safemlx::Array::from_slice(&[1.0f32, 2.0], &[2]);
        let output = input.add(&input, &stream).unwrap();
        output.evaluated().unwrap();
        let pool = ledger();
        let before = pool.snapshot().unwrap();
        let cause = prepare_for_routing(&pool).unwrap_err();
        let mut error: &(dyn std::error::Error + 'static) = &cause;
        let mut predecessor = false;
        loop {
            predecessor |= error.downcast_ref::<safemlx::SchedulerCause>()
                == Some(&safemlx::SchedulerCause::OrdinaryPredecessor);
            let Some(source) = error.source() else {
                break;
            };
            error = source;
        }
        assert!(predecessor, "{cause:?}");
        assert!(!is_ready());
        drop(cause);
        safemlx::reclaim_allocation_owners();
        let after = pool.snapshot().unwrap();
        assert_eq!(after.reservations, before.reservations);
        assert_eq!(after.funding_accounts, before.funding_accounts);
        for (after, before) in after.domains.iter().zip(&before.domains) {
            assert_eq!(after.current_charge_bytes, before.current_charge_bytes);
            assert!(after.historical_peak_bytes >= before.historical_peak_bytes);
        }
    }
}
