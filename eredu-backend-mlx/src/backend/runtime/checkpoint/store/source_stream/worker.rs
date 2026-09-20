//! One admitted process CPU worker for the exact registered source stream.
use super::{MaterializationSourceStreamError, PreparedMaterializationSourceStream};
use crate::backend::managed_memory::scheduler::{self, MlxSchedulerInitializationError};
use eredu_runtime::working_memory::{
    InitializedSharedNative, SharedNativeInitializationCustody, SharedNativeInitializationError,
    SharedNativeInitializer, WorkingMemoryError, WorkingMemoryPool,
};
use safemlx::{
    CpuWorkerCause, CpuWorkerError, CpuWorkerLayout, CpuWorkerTarget, InitializedCpuWorker,
    PreparedCpuWorker,
};
use std::mem::{size_of, size_of_val};

#[derive(Debug)]
struct Initializer {
    layout: CpuWorkerLayout<SharedNativeInitializationCustody>,
    target: CpuWorkerTarget,
}
#[derive(Debug, thiserror::Error)]
enum ConstructorFailure {
    #[error("source worker owner preparation: {0}")]
    Preparation(#[source] CpuWorkerError<(CpuWorkerTarget, SharedNativeInitializationCustody)>),
    #[error("source worker native startup: {0}")]
    Native(#[source] CpuWorkerError<PreparedCpuWorker<SharedNativeInitializationCustody>>),
}
impl SharedNativeInitializer for Initializer {
    type Output = InitializedCpuWorker;
    type Error = ConstructorFailure;
    fn required_storage_bytes(&self) -> Result<usize, WorkingMemoryError> {
        requested(self.layout)
    }
    fn initialize(
        self,
        custody: SharedNativeInitializationCustody,
    ) -> Result<Self::Output, Self::Error> {
        PreparedCpuWorker::with_layout(self.layout, self.target, custody)
            .map_err(ConstructorFailure::Preparation)?
            .try_initialize()
            .map_err(ConstructorFailure::Native)
    }
}
fn requested(
    layout: CpuWorkerLayout<SharedNativeInitializationCustody>,
) -> Result<usize, WorkingMemoryError> {
    let controls = [
        size_of::<&WorkingMemoryPool>(),
        size_of::<&PreparedMaterializationSourceStream>(),
        size_of::<&safemlx::InitializedScheduler>(),
        size_of::<PreparedMaterializationSourceWorker>(),
        size_of::<MaterializationSourceWorkerError>(),
        size_of::<Initializer>(),
        size_of::<Result<InitializedCpuWorker, ConstructorFailure>>(),
        size_of::<Result<CpuWorkerTarget, CpuWorkerCause>>(),
        size_of::<Result<PreparedMaterializationSourceWorker, MaterializationSourceWorkerError>>(),
    ];
    controls
        .into_iter()
        .try_fold(size_of_val(&controls), usize::checked_add)
        .and_then(|n| layout.required_bytes()?.checked_add(n))
        .ok_or(WorkingMemoryError::Overflow)
}
/// Typed source/refusal with the exact failed startup owner retained as its source.
#[derive(Debug, thiserror::Error)]
#[error(transparent)]
pub struct MaterializationSourceWorkerError(Failure);
impl MaterializationSourceWorkerError {
    pub(crate) fn into_backend_failure(self) -> eredu_core::BackendFailure {
        use eredu_core::BackendFailure;
        match self.0 {
            Failure::Native(error) => BackendFailure::from_error(error),
            Failure::Accounting(error) => BackendFailure::from_error(error),
            Failure::Stream(error) => error.into_backend_failure(),
            Failure::Scheduler(error) => BackendFailure::from_error(error),
            Failure::Initialization(error) => BackendFailure::from_error(
                error.retire_output_and_map_error(|error| match error {
                    ConstructorFailure::Preparation(error) => error.cause(),
                    ConstructorFailure::Native(error) => error.cause(),
                }),
            ),
        }
    }
    pub(crate) fn is_busy(&self) -> bool {
        matches!(self.0, Failure::Native(CpuWorkerCause::Busy))
    }
}

#[derive(Debug, thiserror::Error)]
enum Failure {
    #[error("source worker producer: {0}")]
    Native(#[source] CpuWorkerCause),
    #[error("source worker accounting: {0}")]
    Accounting(#[source] WorkingMemoryError),
    #[error("source worker registration: {0}")]
    Stream(#[source] MaterializationSourceStreamError),
    #[error("source worker process Scheduler: {0}")]
    Scheduler(#[source] MlxSchedulerInitializationError),
    #[error("source worker admission/constructor: {0}")]
    Initialization(#[source] SharedNativeInitializationError<Initializer>),
}
/// Birth custody for one actual source worker. Task/FIFO storage, mutable error
/// diagnostics, completion and whole execution fit remain independent contracts.
#[derive(Debug)]
pub struct PreparedMaterializationSourceWorker(InitializedSharedNative<InitializedCpuWorker>);
impl PreparedMaterializationSourceWorker {
    pub(crate) fn worker_owner(&self) -> &InitializedSharedNative<InitializedCpuWorker> {
        &self.0
    }

    /// Authenticate the same pool and actual process worker. This does not certify
    /// its health, dispatch authority or capacity for subsequent work.
    pub fn validate_pool(
        &self,
        pool: &WorkingMemoryPool,
    ) -> Result<(), MaterializationSourceWorkerError> {
        self.0
            .validate_pool(pool)
            .map_err(|e| MaterializationSourceWorkerError(Failure::Accounting(e)))?;
        self.0
            .output()
            .try_borrow()
            .map_err(|e| MaterializationSourceWorkerError(Failure::Native(e)))
    }
}
impl PreparedMaterializationSourceStream {
    /// Query/admit/start one worker on this exact source stream. Both Scheduler
    /// and registration must already have the same pool origin. An ordinary
    /// worker is explicitly refused rather than being charged retroactively.
    pub fn prepare_cpu_worker(
        &self,
        pool: &WorkingMemoryPool,
    ) -> Result<PreparedMaterializationSourceWorker, MaterializationSourceWorkerError> {
        self.validate_pool(pool)
            .map_err(|e| MaterializationSourceWorkerError(Failure::Stream(e)))?;
        let scheduler = scheduler::admitted_owner(pool)
            .map_err(|e| MaterializationSourceWorkerError(Failure::Scheduler(e)))?;
        let layout = PreparedCpuWorker::<SharedNativeInitializationCustody>::layout()
            .map_err(|e| MaterializationSourceWorkerError(Failure::Native(e)))?;
        let target = CpuWorkerTarget::for_stream(scheduler, self.0.output())
            .map_err(|e| MaterializationSourceWorkerError(Failure::Native(e)))?;
        pool.initialize_shared_native(Initializer { layout, target })
            .map(PreparedMaterializationSourceWorker)
            .map_err(|e| MaterializationSourceWorkerError(Failure::Initialization(e)))
    }
    /// Exact dynamic startup/account quote for this existing registered source.
    /// This captures no new source allowance and performs no worker construction.
    pub fn cpu_worker_required_bytes(
        &self,
        pool: &WorkingMemoryPool,
    ) -> Result<u64, MaterializationSourceWorkerError> {
        self.validate_pool(pool)
            .map_err(|e| MaterializationSourceWorkerError(Failure::Stream(e)))?;
        let scheduler = scheduler::admitted_owner(pool)
            .map_err(|e| MaterializationSourceWorkerError(Failure::Scheduler(e)))?;
        let layout = PreparedCpuWorker::<SharedNativeInitializationCustody>::layout()
            .map_err(|e| MaterializationSourceWorkerError(Failure::Native(e)))?;
        let target = CpuWorkerTarget::for_stream(scheduler, self.0.output())
            .map_err(|e| MaterializationSourceWorkerError(Failure::Native(e)))?;
        WorkingMemoryPool::shared_native_initialization_required_bytes(&Initializer {
            layout,
            target,
        })
        .map_err(|e| MaterializationSourceWorkerError(Failure::Accounting(e)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn source_worker_uses_same_pool_and_preserves_birth_after_wrapper_retirement() {
        const CHILD: &str = "EREDU_SOURCE_WORKER_COMPONENT_CHILD";
        if std::env::var_os(CHILD).is_none() {
            let result = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    concat!(
                        "backend::runtime::checkpoint::store::source_stream::worker::tests::",
                        "source_worker_uses_same_pool_and_preserves_birth_after_wrapper_retirement"
                    ),
                    "--nocapture",
                ])
                .env(CHILD, "1")
                .output()
                .unwrap();
            let text = String::from_utf8_lossy(&result.stdout);
            assert!(
                result.status.success(),
                "{text}\n{}",
                String::from_utf8_lossy(&result.stderr)
            );
            assert!(text.contains("SOURCE_WORKER_COMPONENT_OK"), "{text}");
            return;
        }
        let required = std::env::var_os("EREDU_REQUIRE_CPU_WORKER_STARTUP_QUALIFICATION").is_some();
        let layout = PreparedCpuWorker::<SharedNativeInitializationCustody>::layout();
        if required {
            assert!(layout.is_ok(), "{layout:?}");
        }
        if matches!(layout, Err(CpuWorkerCause::UnknownLayout)) {
            println!("SOURCE_WORKER_COMPONENT_OK: typed unknown producer");
            return;
        }
        layout.unwrap();
        let registration = PreparedMaterializationSourceStream::required_bytes();
        if required {
            assert!(registration.is_ok(), "{registration:?}");
        }
        match registration {
            Ok(_) => {}
            Err(super::super::MaterializationSourceStreamError(
                super::super::Failure::Layout(safemlx::StreamRegistrationCause::UnknownLayout)
                | super::super::Failure::Accounting(WorkingMemoryError::UnknownBound),
            )) => {
                assert!(!required);
                println!("SOURCE_WORKER_COMPONENT_OK: typed unknown source qualification");
                return;
            }
            Err(other) => panic!("unexpected source preparation failure: {other:?}"),
        }
        let pool = crate::backend::managed_memory::domain();
        crate::backend::managed_memory::input_allocator::prepare_admitted(&pool).unwrap();
        let source = PreparedMaterializationSourceStream::prepare(&pool).unwrap();
        let quote = source.cpu_worker_required_bytes(&pool).unwrap();
        let before = pool.used_bytes().unwrap();
        let foreign = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
        let wrong = source.prepare_cpu_worker(&foreign).unwrap_err();
        assert!(matches!(wrong.0, Failure::Stream(_)));
        assert_eq!(foreign.used_bytes().unwrap(), 0);
        assert_eq!(pool.used_bytes().unwrap(), before);
        drop(wrong);
        let worker = source.prepare_cpu_worker(&pool).unwrap();
        worker.validate_pool(&pool).unwrap();
        assert!(worker.validate_pool(&foreign).is_err());
        assert_eq!(
            pool.used_bytes().unwrap(),
            before.checked_add(quote).unwrap()
        );
        let second = source.prepare_cpu_worker(&pool).unwrap_err();
        let Failure::Initialization(ref second_source) = second.0 else {
            panic!("{second:?}");
        };
        let Some(ConstructorFailure::Native(error)) = second_source.constructor_failure() else {
            panic!("{second:?}");
        };
        assert_eq!(error.cause(), CpuWorkerCause::AlreadyInitialized);
        assert!(std::error::Error::source(error)
            .unwrap()
            .is::<CpuWorkerCause>());
        assert!(pool.used_bytes().unwrap() > before + quote);
        drop(second);
        assert_eq!(pool.used_bytes().unwrap(), before + quote);
        drop(source);
        worker.validate_pool(&pool).unwrap();
        drop(worker);
        safemlx::reclaim_allocation_owners();
        assert_eq!(pool.used_bytes().unwrap(), before + quote);
        println!("SOURCE_WORKER_COMPONENT_OK: actual birth/account custody");
    }
}
