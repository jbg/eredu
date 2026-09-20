//! Actual admitted runtime and two independent CPU stream/worker identities.
use crate::backend::managed_memory;
use crate::backend::runtime::checkpoint::store::{
    MaterializationSourceStreamError, MaterializationSourceWorkerError,
    PreparedMaterializationSourceStream, PreparedMaterializationSourceWorker,
};
use eredu_runtime::working_memory::{
    InitializedSharedNative, SharedNativeInitializationCustody, SharedNativeInitializationError,
    SharedNativeInitializer, WorkingMemoryError, WorkingMemoryPool,
};
use safemlx::{PreparedInputRuntime, Stream};
use std::mem::{size_of, size_of_val};

#[derive(Debug, Default)]
struct Prefix {
    runtime: Option<PreparedInputRuntime>,
    streams: [Option<PreparedMaterializationSourceStream>; 2],
    workers: [Option<PreparedMaterializationSourceWorker>; 2],
}

#[derive(Debug)]
struct Resources {
    runtime: PreparedInputRuntime,
    streams: [PreparedMaterializationSourceStream; 2],
    workers: [PreparedMaterializationSourceWorker; 2],
    _custody: SharedNativeInitializationCustody,
}

#[derive(Debug, thiserror::Error)]
enum Cause {
    #[error("tile input allocator: {0}")]
    Runtime(#[from] managed_memory::input_allocator::MlxInputAllocatorInitializationError),
    #[error("tile source stream: {0}")]
    Stream(#[from] MaterializationSourceStreamError),
    #[error("tile source worker: {0}")]
    Worker(#[from] MaterializationSourceWorkerError),
}

#[derive(Debug, thiserror::Error)]
#[error("CPU tile resources: {cause}")]
struct ConstructionError {
    #[source]
    cause: Cause,
    _prefix: Prefix,
    _custody: SharedNativeInitializationCustody,
}

#[derive(Debug)]
struct Initializer(WorkingMemoryPool);
impl SharedNativeInitializer for Initializer {
    type Output = Resources;
    type Error = ConstructionError;

    fn required_storage_bytes(&self) -> Result<usize, WorkingMemoryError> {
        // Each actual native child has its own checked source comparison. This
        // account funds only the composing owner and fixed call/error controls.
        let controls = [
            size_of::<Prefix>(),
            size_of::<Resources>(),
            size_of::<Cause>(),
            size_of::<ConstructionError>(),
            size_of::<Result<(), Cause>>(),
            size_of::<CpuTileResources>(),
            size_of::<ResourceError>(),
            size_of::<Result<CpuTileResources, ResourceError>>(),
            size_of::<[&Stream; 2]>(),
        ];
        controls
            .into_iter()
            .try_fold(size_of_val(&controls), usize::checked_add)
            .ok_or(WorkingMemoryError::Overflow)
    }

    fn initialize(
        self,
        custody: SharedNativeInitializationCustody,
    ) -> Result<Resources, ConstructionError> {
        let mut prefix = Prefix::default();
        let result = (|| {
            let (runtime, _) = managed_memory::input_allocator::prepare_admitted(&self.0)?;
            prefix.runtime = Some(runtime);
            for slot in 0..2 {
                let stream = PreparedMaterializationSourceStream::prepare(&self.0)?;
                prefix.streams[slot] = Some(stream);
                let worker = prefix.streams[slot]
                    .as_ref()
                    .expect("constructed stream")
                    .prepare_cpu_worker(&self.0)?;
                prefix.workers[slot] = Some(worker);
            }
            Ok::<_, Cause>(())
        })();
        match result {
            Err(cause) => Err(ConstructionError {
                cause,
                _prefix: prefix,
                _custody: custody,
            }),
            Ok(()) => Ok(Resources {
                runtime: prefix.runtime.take().expect("constructed runtime"),
                streams: prefix
                    .streams
                    .map(|stream| stream.expect("constructed stream")),
                workers: prefix
                    .workers
                    .map(|worker| worker.expect("constructed worker")),
                _custody: custody,
            }),
        }
    }
}

#[derive(Debug, thiserror::Error)]
enum Failure {
    #[error("CPU tile resource domain: {0}")]
    Policy(#[from] WorkingMemoryError),
    #[error("CPU tile resource initialization: {0}")]
    Initialization(#[source] SharedNativeInitializationError<Initializer>),
    #[error("CPU tile stream validation: {0}")]
    Stream(#[from] MaterializationSourceStreamError),
    #[error("CPU tile worker validation: {0}")]
    Worker(#[from] MaterializationSourceWorkerError),
}

#[derive(Debug, thiserror::Error)]
#[error(transparent)]
pub(super) struct ResourceError(#[from] Failure);
impl ResourceError {
    pub(super) fn into_backend_failure(self) -> eredu_core::BackendFailure {
        use eredu_core::BackendFailure;
        match self.0 {
            Failure::Policy(cause) => BackendFailure::from_error(cause),
            Failure::Stream(cause) => cause.into_backend_failure(),
            Failure::Worker(cause) => cause.into_backend_failure(),
            Failure::Initialization(error) => BackendFailure::from_error(
                error.into_parts().1.retire_output_and_map_error(|error| {
                    let ConstructionError { cause, _prefix, _custody } = error;
                    drop(_prefix);
                    drop(_custody);
                    match cause {
                        Cause::Runtime(cause) => BackendFailure::from_error(cause),
                        Cause::Stream(cause) => cause.into_backend_failure(),
                        Cause::Worker(cause) => cause.into_backend_failure(),
                    }
                }),
            ),
        }
    }
}
impl From<WorkingMemoryError> for ResourceError {
    fn from(cause: WorkingMemoryError) -> Self {
        Self(Failure::Policy(cause))
    }
}
impl From<MaterializationSourceStreamError> for ResourceError {
    fn from(cause: MaterializationSourceStreamError) -> Self {
        Self(Failure::Stream(cause))
    }
}
impl From<MaterializationSourceWorkerError> for ResourceError {
    fn from(cause: MaterializationSourceWorkerError) -> Self {
        Self(Failure::Worker(cause))
    }
}

/// Borrowed access to two admitted CPU workers and their original streams. Each
/// native singleton/registration keeps its own source account for its real
/// process lifetime. Dropping these wrappers cannot refund those native owners.
/// Read/recipe metadata, cache/queue/output storage and per-tile work are separate.
#[derive(Debug)]
pub(super) struct CpuTileResources(InitializedSharedNative<Resources>);
impl CpuTileResources {
    pub(super) fn prepare(pool: &WorkingMemoryPool) -> Result<Self, ResourceError> {
        if !pool.same_domain(&managed_memory::domain()) {
            return Err(WorkingMemoryError::IdentityMismatch.into());
        }
        pool.initialize_shared_native(Initializer(pool.clone()))
            .map(Self)
            .map_err(|cause| ResourceError(Failure::Initialization(cause)))
    }

    pub(super) fn runtime(&self) -> &PreparedInputRuntime {
        &self.0.output().runtime
    }

    pub(super) fn streams(&self) -> [&Stream; 2] {
        self.0
            .output()
            .streams
            .each_ref()
            .map(PreparedMaterializationSourceStream::as_stream)
    }

    pub(super) fn validate_pool(&self, pool: &WorkingMemoryPool) -> Result<(), ResourceError> {
        self.0.validate_pool(pool)?;
        // Revalidate the actual admitted singleton; this cannot initialize or
        // promote an ordinary predecessor and allocates no runtime wrapper.
        managed_memory::input_allocator::admitted_initializer(pool)?;
        for (stream, worker) in self.0.output().streams.iter().zip(&self.0.output().workers) {
            stream.validate_pool(pool)?;
            worker.validate_pool(pool)?;
        }
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn original_control_bytes(&self) -> u64 {
        self.0.original_bytes()
    }
}
