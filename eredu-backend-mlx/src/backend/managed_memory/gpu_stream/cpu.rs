//! CPU execution owns the existing registered source stream and worker.
use super::*;
use crate::backend::nn::workspace::MlxCpuMatmulMechanism;
use crate::backend::runtime::checkpoint::store::{prepare_materialization_stream_from_plan,PreparedMaterializationStreamError};
use safemlx::{PreparedStreamCopy,StreamCopyCause};
use std::mem::{size_of, size_of_val};

#[derive(Debug)]
struct Initializer {
    pool: WorkingMemoryPool,
    matmul: Option<MlxCpuMatmulMechanism>,
}
#[derive(Debug)]
struct CpuExecution {
    // The copied context and its paid host owner retire before the source
    // stream/worker. No mutable source or raw Stream clone is published.
    selected: Option<InitializedSharedNative<PreparedStreamCopy<SharedNativeInitializationCustody>>>,
    worker: PreparedMaterializationSourceWorker,
    stream: PreparedMaterializationSourceStream,
}
#[derive(Debug, thiserror::Error)]
enum ConstructorFailure {
    #[error("CPU matrix context plan: {0}")]
    Selection(#[source] StreamCopyCause),
    #[error("CPU matrix context custody: {0}")]
    Selected(#[source] PreparedMaterializationStreamError),
    #[error("CPU execution source: {0}")]
    Stream(#[source] MaterializationSourceStreamError),
    #[error("CPU execution worker: {0}")]
    Worker(#[source] MaterializationSourceWorkerError),
}
impl SharedNativeInitializer for Initializer {
    type Output = CpuExecution;
    type Error = ConstructorFailure;
    fn required_storage_bytes(&self) -> Result<usize, WorkingMemoryError> {
        // Child registration and worker births keep their own source-account
        // admission. This account pays only the enclosing owner/call transports.
        let parts = [
            size_of::<Self>(),
            size_of::<CpuExecution>(),
            size_of::<PreparedCpuExecution>(),
            size_of::<ExecutionStream>(),
            size_of::<PreparedExecutionStreams>(),
            size_of::<WorkingMemoryPool>(),
            size_of::<Result<CpuExecution, ConstructorFailure>>(),
            size_of::<Result<PreparedCpuExecution, MlxGpuStreamError>>(),
            size_of::<Result<Option<PreparedExecutionStreams>, MlxGpuStreamError>>(),
            size_of::<Result<PreparedMaterializationSourceStream, MaterializationSourceStreamError>>(
            ),
            size_of::<Result<PreparedMaterializationSourceWorker, MaterializationSourceWorkerError>>(
            ),
            size_of::<ConstructorFailure>(),
            size_of::<MlxGpuStreamError>(),
        ];
        let selection=match self.matmul {
            Some(choice)=>choice.control_bytes::<SharedNativeInitializationCustody>().ok_or(WorkingMemoryError::UnknownBound)?,
            None=>0,
        };
        parts
            .into_iter()
            .try_fold(selection.checked_add(size_of_val(&parts)).ok_or(WorkingMemoryError::Overflow)?, usize::checked_add)
            .ok_or(WorkingMemoryError::Overflow)
    }
    fn initialize(
        self,
        _custody: SharedNativeInitializationCustody,
    ) -> Result<Self::Output, Self::Error> {
        let stream = PreparedMaterializationSourceStream::prepare(&self.pool)
            .map_err(ConstructorFailure::Stream)?;
        let worker = stream
            .prepare_cpu_worker(&self.pool)
            .map_err(ConstructorFailure::Worker)?;
        let selected=match self.matmul {
            Some(choice)=>{
                let plan=choice.stream_plan(stream.as_stream()).map_err(ConstructorFailure::Selection)?;
                Some(prepare_materialization_stream_from_plan(&self.pool,plan).map_err(ConstructorFailure::Selected)?)
            },
            None=>None,
        };
        Ok(CpuExecution { selected, worker, stream })
    }
}

#[derive(Debug, thiserror::Error)]
#[error("CPU execution ownership: {0}")]
pub(super) struct CpuExecutionError(#[source] SharedNativeInitializationError<Initializer>);

impl CpuExecutionError {
    pub(super) fn into_backend_failure(self) -> eredu_core::BackendFailure {
        eredu_core::BackendFailure::from_error(self.0.retire_output_and_map_error(|error| {
            match error {
                ConstructorFailure::Selection(error) => eredu_core::BackendFailure::from_error(error),
                ConstructorFailure::Selected(error) => eredu_core::BackendFailure::from_error(error),
                ConstructorFailure::Stream(error) => error.into_backend_failure(),
                ConstructorFailure::Worker(error) => error.into_backend_failure(),
            }
        }))
    }
}

#[derive(Debug)]
pub(super) struct PreparedCpuExecution(InitializedSharedNative<CpuExecution>);
impl PreparedCpuExecution {
    pub(super) fn prepare(pool: &WorkingMemoryPool) -> Result<Self, MlxGpuStreamError> {
        Self::prepare_selected(pool,None)
    }
    pub(super) fn prepare_with_matmul(pool:&WorkingMemoryPool,matmul:MlxCpuMatmulMechanism)->Result<Self,MlxGpuStreamError> {
        Self::prepare_selected(pool,Some(matmul))
    }
    fn prepare_selected(pool:&WorkingMemoryPool,matmul:Option<MlxCpuMatmulMechanism>)->Result<Self,MlxGpuStreamError> {
        if !pool.same_domain(&super::super::domain()) {
            return Err(Failure::Accounting(WorkingMemoryError::IdentityMismatch).into());
        }
        pool.initialize_shared_native(Initializer { pool: pool.clone(), matmul })
            .map(Self)
            .map_err(|cause| Failure::CpuExecution(CpuExecutionError(cause)).into())
    }
    pub(super) fn stream(&self) -> &Stream {
        match &self.0.output().selected {
            Some(selected)=>selected.output().as_stream(),
            None=>self.0.output().stream.as_stream(),
        }
    }
    pub(super) fn validate_pool(
        &self,
        pool: &WorkingMemoryPool,
    ) -> Result<(), MlxStreamOwnershipError> {
        self.0
            .validate_pool(pool)
            .map_err(MlxStreamOwnershipError::Accounting)?;
        if let Some(selected)=&self.0.output().selected {
            selected.validate_pool(pool).map_err(MlxStreamOwnershipError::Accounting)?;
            selected.output().owner().validate_pool(pool).map_err(MlxStreamOwnershipError::Accounting)?;
        }
        let source = self.0.output().stream.registration_owner();
        source
            .validate_pool(pool)
            .map_err(MlxStreamOwnershipError::Accounting)?;
        source
            .output()
            .try_borrow()
            .map_err(MlxStreamOwnershipError::CpuStream)?;
        let worker = self.0.output().worker.worker_owner();
        worker
            .validate_pool(pool)
            .map_err(MlxStreamOwnershipError::Accounting)?;
        worker
            .output()
            .try_borrow()
            .map_err(MlxStreamOwnershipError::CpuWorker)
    }
    pub(super) fn observe_idle(
        &self,
        pool: &WorkingMemoryPool,
    ) -> Result<(), MlxStreamOwnershipError> {
        self.validate_pool(pool)?;
        self.0
            .output()
            .worker
            .worker_owner()
            .output()
            .try_observe_idle()
            .map_err(MlxStreamOwnershipError::CpuWorker)
    }
    /// The enclosing initializer and optional selected-context copy are local
    /// wrapper owners. The process registrations retain only their independent
    /// source stream/worker accounts after this execution owner retires.
    #[cfg(test)]
    pub(super) fn wrapper_control_bytes(&self) -> u64 {
        self.0.original_bytes()
            + self.0.output().selected.as_ref().map_or(0,InitializedSharedNative::original_bytes)
    }
    #[cfg(test)]
    pub(super) fn original_bytes(&self) -> u64 {
        self.0.original_bytes()
            + self.0.output().selected.as_ref().map_or(0,InitializedSharedNative::original_bytes)
            + self.0.output().stream.registration_owner().original_bytes()
            + self.0.output().worker.worker_owner().original_bytes()
    }
}
