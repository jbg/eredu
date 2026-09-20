//! Admitted execution stream/worker birth for selected backend targets.
mod cpu;
use crate::backend::runtime::checkpoint::store::{
    MaterializationSourceStreamError, MaterializationSourceWorkerError,
    PreparedMaterializationSourceStream, PreparedMaterializationSourceWorker,
};
use eredu_runtime::working_memory::{
    InitializedSharedNative, SharedNativeInitializationCustody, SharedNativeInitializationError,
    SharedNativeInitializer, WorkingMemoryError, WorkingMemoryPool,
};
use safemlx::{
    GpuStreamRegistrationCause, GpuStreamRegistrationError, GpuStreamRegistrationLayout,
    GpuStreamTarget, PreparedGpuStream, RegisteredGpuStream, Stream,
};

#[derive(Debug)]
struct Initializer {
    layout: GpuStreamRegistrationLayout<SharedNativeInitializationCustody>,
    target: GpuStreamTarget,
}
#[derive(Debug, thiserror::Error)]
enum ConstructorFailure {
    #[error("GPU stream custody preparation: {0}")]
    Preparation(#[source] GpuStreamRegistrationError<SharedNativeInitializationCustody>),
    #[error("GPU stream native constructor: {0}")]
    Native(
        #[source] GpuStreamRegistrationError<PreparedGpuStream<SharedNativeInitializationCustody>>,
    ),
}
impl SharedNativeInitializer for Initializer {
    type Output = RegisteredGpuStream;
    type Error = ConstructorFailure;
    fn required_storage_bytes(&self) -> Result<usize, WorkingMemoryError> {
        self.layout
            .required_bytes()
            .and_then(|n| n.checked_add(std::mem::size_of::<Initializer>()))
            .and_then(|n| n.checked_add(std::mem::size_of::<MlxGpuStreamError>()))
            .and_then(|n| n.checked_add(std::mem::size_of::<PreparedExecutionStreams>()))
            .and_then(|n| {
                n.checked_add(std::mem::size_of::<
                    Result<PreparedExecutionStream, MlxGpuStreamError>,
                >())
            })
            .ok_or(WorkingMemoryError::Overflow)
    }
    fn initialize(
        self,
        custody: SharedNativeInitializationCustody,
    ) -> Result<Self::Output, Self::Error> {
        PreparedGpuStream::with_layout(self.layout, custody)
            .map_err(ConstructorFailure::Preparation)?
            .try_initialize(self.target)
            .map_err(ConstructorFailure::Native)
    }
}
/// Actual prerequisite or constructor refusal, retaining native-prefix custody.
#[derive(Debug, thiserror::Error)]
#[error(transparent)]
pub struct MlxGpuStreamError(#[from] Failure);

impl MlxGpuStreamError {
    /// Retires thread-local constructor wrappers before crossing the public
    /// error boundary. Native registrations retain their own custody; disposing
    /// a wrapper does not release a live registration's reservation.
    pub(crate) fn into_backend_failure(self) -> eredu_core::BackendFailure {
        use eredu_core::BackendFailure;
        match self.0 {
            Failure::Accounting(error) => BackendFailure::from_error(error),
            Failure::Native(error) => BackendFailure::from_error(error),
            Failure::Runtime(error) => BackendFailure::from_error(error),
            Failure::Device(error) => BackendFailure::from_error(error),
            Failure::Scheduler(error) => BackendFailure::from_error(error),
            Failure::CpuExecution(error) => error.into_backend_failure(),
            Failure::Source(error) => error.into_backend_failure(),
            Failure::Worker(error) => error.into_backend_failure(),
            Failure::Constructor(error) => BackendFailure::from_error(
                error.retire_output_and_map_error(|error| match error {
                    ConstructorFailure::Preparation(error) => error.cause(),
                    ConstructorFailure::Native(error) => error.cause(),
                }),
            ),
        }
    }
}
/// Actual prerequisite or constructor refusal, retaining native-prefix custody.
#[derive(Debug, thiserror::Error)]
enum Failure {
    #[error("GPU stream source: {0}")]
    Accounting(#[source] WorkingMemoryError),
    #[error("GPU stream layout: {0}")]
    Native(#[source] GpuStreamRegistrationCause),
    #[error("GPU stream prerequisite: {0}")]
    Runtime(#[source] super::input_allocator::MlxInputAllocatorInitializationError),
    #[error("GPU stream Device owner: {0}")]
    Device(#[source] super::metal_device::MlxMetalDeviceInitializationError),
    #[error("GPU stream Scheduler owner: {0}")]
    Scheduler(#[source] super::scheduler::MlxSchedulerInitializationError),
    #[error("GPU stream birth: {0}")]
    Constructor(#[source] SharedNativeInitializationError<Initializer>),
    #[error(transparent)]
    CpuExecution(cpu::CpuExecutionError),
    #[error("CPU source stream: {0}")]
    Source(#[source] MaterializationSourceStreamError),
    #[error("CPU source worker: {0}")]
    Worker(#[source] MaterializationSourceWorkerError),
}
/// The backend-owned wrapper, alongside its exact shared initializer account.
#[derive(Debug)]
pub(crate) struct PreparedExecutionStream(InitializedSharedNative<RegisteredGpuStream>);
impl PreparedExecutionStream {
    pub(crate) fn prepare(pool: &WorkingMemoryPool) -> Result<Self, MlxGpuStreamError> {
        if !pool.same_domain(&super::domain()) {
            return Err(Failure::Accounting(WorkingMemoryError::IdentityMismatch).into());
        }
        let layout = PreparedGpuStream::<SharedNativeInitializationCustody>::layout()
            .map_err(Failure::Native)?;
        super::input_allocator::prepare_admitted(pool).map_err(Failure::Runtime)?;
        // Loaded modules may subsequently be constructed inside an accepted
        // materialization scope. Prepare the source-owned router worker here,
        // while cold factory authority can still admit its actual native birth.
        super::router::prepare_before_native_construction();
        let device = super::metal_device::admitted_owner(pool).map_err(Failure::Device)?;
        let scheduler = super::scheduler::admitted_owner(pool).map_err(Failure::Scheduler)?;
        let target =
            GpuStreamTarget::for_initialized(device, scheduler).map_err(Failure::Native)?;
        pool.initialize_shared_native(Initializer { layout, target })
            .map(Self)
            .map_err(|error| Failure::Constructor(error).into())
    }
    pub(crate) fn for_factory(pool: &WorkingMemoryPool) -> Result<Option<Self>, MlxGpuStreamError> {
        match Self::prepare(pool) {
            Ok(stream) => Ok(Some(stream)),
            Err(MlxGpuStreamError(Failure::Native(GpuStreamRegistrationCause::UnknownLayout))) => {
                Ok(None)
            }
            Err(MlxGpuStreamError(Failure::Runtime(error))) if error.permits_ordinary_stream() => {
                Ok(None)
            }
            Err(error) => Err(error),
        }
    }
    pub(crate) fn as_stream(&self) -> &Stream {
        self.0.output().as_stream()
    }
    pub(crate) fn validate_pool(
        &self,
        pool: &WorkingMemoryPool,
    ) -> Result<(), MlxStreamOwnershipError> {
        self.0
            .validate_pool(pool)
            .map_err(MlxStreamOwnershipError::Accounting)?;
        self.0
            .output()
            .try_borrow()
            .map_err(MlxStreamOwnershipError::Gpu)
    }
}

#[derive(Debug)]
enum ExecutionStream {
    Gpu(PreparedExecutionStream),
    Cpu(cpu::PreparedCpuExecution),
}
impl ExecutionStream {
    fn as_stream(&self)->&Stream {match self {Self::Gpu(value)=>value.as_stream(),Self::Cpu(value)=>value.stream()}}
    fn validate_pool(&self,pool:&WorkingMemoryPool)->Result<(),MlxStreamOwnershipError>{
        match self {Self::Gpu(value)=>value.validate_pool(pool),Self::Cpu(value)=>value.validate_pool(pool)}
    }
    fn observe_idle(&self,pool:&WorkingMemoryPool)->Result<(),MlxStreamOwnershipError>{
        match self {
            Self::Gpu(value)=>value.0.output().try_observe_idle().map_err(MlxStreamOwnershipError::Gpu),
            Self::Cpu(value)=>value.observe_idle(pool),
        }
    }
}
/// Actual same-domain execution and weight-stream owners retained by the backend.
#[derive(Debug)]
pub(crate) struct PreparedExecutionStreams {
    execution: ExecutionStream,
    source: PreparedMaterializationSourceStream,
    worker: PreparedMaterializationSourceWorker,
}
impl PreparedExecutionStreams {
    pub(crate) fn for_factory(pool: &WorkingMemoryPool) -> Result<Option<Self>, MlxGpuStreamError> {
        let Some(execution) = PreparedExecutionStream::for_factory(pool)? else {
            return Ok(None);
        };
        Self::finish_factory(ExecutionStream::Gpu(execution),pool).map(Some)
    }
    /// Only CPU stream and worker ownership is supplied here. Native equation
    /// and physical-copy source consumers retain their own device qualification.
    /// Shared cold selection for plan and distributed native construction.
    /// This choice is made before either ordinary or managed execution begins.
    pub(crate) fn for_device_factory(pool: &WorkingMemoryPool, device: safemlx::DeviceType)
        -> Result<Option<Self>, MlxGpuStreamError> {
        match device {
            safemlx::DeviceType::Gpu => Self::for_factory(pool),
            safemlx::DeviceType::Cpu => {
                match crate::backend::nn::workspace::MlxCpuMatmulMechanism::select(
                    eredu_nn::CpuMatmulImplementation::Float32Tiles,
                ) {
                    Some(choice) => Self::for_cpu_factory_with_matmul(pool, choice),
                    None => Self::for_cpu_factory(pool),
                }
            }
        }
    }

    pub(crate) fn for_cpu_factory(pool:&WorkingMemoryPool)->Result<Option<Self>,MlxGpuStreamError>{
        Self::for_cpu_factory_selected(pool,None)
    }
    /// Use the same source/worker factory with an explicit cold matrix choice.
    /// This grants no whole-equation or cross-device source qualification.
    pub(crate) fn for_cpu_factory_with_matmul(pool:&WorkingMemoryPool,choice:crate::backend::nn::workspace::MlxCpuMatmulMechanism)->Result<Option<Self>,MlxGpuStreamError>{
        Self::for_cpu_factory_selected(pool,Some(choice))
    }
    fn for_cpu_factory_selected(pool:&WorkingMemoryPool,choice:Option<crate::backend::nn::workspace::MlxCpuMatmulMechanism>)->Result<Option<Self>,MlxGpuStreamError>{
        match super::input_allocator::prepare_admitted(pool) {
            Ok(_)=>{},
            Err(cause) if cause.permits_ordinary_stream()=>return Ok(None),
            Err(cause)=>return Err(Failure::Runtime(cause).into()),
        }
        let execution=match choice {
            Some(choice)=>cpu::PreparedCpuExecution::prepare_with_matmul(pool,choice)?,
            None=>cpu::PreparedCpuExecution::prepare(pool)?,
        };
        Self::finish_factory(ExecutionStream::Cpu(execution),pool).map(Some)
    }
    fn finish_factory(execution:ExecutionStream,pool:&WorkingMemoryPool)->Result<Self,MlxGpuStreamError>{
        let source = PreparedMaterializationSourceStream::prepare(pool).map_err(Failure::Source)?;
        let worker = source.prepare_cpu_worker(pool).map_err(Failure::Worker)?;
        Ok(Self {execution,source,worker})
    }
    /// Pure readiness snapshot of the exact retained source worker and encoder.
    /// The caller separately excludes all aliases able to mutate its session.
    pub(crate) fn observe_idle(
        &self,
        pool: &WorkingMemoryPool,
    ) -> Result<(), MlxStreamOwnershipError> {
        self.validate_pool(pool)?;
        self.worker
            .worker_owner()
            .output()
            .try_observe_idle()
            .map_err(MlxStreamOwnershipError::CpuWorker)?;
        self.execution.observe_idle(pool)
    }
    #[cfg(test)]
    pub(crate) fn retiring_wrapper_control_bytes(&self) -> u64 {
        match &self.execution {
            ExecutionStream::Cpu(execution) => execution.wrapper_control_bytes(),
            ExecutionStream::Gpu(_) => 0,
        }
    }
    pub(crate) fn execution(&self) -> &Stream {
        self.execution.as_stream()
    }
    pub(crate) fn source(&self) -> &Stream {
        self.source.as_stream()
    }
    pub(crate) fn validate_pool(
        &self,
        pool: &WorkingMemoryPool,
    ) -> Result<(), MlxStreamOwnershipError> {
        self.execution.validate_pool(pool)?;
        let source = self.source.registration_owner();
        source
            .validate_pool(pool)
            .map_err(MlxStreamOwnershipError::Accounting)?;
        source
            .output()
            .try_borrow()
            .map_err(MlxStreamOwnershipError::CpuStream)?;
        let worker = self.worker.worker_owner();
        worker
            .validate_pool(pool)
            .map_err(MlxStreamOwnershipError::Accounting)?;
        worker
            .output()
            .try_borrow()
            .map_err(MlxStreamOwnershipError::CpuWorker)?;
        Ok(())
    }
}

/// Fixed read-only owner authentication failure; no constructor prefix is created.
#[derive(Debug, thiserror::Error)]
pub enum MlxStreamOwnershipError {
    /// The retained account belongs to a different source domain.
    #[error("stream account: {0}")]
    Accounting(#[source] WorkingMemoryError),
    /// The GPU registration no longer authenticates or is temporarily busy.
    #[error("GPU stream owner: {0}")]
    Gpu(#[source] GpuStreamRegistrationCause),
    /// The CPU source registration no longer authenticates or is temporarily busy.
    #[error("CPU source stream owner: {0}")]
    CpuStream(#[source] safemlx::StreamRegistrationCause),
    /// The actual process source worker no longer authenticates or is busy.
    #[error("CPU source worker owner: {0}")]
    CpuWorker(#[source] safemlx::CpuWorkerCause),
}

#[cfg(test)]
mod tests;
