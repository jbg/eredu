//! A new source stream's registration/CPU encoder through the existing account.
//! Worker/TLS/event initialization is deliberately a separate requirement.
use eredu_runtime::working_memory::{
    InitializedSharedNative, MemoryLedger, SharedNativeInitializationCustody,
    SharedNativeInitializationError, SharedNativeInitializer, WorkingMemoryError,
};
use safemlx::{
    CpuStreamRegistrationLayout, PreparedCpuStream, RegisteredCpuStream, Stream,
    StreamRegistrationCause, StreamRegistrationError,
};
use std::mem::{size_of, size_of_val};

#[derive(Debug)]
struct Initializer(CpuStreamRegistrationLayout<SharedNativeInitializationCustody>);
#[derive(Debug, thiserror::Error)]
enum ConstructorFailure {
    #[error("source stream owner preparation: {0}")]
    Preparation(#[source] StreamRegistrationError<SharedNativeInitializationCustody>),
    #[error("source stream native registration: {0}")]
    Native(#[source] StreamRegistrationError<PreparedCpuStream<SharedNativeInitializationCustody>>),
}
impl SharedNativeInitializer for Initializer {
    type Output = RegisteredCpuStream;
    type Error = ConstructorFailure;
    fn required_storage_bytes(&self) -> Result<usize, WorkingMemoryError> {
        let controls = [
            size_of::<&MemoryLedger>(),
            size_of::<PreparedMaterializationSourceStream>(),
            size_of::<MaterializationSourceStreamError>(),
            size_of::<Result<Self::Output, Self::Error>>(),
            size_of::<Result<PreparedMaterializationSourceStream, MaterializationSourceStreamError>>(
            ),
        ];
        controls
            .into_iter()
            .try_fold(size_of_val(&controls), usize::checked_add)
            .and_then(|n| self.0.required_bytes()?.checked_add(n))
            .ok_or(WorkingMemoryError::Overflow)
    }
    fn initialize(
        self,
        custody: SharedNativeInitializationCustody,
    ) -> Result<Self::Output, Self::Error> {
        PreparedCpuStream::with_layout(self.0, custody)
            .map_err(ConstructorFailure::Preparation)?
            .try_initialize()
            .map_err(ConstructorFailure::Native)
    }
}
/// Typed source/refusal retaining any actual native constructor prefix.
#[derive(Debug, thiserror::Error)]
#[error(transparent)]
pub struct MaterializationSourceStreamError(Failure);
impl MaterializationSourceStreamError {
    pub(crate) fn into_backend_failure(self) -> eredu_core::BackendFailure {
        use eredu_core::BackendFailure;
        match self.0 {
            Failure::Layout(error) | Failure::Native(error) => BackendFailure::from_error(error),
            Failure::Accounting(error) => BackendFailure::from_error(error),
            Failure::Initialization(error) => {
                BackendFailure::from_error(error.into_parts().1.retire_output_and_map_error(
                    |error| match error {
                        ConstructorFailure::Preparation(error) => error.cause(),
                        ConstructorFailure::Native(error) => error.cause(),
                    },
                ))
            }
        }
    }
}

#[derive(Debug, thiserror::Error)]
enum Failure {
    #[error("source stream producer qualification: {0}")]
    Layout(#[source] StreamRegistrationCause),
    #[error("source stream requested storage: {0}")]
    Accounting(#[source] WorkingMemoryError),
    #[error("source stream constructor identity: {0}")]
    Native(#[source] StreamRegistrationCause),
    #[error("source stream admission/constructor: {0}")]
    Initialization(#[source] SharedNativeInitializationError<Initializer>),
}
/// One newly registered CPU materialization source stream. Registration and its
/// inline encoder have original birth custody; no worker-readiness or execution
/// fit is asserted. An existing supplied stream is never replaced or promoted.
#[derive(Debug)]
pub struct PreparedMaterializationSourceStream(InitializedSharedNative<RegisteredCpuStream>);
impl PreparedMaterializationSourceStream {
    pub(crate) fn registration_owner(&self) -> &InitializedSharedNative<RegisteredCpuStream> {
        &self.0
    }

    fn initializer() -> Result<Initializer, MaterializationSourceStreamError> {
        PreparedCpuStream::<SharedNativeInitializationCustody>::layout()
            .map(Initializer)
            .map_err(|e| MaterializationSourceStreamError(Failure::Layout(e)))
    }
    /// Exact dynamic request, including the existing account's own controls.
    /// Module statics belong to the separately priced native-domain baseline.
    pub fn required_bytes() -> Result<u64, MaterializationSourceStreamError> {
        let plan = Self::initializer()?;
        MemoryLedger::shared_native_initialization_required_bytes(&plan)
            .map_err(|e| MaterializationSourceStreamError(Failure::Accounting(e)))
    }
    /// Cold explicit birth: qualification and source comparison precede the
    /// owner node, C wrapper and native registration. No worker fallback runs.
    pub fn prepare(pool: &MemoryLedger) -> Result<Self, MaterializationSourceStreamError> {
        let plan = Self::initializer()?;
        pool.initialize_shared_native(plan)
            .map(Self)
            .map_err(|e| MaterializationSourceStreamError(Failure::Initialization(e)))
    }
    /// Borrow for ordinary or separately prepared materialization mechanisms.
    /// Any borrowed raw native pointer remains immutable for this borrow.
    pub fn as_stream(&self) -> &Stream {
        self.0.output().as_stream()
    }
    /// Validate the immutable source-account origin before copying into a
    /// prepared materialization context. This is not submission authority.
    pub fn validate_pool(
        &self,
        pool: &MemoryLedger,
    ) -> Result<(), MaterializationSourceStreamError> {
        self.0
            .validate_pool(pool)
            .map_err(|e| MaterializationSourceStreamError(Failure::Accounting(e)))?;
        self.0
            .output()
            .try_borrow()
            .map_err(|e| MaterializationSourceStreamError(Failure::Native(e)))
    }
}

mod worker;
pub use worker::{MaterializationSourceWorkerError, PreparedMaterializationSourceWorker};

#[cfg(test)]
mod tests;
