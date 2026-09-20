//! Quote the implementation already carried by the actual borrowed stream.
use super::{MlxMetalWorkspaceMechanisms, ResidentRecipeRecorder, NativeAllocationFacts, MlxWorkspaceFactError};
use eredu_nn::workspace::*;
use eredu_nn::{Error, workspace::{WorkspaceContext, WorkspaceMetadataError, HostMetadataFunding}};
use std::mem::{size_of, size_of_val};
use crate::backend::nn::workspace::{MlxCpuMatmulMechanism, MlxCpuWorkspaceMechanisms};
use eredu_nn::CpuMatmulImplementation;
use safemlx::{CpuMatmulKernel, DeviceType, Stream, StreamCopyPlan};

#[derive(Debug,thiserror::Error)]
#[error("resident equation stream source: {cause}")]
struct StreamFailure {
    #[source] cause:safemlx::StreamCopyCause,
    _funding:HostMetadataFunding,
}

/// Cold facts only. The caller keeps its exact environment/source loan through
/// native execution; this value never creates or modifies a native context.
#[derive(Clone, Copy, Debug)]
pub(crate) enum ResidentExecutionMechanisms {
    Metal(MlxMetalWorkspaceMechanisms),
    Cpu { ordinary: MlxMetalWorkspaceMechanisms, cpu: MlxCpuWorkspaceMechanisms },
}
impl ResidentExecutionMechanisms {
    /// Loading retains the immutable implementation of this exact stream.
    /// This cold query creates no device, stream, tensor or admission authority.
    pub(crate) fn from_cold_stream(
        ordinary: MlxMetalWorkspaceMechanisms, stream: &Stream,
    ) -> Result<Self, Error> {
        let source = StreamCopyPlan::<()>::capture(stream).map_err(Error::backend)?;
        Self::from_source(ordinary, &source).map_err(Into::into)
    }
    fn from_source(
        ordinary: MlxMetalWorkspaceMechanisms, source: &StreamCopyPlan<()>,
    ) -> Result<Self, WorkspaceMetadataError> {
        Ok(match source.device_type() {
            DeviceType::Gpu => Self::Metal(ordinary),
            DeviceType::Cpu => {
                let implementation = match source.cpu_matmul() {
                    CpuMatmulKernel::PlatformDefault => CpuMatmulImplementation::PlatformDefault,
                    CpuMatmulKernel::Float32Tiles => CpuMatmulImplementation::Float32Tiles,
                    CpuMatmulKernel::Float32AndFloat16Tiles => CpuMatmulImplementation::Float32AndFloat16Tiles,
                };
                let matmul = MlxCpuMatmulMechanism::select(implementation)
                    .ok_or(WorkspaceMetadataError::Unqualified)?;
                Self::Cpu { ordinary, cpu: MlxCpuWorkspaceMechanisms::new(ordinary.allocation(), matmul) }
            }
        })
    }
    /// Common allocator and native host facts. Equation quotation must use the
    /// complete retained enum, never infer a GPU worker from these shared facts.
    pub(crate) fn ordinary(self) -> MlxMetalWorkspaceMechanisms {
        match self { Self::Metal(facts) | Self::Cpu { ordinary: facts, .. } => facts }
    }
    pub(crate) fn allocation(self) -> NativeAllocationFacts { self.ordinary().allocation() }
    pub(crate) fn metal(self) -> Option<MlxMetalWorkspaceMechanisms> {
        match self { Self::Metal(facts) => Some(facts), Self::Cpu { .. } => None }
    }
    pub(crate) fn from_stream(ordinary:MlxMetalWorkspaceMechanisms, stream:&Stream,
        funding:&HostMetadataFunding) -> Result<Self,Error> {
        // The fixed source queries have no owning side effects. Price their
        // actual scalar transports before reading the retained stream value.
        let selected=MlxCpuMatmulMechanism::select(CpuMatmulImplementation::Float32Tiles);
        let default=MlxCpuMatmulMechanism::select(CpuMatmulImplementation::PlatformDefault)
            .ok_or(WorkspaceMetadataError::Overflow)?;
        let selection_controls=selected.unwrap_or(default).control_bytes::<()>()
            .ok_or(WorkspaceMetadataError::Overflow)?;
        let frames=[selection_controls,Stream::device_type_control_bytes().ok_or(WorkspaceMetadataError::Overflow)?,
            size_of::<Self>(),size_of::<Result<Self,Error>>(),size_of::<Option<MlxCpuMatmulMechanism>>(),
            size_of::<MlxCpuMatmulMechanism>(),size_of::<MlxCpuWorkspaceMechanisms>(),
            size_of::<StreamCopyPlan<()>>(),size_of::<Result<StreamCopyPlan<()>,safemlx::StreamCopyCause>>(),
            size_of::<DeviceType>(),size_of::<CpuMatmulKernel>(),
            WorkspaceContext::metadata_source_bytes::<StreamFailure>().ok_or(WorkspaceMetadataError::Overflow)?,
            size_of::<(Self,HostMetadataFunding)>(),
            size_of::<(MlxMetalWorkspaceMechanisms,&Stream,&HostMetadataFunding)>()];
        funding.reserve_metadata(frames.into_iter().try_fold(size_of_val(&frames),usize::checked_add)
            .ok_or(WorkspaceMetadataError::Overflow)?).map_err(WorkspaceMetadataError::Funding)?;
        let source=StreamCopyPlan::<()>::capture(stream).map_err(|cause|Error::backend_retained_source(
            StreamFailure {cause,_funding:funding.clone()}))?;
        Self::from_source(ordinary, &source).map_err(Into::into)
    }

    pub(crate) fn context(self,funding:HostMetadataFunding)->Result<WorkspaceContext,WorkspaceMetadataError> {
        match self {
            Self::Metal(mechanism)=>WorkspaceContext::new_with_metadata_funding(mechanism,funding),
            Self::Cpu {cpu,..}=>WorkspaceContext::new_with_metadata_funding(cpu,funding),
        }
    }
    pub(crate) fn recorder(self, geometry: eredu_core::InferenceGeometry, context: &WorkspaceContext)
        -> Result<ResidentRecipeRecorder, Error> {
        let parts = [size_of::<Self>(), size_of::<eredu_core::InferenceGeometry>(),
            size_of::<ResidentRecipeRecorder>(), size_of::<Result<ResidentRecipeRecorder, Error>>(),
            size_of::<(Self, eredu_core::InferenceGeometry, &WorkspaceContext)>()];
        context.charge_metadata(parts.into_iter().try_fold(size_of_val(&parts), usize::checked_add)
            .ok_or(WorkspaceMetadataError::Overflow)?)?;
        match self {
            Self::Metal(ordinary) => ResidentRecipeRecorder::with_context(geometry, ordinary, context),
            Self::Cpu { ordinary, cpu } => ResidentRecipeRecorder::with_cpu_context(geometry, ordinary, cpu, context),
        }
    }

}


// Dispatch every equation fact to the implementation retained at loading.
// Native source preparation remains with that same implementation.
macro_rules! selected {
    ($self:expr, $method:ident($($arg:expr),* $(,)?)) => {
        match $self {
            ResidentExecutionMechanisms::Metal(facts) => facts.$method($($arg),*),
            ResidentExecutionMechanisms::Cpu { cpu, .. } => cpu.$method($($arg),*),
        }
    };
}
impl WorkspaceMechanisms for ResidentExecutionMechanisms {
    fn prepared_text_input_dtype(&self)->Option<WorkspaceDtype> {
        // Both selected native text-input workers consume [u32]: ordinary
        // Array::try_from_slice and original prepared host-token copying. This
        // element declaration is independent of CPU/Metal or model family.
        match <u32 as safemlx::ArrayElement>::DTYPE {
            safemlx::Dtype::Uint32=>Some(WorkspaceDtype::Uint32),
            _=>None,
        }
    }

    fn output_representation(&self, operation: WorkspaceOperationView<'_>, output: usize)
        -> Option<WorkspaceRepresentation> { selected!(self, output_representation(operation, output)) }
    fn projection_input_observation_mechanism(&self, format: &eredu_nn::LinearFormatSpec)
        -> Result<Option<eredu_nn::ProjectionInputObservationMechanism>, Error> {
        selected!(self, projection_input_observation_mechanism(format))
    }
    fn grouped_observation_schedule(&self, bank: &WorkspaceGroupedBank, tokens: u32)
        -> Result<Option<WorkspaceGroupedObservationSchedule>, Error> {
        selected!(self, grouped_observation_schedule(bank, tokens))
    }
    fn operation_bound(&self, operation: &WorkspaceOperation)
        -> Result<Option<WorkspaceOperationBound>, Error> { selected!(self, operation_bound(operation)) }
    fn host_workspace_bound(&self, operation: &WorkspaceOperation)
        -> Result<Option<WorkspaceHostBound>, Error> { selected!(self, host_workspace_bound(operation)) }
}
impl WorkspaceFactMechanisms for ResidentExecutionMechanisms {
    type Error = MlxWorkspaceFactError;
    fn with_prepared_facts<T>(&self, operation: WorkspaceOperationView<'_>,
        funding: Option<&HostMetadataFunding>,
        visit: impl FnOnce(&dyn WorkspaceFactMechanisms<Error = Self::Error>) -> T)
        -> Result<T, Self::Error> {
        selected!(self, with_prepared_facts(operation, funding, visit))
    }
    fn operation_facts(&self, operation: WorkspaceOperationView<'_>)
        -> Result<Option<WorkspaceOperationFacts>, Self::Error> {
        selected!(self, operation_facts(operation))
    }
    fn write_operation_facts(&self, operation: WorkspaceOperationView<'_>,
        destination: WorkspaceEffectDestination<'_>)
        -> Result<Option<WorkspaceOperationFacts>, Self::Error> {
        selected!(self, write_operation_facts(operation, destination))
    }
    fn host_facts(&self, operation: WorkspaceOperationView<'_>)
        -> Result<Option<WorkspaceHostFacts>, Self::Error> {
        selected!(self, host_facts(operation))
    }
    fn write_host_facts(&self, operation: WorkspaceOperationView<'_>,
        destination: WorkspaceHostDestination<'_>)
        -> Result<Option<WorkspaceHostFacts>, Self::Error> {
        selected!(self, write_host_facts(operation, destination))
    }
}
