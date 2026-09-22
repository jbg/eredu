//! Quote the implementation already carried by the actual borrowed stream.
use super::{
    MlxMetalWorkspaceMechanisms, MlxWorkspaceFactError, NativeAllocationFacts,
    ResidentRecipeRecorder,
};
use crate::backend::nn::workspace::{MlxCpuMatmulMechanism, MlxCpuWorkspaceMechanisms};
use eredu_nn::CpuMatmulImplementation;
use eredu_nn::workspace::*;
use eredu_nn::{
    Error,
    workspace::{HostMetadataFunding, WorkspaceContext, WorkspaceMetadataError},
};
use safemlx::{CpuMatmulKernel, DeviceType, Stream, StreamCopyPlan};
use std::mem::{size_of, size_of_val};
#[path = "resident_mechanism/allocation_population.rs"]
mod allocation_population;

/// Fixed equation causes remain allocation-free; prepared populations retain
/// their actual metadata payer when source construction fails.
#[derive(Debug, thiserror::Error)]
pub enum MlxWorkspacePreparationError {
    /// Allocation-free equation or descriptor failure.
    #[error(transparent)]
    Fixed(#[from] MlxWorkspaceFactError),
    /// Source preparation failure retaining its actual host metadata payer.
    #[error("native allocation source preparation failed: {cause}")]
    Source {
        /// Original neutral workspace failure.
        #[source]
        cause: Error,
        /// Funding retained until the source failure is destroyed.
        _funding: Option<HostMetadataFunding>,
    },
}
pub(crate) type ResidentFactError = MlxWorkspacePreparationError;
impl MlxWorkspacePreparationError {
    fn source(cause: Error, funding: Option<&HostMetadataFunding>) -> Self {
        Self::Source {
            cause,
            _funding: funding.cloned(),
        }
    }
}

#[derive(Debug, thiserror::Error)]
#[error("resident equation stream source: {cause}")]
struct StreamFailure {
    #[source]
    cause: safemlx::StreamCopyCause,
    _funding: HostMetadataFunding,
}

/// Cold facts only. The caller keeps its exact environment/source loan through
/// native execution; this value never creates or modifies a native context.
#[derive(Clone, Copy, Debug)]
pub(crate) enum ResidentExecutionMechanisms {
    Metal(MlxMetalWorkspaceMechanisms),
    Cpu {
        ordinary: MlxMetalWorkspaceMechanisms,
        cpu: MlxCpuWorkspaceMechanisms,
    },
}
impl ResidentExecutionMechanisms {
    /// Selected allocation mechanism only; this is not an execution grant.
    pub(crate) fn uses_original_storage(self) -> bool {
        self.allocation().original_storage
    }
    /// Keeps the selected stream's operators while quoting ordinary backing
    /// allocation and publication instead of a prepaid original arena.
    pub(crate) const fn ordinary_storage(self) -> Self {
        match self {
            Self::Metal(facts) => Self::Metal(facts.ordinary_storage()),
            Self::Cpu { ordinary, cpu } => Self::Cpu {
                ordinary: ordinary.ordinary_storage(),
                cpu: cpu.ordinary_storage(),
            },
        }
    }
    /// Loading retains the immutable implementation of this exact stream.
    /// This cold query creates no device, stream, tensor or admission authority.
    pub(crate) fn from_cold_stream(
        ordinary: MlxMetalWorkspaceMechanisms,
        stream: &Stream,
    ) -> Result<Self, Error> {
        let source = StreamCopyPlan::<()>::capture(stream).map_err(Error::backend)?;
        Self::from_source(ordinary, &source).map_err(Into::into)
    }
    fn from_source(
        ordinary: MlxMetalWorkspaceMechanisms,
        source: &StreamCopyPlan<()>,
    ) -> Result<Self, WorkspaceMetadataError> {
        Ok(match source.device_type() {
            DeviceType::Gpu => Self::Metal(ordinary),
            DeviceType::Cpu => {
                let implementation = match source.cpu_matmul() {
                    CpuMatmulKernel::PlatformDefault => CpuMatmulImplementation::PlatformDefault,
                    CpuMatmulKernel::Float32Tiles => CpuMatmulImplementation::Float32Tiles,
                    CpuMatmulKernel::Float32AndFloat16Tiles => {
                        CpuMatmulImplementation::Float32AndFloat16Tiles
                    }
                };
                let matmul = MlxCpuMatmulMechanism::select(implementation)
                    .ok_or(WorkspaceMetadataError::Unqualified)?;
                Self::Cpu {
                    ordinary,
                    cpu: MlxCpuWorkspaceMechanisms::new(ordinary.allocation(), matmul),
                }
            }
        })
    }
    /// Common allocator and native host facts. Equation quotation must use the
    /// complete retained enum, never infer a GPU worker from these shared facts.
    pub(crate) fn ordinary(self) -> MlxMetalWorkspaceMechanisms {
        match self {
            Self::Metal(facts)
            | Self::Cpu {
                ordinary: facts, ..
            } => facts,
        }
    }
    pub(crate) fn allocation(self) -> NativeAllocationFacts {
        self.ordinary().allocation()
    }
    pub(crate) fn metal(self) -> Option<MlxMetalWorkspaceMechanisms> {
        match self {
            Self::Metal(facts) => Some(facts),
            Self::Cpu { .. } => None,
        }
    }
    pub(crate) fn from_stream(
        ordinary: MlxMetalWorkspaceMechanisms,
        stream: &Stream,
        funding: &HostMetadataFunding,
    ) -> Result<Self, Error> {
        // The fixed source queries have no owning side effects. Price their
        // actual scalar transports before reading the retained stream value.
        let selected = MlxCpuMatmulMechanism::select(CpuMatmulImplementation::Float32Tiles);
        let default = MlxCpuMatmulMechanism::select(CpuMatmulImplementation::PlatformDefault)
            .ok_or(WorkspaceMetadataError::Overflow)?;
        let selection_controls = selected
            .unwrap_or(default)
            .control_bytes::<()>()
            .ok_or(WorkspaceMetadataError::Overflow)?;
        let frames = [
            selection_controls,
            Stream::device_type_control_bytes().ok_or(WorkspaceMetadataError::Overflow)?,
            size_of::<Self>(),
            size_of::<Result<Self, Error>>(),
            size_of::<Option<MlxCpuMatmulMechanism>>(),
            size_of::<MlxCpuMatmulMechanism>(),
            size_of::<MlxCpuWorkspaceMechanisms>(),
            size_of::<StreamCopyPlan<()>>(),
            size_of::<Result<StreamCopyPlan<()>, safemlx::StreamCopyCause>>(),
            size_of::<DeviceType>(),
            size_of::<CpuMatmulKernel>(),
            WorkspaceContext::metadata_source_bytes::<StreamFailure>()
                .ok_or(WorkspaceMetadataError::Overflow)?,
            size_of::<(Self, HostMetadataFunding)>(),
            size_of::<(MlxMetalWorkspaceMechanisms, &Stream, &HostMetadataFunding)>(),
        ];
        funding
            .reserve_metadata(
                frames
                    .into_iter()
                    .try_fold(size_of_val(&frames), usize::checked_add)
                    .ok_or(WorkspaceMetadataError::Overflow)?,
            )
            .map_err(WorkspaceMetadataError::Funding)?;
        let source = StreamCopyPlan::<()>::capture(stream).map_err(|cause| {
            Error::backend_retained_source(StreamFailure {
                cause,
                _funding: funding.clone(),
            })
        })?;
        Self::from_source(ordinary, &source).map_err(Into::into)
    }

    pub(crate) fn context(
        self,
        funding: HostMetadataFunding,
    ) -> Result<WorkspaceContext, WorkspaceMetadataError> {
        WorkspaceContext::new_with_metadata_funding(self, funding)
    }
    pub(crate) fn recorder(
        self,
        geometry: eredu_core::InferenceGeometry,
        context: &WorkspaceContext,
    ) -> Result<ResidentRecipeRecorder, Error> {
        let parts = [
            size_of::<Self>(),
            size_of::<eredu_core::InferenceGeometry>(),
            size_of::<ResidentRecipeRecorder>(),
            size_of::<Result<ResidentRecipeRecorder, Error>>(),
            size_of::<(Self, eredu_core::InferenceGeometry, &WorkspaceContext)>(),
        ];
        context.charge_metadata(
            parts
                .into_iter()
                .try_fold(size_of_val(&parts), usize::checked_add)
                .ok_or(WorkspaceMetadataError::Overflow)?,
        )?;
        match self {
            Self::Metal(ordinary) => {
                ResidentRecipeRecorder::with_context(geometry, ordinary, context)
            }
            Self::Cpu { ordinary, cpu } => {
                ResidentRecipeRecorder::with_cpu_context(geometry, ordinary, cpu, context)
            }
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
    fn prepare_allocation_sources(
        &self,
        operation: WorkspaceOperationView<'_>,
        context: &WorkspaceContext,
    ) -> Result<Option<WorkspaceOperationAllocationSources>, Error> {
        context.charge_metadata(size_of::<(
            Self,
            WorkspaceOperationView<'_>,
            super::facts::Emitter<'_>,
            allocation_population::Description,
            Option<HostMetadataFunding>,
            Option<allocation_population::AllocationPopulations>,
            Result<Option<WorkspaceOperationAllocationSources>, Error>,
        )>())?;
        allocation_population::AllocationPopulations::prepare(*self, operation, context)
            .map_err(Error::backend_retained_source)?
            .map(|source| source.into_operation_sources(operation.outputs.len(), context))
            .transpose()
    }

    fn scratch_allocation_count(
        &self,
        operation: WorkspaceOperationView<'_>,
    ) -> Result<Option<usize>, Error> {
        match self {
            Self::Metal(facts) => facts.scratch_allocation_count(operation),
            Self::Cpu { cpu, .. } => cpu.scratch_allocation_count(operation),
        }
    }

    fn completion_strategy(&self) -> WorkspaceCompletionStrategy {
        if self.uses_original_storage() {
            WorkspaceCompletionStrategy::EnclosingSubmission
        } else {
            WorkspaceCompletionStrategy::OperationSubmissions
        }
    }
    fn allocation_host_control_bytes(
        &self,
        op: WorkspaceOperationView<'_>,
        output: usize,
    ) -> Option<u64> {
        match self {
            Self::Metal(facts) => {
                WorkspaceMechanisms::allocation_host_control_bytes(facts, op, output)
            }
            Self::Cpu { cpu, .. } => {
                WorkspaceMechanisms::allocation_host_control_bytes(cpu, op, output)
            }
        }
    }
    fn scratch_host_control_bytes(
        &self,
        op: WorkspaceOperationView<'_>,
    ) -> Result<Option<u64>, Error> {
        match self {
            Self::Metal(facts) => WorkspaceMechanisms::scratch_host_control_bytes(facts, op),
            Self::Cpu { cpu, .. } => WorkspaceMechanisms::scratch_host_control_bytes(cpu, op),
        }
    }

    fn memory_topology(&self) -> Option<&eredu_core::MemoryTopology> {
        match self {
            Self::Metal(facts) => WorkspaceMechanisms::memory_topology(facts),
            Self::Cpu { cpu, .. } => WorkspaceMechanisms::memory_topology(cpu),
        }
    }
    fn output_placement(
        &self,
        operation: WorkspaceOperationView<'_>,
        output: usize,
    ) -> Option<&eredu_core::MemoryPlacement> {
        match self {
            Self::Metal(facts) => WorkspaceMechanisms::output_placement(facts, operation, output),
            Self::Cpu { cpu, .. } => WorkspaceMechanisms::output_placement(cpu, operation, output),
        }
    }
    fn scratch_placement(
        &self,
        operation: WorkspaceOperationView<'_>,
    ) -> Option<&eredu_core::MemoryPlacement> {
        match self {
            Self::Metal(facts) => WorkspaceMechanisms::scratch_placement(facts, operation),
            Self::Cpu { cpu, .. } => WorkspaceMechanisms::scratch_placement(cpu, operation),
        }
    }
    fn prepared_text_input_dtype(&self) -> Option<WorkspaceDtype> {
        // Both selected native text-input workers consume [u32]: ordinary
        // Array::try_from_slice and original prepared host-token copying. This
        // element declaration is independent of CPU/Metal or model family.
        match <u32 as safemlx::ArrayElement>::DTYPE {
            safemlx::Dtype::Uint32 => Some(WorkspaceDtype::Uint32),
            _ => None,
        }
    }

    fn output_representation(
        &self,
        operation: WorkspaceOperationView<'_>,
        output: usize,
    ) -> Option<WorkspaceRepresentation> {
        selected!(self, output_representation(operation, output))
    }
    fn projection_input_observation_mechanism(
        &self,
        format: &eredu_nn::LinearFormatSpec,
    ) -> Result<Option<eredu_nn::ProjectionInputObservationMechanism>, Error> {
        selected!(self, projection_input_observation_mechanism(format))
    }
    fn grouped_observation_schedule(
        &self,
        bank: &WorkspaceGroupedBank,
        tokens: u32,
    ) -> Result<Option<WorkspaceGroupedObservationSchedule>, Error> {
        selected!(self, grouped_observation_schedule(bank, tokens))
    }
    fn operation_bound(
        &self,
        operation: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, Error> {
        selected!(self, operation_bound(operation))
    }
    fn host_workspace_bound(
        &self,
        operation: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceHostBound>, Error> {
        selected!(self, host_workspace_bound(operation))
    }
}
impl WorkspaceFactMechanisms for ResidentExecutionMechanisms {
    fn allocation_host_control_bytes(
        &self,
        op: WorkspaceOperationView<'_>,
        output: usize,
    ) -> Option<u64> {
        match self {
            Self::Metal(facts) => {
                WorkspaceFactMechanisms::allocation_host_control_bytes(facts, op, output)
            }
            Self::Cpu { cpu, .. } => {
                WorkspaceFactMechanisms::allocation_host_control_bytes(cpu, op, output)
            }
        }
    }
    fn scratch_host_control_bytes(
        &self,
        op: WorkspaceOperationView<'_>,
    ) -> Result<Option<u64>, Self::Error> {
        match self {
            Self::Metal(facts) => {
                WorkspaceFactMechanisms::scratch_host_control_bytes(facts, op).map_err(Into::into)
            }
            Self::Cpu { cpu, .. } => {
                WorkspaceFactMechanisms::scratch_host_control_bytes(cpu, op).map_err(Into::into)
            }
        }
    }
    fn memory_topology(&self) -> Option<&eredu_core::MemoryTopology> {
        match self {
            Self::Metal(facts) => WorkspaceFactMechanisms::memory_topology(facts),
            Self::Cpu { cpu, .. } => WorkspaceFactMechanisms::memory_topology(cpu),
        }
    }
    fn output_placement(
        &self,
        operation: WorkspaceOperationView<'_>,
        output: usize,
    ) -> Option<&eredu_core::MemoryPlacement> {
        match self {
            Self::Metal(facts) => {
                WorkspaceFactMechanisms::output_placement(facts, operation, output)
            }
            Self::Cpu { cpu, .. } => {
                WorkspaceFactMechanisms::output_placement(cpu, operation, output)
            }
        }
    }
    fn scratch_placement(
        &self,
        operation: WorkspaceOperationView<'_>,
    ) -> Option<&eredu_core::MemoryPlacement> {
        match self {
            Self::Metal(facts) => WorkspaceFactMechanisms::scratch_placement(facts, operation),
            Self::Cpu { cpu, .. } => WorkspaceFactMechanisms::scratch_placement(cpu, operation),
        }
    }
    type Error = ResidentFactError;
    fn with_prepared_facts<T>(
        &self,
        operation: WorkspaceOperationView<'_>,
        funding: Option<&HostMetadataFunding>,
        visit: impl FnOnce(&dyn WorkspaceFactMechanisms<Error = Self::Error>) -> T,
    ) -> Result<T, Self::Error> {
        let context = match funding {
            Some(funding) => WorkspaceContext::new_with_metadata_funding(*self, funding.clone()),
            None => Ok(WorkspaceContext::new(*self)),
        }
        .map_err(|cause| Self::Error::source(cause.into(), funding))?;
        self.with_prepared_facts_context(operation, &context, visit)
    }
    fn with_prepared_facts_context<T>(
        &self,
        operation: WorkspaceOperationView<'_>,
        context: &WorkspaceContext,
        visit: impl FnOnce(&dyn WorkspaceFactMechanisms<Error = Self::Error>) -> T,
    ) -> Result<T, Self::Error> {
        if matches!(self, Self::Metal(_)) {
            let bytes = size_of::<(
                Self,
                WorkspaceOperationView<'_>,
                &WorkspaceContext,
                super::facts::Emitter<'_>,
                super::facts::DefaultScratchSources,
                allocation_population::Description,
                Option<HostMetadataFunding>,
                allocation_population::PreparedAllocationFacts<'_>,
                Option<allocation_population::AllocationPopulations>,
                ResidentFactError,
                Result<T, ResidentFactError>,
            )>()
            .checked_add(size_of_val(&visit))
            .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
            context.charge_metadata(bytes).map_err(|cause| {
                Self::Error::source(cause.into(), context.metadata_funding().as_ref())
            })?;
            if let Some(source) =
                allocation_population::AllocationPopulations::prepare(*self, operation, context)?
            {
                return Ok(visit(&allocation_population::PreparedAllocationFacts {
                    base: self,
                    source: &source,
                    operation,
                }));
            }
        }
        Ok(visit(self))
    }
    fn operation_facts(
        &self,
        operation: WorkspaceOperationView<'_>,
    ) -> Result<Option<WorkspaceOperationFacts>, Self::Error> {
        match self {
            Self::Metal(facts) => facts.operation_facts(operation),
            Self::Cpu { cpu, .. } => cpu.operation_facts(operation).map_err(Into::into),
        }
    }
    fn write_operation_facts(
        &self,
        operation: WorkspaceOperationView<'_>,
        destination: WorkspaceEffectDestination<'_>,
    ) -> Result<Option<WorkspaceOperationFacts>, Self::Error> {
        match self {
            Self::Metal(facts) => facts.write_operation_facts(operation, destination),
            Self::Cpu { cpu, .. } => cpu
                .write_operation_facts(operation, destination)
                .map_err(Into::into),
        }
    }
    fn host_facts(
        &self,
        operation: WorkspaceOperationView<'_>,
    ) -> Result<Option<WorkspaceHostFacts>, Self::Error> {
        match self {
            Self::Metal(facts) => facts.host_facts(operation),
            Self::Cpu { cpu, .. } => cpu.host_facts(operation).map_err(Into::into),
        }
    }
    fn write_host_facts(
        &self,
        operation: WorkspaceOperationView<'_>,
        destination: WorkspaceHostDestination<'_>,
    ) -> Result<Option<WorkspaceHostFacts>, Self::Error> {
        match self {
            Self::Metal(facts) => facts.write_host_facts(operation, destination),
            Self::Cpu { cpu, .. } => cpu
                .write_host_facts(operation, destination)
                .map_err(Into::into),
        }
    }
}
