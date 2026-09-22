//! Mechanical projection of live frame sources into the common cold driver.
use super::{
    Error, GenerationSampler, MlxKeyValueState, MlxRealtimeExecution, MlxTensor, RandomState,
};
use crate::backend::runtime::distributed::topology::original_source::parallel::OriginalParallelInvocation;
use crate::{
    backend::{
        nn::workspace::{
            ExistingArrayProjection, MlxParallelWorkspace, MlxParallelWorkspaceMechanisms,
            ProjectedNativeStorage, ResidentExecutionMechanisms, SpeculativeNumericalRecipe,
        },
        runtime::execution::generic::LayerwiseWorkspace,
    },
    composition::moshi::{
        RealtimeOperationPlan, RealtimeOperationRecipe, RealtimeWorkspaceVisitor,
    },
};
use eredu_core::{RealtimeFrameScheduleState, RealtimeSampling};
use eredu_nn::workspace::{
    HostMetadataFunding, WorkspaceContext, WorkspaceMetadataError, WorkspaceTensor,
};
use eredu_runtime::{
    working_memory::{HostSourceConstructionFacts, WorkspaceSamplingRandomState},
    RealtimeIngressSource, RealtimePayloadContract, RealtimePayloadHistory,
};
use safemlx::{PreparedInputRuntime, Stream};
use std::{
    mem::{size_of, size_of_val},
    num::NonZeroU32,
};

#[derive(Debug, thiserror::Error)]
#[error("realtime retained frame source {stage}: {cause}")]
struct SourceFailure {
    stage: &'static str,
    #[source]
    cause: Error,
}
fn source_failure(context: &WorkspaceContext, stage: &'static str, cause: Error) -> Error {
    // The existing cold account pays the concrete source/error representation
    // before retention; static stage labels create no formatted allocation.
    Error::Neural(context.metadata_source(SourceFailure { stage, cause }))
}

// Diagnose the first actual unknown allocation without evaluation or copying.
// The descriptor loan and message are admitted by the existing cold account.
fn incomplete_witness(
    context: &WorkspaceContext,
    role: &'static str,
    ordinal: usize,
    array: &safemlx::Array,
) -> Option<Error> {
    let controls = size_of::<(&WorkspaceContext, &str, usize, &safemlx::Array)>()
        .checked_add(size_of::<Option<Error>>())?;
    if let Err(cause) = context.charge_metadata(controls) {
        return Some(Error::Neural(cause.into()));
    }
    match array.try_descriptor() {
        Err(cause)=>Some(Error::Neural(context.metadata_source(cause))),
        Ok(descriptor) if descriptor.facts().allocation().is_none()=>Some(Error::Neural(
            context.metadata_error(format_args!("realtime {role} backing incomplete at value {ordinal}; shape {:?}; dtype {:?}; allocation missing",
                descriptor.shape(),descriptor.facts().dtype())))),
        Ok(_)=>None,
    }
}

/// Borrowed canonical owners before the unpublished branch is made. Equal
/// geometry or token values cannot stand in for these native sources.
pub(crate) struct RealtimeNativeFrameSource<'a> {
    pub(crate) state: &'a MlxKeyValueState,
    pub(crate) history: &'a RealtimePayloadHistory<MlxTensor>,
    pub(crate) schedule: &'a RealtimeFrameScheduleState,
    pub(crate) sampling: RealtimeSampling,
    pub(crate) samplers: &'a [GenerationSampler],
    pub(crate) random: Option<&'a RandomState>,
}
/// Selected numerical recipe and retained native witnesses. Wrapper, input,
/// observation and branch populations remain separate admission domains.
pub(crate) struct CompiledRealtimeFrameSource {
    pub(crate) operations: RealtimeOperationRecipe,
    pub(super) parallel: Option<OriginalParallelInvocation>,
    pub(crate) host: super::original_observation::RealtimeHostReadPlan,
    pub(crate) initialized_inputs: HostSourceConstructionFacts,
    pub(super) source_program: super::source_program::FrameSourceProgram,
    pub(crate) completion_roots: usize,
    pub(crate) coordinator: eredu_runtime::RealtimeCoordinatorHostSource,
    pub(crate) ingress: eredu_runtime::RealtimeIngressContract,
    pub(crate) shape: FrameShape,
    _state: ProjectedNativeStorage,
    _payloads: ProjectedNativeStorage,
    _funding: HostMetadataFunding,
}
/// Descriptive branch-selection facts. Token values remain borrowed from the
/// actual scheduler frame by the initialized-input constructor; they are never
/// substituted with these shape facts.
pub(super) struct FrameShape {
    batch: usize,
    input: usize,
    audio: Option<usize>,
    text: Option<usize>,
    mask: Option<Vec<bool>>,
    diagnostics: bool,
}
impl FrameShape {
    fn prepare(
        frame: &eredu_core::RealtimeInputFrame,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        context
            .charge_metadata(size_of::<Self>())
            .map_err(|e| Error::Neural(e.into()))?;
        let mask = frame
            .forced_generated_audio_codebooks()
            .map(|values| {
                let mut owned = context.metadata_vec(values.len())?;
                owned.extend_from_slice(values);
                Ok::<_, eredu_nn::Error>(owned)
            })
            .transpose()
            .map_err(Error::Neural)?;
        Ok(Self {
            batch: frame.batch(),
            input: frame.input_audio_tokens().len(),
            audio: frame.forced_generated_audio_tokens().map(<[i32]>::len),
            text: frame.forced_text_tokens().map(<[i32]>::len),
            mask,
            diagnostics: frame.retains_diagnostics(),
        })
    }
    pub(super) fn matches(&self, frame: &eredu_core::RealtimeInputFrame) -> bool {
        self.batch == frame.batch()
            && self.input == frame.input_audio_tokens().len()
            && self.audio == frame.forced_generated_audio_tokens().map(<[i32]>::len)
            && self.text == frame.forced_text_tokens().map(<[i32]>::len)
            && self.mask.as_deref() == frame.forced_generated_audio_codebooks()
            && self.diagnostics == frame.retains_diagnostics()
    }
}
pub(super) struct RetainedFrameSources {
    pub(super) parallel: Option<OriginalParallelInvocation>,
    _state: ProjectedNativeStorage,
    _payloads: ProjectedNativeStorage,
    _funding: HostMetadataFunding,
}
impl CompiledRealtimeFrameSource {
    pub(super) fn funding(&self) -> &HostMetadataFunding {
        &self._funding
    }
    pub(super) fn into_parts(
        self,
    ) -> (
        RealtimeOperationRecipe,
        super::original_observation::RealtimeHostReadPlan,
        RetainedFrameSources,
        eredu_runtime::RealtimeIngressContract,
        FrameShape,
    ) {
        (
            self.operations,
            self.host,
            RetainedFrameSources {
                parallel: self.parallel,
                _state: self._state,
                _payloads: self._payloads,
                _funding: self._funding,
            },
            self.ingress,
            self.shape,
        )
    }
}
#[derive(Debug, thiserror::Error)]
enum MapFailure {
    #[error(transparent)]
    Funding(#[from] eredu_core::HostMetadataFundingError),
    #[error(transparent)]
    Neural(#[from] eredu_nn::Error),
}
struct Visitor<'a> {
    native: RealtimeNativeFrameSource<'a>,
    ingress: RealtimeIngressSource<'a>,
    payload: &'a RealtimePayloadContract,
    runtime: &'a PreparedInputRuntime,
    context: &'a WorkspaceContext,
    mechanism: ResidentExecutionMechanisms,
    parallel: Option<&'a MlxParallelWorkspace>,
    operations: Option<RealtimeOperationPlan>,
    result: Option<CompiledRealtimeFrameSource>,
}
impl RealtimeWorkspaceVisitor for Visitor<'_> {
    fn visit(
        &mut self,
        model: &mut eredu_architectures::moshi::PreparedMoshiWorkspaceFrame<'_>,
        parameters: &mut ExistingArrayProjection<'_>,
        layerwise: Option<&LayerwiseWorkspace>,
    ) -> Result<(), Error> {
        let context = self.context;
        let invalid = |reason: &'static str| {
            Error::Neural(context.metadata_error(format_args!(
                "realtime retained source {reason}; frontier {:?}; history values {}",
                self.native.schedule.frontier(),
                self.native.history.len()
            )))
        };
        if self.result.is_some() {
            return Err(invalid("visitor repeated"));
        }
        if !parameters.is_complete() {
            return Err(invalid("parameter backing incomplete"));
        }
        let operations = self
            .operations
            .take()
            .ok_or_else(|| invalid("operation plan already consumed"))?;
        let batch = NonZeroU32::new(
            u32::try_from(self.ingress.frame().batch())
                .map_err(|_| invalid("batch extent overflow"))?,
        )
        .ok_or_else(|| invalid("zero batch"))?;
        let mut state = self
            .native
            .state
            .project_resident_workspace_with_storage(batch, context)
            .map_err(|cause| source_failure(context, "state projection", Error::Neural(cause)))?;
        if !state.storage.is_complete() {
            let mut cause = None;
            let mut ordinal = 0usize;
            <MlxKeyValueState as eredu_runtime::RuntimeState<
                crate::backend::nn::shared::MlxNeuralBackend,
            >>::visit_all_retained_values(self.native.state, &mut |value| {
                if cause.is_none() {
                    cause = incomplete_witness(context, "state", ordinal, value.as_array());
                }
                ordinal = ordinal.saturating_add(1);
            })
            .map_err(|error| Error::Neural(context.metadata_source(error)))?;
            return Err(cause.unwrap_or_else(|| invalid("projected state backing incomplete")));
        }
        let count = self
            .native
            .history
            .len()
            .checked_add(usize::from(self.native.random.is_some()))
            .ok_or_else(|| invalid("payload source count overflow"))?;
        let mut projection = ExistingArrayProjection::with_source_count(context, count)
            .map_err(|cause| Error::Neural(context.metadata_source(cause)))?;
        let funding = context
            .metadata_funding()
            .ok_or_else(|| invalid("metadata funding missing"))?;
        let mut ordinal = 0usize;
        let history=self.native.history.try_map_with(&funding,&mut |value| {
            let projected=projection.project(value.as_array()).map_err(MapFailure::from)?;
            if !projection.is_complete() {
                return Err(MapFailure::Neural(context.metadata_error(format_args!(
                    "realtime history backing incomplete at value {ordinal}; shape {:?}; dtype {:?}; frontier {:?}",
                    projected.layout().shape(),projected.layout().dtype(),self.native.schedule.frontier()))));
            }
            ordinal=ordinal.checked_add(1).ok_or_else(||MapFailure::Neural(WorkspaceMetadataError::Overflow.into()))?;
            Ok(projected)
        }).map_err(|cause|source_failure(context,"history projection",Error::Neural(context.metadata_source(cause))))?;
        let random = self
            .native
            .random
            .map(|value| {
                projection
                    .project(value.as_array())
                    .and_then(WorkspaceSamplingRandomState::from_key)
            })
            .transpose()
            .map_err(|cause| {
                source_failure(context, "random key projection", Error::Neural(cause))
            })?;
        if !projection.is_complete() {
            return Err(self
                .native
                .random
                .and_then(|value| incomplete_witness(context, "random key", 0, value.as_array()))
                .unwrap_or_else(|| invalid("history/random backing incomplete")));
        }
        let payloads = projection.try_into_storage().map_err(|cause| {
            source_failure(
                context,
                "history/random retained storage",
                Error::Neural(cause),
            )
        })?;
        let source = super::frame_trace::RealtimeFrameTraceSource {
            ingress: self.ingress,
            payload: self.payload,
            state: &mut state.state,
            history: &history,
            schedule: self.native.schedule,
            sampling: self.native.sampling,
            samplers: self.native.samplers,
            random: random.as_ref(),
            native_alias_bytes: MlxTensor::host_clone_bytes()
                .ok_or_else(|| invalid("tensor alias control overflow"))?,
            random_copy_bytes: self
                .native
                .random
                .map(|_| {
                    RandomState::host_clone_bytes()
                        .ok_or_else(|| invalid("random alias control overflow"))
                })
                .transpose()?,
        };
        let traced = super::frame_trace::trace_frame(model, source, self.runtime, context)
            .map_err(|cause| source_failure(context, "complete shared coordinator trace", cause))?;
        let parallel = self
            .parallel
            .map(|source| source.prepare_invocation(&traced.report))
            .transpose()
            .map_err(Error::Neural)?;
        let capture = crate::backend::array_copy::CaptureNativePopulation::default();
        let numerical = match parallel.as_ref() {
            Some(parallel) => {
                SpeculativeNumericalRecipe::inspect_owned_parallel_child_with_sources(
                    &traced.report,
                    traced.completion_roots,
                    self.mechanism,
                    context,
                    capture,
                    layerwise,
                    parallel,
                )
            }
            None => SpeculativeNumericalRecipe::inspect_owned_child_with_sources(
                &traced.report,
                traced.completion_roots,
                self.mechanism,
                context,
                capture,
                layerwise,
            ),
        }
        .map_err(|cause| {
            source_failure(
                context,
                "native numerical qualification",
                Error::Neural(cause),
            )
        })?;
        let operations = operations
            .qualify(numerical, layerwise, context)
            .map_err(|cause| source_failure(context, "selected operation qualification", cause))?;
        let source_program = super::source_program::FrameSourceProgram::prepare(
            traced.initialized_inputs,
            operations.source_facts(),
            context,
        )
        .map_err(|cause| source_failure(context, "input/operation source program", cause))?;
        let shape = FrameShape::prepare(self.ingress.frame(), context)
            .map_err(|cause| source_failure(context, "retained frame shape", cause))?;
        let schedule = self
            .ingress
            .contract()
            .schedule()
            .try_clone_with_host_source(&funding)
            .map_err(|cause| Error::Neural(context.metadata_source(cause)))?;
        let ingress = eredu_runtime::RealtimeIngressContract::new(
            schedule,
            self.ingress.contract().text_domain(),
            self.ingress.contract().audio_domain(),
        )
        .map_err(|cause| Error::Neural(context.metadata_source(cause)))?;
        self.result = Some(CompiledRealtimeFrameSource {
            operations,
            parallel,
            source_program,
            host: traced.host,
            initialized_inputs: traced.initialized_inputs,
            completion_roots: traced.completion_roots,
            coordinator: traced.coordinator,
            ingress,
            shape,
            _state: state.storage,
            _payloads: payloads,
            _funding: funding,
        });
        Ok(())
    }
}
/// Projects actual sources and qualifies their operation slot without creating
/// a device, acquiring a unit, copying native data or advancing state.
#[allow(clippy::too_many_arguments)]
pub(crate) fn compile_frame_source(
    model: &MlxRealtimeExecution,
    native: RealtimeNativeFrameSource<'_>,
    ingress: RealtimeIngressSource<'_>,
    payload: &RealtimePayloadContract,
    runtime: &PreparedInputRuntime,
    stream: &Stream,
    pool: &eredu_runtime::working_memory::MemoryLedger,
    mechanism: ResidentExecutionMechanisms,
    context: &WorkspaceContext,
) -> Result<CompiledRealtimeFrameSource, Error> {
    let frames = [
        size_of::<Visitor<'_>>(),
        size_of::<CompiledRealtimeFrameSource>(),
        size_of::<Result<CompiledRealtimeFrameSource, Error>>(),
        size_of::<RealtimeNativeFrameSource<'_>>(),
        size_of::<ProjectedNativeStorage>() * 2,
        size_of::<RealtimePayloadHistory<WorkspaceTensor>>(),
        size_of::<Option<WorkspaceSamplingRandomState>>(),
        size_of::<RealtimeOperationPlan>(),
        size_of::<Option<MlxParallelWorkspace>>(),
        size_of::<Result<MlxParallelWorkspace, eredu_nn::Error>>(),
        size_of::<Option<OriginalParallelInvocation>>(),
    ];
    context
        .charge_metadata(
            frames
                .into_iter()
                .try_fold(size_of_val(&frames), usize::checked_add)
                .ok_or_else(|| Error::Neural(WorkspaceMetadataError::Overflow.into()))?,
        )
        .map_err(|cause| Error::Neural(cause.into()))?;
    let funding = context
        .metadata_funding()
        .ok_or_else(|| Error::Neural(WorkspaceMetadataError::Unqualified.into()))?;
    let parallel = model
        .parallel_communication()
        .map(|(communication, id)| {
            let source = communication.original_initialized_tensor_source(id, pool, &funding)?;
            MlxParallelWorkspaceMechanisms::new(mechanism, source)
                .prepare_workspace()
                .map_err(Error::Neural)
        })
        .transpose()?;
    let context = parallel
        .as_ref()
        .map_or(context, MlxParallelWorkspace::context);
    let operations = model
        .realtime_operation_plan(stream, mechanism.allocation(), pool, context)
        .map_err(|cause| source_failure(context, "selected operation source", cause))?;
    let mut visitor = Visitor {
        native,
        ingress,
        payload,
        runtime,
        context,
        mechanism,
        parallel: parallel.as_ref(),
        operations: Some(operations),
        result: None,
    };
    model
        .with_workspace_frame(mechanism.allocation(), context, &mut visitor)
        .map_err(|cause| source_failure(context, "model source visit", cause))?;
    visitor.result.ok_or_else(|| {
        Error::Neural(context.metadata_error(format_args!(
            "selected realtime source did not execute its complete cold visit"
        )))
    })
}

use eredu_nn::workspace::WorkspaceMetadataAllocation;
