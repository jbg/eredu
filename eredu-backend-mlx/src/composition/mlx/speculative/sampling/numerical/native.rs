//! One-use phase compiler over the existing original Scope/recovery engine.
use super::*;
use crate::backend::nn::workspace::{MlxMetalWorkspaceMechanisms, SpeculativeNumericalRecipe};
use crate::backend::submission_recovery::{PreparedRecovery, Retention, Status};
use crate::backend::{OriginalCopyEnvironment, OriginalCopyEnvironmentError};
use crate::composition::mlx::speculative::OriginalSpeculativeNumericalSources;
use eredu_nn::workspace::{
    WorkspaceContext, WorkspaceDtype, WorkspaceTensor, WorkspaceTraceReport,
};
use eredu_runtime::generation::{LogitProgramError, SpeculativeLogitProgram, SpeculativeGreedyProgram, SpeculativeCategoricalProgram};
use crate::backend::runtime::generation::MlxSamplingBackend;
use crate::MlxTensor;
use eredu_runtime::working_memory::{
    WorkspaceSamplingBackend, WorkspaceSamplingRandomState,
    OriginalSpeculativeNumericalBudgetCustody as Custody, SpeculativeNumericalRequirements,
};
use program::SpeculativeProbabilityBackend;
use program::{SpeculativeNumericalKind as Kind, SpeculativeNumericalProgram as Program};
use safemlx::{
    OriginalBufferBudget, OriginalScopeObserver, PrefillRoots, PrefillRootsRuntime,
    PreparedOriginalBufferBudget, PreparedPipelineCache, PreparedPipelineCachePlan,
    PreparedPrefillFailure, PreparedSubmissionGraphQuota, PreparedSubmissionRecordQuota,
    RetainedPrefillFailure, SubmissionGraphQuota, SubmissionRecordQuota,
};
use std::cell::RefCell;
mod placement;
mod tensor_layout;
type InputPlacement<'a> = (crate::composition::mlx::speculative::SpeculativeExecutionStreams<'a>,
    eredu_core::speculative::SamplingPlacement);

#[derive(Debug, thiserror::Error)]
enum Cause {
    #[error("numerical source differs from the accepted request, stream or value kind")]
    Source,
    #[error(transparent)]
    Capture(#[from] eredu_runtime::capture::FundedCaptureError<crate::backend::array_copy::CaptureTensorNativeError>),
    #[error(transparent)]
    CaptureNative(#[from] crate::backend::array_copy::CaptureTensorNativeError),
    #[error(transparent)]
    CaptureDrain(#[from] eredu_runtime::capture::FundedCaptureDrainError),
    #[error(transparent)]
    Mask(#[from]eredu_runtime::generation::TokenMaskError),
    #[error("token mask destination allocation failed: {0}")]
    MaskAllocation(#[source]std::collections::TryReserveError),
    #[error("numerical phase has an incomplete producer bound: {0}")]
    Unknown(&'static str),
    #[error("numerical phase geometry overflow")]
    Overflow,
    #[error("numerical phase failed or its completion is not established")]
    Completion,
    #[error("numerical recovery did not complete successfully: {0:?}")]
    RecoveryStatus(Status),
    #[error("numerical exact-owner record retirement did not complete: {0:?}")]
    Retirement(safemlx::SubmissionRetirement),
    #[error(transparent)]
    Policy(#[from] LogitProgramError<Exception>),
    #[error(transparent)]
    TokenInput(#[from] safemlx::OriginalPromptInputCause),
    #[error(transparent)]
    StreamCopy(#[from] safemlx::StreamCopyCause),
    #[error(transparent)]
    Environment(#[from] OriginalCopyEnvironmentError),
    #[error(transparent)]
    Program(#[from] program::SpeculativeNumericalError),
    #[error(transparent)]
    Buffer(#[from] safemlx::OriginalBufferCause),
    #[error(transparent)]
    Graph(#[from] safemlx::SubmissionGraphQuotaCause),
    #[error(transparent)]
    Record(#[from] safemlx::SubmissionRecordQuotaCause),
    #[error(transparent)]
    Pipeline(#[from] safemlx::PipelineCacheCause),
    #[error(transparent)]
    Failure(#[from] safemlx::PrefillFailureCause),
    #[error(transparent)]
    Scope(#[from] safemlx::SubmissionScopeOwnerCause),
    #[error(transparent)]
    Controls(#[from] safemlx::OriginalNativeControlError),
    #[error(transparent)]
    Roots(#[from] safemlx::PrefillRootsError),
    #[error(transparent)]
    RootConstruction(#[from] safemlx::PrefillRootsCause),
    #[error("numerical {cut} {side} input is not a completed source: {source}")]
    Input {
        cut: &'static str,
        side: &'static str,
        #[source]
        source: Exception,
    },
    #[error(transparent)]
    Native(#[from] Exception),
    #[error(transparent)]
    Backend(#[from] Error),
    #[error(transparent)]
    Slice(#[from] safemlx::error::AsSliceError),
    #[error("numerical observation unavailable: {0:?}")]
    Observation(safemlx::ScopedSubmissionProgress),
}
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct Failure {
    #[source]
    cause: Cause,
    custody: Custody,
    funding: WorkspaceMetadataFunding,
}
fn failure(cause: Cause, custody: &Custody, funding: &WorkspaceMetadataFunding) -> Error {
    // Exact closed error owner paid by this phase before any native constructor.
    Error::with_original_control_source(
        eredu_core::BackendFailure::from_error(Failure {
            cause,
            custody: custody.clone(),
            funding: funding.clone(),
        }).with_operation("execute original speculative sampling"),
        false,
    )
}
/// Results preserve their meaning; corrected logits may feed another fixed
/// probability program, while a normalized distribution is not another logit.
#[derive(Debug)]
pub(crate) enum NumericalOutput {
    Key(OriginalNumericalKey),
    KeyPair { current: OriginalNumericalKey, next: OriginalNumericalKey },
    RandomToken { token: u32, advanced: OriginalNumericalKey },
    RandomUnitInterval { value: f32, advanced: OriginalNumericalKey },
    Token(u32),
    Tensor(OriginalNumericalValue),
    Logits(OriginalNumericalValue),
    CapturedLogits { value: OriginalNumericalValue, capture: eredu_core::capture::SharedCapturedStep },
    Distribution(OriginalNumericalValue),
    Probability(f32),
    Correction(Option<OriginalNumericalValue>),
}
pub(crate) struct NumericalProducer;
#[derive(Clone, Copy)]
enum Cut {
    TokenIds(u32),
    TensorRange { axis: u8, start: u32, end: u32, tokens: bool },
    TensorConcatenate,
    CreateKey(u64),
    NextKey,
    UniformUnitInterval,
    KeyAt(u32),
    Categorical(SpeculativeCategoricalProgram),
    Greedy(SpeculativeGreedyProgram),
    Policy(SpeculativeLogitProgram),
    Normalize,
    LogitsRow(u32),
    Probability(u32),
    Difference,
    Logarithm,
}
impl Cut {
    fn name(self) -> &'static str {
        match self {
            Self::TokenIds(_) => "token input",
            Self::TensorRange { tokens: true, .. } => "token range",
            Self::TensorRange { tokens: false, .. } => "tensor range",
            Self::TensorConcatenate => "tensor concatenation",
            Self::CreateKey(_) => "random key",
            Self::NextKey => "next random key",
            Self::UniformUnitInterval => "uniform draw",
            Self::KeyAt(_) => "random key selection",
            Self::Categorical(_) => "categorical sampling",
            Self::Greedy(_) => "greedy sampling",
            Self::Policy(_) => "logit policy",
            Self::Normalize => "normalization",
            Self::LogitsRow(_) => "logits row",
            Self::Probability(_) => "probability selection",
            Self::Difference => "probability difference",
            Self::Logarithm => "logarithm",
        }
    }
}
struct Plan {
    cut: Cut,
    recipe: SpeculativeNumericalRecipe,
    physical: usize,
    graph_bytes: u64,
    record_bytes: u64,
    controls: u64,
    pipeline: PreparedPipelineCachePlan,
}
struct Planning {
    reports: [Option<WorkspaceTraceReport>; 2],
    context: WorkspaceContext,
    capture: Option<capture::Plan>,
    funding: WorkspaceMetadataFunding,
}
// Explicit closed Rc retirement frees its shell before the funding it owns.
struct PlanningOwner(Option<Rc<Planning>>);
impl Clone for PlanningOwner {
    fn clone(&self) -> Self {
        Self(Some(Rc::clone(self.0.as_ref().expect("live planning"))))
    }
}
impl Drop for PlanningOwner {
    fn drop(&mut self) {
        if let Some(value) = self.0.take() {
            drop(Rc::into_inner(value));
        }
    }
}
#[derive(Clone)]
struct Roots(Rc<RefCell<PrefillRoots>>);
struct Domains {
    graph: SubmissionGraphQuota,
    record: SubmissionRecordQuota,
    buffer: OriginalBufferBudget,
    _failure: RetainedPrefillFailure,
    pipeline: PreparedPipelineCache<Custody>,
}
struct Retained {
    capture_roots: Option<capture::Roots>,
    roots: Roots,
    domains: Domains,
    left: Option<OriginalNumericalValue>,
    right: Option<OriginalNumericalValue>,
    seed_stream: Option<ValueStream>,
    planning: PlanningOwner,
    custody: Custody,
}
impl Retention for Retained {
    fn observe(&self, _: Status) {}
}
struct CutOutput {
    value: Array,
    budget: OriginalBufferBudget,
    other: Option<Array>,
    scalar: Option<f32>,
    token: Option<u32>,
}

impl NumericalProducer {
    /// Executes only these deterministic shared equations. It neither invokes
    /// nor certifies arbitrary `process_logits` or random sampler callbacks.
    pub(crate) fn execute(
        sources: &OriginalSpeculativeNumericalSources,
        environment: &OriginalCopyEnvironment<'_>,
        roots_runtime: &PrefillRootsRuntime,
        mechanism: MlxMetalWorkspaceMechanisms,
        kind: Kind,
        left: &OriginalNumericalValue,
        right: Option<&OriginalNumericalValue>,
    ) -> Result<NumericalOutput, Error> {
        Self::execute_with_history(sources, environment, roots_runtime, mechanism,
            kind, left, right, None)
    }
    /// Executes the same worker on the selected environment. An input from
    /// another stream must be a closed completed value in the authenticated
    /// same-device external assignment; no lazy/raw input is imported here.
    pub(crate) fn execute_at(context:crate::composition::mlx::speculative::SpeculativeExecutionStreams<'_>,
        placement:eredu_core::speculative::SamplingPlacement,kind:Kind,
        left:&OriginalNumericalValue,right:Option<&OriginalNumericalValue>,
    )->Result<NumericalOutput,Error>{
        Self::execute_inputs_at(context,placement,kind,Some(left),right)
    }
    pub(super) fn execute_inputs_at(context:crate::composition::mlx::speculative::SpeculativeExecutionStreams<'_>,
        placement:eredu_core::speculative::SamplingPlacement,kind:Kind,
        left:Option<&OriginalNumericalValue>,right:Option<&OriginalNumericalValue>,
    )->Result<NumericalOutput,Error>{
        let (sources,environment)=context.original_numerical_for(placement)
            .ok_or(Error::PrefillControl(eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch))?;
        let (roots,mechanism)=sources.numerical_prerequisites();
        Self::execute_inputs_with_mask(sources,environment,roots,mechanism,kind,left,right,
            None,None,None,None,None,Some((context,placement)))
    }
    pub(crate) fn validate_input_at(value:&OriginalNumericalValue,
        context:crate::composition::mlx::speculative::SpeculativeExecutionStreams<'_>,
        placement:eredu_core::speculative::SamplingPlacement,
    )->Result<(),Error>{
        placement::validate_completed_input(value,context,placement)
    }
    /// The same policy/mask/capture worker with its selected-side source loan.
    pub(super) fn execute_policy_at(context:crate::composition::mlx::speculative::SpeculativeExecutionStreams<'_>,
        placement:eredu_core::speculative::SamplingPlacement,policy:SpeculativeLogitProgram,
        left:&OriginalNumericalValue,history:&[u32],mask:Option<eredu_runtime::generation::TokenMaskPlan<'_>>,
        capture:Option<capture::CaptureSource<'_>>,failed_capture:&mut Option<eredu_core::capture::SharedCapturedStep>,
    )->Result<NumericalOutput,Error>{
        let (sources,environment)=context.original_numerical_for(placement)
            .ok_or(Error::PrefillControl(eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch))?;
        let (roots,mechanism)=sources.numerical_prerequisites();
        Self::execute_inputs_with_mask(sources,environment,roots,mechanism,Kind::ProcessLogits(policy),Some(left),None,
            Some(history),mask,capture,Some(failed_capture),None,Some((context,placement)))
    }
    pub(super) fn execute_with_history(
        sources: &OriginalSpeculativeNumericalSources,
        environment: &OriginalCopyEnvironment<'_>,
        roots_runtime: &PrefillRootsRuntime,
        mechanism: MlxMetalWorkspaceMechanisms,
        kind: Kind,
        left: &OriginalNumericalValue,
        right: Option<&OriginalNumericalValue>,
        history: Option<&[u32]>,
    ) -> Result<NumericalOutput, Error> {
        Self::execute_inputs(sources, environment, roots_runtime, mechanism, kind,
            Some(left), right, history)
    }
    pub(super) fn execute_inputs(
        sources: &OriginalSpeculativeNumericalSources,
        environment: &OriginalCopyEnvironment<'_>,
        roots_runtime: &PrefillRootsRuntime,
        mechanism: MlxMetalWorkspaceMechanisms,
        kind: Kind,
        left: Option<&OriginalNumericalValue>,
        right: Option<&OriginalNumericalValue>,
        history: Option<&[u32]>,
    ) -> Result<NumericalOutput, Error> {
        Self::execute_inputs_with_mask(sources,environment,roots_runtime,mechanism,kind,left,right,history,None,None,None,None,None)
    }
    pub(super) fn execute_controlled_policy(
        sources:&OriginalSpeculativeNumericalSources,environment:&OriginalCopyEnvironment<'_>,roots_runtime:&PrefillRootsRuntime,
        mechanism:MlxMetalWorkspaceMechanisms,policy:SpeculativeLogitProgram,left:&OriginalNumericalValue,
        history:&[u32],mask:eredu_runtime::generation::TokenMaskPlan<'_>,
    )->Result<NumericalOutput,Error> {
        Self::execute_inputs_with_mask(sources,environment,roots_runtime,mechanism,Kind::ProcessLogits(policy),Some(left),None,Some(history),Some(mask),None,None,None,None)
    }
    pub(super) fn execute_captured_policy(
        sources:&OriginalSpeculativeNumericalSources,environment:&OriginalCopyEnvironment<'_>,roots_runtime:&PrefillRootsRuntime,
        mechanism:MlxMetalWorkspaceMechanisms,policy:SpeculativeLogitProgram,left:&OriginalNumericalValue,
        history:&[u32],mask:Option<eredu_runtime::generation::TokenMaskPlan<'_>>,capture:capture::CaptureSource<'_>,
        failed_capture:&mut Option<eredu_core::capture::SharedCapturedStep>,
    )->Result<NumericalOutput,Error> {
        Self::execute_inputs_with_mask(sources,environment,roots_runtime,mechanism,Kind::ProcessLogits(policy),
            Some(left),None,Some(history),mask,Some(capture),Some(failed_capture),None,None)
    }
    pub(super) fn execute_token_ids(
        sources: &OriginalSpeculativeNumericalSources,
        environment: &OriginalCopyEnvironment<'_>,
        roots_runtime: &PrefillRootsRuntime,
        mechanism: MlxMetalWorkspaceMechanisms,
        tokens: &[u32],
    ) -> Result<NumericalOutput, Error> {
        sources.validate_environment(environment)?;
        let length = u32::try_from(tokens.len()).map_err(|_| Error::PrefillControl(
            eredu_runtime::working_memory::WorkingMemoryError::Overflow,
        ))?;
        Self::execute_inputs_with_mask(sources, environment, roots_runtime, mechanism,
            Kind::TokenIds { length }, None, None, None, None, None, None, Some(tokens), None)
    }
    fn execute_inputs_with_mask(
        sources:&OriginalSpeculativeNumericalSources,environment:&OriginalCopyEnvironment<'_>,roots_runtime:&PrefillRootsRuntime,
        mechanism:MlxMetalWorkspaceMechanisms,kind:Kind,left:Option<&OriginalNumericalValue>,right:Option<&OriginalNumericalValue>,
        history:Option<&[u32]>,mask:Option<eredu_runtime::generation::TokenMaskPlan<'_>>,
        capture_source:Option<capture::CaptureSource<'_>>,
        failed_capture:Option<&mut Option<eredu_core::capture::SharedCapturedStep>>,
        tokens: Option<&[u32]>,inputs:Option<InputPlacement<'_>>,
    )->Result<NumericalOutput,Error> {
        let phase_funding = sources.prepare_phase_metadata()?;
        let funding = &phase_funding;
        let result = (|| -> Result<NumericalOutput, Error> {
            sources.validate_environment(environment)?;
            let fixed = planning_control_bytes().ok_or(Error::WorkspacePlanning(
                WorkspaceMetadataFundingError::Overflow,
            ))?;
            funding
                .reserve_metadata(fixed)
                .map_err(Error::WorkspacePlanning)?;
            let cold = |cause| retain_planning_error(cause, funding.clone());
            if let Some((context,placement))=inputs {
                let (selected,destination)=context.original_numerical_for(placement)
                    .ok_or_else(||cold(Cause::Source))?;
                if !std::ptr::eq(selected,sources) || !std::ptr::eq(destination,environment) {
                    return Err(cold(Cause::Source));
                }
            }
            let stream = safemlx::StreamCopyPlan::<()>::capture(environment.stream())
                .map_err(|_| cold(Cause::Source))?;
            let token_shape = if let Kind::TokenIds { length } = kind {
                if tokens.map(<[u32]>::len) != Some(length as usize) { return Err(cold(Cause::Source)); }
                Some([1, i32::try_from(length).map_err(|_| cold(Cause::Source))?])
            } else {
                if tokens.is_some() { return Err(cold(Cause::Source)); }
                None
            };
            let source_shape = match (kind, left) {
                (Kind::TokenIds { .. }, None) => token_shape.as_ref().ok_or_else(|| cold(Cause::Source))?.as_slice(),
                (Kind::CreateKey { .. }, None) => &[2][..],
                (_, Some(left)) => left.value().array.shape(),
                _ => return Err(cold(Cause::Source)),
            };
            let program = Program::new(kind, source_shape).map_err(|e| cold(Cause::Program(e)))?;
            if let Some(capture)=capture_source {capture.validate(sources)?;}
            let capture_host=capture_source.map(|capture|capture.host(program)).transpose()
                .map_err(|cause|sources.retain_startup_error(cause))?;

            if mask.is_some_and(|mask|!matches!(kind,Kind::ProcessLogits(_)) || !mask.matches_shape(source_shape)) { return Err(cold(Cause::Source)); }

            if let Kind::ProcessLogits(policy) = kind {
                if history.map(<[u32]>::len) != Some(policy.history_len()) { return Err(cold(Cause::Source)); }
            }
            let expected = match kind {
                Kind::ProbabilityAt { .. } => Meaning::Probabilities,
                Kind::TokenRange { .. } => Meaning::TokenIds,
                Kind::TensorRange { .. } | Kind::TensorAxisRange { .. } | Kind::TensorConcatenate { .. } => Meaning::Capture,
                Kind::NextKey | Kind::UniformUnitInterval | Kind::KeyAt { .. } => Meaning::RandomKey,
                _ => Meaning::Logits,
            };
            let cpu = stream.device_type() == safemlx::DeviceType::Cpu;
            // CPU sources are closed per actual invocation: eager token input
            // plus fixed-rank views, F32 normalization or known-policy argmax,
            // each with its source-funded completion.
            // ProcessLogits reaches the shared metadata trace: only its proved
            // empty report can return the existing source alias. Any nonempty
            // CPU policy requires its exact recipe in Plan::inspect; the
            // shared vocabulary mask has its own complete CPU Select source.
            // Every other program needs its own complete native recipe.
            if (stream.device_type() != safemlx::DeviceType::Gpu
                && !(cpu && matches!(kind, Kind::TensorConcatenate { .. } | Kind::NextKey | Kind::UniformUnitInterval | Kind::Categorical(_) | Kind::KeyAt { .. } | Kind::CreateKey { .. } | Kind::TokenIds { .. } | Kind::TokenRange { .. }
                    | Kind::TensorRange { .. } | Kind::TensorAxisRange { .. } | Kind::Normalize | Kind::Correction | Kind::Greedy(_) | Kind::ProbabilityAt { .. } | Kind::LogitsRow { .. } | Kind::ProcessLogits(_)) && (capture_source.is_none()||matches!(kind,Kind::ProcessLogits(_)))))
                || usize::from(left.is_some()) + usize::from(right.is_some()) != program.source_count() {
                return Err(cold(Cause::Source));
            }
            for (value, meaning) in left.map(|v|(v,expected)).into_iter().chain(right.map(|v|
                (v, if matches!(kind, Kind::Categorical(_)) { Meaning::RandomKey }
                    else if matches!(kind, Kind::TensorConcatenate { .. }) { Meaning::Capture } else { Meaning::Logits }))) {
                if !value.value().provenance.source().belongs_to_request(sources.request())
                    || value.value().meaning != meaning
                    || (!stream.matches_source(&value.value().stream)
                        && !placement::matches_completed_input(value,inputs,sources,environment,funding)?)
                    || (meaning == Meaning::RandomKey && (value.value().array.shape() != [2]
                        || value.value().array.dtype() != safemlx::Dtype::Uint32)) {
                    return Err(cold(Cause::Source));
                }
            }
            if let Some(right) = right.filter(|_| !matches!(kind, Kind::Categorical(_))) {
                program.validate_source(right.value().array.shape()).map_err(|e| cold(Cause::Program(e)))?;
                if matches!(kind, Kind::TensorConcatenate { .. })
                    && left.is_none_or(|left|left.value().array.dtype()!=right.value().array.dtype()) {
                    return Err(cold(Cause::Source));
                }
            }
            // Composed concatenation, logit policies and capture use the actual CPU model worker.
            // Retain its stream choice and exact completed input layouts; GPU
            // census cannot substitute for these copies or nested completions.
            let cpu_equations = if cpu && matches!(kind,Kind::TensorConcatenate {..}|Kind::ProcessLogits(_)) {
                match crate::backend::nn::workspace::ResidentExecutionMechanisms::from_stream(
                    mechanism,environment.stream(),funding)? {
                    crate::backend::nn::workspace::ResidentExecutionMechanisms::Cpu {cpu,..}=>Some(cpu),
                    _=>return Err(cold(Cause::Source)),
                }
            } else {None};
            let context = match cpu_equations {
                Some(cpu)=>WorkspaceContext::new_with_metadata_funding(cpu,funding.clone()),
                None=>WorkspaceContext::new_with_metadata_funding(mechanism,funding.clone()),
            }.map_err(|cause| Error::Neural(cause.into()))?;
            let metadata = |value: &OriginalNumericalValue| -> Result<WorkspaceTensor, Error> {
                if cpu_equations.is_some() {return tensor_layout::completed_floating(value,&context);}

                let dtype = match (value.value().meaning, value.value().array.dtype()) {
                    (Meaning::RandomKey | Meaning::TokenIds, safemlx::Dtype::Uint32) => WorkspaceDtype::Uint32,
                    (Meaning::TokenIds, safemlx::Dtype::Int32) => WorkspaceDtype::Int32,
                    (_, dtype) => match dtype {
                    safemlx::Dtype::Float32
                    | safemlx::Dtype::Float16
                    | safemlx::Dtype::Bfloat16 => WorkspaceDtype::Float32,
                    _ => return Err(cold(Cause::Source)),
                    },
                };
                Ok(WorkspaceTensor::existing(
                    context.layout(value.value().array.shape(), dtype)?.with_representation(if matches!(kind,Kind::TensorRange{..}|Kind::TensorAxisRange{..}|Kind::TensorConcatenate{..}) {match value.value().array.dtype() {
                        safemlx::Dtype::Float32=>Some(eredu_nn::workspace::WorkspaceRepresentation::new(eredu_nn::workspace::WorkspaceFloatingType::Float32,false)),
                        safemlx::Dtype::Float16=>Some(eredu_nn::workspace::WorkspaceRepresentation::new(eredu_nn::workspace::WorkspaceFloatingType::Float16,false)),
                        safemlx::Dtype::Bfloat16=>Some(eredu_nn::workspace::WorkspaceRepresentation::new(eredu_nn::workspace::WorkspaceFloatingType::Bfloat16,false)),
                        _=>None,
                    }} else {None}),
                    &context,
                )?)
            };
            let left_metadata = left.map(metadata).transpose()?;
            let left_value = || left.ok_or_else(|| cold(Cause::Source));
            let left_meta = || left_metadata.as_ref().ok_or_else(|| cold(Cause::Source));
            let right_metadata = right.map(metadata).transpose()?;
            context.begin_span();
            let capture_plan=capture_source.zip(capture_host.as_ref()).map(|(source,host)|
                capture::Plan::trace(source,host,left_meta()?,&context)).transpose()?;
            let (first_cut, first_report, second_report) = match kind {
                Kind::TokenIds { length } => {
                    // Exact eager source facts are included by Plan::inspect;
                    // this is neither a full/Initialize equation nor a cast.
                    let input = WorkspaceTensor::existing(
                        context.layout(program.shape(), WorkspaceDtype::Uint32)?, &context,
                    )?;
                    (Cut::TokenIds(length), context.finish_report(&[input])?, None)
                }
                Kind::TensorRange { start, end } | Kind::TokenRange { start, end } => {
                    let output = super::tensor::metadata_range(left_meta()?, start, end, &context)?;
                    (Cut::TensorRange { axis: 1, start, end, tokens: matches!(kind, Kind::TokenRange { .. }) },
                        context.finish_report(&[output])?, None)
                }
                Kind::TensorAxisRange { axis, start, end } => {
                    let output = super::tensor::metadata_axis_range(left_meta()?, axis, start, end, &context)?;
                    (Cut::TensorRange { axis, start, end, tokens: false }, context.finish_report(&[output])?, None)
                }
                Kind::TensorConcatenate { .. } => {
                    let output = super::tensor::metadata_concatenate(left_meta()?,
                        right_metadata.as_ref().ok_or_else(||cold(Cause::Source))?, &context)?;
                    (Cut::TensorConcatenate, context.finish_report(&[output])?, None)
                }
                Kind::CreateKey { seed } => {
                    let key = WorkspaceSamplingRandomState::from_seed(&context)?.into_key();
                    (Cut::CreateKey(seed), context.finish_report(&[key])?, None)
                }
                Kind::NextKey => {
                    let mut random = WorkspaceSamplingRandomState::from_key(left_meta()?.clone())?;
                    let next = random.next_key(&context)?;
                    (Cut::NextKey, context.finish_report(&[random.into_key(), next])?, None)
                }
                Kind::UniformUnitInterval => {
                    let mut random = WorkspaceSamplingRandomState::from_key(left_meta()?.clone())?;
                    let draw = random.uniform_unit_interval(&context)?;
                    (Cut::UniformUnitInterval, context.finish_report(&[draw, random.into_key()])?, None)
                }
                Kind::KeyAt { position } => {
                    let key = WorkspaceSamplingRandomState::key_at(left_meta()?, position, &context)?;
                    (Cut::KeyAt(position), context.finish_report(&[key])?, None)
                }
                Kind::Categorical(choice) => {
                    if left_value()?.value().array.dtype() != safemlx::Dtype::Float32 { return Err(cold(Cause::Source)); }
                    let input = left_meta()?.clone();
                    let mut random = WorkspaceSamplingRandomState::from_key(
                        right_metadata.as_ref().ok_or_else(|| cold(Cause::Source))?.clone())?;
                    let token = choice.construct::<WorkspaceSamplingBackend>(&input, &mut random, &context)?;
                    (Cut::Categorical(choice), context.finish_report(&[token, random.into_key()])?, None)
                }
                Kind::Greedy(choice) => {
                    // Existing sampler receipts qualify this actual F32 row.
                    if left_value()?.value().array.dtype() != safemlx::Dtype::Float32 {
                        return Err(cold(Cause::Source));
                    }
                    let input = left_meta()?.clone();
                    let token = choice.construct::<WorkspaceSamplingBackend>(&input, &context)?;
                    (Cut::Greedy(choice), context.finish_report(&[token])?, None)
                }
                Kind::ProcessLogits(policy) => {
                    // Match the actual C source-handle clone used to enter the
                    // ordinary typed sampling worker inside the prepared Graph.
                    let input = capture_plan.as_ref().and_then(|plan|plan.effective.as_ref())
                        .unwrap_or(left_meta()?).clone();
                    let masked=mask.map(|mask|WorkspaceSamplingBackend::apply_prepared_token_mask(&input,mask,&context)).transpose()?;
                    let output = policy.process::<WorkspaceSamplingBackend>(
                        masked.as_ref().unwrap_or(&input), history.ok_or_else(|| cold(Cause::Source))?, &context,
                    ).map_err(|cause| Error::Neural(context.metadata_source(cause)))?;
                    let report = context.finish_report(&[output])?;
                    if report.operations.is_empty() && capture_plan.is_none() {
                        // A proved identity performs no native constructor. The
                        // returned closed Rc alias keeps its original source Q.
                        return Ok(NumericalOutput::Logits(left_value()?.clone()));
                    }
                    // Existing filter/penalty receipts describe the real F32
                    // worker. Other source dtypes keep their closed refusal;
                    // substituting a metadata dtype would not prove this path.
                    if left_value()?.value().array.dtype() != safemlx::Dtype::Float32 {
                        return Err(cold(Cause::Source));
                    }
                    (Cut::Policy(policy), report, None)
                }
                Kind::LogitsRow { row } => {
                    if left_value()?.value().array.dtype() != safemlx::Dtype::Float32 {
                        return Err(cold(Cause::Source));
                    }
                    let output = readout::metadata_row(left_meta()?, row, &context)?;
                    (Cut::LogitsRow(row), context.finish_report(&[output])?, None)
                }
                Kind::Normalize => {
                    if cpu && left_value()?.value().array.dtype()!=safemlx::Dtype::Float32 { return Err(cold(Cause::Source)); }
                    let output =
                        program::normalize::<operators::Metadata>(left_meta()?, &context)?;
                    (Cut::Normalize, context.finish_report(&[output])?, None)
                }
                Kind::ProbabilityAt { token } => {
                    if cpu && left_value()?.value().array.dtype()!=safemlx::Dtype::Float32 {return Err(cold(Cause::Source));}
                    let output = operators::Metadata::select(left_meta()?, token, &context)?;
                    (
                        Cut::Probability(token),
                        context.finish_report(&[output])?,
                        None,
                    )
                }
                Kind::Correction => {
                    if cpu && (left_value()?.value().array.dtype()!=safemlx::Dtype::Float32
                        ||right.ok_or_else(||cold(Cause::Source))?.value().array.dtype()!=safemlx::Dtype::Float32){return Err(cold(Cause::Source));}
                    let (difference, mass) = program::correction::<operators::Metadata>(
                        left_meta()?,
                        right_metadata.as_ref().ok_or_else(|| cold(Cause::Source))?,
                        &context,
                    )?;
                    // The second metadata trace is descriptive only. Its native
                    // counterpart remains behind the completed mass decision.
                    let first = context.finish_report(&[difference, mass])?;
                    let difference = WorkspaceTensor::existing(
                        context.layout(program.shape(), WorkspaceDtype::Float32)?,
                        &context,
                    )?;
                    context.begin_span();
                    let logits =
                        program::correction_logits::<operators::Metadata>(&difference, &context)?;
                    let second = context.finish_report(&[logits])?;
                    (Cut::Difference, first, Some(second))
                }
            };
            let first = Plan::inspect(first_cut, &first_report, mechanism, &context, environment, capture_plan.as_ref(), cpu, cpu_equations)
                .map_err(cold)?;
            let second = second_report
                .as_ref()
                .map(|report| {
                    Plan::inspect(Cut::Logarithm, report, mechanism, &context, environment, None, cpu, None)
                })
                .transpose()
                .map_err(cold)?;
            let sum = |field: fn(&Plan) -> u64| {
                field(&first).checked_add(second.as_ref().map_or(0, field))
            };
            let requirements = SpeculativeNumericalRequirements::new(
                program,
                sum(|p| p.physical as u64),
                sum(|p| p.graph_bytes),
                sum(|p| p.record_bytes),
                sum(|p| p.controls),
            )
            .map_err(|e| retain_planning_error(e, funding.clone()))?;
            let requirements=if let Some(host)=capture_host.as_ref() {
                requirements.with_capture_destination(host).map_err(|cause|sources.retain_startup_error(cause))?
            } else {requirements};
            let source_values = [left.map(|v|v.value().provenance.source()), right.map(|v|v.value().provenance.source())];
            let one = source_values[0].map(|v|[v]);
            let two = source_values[0].zip(source_values[1]).map(|(a,b)|[a,b]);
            let admitted_sources = match program.source_count() {
                0 => &[][..],
                1 => &one.as_ref().ok_or_else(||cold(Cause::Source))?[..],
                2 => &two.as_ref().ok_or_else(||cold(Cause::Source))?[..],
                _ => return Err(cold(Cause::Source)),
            };
            let phase = sources.request().reserve_numerical(requirements, admitted_sources)
                .map_err(|e| retain_planning_error(e, funding.clone()))?;
            let (custody,capture_invocation)=if let Some(host)=capture_host {
                let (custody,invocation)=phase.begin_with_capture(host).map_err(|cause|sources.retain_startup_error(cause))?;
                (custody,Some(invocation))
            } else {(phase.begin(),None)};
            let mut capture_invocation=capture::Invocation::new(capture_invocation,failed_capture);
            if let Some(plan)=&capture_plan {plan.validate(sources)?;}

            // Every invocation owns its stream wrapper. Reusing a predecessor's
            // wrapper would keep that predecessor's entire model/numerical
            // reservation alive after independent output storage has completed.
            // Source identity was checked above; actual views retain their
            // backing owner separately through _readout_source.
            let stream_plan = safemlx::StreamCopyPlan::<Custody>::capture(environment.stream())
                .map_err(|e| failure(e.into(), &custody, funding))?;
            let output_stream = ValueStream::Numerical(stream_plan.realize(custody.clone())
                .map_err(|e| failure(e.into_parts().0.into(), &custody, funding))?);
            let seed_stream = Some(output_stream.clone());
            let planning = PlanningOwner(Some(Rc::new(Planning {
                reports: [Some(first_report), second_report],
                context,
                capture: capture_plan,
                funding: funding.clone(),
            })));
            let output = run_cut(
                &first,
                environment,
                roots_runtime,
                left,
                right,
                &planning,
                &custody,
                funding,
                history,
                mask,
                seed_stream,
                capture_invocation.as_mut(),
                capture_source.and_then(capture::CaptureSource::domain),
                tokens,
            )?;
            Ok(match kind {
                Kind::TokenIds { .. } => NumericalOutput::Tensor(
                    OriginalNumericalValue::completed(output.value, output_stream,
                        Meaning::TokenIds, custody, funding.clone()).with_original_budget(output.budget),
                ),
                Kind::TensorRange { .. } | Kind::TensorAxisRange { .. } | Kind::TokenRange { .. } => {
                    let mut value = OriginalNumericalValue::completed(output.value, output_stream,
                        expected, custody, funding.clone()).with_original_budget(output.budget);
                    Rc::get_mut(value.0.as_mut().expect("unpublished tensor view"))
                        .expect("unique completed tensor view")._readout_source = left.cloned().map(RetainedValues::One);
                    NumericalOutput::Tensor(value)
                }
                Kind::TensorConcatenate { .. } => {
                    let mut value = OriginalNumericalValue::completed(output.value, output_stream,
                        Meaning::Capture, custody, funding.clone()).with_original_budget(output.budget);
                    Rc::get_mut(value.0.as_mut().expect("unpublished concatenated tensor"))
                        .expect("unique completed concatenation")._readout_source = Some(RetainedValues::Two([
                            left.expect("validated left source").clone(), right.expect("validated right source").clone(),
                        ]));
                    NumericalOutput::Tensor(value)
                }
                Kind::CreateKey { .. } | Kind::KeyAt { .. } => NumericalOutput::Key(OriginalNumericalKey::completed(
                    output.value, output_stream, custody, funding.clone(), output.budget)),
                Kind::NextKey => NumericalOutput::KeyPair {
                    current: OriginalNumericalKey::completed(output.value, output_stream.clone(), custody.clone(), funding.clone(), output.budget.clone()),
                    next: OriginalNumericalKey::completed(output.other.ok_or_else(||failure(Cause::Completion,&custody,funding))?,
                        output_stream, custody, funding.clone(), output.budget),
                },
                Kind::UniformUnitInterval => NumericalOutput::RandomUnitInterval {
                    value: output.scalar.filter(|&value| value >= 0.0 && value < 1.0)
                        .ok_or_else(|| failure(Cause::Completion, &custody, funding))?,
                    advanced: OriginalNumericalKey::completed(
                        output.other.ok_or_else(|| failure(Cause::Completion, &custody, funding))?,
                        output_stream, custody, funding.clone(), output.budget),
                },
                Kind::Categorical(_) => NumericalOutput::RandomToken {
                    token: output.token.filter(|&token|token < *program.shape().last().expect("vocabulary") as u32)
                        .ok_or_else(||failure(Cause::Completion,&custody,funding))?,
                    advanced: OriginalNumericalKey::completed(output.other.ok_or_else(||failure(Cause::Completion,&custody,funding))?,
                        output_stream, custody, funding.clone(), output.budget),
                },
                Kind::Greedy(_) => NumericalOutput::Token(output.token
                    .filter(|&token| token < *program.shape().last().expect("source vocabulary") as u32)
                    .ok_or_else(|| failure(Cause::Completion, &custody, funding))?),
                Kind::ProcessLogits(_) => {
                    let capture=capture_invocation.as_mut().map(|invocation|
                        invocation.take_shared_step().map_err(|cause|failure(cause.into(),&custody,funding))?
                            .ok_or_else(||failure(Cause::Completion,&custody,funding))).transpose()?;
                    // A completed policy output can next cross the selected device
                    // boundary. Carry its actual native budget into the closed
                    // value so the existing copy worker can authenticate it.
                    let value=OriginalNumericalValue::completed(output.value,output_stream.clone(),Meaning::Logits,custody,funding.clone())
                        .with_original_budget(output.budget);
                    match capture {Some(capture)=>NumericalOutput::CapturedLogits {value,capture},None=>NumericalOutput::Logits(value)}
                },
                Kind::LogitsRow { .. } => {
                    let mut value = OriginalNumericalValue::completed(
                        output.value, output_stream, Meaning::Logits, custody, funding.clone(),
                    );
                    // The static view keeps its original backing independently
                    // of the new numerical graph/record completion account.
                    Rc::get_mut(value.0.as_mut().expect("unpublished readout"))
                        .expect("unique completed readout")._readout_source = left.cloned().map(RetainedValues::One);
                    NumericalOutput::Logits(value)
                }
                Kind::Normalize => {
                    NumericalOutput::Distribution(OriginalNumericalValue::completed(
                        output.value,
                        output_stream.clone(),
                        Meaning::Probabilities,
                        custody,
                        funding.clone(),
                    ).with_controller_choice(left.and_then(|v|v.controller_choice()).cloned()))
                }
                Kind::ProbabilityAt { .. } => NumericalOutput::Probability(
                    output
                        .scalar
                        .ok_or_else(|| failure(Cause::Completion, &custody, funding))?,
                ),
                Kind::Correction => {
                    let mass = output
                        .scalar
                        .ok_or_else(|| failure(Cause::Completion, &custody, funding))?;
                    if !program::correction_has_mass(mass) {
                        NumericalOutput::Correction(None)
                    } else {
                        let difference = OriginalNumericalValue::completed(
                            output.value,
                            output_stream.clone(),
                            Meaning::Difference,
                            custody.clone(),
                            funding.clone(),
                        );
                        let plan = second
                            .as_ref()
                            .ok_or_else(|| failure(Cause::Unknown("correction second cut"), &custody, funding))?;
                        let output = run_cut(
                            plan,
                            environment,
                            roots_runtime,
                            Some(&difference),
                            None,
                            &planning,
                            &custody,
                            funding,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                        )?;
                        NumericalOutput::Correction(Some(OriginalNumericalValue::completed(
                            output.value,
                            output_stream.clone(),
                            Meaning::Logits,
                            custody,
                            funding.clone(),
                        ).with_controller_choice(left.and_then(|v|v.controller_choice()).cloned())))
                    }
                }
            })
        })();
        result.map_err(|cause| sources.retain_error(cause))
    }
}
impl Plan {
    fn inspect(
        cut: Cut,
        report: &WorkspaceTraceReport,
        mechanism: MlxMetalWorkspaceMechanisms,
        context: &WorkspaceContext,
        environment: &OriginalCopyEnvironment<'_>,
        capture: Option<&capture::Plan>,
        cpu: bool,
        cpu_equations:Option<crate::backend::nn::workspace::MlxCpuWorkspaceMechanisms>,
    ) -> Result<Self, Cause> {
        let ordinary_roots:usize = if matches!(cut, Cut::Difference | Cut::NextKey | Cut::UniformUnitInterval | Cut::Categorical(_)) { 2 } else { 1 };
        let roots=ordinary_roots.checked_add(capture.map_or(0,|capture|capture.roots)).ok_or(Cause::Overflow)?;
        let runtime = environment.input_runtime()?;
        let eager = if let Cut::TokenIds(length) = cut {
            if capture.is_some() { return Err(Cause::Source); }
            Some(safemlx::OriginalPromptInputFacts::inspect(&runtime, length as usize)?)
        } else { None };
        let recipe = match (cpu, eager) {
            (true, Some(source)) => SpeculativeNumericalRecipe::inspect_cpu_token_ids(report, source, mechanism, context),
            (true,None) if matches!(cut,Cut::TensorConcatenate)&&capture.is_none()=>
                SpeculativeNumericalRecipe::inspect_cpu_equations(report,mechanism,
                    cpu_equations.ok_or(Cause::Source)?,context),
            (true, None) if matches!(cut, Cut::TensorRange { .. }) && capture.is_none() =>
                SpeculativeNumericalRecipe::inspect_cpu_range(report, mechanism, context),
            (true, None) if matches!(cut,Cut::Difference)&&capture.is_none()=>
                SpeculativeNumericalRecipe::inspect_cpu_difference(report,mechanism,context),
            (true, None) if matches!(cut,Cut::Logarithm)&&capture.is_none()=>
                SpeculativeNumericalRecipe::inspect_cpu_logarithm(report,mechanism,context),
            (true, None) if matches!(cut, Cut::Normalize) && capture.is_none() =>
                SpeculativeNumericalRecipe::inspect_cpu_normalize(report, mechanism, context),
            (true, None) if matches!(cut, Cut::Greedy(_)) && capture.is_none() =>
                SpeculativeNumericalRecipe::inspect_cpu_greedy(report, mechanism, context),
            (true, None) if matches!(cut, Cut::Probability(_) | Cut::LogitsRow(_)) && capture.is_none() =>
                SpeculativeNumericalRecipe::inspect_cpu_static_index(report, mechanism, context),
            (true, None) if matches!(cut, Cut::CreateKey(_)) && capture.is_none() =>
                SpeculativeNumericalRecipe::inspect_cpu_key(report, mechanism, context),
            (true,None) if matches!(cut,Cut::Categorical(_))&&capture.is_none()=>
                SpeculativeNumericalRecipe::inspect_cpu_categorical(report,mechanism,context),
            (true, None) if matches!(cut, Cut::UniformUnitInterval) && capture.is_none() =>
                SpeculativeNumericalRecipe::inspect_cpu_uniform(report, mechanism, context),
            (true, None) if matches!(cut, Cut::NextKey | Cut::KeyAt(_)) && capture.is_none() =>
                SpeculativeNumericalRecipe::inspect_cpu_split(report, roots, mechanism, context),
            (true,None) if matches!(cut,Cut::Policy(_)) && capture.is_some_and(|capture|capture.cpu_readouts)=>{
                let capture=capture.ok_or(Cause::Source)?;
                SpeculativeNumericalRecipe::inspect_cpu_capture(report,capture.roots,capture.completions,
                    mechanism,cpu_equations.ok_or(Cause::Source)?,context)
            },
            (true, None) if matches!(cut,Cut::Policy(_)) && capture.is_none()=>
                SpeculativeNumericalRecipe::inspect_cpu_equations(report,mechanism,
                    cpu_equations.ok_or(Cause::Source)?,context),
            (true, None) => return Err(Cause::Unknown(cut.name())),
            (false, Some(source)) => SpeculativeNumericalRecipe::inspect_token_ids(report, source, mechanism, context),
            (false, None) => SpeculativeNumericalRecipe::inspect_with_nested(report, roots,
                capture.map_or(0,|capture|capture.completions), mechanism, context),
        }.map_err(Error::from)?;
        let population = OriginalBufferBudget::metal_population_layout(
            &runtime,
            usize::try_from(recipe.storage.mutable_bytes()).map_err(|_| Cause::Overflow)?,
            recipe.storage.maximum_births(),
        )?;
        let physical = population.capacity()
            .checked_add(eager.map_or(0, |source| source.mutable_bytes())).ok_or(Cause::Overflow)?;
        let graph = PreparedSubmissionGraphQuota::<Custody>::layout(recipe.graph_capacity)?;
        let record = PreparedSubmissionRecordQuota::<Custody>::layout(recipe.record_capacity)?;
        let pipeline = PreparedPipelineCachePlan::new(recipe.kernels);
        let roots_layout = PrefillRoots::layout(roots)?;
        let policy_controls = if matches!(cut, Cut::Policy(_)) {
            crate::backend::runtime::generation::processing_control_bytes().ok_or(Cause::Unknown("logit policy controls"))?
        } else { 0 };
        // Price the actual C wrapper, deferred owner and shared Rust allocation
        // before role admission, for the current invocation's stream ownership.
        let output_stream = safemlx::StreamCopyPlan::<Custody>::capture(environment.stream())?;
        let shared_stream = Layout::new::<[std::sync::atomic::AtomicUsize; 2]>()
            .extend(output_stream.shared_body_layout()).map_err(|_| Cause::Overflow)?
            .0.pad_to_align().size();
        let stream_parts = [
            output_stream.control_bytes().ok_or(Cause::Unknown("output stream controls"))?,
            shared_stream,
            output_stream.owner_node_layout().size(),
            output_stream.native_wrapper_bytes(),
        ];
        let seed_stream_controls = stream_parts.into_iter()
            .try_fold(size_of_val(&stream_parts), usize::checked_add).ok_or(Cause::Overflow)?;
        let key_at_controls = if matches!(cut, Cut::KeyAt(_)) {
            crate::backend::random::split_key_at_control_bytes().ok_or(Cause::Unknown("key selection controls"))?
        } else {0};
        let uniform_controls = if matches!(cut, Cut::UniformUnitInterval) {
            crate::backend::random::uniform_unit_interval_control_bytes().ok_or(Cause::Unknown("uniform controls"))?
        } else { 0 };
        let range_controls=if matches!(cut,Cut::TensorRange{..}) {
            super::tensor::native_axis_range_control_bytes().ok_or(Cause::Overflow)?
        } else {0};
        let concatenate_controls=if matches!(cut,Cut::TensorConcatenate){
            safemlx::ops::concatenate_axis_control_bytes().ok_or(Cause::Unknown("concatenation controls"))?
                .checked_add(size_of::<[&Array;2]>()).ok_or(Cause::Overflow)?
        }else{0};
        let parts = [
            concatenate_controls,
            range_controls,
            eager.map(|facts| facts.control_bytes().ok_or(Cause::Unknown("eager input controls"))).transpose()?.unwrap_or(0),
            size_of::<Option<safemlx::OriginalPromptInputFacts>>(),
            size_of::<Result<safemlx::OriginalPromptInputFacts, safemlx::OriginalPromptInputCause>>(),
            eredu_runtime::generation::TokenMaskPlan::control_bytes(),
            size_of::<Option<eredu_runtime::generation::TokenMaskPlan<'_>>>(),
            uniform_controls,
            key_at_controls,
            size_of::<SpeculativeCategoricalProgram>(),
            size_of::<Result<MlxTensor, Exception>>(),
            size_of::<Option<&mut crate::backend::random::RandomState>>(),
            seed_stream_controls,
            size_of::<ValueStream>(), size_of::<Option<ValueStream>>(),
            size_of::<OriginalNumericalKey>(), size_of::<Option<OriginalNumericalKey>>(),
            size_of::<crate::backend::random::RandomState>(),
            size_of::<safemlx::StreamCopyPlan<Custody>>(),
            size_of::<Result<safemlx::PreparedStreamCopy<Custody>, safemlx::StreamCopyError<Custody>>>(),
            policy_controls,
            capture.map_or(0,|capture|capture.controls),
            size_of::<capture::Invocation<'_>>(),
            size_of::<Option<eredu_core::capture::SharedCapturedStep>>(),
            size_of::<Result<Option<eredu_core::capture::SharedCapturedStep>,eredu_runtime::capture::FundedCaptureDrainError>>(),
            size_of::<MlxTensor>(),
            size_of::<Result<MlxTensor, LogitProgramError<Exception>>>(),
            size_of::<SpeculativeLogitProgram>(),
            size_of::<Option<&[u32]>>(),
            size_of::<Self>(),
            size_of::<Domains>(),
            size_of::<Retained>(),
            size_of::<CutOutput>(),
            size_of::<Option<Array>>(),
            size_of::<Result<CutOutput, Cause>>(),
            size_of::<Result<CutOutput, Error>>(),
            size_of::<Cause>(),
            size_of::<Failure>(),
            size_of::<OriginalScopeObserver>(),
            size_of::<safemlx::SubmissionRetirement>(),
            size_of::<Result<safemlx::SubmissionRetirement, Exception>>(),
            size_of::<Option<safemlx::PreparedResidentGraph>>(),
            size_of::<Result<safemlx::PreparedResidentGraph, Exception>>(),
            size_of::<safemlx::EvaluatedArray<'_>>(),
            size_of::<safemlx::error::AsSliceError>(),
            size_of::<Result<&[f32], safemlx::error::AsSliceError>>(),
            size_of::<Result<&[u32], safemlx::error::AsSliceError>>(),
            size_of::<Option<u32>>(),
            rc_bytes::<RefCell<PrefillRoots>>().ok_or(Cause::Overflow)?,
            value_control_bytes().and_then(|n|n.checked_mul(2)).ok_or(Cause::Overflow)?,
            roots_layout.host_bytes().ok_or(Cause::Overflow)?,
            population.control_bytes(),
            PreparedOriginalBufferBudget::<Custody>::layout(&runtime, physical)?
                .total_owner_bytes()
                .ok_or(Cause::Overflow)?,
            PreparedPrefillFailure::<Custody>::layout()?
                .total_bytes()
                .ok_or(Cause::Overflow)?,
            pipeline
                .layout::<Custody>()?
                .required_bytes()
                .ok_or(Cause::Overflow)?,
            safemlx::OriginalNativeControlLayout::inspect()?.fixed_control_bytes,
            safemlx::original_scoped_evaluation_control_bytes().ok_or(Cause::Unknown("scoped evaluation controls"))?,
            OriginalCopyEnvironment::control_bytes().ok_or(Cause::Unknown("copy environment controls"))?,
            recipe
                .completion
                .graph
                .control_bytes()
                .ok_or(Cause::Unknown("resident graph controls"))?,
            eredu_core::BackendFailure::source_retention_peak_bytes::<Failure>()
                .and_then(|n| n.checked_mul(2))
                .ok_or(Cause::Unknown("failure retention controls"))?,
        ];
        let controls = parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
            .and_then(|n| u64::try_from(n).ok())
            .and_then(|n| n.checked_add(recipe.controls))
            .and_then(|n| n.checked_add(PreparedRecovery::<Retained, Custody>::control_bytes()?))
            .and_then(|n| n.checked_add(report.host_workspace_bytes?))
            .ok_or(Cause::Unknown("phase and report host controls"))?;
        Ok(Self {
            cut,
            recipe,
            physical,
            pipeline,
            controls,
            graph_bytes: u64::try_from(graph.total_bytes().ok_or(Cause::Overflow)?)
                .map_err(|_| Cause::Overflow)?,
            record_bytes: u64::try_from(record.total_bytes().ok_or(Cause::Overflow)?)
                .map_err(|_| Cause::Overflow)?,
        })
    }
    fn prepare(
        &self,
        environment: &OriginalCopyEnvironment<'_>,
        runtime: &PrefillRootsRuntime,
        custody: &Custody,
    ) -> Result<(Domains, Roots), Cause> {
        let graph =
            PreparedSubmissionGraphQuota::try_new(self.recipe.graph_capacity, custody.clone())
                .map_err(|e| e.into_parts().0)?
                .try_allocate()
                .map_err(|e| e.into_parts().0)?;
        let record =
            PreparedSubmissionRecordQuota::try_new(self.recipe.record_capacity, custody.clone())
                .map_err(|e| e.into_parts().0)?
                .try_allocate()
                .map_err(|e| e.into_parts().0)?;
        let allocator = environment.input_runtime()?;
        let buffer =
            PreparedOriginalBufferBudget::try_new(&allocator, self.physical, custody.clone())
                .map_err(|e| e.into_parts().0)?
                .try_allocate()
                .map_err(|e| e.into_parts().0)?;
        let pipeline = self
            .pipeline
            .realize(custody.clone())
            .map_err(|e| e.into_parts().0)?;
        pipeline.install(&graph)?;
        let failure = PreparedPrefillFailure::try_new(custody.clone())
            .map_err(|e| e.into_parts().0)?
            .try_allocate()
            .map_err(|e| e.into_parts().0)?;
        let roots = Roots(Rc::new(RefCell::new(PrefillRoots::new_retained(
            runtime,
            self.recipe.completion.traversal.roots(),
            &graph,
            &failure,
        )?)));
        Ok((
            Domains {
                graph,
                record,
                buffer,
                _failure: failure,
                pipeline,
            },
            roots,
        ))
    }
}
fn planning_control_bytes() -> Option<usize> {
    let parts = [
        size_of::<bool>() * 2, // actual device selection and Plan::inspect argument
        size_of::<Option<crate::backend::nn::workspace::MlxCpuWorkspaceMechanisms>>(),
        size_of::<crate::backend::nn::workspace::ResidentExecutionMechanisms>(),
        size_of::<Option<crate::backend::nn::workspace::MlxCpuWorkspaceMechanisms>>(),
        size_of::<(InputPlacement<'_>,SpeculativeLogitProgram,&OriginalNumericalValue,&[u32],
            Option<eredu_runtime::generation::TokenMaskPlan<'_>>,Option<capture::CaptureSource<'_>>,
            &mut Option<eredu_core::capture::SharedCapturedStep>)>(),
        size_of::<(InputPlacement<'_>,Kind,Option<&OriginalNumericalValue>,Option<&OriginalNumericalValue>)>(),
        size_of::<Result<NumericalOutput,Error>>(),
        size_of::<Option<InputPlacement<'_>>>(),
        size_of::<(InputPlacement<'_>,Kind,&OriginalNumericalValue,Option<&OriginalNumericalValue>)>(),
        size_of::<Result<NumericalOutput,Error>>(),
        size_of::<Option<(&OriginalSpeculativeNumericalSources,&OriginalCopyEnvironment<'_>)>>(),
        size_of::<(&OriginalSpeculativeNumericalSources,&OriginalCopyEnvironment<'_>)>(),
        capture::planning_control_bytes()?,
        eredu_runtime::generation::TokenMaskPlan::control_bytes(),
        size_of::<Option<eredu_runtime::generation::TokenMaskPlan<'_>>>(),
        size_of::<Planning>(),
        size_of::<capture::Plan>(), size_of::<Option<capture::Plan>>(),
        size_of::<capture::CaptureSource<'_>>(),
        size_of::<eredu_runtime::working_memory::SpeculativeCaptureHostPlan<'_>>(),
        size_of::<eredu_runtime::working_memory::SpeculativeCapturePreparationError>(),
        rc_bytes::<Planning>()?,
        size_of::<PlanningOwner>(),
        size_of::<[Option<WorkspaceTraceReport>; 2]>(),
        size_of::<[Option<Plan>; 2]>(),
        size_of::<Program>(),
        size_of::<SpeculativeNumericalRequirements>(),
        size_of::<NumericalOutput>(),
        size_of::<Result<NumericalOutput, Error>>(),
        size_of::<Option<&OriginalNumericalValue>>(),
        size_of::<Option<[SpeculativeNumericalSource<'_>; 1]>>(),
        size_of::<Option<[SpeculativeNumericalSource<'_>; 2]>>(),
        size_of::<&[SpeculativeNumericalSource<'_>]>(),
        size_of::<std::iter::Chain<std::option::IntoIter<(&OriginalNumericalValue, Meaning)>, std::option::IntoIter<(&OriginalNumericalValue, Meaning)>>>(),
        size_of::<[SpeculativeNumericalSource<'_>; 2]>(),
        size_of::<[Option<SpeculativeNumericalSource<'_>>; 2]>(),
        size_of::<safemlx::StreamCopyPlan<()>>(),
        size_of::<Result<safemlx::StreamCopyPlan<()>, safemlx::StreamCopyCause>>(),
        size_of::<WorkspaceSamplingRandomState>(),
        size_of::<Result<WorkspaceSamplingRandomState, eredu_nn::Error>>(),
        size_of::<WorkspaceTensor>(),
        size_of::<Option<WorkspaceTensor>>(),
        size_of::<Result<WorkspaceTensor, eredu_nn::Error>>(),
        size_of::<Result<WorkspaceTensor, LogitProgramError<eredu_nn::Error>>>(),
        size_of::<SpeculativeLogitProgram>(),
        size_of::<Option<&[u32]>>(),
        size_of::<Option<[i32; 2]>>(),
        size_of::<[i32; 2]>(),
        size_of::<Option<safemlx::OriginalPromptInputFacts>>(),
        size_of::<Result<safemlx::OriginalPromptInputFacts, safemlx::OriginalPromptInputCause>>(),
        size_of::<Cause>(),
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}
fn run_cut(
    plan: &Plan,
    environment: &OriginalCopyEnvironment<'_>,
    runtime: &PrefillRootsRuntime,
    left: Option<&OriginalNumericalValue>,
    right: Option<&OriginalNumericalValue>,
    planning: &PlanningOwner,
    custody: &Custody,
    funding: &WorkspaceMetadataFunding,
    history: Option<&[u32]>,
    mask:Option<eredu_runtime::generation::TokenMaskPlan<'_>>,
    seed_stream: Option<ValueStream>,
    capture_invocation:Option<&mut eredu_runtime::capture::FundedSpeculativeCaptureInvocation>,
    capture_domain:Option<eredu_core::capture::CaptureTokenDomain<'_>>,
    tokens: Option<&[u32]>,
) -> Result<CutOutput, Error> {
    let err = |cause| failure(cause, custody, funding);
    let (domains, roots) = plan.prepare(environment, runtime, custody).map_err(err)?;
    let graph = domains.graph.clone();
    let record = domains.record.clone();
    let buffer = domains.buffer.clone();
    let capture_plan=planning.0.as_ref().expect("live planning").capture.as_ref();
    let capture_roots=capture_plan.map(|capture|capture::Roots::prepare(capture.roots)).transpose()
        .map_err(|cause|err(cause.into()))?;
    let retained = Retained {
        capture_roots:capture_roots.clone(),
        roots: roots.clone(),
        domains,
        left: left.cloned(),
        right: right.cloned(),
        seed_stream,
        planning: planning.clone(),
        custody: custody.clone(),
    };
    let pending = PreparedRecovery::new(retained, custody.clone())
        .map_err(|e| err(e.cause.into()))?
        .with_graph_quota(Some(graph))
        .with_record_quota(Some(record));
    let mut recovery = pending.try_begin().map_err(|e| err(e.cause.into()))?;
    recovery
        .configure_scope(|scope| -> Result<(), Cause> {
            scope
                .enable_scoped_observation()
                .map_err(Cause::Observation)?;
            scope.require_original_native_controls()?;
            // The roots bind their retained failure exactly once, before
            // enabling original work; a second binding correctly refuses.
            roots.0.borrow_mut().bind_scope(scope)?;
            scope.enable_original_native_controls()?;
            scope.bind_original_buffer_budget(&buffer)?;
            Ok(())
        })
        .map_err(err)?;
    let observer = OriginalScopeObserver::require_current().map_err(|e| err(e.into()))?;
    let result = (|| -> Result<CutOutput, Cause> {
        if let Some(left) = left {
            safemlx::OperationEvent::validate_traversal_leaf(&left.value().array, &observer)
                .map_err(|source| Cause::Input { cut: plan.cut.name(), side: "left", source })?;
        }
        let left_value = || left.ok_or(Cause::Source);
        if let Some(right) = right {
            safemlx::OperationEvent::validate_traversal_leaf(&right.value().array, &observer)
                .map_err(|source| Cause::Input { cut: plan.cut.name(), side: "right", source })?;
        }
        // Eager input storage is priced by OriginalPromptInputFacts and routed
        // directly through the admitted Graph. The lazy constructor bank has
        // no token-source slots and must not intercept that separate producer.
        let mut bank = if matches!(plan.cut, Cut::TokenIds(_)) {
            None
        } else {
            Some(safemlx::OperationEvent::prepare_resident_graph(
                plan.recipe.completion.graph,
                &observer,
            )?)
        };
        if plan.recipe.completion.nested_completions != 0 {
            let nested=plan.recipe.completion.nested_traversal().ok_or(Cause::Unknown("nested traversal"))?;
            bank.as_mut().ok_or(Cause::Source)?
                .configure_nested_completions(&nested,plan.recipe.completion.nested_completions)?;
        }
        let stream = environment.stream();
        let effective=match (capture_plan,capture_invocation,capture_roots.as_ref()) {
            (Some(capture),Some(invocation),Some(roots))=>capture::observe(capture,&plan.recipe,invocation,
                &left_value()?.value().array,stream,roots,custody,&observer,capture_domain)?,
            (None,None,None)=>None,
            _=>return Err(Cause::Source),
        };
        let (value, mass) = match plan.cut {
            Cut::TokenIds(length) => {
                let tokens = tokens.filter(|tokens| tokens.len() == length as usize).ok_or(Cause::Source)?;
                (Array::try_from_original_prompt_ids(tokens)?, None)
            }
            Cut::TensorRange { axis, start, end, tokens: _ } => {
                (super::tensor::native_axis_range(&left_value()?.value().array, axis, start, end, stream)?, None)
            }
            Cut::TensorConcatenate => (
                safemlx::ops::concatenate_axis(&[&left_value()?.value().array,
                    &right.ok_or(Cause::Source)?.value().array], 1, stream)?, None),
            Cut::CreateKey(seed) => (safemlx::random::key(seed)?, None),
            Cut::NextKey => {
                let mut random = crate::backend::random::RandomState::from_key(left_value()?.value().array.clone());
                let next = random.next_key(stream)?;
                (random.into_key(), Some(next))
            }
            Cut::UniformUnitInterval => {
                let mut random = crate::backend::random::RandomState::from_key(left_value()?.value().array.clone());
                let draw = random.uniform_unit_interval(stream)?;
                (draw, Some(random.into_key()))
            }
            Cut::KeyAt(position) => (crate::backend::random::split_key_at(&left_value()?.value().array, position as usize, stream)?, None),
            Cut::Categorical(choice) => {
                let input = MlxTensor::from_array(left_value()?.value().array.clone());
                let mut random = crate::backend::random::RandomState::from_key(right.ok_or(Cause::Source)?.value().array.clone());
                let token = choice.construct::<MlxSamplingBackend>(&input, &mut random, stream)?;
                (token.into_array(), Some(random.into_key()))
            }
            Cut::Greedy(choice) => {
                let input = MlxTensor::from_array(left_value()?.value().array.clone());
                (choice.construct::<MlxSamplingBackend>(&input, stream)?.into_array(), None)
            }
            Cut::Policy(policy) => {
                let input = MlxTensor::from_array(match effective {
                    Some(value)=>value,
                    None=>left_value()?.value().array.clone(),
                });
                let masked=if let Some(mask)=mask {
                    if !mask.matches_shape(input.as_array().shape()) { return Err(Cause::Source); }
                    if mask.is_identity() { Some(input.clone()) } else {
                        let mut invalid=Vec::new();
                        invalid.try_reserve_exact(mask.elements()).map_err(Cause::MaskAllocation)?;
                        if invalid.capacity()!=mask.elements() { return Err(eredu_runtime::generation::TokenMaskError::Destination.into()); }
                        mask.fill(&mut invalid)?;
                        Some(crate::backend::runtime::generation::apply_token_mask(&input,&invalid,stream)?)
                    }
                } else {None};
                let output = policy.process::<MlxSamplingBackend>(
                    masked.as_ref().unwrap_or(&input), history.ok_or(Cause::Source)?, stream,
                )?;
                (output.into_array(), None)
            }
            Cut::LogitsRow(row) => (
                crate::composition::mlx::prepared_speculative::embedded_logits::row(
                    &left_value()?.value().array, row as usize, stream,
                )?, None,
            ),
            Cut::Normalize => (
                program::normalize::<operators::Native>(&left_value()?.value().array, stream)?,
                None,
            ),
            Cut::Probability(token) => (
                operators::Native::select(&left_value()?.value().array, token, stream)?,
                None,
            ),
            Cut::Difference => {
                let (difference, mass) = program::correction::<operators::Native>(
                    &left_value()?.value().array,
                    &right.ok_or(Cause::Source)?.value().array,
                    stream,
                )?;
                (difference, Some(mass))
            }
            Cut::Logarithm => (
                program::correction_logits::<operators::Native>(&left_value()?.value().array, stream)?,
                None,
            ),
        };
        drop(bank);
        {
            let mut roots = roots.0.try_borrow_mut().map_err(|_| Cause::Source)?;
            roots.append(&value)?;
            if let Some(mass) = &mass {
                roots.append(mass)?;
            }
            // Includes the normalized row even when the logical ledger skipped
            // every selection. Every constructed alias/intermediate settles
            // before the shared frame or processed logits can escape.
            if let Some(capture)=&capture_roots {capture.append_to(&mut roots)?;}
            roots.complete_current_scope_on_stream_prepared(
                stream,
                &plan.recipe.completion.traversal,
            )?;
            // This prepared entry already performs the exact scoped wait and
            // validates every retained root. Ordinary wait/validate refuse an
            // original collector and must never be used as fallback here.
        }
        let scalar = if matches!(plan.cut, Cut::Probability(_) | Cut::Difference | Cut::UniformUnitInterval) {
            // Correction reads its second root (mass); a uniform phase reads
            // its first F32 root, while the second root is the advanced U32 key.
            let source = if matches!(plan.cut, Cut::Difference) {
                mass.as_ref().ok_or(Cause::Completion)?
            } else { &value };
            let scalar = source.completed_in_original_scope(&observer)?;
            let values = scalar
                .try_as_slice::<f32>()
                .map_err(|_| Cause::Completion)?;
            Some(
                *values
                    .first()
                    .filter(|_| values.len() == 1)
                    .ok_or(Cause::Completion)?,
            )
        } else {
            None
        };
        let token = if matches!(plan.cut, Cut::Greedy(_) | Cut::Categorical(_)) {
            let completed = value.completed_in_original_scope(&observer)?;
            let values = completed.try_as_slice::<u32>().map_err(|_| Cause::Completion)?;
            Some(*values.first().filter(|_| values.len() == 1).ok_or(Cause::Completion)?)
        } else { None };
        Ok(CutOutput { value, budget: buffer.clone(), other: mass, scalar, token })
    })();
    // The recovery payload becomes the sole collector owner before terminal
    // cleanup. Its native roots/event and Rc shell are destroyed under that
    // worker's no-hooks retirement guard, before exact Record drainage.
    drop(capture_roots);
    drop(roots);
    recovery.seal();
    // Only the successful prepared-root completion enters the shared waiting
    // path. Callback/native failure still drops the live recovery nonblocking.
    let output = result.map_err(err)?;
    // The final root event may become visible before every accepted record
    // publishes its terminal transition. One progress poll is not completion.
    let status = recovery.finish();
    if !status.settled || status.failed || status.blocked {
        return Err(err(Cause::RecoveryStatus(status)));
    }
    let retired = observer
        .retire_completed_records()
        .map_err(|e| err(e.into()))?;
    if !matches!(retired, safemlx::SubmissionRetirement::CompleteSnapshot) {
        return Err(err(Cause::Retirement(retired)));
    }
    Ok(output)
}

#[cfg(test)]
pub(super) fn is_program_source_failure(error: &(dyn std::error::Error + 'static)) -> bool {
    let mut next = Some(error);
    while let Some(cause) = next {
        if matches!(cause.downcast_ref::<Cause>(),
            Some(Cause::Program(program::SpeculativeNumericalError::Source))) {
            return true;
        }
        next = cause.source();
    }
    false
}
