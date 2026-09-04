//! MLX implementation of backend-neutral realtime loading and execution.

use std::{num::NonZeroUsize, ops::Deref, sync::Arc};

use eredu_architectures::moshi::{
    inspect_moshi_realtime, select_inspected_moshi_realtime, MoshiRealtimeExecution,
    MoshiRealtimeRequest, PreparedMoshiRealtime, RealtimePreparationPlan,
};
use eredu_checkpoint::{LinearFormat, SourceTensorEncoding};
use eredu_core::cache::{StateComponentPolicy, StateComponentRole};
use eredu_core::{
    backend::Completion,
    realtime::{RealtimeDecisionDiagnostics, RealtimeInputFrame, RealtimeOutputFrame},
    CompletionCancellationMode, ParallelRankTopology, ParallelTopology, SessionCapabilities,
};
use eredu_runtime::{
    execute_realtime_frame, CommunicationCompletionCapabilities, CompletedRealtimeFrame,
    ExecutionResidency, GenerationSampler, MaterializedRealtimeInput, PipelineActivationDtype,
    PreparedRealtimeFrameExecutor, PrepublicationRealtimeFrame, RealtimeArchitectureRequirements,
    RealtimeCompletionCreationError, RealtimeFrameCompletionMechanism, RealtimeFrameHostObserver,
    RealtimeFrameTensorMechanisms, RealtimeHostTokenMaterializer, RealtimeMechanism,
    RealtimeMechanismCapabilities, RealtimeObservationRequirements, RealtimePayloadBranch,
    RealtimePayloadHistory, RealtimeSessionBranch, StateComponentMechanism,
    StateComponentPlacement, StateMechanismCapabilities, WeightLoweringCapability,
    WeightLoweringKind,
};
use safemlx::{
    ops::{indexing::TryIndexOp, stack_axis},
    random,
    transforms::{async_eval_with_event, eval},
    Array, Dtype, Event, Stream,
};

use crate::backend::runtime::distributed::Group;
use crate::{
    backend::random::RandomState,
    backend::runtime::{
        cache::state::{MlxKeyValueState, MlxKeyValueTransactionBranch},
        generation::MlxSamplingBackend,
    },
    backend::{
        error::Error,
        nn::tensor::{TokenValidationBatch, TokenValidationScope},
    },
    composition::moshi::{self as neutral_moshi, MlxRealtimeExecution},
    MlxLoadRequest, MlxTensor,
};

/// MLX stream and collective mechanisms for neutral realtime execution.
#[derive(Clone)]
pub struct MlxRealtimeExecutionContext {
    stream: Stream,
    weights_stream: Stream,
    world_group: Option<Arc<Group>>,
}

impl MlxRealtimeExecutionContext {
    /// Selects execution and weight-materialization streams for one backend.
    pub fn new(stream: &Stream, weights_stream: &Stream) -> Self {
        Self {
            stream: stream.clone(),
            weights_stream: weights_stream.clone(),
            world_group: None,
        }
    }

    /// Supplies the native world group used to realize architecture-selected resources.
    pub fn with_tensor_parallel_group(mut self, group: Arc<Group>) -> Self {
        self.world_group = Some(group);
        self
    }

    /// Selected MLX execution stream.
    pub const fn stream(&self) -> &Stream {
        &self.stream
    }

    /// Selected MLX checkpoint materialization stream.
    pub const fn weights_stream(&self) -> &Stream {
        &self.weights_stream
    }

    /// Fail-closed capabilities of this concrete session mechanism route.
    pub const fn session_capabilities() -> eredu_core::SessionCapabilities {
        realtime_session_capabilities()
    }

    /// Inspects and selects architecture semantics without creating native work.
    ///
    /// `collectives_supported` reports static route capability; an actual group
    /// is realized only later while materializing an already selected topology.
    pub fn select_realtime_execution(
        preparation: RealtimePreparationPlan,
        options: &MlxLoadRequest,
        collectives_supported: bool,
    ) -> Result<PreparedMoshiRealtime, Error> {
        select_realtime_model(preparation, options, collectives_supported)
    }

    /// Materializes an already selected architecture through MLX mechanisms.
    pub fn materialize_realtime_execution(
        &self,
        selected: PreparedMoshiRealtime,
        options: MlxLoadRequest,
    ) -> Result<MoshiRealtimeExecution<MlxRealtimeExecution>, Error> {
        validate_realtime_session_requirements(&options)?;
        materialize_realtime_model(
            selected,
            options,
            self.world_group.clone(),
            &self.stream,
            &self.weights_stream,
        )
    }

    /// Creates backend-native cache state from the selected neutral layout.
    pub fn new_realtime_model_state(
        &self,
        model: &MoshiRealtimeExecution<MlxRealtimeExecution>,
    ) -> Result<MlxKeyValueState, Error> {
        model.executor().new_realtime_state()
    }

    /// Realizes backend-native random state from an optional portable seed.
    ///
    /// Architecture and runtime composition decide whether randomness is
    /// required; this mechanism only creates the opaque backend state.
    pub fn realize_random_state(&self, seed: Option<u64>) -> Result<Option<RandomState>, Error> {
        Ok(seed
            .map(|seed| random::key(seed).map(RandomState::from_key))
            .transpose()?)
    }

    /// Submits one portable frame on an unpublished neutral scheduler branch.
    pub fn submit_realtime_frame(
        &self,
        model: &mut MoshiRealtimeExecution<MlxRealtimeExecution>,
        frame: &RealtimeInputFrame,
        branch: &mut MlxFrameSessionBranch,
    ) -> Result<MlxPrepublicationFrame, Error> {
        submit_scheduled_realtime_frame(model, branch, frame, &self.stream)
    }
}

const fn realtime_session_capabilities() -> eredu_core::SessionCapabilities {
    eredu_core::SessionCapabilities::new(true, true, false)
}

fn validate_realtime_session_requirements(options: &MlxLoadRequest) -> Result<(), Error> {
    options
        .required_session_capabilities
        .validate(&realtime_session_capabilities())?;
    Ok(())
}

fn materialize_realtime_model(
    selected: PreparedMoshiRealtime,
    options: MlxLoadRequest,
    world: Option<Arc<Group>>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<MoshiRealtimeExecution<MlxRealtimeExecution>, Error> {
    if !selected.selected().topology().is_replicated() {
        options
            .parallel_rank_context()?
            .ok_or_else(|| {
                Error::Parallel("parallel realtime execution has no MLX rank/device context".into())
            })?
            .validate_execution_stream(stream)?;
    }
    neutral_moshi::materialize_selected(selected, world, stream, weights_stream)
}

fn select_realtime_model(
    preparation: RealtimePreparationPlan,
    options: &MlxLoadRequest,
    collectives_supported: bool,
) -> Result<PreparedMoshiRealtime, Error> {
    validate_realtime_session_requirements(options)?;
    let request = mlx_realtime_request(options)?;
    let inspected = inspect_moshi_realtime(preparation, request)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let capabilities = mlx_realtime_capabilities(inspected.requirements(), collectives_supported);
    select_inspected_moshi_realtime(inspected, &capabilities)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))
}

fn mlx_realtime_request(options: &MlxLoadRequest) -> Result<MoshiRealtimeRequest, Error> {
    let rank = options.parallel_topology().unwrap_or_else(|| {
        ParallelRankTopology::new(
            ParallelTopology::new(1, 1, 1, 1).expect("replicated topology is valid"),
            0,
        )
        .expect("rank zero belongs to replicated topology")
    });
    let (maximum_batch_size, maximum_sequence_length) = options
        .partitioned_invocation_limits()?
        .unwrap_or((i32::MAX, i32::MAX));
    let maximum_batch_size = NonZeroUsize::new(
        usize::try_from(maximum_batch_size)
            .map_err(|_| Error::Parallel("negative realtime batch limit".into()))?,
    )
    .ok_or_else(|| Error::Parallel("realtime batch limit must be positive".into()))?;
    let maximum_sequence_length = NonZeroUsize::new(
        usize::try_from(maximum_sequence_length)
            .map_err(|_| Error::Parallel("negative realtime sequence limit".into()))?,
    )
    .ok_or_else(|| Error::Parallel("realtime sequence limit must be positive".into()))?;
    let activation_dtype = match (
        rank.topology().is_replicated(),
        options.pipeline_wire_contract(),
    ) {
        (false, None) => {
            return Err(Error::Parallel(
                "parallel Moshi wire contract is missing".into(),
            ))
        }
        (_, Some(wire)) => wire.activation_dtype(),
        (true, None) => PipelineActivationDtype::Float32,
    };
    Ok(MoshiRealtimeRequest::new(
        options.quantization(),
        options.weight_residency().layers(),
        options.state_residency().clone(),
        rank,
        maximum_batch_size,
        maximum_sequence_length,
        activation_dtype,
        options.realtime_completion_policy()?,
        RealtimeObservationRequirements::new(true, []),
    )
    .with_independently_addressable_parameters(
        options.weight_residency().parameter_bank_cache().is_some(),
    ))
}

fn mlx_realtime_capabilities(
    requirements: &RealtimeArchitectureRequirements,
    collectives_supported: bool,
) -> RealtimeMechanismCapabilities {
    let state =
        StateMechanismCapabilities::new((0..requirements.state_layout().len()).flat_map(|layer| {
            requirements
                .state_layout()
                .components(layer)
                .expect("validated state layout exposes every layer")
                .iter()
                .filter(mlx_supports_realtime_state_component)
                .cloned()
                .map(move |component| {
                    StateComponentMechanism::new(
                        layer,
                        component,
                        Some(StateComponentPlacement::Device),
                        None,
                    )
                })
                .collect::<Vec<_>>()
        }))
        .with_transactions(true, true)
        .with_reset(true)
        .with_observation_retention(true);
    let lowerings = requirements
        .executions()
        .iter()
        .flat_map(|execution| execution.weight_lowerings())
        .filter(|lowering| mlx_supports_realtime_lowering(lowering.descriptor(), lowering.kind()))
        .map(|lowering| {
            WeightLoweringCapability::new(lowering.descriptor().clone(), lowering.kind())
        })
        .collect();
    let mechanisms = mlx_realtime_mechanisms(collectives_supported);
    RealtimeMechanismCapabilities::new(
        eredu_nn::NeuralOperatorCapabilities::NONE,
        mechanisms,
        [
            ExecutionResidency::FullyResident,
            ExecutionResidency::LayerwiseHost,
            ExecutionResidency::DenseDiskStream,
        ],
        lowerings,
        state,
        NonZeroUsize::new(usize::MAX).expect("usize maximum is positive"),
        CommunicationCompletionCapabilities::new([
            CompletionCancellationMode::QuarantineUntilComplete,
        ])
        .expect("MLX quarantine completion capability is valid"),
        SessionCapabilities::new(true, true, false),
    )
}

fn mlx_realtime_mechanisms(collectives_supported: bool) -> Vec<RealtimeMechanism> {
    let mut mechanisms = vec![
        RealtimeMechanism::TensorOperations,
        RealtimeMechanism::NeuralOperations,
        RealtimeMechanism::ParameterMaterialization,
        RealtimeMechanism::ParameterStorage,
        RealtimeMechanism::StateStorage,
        RealtimeMechanism::CoordinateStorage,
        RealtimeMechanism::Sampling,
        RealtimeMechanism::Randomness,
        RealtimeMechanism::HostConversion,
        RealtimeMechanism::ExactCompletion,
        RealtimeMechanism::ResourceRetention,
        RealtimeMechanism::Transfer,
    ];
    if collectives_supported {
        mechanisms.push(RealtimeMechanism::Collectives);
    }
    mechanisms
}

fn mlx_supports_realtime_state_component(component: &&StateComponentPolicy) -> bool {
    matches!(
        component.role(),
        StateComponentRole::AttentionKeys
            | StateComponentRole::AttentionValues
            | StateComponentRole::CompressedLatent
            | StateComponentRole::RotaryKeys
            | StateComponentRole::Fixed(_)
    )
}

fn mlx_supports_realtime_lowering(
    descriptor: &eredu_runtime::WeightLoweringDescriptor,
    kind: WeightLoweringKind,
) -> bool {
    if !matches!(descriptor.source(), SourceTensorEncoding::Safetensors(_)) {
        return false;
    }
    match kind {
        WeightLoweringKind::Direct | WeightLoweringKind::Derived => matches!(
            descriptor.executable(),
            LinearFormat::Dense
                | LinearFormat::Affine(_)
                | LinearFormat::MxFp4
                | LinearFormat::GgufIQuant { .. }
                | LinearFormat::E4M3BlockFp8(_)
        ),
        WeightLoweringKind::Transform | WeightLoweringKind::DerivedTransform => matches!(
            descriptor.executable(),
            LinearFormat::Affine(_) | LinearFormat::MxFp4
        ),
        _ => false,
    }
}

fn array_i32_host(array: &Array) -> Result<Vec<i32>, Error> {
    let evaluated = array.evaluated()?;
    match array.dtype() {
        Dtype::Int32 => Ok(evaluated.as_slice::<i32>().to_vec()),
        Dtype::Uint32 => evaluated
            .as_slice::<u32>()
            .iter()
            .map(|value| i32::try_from(*value).map_err(|error| Error::Parallel(error.to_string())))
            .collect(),
        Dtype::Int64 => evaluated
            .as_slice::<i64>()
            .iter()
            .map(|value| i32::try_from(*value).map_err(|error| Error::Parallel(error.to_string())))
            .collect(),
        Dtype::Uint64 => evaluated
            .as_slice::<u64>()
            .iter()
            .map(|value| i32::try_from(*value).map_err(|error| Error::Parallel(error.to_string())))
            .collect(),
        dtype => Err(Error::Parallel(format!(
            "realtime token observation expected integer values, got {dtype:?}"
        ))),
    }
}

fn array_f32_host(array: &Array, stream: &Stream) -> Result<Vec<f32>, Error> {
    let array = if array.dtype() == Dtype::Float32 {
        array.clone()
    } else {
        array.as_dtype(Dtype::Float32, stream)?
    };
    Ok(array.evaluated()?.as_slice::<f32>().to_vec())
}

/// MLX host observer used by the neutral prepublication transition.
#[derive(Clone)]
pub struct MlxRealtimeHostObserver {
    stream: Stream,
}

impl MlxRealtimeHostObserver {
    /// Creates an observer on the selected execution stream.
    pub fn new(stream: &Stream) -> Self {
        Self {
            stream: stream.clone(),
        }
    }
}

impl RealtimeFrameHostObserver<MlxTensor> for MlxRealtimeHostObserver {
    type Output = RealtimeOutputFrame;
    type Error = Error;

    fn observe(
        &mut self,
        frame: &CompletedRealtimeFrame<MlxTensor, MlxTensor>,
    ) -> Result<Self::Output, Self::Error> {
        let text = frame.text().as_array();
        let batch = usize::try_from(text.dim(0))
            .map_err(|_| Error::Parallel("negative realtime output batch".into()))?;
        let diagnostics = frame
            .diagnostics()
            .iter()
            .enumerate()
            .map(|(prediction, logits)| {
                let logits = logits.as_array();
                let shape = logits
                    .shape()
                    .iter()
                    .map(|dimension| {
                        usize::try_from(*dimension).map_err(|_| {
                            Error::Parallel("negative realtime diagnostic dimension".into())
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                RealtimeDecisionDiagnostics::new(
                    prediction,
                    shape,
                    array_f32_host(logits, &self.stream)?,
                )
                .map_err(|error| Error::ArchitectureModel(error.to_string()))
            })
            .collect::<Result<Vec<_>, Error>>()?;
        Ok(RealtimeOutputFrame::new(
            batch,
            array_i32_host(text)?,
            array_i32_host(frame.decision_audio().as_array())?,
            array_i32_host(frame.sampled_audio().as_array())?,
            frame
                .aligned_audio()
                .map(MlxTensor::as_array)
                .map(array_i32_host)
                .transpose()?,
            diagnostics,
        ))
    }
}

/// Neutral scheduler branch specialized to MLX model mechanisms.
pub type MlxFrameSessionBranch = RealtimeSessionBranch<
    RealtimePayloadBranch<MlxKeyValueTransactionBranch, MlxTensor>,
    GenerationSampler,
    RandomState,
    MlxRealtimeCompletion,
>;

/// MLX submission whose host observation must succeed before publication.
pub type MlxPrepublicationFrame =
    PrepublicationRealtimeFrame<MlxTensor, MlxRealtimeCompletion, MlxRealtimeHostObserver>;

/// Exact MLX event retaining generated token arrays.
#[derive(Clone)]
pub struct MlxRealtimeCompletion {
    inner: Arc<MlxRealtimeCompletionInner>,
}

struct MlxRealtimeCompletionInner {
    event: Event,
    retained: Vec<Array>,
    token_validations: TokenValidationBatch,
    _execution_resources: Option<Arc<neutral_moshi::SelectedRealtimeResources>>,
}

#[derive(Debug, Eq, PartialEq)]
enum CompletionSubmissionFailure<E> {
    Drained { submission: E },
    DrainReported { submission: E, drain: E },
}

/// Owns every submitted root until either an exact completion exists or a
/// synchronous fallback has returned. MLX can begin native work before event
/// construction reports an error, so an event-creation error alone is not a
/// safe pre-submission boundary.
fn submit_or_synchronously_drain<R, C, E>(
    retained: Vec<R>,
    submit: impl FnOnce(&[R]) -> Result<C, E>,
    drain: impl FnOnce(&[R]) -> Result<(), E>,
) -> Result<(C, Vec<R>), CompletionSubmissionFailure<E>> {
    match submit(&retained) {
        Ok(completion) => Ok((completion, retained)),
        Err(submission) => match drain(&retained) {
            Ok(()) => Err(CompletionSubmissionFailure::Drained { submission }),
            Err(drain) => Err(CompletionSubmissionFailure::DrainReported { submission, drain }),
        },
    }
}

impl MlxRealtimeCompletion {
    #[cfg(test)]
    fn submit_retained(
        retained: Vec<Array>,
        token_validations: TokenValidationBatch,
    ) -> Result<Self, Error> {
        Self::submit_retained_with_resources(retained, token_validations, None)
    }

    fn submit_retained_with_resources(
        mut retained: Vec<Array>,
        token_validations: TokenValidationBatch,
        execution_resources: Option<Arc<neutral_moshi::SelectedRealtimeResources>>,
    ) -> Result<Self, Error> {
        retained.extend(token_validations.arrays().cloned());
        let (event, retained) = submit_or_synchronously_drain(
            retained,
            |retained| async_eval_with_event(retained.iter()),
            |retained| eval(retained.iter()),
        )
        .map_err(|failure| match failure {
            CompletionSubmissionFailure::Drained { submission } => Error::Parallel(format!(
                "MLX realtime completion event creation failed after possible native submission; \
                 retained work was synchronously drained: {submission}"
            )),
            CompletionSubmissionFailure::DrainReported { submission, drain } => {
                Error::Parallel(format!(
                    "MLX realtime completion event creation failed after possible native \
                     submission ({submission}); synchronous drain reported: {drain}"
                ))
            }
        })?;
        Ok(Self {
            inner: Arc::new(MlxRealtimeCompletionInner {
                event,
                retained,
                token_validations,
                _execution_resources: execution_resources,
            }),
        })
    }

    /// Number of array handles retained through exact completion.
    pub fn retained_resources(&self) -> usize {
        self.inner.retained.len()
    }
}

impl Completion for MlxRealtimeCompletion {
    type Error = Error;

    fn is_complete(&self) -> Result<bool, Self::Error> {
        let complete = self.inner.event.is_complete()?;
        if complete {
            // Readiness is a semantic publication gate, not merely native
            // event readiness. A completed invalid token scope must therefore
            // fail before any scheduler is allowed to commit its branch.
            self.inner.token_validations.validate_completed()?;
        }
        Ok(complete)
    }

    fn wait(&self) -> Result<(), Self::Error> {
        self.inner.event.synchronize()?;
        self.inner.token_validations.validate_completed()?;
        Ok(())
    }
}

#[cfg(test)]
mod completion_ownership_tests {
    use super::{
        submit_or_synchronously_drain, CompletionSubmissionFailure, MlxRealtimeCompletion,
    };
    use crate::backend::nn::tensor::{validate_token_domain, TokenValidationScope};
    use eredu_core::backend::Completion;
    use safemlx::{Array, Device, DeviceType, Stream};
    use std::{cell::RefCell, rc::Rc};

    #[derive(Debug)]
    struct RetainedRoot {
        calls: Rc<RefCell<Vec<&'static str>>>,
    }

    impl Drop for RetainedRoot {
        fn drop(&mut self) {
            self.calls.borrow_mut().push("drop");
        }
    }

    #[test]
    fn successful_exact_submission_does_not_run_the_synchronous_drain() {
        let calls = Rc::new(RefCell::new(Vec::new()));
        let retained = vec![RetainedRoot {
            calls: Rc::clone(&calls),
        }];

        let (completion, retained) = submit_or_synchronously_drain(
            retained,
            |roots| {
                assert_eq!(roots.len(), 1);
                calls.borrow_mut().push("submit");
                Ok::<_, &'static str>("event")
            },
            |_| {
                calls.borrow_mut().push("drain");
                Ok(())
            },
        )
        .unwrap();

        assert_eq!(completion, "event");
        assert_eq!(&*calls.borrow(), &["submit"]);
        drop(retained);
        assert_eq!(&*calls.borrow(), &["submit", "drop"]);
    }

    #[test]
    fn failed_event_creation_retains_roots_through_the_synchronous_drain() {
        let calls = Rc::new(RefCell::new(Vec::new()));
        let retained = vec![RetainedRoot {
            calls: Rc::clone(&calls),
        }];

        let error = submit_or_synchronously_drain(
            retained,
            |_| {
                calls.borrow_mut().push("submit");
                Err::<(), _>("submission")
            },
            |roots| {
                assert_eq!(roots.len(), 1);
                assert_eq!(&*calls.borrow(), &["submit"]);
                calls.borrow_mut().push("drain");
                Ok(())
            },
        )
        .unwrap_err();

        assert_eq!(
            error,
            CompletionSubmissionFailure::Drained {
                submission: "submission"
            }
        );
        assert_eq!(&*calls.borrow(), &["submit", "drain", "drop"]);
    }

    #[test]
    fn synchronous_drain_failure_is_reported_only_after_retained_roots_are_held() {
        let calls = Rc::new(RefCell::new(Vec::new()));
        let retained = vec![RetainedRoot {
            calls: Rc::clone(&calls),
        }];

        let error = submit_or_synchronously_drain(
            retained,
            |_| {
                calls.borrow_mut().push("submit");
                Err::<(), _>("submission")
            },
            |roots| {
                assert_eq!(roots.len(), 1);
                calls.borrow_mut().push("drain");
                Err("asynchronous execution failure")
            },
        )
        .unwrap_err();

        assert_eq!(
            error,
            CompletionSubmissionFailure::DrainReported {
                submission: "submission",
                drain: "asynchronous execution failure"
            }
        );
        assert_eq!(&*calls.borrow(), &["submit", "drain", "drop"]);
    }

    #[test]
    fn completed_invalid_token_scope_fails_readiness_before_publication() {
        let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let scope = TokenValidationScope::begin().unwrap();
        let tokens =
            validate_token_domain(&Array::from_slice(&[5_u32], &[1]), 5, None, &stream).unwrap();
        let validations = scope.finish();
        let expected_retained = 1 + validations.arrays().count();
        let completion = MlxRealtimeCompletion::submit_retained(vec![tokens], validations).unwrap();

        assert_eq!(completion.retained_resources(), expected_retained);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        let readiness_error = loop {
            match completion.is_complete() {
                Ok(false) => {
                    assert!(std::time::Instant::now() < deadline);
                    std::thread::yield_now();
                }
                Ok(true) => panic!("invalid token scope must not become publishable"),
                Err(error) => break error.to_string(),
            }
        };
        assert!(
            readiness_error.contains("outside 0..5"),
            "{readiness_error}"
        );
        let wait_error = completion.wait().unwrap_err().to_string();
        assert!(wait_error.contains("outside 0..5"), "{wait_error}");
    }
}

impl Drop for MlxRealtimeCompletionInner {
    fn drop(&mut self) {
        match self.event.is_complete() {
            Ok(true) => {}
            Ok(false) | Err(_) => {
                let _ = self.event.synchronize();
            }
        }
    }
}

/// Family-blind MLX materialization and tensor operations for prepared frames.
pub struct MlxRealtimeFrameTensorMechanisms<'a> {
    stream: &'a Stream,
}

impl<'a> MlxRealtimeFrameTensorMechanisms<'a> {
    /// Binds portable frame tensor operations to one selected MLX stream.
    pub const fn new(stream: &'a Stream) -> Self {
        Self { stream }
    }
}

impl RealtimeHostTokenMaterializer for MlxRealtimeFrameTensorMechanisms<'_> {
    type Tensor = MlxTensor;
    type Error = Error;

    fn materialize_i32(
        &mut self,
        values: &[i32],
        shape: [usize; 2],
    ) -> Result<Self::Tensor, Self::Error> {
        let shape = shape
            .into_iter()
            .map(|dimension| {
                i32::try_from(dimension)
                    .map_err(|_| Error::Parallel("realtime tensor dimension exceeds i32".into()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Array::from_slice(values, &shape)
            .copy(self.stream)
            .map(MlxTensor::from_array)
            .map_err(Into::into)
    }
}

impl RealtimeFrameTensorMechanisms for MlxRealtimeFrameTensorMechanisms<'_> {
    type Tensor = MlxTensor;
    type Error = Error;

    fn column(
        &mut self,
        matrix: &Self::Tensor,
        column: usize,
    ) -> Result<Self::Tensor, Self::Error> {
        let column = i32::try_from(column)
            .map_err(|_| Error::Parallel("realtime tensor column exceeds i32".into()))?;
        matrix
            .as_array()
            .try_index_device((.., column..column + 1), self.stream)
            .map(MlxTensor::from_array)
            .map_err(Into::into)
    }

    fn filled_column(&mut self, token: i32, batch: usize) -> Result<Self::Tensor, Self::Error> {
        let batch = i32::try_from(batch)
            .map_err(|_| Error::Parallel("realtime tensor batch exceeds i32".into()))?;
        Array::full::<i32>(&[batch, 1], Array::from_int(token), self.stream)
            .map(MlxTensor::from_array)
            .map_err(Into::into)
    }

    fn stack_columns(
        &mut self,
        columns: &[Self::Tensor],
        batch: usize,
    ) -> Result<Self::Tensor, Self::Error> {
        if columns.is_empty() {
            let batch = i32::try_from(batch)
                .map_err(|_| Error::Parallel("realtime tensor batch exceeds i32".into()))?;
            return Array::zeros::<i32>(&[batch, 0], self.stream)
                .map(MlxTensor::from_array)
                .map_err(Into::into);
        }
        let columns = columns.iter().map(MlxTensor::as_array).collect::<Vec<_>>();
        stack_axis(&columns, 1, self.stream)?
            .squeeze_axes(&[-1], self.stream)
            .map(MlxTensor::from_array)
            .map_err(Into::into)
    }
}

/// Exact MLX completion creation for one shared-coordinator Moshi frame.
pub struct MlxRealtimeFrameCompletionMechanism {
    token_validations: Option<TokenValidationScope>,
    execution_resources: Option<Arc<neutral_moshi::SelectedRealtimeResources>>,
}

impl MlxRealtimeFrameCompletionMechanism {
    /// Starts the one token-validation scope owned by this frame submission.
    pub fn begin() -> Result<Self, Error> {
        TokenValidationScope::begin()
            .map(|token_validations| Self {
                token_validations: Some(token_validations),
                execution_resources: None,
            })
            .map_err(Into::into)
    }

    fn begin_for_execution(
        resources: Arc<neutral_moshi::SelectedRealtimeResources>,
    ) -> Result<Self, Error> {
        Self::begin().map(|mut mechanism| {
            mechanism.execution_resources = Some(resources);
            mechanism
        })
    }
}

impl<T>
    RealtimeFrameCompletionMechanism<
        MlxTensor,
        T,
        (
            MlxTensor,
            eredu_architectures::moshi::ForwardContext<MlxTensor>,
        ),
    > for MlxRealtimeFrameCompletionMechanism
where
    T: Deref<Target = MlxKeyValueState>,
{
    type Completion = MlxRealtimeCompletion;
    type Error = Error;

    fn complete(
        &mut self,
        input: MaterializedRealtimeInput<MlxTensor>,
        output: &CompletedRealtimeFrame<MlxTensor, MlxTensor>,
        model_state: &T,
        payload_history: &RealtimePayloadHistory<MlxTensor>,
        execution: Option<(
            MlxTensor,
            eredu_architectures::moshi::ForwardContext<MlxTensor>,
        )>,
    ) -> Result<Self::Completion, RealtimeCompletionCreationError<Self::Completion, Self::Error>>
    {
        let token_validations = self.token_validations.take().ok_or_else(|| {
            RealtimeCompletionCreationError::before_submission(Error::Parallel(
                "MLX realtime completion scope was already consumed".into(),
            ))
        })?;
        let (_, _, input_audio, forced_audio, forced_text, _, _) = input.into_parts();
        let mut retained = vec![input_audio.into_array()];
        retained.extend(forced_audio.map(MlxTensor::into_array));
        retained.extend(forced_text.map(MlxTensor::into_array));
        retained.extend([
            output.text().as_array().clone(),
            output.decision_audio().as_array().clone(),
            output.sampled_audio().as_array().clone(),
        ]);
        retained.extend(output.aligned_audio().map(|value| value.as_array().clone()));
        retained.extend(
            output
                .diagnostics()
                .iter()
                .map(|value| value.as_array().clone()),
        );
        retained.extend(
            payload_history
                .retained_values()
                .map(|value| value.as_array().clone()),
        );
        retained.extend(model_state.retained_arrays().into_iter().cloned());
        if let Some((text_logits, forward)) = execution {
            retained.push(text_logits.into_array());
            retained.extend(
                forward
                    .temporal_mask()
                    .map(|value| value.as_array().clone()),
            );
            retained.extend(
                forward
                    .temporal_output()
                    .map(|value| value.as_array().clone()),
            );
            retained.extend(forward.text_logits().map(|value| value.as_array().clone()));
            retained.extend(
                forward
                    .previous_depth_token()
                    .map(|value| value.as_array().clone()),
            );
        }
        // `submit_retained` converts a failed event creation into this
        // pre-submission category only after synchronously draining every
        // retained root, so no unowned native work crosses this boundary.
        MlxRealtimeCompletion::submit_retained_with_resources(
            retained,
            token_validations.finish(),
            self.execution_resources.take(),
        )
        .map_err(RealtimeCompletionCreationError::before_submission)
    }

    fn retained_resources(&self, completion: &Self::Completion) -> usize {
        completion.retained_resources()
    }
}

struct MlxSelectedRealtimeFrameExecutor<'a> {
    model: &'a mut MlxRealtimeExecution,
}

impl
    PreparedRealtimeFrameExecutor<
        MlxSamplingBackend,
        GenerationSampler,
        MlxKeyValueTransactionBranch,
    > for MlxSelectedRealtimeFrameExecutor<'_>
{
    type Error = Error;
    type Retained = (
        MlxTensor,
        eredu_architectures::moshi::ForwardContext<MlxTensor>,
    );

    fn execute(
        &mut self,
        model_state: &mut MlxKeyValueTransactionBranch,
        temporal: &[MlxTensor],
        driver: &mut eredu_runtime::SequentialDecisionDriver<MlxSamplingBackend, GenerationSampler>,
        context: &Stream,
    ) -> Result<Self::Retained, Self::Error> {
        self.model
            .execute_selected_realtime(&mut *model_state, temporal, driver, context)
    }
}

fn submit_scheduled_realtime_frame(
    model: &mut MoshiRealtimeExecution<MlxRealtimeExecution>,
    branch: &mut MlxFrameSessionBranch,
    frame: &RealtimeInputFrame,
    stream: &Stream,
) -> Result<MlxPrepublicationFrame, Error> {
    let ingress = eredu_architectures::moshi::realtime_ingress_contract(model.execution_config())
        .map_err(Error::ArchitectureModel)?;
    let payload_contract = branch
        .payload_contract(&ingress)
        .map_err(|error| Error::Parallel(error.to_string()))?;
    let mut host = MlxRealtimeFrameTensorMechanisms::new(stream);
    let mut tensors = MlxRealtimeFrameTensorMechanisms::new(stream);
    let mut completion = MlxRealtimeFrameCompletionMechanism::begin_for_execution(
        model.executor().completion_resources(),
    )?;
    let mut executor = MlxSelectedRealtimeFrameExecutor {
        model: model.executor_mut(),
    };
    let submitted = execute_realtime_frame::<MlxSamplingBackend, _, _, _, _, _, _, _, _>(
        &ingress,
        &payload_contract,
        frame,
        branch.generation_mut(),
        &eredu_architectures::moshi::realtime_decision_execution(),
        &mut host,
        &mut tensors,
        &mut executor,
        &mut completion,
        stream,
    )
    .map_err(|error| Error::Parallel(error.to_string()))?;
    Ok(PrepublicationRealtimeFrame::new(
        submitted,
        MlxRealtimeHostObserver::new(stream),
    ))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use eredu_architectures::moshi::EffectiveModelType;
    use eredu_checkpoint::{
        schema::StoredDtypeConstraint, AffineQuantization, StoredDtype, WeightQuantization,
    };
    use eredu_core::{
        scheduler::{RequestId, RequestStatus, SchedulerLimits},
        RealtimeFrameConvention, RealtimeFrameForcing, RealtimeFrameScheduleState,
        RealtimeInputFrame, RealtimeSampling, RealtimeSpeechConfig,
    };
    use eredu_runtime::{
        DenseDiskStreamLoadOptions, ExecutionResidency, LayerwiseLoadOptions,
        RealtimeGenerationState, RealtimeModelSessionIdentity, RealtimePayloadState,
        RealtimeSessionScheduler, WeightResidency,
    };
    use safetensors::tensor::{serialize_to_file, Dtype as SafeDtype, TensorView};
    use std::path::Path;

    fn prepare(path: &Path) -> RealtimePreparationPlan {
        eredu_architectures::moshi::prepare_realtime_model(path)
            .unwrap_or_else(|error| panic!("prepare realtime artifact {}: {error}", path.display()))
    }

    struct SelectedTestModel {
        backend: MlxRealtimeExecutionContext,
        model: MoshiRealtimeExecution<MlxRealtimeExecution>,
    }

    type TestScheduler = RealtimeSessionScheduler<
        RealtimePayloadState<MlxKeyValueState, MlxTensor>,
        GenerationSampler,
        RandomState,
        MlxRealtimeCompletion,
        MlxPrepublicationFrame,
    >;

    fn load_selected_test_model(
        backend: MlxRealtimeExecutionContext,
        preparation: RealtimePreparationPlan,
        options: MlxLoadRequest,
    ) -> SelectedTestModel {
        let selected =
            MlxRealtimeExecutionContext::select_realtime_execution(preparation, &options, false)
                .expect("select realtime model");
        let model = backend
            .materialize_realtime_execution(selected, options)
            .expect("load selected realtime model");
        SelectedTestModel { backend, model }
    }

    fn selected_scheduler(
        model: &SelectedTestModel,
        request: RequestId,
        sampling: RealtimeSampling,
    ) -> TestScheduler {
        let mut scheduler = TestScheduler::new(
            RealtimeModelSessionIdentity::from_selected(model.model.selected()),
            SchedulerLimits::new(1, 1).unwrap(),
        )
        .unwrap();
        let schedule = model.model.execution_config().frame_schedule().clone();
        let samplers =
            eredu_architectures::moshi::realtime_generation_samplers(&schedule, sampling).unwrap();
        let model_state = RealtimePayloadState::fresh(
            model
                .backend
                .new_realtime_model_state(&model.model)
                .unwrap(),
            schedule.clone(),
        );
        let random = model
            .backend
            .realize_random_state(sampling.is_stochastic().then_some(sampling.seed()))
            .unwrap();
        scheduler
            .register(
                request,
                RealtimeGenerationState::new(model_state, schedule, sampling, samplers, random)
                    .unwrap(),
            )
            .unwrap();
        scheduler
    }

    fn drive_selected_frame(
        model: &mut SelectedTestModel,
        scheduler: &mut TestScheduler,
        request: RequestId,
        frame: RealtimeInputFrame,
    ) -> RealtimeOutputFrame {
        scheduler.enqueue(request, frame).unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
        loop {
            let mut progress = scheduler
                .run_local_turn(std::time::Instant::now(), |_, frame, branch| {
                    model
                        .backend
                        .submit_realtime_frame(&mut model.model, frame, branch)
                })
                .unwrap();
            if let Some((_, _, output)) = progress.committed.pop() {
                return output.into_host_output().unwrap();
            }
            assert!(std::time::Instant::now() < deadline);
            std::thread::yield_now();
        }
    }

    #[test]
    fn realtime_session_capabilities_fail_closed_for_activation_inspection() {
        let available = realtime_session_capabilities();
        assert!(available.persistent_cache());
        assert!(available.output_observation());
        assert!(!available.activation_inspection());

        let options = MlxLoadRequest::default().with_required_session_capabilities(
            eredu_core::SessionCapabilities::default().with_activation_inspection(true),
        );
        let error = validate_realtime_session_requirements(&options).unwrap_err();
        match error {
            Error::SessionCapability(error) => {
                assert_eq!(error.capability(), "activation_inspection")
            }
            error => panic!("expected session capability error, got {error:?}"),
        }
    }

    #[test]
    fn realtime_capabilities_fail_closed_for_unimplemented_mechanisms_and_lowerings() {
        let mechanisms = mlx_realtime_mechanisms(false);
        assert!(!mechanisms.contains(&RealtimeMechanism::Collectives));
        assert!(!mechanisms.contains(&RealtimeMechanism::Observation));
        assert!(!mechanisms.contains(&RealtimeMechanism::Timing));

        let unsupported_transform = eredu_runtime::WeightLoweringDescriptor::new(
            SourceTensorEncoding::Safetensors(StoredDtype::F32),
            LinearFormat::Dense,
            vec![2, 2],
            vec![2, 2],
            None,
        )
        .unwrap();
        assert!(!mlx_supports_realtime_lowering(
            &unsupported_transform,
            WeightLoweringKind::Transform,
        ));
    }

    #[test]
    fn prepared_frame_tensor_mechanisms_preserve_canonical_matrix_geometry() {
        let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let mut host = MlxRealtimeFrameTensorMechanisms::new(&stream);
        let matrix = host
            .materialize_i32(&[1, 2, 3, 4], [2, 2])
            .expect("materialize portable frame matrix");
        assert_eq!(matrix.as_array().shape(), &[2, 2]);
        assert_eq!(matrix.as_array().dtype(), Dtype::Int32);

        let mut tensors = MlxRealtimeFrameTensorMechanisms::new(&stream);
        let left = tensors.column(&matrix, 0).expect("select left column");
        let right = tensors.column(&matrix, 1).expect("select right column");
        assert_eq!(left.as_array().shape(), &[2, 1]);
        let stacked = tensors
            .stack_columns(&[left, right], 2)
            .expect("stack canonical columns");
        assert_eq!(stacked.as_array().shape(), &[2, 2]);
        let empty = tensors
            .stack_columns(&[], 2)
            .expect("stack empty generated-codebook set");
        assert_eq!(empty.as_array().shape(), &[2, 0]);
        let padding = tensors
            .filled_column(7, 2)
            .expect("materialize padding column");
        assert_eq!(padding.as_array().shape(), &[2, 1]);
    }

    const TINY_NATIVE_CONFIG: &str = r#"{
        "model_type": "moshi",
        "dim": 32,
        "text_card": 32,
        "n_q": 2,
        "dep_q": 1,
        "generated_audio_codebooks": 1,
        "card": 32,
        "num_heads": 4,
        "num_layers": 1,
        "dim_feedforward": 48,
        "causal": true,
        "context": 7,
        "max_period": 10000.0,
        "positional_embedding": "rope",
        "depformer_dim": 32,
        "depformer_dim_feedforward": 48,
        "depformer_num_heads": 4,
        "depformer_num_layers": 1,
        "depformer_context": 3,
        "depformer_max_period": 10000.0,
        "depformer_pos_emb": "none",
        "delays": [0, 0, 1]
    }"#;

    #[derive(Debug, Eq, PartialEq)]
    struct TinyFrameTokens {
        text: Vec<i32>,
        sampled_audio: Vec<i32>,
        output_audio: Option<Vec<i32>>,
    }

    pub(crate) fn write_tiny_native_artifact(
        directory: &Path,
        quantization: Option<WeightQuantization>,
    ) {
        let mut config_json = serde_json::from_str::<serde_json::Value>(TINY_NATIVE_CONFIG)
            .expect("tiny native JSON");
        if let Some(quantization) = quantization {
            config_json.as_object_mut().unwrap().insert(
                "quantization".into(),
                serde_json::to_value(quantization).expect("serialize tiny quantization"),
            );
        }
        let config_json = serde_json::to_string_pretty(&config_json).unwrap();
        let config = eredu_architectures::moshi::MoshiConfig::from_json(&config_json)
            .expect("tiny native Moshi config");
        let plan = eredu_architectures::moshi::safetensors_plan(&config)
            .expect("tiny native SafeTensors plan");
        assert!(plan.layout_groups.is_empty());

        // Derive every physical name and shape from the strict architecture
        // catalog. Zero matrices make greedy decisions exact across dense and
        // load-time packed execution; unit normalization scales remain valid.
        let tensors = plan
            .common_tensors
            .iter()
            .map(|constraint| {
                let dtype = match &constraint.dtype {
                    StoredDtypeConstraint::Exact(dtype) => dtype.clone(),
                    StoredDtypeConstraint::Floating => StoredDtype::F32,
                    StoredDtypeConstraint::OneOf(dtypes) => dtypes
                        .iter()
                        .find(|dtype| **dtype == StoredDtype::F32)
                        .or_else(|| dtypes.first())
                        .cloned()
                        .expect("validated catalog dtype set"),
                };
                let elements = constraint.shape.iter().product::<usize>();
                let (dtype, bytes) = match dtype {
                    StoredDtype::F32 => {
                        let value = if constraint.key.contains("norm")
                            || constraint.key.ends_with(".scales")
                        {
                            1.0f32
                        } else {
                            0.0f32
                        };
                        (
                            SafeDtype::F32,
                            std::iter::repeat_n(value, elements)
                                .flat_map(f32::to_le_bytes)
                                .collect::<Vec<_>>(),
                        )
                    }
                    StoredDtype::U32 => (
                        SafeDtype::U32,
                        std::iter::repeat_n(0u32, elements)
                            .flat_map(u32::to_le_bytes)
                            .collect::<Vec<_>>(),
                    ),
                    StoredDtype::U8 => (
                        SafeDtype::U8,
                        vec![
                            if constraint.key.ends_with(".scales") {
                                127
                            } else {
                                0
                            };
                            elements
                        ],
                    ),
                    dtype => panic!("tiny native writer does not support {dtype:?}"),
                };
                (
                    constraint.key.clone(),
                    constraint.shape.clone(),
                    dtype,
                    bytes,
                )
            })
            .collect::<Vec<_>>();
        let views = tensors.iter().map(|(name, shape, dtype, bytes)| {
            (
                name.as_str(),
                TensorView::new(*dtype, shape.clone(), bytes).expect("catalog-derived tensor view"),
            )
        });
        std::fs::write(directory.join("config.json"), config_json)
            .expect("write tiny native config");
        serialize_to_file(views, None, &directory.join("model.safetensors"))
            .expect("write tiny native SafeTensors artifact");
    }

    fn artifact_files(directory: &Path) -> std::collections::BTreeSet<String> {
        std::fs::read_dir(directory)
            .expect("read tiny artifact directory")
            .map(|entry| {
                entry
                    .expect("tiny artifact entry")
                    .file_name()
                    .to_string_lossy()
                    .into_owned()
            })
            .collect()
    }

    fn host_i32(array: &Array) -> Vec<i32> {
        array
            .evaluated()
            .expect("evaluate realtime token array")
            .as_slice::<i32>()
            .to_vec()
    }

    fn run_tiny_realtime_frames(model: &mut SelectedTestModel) -> Vec<TinyFrameTokens> {
        let request = RequestId::new(8);
        let mut scheduler = selected_scheduler(model, request, RealtimeSampling::greedy());
        let inputs = [
            RealtimeInputFrame::new(1, vec![1]),
            RealtimeInputFrame::new(1, vec![2])
                .with_forced_text(vec![7])
                .with_forced_generated_audio(vec![9]),
            RealtimeInputFrame::new(1, vec![3]).with_forced_text(vec![11]),
            RealtimeInputFrame::new(1, vec![4]).with_forced_generated_audio(vec![13]),
            RealtimeInputFrame::new(1, vec![5]),
        ];
        let frames = inputs
            .into_iter()
            .map(|input| {
                let output = drive_selected_frame(model, &mut scheduler, request, input);
                TinyFrameTokens {
                    text: output.text_tokens().to_vec(),
                    sampled_audio: output.sampled_audio_tokens().to_vec(),
                    output_audio: output.output_audio_tokens().map(<[i32]>::to_vec),
                }
            })
            .collect();
        scheduler
            .finish(request)
            .expect("finish tiny realtime request");
        frames
    }

    fn verify_tiny_native_hardware_matrix() {
        let directory = tempfile::tempdir().expect("tiny native artifact directory");
        write_tiny_native_artifact(directory.path(), None);
        let original_files = artifact_files(directory.path());
        assert_eq!(
            original_files,
            std::collections::BTreeSet::from([
                "config.json".to_string(),
                "model.safetensors".to_string(),
            ])
        );

        let device = safemlx::Device::new(safemlx::DeviceType::Gpu, 0);
        let weights_device = safemlx::Device::new(safemlx::DeviceType::Cpu, 0);
        let stream = Stream::new_with_device(&device);
        let weights_stream = Stream::new_with_device(&weights_device);
        let policies = [
            (
                WeightResidency::fully_resident(),
                ExecutionResidency::FullyResident,
            ),
            (
                WeightResidency::layerwise_host(LayerwiseLoadOptions::default()),
                ExecutionResidency::LayerwiseHost,
            ),
            (
                WeightResidency::dense_disk_stream(
                    DenseDiskStreamLoadOptions::new(1 << 20, 1 << 20, 1, 1)
                        .expect("tiny dense stream policy"),
                ),
                ExecutionResidency::DenseDiskStream,
            ),
        ];
        let expected = vec![
            TinyFrameTokens {
                text: vec![0],
                sampled_audio: vec![0],
                output_audio: None,
            },
            TinyFrameTokens {
                text: vec![7],
                sampled_audio: vec![9],
                output_audio: Some(vec![0]),
            },
            TinyFrameTokens {
                text: vec![11],
                sampled_audio: vec![0],
                output_audio: Some(vec![9]),
            },
            TinyFrameTokens {
                text: vec![0],
                sampled_audio: vec![13],
                output_audio: Some(vec![0]),
            },
            TinyFrameTokens {
                text: vec![0],
                sampled_audio: vec![0],
                output_audio: Some(vec![13]),
            },
        ];

        for (residency, execution) in policies {
            let backend = MlxRealtimeExecutionContext::new(&stream, &weights_stream);
            let mut model = load_selected_test_model(
                backend,
                prepare(directory.path()),
                MlxLoadRequest::default().with_weight_residency(residency),
            );
            assert_eq!(model.model.executor().metadata().residency(), execution);
            assert_eq!(run_tiny_realtime_frames(&mut model), expected);
            let report = model
                .model
                .executor()
                .residency_report()
                .expect("tiny residency report");
            assert!(report.initialized());
            assert!(report.weight_store().physical_reads > 0);
            if execution == ExecutionResidency::DenseDiskStream {
                let dense = model
                    .model
                    .executor()
                    .dense_stream_report()
                    .expect("tiny dense stream report")
                    .expect("selected dense-stream policy has a report");
                assert!(dense.planned_layer_count() > 0);
                assert!(dense.decode_forwards() > 0);
            }
        }

        for (request, quantization) in [
            (
                eredu_core::QuantizationRequest::Affine {
                    group_size: 32,
                    bits: 4,
                },
                WeightQuantization::Affine(AffineQuantization::new(32, 4).unwrap()),
            ),
            (
                eredu_core::QuantizationRequest::MxFp4,
                WeightQuantization::MxFp4,
            ),
        ] {
            let backend = MlxRealtimeExecutionContext::new(&stream, &weights_stream);
            let mut model = load_selected_test_model(
                backend,
                prepare(directory.path()),
                MlxLoadRequest::with_quantization(request),
            );
            let metadata = model.model.executor().metadata();
            assert_eq!(metadata.quantization(), Some(quantization));
            let materialization = metadata
                .materialization()
                .expect("load-time quantization telemetry");
            assert!(materialization.transformed_weights > 0);
            assert!(materialization.source_bytes_read > 0);
            assert!(materialization.output_bytes > 0);
            assert_eq!(run_tiny_realtime_frames(&mut model), expected);
            drop(model);
            assert_eq!(
                artifact_files(directory.path()),
                original_files,
                "load-time {quantization:?} created a disk artifact"
            );
        }

        for quantization in [
            WeightQuantization::Affine(AffineQuantization::new(32, 4).unwrap()),
            WeightQuantization::MxFp4,
        ] {
            let packed_directory = tempfile::tempdir().expect("tiny packed artifact directory");
            write_tiny_native_artifact(packed_directory.path(), Some(quantization));
            let original_files = artifact_files(packed_directory.path());
            let backend = MlxRealtimeExecutionContext::new(&stream, &weights_stream);
            let mut model = load_selected_test_model(
                backend,
                prepare(packed_directory.path()),
                MlxLoadRequest::default(),
            );
            let metadata = model.model.executor().metadata();
            assert_eq!(metadata.quantization(), Some(quantization));
            assert_eq!(metadata.materialization(), None);
            assert_eq!(run_tiny_realtime_frames(&mut model), expected);
            drop(model);
            assert_eq!(artifact_files(packed_directory.path()), original_files);
        }
    }

    #[test]
    #[ignore = "requires local MLX Metal execution"]
    fn moshi_mlx_scheduler_transaction_rollback_release_resume() {
        let directory = tempfile::tempdir().expect("tiny scheduler artifact directory");
        write_tiny_native_artifact(directory.path(), None);
        let execution = crate::backend::ExecutionContext::new(safemlx::Device::new(
            safemlx::DeviceType::Gpu,
            0,
        ));
        let weights = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let backend = MlxRealtimeExecutionContext::new(execution.stream(), &weights);
        let mut model = load_selected_test_model(
            backend,
            prepare(directory.path()),
            MlxLoadRequest::default(),
        );
        let request = RequestId::new(81);
        let mut scheduler = selected_scheduler(&model, request, RealtimeSampling::greedy());

        drive_selected_frame(
            &mut model,
            &mut scheduler,
            request,
            RealtimeInputFrame::new(1, vec![1]),
        );
        assert_eq!(
            scheduler
                .request_state(request)
                .unwrap()
                .generation()
                .schedule_state()
                .frontier(),
            1
        );
        let released = scheduler.release(request).unwrap();
        scheduler.resume(request, released).unwrap();

        drive_selected_frame(
            &mut model,
            &mut scheduler,
            request,
            RealtimeInputFrame::new(1, vec![2])
                .with_forced_text(vec![7])
                .with_forced_generated_audio(vec![9]),
        );
        assert_eq!(
            scheduler
                .request_state(request)
                .unwrap()
                .generation()
                .schedule_state()
                .frontier(),
            2
        );
        let released = scheduler.release(request).unwrap();
        scheduler.resume(request, released).unwrap();

        scheduler
            .enqueue(request, RealtimeInputFrame::new(1, vec![-1]))
            .unwrap();
        let error = match scheduler.run_local_turn(std::time::Instant::now(), |_, frame, branch| {
            model
                .backend
                .submit_realtime_frame(&mut model.model, frame, branch)
        }) {
            Ok(_) => panic!("invalid realtime input unexpectedly submitted"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("Audio token -1"), "{error}");
        assert_eq!(
            scheduler.request_status(request),
            Some(RequestStatus::Failed)
        );
        assert!(scheduler.request_state(request).is_none());
    }

    #[test]
    #[ignore = "runs the MLX operator, transaction, and tiny native model conformance suite"]
    fn moshi_mlx_conformance_suite() {
        verify_tiny_native_hardware_matrix();
        const TESTS: &[&str] = &[
            "backend::nn::shared::neutral_semantic_operator_tests::mlx_dense_fused_projection_equivalence",
            "backend::nn::shared::neutral_semantic_operator_tests::mlx_affine_fused_projection_equivalence",
            "backend::nn::shared::neutral_semantic_operator_tests::mlx_mxfp4_fused_projection_equivalence",
            "backend::nn::shared::neutral_semantic_operator_tests::mlx_sentinel_embedding_validation",
            "backend::nn::shared::neutral_semantic_operator_tests::mlx_multi_table_embedding_sum_is_ordered_and_sentinel_safe",
            "backend::runtime::cache::state::semantic_transaction_tests::paged_depth_segment_reset_preserves_temporal_pages_and_later_rollback",
            "backend::runtime::cache::state::semantic_transaction_tests::mlx_realtime_transaction_paged_rollback_release_resume",
            "backend::runtime::residency::manager::tests::cross_unit_alias_reacquisition_reuses_one_pinned_owner_read",
            "backend::runtime::generation::backend::tests::mlx_token_domain_validation_is_deferred_to_completion",
            "composition::mlx::realtime::tests::mlx_realtime_input_domains_are_deferred_and_strict",
        ];
        let executable = std::env::current_exe().expect("current unit-test executable");
        for test in TESTS {
            let output = std::process::Command::new(&executable)
                .args(["--exact", test, "--ignored", "--nocapture"])
                .output()
                .unwrap_or_else(|error| {
                    panic!("failed to launch MLX conformance test {test}: {error}")
                });
            assert!(
                output.status.success(),
                "MLX conformance test {test} failed\n--- stdout ---\n{}\n--- stderr ---\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr),
            );
        }
    }

    #[test]
    fn partial_forcing_and_initialization_only_transition_are_portable() {
        let schedule = RealtimeSpeechConfig::new(
            4,
            2,
            2,
            3,
            100,
            64,
            RealtimeFrameConvention::AbsoluteDelayedSlots,
            vec![0, 0, 1, 0, 1],
        )
        .unwrap();
        let mut state = RealtimeFrameScheduleState::new(schedule.clone());
        let transition = state
            .advance(
                &schedule,
                &RealtimeFrameForcing::new(false, vec![true, false]),
            )
            .unwrap();
        assert!(!transition.model_call_required());
        assert_eq!(transition.forced_placements().len(), 1);
    }

    #[test]
    fn portable_realtime_input_domains_are_strict_before_materialization() {
        let config = eredu_architectures::moshi::MoshiConfig::from_json(TINY_NATIVE_CONFIG)
            .expect("tiny native Moshi config");
        let ingress = eredu_architectures::moshi::realtime_ingress_contract(&config).unwrap();
        ingress
            .validate(
                &RealtimeInputFrame::new(1, vec![32])
                    .with_forced_text(vec![32])
                    .with_forced_generated_audio(vec![32]),
            )
            .unwrap();
        for input in [
            RealtimeInputFrame::new(1, vec![-1]),
            RealtimeInputFrame::new(1, vec![33]),
            RealtimeInputFrame::new(1, vec![0]).with_forced_text(vec![33]),
            RealtimeInputFrame::new(1, vec![0]).with_forced_generated_audio(vec![33]),
        ] {
            assert!(ingress.validate(&input).is_err());
        }
    }

    fn required_fixture_array<'a>(
        fixture: &'a std::collections::HashMap<String, Array>,
        key: &str,
    ) -> &'a Array {
        fixture
            .get(key)
            .unwrap_or_else(|| panic!("teacher-forced fixture is missing tensor {key}"))
    }

    fn assert_token_array_equal(actual: &Array, expected: &Array, label: &str, stream: &Stream) {
        assert_eq!(
            actual.shape(),
            expected.shape(),
            "shape mismatch for {label}"
        );
        assert!(
            actual
                .eq(expected, stream)
                .expect("token comparison")
                .all(None, stream)
                .expect("token comparison reduction")
                .item::<bool>(stream),
            "token fixture differs at {label}"
        );
    }

    fn run_personaplex_frame_fixture(
        model: &mut SelectedTestModel,
        fixture: &std::collections::HashMap<String, Array>,
        prefix: &str,
        forced: bool,
    ) {
        let stream = model.backend.stream().clone();
        let user_key = if forced {
            format!("{prefix}.user_audio")
        } else {
            format!("{prefix}.input_audio")
        };
        let user = required_fixture_array(fixture, &user_key);
        let agent =
            forced.then(|| required_fixture_array(fixture, &format!("{prefix}.agent_audio")));
        let text = forced.then(|| required_fixture_array(fixture, &format!("{prefix}.text")));
        let request = RequestId::new(91);
        let mut scheduler = selected_scheduler(model, request, RealtimeSampling::greedy());
        let mut sampled = Vec::new();
        let mut output_audio = Vec::new();
        let mut emitted_steps = Vec::new();
        for step in 0..user.dim(2) {
            let user_step = user
                .try_index_device((.., .., step), &stream)
                .expect("PersonaPlex user frame");
            let mut input = RealtimeInputFrame::new(
                usize::try_from(user_step.dim(0)).unwrap(),
                host_i32(&user_step),
            );
            if let (Some(agent), Some(text)) = (agent, text) {
                let agent_step = agent
                    .try_index_device((.., .., step), &stream)
                    .expect("PersonaPlex agent frame");
                let text_step = text
                    .try_index_device((.., .., step), &stream)
                    .expect("PersonaPlex text frame");
                input = input
                    .with_forced_generated_audio(host_i32(&agent_step))
                    .with_forced_text(host_i32(&text_step));
            }
            let output = drive_selected_frame(model, &mut scheduler, request, input);
            if step > 0 {
                let values = output
                    .text_tokens()
                    .iter()
                    .chain(output.sampled_audio_tokens())
                    .copied()
                    .collect::<Vec<_>>();
                sampled.push(Array::from_slice(
                    &values,
                    &[
                        i32::try_from(output.batch()).unwrap(),
                        i32::try_from(values.len() / output.batch()).unwrap(),
                    ],
                ));
            }
            if let Some(audio) = output.output_audio_tokens() {
                output_audio.push(Array::from_slice(
                    audio,
                    &[
                        i32::try_from(output.batch()).unwrap(),
                        i32::try_from(audio.len() / output.batch()).unwrap(),
                    ],
                ));
                emitted_steps.push(step);
            }
        }
        scheduler.finish(request).unwrap();
        let sampled = stack_axis(&sampled, 2, &stream).expect("PersonaPlex sampled transcript");
        let output_audio =
            stack_axis(&output_audio, 2, &stream).expect("PersonaPlex delayed audio transcript");
        let emitted_steps = Array::from_slice(&emitted_steps, &[output_audio.dim(2)]);
        async_eval_with_event([&sampled, &output_audio, &emitted_steps])
            .unwrap()
            .synchronize()
            .unwrap();
        assert_token_array_equal(
            &sampled,
            required_fixture_array(fixture, &format!("{prefix}.expected_sampled")),
            &format!("{prefix}.expected_sampled"),
            &stream,
        );
        assert_token_array_equal(
            &output_audio,
            required_fixture_array(fixture, &format!("{prefix}.expected_output_audio")),
            &format!("{prefix}.expected_output_audio"),
            &stream,
        );
        assert_token_array_equal(
            &emitted_steps,
            required_fixture_array(fixture, &format!("{prefix}.expected_emitted_steps")),
            &format!("{prefix}.expected_emitted_steps"),
            &stream,
        );
    }

    fn run_native_seeded_fixture(
        model: &mut SelectedTestModel,
        fixture: &std::collections::HashMap<String, Array>,
    ) {
        let stream = model.backend.stream().clone();
        let input = required_fixture_array(fixture, "generation.input_audio");
        let seed = required_fixture_array(fixture, "generation.seeded.seed")
            .clone()
            .item::<i64>(&stream) as u64;
        let text_temperature =
            required_fixture_array(fixture, "generation.seeded.text_temperature")
                .clone()
                .item::<f32>(&stream);
        let audio_temperature =
            required_fixture_array(fixture, "generation.seeded.audio_temperature")
                .clone()
                .item::<f32>(&stream);
        let request = RequestId::new(92);
        let sampling = RealtimeSampling::new(text_temperature, audio_temperature, seed).unwrap();
        let mut scheduler = selected_scheduler(model, request, sampling);
        let mut text = Vec::new();
        let mut audio = Vec::new();
        for step in 0..input.dim(2) {
            let frame = input
                .try_index_device((.., .., step), &stream)
                .expect("native seeded input frame");
            let output = drive_selected_frame(
                model,
                &mut scheduler,
                request,
                RealtimeInputFrame::new(usize::try_from(frame.dim(0)).unwrap(), host_i32(&frame)),
            );
            text.push(Array::from_slice(
                output.text_tokens(),
                &[i32::try_from(output.batch()).unwrap()],
            ));
            if let Some(tokens) = output.output_audio_tokens() {
                audio.push(Array::from_slice(
                    tokens,
                    &[
                        i32::try_from(output.batch()).unwrap(),
                        i32::try_from(tokens.len() / output.batch()).unwrap(),
                    ],
                ));
            }
        }
        scheduler.finish(request).unwrap();
        let text = stack_axis(&text, 1, &stream).unwrap();
        let audio = if audio.is_empty() {
            Array::zeros::<i32>(
                &[
                    input.dim(0),
                    model
                        .model
                        .execution_config()
                        .frame_schedule()
                        .generated_audio_codebooks() as i32,
                    0,
                ],
                &stream,
            )
            .unwrap()
        } else {
            stack_axis(&audio, 2, &stream).unwrap()
        };
        async_eval_with_event([&text, &audio])
            .unwrap()
            .synchronize()
            .unwrap();
        assert_token_array_equal(
            &text,
            required_fixture_array(fixture, "generation.seeded.expected_text"),
            "generation.seeded.expected_text",
            &stream,
        );
        assert_token_array_equal(
            &audio,
            required_fixture_array(fixture, "generation.seeded.expected_audio"),
            "generation.seeded.expected_audio",
            &stream,
        );
    }

    #[test]
    #[ignore = "requires released PersonaPlex artifact and PyTorch realtime fixture"]
    fn moshi_personaplex_prompt_realtime_and_residency_parity() {
        let model_path = std::env::var_os("EREDU_PERSONAPLEX_FIXTURE").expect(
            "EREDU_PERSONAPLEX_FIXTURE must point at a released artifact when this ignored fixture test is explicitly enabled",
        );
        let fixture_path = std::env::var_os("EREDU_PERSONAPLEX_TEACHER_FIXTURE")
            .expect("EREDU_PERSONAPLEX_TEACHER_FIXTURE must accompany the model fixture");
        let execution = crate::backend::ExecutionContext::new(safemlx::Device::new(
            safemlx::DeviceType::Gpu,
            0,
        ));
        let weights = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let fixture = Array::load_safetensors(Path::new(&fixture_path), execution.stream())
            .expect("load PersonaPlex parity fixture");
        for residency in [
            WeightResidency::fully_resident(),
            WeightResidency::layerwise_host(LayerwiseLoadOptions::default()),
            WeightResidency::dense_disk_stream(
                DenseDiskStreamLoadOptions::new(1 << 30, 1 << 30, 1, 1).unwrap(),
            ),
        ] {
            let backend = MlxRealtimeExecutionContext::new(execution.stream(), &weights);
            let mut model = load_selected_test_model(
                backend,
                prepare(Path::new(&model_path)),
                MlxLoadRequest::default().with_weight_residency(residency),
            );
            assert_eq!(
                model.model.execution_config().effective_model_type(),
                EffectiveModelType::PersonaPlex
            );
            run_personaplex_frame_fixture(&mut model, &fixture, "generation", false);
            run_personaplex_frame_fixture(&mut model, &fixture, "prompt", true);
        }
    }

    #[test]
    #[ignore = "requires released native Moshi artifact and seeded MLX fixture"]
    fn moshi_native_multiframe_seeded_realtime_parity() {
        let model_path = std::env::var_os("EREDU_MOSHI_FIXTURE").expect(
            "EREDU_MOSHI_FIXTURE must point at a released artifact when this ignored fixture test is explicitly enabled",
        );
        let fixture_path = std::env::var_os("EREDU_MOSHI_TEACHER_FIXTURE")
            .expect("EREDU_MOSHI_TEACHER_FIXTURE must accompany the model fixture");
        let execution = crate::backend::ExecutionContext::new(safemlx::Device::new(
            safemlx::DeviceType::Gpu,
            0,
        ));
        let weights = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let backend = MlxRealtimeExecutionContext::new(execution.stream(), &weights);
        let mut model = load_selected_test_model(
            backend,
            prepare(Path::new(&model_path)),
            MlxLoadRequest::default(),
        );
        let fixture = Array::load_safetensors(Path::new(&fixture_path), execution.stream())
            .expect("load native seeded fixture");
        run_native_seeded_fixture(&mut model, &fixture);
    }

    #[test]
    #[ignore = "requires EREDU_MOSHI_FIXTURE and an MLX runtime"]
    fn moshi_neutral_session_hook() {
        let fixture = std::env::var_os("EREDU_MOSHI_FIXTURE").expect(
            "EREDU_MOSHI_FIXTURE must point at a released artifact when this ignored fixture test is explicitly enabled",
        );
        assert!(
            Path::new(&fixture).exists(),
            "EREDU_MOSHI_FIXTURE does not exist: {}",
            Path::new(&fixture).display()
        );
        let device = safemlx::Device::new(safemlx::DeviceType::Cpu, 0);
        let stream = Stream::new_with_device(&device);
        let backend = MlxRealtimeExecutionContext::new(&stream, &stream);
        let scheduler_model = load_selected_test_model(
            backend,
            prepare(Path::new(&fixture)),
            MlxLoadRequest::default(),
        );
        let request = RequestId::new(93);
        let _scheduler = selected_scheduler(&scheduler_model, request, RealtimeSampling::greedy());
    }

    #[test]
    #[ignore = "requires EREDU_PERSONAPLEX_FIXTURE and an MLX runtime"]
    fn moshi_personaplex_fixture_session_hook() {
        let fixture = std::env::var_os("EREDU_PERSONAPLEX_FIXTURE").expect(
            "EREDU_PERSONAPLEX_FIXTURE must point at a released artifact when this ignored fixture test is explicitly enabled",
        );
        assert!(
            Path::new(&fixture).exists(),
            "EREDU_PERSONAPLEX_FIXTURE does not exist: {}",
            Path::new(&fixture).display()
        );
        let device = safemlx::Device::new(safemlx::DeviceType::Cpu, 0);
        let stream = Stream::new_with_device(&device);
        let backend = MlxRealtimeExecutionContext::new(&stream, &stream);
        let scheduler_model = load_selected_test_model(
            backend,
            prepare(Path::new(&fixture)),
            MlxLoadRequest::default(),
        );
        assert_eq!(
            scheduler_model
                .model
                .execution_config()
                .effective_model_type(),
            EffectiveModelType::PersonaPlex
        );
        let request = RequestId::new(94);
        let _scheduler = selected_scheduler(&scheduler_model, request, RealtimeSampling::greedy());
    }
}
