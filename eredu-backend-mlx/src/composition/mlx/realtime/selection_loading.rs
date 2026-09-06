use super::*;

/// Exact cold realtime preparation inseparably paired with its native target.
///
/// Only cold selection constructs this value. Materialization accepts no new
/// load request, so a caller cannot replace the admitted rank/device binding.
///
/// ```compile_fail
/// use eredu_backend_mlx::native::MlxPreparedRealtimeExecution;
/// fn discard_selected_target(mut prepared: MlxPreparedRealtimeExecution) {
///     prepared.rank_context = None;
/// }
/// ```
pub struct MlxPreparedRealtimeExecution {
    source: PreparedMoshiRealtimeSource,
    rank_context: Option<crate::backend::MlxRankContext>,
}

/// MLX stream and collective mechanisms for neutral realtime execution.
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
    ) -> Result<MlxPreparedRealtimeExecution, Error> {
        select_realtime_model(preparation, options, collectives_supported)
    }

    /// Materializes an already selected architecture through MLX mechanisms.
    pub fn materialize_realtime_execution(
        &self,
        selected: MlxPreparedRealtimeExecution,
    ) -> Result<MoshiRealtimeExecution<MlxRealtimeExecution>, Error> {
        materialize_realtime_model(
            selected,
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
        model
            .executor()
            .validate_context(&self.stream, self.world_group.as_deref())?;
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
        model
            .executor()
            .validate_context(&self.stream, self.world_group.as_deref())?;
        submit_scheduled_realtime_frame(model, branch, frame, &self.stream)
    }
}

pub(super) const fn realtime_session_capabilities() -> eredu_core::SessionCapabilities {
    eredu_core::SessionCapabilities::new(true, true, false)
}

fn materialize_realtime_model(
    selected: MlxPreparedRealtimeExecution,
    world: Option<Arc<Group>>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<MoshiRealtimeExecution<MlxRealtimeExecution>, Error> {
    let MlxPreparedRealtimeExecution {
        source,
        rank_context,
    } = selected;
    if let Some(rank) = rank_context {
        rank.validate_execution_stream(stream)?;
    }
    neutral_moshi::materialize_selected(source, world, stream, weights_stream)
}

fn select_realtime_model(
    preparation: RealtimePreparationPlan,
    options: &MlxLoadRequest,
    collectives_supported: bool,
) -> Result<MlxPreparedRealtimeExecution, Error> {
    let (normalized, rank_context) = options.checked_normalized()?;
    let request = moshi_realtime_request_from_normalized(
        normalized,
        RealtimeObservationRequirements::new(true, []),
    )
    .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let inspected = inspect_moshi_realtime(preparation, request)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let capabilities = mlx_realtime_capabilities(inspected.requirements(), collectives_supported);
    let selected = select_inspected_moshi_realtime(inspected, &capabilities)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let source = prepare_selected_moshi_realtime_source(selected)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    Ok(MlxPreparedRealtimeExecution {
        source,
        rank_context,
    })
}

fn mlx_realtime_capabilities(
    requirements: &RealtimeArchitectureRequirements,
    collectives_supported: bool,
) -> RealtimeMechanismCapabilities {
    eredu_runtime::synthesize_realtime_capabilities(
        requirements,
        &MlxRealtimeSupport {
            collectives_supported,
        },
    )
}

struct MlxRealtimeSupport {
    collectives_supported: bool,
}

impl eredu_runtime::RealtimeMechanismSupport for MlxRealtimeSupport {
    fn facts(&self) -> eredu_runtime::RealtimeMechanismFacts {
        eredu_runtime::RealtimeMechanismFacts::new(
            eredu_nn::NeuralOperatorCapabilities::NONE,
            mlx_realtime_mechanisms(self.collectives_supported),
            [
                ExecutionResidency::FullyResident,
                ExecutionResidency::LayerwiseHost,
                ExecutionResidency::DenseDiskStream,
            ],
            NonZeroUsize::new(usize::MAX).expect("usize maximum is positive"),
            CommunicationCompletionCapabilities::new([
                CompletionCancellationMode::QuarantineUntilComplete,
            ])
            .expect("MLX quarantine completion capability is valid"),
            SessionCapabilities::new(true, true, false),
        )
        .with_state_lifecycle(
            eredu_runtime::StateLifecycleCapabilities::new()
                .with_transactions(true, true)
                .with_reset(true)
                .with_observation_retention(true),
        )
    }

    fn supports_lowering(
        &self,
        descriptor: &eredu_runtime::WeightLoweringDescriptor,
        kind: WeightLoweringKind,
    ) -> bool {
        mlx_supports_realtime_lowering(descriptor, kind)
    }

    fn state_component_placements(
        &self,
        component: &StateComponentPolicy,
    ) -> (
        Option<StateComponentPlacement>,
        Option<StateComponentPlacement>,
    ) {
        (
            mlx_supports_realtime_state_component(component)
                .then_some(StateComponentPlacement::Device),
            None,
        )
    }
}

pub(super) fn mlx_realtime_mechanisms(collectives_supported: bool) -> Vec<RealtimeMechanism> {
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

fn mlx_supports_realtime_state_component(component: &StateComponentPolicy) -> bool {
    matches!(
        component.role(),
        StateComponentRole::AttentionKeys
            | StateComponentRole::AttentionValues
            | StateComponentRole::CompressedLatent
            | StateComponentRole::RotaryKeys
            | StateComponentRole::Fixed(_)
    )
}

pub(super) fn mlx_supports_realtime_lowering(
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
