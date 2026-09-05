use super::*;

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
    ) -> Result<PreparedMoshiRealtimeSource, Error> {
        select_realtime_model(preparation, options, collectives_supported)
    }

    /// Materializes an already selected architecture through MLX mechanisms.
    pub fn materialize_realtime_execution(
        &self,
        selected: PreparedMoshiRealtimeSource,
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

pub(super) const fn realtime_session_capabilities() -> eredu_core::SessionCapabilities {
    eredu_core::SessionCapabilities::new(true, true, false)
}

pub(super) fn validate_realtime_session_requirements(
    options: &MlxLoadRequest,
) -> Result<(), Error> {
    options
        .required_session_capabilities()
        .validate(&realtime_session_capabilities())?;
    Ok(())
}

fn materialize_realtime_model(
    selected: PreparedMoshiRealtimeSource,
    options: MlxLoadRequest,
    world: Option<Arc<Group>>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<MoshiRealtimeExecution<MlxRealtimeExecution>, Error> {
    if !selected.selected().selected().topology().is_replicated() {
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
) -> Result<PreparedMoshiRealtimeSource, Error> {
    validate_realtime_session_requirements(options)?;
    let request = moshi_realtime_request_from_normalized(
        options.checked_normalized()?.0,
        RealtimeObservationRequirements::new(true, []),
    )
    .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let inspected = inspect_moshi_realtime(preparation, request)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let capabilities = mlx_realtime_capabilities(inspected.requirements(), collectives_supported);
    let selected = select_inspected_moshi_realtime(inspected, &capabilities)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    prepare_selected_moshi_realtime_source(selected)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))
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
