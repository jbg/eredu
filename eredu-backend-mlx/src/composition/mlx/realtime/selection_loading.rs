use super::*;

/// MLX stream and collective mechanisms for neutral realtime execution.
pub struct MlxRealtimeExecutionContext {
    stream: Stream,
    weights_stream: Stream,
    world_group: Option<Arc<Group>>,
}

impl From<eredu_core::realtime_mechanisms::RealtimeContexts<crate::native::ExecutionContext>>
    for MlxRealtimeExecutionContext
{
    fn from(
        contexts: eredu_core::realtime_mechanisms::RealtimeContexts<
            crate::native::ExecutionContext,
        >,
    ) -> Self {
        Self::new(contexts.execution.stream(), contexts.weights.stream())
    }
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
    pub fn with_native_world(mut self, world: &safemlx::distributed::Group) -> Self {
        self.world_group = Some(Arc::new(Group::uncontracted(world)));
        self
    }

    /// Selected MLX execution stream.
    pub const fn stream(&self) -> &Stream {
        &self.stream
    }

    /// Opens ordinary component-binding mechanisms on the same streams.
    pub(super) fn component_binder(&self) -> crate::native::MlxComponentBinder {
        crate::native::MlxComponentBinder::new(&self.weights_stream, &self.stream)
    }

    pub(super) fn world_identity(&self) -> Option<(usize, usize)> {
        self.world_group
            .as_deref()
            .map(|group| (group.size(), group.rank()))
    }

    pub(super) fn open_materialization(
        &self,
        rank: Option<&crate::backend::MlxRankContext>,
    ) -> Result<MlxRealtimeBinding<'_>, Error> {
        if let Some(rank) = rank {
            rank.validate_execution_stream(&self.stream)?;
        }
        crate::backend::ordinary_retirement::reclaim();
        Ok(MlxRealtimeBinding { context: self })
    }

    pub(super) fn queue(&self) -> MlxRealtimeQueue {
        MlxRealtimeQueue {
            stream: self.stream.clone(),
            world: self.world_group.clone(),
            _weights_stream: self.weights_stream.clone(),
        }
    }

    /// Validates the retained queue against this exact native placement.
    pub fn validate_execution(
        &self,
        queue: &MlxRealtimeQueue,
    ) -> Result<(), eredu_core::BackendFailure> {
        let device = queue
            .stream
            .get_device()
            .map_err(eredu_core::BackendFailure::from_error)?;
        crate::backend::validate_native_execution_target(
            &device,
            queue.world.as_deref().map(Group::native_group),
            &self.stream,
            self.world_group.as_deref().map(Group::native_group),
        )
        .map_err(eredu_core::BackendFailure::from_error)
    }
}

/// Side-effect-free MLX support for realtime loading.
pub struct MlxRealtimeSupport {
    collectives_supported: bool,
}

impl MlxRealtimeSupport {
    /// Declares whether the selected route provides collectives.
    pub const fn new(collectives_supported: bool) -> Self {
        Self {
            collectives_supported,
        }
    }
}

impl RealtimeLoadSupport for MlxRealtimeSupport {
    fn independent_state_support(
        &self,
    ) -> &impl eredu_core::preparation::ReplicatedTextMechanismSupport {
        &crate::composition::mlx::replicated_text::MlxReplicatedTextSupport
    }
}

impl RealtimeMechanismSupport for MlxRealtimeSupport {
    fn facts(&self) -> RealtimeMechanismFacts {
        RealtimeMechanismFacts::new(
            eredu_nn::NeuralOperatorCapabilities::NONE,
            mlx_realtime_mechanisms(self.collectives_supported),
            [
                WeightResidencyMechanism::Resident,
                WeightResidencyMechanism::Windowed,
                WeightResidencyMechanism::DiskStreamed,
            ],
            NonZeroUsize::new(usize::MAX).expect("usize maximum is positive"),
            CommunicationCompletionCapabilities::new([
                CompletionCancellationMode::QuarantineUntilComplete,
            ])
            .expect("MLX quarantine completion capability is valid"),
            eredu_core::SessionCapabilities::new(true, true, false),
        )
        .with_state_lifecycle(
            StateLifecycleCapabilities::new()
                .with_transactions(true, true)
                .with_reset(true)
                .with_observation_retention(true),
        )
    }

    fn supports_lowering(
        &self,
        descriptor: &WeightLoweringDescriptor,
        transforms_payload: bool,
    ) -> bool {
        mlx_supports_realtime_lowering(descriptor, transforms_payload)
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
    descriptor: &WeightLoweringDescriptor,
    transforms_payload: bool,
) -> bool {
    if !matches!(descriptor.source(), SourceTensorEncoding::Safetensors(_)) {
        return false;
    }
    if transforms_payload {
        matches!(
            descriptor.executable(),
            LinearFormat::Affine(_) | LinearFormat::MxFp4
        )
    } else {
        matches!(
            descriptor.executable(),
            LinearFormat::Dense
                | LinearFormat::Affine(_)
                | LinearFormat::MxFp4
                | LinearFormat::GgufIQuant { .. }
                | LinearFormat::E4M3BlockFp8(_)
        )
    }
}

/// Materialization authority available after native target validation.
pub struct MlxRealtimeBinding<'a> {
    context: &'a MlxRealtimeExecutionContext,
}

impl<'a>
    eredu_core::realtime_mechanisms::RealtimeBindingMechanisms<
        crate::backend::nn::shared::MlxNeuralBackend,
    > for MlxRealtimeBinding<'a>
{
    type Queue = MlxRealtimeQueue;

    fn execution_queue(&self) -> Self::Queue {
        self.context.queue()
    }
    type Error = Error;
    type Group = Group;
    type Route = crate::backend::runtime::distributed::topology::CommunicationRouteRealization;
    type TensorMetadata = crate::backend::nn::shared::MlxCommunicationTensorMetadata;
    type Sampling = MlxSamplingBackend;
    type Materializer = crate::native::MlxComponentBinder;

    fn context(&self) -> &Stream {
        self.context.stream()
    }

    type CommunicationSetup =
        crate::backend::runtime::distributed::topology::NativeWorldManifestTransport<'a>;

    fn open_communication_setup(&self) -> Result<Self::CommunicationSetup, Error> {
        let world = self.context.world_group.as_deref().ok_or_else(|| {
            Error::Parallel("selected tensor parallelism requires a native world group".into())
        })?;
        Ok(Self::CommunicationSetup::new(
            world.native_group(),
            self.context.stream(),
        ))
    }

    type Rank = crate::backend::MlxDistributedSession;

    fn realize_rank(
        &self,
        prepared: eredu_core::communication::WorldCommunicationBinding<Self::CommunicationSetup>,
    ) -> Result<Self::Rank, Error> {
        crate::backend::MlxDistributedSession::from_prepared(prepared)
    }

    fn materializer(&self) -> Self::Materializer {
        self.context.component_binder()
    }
}
