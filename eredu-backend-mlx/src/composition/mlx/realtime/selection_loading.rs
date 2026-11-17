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

    /// Binds retained portable construction after validating the native target.
    pub fn bind_realtime<P, E, F>(
        &self,
        prepared: P,
        rank: Option<crate::backend::MlxRankContext>,
    ) -> Result<E, Error>
    where
        for<'a> P: eredu_runtime::PreparedRealtimeBinding<
            crate::backend::nn::shared::MlxNeuralBackend,
            MlxKeyValueState,
            MlxRealtimeBinding<'a>,
            Execution = E,
            Error = F,
        >,
        F: std::error::Error + Send + Sync + 'static,
    {
        let selected = prepared.realization();
        let size = selected.topology().world_size();
        let selected_rank = selected.rank();
        match (rank, self.world_group.as_deref()) {
            (None, None) if size == 1 => {}
            (Some(rank), world)
                if (rank.world_size(), rank.global_rank()) == (size, selected_rank)
                    && match world {
                        Some(world) => {
                            size > 1 && (world.size(), world.rank()) == (size, selected_rank)
                        }
                        None => size == 1,
                    } =>
            {
                rank.validate_execution_stream(&self.stream)?;
            }
            _ => {
                return Err(Error::Parallel(
                    "native rank or world group does not match the selected realtime target".into(),
                ))
            }
        }
        crate::initialize_allocator_cache_policy(false)?;
        crate::backend::ordinary_retirement::reclaim();
        let result = prepared
            .bind(&MlxRealtimeBinding { context: self })
            .map_err(|error| Error::Other(Box::new(error)));
        let trim = crate::trim_allocator_cache();
        result.and_then(|execution| {
            trim?;
            Ok(execution)
        })
    }

    /// Validates an execution handle before portable generation startup.
    pub fn validate_execution(
        &self,
        execution: &MlxRealtimeExecution,
    ) -> Result<(), eredu_core::BackendFailure> {
        execution
            .validate_context(&self.stream, self.world_group.as_deref())
            .map_err(eredu_core::BackendFailure::from_error)
    }

    /// Submits a portable plan on an unpublished scheduler branch.
    pub fn submit_realtime<P>(
        &self,
        execution: &MlxRealtimeExecution,
        plan: RealtimeFramePlan<P>,
        frame: &RealtimeInputFrame,
        branch: &mut MlxFrameSessionBranch,
    ) -> Result<MlxPrepublicationFrame<P>, Error>
    where
        P: RealtimeHostOutputProjection<observation::MlxHostChannels>,
    {
        execution.validate_context(&self.stream, self.world_group.as_deref())?;
        let mut executor = execution;
        submit_scheduled_realtime_frame(
            plan,
            &mut executor,
            || execution.retain_frame(),
            branch,
            frame,
            &self.stream,
        )
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

impl eredu_runtime::RealtimeLoadSupport for MlxRealtimeSupport {
    fn independent_state_support(&self) -> &impl eredu_runtime::ReplicatedTextMechanismSupport {
        &crate::composition::mlx::replicated_text::MlxReplicatedTextSupport
    }
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
            eredu_core::SessionCapabilities::new(true, true, false),
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

/// Materialization authority available after native target validation.
pub struct MlxRealtimeBinding<'a> {
    context: &'a MlxRealtimeExecutionContext,
}

impl
    eredu_runtime::RealtimeBindingMechanisms<
        crate::backend::nn::shared::MlxNeuralBackend,
        MlxKeyValueState,
    > for MlxRealtimeBinding<'_>
{
    type Execution = MlxRealtimeExecution;
    type Error = Error;
    type Group = Group;
    type Route = crate::backend::runtime::distributed::topology::CommunicationRouteRealization;
    type TensorMetadata = crate::backend::nn::shared::MlxCommunicationTensorMetadata;
    type Sampling = MlxSamplingBackend;
    type Sampler = GenerationSampler;
    type Materializer<A>
        = construction::MlxRealtimeConstructionMechanisms<A::Unit>
    where
        A: eredu_runtime::LayeredArchitecture<
                crate::backend::nn::shared::MlxNeuralBackend,
                MlxKeyValueState,
            > + 'static,
        A::Unit: 'static;

    fn context(&self) -> &Stream {
        self.context.stream()
    }

    fn realize_tensor(
        &self,
        manifest: eredu_runtime::CommunicationManifest,
        tensor_group: eredu_core::CollectiveGroupId,
    ) -> Result<
        eredu_runtime::RealtimeTensorResources<
            crate::backend::nn::shared::MlxNeuralBackend,
            Self::Group,
            Self::Route,
            Self::TensorMetadata,
        >,
        Error,
    > {
        let world = self.context.world_group.as_deref().ok_or_else(|| {
            Error::Parallel("selected tensor parallelism requires a native world group".into())
        })?;
        let distributed = crate::backend::MlxDistributedSession::from_manifest(
            &manifest,
            world.native_group(),
            self.context.stream(),
        )?;
        let (communication, parallel, executor) =
            distributed.into_component_communication(manifest, Some(tensor_group))?;
        let parallel = parallel
            .ok_or_else(|| Error::Parallel("selected tensor group was not realized".into()))?;
        Ok(eredu_runtime::RealtimeTensorResources {
            communication,
            parallel,
            executor,
        })
    }

    fn materializer<A>(
        &self,
        source: eredu_checkpoint::store::SharedCheckpointSource,
        residency: eredu_runtime::LayerWeightResidency,
        transform: Option<eredu_checkpoint::WeightQuantization>,
        layout: Option<&eredu_runtime::LocalModelLayout>,
    ) -> Self::Materializer<A>
    where
        A: eredu_runtime::LayeredArchitecture<
                crate::backend::nn::shared::MlxNeuralBackend,
                MlxKeyValueState,
            > + 'static,
        A::Unit: 'static,
    {
        construction::MlxRealtimeConstructionMechanisms::new(
            source,
            residency,
            transform,
            &self.context.weights_stream,
            layout.cloned(),
        )
    }

    fn materialization_metadata<A>(
        &self,
        materializer: &Self::Materializer<A>,
    ) -> Result<eredu_runtime::LayerwiseModelMetadata, Error>
    where
        A: eredu_runtime::LayeredArchitecture<
                crate::backend::nn::shared::MlxNeuralBackend,
                MlxKeyValueState,
            > + 'static,
        A::Unit: 'static,
    {
        materializer.metadata().cloned().ok_or_else(|| {
            Error::ArchitectureModel("realtime construction produced no metadata".into())
        })
    }

    fn bind_executor<E, F, R>(
        &self,
        executor: E,
        metadata: eredu_runtime::LayerwiseModelMetadata,
        source: eredu_checkpoint::store::SharedCheckpointSource,
    ) -> MlxRealtimeExecution
    where
        E: PreparedRealtimeFrameExecutor<
                MlxSamplingBackend,
                GenerationSampler,
                MlxKeyValueState,
                Error = F,
                Retained = R,
            > + eredu_runtime::RealtimeResidencyTelemetry<Error = Error>
            + 'static,
        F: std::fmt::Display,
        R: eredu_nn::RetainedTensorRoots<MlxTensor> + 'static,
    {
        MlxRealtimeExecution::bind(
            executor,
            metadata,
            source,
            self.context.world_group.clone(),
            self.context.stream(),
            self.context.weights_stream.clone(),
        )
    }
}
