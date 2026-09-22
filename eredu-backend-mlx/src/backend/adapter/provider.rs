use super::*;

mod original_copy;
pub(crate) use original_copy::{
    OriginalCopyEnvironment, OriginalCopyEnvironmentError, PreparedOriginalCopyEnvironment,
    PreparedOriginalCopyEnvironmentError, RetainedOriginalCopyEnvironment,
};

/// MLX backend selected for a complete model/session.
pub struct MlxBackend<'a> {
    streams: BackendStreams,
    realized_device: Option<MlxDeviceIdentity>,
    world: Option<&'a safemlx::distributed::Group>,
    memory_ledger: eredu_runtime::working_memory::MemoryLedger,
}

enum BackendStreams {
    Ordinary { stream: Stream, weights: Stream },
    Prepared(crate::backend::managed_memory::gpu_stream::PreparedExecutionStreams),
}

impl MlxBackend<'static> {
    pub(crate) fn for_prepared_execution_plan(
        streams: crate::backend::managed_memory::gpu_stream::PreparedExecutionStreams,
        realized_device: MlxDeviceIdentity,
    ) -> Self {
        Self {
            streams: BackendStreams::Prepared(streams),
            realized_device: Some(realized_device),
            world: None,
            memory_ledger: crate::backend::managed_memory::ledger(),
        }
    }

    pub(crate) fn new(stream: &Stream, weights_stream: &Stream) -> Self {
        Self {
            streams: BackendStreams::Ordinary {
                stream: stream.clone(),
                weights: weights_stream.clone(),
            },
            realized_device: None,
            world: None,
            memory_ledger: crate::backend::managed_memory::ledger(),
        }
    }

    pub(crate) fn for_execution_plan(
        stream: &Stream,
        weights_stream: &Stream,
        realized_device: MlxDeviceIdentity,
    ) -> Self {
        Self {
            streams: BackendStreams::Ordinary {
                stream: stream.clone(),
                weights: weights_stream.clone(),
            },
            realized_device: Some(realized_device),
            world: None,
            memory_ledger: crate::backend::managed_memory::ledger(),
        }
    }
}

impl<'a> MlxBackend<'a> {
    pub(crate) fn with_prepared_distributed_world(
        streams: crate::backend::managed_memory::gpu_stream::PreparedExecutionStreams,
        world: &'a safemlx::distributed::Group,
    ) -> Self {
        Self {
            streams: BackendStreams::Prepared(streams),
            realized_device: None,
            world: Some(world),
            memory_ledger: crate::backend::managed_memory::ledger(),
        }
    }

    pub(crate) fn with_distributed_world(
        stream: &Stream,
        weights_stream: &Stream,
        world: &'a safemlx::distributed::Group,
    ) -> Self {
        Self {
            streams: BackendStreams::Ordinary {
                stream: stream.clone(),
                weights: weights_stream.clone(),
            },
            realized_device: None,
            world: Some(world),
            memory_ledger: crate::backend::managed_memory::ledger(),
        }
    }
    #[cfg(test)]
    pub(crate) fn retiring_stream_wrapper_control_bytes(&self) -> u64 {
        match &self.streams {
            BackendStreams::Prepared(streams) => streams.retiring_wrapper_control_bytes(),
            BackendStreams::Ordinary { .. } => 0,
        }
    }
    pub(crate) fn stream(&self) -> &Stream {
        match &self.streams {
            BackendStreams::Ordinary { stream, .. } => stream,
            BackendStreams::Prepared(streams) => streams.execution(),
        }
    }
    pub(crate) fn weights_stream(&self) -> &Stream {
        match &self.streams {
            BackendStreams::Ordinary { weights, .. } => weights,
            BackendStreams::Prepared(streams) => streams.source(),
        }
    }
    pub(crate) fn validate_original_stream_owners(&self) -> Result<(), Error> {
        match &self.streams {
            BackendStreams::Prepared(streams) => streams
                .validate_pool(&self.memory_ledger)
                .map_err(Error::GpuStreamOwnership),
            BackendStreams::Ordinary { .. } => Err(Error::PrefillControl(
                eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
            )),
        }
    }

    pub(crate) fn observe_original_streams_idle(
        &self,
    ) -> Result<(), crate::backend::managed_memory::gpu_stream::MlxStreamOwnershipError> {
        use crate::backend::managed_memory::gpu_stream::MlxStreamOwnershipError;
        match &self.streams {
            BackendStreams::Prepared(streams) => streams.observe_idle(&self.memory_ledger),
            BackendStreams::Ordinary { .. } => Err(MlxStreamOwnershipError::Accounting(
                eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
            )),
        }
    }

    pub(crate) fn memory_ledger(&self) -> &eredu_runtime::working_memory::MemoryLedger {
        &self.memory_ledger
    }

    #[cfg(test)]
    pub(crate) fn with_memory_ledger(
        mut self,
        pool: eredu_runtime::working_memory::MemoryLedger,
    ) -> Self {
        self.memory_ledger = pool;
        self
    }

    pub(crate) fn matches_prepared_target(&self, target: &MlxPreparedTarget) -> bool {
        target.matches(self.stream(), self.world)
    }

    pub(crate) fn validate_prepared_target(&self, target: &MlxPreparedTarget) -> Result<(), Error> {
        target.validate(self.stream(), self.world)
    }

    fn realize_selected_communication(
        &self,
        manifest: Option<&eredu_runtime::CommunicationManifest>,
        rank: Option<crate::backend::MlxRankContext>,
    ) -> Result<Option<MlxDistributedSession>, Error> {
        let Some(manifest) = manifest else {
            return match rank {
                None => Ok(None),
                Some(_) => Err(Error::Parallel(
                    "distributed MLX preparation has no architecture communication manifest".into(),
                )),
            };
        };
        let rank = rank.ok_or_else(|| {
            Error::Parallel("communication manifest has no MLX rank/device context".into())
        })?;
        #[cfg(test)]
        crate::tests::support::path_instrumentation::communication_realization_attempt();
        let world = self.world.ok_or_else(|| {
            Error::Parallel(
                "distributed model preparation requires native::distributed_backend".into(),
            )
        })?;
        rank.validate_execution_stream(self.stream())?;
        #[cfg(test)]
        crate::tests::support::path_instrumentation::manifest_communication_realization_attempt();
        MlxDistributedSession::from_manifest(manifest, world, self.stream()).map(Some)
    }

    fn prepare_layerwise_source(
        &self,
        sources: &eredu_architectures::prepared_sources::PreparedModelSources,
    ) -> Result<Option<crate::backend::runtime::execution::generic::PreparedLayerwiseManager>, Error>
    {
        // A safely retired previous model must release its unquoted exclusion
        // before the new source initializer attempts admission. Still-live model
        // owners remain authoritative; this never forces their retirement.
        crate::backend::nn::shared::MlxNeuralBackend::reclaim_retired_resources();
        crate::backend::runtime::execution::generic::prepare_layerwise_manager(
            sources,
            self.memory_ledger(),
            self.weights_stream(),
            self.stream(),
        )
    }

    /// Explicit prepared foreground source entry for the disk operation join.
    /// The ordinary/public path does not select it until request slots and native
    /// disk capacity are complete. The existing move-only loading transport can
    /// carry its returned manager without retaining incoming source caches.
    pub(in crate::backend) fn prepare_foreground_layerwise_source(
        &self,
        sources: &eredu_architectures::prepared_sources::PreparedModelSources,
    ) -> Result<Option<crate::backend::runtime::execution::generic::PreparedLayerwiseManager>, Error>
    {
        crate::backend::nn::shared::MlxNeuralBackend::reclaim_retired_resources();
        crate::backend::runtime::execution::generic::prepare_foreground_layerwise_manager(
            sources,
            self.memory_ledger(),
            self.weights_stream(),
            self.stream(),
        )
    }

    pub(in crate::backend) fn materialize_after_communication(
        &self,
        capabilities: SessionCapabilities,
        manifest: Option<&eredu_runtime::CommunicationManifest>,
        rank: Option<crate::backend::MlxRankContext>,
        materialize: impl FnOnce(Option<MlxDistributedSession>) -> Result<MlxModel, Error>,
    ) -> Result<PreparedModel<MlxModel>, Error> {
        // This is an ordinary host entry, after cold selection and before new
        // native allocations. Retire a previous executable here rather than
        // keeping its weights until the replacement session is constructed.
        crate::backend::nn::shared::MlxNeuralBackend::reclaim_retired_resources();
        let memory =
            crate::backend::managed_memory::NativeMemoryOwner::acquire(self.memory_ledger())?;
        crate::backend::submission_recovery::detached_retained(memory.clone(), || {
            let distributed = self.realize_selected_communication(manifest, rank)?;
            materialize(distributed)
                .and_then(|model| model.with_memory_owner(memory))
                .map(|model| PreparedModel::new(model, capabilities))
        })
    }

    /// Ordinary model construction while a cold semantic owner borrows the exact
    /// selected configuration. The additional source/selection copy occurs only
    /// inside the existing load owner's recovery scope. It is ordinary load cost,
    /// not part of original prepared-media semantic residence or managed loading.
    pub fn prepare_model_borrowed(
        &self,
        config: &crate::composition::mlx::loading::MlxModelConfig,
    ) -> Result<PreparedModel<MlxModel>, Error> {
        let layerwise_manager = self.prepare_layerwise_source(&config.sources)?;
        let construction_sources =
            crate::composition::mlx::loading::PreparedNativeConstructionSources::prepare(
                config.sources.selected().text_realization().state(),
                layerwise_manager,
                self.memory_ledger(),
                self.stream(),
            )?;
        let addressable_manager = crate::composition::mlx::loading::prepare_addressable_source(
            &config.sources,
            self.memory_ledger(),
            self.weights_stream(),
            self.stream(),
        )?;
        self.materialize_after_communication(
            config.sources.selected().session_capabilities(),
            config.sources.selected().communication_manifest(),
            config.rank_context,
            |distributed| {
                crate::composition::mlx::loading::materialize_model_plan_with_construction_sources(
                    config.sources.clone(),
                    distributed,
                    self.stream(),
                    self.weights_stream(),
                    Some(construction_sources),
                    addressable_manager,
                )
            },
        )
    }

    /// Waits for all work submitted to this backend's execution queue.
    pub fn synchronize(&self) -> Result<(), Error> {
        self.stream().synchronize().map_err(Into::into)
    }
}

impl<'a> BackendProvider for MlxBackend<'a> {
    type ModelConfig = crate::composition::mlx::loading::MlxModelConfig;
    type Model = MlxModel;
    type Session = MlxModelSession;
    type Error = Error;

    fn into_backend_failure(error: Self::Error) -> eredu_core::BackendFailure {
        error.into_backend_failure()
    }

    fn descriptor(&self) -> BackendDescriptor {
        BackendDescriptor::new("mlx", env!("CARGO_PKG_VERSION"))
    }

    fn devices(&self) -> Result<Vec<(DeviceDescriptor, DeviceCapabilities)>, Self::Error> {
        let device = self.stream().get_device()?;
        let identity = match &self.realized_device {
            Some(identity) => {
                identity.validate_device(&device)?;
                identity.clone()
            }
            None => infer_native_device_identity(&device)?,
        };
        Ok(vec![(
            identity.descriptor(),
            device_capabilities(self.world.is_some()),
        )])
    }

    fn prepare_model(
        &self,
        config: Self::ModelConfig,
    ) -> Result<PreparedModel<Self::Model>, Self::Error> {
        let layerwise_manager = self.prepare_layerwise_source(&config.sources)?;
        let construction_sources =
            crate::composition::mlx::loading::PreparedNativeConstructionSources::prepare(
                config.sources.selected().text_realization().state(),
                layerwise_manager,
                self.memory_ledger(),
                self.stream(),
            )?;
        let addressable_manager = crate::composition::mlx::loading::prepare_addressable_source(
            &config.sources,
            self.memory_ledger(),
            self.weights_stream(),
            self.stream(),
        )?;
        let capabilities = config.sources.selected().session_capabilities();
        let rank = config.rank_context;
        let manifest = config.sources.selected().communication_manifest().cloned();
        self.materialize_after_communication(capabilities, manifest.as_ref(), rank, |distributed| {
            crate::composition::mlx::loading::materialize_model_plan_with_construction_sources(
                config.sources,
                distributed,
                self.stream(),
                self.weights_stream(),
                Some(construction_sources),
                addressable_manager,
            )
        })
    }

    fn create_session(
        &self,
        model: PreparedModel<Self::Model>,
    ) -> Result<Self::Session, Self::Error> {
        self.validate_prepared_target(model.native_target())?;
        let admitted = model.capabilities();
        let session = MlxModelSession::from_model(model.into_inner(), admitted)?;
        #[cfg(all(feature = "metal", not(feature = "cuda")))]
        if self
            .memory_ledger
            .same_ledger(&crate::backend::managed_memory::ledger())
            && self.stream().device_type()? == safemlx::DeviceType::Gpu
        {
            // These shared source families are part of the initialized backend,
            // before any request can reserve execution storage. Their exact paid
            // owners outlive sessions; no lazy unquoted source is born mid-run.
            crate::backend::nn::shared::MlxNeuralBackend::reclaim_retired_resources();
            crate::backend::managed_memory::input_allocator::prepare_kernel_families(
                &self.memory_ledger,
            )
            .map_err(|cause| Error::Other(Box::new(cause)))?;
        }
        Ok(session)
    }

    fn session_capability_mismatch(
        &self,
        admitted: SessionCapabilities,
        realized: SessionCapabilities,
    ) -> Self::Error {
        Error::ArchitectureModel(format!(
            "realized MLX session capabilities {realized:?} do not match pre-materialization admission {admitted:?}"
        ))
    }
}
impl ModelLoadingBackend for MlxBackend<'_> {
    type LoadOptions = MlxLoadRequest;
    type SelectedPreparation = crate::composition::mlx::loading::MlxSelectedPreparation;
    type ConfigurationResolver = eredu_architectures::configuration::ModelConfigurations;

    fn configuration_resolver(&self) -> &Self::ConfigurationResolver {
        &eredu_architectures::configuration::MODEL_CONFIGURATIONS
    }

    fn inspect_model_artifact(
        &self,
        path: &std::path::Path,
    ) -> Result<
        eredu_core::ArtifactInspection<
            eredu_architectures::processor_plan::ArtifactArchitecturePlan,
        >,
        eredu_core::ModelLoadError<Self::Error>,
    > {
        self.memory_ledger()
            .inspect_artifact_for_loading(
                path,
                self.configuration_resolver(),
                eredu_checkpoint::safetensors::SafetensorsDiscoveryLimits::default(),
                eredu_runtime::working_memory::DependencyMemoryPolicy::default(),
            )
            .map_err(|error| eredu_core::ModelLoadError::Backend(Error::ArtifactInspection(error)))
    }

    fn select_preparation(
        &self,
        inspection: &eredu_core::ArtifactInspection<
            eredu_architectures::processor_plan::ArtifactArchitecturePlan,
        >,
        options: &Self::LoadOptions,
    ) -> Result<Self::SelectedPreparation, Self::Error> {
        crate::composition::mlx::loading::select_preparation(inspection, options.clone())
    }

    fn selected_preparation_admission(
        &self,
        selected: &Self::SelectedPreparation,
    ) -> eredu_core::PreparationAdmission {
        selected.neutral().admission()
    }

    fn model_config(
        &self,
        selected: eredu_core::SelectedModelPreparation<Self>,
    ) -> Result<Self::ModelConfig, Self::Error> {
        crate::composition::mlx::loading::MlxModelConfig::new(selected, self.memory_ledger())
    }
}
