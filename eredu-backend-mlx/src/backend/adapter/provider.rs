use super::*;

/// MLX backend selected for a complete model/session.
pub struct MlxBackend<'a> {
    stream: Stream,
    weights_stream: Stream,
    realized_device: Option<MlxDeviceIdentity>,
    world: Option<&'a safemlx::distributed::Group>,
}

impl MlxBackend<'static> {
    pub(crate) fn new(stream: &Stream, weights_stream: &Stream) -> Self {
        Self {
            stream: stream.clone(),
            weights_stream: weights_stream.clone(),
            realized_device: None,
            world: None,
        }
    }

    pub(crate) fn for_execution_plan(
        stream: &Stream,
        weights_stream: &Stream,
        realized_device: MlxDeviceIdentity,
    ) -> Self {
        Self {
            stream: stream.clone(),
            weights_stream: weights_stream.clone(),
            realized_device: Some(realized_device),
            world: None,
        }
    }
}

impl<'a> MlxBackend<'a> {
    pub(crate) fn with_distributed_world(
        stream: &Stream,
        weights_stream: &Stream,
        world: &'a safemlx::distributed::Group,
    ) -> Self {
        Self {
            stream: stream.clone(),
            weights_stream: weights_stream.clone(),
            realized_device: None,
            world: Some(world),
        }
    }
    pub(crate) const fn stream(&self) -> &Stream {
        &self.stream
    }

    pub(crate) const fn weights_stream(&self) -> &Stream {
        &self.weights_stream
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
        rank.validate_execution_stream(&self.stream)?;
        #[cfg(test)]
        crate::tests::support::path_instrumentation::manifest_communication_realization_attempt();
        MlxDistributedSession::from_manifest(manifest, world, &self.stream).map(Some)
    }

    pub(in crate::backend) fn materialize_after_communication(
        &self,
        capabilities: SessionCapabilities,
        manifest: Option<&eredu_runtime::CommunicationManifest>,
        rank: Option<crate::backend::MlxRankContext>,
        materialize: impl FnOnce(Option<MlxDistributedSession>) -> Result<MlxModel, Error>,
    ) -> Result<PreparedModel<MlxModel>, Error> {
        let distributed = self.realize_selected_communication(manifest, rank)?;
        materialize(distributed).map(|model| PreparedModel::new(model, capabilities))
    }

    /// Waits for all work submitted to this backend's execution queue.
    pub fn synchronize(&self) -> Result<(), Error> {
        self.stream.synchronize().map_err(Into::into)
    }
}

impl<'a> BackendProvider for MlxBackend<'a> {
    type ModelConfig = crate::composition::mlx::loading::MlxModelConfig;
    type Model = MlxModel;
    type Session = MlxModelSession;
    type Error = Error;

    fn descriptor(&self) -> BackendDescriptor {
        BackendDescriptor::new("mlx", env!("CARGO_PKG_VERSION"))
    }

    fn devices(&self) -> Result<Vec<(DeviceDescriptor, DeviceCapabilities)>, Self::Error> {
        let device = self.stream.get_device()?;
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
        let capabilities = config.sources.selected().session_capabilities();
        let rank = config.rank_context;
        let manifest = config.sources.selected().communication_manifest().cloned();
        self.materialize_after_communication(capabilities, manifest.as_ref(), rank, |distributed| {
            crate::composition::mlx::loading::materialize_model_plan(
                config.sources,
                distributed,
                &self.stream,
                &self.weights_stream,
            )
        })
    }

    fn create_session(
        &self,
        model: PreparedModel<Self::Model>,
    ) -> Result<Self::Session, Self::Error> {
        let admitted = model.capabilities();
        MlxModelSession::from_model(model.into_inner(), admitted)
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
        crate::composition::mlx::loading::MlxModelConfig::new(selected)
    }
}
