use super::*;

/// MLX backend selected for a complete model/session.
pub struct MlxBackend<'a, Dispatch = std::convert::Infallible> {
    stream: Stream,
    weights_stream: Stream,
    realized_device: Option<MlxDeviceIdentity>,
    world: Option<std::borrow::Cow<'a, safemlx::distributed::Group>>,
    assistant_dispatch: std::marker::PhantomData<fn() -> Dispatch>,
}

impl MlxBackend<'static> {
    pub(crate) fn new(stream: &Stream, weights_stream: &Stream) -> Self {
        Self::from_streams(stream, weights_stream)
    }
}

impl<Dispatch> MlxBackend<'static, Dispatch> {
    pub(crate) fn from_streams(stream: &Stream, weights_stream: &Stream) -> Self {
        Self {
            stream: stream.clone(),
            weights_stream: weights_stream.clone(),
            realized_device: None,
            world: None,
            assistant_dispatch: std::marker::PhantomData,
        }
    }
    pub(crate) fn for_execution_plan(
        stream: &Stream,
        weights_stream: &Stream,
        realized_device: MlxDeviceIdentity,
        world: Option<safemlx::distributed::Group>,
    ) -> Self {
        Self {
            stream: stream.clone(),
            weights_stream: weights_stream.clone(),
            realized_device: Some(realized_device),
            world: world.map(std::borrow::Cow::Owned),
            assistant_dispatch: std::marker::PhantomData,
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
            world: Some(std::borrow::Cow::Borrowed(world)),
            assistant_dispatch: std::marker::PhantomData,
        }
    }
}

impl<'a, Dispatch> MlxBackend<'a, Dispatch> {
    pub(crate) const fn stream(&self) -> &Stream {
        &self.stream
    }

    pub(crate) const fn weights_stream(&self) -> &Stream {
        &self.weights_stream
    }

    pub(crate) fn validate_prepared_target(&self, target: &MlxPreparedTarget) -> Result<(), Error> {
        target.validate(&self.stream, self.world.as_deref())
    }

    pub(crate) fn open_model_communication(
        &self,
        rank: crate::backend::MlxRankContext,
    ) -> Result<
        crate::backend::runtime::distributed::topology::NativeWorldManifestTransport<'_>,
        Error,
    > {
        use crate::backend::runtime::distributed::topology::NativeWorldManifestTransport;
        let world = self.world.as_deref().ok_or_else(|| {
            Error::Parallel("distributed model preparation requires a bound native world".into())
        })?;
        rank.validate_execution_stream(&self.stream)?;
        Ok(NativeWorldManifestTransport::new(world, &self.stream))
    }

    /// Waits for all work submitted to this backend's execution queue.
    pub fn synchronize(&self) -> Result<(), Error> {
        self.stream.synchronize().map_err(Into::into)
    }
}

impl<'a, Dispatch> BackendProvider for MlxBackend<'a, Dispatch> {
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

impl<Dispatch> ModelLoadingBackend for MlxBackend<'_, Dispatch> {
    type LoadPlacement = crate::backend::DeviceAssignment;
    type Binding = crate::composition::mlx::loading::MlxModelPlacement;
    type LoadMechanisms = crate::composition::mlx::loading::MlxPreparationMechanisms;

    fn load_mechanisms(&self) -> crate::composition::mlx::loading::MlxPreparationMechanisms {
        crate::composition::mlx::loading::MlxPreparationMechanisms
    }

    fn bind_load_placement(
        &self,
        rank: Option<eredu_core::ParallelRankTopology>,
        device: Option<Self::LoadPlacement>,
    ) -> Result<Self::Binding, Self::Error> {
        let rank = crate::composition::mlx::loading::direct_rank(rank, device)?;
        Ok(Self::Binding::direct(rank))
    }

    fn validate_model_binding(&self, binding: &Self::Binding) -> Result<(), Self::Error> {
        if !cfg!(feature = "distributed") && binding.rank_context.is_some() {
            return Err(Error::DistributedModelExecutionDisabled);
        }
        Ok(())
    }
}
