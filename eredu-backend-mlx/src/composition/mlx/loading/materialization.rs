use super::super::replicated_text as binding;
use super::*;
use crate::backend::distributed::MlxDistributedSession;
use crate::backend::runtime::cache::MlxCacheBackend;
impl<Dispatch> eredu_core::checkpoint_mechanisms::ModelBindingResources
    for crate::backend::MlxBackend<'_, Dispatch>
{
    type Placement = MlxModelPlacement;
    type Communication = MlxDistributedSession;
    type TextValues = binding::MlxTextValues;
    type CommunicationSetup<'a>
        = crate::backend::runtime::distributed::topology::NativeWorldManifestTransport<'a>
    where
        Self: 'a;
    type Target = crate::backend::MlxPreparedTarget;
    type BindingError = Error;

    fn has_native_rank(placement: &MlxModelPlacement) -> bool {
        placement.rank_context.is_some()
    }

    fn open_communication_setup(
        &self,
        placement: &MlxModelPlacement,
    ) -> Result<Self::CommunicationSetup<'_>, Error> {
        if !cfg!(feature = "distributed") {
            return Err(Error::DistributedModelExecutionDisabled);
        }
        let rank = placement.rank_context.ok_or_else(|| {
            Error::Parallel("native world initialization requires a retained rank".into())
        })?;
        self.open_model_communication(rank)
    }

    fn reclaim_retired_bindings(&self) {
        MlxNeuralBackend::reclaim_retired_resources();
    }

    fn realize_model_communication<'a>(
        &'a self,
        communication: eredu_core::communication::WorldCommunicationBinding<
            Self::CommunicationSetup<'a>,
        >,
    ) -> Result<Self::Communication, Error> {
        MlxDistributedSession::from_prepared(communication)
    }

    fn bound_communication_manifest(
        communication: &Self::Communication,
    ) -> &eredu_core::communication::CommunicationManifest {
        communication.communication_manifest()
    }

    fn allocator_cache_defaults(
        &self,
    ) -> Option<&dyn eredu_core::capability::NativeAllocatorCacheDefaults> {
        Some(&crate::MlxAllocatorCacheDefaults)
    }

    fn open_binding_target(
        &self,
        _: &Self::Placement,
        communication: Option<&MlxDistributedSession>,
    ) -> Result<Self::Target, Error> {
        crate::backend::MlxPreparedTarget::new(self.stream(), communication)
    }

    fn floating_state_element(
        &self,
        dtype: &eredu_core::checkpoint::TensorDtype,
    ) -> Option<eredu_nn::TensorElementType> {
        eredu_core::preparation::StateStorageDtype::from_floating_source(dtype, |_| true)
            .map(eredu_core::preparation::StateStorageDtype::element_type)
    }

    fn observation_mechanisms(&self) -> eredu_core::ObservationMechanisms {
        eredu_core::ObservationMechanisms {
            activation_tensors: true,
            routing_tensors: true,
            routed_unit_tensors: true,
            floating_to_f32: true,
        }
    }

    fn capture_capabilities(&self) -> eredu_core::capture::CaptureCapabilities {
        super::super::session::capture_capabilities()
    }

    fn finish_binding(&self) -> Result<(), Error> {
        crate::trim_allocator_cache().map_err(Into::into)
    }
}

impl<Dispatch> eredu_core::checkpoint_mechanisms::TextBindingResources
    for crate::backend::MlxBackend<'_, Dispatch>
{
    type Neural = MlxNeuralBackend;
    type CacheStorage = MlxCacheBackend;
    type PredictionMaterializer = binding::MlxEmbeddedPredictionMaterializer;
    #[cfg(feature = "distributed")]
    type RankPlacement = eredu_core::checkpoint_mechanisms::RankedTextPlacement;
    #[cfg(not(feature = "distributed"))]
    type RankPlacement = eredu_core::checkpoint_mechanisms::LocalTextPlacement;
}

impl<'a, Dispatch> eredu_core::checkpoint_mechanisms::TextBindingContext<'a>
    for crate::backend::MlxBackend<'_, Dispatch>
{
    type TextResources = binding::MlxLocalTextBinding<'a>;

    fn text_execution(&'a self) -> &'a Stream {
        self.stream()
    }

    fn text_weights(&'a self) -> &'a Stream {
        self.weights_stream()
    }

    fn text_resources(&'a self) -> Self::TextResources {
        binding::MlxLocalTextBinding {
            stream: self.stream(),
            weights_stream: self.weights_stream(),
        }
    }
}
