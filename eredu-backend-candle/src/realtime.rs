//! Native local mechanisms for backend-neutral realtime binding.

#[cfg(feature = "audio")]
mod media;
mod submission;

use std::{convert::Infallible, num::NonZeroUsize};

use eredu_core::{
    backend::local_text::NoTextTransport,
    cache::{mechanisms::LiveCacheBackend, StateComponentPolicy, StateComponentRole},
    checkpoint::TensorDtype,
    communication::{
        CommunicationBackend, CommunicationCompletionCapabilities, CommunicationManifest,
        CommunicationTensorMetadata, RankCommunication, RankCommunicationResources,
        WorldCommunicationBinding,
    },
    eager_completion::EagerCompletion,
    preparation::{
        ReplicatedTextMechanismSupport, StateComponentPlacement, StateLifecycleCapabilities,
        WeightResidencyMechanism,
    },
    realtime_mechanisms::{
        CategoricalRealtimeNative, RealtimeBindingMechanisms, RealtimeBindingTarget,
        RealtimeContextFactory, RealtimeContexts, RealtimeFrameTensorMechanisms,
        RealtimeHostTokenMaterializer, RealtimeLoadSupport, RealtimeMechanism,
        RealtimeMechanismFacts, RealtimeMechanismSupport, RealtimeRandomRoot,
        RealtimeStartupMechanisms, SpeechRealtimeNative,
    },
    sampling::{host_random::HostRandomState, host_tensor::HostTensorSampling},
    BackendFailure, BackendFailureKind, CompletionCancellationMode, DevicePlan,
    SessionCapabilities, Submission, SubmissionBackend,
};
use eredu_nn::Tensor;

use crate::{
    CandleBackendFactory, CandleCacheStorage, CandleContext, CandleNeuralBackend,
    CandlePreparation, CandleTensor,
};

use submission::CandleRealtimeCompletion;

pub(crate) fn unsupported(detail: &'static str) -> BackendFailure {
    BackendFailure::new(
        BackendFailureKind::Unsupported,
        std::io::Error::other(detail),
    )
}

#[doc(hidden)]
pub struct CandleRealtimeSupport {
    preparation: CandlePreparation,
    enabled: bool,
}

impl RealtimeLoadSupport for CandleRealtimeSupport {
    fn independent_state_support(&self) -> &impl ReplicatedTextMechanismSupport {
        &self.preparation
    }
}

impl RealtimeMechanismSupport for CandleRealtimeSupport {
    fn facts(&self) -> RealtimeMechanismFacts {
        use RealtimeMechanism::*;
        let mechanisms = [
            TensorOperations,
            NeuralOperations,
            ParameterMaterialization,
            ParameterStorage,
            StateStorage,
            CoordinateStorage,
            Sampling,
            Randomness,
            HostConversion,
            ExactCompletion,
            ResourceRetention,
            Transfer,
        ];
        let residencies = [
            WeightResidencyMechanism::Resident,
            WeightResidencyMechanism::Windowed,
            WeightResidencyMechanism::DiskStreamed,
        ];
        let completion = [CompletionCancellationMode::QuarantineUntilComplete];
        let operators = if self.enabled {
            self.preparation
                .operators()
                .union(CandleNeuralBackend::REALTIME_OPERATORS)
        } else {
            eredu_nn::NeuralOperatorCapabilities::NONE
        };
        RealtimeMechanismFacts::new(
            operators,
            mechanisms.into_iter().filter(|_| self.enabled),
            residencies.into_iter().filter(|_| self.enabled),
            NonZeroUsize::MIN,
            CommunicationCompletionCapabilities::new(
                completion.into_iter().filter(|_| self.enabled),
            )
            .expect("unique realtime completion modes"),
            SessionCapabilities::new(self.enabled, self.enabled, false),
        )
        .with_state_lifecycle(
            StateLifecycleCapabilities::new()
                .with_transactions(self.enabled, self.enabled)
                .with_reset(self.enabled)
                .with_observation_retention(self.enabled),
        )
    }

    fn supports_lowering(
        &self,
        descriptor: &eredu_checkpoint::lowering::WeightLoweringDescriptor,
        transforms_payload: bool,
    ) -> bool {
        self.enabled
            && if transforms_payload {
                self.preparation.supports_transform(descriptor)
            } else {
                self.preparation.supports_direct(descriptor)
            }
    }

    fn state_component_placements(
        &self,
        component: &StateComponentPolicy,
    ) -> (
        Option<StateComponentPlacement>,
        Option<StateComponentPlacement>,
    ) {
        let supported = self.enabled
            && matches!(
                component.role(),
                StateComponentRole::AttentionKeys
                    | StateComponentRole::AttentionValues
                    | StateComponentRole::CompressedLatent
                    | StateComponentRole::RotaryKeys
                    | StateComponentRole::Fixed(_)
            );
        (supported.then_some(StateComponentPlacement::Device), None)
    }
}

impl RealtimeContextFactory for CandleBackendFactory {
    type Context = CandleContext;
    type Support = CandleRealtimeSupport;
    type MediaGraphSupport = CandleRealtimeSupport;
    type Mechanisms = CandleContext;

    fn realtime_support(&self) -> Self::Support {
        CandleRealtimeSupport {
            preparation: CandlePreparation::new(self.placement),
            enabled: true,
        }
    }

    fn media_graph_support(&self) -> Self::MediaGraphSupport {
        CandleRealtimeSupport {
            enabled: cfg!(feature = "audio"),
            ..self.realtime_support()
        }
    }

    fn open_realtime_execution(
        &self,
        device: &DevicePlan,
    ) -> Result<CandleContext, BackendFailure> {
        use eredu_core::automatic::ColdPlanPlacement as _;
        self.plan_placement(device, None)
            .map_err(BackendFailure::from_error)?;
        self.open()
    }

    fn open_realtime_host(&self) -> Result<CandleContext, BackendFailure> {
        Ok(CandleContext::cpu())
    }
}

impl From<RealtimeContexts<CandleContext>> for CandleContext {
    fn from(contexts: RealtimeContexts<CandleContext>) -> Self {
        // Resident parameters and frame operations share the execution context.
        contexts.execution
    }
}

impl RealtimeStartupMechanisms for CandleContext {
    type Random = HostRandomState;
    type Completion = CandleRealtimeCompletion;

    fn realize_random(&self, seed: u64) -> Result<HostRandomState, BackendFailure> {
        Ok(HostRandomState::with_seed(seed))
    }
}

impl RealtimeRandomRoot<CandleTensor> for CandleContext {
    fn random_root(_: &HostRandomState) -> Option<CandleTensor> {
        None
    }
}

impl CommunicationBackend for CandleNeuralBackend {
    type CommunicationGroup = Infallible;
    type CommunicationRoute = Infallible;
    type CommunicationCompletion = EagerCompletion<eredu_nn::Error>;
    type CommunicationError = eredu_nn::Error;

    fn submit_local_dependencies<'a, I>(
        values: I,
        executor: &CandleContext,
    ) -> Result<Submission<(), Self::CommunicationCompletion>, Self::CommunicationError>
    where
        I: IntoIterator<Item = &'a CandleTensor>,
    {
        Ok(Submission {
            output: (),
            completion: Self::submit(executor, values)?,
        })
    }
}

#[doc(hidden)]
pub struct CandleRealtimeMetadata;

impl CommunicationTensorMetadata<CandleNeuralBackend> for CandleRealtimeMetadata {
    fn dtype(&self, value: &CandleTensor) -> TensorDtype {
        if value.boolean {
            TensorDtype::Bool
        } else {
            eredu_core::artifact::stored_to_tensor_dtype(&CandleCacheStorage::dtype(value))
        }
    }
    fn shape(&self, value: &CandleTensor) -> Vec<usize> {
        value.value.dims().to_vec()
    }
}

#[doc(hidden)]
pub struct NoCandleRank(Infallible);

impl RankCommunication for NoCandleRank {
    type Backend = CandleNeuralBackend;
    type Group = Infallible;
    type Route = Infallible;
    type Metadata = CandleRealtimeMetadata;
    type Parallel = Infallible;
    type Error = BackendFailure;

    fn bound_group(&self, _: eredu_core::CollectiveGroupId) -> Option<&Infallible> {
        match self.0 {}
    }
    fn tensor_context(&self, _: eredu_core::CollectiveGroupId) -> Option<Infallible> {
        match self.0 {}
    }
    fn coordinated_tensor_context(&self, _: eredu_core::CollectiveGroupId) -> Option<Infallible> {
        match self.0 {}
    }
    fn into_rank_resources(
        self,
        _: &CommunicationManifest,
    ) -> Result<
        RankCommunicationResources<Self::Backend, Infallible, Infallible, Self::Metadata>,
        BackendFailure,
    > {
        match self.0 {}
    }
}

#[doc(hidden)]
pub struct CandleRealtimeBinding<'a>(&'a CandleContext);

impl RealtimeBindingMechanisms<CandleNeuralBackend> for CandleRealtimeBinding<'_> {
    type Queue = CandleContext;
    type Error = BackendFailure;
    type Group = Infallible;
    type Route = Infallible;
    type TensorMetadata = CandleRealtimeMetadata;
    type Sampling = HostTensorSampling<CandleTensor>;
    type Materializer = CandleContext;
    type CommunicationSetup = NoTextTransport;
    type Rank = NoCandleRank;

    fn execution_queue(&self) -> CandleContext {
        self.0.clone()
    }
    fn context(&self) -> &CandleContext {
        self.0
    }
    fn open_communication_setup(&self) -> Result<NoTextTransport, BackendFailure> {
        Err(unsupported("Candle realtime execution is local"))
    }
    fn realize_rank(
        &self,
        _: WorldCommunicationBinding<NoTextTransport>,
    ) -> Result<NoCandleRank, BackendFailure> {
        Err(unsupported("Candle realtime execution is local"))
    }
    fn materializer(&self) -> CandleContext {
        self.0.clone()
    }
}

impl RealtimeBindingTarget<Infallible, CandleContext> for CandleContext {
    type Binding<'a> = CandleRealtimeBinding<'a>;

    fn native_rank_identity(rank: &Infallible) -> (usize, usize) {
        match *rank {}
    }
    fn world_identity(&self) -> Option<(usize, usize)> {
        None
    }

    fn open_binding(&self, _: Option<&Infallible>) -> Result<Self::Binding<'_>, BackendFailure> {
        self.check_healthy()?;
        Ok(CandleRealtimeBinding(self))
    }
    fn finish_binding(&self) -> Result<(), BackendFailure> {
        self.synchronize()
    }

    fn validate_execution(&self, queue: &CandleContext) -> Result<(), BackendFailure> {
        self.check_healthy()?;
        if !self.same_context(queue) {
            return Err(BackendFailure::new(
                BackendFailureKind::InvalidSession,
                std::io::Error::other("realtime session belongs to a different Candle context"),
            ));
        }
        Ok(())
    }
}

#[doc(hidden)]
pub struct CandleFrameTensors<'a>(&'a CandleContext);

impl RealtimeHostTokenMaterializer for CandleFrameTensors<'_> {
    type Tensor = CandleTensor;
    type Error = BackendFailure;
    fn materialize_i32(
        &mut self,
        values: &[i32],
        shape: [usize; 2],
    ) -> Result<CandleTensor, BackendFailure> {
        let [batch, columns] = shape;
        let shape = [
            i32::try_from(batch).map_err(BackendFailure::from_error)?,
            i32::try_from(columns).map_err(BackendFailure::from_error)?,
        ];
        CandleTensor::from_i32_slice(values, &shape, self.0).map_err(BackendFailure::from_error)
    }
}
impl RealtimeFrameTensorMechanisms for CandleFrameTensors<'_> {
    type Tensor = CandleTensor;
    type Error = BackendFailure;

    fn filled_column(&mut self, token: i32, batch: usize) -> Result<CandleTensor, BackendFailure> {
        let batch = i32::try_from(batch).map_err(BackendFailure::from_error)?;
        CandleTensor::full_i32(token, &[batch, 1], self.0).map_err(BackendFailure::from_error)
    }
}

impl CategoricalRealtimeNative for CandleContext {
    type Neural = CandleNeuralBackend;
    type FrameQueue = CandleContext;
    type BindingError = BackendFailure;
    type Group = Infallible;
    type Route = Infallible;
    type TensorMetadata = CandleRealtimeMetadata;
    type Materializer = CandleContext;
    type Rank = Infallible;
    type FrameSampling = HostTensorSampling<CandleTensor>;
    type FrameError = BackendFailure;
    type FrameTensors<'a> = CandleFrameTensors<'a>;

    fn frame_tensors(queue: &CandleContext) -> Self::FrameTensors<'_> {
        CandleFrameTensors(queue)
    }
}

impl SpeechRealtimeNative for CandleContext {
    type Neural = CandleNeuralBackend;
    type CacheBackend = CandleCacheStorage;
    type Components = CandleContext;
    type Queue = CandleContext;

    fn state_copy_queue(&self) -> CandleContext {
        self.clone()
    }
    fn components(&self) -> CandleContext {
        self.clone()
    }
    fn speech_queue(&self) -> CandleContext {
        self.clone()
    }
}
