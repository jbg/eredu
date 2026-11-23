//! MLX resource binding and completion for portable realtime execution.

use crate::backend::runtime::cache::MlxCacheBackend;
use crate::{
    backend::{nn::shared::MlxNeuralBackend, runtime::generation::MlxSamplingBackend},
    native::{
        ExecutionContext, MlxComponentBinder, MlxRealtimeBinding, MlxRealtimeExecutionContext,
        MlxRealtimeQueue,
    },
    MlxBackendFactory, MlxTensor,
};
use eredu_core::realtime_mechanisms::{CategoricalRealtimeNative, SpeechRealtimeNative};
use eredu_core::{
    realtime_mechanisms::{
        RealtimeBindingMechanisms, RealtimeBindingTarget, RealtimeContextFactory,
        RealtimeRandomRoot,
    },
    BackendFailure, DevicePlan,
};

use super::MlxRealtimeSupport;
use safemlx::Stream;

impl CategoricalRealtimeNative for MlxRealtimeExecutionContext {
    type Neural = MlxNeuralBackend;
    type FrameQueue = MlxRealtimeQueue;
    type BindingError =
        <MlxRealtimeBinding<'static> as RealtimeBindingMechanisms<Self::Neural>>::Error;
    type Group = <MlxRealtimeBinding<'static> as RealtimeBindingMechanisms<Self::Neural>>::Group;
    type Route = <MlxRealtimeBinding<'static> as RealtimeBindingMechanisms<Self::Neural>>::Route;
    type TensorMetadata =
        <MlxRealtimeBinding<'static> as RealtimeBindingMechanisms<Self::Neural>>::TensorMetadata;
    type Materializer = MlxComponentBinder;
    type Rank = crate::backend::MlxRankContext;

    type FrameSampling = MlxSamplingBackend;
    type FrameError = crate::backend::error::Error;
    type FrameTensors<'a> = super::frame_execution::MlxRealtimeFrameTensorMechanisms<'a>;

    fn frame_tensors(queue: &Self::FrameQueue) -> Self::FrameTensors<'_> {
        super::frame_execution::MlxRealtimeFrameTensorMechanisms::new(&queue.stream)
    }
}

impl RealtimeBindingTarget<crate::backend::MlxRankContext, MlxRealtimeQueue>
    for MlxRealtimeExecutionContext
{
    type Binding<'a> = MlxRealtimeBinding<'a>;

    fn native_rank_identity(rank: &crate::backend::MlxRankContext) -> (usize, usize) {
        (rank.world_size(), rank.global_rank())
    }

    fn world_identity(&self) -> Option<(usize, usize)> {
        self.world_identity()
    }

    fn allocator_cache_defaults(
        &self,
    ) -> Option<&dyn eredu_core::capability::NativeAllocatorCacheDefaults> {
        Some(&crate::MlxAllocatorCacheDefaults)
    }

    fn open_binding(
        &self,
        native_rank: Option<&crate::backend::MlxRankContext>,
    ) -> Result<Self::Binding<'_>, BackendFailure> {
        self.open_materialization(native_rank)
            .map_err(BackendFailure::from_error)
    }

    fn finish_binding(&self) -> Result<(), BackendFailure> {
        crate::trim_allocator_cache().map_err(BackendFailure::from_error)
    }

    fn validate_execution(&self, queue: &MlxRealtimeQueue) -> Result<(), BackendFailure> {
        self.validate_execution(queue)
    }
}

impl RealtimeRandomRoot<MlxTensor> for MlxRealtimeExecutionContext {
    fn random_root(random: &crate::backend::random::RandomState) -> Option<MlxTensor> {
        Some(MlxTensor::from_array(random.as_array().clone()))
    }
}

impl SpeechRealtimeNative for MlxRealtimeExecutionContext {
    type Neural = MlxNeuralBackend;
    type CacheBackend = MlxCacheBackend;

    fn state_copy_queue(&self) -> Stream {
        self.stream().clone()
    }

    type Components = MlxComponentBinder;
    type Queue = MlxRealtimeQueue;

    fn components(&self) -> Self::Components {
        self.component_binder()
    }

    fn speech_queue(&self) -> Self::Queue {
        self.queue()
    }
}

impl RealtimeContextFactory for MlxBackendFactory {
    type Context = ExecutionContext;
    type MediaGraphSupport = MlxRealtimeSupport;
    type Support = MlxRealtimeSupport;
    type Mechanisms = MlxRealtimeExecutionContext;

    fn realtime_support(&self) -> Self::Support {
        MlxRealtimeSupport::new(false)
    }

    fn media_graph_support(&self) -> Self::MediaGraphSupport {
        MlxRealtimeSupport::new(crate::native::MlxRingTransport::is_available())
    }

    fn open_realtime_execution(
        &self,
        device: &DevicePlan,
    ) -> Result<ExecutionContext, BackendFailure> {
        self.execution_context(device)
    }

    fn open_realtime_host(&self) -> Result<ExecutionContext, BackendFailure> {
        ExecutionContext::cpu().map_err(BackendFailure::from_error)
    }
}

#[cfg(feature = "audio")]
impl eredu_core::realtime_mechanisms::MediaGraphNative for MlxRealtimeExecutionContext {
    type Neural = MlxNeuralBackend;
    type World = crate::native::MlxRingTransport;
    type Graph = MlxComponentBinder;

    fn graph(&self) -> Self::Graph {
        self.component_binder()
    }

    fn media_lowering(&self) -> impl eredu_core::input::MediaInputLowering<MlxTensor> {
        crate::native::MlxMediaInputLowering
    }

    fn pipeline_world(&self) -> Self::World {
        crate::native::MlxRingTransport
    }
}
