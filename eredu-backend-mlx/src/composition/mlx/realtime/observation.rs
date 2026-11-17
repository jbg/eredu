use super::*;
pub(super) use crate::backend::runtime::realtime::MlxRealtimeChannelObserver;

pub(super) type MlxHostChannels = <MlxRealtimeChannelObserver as RealtimeFrameHostObserver<
    CompletedRealtimeFrame<MlxTensor, MlxTensor>,
>>::Output;

/// Neutral scheduler branch specialized to MLX model mechanisms.
pub type MlxFrameSessionBranch = RealtimeSessionBranch<
    RealtimePayloadBranch<MlxKeyValueTransactionBranch, MlxTensor>,
    eredu_core::RealtimeFrameScheduleState,
    eredu_core::RealtimeSampling,
    GenerationSampler,
    RandomState,
    MlxRealtimeCompletion,
>;

/// MLX submission whose host observation must succeed before publication.
pub type MlxPrepublicationFrame<P> = PrepublicationRealtimeFrame<
    CompletedRealtimeFrame<MlxTensor, MlxTensor>,
    MlxRealtimeCompletion,
    eredu_runtime::MappedRealtimeFrameHostObserver<MlxRealtimeChannelObserver, P>,
>;
