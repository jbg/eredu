use super::*;
use crate::backend::runtime::realtime::MlxRealtimeChannelObserver;

/// MLX host observer used by the neutral prepublication transition.
#[derive(Clone)]
pub struct MlxRealtimeHostObserver {
    channels: MlxRealtimeChannelObserver,
    projection: eredu_architectures::moshi::MoshiRealtimeOutputProjection,
}

impl MlxRealtimeHostObserver {
    /// Creates an observer on the selected execution stream.
    pub fn new(stream: &Stream, contract: &eredu_runtime::RealtimeIngressContract) -> Self {
        Self {
            channels: MlxRealtimeChannelObserver::new(stream),
            projection: eredu_architectures::moshi::MoshiRealtimeOutputProjection::new(contract),
        }
    }
}

impl RealtimeFrameHostObserver<CompletedRealtimeFrame<MlxTensor, MlxTensor>>
    for MlxRealtimeHostObserver
{
    type Output = RealtimeOutputFrame;
    type Error = Error;

    fn observe(
        &mut self,
        frame: &CompletedRealtimeFrame<MlxTensor, MlxTensor>,
    ) -> Result<Self::Output, Self::Error> {
        let host_frame = self.channels.observe(frame)?;
        self.projection
            .project(&host_frame)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))
    }
}

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
pub type MlxPrepublicationFrame = PrepublicationRealtimeFrame<
    CompletedRealtimeFrame<MlxTensor, MlxTensor>,
    MlxRealtimeCompletion,
    MlxRealtimeHostObserver,
>;
