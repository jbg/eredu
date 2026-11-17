//! Host operations for finite speech tasks over retained execution.

use eredu_core::BackendFailure;
use eredu_nn::{DistributedNeuralBackend, GroupedNeuralBackend, NeuralBackend, Tensor};
use eredu_runtime::{RealtimeStartupMechanisms, SamplingBackend, SubmissionBackend};

use super::{
    AudioAdvance, AudioExecutionKernel, AudioIngress, AudioIngressError, AudioIngressRequest,
    AudioPcmTranscription, AudioRequest, AudioSession, AudioSessionError, AudioTranscriptionError,
    PreparedResidentAudio, ResidentAudioBinding, ResidentAudioError,
};
use crate::{lfm3::omni::OmniTurnError, processor_execution::MediaGraphMechanisms};

type Context<B> = <<B as NeuralBackend>::Tensor as Tensor>::Context;

/// The selected startup action for a local Audio task.
pub enum LocalAudioTask<'a> {
    /// Start streaming transcription.
    Transcribe,
    /// Start finite PCM ingress and retain the sampling seed for its response.
    Ingress(AudioIngressRequest<'a>, u64),
    /// Prefill and start a response immediately.
    Respond(AudioRequest<'a>, u64),
}

/// Ready host operations for the selected task.
pub enum BoundAudioTask {
    /// Streaming transcription.
    Transcribe(Box<dyn AudioTaskTranscription>),
    /// PCM ingress before a finite response.
    Ingress(Box<dyn AudioTaskIngress>),
    /// A controlled response after prefill.
    Response(Box<dyn AudioTaskResponse>),
}

/// Portable failures preparing or advancing a finite speech task.
#[derive(Debug, thiserror::Error)]
pub enum AudioTaskError {
    /// The retained speech graph or its initial request failed.
    #[error(transparent)]
    Startup(#[from] ResidentAudioError),
    /// A finite multimodal turn failed to start.
    #[error(transparent)]
    Multimodal(#[from] OmniTurnError),
    /// Streaming transcription failed.
    #[error(transparent)]
    Transcription(#[from] AudioTranscriptionError),
    /// Finite PCM ingress failed.
    #[error(transparent)]
    Ingress(#[from] AudioIngressError),
    /// Controlled response advancement failed.
    #[error(transparent)]
    Response(#[from] AudioSessionError),
    /// The task binding encountered a native failure.
    #[error(transparent)]
    Backend(#[from] BackendFailure),
}

/// Host PCM operations for streaming transcription.
pub trait AudioTaskTranscription {
    /// Capacity of the next PCM submission.
    fn available_samples(&self) -> usize;
    /// Submits PCM without advancing the transcription frontier.
    fn submit(&mut self, samples: &[f32], rate: u32) -> Result<(), AudioTaskError>;
    /// Closes the PCM input.
    fn finish_input(&mut self) -> Result<(), AudioTaskError>;
    /// Publishes one transcript block if ready.
    fn advance(&mut self) -> Result<Option<Vec<i32>>, AudioTaskError>;
    /// Reports whether the final block has been published.
    fn is_finished(&self) -> bool;
}

/// Host PCM operations for finite response ingress.
pub trait AudioTaskIngress {
    /// Submits the next PCM chunk.
    fn submit(&mut self, samples: &[f32], rate: u32) -> Result<(), AudioTaskError>;
    /// Consumes the ingress and returns the prefilled response.
    fn finish(self: Box<Self>) -> Result<Box<dyn AudioTaskResponse>, AudioTaskError>;
}

/// Host output of one controlled response advance.
pub trait AudioTaskResponse {
    /// Advances one bounded step and returns any committed output.
    fn advance(&mut self) -> Result<AudioAdvance, AudioTaskError>;
}

impl<K: AudioExecutionKernel> AudioTaskResponse for AudioSession<K>
where
    <K::SamplerBackend as SamplingBackend>::RandomState: Clone,
{
    fn advance(&mut self) -> Result<AudioAdvance, AudioTaskError> {
        AudioSession::advance(self).map_err(Into::into)
    }
}

impl<B, H> AudioTaskTranscription for AudioPcmTranscription<B, H>
where
    B: GroupedNeuralBackend + DistributedNeuralBackend + SubmissionBackend<Executor = Context<B>>,
    H: ResidentAudioBinding<B>,
{
    fn available_samples(&self) -> usize {
        AudioPcmTranscription::available_samples(self)
    }

    fn submit(&mut self, samples: &[f32], rate: u32) -> Result<(), AudioTaskError> {
        AudioPcmTranscription::submit(self, samples, rate).map_err(Into::into)
    }

    fn finish_input(&mut self) -> Result<(), AudioTaskError> {
        AudioPcmTranscription::finish_input(self).map_err(Into::into)
    }

    fn advance(&mut self) -> Result<Option<Vec<i32>>, AudioTaskError> {
        AudioPcmTranscription::advance(self, false)
            .map(|block| block.map(|block| block.tokens))
            .map_err(Into::into)
    }

    fn is_finished(&self) -> bool {
        AudioPcmTranscription::is_finished(self)
    }
}

impl<B, H> AudioTaskIngress for AudioIngress<B, H>
where
    B: GroupedNeuralBackend
        + DistributedNeuralBackend
        + SubmissionBackend<Executor = Context<B>>
        + 'static,
    H: ResidentAudioBinding<B> + 'static,
    <H::Sampling as SamplingBackend>::RandomState: Clone,
{
    fn submit(&mut self, samples: &[f32], rate: u32) -> Result<(), AudioTaskError> {
        AudioIngress::submit(self, samples, rate)
            .map(|_| ())
            .map_err(Into::into)
    }

    fn finish(self: Box<Self>) -> Result<Box<dyn AudioTaskResponse>, AudioTaskError> {
        (*self)
            .finish()
            .map(|session| Box::new(session) as _)
            .map_err(Into::into)
    }
}

impl LocalAudioTask<'_> {
    /// Binds the retained local graph and starts the selected host operation.
    pub fn bind(
        self,
        selected: PreparedResidentAudio,
        mechanisms: &impl MediaGraphMechanisms,
    ) -> Result<BoundAudioTask, AudioTaskError> {
        let graph = mechanisms.graph();
        let model = selected.bind_local(graph.clone())?;
        Ok(match self {
            Self::Transcribe => {
                BoundAudioTask::Transcribe(Box::new(model.prepare_pcm_transcription()?))
            }
            Self::Ingress(request, seed) => BoundAudioTask::Ingress(Box::new(
                model.prepare_ingress(request, graph.realize_random(seed)?)?,
            )),
            Self::Respond(request, seed) => BoundAudioTask::Response(Box::new(
                model.start(request, graph.realize_random(seed)?)?,
            )),
        })
    }
}
