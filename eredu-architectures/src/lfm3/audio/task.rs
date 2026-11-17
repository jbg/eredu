//! Host operations for finite speech tasks over retained execution.

use eredu_core::BackendFailure;
use eredu_nn::{
    DistributedNeuralBackend, GroupedNeuralBackend, NeuralBackend, RetainedTensorRoots, Tensor,
    TensorParallelGroupedNeuralBackend,
};
use eredu_runtime::{
    ComponentRankMechanisms, LocalComponentMechanisms, PipelineTransportProvider,
    RealtimeGraphMechanisms, RealtimeStartupMechanisms, SamplingBackend, SubmissionBackend,
};

use super::{
    AudioAdvance, AudioExecutionKernel, AudioIngress, AudioIngressError, AudioIngressRequest,
    AudioPcmTranscription, AudioRequest, AudioSession, AudioSessionError, AudioTaskBinder,
    AudioTranscriptionError, PreparedResidentAudio, ResidentAudioBinding, ResidentAudioError,
};
use crate::{
    lfm3::omni::{OmniTurnError, PreparedOmniTurn},
    processor_execution::MediaInputLowering,
};

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

/// Architecture-owned prepared execution and its host input.
pub enum AudioTaskExecution<'a> {
    /// A selected local Audio graph and its startup action.
    Local {
        /// Exact retained model selection.
        selected: Box<PreparedResidentAudio>,
        /// Startup action after binding the graph.
        task: LocalAudioTask<'a>,
    },
    /// A selected multimodal turn.
    Omni {
        /// Exact selected vision and speech inputs.
        turn: Box<PreparedOmniTurn>,
        /// Encodes processor-generated text, such as timestamps.
        encode_text: &'a mut dyn FnMut(&str) -> Result<Vec<u32>, std::io::Error>,
    },
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

/// Native binding entry point for a prepared finite speech task.
pub trait AudioTaskBinding {
    /// Realizes the selected task and returns only host-facing operations.
    fn bind(&self, task: AudioTaskExecution<'_>) -> Result<BoundAudioTask, AudioTaskError>;
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

impl<B, G, M, P> AudioTaskBinding for AudioTaskBinder<B, G, M, P>
where
    B: TensorParallelGroupedNeuralBackend
        + DistributedNeuralBackend
        + SubmissionBackend<Executor = Context<B>>
        + 'static,
    G: RealtimeGraphMechanisms<B> + RealtimeStartupMechanisms<Random: Clone> + Clone + 'static,
    G::State: RetainedTensorRoots<B::Tensor>,
    G::Components:
        LocalComponentMechanisms<B, G::State> + ComponentRankMechanisms<B, G::State, P::Transport>,
    G::Sampling: SamplingBackend<RandomState = G::Random>,
    M: MediaInputLowering<B::Tensor>,
    P: PipelineTransportProvider<Transport: 'static>,
{
    fn bind(&self, task: AudioTaskExecution<'_>) -> Result<BoundAudioTask, AudioTaskError> {
        Ok(match task {
            AudioTaskExecution::Local { selected, task } => {
                let model = selected.bind_local(self.graph.clone())?;
                match task {
                    LocalAudioTask::Transcribe => {
                        BoundAudioTask::Transcribe(Box::new(model.prepare_pcm_transcription()?))
                    }
                    LocalAudioTask::Ingress(request, seed) => BoundAudioTask::Ingress(Box::new(
                        model.prepare_ingress(request, self.graph.realize_random(seed)?)?,
                    )),
                    LocalAudioTask::Respond(request, seed) => BoundAudioTask::Response(Box::new(
                        model.start(request, self.graph.realize_random(seed)?)?,
                    )),
                }
            }
            AudioTaskExecution::Omni { turn, encode_text } => {
                BoundAudioTask::Response(turn.start(self, encode_text)?)
            }
        })
    }
}
