//! PCM carry and shared perception/transcription advance as one candidate.
//!
//! Submission only admits host PCM. Advancement borrows the committed parent;
//! the shared execution owner retains the returned state with the rest of the
//! neural transition and establishes completion before publishing it.

use eredu_core::{execution_control::SnapshotEstimate, BackendFailure};
use eredu_media::audio::streaming::{PcmFinishPolicy, PcmFrameSource};
use eredu_runtime::execution_control::{
    snapshot_value_estimate, SnapshotReservation, SnapshotValue,
};

use eredu_nn::RetainedTensorRoots;

use eredu_nn::{NeuralBackend, Tensor};
use eredu_runtime::{RealtimeGenerationBranch, RealtimeScheduleState};

use super::{
    config::PreprocessingConfig,
    frontend::{FrontendError, FrontendOptions, FrontendStep, PcmPrecision, VoiceChatFrontend},
    perception::PerceptionState,
    perception_rnnt::{
        PerceptionCacheState, PerceptionRnnt, PerceptionRnntError, PerceptionRnntFrame,
        PerceptionRnntLimits, PerceptionRnntState,
    },
    state::StateBranch,
};

/// Host PCM policy and frontier of an unpublished input candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PcmInputStatus {
    /// Precision applied to accepted PCM before feature extraction.
    pub pcm_precision: PcmPrecision,
    /// Maximum accepted, unprocessed PCM samples.
    pub queue_capacity_samples: usize,
    /// Accepted samples waiting for perception advancement.
    pub queued_samples: usize,
    /// Number of input and continuation cadence frames consumed by this candidate.
    pub frames: u64,
    /// Whether the complete input finish policy has entered this frontend.
    pub finished: bool,
    /// Unconsumed synthetic cadence frames in the declared finish policy.
    pub remaining_continuation_frames: u64,
    /// Whether ingress is finished and all source and continuation frames are consumed.
    pub drained: bool,
}

/// Host PCM access to an unpublished component transaction.
///
/// Admission changes only the branch's input candidate. The enclosing session
/// retains submitted PCM until that same branch reaches its semantic commit.
pub trait PcmInputState {
    /// Reports candidate input policy and frontier without exposing neural caches.
    fn pcm_input_status(&self) -> PcmInputStatus;

    /// Admits host PCM into the candidate before execution begins.
    fn submit_pcm(&mut self, samples: &[f32], sample_rate: u32) -> Result<(), InputError>;

    /// Declares final partial-frame handling and bounded silence continuation.
    fn finish_pcm(&mut self, policy: PcmFinishPolicy) -> Result<(), InputError>;
}

impl<T: Tensor, N, C> PcmInputState for StateBranch<T, N, C> {
    fn pcm_input_status(&self) -> PcmInputStatus {
        let frontend = self.input.frontend();
        let options = frontend.options();
        PcmInputStatus {
            pcm_precision: options.pcm_precision,
            queue_capacity_samples: options.queue_capacity_samples,
            queued_samples: frontend.queued_samples(),
            frames: frontend.frames(),
            finished: frontend.is_finished(),
            remaining_continuation_frames: frontend.remaining_continuation_frames(),
            drained: frontend.is_drained(),
        }
    }

    fn submit_pcm(&mut self, samples: &[f32], sample_rate: u32) -> Result<(), InputError> {
        self.input.submit(samples, sample_rate)
    }

    fn finish_pcm(&mut self, policy: PcmFinishPolicy) -> Result<(), InputError> {
        self.input.finish(policy)
    }
}

impl<M: PcmInputState, Q: RealtimeScheduleState, P, S, R, C> PcmInputState
    for RealtimeGenerationBranch<M, Q, P, S, R, C>
{
    fn pcm_input_status(&self) -> PcmInputStatus {
        self.model_state().pcm_input_status()
    }

    fn submit_pcm(&mut self, samples: &[f32], sample_rate: u32) -> Result<(), InputError> {
        self.model_state_mut().submit_pcm(samples, sample_rate)
    }

    fn finish_pcm(&mut self, policy: PcmFinishPolicy) -> Result<(), InputError> {
        self.model_state_mut().finish_pcm(policy)
    }
}

/// Host preprocessing carry and both neural input caches at the same audio frontier.
#[derive(Debug, Clone)]
pub struct InputState<T, C = PerceptionState<T>> {
    frontend: VoiceChatFrontend,
    neural: PerceptionRnntState<T, C>,
}

impl<T: SnapshotValue, C: SnapshotValue> SnapshotValue for InputState<T, C> {
    fn estimate_snapshot_value(&self) -> Result<SnapshotEstimate, BackendFailure> {
        snapshot_value_estimate::<Self>([
            self.frontend.estimate_snapshot_value(),
            self.neural.estimate_snapshot_value(),
        ])
    }
    fn copy_snapshot_value(&self, retention: &SnapshotReservation) -> Result<Self, BackendFailure> {
        Ok(Self {
            frontend: self.frontend.copy_snapshot_value(retention)?,
            neural: self.neural.copy_snapshot_value(retention)?,
        })
    }
}

impl<T, C: RetainedTensorRoots<T>> RetainedTensorRoots<T> for InputState<T, C> {
    fn visit_retained_tensors<'a>(&'a self, visitor: &mut dyn FnMut(&'a T)) {
        self.neural.visit_retained_tensors(visitor);
    }
}

impl<T> InputState<T> {
    /// Creates input state from the admitted checkpoint's exact host coefficients.
    /// Construction performs no source-store reads or native allocation.
    pub fn from_retained(
        artifacts: &super::artifact::RetainedVoiceChatArtifacts,
        options: FrontendOptions,
    ) -> Result<Self, InputError> {
        Ok(Self {
            frontend: VoiceChatFrontend::from_retained(artifacts, options)?,
            neural: PerceptionRnntState::default(),
        })
    }

    /// Creates empty host and neural state, without allocating backend tensors.
    pub fn new(
        preprocessing: &PreprocessingConfig,
        options: FrontendOptions,
    ) -> Result<Self, InputError> {
        Ok(Self {
            frontend: VoiceChatFrontend::new(preprocessing, options)?,
            neural: PerceptionRnntState::default(),
        })
    }
}

impl<T, C> InputState<T, C> {
    /// Admits a bounded PCM chunk without advancing perception or transcription.
    pub fn submit(&mut self, samples: &[f32], sample_rate: u32) -> Result<(), InputError> {
        Ok(self.frontend.submit(samples, sample_rate)?)
    }

    /// Closes PCM ingress with explicit partial handling and finite silence continuation.
    pub fn finish(&mut self, policy: PcmFinishPolicy) -> Result<(), InputError> {
        Ok(self.frontend.finish(policy)?)
    }

    /// Read-only host carry and queued-sample accounting.
    pub fn frontend(&self) -> &VoiceChatFrontend {
        &self.frontend
    }

    /// Paired caches and retained transcription hypothesis.
    pub fn neural(&self) -> &PerceptionRnntState<T, C> {
        &self.neural
    }
}

impl<T, C: PerceptionCacheState> InputState<T, C> {
    /// Joins retained host carry and exact stage-local neural caches at one frontier.
    /// Construction performs no artifact reads, tensor allocation or advancement.
    pub fn from_parts(
        frontend: VoiceChatFrontend,
        neural: PerceptionRnntState<T, C>,
    ) -> Result<Self, InputError> {
        if frontend.frames() != neural.frames() {
            return Err(InputError::State("PCM and neural input frontiers differ"));
        }
        Ok(Self { frontend, neural })
    }
}

/// One unpublished audio frame with exact input-sample attribution.
#[derive(Debug, Clone)]
pub struct InputFrame<T, C = PerceptionState<T>> {
    /// Candidate PCM carry and both neural caches, adopted together at commit.
    pub state: InputState<T, C>,
    /// Zero-based live-audio frame identity, independent of tool/model positions.
    pub frame_index: u64,
    /// Absolute input sample index at the source sampling rate.
    pub start_sample: u64,
    /// Source samples in this frame, excluding explicit final padding.
    pub valid_samples: usize,
    /// Original PCM or an explicitly requested synthetic silence cadence.
    pub source: PcmFrameSource,
    /// Unprojected `[1, 1, encoder_width]` features consumed by RNNT.
    pub encoder: T,
    /// Projected `[1, 1, output_width]` features for Nemotron fusion.
    pub projected: T,
    /// Newly decoded transcription tokens, in order.
    pub emitted: Vec<usize>,
    /// First joint decision was blank.
    pub first_blank: bool,
    /// Final evaluated joint decision was blank.
    pub ended_blank: bool,
    /// Decoding stopped on a nonblank decision at the symbol bound.
    pub hit_symbol_limit: bool,
}

impl<T, C: RetainedTensorRoots<T>> RetainedTensorRoots<T> for InputFrame<T, C> {
    fn visit_retained_tensors<'a>(&'a self, visitor: &mut dyn FnMut(&'a T)) {
        self.state.visit_retained_tensors(visitor);
        visitor(&self.encoder);
        visitor(&self.projected);
    }
}

/// Portable input validation and component errors; every failure preserves the parent.
#[derive(Debug, thiserror::Error)]
pub enum InputError {
    /// Host PCM admission or preprocessing failed.
    #[error(transparent)]
    Frontend(#[from] FrontendError),
    /// Shared perception/transcription failed without adopting its candidate.
    #[error(transparent)]
    PerceptionRnnt(#[from] PerceptionRnntError),
    /// A retained frontier or host feature dimension is incompatible.
    #[error("{0}")]
    State(&'static str),
    /// Distributed host admission or completion failed, preserving its source.
    #[error(transparent)]
    Backend(#[from] BackendFailure),
    /// Tensor creation failed, preserving the original backend source.
    #[error(transparent)]
    Neural(#[from] eredu_nn::Error),
}

/// Shared host-PCM advancement over ordinary or partitioned neural input execution.
pub trait InputExecution<B: NeuralBackend> {
    /// Exact perception cache owner retained with the host frontend and RNNT state.
    type PerceptionCache: PerceptionCacheState + Clone + RetainedTensorRoots<B::Tensor>;
    /// Consumes at most one cadence frame into an unpublished candidate.
    /// Incomplete input returns `None` without consuming PCM or neural authority.
    fn advance_pcm(
        &mut self,
        previous: &InputState<B::Tensor, Self::PerceptionCache>,
        limits: PerceptionRnntLimits,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Option<InputFrame<B::Tensor, Self::PerceptionCache>>, InputError>;
}

pub(crate) struct PreparedInput {
    pub(crate) frontend: FrontendStep,
    pub(crate) shape: [i32; 3],
}
pub(crate) fn prepare_frontend<T, C: PerceptionCacheState>(
    previous: &InputState<T, C>,
) -> Result<Option<PreparedInput>, InputError> {
    if previous.frontend.frames() != previous.neural.frames() {
        return Err(InputError::State("PCM and neural input frontiers differ"));
    }
    let Some(frontend) = previous.frontend.advance()? else {
        return Ok(None);
    };
    let frames = i32::try_from(frontend.mel.frames)
        .map_err(|_| InputError::State("mel time exceeds tensor indexing"))?;
    let bins = i32::try_from(frontend.mel.mel_bins)
        .map_err(|_| InputError::State("mel bins exceed tensor indexing"))?;
    Ok(Some(PreparedInput {
        frontend,
        shape: [1, frames, bins],
    }))
}
pub(crate) fn materialize<B: NeuralBackend>(
    prepared: &PreparedInput,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<B::Tensor, InputError> {
    Ok(B::Tensor::from_f32_slice(
        &prepared.frontend.mel.values,
        &prepared.shape,
        context,
    )?)
}
pub(crate) fn compose_frame<T, C: PerceptionCacheState>(
    prepared: PreparedInput,
    neural: PerceptionRnntFrame<T, C>,
) -> Result<InputFrame<T, C>, InputError> {
    let frontend = prepared.frontend;
    if frontend.state.frames() != neural.state.frames() {
        return Err(InputError::State(
            "PCM and neural candidate frontiers differ",
        ));
    }
    Ok(InputFrame {
        state: InputState {
            frontend: frontend.state,
            neural: neural.state,
        },
        frame_index: frontend.frame_index,
        start_sample: frontend.start_sample,
        valid_samples: frontend.valid_samples,
        source: frontend.source,
        encoder: neural.encoder,
        projected: neural.projected,
        emitted: neural.emitted,
        first_blank: neural.first_blank,
        ended_blank: neural.ended_blank,
        hit_symbol_limit: neural.hit_symbol_limit,
    })
}
impl<
        B: NeuralBackend,
        P: super::perception_layered::PerceptionExecution<B>,
        J: super::rnnt::RnntExecution<B>,
    > InputExecution<B> for PerceptionRnnt<B, P, J>
{
    type PerceptionCache = PerceptionState<B::Tensor>;
    fn advance_pcm(
        &mut self,
        previous: &InputState<B::Tensor>,
        limits: PerceptionRnntLimits,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Option<InputFrame<B::Tensor>>, InputError> {
        let Some(prepared) = prepare_frontend(previous)? else {
            return Ok(None);
        };
        self.validate_advance(
            &prepared.shape,
            prepared.frontend.drop_preencoded,
            &previous.neural,
            limits,
        )?;
        let mel = materialize::<B>(&prepared, context)?;
        let neural = self.advance(
            &mel,
            prepared.frontend.drop_preencoded,
            &previous.neural,
            limits,
            context,
        )?;
        Ok(Some(compose_frame(prepared, neural)?))
    }
}
