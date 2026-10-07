//! Observed component frontiers for checked VoiceChat schedule receipts.
//!
//! These reads do not advance a component. Access to an existing bounded
//! backbone state may materialize its selected local slots. Publication and
//! native completion remain owned by the enclosing realtime transaction.

use super::backbone_execution::{BackboneExecution, BackboneStateError};
use super::tts_transformer_layered::{CausalTransformerExecution, TransformerError};
use eredu_nn::NeuralBackend;

use super::{
    input::InputState,
    perception_rnnt::PerceptionCacheState,
    schedule::{Progress, ScheduleError, ScheduleState, StepPlan},
    speech::SpeechState,
};

/// A component has no observable frontier, or retained components disagree.
#[derive(Debug, thiserror::Error)]
pub enum ProgressError {
    /// Selected backbone state validation failed.
    #[error(transparent)]
    Backbone(#[from] BackboneStateError),
    /// Selected causal state validation failed.
    #[error(transparent)]
    Tts(#[from] TransformerError),
    /// Components cannot describe one complete candidate boundary.
    #[error("invalid VoiceChat component progress: {0}")]
    Inconsistent(&'static str),
    /// Observed execution does not satisfy its previously admitted schedule.
    #[error(transparent)]
    Schedule(#[from] ScheduleError),
}

/// Reads the actual frontiers of one candidate's input, local backbone and speech.
///
/// The selected backbone provider validates and reports its state frontier.
/// Ordinary and tensor-parallel layered providers compare every stateful local
/// layer and skip stateless MLP slots without accessing remote state. TTS guidance occupies
/// batch rows in the same selected state. Its provider validates local cache
/// ownership and returns the logical invocation frontier, including on stages
/// without attention caches. Absent speech requires the cold TTS frontier.
pub fn observe<B, S: ?Sized, C: ?Sized, X, R>(
    input: &InputState<B::Tensor, impl PerceptionCacheState>,
    provider: &R,
    backbone: &mut S,
    transformer: &X,
    tts_state: &C,
    speech: Option<&SpeechState<B::Tensor, impl Sized>>,
) -> Result<Progress, ProgressError>
where
    B: NeuralBackend,
    R: BackboneExecution<B, S>,
    X: CausalTransformerExecution<B, C>,
{
    let backbone_positions = provider.position(backbone)?;
    let tts_position = frontier(transformer.causal_position(tts_state)?)?;
    let mut observed = Progress {
        audio_frames: input.frontend().frames(),
        perception_frames: input.neural().perception().frames(),
        rnnt_frames: input.neural().rnnt().frames(),
        backbone_positions,
        ..Progress::default()
    };
    if observed.audio_frames != observed.perception_frames
        || observed.audio_frames != observed.rnnt_frames
    {
        return Err(ProgressError::Inconsistent(
            "input components have different frontiers",
        ));
    }
    match speech {
        Some(speech) => {
            if frontier(speech.model_position())? != tts_position {
                return Err(ProgressError::Inconsistent(
                    "TTS caches differ from retained speech state",
                ));
            }
            observed.tts_conditional_positions = tts_position;
            observed.tts_unconditional_positions = if speech.tts_config().guidance_enabled {
                tts_position
            } else {
                0
            };
            observed.codec_frames = speech.codec().frames_seen();
            observed.output_samples = speech.codec().samples_emitted();
            observed.codec_finished = speech.codec().is_finished();
        }
        None if tts_position != 0 => {
            return Err(ProgressError::Inconsistent(
                "advanced TTS caches have no speech state",
            ));
        }
        None => {}
    }
    Ok(observed)
}

/// Completes an admitted plan using component observations, without publishing it.
/// The enclosing branch must still retain every native resource through its exact
/// completion, including when observation or receipt validation fails.
pub fn complete<B, S: ?Sized, C: ?Sized, X, R>(
    plan: StepPlan,
    input: &InputState<B::Tensor, impl PerceptionCacheState>,
    provider: &R,
    backbone: &mut S,
    transformer: &X,
    tts_state: &C,
    speech: Option<&SpeechState<B::Tensor, impl Sized>>,
) -> Result<ScheduleState, ProgressError>
where
    B: NeuralBackend,
    R: BackboneExecution<B, S>,
    X: CausalTransformerExecution<B, C>,
{
    Ok(plan.complete(observe::<B, S, C, X, R>(
        input,
        provider,
        backbone,
        transformer,
        tts_state,
        speech,
    )?)?)
}

fn frontier(position: i32) -> Result<u64, ProgressError> {
    u64::try_from(position).map_err(|_| ProgressError::Inconsistent("negative component frontier"))
}
