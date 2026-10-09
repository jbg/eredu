//! EAR-TTS refinement using the ordinary realtime branch's retained random stream.
//!
//! Every active slot consumes one uniform tensor for Gumbel-Max and then one
//! standard-normal latent tensor. The architecture owns truncation and ordering;
//! backends provide numerical mechanisms. Candidate RNG is adopted only when the
//! requested refinement succeeds, independently of native completion ownership.
//! Supplying [`SpeechSamplingCoordination`] selects owner-only draws and shared
//! phase agreement within that same transaction. Pass `None` for local execution;
//! each coordinated operation requires fresh recorded invocation authority.

use super::tts_conditioning_layered::ConditioningExecution;
use super::tts_mog_layered::MogExecution;
use super::tts_rvq_layered::RvqExecution;
use super::tts_transformer_layered::TransformerExecution;
use eredu_codec::ear::AudioDecoderExecution;
use eredu_nn::{NeuralBackend, Tensor};
use eredu_runtime::{DenseSamplingBackend, RealtimeGenerationBranch, RealtimeScheduleState};

use super::{
    speech::{SpeechComponents, SpeechFrame, SpeechOutput},
    tts_generator::{TtsFrame, TtsGenerator},
    tts_mog::TtsError,
};

/// Sampling admission or refinement failure, without exposing native error types.
#[derive(Debug, thiserror::Error)]
pub enum SpeechSamplingError {
    /// A submitted transition cannot make further random decisions.
    #[error("speech sampling requires a branch without submission completion")]
    Submitted,
    /// Active refinement requires an explicitly retained random stream.
    #[error("speech sampling requires a retained random stream")]
    MissingRandomState,
    /// A neural or sampling mechanism failed; its original source is retained.
    #[error(transparent)]
    Refinement(#[from] TtsError),
}

#[path = "speech_coordination.rs"]
mod coordination;
use super::tts_rvq_layered::BorrowedRefinementSampling;
use coordination::sampled;
pub use coordination::SpeechSamplingCoordination;

impl<
        B: NeuralBackend,
        X: TransformerExecution<B>,
        Y: MogExecution<B>,
        Z: RvqExecution<B>,
        W: ConditioningExecution<B>,
        E: TransformerExecution<B>,
    > TtsGenerator<B, X, Y, Z, W, E>
{
    /// Advances one active slot using the enclosing realtime branch's RNG. The
    /// three empty schedule slots consume no draws. On any error, neither the
    /// supplied frame nor the branch RNG advances; native completion still
    /// belongs to the enclosing submission owner.
    pub fn advance_sampled_frame<D, M, Q, P, S, C>(
        &mut self,
        frame: &TtsFrame<B::Tensor>,
        branch: &mut RealtimeGenerationBranch<M, Q, P, S, D::RandomState, C>,
        coordination: Option<&mut SpeechSamplingCoordination<'_, B::Tensor>>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<TtsFrame<B::Tensor>, SpeechSamplingError>
    where
        D: DenseSamplingBackend<
            Logits = B::Tensor,
            Token = B::Tensor,
            Context = <B::Tensor as Tensor>::Context,
        >,
        D::RandomState: Clone,
        D::Error: std::error::Error + Send + Sync + 'static,
        Q: RealtimeScheduleState,
    {
        sampled::<B, D, _, _, _, _, _, _>(
            branch,
            frame.refinement().next_slot().is_some(),
            self.config().latent_size,
            1,
            coordination,
            context,
            |draw| self.advance_frame(frame, BorrowedRefinementSampling(draw), context),
        )
    }

    /// Completes a partial frame through the same slot transitions as controlled
    /// advancement. The whole call adopts RNG atomically, including all dependent
    /// MoG and RVQ operations. An already complete frame consumes no randomness.
    pub fn generate_sampled_frame<D, M, Q, P, S, C>(
        &mut self,
        frame: &TtsFrame<B::Tensor>,
        branch: &mut RealtimeGenerationBranch<M, Q, P, S, D::RandomState, C>,
        coordination: Option<&mut SpeechSamplingCoordination<'_, B::Tensor>>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<TtsFrame<B::Tensor>, SpeechSamplingError>
    where
        D: DenseSamplingBackend<
            Logits = B::Tensor,
            Token = B::Tensor,
            Context = <B::Tensor as Tensor>::Context,
        >,
        D::RandomState: Clone,
        D::Error: std::error::Error + Send + Sync + 'static,
        Q: RealtimeScheduleState,
    {
        sampled::<B, D, _, _, _, _, _, _>(
            branch,
            frame.refinement().next_slot().is_some(),
            self.config().latent_size,
            2,
            coordination,
            context,
            |draw| self.generate_frame(frame, BorrowedRefinementSampling(draw), context),
        )
    }
}

impl<
        B: NeuralBackend,
        X: TransformerExecution<B>,
        Y: MogExecution<B>,
        Z: RvqExecution<B>,
        W: ConditioningExecution<B>,
        E: TransformerExecution<B>,
        A: AudioDecoderExecution<B>,
    > SpeechComponents<'_, B, X, Y, Z, W, A, E>
{
    /// Completes retained refinement, applies speech feedback and renders one
    /// codec frame under a single RNG adoption boundary. A synchronous rendering
    /// failure preserves the supplied frame, codec carry and branch random state,
    /// so the retained frame can be retried after its owner establishes safe
    /// native completion, with fresh execution authority when the selected codec
    /// requires it. An already refined frame consumes no random draws.
    ///
    /// The returned waveform and state are unpublished. This operation does not
    /// repeat the causal TTS invocation or establish submission completion.
    pub fn render_sampled_frame<D, M, Q, P, S, C>(
        &mut self,
        frame: &SpeechFrame<B::Tensor, A::NeuralState>,
        branch: &mut RealtimeGenerationBranch<M, Q, P, S, D::RandomState, C>,
        coordination: Option<&mut SpeechSamplingCoordination<'_, B::Tensor>>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<SpeechOutput<B::Tensor, A::NeuralState>, SpeechSamplingError>
    where
        D: DenseSamplingBackend<
            Logits = B::Tensor,
            Token = B::Tensor,
            Context = <B::Tensor as Tensor>::Context,
        >,
        D::RandomState: Clone,
        D::Error: std::error::Error + Send + Sync + 'static,
        Q: RealtimeScheduleState,
    {
        sampled::<B, D, _, _, _, _, _, _>(
            branch,
            frame.tts().refinement().next_slot().is_some(),
            self.generator.config().latent_size,
            3,
            coordination,
            context,
            |draw| {
                let complete =
                    self.generate_frame(frame, BorrowedRefinementSampling(draw), context)?;
                self.render_with_sampling(&complete, BorrowedRefinementSampling(draw), context)
            },
        )
    }

    /// Advances one active slot from this speech frame on the shared branch RNG.
    /// Sampling and subsequent MoG/RVQ failures leave its retained RNG unchanged.
    pub fn advance_sampled_frame<D, M, Q, P, S, C>(
        &mut self,
        frame: &SpeechFrame<B::Tensor, A::NeuralState>,
        branch: &mut RealtimeGenerationBranch<M, Q, P, S, D::RandomState, C>,
        coordination: Option<&mut SpeechSamplingCoordination<'_, B::Tensor>>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<SpeechFrame<B::Tensor, A::NeuralState>, SpeechSamplingError>
    where
        D: DenseSamplingBackend<
            Logits = B::Tensor,
            Token = B::Tensor,
            Context = <B::Tensor as Tensor>::Context,
        >,
        D::RandomState: Clone,
        D::Error: std::error::Error + Send + Sync + 'static,
        Q: RealtimeScheduleState,
    {
        sampled::<B, D, _, _, _, _, _, _>(
            branch,
            frame.tts().refinement().next_slot().is_some(),
            self.generator.config().latent_size,
            1,
            coordination,
            context,
            |draw| self.advance_frame(frame, BorrowedRefinementSampling(draw), context),
        )
    }

    /// Finishes the same retained refinement with an atomic RNG adoption boundary
    /// spanning the requested slots. Waveform rendering and publication remain
    /// separate operations over this unpublished candidate.
    pub fn generate_sampled_frame<D, M, Q, P, S, C>(
        &mut self,
        frame: &SpeechFrame<B::Tensor, A::NeuralState>,
        branch: &mut RealtimeGenerationBranch<M, Q, P, S, D::RandomState, C>,
        coordination: Option<&mut SpeechSamplingCoordination<'_, B::Tensor>>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<SpeechFrame<B::Tensor, A::NeuralState>, SpeechSamplingError>
    where
        D: DenseSamplingBackend<
            Logits = B::Tensor,
            Token = B::Tensor,
            Context = <B::Tensor as Tensor>::Context,
        >,
        D::RandomState: Clone,
        D::Error: std::error::Error + Send + Sync + 'static,
        Q: RealtimeScheduleState,
    {
        sampled::<B, D, _, _, _, _, _, _>(
            branch,
            frame.tts().refinement().next_slot().is_some(),
            self.generator.config().latent_size,
            2,
            coordination,
            context,
            |draw| self.generate_frame(frame, BorrowedRefinementSampling(draw), context),
        )
    }
}
