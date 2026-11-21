//! Portable generation request ownership over native numerical state.
use super::specialized::TextModeBinding;
use crate::{capture::CaptureSession, speculative::shared_backbone::SharedBackboneMechanisms};
use eredu_core::refinement::RefinementBackend;

/// Numerical state for the selected text execution mechanisms.
pub type TextSamplingFor<B> = TextSamplingState<
    <B as TextModeBinding>::Refinement,
    <B as TextModeBinding>::Shared,
    crate::generation::TextSampler,
>;

/// Request state for a backend used by the portable text driver.
pub type TextRequestFor<B> = TextRequestState<
    <B as TextModeBinding>::Refinement,
    <B as TextModeBinding>::Shared,
    crate::generation::TextSampler,
>;

/// The portable driver owns the request cursor, numerical state, and observation.
pub struct TextRequestState<R: RefinementBackend, M: SharedBackboneMechanisms, S> {
    pub(super) sampling: TextSamplingState<R, M, S>,
    pub(super) readout_selection: Option<Vec<eredu_core::refinement::RefinementScoreSelection>>,
    pub(super) readout_rows: Option<eredu_core::refinement::RefinementScoreRows>,
    pub(super) capture: Option<CaptureSession>,
}

impl<R: RefinementBackend, M: SharedBackboneMechanisms, S> TextRequestState<R, M, S> {
    /// Installs native sampler and random storage for a portable generation request.
    pub fn start(
        config: eredu_core::TextGenerationConfig,
        sampler: S,
        prng: Option<R::RandomState>,
    ) -> Self {
        Self {
            sampling: TextSamplingState {
                temperature: config.sampling().temperature,
                prng,
                sampler,
                next_prediction: 0,
                parameter_epoch: None,
                refinement_seed: config.seed(),
                proposal_random: None,
                draft_events: config.draft_events(),
                canvas: config.refinement_canvas().cloned(),
                decoding: super::TextModeState {
                    mode: config.decoding_mode().unwrap_or_default(),
                    speculative: None,
                    committed_token_cached: false,
                    next_logits: None,
                    refinement_random: None,
                    refinement_policy: config.refinement_policy().unwrap_or_default(),
                    refinement: None,
                },
            },
            capture: None,
            readout_selection: None,
            readout_rows: None,
        }
    }
}

/// Retained ordinary sampling and mode state, independent of the native storage types.
pub struct TextSamplingState<R, M, S>
where
    R: eredu_core::refinement::RefinementBackend,
    M: crate::speculative::shared_backbone::SharedBackboneMechanisms,
{
    /// Current physical decoding mode and its retained values.
    pub decoding: super::TextModeState<R, M>,
    /// Selected positive sampling temperature or greedy setting.
    pub temperature: f32,
    /// Explicit sampling stream, if used.
    pub prng: Option<R::RandomState>,
    /// Configured sampler and penalty history.
    pub sampler: S,
    /// Next absolute prediction position.
    pub next_prediction: u64,
    /// Parameter version bound at submission.
    pub parameter_epoch: Option<u64>,
    /// Seed retained for the refinement stream.
    pub refinement_seed: u64,
    /// Independently advancing proposal stream.
    pub proposal_random: Option<R::RandomState>,
    /// Whether draft events are requested.
    pub draft_events: bool,
    /// Retained immutable conditioning canvas.
    pub canvas: Option<std::sync::Arc<eredu_core::refinement::RefinementCanvas>>,
}
