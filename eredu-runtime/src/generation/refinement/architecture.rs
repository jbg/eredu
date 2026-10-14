//! Architecture-owned input equations for iterative token-block execution.

use crate::{LayeredArchitecture, RuntimeState};
use eredu_nn::NeuralBackend;
use std::num::NonZeroU32;

/// Positive trained score geometry retained by cold execution selection.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct BlockRefinementGeometry {
    /// Number of tentative positions per model invocation.
    pub canvas_length: NonZeroU32,
    /// Complete canonical token domain width.
    pub vocabulary_size: NonZeroU32,
}

/// Shared context and refinement execution over one retained model state.
///
/// The architecture supplies each borrowed input, visibility and score geometry;
/// the ordinary layered session supplies residency, observation, transport and
/// exact completion. Sampler state belongs to `BlockRefinementState`, independently
/// of model parameters and authoritative context history.
pub trait BlockRefinementArchitecture<B, S>: LayeredArchitecture<B, S>
where
    B: NeuralBackend,
    S: RuntimeState<B>,
{
    /// Trained number of tentative positions, including a truncated final output block.
    fn refinement_geometry(&self) -> BlockRefinementGeometry;
    /// Authoritative text context input. Prepared media uses the architecture's
    /// composite ingress before this execution seam.
    fn context_input<'a>(tokens: &'a B::Tensor, mask: Option<&'a B::Tensor>) -> Self::Input<'a>;
    /// Tentative canvas and embedding-precision feedback, without context mutation.
    fn refinement_input<'a>(
        canvas: &'a B::Tensor,
        feedback: Option<&'a B::Tensor>,
        mask: Option<&'a B::Tensor>,
    ) -> Self::Input<'a>;
}

/// Architecture-declared ordinary refinement, separate from speculative drafting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RefinementExecutionPolicy {
    /// Complete trained invocation geometry.
    pub geometry: BlockRefinementGeometry,
    /// Validated numerical defaults.
    pub policy: eredu_core::refinement::RefinementPolicy,
    /// Whether the same architecture also supplies causal conditional sampling.
    pub causal_sampling: bool,
    /// Whether context scores seed the first tentative masked position.
    pub seed_first: bool,
}
impl RefinementExecutionPolicy {
    /// Independent numerical stream used by the ordinary refinement policy.
    /// Entropy feedback retains its published root seed; masked refinement uses
    /// a separate stream from causal target sampling.
    pub const fn randomness_substream(self) -> Option<usize> {
        use eredu_core::refinement::RefinementPolicy;
        match self.policy {
            RefinementPolicy::EntropyFeedback(_) => None,
            RefinementPolicy::MaskedUnmasking { .. } => Some(1),
        }
    }
    /// Resolves request controls without touching parameters or mutable state.
    pub fn prepare(
        self,
        requested: Option<eredu_core::refinement::RefinementPolicy>,
        temperature: f32,
    ) -> Result<eredu_core::refinement::RefinementPolicy, eredu_core::decoding::DecodingModeError>
    {
        use eredu_core::refinement::RefinementPolicy as P;
        let policy = match (self.policy, requested) {
            (P::EntropyFeedback(default), None) => P::EntropyFeedback(default),
            (P::EntropyFeedback(_), Some(policy @ P::EntropyFeedback(_))) => policy,
            (
                P::MaskedUnmasking {
                    config,
                    mask_token_id,
                },
                None,
            ) => {
                let mut options = config.options();
                options.temperature = temperature;
                P::MaskedUnmasking {
                    config: options.try_into().map_err(|_| {
                        eredu_core::decoding::DecodingModeError::Unsupported(
                            "refinement temperature",
                        )
                    })?,
                    mask_token_id,
                }
            }
            (
                P::MaskedUnmasking {
                    mask_token_id: expected,
                    ..
                },
                Some(policy @ P::MaskedUnmasking { mask_token_id, .. }),
            ) if expected == mask_token_id => policy,
            _ => {
                return Err(eredu_core::decoding::DecodingModeError::Unsupported(
                    "refinement equations differ from retained architecture",
                ));
            }
        };
        Ok(policy)
    }
}
