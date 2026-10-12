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
