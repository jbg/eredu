//! Backend-neutral causal-model session contracts.

use eredu_nn::Tensor;

/// Monomorphized causal model used by generation sessions.
///
/// Prepared input, mutable state, tensor, execution context, and error remain
/// concrete associated types. Backends may therefore preserve native graphs
/// and caches without virtual dispatch or host conversion in token loops.
pub trait CausalModel<S> {
    /// Backend-native tensor handle containing logits and decode token ids.
    type Tensor: Tensor;
    /// Borrowed, tokenizer/media-prepared prefill input.
    type Input<'a>: Copy;
    /// Concrete model or backend failure.
    type Error;

    /// Computes initial logits and updates mutable state.
    fn prefill_input_logits(
        &mut self,
        input: Self::Input<'_>,
        state: &mut S,
        context: &<Self::Tensor as Tensor>::Context,
    ) -> Result<Self::Tensor, Self::Error>;

    /// Computes logits for decode tokens using existing mutable state.
    fn decode_logits(
        &mut self,
        input_tokens: &Self::Tensor,
        state: &mut S,
        context: &<Self::Tensor as Tensor>::Context,
    ) -> Result<Self::Tensor, Self::Error>;

    /// Adjusts prefill logits before backend-native sampling.
    fn adjust_prefill_logits(
        &mut self,
        logits: Self::Tensor,
        _state: &mut S,
        _context: &<Self::Tensor as Tensor>::Context,
    ) -> Result<Self::Tensor, Self::Error> {
        Ok(logits)
    }
}
