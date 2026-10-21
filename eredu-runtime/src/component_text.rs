//! Typed text and refinement request construction for the shared component adapter.
use crate::*;
use eredu_nn::NeuralBackend;
use std::borrow::Borrow;

/// Explicit text request capability, independent of mutable-state storage.
pub trait TextComponentKernel<B: NeuralBackend>: ComponentKernel<B> {
    /// Whether the selected equations admit ordinary chunked prefill.
    fn supports_chunked_prefill() -> bool {
        false
    }
    /// Constructs the family request without allocating native resources.
    fn text_input<'a>(tokens: &'a B::Tensor, mask: Option<&'a B::Tensor>) -> Self::Input<'a>;
    /// Exact iterative geometry when the selected kernel supports block refinement.
    fn block_refinement_geometry(
        &self,
    ) -> Option<generation::refinement::architecture::BlockRefinementGeometry> {
        None
    }
    /// Constructs a supported invocation without changing parameters or state.
    fn invocation_input<'a>(
        tokens: &'a B::Tensor,
        feedback: Option<&'a B::Tensor>,
        mask: Option<&'a B::Tensor>,
        invocation: eredu_core::decoding::AttentionInvocation,
    ) -> Option<Self::Input<'a>> {
        (invocation == eredu_core::decoding::AttentionInvocation::CAUSAL && feedback.is_none())
            .then(|| Self::text_input(tokens, mask))
    }
    /// Projection of a complete output into the text session's score rows.
    fn text_output_selection(&self) -> ReplicatedTextOutputSelection {
        ReplicatedTextOutputSelection::LastSequencePosition
    }
}
impl<B, D, K, S, R> ReplicatedTextArchitecture<B, S> for ComponentAdapter<D, K, R>
where
    B: NeuralBackend,
    D: Borrow<ComponentDefinition> + 'static,
    R: ComponentResourceAccess<B, K> + 'static,
    S: RuntimeState<B>,
    K: TextComponentKernel<B> + ComponentEquations<B, S> + 'static,
{
    fn supports_chunked_prefill() -> bool {
        K::supports_chunked_prefill()
    }
    fn text_input<'a>(tokens: &'a B::Tensor, mask: Option<&'a B::Tensor>) -> Self::Input<'a> {
        K::text_input(tokens, mask)
    }
    fn block_refinement_geometry(
        &self,
    ) -> Option<generation::refinement::architecture::BlockRefinementGeometry> {
        self.kernel().block_refinement_geometry()
    }
    fn invocation_input<'a>(
        tokens: &'a B::Tensor,
        feedback: Option<&'a B::Tensor>,
        mask: Option<&'a B::Tensor>,
        invocation: eredu_core::decoding::AttentionInvocation,
    ) -> Option<Self::Input<'a>> {
        K::invocation_input(tokens, feedback, mask, invocation)
    }
    fn text_output_selection(&self) -> ReplicatedTextOutputSelection {
        self.kernel().text_output_selection()
    }
}

/// Explicit context/refinement input capability over the ordinary component lifecycle.
pub trait RefinementComponentKernel<B: NeuralBackend>: ComponentKernel<B> {
    /// Trained iterative score geometry.
    fn refinement_geometry(&self) -> generation::refinement::architecture::BlockRefinementGeometry;
    /// Authoritative context invocation.
    fn context_input<'a>(tokens: &'a B::Tensor, mask: Option<&'a B::Tensor>) -> Self::Input<'a>;
    /// Tentative canvas invocation, independently borrowing feedback and visibility.
    fn refinement_input<'a>(
        tokens: &'a B::Tensor,
        feedback: Option<&'a B::Tensor>,
        mask: Option<&'a B::Tensor>,
    ) -> Self::Input<'a>;
}
impl<B, D, K, S, R> generation::refinement::architecture::BlockRefinementArchitecture<B, S>
    for ComponentAdapter<D, K, R>
where
    B: NeuralBackend,
    D: Borrow<ComponentDefinition> + 'static,
    R: ComponentResourceAccess<B, K> + 'static,
    S: RuntimeState<B>,
    K: RefinementComponentKernel<B> + ComponentEquations<B, S> + 'static,
{
    fn refinement_geometry(&self) -> generation::refinement::architecture::BlockRefinementGeometry {
        self.kernel().refinement_geometry()
    }
    fn context_input<'a>(tokens: &'a B::Tensor, mask: Option<&'a B::Tensor>) -> Self::Input<'a> {
        K::context_input(tokens, mask)
    }
    fn refinement_input<'a>(
        tokens: &'a B::Tensor,
        feedback: Option<&'a B::Tensor>,
        mask: Option<&'a B::Tensor>,
    ) -> Self::Input<'a> {
        K::refinement_input(tokens, feedback, mask)
    }
}
