//! Typed causal-text request construction for the shared component adapter.
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
}
