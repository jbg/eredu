//! Portable composition of Audio task mechanisms.

use std::marker::PhantomData;

/// Composes graph realization, media lowering, and transport for finite Audio tasks.
pub struct AudioTaskBinder<B, G, M, P> {
    pub(in crate::lfm3) graph: G,
    pub(in crate::lfm3) media: M,
    pub(in crate::lfm3) pipeline: P,
    backend: PhantomData<fn() -> B>,
}

impl<B, G, M, P> AudioTaskBinder<B, G, M, P>
where
    B: eredu_nn::NeuralBackend,
    G: eredu_runtime::RealtimeGraphMechanisms<B>,
{
    /// Retains independent mechanisms for the architecture-owned task driver.
    pub fn new(graph: G, media: M, pipeline: P) -> Self {
        Self {
            graph,
            media,
            pipeline,
            backend: PhantomData,
        }
    }
}
