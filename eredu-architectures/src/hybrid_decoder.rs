//! Pinned shared resources for heterogeneous text decoders.

use eredu_nn::NeuralBackend;

use crate::decoder::{DecoderBoundary, NormalizedBoundary, StaticModules};

/// Pinned decoder modules with architecture-owned shared execution parameters.
#[derive(Debug, eredu_nn::Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct HybridStaticModules<B: NeuralBackend, E, D: DecoderBoundary<B> = NormalizedBoundary<B>> {
    /// Shared token embedding, normalization, and vocabulary projection.
    pub base: StaticModules<B, D>,
    /// Architecture-owned pinned modules shared by execution units.
    pub extension: E,
}

impl<B: NeuralBackend, E: Clone, D: DecoderBoundary<B> + Clone> Clone
    for HybridStaticModules<B, E, D>
{
    fn clone(&self) -> Self {
        Self {
            base: self.base.clone(),
            extension: self.extension.clone(),
        }
    }
}
