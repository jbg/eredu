//! Architecture-owned transformations at the shared decoder's residual boundaries.
use super::ComponentInstrumentation;
use eredu_nn::{Error, NeuralBackend, Parameterized, Tensor};

/// Converts ordinary embeddings to unit activations and complete residuals to
/// vocabulary-head inputs. The caller retains pre-collapse values for prediction.
/// Implementations operate independently at each sequence position, so the shared
/// causal-text path may narrow to the final position before invoking the readout.
pub trait DecoderBoundary<B: NeuralBackend>: Parameterized<B::Tensor> {
    /// Width shared by ordinary embeddings and the vocabulary-head input.
    fn embedding_width(&self) -> i32;
    /// Expands prepared embeddings, including embeddings with media replacements.
    fn expand(
        &self,
        embeddings: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>;

    /// Applies the architecture's final residual transformation with observations
    /// at the values actually consumed by its vocabulary projection.
    fn collapse(
        &mut self,
        residual: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error>;
}

/// Ordinary single-stream decoder boundary.
#[derive(Debug, eredu_nn::Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct NormalizedBoundary<B: NeuralBackend> {
    #[parameter(skip)]
    dimensions: i32,
    /// Architecture-selected final normalization, including its scale convention.
    pub norm: B::Normalization,
}
impl<B: NeuralBackend> Clone for NormalizedBoundary<B> {
    fn clone(&self) -> Self {
        Self {
            dimensions: self.dimensions,
            norm: self.norm.clone(),
        }
    }
}
impl<B: NeuralBackend> NormalizedBoundary<B> {
    /// Constructs an ordinary boundary from its exact normalization policy.
    pub fn new(
        spec: eredu_nn::NormalizationConstructionSpec,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        spec.validate()?;
        Ok(Self {
            dimensions: spec.dimensions,
            norm: B::normalization(spec, context)?,
        })
    }
}
impl<B: NeuralBackend> DecoderBoundary<B> for NormalizedBoundary<B> {
    fn embedding_width(&self) -> i32 {
        self.dimensions
    }
    fn expand(
        &self,
        embeddings: &B::Tensor,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        Ok(embeddings.clone())
    }
    fn collapse(
        &mut self,
        residual: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        instrumentation.normalize_readout(residual, &mut self.norm, context)
    }
}
