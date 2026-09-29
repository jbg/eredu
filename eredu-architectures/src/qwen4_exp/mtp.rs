//! Embedded prediction fusion over the retained target residual streams.
//! SGLang equation reference: 7bdd8fe6ec2a94d6a0d11b885ca9fe6a9f610d53.
use eredu_nn::residual_streams::ResidualStreamGeometry;
use eredu_nn::{
    Error, LinearOperator, LinearSpec, NeuralBackend, NormalizationConstructionSpec,
    NormalizationOperator, Tensor,
};

/// Exact checkpoint parameters for residual-linear-shared prediction fusion.
#[derive(Debug, Clone)]
pub struct PredictionFusionSpec {
    /// Target residual shape retained before final collapse.
    pub geometry: ResidualStreamGeometry,
    /// Learned-offset RMS normalization of the ordinary token embedding.
    pub embedding_norm: NormalizationConstructionSpec,
    /// Learned-offset RMS normalization across the **complete flattened residual**.
    /// This deliberately has no per-stream groups.
    pub hidden_norm: NormalizationConstructionSpec,
    /// Projection of normalized next-token embeddings.
    pub embedding_projection: LinearSpec,
    /// Shared projection applied independently to each normalized residual stream.
    pub hidden_projection: LinearSpec,
}
impl PredictionFusionSpec {
    /// Checks the distinct full-residual and per-stream arithmetic geometry.
    pub fn validate(&self) -> Result<(), Error> {
        self.embedding_norm.validate()?;
        self.hidden_norm.validate()?;
        let hidden = self.geometry.hidden_size();
        if self.geometry.streams() <= 1
            || self.embedding_norm.dimensions != hidden
            || self.embedding_norm.groups.is_some()
            || self.hidden_norm.dimensions != self.geometry.flattened_width()
            || self.hidden_norm.groups.is_some()
        {
            return Err(Error::backend("prediction fusion requires full residual normalization before per-stream projection"));
        }
        for linear in [&self.embedding_projection, &self.hidden_projection] {
            if linear.input != hidden || linear.output != hidden || linear.bias.is_some() {
                return Err(Error::backend(
                    "prediction fusion projection must be bias-free hidden-to-hidden",
                ));
            }
            linear.format.validate_for_weight(&linear.weight)?;
        }
        Ok(())
    }
}
/// Stateless learned fusion; prediction decoder state remains separately owned.
#[derive(Debug, Clone, eredu_nn::Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct PredictionFusion<B: NeuralBackend> {
    #[parameter(skip)]
    geometry: ResidualStreamGeometry,
    embedding_norm: B::Normalization,
    hidden_norm: B::Normalization,
    embedding_projection: B::Linear,
    hidden_projection: B::Linear,
}
impl<B: NeuralBackend> PredictionFusion<B> {
    /// Constructs the exact fusion parameters through ordinary generic binding.
    pub fn new(
        spec: PredictionFusionSpec,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        spec.validate()?;
        B::require_operator_capabilities(
            "qwen4_exp prediction fusion",
            eredu_nn::NeuralOperatorCapabilities::BROADCAST_TO,
        )?;
        Ok(Self {
            geometry: spec.geometry,
            embedding_norm: B::normalization(spec.embedding_norm, context)?,
            hidden_norm: B::normalization(spec.hidden_norm, context)?,
            embedding_projection: B::linear(spec.embedding_projection, context)?,
            hidden_projection: B::linear(spec.hidden_projection, context)?,
        })
    }
    /// Fuses ordinary shared token embeddings with complete pre-collapse target
    /// streams, returning the same `[batch,tokens,streams,hidden]` residual shape.
    pub fn forward(
        &mut self,
        embeddings: &B::Tensor,
        target: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.geometry.validate_streams(target.shape())?;
        if embeddings.shape() != [target.dim(0), target.dim(1), self.geometry.hidden_size()] {
            return Err(Error::backend(
                "prediction token embeddings must align with retained target streams",
            ));
        }
        let embedding = self.embedding_norm.forward(embeddings, context)?;
        let embedding = self.geometry.expand(
            &self.embedding_projection.forward(&embedding, context)?,
            context,
        )?;
        let hidden = self
            .hidden_norm
            .forward(&self.geometry.flatten(target, context)?, context)?;
        let hidden = self.geometry.unflatten(&hidden, context)?;
        self.hidden_projection
            .forward(&hidden, context)?
            .add(&embedding, context)
    }
}

mod decoder;
mod parallel;
pub use decoder::{
    PredictionForward, PredictionInput, PredictionLimits, PredictionShared, PredictionSpec,
    PredictionUnit, PredictionUnitSpec,
};
pub use parallel::PredictionTensorPartition;
