//! Embedded prediction fusion over the retained target residual streams.
//! SGLang equation reference: 7bdd8fe6ec2a94d6a0d11b885ca9fe6a9f610d53.
use eredu_nn::residual_streams::ResidualStreamGeometry;
use eredu_nn::{
    DeclareLogicalParameters, Error, LinearFormatSpec, LinearOperator, LinearSpec,
    LogicalParameterDeclaration, NeuralBackend, NormalizationConstructionSpec,
    NormalizationOperator, NormalizationScale, ParameterSpec, Tensor,
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
    /// Declares prediction-only fusion geometry without decoder state or invocation limits.
    pub fn from_config(
        config: &super::config::Config,
        mut format: impl FnMut(&str) -> Result<LinearFormatSpec, Error>,
    ) -> Result<Self, Error> {
        let geometry = ResidualStreamGeometry::new(config.residual.streams, config.hidden_size)?;
        let parameter = |name: &str| {
            let mut weight = ParameterSpec::trainable(name).map_err(Error::backend)?;
            weight.group = Some("mtp.fusion".into());
            Ok::<_, Error>(weight)
        };
        let norm = |name: &str, dimensions| -> Result<_, Error> {
            Ok(NormalizationConstructionSpec {
                reduction: eredu_nn::NormalizationReduction::Native,
                dimensions,
                groups: None,
                epsilon: config.norm_epsilon,
                scale: NormalizationScale::LearnedOffset {
                    weight: parameter(name)?,
                    offset: 1.,
                },
            })
        };
        let mut projection = |name: &str| -> Result<_, Error> {
            Ok(LinearSpec {
                input: geometry.hidden_size(),
                output: geometry.hidden_size(),
                weight: parameter(name)?,
                bias: None,
                format: format(name)?,
            })
        };
        let spec = Self {
            geometry,
            embedding_norm: norm("mtp.pre_fc_norm_embedding.weight", geometry.hidden_size())?,
            hidden_norm: norm("mtp.pre_fc_norm_hidden.weight", geometry.flattened_width())?,
            embedding_projection: projection("mtp.fc_embedding.weight")?,
            hidden_projection: projection("mtp.fc_hidden.weight")?,
        };
        spec.validate()?;
        Ok(spec)
    }
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
            return Err(Error::backend(
                "prediction fusion requires full residual normalization before per-stream projection",
            ));
        }
        for linear in [&self.embedding_projection, &self.hidden_projection] {
            if linear.input != hidden || linear.output != hidden || linear.bias.is_some() {
                return Err(Error::backend(
                    "prediction fusion projection must be bias-free hidden-to-hidden",
                ));
            }
            linear.format.validate_for_weight(&linear.weight)?;
        }
        eredu_nn::validate_logical_parameter_topology(&self.logical_parameters()?)
            .map_err(Error::backend_source)
    }
}
impl DeclareLogicalParameters for PredictionFusionSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let mut p = self.embedding_norm.logical_parameters()?;
        p.extend(self.hidden_norm.logical_parameters()?);
        p.extend(self.embedding_projection.logical_parameters()?);
        p.extend(self.hidden_projection.logical_parameters()?);
        Ok(p)
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

pub(crate) mod declarations;
mod decoder;
mod parallel;
mod parameters;

/// Execution-group identity for independent prediction decoder depths.
pub(crate) const PREDICTION_EXECUTION_GROUP: &str = "prediction";
pub use decoder::{
    PredictionForward, PredictionInput, PredictionLimits, PredictionShared, PredictionSharedSpec,
    PredictionSpec, PredictionUnit, PredictionUnitSpec,
};
pub use parallel::PredictionTensorPartition;
