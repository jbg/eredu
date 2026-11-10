//! Pinned vocabulary and residual-collapse declarations, independent of execution units.
use super::{
    config::Config,
    residual::{mixer_spec, ResidualBoundary},
};
use crate::decoder::StaticModules;
use eredu_nn::{
    residual_streams::GatedResidualSpec, DeclareLogicalParameters, EmbeddingSpec, Error,
    LinearFormatSpec, LinearSpec, LogicalParameterDeclaration, NeuralBackend, ParameterSpec,
    Tensor,
};
use std::borrow::Cow;

/// Canonical static slots; a tied output consumes the embedding slot directly.
#[derive(Debug, Clone)]
pub struct StaticParameters {
    /// Shared token lookup, including its retained encoding.
    pub embedding: EmbeddingSpec,
    /// Independent vocabulary projection; absent when the embedding owns output weights.
    pub head: Option<LinearSpec>,
    /// Final residual collapse, without an additional normalization after mixing.
    pub readout: GatedResidualSpec,
}
impl StaticParameters {
    /// Declares geometry, identities, ownership and sharing without native resources.
    pub fn from_config(
        config: &Config,
        mut format: impl FnMut(&str) -> Result<LinearFormatSpec, Error>,
    ) -> Result<Self, Error> {
        let parameter = |name: &str, owner: &str| -> Result<ParameterSpec, Error> {
            let mut parameter = ParameterSpec::trainable(name).map_err(Error::backend)?;
            parameter.group = Some(owner.into());
            Ok(parameter)
        };
        let embedding = EmbeddingSpec {
            vocabulary: config.vocabulary,
            dimensions: config.hidden_size,
            weight: parameter("model.embed_tokens.weight", "model.embed_tokens")?,
            format: format("model.embed_tokens.weight")?,
        };
        let head = if config.tied_embeddings {
            None
        } else {
            Some(LinearSpec {
                input: embedding.dimensions,
                output: embedding.vocabulary,
                weight: parameter("lm_head.weight", "lm_head")?,
                bias: None,
                format: format("lm_head.weight")?,
            })
        };
        let spec = Self {
            embedding,
            head,
            readout: mixer_spec(config, "model.hyper_connection_mixer", false, format)?,
        };
        spec.validate()?;
        Ok(spec)
    }
    /// The output invocation borrows an independent head or references the canonical table.
    /// A tied projection adds no logical slot; placement resolves any materialized replicas.
    pub fn output_projection(&self) -> Cow<'_, LinearSpec> {
        match &self.head {
            Some(head) => Cow::Borrowed(head),
            None => Cow::Owned(self.embedding.output_projection()),
        }
    }
    /// Rejects incompatible geometry, identities and sharing before backend allocation.
    pub fn validate(&self) -> Result<(), Error> {
        self.logical_parameters().map(|_| ())
    }
    fn validate_geometry(&self) -> Result<(), Error> {
        crate::decoder::validate_vocabulary_specs(&self.embedding, self.head.as_ref())?;
        self.readout.validate()?;
        if self.readout.injection.is_some()
            || self.embedding.dimensions != self.readout.geometry.hidden_size()
            || self.head.as_ref().is_some_and(|head| head.bias.is_some())
        {
            return Err(Error::backend(
                "Qwen4 static vocabulary/collapse geometry or equation mismatch",
            ));
        }
        Ok(())
    }
    /// Constructs exactly these slots through ordinary neural backend mechanisms.
    pub fn materialize<B: NeuralBackend>(
        &self,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<StaticModules<B, ResidualBoundary<B>>, Error> {
        self.validate()?;
        let readout = ResidualBoundary::new(self.readout.clone(), context)?;
        StaticModules::from_boundary(self.embedding.clone(), self.head.clone(), readout, context)
    }
}
impl DeclareLogicalParameters for StaticParameters {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        self.validate_geometry()?;
        let mut parameters = self.embedding.logical_parameters()?;
        parameters.extend(self.readout.logical_parameters()?);
        if let Some(head) = &self.head {
            parameters.extend(head.logical_parameters()?);
        }
        eredu_nn::validate_logical_parameter_topology(&parameters)
            .map_err(Error::backend_source)?;
        Ok(parameters)
    }
}
