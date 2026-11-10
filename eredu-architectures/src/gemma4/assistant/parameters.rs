//! Assistant-owned fusion and vocabulary slots, independent of shared-KV invocation state.

use super::AssistantConfig;
use eredu_checkpoint::LinearFormat;
use eredu_nn::{
    DeclareLogicalParameters, EmbeddingSpec, Error, LinearSpec, LogicalParameterDeclaration,
    LogicalParameterRole, NormalizationConstructionSpec, ParameterSpec,
};

#[derive(Debug, Clone)]
pub(super) struct OrderingSpec {
    pub parameter: ParameterSpec,
    pub shape: [i32; 1],
}
impl DeclareLogicalParameters for OrderingSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        Ok(vec![LogicalParameterDeclaration::new(
            &self.parameter,
            &self.shape,
            LogicalParameterRole::TokenMapping,
        )?])
    }
}
#[derive(Debug, Clone)]
pub(super) struct OrderedParameters {
    pub centroids: LinearSpec,
    pub ordering: OrderingSpec,
    pub table: EmbeddingSpec,
}
impl DeclareLogicalParameters for OrderedParameters {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let mut parameters = self.centroids.logical_parameters()?;
        parameters.extend(self.ordering.logical_parameters()?);
        parameters.extend(self.table.logical_parameters()?);
        Ok(parameters)
    }
}
#[derive(Debug, Clone)]
pub(super) enum Vocabulary {
    Tied(EmbeddingSpec),
    Output(LinearSpec),
    Ordered(OrderedParameters),
}
impl DeclareLogicalParameters for Vocabulary {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        match self {
            Self::Tied(spec) => spec.logical_parameters(),
            Self::Output(spec) => spec.logical_parameters(),
            Self::Ordered(spec) => spec.logical_parameters(),
        }
    }
}
#[derive(Debug, Clone)]
pub(super) struct StaticParameters {
    pub pre: LinearSpec,
    pub post: LinearSpec,
    pub norm: NormalizationConstructionSpec,
    pub vocabulary: Vocabulary,
}
impl StaticParameters {
    pub fn project_formats(&self, config: &AssistantConfig) -> Result<Self, Error> {
        config.validate().map_err(Error::backend)?;
        let args = &config.text_config;
        let encoding = |name: &str| {
            config
                .quantization
                .map(Into::into)
                .unwrap_or_else(|| args.linear_format_for(name))
        };
        let linear = |spec: &mut LinearSpec| -> Result<(), Error> {
            spec.format = crate::linear_format::project_matrix_format(
                &spec.weight,
                &spec.format,
                encoding(spec.weight.id.as_str()),
            )?;
            Ok(())
        };
        let mut projected = self.clone();
        linear(&mut projected.pre)?;
        linear(&mut projected.post)?;
        match &mut projected.vocabulary {
            Vocabulary::Tied(table) => {
                table.format = crate::linear_format::project_matrix_format(
                    &table.weight,
                    &table.format,
                    args.linear_format_for(table.weight.id.as_str()),
                )?;
            }
            Vocabulary::Output(head) => linear(head)?,
            Vocabulary::Ordered(ordered) => {
                if args.linear_format_for(ordered.table.weight.id.as_str()) != LinearFormat::Dense
                    || args.linear_format_for(ordered.centroids.weight.id.as_str())
                        != LinearFormat::Dense
                {
                    return Err(Error::backend(
                        "ordered assistant vocabulary slots require dense storage",
                    ));
                }
            }
        }
        eredu_nn::validate_logical_parameter_topology(&projected.logical_parameters()?)
            .map_err(Error::backend_source)?;
        Ok(projected)
    }
    pub fn new(config: &AssistantConfig) -> Result<Self, Error> {
        config.validate().map_err(Error::backend)?;
        let args = &config.text_config;
        let format = |name: &str| {
            config
                .quantization
                .map(Into::into)
                .unwrap_or_else(|| args.linear_format_for(name))
        };
        let linear = |name: &str, input, output, encoding| -> Result<LinearSpec, Error> {
            Ok(LinearSpec {
                input,
                output,
                weight: ParameterSpec::trainable(name).map_err(Error::backend)?,
                bias: None,
                format: crate::linear_format::standard_linear_format(name, encoding)?,
            })
        };
        let table = |encoding| -> Result<EmbeddingSpec, Error> {
            Ok(EmbeddingSpec {
                vocabulary: args.vocab_size,
                dimensions: args.hidden_size,
                weight: ParameterSpec::trainable("model.embed_tokens.weight")
                    .map_err(Error::backend)?,
                format: crate::linear_format::standard_linear_format(
                    "model.embed_tokens.weight",
                    encoding,
                )?,
            })
        };
        let vocabulary = if config.use_ordered_embeddings {
            if args.linear_format_for("model.embed_tokens.weight") != LinearFormat::Dense
                || args.linear_format_for("masked_embedding.centroids.weight")
                    != LinearFormat::Dense
            {
                return Err(Error::backend(
                    "ordered assistant vocabulary slots require dense storage",
                ));
            }
            Vocabulary::Ordered(OrderedParameters {
                centroids: linear(
                    "masked_embedding.centroids.weight",
                    args.hidden_size,
                    config.num_centroids,
                    LinearFormat::Dense,
                )?,
                ordering: OrderingSpec {
                    parameter: ParameterSpec::trainable("masked_embedding.token_ordering")
                        .map_err(Error::backend)?,
                    shape: [args.vocab_size],
                },
                table: table(LinearFormat::Dense)?,
            })
        } else if config.tie_word_embeddings {
            Vocabulary::Tied(table(args.linear_format_for("model.embed_tokens.weight"))?)
        } else {
            Vocabulary::Output(linear(
                "lm_head.weight",
                args.hidden_size,
                args.vocab_size,
                format("lm_head.weight"),
            )?)
        };
        let input = config
            .backbone_hidden_size
            .checked_mul(2)
            .ok_or_else(|| Error::backend("assistant input width exceeds i32"))?;
        let parameters = Self {
            pre: linear(
                "pre_projection.weight",
                input,
                args.hidden_size,
                format("pre_projection.weight"),
            )?,
            post: linear(
                "post_projection.weight",
                args.hidden_size,
                config.backbone_hidden_size,
                format("post_projection.weight"),
            )?,
            norm: NormalizationConstructionSpec::learned(
                args.hidden_size,
                args.rms_norm_eps,
                ParameterSpec::trainable("model.norm.weight").map_err(Error::backend)?,
            ),
            vocabulary,
        };
        let declared = parameters.logical_parameters()?;
        eredu_nn::validate_logical_parameter_topology(&declared).map_err(Error::backend_source)?;
        Ok(parameters)
    }
}
impl DeclareLogicalParameters for StaticParameters {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let mut parameters = self.pre.logical_parameters()?;
        parameters.extend(self.post.logical_parameters()?);
        parameters.extend(self.norm.logical_parameters()?);
        parameters.extend(self.vocabulary.logical_parameters()?);
        Ok(parameters)
    }
}
