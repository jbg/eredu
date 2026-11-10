//! Retained vocabulary and normalization ownership shared by cold and native consumers.
use super::super::TextArgs;
use crate::decoder::StaticModuleConstruction;
use eredu_nn::{
    DeclareLogicalParameters, EmbeddingSpec, Error, LinearSpec, LogicalParameterDeclaration,
    NormalizationConstructionSpec, ParameterSpec,
};

#[derive(Debug, Clone)]
pub(crate) struct PinnedParameters {
    pub ordinary: StaticModuleConstruction,
    pub embedding_norm: NormalizationConstructionSpec,
}
impl PinnedParameters {
    pub(in crate::inkling) fn new(args: &TextArgs) -> Result<Self, Error> {
        let parameter = |name| ParameterSpec::trainable(name).map_err(Error::backend);
        let format =
            |name| crate::linear_format::standard_linear_format(name, args.linear_format_for(name));
        let parameters = Self {
            ordinary: StaticModuleConstruction::new(
                EmbeddingSpec {
                    vocabulary: args.vocab_size,
                    dimensions: args.hidden_size,
                    weight: parameter("model.embed_tokens.weight")?,
                    format: format("model.embed_tokens.weight")?,
                },
                Some(LinearSpec {
                    input: args.hidden_size,
                    output: args.vocab_size,
                    weight: parameter("lm_head.weight")?,
                    bias: None,
                    format: format("lm_head.weight")?,
                }),
                NormalizationConstructionSpec::learned(
                    args.hidden_size,
                    args.rms_norm_eps,
                    parameter("model.norm.weight")?,
                ),
            )?,
            embedding_norm: NormalizationConstructionSpec::learned(
                args.hidden_size,
                args.rms_norm_eps,
                parameter("model.embed_norm.weight")?,
            ),
        };
        parameters.validate()?;
        Ok(parameters)
    }
    fn validate(&self) -> Result<(), Error> {
        let parameters = self.logical_parameters()?;
        eredu_nn::validate_logical_parameter_topology(&parameters)
            .map_err(Error::backend_source)?;
        Ok(())
    }
    pub(in crate::inkling) fn project_formats(&mut self, args: &TextArgs) -> Result<(), Error> {
        let mut embedding = self.ordinary.embedding_spec().clone();
        let mut head = self.ordinary.head_spec().cloned();
        let project = |name: &str, previous: &eredu_nn::LinearFormatSpec| -> Result<_, Error> {
            let mut format =
                crate::linear_format::standard_linear_format(name, args.linear_format_for(name))?
                    .with_arithmetic(previous.arithmetic())?;
            if previous.exact_zero_inputs() {
                format = format.with_exact_zero_inputs();
            }
            Ok(format)
        };
        embedding.format = project(embedding.weight.id.as_str(), &embedding.format)?;
        if let Some(head) = &mut head {
            head.format = project(head.weight.id.as_str(), &head.format)?;
        }
        self.ordinary = StaticModuleConstruction::new(
            embedding,
            head,
            self.ordinary.normalization_spec().clone(),
        )?;
        self.validate()
    }
}
impl DeclareLogicalParameters for PinnedParameters {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let mut parameters = self.ordinary.logical_parameters()?;
        parameters.extend(self.embedding_norm.logical_parameters()?);
        Ok(parameters)
    }
}
