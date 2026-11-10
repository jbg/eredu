//! Embedded speech/character parameter declarations composed from shared decoder specs.
use super::tts_transformer::TransformerConfig;
use crate::decoder::topology::EmbeddedParameters;
use eredu_nn::{DeclareLogicalParameters, Error, LogicalParameterDeclaration};
/// Complete parameter ownership for a causal speech or bidirectional character stack.
#[derive(Debug, Clone)]
pub struct TransformerSpec {
    pub(crate) parameters: EmbeddedParameters,
}
impl TransformerSpec {
    /// Validates all block and final-normalization identities before allocation.
    pub fn new(config: &TransformerConfig) -> Result<Self, Error> {
        let declaration = Self {
            parameters: EmbeddedParameters::new(config)?,
        };
        config.validate_formats(&declaration)?;
        Ok(declaration)
    }
}
impl DeclareLogicalParameters for TransformerSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        self.parameters.logical_parameters()
    }
}

impl TransformerSpec {
    /// Projects physical format and arithmetic extensions through retained affine identities.
    pub(crate) fn with_policy(&self, config: &TransformerConfig) -> Result<Self, Error> {
        use crate::decoder::Config;
        let projection = |spec: &mut eredu_nn::LinearSpec| -> Result<(), Error> {
            spec.format = crate::linear_format::project_matrix_format(
                &spec.weight,
                &spec.format,
                config.linear_format(spec.weight.id.as_str()),
            )?
            .with_compute_element(config.linear_compute_element())?;
            Ok(())
        };
        let declaration = Self {
            parameters:
                self.parameters.map_blocks(config, |_, block| {
                    let mut block = block.clone();
                    let crate::decoder::topology::AttentionInputParameters::Split {
                        query,
                        key,
                        value,
                    } = &mut block.attention.input
                    else {
                        return Err(Error::backend(
                            "TTS requires independent attention projections",
                        ));
                    };
                    projection(query)?;
                    projection(key)?;
                    if let Some(value) = value {
                        projection(value)?;
                    }
                    if let Some((gate, _)) = &mut block.attention.gate {
                        projection(gate)?;
                    }
                    projection(&mut block.attention.output)?;
                    if let Some(feed) = &mut block.feed_forward {
                        match feed {
                            eredu_nn::GatedProductSpec::Split(spec) => {
                                projection(&mut spec.gate)?;
                                projection(&mut spec.up)?;
                                projection(&mut spec.down)?;
                            }
                            eredu_nn::GatedProductSpec::Fused(_) => {
                                return Err(Error::backend(
                                    "TTS requires independent gated projections",
                                ))
                            }
                        }
                    }
                    Ok(block)
                })?,
        };
        config.validate_formats(&declaration)?;
        Ok(declaration)
    }
    /// Projects one retained stack into a selected head/intermediate domain.
    pub(crate) fn with_geometry(&self, config: &TransformerConfig) -> Result<Self, Error> {
        use crate::decoder::Config;
        Ok(Self {
            parameters: self.parameters.map_blocks(config, |_, block| {
                let old = block.attention.geometry;
                block.with_geometry(
                    eredu_nn::AttentionHeadGeometry::new(
                        config.num_attention_heads(),
                        config.num_key_value_heads(),
                        old.key_dimensions(),
                        old.value_dimensions(),
                    )?,
                    Some(config.intermediate_size()),
                    config.query_key_norm_per_head_weights(),
                )
            })?,
        })
    }
}

/// Retained canonical parameters, normalized execution policy and component boundaries.
/// Floating arithmetic is an execution extension; canonical source slots retain their formats.
#[derive(Debug, Clone)]
pub struct TransformerDeclaration {
    config: TransformerConfig,
    spec: TransformerSpec,
    definition: eredu_runtime::ComponentDefinition,
}
impl TransformerDeclaration {
    /// Declares one causal speech or bidirectional character stack before allocation.
    pub fn new(
        config: TransformerConfig,
    ) -> Result<Self, super::tts_transformer_layered::TransformerError> {
        config
            .precision()
            .validate()
            .map_err(super::tts_transformer_layered::TransformerError::Configuration)?;
        let canonical = config.clone().with_precision(Default::default());
        let spec = TransformerSpec::new(&canonical)?;
        let definition =
            super::tts_transformer_layered::transformer_component_definition_with_state(
                &canonical,
                &spec,
                super::tts_transformer_layered::transformer_state_layout(
                    &config,
                    &spec.parameters.blocks,
                )?,
            )?;
        Ok(Self {
            config,
            spec,
            definition,
        })
    }
    /// Retained global equations, positional policy and selected physical formats.
    pub fn config(&self) -> &TransformerConfig {
        &self.config
    }
    /// Complete source-free ownership and invocation/state projection.
    pub fn definition(&self) -> &eredu_runtime::ComponentDefinition {
        &self.definition
    }
    /// Selects arithmetic without changing canonical source identities, geometry or bindings.
    pub fn with_precision(
        mut self,
        precision: super::precision::SpeechPrecision,
    ) -> Result<Self, super::tts_transformer_layered::TransformerError> {
        precision
            .validate()
            .map_err(super::tts_transformer_layered::TransformerError::Configuration)?;
        self.config = self.config.with_precision(precision);
        Ok(self)
    }
    /// Projects admitted physical companions through the retained affine identities.
    pub fn with_formats(
        mut self,
        formats: std::collections::BTreeMap<String, eredu_checkpoint::LinearFormat>,
    ) -> Result<Self, super::tts_transformer_layered::TransformerError> {
        self.config = self.config.with_formats(formats);
        let canonical = self.config.clone().with_precision(Default::default());
        self.spec = self.spec.with_policy(&canonical)?;
        self.definition =
            super::tts_transformer_layered::transformer_component_definition_with_state(
                &canonical,
                &self.spec,
                self.definition.state_layout().clone(),
            )?;
        Ok(self)
    }
    pub(crate) fn spec(&self) -> &TransformerSpec {
        &self.spec
    }
    pub(crate) fn with_state_layout(
        mut self,
        state: eredu_runtime::StateLayout,
    ) -> Result<Self, super::tts_transformer_layered::TransformerError> {
        self.definition = self.definition.with_state_layout(state)?;
        Ok(self)
    }
    pub(crate) fn into_parts(
        self,
    ) -> (
        TransformerConfig,
        TransformerSpec,
        eredu_runtime::ComponentDefinition,
    ) {
        (self.config, self.spec, self.definition)
    }
}
impl eredu_runtime::ComponentPreparationDeclaration for TransformerDeclaration {
    fn parameters(&self) -> &eredu_runtime::ArchitectureParameterDescription {
        self.definition.parameters()
    }
}
/// Immutable source-free scopes for the validated published VoiceChat profile.
#[derive(Debug)]
pub(crate) struct ReleasedTransformers {
    pub causal: TransformerDeclaration,
    pub character: TransformerDeclaration,
}
pub(crate) fn released_declarations(
) -> Result<&'static ReleasedTransformers, super::tts_transformer_layered::TransformerError> {
    static DECLARATIONS: std::sync::OnceLock<Result<ReleasedTransformers, String>> =
        std::sync::OnceLock::new();
    DECLARATIONS
        .get_or_init(|| {
            let c = super::config::released_configuration().map_err(|e| e.to_string())?;
            let new = |g| {
                TransformerConfig::new(g)
                    .map_err(|e| e.to_string())
                    .and_then(|c| TransformerDeclaration::new(c).map_err(|e| e.to_string()))
            };
            Ok(ReleasedTransformers {
                causal: new(&c.tts.backbone)?,
                character: new(&c.tts.character_encoder)?,
            })
        })
        .as_ref()
        .map_err(|e| Error::backend(e.clone()).into())
}
