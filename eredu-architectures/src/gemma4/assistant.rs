//! External Gemma 4 assistant equations and forkable draft state.

use std::collections::HashMap;

use eredu_checkpoint::{
    schema::{
        matrix_for_linear_format, CatalogPolicy, GgufCheckpointPlan, GgufTensorConstraint,
        GgufTypeConstraint, SafetensorsCheckpointPlan, SafetensorsTensorConstraint,
        StoredDtypeConstraint, TensorOperation,
    },
    StoredDtype, WeightQuantization,
};
use eredu_core::{AttentionPolicy, LayerSchedule};
use eredu_gguf::MetadataValue;
use eredu_nn::{
    multimodal::{masked_output_projection, MaskedOutputProjectionInput},
    AttentionCache, AttentionStateSource, DeclareLogicalParameters, EmbeddingOperator, Error,
    GroupedNeuralBackend, LinearOperator, NeuralBackend, NormalizationOperator, Parameter,
    Parameterized, RotaryPosition, Tensor,
};
use serde::Deserialize;

use super::{BlockInput, DenseBlock, ModelArgs, SharedAttentionStates};

mod parameters;
use parameters::{StaticParameters, Vocabulary};

/// Complete typed Gemma assistant construction, shared by cold consumers and materialization.
#[derive(Debug, Clone)]
pub struct AssistantDefinition {
    config: AssistantConfig,
    static_parameters: StaticParameters,
    source_per_layer: Option<super::text::pinned::PerLayerSpec>,
    blocks: Vec<super::text::specs::DecoderBlockSpec>,
    parameters: crate::external_assistant::AssistantParameterDeclarations,
}
impl AssistantDefinition {
    /// Validates and retains every construction scope without native resources.
    pub fn new(config: AssistantConfig) -> Result<Self, Error> {
        let static_parameters = StaticParameters::new(&config)?;
        let source_per_layer = super::text::pinned::PerLayerSpec::at(&config.text_config, "model")?;
        let blocks = (0..config.text_config.num_hidden_layers())
            .map(|layer| {
                super::text::specs::DecoderBlockSpec::from_scopes(
                    &config.text_config,
                    layer,
                    super::text::specs::BlockSpec::at(&config.text_config, layer, "model.layers")?,
                    super::text::specs::DenseSpec::at(&config.text_config, layer, "model.layers")?,
                    super::text::specs::AttentionSpec::at(
                        &config.text_config,
                        layer,
                        "model.layers",
                    )?,
                    (config.text_config.layer_policy(layer).is_some_and(|p| {
                        p.feed_forward == super::FeedForwardPolicy::DenseWithSparseMoe
                    }))
                    .then(|| {
                        super::text::routed::Declaration::at(
                            &config.text_config,
                            layer,
                            "model.layers",
                        )
                    })
                    .transpose()?,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        Self::from_scopes(config, static_parameters, source_per_layer, blocks)
    }
    fn from_scopes(
        config: AssistantConfig,
        static_parameters: StaticParameters,
        source_per_layer: Option<super::text::pinned::PerLayerSpec>,
        blocks: Vec<super::text::specs::DecoderBlockSpec>,
    ) -> Result<Self, Error> {
        use eredu_runtime::ReplicatedTextParameterOwner;
        let mut parameters = crate::external_assistant::AssistantParameterDeclarations::new();
        parameters.append(
            ReplicatedTextParameterOwner::StaticRole("external_assistant".into()),
            &static_parameters,
        )?;
        for (unit, block) in blocks.iter().enumerate() {
            parameters.append(
                ReplicatedTextParameterOwner::ExecutionUnit {
                    group: "assistant.decoder".into(),
                    unit,
                },
                block,
            )?;
        }
        parameters.validate()?;
        let mut declared = parameters.logical_parameters()?;
        if let Some(scope) = &source_per_layer {
            declared.extend(scope.logical_parameters()?);
        }
        eredu_nn::validate_logical_parameter_topology(&declared).map_err(Error::backend_source)?;
        // Alias normalization is a family source rule; the shared resolver owns
        // matrix eligibility, encoding equality and sharing validation.
        let references = config
            .text_config
            .quantized_weight_configs
            .iter()
            .flat_map(|formats| formats.keys())
            .chain(config.text_config.quantized_weights.iter().flatten())
            .map(|name| {
                let canonical = super::config::canonical_weight_name(name);
                let parameter = declared
                    .iter()
                    .find(|p| {
                        super::config::canonical_weight_name(p.parameter.id.as_str()) == canonical
                    })
                    .ok_or_else(|| {
                        Error::backend(format!("undeclared assistant parameter {name}"))
                    })?;
                Ok((
                    parameter.parameter.id.to_string(),
                    Some(config.text_config.linear_format_for(name)),
                ))
            })
            .collect::<Result<Vec<_>, Error>>()?;
        crate::format_references::validate_matrices(
            declared,
            references
                .iter()
                .map(|(name, format)| (name.as_str(), *format)),
        )?;

        Ok(Self {
            config,
            static_parameters,
            source_per_layer,
            blocks,
            parameters,
        })
    }
    /// Applies canonical checkpoint formats to a complete assistant
    /// configuration.
    pub fn with_checkpoint_formats(
        &self,
        formats: HashMap<String, WeightQuantization>,
    ) -> Result<AssistantConfig, AssistantConfigError> {
        let mut formats = formats;
        // Required but unused GGUF vocabulary sources have no executable matrix owner.
        for (_, target, _) in self.gguf_source_vocabulary_extensions() {
            formats.remove(target);
        }
        for routed in self.blocks.iter().filter_map(|block| block.routed.as_ref()) {
            let names =
                super::checkpoint::expert_layout_names(routed, |base| format!("{base}.weight"))
                    .map_err(AssistantConfigError::Invalid)?;
            let gate = formats.remove(&names.separate_gate);
            let up = formats.remove(&names.separate_up);
            if gate.is_some() || up.is_some() {
                let format = match (gate, up) {
                    (Some(gate), Some(up)) if gate == up => gate,
                    _ => {
                        return Err(AssistantConfigError::Invalid(
                            "split assistant expert gate/up encodings differ or are incomplete"
                                .into(),
                        ));
                    }
                };
                let target = super::config::canonical_weight_name(&names.target_gate_up);
                if formats
                    .keys()
                    .any(|name| super::config::canonical_weight_name(name) == target)
                {
                    return Err(AssistantConfigError::Invalid(
                        "assistant fused and split expert encodings conflict".into(),
                    ));
                }
                formats.insert(names.target_gate_up, format);
            }
        }
        let mut declared = self
            .parameters
            .logical_parameters()
            .map_err(|e| AssistantConfigError::Invalid(e.to_string()))?;
        if let Some(scope) = &self.source_per_layer {
            declared.extend(
                scope
                    .logical_parameters()
                    .map_err(|e| AssistantConfigError::Invalid(e.to_string()))?,
            );
        }
        for (name, format) in &formats {
            let canonical = super::config::canonical_weight_name(name);
            let parameter = declared
                .iter()
                .find(|p| {
                    p.linear_format.is_some()
                        && super::config::canonical_weight_name(p.parameter.id.as_str())
                            == canonical
                })
                .ok_or_else(|| {
                    AssistantConfigError::Invalid(format!(
                        "{name} is not a declared assistant matrix"
                    ))
                })?;
            format
                .validate()
                .map_err(|e| AssistantConfigError::Invalid(e.to_string()))?;
            matrix_for_linear_format(
                name,
                std::iter::empty::<String>(),
                parameter.shape.clone(),
                (*format).into(),
                None,
            )
            .map_err(|e| AssistantConfigError::Invalid(e.to_string()))?;
        }
        let mut target = self.config.clone();
        target.text_config.quantized_weight_configs = (!formats.is_empty()).then_some(formats);
        target.validate()?;
        Ok(target)
    }

    /// Selects physical storage on retained construction scopes, preserving all source semantics.
    pub fn project_formats(&self, config: AssistantConfig) -> Result<Self, Error> {
        let neutral_text = |args: &ModelArgs| {
            let mut args = args.clone();
            args.weight_quantization = None;
            args.quantized_weights = None;
            args.quantized_weight_configs = None;
            args
        };
        let source = &self.config;
        if source.model_type != config.model_type
            || source.backbone_hidden_size != config.backbone_hidden_size
            || source.use_ordered_embeddings != config.use_ordered_embeddings
            || source.num_centroids != config.num_centroids
            || source.centroid_intermediate_top_k != config.centroid_intermediate_top_k
            || source.tie_word_embeddings != config.tie_word_embeddings
            || source.block_size != config.block_size
            || neutral_text(&source.text_config) != neutral_text(&config.text_config)
        {
            return Err(Error::backend(
                "assistant encoding projection changes retained geometry, ownership or execution policy",
            ));
        }
        let static_parameters = self.static_parameters.project_formats(&config)?;
        let source_per_layer = self
            .source_per_layer
            .as_ref()
            .map(|spec| spec.project_formats(&config.text_config))
            .transpose()?;
        let blocks = self
            .blocks
            .iter()
            .enumerate()
            .map(|(layer, source)| {
                super::text::specs::DecoderBlockSpec::from_scopes(
                    &config.text_config,
                    layer,
                    source.block.project_formats(&config.text_config)?,
                    source.dense.project_formats(&config.text_config)?,
                    source.attention.project_formats(&config.text_config)?,
                    source
                        .routed
                        .as_ref()
                        .map(|spec| spec.project_formats(&config.text_config))
                        .transpose()?,
                )
            })
            .collect::<Result<Vec<_>, Error>>()?;
        let projected = Self::from_scopes(config, static_parameters, source_per_layer, blocks)?;
        self.parameters.validate_projection(&projected.parameters)?;
        Ok(projected)
    }
    /// Logical facts projected from these exact retained construction scopes.
    pub fn parameter_declarations(
        &self,
    ) -> &crate::external_assistant::AssistantParameterDeclarations {
        &self.parameters
    }
    pub(crate) fn source_recipes(
        &self,
        catalog: &dyn eredu_checkpoint::recipe::RecipeCatalog,
    ) -> Result<
        std::collections::BTreeMap<String, eredu_checkpoint::recipe::DerivedWeightRecipe>,
        String,
    > {
        let mut recipes = std::collections::BTreeMap::new();
        for block in &self.blocks {
            let Some(routed) = &block.routed else {
                continue;
            };
            let names = super::checkpoint::expert_layout_names(routed, |base| {
                catalog
                    .tensor_metadata(base)
                    .or_else(|_| catalog.tensor_metadata(&format!("{base}.weight")))
                    .map(|metadata| metadata.name)
                    .unwrap_or_else(|_| format!("{base}.weight"))
            })?;
            let needs_global_projection = routed
                .routed
                .experts
                .packed_parameters()
                .map_err(|e| e.to_string())?
                .gate_up()
                .linear_format
                .is_some_and(|format| format.encoding() == eredu_checkpoint::LinearFormat::NvFp4);
            if !needs_global_projection
                && catalog.tensor_metadata(&names.target_gate_up).is_ok()
                && catalog.tensor_metadata(&names.target_down).is_ok()
            {
                continue;
            }
            let resolved =
                eredu_checkpoint::expert::resolve_gated_product_expert_recipes(catalog, &names)
                    .map_err(|error| error.to_string())?;
            for (target, recipe) in
                eredu_checkpoint::expert::gated_product_materialization_recipes(catalog, &resolved)?
            {
                if recipes.insert(target.clone(), recipe).is_some() {
                    return Err(format!("duplicate assistant source recipe {target:?}"));
                }
            }
        }
        Ok(recipes)
    }

    pub(crate) fn build_unit<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend>(
        &self,
        ordinal: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<DenseBlock<B>, Error> {
        let spec = self
            .blocks
            .get(ordinal)
            .ok_or_else(|| Error::backend("invalid assistant decoder unit"))?;
        DenseBlock::from_spec(ordinal, spec.clone(), context)
    }
}

/// Invalid Gemma assistant configuration.
#[derive(Debug, thiserror::Error)]
pub enum AssistantConfigError {
    /// JSON decoding failed.
    #[error("invalid Gemma 4 assistant configuration: {0}")]
    Json(#[from] serde_json::Error),
    /// Nested text configuration is invalid.
    #[error(transparent)]
    Text(#[from] super::ConfigError),
    /// Assistant geometry is unsupported.
    #[error("{0}")]
    Invalid(String),
}

/// Why a Gemma 4 assistant cannot share state with a target decoder.
#[derive(Debug, thiserror::Error)]
pub enum AssistantCompatibilityError {
    /// The assistant captures a different target hidden width.
    #[error(
        "Gemma 4 assistant target hidden width {assistant} does not match decoder width {target}"
    )]
    HiddenWidth {
        /// Width captured by the assistant.
        assistant: i32,
        /// Target decoder width.
        target: i32,
    },
    /// Assistant and target token tables have different row counts.
    #[error("Gemma 4 assistant vocabulary {assistant} does not match target vocabulary {target}")]
    Vocabulary {
        /// Assistant token-table rows.
        assistant: i32,
        /// Target token-table rows.
        target: i32,
    },
    /// A mutable or otherwise invalid assistant config has no proposal slot.
    #[error("Gemma 4 assistant block size must include at least one proposal")]
    EmptyProposalBlock,
    /// No target layer publishes the attention-state class an assistant layer consumes.
    #[error(
        "Gemma 4 assistant layer {assistant_layer} requires {attention:?} shared state with no target publisher"
    )]
    MissingPublisher {
        /// Zero-based assistant layer.
        assistant_layer: usize,
        /// Required full/sliding attention-state class.
        attention: AttentionPolicy,
    },
    /// The selected publisher uses different KV heads, head width, or rotary policy.
    #[error(
        "Gemma 4 assistant layer {assistant_layer} shared-KV or rotary geometry does not match target publisher {target_layer}"
    )]
    PublisherGeometry {
        /// Zero-based assistant layer.
        assistant_layer: usize,
        /// Zero-based target publisher layer.
        target_layer: usize,
    },
}

/// Architecture-owned proof that every assistant shared-KV input has a compatible target publisher.
#[derive(Debug, Clone, Eq, PartialEq)]
#[must_use = "the compatibility proof should gate assistant/target composition"]
pub struct AssistantCompatibility {
    target_publisher_layers: Box<[usize]>,
}

impl AssistantCompatibility {
    /// Returns the selected target publisher for each assistant layer in order.
    pub fn target_publisher_layers(&self) -> &[usize] {
        &self.target_publisher_layers
    }
}

#[derive(Debug, Deserialize)]
struct AssistantSource {
    model_type: String,
    backbone_hidden_size: i32,
    #[serde(default)]
    use_ordered_embeddings: bool,
    #[serde(default = "default_num_centroids")]
    num_centroids: i32,
    #[serde(default = "default_centroid_top_k")]
    centroid_intermediate_top_k: i32,
    #[serde(default = "default_true")]
    tie_word_embeddings: bool,
    #[serde(default = "default_block_size")]
    block_size: usize,
    text_config: serde_json::Value,
    #[serde(default)]
    quantization: Option<WeightQuantization>,
}

const ASSISTANT_MODEL_TYPE: &str = "gemma4_assistant";
const ASSISTANT_TEXT_MODEL_TYPE: &str = "gemma4_text";

const fn default_num_centroids() -> i32 {
    2048
}

const fn default_centroid_top_k() -> i32 {
    32
}

const fn default_block_size() -> usize {
    4
}

const fn default_true() -> bool {
    true
}

/// Validated external assistant policy.
#[derive(Debug, Clone)]
pub struct AssistantConfig {
    /// Assistant artifact identity.
    pub model_type: String,
    /// Target decoder hidden width captured for drafting.
    pub backbone_hidden_size: i32,
    /// Whether centroid-selected vocabulary projection is used.
    pub use_ordered_embeddings: bool,
    /// Centroid count.
    pub num_centroids: i32,
    /// Selected centroid count.
    pub centroid_intermediate_top_k: i32,
    /// Whether the ordinary output head shares the assistant embedding table.
    pub tie_word_embeddings: bool,
    /// Maximum speculative block size including its anchor.
    pub block_size: usize,
    /// Normalized assistant decoder geometry.
    pub text_config: ModelArgs,
    /// Uniform assistant weight encoding when present.
    pub quantization: Option<WeightQuantization>,
}

impl AssistantConfig {
    /// Parses and validates a released assistant configuration.
    pub fn from_json(bytes: &[u8]) -> Result<Self, AssistantConfigError> {
        let source: AssistantSource = serde_json::from_slice(bytes)?;
        if source.model_type != ASSISTANT_MODEL_TYPE {
            return Err(AssistantConfigError::Invalid(format!(
                "unsupported Gemma 4 assistant model_type {:?}",
                source.model_type
            )));
        }
        let mut text_value = source.text_config;
        let object = text_value.as_object_mut().ok_or_else(|| {
            AssistantConfigError::Invalid("assistant text_config must be an object".into())
        })?;
        let text_model_type = object
            .get("model_type")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| {
                AssistantConfigError::Invalid(
                    "Gemma 4 assistant text_config is missing string model_type".into(),
                )
            })?;
        if text_model_type != ASSISTANT_TEXT_MODEL_TYPE {
            return Err(AssistantConfigError::Invalid(format!(
                "unsupported Gemma 4 assistant text_config.model_type {text_model_type:?}"
            )));
        }
        object.insert(
            "model_type".into(),
            serde_json::Value::String("gemma4".into()),
        );
        let mut text_config = ModelArgs::from_hf_json(&serde_json::to_vec(&text_value)?)?;
        let policies = text_config
            .layer_schedule
            .iter()
            .copied()
            .map(|mut policy| {
                policy.key_value = AttentionStateSource::Shared;
                policy
            })
            .collect::<Vec<_>>();
        text_config.layer_schedule = LayerSchedule::new(policies.len(), policies)
            .map_err(|error| AssistantConfigError::Invalid(error.to_string()))?;
        text_config.tie_word_embeddings = source.tie_word_embeddings;
        if source.quantization.is_some() {
            text_config.weight_quantization = source.quantization;
            text_config.quantized_weights = None;
            text_config.quantized_weight_configs = None;
        }
        let config = Self {
            model_type: source.model_type,
            backbone_hidden_size: source.backbone_hidden_size,
            use_ordered_embeddings: source.use_ordered_embeddings,
            num_centroids: source.num_centroids,
            centroid_intermediate_top_k: source.centroid_intermediate_top_k,
            tie_word_embeddings: source.tie_word_embeddings,
            block_size: source.block_size,
            text_config,
            quantization: source.quantization,
        };
        config.validate()?;
        Ok(config)
    }

    /// Parses the released assistant GGUF without depending on a backend reader.
    pub fn from_gguf_metadata<C: crate::GgufTensorCatalog + ?Sized>(
        catalog: &C,
        metadata: &HashMap<String, MetadataValue>,
    ) -> Result<Self, AssistantConfigError> {
        let mut text_metadata = metadata.clone();
        for (key, value) in metadata {
            if let Some(suffix) = key.strip_prefix("gemma4-assistant.") {
                text_metadata.insert(format!("gemma4.{suffix}"), value.clone());
            }
        }
        let mut text_config = ModelArgs::from_gguf_metadata(catalog, &text_metadata)?;
        let policies = text_config
            .layer_schedule
            .iter()
            .copied()
            .map(|mut policy| {
                policy.key_value = AttentionStateSource::Shared;
                policy
            })
            .collect::<Vec<_>>();
        text_config.layer_schedule = LayerSchedule::new(policies.len(), policies)
            .map_err(|error| AssistantConfigError::Invalid(error.to_string()))?;
        let integer = |suffix: &str| {
            let key = format!("gemma4-assistant.{suffix}");
            metadata
                .get(&key)
                .and_then(MetadataValue::as_i64)
                .ok_or_else(|| {
                    AssistantConfigError::Invalid(format!(
                        "Gemma 4 assistant GGUF requires integer metadata {key:?}"
                    ))
                })
        };
        let block_size = metadata
            .get("gemma4-assistant.nextn_predict_layers")
            .and_then(MetadataValue::as_i64)
            .unwrap_or(3)
            .checked_add(1)
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| {
                AssistantConfigError::Invalid(
                    "Gemma 4 assistant GGUF draft block size is invalid".into(),
                )
            })?;
        let config = Self {
            model_type: ASSISTANT_MODEL_TYPE.into(),
            backbone_hidden_size: i32::try_from(integer("embedding_length_out")?).map_err(
                |_| AssistantConfigError::Invalid("assistant target width exceeds i32".into()),
            )?,
            use_ordered_embeddings: catalog.contains("nextn.centroids.weight")
                || catalog.contains("mtp.centroids.weight"),
            num_centroids: default_num_centroids(),
            centroid_intermediate_top_k: default_centroid_top_k(),
            tie_word_embeddings: !catalog.contains("output.weight"),
            block_size,
            text_config,
            quantization: None,
        };
        config.validate()?;
        Ok(config)
    }

    /// Validates output masking, drafting, and target-fusion geometry.
    pub fn validate(&self) -> Result<(), AssistantConfigError> {
        if self.backbone_hidden_size <= 0 || self.block_size < 2 {
            return Err(AssistantConfigError::Invalid(
                "Gemma 4 assistant requires positive target width and block size at least two"
                    .into(),
            ));
        }
        if self.use_ordered_embeddings
            && (self.quantization.is_some()
                || self.num_centroids <= 0
                || self.centroid_intermediate_top_k <= 0
                || self.centroid_intermediate_top_k > self.num_centroids
                || self.text_config.vocab_size % self.num_centroids != 0)
        {
            return Err(AssistantConfigError::Invalid(
                "ordered Gemma 4 assistant embeddings require dense weights and integral centroid groups"
                    .into(),
            ));
        }
        if self
            .text_config
            .layer_schedule
            .iter()
            .any(|policy| policy.key_value != AttentionStateSource::Shared)
        {
            return Err(AssistantConfigError::Invalid(
                "Gemma 4 assistant layers must consume shared target KV".into(),
            ));
        }
        Ok(())
    }

    /// Derives the complete assistant configuration for load-time
    /// quantization, replacing any checkpoint-specific text formats.
    pub fn load_time_quantization(
        &self,
        quantization: WeightQuantization,
    ) -> Result<Self, AssistantConfigError> {
        quantization
            .validate()
            .map_err(|error| AssistantConfigError::Invalid(error.to_string()))?;
        if self.use_ordered_embeddings {
            return Err(AssistantConfigError::Invalid(
                "ordered Gemma 4 assistant embeddings cannot be quantized".into(),
            ));
        }
        let mut target = self.clone();
        target.quantization = Some(quantization);
        target.text_config.weight_quantization = Some(quantization);
        target.text_config.quantized_weights = None;
        target.text_config.quantized_weight_configs = None;
        target.validate()?;
        Ok(target)
    }

    /// Proves that this assistant can consume shared state from `target`.
    ///
    /// This relationship is independent of tensor storage and execution backend.
    pub fn prove_compatibility(
        &self,
        target: &ModelArgs,
    ) -> Result<AssistantCompatibility, AssistantCompatibilityError> {
        if self.backbone_hidden_size != target.hidden_size {
            return Err(AssistantCompatibilityError::HiddenWidth {
                assistant: self.backbone_hidden_size,
                target: target.hidden_size,
            });
        }
        if self.text_config.vocab_size != target.vocab_size {
            return Err(AssistantCompatibilityError::Vocabulary {
                assistant: self.text_config.vocab_size,
                target: target.vocab_size,
            });
        }
        if self.block_size < 2 {
            return Err(AssistantCompatibilityError::EmptyProposalBlock);
        }

        let mut target_publisher_layers = Vec::with_capacity(self.text_config.num_hidden_layers());
        for (assistant_layer, draft_policy) in self.text_config.layer_schedule.iter().enumerate() {
            let Some((target_layer, target_policy)) = target
                .layer_schedule
                .iter()
                .enumerate()
                .find(|(_, policy)| {
                    policy.attention == draft_policy.attention && policy.key_value.publishes_state()
                })
            else {
                return Err(AssistantCompatibilityError::MissingPublisher {
                    assistant_layer,
                    attention: draft_policy.attention,
                });
            };
            if draft_policy.num_key_value_heads != target_policy.num_key_value_heads
                || draft_policy.head_dim != target_policy.head_dim
                || self
                    .text_config
                    .rope_theta_for(draft_policy.attention)
                    .to_bits()
                    != target.rope_theta_for(target_policy.attention).to_bits()
            {
                return Err(AssistantCompatibilityError::PublisherGeometry {
                    assistant_layer,
                    target_layer,
                });
            }
            target_publisher_layers.push(target_layer);
        }
        Ok(AssistantCompatibility {
            target_publisher_layers: target_publisher_layers.into_boxed_slice(),
        })
    }
}

/// Builds the strict released assistant SafeTensors catalog.
pub fn assistant_safetensors_plan(
    definition: &AssistantDefinition,
) -> Result<SafetensorsCheckpointPlan, String> {
    let mut common = Vec::new();
    let mut groups = Vec::new();
    let mut static_slots = definition
        .static_parameters
        .logical_parameters()
        .map_err(|e| e.to_string())?;
    if let Some(scope) = &definition.source_per_layer {
        static_slots.extend(scope.logical_parameters().map_err(|e| e.to_string())?);
    }
    for parameter in static_slots {
        let name = parameter.parameter.id.as_str();
        let dense_centroids = matches!(&definition.static_parameters.vocabulary,
            Vocabulary::Ordered(spec) if spec.centroids.weight.id == parameter.parameter.id);
        if dense_centroids {
            common.push(SafetensorsTensorConstraint::required(
                name,
                parameter.shape,
                StoredDtypeConstraint::Floating,
            ));
        } else if let Some(format) = parameter.linear_format {
            common.extend(
                matrix_for_linear_format(
                    name,
                    std::iter::empty::<String>(),
                    parameter.shape,
                    format.encoding(),
                    None,
                )
                .map_err(|e| e.to_string())?,
            );
        } else {
            let dtype = if parameter.role == eredu_nn::LogicalParameterRole::TokenMapping {
                StoredDtypeConstraint::Exact(StoredDtype::I32)
            } else {
                StoredDtypeConstraint::Floating
            };
            common.push(SafetensorsTensorConstraint::required(
                name,
                parameter.shape,
                dtype,
            ));
        }
    }
    for (layer, block) in definition.blocks.iter().enumerate() {
        super::checkpoint::add_safetensors_block(
            layer,
            block,
            &format!("model.language_model.layers.{layer}"),
            &mut common,
            &mut groups,
        )?;
    }
    // Ordinary assistant slots use canonical names; bank alternatives retain
    // the released and canonical source spellings declared by the format extension.
    let canonical = common
        .iter()
        .filter_map(|tensor| {
            tensor
                .aliases
                .first()
                .map(|alias| (tensor.key.clone(), alias.clone()))
        })
        .collect::<std::collections::BTreeMap<_, _>>();
    for tensor in &mut common {
        if let Some(target) = canonical.get(&tensor.key) {
            tensor.key = target.clone();
        }
        if let Some(companion) = &mut tensor.linear_companion {
            if let Some(target) = canonical.get(&companion.primary) {
                companion.primary = target.clone();
            }
        }
        tensor.aliases.clear();
    }
    for group in &mut groups {
        for variant in &mut group.variants {
            for tensor in &mut variant.tensors {
                super::checkpoint::add_mlx_vlm_aliases(tensor);
            }
        }
    }
    SafetensorsCheckpointPlan::new(
        "Gemma 4 assistant SafeTensors",
        common,
        groups,
        CatalogPolicy::strict(),
    )
    .map_err(|e| e.to_string())
}

/// Builds the strict released assistant GGUF catalog.
pub fn assistant_gguf_plan(definition: &AssistantDefinition) -> Result<GgufCheckpointPlan, String> {
    let mut common = Vec::new();
    for (parameter, source) in definition.gguf_slots()? {
        let operation = if parameter.role == eredu_nn::LogicalParameterRole::TokenMapping {
            TensorOperation::I32
        } else if parameter.linear_format.is_some() {
            TensorOperation::Matrix
        } else {
            TensorOperation::Vector
        };
        let alias = source
            .strip_prefix("mtp.")
            .map(|local| format!("nextn.{local}"));
        let mut tensor = GgufTensorConstraint::required(
            source,
            parameter.shape,
            GgufTypeConstraint::OperationClass(operation),
        );
        if let Some(alias) = alias {
            tensor = tensor.with_aliases([alias]);
        }
        common.push(tensor);
    }
    for (source, _, shape) in definition.gguf_source_vocabulary_extensions() {
        common.push(GgufTensorConstraint::required(
            source,
            shape,
            GgufTypeConstraint::OperationClass(TensorOperation::Matrix),
        ));
    }
    let mut groups = Vec::new();
    let mut catalog = CatalogPolicy::strict();
    catalog.allowed_prefixes.push("rope_freqs.".into());
    super::checkpoint::add_gguf_blocks(&definition.blocks, &mut common, &mut groups, &mut catalog)?;
    GgufCheckpointPlan::new("Gemma 4 assistant GGUF", common, groups, catalog)
        .map_err(|e| e.to_string())
}

impl AssistantDefinition {
    fn gguf_slots(
        &self,
    ) -> Result<Vec<(eredu_nn::LogicalParameterDeclaration<'_>, String)>, String> {
        let declared = &self.static_parameters;
        let mut slots = Vec::new();
        for (spec, source) in [
            (&declared.pre, "mtp.pre_projection.weight"),
            (&declared.post, "mtp.post_projection.weight"),
        ] {
            slots.extend(
                spec.logical_parameters()
                    .map_err(|e| e.to_string())?
                    .into_iter()
                    .map(|p| (p, source.to_owned())),
            );
        }
        slots.extend(
            declared
                .norm
                .logical_parameters()
                .map_err(|e| e.to_string())?
                .into_iter()
                .map(|p| (p, "output_norm.weight".into())),
        );
        let vocabulary = declared
            .vocabulary
            .logical_parameters()
            .map_err(|e| e.to_string())?;
        for parameter in vocabulary {
            let source = match &declared.vocabulary {
                Vocabulary::Tied(_) => "token_embd.weight",
                Vocabulary::Output(_) => "output.weight",
                Vocabulary::Ordered(ordered) => {
                    if parameter.parameter.id == ordered.ordering.parameter.id {
                        "mtp.token_ordering.weight"
                    } else if parameter.parameter.id == ordered.centroids.weight.id {
                        "mtp.centroids.weight"
                    } else {
                        "token_embd.weight"
                    }
                }
            };
            slots.push((parameter, source.into()));
        }
        if let Some(scope) = &self.source_per_layer {
            slots.extend(super::checkpoint::per_layer_gguf_slots(scope)?);
        }
        Ok(slots)
    }
    /// Released GGUF carries vocabulary source tables even when the assistant
    /// equation reads only its independent output or centroid-selected table.
    /// These source-only requirements reference the retained vocabulary geometry.
    fn gguf_source_vocabulary_extensions(&self) -> Vec<(&'static str, &'static str, Vec<usize>)> {
        match &self.static_parameters.vocabulary {
            Vocabulary::Output(spec) => vec![(
                "token_embd.weight",
                "model.embed_tokens.weight",
                vec![spec.output as usize, spec.input as usize],
            )],
            Vocabulary::Ordered(spec) if !self.config.text_config.tie_word_embeddings => vec![(
                "output.weight",
                "lm_head.weight",
                vec![
                    spec.table.vocabulary as usize,
                    spec.table.dimensions as usize,
                ],
            )],
            _ => Vec::new(),
        }
    }
    /// Source aliases resolve against the same scopes as admission and materialization.
    pub fn gguf_names(&self) -> Result<super::TextGgufNames, String> {
        let mut bindings = Vec::new();
        for (parameter, source) in self.gguf_slots()? {
            let target = parameter.parameter.id.to_string();
            if let Some(local) = source.strip_prefix("mtp.") {
                bindings.push((
                    format!("nextn.{local}"),
                    target.clone(),
                    parameter.linear_format.is_some(),
                ));
            }
            // Token ordering has an integer source suffix but no matrix companions.
            bindings.push((source, target, parameter.linear_format.is_some()));
        }
        for (source, target, _) in self.gguf_source_vocabulary_extensions() {
            bindings.push((source.into(), target.into(), true));
        }
        super::TextGgufNames::from_scopes(bindings, &self.blocks)
    }
}

/// Forkable assistant progress for one speculative branch.
#[derive(Debug, Clone)]
pub struct AssistantState<T> {
    /// Shared target K/V captures keyed by attention policy.
    pub shared_kv: SharedAttentionStates<T>,
    /// Committed target cache length.
    pub kv_offset: i32,
    /// Target-width hidden capture from the prior position.
    pub hidden: T,
}

/// Output from one assistant proposal step.
#[derive(Debug, Clone)]
pub struct AssistantOutput<T> {
    /// Target-width hidden state passed to the next proposal.
    pub hidden: T,
    /// Vocabulary logits for the proposed token.
    pub logits: T,
}

#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
struct MaskedHead<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> {
    centroids: B::Linear,
    token_ordering: Parameter<B::Tensor>,
    output_weight: Parameter<B::Tensor>,
    #[parameter(skip)]
    top_k: i32,
}

impl<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> MaskedHead<B> {
    fn new(
        spec: &parameters::OrderedParameters,
        top_k: i32,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Ok(Self {
            centroids: B::linear(spec.centroids.clone(), context)?,
            token_ordering: Parameter::unloaded_i32(
                spec.ordering.parameter.clone(),
                &spec.ordering.shape,
                context,
            )?,
            output_weight: Parameter::unloaded(
                spec.table.weight.clone(),
                &[spec.table.vocabulary, spec.table.dimensions],
                context,
            )?,
            top_k,
        })
    }

    fn forward(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let centroid_logits = self.centroids.forward(hidden, context)?;
        masked_output_projection(
            MaskedOutputProjectionInput {
                hidden,
                output_weight: self.output_weight.as_ref(),
                centroid_logits: &centroid_logits,
                token_ordering: self.token_ordering.as_ref(),
                top_centroids: self.top_k,
                mask_margin: 1.0,
            },
            context,
        )
    }
}

/// Neutral external Gemma assistant, built from ordinary shared-KV blocks.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct Assistant<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> {
    #[parameter(skip)]
    config: AssistantConfig,
    layers: Vec<DenseBlock<B>>,
    final_norm: B::Normalization,
    pre_projection: B::Linear,
    post_projection: B::Linear,
    tied_embedding: Option<B::Embedding>,
    output_head: Option<B::Linear>,
    masked_head: Option<MaskedHead<B>>,
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> Assistant<B> {
    /// Builds an unloaded assistant under released SafeTensors identities.
    pub fn new(
        definition: &AssistantDefinition,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let config = definition.config.clone();
        let declared = definition.static_parameters.clone();
        let layers = (0..definition.blocks.len())
            .map(|layer| definition.build_unit::<B>(layer, context))
            .collect::<Result<Vec<_>, _>>()?;
        let (tied_embedding, output_head, masked_head) = match &declared.vocabulary {
            Vocabulary::Tied(spec) => (Some(B::embedding(spec.clone(), context)?), None, None),
            Vocabulary::Output(spec) => (None, Some(B::linear(spec.clone(), context)?), None),
            Vocabulary::Ordered(spec) => (
                None,
                None,
                Some(MaskedHead::new(
                    spec,
                    config.centroid_intermediate_top_k,
                    context,
                )?),
            ),
        };
        Ok(Self {
            layers,
            final_norm: B::normalization(declared.norm, context)?,
            pre_projection: B::linear(declared.pre, context)?,
            post_projection: B::linear(declared.post, context)?,
            tied_embedding,
            output_head,
            masked_head,
            config,
        })
    }

    /// Maximum number of proposals after the anchor token.
    pub fn max_proposals(&self) -> usize {
        self.config.block_size.saturating_sub(1)
    }

    /// Starts one proposal branch from committed target captures.
    pub fn begin_round(
        &self,
        shared_kv: SharedAttentionStates<B::Tensor>,
        kv_offset: i32,
        hidden: B::Tensor,
    ) -> AssistantState<B::Tensor> {
        AssistantState {
            shared_kv,
            kv_offset,
            hidden,
        }
    }

    /// Produces one proposal distribution and advances fork-local hidden state.
    pub fn draft_step<C: AttentionCache<B::Tensor>>(
        &mut self,
        scaled_target_token_embedding: &B::Tensor,
        state: &mut AssistantState<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let input = B::Tensor::concatenate(
            &[scaled_target_token_embedding.clone(), state.hidden.clone()],
            -1,
            context,
        )?;
        let output = self.forward::<C>(&input, state, context)?;
        state.hidden = output.hidden;
        state.kv_offset = state.kv_offset.saturating_add(1);
        Ok(output.logits)
    }

    fn forward<C: AttentionCache<B::Tensor>>(
        &mut self,
        input: &B::Tensor,
        state: &mut AssistantState<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<AssistantOutput<B::Tensor>, Error> {
        let mut hidden = self.pre_projection.forward(input, context)?;
        let query_offset = state.kv_offset.saturating_sub(1);
        for (layer_index, layer) in self.layers.iter_mut().enumerate() {
            let policy = self
                .config
                .text_config
                .layer_policy(layer_index)
                .ok_or_else(|| Error::backend("missing assistant layer policy"))?;
            let key_length = state
                .shared_kv
                .get(&policy.attention)
                .ok_or_else(|| Error::backend("missing shared K/V state for assistant layer"))?
                .0
                .dim(2);
            let mask = drafter_mask::<B::Tensor>(
                policy.attention,
                hidden.dim(1),
                query_offset,
                key_length,
                context,
            )?;
            hidden = layer.forward::<C>(
                BlockInput {
                    hidden: &hidden,
                    mask: mask.as_ref(),
                    cache: None,
                    shared: &mut state.shared_kv,
                    per_layer_input: None,
                    rotary_position: Some(RotaryPosition::Offset(query_offset)),
                },
                context,
            )?;
        }
        let hidden = self.final_norm.forward(&hidden, context)?;
        let next_hidden = self.post_projection.forward(&hidden, context)?;
        let logits = if let Some(masked) = self.masked_head.as_mut() {
            masked.forward(&hidden, context)?
        } else if let Some(head) = self.output_head.as_mut() {
            head.forward(&hidden, context)?
        } else {
            self.tied_embedding
                .as_mut()
                .ok_or_else(|| Error::backend("assistant has no output head"))?
                .as_linear(&hidden, context)?
        };
        Ok(AssistantOutput {
            hidden: next_hidden,
            logits,
        })
    }
}

fn drafter_mask<T: Tensor>(
    policy: eredu_core::AttentionPolicy,
    query_length: i32,
    query_offset: i32,
    key_length: i32,
    context: &T::Context,
) -> Result<Option<T>, Error> {
    let Some(window) = policy.window().map(|window| window.get() as i32) else {
        return Ok(None);
    };
    if key_length <= window && query_offset + query_length <= key_length + window {
        return Ok(None);
    }
    let mut values = Vec::with_capacity((query_length * key_length) as usize);
    for query in query_offset..query_offset + query_length {
        for key in 0..key_length {
            let distance = query - key;
            values.push(if distance > -window && distance < window {
                0.0
            } else {
                f32::NEG_INFINITY
            });
        }
    }
    Ok(Some(T::from_f32_slice(
        &values,
        &[1, 1, query_length, key_length],
        context,
    )?))
}
