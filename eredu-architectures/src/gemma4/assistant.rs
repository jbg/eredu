//! External Gemma 4 assistant equations and forkable draft state.

use std::collections::HashMap;

use eredu_checkpoint::{
    schema::{GgufCheckpointPlan, GgufTensorConstraint, GgufTypeConstraint, TensorOperation},
    LinearFormat, WeightQuantization,
};
use eredu_core::LayerSchedule;
use eredu_gguf::MetadataValue;
use eredu_nn::{
    multimodal::{masked_output_projection, MaskedOutputProjectionInput},
    AttentionCache, AttentionStateSource, EmbeddingOperator, EmbeddingSpec, Error, LinearOperator,
    LinearSpec, NeuralBackend, NormalizationOperator, NormalizationSpec, Parameter, ParameterSpec,
    Parameterized, RotaryPosition, RoutedNeuralBackend, Tensor,
};
use serde::Deserialize;

use super::{BlockInput, DenseBlock, ModelArgs, SharedAttentionStates};

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

#[derive(Debug, Deserialize)]
struct AssistantSource {
    #[serde(default = "default_model_type")]
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

fn default_model_type() -> String {
    "gemma4_assistant".into()
}

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
        if source.model_type != "gemma4_assistant" {
            return Err(AssistantConfigError::Invalid(format!(
                "unsupported Gemma 4 assistant model_type {:?}",
                source.model_type
            )));
        }
        let mut text_value = source.text_config;
        let object = text_value.as_object_mut().ok_or_else(|| {
            AssistantConfigError::Invalid("assistant text_config must be an object".into())
        })?;
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
    pub fn from_gguf_metadata<C: super::GgufTensorCatalog + ?Sized>(
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
            model_type: default_model_type(),
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
}

/// Builds the strict released assistant GGUF catalog.
pub fn assistant_gguf_plan(config: &AssistantConfig) -> Result<GgufCheckpointPlan, String> {
    let mut plan = super::gguf_plan(&config.text_config)?;
    let hidden = usize::try_from(config.text_config.hidden_size)
        .map_err(|_| "Gemma 4 assistant hidden size is invalid".to_string())?;
    let backbone = usize::try_from(config.backbone_hidden_size)
        .map_err(|_| "Gemma 4 assistant target width is invalid".to_string())?;
    let matrix = |name: &str, aliases: &[&str], shape: Vec<usize>| {
        GgufTensorConstraint::required(
            name,
            shape,
            GgufTypeConstraint::OperationClass(TensorOperation::Matrix),
        )
        .with_aliases(aliases.iter().copied())
    };
    plan.common_tensors.extend([
        matrix(
            "mtp.pre_projection.weight",
            &["nextn.pre_projection.weight"],
            vec![hidden, backbone * 2],
        ),
        matrix(
            "mtp.post_projection.weight",
            &["nextn.post_projection.weight"],
            vec![backbone, hidden],
        ),
    ]);
    if config.use_ordered_embeddings {
        let centroids = usize::try_from(config.num_centroids)
            .map_err(|_| "Gemma 4 assistant centroid count is invalid".to_string())?;
        let vocabulary = usize::try_from(config.text_config.vocab_size)
            .map_err(|_| "Gemma 4 assistant vocabulary is invalid".to_string())?;
        plan.common_tensors.extend([
            matrix(
                "mtp.centroids.weight",
                &["nextn.centroids.weight"],
                vec![centroids, hidden],
            ),
            GgufTensorConstraint::required(
                "mtp.token_ordering.weight",
                vec![vocabulary],
                GgufTypeConstraint::OperationClass(TensorOperation::I32),
            )
            .with_aliases(["nextn.token_ordering.weight"]),
        ]);
    }
    GgufCheckpointPlan::new(
        "Gemma 4 assistant GGUF",
        plan.common_tensors,
        plan.layout_groups,
        plan.catalog_policy,
    )
    .map_err(|error| error.to_string())
}

/// Translates one assistant GGUF tensor to its neutral parameter identity.
pub fn translate_assistant_gguf_weight_name(name: &str) -> String {
    if matches!(
        name,
        "mtp.token_ordering.weight" | "nextn.token_ordering.weight"
    ) {
        return "masked_embedding.token_ordering".into();
    }
    for (source, target) in [
        ("mtp.pre_projection", "pre_projection"),
        ("nextn.pre_projection", "pre_projection"),
        ("mtp.post_projection", "post_projection"),
        ("nextn.post_projection", "post_projection"),
        ("mtp.centroids", "masked_embedding.centroids"),
        ("nextn.centroids", "masked_embedding.centroids"),
    ] {
        if name == source || name.starts_with(&format!("{source}.")) {
            return name.replacen(source, target, 1);
        }
    }
    super::translate_gguf_weight_name(name)
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
struct MaskedHead<B: NeuralBackend> {
    centroids: B::Linear,
    token_ordering: Parameter<B::Tensor>,
    output_weight: Parameter<B::Tensor>,
    #[parameter(skip)]
    top_k: i32,
}

impl<B: NeuralBackend> MaskedHead<B> {
    fn new(
        config: &AssistantConfig,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let args = &config.text_config;
        Ok(Self {
            centroids: B::linear(
                LinearSpec {
                    input: args.hidden_size,
                    output: config.num_centroids,
                    weight: ParameterSpec::trainable("masked_embedding.centroids.weight")
                        .map_err(Error::backend)?,
                    bias: None,
                    format: LinearFormat::Dense,
                },
                context,
            )?,
            token_ordering: Parameter::unloaded_i32(
                ParameterSpec::trainable("masked_embedding.token_ordering")
                    .map_err(Error::backend)?,
                &[args.vocab_size],
                context,
            )?,
            output_weight: Parameter::unloaded(
                ParameterSpec::trainable("model.embed_tokens.weight").map_err(Error::backend)?,
                &[args.vocab_size, args.hidden_size],
                context,
            )?,
            top_k: config.centroid_intermediate_top_k,
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
pub struct Assistant<B: RoutedNeuralBackend> {
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

impl<B: RoutedNeuralBackend> Assistant<B> {
    /// Builds an unloaded assistant under released SafeTensors identities.
    pub fn new(
        config: AssistantConfig,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        config.validate().map_err(Error::backend)?;
        let args = &config.text_config;
        let layers = (0..args.num_hidden_layers())
            .map(|layer| DenseBlock::new_at(args, layer, "model.layers", context))
            .collect::<Result<Vec<_>, _>>()?;
        let format = |name: &str| {
            config
                .quantization
                .map(LinearFormat::from)
                .unwrap_or_else(|| args.linear_format_for(name))
        };
        let linear = |name: &str, input, output| {
            B::linear(
                LinearSpec {
                    input,
                    output,
                    weight: ParameterSpec::trainable(name).map_err(Error::backend)?,
                    bias: None,
                    format: format(name),
                },
                context,
            )
        };
        let masked_head = config
            .use_ordered_embeddings
            .then(|| MaskedHead::new(&config, context))
            .transpose()?;
        let tied_embedding = (config.tie_word_embeddings && !config.use_ordered_embeddings)
            .then(|| {
                B::embedding(
                    EmbeddingSpec {
                        vocabulary: args.vocab_size,
                        dimensions: args.hidden_size,
                        weight: ParameterSpec::trainable("model.embed_tokens.weight")
                            .map_err(Error::backend)?,
                        quantization: args
                            .linear_format_for("model.embed_tokens.weight")
                            .weight_quantization(),
                    },
                    context,
                )
            })
            .transpose()?;
        let output_head = (!config.tie_word_embeddings && !config.use_ordered_embeddings)
            .then(|| linear("lm_head.weight", args.hidden_size, args.vocab_size))
            .transpose()?;
        Ok(Self {
            layers,
            final_norm: B::rms_norm(
                NormalizationSpec {
                    dimensions: args.hidden_size,
                    epsilon: args.rms_norm_eps,
                    weight: ParameterSpec::trainable("model.norm.weight")
                        .map_err(Error::backend)?,
                },
                context,
            )?,
            pre_projection: linear(
                "pre_projection.weight",
                2 * config.backbone_hidden_size,
                args.hidden_size,
            )?,
            post_projection: linear(
                "post_projection.weight",
                args.hidden_size,
                config.backbone_hidden_size,
            )?,
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

#[cfg(test)]
mod tests {
    use super::*;

    const CONFIG: &str = r#"{
      "model_type":"gemma4_assistant",
      "backbone_hidden_size":32,
      "use_ordered_embeddings":false,
      "tie_word_embeddings":false,
      "block_size":4,
      "text_config":{
        "model_type":"gemma4_text","hidden_size":32,"num_hidden_layers":1,
        "intermediate_size":64,"num_attention_heads":4,"num_key_value_heads":2,
        "head_dim":8,"rms_norm_eps":0.00001,"vocab_size":32,
        "max_position_embeddings":128,"tie_word_embeddings":false,
        "attention_k_eq_v":false,"layer_types":["full_attention"]
      }
    }"#;

    #[test]
    fn normalizes_every_assistant_layer_to_shared_target_state() {
        let config = AssistantConfig::from_json(CONFIG.as_bytes()).unwrap();
        assert_eq!(config.block_size, 4);
        assert!(config
            .text_config
            .layer_schedule
            .iter()
            .all(|policy| policy.key_value == AttentionStateSource::Shared));
    }

    #[test]
    fn ordered_head_rejects_non_integral_or_quantized_centroids() {
        let mut value: serde_json::Value = serde_json::from_str(CONFIG).unwrap();
        value["use_ordered_embeddings"] = true.into();
        value["num_centroids"] = 3.into();
        assert!(AssistantConfig::from_json(&serde_json::to_vec(&value).unwrap()).is_err());
        value["num_centroids"] = 4.into();
        value["quantization"] = serde_json::json!({"bits":4,"group_size":32});
        assert!(AssistantConfig::from_json(&serde_json::to_vec(&value).unwrap()).is_err());
    }
}
