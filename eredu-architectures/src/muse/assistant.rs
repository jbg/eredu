//! Backend-neutral lossless DFlash assistant equations and committed context state.

/// Released Spark assistant identities, compatibility, and checkpoint contracts.
pub mod spark;

/// Proposal-local dynamic convolution shared by DFlash assistant bodies.
pub mod block_convolution;

/// Conditional candidate-transition scoring for DFlash2.
pub mod candidate_selector;

/// Causal dense Muse assistant body over shared decoder blocks.
pub mod dspark;

/// Shared retained-source construction and execution dispatch for Muse assistants.
pub mod integration;

/// Shared completion-safe parameter-unit traversal for assistant neural bodies.
pub mod body;

mod heads;

/// Shared body/head parameter topology and construction.
pub mod execution;

/// Which proposal slots participate in sampling and verification.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum ProposalLayout {
    /// Every block position predicts, including the position seeded by the anchor.
    SampleFromAnchor,
    /// The first position is the verified anchor; subsequent positions predict.
    AnchorThenMasks,
}

mod decoder_config;
pub use decoder_config::{
    DFlashContextProjectionLayout, DFlashDecoderConfig, DFlashDecoderEquation, DFlashSublayerPolicy,
};

use std::collections::HashMap;

use eredu_checkpoint::{
    schema::{
        matrix_for_linear_format, CatalogPolicy, GgufCheckpointPlan, GgufTensorConstraint,
        GgufTypeConstraint, SafetensorsCheckpointPlan, SafetensorsTensorConstraint,
        StoredDtypeConstraint, TensorOperation,
    },
    WeightQuantization,
};
use eredu_gguf::{MetadataArray, MetadataValue};
use eredu_nn::{
    Error, Index, LinearOperator, LinearSpec, NeuralBackend,
    NormalizationOperator, ParameterSpec, Parameterized, RotaryOperator, RotaryPosition,
    RotarySpec, Tensor,
};
use serde::Deserialize;
use serde_json::Value;

/// Invalid or unsupported DFlash assistant configuration.
#[derive(Debug, thiserror::Error)]
pub enum DFlashConfigError {
    /// JSON decoding failed.
    #[error("invalid Muse-Glimmer assistant configuration: {0}")]
    Json(#[from] serde_json::Error),
    /// Released geometry or policy is not exact.
    #[error("{0}")]
    Invalid(String),
}

/// Why a Muse-Glimmer DFlash assistant cannot be paired with a target decoder.
#[derive(Debug, thiserror::Error)]
pub enum DFlashCompatibilityError {
    /// The target-state encoder and decoder hidden widths differ.
    #[error("Muse-Glimmer DFlash hidden width {assistant} does not match target width {target}")]
    HiddenWidth {
        /// Assistant encoder width.
        assistant: i32,
        /// Target decoder width.
        target: i32,
    },
    /// A requested target state does not exist.
    #[error("Muse-Glimmer DFlash target layer {layer} is outside decoder depth {target_layers}")]
    TargetLayer {
        /// Zero-based target state requested by the assistant.
        layer: usize,
        /// Number of target decoder layers.
        target_layers: usize,
    },
    /// The assistant mask token is not representable by the target token table.
    #[error(
        "Muse-Glimmer DFlash mask token {mask_token_id} is outside target vocabulary {vocabulary}"
    )]
    Vocabulary {
        /// Assistant mask token.
        mask_token_id: u32,
        /// Target vocabulary row count.
        vocabulary: i32,
    },
    /// The public DFlash transaction requires anchor plus fifteen mask positions.
    #[error("Muse-Glimmer DFlash block size must be 16, found {0}")]
    BlockSize(usize),
}

/// Architecture-owned proof of DFlash target-state and vocabulary compatibility.
#[derive(Debug, Clone, Eq, PartialEq)]
#[must_use = "the compatibility proof should gate assistant/target composition"]
pub struct DFlashCompatibility {
    target_layer_ids: Box<[usize]>,
    target_vocabulary: u32,
}

impl DFlashCompatibility {
    /// Returns the validated zero-based target states consumed by DFlash.
    pub fn target_layer_ids(&self) -> &[usize] {
        &self.target_layer_ids
    }

    /// Returns the validated target token-table row count.
    pub const fn target_vocabulary(&self) -> u32 {
        self.target_vocabulary
    }
}

/// Canonical DFlash checkpoint geometry.
#[derive(Debug, Clone)]
pub struct DFlashConfig {
    /// Assistant artifact identity.
    pub model_type: String,
    /// Assistant hidden width.
    pub hidden_size: i32,
    /// SwiGLU intermediate width.
    pub intermediate_size: i32,
    /// Assistant block count.
    pub num_hidden_layers: i32,
    /// Query head count.
    pub num_attention_heads: i32,
    /// Key/value head count.
    pub num_key_value_heads: i32,
    /// Per-head width.
    pub head_dim: i32,
    /// RMS normalization epsilon.
    pub rms_norm_eps: f32,
    /// Rotary base.
    pub rope_theta: f32,
    /// Declared maximum positions.
    pub max_position_embeddings: i32,
    /// Accepted-context attention window.
    pub sliding_window: i32,
    /// Anchor-plus-mask proposal width.
    pub block_size: usize,
    /// Target vocabulary mask token.
    pub mask_token_id: u32,
    /// Zero-based post-block target states consumed by the encoder.
    pub target_layer_ids: Vec<usize>,
    /// Uniform assistant quantization when present.
    pub quantization: Option<WeightQuantization>,
    /// Per-weight mixed quantization overrides.
    pub quantized_weights: HashMap<String, WeightQuantization>,
}

#[derive(Debug, Deserialize)]
struct HfConfig {
    model_type: String,
    hidden_size: i32,
    intermediate_size: i32,
    num_hidden_layers: i32,
    num_attention_heads: i32,
    num_key_value_heads: i32,
    head_dim: i32,
    rms_norm_eps: f32,
    max_position_embeddings: i32,
    sliding_window: i32,
    block_size: usize,
    mask_token_id: u32,
    target_layer_ids: Vec<usize>,
    layer_types: Vec<String>,
    hidden_act: String,
    attention_dropout: f32,
    rope_parameters: HashMap<String, Value>,
    #[serde(default)]
    quantization: Option<WeightQuantization>,
}

impl DFlashConfig {
    /// Parses and strictly validates the released Hugging Face assistant config.
    pub fn from_hf_json(bytes: &[u8]) -> Result<Self, DFlashConfigError> {
        let source: HfConfig = serde_json::from_slice(bytes)?;
        let rope_theta = source
            .rope_parameters
            .get("rope_theta")
            .and_then(Value::as_f64)
            .map(|value| value as f32)
            .ok_or_else(|| DFlashConfigError::Invalid("DFlash rope_theta is missing".into()))?;
        let config = Self {
            model_type: source.model_type,
            hidden_size: source.hidden_size,
            intermediate_size: source.intermediate_size,
            num_hidden_layers: source.num_hidden_layers,
            num_attention_heads: source.num_attention_heads,
            num_key_value_heads: source.num_key_value_heads,
            head_dim: source.head_dim,
            rms_norm_eps: source.rms_norm_eps,
            rope_theta,
            max_position_embeddings: source.max_position_embeddings,
            sliding_window: source.sliding_window,
            block_size: source.block_size,
            mask_token_id: source.mask_token_id,
            target_layer_ids: source.target_layer_ids,
            quantization: source.quantization,
            quantized_weights: HashMap::new(),
        };
        if source.layer_types != vec!["sliding_attention"; 5]
            || source.hidden_act != "silu"
            || source.attention_dropout != 0.0
        {
            return Err(DFlashConfigError::Invalid(
                "DFlash layer, activation, or dropout policy is unsupported".into(),
            ));
        }
        config.validate_released()?;
        Ok(config)
    }

    /// Parses and validates the released DFlash GGUF metadata.
    pub fn from_gguf_metadata(
        metadata: &HashMap<String, MetadataValue>,
    ) -> Result<Self, DFlashConfigError> {
        let integer = |key: &str| {
            metadata
                .get(key)
                .and_then(MetadataValue::as_i64)
                .ok_or_else(|| {
                    DFlashConfigError::Invalid(format!(
                        "DFlash GGUF requires integer metadata {key:?}"
                    ))
                })
        };
        let float = |key: &str| {
            metadata
                .get(key)
                .and_then(MetadataValue::as_f32)
                .ok_or_else(|| {
                    DFlashConfigError::Invalid(format!(
                        "DFlash GGUF requires float metadata {key:?}"
                    ))
                })
        };
        if metadata
            .get("general.architecture")
            .and_then(MetadataValue::as_str)
            != Some("dflash")
        {
            return Err(DFlashConfigError::Invalid(
                "Muse-Glimmer assistant GGUF requires general.architecture=dflash".into(),
            ));
        }
        let target_layers = match metadata.get("dflash.target_layers") {
            Some(MetadataValue::Array(MetadataArray::Int32(values))) => values,
            _ => {
                return Err(DFlashConfigError::Invalid(
                    "DFlash GGUF requires Int32 dflash.target_layers".into(),
                ))
            }
        };
        let target_layer_ids = target_layers
            .iter()
            .map(|value| {
                usize::try_from(*value)
                    .ok()
                    .and_then(|value| value.checked_sub(1))
            })
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| DFlashConfigError::Invalid("invalid DFlash target layer ids".into()))?;
        let i32_value = |key: &str| {
            i32::try_from(integer(key)?)
                .map_err(|_| DFlashConfigError::Invalid(format!("{key} exceeds i32")))
        };
        let config = Self {
            model_type: "muse_glimmer_assistant".into(),
            hidden_size: i32_value("dflash.embedding_length")?,
            intermediate_size: i32_value("dflash.feed_forward_length")?,
            num_hidden_layers: i32_value("dflash.block_count")?,
            num_attention_heads: i32_value("dflash.attention.head_count")?,
            num_key_value_heads: i32_value("dflash.attention.head_count_kv")?,
            head_dim: i32_value("dflash.attention.key_length")?,
            rms_norm_eps: float("dflash.attention.layer_norm_rms_epsilon")?,
            rope_theta: float("dflash.rope.freq_base")?,
            max_position_embeddings: i32_value("dflash.context_length")?,
            sliding_window: i32_value("dflash.attention.sliding_window")?,
            block_size: usize::try_from(integer("dflash.block_size")?)
                .map_err(|_| DFlashConfigError::Invalid("DFlash block size is invalid".into()))?,
            mask_token_id: 201818,
            target_layer_ids,
            quantization: None,
            quantized_weights: HashMap::new(),
        };
        config.validate_released()?;
        Ok(config)
    }

    /// Validates exact lossless assistant/target compatibility.
    pub fn validate_released(&self) -> Result<(), DFlashConfigError> {
        if self.model_type != "muse_glimmer_assistant"
            || self.hidden_size != 6656
            || self.intermediate_size != 19968
            || self.num_hidden_layers != 5
            || self.num_attention_heads != 32
            || self.num_key_value_heads != 8
            || self.head_dim != 128
            || self.block_size != 16
            || self.target_layer_ids != [1, 13, 25, 37, 49]
            || self.sliding_window != 2048
            || self.max_position_embeddings != 131072
            || self.rope_theta != 500_000.0
            || !self.rms_norm_eps.is_finite()
            || self.rms_norm_eps <= 0.0
        {
            return Err(DFlashConfigError::Invalid(
                "assistant does not match released Muse-Glimmer DFlash geometry".into(),
            ));
        }
        Ok(())
    }

    /// Derives the complete assistant configuration for load-time
    /// quantization, replacing any checkpoint-specific formats.
    pub fn load_time_quantization(
        &self,
        quantization: WeightQuantization,
    ) -> Result<Self, DFlashConfigError> {
        quantization
            .validate()
            .map_err(|error| DFlashConfigError::Invalid(error.to_string()))?;
        let mut target = self.clone();
        target.quantization = Some(quantization);
        target.quantized_weights.clear();
        target.validate_released()?;
        Ok(target)
    }

    /// Applies canonical checkpoint formats to a complete DFlash
    /// configuration.
    pub fn with_checkpoint_formats(
        &self,
        formats: HashMap<String, WeightQuantization>,
    ) -> Result<Self, DFlashConfigError> {
        let mut target = self.clone();
        target.quantized_weights = formats;
        target.validate_released()?;
        Ok(target)
    }

    /// Proves that the target exposes every state and token required by DFlash.
    ///
    /// This relationship is independent of tensor storage and execution backend.
    pub fn prove_compatibility(
        &self,
        target: &super::DecoderConfig,
    ) -> Result<DFlashCompatibility, DFlashCompatibilityError> {
        if self.hidden_size != target.hidden_size {
            return Err(DFlashCompatibilityError::HiddenWidth {
                assistant: self.hidden_size,
                target: target.hidden_size,
            });
        }
        let target_layers = usize::try_from(target.num_hidden_layers).unwrap_or(0);
        if let Some(layer) = self
            .target_layer_ids
            .iter()
            .copied()
            .find(|layer| *layer >= target_layers)
        {
            return Err(DFlashCompatibilityError::TargetLayer {
                layer,
                target_layers,
            });
        }
        let target_vocabulary = u32::try_from(target.vocab_size).unwrap_or(0);
        if self.mask_token_id >= target_vocabulary {
            return Err(DFlashCompatibilityError::Vocabulary {
                mask_token_id: self.mask_token_id,
                vocabulary: target.vocab_size,
            });
        }
        if self.block_size != 16 {
            return Err(DFlashCompatibilityError::BlockSize(self.block_size));
        }
        Ok(DFlashCompatibility {
            target_layer_ids: self.target_layer_ids.clone().into_boxed_slice(),
            target_vocabulary,
        })
    }

    fn linear_format_for(&self, name: &str) -> eredu_checkpoint::LinearFormat {
        self.quantized_weights
            .get(name)
            .copied()
            .or(self.quantization)
            .map(Into::into)
            .unwrap_or(eredu_checkpoint::LinearFormat::Dense)
    }
}

/// Builds the strict released DFlash SafeTensors catalog.
pub fn dflash_safetensors_plan(config: &DFlashConfig) -> Result<SafetensorsCheckpointPlan, String> {
    fn add_matrix(
        config: &DFlashConfig,
        tensors: &mut Vec<SafetensorsTensorConstraint>,
        name: String,
        shape: Vec<usize>,
    ) -> Result<(), String> {
        let format = config.linear_format_for(&name);
        tensors.extend(
            matrix_for_linear_format(name, std::iter::empty::<String>(), shape, format, None)
                .map_err(|error| error.to_string())?,
        );
        Ok(())
    }

    config
        .validate_released()
        .map_err(|error| error.to_string())?;
    let dimension = |value: i32, name: &str| {
        usize::try_from(value)
            .ok()
            .filter(|value| *value > 0)
            .ok_or_else(|| format!("DFlash {name} must be positive"))
    };
    let hidden = dimension(config.hidden_size, "hidden size")?;
    let intermediate = dimension(config.intermediate_size, "intermediate size")?;
    let head = dimension(config.head_dim, "head width")?;
    let query = dimension(config.num_attention_heads, "query heads")?
        .checked_mul(head)
        .ok_or_else(|| "DFlash query width overflows".to_string())?;
    let key_value = dimension(config.num_key_value_heads, "KV heads")?
        .checked_mul(head)
        .ok_or_else(|| "DFlash KV width overflows".to_string())?;
    let encoded = hidden
        .checked_mul(config.target_layer_ids.len())
        .ok_or_else(|| "DFlash encoder width overflows".to_string())?;
    let vector = |name: String, shape: Vec<usize>| {
        SafetensorsTensorConstraint::required(name, shape, StoredDtypeConstraint::Floating)
    };
    let mut tensors = vec![
        vector("encoder.output_norm_enc.weight".into(), vec![hidden]),
        vector("norm.weight".into(), vec![hidden]),
    ];
    add_matrix(
        config,
        &mut tensors,
        "encoder.fc.weight".into(),
        vec![hidden, encoded],
    )?;
    for layer in 0..config.num_hidden_layers as usize {
        let root = format!("layers.{layer}");
        tensors.extend([
            vector(format!("{root}.input_layernorm.weight"), vec![hidden]),
            vector(
                format!("{root}.post_attention_layernorm.weight"),
                vec![hidden],
            ),
            vector(format!("{root}.self_attn.q_norm.weight"), vec![head]),
            vector(format!("{root}.self_attn.k_norm.weight"), vec![head]),
        ]);
        for (local, shape) in [
            ("self_attn.q_proj.weight", vec![query, hidden]),
            ("self_attn.k_proj.weight", vec![key_value, hidden]),
            ("self_attn.v_proj.weight", vec![key_value, hidden]),
            ("self_attn.o_proj.weight", vec![hidden, query]),
            ("mlp.gate_proj.weight", vec![intermediate, hidden]),
            ("mlp.up_proj.weight", vec![intermediate, hidden]),
            ("mlp.down_proj.weight", vec![hidden, intermediate]),
        ] {
            add_matrix(config, &mut tensors, format!("{root}.{local}"), shape)?;
        }
    }
    SafetensorsCheckpointPlan::new(
        "Muse-Glimmer DFlash SafeTensors",
        tensors,
        Vec::new(),
        CatalogPolicy::strict(),
    )
    .map_err(|error| error.to_string())
}

/// Builds the strict released DFlash GGUF tensor catalog.
pub fn dflash_gguf_plan(config: &DFlashConfig) -> Result<GgufCheckpointPlan, String> {
    config
        .validate_released()
        .map_err(|error| error.to_string())?;
    let dimension = |value: i32, name: &str| {
        usize::try_from(value)
            .ok()
            .filter(|value| *value > 0)
            .ok_or_else(|| format!("DFlash {name} must be positive"))
    };
    let hidden = dimension(config.hidden_size, "hidden size")?;
    let intermediate = dimension(config.intermediate_size, "intermediate size")?;
    let head = dimension(config.head_dim, "head width")?;
    let query = dimension(config.num_attention_heads, "query heads")?
        .checked_mul(head)
        .ok_or_else(|| "DFlash query width overflows".to_string())?;
    let key_value = dimension(config.num_key_value_heads, "KV heads")?
        .checked_mul(head)
        .ok_or_else(|| "DFlash KV width overflows".to_string())?;
    let matrix = |name: String, shape: Vec<usize>| {
        GgufTensorConstraint::required(
            name,
            shape,
            GgufTypeConstraint::OperationClass(TensorOperation::Matrix),
        )
    };
    let vector = |name: String, shape: Vec<usize>| {
        GgufTensorConstraint::required(
            name,
            shape,
            GgufTypeConstraint::OperationClass(TensorOperation::Vector),
        )
    };
    let encoded = hidden
        .checked_mul(config.target_layer_ids.len())
        .ok_or_else(|| "DFlash encoder width overflows".to_string())?;
    let mut tensors = vec![
        matrix("fc.weight".into(), vec![hidden, encoded]),
        vector("enc.output_norm.weight".into(), vec![hidden]),
        vector("output_norm.weight".into(), vec![hidden]),
    ];
    for layer in 0..config.num_hidden_layers as usize {
        let root = format!("blk.{layer}");
        tensors.extend([
            vector(format!("{root}.attn_norm.weight"), vec![hidden]),
            vector(format!("{root}.ffn_norm.weight"), vec![hidden]),
            vector(format!("{root}.attn_q_norm.weight"), vec![head]),
            vector(format!("{root}.attn_k_norm.weight"), vec![head]),
            matrix(format!("{root}.attn_q.weight"), vec![query, hidden]),
            matrix(format!("{root}.attn_k.weight"), vec![key_value, hidden]),
            matrix(format!("{root}.attn_v.weight"), vec![key_value, hidden]),
            matrix(format!("{root}.attn_output.weight"), vec![hidden, query]),
            matrix(
                format!("{root}.ffn_gate.weight"),
                vec![intermediate, hidden],
            ),
            matrix(format!("{root}.ffn_up.weight"), vec![intermediate, hidden]),
            matrix(
                format!("{root}.ffn_down.weight"),
                vec![hidden, intermediate],
            ),
        ]);
    }
    GgufCheckpointPlan::new(
        "Muse-Glimmer DFlash GGUF",
        tensors,
        Vec::new(),
        CatalogPolicy::strict(),
    )
    .map_err(|error| error.to_string())
}

/// Translates one released DFlash GGUF tensor to neutral parameter identity.
pub fn translate_dflash_gguf_weight_name(name: &str) -> String {
    match name {
        "fc.weight" => return "encoder.fc.weight".into(),
        "enc.output_norm.weight" => return "encoder.output_norm_enc.weight".into(),
        "output_norm.weight" => return "norm.weight".into(),
        _ => {}
    }
    let Some(rest) = name.strip_prefix("blk.") else {
        return name.into();
    };
    let Some((layer, parameter)) = rest.split_once('.') else {
        return name.into();
    };
    for (source, target) in [
        ("attn_norm", "input_layernorm"),
        ("ffn_norm", "post_attention_layernorm"),
        ("attn_q_norm", "self_attn.q_norm"),
        ("attn_k_norm", "self_attn.k_norm"),
        ("attn_q", "self_attn.q_proj"),
        ("attn_k", "self_attn.k_proj"),
        ("attn_v", "self_attn.v_proj"),
        ("attn_output", "self_attn.o_proj"),
        ("ffn_gate", "mlp.gate_proj"),
        ("ffn_up", "mlp.up_proj"),
        ("ffn_down", "mlp.down_proj"),
    ] {
        if parameter == source || parameter.starts_with(&format!("{source}.")) {
            return format!("layers.{layer}.{}", parameter.replacen(source, target, 1));
        }
    }
    name.into()
}

/// Per-layer K/V projection of committed target context.
#[derive(Debug, Clone)]
pub struct DFlashLayerContext<T> {
    /// Normalized rotary keys.
    pub keys: T,
    /// Projected values.
    pub values: T,
}

/// Canonical committed target context; transient proposal K/V is never stored here.
#[derive(Debug, Clone)]
pub struct DFlashContext<T> {
    /// Encoded concatenated target taps retained for diagnostics and continuation.
    pub encoded: T,
    /// Per-assistant-layer committed K/V.
    pub layers: Vec<DFlashLayerContext<T>>,
    /// Absolute first retained target position.
    pub start: i32,
    /// Absolute committed target frontier.
    pub end: i32,
}

impl<T> DFlashContext<T> {
    /// Returns the retained committed context length.
    pub fn retained_len(&self) -> i32 {
        self.end - self.start
    }
}

#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
struct DFlashAttention<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> {
    query: B::Linear,
    key: B::Linear,
    value: B::Linear,
    output: B::Linear,
    query_norm: B::Normalization,
    key_norm: B::Normalization,
    rotary: B::Rotary,
    #[parameter(skip)]
    query_heads: i32,
    #[parameter(skip)]
    key_value_heads: i32,
    #[parameter(skip)]
    head_dim: i32,
    #[parameter(skip)]
    scale: f32,
    #[parameter(skip)]
    arithmetic: eredu_nn::AttentionArithmetic,
}

impl<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> DFlashAttention<B> {
    fn new(
        config: &DFlashDecoderConfig,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let prefix = format!("layers.{layer}.self_attn");
        let linear = |field: &str, input, output| {
            let name = format!("{prefix}.{field}.weight");
            B::linear(
                LinearSpec {
                    input,
                    output,
                    weight: ParameterSpec::trainable(&name).map_err(Error::backend)?,
                    bias: None,
                    format: crate::linear_format::standard_linear_format(
                        &name,
                        config.linear_format_for(&name),
                    )?,
                },
                context,
            )
        };
        let norm = |field: &str| {
            B::normalization(
                config.equation.normalization(
                    config.head_dim,
                    config.rms_norm_eps,
                    ParameterSpec::trainable(format!("{prefix}.{field}.weight"))
                        .map_err(Error::backend)?,
                ),
                context,
            )
        };
        Ok(Self {
            query: linear(
                "q_proj",
                config.hidden_size,
                config.num_attention_heads * config.head_dim,
            )?,
            key: linear(
                "k_proj",
                config.hidden_size,
                config.num_key_value_heads * config.head_dim,
            )?,
            value: linear(
                "v_proj",
                config.hidden_size,
                config.num_key_value_heads * config.head_dim,
            )?,
            output: linear(
                "o_proj",
                config.num_attention_heads * config.head_dim,
                config.hidden_size,
            )?,
            query_norm: norm("q_norm")?,
            key_norm: norm("k_norm")?,
            rotary: B::rotary(
                RotarySpec {
                    arithmetic: match config.equation {
                        DFlashDecoderEquation::InputRounded => eredu_nn::RotaryArithmetic::Native,
                        DFlashDecoderEquation::Float32ResidualNorm => eredu_nn::RotaryArithmetic::InputProducts,
                    },
                    dimensions: config.head_dim,
                    base: config.rope_theta,
                    traditional: false,
                    algorithm: eredu_nn::RotaryAlgorithm::Default,
                },
                context,
            )?,
            query_heads: config.num_attention_heads,
            key_value_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            scale: (config.head_dim as f32).sqrt().recip(),
            arithmetic: match config.equation {
                DFlashDecoderEquation::InputRounded => eredu_nn::AttentionArithmetic::Fused,
                DFlashDecoderEquation::Float32ResidualNorm => eredu_nn::AttentionArithmetic::Float32,
            },
        })
    }

    fn project_context(
        &mut self,
        hidden: &B::Tensor,
        offset: i32,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<DFlashLayerContext<B::Tensor>, Error> {
        let batch = hidden.dim(0);
        let length = hidden.dim(1);
        let keys = self.key.forward(hidden, context)?.reshape(
            &[batch, length, self.key_value_heads, self.head_dim],
            context,
        )?;
        let keys = self
            .key_norm
            .forward(&keys.transpose_axes(&[0, 2, 1, 3], context)?, context)?;
        let keys = self
            .rotary
            .forward(&keys, RotaryPosition::Offset(offset), context)?;
        let values = self
            .value
            .forward(hidden, context)?
            .reshape(
                &[batch, length, self.key_value_heads, self.head_dim],
                context,
            )?
            .transpose_axes(&[0, 2, 1, 3], context)?;
        Ok(DFlashLayerContext { keys, values })
    }

    fn forward(
        &mut self,
        hidden: &B::Tensor,
        committed: &DFlashLayerContext<B::Tensor>,
        committed_len: i32,
        absolute_end: i32,
        window: i32,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut crate::decoder::ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        let batch = hidden.dim(0);
        let block = hidden.dim(1);
        let reshape = |value: B::Tensor, heads| {
            value
                .reshape(&[batch, block, heads, self.head_dim], context)?
                .transpose_axes(&[0, 2, 1, 3], context)
        };
        let queries = reshape(self.query.forward(hidden, context)?, self.query_heads)?;
        let queries = self.query_norm.forward(&queries, context)?;
        let queries =
            self.rotary
                .forward(&queries, RotaryPosition::Offset(absolute_end), context)?;
        instrumentation.observe("attention.queries", &queries)?;
        let block_keys = reshape(self.key.forward(hidden, context)?, self.key_value_heads)?;
        let block_keys = self.key_norm.forward(&block_keys, context)?;
        let block_keys =
            self.rotary
                .forward(&block_keys, RotaryPosition::Offset(absolute_end), context)?;
        let block_values = reshape(self.value.forward(hidden, context)?, self.key_value_heads)?;
        let keys = B::Tensor::concatenate(&[committed.keys.clone(), block_keys], 2, context)?;
        let values = B::Tensor::concatenate(&[committed.values.clone(), block_values], 2, context)?;
        let mask = proposal_attention_mask::<B::Tensor>(
            committed_len,
            block,
            absolute_end,
            window,
            ProposalAttentionPolicy::Bidirectional,
            context,
        )?;
        let attended = B::attention_with_sinks(eredu_nn::AttentionRequest {
            queries, keys, values, scale: self.scale, mask: Some(&mask), sinks: None,
            softcap: None, arithmetic: self.arithmetic,
        }, context)?;
        instrumentation.observe("attention.attended_heads", &attended)?;
        let attended = attended
            .transpose_axes(&[0, 2, 1, 3], context)?
            .reshape(&[batch, block, self.query_heads * self.head_dim], context)?;
        self.output.forward(&attended, context)
    }
}

#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
struct DFlashBlock<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> {
    #[parameter(skip)]
    equation: DFlashDecoderEquation,
    input_norm: B::Normalization,
    attention: DFlashAttention<B>,
    post_attention_norm: B::Normalization,
    gate: B::Linear,
    up: B::Linear,
    down: B::Linear,
    attention_conv: Option<block_convolution::BlockConvolution<B>>,
    mlp_conv: Option<block_convolution::BlockConvolution<B>>,
}

impl<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> DFlashBlock<B> {
    fn new(
        config: &DFlashDecoderConfig,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let root = format!("layers.{layer}");
        let norm = |field: &str| {
            B::normalization(
                config.equation.normalization(
                    config.hidden_size,
                    config.rms_norm_eps,
                    ParameterSpec::trainable(format!("{root}.{field}.weight"))
                        .map_err(Error::backend)?,
                ),
                context,
            )
        };
        let linear = |field: &str, input, output| {
            let name = format!("{root}.mlp.{field}.weight");
            B::linear(
                LinearSpec {
                    input,
                    output,
                    weight: ParameterSpec::trainable(&name).map_err(Error::backend)?,
                    bias: None,
                    format: crate::linear_format::standard_linear_format(
                        &name,
                        config.linear_format_for(&name),
                    )?,
                },
                context,
            )
        };
        let convolution = |field: &str| match config.sublayer {
            DFlashSublayerPolicy::Direct => Ok(None),
            DFlashSublayerPolicy::BlockConvolution { taps, group_size } => {
                block_convolution::BlockConvolution::new_with_format(
                    block_convolution::BlockConvolutionGeometry {
                        hidden: config.hidden_size,
                        block_size: config.block_size as i32,
                        taps,
                        group_size,
                    },
                    format!("{root}.{field}"),
                    config.linear_format_for(&format!("{root}.{field}.kernel_projection.weight")),
                    context,
                )
                .map(Some)
            }
        };
        Ok(Self {
            equation: config.equation,
            input_norm: norm("input_layernorm")?,
            attention: DFlashAttention::new(config, layer, context)?,
            post_attention_norm: norm("post_attention_layernorm")?,
            gate: linear("gate_proj", config.hidden_size, config.intermediate_size)?,
            up: linear("up_proj", config.hidden_size, config.intermediate_size)?,
            down: linear("down_proj", config.intermediate_size, config.hidden_size)?,
            attention_conv: convolution("attention_conv")?,
            mlp_conv: convolution("mlp_conv")?,
        })
    }

    fn forward_observed(
        &mut self,
        state: &body::AssistantBodyState<B::Tensor>,
        committed: &DFlashLayerContext<B::Tensor>,
        committed_len: i32,
        absolute_end: i32,
        window: i32,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut crate::decoder::ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<body::AssistantBodyState<B::Tensor>, Error> {
        let hidden = &state.hidden;
        let dtype = hidden.element_type().ok_or_else(|| Error::backend("assistant state has no dtype"))?;
        let input = match &state.residual {
            Some(residual) => hidden.cast_float(eredu_nn::TensorElementType::F32, context)?
                .add(&residual.cast_float(eredu_nn::TensorElementType::F32, context)?, context)?,
            None => hidden.clone(),
        };
        let residual = input.cast_float(dtype, context)?;
        instrumentation.observe("input", &residual)?;
        let normalized = self.input_norm.forward(&input, context)?.cast_float(dtype, context)?;
        let (normalized, attention_kernel) = match &mut self.attention_conv {
            Some(conv) => {
                let (input, kernel) = conv.prepare(&normalized, context)?;
                (input, Some(kernel))
            }
            None => (normalized, None),
        };
        let attended = self.attention.forward(
            &normalized,
            committed,
            committed_len,
            absolute_end,
            window,
            context,
            instrumentation,
        )?;
        let attended = match (&self.attention_conv, attention_kernel) {
            (Some(conv), Some(kernel)) => conv.finish(&attended, kernel, context)?,
            _ => attended,
        };
        let sum = match self.equation {
            DFlashDecoderEquation::InputRounded => residual.add(&attended, context)?,
            DFlashDecoderEquation::Float32ResidualNorm => residual.cast_float(eredu_nn::TensorElementType::F32, context)?
                .add(&attended.cast_float(eredu_nn::TensorElementType::F32, context)?, context)?,
        };
        let hidden = sum.cast_float(dtype, context)?;
        let normalized = self.post_attention_norm.forward(&sum, context)?.cast_float(dtype, context)?;
        let (normalized, mlp_kernel) = match &mut self.mlp_conv {
            Some(conv) => {
                let (input, kernel) = conv.prepare(&normalized, context)?;
                (input, Some(kernel))
            }
            None => (normalized, None),
        };
        let gate = self.gate.forward(&normalized, context)?;
        let up = self.up.forward(&normalized, context)?;
        let feed_forward =
            B::gated_product(gate, up, eredu_nn::GatedProductPolicy::default(), context)?;
        let feed_forward = self.down.forward(&feed_forward, context)?;
        let feed_forward = match (&self.mlp_conv, mlp_kernel) {
            (Some(conv), Some(kernel)) => conv.finish(&feed_forward, kernel, context)?,
            _ => feed_forward,
        };
        Ok(match self.equation {
            DFlashDecoderEquation::InputRounded => {
                let output = hidden.add(&feed_forward, context)?;
                instrumentation.observe("output", &output)?;
                body::AssistantBodyState { hidden: output, residual: None }
            }
            DFlashDecoderEquation::Float32ResidualNorm => {
                if instrumentation.enabled() {
                    instrumentation.observe("output", &hidden.add(&feed_forward, context)?)?;
                }
                body::AssistantBodyState { hidden: feed_forward, residual: Some(hidden) }
            }
        })
    }
}

/// Neutral DFlash assistant body. The target embedding and output head remain target-owned.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct DFlash<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> {
    units: eredu_runtime::ResidentUnitWindow<body::DFlashBodyUnit<B>>,
    #[parameter(skip)]
    definition: body::DFlashBodyDefinition,
}

impl<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> DFlash<B> {
    /// Builds the unloaded released assistant body.
    pub fn new(
        config: DFlashConfig,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        config.validate_released().map_err(Error::backend)?;
        Self::from_decoder_config(config.into(), context)
    }
    /// Constructs the shared body from normalized architecture equations.
    pub fn from_decoder_config(
        config: DFlashDecoderConfig,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let definition = body::DFlashBodyDefinition::new(config)?;
        let units = body::AssistantBodyDefinition::<B>::build_units(&definition, context)?;
        Ok(Self {
            units: eredu_runtime::ResidentUnitWindow::new(units),
            definition,
        })
    }
    /// Returns target layers that the ordinary target forward must capture.
    pub fn target_layer_ids(&self) -> &[usize] {
        self.definition.target_layer_ids()
    }
    /// Separates validated geometry from resident units for ordinary parameter policies.
    pub fn into_body_parts(
        self,
    ) -> (
        body::DFlashBodyDefinition,
        eredu_runtime::ResidentUnitWindow<body::DFlashBodyUnit<B>>,
    ) {
        (self.definition, self.units)
    }
    /// Concatenates ordered target taps into the assistant encoder input.
    pub fn assemble_target_states(
        &self,
        states: &[B::Tensor],
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        body::AssistantBodyDefinition::<B>::geometry(&self.definition)
            .assemble_target_states(states, context)
    }
    /// Encodes and appends newly committed target taps, retaining one assistant window.
    pub fn update_context(
        &mut self,
        previous: Option<DFlashContext<B::Tensor>>,
        pending: &B::Tensor,
        end: i32,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<DFlashContext<B::Tensor>, Error> {
        body::update_context::<B, _, _>(
            &self.definition,
            &mut self.units,
            previous,
            pending,
            end,
            context,
        )
    }
    /// Runs one anchor-plus-mask proposal block and returns mask-position states.
    pub fn proposal_states(
        &mut self,
        embeddings: &B::Tensor,
        committed: &DFlashContext<B::Tensor>,
        end: i32,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        body::proposal_states::<B, _, _>(
            &self.definition,
            &mut self.units,
            embeddings,
            committed,
            end,
            context,
            &mut crate::decoder::ComponentInstrumentation::disabled(),
        )
    }
}

/// Appends independently projected committed context; proposal state never enters it.
fn append_projected_context<T: Tensor>(
    previous: Option<DFlashContext<T>>,
    encoded: T,
    projected: Vec<DFlashLayerContext<T>>,
    absolute_end: i32,
    window: i32,
    context: &T::Context,
) -> Result<DFlashContext<T>, Error> {
    let (encoded, layers) = match previous {
        Some(previous) => {
            if previous.layers.len() != projected.len()
                || previous.encoded.dim(1) != previous.retained_len()
            {
                return Err(Error::backend("invalid cached DFlash context geometry"));
            }
            let encoded = retain_sequence_tail(
                T::concatenate(&[previous.encoded, encoded], 1, context)?,
                1,
                window,
                context,
            )?;
            let layers = previous
                .layers
                .into_iter()
                .zip(projected)
                .map(|(previous, pending)| {
                    Ok(DFlashLayerContext {
                        keys: retain_sequence_tail(
                            T::concatenate(&[previous.keys, pending.keys], 2, context)?,
                            2,
                            window,
                            context,
                        )?,
                        values: retain_sequence_tail(
                            T::concatenate(&[previous.values, pending.values], 2, context)?,
                            2,
                            window,
                            context,
                        )?,
                    })
                })
                .collect::<Result<Vec<_>, Error>>()?;
            (encoded, layers)
        }
        None => {
            let encoded = retain_sequence_tail(encoded, 1, window, context)?;
            let layers = projected
                .into_iter()
                .map(|layer| {
                    Ok(DFlashLayerContext {
                        keys: retain_sequence_tail(layer.keys, 2, window, context)?,
                        values: retain_sequence_tail(layer.values, 2, window, context)?,
                    })
                })
                .collect::<Result<Vec<_>, Error>>()?;
            (encoded, layers)
        }
    };
    let retained = encoded.dim(1);
    Ok(DFlashContext {
        encoded,
        layers,
        start: absolute_end - retained,
        end: absolute_end,
    })
}

fn context_append_start(
    previous_end: Option<i32>,
    pending_len: i32,
    absolute_end: i32,
) -> Result<i32, Error> {
    if pending_len <= 0 || absolute_end < pending_len {
        return Err(Error::backend("invalid DFlash context range"));
    }
    let start = absolute_end - pending_len;
    if previous_end.is_some_and(|previous_end| previous_end != start) {
        return Err(Error::backend("DFlash context/cache frontier mismatch"));
    }
    Ok(start)
}

fn retain_sequence_tail<T: Tensor>(
    value: T,
    axis: usize,
    window: i32,
    context: &T::Context,
) -> Result<T, Error> {
    let length = value.dim(axis);
    let start = (length - window).max(0);
    let mut indexes = vec![Index::Full; value.shape().len()];
    indexes[axis] = Index::Range(start, length);
    value.index(&indexes, context)?.compact(context)
}

#[derive(Clone, Copy)]
enum ProposalAttentionPolicy {
    Causal,
    Bidirectional,
}

fn proposal_attention_mask<T: Tensor>(
    context_len: i32,
    block_len: i32,
    context_end: i32,
    window: i32,
    policy: ProposalAttentionPolicy,
    context: &T::Context,
) -> Result<T, Error> {
    if context_len <= 0 || block_len <= 0 || window <= 0 || context_end < context_len {
        return Err(Error::backend("invalid DFlash attention mask geometry"));
    }
    let key_len = context_len + block_len;
    let context_start = context_end - context_len;
    let mut values = Vec::with_capacity((block_len * key_len) as usize);
    for query in 0..block_len {
        let query_position = context_end + query;
        for key in 0..key_len {
            let key_position = context_start + key;
            let in_block = key >= context_len;
            let allowed = match policy {
                ProposalAttentionPolicy::Bidirectional if in_block => true,
                _ => key_position <= query_position && query_position - key_position < window,
            };
            values.push(if allowed { 0.0 } else { f32::NEG_INFINITY });
        }
    }
    T::from_f32_slice(&values, &[1, 1, block_len, key_len], context)
}
