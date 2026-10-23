//! EAR-TTS Gemma3 causal and T5Gemma bidirectional blocks over shared decoder operations.
//!
//! Parameters can be materialized one block at a time. The resident assembly is
//! a convenience for component execution; cache ownership, residency, submission
//! completion and publication remain with the existing execution drivers.
//! These embedded-input components have no vocabulary axis or text output head;
//! use their block groups and final norm rather than full language-model statics.

use super::config::GemmaConfig;
use crate::decoder::{
    AttentionProjection, BlockParameterFields, Config, PartitionedConfig, TransformerBlock,
};
use eredu_checkpoint::{LinearFormat, WeightQuantization};
use eredu_core::{AttentionPolicy, LayerSchedule};
use eredu_nn::{
    AttentionCache, AttentionRequest, Error, GatedProductPolicy, NeuralBackend,
    NormalizationConstructionSpec, NormalizationScale, ParameterSpec, RotaryAlgorithm,
    RotaryArithmetic, RotarySpec, Tensor, TensorElementType,
};
use std::collections::BTreeMap;

/// Normalized equations and exact parameter namespace for one TTS transformer.
#[derive(Debug, Clone)]
pub struct TransformerConfig {
    precision: super::precision::SpeechPrecision,
    geometry: GemmaConfig,
    schedule: LayerSchedule<AttentionPolicy>,
    formats: BTreeMap<String, LinearFormat>,
}
impl TransformerConfig {
    /// Validates Gemma3 causal or T5Gemma encoder policy before native construction.
    pub fn new(geometry: &GemmaConfig) -> Result<Self, Error> {
        let positive = |n: usize| n > 0 && n <= i32::MAX as usize;
        if ![
            geometry.hidden_size,
            geometry.intermediate_size,
            geometry.num_hidden_layers,
            geometry.num_attention_heads,
            geometry.num_key_value_heads,
            geometry.head_dim,
            geometry.max_position_embeddings,
            geometry.query_pre_attn_scalar,
            geometry.sliding_window,
        ]
        .into_iter()
        .all(positive)
            || !geometry.head_dim.is_multiple_of(2)
            || geometry.num_hidden_layers > 256
            || !geometry
                .num_attention_heads
                .is_multiple_of(geometry.num_key_value_heads)
            || geometry
                .num_attention_heads
                .checked_mul(geometry.head_dim)
                .is_none_or(|n| n > i32::MAX as usize)
            || geometry.layer_types.len() != geometry.num_hidden_layers
            || geometry.hidden_activation != "gelu_pytorch_tanh"
            || geometry.rope_scaling.is_some()
            || !geometry.rms_norm_eps.is_finite()
            || geometry.rms_norm_eps <= 0.0
            || !(geometry.rms_norm_eps as f32).is_finite()
            || geometry.rms_norm_eps as f32 <= 0.0
            || !geometry.rope_theta.is_finite()
            || geometry.rope_theta <= 0.0
            || !(geometry.rope_theta as f32).is_finite()
            || geometry.rope_theta as f32 <= 0.0
            || geometry
                .rope_local_base_freq
                .is_some_and(|n| !(n as f32).is_finite() || n as f32 <= 0.0)
            || geometry
                .attn_logit_softcapping
                .is_some_and(|n| !(n as f32).is_finite() || n as f32 <= 0.0)
            || (geometry.causal && geometry.rope_local_base_freq.is_none())
        {
            return Err(Error::backend(
                "invalid EAR-TTS transformer geometry or equations",
            ));
        }
        let policies = geometry
            .layer_types
            .iter()
            .map(|kind| match kind.as_str() {
                "full_attention" => Ok(AttentionPolicy::Full),
                "sliding_attention" => {
                    AttentionPolicy::sliding(geometry.sliding_window as u32).map_err(Error::backend)
                }
                _ => Err(Error::backend("unknown EAR-TTS attention layer kind")),
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            geometry: geometry.clone(),
            schedule: LayerSchedule::new(geometry.num_hidden_layers, policies)
                .map_err(Error::backend)?,
            formats: BTreeMap::new(),
            precision: Default::default(),
        })
    }
    /// Selects projection storage/arithmetic while norms and caches remain F32.
    pub fn with_precision(mut self, precision: super::precision::SpeechPrecision) -> Self {
        self.precision = precision;
        self
    }
    /// Retained floating policy.
    pub fn precision(&self) -> super::precision::SpeechPrecision {
        self.precision
    }
    /// Installs physical formats selected by the retained checkpoint recipe.
    pub fn with_formats(mut self, formats: BTreeMap<String, LinearFormat>) -> Self {
        self.formats = formats;
        self
    }
    /// Exact admitted dimensions and positional capacity.
    pub fn geometry(&self) -> &GemmaConfig {
        &self.geometry
    }
    /// Builds one independently materializable shared decoder block.
    pub fn build_block<B: NeuralBackend>(
        &self,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<TransformerBlock<B>, Error> {
        if layer >= self.geometry.num_hidden_layers {
            return Err(Error::backend("TTS layer is outside its stack"));
        }
        TransformerBlock::new(self, layer, context)
    }
    /// Builds the separately materializable final hidden-state normalization.
    pub fn build_final_norm<B: NeuralBackend>(
        &self,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Normalization, Error> {
        B::normalization(
            NormalizationConstructionSpec {
                reduction: eredu_nn::NormalizationReduction::Native,
                groups: Some(1),
                dimensions: self.hidden_size(),
                epsilon: self.rms_norm_epsilon(),
                scale: NormalizationScale::LearnedOffset {
                    weight: ParameterSpec::trainable(format!(
                        "{}.norm.weight",
                        self.parameter_root()
                    ))
                    .map_err(Error::backend)?,
                    offset: 1.0,
                },
            },
            context,
        )
    }
}
impl Config for TransformerConfig {
    fn linear_compute_element(&self) -> Option<eredu_nn::TensorElementType> {
        self.precision.projection_element()
    }
    fn model_family(&self) -> &'static str {
        "nemotron_voicechat"
    }
    fn model_identity(&self) -> &str {
        self.parameter_root()
    }
    fn architecture_fingerprint(&self) -> String {
        format!(
            "voicechat-tts-transformer:{:?}:{:?}",
            self.geometry,
            (&self.formats, self.precision)
        )
    }
    fn parameter_root(&self) -> &str {
        if self.geometry.causal {
            "tts_model.tts_model.backbone"
        } else {
            "tts_model.tts_model.embed_subword.backbone.encoder"
        }
    }
    fn validate_config(&self) -> Result<(), Error> {
        Self::new(&self.geometry).map(|_| ())
    }
    fn hidden_size(&self) -> i32 {
        self.geometry.hidden_size as i32
    }
    fn intermediate_size(&self) -> i32 {
        self.geometry.intermediate_size as i32
    }
    fn num_hidden_layers(&self) -> i32 {
        self.geometry.num_hidden_layers as i32
    }
    fn num_attention_heads(&self) -> i32 {
        self.geometry.num_attention_heads as i32
    }
    fn num_key_value_heads(&self) -> i32 {
        self.geometry.num_key_value_heads as i32
    }
    fn head_dim(&self) -> i32 {
        self.geometry.head_dim as i32
    }
    fn rms_norm_epsilon(&self) -> f32 {
        self.geometry.rms_norm_eps as f32
    }
    fn normalization_offset(&self) -> f32 {
        1.0
    }
    fn normalization_groups(&self) -> Option<i32> {
        Some(1)
    }
    fn query_key_norm_epsilon(&self) -> Option<f32> {
        self.geometry.causal.then_some(self.rms_norm_epsilon())
    }
    fn query_key_normalization_offset(&self) -> f32 {
        1.0
    }
    fn query_key_normalization_groups(&self) -> Option<i32> {
        Some(1)
    }
    fn attention_scale(&self) -> f32 {
        (self.geometry.query_pre_attn_scalar as f32).sqrt().recip()
    }
    fn attention_softcap(&self) -> Option<f32> {
        if self.geometry.causal {
            None
        } else {
            self.geometry.attn_logit_softcapping.map(|n| n as f32)
        }
    }
    fn attention_arithmetic(&self) -> eredu_nn::AttentionArithmetic {
        eredu_nn::AttentionArithmetic::InputScores
    }
    fn vocabulary_size(&self) -> i32 {
        // No vocabulary axis: full language-model static construction rejects
        // this geometry before allocating an embedding or projection table.
        0
    }
    fn attention_bias(&self, _: AttentionProjection) -> bool {
        self.geometry.attention_bias
    }
    fn mlp_bias(&self) -> bool {
        false
    }
    fn tie_word_embeddings(&self) -> bool {
        true
    }
    fn attention_schedule(&self) -> &LayerSchedule<AttentionPolicy> {
        &self.schedule
    }
    fn weight_quantization(&self, _: &str) -> Option<WeightQuantization> {
        None
    }
    fn linear_format(&self, name: &str) -> LinearFormat {
        self.formats
            .get(name)
            .copied()
            .unwrap_or(LinearFormat::Dense)
    }
    fn gated_product_policy(&self) -> Option<GatedProductPolicy> {
        Some(GatedProductPolicy::ordinary_gelu_approximate())
    }
    fn rotary_spec(&self, dimensions: i32) -> RotarySpec {
        RotarySpec {
            arithmetic: RotaryArithmetic::InputProducts,
            dimensions,
            base: self.geometry.rope_theta as f32,
            traditional: false,
            algorithm: RotaryAlgorithm::Default,
        }
    }
    fn layer_rotary_spec(&self, layer: usize, dimensions: i32) -> RotarySpec {
        let mut spec = self.rotary_spec(dimensions);
        if self.geometry.causal && self.schedule.get(layer).and_then(|p| p.window()).is_some() {
            spec.base = self.geometry.rope_local_base_freq.unwrap() as f32;
        }
        spec
    }
    fn block_parameter_fields(&self) -> BlockParameterFields<'_> {
        BlockParameterFields {
            input_norm: if self.geometry.causal {
                "input_layernorm"
            } else {
                "pre_self_attn_layernorm"
            },
            post_attention_norm: "pre_feedforward_layernorm",
            ..Default::default()
        }
    }
    fn attention_output_normalization(&self, layer: usize) -> Option<String> {
        Some(format!(
            "{}.layers.{layer}.{}.weight",
            self.parameter_root(),
            if self.geometry.causal {
                "post_attention_layernorm"
            } else {
                "post_self_attn_layernorm"
            }
        ))
    }
    fn feed_forward_output_normalization(&self, layer: usize) -> Option<String> {
        Some(format!(
            "{}.layers.{layer}.post_feedforward_layernorm.weight",
            self.parameter_root()
        ))
    }
}
impl PartitionedConfig for TransformerConfig {
    fn set_local_geometry(&mut self, query: i32, kv: i32, intermediate: i32) -> Result<(), Error> {
        if query <= 0 || kv <= 0 || intermediate <= 0 || query % kv != 0 {
            return Err(Error::backend("invalid TTS partition geometry"));
        }
        self.geometry.num_attention_heads = query as usize;
        self.geometry.num_key_value_heads = kv as usize;
        self.geometry.intermediate_size = intermediate as usize;
        Ok(())
    }
}

/// Resident component assembly using the same layered equations as bounded execution.
pub struct TtsTransformer<B: NeuralBackend> {
    runtime: eredu_runtime::ResidentRuntime<
        eredu_runtime::ComponentAdapter<
            eredu_runtime::ComponentDefinition,
            super::tts_transformer_layered::TransformerKernel<B>,
        >,
        B,
        super::tts_transformer_layered::TransformerStateMarker<B>,
    >,
}
impl<B: NeuralBackend> eredu_nn::Parameterized<B::Tensor> for TtsTransformer<B> {
    fn visit_parameters<'a, V>(&'a self, visitor: &mut V)
    where
        V: eredu_nn::ParameterVisitor<'a, B::Tensor>,
    {
        for group in self.runtime.units() {
            for unit in group {
                unit.visit_parameters(visitor);
            }
        }
    }
    fn visit_parameters_mut<'a, V>(&'a mut self, visitor: &mut V)
    where
        V: eredu_nn::ParameterVisitorMut<'a, B::Tensor>,
    {
        for group in self.runtime.units_mut() {
            for unit in group {
                unit.visit_parameters_mut(visitor);
            }
        }
    }
    fn set_trainable(&mut self, trainable: bool) {
        for group in self.runtime.units_mut() {
            for unit in group {
                unit.set_trainable(trainable);
            }
        }
    }
}
impl<B: NeuralBackend> TtsTransformer<B> {
    /// Constructs unloaded units without source access.
    pub fn new(
        config: TransformerConfig,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, super::tts_transformer_layered::TransformerError> {
        Ok(Self {
            runtime: eredu_runtime::ResidentRuntime::new(
                super::tts_transformer_layered::TransformerKernel::new(config)?.into_component()?,
                context,
            )
            .map_err(eredu_runtime::ResidentUnitsError::into_architecture)?,
        })
    }
    /// Adopts independently materialized blocks and normalization without replacements.
    pub(super) fn from_parts(
        config: TransformerConfig,
        blocks: Vec<TransformerBlock<B>>,
        norm: B::Normalization,
    ) -> Result<Self, super::tts_transformer_layered::TransformerError> {
        use super::tts_transformer_layered::{TransformerKernel, TransformerUnit};
        let units = vec![
            blocks.into_iter().map(TransformerUnit::Block).collect(),
            vec![TransformerUnit::Norm(norm)],
        ];
        Ok(Self {
            runtime: eredu_runtime::ResidentRuntime::from_units(
                TransformerKernel::new(config)?.into_component()?,
                units,
            )
            .map_err(eredu_runtime::ResidentUnitsError::into_architecture)?,
        })
    }
    /// Exact retained transformer equations and physical formats.
    pub fn config(&self) -> &TransformerConfig {
        self.runtime.architecture().kernel().config()
    }
    /// Runs independent character rows with bidirectional key-padding masks.
    pub fn forward_encoder(
        &mut self,
        embedded: &B::Tensor,
        key_mask: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, super::tts_transformer_layered::TransformerError> {
        use super::tts_transformer_layered::{TransformerInput, TransformerRuntimeState};
        let mut empty = [];
        let mut state =
            TransformerRuntimeState::<B, NoTransformerCache<B>>::new(self.config(), &mut empty)?;
        self.runtime.forward_with_state(
            TransformerInput {
                embedded,
                key_mask: Some(key_mask),
            },
            &mut state,
            context,
        )
    }
}
impl<B: NeuralBackend> super::tts_transformer_layered::TransformerExecution<B>
    for TtsTransformer<B>
{
    fn config(&self) -> &TransformerConfig {
        self.config()
    }
    fn forward_encoder(
        &mut self,
        embedded: &B::Tensor,
        key_mask: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, super::tts_transformer_layered::TransformerError> {
        self.forward_encoder(embedded, key_mask, context)
    }
}

impl<B: NeuralBackend, C: AttentionCache<B::Tensor> + eredu_runtime::RuntimeLayerState<B>>
    super::tts_transformer_layered::CausalTransformerExecution<B, [C]> for TtsTransformer<B>
{
    fn causal_position(
        &self,
        caches: &[C],
    ) -> Result<i32, super::tts_transformer_layered::TransformerError> {
        super::tts_transformer_layered::validate_causal_caches::<B, C>(self.config(), caches)
    }
    /// Advances caller-owned unpublished cache branches; the caller owns rollback.
    fn forward_causal(
        &mut self,
        embedded: &B::Tensor,
        caches: &mut [C],
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, super::tts_transformer_layered::TransformerError> {
        use super::tts_transformer_layered::{TransformerInput, TransformerRuntimeState};
        let mut state = TransformerRuntimeState::<B, C>::new(self.config(), caches)?;
        self.runtime.forward_with_state(
            TransformerInput {
                embedded,
                key_mask: None,
            },
            &mut state,
            context,
        )
    }
}

pub(super) fn encoder_mask<T: Tensor>(
    keys: &T,
    sequence: i32,
    window: Option<i32>,
    dtype: TensorElementType,
    context: &T::Context,
) -> Result<T, Error> {
    let minimum = match dtype {
        TensorElementType::F16 => -65504.0,
        TensorElementType::Bf16 => -3.3895314e38,
        _ => f32::MIN,
    };
    let denied = T::full_f32(minimum, &[], context)?.cast_float(dtype, context)?;
    let keys = keys.reshape(&[keys.dim(0), 1, 1, sequence], context)?;
    if window.is_none() {
        let allowed = T::full_f32(0.0, &[], context)?.cast_float(dtype, context)?;
        return T::where_condition(&keys, &allowed, &denied, context);
    }
    let count = (sequence as usize)
        .checked_mul(sequence as usize)
        .ok_or_else(|| Error::backend("character attention mask extent overflows"))?;
    let mut values = Vec::new();
    values.try_reserve_exact(count).map_err(Error::backend)?;
    for q in 0..sequence {
        for k in 0..sequence {
            values.push(
                if window.is_none_or(|w| (q as i64 - k as i64).abs() < w as i64) {
                    0.0
                } else {
                    minimum
                },
            );
        }
    }
    let allowed = T::from_f32_slice(&values, &[1, 1, sequence, sequence], context)?
        .cast_float(dtype, context)?;
    T::where_condition(&keys, &allowed, &denied, context)
}

/// Type witness for stateless character attention; no native cache is allocated.
pub struct NoTransformerCache<B>(std::marker::PhantomData<B>);
impl<B: NeuralBackend> AttentionCache<B::Tensor> for NoTransformerCache<B> {
    fn offset(&self) -> i32 {
        0
    }
    fn max_size(&self) -> Option<i32> {
        None
    }
    fn update_for_attention(
        &mut self,
        _: B::Tensor,
        _: B::Tensor,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<(B::Tensor, B::Tensor), Error> {
        Err(Error::backend("uncached attention cannot append state"))
    }
    fn attention(
        &mut self,
        _: AttentionRequest<'_, B::Tensor>,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        Err(Error::backend("uncached attention has no state owner"))
    }
}

impl<B: NeuralBackend> eredu_runtime::RuntimeLayerState<B> for NoTransformerCache<B> {
    type RetainedValues<'a>
        = std::iter::Empty<&'a B::Tensor>
    where
        Self: 'a,
        B::Tensor: 'a;
    fn retained_values(&self) -> Self::RetainedValues<'_> {
        std::iter::empty()
    }
}
