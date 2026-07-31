//! Thinking Machines Lab Inkling multimodal model support.
//!
//! The released checkpoint is a multimodal conditional-generation model.  This
//! module owns the decoder, the native dMel audio and hMLP vision towers, and
//! the native safetensors loader. Multi-token-prediction draft layers are not
//! needed for ordinary autoregressive generation and are skipped by the loader.

#![allow(missing_docs)]

use std::{collections::HashMap, path::Path};

use safemlx::{
    error::Exception,
    macros::ModuleParameters,
    module::{Module, ModuleParameters as ModuleParametersTrait, ModuleParametersExt, Param},
    nn,
    ops::{
        arange, argpartition_axis, broadcast_to, clip, concatenate_axis,
        indexing::{take_along_axis, NewAxis, TryIndexOp},
        matmul, r#where, sigmoid, softmax_axis, sum_axis, GgufCheckpoint, GgufMetadataValue,
    },
    quantization::MaybeQuantized,
    Array, Dtype, Stream,
};
use serde::Deserialize;
use serde_json::Value;
use tokenizers::Tokenizer;

use crate::{
    api::{
        common::{
            self,
            convolution::{causal_depthwise_conv1d, CausalConv1dCache, DepthwiseConv1d},
            generation::CausalLm,
            layers::SwiGluMlp,
            moe::PackedSwiGluExperts,
        },
        input,
        qwen3::{gguf_i32, gguf_string},
    },
    error::Error,
    runtime::cache::residency::{
        CacheRankIdentity, CacheResidencyManager, CacheResidencyReport, PagedCacheOptions,
    },
    runtime::cache::{
        BlockwiseAttentionAccumulator, ConcatKeyValueCache, KeyValueCache, PagedKeyValueCache,
        SlidingKeyValueCache,
    },
    runtime::checkpoint::load::{
        for_each_safetensor_array, gguf_metadata, gguf_quantization_configs, load_array_strict,
        load_named_array_strict, safetensors_files, StrictLoadConfig, StrictLoadReport,
    },
    runtime::checkpoint::quantization::WeightQuantization,
};

fn default_model_type() -> String {
    "inkling_mm_model".into()
}

fn default_true() -> bool {
    true
}

fn default_rms_norm_eps() -> f32 {
    1e-6
}

fn default_head_dim() -> i32 {
    128
}

fn default_sconv_kernel_size() -> i32 {
    4
}

fn default_rel_extent() -> i32 {
    1024
}

fn default_sliding_window() -> i32 {
    512
}

fn default_route_scale() -> f32 {
    1.0
}

fn default_logit_scale() -> f32 {
    1.0
}

fn default_image_token_id() -> u32 {
    200_054
}

fn default_audio_token_id() -> u32 {
    200_053
}

#[derive(Debug, Clone, Deserialize)]
/// Decoder fields from Inkling's nested `text_config`.
pub struct TextArgs {
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    pub vocab_size: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    #[serde(default = "default_head_dim")]
    pub head_dim: i32,
    #[serde(default)]
    pub swa_num_attention_heads: Option<i32>,
    #[serde(default)]
    pub swa_num_key_value_heads: Option<i32>,
    #[serde(default)]
    pub swa_head_dim: Option<i32>,
    #[serde(default = "default_sliding_window", alias = "sliding_window")]
    pub sliding_window_size: i32,
    #[serde(default)]
    pub local_layer_ids: Option<Vec<i32>>,
    #[serde(default)]
    pub layer_types: Option<Vec<String>>,
    #[serde(default)]
    pub dense_mlp_idx: i32,
    #[serde(default)]
    pub mlp_layer_types: Option<Vec<String>>,
    #[serde(default = "default_sconv_kernel_size", alias = "conv_kernel_size")]
    pub sconv_kernel_size: i32,
    #[serde(default = "default_true")]
    pub use_sconv: bool,
    #[serde(default = "default_rel_extent")]
    pub rel_extent: i32,
    pub d_rel: i32,
    #[serde(default)]
    pub log_scaling_n_floor: Option<i32>,
    #[serde(default)]
    pub log_scaling_alpha: f32,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_true")]
    pub use_embed_norm: bool,
    #[serde(default)]
    pub unpadded_vocab_size: Option<i32>,
    #[serde(default = "default_logit_scale")]
    pub logits_mup_width_multiplier: f32,
    #[serde(default)]
    pub final_logit_softcapping: Option<f32>,
    #[serde(default)]
    pub intermediate_size: i32,
    #[serde(default)]
    pub dense_intermediate_size: Option<i32>,
    #[serde(default)]
    pub moe_intermediate_size: Option<i32>,
    #[serde(default, alias = "num_experts")]
    pub n_routed_experts: i32,
    #[serde(default)]
    pub num_experts_per_tok: i32,
    #[serde(default)]
    pub n_shared_experts: i32,
    #[serde(default = "default_route_scale")]
    pub route_scale: f32,
    #[serde(default = "default_true")]
    pub shared_expert_sink: bool,
    #[serde(default = "default_true")]
    pub use_gate_bias: bool,
    #[serde(default = "default_true")]
    pub norm_after_topk: bool,
    #[serde(default = "default_true")]
    pub use_global_scale: bool,
    #[serde(default = "default_gate_activation")]
    pub gate_activation: String,
    #[serde(default = "default_hidden_activation")]
    pub hidden_act: String,
    #[serde(default)]
    pub attention_dropout: f32,
    #[serde(default)]
    pub q_bias: bool,
    #[serde(default)]
    pub o_bias: bool,
    #[serde(default)]
    pub model_max_length: Option<i32>,
    /// Exact per-weight formats for mixed GGUF checkpoints.
    #[serde(skip)]
    pub quantized_weight_configs: Option<HashMap<String, WeightQuantization>>,
}

fn default_gate_activation() -> String {
    "sigmoid".into()
}

fn default_hidden_activation() -> String {
    "silu".into()
}

fn default_audio_mode() -> String {
    "dmel".into()
}

fn default_vision_encoder_type() -> String {
    "hmlp".into()
}

impl TextArgs {
    pub(crate) fn weight_quantization_for(&self, name: &str) -> Option<WeightQuantization> {
        self.quantized_weight_configs
            .as_ref()
            .and_then(|configs| configs.get(name).copied())
    }
    fn dense_intermediate_size(&self) -> i32 {
        self.dense_intermediate_size
            .unwrap_or(self.intermediate_size)
    }

    pub(crate) fn moe_intermediate_size(&self) -> i32 {
        self.moe_intermediate_size.unwrap_or(self.intermediate_size)
    }

    fn is_local(&self, layer: i32) -> bool {
        if let Some(ids) = &self.local_layer_ids {
            return ids.contains(&layer);
        }
        if let Some(types) = &self.layer_types {
            return types
                .get(layer as usize)
                .is_some_and(|kind| kind.contains("sliding"));
        }
        (layer + 1) % 6 != 0
    }

    pub(crate) fn is_dense(&self, layer: i32) -> bool {
        if let Some(types) = &self.mlp_layer_types {
            return types
                .get(layer as usize)
                .is_some_and(|kind| kind == "dense");
        }
        layer < self.dense_mlp_idx
    }

    fn q_heads(&self, local: bool) -> i32 {
        if local {
            self.swa_num_attention_heads
                .unwrap_or(self.num_attention_heads)
        } else {
            self.num_attention_heads
        }
    }

    fn kv_heads(&self, local: bool) -> i32 {
        if local {
            self.swa_num_key_value_heads
                .unwrap_or(self.num_key_value_heads)
        } else {
            self.num_key_value_heads
        }
    }

    fn attention_head_dim(&self, local: bool) -> i32 {
        if local {
            self.swa_head_dim.unwrap_or(self.head_dim)
        } else {
            self.head_dim
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
/// Released top-level Inkling configuration.
pub struct ModelArgs {
    #[serde(default = "default_model_type")]
    pub model_type: String,
    pub text_config: TextArgs,
    #[serde(default)]
    pub audio_config: Option<AudioArgs>,
    #[serde(default)]
    pub vision_config: Option<VisionArgs>,
    #[serde(default = "default_image_token_id")]
    pub image_token_id: u32,
    #[serde(default = "default_audio_token_id")]
    pub audio_token_id: u32,
    #[serde(default)]
    pub eos_token_id: Option<u32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AudioArgs {
    #[serde(alias = "decoder_dmodel")]
    pub text_hidden_size: i32,
    #[serde(alias = "n_mel_bins")]
    pub num_codebooks: i32,
    #[serde(alias = "mel_vocab_size")]
    pub codebook_size: i32,
    #[serde(default)]
    pub bias: bool,
    #[serde(default = "default_true")]
    pub use_audio_norm: bool,
    #[serde(default = "default_audio_mode")]
    pub audio_mode: String,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct VisionArgs {
    #[serde(default = "default_vision_encoder_type")]
    pub vision_encoder_type: String,
    #[serde(alias = "decoder_dmodel")]
    pub text_hidden_size: i32,
    pub patch_size: i32,
    pub temporal_patch_size: i32,
    #[serde(alias = "n_channels")]
    pub num_channels: i32,
    #[serde(alias = "n_layers")]
    pub num_hidden_layers: i32,
    #[serde(default = "default_true")]
    pub use_vision_norm: bool,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
}

#[derive(Debug, Clone)]
/// Global or bounded KV state selected per decoder layer.
pub enum InklingKvCache {
    Global(ConcatKeyValueCache),
    Sliding(SlidingKeyValueCache),
    Paged(PagedKeyValueCache),
}

impl KeyValueCache for InklingKvCache {
    fn offset(&self) -> i32 {
        match self {
            Self::Global(cache) => cache.offset(),
            Self::Sliding(cache) => cache.offset(),
            Self::Paged(cache) => cache.offset(),
        }
    }

    fn max_size(&self) -> Option<i32> {
        match self {
            Self::Global(cache) => cache.max_size(),
            Self::Sliding(cache) => cache.max_size(),
            Self::Paged(cache) => cache.max_size(),
        }
    }

    fn retained_arrays(&self) -> Vec<&Array> {
        match self {
            Self::Global(cache) => cache.retained_arrays(),
            Self::Sliding(cache) => cache.retained_arrays(),
            Self::Paged(cache) => cache.retained_arrays(),
        }
    }

    fn update_and_fetch(
        &mut self,
        keys: Array,
        values: Array,
        stream: &Stream,
    ) -> Result<(Array, Array), Exception> {
        match self {
            Self::Global(cache) => cache.update_and_fetch(keys, values, stream),
            Self::Sliding(cache) => cache.update_and_fetch(keys, values, stream),
            Self::Paged(cache) => cache.update_and_fetch(keys, values, stream),
        }
    }

    fn update_for_attention(
        &mut self,
        keys: Array,
        values: Array,
        stream: &Stream,
    ) -> Result<(Array, Array), Exception> {
        match self {
            Self::Global(cache) => cache.update_for_attention(keys, values, stream),
            Self::Sliding(cache) => cache.update_for_attention(keys, values, stream),
            Self::Paged(cache) => cache.update_for_attention(keys, values, stream),
        }
    }

    fn is_paged(&self) -> bool {
        matches!(self, Self::Paged(_))
    }
}

#[derive(Debug, Clone)]
/// Incremental state for one Inkling decoder layer.
pub struct LayerCache {
    pub kv: InklingKvCache,
    pub convolutions: [CausalConv1dCache; 4],
}

impl LayerCache {
    fn new(local: bool, window: i32) -> Self {
        Self {
            kv: if local {
                InklingKvCache::Sliding(SlidingKeyValueCache::new(window))
            } else {
                InklingKvCache::Global(ConcatKeyValueCache::new())
            },
            convolutions: std::array::from_fn(|_| CausalConv1dCache::default()),
        }
    }

    fn new_paged(
        args: &TextArgs,
        layer: usize,
        manager: CacheResidencyManager,
        rank: Option<CacheRankIdentity>,
    ) -> Result<Self, Exception> {
        let local = args.is_local(layer as i32);
        Ok(Self {
            kv: InklingKvCache::Paged(PagedKeyValueCache::new_with_layout(
                manager,
                layer,
                local.then_some(args.sliding_window_size),
                0,
                rank,
            )?),
            convolutions: std::array::from_fn(|_| CausalConv1dCache::default()),
        })
    }
}

#[derive(Debug, Clone)]
/// Heterogeneous Inkling generation cache.
pub struct Cache {
    pub layers: Vec<LayerCache>,
}

impl Cache {
    pub(crate) fn new(args: &TextArgs) -> Self {
        Self {
            layers: (0..args.num_hidden_layers)
                .map(|layer| LayerCache::new(args.is_local(layer), args.sliding_window_size))
                .collect(),
        }
    }

    pub(crate) fn new_paged(
        args: &TextArgs,
        options: PagedCacheOptions,
        rank: Option<CacheRankIdentity>,
    ) -> Result<Self, Exception> {
        let manager = CacheResidencyManager::new(options)
            .map_err(|error| Exception::custom(error.to_string()))?;
        Self::new_paged_with_manager(args, manager, rank)
    }

    pub(crate) fn new_paged_with_manager(
        args: &TextArgs,
        manager: CacheResidencyManager,
        rank: Option<CacheRankIdentity>,
    ) -> Result<Self, Exception> {
        let layer_count = usize::try_from(args.num_hidden_layers)
            .map_err(|_| Exception::custom("invalid Inkling cache layer count"))?;
        Ok(Self {
            layers: (0..layer_count)
                .map(|layer| LayerCache::new_paged(args, layer, manager.clone(), rank))
                .collect::<Result<Vec<_>, _>>()?,
        })
    }

    pub fn residency_report(&self) -> Result<Option<CacheResidencyReport>, Exception> {
        self.layers
            .iter()
            .find_map(|layer| match &layer.kv {
                InklingKvCache::Paged(cache) => Some(cache.report()),
                _ => None,
            })
            .transpose()
    }

    pub fn offset(&self) -> i32 {
        self.layers.first().map_or(0, |layer| layer.kv.offset())
    }

    pub(crate) fn reset(&mut self) -> Result<(), Exception> {
        let paged_manager = self.layers.iter().find_map(|layer| match &layer.kv {
            InklingKvCache::Paged(cache) => Some(cache.manager().clone()),
            _ => None,
        });
        if let Some(manager) = paged_manager {
            manager
                .clear()
                .map_err(|error| Exception::custom(error.to_string()))?;
        }
        for layer in &mut self.layers {
            match &mut layer.kv {
                InklingKvCache::Global(cache) => cache.clear(),
                InklingKvCache::Sliding(cache) => cache.clear(),
                InklingKvCache::Paged(cache) => cache.reset_local_after_manager_clear(),
            }
            layer.convolutions = std::array::from_fn(|_| CausalConv1dCache::default());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, ModuleParameters)]
struct InklingAttention {
    n_heads: i32,
    n_kv_heads: i32,
    head_dim: i32,
    d_rel: i32,
    rel_extent: i32,
    local: bool,
    sliding_window: i32,
    log_scaling_n_floor: Option<i32>,
    log_scaling_alpha: f32,
    #[param]
    q_proj: MaybeQuantized<nn::Linear>,
    #[param]
    k_proj: MaybeQuantized<nn::Linear>,
    #[param]
    v_proj: MaybeQuantized<nn::Linear>,
    #[param]
    r_proj: MaybeQuantized<nn::Linear>,
    #[param]
    o_proj: MaybeQuantized<nn::Linear>,
    #[param]
    q_norm: nn::RmsNorm,
    #[param]
    k_norm: nn::RmsNorm,
    #[param]
    rel_proj: Param<Array>,
    #[param]
    k_sconv: DepthwiseConv1d,
    #[param]
    v_sconv: DepthwiseConv1d,
}

impl InklingAttention {
    fn new(args: &TextArgs, layer: i32, stream: &Stream) -> Result<Self, Exception> {
        let local = args.is_local(layer);
        let n_heads = args.q_heads(local);
        let n_kv_heads = args.kv_heads(local);
        let head_dim = args.attention_head_dim(local);
        let rel_extent = if local {
            args.sliding_window_size
        } else {
            args.rel_extent
        };
        let prefix = format!("model.layers.{layer}.self_attn");
        Ok(Self {
            n_heads,
            n_kv_heads,
            head_dim,
            d_rel: args.d_rel,
            rel_extent,
            local,
            sliding_window: args.sliding_window_size,
            log_scaling_n_floor: args.log_scaling_n_floor,
            log_scaling_alpha: args.log_scaling_alpha,
            q_proj: common::linear::unloaded_maybe_quantized_linear(
                args.hidden_size,
                n_heads * head_dim,
                false,
                args.weight_quantization_for(&format!("{prefix}.q_proj.weight")),
                stream,
            )?,
            k_proj: common::linear::unloaded_maybe_quantized_linear(
                args.hidden_size,
                n_kv_heads * head_dim,
                false,
                args.weight_quantization_for(&format!("{prefix}.k_proj.weight")),
                stream,
            )?,
            v_proj: common::linear::unloaded_maybe_quantized_linear(
                args.hidden_size,
                n_kv_heads * head_dim,
                false,
                args.weight_quantization_for(&format!("{prefix}.v_proj.weight")),
                stream,
            )?,
            r_proj: common::linear::unloaded_maybe_quantized_linear(
                args.hidden_size,
                n_heads * args.d_rel,
                false,
                args.weight_quantization_for(&format!("{prefix}.r_proj.weight")),
                stream,
            )?,
            o_proj: common::linear::unloaded_maybe_quantized_linear(
                n_heads * head_dim,
                args.hidden_size,
                false,
                args.weight_quantization_for(&format!("{prefix}.o_proj.weight")),
                stream,
            )?,
            q_norm: nn::RmsNorm::unloaded(head_dim, args.rms_norm_eps, Dtype::Float32, stream)?,
            k_norm: nn::RmsNorm::unloaded(head_dim, args.rms_norm_eps, Dtype::Float32, stream)?,
            rel_proj: Param::<Array>::unloaded(&[args.d_rel, rel_extent], Dtype::Float32, stream)?,
            k_sconv: DepthwiseConv1d::new(
                n_kv_heads * head_dim,
                args.sconv_kernel_size,
                false,
                stream,
            )?,
            v_sconv: DepthwiseConv1d::new(
                n_kv_heads * head_dim,
                args.sconv_kernel_size,
                false,
                stream,
            )?,
        })
    }

    fn repeat_kv(&self, states: &Array, stream: &Stream) -> Result<Array, Exception> {
        if self.n_heads == self.n_kv_heads {
            return Ok(states.clone());
        }
        let shape = states.shape();
        let repeats = self.n_heads / self.n_kv_heads;
        broadcast_to(
            &states.reshape(&[shape[0], self.n_kv_heads, 1, shape[2], shape[3]], stream)?,
            &[shape[0], self.n_kv_heads, repeats, shape[2], shape[3]],
            stream,
        )?
        .reshape(&[shape[0], self.n_heads, shape[2], shape[3]], stream)
    }

    fn forward(
        &mut self,
        hidden: &Array,
        cache: Option<&mut LayerCache>,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let batch = hidden.dim(0);
        let seq_len = hidden.dim(1);
        let q_offset = cache.as_ref().map_or(0, |cache| cache.kv.offset());

        let q = self.q_proj.forward(hidden, stream)?;
        let mut k = self.k_proj.forward(hidden, stream)?;
        let mut v = self.v_proj.forward(hidden, stream)?;
        let relative = self.r_proj.forward(hidden, stream)?;

        if let Some(cache) = cache {
            k = short_convolution(&self.k_sconv, &k, Some(&mut cache.convolutions[0]), stream)?;
            v = short_convolution(&self.v_sconv, &v, Some(&mut cache.convolutions[1]), stream)?;
            let q = self
                .q_norm
                .forward(
                    &q.reshape(&[batch, seq_len, self.n_heads, self.head_dim], stream)?,
                    stream,
                )?
                .transpose_axes(&[0, 2, 1, 3], stream)?;
            let k = self
                .k_norm
                .forward(
                    &k.reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim], stream)?,
                    stream,
                )?
                .transpose_axes(&[0, 2, 1, 3], stream)?;
            let v = v
                .reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim], stream)?
                .transpose_axes(&[0, 2, 1, 3], stream)?;
            if cache.kv.is_paged() {
                cache.kv.update_for_attention(k, v, stream)?;
                let InklingKvCache::Paged(cache) = &cache.kv else {
                    unreachable!("paged Inkling cache variant checked above")
                };
                self.attend_paged(q, &relative, batch, seq_len, q_offset, cache, stream)
            } else {
                let (k, v) = cache.kv.update_and_fetch(k, v, stream)?;
                let key_len = k.dim(2);
                let key_offset = q_offset + seq_len - key_len;
                self.attend_chunked(
                    q, k, v, &relative, batch, seq_len, q_offset, key_offset, stream,
                )
            }
        } else {
            k = short_convolution(&self.k_sconv, &k, None, stream)?;
            v = short_convolution(&self.v_sconv, &v, None, stream)?;
            let q = self
                .q_norm
                .forward(
                    &q.reshape(&[batch, seq_len, self.n_heads, self.head_dim], stream)?,
                    stream,
                )?
                .transpose_axes(&[0, 2, 1, 3], stream)?;
            let k = self
                .k_norm
                .forward(
                    &k.reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim], stream)?,
                    stream,
                )?
                .transpose_axes(&[0, 2, 1, 3], stream)?;
            let v = v
                .reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim], stream)?
                .transpose_axes(&[0, 2, 1, 3], stream)?;
            self.attend_chunked(q, k, v, &relative, batch, seq_len, 0, 0, stream)
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn attend_chunked(
        &mut self,
        q: Array,
        k: Array,
        v: Array,
        relative: &Array,
        batch: i32,
        query_len: i32,
        query_offset: i32,
        key_offset: i32,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        // Bound the eagerly materialized score/bias tensors. Local layers also
        // discard keys outside the earliest query's window for each chunk.
        const TARGET_SCORE_ELEMENTS: i32 = 16 * 1024 * 1024;
        let total_key_len = k.dim(2).max(1);
        let chunk_size = if self.local {
            256
        } else {
            (TARGET_SCORE_ELEMENTS / (self.n_heads * total_key_len)).clamp(1, 256)
        };
        let key_limit = key_offset + k.dim(2);
        let mut outputs = Vec::new();
        let mut start = 0;
        while start < query_len {
            let end = (start + chunk_size).min(query_len);
            let query_abs_start = query_offset + start;
            let query_abs_end = query_offset + end;
            let chunk_key_start = if self.local {
                (query_abs_start - self.sliding_window + 1).max(key_offset)
            } else {
                key_offset
            };
            let chunk_key_end = query_abs_end.min(key_limit);
            let key_start_index = chunk_key_start - key_offset;
            let key_end_index = chunk_key_end - key_offset;
            let mut q_chunk = q.try_index_device((.., .., start..end, ..), stream)?;
            let k_chunk =
                k.try_index_device((.., .., key_start_index..key_end_index, ..), stream)?;
            let v_chunk =
                v.try_index_device((.., .., key_start_index..key_end_index, ..), stream)?;
            let relative_chunk = relative.try_index_device((.., start..end, ..), stream)?;
            let (bias, mask, tau) = self.position_data(
                &relative_chunk,
                batch,
                end - start,
                key_end_index - key_start_index,
                query_abs_start,
                chunk_key_start,
                stream,
            )?;
            if let Some(tau) = tau {
                q_chunk = q_chunk.multiply(tau, stream)?;
            }
            outputs.push(self.attend(
                q_chunk,
                k_chunk,
                v_chunk,
                bias,
                mask,
                batch,
                end - start,
                stream,
            )?);
            start = end;
        }
        concatenate_axis(&outputs, 1, stream)
    }

    #[allow(clippy::too_many_arguments)]
    fn attend_paged(
        &mut self,
        q: Array,
        relative: &Array,
        batch: i32,
        query_len: i32,
        query_offset: i32,
        cache: &PagedKeyValueCache,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        const TARGET_SCORE_ELEMENTS: i32 = 16 * 1024 * 1024;
        let chunk_size = if self.local {
            256
        } else {
            (TARGET_SCORE_ELEMENTS / (self.n_heads * cache.offset().max(1))).clamp(1, 256)
        };
        let mut outputs = Vec::new();
        let mut scanned_blocks = 0;
        let mut scanned_bytes = 0;
        let mut scratch_bytes = 0;
        let mut start = 0;
        while start < query_len {
            let end = (start + chunk_size).min(query_len);
            let query_abs_start = query_offset + start;
            let mut q_chunk = q.try_index_device((.., .., start..end, ..), stream)?;
            let relative_chunk = relative.try_index_device((.., start..end, ..), stream)?;
            if let Some(tau) = self.global_query_scale(query_abs_start, end - start, stream)? {
                q_chunk = q_chunk.multiply(tau, stream)?;
            }
            let mut accumulator = BlockwiseAttentionAccumulator::new(
                &q_chunk,
                1.0 / self.head_dim as f32,
                None,
                query_abs_start as i64,
                self.local.then_some(self.sliding_window),
                0,
                None,
                cache.offset() as i64,
                stream,
            )?;
            let visible_start = if self.local {
                (query_abs_start - self.sliding_window + 1).max(0) as i64
            } else {
                0
            };
            let (blocks, bytes) = cache.visit_attention_blocks(
                visible_start,
                cache.offset() as i64,
                stream,
                |block| {
                    let (bias, _, _) = self.position_data(
                        &relative_chunk,
                        batch,
                        end - start,
                        i32::try_from(block.end - block.start).map_err(|_| {
                            Exception::custom("Inkling paged cache block length overflow")
                        })?,
                        query_abs_start,
                        i32::try_from(block.start).map_err(|_| {
                            Exception::custom("Inkling paged cache position overflow")
                        })?,
                        stream,
                    )?;
                    accumulator.accumulate_with_bias(block, Some(&bias), stream)
                },
            )?;
            scanned_blocks += blocks;
            scanned_bytes += bytes;
            scratch_bytes = scratch_bytes.max(
                batch as u64
                    * self.n_heads as u64
                    * (end - start) as u64
                    * cache.manager().options().block_size_tokens() as u64
                    * 4,
            );
            outputs.push(accumulator.finish(stream)?);
            start = end;
        }
        cache.record_architecture_attention_scan(
            query_len > 1,
            scanned_blocks,
            scanned_bytes,
            scratch_bytes,
        )?;
        let attended = concatenate_axis(&outputs, 2, stream)?
            .transpose_axes(&[0, 2, 1, 3], stream)?
            .reshape(&[batch, query_len, self.n_heads * self.head_dim], stream)?;
        self.o_proj.forward(&attended, stream)
    }

    fn global_query_scale(
        &self,
        query_offset: i32,
        query_len: i32,
        stream: &Stream,
    ) -> Result<Option<Array>, Exception> {
        if self.local {
            return Ok(None);
        }
        let Some(floor) = self.log_scaling_n_floor else {
            return Ok(None);
        };
        let positions =
            arange::<i32, i32>(query_offset + 1, query_offset + query_len + 1, 1, stream)?
                .as_dtype(Dtype::Float32, stream)?;
        let ratio = positions.divide(Array::from_f32(floor as f32), stream)?;
        let ratio = safemlx::ops::maximum(ratio, Array::from_f32(1.0), stream)?;
        Ok(Some(
            ratio
                .log(stream)?
                .multiply(Array::from_f32(self.log_scaling_alpha), stream)?
                .add(Array::from_f32(1.0), stream)?
                .reshape(&[1, 1, query_len, 1], stream)?,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn position_data(
        &self,
        relative: &Array,
        batch: i32,
        query_len: i32,
        key_len: i32,
        query_offset: i32,
        key_offset: i32,
        stream: &Stream,
    ) -> Result<(Array, Array, Option<Array>), Exception> {
        let relative = relative.reshape(&[batch, query_len, self.n_heads, self.d_rel], stream)?;
        let profiles = matmul(relative, self.rel_proj.as_ref(), stream)?
            .transpose_axes(&[0, 2, 1, 3], stream)?;
        let q_positions = arange::<i32, i32>(query_offset, query_offset + query_len, 1, stream)?
            .try_index_device((.., NewAxis), stream)?;
        let k_positions = arange::<i32, i32>(key_offset, key_offset + key_len, 1, stream)?
            .try_index_device((NewAxis, ..), stream)?;
        let distances = q_positions.subtract(k_positions, stream)?;
        let mut valid = distances.ge(Array::from_int(0), stream)?;
        if self.local {
            valid = valid.logical_and(
                &distances.lt(Array::from_int(self.sliding_window), stream)?,
                stream,
            )?;
        }
        let gather = clip(&distances, (0, self.rel_extent - 1), stream)?
            .as_dtype(Dtype::Int32, stream)?
            .try_index_device((NewAxis, NewAxis, .., ..), stream)?;
        let gather = broadcast_to(&gather, &[batch, self.n_heads, query_len, key_len], stream)?;
        let mut bias = take_along_axis(&profiles, &gather, -1, stream)?;
        let relative_valid = distances.ge(Array::from_int(0), stream)?.logical_and(
            &distances.lt(Array::from_int(self.rel_extent), stream)?,
            stream,
        )?;
        bias = r#where(&relative_valid, bias, Array::from_f32(0.0), stream)?;
        let tau = if !self.local {
            if let Some(tau) = self.global_query_scale(query_offset, query_len, stream)? {
                bias = bias.multiply(&tau, stream)?;
                Some(tau)
            } else {
                None
            }
        } else {
            None
        };
        Ok((bias, valid, tau))
    }

    #[allow(clippy::too_many_arguments)]
    fn attend(
        &mut self,
        q: Array,
        k: Array,
        v: Array,
        bias: Array,
        valid: Array,
        batch: i32,
        seq_len: i32,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let k = self.repeat_kv(&k, stream)?;
        let v = self.repeat_kv(&v, stream)?;
        let mut scores = matmul(
            &q.multiply(Array::from_f32(1.0 / self.head_dim as f32), stream)?,
            &k.swap_axes(-1, -2, stream)?,
            stream,
        )?
        .add(bias, stream)?;
        scores = r#where(&valid, scores, Array::from_f32(f32::NEG_INFINITY), stream)?;
        let probabilities = softmax_axis(scores, -1, true, stream)?;
        let attended = matmul(probabilities, v, stream)?
            .transpose_axes(&[0, 2, 1, 3], stream)?
            .reshape(&[batch, seq_len, self.n_heads * self.head_dim], stream)?;
        self.o_proj.forward(&attended, stream)
    }
}

fn short_convolution(
    convolution: &DepthwiseConv1d,
    input: &Array,
    cache: Option<&mut CausalConv1dCache>,
    stream: &Stream,
) -> Result<Array, Exception> {
    let dtype = input.dtype();
    let input = input.as_dtype(Dtype::Float32, stream)?;
    causal_depthwise_conv1d(convolution, &input, cache, stream)?
        .add(&input, stream)?
        .as_dtype(dtype, stream)
}

#[derive(Debug, Clone, ModuleParameters)]
struct InklingRouter {
    num_routed: i32,
    num_shared: i32,
    top_k: i32,
    route_scale: f32,
    #[param]
    weight: Param<Array>,
    #[param]
    bias: Param<Array>,
    #[param]
    global_scale: Param<Array>,
}

impl InklingRouter {
    fn new(args: &TextArgs, stream: &Stream) -> Result<Self, Exception> {
        Ok(Self {
            num_routed: args.n_routed_experts,
            num_shared: args.n_shared_experts,
            top_k: args.num_experts_per_tok,
            route_scale: args.route_scale,
            weight: Param::<Array>::unloaded(
                &[
                    args.n_routed_experts + args.n_shared_experts,
                    args.hidden_size,
                ],
                Dtype::Float32,
                stream,
            )?,
            bias: Param::<Array>::unloaded(&[args.n_routed_experts], Dtype::Float32, stream)?,
            global_scale: Param::<Array>::unloaded(&[1], Dtype::Float32, stream)?,
        })
    }

    fn forward(&self, hidden: &Array, stream: &Stream) -> Result<(Array, Array, Array), Exception> {
        let flat = hidden.reshape(&[-1, hidden.dim(-1)], stream)?;
        let logits = matmul(&flat, &self.weight.as_ref().transpose(stream)?, stream)?;
        let routed = logits.try_index_device((.., ..self.num_routed), stream)?;
        let shared = logits.try_index_device((.., self.num_routed..), stream)?;
        let choice = sigmoid(&routed, stream)?.add(self.bias.as_ref(), stream)?;
        let indices = argpartition_axis(choice, -self.top_k, -1, stream)?
            .try_index_device((.., -self.top_k..), stream)?;
        let selected_logits = take_along_axis(&routed, &indices, -1, stream)?;
        let all_logits = concatenate_axis(&[selected_logits, shared], -1, stream)?;
        let weights = softmax_axis(nn::log_sigmoid(all_logits, stream)?, -1, true, stream)?
            .multiply(Array::from_f32(self.route_scale), stream)?
            .multiply(self.global_scale.as_ref(), stream)?;
        let routed_weights = weights.try_index_device((.., ..self.top_k), stream)?;
        let shared_weights = weights.try_index_device((.., self.top_k..), stream)?;
        Ok((indices, routed_weights, shared_weights))
    }
}

#[derive(Debug, Clone, ModuleParameters)]
struct InklingMoe {
    #[param]
    router: InklingRouter,
    #[param]
    experts: PackedSwiGluExperts,
    #[param]
    shared_experts: PackedSwiGluExperts,
}

impl InklingMoe {
    fn new(args: &TextArgs, layer: i32, stream: &Stream) -> Result<Self, Exception> {
        let intermediate = args.moe_intermediate_size();
        let prefix = format!("model.layers.{layer}.moe");
        Ok(Self {
            router: InklingRouter::new(args, stream)?,
            experts: PackedSwiGluExperts::new(
                args.n_routed_experts,
                args.hidden_size,
                intermediate,
                args.weight_quantization_for(&format!("{prefix}.experts.gate_up_proj")),
                args.weight_quantization_for(&format!("{prefix}.experts.down_proj")),
                stream,
            )?,
            shared_experts: PackedSwiGluExperts::new(
                args.n_shared_experts,
                args.hidden_size,
                intermediate,
                args.weight_quantization_for(&format!("{prefix}.shared_experts.gate_up_proj")),
                args.weight_quantization_for(&format!("{prefix}.shared_experts.down_proj")),
                stream,
            )?,
        })
    }

    fn forward(&mut self, hidden: &Array, stream: &Stream) -> Result<Array, Exception> {
        let shape = hidden.shape().to_vec();
        let flat = hidden.reshape(&[-1, hidden.dim(-1)], stream)?;
        let (indices, weights, shared_weights) = self.router.forward(hidden, stream)?;
        let routed = self.experts.forward(&flat, &indices, &weights, stream)?;
        let tokens = flat.dim(0);
        let shared_indices = broadcast_to(
            &arange::<i32, i32>(0, self.router.num_shared, 1, stream)?
                .try_index_device((NewAxis, ..), stream)?,
            &[tokens, self.router.num_shared],
            stream,
        )?;
        let shared =
            self.shared_experts
                .forward(&flat, &shared_indices, &shared_weights, stream)?;
        routed.add(shared, stream)?.reshape(&shape, stream)
    }

    fn forward_with_expert_executor<F>(
        &mut self,
        hidden: &Array,
        stream: &Stream,
        execute: F,
    ) -> Result<Array, Exception>
    where
        F: FnOnce(&Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        let shape = hidden.shape().to_vec();
        let flat = hidden.reshape(&[-1, hidden.dim(-1)], stream)?;
        let (indices, weights, shared_weights) = self.router.forward(hidden, stream)?;
        let routed = execute(&flat, &indices, &weights, stream)?;
        let tokens = flat.dim(0);
        let shared_indices = broadcast_to(
            &arange::<i32, i32>(0, self.router.num_shared, 1, stream)?
                .try_index_device((NewAxis, ..), stream)?,
            &[tokens, self.router.num_shared],
            stream,
        )?;
        let shared =
            self.shared_experts
                .forward(&flat, &shared_indices, &shared_weights, stream)?;
        routed.add(shared, stream)?.reshape(&shape, stream)
    }
}

#[derive(Debug, Clone, ModuleParameters)]
pub(crate) struct DecoderLayer {
    #[param]
    input_layernorm: nn::RmsNorm,
    #[param]
    self_attn: InklingAttention,
    #[param]
    attn_sconv: DepthwiseConv1d,
    #[param]
    post_attention_layernorm: nn::RmsNorm,
    #[param]
    dense: Option<SwiGluMlp>,
    #[param]
    dense_global_scale: Param<Option<Array>>,
    #[param]
    moe: Option<InklingMoe>,
    #[param]
    mlp_sconv: DepthwiseConv1d,
}

impl DecoderLayer {
    pub(crate) fn new(args: &TextArgs, layer: i32, stream: &Stream) -> Result<Self, Exception> {
        let dense = args.is_dense(layer);
        Ok(Self {
            input_layernorm: nn::RmsNorm::unloaded(
                args.hidden_size,
                args.rms_norm_eps,
                Dtype::Float32,
                stream,
            )?,
            self_attn: InklingAttention::new(args, layer, stream)?,
            attn_sconv: DepthwiseConv1d::new(
                args.hidden_size,
                args.sconv_kernel_size,
                false,
                stream,
            )?,
            post_attention_layernorm: nn::RmsNorm::unloaded(
                args.hidden_size,
                args.rms_norm_eps,
                Dtype::Float32,
                stream,
            )?,
            dense: if dense {
                let prefix = format!("model.layers.{layer}.dense");
                Some(SwiGluMlp {
                    gate_proj: common::linear::unloaded_maybe_quantized_linear(
                        args.hidden_size,
                        args.dense_intermediate_size(),
                        false,
                        args.weight_quantization_for(&format!("{prefix}.gate_proj.weight")),
                        stream,
                    )?,
                    down_proj: common::linear::unloaded_maybe_quantized_linear(
                        args.dense_intermediate_size(),
                        args.hidden_size,
                        false,
                        args.weight_quantization_for(&format!("{prefix}.down_proj.weight")),
                        stream,
                    )?,
                    up_proj: common::linear::unloaded_maybe_quantized_linear(
                        args.hidden_size,
                        args.dense_intermediate_size(),
                        false,
                        args.weight_quantization_for(&format!("{prefix}.up_proj.weight")),
                        stream,
                    )?,
                })
            } else {
                None
            },
            dense_global_scale: if dense {
                Param::<Option<Array>>::unloaded_some(&[1], Dtype::Float32, stream)?
            } else {
                Param::new(None)
            },
            moe: if dense {
                None
            } else {
                Some(InklingMoe::new(args, layer, stream)?)
            },
            mlp_sconv: DepthwiseConv1d::new(
                args.hidden_size,
                args.sconv_kernel_size,
                false,
                stream,
            )?,
        })
    }

    pub(crate) fn forward(
        &mut self,
        hidden: &Array,
        cache: Option<&mut LayerCache>,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        match cache {
            Some(cache) => {
                let normalized = self.input_layernorm.forward(hidden, stream)?;
                let attention = self.self_attn.forward(&normalized, Some(cache), stream)?;
                let attention = short_convolution(
                    &self.attn_sconv,
                    &attention,
                    Some(&mut cache.convolutions[2]),
                    stream,
                )?;
                let hidden = hidden.add(attention, stream)?;
                let normalized = self.post_attention_layernorm.forward(&hidden, stream)?;
                let mlp = self.forward_mlp(&normalized, stream)?;
                let mlp = short_convolution(
                    &self.mlp_sconv,
                    &mlp,
                    Some(&mut cache.convolutions[3]),
                    stream,
                )?;
                hidden.add(mlp, stream)
            }
            None => {
                let normalized = self.input_layernorm.forward(hidden, stream)?;
                let attention = self.self_attn.forward(&normalized, None, stream)?;
                let attention = short_convolution(&self.attn_sconv, &attention, None, stream)?;
                let hidden = hidden.add(attention, stream)?;
                let normalized = self.post_attention_layernorm.forward(&hidden, stream)?;
                let mlp = self.forward_mlp(&normalized, stream)?;
                let mlp = short_convolution(&self.mlp_sconv, &mlp, None, stream)?;
                hidden.add(mlp, stream)
            }
        }
    }

    pub(crate) fn forward_with_expert_executor<F>(
        &mut self,
        hidden: &Array,
        cache: Option<&mut LayerCache>,
        stream: &Stream,
        execute: F,
    ) -> Result<Array, Exception>
    where
        F: FnOnce(&Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        match cache {
            Some(cache) => {
                let normalized = self.input_layernorm.forward(hidden, stream)?;
                let attention = self.self_attn.forward(&normalized, Some(cache), stream)?;
                let attention = short_convolution(
                    &self.attn_sconv,
                    &attention,
                    Some(&mut cache.convolutions[2]),
                    stream,
                )?;
                let hidden = hidden.add(attention, stream)?;
                let normalized = self.post_attention_layernorm.forward(&hidden, stream)?;
                let mlp = self.forward_mlp_with_expert_executor(&normalized, stream, execute)?;
                let mlp = short_convolution(
                    &self.mlp_sconv,
                    &mlp,
                    Some(&mut cache.convolutions[3]),
                    stream,
                )?;
                hidden.add(mlp, stream)
            }
            None => {
                let normalized = self.input_layernorm.forward(hidden, stream)?;
                let attention = self.self_attn.forward(&normalized, None, stream)?;
                let attention = short_convolution(&self.attn_sconv, &attention, None, stream)?;
                let hidden = hidden.add(attention, stream)?;
                let normalized = self.post_attention_layernorm.forward(&hidden, stream)?;
                let mlp = self.forward_mlp_with_expert_executor(&normalized, stream, execute)?;
                let mlp = short_convolution(&self.mlp_sconv, &mlp, None, stream)?;
                hidden.add(mlp, stream)
            }
        }
    }

    fn forward_mlp(&mut self, hidden: &Array, stream: &Stream) -> Result<Array, Exception> {
        if let Some(dense) = &mut self.dense {
            let output = dense.forward(hidden, stream)?;
            return match self.dense_global_scale.as_ref() {
                Some(scale) => output.multiply(scale, stream),
                None => Ok(output),
            };
        }
        self.moe
            .as_mut()
            .expect("validated sparse layer")
            .forward(hidden, stream)
    }

    fn forward_mlp_with_expert_executor<F>(
        &mut self,
        hidden: &Array,
        stream: &Stream,
        execute: F,
    ) -> Result<Array, Exception>
    where
        F: FnOnce(&Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        if let Some(dense) = &mut self.dense {
            let output = dense.forward(hidden, stream)?;
            return match self.dense_global_scale.as_ref() {
                Some(scale) => output.multiply(scale, stream),
                None => Ok(output),
            };
        }
        self.moe
            .as_mut()
            .expect("validated sparse layer")
            .forward_with_expert_executor(hidden, stream, execute)
    }
}

#[derive(Debug, Clone, ModuleParameters)]
struct TextModel {
    #[param]
    embed_tokens: MaybeQuantized<nn::Embedding>,
    #[param]
    embed_norm: nn::RmsNorm,
    #[param]
    layers: Vec<DecoderLayer>,
    #[param]
    norm: nn::RmsNorm,
}

impl TextModel {
    fn new(args: &TextArgs, stream: &Stream) -> Result<Self, Exception> {
        Ok(Self {
            embed_tokens: common::linear::unloaded_maybe_quantized_embedding(
                args.vocab_size,
                args.hidden_size,
                args.weight_quantization_for("model.embed_tokens.weight"),
                stream,
            )?,
            embed_norm: nn::RmsNorm::unloaded(
                args.hidden_size,
                args.rms_norm_eps,
                Dtype::Float32,
                stream,
            )?,
            layers: (0..args.num_hidden_layers)
                .map(|layer| DecoderLayer::new(args, layer, stream))
                .collect::<Result<Vec<_>, _>>()?,
            norm: nn::RmsNorm::unloaded(
                args.hidden_size,
                args.rms_norm_eps,
                Dtype::Float32,
                stream,
            )?,
        })
    }

    fn embed(&mut self, tokens: &Array, stream: &Stream) -> Result<Array, Exception> {
        let embedded = self.embed_tokens.forward(tokens, stream)?;
        self.embed_norm.forward(&embedded, stream)
    }

    fn forward(
        &mut self,
        tokens: &Array,
        inputs_embeds: Option<&Array>,
        cache: Option<&mut Cache>,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let mut hidden = match inputs_embeds {
            Some(embeddings) => embeddings.clone(),
            None => self.embed(tokens, stream)?,
        };
        if let Some(cache) = cache {
            for (layer, cache) in self.layers.iter_mut().zip(cache.layers.iter_mut()) {
                hidden = layer.forward(&hidden, Some(cache), stream)?;
            }
        } else {
            for layer in &mut self.layers {
                hidden = layer.forward(&hidden, None, stream)?;
            }
        }
        self.norm.forward(&hidden, stream)
    }

    fn forward_with_expert_executor<F>(
        &mut self,
        tokens: &Array,
        cache: &mut Cache,
        mut execute: F,
        stream: &Stream,
    ) -> Result<Array, Exception>
    where
        F: FnMut(usize, &Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        let mut hidden = self.embed(tokens, stream)?;
        for (index, (layer, layer_cache)) in self
            .layers
            .iter_mut()
            .zip(cache.layers.iter_mut())
            .enumerate()
        {
            hidden = if layer.moe.is_some() {
                layer.forward_with_expert_executor(
                    &hidden,
                    Some(layer_cache),
                    stream,
                    |flat, ids, weights, stream| execute(index, flat, ids, weights, stream),
                )?
            } else {
                layer.forward(&hidden, Some(layer_cache), stream)?
            };
        }
        self.norm.forward(&hidden, stream)
    }
}

#[derive(Debug, Clone, ModuleParameters)]
pub(crate) struct AudioModel {
    num_codebooks: i32,
    codebook_size: i32,
    #[param]
    encoder: nn::Embedding,
    #[param]
    final_norm: nn::RmsNorm,
}

impl AudioModel {
    pub(crate) fn new(args: &AudioArgs, stream: &Stream) -> Result<Self, Exception> {
        Ok(Self {
            num_codebooks: args.num_codebooks,
            codebook_size: args.codebook_size,
            encoder: nn::Embedding::unloaded(
                args.num_codebooks * args.codebook_size,
                args.text_hidden_size,
                Dtype::Float32,
                stream,
            )?,
            final_norm: nn::RmsNorm::unloaded(
                args.text_hidden_size,
                args.rms_norm_eps,
                Dtype::Float32,
                stream,
            )?,
        })
    }

    pub(crate) fn forward(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let input_ids = match input_ids.ndim() {
            2 => input_ids.try_index_device(NewAxis, stream)?,
            3 if input_ids.dim(0) == 1 => input_ids.clone(),
            _ => {
                return Err(Exception::custom(format!(
                    "Inkling audio IDs must be [frames, {}] or [1, frames, {}], got {:?}",
                    self.num_codebooks,
                    self.num_codebooks,
                    input_ids.shape()
                )))
            }
        };
        if input_ids.dim(-1) != self.num_codebooks {
            return Err(Exception::custom(format!(
                "Inkling audio IDs require {} dMel codebooks, got {:?}",
                self.num_codebooks,
                input_ids.shape()
            )));
        }
        let offsets = arange::<i32, i32>(
            0,
            self.num_codebooks * self.codebook_size,
            self.codebook_size,
            stream,
        )?
        .reshape(&[1, 1, self.num_codebooks], stream)?;
        let indices = input_ids
            .as_dtype(Dtype::Int32, stream)?
            .add(offsets, stream)?;
        let embedded = self.encoder.forward(&indices, stream)?;
        let mut embedded = sum_axis(&embedded, -2, false, stream)?;
        embedded = self.final_norm.forward(&embedded, stream)?;
        if let Some(mask) = mask {
            if mask.ndim() != 2 || mask.dim(0) != 1 || mask.dim(1) != embedded.dim(1) {
                return Err(Exception::custom(format!(
                    "Inkling audio mask must be [1, frames], got {:?}",
                    mask.shape()
                )));
            }
            let valid = mask.sum(None, stream)?.item::<i32>(stream);
            embedded = embedded.try_index_device((.., ..valid, ..), stream)?;
        }
        Ok(embedded)
    }
}

#[derive(Debug, Clone, ModuleParameters)]
pub(crate) struct VisionLayer {
    t_fold: i32,
    hw_fold: i32,
    #[param]
    projection: nn::Linear,
    #[param]
    layer_norm: Option<nn::RmsNorm>,
}

impl VisionLayer {
    pub(crate) fn new(
        input_dim: i32,
        output_dim: i32,
        t_fold: i32,
        hw_fold: i32,
        add_norm: bool,
        eps: f32,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        Ok(Self {
            t_fold,
            hw_fold,
            projection: nn::Linear::unloaded(input_dim, output_dim, false, Dtype::Float32, stream)?,
            layer_norm: add_norm
                .then(|| nn::RmsNorm::unloaded(output_dim, eps, Dtype::Float32, stream))
                .transpose()?,
        })
    }

    pub(crate) fn forward(&mut self, hidden: &Array, stream: &Stream) -> Result<Array, Exception> {
        let mut hidden = if self.t_fold > 1 || self.hw_fold > 1 {
            let shape = hidden.shape();
            if shape.len() != 5
                || shape[1] % self.t_fold != 0
                || shape[2] % self.hw_fold != 0
                || shape[3] % self.hw_fold != 0
            {
                return Err(Exception::custom(format!(
                    "Inkling hMLP fold ({}, {}) is incompatible with {:?}",
                    self.t_fold, self.hw_fold, shape
                )));
            }
            let (batch, time, height, width, channels) =
                (shape[0], shape[1], shape[2], shape[3], shape[4]);
            hidden
                .reshape(
                    &[
                        batch,
                        time / self.t_fold,
                        self.t_fold,
                        height / self.hw_fold,
                        self.hw_fold,
                        width / self.hw_fold,
                        self.hw_fold,
                        channels,
                    ],
                    stream,
                )?
                .transpose_axes(&[0, 1, 3, 5, 2, 4, 6, 7], stream)?
                .reshape(
                    &[
                        batch,
                        time / self.t_fold,
                        height / self.hw_fold,
                        width / self.hw_fold,
                        self.t_fold * self.hw_fold * self.hw_fold * channels,
                    ],
                    stream,
                )?
        } else {
            hidden.clone()
        };
        hidden = self.projection.forward(&hidden, stream)?;
        if let Some(norm) = &mut self.layer_norm {
            hidden = norm.forward(&hidden, stream)?;
            hidden = nn::gelu(hidden, stream)?;
        }
        Ok(hidden)
    }
}

#[derive(Debug, Clone, ModuleParameters)]
pub(crate) struct VisionModel {
    pub(crate) text_hidden_size: i32,
    #[param]
    pub(crate) layers: Vec<VisionLayer>,
    #[param]
    pub(crate) final_norm: nn::RmsNorm,
}

impl VisionModel {
    pub(crate) fn new(args: &VisionArgs, stream: &Stream) -> Result<Self, Error> {
        if (
            args.temporal_patch_size,
            args.patch_size,
            args.num_hidden_layers,
            args.num_channels,
        ) != (2, 40, 4, 3)
        {
            return Err(Error::UnsupportedArchitecture(format!(
                "Inkling hMLP currently supports the released (temporal_patch_size=2, patch_size=40, n_layers=4, channels=3) tower, got ({}, {}, {}, {})",
                args.temporal_patch_size, args.patch_size, args.num_hidden_layers, args.num_channels
            )));
        }
        // `plan_out_scales` for the released tower selects reduction scales
        // [1, 25, 100, 1600, 3200].
        let specs = [
            (75, 128, 1, 5),
            (512, 512, 1, 2),
            (8192, 4800, 1, 4),
            (9600, args.text_hidden_size, 2, 1),
        ];
        let mut layers = Vec::with_capacity(specs.len());
        for (index, (input_dim, output_dim, t_fold, hw_fold)) in specs.into_iter().enumerate() {
            layers.push(VisionLayer::new(
                input_dim,
                output_dim,
                t_fold,
                hw_fold,
                index + 1 != specs.len(),
                args.rms_norm_eps,
                stream,
            )?);
        }
        Ok(Self {
            text_hidden_size: args.text_hidden_size,
            layers,
            final_norm: nn::RmsNorm::unloaded(
                args.text_hidden_size,
                args.rms_norm_eps,
                Dtype::Float32,
                stream,
            )?,
        })
    }

    fn forward(&mut self, pixels: &Array, stream: &Stream) -> Result<Array, Exception> {
        if pixels.ndim() != 5 || pixels.shape()[1..] != [2, 40, 40, 3] {
            return Err(Exception::custom(format!(
                "Inkling image patches must be [patches, 2, 40, 40, 3], got {:?}",
                pixels.shape()
            )));
        }
        let mut hidden = pixels.clone();
        for layer in &mut self.layers {
            hidden = layer.forward(&hidden, stream)?;
        }
        hidden = self.final_norm.forward(&hidden, stream)?;
        hidden
            .reshape(&[-1, self.text_hidden_size], stream)?
            .try_index_device(NewAxis, stream)
    }
}

#[derive(Debug, Clone, ModuleParameters)]
/// Inkling causal language model.
pub struct Model {
    pub args: ModelArgs,
    #[param]
    model: TextModel,
    #[param]
    audio: Option<AudioModel>,
    #[param]
    visual: Option<VisionModel>,
    #[param]
    lm_head: MaybeQuantized<nn::Linear>,
}

impl Model {
    pub fn new(args: ModelArgs, stream: &Stream) -> Result<Self, Error> {
        validate_args(&args)?;
        Ok(Self {
            model: TextModel::new(&args.text_config, stream)?,
            audio: args
                .audio_config
                .as_ref()
                .map(|config| AudioModel::new(config, stream))
                .transpose()?,
            visual: args
                .vision_config
                .as_ref()
                .map(|config| VisionModel::new(config, stream))
                .transpose()?,
            lm_head: common::linear::unloaded_maybe_quantized_linear(
                args.text_config.hidden_size,
                args.text_config.vocab_size,
                false,
                args.text_config.weight_quantization_for("lm_head.weight"),
                stream,
            )?,
            args,
        })
    }

    pub fn model_type(&self) -> &str {
        &self.args.model_type
    }

    pub fn new_cache(&self) -> Cache {
        Cache::new(&self.args.text_config)
    }

    pub fn new_paged_cache(&self, options: PagedCacheOptions) -> Result<Cache, Exception> {
        Cache::new_paged(&self.args.text_config, options, None)
    }

    pub(crate) fn new_paged_cache_with_manager(
        &self,
        manager: CacheResidencyManager,
        rank: Option<CacheRankIdentity>,
    ) -> Result<Cache, Exception> {
        Cache::new_paged_with_manager(&self.args.text_config, manager, rank)
    }

    pub(crate) fn forward_logits(
        &mut self,
        tokens: &Array,
        inputs_embeds: Option<&Array>,
        cache: Option<&mut Cache>,
        last_token_only: bool,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let mut hidden = self.model.forward(tokens, inputs_embeds, cache, stream)?;
        if last_token_only {
            hidden = hidden.try_index_device((.., -1, ..), stream)?;
        }
        hidden = hidden.divide(
            Array::from_f32(self.args.text_config.logits_mup_width_multiplier),
            stream,
        )?;
        let mut logits = self.lm_head.forward(&hidden, stream)?;
        if let Some(size) = self.args.text_config.unpadded_vocab_size {
            if size < logits.dim(-1) {
                logits = match logits.ndim() {
                    2 => logits.try_index_device((.., ..size), stream)?,
                    3 => logits.try_index_device((.., .., ..size), stream)?,
                    rank => {
                        return Err(Exception::custom(format!(
                            "Inkling logits have unsupported rank {rank}"
                        )))
                    }
                };
            }
        }
        Ok(logits)
    }

    pub(crate) fn forward_cached_expert_parallel<F>(
        &mut self,
        tokens: &Array,
        cache: &mut Cache,
        execute: F,
        stream: &Stream,
    ) -> Result<Array, Exception>
    where
        F: FnMut(usize, &Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        let mut hidden = self
            .model
            .forward_with_expert_executor(tokens, cache, execute, stream)?;
        hidden = hidden.divide(
            Array::from_f32(self.args.text_config.logits_mup_width_multiplier),
            stream,
        )?;
        let mut logits = self.lm_head.forward(&hidden, stream)?;
        if let Some(size) = self.args.text_config.unpadded_vocab_size {
            if size < logits.dim(-1) {
                logits = logits.try_index_device((.., .., ..size), stream)?;
            }
        }
        Ok(logits)
    }

    fn prepare_typed_prefill(
        &mut self,
        input: input::ModelInput<'_>,
        stream: &Stream,
    ) -> Result<input::PreparedPrefill, Exception> {
        let modality_tokens = [
            input::ModalityToken {
                modality: input::Modality::Image,
                token_id: self.args.image_token_id,
            },
            input::ModalityToken {
                modality: input::Modality::Audio,
                token_id: self.args.audio_token_id,
            },
        ];
        let embed_tokens = &mut self.model;
        let audio = &mut self.audio;
        let visual = &mut self.visual;
        input::prepare_decoder_prefill(
            input,
            &modality_tokens,
            self.args.text_config.hidden_size,
            "Inkling",
            stream,
            |tokens, stream| embed_tokens.embed(tokens, stream),
            |part, stream| match (part.modality, part.payload) {
                (_, input::InputPayload::Embeddings(embeddings)) => Ok(vec![embeddings.clone()]),
                (input::Modality::Image, input::InputPayload::Tensor(pixels)) => Ok(vec![visual
                    .as_mut()
                    .ok_or_else(|| {
                        Exception::custom(
                            "Inkling image input requires vision_config and vision weights",
                        )
                    })?
                    .forward(pixels, stream)?]),
                (input::Modality::Audio, input::InputPayload::Tensor(ids)) => Ok(vec![audio
                    .as_mut()
                    .ok_or_else(|| {
                        Exception::custom(
                            "Inkling audio input requires audio_config and audio weights",
                        )
                    })?
                    .forward(ids, part.metadata.audio_mask, stream)?]),
                (modality, input::InputPayload::Tensor(_)) => Err(Exception::custom(format!(
                    "Inkling does not support {} tensor inputs",
                    modality.as_str()
                ))),
                (modality, input::InputPayload::TokenIds(_)) => Err(Exception::custom(format!(
                    "Inkling {} input does not accept token-id payloads",
                    modality.as_str()
                ))),
            },
        )
    }
}

impl CausalLm<Cache> for Model {
    fn prefill_input_logits(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        match self.prepare_typed_prefill(input, stream)? {
            input::PreparedPrefill::Text(tokens) => {
                self.forward_logits(&tokens, None, Some(cache), true, stream)
            }
            input::PreparedPrefill::Embeddings { tokens, embeddings } => {
                self.forward_logits(&tokens, Some(&embeddings), Some(cache), true, stream)
            }
        }
    }

    fn decode_logits(
        &mut self,
        input_tokens: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        self.forward_logits(input_tokens, None, Some(cache), true, stream)
    }
}

/// Inkling token generation iterator.
pub type Generate<'a, S = crate::runtime::generation::sampler::DefaultSampler> =
    common::generation::Generate<'a, Model, Cache, S>;

pub(crate) struct LoadedInklingGguf {
    pub(crate) model: Model,
    pub(crate) eos_token_ids: Vec<u32>,
}

pub(crate) struct PreparedInklingGguf {
    pub(crate) args: ModelArgs,
    pub(crate) eos_token_ids: Vec<u32>,
}

/// Loads the text decoder from an `inkling` GGUF checkpoint.
pub fn load_gguf(
    gguf_file: impl AsRef<Path>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Model, Error> {
    let checkpoint = GgufCheckpoint::open(gguf_file)?;
    let metadata = gguf_metadata(&checkpoint);
    Ok(load_gguf_checkpoint(&checkpoint, metadata, stream, weights_stream)?.model)
}

pub(crate) fn load_gguf_checkpoint(
    checkpoint: &GgufCheckpoint,
    metadata: HashMap<String, GgufMetadataValue>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<LoadedInklingGguf, Error> {
    let prepared = prepare_gguf_checkpoint(checkpoint, &metadata, weights_stream)?;
    let mut model = Model::new(prepared.args, stream)?;
    let config = StrictLoadConfig::default();
    let mut report = StrictLoadReport::default();
    let mut materializer = checkpoint.materializer();
    for tensor in checkpoint.catalog().tensors() {
        let physical = &tensor.descriptor().name;
        if physical.contains("ffn_gate_exps")
            || physical.contains("ffn_up_exps")
            || physical.contains("ffn_gate_shexp")
            || physical.contains("ffn_up_shexp")
        {
            continue;
        }
        for (name, mut value) in materializer.converted_tensor(physical)?.into_arrays() {
            let mut translated = translate_gguf_weight_name(&name);
            if let Some(prefix) = translated.strip_suffix(".global_scale") {
                let layer = prefix
                    .strip_prefix("model.layers.")
                    .and_then(|value| value.parse::<i32>().ok())
                    .ok_or_else(|| {
                        Error::UnsupportedArchitecture(format!(
                            "invalid Inkling GGUF global-scale name {translated:?}"
                        ))
                    })?;
                translated = if model.args.text_config.is_dense(layer) {
                    format!("model.layers.{layer}.dense_global_scale")
                } else {
                    format!("model.layers.{layer}.moe.router.global_scale")
                };
            }
            if translated.contains("_sconv.weight") && value.ndim() == 2 {
                value = value.reshape(&[value.dim(0), 1, value.dim(1)], weights_stream)?;
            }
            load_named_array_strict(&mut model, translated, value, None, &config, &mut report)?;
        }
    }
    for layer in model.args.text_config.dense_mlp_idx..model.args.text_config.num_hidden_layers {
        let source = format!("blk.{layer}");
        for (source_gate, source_up, target) in [
            ("ffn_gate_exps", "ffn_up_exps", "moe.experts.gate_up_proj"),
            (
                "ffn_gate_shexp",
                "ffn_up_shexp",
                "moe.shared_experts.gate_up_proj",
            ),
        ] {
            let gate = materializer
                .converted_tensor(&format!("{source}.{source_gate}.weight"))?
                .into_arrays()
                .into_iter()
                .collect::<HashMap<_, _>>();
            let up = materializer
                .converted_tensor(&format!("{source}.{source_up}.weight"))?
                .into_arrays()
                .into_iter()
                .collect::<HashMap<_, _>>();
            for (source_suffix, target_suffix) in
                [("weight", ""), ("scales", "_scales"), ("biases", "_biases")]
            {
                let gate_name = format!("{source}.{source_gate}.{source_suffix}");
                let up_name = format!("{source}.{source_up}.{source_suffix}");
                match (gate.get(&gate_name), up.get(&up_name)) {
                    (Some(gate), Some(up)) => load_named_array_strict(
                        &mut model,
                        format!("model.layers.{layer}.{target}{target_suffix}"),
                        concatenate_axis(&[gate.clone(), up.clone()], 1, weights_stream)?,
                        None,
                        &config,
                        &mut report,
                    )?,
                    (None, None) if source_suffix == "biases" => {}
                    _ => {
                        return Err(Error::UnsupportedArchitecture(format!(
                            "Inkling GGUF has incomplete gate/up tensors under {source}"
                        )))
                    }
                }
            }
        }
    }
    report.finish(&model, &config)?;
    model.copy_to_stream(stream)?;
    Ok(LoadedInklingGguf {
        model,
        eos_token_ids: prepared.eos_token_ids,
    })
}

pub(crate) fn prepare_gguf_checkpoint(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    stream: &Stream,
) -> Result<PreparedInklingGguf, Error> {
    let architecture = gguf_string(metadata, "general.architecture")?;
    if architecture != "inkling" {
        return Err(Error::UnsupportedArchitecture(format!(
            "GGUF architecture {architecture:?}; this loader supports inkling"
        )));
    }
    checkpoint
        .catalog()
        .translated_outputs(translate_gguf_weight_name)
        .map_err(safemlx::error::IoError::from)?;
    let mut configs = gguf_quantization_configs(checkpoint, translate_gguf_weight_name)?;
    let mut args = args_from_gguf(metadata, stream)?;
    for layer in args.text_config.dense_mlp_idx..args.text_config.num_hidden_layers {
        for prefix in [
            format!("model.layers.{layer}.moe.experts"),
            format!("model.layers.{layer}.moe.shared_experts"),
        ] {
            let gate = format!("{prefix}.gate_proj");
            let up = format!("{prefix}.up_proj");
            match (configs.remove(&gate), configs.remove(&up)) {
                (Some(gate), Some(up)) if gate == up => {
                    configs.insert(format!("{prefix}.gate_up_proj"), gate);
                }
                (None, None) => {}
                (gate, up) => {
                    return Err(Error::UnsupportedArchitecture(format!(
                        "Inkling GGUF gate/up formats differ under {prefix}: {gate:?} vs {up:?}"
                    )))
                }
            }
        }
    }
    args.text_config.quantized_weight_configs = Some(configs);
    validate_args(&args)?;
    Ok(PreparedInklingGguf {
        args,
        eos_token_ids: crate::api::gguf_eos_token_ids(metadata)?,
    })
}

fn args_from_gguf(
    metadata: &HashMap<String, GgufMetadataValue>,
    stream: &Stream,
) -> Result<ModelArgs, Error> {
    let key = |suffix: &str| format!("inkling.{suffix}");
    let layers = gguf_i32(metadata, &key("block_count"), stream)?;
    let pattern = gguf_bool_pattern(metadata, &key("attention.sliding_window_pattern"), layers)?;
    let kv_values = metadata
        .get(&key("attention.head_count_kv"))
        .and_then(GgufMetadataValue::to_i64_vec)
        .ok_or_else(|| {
            Error::UnsupportedArchitecture("Inkling GGUF is missing attention.head_count_kv".into())
        })?;
    let kv_values = if kv_values.len() == 1 {
        vec![kv_values[0]; layers as usize]
    } else {
        kv_values
    };
    if kv_values.len() != layers as usize {
        return Err(Error::UnsupportedArchitecture(
            "Inkling GGUF attention.head_count_kv length does not match block_count".into(),
        ));
    }
    let global_kv = kv_values
        .iter()
        .zip(&pattern)
        .find_map(|(value, local)| (!local).then_some(*value))
        .unwrap_or(kv_values[0]);
    let local_kv = kv_values
        .iter()
        .zip(&pattern)
        .find_map(|(value, local)| local.then_some(*value))
        .unwrap_or(global_kv);
    let hidden_size = gguf_i32(metadata, &key("embedding_length"), stream)?;
    let heads = gguf_i32(metadata, &key("attention.head_count"), stream)?;
    let head_dim =
        gguf_optional_i32(metadata, &key("attention.key_length"))?.unwrap_or(hidden_size / heads);
    let vocab_size = gguf_vocab_size(metadata, &key("vocab_size"), stream)?;
    Ok(ModelArgs {
        model_type: "inkling_mm_model".into(),
        text_config: TextArgs {
            hidden_size,
            num_hidden_layers: layers,
            vocab_size,
            num_attention_heads: heads,
            num_key_value_heads: i32::try_from(global_kv).map_err(|_| {
                Error::UnsupportedArchitecture("Inkling global KV heads exceed i32".into())
            })?,
            head_dim,
            swa_num_attention_heads: Some(heads),
            swa_num_key_value_heads: Some(i32::try_from(local_kv).map_err(|_| {
                Error::UnsupportedArchitecture("Inkling local KV heads exceed i32".into())
            })?),
            swa_head_dim: Some(head_dim),
            sliding_window_size: gguf_i32(metadata, &key("attention.sliding_window"), stream)?,
            local_layer_ids: Some(
                pattern
                    .iter()
                    .enumerate()
                    .filter_map(|(layer, local)| local.then_some(layer as i32))
                    .collect(),
            ),
            layer_types: None,
            dense_mlp_idx: gguf_i32(metadata, &key("dense_block_count"), stream)?,
            mlp_layer_types: None,
            sconv_kernel_size: gguf_i32(metadata, &key("shortconv_kernel"), stream)?,
            use_sconv: true,
            rel_extent: gguf_i32(metadata, &key("rel_extent"), stream)?,
            d_rel: gguf_i32(metadata, &key("d_rel"), stream)?,
            log_scaling_n_floor: gguf_optional_i32(metadata, &key("log_scaling_n_floor"))?
                .filter(|value| *value > 0),
            log_scaling_alpha: gguf_optional_f32(metadata, &key("log_scaling_alpha"))?
                .unwrap_or(0.0),
            rms_norm_eps: gguf_f32(metadata, &key("attention.layer_norm_rms_epsilon"))?,
            use_embed_norm: true,
            unpadded_vocab_size: gguf_optional_i32(metadata, &key("unpadded_vocab_size"))?,
            logits_mup_width_multiplier: gguf_optional_f32(metadata, &key("logit_scale_denom"))?
                .unwrap_or(1.0),
            final_logit_softcapping: None,
            intermediate_size: gguf_i32(metadata, &key("expert_feed_forward_length"), stream)?,
            dense_intermediate_size: Some(gguf_i32(metadata, &key("feed_forward_length"), stream)?),
            moe_intermediate_size: None,
            n_routed_experts: gguf_i32(metadata, &key("expert_count"), stream)?,
            num_experts_per_tok: gguf_i32(metadata, &key("expert_used_count"), stream)?,
            n_shared_experts: gguf_i32(metadata, &key("expert_shared_count"), stream)?,
            route_scale: gguf_optional_f32(metadata, &key("expert_weights_scale"))?.unwrap_or(1.0),
            shared_expert_sink: true,
            use_gate_bias: true,
            norm_after_topk: true,
            use_global_scale: true,
            gate_activation: "sigmoid".into(),
            hidden_act: "silu".into(),
            attention_dropout: 0.0,
            q_bias: false,
            o_bias: false,
            model_max_length: Some(gguf_i32(metadata, &key("context_length"), stream)?),
            quantized_weight_configs: None,
        },
        audio_config: None,
        vision_config: None,
        image_token_id: default_image_token_id(),
        audio_token_id: default_audio_token_id(),
        eos_token_id: crate::api::gguf_eos_token_ids(metadata)?.first().copied(),
    })
}

fn gguf_bool_pattern(
    metadata: &HashMap<String, GgufMetadataValue>,
    key: &str,
    layers: i32,
) -> Result<Vec<bool>, Error> {
    let values = match metadata.get(key) {
        Some(GgufMetadataValue::Array(safemlx::ops::GgufMetadataArray::Bool(values))) => {
            values.clone()
        }
        Some(value) => value
            .to_i64_vec()
            .map(|values| values.into_iter().map(|value| value != 0).collect())
            .ok_or_else(|| {
                Error::UnsupportedArchitecture(format!(
                    "Inkling GGUF metadata key {key:?} must be a bool array"
                ))
            })?,
        None => (0..layers).map(|layer| (layer + 1) % 6 != 0).collect(),
    };
    if values.len() != layers as usize {
        return Err(Error::UnsupportedArchitecture(format!(
            "Inkling GGUF {key:?} has {} values for {layers} layers",
            values.len()
        )));
    }
    Ok(values)
}

fn gguf_vocab_size(
    metadata: &HashMap<String, GgufMetadataValue>,
    fallback: &str,
    stream: &Stream,
) -> Result<i32, Error> {
    match metadata
        .get("tokenizer.ggml.tokens")
        .and_then(GgufMetadataValue::as_strings)
    {
        Some(tokens) => i32::try_from(tokens.len()).map_err(|_| {
            Error::UnsupportedArchitecture("GGUF tokenizer vocabulary exceeds i32".into())
        }),
        None if metadata.contains_key("tokenizer.ggml.tokens") => {
            Err(Error::UnsupportedArchitecture(
                "GGUF tokenizer.ggml.tokens metadata has the wrong type".into(),
            ))
        }
        None => gguf_i32(metadata, fallback, stream),
    }
}

fn gguf_optional_i32(
    metadata: &HashMap<String, GgufMetadataValue>,
    key: &str,
) -> Result<Option<i32>, Error> {
    metadata
        .get(key)
        .map(|value| {
            value
                .as_i64()
                .and_then(|value| i32::try_from(value).ok())
                .ok_or_else(|| {
                    Error::UnsupportedArchitecture(format!(
                        "GGUF metadata key {key:?} must be an i32 scalar"
                    ))
                })
        })
        .transpose()
}

fn gguf_f32(metadata: &HashMap<String, GgufMetadataValue>, key: &str) -> Result<f32, Error> {
    gguf_optional_f32(metadata, key)?.ok_or_else(|| {
        Error::UnsupportedArchitecture(format!("GGUF metadata is missing required key {key:?}"))
    })
}

fn gguf_optional_f32(
    metadata: &HashMap<String, GgufMetadataValue>,
    key: &str,
) -> Result<Option<f32>, Error> {
    metadata
        .get(key)
        .map(|value| {
            value.as_f32().ok_or_else(|| {
                Error::UnsupportedArchitecture(format!(
                    "GGUF metadata key {key:?} must be a numeric scalar"
                ))
            })
        })
        .transpose()
}

pub(crate) fn translate_gguf_weight_name(name: &str) -> String {
    for (source, target) in [
        ("token_embd", "model.embed_tokens"),
        ("token_embd_norm", "model.embed_norm"),
        ("output_norm", "model.norm"),
        ("output", "lm_head"),
    ] {
        if name == source || name.starts_with(&format!("{source}.")) {
            return name.replacen(source, target, 1);
        }
    }
    let Some(rest) = name.strip_prefix("blk.") else {
        return name.to_string();
    };
    let Some((layer, parameter)) = rest.split_once('.') else {
        return name.to_string();
    };
    for (source, target) in [
        ("ffn_gate_exps", "moe.experts.gate_proj"),
        ("ffn_up_exps", "moe.experts.up_proj"),
        ("ffn_down_exps", "moe.experts.down_proj"),
        ("ffn_gate_shexp", "moe.shared_experts.gate_proj"),
        ("ffn_up_shexp", "moe.shared_experts.up_proj"),
        ("ffn_down_shexp", "moe.shared_experts.down_proj"),
    ] {
        if parameter == source || parameter.starts_with(&format!("{source}.")) {
            let suffix = parameter.strip_prefix(source).unwrap_or_default();
            let suffix = match suffix {
                ".weight" => "",
                ".scales" => "_scales",
                ".biases" => "_biases",
                other => other,
            };
            return format!("model.layers.{layer}.{target}{suffix}");
        }
    }
    for (source, target) in [
        ("attn_norm", "input_layernorm"),
        ("attn_q", "self_attn.q_proj"),
        ("attn_k", "self_attn.k_proj"),
        ("attn_v", "self_attn.v_proj"),
        ("attn_r", "self_attn.r_proj"),
        ("attn_output", "self_attn.o_proj"),
        ("attn_q_norm", "self_attn.q_norm"),
        ("attn_k_norm", "self_attn.k_norm"),
        ("shortconv_k", "self_attn.k_sconv"),
        ("shortconv_v", "self_attn.v_sconv"),
        ("shortconv_attn", "attn_sconv"),
        ("shortconv_mlp", "mlp_sconv"),
        ("ffn_norm", "post_attention_layernorm"),
        ("ffn_gate", "dense.gate_proj"),
        ("ffn_up", "dense.up_proj"),
        ("ffn_down", "dense.down_proj"),
        ("ffn_gate_inp", "moe.router.weight"),
    ] {
        if parameter == source || parameter.starts_with(&format!("{source}.")) {
            let mut translated = parameter.replacen(source, target, 1);
            if source == "ffn_gate_inp" && translated.ends_with(".weight") {
                translated.truncate(translated.len() - ".weight".len());
            }
            return format!("model.layers.{layer}.{translated}");
        }
    }
    if parameter == "attn_rel_proj.weight" || parameter == "attn_rel_proj" {
        return format!("model.layers.{layer}.self_attn.rel_proj");
    }
    if parameter == "ffn_gscale" || parameter == "ffn_gscale.weight" {
        return format!("model.layers.{layer}.global_scale");
    }
    if parameter == "ffn_exp_probs_b.bias" || parameter == "ffn_exp_probs_b" {
        return format!("model.layers.{layer}.moe.router.bias");
    }
    name.to_string()
}

pub fn load_tokenizer(model_dir: impl AsRef<Path>) -> Result<Tokenizer, Error> {
    Tokenizer::from_file(model_dir.as_ref().join("tokenizer.json")).map_err(Into::into)
}

pub fn get_model_args(model_dir: impl AsRef<Path>) -> Result<ModelArgs, Error> {
    let value: Value =
        serde_json::from_reader(std::fs::File::open(model_dir.as_ref().join("config.json"))?)?;
    validate_model_config_value(&value)?;
    Ok(serde_json::from_value(value)?)
}

pub fn validate_model_config_value(value: &Value) -> Result<(), Error> {
    let args: ModelArgs = serde_json::from_value(value.clone()).map_err(|error| {
        Error::UnsupportedArchitecture(format!("invalid Inkling config: {error}"))
    })?;
    validate_args(&args)
}

fn validate_args(args: &ModelArgs) -> Result<(), Error> {
    let text = &args.text_config;
    if args.model_type != "inkling_mm_model" {
        return Err(Error::UnsupportedArchitecture(format!(
            "expected Inkling model_type inkling_mm_model, got {:?}",
            args.model_type
        )));
    }
    for (name, value) in [
        ("hidden_size", text.hidden_size),
        ("num_hidden_layers", text.num_hidden_layers),
        ("vocab_size", text.vocab_size),
        ("num_attention_heads", text.num_attention_heads),
        ("num_key_value_heads", text.num_key_value_heads),
        ("head_dim", text.head_dim),
        ("d_rel", text.d_rel),
        ("rel_extent", text.rel_extent),
        ("sliding_window_size", text.sliding_window_size),
        ("sconv_kernel_size", text.sconv_kernel_size),
        ("n_routed_experts", text.n_routed_experts),
        ("num_experts_per_tok", text.num_experts_per_tok),
        ("n_shared_experts", text.n_shared_experts),
    ] {
        if value <= 0 {
            return Err(Error::UnsupportedArchitecture(format!(
                "Inkling {name} must be positive, got {value}"
            )));
        }
    }
    if !text.use_sconv
        || !text.use_embed_norm
        || !text.shared_expert_sink
        || !text.use_gate_bias
        || !text.norm_after_topk
        || !text.use_global_scale
        || text.gate_activation != "sigmoid"
        || text.hidden_act != "silu"
        || text.attention_dropout != 0.0
        || text.q_bias
        || text.o_bias
        || text
            .final_logit_softcapping
            .is_some_and(|value| value != 0.0)
    {
        return Err(Error::UnsupportedArchitecture(
            "Inkling config uses an unsupported attention, convolution, routing, or logit variant"
                .into(),
        ));
    }
    if text.num_attention_heads % text.num_key_value_heads != 0
        || text.q_heads(true) % text.kv_heads(true) != 0
        || text.attention_head_dim(true) != text.head_dim
    {
        return Err(Error::UnsupportedArchitecture(
            "Inkling attention head configuration is inconsistent".into(),
        ));
    }
    if text.dense_intermediate_size() <= 0 || text.moe_intermediate_size() <= 0 {
        return Err(Error::UnsupportedArchitecture(
            "Inkling dense and MoE intermediate sizes must be positive".into(),
        ));
    }
    if text.num_experts_per_tok > text.n_routed_experts
        || !(0..=text.num_hidden_layers).contains(&text.dense_mlp_idx)
        || text.local_layer_ids.as_ref().is_some_and(|ids| {
            ids.iter()
                .any(|layer| !(0..text.num_hidden_layers).contains(layer))
        })
        || text
            .layer_types
            .as_ref()
            .is_some_and(|types| types.len() != text.num_hidden_layers as usize)
        || text
            .mlp_layer_types
            .as_ref()
            .is_some_and(|types| types.len() != text.num_hidden_layers as usize)
    {
        return Err(Error::UnsupportedArchitecture(
            "Inkling layer schedule or expert top-k configuration is inconsistent".into(),
        ));
    }
    if let Some(audio) = &args.audio_config {
        if audio.text_hidden_size != text.hidden_size
            || audio.num_codebooks <= 0
            || audio.codebook_size <= 0
            || audio.bias
            || !audio.use_audio_norm
            || audio.audio_mode != "dmel"
        {
            return Err(Error::UnsupportedArchitecture(
                "Inkling audio configuration is inconsistent with the text decoder".into(),
            ));
        }
    }
    if let Some(vision) = &args.vision_config {
        if vision.text_hidden_size != text.hidden_size
            || vision.vision_encoder_type != "hmlp"
            || !vision.use_vision_norm
        {
            return Err(Error::UnsupportedArchitecture(
                "Inkling vision hidden size does not match the text decoder".into(),
            ));
        }
        if (
            vision.temporal_patch_size,
            vision.patch_size,
            vision.num_hidden_layers,
            vision.num_channels,
        ) != (2, 40, 4, 3)
        {
            return Err(Error::UnsupportedArchitecture(
                "Inkling vision configuration is not the released 4-layer 2x40x40 hMLP tower"
                    .into(),
            ));
        }
    }
    Ok(())
}

pub fn load_model(
    model_dir: impl AsRef<Path>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Model, Error> {
    let model_dir = model_dir.as_ref();
    let args = get_model_args(model_dir)?;
    let mut model = Model::new(args, stream)?;
    let config = StrictLoadConfig::default().allow_unused_prefix("model.mtp.");
    let mut report = StrictLoadReport::default();
    let mut params = model.parameters_mut().flatten();
    for file in safetensors_files(model_dir)? {
        for_each_safetensor_array(file, weights_stream, |key, value| {
            for (key, value) in transform_weight(key, value, stream)? {
                load_array_strict(&mut params, key, value, &config, &mut report);
            }
            Ok(())
        })?;
    }
    drop(params);
    report.finish(&model, &config)?;
    model.copy_to_stream(stream)?;
    Ok(model)
}

pub(crate) fn transform_weight(
    key: String,
    mut value: Array,
    stream: &Stream,
) -> Result<Vec<(String, Array)>, Error> {
    if key.ends_with("_sconv.weight") {
        value = value.as_dtype(Dtype::Float32, stream)?;
    }
    if let Some(suffix) = key.strip_prefix("model.audio.") {
        return Ok(vec![(format!("audio.{suffix}"), value)]);
    }
    if let Some(suffix) = key.strip_prefix("model.visual.") {
        let mut suffix = suffix.to_string();
        for layer in 0..4 {
            suffix = suffix
                .replace(
                    &format!("layers.linear_{layer}.weight"),
                    &format!("layers.{layer}.projection.weight"),
                )
                .replace(
                    &format!("layers.norm_{layer}.weight"),
                    &format!("layers.{layer}.layer_norm.weight"),
                );
        }
        return Ok(vec![(format!("visual.{suffix}"), value)]);
    }
    if !key.starts_with("model.llm.") {
        return Ok(vec![(key, value)]);
    }
    let mut key = key.replacen("model.llm.", "model.", 1);
    key = key
        .replace("model.embed.weight", "model.embed_tokens.weight")
        .replace("model.unembed.weight", "lm_head.weight")
        .replace(".attn_norm.weight", ".input_layernorm.weight")
        .replace(".mlp_norm.weight", ".post_attention_layernorm.weight")
        .replace(".attn.wq_du.weight", ".self_attn.q_proj.weight")
        .replace(".attn.wk_dv.weight", ".self_attn.k_proj.weight")
        .replace(".attn.wv_dv.weight", ".self_attn.v_proj.weight")
        .replace(".attn.wr_du.weight", ".self_attn.r_proj.weight")
        .replace(".attn.wo_ud.weight", ".self_attn.o_proj.weight")
        .replace(".attn.q_norm.weight", ".self_attn.q_norm.weight")
        .replace(".attn.k_norm.weight", ".self_attn.k_norm.weight")
        .replace(".attn.rel_logits_proj.proj", ".self_attn.rel_proj")
        .replace(".attn.k_sconv.weight", ".self_attn.k_sconv.weight")
        .replace(".attn.v_sconv.weight", ".self_attn.v_sconv.weight")
        .replace(".mlp.w2_md.weight", ".dense.down_proj.weight")
        .replace(".mlp.global_scale", ".dense_global_scale")
        .replace(".mlp.gate.weight", ".moe.router.weight")
        .replace(".mlp.gate.bias", ".moe.router.bias")
        .replace(".mlp.gate.global_scale", ".moe.router.global_scale")
        .replace(".mlp.experts.w2_weight", ".moe.experts.down_proj")
        .replace(
            ".mlp.shared_experts.shared_w2_weight",
            ".moe.shared_experts.down_proj",
        );

    if key.ends_with(".mlp.w13_dn.weight") {
        let prefix = key.trim_end_matches(".mlp.w13_dn.weight");
        let (gate, up) = deinterleave_w13(value, stream)?;
        return Ok(vec![
            (format!("{prefix}.dense.gate_proj.weight"), gate),
            (format!("{prefix}.dense.up_proj.weight"), up),
        ]);
    }
    if key.ends_with(".mlp.experts.w13_weight") {
        let prefix = key.trim_end_matches(".mlp.experts.w13_weight");
        let (gate, up) = deinterleave_w13(value, stream)?;
        return Ok(vec![(
            format!("{prefix}.moe.experts.gate_up_proj"),
            concatenate_axis(&[gate, up], -2, stream)?,
        )]);
    }
    if key.ends_with(".mlp.shared_experts.shared_w13_weight") {
        let prefix = key.trim_end_matches(".mlp.shared_experts.shared_w13_weight");
        let (gate, up) = deinterleave_w13(value, stream)?;
        return Ok(vec![(
            format!("{prefix}.moe.shared_experts.gate_up_proj"),
            concatenate_axis(&[gate, up], -2, stream)?,
        )]);
    }
    Ok(vec![(key, value)])
}

fn deinterleave_w13(value: Array, stream: &Stream) -> Result<(Array, Array), Error> {
    let shape = value.shape().to_vec();
    if shape.len() < 2 || shape[shape.len() - 2] % 2 != 0 {
        return Err(Error::UnsupportedArchitecture(format!(
            "Inkling w13 tensor has invalid shape {shape:?}"
        )));
    }
    let rows = shape[shape.len() - 2] / 2;
    let hidden = shape[shape.len() - 1];
    let (reshaped, gate_index, up_index) = if shape.len() == 2 {
        (
            value.reshape(&[rows, 2, hidden], stream)?,
            (.., 0, ..),
            (.., 1, ..),
        )
    } else if shape.len() == 3 {
        let experts = shape[0];
        let reshaped = value.reshape(&[experts, rows, 2, hidden], stream)?;
        let gate = reshaped.try_index_device((.., .., 0, ..), stream)?;
        let up = reshaped.try_index_device((.., .., 1, ..), stream)?;
        return Ok((gate, up));
    } else {
        return Err(Error::UnsupportedArchitecture(format!(
            "Inkling w13 tensor rank {} is unsupported",
            shape.len()
        )));
    };
    Ok((
        reshaped.try_index_device(gate_index, stream)?,
        reshaped.try_index_device(up_index, stream)?,
    ))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use safemlx::{
        ops::{GgufMetadataArray, GgufMetadataValue},
        Device, DeviceType, ExecutionContext,
    };
    use serde_json::json;

    #[test]
    fn gguf_names_translate_to_text_parameter_tree() {
        assert_eq!(
            super::translate_gguf_weight_name("blk.4.attn_rel_proj.weight"),
            "model.layers.4.self_attn.rel_proj"
        );
        assert_eq!(
            super::translate_gguf_weight_name("blk.4.ffn_gate_exps.scales"),
            "model.layers.4.moe.experts.gate_proj_scales"
        );
        assert_eq!(
            super::translate_gguf_weight_name("blk.4.ffn_gscale"),
            "model.layers.4.global_scale"
        );
    }

    #[test]
    fn draft_gguf_metadata_builds_text_config() {
        let metadata = HashMap::from([
            (
                "general.architecture".into(),
                GgufMetadataValue::String("inkling".into()),
            ),
            ("inkling.block_count".into(), GgufMetadataValue::Uint32(2)),
            (
                "inkling.embedding_length".into(),
                GgufMetadataValue::Uint32(32),
            ),
            (
                "inkling.feed_forward_length".into(),
                GgufMetadataValue::Uint32(64),
            ),
            (
                "inkling.expert_feed_forward_length".into(),
                GgufMetadataValue::Uint32(32),
            ),
            (
                "inkling.attention.head_count".into(),
                GgufMetadataValue::Uint32(4),
            ),
            (
                "inkling.attention.head_count_kv".into(),
                GgufMetadataValue::Array(GgufMetadataArray::Uint32(vec![2, 1])),
            ),
            (
                "inkling.attention.key_length".into(),
                GgufMetadataValue::Uint32(8),
            ),
            (
                "inkling.attention.sliding_window".into(),
                GgufMetadataValue::Uint32(8),
            ),
            (
                "inkling.attention.sliding_window_pattern".into(),
                GgufMetadataValue::Array(GgufMetadataArray::Bool(vec![true, false])),
            ),
            (
                "inkling.attention.layer_norm_rms_epsilon".into(),
                GgufMetadataValue::Float32(1e-6),
            ),
            (
                "inkling.context_length".into(),
                GgufMetadataValue::Uint32(128),
            ),
            ("inkling.vocab_size".into(), GgufMetadataValue::Uint32(64)),
            ("inkling.d_rel".into(), GgufMetadataValue::Uint32(4)),
            ("inkling.rel_extent".into(), GgufMetadataValue::Uint32(16)),
            (
                "inkling.shortconv_kernel".into(),
                GgufMetadataValue::Uint32(4),
            ),
            (
                "inkling.dense_block_count".into(),
                GgufMetadataValue::Uint32(1),
            ),
            ("inkling.expert_count".into(), GgufMetadataValue::Uint32(4)),
            (
                "inkling.expert_used_count".into(),
                GgufMetadataValue::Uint32(2),
            ),
            (
                "inkling.expert_shared_count".into(),
                GgufMetadataValue::Uint32(1),
            ),
        ]);
        let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let args = super::args_from_gguf(&metadata, context.stream()).unwrap();
        super::validate_args(&args).unwrap();
        assert_eq!(args.text_config.local_layer_ids, Some(vec![0]));
        assert_eq!(args.text_config.swa_num_key_value_heads, Some(2));
        assert_eq!(args.text_config.num_key_value_heads, 1);
        assert!(args.audio_config.is_none());
        assert!(args.vision_config.is_none());
    }

    #[test]
    fn released_config_shape_is_accepted() {
        let config = json!({
            "model_type":"inkling_mm_model",
            "eos_token_id":200006,
            "text_config":{
                "hidden_size":32,"num_hidden_layers":3,"vocab_size":64,
                "num_attention_heads":4,"num_key_value_heads":2,"head_dim":8,
                "swa_num_attention_heads":4,"swa_num_key_value_heads":2,"swa_head_dim":8,
                "sliding_window_size":8,"local_layer_ids":[0,1],"dense_mlp_idx":1,
                "sconv_kernel_size":4,"d_rel":4,"rel_extent":16,
                "intermediate_size":24,"dense_intermediate_size":48,
                "n_routed_experts":4,"num_experts_per_tok":2,"n_shared_experts":1,
                "route_scale":8.0,"use_sconv":true,"use_embed_norm":true,
                "shared_expert_sink":true,"use_gate_bias":true,"norm_after_topk":true,
                "use_global_scale":true,"gate_activation":"sigmoid"
            }
        });
        super::validate_model_config_value(&config).unwrap();
        let support = crate::api::check_model_config(&config);
        let crate::api::ModelConfigSupport::Supported(support) = support else {
            panic!("released Inkling metadata did not dispatch")
        };
        assert_eq!(support.kind, crate::api::ModelKind::Inkling);
        assert_eq!(support.effective_model_type, "inkling_mm_model");
    }

    #[test]
    fn cache_schedule_matches_local_and_global_layers() {
        let config = json!({
            "model_type":"inkling_mm_model",
            "image_token_id":200054,
            "audio_token_id":200053,
            "text_config":{
                "hidden_size":32,"num_hidden_layers":3,"vocab_size":64,
                "num_attention_heads":4,"num_key_value_heads":2,"head_dim":8,
                "swa_num_attention_heads":4,"swa_num_key_value_heads":2,"swa_head_dim":8,
                "sliding_window_size":8,"local_layer_ids":[0,1],"dense_mlp_idx":1,
                "sconv_kernel_size":4,"d_rel":4,"rel_extent":16,
                "intermediate_size":24,"dense_intermediate_size":48,
                "n_routed_experts":4,"num_experts_per_tok":2,"n_shared_experts":1,
                "route_scale":8.0,"use_sconv":true,"use_embed_norm":true,
                "shared_expert_sink":true,"use_gate_bias":true,"norm_after_topk":true,
                "use_global_scale":true,"gate_activation":"sigmoid"
            },
            "audio_config":{
                "decoder_dmodel":32,"n_mel_bins":80,"mel_vocab_size":16
            },
            "vision_config":{
                "decoder_dmodel":32,"patch_size":40,"temporal_patch_size":2,
                "n_channels":3,"n_layers":4
            }
        });
        let args: super::ModelArgs = serde_json::from_value(config).unwrap();
        super::validate_args(&args).unwrap();
        let cache = super::Cache::new(&args.text_config);
        assert_eq!(cache.layers.len(), 3);
        assert!(matches!(
            cache.layers[0].kv,
            super::InklingKvCache::Sliding(_)
        ));
        assert!(matches!(
            cache.layers[2].kv,
            super::InklingKvCache::Global(_)
        ));
    }
}
