//! Kimi Linear's established no-RoPE latent-attention operator.

use eredu_checkpoint::WeightQuantization;
use eredu_nn::{CompressedAttentionCache as _, CompressedAttentionState, CompressedAttentionView};
use eredu_runtime::ActivationObserver as RuntimeActivationObserver;
use safemlx::{
    builder::Builder,
    error::Exception,
    fast::ScaledDotProductAttentionMask,
    macros::ModuleParameters,
    module::{Module, Param},
    nn,
    ops::{
        broadcast_to, concatenate_axis, einsum, grouped_matmul,
        indexing::{NewAxis, TryIndexOp},
        quantized_packed_dimension, r#where, softmax_axis,
    },
    transforms::eval,
    Array, Dtype, Stream,
};

use crate::{
    backend::mlx::nn as common,
    backend::mlx::nn::linear::PhysicalLinear as Linear,
    backend::mlx::nn::tensor::{
        create_causal_mask,
        rope::{initialize_rope, RopeVariant},
    },
    backend::mlx::runtime::cache::{
        BlockwiseAttentionAccumulator, CompressedLatentCache, KeyValueAttentionBlock,
    },
};

use super::model::ModelArgs;
use eredu_checkpoint::LinearFormat as WeightFormat;

type ObserverOption<'a> = Option<&'a mut dyn RuntimeActivationObserver<Array, Exception>>;

fn activation_name(prefix: &str, suffix: &str) -> String {
    if suffix.is_empty() {
        prefix.to_string()
    } else {
        format!("{prefix}.{suffix}")
    }
}

#[inline]
fn observe_activation(
    observer: &mut ObserverOption<'_>,
    prefix: &str,
    suffix: &str,
    value: &Array,
) -> Result<(), Exception> {
    if let Some(observer) = observer.as_mut() {
        observer.observe(&activation_name(prefix, suffix), value)?;
    }
    Ok(())
}

#[derive(Debug, Clone, ModuleParameters)]
/// Per-head MLA reconstruction matrix used by split-projection GGUFs.
pub(crate) struct PackedHeadProjection {
    /// Head count represented by the leading weight dimension.
    pub num_heads: i32,
    /// Optional per-weight affine encoding.
    pub affine: Option<WeightQuantization>,
    #[param]
    /// Weight shaped `[heads, output, input]` before affine packing.
    pub weight: Param<Array>,
    #[param]
    /// Affine scales.
    pub scales: Param<Option<Array>>,
    #[param]
    /// Affine biases.
    pub biases: Param<Option<Array>>,
}

impl PackedHeadProjection {
    fn new(
        num_heads: i32,
        input_dims: i32,
        output_dims: i32,
        format: WeightFormat,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        let affine = match format {
            WeightFormat::Affine(config) => Some(WeightQuantization::Affine(config)),
            WeightFormat::MxFp4 => Some(WeightQuantization::MxFp4),
            WeightFormat::Dense
            | WeightFormat::GgufIQuant { .. }
            | WeightFormat::E4M3BlockFp8(_) => None,
        };
        let packed_input = affine.map_or(input_dims, |quantization| {
            quantized_packed_dimension(input_dims, quantization.bits())
        });
        Ok(Self {
            num_heads,
            affine,
            weight: Param::<Array>::unloaded(
                &[num_heads, output_dims, packed_input],
                if affine.is_some() {
                    Dtype::Uint32
                } else {
                    Dtype::Float32
                },
                stream,
            )?,
            scales: if let Some(quantization) = affine {
                Param::<Option<Array>>::unloaded_some(
                    &[
                        num_heads,
                        output_dims,
                        input_dims / quantization.group_size(),
                    ],
                    if quantization == WeightQuantization::MxFp4 {
                        Dtype::Uint8
                    } else {
                        Dtype::Float32
                    },
                    stream,
                )?
            } else {
                Param::new(None)
            },
            biases: if let Some(quantization) = affine.filter(|q| q.has_biases()) {
                Param::<Option<Array>>::unloaded_some(
                    &[
                        num_heads,
                        output_dims,
                        input_dims / quantization.group_size(),
                    ],
                    Dtype::Float32,
                    stream,
                )?
            } else {
                Param::new(None)
            },
        })
    }

    fn forward(
        &mut self,
        input: &Array,
        transpose: bool,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let shape = input.shape();
        let routes = input.size() as i32 / input.dim(-1);
        let mut ids = Vec::with_capacity(routes as usize);
        for _ in 0..routes / self.num_heads {
            ids.extend(0..self.num_heads as u32);
        }
        let group_ids = Array::from_slice(&ids, &[routes]);
        let input = input.reshape(&[routes, input.dim(-1)], stream)?;
        let output = if let Some(affine) = self.affine {
            common::moe::packed_grouped_linear_with_transpose(
                &input,
                self.weight.as_ref(),
                self.scales.as_ref().as_ref().expect("packed head scales"),
                self.biases.as_ref().as_ref(),
                &group_ids,
                affine,
                transpose,
                stream,
            )?
        } else {
            let weight = if transpose {
                self.weight.as_ref().swap_axes(-1, -2, stream)?
            } else {
                self.weight.as_ref().clone()
            };
            grouped_matmul(&input, &weight, &group_ids, true, stream)?
        };
        let mut output_shape = shape.to_vec();
        *output_shape.last_mut().expect("head projection rank") = output.dim(-1);
        output.reshape(&output_shape, stream)
    }
}

#[derive(Debug, Clone, ModuleParameters)]
/// Kimi Linear no-RoPE multi-head latent attention.
pub(crate) struct KimiLatentAttention {
    /// Query head count.
    pub num_heads: i32,
    /// Non-positional width per head.
    pub qk_nope_head_dim: i32,
    /// Rotary width per head.
    pub qk_rope_head_dim: i32,
    /// Value width per head.
    pub v_head_dim: i32,
    /// Compressed latent width.
    pub kv_lora_rank: i32,
    /// Attention score scale.
    pub softmax_scale: f32,
    /// Whether to leave the nominal positional subspace unrotated.
    pub use_nope: bool,
    #[param]
    /// Direct query projection for compatible no-query-LoRA checkpoints.
    pub q_proj: Option<Linear>,
    #[param]
    /// Query LoRA down projection.
    pub q_a_proj: Option<Linear>,
    #[param]
    /// Query LoRA normalization.
    pub q_a_layernorm: Option<nn::RmsNorm>,
    #[param]
    /// Query LoRA up projection.
    pub q_b_proj: Option<Linear>,
    #[param]
    /// Combined compressed latent and shared rotary-key projection.
    pub kv_a_proj_with_mqa: Linear,
    #[param]
    /// Compressed latent normalization.
    pub kv_a_layernorm: nn::RmsNorm,
    #[param]
    /// Per-head non-positional key and value reconstruction.
    pub kv_b_proj: Option<Linear>,
    #[param]
    /// Split non-positional key reconstruction used by modern GGUFs.
    pub k_b_proj: Option<PackedHeadProjection>,
    #[param]
    /// Split value reconstruction used by modern GGUFs.
    pub v_b_proj: Option<PackedHeadProjection>,
    #[param]
    /// Attention output projection.
    pub o_proj: Linear,
    #[param]
    /// Rotary embedding applied only to the positional subspace.
    pub rope: RopeVariant,
}

impl KimiLatentAttention {
    /// Creates MLA with an optional identity positional-subspace policy.
    pub(crate) fn new_with_nope(
        args: &ModelArgs,
        layer: i32,
        use_nope: bool,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        let prefix = format!("model.layers.{layer}.self_attn");
        let format =
            |projection: &str| args.weight_format_for(&format!("{prefix}.{projection}.weight"));
        let q_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim;
        let (q_proj, q_a_proj, q_a_layernorm, q_b_proj) = match args.q_lora_rank {
            Some(rank) => (
                None,
                Some(Linear::unloaded(
                    args.hidden_size,
                    rank,
                    false,
                    format("q_a_proj"),
                    stream,
                )?),
                Some(nn::RmsNorm::unloaded(
                    rank,
                    args.rms_norm_eps,
                    Dtype::Float32,
                    stream,
                )?),
                Some(Linear::unloaded(
                    rank,
                    args.num_attention_heads * q_head_dim,
                    false,
                    format("q_b_proj"),
                    stream,
                )?),
            ),
            None => (
                Some(Linear::unloaded(
                    args.hidden_size,
                    args.num_attention_heads * q_head_dim,
                    false,
                    format("q_proj"),
                    stream,
                )?),
                None,
                None,
                None,
            ),
        };
        let rope_config = None;
        let scale = (q_head_dim as f32).sqrt().recip();
        Ok(Self {
            num_heads: args.num_attention_heads,
            qk_nope_head_dim: args.qk_nope_head_dim,
            qk_rope_head_dim: args.qk_rope_head_dim,
            v_head_dim: args.v_head_dim,
            kv_lora_rank: args.kv_lora_rank,
            softmax_scale: scale,
            use_nope,
            q_proj,
            q_a_proj,
            q_a_layernorm,
            q_b_proj,
            kv_a_proj_with_mqa: Linear::unloaded(
                args.hidden_size,
                args.kv_lora_rank + args.qk_rope_head_dim,
                false,
                format("kv_a_proj_with_mqa"),
                stream,
            )?,
            kv_a_layernorm: nn::RmsNorm::unloaded(
                args.kv_lora_rank,
                args.rms_norm_eps,
                Dtype::Float32,
                stream,
            )?,
            kv_b_proj: if args.split_kv_b {
                None
            } else {
                Some(Linear::unloaded(
                    args.kv_lora_rank,
                    args.num_attention_heads * (args.qk_nope_head_dim + args.v_head_dim),
                    false,
                    format("kv_b_proj"),
                    stream,
                )?)
            },
            k_b_proj: if args.split_kv_b {
                Some(PackedHeadProjection::new(
                    args.num_attention_heads,
                    args.qk_nope_head_dim,
                    args.kv_lora_rank,
                    format("k_b_proj"),
                    stream,
                )?)
            } else {
                None
            },
            v_b_proj: if args.split_kv_b {
                Some(PackedHeadProjection::new(
                    args.num_attention_heads,
                    args.kv_lora_rank,
                    args.v_head_dim,
                    format("v_b_proj"),
                    stream,
                )?)
            } else {
                None
            },
            o_proj: Linear::unloaded(
                args.num_attention_heads * args.v_head_dim,
                args.hidden_size,
                false,
                format("o_proj"),
                stream,
            )?,
            rope: initialize_rope(
                args.qk_rope_head_dim,
                args.rope_theta,
                false,
                &rope_config,
                args.model_max_length,
                stream,
            )?,
        })
    }

    fn project_queries(
        &mut self,
        x: &Array,
        stream: &Stream,
        prefix: &str,
        observer: &mut ObserverOption<'_>,
    ) -> Result<Array, Exception> {
        if let Some(q_proj) = &mut self.q_proj {
            let query = q_proj.forward(x, stream)?;
            observe_activation(observer, prefix, "q_proj", &query)?;
            Ok(query)
        } else {
            let q = self
                .q_a_proj
                .as_mut()
                .expect("query LoRA down projection")
                .forward(x, stream)?;
            observe_activation(observer, prefix, "q_a_proj", &q)?;
            let q = self
                .q_a_layernorm
                .as_mut()
                .expect("query LoRA norm")
                .forward(&q, stream)?;
            observe_activation(observer, prefix, "q_a_layernorm", &q)?;
            let q = self
                .q_b_proj
                .as_mut()
                .expect("query LoRA up projection")
                .forward(&q, stream)?;
            observe_activation(observer, prefix, "q_b_proj", &q)?;
            Ok(q)
        }
    }

    fn reconstruct_keys_values(
        &mut self,
        latent: &Array,
        rotary_key: &Array,
        stream: &Stream,
        prefix: &str,
        observer: &mut ObserverOption<'_>,
    ) -> Result<(Array, Array), Exception> {
        let batch = latent.dim(0);
        let sequence = latent.dim(1);
        let (k_nope, values) = if let Some(kv_b_proj) = &mut self.kv_b_proj {
            let kv_projected = kv_b_proj.forward(latent, stream)?;
            observe_activation(observer, prefix, "kv_b_proj", &kv_projected)?;
            let kv = kv_projected.reshape(
                &[
                    batch,
                    sequence,
                    self.num_heads,
                    self.qk_nope_head_dim + self.v_head_dim,
                ],
                stream,
            )?;
            (
                kv.try_index_device((.., .., .., ..self.qk_nope_head_dim), stream)?,
                kv.try_index_device((.., .., .., self.qk_nope_head_dim..), stream)?
                    .transpose_axes(&[0, 2, 1, 3], stream)?,
            )
        } else {
            let latent_heads = broadcast_to(
                latent.try_index_device((.., .., NewAxis, ..), stream)?,
                &[batch, sequence, self.num_heads, self.kv_lora_rank],
                stream,
            )?;
            let k_nope = self
                .k_b_proj
                .as_mut()
                .expect("split MLA key projection")
                .forward(&latent_heads, false, stream)?;
            observe_activation(observer, prefix, "k_b_proj", &k_nope)?;
            let values = self
                .v_b_proj
                .as_mut()
                .expect("split MLA value projection")
                .forward(&latent_heads, true, stream)?
                .transpose_axes(&[0, 2, 1, 3], stream)?;
            observe_activation(observer, prefix, "v_b_proj", &values)?;
            (k_nope, values)
        };
        observe_activation(observer, prefix, "keys_nope", &k_nope)?;
        observe_activation(observer, prefix, "values", &values)?;
        let keys = concatenate_axis(
            &[
                k_nope,
                broadcast_to(
                    rotary_key.try_index_device((.., .., NewAxis, ..), stream)?,
                    &[batch, sequence, self.num_heads, self.qk_rope_head_dim],
                    stream,
                )?,
            ],
            -1,
            stream,
        )?
        .transpose_axes(&[0, 2, 1, 3], stream)?;
        observe_activation(observer, prefix, "keys", &keys)?;
        Ok((keys, values))
    }

    #[allow(clippy::unnecessary_unwrap)]
    fn forward_impl(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        mut cache: Option<&mut CompressedLatentCache>,
        stream: &Stream,
        prefix: &str,
        observer: &mut ObserverOption<'_>,
    ) -> Result<Array, Exception> {
        observe_activation(observer, prefix, "input", x)?;
        let b = x.dim(0);
        let l = x.dim(1);
        let q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim;
        let q = self
            .project_queries(x, stream, prefix, observer)?
            .reshape(&[b, l, self.num_heads, q_head_dim], stream)?;
        observe_activation(observer, prefix, "queries", &q)?;
        let q_nope = q.try_index_device((.., .., .., ..self.qk_nope_head_dim), stream)?;
        observe_activation(observer, prefix, "queries_nope", &q_nope)?;
        let q_pe = q
            .try_index_device((.., .., .., self.qk_nope_head_dim..), stream)?
            .transpose_axes(&[0, 2, 1, 3], stream)?;
        observe_activation(observer, prefix, "queries_rope_input", &q_pe)?;

        let kv = self.kv_a_proj_with_mqa.forward(x, stream)?;
        observe_activation(observer, prefix, "kv_a_proj_with_mqa", &kv)?;
        let latent_raw = kv.try_index_device((.., .., ..self.kv_lora_rank), stream)?;
        observe_activation(observer, prefix, "latent_raw", &latent_raw)?;
        let latent = latent_raw;
        let latent = self.kv_a_layernorm.forward(&latent, stream)?;
        observe_activation(observer, prefix, "kv_a_layernorm", &latent)?;
        let k_pe = kv
            .try_index_device((.., .., self.kv_lora_rank..), stream)?
            .try_index_device((.., NewAxis, .., ..), stream)?;
        observe_activation(observer, prefix, "keys_rope_input", &k_pe)?;

        let offset = cache.as_ref().map_or(0, |cache| cache.offset());
        let q_pe = if self.use_nope {
            q_pe
        } else {
            self.rope.forward(
                nn::RopeInputBuilder::new(&q_pe).offset(offset).build()?,
                stream,
            )?
        };
        observe_activation(observer, prefix, "queries_rope", &q_pe)?;
        let k_pe = if self.use_nope {
            k_pe
        } else {
            self.rope.forward(
                nn::RopeInputBuilder::new(&k_pe).offset(offset).build()?,
                stream,
            )?
        };
        observe_activation(observer, prefix, "keys_rope", &k_pe)?;
        let new_k_pe = k_pe.try_index_device((.., 0, .., ..), stream)?;

        let view = match cache.as_deref_mut() {
            Some(cache) => cache
                .append(
                    CompressedAttentionState {
                        latent: latent.clone(),
                        rotary: new_k_pe.clone(),
                    },
                    stream,
                )
                .map_err(|error| Exception::custom(error.to_string()))?,
            None => CompressedAttentionView::Resident(CompressedAttentionState {
                latent: latent.clone(),
                rotary: new_k_pe.clone(),
            }),
        };
        let paged = view.is_paged();
        if paged && observer.is_some() {
            return Err(Exception::custom(
                "attention-probability inspection is unavailable for paged compressed-latent attention",
            ));
        }
        let observed = view.observable();
        let cached_latent = observed.latent.clone();
        let cached_k_pe = observed.rotary.clone();
        observe_activation(observer, prefix, "latent_cache", &cached_latent)?;
        observe_activation(observer, prefix, "rotary_key_cache", &cached_k_pe)?;
        if let Some(mask) = mask {
            observe_activation(observer, prefix, "attention_mask", mask)?;
        }

        // Every multi-token prefill reconstructs K/V transiently and stays on
        // MLX's fused attention path. Initial prefill uses the compact causal
        // mode; cached chunks use the explicit offset-aware mask constructed by
        // `TextModel`. Persistent state remains compressed and head-independent.
        let attended = if paged {
            let queries = concatenate_axis(
                &[q_nope, q_pe.transpose_axes(&[0, 2, 1, 3], stream)?],
                -1,
                stream,
            )?
            .transpose_axes(&[0, 2, 1, 3], stream)?;
            let mut accumulator = BlockwiseAttentionAccumulator::new(
                &queries,
                self.softmax_scale,
                mask,
                offset as i64,
                None,
                0,
                None,
                offset as i64 + l as i64,
                stream,
            )?;
            cache
                .expect("paged view requires a cache")
                .visit_blocks(l, stream, |block| {
                    let result = (|| -> Result<u64, Exception> {
                        let mut no_observer = None;
                        let (keys, values) = self.reconstruct_keys_values(
                            &block.state.latent,
                            &block.state.rotary,
                            stream,
                            prefix,
                            &mut no_observer,
                        )?;
                        let scratch = keys.nbytes() as u64 + values.nbytes() as u64;
                        let kv_block =
                            KeyValueAttentionBlock::unleased(block.start, block.end, keys, values);
                        accumulator.accumulate(&kv_block, stream)?;
                        accumulator.submit()?;
                        Ok(scratch)
                    })();
                    result.map_err(eredu_nn::Error::backend)
                })
                .map_err(|error| Exception::custom(error.to_string()))?;
            let output = accumulator.finish(stream)?;
            eval([&output])?;
            output.transpose_axes(&[0, 2, 1, 3], stream)?
        } else if l > 1 {
            let (keys, values) = self.reconstruct_keys_values(
                &cached_latent,
                &cached_k_pe,
                stream,
                prefix,
                observer,
            )?;
            let queries = concatenate_axis(
                &[q_nope, q_pe.transpose_axes(&[0, 2, 1, 3], stream)?],
                -1,
                stream,
            )?
            .transpose_axes(&[0, 2, 1, 3], stream)?;
            observe_activation(observer, prefix, "queries_combined", &queries)?;
            if observer.is_some() {
                let generated_causal_mask = if mask.is_none() {
                    Some(create_causal_mask(l, Some(offset), None, None, stream)?)
                } else {
                    None
                };
                if let Some(mask) = generated_causal_mask.as_ref() {
                    observe_activation(observer, prefix, "attention_mask", mask)?;
                }
                let probability_mask = mask.or(generated_causal_mask.as_ref());
                let probabilities = common::attention::attention_probabilities(
                    &queries,
                    &keys,
                    self.softmax_scale,
                    probability_mask,
                    stream,
                )?;
                observe_activation(observer, prefix, "attention_probs", &probabilities)?;
            }
            safemlx::fast::scaled_dot_product_attention(
                queries,
                keys,
                values,
                self.softmax_scale,
                Some(match mask {
                    Some(mask) => ScaledDotProductAttentionMask::Array(mask),
                    None => ScaledDotProductAttentionMask::Causal,
                }),
                None,
                stream,
            )?
            .transpose_axes(&[0, 2, 1, 3], stream)?
        } else {
            if self.kv_b_proj.is_none() {
                let q_latent = self
                    .k_b_proj
                    .as_mut()
                    .expect("split MLA key projection")
                    .forward(&q_nope, true, stream)?;
                observe_activation(observer, prefix, "queries_latent", &q_latent)?;
                let mut scores = einsum("blhc,btc->bhlt", [&q_latent, &cached_latent], stream)?
                    .add(
                        einsum("bhlr,btr->bhlt", [&q_pe, &cached_k_pe], stream)?,
                        stream,
                    )?
                    .multiply(Array::from_f32(self.softmax_scale), stream)?;
                if let Some(mask) = mask {
                    if mask.dtype() == Dtype::Bool {
                        scores = r#where(
                            mask,
                            &scores,
                            Array::from_f32(scores.dtype().finfo_min()? as f32),
                            stream,
                        )?;
                    } else {
                        scores = scores.add(mask, stream)?;
                    }
                }
                observe_activation(observer, prefix, "attention_scores", &scores)?;
                let probabilities = softmax_axis(scores, -1, true, stream)?;
                observe_activation(observer, prefix, "attention_probs", &probabilities)?;
                let context = einsum("bhlt,btc->blhc", [&probabilities, &cached_latent], stream)?;
                observe_activation(observer, prefix, "latent_context", &context)?;
                let values = self
                    .v_b_proj
                    .as_mut()
                    .expect("split MLA value projection")
                    .forward(&context, true, stream)?;
                observe_activation(observer, prefix, "v_b_proj", &values)?;
                values
            } else {
                let kv_b_proj = self.kv_b_proj.as_mut().expect("fused MLA projection");
                let fp8_group_ids = kv_b_proj.weight_scale_inv.as_ref().as_ref().map(|_| {
                    let mut ids = Vec::with_capacity((b * l * self.num_heads) as usize);
                    for _ in 0..b * l {
                        ids.extend(0..self.num_heads as u32);
                    }
                    Array::from_slice(&ids, &[b * l * self.num_heads])
                });

                let mut absorbed_weight = None;
                let q_latent = if let (Some(scale), Some(group_ids)) = (
                    kv_b_proj.weight_scale_inv.as_ref().as_ref(),
                    fp8_group_ids.as_ref(),
                ) {
                    common::fp8::segmented_transposed_linear(
                        &q_nope
                            .reshape(&[b * l * self.num_heads, self.qk_nope_head_dim], stream)?,
                        kv_b_proj.weight.as_ref(),
                        scale,
                        group_ids,
                        self.qk_nope_head_dim + self.v_head_dim,
                        0,
                        stream,
                    )?
                    .reshape(&[b, l, self.num_heads, self.kv_lora_rank], stream)?
                } else {
                    let weight = kv_b_proj.dequantized_weight(stream)?.reshape(
                        &[
                            self.num_heads,
                            self.qk_nope_head_dim + self.v_head_dim,
                            self.kv_lora_rank,
                        ],
                        stream,
                    )?;
                    let wk = weight.try_index_device((.., ..self.qk_nope_head_dim, ..), stream)?;
                    let q_latent = einsum("blhd,hdc->blhc", [&q_nope, &wk], stream)?;
                    absorbed_weight = Some(weight);
                    q_latent
                };
                observe_activation(observer, prefix, "queries_latent", &q_latent)?;
                let mut scores = einsum("blhc,btc->bhlt", [&q_latent, &cached_latent], stream)?
                    .add(
                        einsum("bhlr,btr->bhlt", [&q_pe, &cached_k_pe], stream)?,
                        stream,
                    )?
                    .multiply(Array::from_f32(self.softmax_scale), stream)?;
                if let Some(mask) = mask {
                    if mask.dtype() == Dtype::Bool {
                        scores = r#where(
                            mask,
                            &scores,
                            Array::from_f32(scores.dtype().finfo_min()? as f32),
                            stream,
                        )?;
                    } else {
                        scores = scores.add(mask, stream)?;
                    }
                }
                observe_activation(observer, prefix, "attention_scores", &scores)?;
                let probabilities = softmax_axis(scores, -1, true, stream)?;
                observe_activation(observer, prefix, "attention_probs", &probabilities)?;
                let context = einsum("bhlt,btc->blhc", [&probabilities, &cached_latent], stream)?;
                observe_activation(observer, prefix, "latent_context", &context)?;
                if let (Some(scale), Some(group_ids)) = (
                    kv_b_proj.weight_scale_inv.as_ref().as_ref(),
                    fp8_group_ids.as_ref(),
                ) {
                    common::fp8::segmented_linear(
                        &context.reshape(&[b * l * self.num_heads, self.kv_lora_rank], stream)?,
                        kv_b_proj.weight.as_ref(),
                        scale,
                        group_ids,
                        self.qk_nope_head_dim + self.v_head_dim,
                        self.qk_nope_head_dim,
                        self.v_head_dim,
                        stream,
                    )?
                    .reshape(&[b, l, self.num_heads, self.v_head_dim], stream)?
                } else {
                    let weight = absorbed_weight.expect("dense absorbed MLA weight initialized");
                    let wv = weight.try_index_device((.., self.qk_nope_head_dim.., ..), stream)?;
                    einsum("blhc,hvc->blhv", [&context, &wv], stream)?
                }
            }
        };
        observe_activation(observer, prefix, "attention", &attended)?;
        let attended = attended.reshape(&[b, l, self.num_heads * self.v_head_dim], stream)?;
        observe_activation(observer, prefix, "o_proj_input", &attended)?;
        let output = self.o_proj.forward(&attended, stream)?;
        observe_activation(observer, prefix, "o_proj", &output)?;
        Ok(output)
    }

    /// Runs shared MLA with optional activation observation.
    pub(crate) fn forward_shared(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<&mut CompressedLatentCache>,
        stream: &Stream,
        prefix: &str,
        observer: &mut Option<&mut dyn RuntimeActivationObserver<Array, Exception>>,
    ) -> Result<Array, Exception> {
        self.forward_impl(x, mask, cache, stream, prefix, observer)
    }
}
