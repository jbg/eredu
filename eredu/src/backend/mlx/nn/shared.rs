//! MLX implementation of backend-neutral neural operators.

use eredu_checkpoint::WeightQuantization;

use std::ops::{Deref, DerefMut};

use eredu_nn::{
    AttentionCache, Backend, EmbeddingOperator, Error as ComputeError, LinearOperator, LinearSpec,
    NormalizationOperator, RopeValue, RotaryOperator, RotarySpec,
};
use safemlx::{
    builder::Builder,
    distributed::Group,
    fast::ScaledDotProductAttentionMask,
    module::{Module, ModuleParam, ModuleParamMut, ModuleParamRef, ModuleParameters},
    nn,
    quantization::MaybeQuantized,
    Array, Dtype, Stream,
};

use crate::backend::mlx::{
    nn::{self as common, tensor::rope::RopeVariant},
    runtime::cache::{
        ConcatKeyValueCache, KeyValueCache, PagedKeyValueCache, SlidingKeyValueCache,
    },
};

fn compute<T>(result: Result<T, safemlx::error::Exception>) -> Result<T, ComputeError> {
    result.map_err(ComputeError::backend)
}

macro_rules! delegate_parameters {
    ($type:ty, $field:tt) => {
        impl ModuleParameters for $type {
            fn num_parameters(&self) -> usize {
                self.$field.num_parameters()
            }
            fn parameters(&self) -> ModuleParamRef<'_> {
                self.$field.parameters()
            }
            fn parameters_mut(&mut self) -> ModuleParamMut<'_> {
                self.$field.parameters_mut()
            }
            fn trainable_parameters(&self) -> ModuleParamRef<'_> {
                self.$field.trainable_parameters()
            }
            fn update(&mut self, parameters: ModuleParam) {
                self.$field.update(parameters);
            }
            fn freeze_parameters(&mut self, recursive: bool) {
                self.$field.freeze_parameters(recursive);
            }
            fn unfreeze_parameters(&mut self, recursive: bool) {
                self.$field.unfreeze_parameters(recursive);
            }
            fn all_frozen(&self) -> Option<bool> {
                self.$field.all_frozen()
            }
            fn any_frozen(&self) -> Option<bool> {
                self.$field.any_frozen()
            }
        }
    };
}

/// MLX dense-or-quantized affine projection.
#[derive(Debug, Clone)]
pub struct MlxLinear(pub MaybeQuantized<nn::Linear>);

impl Deref for MlxLinear {
    type Target = MaybeQuantized<nn::Linear>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for MlxLinear {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
delegate_parameters!(MlxLinear, 0);

impl LinearOperator<Array> for MlxLinear {
    fn forward(&mut self, input: &Array, context: &Stream) -> Result<Array, ComputeError> {
        compute(self.0.forward(input, context))
    }
}

/// MLX dense-or-quantized token embedding.
#[derive(Debug, Clone)]
pub struct MlxEmbedding(pub MaybeQuantized<nn::Embedding>);

impl Deref for MlxEmbedding {
    type Target = MaybeQuantized<nn::Embedding>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for MlxEmbedding {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
delegate_parameters!(MlxEmbedding, 0);

impl EmbeddingOperator<Array> for MlxEmbedding {
    fn forward(&mut self, input: &Array, context: &Stream) -> Result<Array, ComputeError> {
        compute(self.0.forward(input, context))
    }

    fn as_linear(&mut self, input: &Array, context: &Stream) -> Result<Array, ComputeError> {
        compute(match &mut self.0 {
            MaybeQuantized::Original(embedding) => embedding.as_linear(input, context),
            MaybeQuantized::Quantized(embedding) => embedding.as_linear(input, context),
        })
    }
}

/// MLX fused RMS normalization.
#[derive(Debug, Clone)]
pub struct MlxRmsNorm(pub nn::RmsNorm);

impl Deref for MlxRmsNorm {
    type Target = nn::RmsNorm;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for MlxRmsNorm {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
delegate_parameters!(MlxRmsNorm, 0);

impl NormalizationOperator<Array> for MlxRmsNorm {
    fn forward(&mut self, input: &Array, context: &Stream) -> Result<Array, ComputeError> {
        compute(self.0.forward(input, context))
    }
}

/// MLX RoPE variant selected from model metadata.
#[derive(Debug, Clone)]
pub struct MlxRotary(pub RopeVariant);

impl Deref for MlxRotary {
    type Target = RopeVariant;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for MlxRotary {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
delegate_parameters!(MlxRotary, 0);

impl RotaryOperator<Array> for MlxRotary {
    fn forward(
        &mut self,
        input: &Array,
        offset: i32,
        context: &Stream,
    ) -> Result<Array, ComputeError> {
        let rope_input = nn::RopeInputBuilder::new(input)
            .offset(offset)
            .build()
            .map_err(ComputeError::backend)?;
        compute(self.0.forward(rope_input, context))
    }
}

/// Zero-sized MLX backend selector. All calls are statically dispatched.
#[derive(Debug, Clone, Copy)]
pub struct MlxBackend;

impl Backend for MlxBackend {
    type Tensor = Array;
    type Linear = MlxLinear;
    type Embedding = MlxEmbedding;
    type Normalization = MlxRmsNorm;
    type Rotary = MlxRotary;
    type ParallelContext = Group;

    fn linear(spec: LinearSpec<'_>, context: &Stream) -> Result<MlxLinear, ComputeError> {
        compute(common::linear::unloaded_maybe_quantized_linear(
            spec.input,
            spec.output,
            spec.bias,
            spec.quantization,
            context,
        ))
        .map(MlxLinear)
    }

    fn embedding(
        vocabulary: i32,
        dimensions: i32,
        _weight_name: &str,
        quantization: Option<WeightQuantization>,
        context: &Stream,
    ) -> Result<MlxEmbedding, ComputeError> {
        compute(common::linear::unloaded_maybe_quantized_embedding(
            vocabulary,
            dimensions,
            quantization,
            context,
        ))
        .map(MlxEmbedding)
    }

    fn rms_norm(
        dimensions: i32,
        epsilon: f32,
        context: &Stream,
    ) -> Result<MlxRmsNorm, ComputeError> {
        compute(nn::RmsNorm::unloaded(
            dimensions,
            epsilon,
            Dtype::Float32,
            context,
        ))
        .map(MlxRmsNorm)
    }

    fn rotary(spec: RotarySpec<'_>, context: &Stream) -> Result<MlxRotary, ComputeError> {
        let scaling = spec.scaling.map(|values| {
            values
                .iter()
                .map(|(key, value)| {
                    let value = match value {
                        RopeValue::Float(value) => RopeValue::Float(*value),
                        RopeValue::String(value) => RopeValue::String(value.clone()),
                        RopeValue::Bool(value) => RopeValue::Bool(*value),
                    };
                    (key.clone(), value)
                })
                .collect()
        });
        compute(crate::backend::mlx::nn::tensor::rope::initialize_rope(
            spec.dimensions,
            spec.base,
            spec.traditional,
            &scaling,
            spec.max_positions,
            context,
        ))
        .map(MlxRotary)
    }

    fn silu(input: Array, context: &Stream) -> Result<Array, ComputeError> {
        compute(common::layers::silu(input, context))
    }

    fn attention(
        queries: Array,
        keys: Array,
        values: Array,
        scale: f32,
        mask: Option<&Array>,
        context: &Stream,
    ) -> Result<Array, ComputeError> {
        compute(safemlx::fast::scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale,
            mask.map(ScaledDotProductAttentionMask::Array),
            None,
            context,
        ))
    }

    fn sliding_window_attention(
        queries: Array,
        keys: Array,
        values: Array,
        scale: f32,
        window: i32,
        position_offset: i32,
        context: &Stream,
    ) -> Result<Array, ComputeError> {
        let batch = queries.dim(0);
        let sequence = queries.dim(2);
        compute(common::attention::sliding_window_prefill_attention(
            queries,
            keys,
            values,
            scale,
            window,
            position_offset,
            batch,
            sequence,
            context,
        ))
    }

    fn causal_mask(
        sequence: i32,
        offset: i32,
        window: Option<i32>,
        context: &Stream,
    ) -> Result<Array, ComputeError> {
        compute(crate::backend::mlx::nn::tensor::create_causal_mask(
            sequence,
            Some(offset),
            window,
            None,
            context,
        ))
    }

    fn row_parallel_linear(
        linear: &mut MlxLinear,
        input: &Array,
        parallel: &Group,
        context: &Stream,
    ) -> Result<Array, ComputeError> {
        compute(crate::backend::mlx::nn::parallel::forward_row_parallel(
            &mut linear.0,
            input,
            parallel,
            context,
        ))
    }
}

macro_rules! impl_attention_cache {
    ($type:ty) => {
        impl AttentionCache<Array> for $type {
            fn offset(&self) -> i32 {
                KeyValueCache::offset(self)
            }
            fn max_size(&self) -> Option<i32> {
                KeyValueCache::max_size(self)
            }
            fn update_for_attention(
                &mut self,
                keys: Array,
                values: Array,
                context: &Stream,
            ) -> Result<(Array, Array), ComputeError> {
                compute(KeyValueCache::update_for_attention(
                    self, keys, values, context,
                ))
            }
            fn attention(
                &mut self,
                queries: Array,
                keys: Array,
                values: Array,
                scale: f32,
                mask: Option<&Array>,
                context: &Stream,
            ) -> Result<Array, ComputeError> {
                if let Some(output) = compute(KeyValueCache::paged_attention(
                    self, &queries, scale, mask, None, context,
                ))? {
                    return Ok(output);
                }
                compute(
                    crate::backend::mlx::nn::tensor::scaled_dot_product_attention(
                        queries,
                        keys,
                        values,
                        Some(self),
                        scale,
                        mask,
                        context,
                    ),
                )
            }
        }
    };
}

impl_attention_cache!(ConcatKeyValueCache);
impl_attention_cache!(SlidingKeyValueCache);
impl_attention_cache!(PagedKeyValueCache);
