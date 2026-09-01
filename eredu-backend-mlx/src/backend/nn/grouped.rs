//! Group selection and packed grouped-projection implementations.

use eredu_checkpoint::WeightQuantization;
use eredu_nn::{GatedProductActivation, GatedProductPolicy, TensorParallelGroupedOutput};

use eredu_backend_mlx_macros::PhysicalParameters;
use safemlx::{
    error::Exception,
    ops::{
        arange, argpartition_axis, concatenate_axis, gather_qmm_with_mode,
        indexing::{scatter_single, take_along_axis, topk_axis, NewAxis, TryIndexOp},
        matmul, mean_axis, quantized_matmul_with_mode, quantized_packed_dimension, r#where, rsqrt,
        sigmoid, softmax_axis, sum_axis, zeros_dtype, QuantizationMode,
    },
    Array, Dtype, Stream,
};

use crate::{
    module::PhysicalParam,
    native_quantization::{native_grouped_linear, NativeQuantizedTensor},
};

use super::grouping::{
    gather_grouped_rows, gather_selection_values, grouped_matmul, topk_group_plan,
    GroupedSelectionPlan,
};
use super::layers::{relu2, silu};

/// Applies one affine- or MXFP4-packed group projection to group-major rows.
///
/// The packed weight and its metadata keep the group dimension leading, so
/// this is usable by any checkpoint layout once split groups have been
/// assembled into `[groups, output, input]` banks.
pub fn packed_grouped_linear(
    input: &Array,
    weight: &Array,
    scales: &Array,
    biases: Option<&Array>,
    group_ids: &Array,
    quantization: WeightQuantization,
    stream: &Stream,
) -> Result<Array, Exception> {
    packed_grouped_linear_with_transpose(
        input,
        weight,
        scales,
        biases,
        group_ids,
        quantization,
        true,
        stream,
    )
}

/// Applies a packed grouped projection in either matrix direction.
#[allow(clippy::too_many_arguments)]
pub fn packed_grouped_linear_with_transpose(
    input: &Array,
    weight: &Array,
    scales: &Array,
    biases: Option<&Array>,
    group_ids: &Array,
    quantization: WeightQuantization,
    transpose: bool,
    stream: &Stream,
) -> Result<Array, Exception> {
    packed_grouped_linear_with_options(
        input,
        weight,
        scales,
        biases,
        group_ids,
        quantization,
        transpose,
        true,
        stream,
    )
}

/// Applies a packed grouped projection with explicit selection-order metadata.
#[allow(clippy::too_many_arguments)]
pub fn packed_grouped_linear_with_options(
    input: &Array,
    weight: &Array,
    scales: &Array,
    biases: Option<&Array>,
    group_ids: &Array,
    quantization: WeightQuantization,
    transpose: bool,
    sorted_indices: bool,
    stream: &Stream,
) -> Result<Array, Exception> {
    let mode =
        crate::backend::runtime::checkpoint::quantization::mlx_quantization_mode(quantization)
            .map_err(|error| Exception::custom(error.to_string()))?;
    let selections = input.dim(0);
    let out_features = if transpose {
        weight.dim(-2)
    } else {
        weight.dim(-1) * 32 / quantization.bits()
    };
    if quantization.group_size() == 16 {
        if !transpose {
            return Err(Exception::custom(
                "group-16 affine group projections require transposed packed weights",
            ));
        }
        let selected_weight = weight.take_axis(group_ids, 0, stream)?;
        let selected_scales = scales.take_axis(group_ids, 0, stream)?;
        let selected_biases = biases
            .map(|biases| biases.take_axis(group_ids, 0, stream))
            .transpose()?;
        return quantized_matmul_with_mode(
            input.reshape(&[selections, 1, input.dim(-1)], stream)?,
            &selected_weight,
            &selected_scales,
            selected_biases.as_ref(),
            true,
            quantization.group_size(),
            quantization.bits(),
            mode,
            stream,
        )?
        .reshape(&[selections, out_features], stream);
    }

    let lhs_indices = arange::<i32, u32>(0, selections, 1, stream)?;
    gather_qmm_with_mode(
        input.reshape(&[selections, 1, input.dim(-1)], stream)?,
        weight,
        scales,
        biases,
        Some(&lhs_indices),
        Some(group_ids),
        transpose,
        quantization.group_size(),
        quantization.bits(),
        sorted_indices,
        mode,
        stream,
    )?
    .reshape(&[selections, out_features], stream)
}

/// Selector score transform used before top-k group selection.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
#[non_exhaustive]
pub enum TopKGroupScoring {
    /// Softmax scores before top-k selection.
    Softmax,
    /// Select raw logits first, then softmax only the selected entries.
    SelectedSoftmax,
    /// Sigmoid selector scores.
    Sigmoid,
    /// Square-root softplus selector scores.
    SqrtSoftplus,
}

impl TopKGroupScoring {
    fn requires_fp32(self) -> bool {
        matches!(self, Self::Sigmoid | Self::SqrtSoftplus)
    }

    fn apply(self, logits: Array, stream: &Stream) -> Result<Array, Exception> {
        match self {
            Self::Softmax => softmax_axis(logits, -1, true, stream),
            Self::SelectedSoftmax => Ok(logits),
            Self::Sigmoid => sigmoid(logits, stream),
            Self::SqrtSoftplus => super::layers::softplus(logits, stream)?.sqrt(stream),
        }
    }
}

/// Configuration for reusable top-k grouped selection.
#[derive(Debug, Clone, Copy)]
pub struct TopKGroupSelectorConfig {
    /// Number of selected groups per token.
    top_k: i32,
    /// Total number of selectable groups.
    group_count: i32,
    /// Hidden dimension consumed by the selector projection.
    hidden_size: i32,
    /// Score transform to apply to selector logits.
    score_function: TopKGroupScoring,
    /// Whether selected top-k weights are normalized after gathering.
    norm_topk_prob: bool,
    /// Optional epsilon added to the normalization denominator.
    normalization_epsilon: f32,
    /// Final multiplier applied to gathered selection weights.
    coefficient_scale: f32,
    /// Number of selection groups.
    n_group: i32,
    /// Number of selection groups selected before group top-k.
    topk_group: i32,
    /// Whether to allocate an ordinary per-group projection output bias.
    projection_bias: bool,
    /// Whether to allocate a selection-only group score correction bias.
    score_correction_bias: bool,
    /// Optional epsilon for weightless RMS normalization before projection.
    input_rms_epsilon: Option<f32>,
    /// Whether normalized inputs receive an additional inverse-sqrt-width scale.
    input_inverse_sqrt_dimensions: bool,
    /// Whether to allocate learned per-group selection multipliers.
    learned_coefficient_scale: bool,
}

impl TopKGroupSelectorConfig {
    /// Creates and validates a complete grouped-selector configuration.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        top_k: i32,
        group_count: i32,
        hidden_size: i32,
        score_function: TopKGroupScoring,
        norm_topk_prob: bool,
        normalization_epsilon: f32,
        coefficient_scale: f32,
        n_group: i32,
        topk_group: i32,
        projection_bias: bool,
        score_correction_bias: bool,
        input_rms_epsilon: Option<f32>,
        input_inverse_sqrt_dimensions: bool,
        learned_coefficient_scale: bool,
    ) -> Result<Self, Exception> {
        if top_k <= 0 || group_count <= 0 || top_k > group_count || hidden_size <= 0 {
            return Err(Exception::custom(
                "grouped selector requires positive dimensions and top_k no larger than group_count",
            ));
        }
        if n_group <= 0 || topk_group <= 0 || topk_group > n_group || group_count % n_group != 0 {
            return Err(Exception::custom(
                "grouped selector partitions must divide group_count and select a non-empty subset",
            ));
        }
        if !normalization_epsilon.is_finite()
            || normalization_epsilon < 0.0
            || !coefficient_scale.is_finite()
            || input_rms_epsilon.is_some_and(|epsilon| !epsilon.is_finite() || epsilon < 0.0)
        {
            return Err(Exception::custom(
                "grouped selector scaling values must be finite and epsilons non-negative",
            ));
        }
        Ok(Self {
            top_k,
            group_count,
            hidden_size,
            score_function,
            norm_topk_prob,
            normalization_epsilon,
            coefficient_scale,
            n_group,
            topk_group,
            projection_bias,
            score_correction_bias,
            input_rms_epsilon,
            input_inverse_sqrt_dimensions,
            learned_coefficient_scale,
        })
    }
}

/// Reusable top-k grouped selector.
#[derive(Debug, Clone, PhysicalParameters)]
#[module(root = crate)]
pub struct TopKGroupSelector {
    /// Number of selected groups per token.
    pub(crate) top_k: i32,
    /// Total number of selectable groups.
    pub(crate) group_count: i32,
    /// Logical input width of the selector projection.
    pub(crate) input_dims: i32,
    /// Selector score transform.
    pub(crate) score_function: TopKGroupScoring,
    /// Whether selected probabilities are normalized.
    pub(crate) norm_topk_prob: bool,
    /// Optional epsilon added to the normalization denominator.
    pub(crate) normalization_epsilon: f32,
    /// Final multiplier applied to selection weights.
    pub(crate) coefficient_scale: f32,
    /// Number of selection groups.
    pub(crate) n_group: i32,
    /// Number of selected partitions.
    pub(crate) topk_group: i32,
    #[param]
    /// Selector projection weight.
    pub(crate) weight: PhysicalParam<Array>,
    #[param]
    /// Optional ordinary projection output bias applied to selector logits.
    pub(crate) bias: PhysicalParam<Option<Array>>,
    #[param]
    /// Optional affine scales for a packed selector projection.
    pub(crate) scales: PhysicalParam<Option<Array>>,
    #[param]
    /// Optional affine biases for a packed selector projection.
    pub(crate) biases: PhysicalParam<Option<Array>>,
    #[param]
    /// Optional score correction bias used only when choosing groups.
    pub(crate) e_score_correction_bias: PhysicalParam<Option<Array>>,
    #[param]
    /// Optional learned feature scale applied to RMS-normalized inputs.
    pub(crate) input_scale: PhysicalParam<Option<Array>>,
    #[param]
    /// Optional learned multiplier gathered for each selected group.
    pub(crate) learned_coefficient_scale: PhysicalParam<Option<Array>>,
    /// Optional selector-input RMS epsilon.
    pub(crate) input_rms_epsilon: Option<f32>,
    /// Whether normalized selector inputs are divided by the square root of width.
    pub(crate) input_inverse_sqrt_dimensions: bool,
    /// Affine group size, or zero for a dense selector.
    pub(crate) group_size: i32,
    /// Affine bit width, or zero for a dense selector.
    pub(crate) bits: i32,
    /// Packed quantization encoding.
    pub(crate) mode: QuantizationMode,
    /// Checkpoint-native GGML encoding and byte order.
    pub(crate) iquant: Option<WeightQuantization>,
}

/// Selected group ids plus the score and weight arrays produced by a top-k selector.
pub struct GroupSelectionOutput {
    /// Selected group ids with shape `[tokens, top_k]`.
    pub(crate) indices: Array,
    /// Selector probabilities or scores gathered at the selected ids.
    pub(crate) scores: Array,
    /// Final selection weights after optional normalization/scaling.
    pub(crate) weights: Array,
}

impl TopKGroupSelector {
    /// Creates an unloaded dense or affine-packed selector.
    pub fn new_with_quantization(
        config: TopKGroupSelectorConfig,
        quantization: Option<WeightQuantization>,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        Self::new_with_quantization_and_dtype(config, quantization, Dtype::Float32, stream)
    }

    /// Creates an unloaded dense or affine-packed selector with an explicit dense dtype.
    pub fn new_with_quantization_and_dtype(
        config: TopKGroupSelectorConfig,
        quantization: Option<WeightQuantization>,
        dense_dtype: Dtype,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        if let Some(quantization) = quantization {
            if config.hidden_size <= 0 || config.hidden_size % quantization.group_size() != 0 {
                return Err(Exception::custom(format!(
                    "affine selector hidden dimension {} is not divisible by group size {}",
                    config.hidden_size,
                    quantization.group_size()
                )));
            }
        }
        let affine = quantization.filter(|q| !matches!(q, WeightQuantization::GgufIQuant { .. }));
        let mode = affine
            .map(crate::backend::runtime::checkpoint::quantization::mlx_quantization_mode)
            .transpose()
            .map_err(|error| Exception::custom(error.to_string()))?
            .unwrap_or(QuantizationMode::Affine);
        Ok(Self {
            top_k: config.top_k,
            group_count: config.group_count,
            input_dims: config.hidden_size,
            score_function: config.score_function,
            norm_topk_prob: config.norm_topk_prob,
            normalization_epsilon: config.normalization_epsilon,
            coefficient_scale: config.coefficient_scale,
            n_group: config.n_group,
            topk_group: config.topk_group,
            weight: match quantization {
                Some(WeightQuantization::GgufIQuant { ggml_type, .. }) => {
                    let (block_values, block_bytes) =
                        ggml_type.block_and_bytes().map_err(|_| {
                            Exception::custom(format!(
                                "{ggml_type:?} has no native selector block geometry"
                            ))
                        })?;
                    PhysicalParam::<Array>::unloaded(
                        &[
                            config.group_count,
                            config.hidden_size / block_values as i32 * block_bytes as i32,
                        ],
                        Dtype::Uint8,
                        stream,
                    )?
                }
                Some(quantization) => PhysicalParam::<Array>::unloaded(
                    &[
                        config.group_count,
                        quantized_packed_dimension(config.hidden_size, quantization.bits()),
                    ],
                    Dtype::Uint32,
                    stream,
                )?,
                None => PhysicalParam::<Array>::unloaded(
                    &[config.group_count, config.hidden_size],
                    dense_dtype,
                    stream,
                )?,
            },
            bias: if config.projection_bias {
                PhysicalParam::<Option<Array>>::unloaded_some(
                    &[config.group_count],
                    dense_dtype,
                    stream,
                )?
            } else {
                PhysicalParam::new(None)
            },
            scales: if let Some(quantization) = affine {
                PhysicalParam::<Option<Array>>::unloaded_some(
                    &[
                        config.group_count,
                        config.hidden_size / quantization.group_size(),
                    ],
                    if quantization == WeightQuantization::MxFp4 {
                        Dtype::Uint8
                    } else {
                        Dtype::Float16
                    },
                    stream,
                )?
            } else {
                PhysicalParam::new(None)
            },
            biases: if let Some(quantization) = affine.filter(|q| q.has_biases()) {
                PhysicalParam::<Option<Array>>::unloaded_some(
                    &[
                        config.group_count,
                        config.hidden_size / quantization.group_size(),
                    ],
                    Dtype::Float16,
                    stream,
                )?
            } else {
                PhysicalParam::new(None)
            },
            e_score_correction_bias: if config.score_correction_bias {
                PhysicalParam::<Option<Array>>::unloaded_some(
                    &[config.group_count],
                    dense_dtype,
                    stream,
                )?
            } else {
                PhysicalParam::new(None)
            },
            input_scale: if config.input_rms_epsilon.is_some() {
                PhysicalParam::<Option<Array>>::unloaded_some(
                    &[config.hidden_size],
                    dense_dtype,
                    stream,
                )?
            } else {
                PhysicalParam::new(None)
            },
            learned_coefficient_scale: if config.learned_coefficient_scale {
                PhysicalParam::<Option<Array>>::unloaded_some(
                    &[config.group_count],
                    dense_dtype,
                    stream,
                )?
            } else {
                PhysicalParam::new(None)
            },
            input_rms_epsilon: config.input_rms_epsilon,
            input_inverse_sqrt_dimensions: config.input_inverse_sqrt_dimensions,
            group_size: affine.map_or(0, WeightQuantization::group_size),
            bits: affine.map_or(0, WeightQuantization::bits),
            mode,
            iquant: quantization.filter(|q| matches!(q, WeightQuantization::GgufIQuant { .. })),
        })
    }

    /// Returns selected ids, pre-normalization selected scores, and final selection weights.
    pub fn select_with_selection_bias(
        &mut self,
        hidden_states: &Array,
        selection_bias: Option<&Array>,
        stream: &Stream,
    ) -> Result<GroupSelectionOutput, Exception> {
        let flat = self.transform_input(hidden_states, stream)?;
        let logits = if let Some(iquant) = self.iquant {
            let (ggml_type, endian) = iquant.gguf_iquant().expect("IQ selector format");
            NativeQuantizedTensor::from_iq_array(
                self.weight.value.clone(),
                &[self.group_count, self.input_dims],
                ggml_type,
                endian,
            )?
            .linear(&flat, true, stream)?
        } else if let Some(scales) = self.scales.as_ref() {
            let input = if self.score_function.requires_fp32() {
                flat.as_dtype(Dtype::Float32, stream)?
            } else {
                flat
            };
            quantized_matmul_with_mode(
                &input,
                self.weight.as_ref(),
                scales,
                self.biases.as_ref().as_ref(),
                true,
                self.group_size,
                self.bits,
                self.mode,
                stream,
            )?
        } else if self.score_function.requires_fp32() {
            matmul(
                &flat.as_dtype(Dtype::Float32, stream)?,
                &self
                    .weight
                    .as_ref()
                    .as_dtype(Dtype::Float32, stream)?
                    .transpose(stream)?,
                stream,
            )?
        } else {
            matmul(&flat, self.weight.as_ref().transpose(stream)?, stream)?
        };
        let logits = match self.bias.as_ref() {
            Some(bias) => logits.add(bias, stream)?,
            None => logits,
        };
        let scores = self.score_function.apply(logits, stream)?;
        let mut scores_for_choice = scores.clone();
        if let Some(bias) = self.e_score_correction_bias.as_ref() {
            scores_for_choice = scores_for_choice.add(bias, stream)?;
        }
        if let Some(bias) = selection_bias {
            scores_for_choice = scores_for_choice.add(bias, stream)?;
        }

        let top_k_index = self.topk_indices(&scores_for_choice, stream)?;
        let mut top_k_weights = take_along_axis(&scores, &top_k_index, -1, stream)?;
        if self.score_function == TopKGroupScoring::SelectedSoftmax {
            top_k_weights = softmax_axis(&top_k_weights, -1, true, stream)?;
        }
        let selected_scores = top_k_weights.clone();
        if self.norm_topk_prob {
            let mut denominator = sum_axis(&top_k_weights, -1, true, stream)?;
            if self.normalization_epsilon != 0.0 {
                denominator =
                    denominator.add(Array::from_f32(self.normalization_epsilon), stream)?;
            }
            top_k_weights = top_k_weights.divide(denominator, stream)?;
        }
        if self.coefficient_scale != 1.0 {
            top_k_weights =
                top_k_weights.multiply(Array::from_f32(self.coefficient_scale), stream)?;
        }
        if let Some(scale) = self.learned_coefficient_scale.as_ref() {
            top_k_weights =
                top_k_weights.multiply(scale.take_axis(&top_k_index, 0, stream)?, stream)?;
        }
        Ok(GroupSelectionOutput {
            indices: top_k_index,
            scores: selected_scores,
            weights: top_k_weights,
        })
    }

    /// Returns caller-selected ids, their raw transformed scores, and final
    /// normalized/scaled selection weights.
    pub fn select_indices(
        &mut self,
        hidden_states: &Array,
        group_indices: &Array,
        stream: &Stream,
    ) -> Result<GroupSelectionOutput, Exception> {
        let flat = self.transform_input(hidden_states, stream)?;
        let logits = if let Some(iquant) = self.iquant {
            let (ggml_type, endian) = iquant.gguf_iquant().expect("IQ selector format");
            NativeQuantizedTensor::from_iq_array(
                self.weight.value.clone(),
                &[self.group_count, self.input_dims],
                ggml_type,
                endian,
            )?
            .linear(&flat, true, stream)?
        } else if let Some(scales) = self.scales.as_ref() {
            let input = if self.score_function.requires_fp32() {
                flat.as_dtype(Dtype::Float32, stream)?
            } else {
                flat
            };
            quantized_matmul_with_mode(
                &input,
                self.weight.as_ref(),
                scales,
                self.biases.as_ref().as_ref(),
                true,
                self.group_size,
                self.bits,
                self.mode,
                stream,
            )?
        } else if self.score_function.requires_fp32() {
            matmul(
                &flat.as_dtype(Dtype::Float32, stream)?,
                &self
                    .weight
                    .as_ref()
                    .as_dtype(Dtype::Float32, stream)?
                    .transpose(stream)?,
                stream,
            )?
        } else {
            matmul(&flat, self.weight.as_ref().transpose(stream)?, stream)?
        };
        let logits = match self.bias.as_ref() {
            Some(bias) => logits.add(bias, stream)?,
            None => logits,
        };
        let scores = self.score_function.apply(logits, stream)?;
        let group_indices = group_indices.reshape(&[-1, self.top_k], stream)?;
        let mut weights = take_along_axis(scores, &group_indices, -1, stream)?;
        if self.score_function == TopKGroupScoring::SelectedSoftmax {
            weights = softmax_axis(&weights, -1, true, stream)?;
        }
        let selected_scores = weights.clone();
        if self.norm_topk_prob {
            let denominator = weights
                .sum_axis(-1, true, stream)?
                .add(Array::from_f32(self.normalization_epsilon), stream)?;
            weights = weights.divide(denominator, stream)?;
        }
        if self.coefficient_scale != 1.0 {
            weights = weights.multiply(Array::from_f32(self.coefficient_scale), stream)?;
        }
        if let Some(scale) = self.learned_coefficient_scale.as_ref() {
            weights = weights.multiply(scale.take_axis(&group_indices, 0, stream)?, stream)?;
        }
        Ok(GroupSelectionOutput {
            indices: group_indices,
            scores: selected_scores,
            weights,
        })
    }

    fn transform_input(&self, hidden_states: &Array, stream: &Stream) -> Result<Array, Exception> {
        let flat = hidden_states.reshape(&[-1, hidden_states.dim(-1)], stream)?;
        let Some(scale) = self.input_scale.as_ref() else {
            return Ok(flat);
        };
        let epsilon = self
            .input_rms_epsilon
            .expect("selector input scale requires an RMS epsilon");
        let variance = mean_axis(&flat.square(stream)?, -1, true, stream)?;
        let normalized = flat.multiply(
            rsqrt(variance.add(Array::from_f32(epsilon), stream)?, stream)?,
            stream,
        )?;
        let scaled = normalized.multiply(scale, stream)?;
        if self.input_inverse_sqrt_dimensions {
            scaled.multiply(
                Array::from_f32((self.input_dims as f32).sqrt().recip()),
                stream,
            )
        } else {
            Ok(scaled)
        }
    }

    fn topk_indices(&self, scores_for_choice: &Array, stream: &Stream) -> Result<Array, Exception> {
        if self.n_group == 1 && self.topk_group == 1 {
            return argpartition_axis(scores_for_choice, -self.top_k, -1, stream)?
                .try_index_device((.., -self.top_k..), stream);
        }
        if self.n_group <= 0
            || self.topk_group <= 0
            || self.topk_group > self.n_group
            || self.group_count % self.n_group != 0
        {
            return Err(Exception::custom("invalid grouped selector configuration"));
        }

        let tokens = scores_for_choice.dim(0);
        let entries_per_partition = self.group_count / self.n_group;
        let grouped =
            scores_for_choice.reshape(&[tokens, self.n_group, entries_per_partition], stream)?;
        let group_top = 2.min(entries_per_partition);
        let group_scores = sum_axis(
            &topk_axis(grouped, group_top, -1, stream)?,
            -1,
            false,
            stream,
        )?;
        let group_idx = argpartition_axis(&group_scores, -self.topk_group, -1, stream)?
            .try_index_device((.., -self.topk_group..), stream)?;

        let partition_ids: Vec<i32> = (0..self.group_count)
            .map(|group| group / entries_per_partition)
            .collect();
        let partition_ids = Array::from_slice(&partition_ids, &[1, 1, self.group_count]);
        let selected_groups = group_idx.try_index_device((.., .., NewAxis), stream)?;
        let group_mask = selected_groups.eq(partition_ids, stream)?;
        let group_mask = sum_axis(
            &group_mask.as_dtype(Dtype::Int32, stream)?,
            1,
            false,
            stream,
        )?
        .gt(Array::from_int(0), stream)?;
        let masked_scores = r#where(
            &group_mask,
            scores_for_choice,
            Array::from_f32(f32::NEG_INFINITY),
            stream,
        )?;
        argpartition_axis(masked_scores, -self.top_k, -1, stream)?
            .try_index_device((.., -self.top_k..), stream)
    }
}

/// Applies selection weights and reduces group-major selection outputs back to source tokens.
pub(crate) fn weighted_group_sum(
    current: Array,
    top_k_weights: &Array,
    plan: &GroupedSelectionPlan,
    num_tokens: i32,
    stream: &Stream,
) -> Result<Array, Exception> {
    let weights = gather_selection_values(top_k_weights, plan, stream)?
        .try_index_device((.., NewAxis), stream)?;
    let weighted = current.multiply(weights, stream)?;

    // Each selection index is unique, so restore the group-major rows with a
    // collision-free scatter and reduce the original top-k slots in their
    // stable order. A segment sum can use unordered GPU atomics here; the
    // resulting roundoff was sufficient to change near-tied downstream selection
    // decisions between identical passes.
    let selections = weighted.dim(0);
    let width = weighted.dim(-1);
    let ordered = scatter_single(
        zeros_dtype(&[selections, width], weighted.dtype(), stream)?,
        &plan.selection_indices,
        weighted.reshape(&[selections, 1, width], stream)?,
        0,
        stream,
    )?;
    let top_k = top_k_weights.dim(-1);
    let ordered = ordered.reshape(&[num_tokens, top_k, width], stream)?;
    sum_axis(ordered, 1, false, stream)
}

/// Packed grouped ReLU2 bank with dense, affine, MXFP4, or GGUF-native IQ storage.
#[derive(Debug, Clone, PhysicalParameters)]
#[module(root = crate)]
pub struct PackedRelu2Groups {
    /// Number of groups.
    pub group_count: i32,
    /// Input and output feature dimension.
    pub hidden_size: i32,
    /// Per-group intermediate dimension.
    pub intermediate_size: i32,
    /// Optional affine or MXFP4 settings for the up-projection bank.
    pub up_quantization: Option<WeightQuantization>,
    /// Optional affine or MXFP4 settings for the down-projection bank.
    pub down_quantization: Option<WeightQuantization>,
    /// Optional checkpoint-native IQ settings for the up-projection bank.
    pub up_iquant: Option<WeightQuantization>,
    /// Optional checkpoint-native IQ settings for the down-projection bank.
    pub down_iquant: Option<WeightQuantization>,
    #[param]
    /// Group up-projection weights.
    pub up_proj: PhysicalParam<Array>,
    #[param]
    /// Group up-projection packed scales.
    pub up_proj_scales: PhysicalParam<Option<Array>>,
    #[param]
    /// Group up-projection affine biases, absent for MXFP4.
    pub up_proj_biases: PhysicalParam<Option<Array>>,
    #[param]
    /// Group down-projection weights.
    pub down_proj: PhysicalParam<Array>,
    #[param]
    /// Group down-projection packed scales.
    pub down_proj_scales: PhysicalParam<Option<Array>>,
    #[param]
    /// Group down-projection affine biases, absent for MXFP4.
    pub down_proj_biases: PhysicalParam<Option<Array>>,
}

impl PackedRelu2Groups {
    /// Creates an unloaded dense, packed, or checkpoint-native IQ group bank.
    pub fn new(
        group_count: i32,
        hidden_size: i32,
        intermediate_size: i32,
        quantization: [Option<WeightQuantization>; 2],
        stream: &Stream,
    ) -> Result<Self, Exception> {
        Self::new_with_dtype(
            group_count,
            hidden_size,
            intermediate_size,
            quantization,
            Dtype::Float32,
            stream,
        )
    }

    /// Creates an unloaded group bank with an explicit dense weight dtype.
    pub fn new_with_dtype(
        group_count: i32,
        hidden_size: i32,
        intermediate_size: i32,
        quantization: [Option<WeightQuantization>; 2],
        dense_dtype: Dtype,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        let split = |quantization| {
            Ok::<_, Exception>(match quantization {
                Some(iq @ WeightQuantization::GgufIQuant { .. }) => (None, Some(iq)),
                packed => (packed, None),
            })
        };
        let (up_quantization, up_iquant) = split(quantization[0])?;
        let (down_quantization, down_iquant) = split(quantization[1])?;
        let projection = |out_features: i32,
                          in_features: i32,
                          quantization: Option<WeightQuantization>,
                          iquant: Option<WeightQuantization>|
         -> Result<GroupProjectionParams, Exception> {
            if let Some(iquant) = iquant {
                let (ggml_type, _) = iquant.gguf_iquant().expect("IQ group format");
                let (block_values, block_bytes) = ggml_type
                    .block_and_bytes()
                    .expect("canonical IQ block geometry");
                return Ok((
                    PhysicalParam::<Array>::unloaded(
                        &[
                            group_count,
                            out_features,
                            in_features / block_values as i32 * block_bytes as i32,
                        ],
                        Dtype::Uint8,
                        stream,
                    )?,
                    PhysicalParam::new(None),
                    PhysicalParam::new(None),
                ));
            }
            match quantization {
                Some(quantization) => Ok((
                    PhysicalParam::<Array>::unloaded(
                        &[
                            group_count,
                            out_features,
                            quantized_packed_dimension(in_features, quantization.bits()),
                        ],
                        Dtype::Uint32,
                        stream,
                    )?,
                    PhysicalParam::<Option<Array>>::unloaded_some(
                        &[
                            group_count,
                            out_features,
                            in_features / quantization.group_size(),
                        ],
                        if quantization == WeightQuantization::MxFp4 {
                            Dtype::Uint8
                        } else {
                            Dtype::Float16
                        },
                        stream,
                    )?,
                    if quantization.has_biases() {
                        PhysicalParam::<Option<Array>>::unloaded_some(
                            &[
                                group_count,
                                out_features,
                                in_features / quantization.group_size(),
                            ],
                            Dtype::Float16,
                            stream,
                        )?
                    } else {
                        PhysicalParam::new(None)
                    },
                )),
                None => Ok((
                    PhysicalParam::<Array>::unloaded(
                        &[group_count, out_features, in_features],
                        dense_dtype,
                        stream,
                    )?,
                    PhysicalParam::new(None),
                    PhysicalParam::new(None),
                )),
            }
        };
        let (up_proj, up_proj_scales, up_proj_biases) =
            projection(intermediate_size, hidden_size, up_quantization, up_iquant)?;
        let (down_proj, down_proj_scales, down_proj_biases) = projection(
            hidden_size,
            intermediate_size,
            down_quantization,
            down_iquant,
        )?;
        Ok(Self {
            group_count,
            hidden_size,
            intermediate_size,
            up_quantization,
            down_quantization,
            up_iquant,
            down_iquant,
            up_proj,
            up_proj_scales,
            up_proj_biases,
            down_proj,
            down_proj_scales,
            down_proj_biases,
        })
    }

    /// Evaluates selected groups and reduces their outputs back to tokens.
    pub fn forward(
        &mut self,
        hidden_states: &Array,
        top_k_index: &Array,
        top_k_weights: &Array,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let num_tokens = hidden_states.dim(0);
        let plan = topk_group_plan(top_k_index, stream)?;
        let hidden = gather_grouped_rows(hidden_states, &plan, stream)?;
        let hidden = if let Some(iquant) = self.up_iquant {
            let (ggml_type, endian) = iquant.gguf_iquant().expect("IQ group format");
            let native = NativeQuantizedTensor::from_iq_array(
                self.up_proj.value.clone(),
                &[self.group_count, self.intermediate_size, self.hidden_size],
                ggml_type,
                endian,
            )?;
            native_grouped_linear(&hidden, &native, &plan.sorted_group_ids, stream)?
        } else {
            match self.up_quantization {
                Some(quantization) => packed_grouped_linear(
                    &hidden,
                    &self.up_proj,
                    self.up_proj_scales
                        .as_ref()
                        .as_ref()
                        .expect("quantized group scales"),
                    self.up_proj_biases.as_ref().as_ref(),
                    &plan.sorted_group_ids,
                    quantization,
                    stream,
                )?,
                None => grouped_matmul(
                    &hidden,
                    &self.up_proj.as_ref().swap_axes(-1, -2, stream)?,
                    &plan.sorted_group_ids,
                    true,
                    stream,
                )?,
            }
        };
        let hidden = relu2(hidden, stream)?;
        let current = if let Some(iquant) = self.down_iquant {
            let (ggml_type, endian) = iquant.gguf_iquant().expect("IQ group format");
            let native = NativeQuantizedTensor::from_iq_array(
                self.down_proj.value.clone(),
                &[self.group_count, self.hidden_size, self.intermediate_size],
                ggml_type,
                endian,
            )?;
            native_grouped_linear(&hidden, &native, &plan.sorted_group_ids, stream)?
        } else {
            match self.down_quantization {
                Some(quantization) => packed_grouped_linear(
                    &hidden,
                    &self.down_proj,
                    self.down_proj_scales
                        .as_ref()
                        .as_ref()
                        .expect("quantized group scales"),
                    self.down_proj_biases.as_ref().as_ref(),
                    &plan.sorted_group_ids,
                    quantization,
                    stream,
                )?,
                None => grouped_matmul(
                    &hidden,
                    &self.down_proj.as_ref().swap_axes(-1, -2, stream)?,
                    &plan.sorted_group_ids,
                    true,
                    stream,
                )?,
            }
        };
        weighted_group_sum(current, top_k_weights, &plan, num_tokens, stream)
    }

    /// Returns the rank-local ReLU2 contribution for one tensor-parallel sum.
    pub fn forward_tensor_parallel(
        &mut self,
        hidden_states: &Array,
        top_k_index: &Array,
        top_k_weights: &Array,
        partitions: usize,
        stream: &Stream,
    ) -> Result<TensorParallelGroupedOutput<Array>, Exception> {
        if partitions == 0 {
            return Err(Exception::custom(
                "tensor-parallel partition count must be positive",
            ));
        }
        self.forward(hidden_states, top_k_index, top_k_weights, stream)
            .map(|reducible| TensorParallelGroupedOutput::new(reducible, None))
    }
}

const GROUPED_PROJECTION_CHUNK_THRESHOLD: i32 = 64;
const GROUPED_PROJECTION_CHUNK_TOKENS: i32 = 32;

/// Packed gated-product bank with optional MLX affine or MXFP4 projections.
#[derive(Debug, Clone, PhysicalParameters)]
#[module(root = crate)]
pub struct PackedGatedProductGroups {
    /// Number of groups.
    pub group_count: i32,
    /// Input and output feature dimension.
    pub hidden_dim: i32,
    /// Per-group intermediate dimension.
    pub intermediate_dim: i32,
    /// Exact gate activation, bounds, sigmoid multiplier, and up offset.
    pub policy: GatedProductPolicy,
    /// Optional encoding for the concatenated gate/up projection.
    pub gate_up_affine: Option<WeightQuantization>,
    /// Optional encoding for the down projection.
    pub down_affine: Option<WeightQuantization>,
    /// Optional checkpoint-native IQ encoding for the gate/up projection.
    pub gate_up_iquant: Option<WeightQuantization>,
    /// Optional checkpoint-native IQ encoding for the down projection.
    pub down_iquant: Option<WeightQuantization>,
    /// Whether weights/scales use checkpoint-native block FP8 with E8M0 scales.
    pub native_fp8_e8m0: bool,
    #[param]
    /// Concatenated gate/up weights shaped `[groups, 2 * intermediate, hidden]`.
    pub gate_up_proj: PhysicalParam<Array>,
    #[param]
    /// Optional ordinary gate/up output bias shaped `[groups, 2 * intermediate]`.
    pub gate_up_proj_bias: PhysicalParam<Option<Array>>,
    #[param]
    /// Gate/up quantization scales.
    pub gate_up_proj_scales: PhysicalParam<Option<Array>>,
    #[param]
    /// Gate/up quantization biases.
    pub gate_up_proj_biases: PhysicalParam<Option<Array>>,
    #[param]
    /// Down weights shaped `[groups, hidden, intermediate]`.
    pub down_proj: PhysicalParam<Array>,
    #[param]
    /// Optional ordinary down output bias shaped `[groups, hidden]`.
    pub down_proj_bias: PhysicalParam<Option<Array>>,
    #[param]
    /// Down quantization scales.
    pub down_proj_scales: PhysicalParam<Option<Array>>,
    #[param]
    /// Down quantization biases.
    pub down_proj_biases: PhysicalParam<Option<Array>>,
}

type GroupProjectionParams = (
    PhysicalParam<Array>,
    PhysicalParam<Option<Array>>,
    PhysicalParam<Option<Array>>,
);

impl PackedGatedProductGroups {
    /// Creates an unloaded packed group bank.
    pub fn new(
        group_count: i32,
        hidden_dim: i32,
        intermediate_dim: i32,
        gate_up_affine: Option<WeightQuantization>,
        down_affine: Option<WeightQuantization>,
        projection_biases: [bool; 2],
        stream: &Stream,
    ) -> Result<Self, Exception> {
        Self::new_with_dtype(
            group_count,
            hidden_dim,
            intermediate_dim,
            gate_up_affine,
            down_affine,
            projection_biases,
            Dtype::Float32,
            stream,
        )
    }

    /// Creates an unloaded packed group bank with an explicit dense weight dtype.
    pub fn new_with_dtype(
        group_count: i32,
        hidden_dim: i32,
        intermediate_dim: i32,
        gate_up_affine: Option<WeightQuantization>,
        down_affine: Option<WeightQuantization>,
        projection_biases: [bool; 2],
        dense_dtype: Dtype,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        let (gate_up_affine, gate_up_iquant) = match gate_up_affine {
            Some(iq @ WeightQuantization::GgufIQuant { .. }) => (None, Some(iq)),
            affine => (affine, None),
        };
        let (down_affine, down_iquant) = match down_affine {
            Some(iq @ WeightQuantization::GgufIQuant { .. }) => (None, Some(iq)),
            affine => (affine, None),
        };
        let projection = |out_features: i32,
                          in_features: i32,
                          quantization: Option<WeightQuantization>,
                          iquant: Option<WeightQuantization>|
         -> Result<GroupProjectionParams, Exception> {
            if let Some(iquant) = iquant {
                let (ggml_type, _) = iquant.gguf_iquant().expect("IQ group format");
                let (block_values, block_bytes) = ggml_type
                    .block_and_bytes()
                    .expect("canonical IQ block geometry");
                Ok((
                    PhysicalParam::<Array>::unloaded(
                        &[
                            group_count,
                            out_features,
                            in_features / block_values as i32 * block_bytes as i32,
                        ],
                        Dtype::Uint8,
                        stream,
                    )?,
                    PhysicalParam::new(None),
                    PhysicalParam::new(None),
                ))
            } else if let Some(quantization) = quantization {
                if in_features % quantization.group_size() != 0 {
                    return Err(Exception::custom(format!(
                        "packed group input width {in_features} is not divisible by {quantization:?} group size {}",
                        quantization.group_size(),
                    )));
                }
                Ok((
                    PhysicalParam::<Array>::unloaded(
                        &[
                            group_count,
                            out_features,
                            quantized_packed_dimension(in_features, quantization.bits()),
                        ],
                        Dtype::Uint32,
                        stream,
                    )?,
                    PhysicalParam::<Option<Array>>::unloaded_some(
                        &[
                            group_count,
                            out_features,
                            in_features / quantization.group_size(),
                        ],
                        if quantization == WeightQuantization::MxFp4 {
                            Dtype::Uint8
                        } else {
                            Dtype::Float16
                        },
                        stream,
                    )?,
                    if quantization.has_biases() {
                        PhysicalParam::<Option<Array>>::unloaded_some(
                            &[
                                group_count,
                                out_features,
                                in_features / quantization.group_size(),
                            ],
                            Dtype::Float16,
                            stream,
                        )?
                    } else {
                        PhysicalParam::new(None)
                    },
                ))
            } else {
                Ok((
                    PhysicalParam::<Array>::unloaded(
                        &[group_count, out_features, in_features],
                        dense_dtype,
                        stream,
                    )?,
                    PhysicalParam::new(None),
                    PhysicalParam::new(None),
                ))
            }
        };
        let (gate_up_proj, gate_up_proj_scales, gate_up_proj_biases) = projection(
            2 * intermediate_dim,
            hidden_dim,
            gate_up_affine,
            gate_up_iquant,
        )?;
        let (down_proj, down_proj_scales, down_proj_biases) =
            projection(hidden_dim, intermediate_dim, down_affine, down_iquant)?;
        Ok(Self {
            group_count,
            hidden_dim,
            intermediate_dim,
            policy: GatedProductPolicy::ordinary_silu(),
            gate_up_affine,
            down_affine,
            gate_up_iquant,
            down_iquant,
            native_fp8_e8m0: false,
            gate_up_proj,
            gate_up_proj_bias: if projection_biases[0] {
                PhysicalParam::<Option<Array>>::unloaded_some(
                    &[group_count, 2 * intermediate_dim],
                    dense_dtype,
                    stream,
                )?
            } else {
                PhysicalParam::new(None)
            },
            gate_up_proj_scales,
            gate_up_proj_biases,
            down_proj,
            down_proj_bias: if projection_biases[1] {
                PhysicalParam::<Option<Array>>::unloaded_some(
                    &[group_count, hidden_dim],
                    dense_dtype,
                    stream,
                )?
            } else {
                PhysicalParam::new(None)
            },
            down_proj_scales,
            down_proj_biases,
        })
    }

    /// Selects a validated gated-product equation.
    pub fn with_policy(mut self, policy: GatedProductPolicy) -> Result<Self, Exception> {
        policy
            .validate()
            .map_err(|error| Exception::custom(error.to_string()))?;
        self.policy = policy;
        Ok(self)
    }

    /// Rebuilds projection storage for native block-FP8 group tensors.
    pub fn with_native_fp8_e8m0(mut self, stream: &Stream) -> Result<Self, Exception> {
        let ceil128 = |value: i32| (value + 127) / 128;
        self.gate_up_proj = PhysicalParam::<Array>::unloaded(
            &[self.group_count, 2 * self.intermediate_dim, self.hidden_dim],
            Dtype::Uint8,
            stream,
        )?;
        self.gate_up_proj_scales = PhysicalParam::<Option<Array>>::unloaded_some(
            &[
                self.group_count,
                ceil128(2 * self.intermediate_dim),
                ceil128(self.hidden_dim),
            ],
            Dtype::Uint8,
            stream,
        )?;
        self.down_proj = PhysicalParam::<Array>::unloaded(
            &[self.group_count, self.hidden_dim, self.intermediate_dim],
            Dtype::Uint8,
            stream,
        )?;
        self.down_proj_scales = PhysicalParam::<Option<Array>>::unloaded_some(
            &[
                self.group_count,
                ceil128(self.hidden_dim),
                ceil128(self.intermediate_dim),
            ],
            Dtype::Uint8,
            stream,
        )?;
        self.native_fp8_e8m0 = true;
        Ok(self)
    }

    fn forward_chunk(
        &mut self,
        hidden_states: &Array,
        top_k_index: &Array,
        top_k_weights: &Array,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let num_tokens = hidden_states.dim(0);
        let plan = topk_group_plan(top_k_index, stream)?;
        let hidden = gather_grouped_rows(hidden_states, &plan, stream)?;
        let gate_up = if self.native_fp8_e8m0 {
            crate::backend::nn::fp8::grouped_linear(
                &hidden,
                self.gate_up_proj.as_ref(),
                self.gate_up_proj_scales
                    .as_ref()
                    .as_ref()
                    .expect("native FP8 gate/up scales"),
                &plan.sorted_group_ids,
                stream,
            )?
        } else if let Some(iquant) = self.gate_up_iquant {
            let (ggml_type, endian) = iquant.gguf_iquant().expect("IQ group format");
            let native = NativeQuantizedTensor::from_iq_array(
                self.gate_up_proj.value.clone(),
                &[self.group_count, 2 * self.intermediate_dim, self.hidden_dim],
                ggml_type,
                endian,
            )?;
            native_grouped_linear(&hidden, &native, &plan.sorted_group_ids, stream)?
        } else if let Some(quantization) = self.gate_up_affine {
            packed_grouped_linear(
                &hidden,
                self.gate_up_proj.as_ref(),
                self.gate_up_proj_scales
                    .as_ref()
                    .as_ref()
                    .expect("quantized gate/up scales"),
                self.gate_up_proj_biases.as_ref().as_ref(),
                &plan.sorted_group_ids,
                quantization,
                stream,
            )?
        } else {
            grouped_matmul(
                &hidden,
                &self.gate_up_proj.as_ref().swap_axes(-1, -2, stream)?,
                &plan.sorted_group_ids,
                true,
                stream,
            )?
        };
        let gate_up = match self.gate_up_proj_bias.as_ref() {
            Some(bias) => {
                gate_up.add(bias.take_axis(&plan.sorted_group_ids, 0, stream)?, stream)?
            }
            None => gate_up,
        };
        let mut gate = gate_up.try_index_device((.., ..self.intermediate_dim), stream)?;
        let mut up = gate_up.try_index_device((.., self.intermediate_dim..), stream)?;
        if let Some(bound) = self.policy.gate_upper_bound() {
            gate = safemlx::ops::clip(gate, ((), bound), stream)?;
        }
        if let Some(bound) = self.policy.up_absolute_bound() {
            up = safemlx::ops::clip(up, (-bound, bound), stream)?;
        }
        if self.policy.up_offset() != 0.0 {
            up = up.add(Array::from_f32(self.policy.up_offset()), stream)?;
        }
        let gate = match self.policy.activation() {
            GatedProductActivation::Silu if self.policy.sigmoid_multiplier() == 1.0 => {
                silu(gate, stream)?
            }
            GatedProductActivation::Silu => gate.multiply(
                sigmoid(
                    gate.multiply(Array::from_f32(self.policy.sigmoid_multiplier()), stream)?,
                    stream,
                )?,
                stream,
            )?,
            GatedProductActivation::GeluApproximate => {
                super::layers::gelu_approximate(gate, stream)?
            }
            _ => {
                return Err(Exception::custom(
                    "unsupported grouped gated-product activation",
                ))
            }
        };
        let activated = gate.multiply(up, stream)?;
        let output = if self.native_fp8_e8m0 {
            crate::backend::nn::fp8::grouped_linear(
                &activated,
                self.down_proj.as_ref(),
                self.down_proj_scales
                    .as_ref()
                    .as_ref()
                    .expect("native FP8 down scales"),
                &plan.sorted_group_ids,
                stream,
            )?
        } else if let Some(iquant) = self.down_iquant {
            let (ggml_type, endian) = iquant.gguf_iquant().expect("IQ group format");
            let native = NativeQuantizedTensor::from_iq_array(
                self.down_proj.value.clone(),
                &[self.group_count, self.hidden_dim, self.intermediate_dim],
                ggml_type,
                endian,
            )?;
            native_grouped_linear(&activated, &native, &plan.sorted_group_ids, stream)?
        } else if let Some(quantization) = self.down_affine {
            packed_grouped_linear(
                &activated,
                self.down_proj.as_ref(),
                self.down_proj_scales
                    .as_ref()
                    .as_ref()
                    .expect("quantized down scales"),
                self.down_proj_biases.as_ref().as_ref(),
                &plan.sorted_group_ids,
                quantization,
                stream,
            )?
        } else {
            grouped_matmul(
                &activated,
                &self.down_proj.as_ref().swap_axes(-1, -2, stream)?,
                &plan.sorted_group_ids,
                true,
                stream,
            )?
        };
        let output = match self.down_proj_bias.as_ref() {
            Some(bias) => output.add(bias.take_axis(&plan.sorted_group_ids, 0, stream)?, stream)?,
            None => output,
        };
        weighted_group_sum(output, top_k_weights, &plan, num_tokens, stream)
    }

    /// Evaluates selected groups and reduces selection outputs back to source tokens.
    pub fn forward(
        &mut self,
        hidden_states: &Array,
        top_k_index: &Array,
        top_k_weights: &Array,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let num_tokens = hidden_states.dim(0);
        if num_tokens <= GROUPED_PROJECTION_CHUNK_THRESHOLD {
            return self.forward_chunk(hidden_states, top_k_index, top_k_weights, stream);
        }
        let mut outputs = Vec::new();
        let mut start = 0;
        while start < num_tokens {
            let end = (start + GROUPED_PROJECTION_CHUNK_TOKENS).min(num_tokens);
            outputs.push(self.forward_chunk(
                &hidden_states.try_index_device((start..end, ..), stream)?,
                &top_k_index.try_index_device((start..end, ..), stream)?,
                &top_k_weights.try_index_device((start..end, ..), stream)?,
                stream,
            )?);
            start = end;
        }
        concatenate_axis(&outputs, 0, stream)
    }

    /// Separates the rank-local projection contribution from replicated grouped
    /// down bias so the latter can be added literally once after all-sum.
    pub fn forward_tensor_parallel(
        &mut self,
        hidden_states: &Array,
        top_k_index: &Array,
        top_k_weights: &Array,
        partitions: usize,
        stream: &Stream,
    ) -> Result<TensorParallelGroupedOutput<Array>, Exception> {
        if partitions == 0 {
            return Err(Exception::custom(
                "tensor-parallel partition count must be positive",
            ));
        }
        let output = self.forward(hidden_states, top_k_index, top_k_weights, stream)?;
        let Some(bias) = self.down_proj_bias.as_ref() else {
            return Ok(TensorParallelGroupedOutput::new(output, None));
        };
        let plan = topk_group_plan(top_k_index, stream)?;
        let selected_bias = bias.take_axis(&plan.sorted_group_ids, 0, stream)?;
        let bias = weighted_group_sum(
            selected_bias,
            top_k_weights,
            &plan,
            hidden_states.dim(0),
            stream,
        )?;
        Ok(TensorParallelGroupedOutput::new(
            output.subtract(&bias, stream)?,
            Some(bias),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{backend::ExecutionContext, module::PhysicalParam};
    use safemlx::{transforms::eval, Device, DeviceType};

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn mlx_selected_softmax_selector_applies_input_and_group_scales() {
        let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = execution.stream();
        let mut selector = TopKGroupSelector::new_with_quantization(
            TopKGroupSelectorConfig::new(
                2,
                3,
                2,
                TopKGroupScoring::SelectedSoftmax,
                false,
                0.0,
                1.0,
                1,
                1,
                false,
                false,
                Some(0.0),
                true,
                true,
            )
            .unwrap(),
            None,
            stream,
        )
        .unwrap();
        selector.weight = PhysicalParam::new(Array::from_slice(
            &[1.0_f32, 0.0, 0.0, 1.0, -1.0, 0.0],
            &[3, 2],
        ));
        selector.input_scale = PhysicalParam::new(Some(Array::from_slice(&[2.0_f32, 1.0], &[2])));
        selector.learned_coefficient_scale =
            PhysicalParam::new(Some(Array::from_slice(&[2.0_f32, 3.0, 5.0], &[3])));
        let output = selector
            .select_with_selection_bias(&Array::from_slice(&[3.0_f32, 4.0], &[1, 2]), None, stream)
            .unwrap();
        eval([&output.indices, &output.scores, &output.weights]).unwrap();
        let first = 0.4_f32.exp() / (0.4_f32.exp() + 1.0);
        let mut seen = [false; 2];
        for selection in 0..2 {
            let group = output
                .indices
                .try_index_device((0, selection), stream)
                .unwrap()
                .item::<i32>(stream);
            let score = output
                .scores
                .try_index_device((0, selection), stream)
                .unwrap()
                .item::<f32>(stream);
            let weight = output
                .weights
                .try_index_device((0, selection), stream)
                .unwrap()
                .item::<f32>(stream);
            let (expected_score, expected_weight) = match group {
                0 => (first, 2.0 * first),
                1 => (1.0 - first, 3.0 * (1.0 - first)),
                other => panic!("unexpected selected group {other}"),
            };
            seen[group as usize] = true;
            assert!((score - expected_score).abs() < 1e-5);
            assert!((weight - expected_weight).abs() < 1e-5);
        }
        assert_eq!(seen, [true, true]);
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn mlx_projection_bias_affects_selected_softmax_while_correction_is_selection_only() {
        let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = execution.stream();
        let mut selector = TopKGroupSelector::new_with_quantization(
            TopKGroupSelectorConfig::new(
                2,
                3,
                1,
                TopKGroupScoring::SelectedSoftmax,
                false,
                0.0,
                1.0,
                1,
                1,
                true,
                true,
                None,
                false,
                false,
            )
            .unwrap(),
            None,
            stream,
        )
        .unwrap();
        selector.weight = PhysicalParam::new(Array::from_slice(&[0.0_f32; 3], &[3, 1]));
        selector.bias = PhysicalParam::new(Some(Array::from_slice(
            &[0.0_f32, 2.0_f32.ln(), 4.0_f32.ln()],
            &[3],
        )));
        selector.e_score_correction_bias =
            PhysicalParam::new(Some(Array::from_slice(&[10.0_f32, 0.0, 0.0], &[3])));

        let output = selector
            .select_with_selection_bias(&Array::from_slice(&[1.0_f32], &[1, 1]), None, stream)
            .unwrap();
        eval([&output.indices, &output.scores, &output.weights]).unwrap();

        let mut seen = [false; 3];
        for selection in 0..2 {
            let group = output
                .indices
                .try_index_device((0, selection), stream)
                .unwrap()
                .item::<i32>(stream);
            let score = output
                .scores
                .try_index_device((0, selection), stream)
                .unwrap()
                .item::<f32>(stream);
            let weight = output
                .weights
                .try_index_device((0, selection), stream)
                .unwrap()
                .item::<f32>(stream);
            let expected = match group {
                0 => 0.2,
                2 => 0.8,
                other => panic!("unexpected selected group {other}"),
            };
            seen[group as usize] = true;
            assert!((score - expected).abs() < 1e-5);
            assert!((weight - expected).abs() < 1e-5);
        }
        assert_eq!(seen, [true, false, true]);
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn mlx_gated_product_policy_applies_projection_biases_at_exact_stages() {
        let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = execution.stream();
        let policy =
            GatedProductPolicy::new(GatedProductActivation::Silu, Some(2.0), Some(1.5), 1.7, 1.0)
                .unwrap();
        let mut bank = PackedGatedProductGroups::new(1, 1, 1, None, None, [true, true], stream)
            .unwrap()
            .with_policy(policy)
            .unwrap();
        bank.gate_up_proj = PhysicalParam::new(Array::from_slice(&[0.0_f32, 0.0], &[1, 2, 1]));
        bank.gate_up_proj_bias =
            PhysicalParam::new(Some(Array::from_slice(&[2.5_f32, -3.0], &[1, 2])));
        bank.down_proj = PhysicalParam::new(Array::from_slice(&[2.0_f32], &[1, 1, 1]));
        bank.down_proj_bias = PhysicalParam::new(Some(Array::from_slice(&[5.0_f32], &[1, 1])));

        let output = bank
            .forward(
                &Array::from_slice(&[7.0_f32], &[1, 1]),
                &Array::from_slice(&[0_i32], &[1, 1]),
                &Array::from_slice(&[0.25_f32], &[1, 1]),
                stream,
            )
            .unwrap();
        eval([&output]).unwrap();
        let gate = 2.0 / (1.0 + (-3.4_f32).exp());
        let expected = 0.25 * (2.0 * gate * -0.5 + 5.0);
        assert!((output.item::<f32>(stream) - expected).abs() < 1e-5);
    }

    #[test]
    #[ignore = "requires MLX runtime construction"]
    fn mlx_mxfp4_group_bank_rejects_indivisible_projection_width() {
        let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = execution.stream();
        let error = PackedGatedProductGroups::new(
            1,
            33,
            32,
            Some(WeightQuantization::MxFp4),
            Some(WeightQuantization::MxFp4),
            [false, false],
            stream,
        )
        .unwrap_err();
        assert!(error.to_string().contains("not divisible"));
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn mlx_tensor_parallel_down_bias_is_route_weighted_exactly_once() {
        let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = execution.stream();
        let rank = |down_weight: f32| {
            let mut bank =
                PackedGatedProductGroups::new(1, 1, 1, None, None, [false, true], stream).unwrap();
            bank.gate_up_proj = PhysicalParam::new(Array::from_slice(&[1.0_f32, 1.0], &[1, 2, 1]));
            bank.down_proj = PhysicalParam::new(Array::from_slice(&[down_weight], &[1, 1, 1]));
            bank.down_proj_bias = PhysicalParam::new(Some(Array::from_slice(&[5.0_f32], &[1, 1])));
            bank
        };
        let input = Array::from_slice(&[1.0_f32], &[1, 1]);
        let group = Array::from_slice(&[0_i32], &[1, 1]);
        let route_weight = Array::from_slice(&[0.25_f32], &[1, 1]);
        let mut rank_zero = rank(2.0);
        let mut rank_one = rank(3.0);
        let output_zero = rank_zero
            .forward_tensor_parallel(&input, &group, &route_weight, 2, stream)
            .unwrap();
        let output_one = rank_one
            .forward_tensor_parallel(&input, &group, &route_weight, 2, stream)
            .unwrap();
        let bias = output_zero.post_reduce().unwrap();
        eval([output_zero.reducible(), output_one.reducible(), bias]).unwrap();

        let gated = 1.0 / (1.0 + (-1.0_f32).exp());
        let expected = 0.25 * (5.0 * gated + 5.0);
        let reduced = output_zero.reducible().clone().item::<f32>(stream)
            + output_one.reducible().clone().item::<f32>(stream)
            + bias.clone().item::<f32>(stream);
        assert!((reduced - expected).abs() < 1e-5);
    }
}
