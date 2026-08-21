//! Mixture-of-experts routing and packed expert implementations.

use eredu_checkpoint::WeightQuantization;
use eredu_nn::{GatedProductActivation, GatedProductPolicy, TensorParallelExpertOutput};

use eredu_runtime::ActivationObserver as RuntimeActivationObserver;
use safemlx::{
    error::Exception,
    macros::ModuleParameters,
    module::Param,
    native_quantization::{native_grouped_linear, NativeQuantizedTensor},
    ops::{
        arange, argpartition_axis, concatenate_axis, gather_grouped_rows, gather_qmm_with_mode,
        gather_route_values, grouped_matmul,
        indexing::{scatter_single, take_along_axis, topk_axis, NewAxis, TryIndexOp},
        matmul, mean_axis, quantized_matmul_with_mode, quantized_packed_dimension, r#where, rsqrt,
        sigmoid, softmax_axis, sum_axis, topk_route_plan, zeros_dtype, GroupedRoutePlan,
        QuantizationMode,
    },
    Array, Dtype, Stream,
};

use crate::{
    backend::mlx::error::Error,
    backend::mlx::runtime::checkpoint::quantization::{quantize_tensor, QuantizedTensor},
};

use super::layers::{relu2, silu};

/// Applies one affine- or MXFP4-packed expert projection to expert-major rows.
///
/// The packed weight and its metadata keep the expert dimension leading, so
/// this is usable by any checkpoint layout once split experts have been
/// assembled into `[experts, output, input]` banks.
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

/// Applies a packed grouped projection with explicit route-order metadata.
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
    let routes = input.dim(0);
    let out_features = if transpose {
        weight.dim(-2)
    } else {
        weight.dim(-1) * 32 / quantization.bits()
    };
    if quantization.group_size() == 16 {
        if !transpose {
            return Err(Exception::custom(
                "group-16 affine expert projections require transposed packed weights",
            ));
        }
        let selected_weight = weight.take_axis(group_ids, 0, stream)?;
        let selected_scales = scales.take_axis(group_ids, 0, stream)?;
        let selected_biases = biases
            .map(|biases| biases.take_axis(group_ids, 0, stream))
            .transpose()?;
        return quantized_matmul_with_mode(
            input.reshape(&[routes, 1, input.dim(-1)], stream)?,
            &selected_weight,
            &selected_scales,
            selected_biases.as_ref(),
            true,
            quantization.group_size(),
            quantization.bits(),
            crate::backend::mlx::runtime::checkpoint::quantization::mlx_quantization_mode(
                quantization,
            ),
            stream,
        )?
        .reshape(&[routes, out_features], stream);
    }

    let lhs_indices = arange::<i32, u32>(0, routes, 1, stream)?;
    gather_qmm_with_mode(
        input.reshape(&[routes, 1, input.dim(-1)], stream)?,
        weight,
        scales,
        biases,
        Some(&lhs_indices),
        Some(group_ids),
        transpose,
        quantization.group_size(),
        quantization.bits(),
        sorted_indices,
        crate::backend::mlx::runtime::checkpoint::quantization::mlx_quantization_mode(quantization),
        stream,
    )?
    .reshape(&[routes, out_features], stream)
}

/// Quantizes a floating-point rank-3 packed expert bank while preserving its
/// leading expert dimension in the emitted weight, scale, and bias tensors.
pub fn quantize_expert_bank(
    value: &Array,
    quantization: WeightQuantization,
    stream: &Stream,
) -> Result<QuantizedTensor, Error> {
    if value.ndim() != 3 || !value.dtype().is_float() {
        return Err(Error::Quantization(format!(
            "expected a floating-point rank-3 expert bank, got shape {:?} and dtype {:?}",
            value.shape(),
            value.dtype()
        )));
    }
    let shape = value.shape();
    let experts = shape[0];
    let output_dims = shape[1];
    let input_dims = shape[2];
    let matrix = value.reshape(&[experts * output_dims, input_dims], stream)?;
    let quantized = quantize_tensor(&matrix, quantization, stream)?;
    Ok(QuantizedTensor {
        weight: quantized.weight.reshape(
            &[
                experts,
                output_dims,
                quantized_packed_dimension(input_dims, quantization.bits()),
            ],
            stream,
        )?,
        scales: quantized.scales.reshape(
            &[experts, output_dims, input_dims / quantization.group_size()],
            stream,
        )?,
        biases: quantized
            .biases
            .map(|biases| {
                biases.reshape(
                    &[experts, output_dims, input_dims / quantization.group_size()],
                    stream,
                )
            })
            .transpose()?,
    })
}

/// Router score transform used before top-k expert selection.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum TopKRouterScoreFunction {
    /// Softmax scores before top-k selection.
    Softmax,
    /// Select raw logits first, then softmax only the selected routes.
    SelectedSoftmax,
    /// Sigmoid router scores.
    Sigmoid,
    /// Square-root softplus router scores.
    SqrtSoftplus,
}

impl TopKRouterScoreFunction {
    fn requires_fp32(self) -> bool {
        matches!(self, Self::Sigmoid | Self::SqrtSoftplus)
    }

    fn apply(self, logits: Array, stream: &Stream) -> Result<Array, Exception> {
        match self {
            Self::Softmax => softmax_axis(logits, -1, true, stream),
            Self::SelectedSoftmax => Ok(logits),
            Self::Sigmoid => sigmoid(logits, stream),
            Self::SqrtSoftplus => safemlx::nn::softplus(logits, stream)?.sqrt(stream),
        }
    }
}

/// Configuration for a reusable top-k MoE router.
#[derive(Debug, Clone, Copy)]
pub struct TopKRouterConfig {
    /// Number of selected experts per token.
    pub top_k: i32,
    /// Total number of routed experts.
    pub num_experts: i32,
    /// Hidden dimension consumed by the router projection.
    pub hidden_size: i32,
    /// Score transform to apply to router logits.
    pub score_function: TopKRouterScoreFunction,
    /// Whether selected top-k weights are normalized after gathering.
    pub norm_topk_prob: bool,
    /// Optional epsilon added to the normalization denominator.
    pub normalization_epsilon: f32,
    /// Final multiplier applied to gathered routing weights.
    pub routed_scaling_factor: f32,
    /// Number of routing groups.
    pub n_group: i32,
    /// Number of routing groups selected before expert top-k.
    pub topk_group: i32,
    /// Whether to allocate an ordinary per-expert projection output bias.
    pub projection_bias: bool,
    /// Whether to allocate a selection-only expert score correction bias.
    pub score_correction_bias: bool,
    /// Optional epsilon for weightless RMS normalization before projection.
    pub input_rms_epsilon: Option<f32>,
    /// Whether normalized inputs receive an additional inverse-sqrt-width scale.
    pub input_inverse_sqrt_dimensions: bool,
    /// Whether to allocate learned per-expert route multipliers.
    pub route_scale: bool,
}

#[derive(Debug, Clone, ModuleParameters)]
/// Reusable top-k router for sparse MoE layers.
pub struct TopKRouter {
    /// Number of selected experts per token.
    pub top_k: i32,
    /// Total number of routed experts.
    pub num_experts: i32,
    /// Logical input width of the router projection.
    pub input_dims: i32,
    /// Router score transform.
    pub score_function: TopKRouterScoreFunction,
    /// Whether selected probabilities are normalized.
    pub norm_topk_prob: bool,
    /// Optional epsilon added to the normalization denominator.
    pub normalization_epsilon: f32,
    /// Final multiplier applied to routing weights.
    pub routed_scaling_factor: f32,
    /// Number of routing groups.
    pub n_group: i32,
    /// Number of selected routing groups.
    pub topk_group: i32,
    #[param]
    /// Router projection weight.
    pub weight: Param<Array>,
    #[param]
    /// Optional ordinary projection output bias applied to router logits.
    pub bias: Param<Option<Array>>,
    #[param]
    /// Optional affine scales for a packed router projection.
    pub scales: Param<Option<Array>>,
    #[param]
    /// Optional affine biases for a packed router projection.
    pub biases: Param<Option<Array>>,
    #[param]
    /// Optional score correction bias used only when choosing experts.
    pub e_score_correction_bias: Param<Option<Array>>,
    #[param]
    /// Optional learned feature scale applied to RMS-normalized inputs.
    pub input_scale: Param<Option<Array>>,
    #[param]
    /// Optional learned multiplier gathered for each selected expert.
    pub route_scale: Param<Option<Array>>,
    /// Optional router-input RMS epsilon.
    pub input_rms_epsilon: Option<f32>,
    /// Whether normalized router inputs are divided by the square root of width.
    pub input_inverse_sqrt_dimensions: bool,
    /// Affine group size, or zero for a dense router.
    pub group_size: i32,
    /// Affine bit width, or zero for a dense router.
    pub bits: i32,
    /// Packed quantization encoding.
    pub mode: QuantizationMode,
    /// Checkpoint-native GGML encoding and byte order.
    pub iquant: Option<WeightQuantization>,
}

/// Selected expert ids plus the score and weight arrays produced by a top-k router.
pub struct TopKRouterOutput {
    /// Selected expert ids with shape `[tokens, top_k]`.
    pub indices: Array,
    /// Router probabilities or scores gathered at the selected ids.
    pub scores: Array,
    /// Final routing weights after optional normalization/scaling.
    pub weights: Array,
}

/// Selects the largest router logits and normalizes only the selected values.
/// The softmax is applied after top-k selection rather than across every
/// candidate expert.
pub fn top_k_softmax_routing(
    logits: &Array,
    top_k: i32,
    stream: &Stream,
) -> Result<(Array, Array), Exception> {
    let indices =
        argpartition_axis(logits, -top_k, -1, stream)?.try_index_device((.., -top_k..), stream)?;
    let selected = take_along_axis(logits, &indices, -1, stream)?;
    Ok((indices, softmax_axis(&selected, -1, true, stream)?))
}

impl TopKRouter {
    /// Creates an unloaded router.
    pub fn new(config: TopKRouterConfig, stream: &Stream) -> Result<Self, Exception> {
        Self::new_with_quantization(config, None, stream)
    }

    /// Creates an unloaded dense or affine-packed router.
    pub fn new_with_quantization(
        config: TopKRouterConfig,
        quantization: Option<WeightQuantization>,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        Self::new_with_quantization_and_dtype(config, quantization, Dtype::Float32, stream)
    }

    /// Creates an unloaded dense or affine-packed router with an explicit dense dtype.
    pub fn new_with_quantization_and_dtype(
        config: TopKRouterConfig,
        quantization: Option<WeightQuantization>,
        dense_dtype: Dtype,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        if let Some(quantization) = quantization {
            if config.hidden_size <= 0 || config.hidden_size % quantization.group_size() != 0 {
                return Err(Exception::custom(format!(
                    "affine router hidden dimension {} is not divisible by group size {}",
                    config.hidden_size,
                    quantization.group_size()
                )));
            }
        }
        let affine = quantization.filter(|q| !matches!(q, WeightQuantization::GgufIQuant { .. }));
        Ok(Self {
            top_k: config.top_k,
            num_experts: config.num_experts,
            input_dims: config.hidden_size,
            score_function: config.score_function,
            norm_topk_prob: config.norm_topk_prob,
            normalization_epsilon: config.normalization_epsilon,
            routed_scaling_factor: config.routed_scaling_factor,
            n_group: config.n_group,
            topk_group: config.topk_group,
            weight: match quantization {
                Some(WeightQuantization::GgufIQuant { ggml_type, .. }) => {
                    let (block_values, block_bytes) =
                        ggml_type.block_and_bytes().map_err(|_| {
                            Exception::custom(format!(
                                "{ggml_type:?} has no native router block geometry"
                            ))
                        })?;
                    Param::<Array>::unloaded(
                        &[
                            config.num_experts,
                            config.hidden_size / block_values as i32 * block_bytes as i32,
                        ],
                        Dtype::Uint8,
                        stream,
                    )?
                }
                Some(quantization) => Param::<Array>::unloaded(
                    &[
                        config.num_experts,
                        quantized_packed_dimension(config.hidden_size, quantization.bits()),
                    ],
                    Dtype::Uint32,
                    stream,
                )?,
                None => Param::<Array>::unloaded(
                    &[config.num_experts, config.hidden_size],
                    dense_dtype,
                    stream,
                )?,
            },
            bias: if config.projection_bias {
                Param::<Option<Array>>::unloaded_some(&[config.num_experts], dense_dtype, stream)?
            } else {
                Param::new(None)
            },
            scales: if let Some(quantization) = affine {
                Param::<Option<Array>>::unloaded_some(
                    &[
                        config.num_experts,
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
                Param::new(None)
            },
            biases: if let Some(quantization) = affine.filter(|q| q.has_biases()) {
                Param::<Option<Array>>::unloaded_some(
                    &[
                        config.num_experts,
                        config.hidden_size / quantization.group_size(),
                    ],
                    Dtype::Float16,
                    stream,
                )?
            } else {
                Param::new(None)
            },
            e_score_correction_bias: if config.score_correction_bias {
                Param::<Option<Array>>::unloaded_some(&[config.num_experts], dense_dtype, stream)?
            } else {
                Param::new(None)
            },
            input_scale: if config.input_rms_epsilon.is_some() {
                Param::<Option<Array>>::unloaded_some(&[config.hidden_size], dense_dtype, stream)?
            } else {
                Param::new(None)
            },
            route_scale: if config.route_scale {
                Param::<Option<Array>>::unloaded_some(&[config.num_experts], dense_dtype, stream)?
            } else {
                Param::new(None)
            },
            input_rms_epsilon: config.input_rms_epsilon,
            input_inverse_sqrt_dimensions: config.input_inverse_sqrt_dimensions,
            group_size: affine.map_or(0, WeightQuantization::group_size),
            bits: affine.map_or(0, WeightQuantization::bits),
            mode: affine.map_or(
                QuantizationMode::Affine,
                crate::backend::mlx::runtime::checkpoint::quantization::mlx_quantization_mode,
            ),
            iquant: quantization.filter(|q| matches!(q, WeightQuantization::GgufIQuant { .. })),
        })
    }

    /// Returns selected expert ids and per-route weights.
    pub fn forward(
        &mut self,
        hidden_states: &Array,
        stream: &Stream,
    ) -> Result<(Array, Array), Exception> {
        self.forward_with_selection_bias(hidden_states, None, stream)
    }

    /// Returns selected expert ids and weights using an optional bias only for selection.
    ///
    /// The gathered route weights always come from the unbiased transformed scores.
    pub fn forward_with_selection_bias(
        &mut self,
        hidden_states: &Array,
        selection_bias: Option<&Array>,
        stream: &Stream,
    ) -> Result<(Array, Array), Exception> {
        let output =
            self.forward_routes_with_selection_bias(hidden_states, selection_bias, stream)?;
        Ok((output.indices, output.weights))
    }

    /// Returns selected ids, pre-normalization selected scores, and final route weights.
    pub fn forward_routes_with_selection_bias(
        &mut self,
        hidden_states: &Array,
        selection_bias: Option<&Array>,
        stream: &Stream,
    ) -> Result<TopKRouterOutput, Exception> {
        let flat = self.transform_input(hidden_states, stream)?;
        let logits = if let Some(iquant) = self.iquant {
            let (ggml_type, endian) = iquant.gguf_iquant().expect("IQ router format");
            NativeQuantizedTensor::from_iq_array(
                self.weight.value.clone(),
                &[self.num_experts, self.input_dims],
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
        if self.score_function == TopKRouterScoreFunction::SelectedSoftmax {
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
        if self.routed_scaling_factor != 1.0 {
            top_k_weights =
                top_k_weights.multiply(Array::from_f32(self.routed_scaling_factor), stream)?;
        }
        if let Some(scale) = self.route_scale.as_ref() {
            top_k_weights =
                top_k_weights.multiply(scale.take_axis(&top_k_index, 0, stream)?, stream)?;
        }
        Ok(TopKRouterOutput {
            indices: top_k_index,
            scores: selected_scores,
            weights: top_k_weights,
        })
    }

    /// Computes routing weights for caller-provided global expert ids.
    ///
    /// In hash-router form, the checkpoint's
    /// token-id table chooses experts, while the ordinary router projection
    /// still supplies their normalized contribution weights.
    pub fn forward_with_routing_indices(
        &mut self,
        hidden_states: &Array,
        expert_indices: &Array,
        stream: &Stream,
    ) -> Result<(Array, Array), Exception> {
        let output =
            self.forward_routes_with_routing_indices(hidden_states, expert_indices, stream)?;
        Ok((output.indices, output.weights))
    }

    /// Returns caller-selected ids, their raw transformed scores, and final
    /// normalized/scaled route weights.
    pub fn forward_routes_with_routing_indices(
        &mut self,
        hidden_states: &Array,
        expert_indices: &Array,
        stream: &Stream,
    ) -> Result<TopKRouterOutput, Exception> {
        let flat = self.transform_input(hidden_states, stream)?;
        let logits = if let Some(iquant) = self.iquant {
            let (ggml_type, endian) = iquant.gguf_iquant().expect("IQ router format");
            NativeQuantizedTensor::from_iq_array(
                self.weight.value.clone(),
                &[self.num_experts, self.input_dims],
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
        let expert_indices = expert_indices.reshape(&[-1, self.top_k], stream)?;
        let mut weights = take_along_axis(scores, &expert_indices, -1, stream)?;
        if self.score_function == TopKRouterScoreFunction::SelectedSoftmax {
            weights = softmax_axis(&weights, -1, true, stream)?;
        }
        let selected_scores = weights.clone();
        if self.norm_topk_prob {
            let denominator = weights
                .sum_axis(-1, true, stream)?
                .add(Array::from_f32(self.normalization_epsilon), stream)?;
            weights = weights.divide(denominator, stream)?;
        }
        if self.routed_scaling_factor != 1.0 {
            weights = weights.multiply(Array::from_f32(self.routed_scaling_factor), stream)?;
        }
        if let Some(scale) = self.route_scale.as_ref() {
            weights = weights.multiply(scale.take_axis(&expert_indices, 0, stream)?, stream)?;
        }
        Ok(TopKRouterOutput {
            indices: expert_indices,
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
            .expect("router input scale requires an RMS epsilon");
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

    /// Returns selected expert ids and weights while reporting router internals.
    pub fn forward_with_observer(
        &mut self,
        hidden_states: &Array,
        stream: &Stream,
        prefix: &str,
        observer: &mut dyn RuntimeActivationObserver<Array, Exception>,
    ) -> Result<TopKRouterOutput, Exception> {
        let flat = self.transform_input(hidden_states, stream)?;
        let logits = if let Some(scales) = self.scales.as_ref() {
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
        observer.observe(&format!("{prefix}.router_logits"), &logits)?;
        let scores = self.score_function.apply(logits, stream)?;
        observer.observe(&format!("{prefix}.router_scores"), &scores)?;

        let mut scores_for_choice = scores.clone();
        if let Some(bias) = self.e_score_correction_bias.as_ref() {
            scores_for_choice = scores_for_choice.add(bias, stream)?;
            observer.observe(
                &format!("{prefix}.router_scores_for_choice"),
                &scores_for_choice,
            )?;
        }

        let top_k_index = self.topk_indices(&scores_for_choice, stream)?;
        observer.observe(&format!("{prefix}.top_k_experts"), &top_k_index)?;
        let mut top_k_weights = take_along_axis(&scores, &top_k_index, -1, stream)?;
        if self.score_function == TopKRouterScoreFunction::SelectedSoftmax {
            top_k_weights = softmax_axis(&top_k_weights, -1, true, stream)?;
        }
        let top_k_scores = top_k_weights.clone();
        observer.observe(&format!("{prefix}.top_k_scores"), &top_k_weights)?;
        if self.norm_topk_prob {
            let mut denominator = sum_axis(&top_k_weights, -1, true, stream)?;
            if self.normalization_epsilon != 0.0 {
                denominator =
                    denominator.add(Array::from_f32(self.normalization_epsilon), stream)?;
            }
            top_k_weights = top_k_weights.divide(denominator, stream)?;
            observer.observe(
                &format!("{prefix}.top_k_weights_normalized"),
                &top_k_weights,
            )?;
        }
        if self.routed_scaling_factor != 1.0 {
            top_k_weights =
                top_k_weights.multiply(Array::from_f32(self.routed_scaling_factor), stream)?;
            observer.observe(&format!("{prefix}.top_k_weights_scaled"), &top_k_weights)?;
        }
        if let Some(scale) = self.route_scale.as_ref() {
            top_k_weights =
                top_k_weights.multiply(scale.take_axis(&top_k_index, 0, stream)?, stream)?;
            observer.observe(
                &format!("{prefix}.top_k_weights_expert_scaled"),
                &top_k_weights,
            )?;
        }
        Ok(TopKRouterOutput {
            indices: top_k_index,
            scores: top_k_scores,
            weights: top_k_weights,
        })
    }

    fn topk_indices(&self, scores_for_choice: &Array, stream: &Stream) -> Result<Array, Exception> {
        if self.n_group == 1 && self.topk_group == 1 {
            return argpartition_axis(scores_for_choice, -self.top_k, -1, stream)?
                .try_index_device((.., -self.top_k..), stream);
        }
        if self.n_group <= 0
            || self.topk_group <= 0
            || self.topk_group > self.n_group
            || self.num_experts % self.n_group != 0
        {
            return Err(Exception::custom(
                "invalid grouped MoE router configuration",
            ));
        }

        let tokens = scores_for_choice.dim(0);
        let experts_per_group = self.num_experts / self.n_group;
        let grouped =
            scores_for_choice.reshape(&[tokens, self.n_group, experts_per_group], stream)?;
        let group_top = 2.min(experts_per_group);
        let group_scores = sum_axis(
            &topk_axis(grouped, group_top, -1, stream)?,
            -1,
            false,
            stream,
        )?;
        let group_idx = argpartition_axis(&group_scores, -self.topk_group, -1, stream)?
            .try_index_device((.., -self.topk_group..), stream)?;

        let expert_group_ids: Vec<i32> = (0..self.num_experts)
            .map(|expert| expert / experts_per_group)
            .collect();
        let expert_group_ids = Array::from_slice(&expert_group_ids, &[1, 1, self.num_experts]);
        let selected_groups = group_idx.try_index_device((.., .., NewAxis), stream)?;
        let group_mask = selected_groups.eq(expert_group_ids, stream)?;
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

    /// Sets training mode.
    pub fn training_mode(&mut self, _mode: bool) {}
}

/// Applies route weights and reduces expert-major route outputs back to source tokens.
pub fn weighted_route_sum(
    current: Array,
    top_k_weights: &Array,
    plan: &GroupedRoutePlan,
    num_tokens: i32,
    stream: &Stream,
) -> Result<Array, Exception> {
    let weights = gather_route_values(top_k_weights, plan, stream)?
        .try_index_device((.., NewAxis), stream)?;
    let weighted = current.multiply(weights, stream)?;

    // Each route index is unique, so restore the expert-major rows with a
    // collision-free scatter and reduce the original top-k slots in their
    // stable order. A segment sum can use unordered GPU atomics here; the
    // resulting roundoff was sufficient to change near-tied downstream routing
    // decisions between identical passes.
    let routes = weighted.dim(0);
    let width = weighted.dim(-1);
    let ordered = scatter_single(
        zeros_dtype(&[routes, width], weighted.dtype(), stream)?,
        &plan.route_indices,
        weighted.reshape(&[routes, 1, width], stream)?,
        0,
        stream,
    )?;
    let top_k = top_k_weights.dim(-1);
    let ordered = ordered.reshape(&[num_tokens, top_k, width], stream)?;
    sum_axis(ordered, 1, false, stream)
}

#[derive(Debug, Clone, ModuleParameters)]
/// Packed routed ReLU2 expert bank with dense, affine, MXFP4, or GGUF-native IQ storage.
pub struct PackedRelu2Experts {
    /// Number of routed experts.
    pub num_experts: i32,
    /// Model hidden dimension.
    pub hidden_size: i32,
    /// Per-expert intermediate dimension.
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
    /// Expert up-projection weights.
    pub up_proj: Param<Array>,
    #[param]
    /// Expert up-projection packed scales.
    pub up_proj_scales: Param<Option<Array>>,
    #[param]
    /// Expert up-projection affine biases, absent for MXFP4.
    pub up_proj_biases: Param<Option<Array>>,
    #[param]
    /// Expert down-projection weights.
    pub down_proj: Param<Array>,
    #[param]
    /// Expert down-projection packed scales.
    pub down_proj_scales: Param<Option<Array>>,
    #[param]
    /// Expert down-projection affine biases, absent for MXFP4.
    pub down_proj_biases: Param<Option<Array>>,
}

impl PackedRelu2Experts {
    /// Creates an unloaded dense, packed, or checkpoint-native IQ expert bank.
    pub fn new(
        num_experts: i32,
        hidden_size: i32,
        intermediate_size: i32,
        quantization: [Option<WeightQuantization>; 2],
        stream: &Stream,
    ) -> Result<Self, Exception> {
        Self::new_with_dtype(
            num_experts,
            hidden_size,
            intermediate_size,
            quantization,
            Dtype::Float32,
            stream,
        )
    }

    /// Creates an unloaded expert bank with an explicit dense weight dtype.
    pub fn new_with_dtype(
        num_experts: i32,
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
         -> Result<ExpertProjectionParams, Exception> {
            if let Some(iquant) = iquant {
                let (ggml_type, _) = iquant.gguf_iquant().expect("IQ expert format");
                let (block_values, block_bytes) = ggml_type
                    .block_and_bytes()
                    .expect("canonical IQ block geometry");
                return Ok((
                    Param::<Array>::unloaded(
                        &[
                            num_experts,
                            out_features,
                            in_features / block_values as i32 * block_bytes as i32,
                        ],
                        Dtype::Uint8,
                        stream,
                    )?,
                    Param::new(None),
                    Param::new(None),
                ));
            }
            match quantization {
                Some(quantization) => Ok((
                    Param::<Array>::unloaded(
                        &[
                            num_experts,
                            out_features,
                            quantized_packed_dimension(in_features, quantization.bits()),
                        ],
                        Dtype::Uint32,
                        stream,
                    )?,
                    Param::<Option<Array>>::unloaded_some(
                        &[
                            num_experts,
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
                        Param::<Option<Array>>::unloaded_some(
                            &[
                                num_experts,
                                out_features,
                                in_features / quantization.group_size(),
                            ],
                            Dtype::Float16,
                            stream,
                        )?
                    } else {
                        Param::new(None)
                    },
                )),
                None => Ok((
                    Param::<Array>::unloaded(
                        &[num_experts, out_features, in_features],
                        dense_dtype,
                        stream,
                    )?,
                    Param::new(None),
                    Param::new(None),
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
            num_experts,
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

    /// Evaluates routed experts and reduces route outputs back to tokens.
    pub fn forward(
        &mut self,
        hidden_states: &Array,
        top_k_index: &Array,
        top_k_weights: &Array,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let num_tokens = hidden_states.dim(0);
        let plan = topk_route_plan(top_k_index, self.num_experts, stream)?;
        let hidden = gather_grouped_rows(hidden_states, &plan, stream)?;
        let hidden = if let Some(iquant) = self.up_iquant {
            let (ggml_type, endian) = iquant.gguf_iquant().expect("IQ expert format");
            let native = NativeQuantizedTensor::from_iq_array(
                self.up_proj.value.clone(),
                &[self.num_experts, self.intermediate_size, self.hidden_size],
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
                        .expect("quantized expert scales"),
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
            let (ggml_type, endian) = iquant.gguf_iquant().expect("IQ expert format");
            let native = NativeQuantizedTensor::from_iq_array(
                self.down_proj.value.clone(),
                &[self.num_experts, self.hidden_size, self.intermediate_size],
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
                        .expect("quantized expert scales"),
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
        weighted_route_sum(current, top_k_weights, &plan, num_tokens, stream)
    }

    /// Returns the rank-local ReLU2 contribution for one tensor-parallel sum.
    pub fn forward_tensor_parallel(
        &mut self,
        hidden_states: &Array,
        top_k_index: &Array,
        top_k_weights: &Array,
        partitions: usize,
        stream: &Stream,
    ) -> Result<TensorParallelExpertOutput<Array>, Exception> {
        if partitions == 0 {
            return Err(Exception::custom(
                "tensor-parallel partition count must be positive",
            ));
        }
        self.forward(hidden_states, top_k_index, top_k_weights, stream)
            .map(|reducible| TensorParallelExpertOutput {
                reducible,
                post_reduce: None,
            })
    }

    /// Sets training mode.
    pub fn training_mode(&mut self, _mode: bool) {}
}

const ROUTED_EXPERT_CHUNK_THRESHOLD: i32 = 64;
const ROUTED_EXPERT_CHUNK_TOKENS: i32 = 32;

#[derive(Debug, Clone, ModuleParameters)]
/// Packed gated-product expert bank with optional MLX affine or MXFP4 projections.
pub struct PackedGatedProductExperts {
    /// Number of experts.
    pub num_experts: i32,
    /// Model hidden dimension.
    pub hidden_dim: i32,
    /// Per-expert intermediate dimension.
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
    /// Concatenated gate/up weights shaped `[experts, 2 * intermediate, hidden]`.
    pub gate_up_proj: Param<Array>,
    #[param]
    /// Optional ordinary gate/up output bias shaped `[experts, 2 * intermediate]`.
    pub gate_up_proj_bias: Param<Option<Array>>,
    #[param]
    /// Gate/up quantization scales.
    pub gate_up_proj_scales: Param<Option<Array>>,
    #[param]
    /// Gate/up quantization biases.
    pub gate_up_proj_biases: Param<Option<Array>>,
    #[param]
    /// Down weights shaped `[experts, hidden, intermediate]`.
    pub down_proj: Param<Array>,
    #[param]
    /// Optional ordinary down output bias shaped `[experts, hidden]`.
    pub down_proj_bias: Param<Option<Array>>,
    #[param]
    /// Down quantization scales.
    pub down_proj_scales: Param<Option<Array>>,
    #[param]
    /// Down quantization biases.
    pub down_proj_biases: Param<Option<Array>>,
}

type ExpertProjectionParams = (Param<Array>, Param<Option<Array>>, Param<Option<Array>>);

impl PackedGatedProductExperts {
    /// Creates an unloaded packed expert bank.
    pub fn new(
        num_experts: i32,
        hidden_dim: i32,
        intermediate_dim: i32,
        gate_up_affine: Option<WeightQuantization>,
        down_affine: Option<WeightQuantization>,
        projection_biases: [bool; 2],
        stream: &Stream,
    ) -> Result<Self, Exception> {
        Self::new_with_dtype(
            num_experts,
            hidden_dim,
            intermediate_dim,
            gate_up_affine,
            down_affine,
            projection_biases,
            Dtype::Float32,
            stream,
        )
    }

    /// Creates an unloaded packed expert bank with an explicit dense weight dtype.
    pub fn new_with_dtype(
        num_experts: i32,
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
         -> Result<ExpertProjectionParams, Exception> {
            if let Some(iquant) = iquant {
                let (ggml_type, _) = iquant.gguf_iquant().expect("IQ expert format");
                let (block_values, block_bytes) = ggml_type
                    .block_and_bytes()
                    .expect("canonical IQ block geometry");
                Ok((
                    Param::<Array>::unloaded(
                        &[
                            num_experts,
                            out_features,
                            in_features / block_values as i32 * block_bytes as i32,
                        ],
                        Dtype::Uint8,
                        stream,
                    )?,
                    Param::new(None),
                    Param::new(None),
                ))
            } else if let Some(quantization) = quantization {
                if in_features % quantization.group_size() != 0 {
                    return Err(Exception::custom(format!(
                        "packed expert input width {in_features} is not divisible by {quantization:?} group size {}",
                        quantization.group_size(),
                    )));
                }
                Ok((
                    Param::<Array>::unloaded(
                        &[
                            num_experts,
                            out_features,
                            quantized_packed_dimension(in_features, quantization.bits()),
                        ],
                        Dtype::Uint32,
                        stream,
                    )?,
                    Param::<Option<Array>>::unloaded_some(
                        &[
                            num_experts,
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
                        Param::<Option<Array>>::unloaded_some(
                            &[
                                num_experts,
                                out_features,
                                in_features / quantization.group_size(),
                            ],
                            Dtype::Float16,
                            stream,
                        )?
                    } else {
                        Param::new(None)
                    },
                ))
            } else {
                Ok((
                    Param::<Array>::unloaded(
                        &[num_experts, out_features, in_features],
                        dense_dtype,
                        stream,
                    )?,
                    Param::new(None),
                    Param::new(None),
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
            num_experts,
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
                Param::<Option<Array>>::unloaded_some(
                    &[num_experts, 2 * intermediate_dim],
                    dense_dtype,
                    stream,
                )?
            } else {
                Param::new(None)
            },
            gate_up_proj_scales,
            gate_up_proj_biases,
            down_proj,
            down_proj_bias: if projection_biases[1] {
                Param::<Option<Array>>::unloaded_some(
                    &[num_experts, hidden_dim],
                    dense_dtype,
                    stream,
                )?
            } else {
                Param::new(None)
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

    /// Rebuilds projection storage for native block-FP8 expert tensors.
    pub fn with_native_fp8_e8m0(mut self, stream: &Stream) -> Result<Self, Exception> {
        let ceil128 = |value: i32| (value + 127) / 128;
        self.gate_up_proj = Param::<Array>::unloaded(
            &[self.num_experts, 2 * self.intermediate_dim, self.hidden_dim],
            Dtype::Uint8,
            stream,
        )?;
        self.gate_up_proj_scales = Param::<Option<Array>>::unloaded_some(
            &[
                self.num_experts,
                ceil128(2 * self.intermediate_dim),
                ceil128(self.hidden_dim),
            ],
            Dtype::Uint8,
            stream,
        )?;
        self.down_proj = Param::<Array>::unloaded(
            &[self.num_experts, self.hidden_dim, self.intermediate_dim],
            Dtype::Uint8,
            stream,
        )?;
        self.down_proj_scales = Param::<Option<Array>>::unloaded_some(
            &[
                self.num_experts,
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
        let plan = topk_route_plan(top_k_index, self.num_experts, stream)?;
        let hidden = gather_grouped_rows(hidden_states, &plan, stream)?;
        let gate_up = if self.native_fp8_e8m0 {
            crate::backend::mlx::nn::fp8::grouped_linear(
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
            let (ggml_type, endian) = iquant.gguf_iquant().expect("IQ expert format");
            let native = NativeQuantizedTensor::from_iq_array(
                self.gate_up_proj.value.clone(),
                &[self.num_experts, 2 * self.intermediate_dim, self.hidden_dim],
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
            GatedProductActivation::GeluApproximate => safemlx::nn::gelu_approximate(gate, stream)?,
        };
        let activated = gate.multiply(up, stream)?;
        let output = if self.native_fp8_e8m0 {
            crate::backend::mlx::nn::fp8::grouped_linear(
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
            let (ggml_type, endian) = iquant.gguf_iquant().expect("IQ expert format");
            let native = NativeQuantizedTensor::from_iq_array(
                self.down_proj.value.clone(),
                &[self.num_experts, self.hidden_dim, self.intermediate_dim],
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
        weighted_route_sum(output, top_k_weights, &plan, num_tokens, stream)
    }

    /// Evaluates selected experts and reduces route outputs back to source tokens.
    pub fn forward(
        &mut self,
        hidden_states: &Array,
        top_k_index: &Array,
        top_k_weights: &Array,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let num_tokens = hidden_states.dim(0);
        if num_tokens <= ROUTED_EXPERT_CHUNK_THRESHOLD {
            return self.forward_chunk(hidden_states, top_k_index, top_k_weights, stream);
        }
        let mut outputs = Vec::new();
        let mut start = 0;
        while start < num_tokens {
            let end = (start + ROUTED_EXPERT_CHUNK_TOKENS).min(num_tokens);
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

    /// Separates the rank-local projection contribution from replicated routed
    /// down bias so the latter can be added literally once after all-sum.
    pub fn forward_tensor_parallel(
        &mut self,
        hidden_states: &Array,
        top_k_index: &Array,
        top_k_weights: &Array,
        partitions: usize,
        stream: &Stream,
    ) -> Result<TensorParallelExpertOutput<Array>, Exception> {
        if partitions == 0 {
            return Err(Exception::custom(
                "tensor-parallel partition count must be positive",
            ));
        }
        let output = self.forward(hidden_states, top_k_index, top_k_weights, stream)?;
        let Some(bias) = self.down_proj_bias.as_ref() else {
            return Ok(TensorParallelExpertOutput {
                reducible: output,
                post_reduce: None,
            });
        };
        let plan = topk_route_plan(top_k_index, self.num_experts, stream)?;
        let routed_bias = bias.take_axis(&plan.sorted_group_ids, 0, stream)?;
        let bias = weighted_route_sum(
            routed_bias,
            top_k_weights,
            &plan,
            hidden_states.dim(0),
            stream,
        )?;
        Ok(TensorParallelExpertOutput {
            reducible: output.subtract(&bias, stream)?,
            post_reduce: Some(bias),
        })
    }

    /// Sets training mode.
    pub fn training_mode(&mut self, _mode: bool) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use safemlx::{module::Param, transforms::eval, Device, DeviceType, ExecutionContext};

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn mlx_selected_softmax_router_applies_input_and_expert_scales() {
        let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = execution.stream();
        let mut router = TopKRouter::new(
            TopKRouterConfig {
                top_k: 2,
                num_experts: 3,
                hidden_size: 2,
                score_function: TopKRouterScoreFunction::SelectedSoftmax,
                norm_topk_prob: false,
                normalization_epsilon: 0.0,
                routed_scaling_factor: 1.0,
                n_group: 1,
                topk_group: 1,
                projection_bias: false,
                score_correction_bias: false,
                input_rms_epsilon: Some(0.0),
                input_inverse_sqrt_dimensions: true,
                route_scale: true,
            },
            stream,
        )
        .unwrap();
        router.weight = Param::new(Array::from_slice(
            &[1.0_f32, 0.0, 0.0, 1.0, -1.0, 0.0],
            &[3, 2],
        ));
        router.input_scale = Param::new(Some(Array::from_slice(&[2.0_f32, 1.0], &[2])));
        router.route_scale = Param::new(Some(Array::from_slice(&[2.0_f32, 3.0, 5.0], &[3])));
        let output = router
            .forward_routes_with_selection_bias(
                &Array::from_slice(&[3.0_f32, 4.0], &[1, 2]),
                None,
                stream,
            )
            .unwrap();
        eval([&output.indices, &output.scores, &output.weights]).unwrap();
        let first = 0.4_f32.exp() / (0.4_f32.exp() + 1.0);
        let mut seen = [false; 2];
        for route in 0..2 {
            let expert = output
                .indices
                .try_index_device((0, route), stream)
                .unwrap()
                .item::<i32>(stream);
            let score = output
                .scores
                .try_index_device((0, route), stream)
                .unwrap()
                .item::<f32>(stream);
            let weight = output
                .weights
                .try_index_device((0, route), stream)
                .unwrap()
                .item::<f32>(stream);
            let (expected_score, expected_weight) = match expert {
                0 => (first, 2.0 * first),
                1 => (1.0 - first, 3.0 * (1.0 - first)),
                other => panic!("unexpected selected expert {other}"),
            };
            seen[expert as usize] = true;
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
        let mut router = TopKRouter::new(
            TopKRouterConfig {
                top_k: 2,
                num_experts: 3,
                hidden_size: 1,
                score_function: TopKRouterScoreFunction::SelectedSoftmax,
                norm_topk_prob: false,
                normalization_epsilon: 0.0,
                routed_scaling_factor: 1.0,
                n_group: 1,
                topk_group: 1,
                projection_bias: true,
                score_correction_bias: true,
                input_rms_epsilon: None,
                input_inverse_sqrt_dimensions: false,
                route_scale: false,
            },
            stream,
        )
        .unwrap();
        router.weight = Param::new(Array::from_slice(&[0.0_f32; 3], &[3, 1]));
        router.bias = Param::new(Some(Array::from_slice(
            &[0.0_f32, 2.0_f32.ln(), 4.0_f32.ln()],
            &[3],
        )));
        router.e_score_correction_bias =
            Param::new(Some(Array::from_slice(&[10.0_f32, 0.0, 0.0], &[3])));

        let output = router
            .forward_routes_with_selection_bias(
                &Array::from_slice(&[1.0_f32], &[1, 1]),
                None,
                stream,
            )
            .unwrap();
        eval([&output.indices, &output.scores, &output.weights]).unwrap();

        let mut seen = [false; 3];
        for route in 0..2 {
            let expert = output
                .indices
                .try_index_device((0, route), stream)
                .unwrap()
                .item::<i32>(stream);
            let score = output
                .scores
                .try_index_device((0, route), stream)
                .unwrap()
                .item::<f32>(stream);
            let weight = output
                .weights
                .try_index_device((0, route), stream)
                .unwrap()
                .item::<f32>(stream);
            let expected = match expert {
                0 => 0.2,
                2 => 0.8,
                other => panic!("unexpected selected expert {other}"),
            };
            seen[expert as usize] = true;
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
        let mut bank = PackedGatedProductExperts::new(1, 1, 1, None, None, [true, true], stream)
            .unwrap()
            .with_policy(policy)
            .unwrap();
        bank.gate_up_proj = Param::new(Array::from_slice(&[0.0_f32, 0.0], &[1, 2, 1]));
        bank.gate_up_proj_bias = Param::new(Some(Array::from_slice(&[2.5_f32, -3.0], &[1, 2])));
        bank.down_proj = Param::new(Array::from_slice(&[2.0_f32], &[1, 1, 1]));
        bank.down_proj_bias = Param::new(Some(Array::from_slice(&[5.0_f32], &[1, 1])));

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
    fn mlx_mxfp4_expert_bank_rejects_indivisible_projection_width() {
        let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = execution.stream();
        let error = PackedGatedProductExperts::new(
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
                PackedGatedProductExperts::new(1, 1, 1, None, None, [false, true], stream).unwrap();
            bank.gate_up_proj = Param::new(Array::from_slice(&[1.0_f32, 1.0], &[1, 2, 1]));
            bank.down_proj = Param::new(Array::from_slice(&[down_weight], &[1, 1, 1]));
            bank.down_proj_bias = Param::new(Some(Array::from_slice(&[5.0_f32], &[1, 1])));
            bank
        };
        let input = Array::from_slice(&[1.0_f32], &[1, 1]);
        let expert = Array::from_slice(&[0_i32], &[1, 1]);
        let route_weight = Array::from_slice(&[0.25_f32], &[1, 1]);
        let mut rank_zero = rank(2.0);
        let mut rank_one = rank(3.0);
        let output_zero = rank_zero
            .forward_tensor_parallel(&input, &expert, &route_weight, 2, stream)
            .unwrap();
        let output_one = rank_one
            .forward_tensor_parallel(&input, &expert, &route_weight, 2, stream)
            .unwrap();
        let bias = output_zero.post_reduce.as_ref().unwrap();
        eval([&output_zero.reducible, &output_one.reducible, bias]).unwrap();

        let gated = 1.0 / (1.0 + (-1.0_f32).exp());
        let expected = 0.25 * (5.0 * gated + 5.0);
        let reduced = output_zero.reducible.item::<f32>(stream)
            + output_one.reducible.item::<f32>(stream)
            + bias.clone().item::<f32>(stream);
        assert!((reduced - expected).abs() < 1e-5);
    }
}
