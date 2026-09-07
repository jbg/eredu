use super::*;

mod intervention;

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
            Self::SqrtSoftplus => super::super::layers::softplus(logits, stream)?.sqrt(stream),
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
        let logits = self.project_logits(hidden_states, stream)?;
        let scores = self.score_function.apply(logits, stream)?;
        let mut scores_for_choice = scores.clone();
        if let Some(bias) = self.e_score_correction_bias.as_ref() {
            scores_for_choice = scores_for_choice.add(bias, stream)?;
        }
        if let Some(bias) = selection_bias {
            scores_for_choice = scores_for_choice.add(bias, stream)?;
        }

        let top_k_index = self.topk_indices(&scores_for_choice, stream)?;
        self.weights_for_indices(&scores, top_k_index, stream)
    }

    /// Returns caller-selected ids, their raw transformed scores, and final
    /// normalized/scaled selection weights.
    pub fn select_indices(
        &mut self,
        hidden_states: &Array,
        group_indices: &Array,
        stream: &Stream,
    ) -> Result<GroupSelectionOutput, Exception> {
        let logits = self.project_logits(hidden_states, stream)?;
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

    fn project_logits(&self, hidden_states: &Array, stream: &Stream) -> Result<Array, Exception> {
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
        Ok(logits)
    }

    fn weights_for_indices(
        &self,
        scores: &Array,
        top_k_index: Array,
        stream: &Stream,
    ) -> Result<GroupSelectionOutput, Exception> {
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
