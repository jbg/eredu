use super::*;

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

pub(super) type GroupProjectionParams = (
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
                super::super::layers::gelu_approximate(gate, stream)?
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
