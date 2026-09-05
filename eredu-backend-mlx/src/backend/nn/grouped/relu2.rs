use super::gated_product::GroupProjectionParams;
use super::*;

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
