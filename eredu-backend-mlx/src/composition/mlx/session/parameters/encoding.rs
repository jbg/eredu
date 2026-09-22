//! Native interpretation of architecture-declared effective parameter layouts.
use super::*;
use eredu_checkpoint::LinearFormat;
use eredu_nn::LinearCompanionRole;
use safemlx::Stream;

#[cfg(test)]
#[path = "encoding_tests.rs"]
mod tests;

#[derive(Clone)]
pub(super) struct EffectiveLayout {
    pub(super) format: LinearFormat,
    pub(super) shape: Vec<u64>,
    pub(super) row_layout: eredu_nn::LinearRowLayout,
    scale: Option<String>,
    affine_bias: Option<String>,
    pub(super) bank_loan: CaptureUsage,
}
impl EffectiveLayout {
    pub(super) fn new(format: LinearFormat, shape: Vec<u64>) -> Self {
        Self {
            format,
            shape,
            row_layout: eredu_nn::LinearRowLayout::Contiguous,
            scale: None,
            affine_bias: None,
            bank_loan: CaptureUsage::default(),
        }
    }
    pub(super) fn bind_companion(&mut self, role: LinearCompanionRole, id: String) {
        match role {
            LinearCompanionRole::Scale => self.scale = Some(id),
            LinearCompanionRole::AffineBias => self.affine_bias = Some(id),
        }
    }
    pub(super) fn input_transform(&self, actual_dtype: Dtype) -> ProjectionInputTransform {
        if matches!(self.format, LinearFormat::E4M3BlockFp8(_)) && actual_dtype == Dtype::Uint8 {
            ProjectionInputTransform::BlockFp8E4m3 {
                block_width: 128,
                floor: eredu_core::component::ComponentScalar::new(1.0e-4),
                magnitude: eredu_core::component::ComponentScalar::new(448.0),
            }
        } else {
            ProjectionInputTransform::Identity
        }
    }
    pub(super) fn supported(&self, actual_shape: &[u64], actual_dtype: Dtype) -> bool {
        if dtype(actual_dtype).is_some() {
            return self.shape == actual_shape;
        }
        match self.format {
            LinearFormat::Dense => false,
            LinearFormat::Affine(config) => {
                actual_dtype == Dtype::Uint32
                    && self.scale.is_some()
                    && self.affine_bias.is_some()
                    && self.packed_geometry(actual_shape, config.bits, config.group_size)
            }
            LinearFormat::MxFp4 => {
                actual_dtype == Dtype::Uint32
                    && self.scale.is_some()
                    && self.packed_geometry(actual_shape, 4, 32)
            }
            LinearFormat::E4M3BlockFp8(_) => {
                self.matrix_geometry()
                    && self
                        .row_layout
                        .rows_per_partition(self.shape[self.shape.len() - 2] as usize)
                        .is_ok()
                    && self.shape == actual_shape
                    && actual_dtype == Dtype::Uint8
                    && self.scale.is_some()
            }
            LinearFormat::GgufIQuant { ggml_type, .. } => {
                self.matrix_geometry()
                    && actual_dtype == Dtype::Uint8
                    && crate::native_quantization::NativeQuantizationFormat::from_ggml_type(
                        ggml_type,
                    )
                    .is_some()
                    && ggml_type.block_and_bytes().is_ok_and(|(block, bytes)| {
                        let width = *self.shape.last().expect("matrix geometry");
                        width % block as u64 == 0
                            && elements(&self.shape)
                                .ok()
                                .and_then(|count| (count / block as u64).checked_mul(bytes as u64))
                                .is_some_and(|expected| {
                                    elements(actual_shape).ok() == Some(expected)
                                })
                    })
            }
        }
    }
    fn matrix_geometry(&self) -> bool {
        (2..=3).contains(&self.shape.len())
            && self
                .shape
                .iter()
                .all(|axis| *axis > 0 && *axis <= i32::MAX as u64)
    }
    fn packed_geometry(&self, actual: &[u64], bits: i32, group: i32) -> bool {
        if !(2..=3).contains(&self.shape.len())
            || actual.len() != self.shape.len()
            || bits <= 0
            || group <= 0
        {
            return false;
        }
        let axis = self.shape.len() - 1;
        self.shape[..axis] == actual[..axis]
            && self.shape[axis] % group as u64 == 0
            && self.shape[axis]
                .checked_mul(bits as u64)
                .is_some_and(|width| width % 32 == 0 && width / 32 == actual[axis])
    }
    pub(super) fn extend_dependencies(&self, ids: &mut BTreeSet<String>) {
        ids.extend(self.scale.iter().cloned());
        ids.extend(self.affine_bias.iter().cloned());
    }
    pub(super) fn selection_metadata_bytes(&self, id: &str) -> Result<u64, ParameterError> {
        let mut bytes = id.len() as u64;
        for companion in [&self.scale, &self.affine_bias].into_iter().flatten() {
            bytes = add(bytes, companion.len() as u64)?;
        }
        Ok(add(1024, mul(bytes, 8)?)?)
    }
    pub(super) fn host_conversion_bytes(&self) -> Result<u64, ParameterError> {
        // The native GGUF decoder has an explicit host implementation. Account
        // for packed reads, temporary decoded vectors and native re-upload.
        if matches!(self.format, LinearFormat::GgufIQuant { .. }) {
            Ok(add(
                mul(elements(&self.shape)?, 32)?,
                self.bank_loan.host_bytes,
            )?)
        } else {
            Ok(self.bank_loan.host_bytes)
        }
    }
    pub(super) fn conversion_and_loan_bytes(&self) -> Result<u64, ParameterError> {
        // FP8 decoding expands both scale axes to complete 128-value blocks.
        // This scratch extent can greatly exceed a small logical matrix.
        if matches!(self.format, LinearFormat::E4M3BlockFp8(_)) {
            let (groups, rows, columns) = match self.shape.as_slice() {
                [rows, columns] => (1, *rows, *columns),
                [groups, rows, columns] => (*groups, *rows, *columns),
                _ => return Err(ParameterError::Invalid("FP8 matrix geometry".into())),
            };
            let rows = self
                .row_layout
                .scale_rows(
                    usize::try_from(rows).map_err(|_| ParameterError::Overflow)?,
                    128,
                )
                .map_err(|error| ParameterError::Invalid(error.to_string()))?
                as u64;
            let rows = mul(rows, 128)?;
            let columns = mul(add(columns, 127)? / 128, 128)?;
            Ok(add(
                mul(mul(mul(groups, rows)?, columns)?, 8)?,
                self.bank_loan.retained_bytes,
            )?)
        } else {
            Ok(self.bank_loan.retained_bytes)
        }
    }
    pub(super) fn effective(
        &self,
        id: &str,
        values: &BTreeMap<String, MlxTensor>,
        stream: &Stream,
    ) -> Result<Array, Error> {
        let weight = values.get(id).map(MlxTensor::as_array).ok_or_else(|| {
            Error::ArchitectureModel(format!("loaded effective parameter slot {id} disappeared"))
        })?;
        self.decode_sources(
            weight,
            self.scale
                .as_ref()
                .and_then(|id| values.get(id))
                .map(MlxTensor::as_array),
            self.affine_bias
                .as_ref()
                .and_then(|id| values.get(id))
                .map(MlxTensor::as_array),
            stream,
        )
    }
    pub(super) fn source_ids<'a>(&'a self, primary: &'a str) -> [Option<&'a str>; 3] {
        [
            Some(primary),
            self.scale.as_deref(),
            self.affine_bias.as_deref(),
        ]
    }
    pub(super) fn decode_sources(
        &self,
        weight: &Array,
        scale: Option<&Array>,
        bias: Option<&Array>,
        stream: &Stream,
    ) -> Result<Array, Error> {
        eredu_nn::parameter_values::decode_parameter(
            NativeDecoding {
                layout: self,
                weight,
                scale,
                bias,
                stream,
            },
            eredu_nn::parameter_values::ParameterDecoding {
                format: self.format,
                row_layout: self.row_layout,
            },
        )
    }
}

struct NativeDecoding<'a> {
    layout: &'a EffectiveLayout,
    weight: &'a Array,
    scale: Option<&'a Array>,
    bias: Option<&'a Array>,
    stream: &'a Stream,
}
impl NativeDecoding<'_> {
    fn shape(&self) -> Result<Vec<i32>, Error> {
        self.layout
            .shape
            .iter()
            .map(|axis| {
                i32::try_from(*axis).map_err(|_| {
                    Error::ArchitectureModel(
                        "effective parameter shape exceeds native extent".into(),
                    )
                })
            })
            .collect()
    }
}
impl eredu_nn::parameter_values::ParameterDecodingMechanism for NativeDecoding<'_> {
    type Value = Array;
    type Error = Error;
    fn weight(&self) -> Result<&Array, Error> {
        Ok(self.weight)
    }
    fn is_floating(&self, value: &Array) -> bool {
        dtype(value.dtype()).is_some()
    }
    fn alias(&self, value: &Array) -> Result<Array, Error> {
        Ok(value.clone())
    }
    fn scale(&self) -> Result<&Array, Error> {
        self.scale
            .ok_or_else(|| Error::ArchitectureModel("packed scale is absent".into()))
    }
    fn affine_bias(&self) -> Result<&Array, Error> {
        self.bias
            .ok_or_else(|| Error::ArchitectureModel("affine bias is absent".into()))
    }
    fn cast_f32(&self, value: &Array) -> Result<Array, Error> {
        Ok(value.as_dtype(Dtype::Float32, self.stream)?)
    }
    fn affine(
        &self,
        weight: &Array,
        scale: &Array,
        bias: &Array,
        config: eredu_checkpoint::AffineQuantization,
    ) -> Result<Array, Error> {
        Ok(safemlx::ops::dequantize_with_mode(
            weight,
            scale,
            Some(bias),
            config.group_size,
            config.bits,
            safemlx::ops::QuantizationMode::Affine,
            self.stream,
        )?)
    }
    fn mx_fp4(&self, weight: &Array, scale: &Array) -> Result<Array, Error> {
        Ok(safemlx::ops::dequantize_with_mode(
            weight,
            scale,
            None,
            32,
            4,
            safemlx::ops::QuantizationMode::MxFp4,
            self.stream,
        )?)
    }
    fn block_fp8(
        &self,
        weight: &Array,
        scale: &Array,
        decoding: eredu_nn::parameter_values::ParameterDecoding,
    ) -> Result<Array, Error> {
        Ok(crate::backend::nn::fp8::dequantize_with_row_layout(
            weight,
            scale,
            decoding.row_layout,
            self.stream,
        )?)
    }
    fn gguf(
        &self,
        weight: &Array,
        decoding: eredu_nn::parameter_values::ParameterDecoding,
    ) -> Result<Array, Error> {
        let LinearFormat::GgufIQuant { ggml_type, endian } = decoding.format else {
            return Err(Error::ArchitectureModel(
                "GGUF parameter decoder has a different format".into(),
            ));
        };
        Ok(
            crate::native_quantization::NativeQuantizedTensor::from_iq_array(
                weight.clone(),
                &self.shape()?,
                ggml_type,
                endian,
            )?
            .dequantize(self.stream)?,
        )
    }
    fn restore_shape(&self, value: Array) -> Result<Array, Error> {
        Ok(value.reshape(&self.shape()?, self.stream)?)
    }
    fn invalid_dense(&self) -> Error {
        Error::ArchitectureModel("dense parameter has a non-floating dtype".into())
    }
}

/// Selected F32 host values; the caller validates geometry and prepays storage.
pub(super) fn read_effective(
    tensor: &Array,
    region: &ParameterRegion,
    stream: &Stream,
) -> Result<Vec<f32>, Error> {
    eredu_nn::parameter_values::read_parameter(NativeValues {
        tensor,
        region,
        projection: None,
        stream,
    })
}

/// Native contraction of an already decoded effective tensor. Only the result
/// reaches the host; the caller owns geometry validation and all reservations.
pub(super) fn project_effective(
    tensor: &Array,
    projection: &ParameterProjection,
    stream: &Stream,
) -> Result<Vec<f32>, Error> {
    eredu_nn::parameter_values::project_parameter(NativeValues {
        tensor,
        region: &projection.region,
        projection: Some(projection),
        stream,
    })
}

struct NativeValues<'a> {
    tensor: &'a Array,
    region: &'a ParameterRegion,
    projection: Option<&'a ParameterProjection>,
    stream: &'a Stream,
}

/// Fixed host adapter frames for the shared decoder. Native constructor
/// populations and any codec staging require their independent source facts.
pub(super) fn decoding_host_control_bytes(layout: &EffectiveLayout) -> Option<usize> {
    let shape_bytes = if matches!(layout.format, LinearFormat::GgufIQuant { .. }) {
        layout
            .shape
            .len()
            .checked_mul(std::mem::size_of::<i32>())?
            .checked_mul(2)?
    } else {
        0
    };
    std::mem::size_of::<NativeDecoding<'_>>()
        .checked_add(std::mem::size_of::<Result<Array, Error>>())?
        .checked_add(shape_bytes)
}
pub(super) fn read_host_control_bytes(region: &ParameterRegion) -> Option<usize> {
    use std::mem::{size_of, size_of_val};
    let frames = [
        size_of::<NativeValues<'_>>(),
        size_of::<Vec<ArrayIndexOp<'_>>>(),
        size_of::<Array>(),
        size_of::<Result<Array, Error>>(),
        size_of::<Result<Vec<f32>, Error>>(),
        region
            .shape
            .len()
            .checked_mul(size_of::<ArrayIndexOp<'_>>())?,
        safemlx::EvaluatedArray::completed_readback_control_bytes::<f32>()?,
        crate::backend::runtime::cache::completed_borrow_control_bytes()?,
    ];
    frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)
}

/// Host index/axis vectors and fixed transports of the selected shared worker.
/// The completed output buffer has independent result custody.
pub(super) fn projection_host_control_bytes(projection: &ParameterProjection) -> Option<usize> {
    use std::mem::{size_of, size_of_val};
    let rank = projection.region.shape.len();
    let frames = [
        size_of::<NativeValues<'_>>(),
        size_of::<Vec<ArrayIndexOp<'_>>>(),
        size_of::<Vec<i32>>(),
        size_of::<[i32; 2]>(),
        size_of::<Array>(),
        size_of::<Result<Array, Error>>(),
        size_of::<Result<Vec<f32>, Error>>(),
        rank.checked_mul(size_of::<ArrayIndexOp<'_>>())?,
        rank.checked_mul(size_of::<i32>())?,
        safemlx::EvaluatedArray::completed_readback_control_bytes::<f32>()?,
        crate::backend::runtime::cache::completed_borrow_control_bytes()?,
    ];
    frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)
}
impl eredu_nn::parameter_values::ParameterValueMechanism for NativeValues<'_> {
    type Value = Array;
    type Output = Vec<f32>;
    type Error = Error;
    fn select(&self) -> Result<Array, Error> {
        Ok(self
            .tensor
            .try_index_device(native_indices(self.region).as_slice(), self.stream)?)
    }
    fn cast_f32(&self, value: Array) -> Result<Array, Error> {
        Ok(value.as_dtype(Dtype::Float32, self.stream)?)
    }
    fn contiguous(&self, value: Array) -> Result<Array, Error> {
        Ok(value.contiguous(false, self.stream)?)
    }
    fn complete_and_read(&self, value: Array) -> Result<Vec<f32>, Error> {
        Ok(
            crate::backend::runtime::cache::complete_and_borrow(&value, self.stream)?
                .as_slice::<f32>()
                .to_vec(),
        )
    }
}
impl eredu_nn::parameter_values::ParameterProjectionMechanism for NativeValues<'_> {
    fn transpose_weights(&self, value: Array) -> Result<Array, Error> {
        let projection = self.projection.expect("selected projection worker");
        let mut axes = Vec::with_capacity(self.region.shape.len());
        for axis in 0..self.region.shape.len() {
            if axis != projection.axis {
                axes.push(axis as i32);
            }
        }
        axes.push(projection.axis as i32);
        Ok(value.transpose_axes(&axes, self.stream)?)
    }
    fn reshape_weights(&self, value: Array) -> Result<Array, Error> {
        let projection = self.projection.expect("selected projection worker");
        Ok(value.reshape(
            &[-1, self.region.shape[projection.axis] as i32],
            self.stream,
        )?)
    }
    fn coefficients(&self) -> Result<Array, Error> {
        let projection = self.projection.expect("selected projection worker");
        Ok(Array::try_from_slice(
            &projection.coefficients,
            &[
                projection.directions as i32,
                self.region.shape[projection.axis] as i32,
            ],
        )?)
    }
    fn transpose_coefficients(&self, value: Array) -> Result<Array, Error> {
        Ok(value.transpose(self.stream)?)
    }
    fn matmul(&self, weights: &Array, coefficients: &Array) -> Result<Array, Error> {
        Ok(weights.matmul(coefficients, self.stream)?)
    }
}
