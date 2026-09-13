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
        let get = |id: &str| {
            values.get(id).map(MlxTensor::as_array).ok_or_else(|| {
                Error::ArchitectureModel(format!(
                    "loaded effective parameter slot {id} disappeared"
                ))
            })
        };
        let weight = get(id)?;
        // A published edit is a dense copy of this affected parameter. Its
        // original packed slots remain retained for exact restoration.
        if dtype(weight.dtype()).is_some() {
            return Ok(weight.clone());
        }
        let scale = || {
            self.scale
                .as_deref()
                .ok_or_else(|| Error::ArchitectureModel("packed scale is absent".into()))
                .and_then(get)
        };
        let result = match self.format {
            LinearFormat::Dense => {
                return Err(Error::ArchitectureModel(
                    "dense parameter has a non-floating dtype".into(),
                ))
            }
            LinearFormat::Affine(config) => {
                let bias = self
                    .affine_bias
                    .as_deref()
                    .ok_or_else(|| Error::ArchitectureModel("affine bias is absent".into()))?;
                safemlx::ops::dequantize_with_mode(
                    weight,
                    scale()?.as_dtype(Dtype::Float32, stream)?,
                    Some(&get(bias)?.as_dtype(Dtype::Float32, stream)?),
                    config.group_size,
                    config.bits,
                    safemlx::ops::QuantizationMode::Affine,
                    stream,
                )?
            }
            LinearFormat::MxFp4 => safemlx::ops::dequantize_with_mode(
                weight,
                scale()?,
                None,
                32,
                4,
                safemlx::ops::QuantizationMode::MxFp4,
                stream,
            )?,
            LinearFormat::E4M3BlockFp8(_) => crate::backend::nn::fp8::dequantize_with_row_layout(
                weight,
                scale()?,
                self.row_layout,
                stream,
            )?,
            LinearFormat::GgufIQuant { ggml_type, endian } => {
                let shape = self
                    .shape
                    .iter()
                    .map(|axis| *axis as i32)
                    .collect::<Vec<_>>();
                crate::native_quantization::NativeQuantizedTensor::from_iq_array(
                    weight.clone(),
                    &shape,
                    ggml_type,
                    endian,
                )?
                .dequantize(stream)?
                .reshape(&shape, stream)?
            }
        };
        Ok(result.as_dtype(Dtype::Float32, stream)?)
    }
}

/// Selected F32 host values; the caller validates geometry and prepays storage.
pub(super) fn read_effective(
    tensor: &Array,
    region: &ParameterRegion,
    stream: &Stream,
) -> Result<Vec<f32>, Error> {
    let output = tensor
        .try_index_device(native_indices(region).as_slice(), stream)?
        .as_dtype(Dtype::Float32, stream)?
        .contiguous(false, stream)?;
    Ok(output.evaluated()?.as_slice::<f32>().to_vec())
}

/// Native contraction of an already decoded effective tensor. Only the result
/// reaches the host; the caller owns geometry validation and all reservations.
pub(super) fn project_effective(
    tensor: &Array,
    projection: &ParameterProjection,
    stream: &Stream,
) -> Result<Vec<f32>, Error> {
    let mut axes: Vec<_> = (0..projection.region.shape.len())
        .filter(|axis| *axis != projection.axis)
        .map(|axis| axis as i32)
        .collect();
    axes.push(projection.axis as i32);
    let width = projection.region.shape[projection.axis] as i32;
    let weights = tensor
        .try_index_device(native_indices(&projection.region).as_slice(), stream)?
        .as_dtype(Dtype::Float32, stream)?
        .transpose_axes(&axes, stream)?
        .reshape(&[-1, width], stream)?;
    let coefficients = Array::from_slice(
        &projection.coefficients,
        &[projection.directions as i32, width],
    )
    .transpose(stream)?;
    let result = weights
        .matmul(&coefficients, stream)?
        .contiguous(false, stream)?;
    let values = result.evaluated()?.as_slice::<f32>().to_vec();
    Ok(values)
}
