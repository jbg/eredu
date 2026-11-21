//! Native interpretation of architecture-declared effective parameter layouts.
use super::*;
use eredu_checkpoint::LinearFormat;
use eredu_nn::LinearCompanionRole;
use safemlx::Stream;

pub(super) fn input_transform(
    layout: &EffectiveParameterLayout,
    actual_dtype: Dtype,
) -> ProjectionInputTransform {
    if matches!(layout.format, LinearFormat::E4M3BlockFp8(_)) && actual_dtype == Dtype::Uint8 {
        ProjectionInputTransform::BlockFp8E4m3 {
            block_width: 128,
            floor: eredu_core::component::ComponentScalar::new(1.0e-4),
            magnitude: eredu_core::component::ComponentScalar::new(448.0),
        }
    } else {
        ProjectionInputTransform::Identity
    }
}
pub(super) fn supported(
    layout: &EffectiveParameterLayout,
    actual_shape: &[u64],
    actual_dtype: Dtype,
) -> bool {
    if dtype(actual_dtype).is_some() {
        return layout.shape == actual_shape;
    }
    match layout.format {
        LinearFormat::Dense => false,
        LinearFormat::Affine(config) => {
            actual_dtype == Dtype::Uint32
                && layout.companion(LinearCompanionRole::Scale).is_some()
                && layout.companion(LinearCompanionRole::AffineBias).is_some()
                && packed_geometry(layout, actual_shape, config.bits, config.group_size)
        }
        LinearFormat::MxFp4 => {
            actual_dtype == Dtype::Uint32
                && layout.companion(LinearCompanionRole::Scale).is_some()
                && packed_geometry(layout, actual_shape, 4, 32)
        }
        LinearFormat::NvFp4 => {
            actual_dtype == Dtype::Uint32
                && layout.companion(LinearCompanionRole::Scale).is_some()
                && layout.companion(LinearCompanionRole::GlobalScale).is_some()
                && packed_geometry(layout, actual_shape, 4, 16)
                && layout
                    .row_layout
                    .rows_per_partition(layout.shape[layout.shape.len() - 2] as usize)
                    .is_ok()
        }
        LinearFormat::E4M3BlockFp8(_) => {
            matrix_geometry(layout)
                && layout
                    .row_layout
                    .rows_per_partition(layout.shape[layout.shape.len() - 2] as usize)
                    .is_ok()
                && layout.shape == actual_shape
                && actual_dtype == Dtype::Uint8
                && layout.companion(LinearCompanionRole::Scale).is_some()
        }
        LinearFormat::GgufIQuant { ggml_type, .. } => {
            matrix_geometry(layout)
                && actual_dtype == Dtype::Uint8
                && crate::native_quantization::NativeQuantizationFormat::from_ggml_type(ggml_type)
                    .is_some()
                && ggml_type.block_and_bytes().is_ok_and(|(block, bytes)| {
                    let width = *layout.shape.last().expect("matrix geometry");
                    width.is_multiple_of(block)
                        && elements(&layout.shape)
                            .ok()
                            .and_then(|count| (count / block).checked_mul(bytes))
                            .is_some_and(|expected| elements(actual_shape).ok() == Some(expected))
                })
        }
    }
}
fn matrix_geometry(layout: &EffectiveParameterLayout) -> bool {
    (2..=3).contains(&layout.shape.len())
        && layout
            .shape
            .iter()
            .all(|axis| *axis > 0 && *axis <= i32::MAX as u64)
}
fn packed_geometry(
    layout: &EffectiveParameterLayout,
    actual: &[u64],
    bits: i32,
    group: i32,
) -> bool {
    if !(2..=3).contains(&layout.shape.len())
        || actual.len() != layout.shape.len()
        || bits <= 0
        || group <= 0
    {
        return false;
    }
    let axis = layout.shape.len() - 1;
    layout.shape[..axis] == actual[..axis]
        && layout.shape[axis].is_multiple_of(group as u64)
        && layout.shape[axis]
            .checked_mul(bits as u64)
            .is_some_and(|width| width % 32 == 0 && width / 32 == actual[axis])
}
pub(super) fn host_conversion_bytes(
    layout: &EffectiveParameterLayout,
) -> Result<u64, ParameterError> {
    // The native GGUF decoder has an explicit host implementation. Account
    // for packed reads, temporary decoded vectors and native re-upload.
    if matches!(
        layout.format,
        LinearFormat::GgufIQuant { .. } | LinearFormat::NvFp4
    ) {
        Ok(mul(elements(&layout.shape)?, 32)?)
    } else {
        Ok(0)
    }
}
pub(super) fn conversion_bytes(layout: &EffectiveParameterLayout) -> Result<u64, ParameterError> {
    // FP8 decoding expands both scale axes to complete 128-value blocks.
    // This scratch extent can greatly exceed a small logical matrix.
    if matches!(layout.format, LinearFormat::E4M3BlockFp8(_)) {
        let (groups, rows, columns) = match layout.shape.as_slice() {
            [rows, columns] => (1, *rows, *columns),
            [groups, rows, columns] => (*groups, *rows, *columns),
            _ => return Err(ParameterError::Invalid("FP8 matrix geometry".into())),
        };
        let rows = layout
            .row_layout
            .scale_rows(
                usize::try_from(rows).map_err(|_| ParameterError::Overflow)?,
                128,
            )
            .map_err(|error| ParameterError::Invalid(error.to_string()))? as u64;
        let rows = mul(rows, 128)?;
        let columns = mul(add(columns, 127)? / 128, 128)?;
        Ok(mul(mul(mul(groups, rows)?, columns)?, 8)?)
    } else {
        Ok(0)
    }
}
pub(super) fn effective(
    layout: &EffectiveParameterLayout,
    id: &str,
    values: &BTreeMap<String, MlxTensor>,
    stream: &Stream,
) -> Result<Array, Error> {
    let get = |id: &str| {
        values.get(id).map(MlxTensor::as_array).ok_or_else(|| {
            Error::ArchitectureModel(format!("loaded effective parameter slot {id} disappeared"))
        })
    };
    let weight = get(id)?;
    // A published edit is a dense copy of this affected parameter. Its
    // original packed slots remain retained for exact restoration.
    if dtype(weight.dtype()).is_some() {
        return Ok(weight.clone());
    }
    let scale = || {
        layout
            .companion(LinearCompanionRole::Scale)
            .ok_or_else(|| Error::ArchitectureModel("packed scale is absent".into()))
            .and_then(get)
    };
    let result = match layout.format {
        LinearFormat::Dense => {
            return Err(Error::ArchitectureModel(
                "dense parameter has a non-floating dtype".into(),
            ));
        }
        LinearFormat::Affine(config) => {
            let bias = layout
                .companion(LinearCompanionRole::AffineBias)
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
        LinearFormat::NvFp4 => crate::backend::nn::nvfp4::dequantize(
            weight,
            scale()?,
            get(layout
                .companion(LinearCompanionRole::GlobalScale)
                .ok_or_else(|| Error::ArchitectureModel("NVFP4 global scale is absent".into()))?)?,
            layout.row_layout,
            stream,
        )?,
        LinearFormat::E4M3BlockFp8(_) => crate::backend::nn::fp8::dequantize_with_row_layout(
            weight,
            scale()?,
            layout.row_layout,
            stream,
        )?,
        LinearFormat::GgufIQuant { ggml_type, endian } => {
            let shape = layout
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
