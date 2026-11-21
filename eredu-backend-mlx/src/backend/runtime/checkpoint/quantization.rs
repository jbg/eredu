//! MLX operators for packing dense weights.

use std::num::NonZeroI32;

use eredu_checkpoint::{
    checkpoint_conversion::{PackedSafetensorsWeight, SafetensorsWeightPacker},
    WeightQuantization,
};
use safemlx::{ops, Array, EvaluatedArray, Stream};
use safetensors::tensor::TensorView;

use crate::backend::error::Error;

/// Tensors produced from one dense quantized matrix.
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    /// Packed unsigned-integer weights.
    pub weight: Array,
    /// Per-group scales.
    pub scales: Array,
    /// Per-group affine biases.
    pub biases: Option<Array>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct QuantizationMatrixGeometry {
    leading_size: NonZeroI32,
    input_dims: NonZeroI32,
}

impl QuantizationMatrixGeometry {
    fn from_shape(shape: &[i32]) -> Result<Self, Error> {
        let Some((&input_dims, leading_dims)) = shape.split_last() else {
            return Err(Error::Quantization(
                "quantization requires a tensor with at least two dimensions".into(),
            ));
        };
        if leading_dims.is_empty() {
            return Err(Error::Quantization(
                "quantization requires a tensor with at least two dimensions".into(),
            ));
        }
        let input_dims = positive_dimension(input_dims).ok_or_else(|| {
            Error::Quantization(format!(
                "quantization input dimension must be nonzero, got shape {shape:?}"
            ))
        })?;
        let leading_size = leading_dims.iter().try_fold(1_i32, |size, &dimension| {
            let dimension = positive_dimension(dimension).ok_or_else(|| {
                Error::Quantization(format!(
                    "quantization leading dimensions must be nonzero, got shape {shape:?}"
                ))
            })?;
            size.checked_mul(dimension.get()).ok_or_else(|| {
                Error::Quantization(format!(
                    "quantization leading geometry cannot be flattened into an MLX matrix: {shape:?}"
                ))
            })
        })?;
        let leading_size = NonZeroI32::new(leading_size).ok_or_else(|| {
            Error::Quantization("quantization leading geometry must be nonzero".into())
        })?;
        Ok(Self {
            leading_size,
            input_dims,
        })
    }
}

fn positive_dimension(dimension: i32) -> Option<NonZeroI32> {
    NonZeroI32::new(dimension).filter(|_| dimension > 0)
}

fn packed_dimension(input_dims: NonZeroI32, bits: NonZeroI32) -> Result<NonZeroI32, Error> {
    i64::from(input_dims.get())
        .checked_mul(i64::from(bits.get()))
        .and_then(|bits| bits.checked_div(32))
        .and_then(|values| i32::try_from(values).ok())
        .and_then(NonZeroI32::new)
        .ok_or_else(|| Error::Quantization("quantized packed dimension overflow".into()))
}

/// Quantizes one floating-point weight using an explicit execution stream.
///
/// The last dimension is grouped and packed. Leading dimensions, including
/// bank dimensions are retained. Both on-the-fly model loading and
/// checkpoint conversion call this function.
pub fn quantize_tensor(
    weight: &Array,
    config: impl Into<WeightQuantization>,
    stream: &Stream,
) -> Result<QuantizedTensor, Error> {
    let config = config.into();
    config.validate()?;
    let mode = mlx_quantization_mode(config)?;
    if weight.ndim() < 2 || !weight.dtype().is_float() {
        return Err(Error::Quantization(format!(
            "expected a floating-point weight with at least two dimensions, got shape {:?} and dtype {:?}",
            weight.shape(),
            weight.dtype()
        )));
    }
    let geometry = QuantizationMatrixGeometry::from_shape(weight.shape())?;
    let input_dims = geometry.input_dims;
    let group_size = positive_dimension(config.group_size())
        .ok_or_else(|| Error::Quantization("quantization group size must be positive".into()))?;
    let bits = positive_dimension(config.bits())
        .ok_or_else(|| Error::Quantization("quantization bit width must be positive".into()))?;
    if input_dims.get() % group_size.get() != 0 || input_dims.get() % 32 != 0 {
        return Err(Error::Quantization(format!(
            "input dimension {} must be divisible by group_size {} and 32",
            input_dims, group_size
        )));
    }
    let original_shape = weight.shape();
    let matrix = if weight.ndim() == 2 {
        weight.clone()
    } else {
        weight.reshape(&[geometry.leading_size.get(), input_dims.get()], stream)?
    };
    let packed_dims = packed_dimension(input_dims, bits)?;
    let group_dims = NonZeroI32::new(input_dims.get() / group_size.get()).ok_or_else(|| {
        Error::Quantization("quantization group dimension must be nonzero".into())
    })?;
    let arrays = ops::quantize_with_mode(&matrix, group_size.get(), bits.get(), mode, stream)?;
    let restore_shape = |array: Array, last_dim: NonZeroI32| -> Result<Array, Error> {
        if weight.ndim() == 2 {
            Ok(array)
        } else {
            let mut shape = original_shape[..original_shape.len() - 1].to_vec();
            shape.push(last_dim.get());
            Ok(array.reshape(&shape, stream)?)
        }
    };
    Ok(QuantizedTensor {
        weight: restore_shape(arrays.weight, packed_dims)?,
        scales: restore_shape(arrays.scales, group_dims)?,
        biases: arrays
            .biases
            .map(|biases| restore_shape(biases, group_dims))
            .transpose()?,
    })
}

/// Maps a producible checkpoint quantization format to its MLX operator mode.
///
/// Checkpoint-native GGUF blocks are valid stored formats, but dense MLX
/// quantization cannot produce them.
pub fn mlx_quantization_mode(config: WeightQuantization) -> Result<ops::QuantizationMode, Error> {
    match config {
        WeightQuantization::Affine(_) => Ok(ops::QuantizationMode::Affine),
        WeightQuantization::MxFp4 => Ok(ops::QuantizationMode::MxFp4),
        WeightQuantization::NvFp4 => Ok(ops::QuantizationMode::NvFp4),
        WeightQuantization::GgufIQuant { .. } => Err(Error::Quantization(
            "checkpoint-native GGUF blocks cannot be produced by dense quantization".into(),
        )),
    }
}

/// MLX execution for the selected weights in an offline checkpoint conversion.
pub struct MlxSafetensorsWeightPacker<'a> {
    stream: &'a Stream,
}

impl<'a> MlxSafetensorsWeightPacker<'a> {
    /// Uses the given stream to pack dense weights.
    pub fn new(stream: &'a Stream) -> Self {
        Self { stream }
    }
}

impl SafetensorsWeightPacker for MlxSafetensorsWeightPacker<'_> {
    type Pending = Array;
    type Ready = EvaluatedArray<'static>;
    type Error = Error;

    fn pack(
        &mut self,
        input: TensorView<'_>,
        quantization: WeightQuantization,
    ) -> Result<PackedSafetensorsWeight<Array>, Error> {
        for &dimension in input.shape() {
            i32::try_from(dimension).map_err(|_| {
                Error::Quantization("input dimension exceeds MLX shape range".into())
            })?;
        }
        let input = Array::try_from(input).map_err(|error| Error::Other(Box::new(error)))?;
        let packed = quantize_tensor(&input, quantization, self.stream)?;
        Ok(PackedSafetensorsWeight {
            weight: packed.weight.contiguous(false, self.stream)?,
            scales: packed.scales.contiguous(false, self.stream)?,
            biases: packed
                .biases
                .map(|biases| biases.contiguous(false, self.stream))
                .transpose()?,
        })
    }

    fn packed_bytes(&self, tensor: &Array) -> usize {
        tensor.nbytes()
    }

    fn materialize(&mut self, arrays: Vec<Array>) -> Result<Vec<Self::Ready>, Error> {
        Ok(Array::into_evaluated_batch(arrays)?)
    }

    fn view<'a>(&self, tensor: &'a Self::Ready) -> Result<TensorView<'a>, Error> {
        tensor
            .safetensors_view()
            .map_err(|error| Error::Other(Box::new(error)))
    }
}
