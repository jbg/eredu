//! The same final physical parameter declarations for loaded and unloaded banks.
use super::*;

/// A constructor receives explicit values or constructs ordinary placeholders.
/// This interface carries no implicit source lookup or allocation authority.
pub(crate) trait ParameterFactory {
    type Error;
    fn array(&mut self, name: &str, shape: &[i32], dtype: Dtype,
        floating_shape: Option<&[i32]>) -> Result<Array, Self::Error>;
    fn geometry(&self, message: &'static str) -> Self::Error;
}
pub(crate) struct UnloadedParameters<'a>(pub &'a Stream);
impl ParameterFactory for UnloadedParameters<'_> {
    type Error = Exception;
    fn array(&mut self, _: &str, shape: &[i32], dtype: Dtype, _: Option<&[i32]>)
        -> Result<Array, Exception> { zeros_dtype(shape, dtype, self.0) }
    fn geometry(&self, message: &'static str) -> Exception { Exception::custom(message) }
}

pub(super) fn projection<F: ParameterFactory>(factory: &mut F, names: [&str; 3],
    groups: i32, output: i32, input: i32, quantization: Option<WeightQuantization>,
    iquant: Option<WeightQuantization>, dense_dtype: Dtype,
    fp8: Option<(eredu_checkpoint::BlockFp8Format, eredu_nn::LinearRowLayout)>)
    -> Result<gated_product::GroupProjectionParams, F::Error> {
    let logical = [groups, output, input];
    let (weight_shape, weight_dtype, scale, affine_bias) = if let Some((format, rows)) = fp8 {
        if format.block_rows != 128 || format.block_columns != 128 {
            return Err(factory.geometry("MLX block-FP8 groups require [128, 128] blocks"));
        }
        let scale_rows = rows.scale_rows_fixed(output as usize, 128)
            .ok().and_then(|rows| i32::try_from(rows).ok())
            .ok_or_else(||factory.geometry("FP8 row scale count exceeds i32"))?;
        let scale_dtype = match format.scale_encoding {
            eredu_checkpoint::BlockFp8ScaleEncoding::FloatingPoint => Dtype::Float32,
            eredu_checkpoint::BlockFp8ScaleEncoding::Ue8m0 => Dtype::Uint8,
        };
        let columns = input.checked_add(127).map(|n| n / 128)
            .ok_or_else(||factory.geometry("FP8 scale columns overflow"))?;
        (logical, Dtype::Uint8, Some(([groups, scale_rows, columns], scale_dtype)), None)
    } else if let Some(iquant) = iquant {
        let (ty, _) = iquant.gguf_iquant().ok_or_else(||factory.geometry("invalid IQ group format"))?;
        let (values, bytes) = ty.block_and_bytes().map_err(|_|factory.geometry("invalid IQ block geometry"))?;
        let columns = input.checked_div(values as i32).and_then(|n| n.checked_mul(bytes as i32))
            .ok_or_else(||factory.geometry("IQ packed group extent overflow"))?;
        ([groups, output, columns], Dtype::Uint8, None, None)
    } else if let Some(quantization) = quantization {
        if input % quantization.group_size() != 0 {
            return Err(factory.geometry("packed group input width is not divisible by its quantization group size"));
        }
        let shape = [groups, output, input / quantization.group_size()];
        ([groups, output, quantized_packed_dimension(input, quantization.bits())], Dtype::Uint32,
            Some((shape, if quantization == WeightQuantization::MxFp4 {Dtype::Uint8} else {Dtype::Float16})),
            quantization.has_biases().then_some((shape, Dtype::Float16)))
    } else { (logical, dense_dtype, None, None) };
    let weight = PhysicalParam::new(factory.array(names[0], &weight_shape, weight_dtype, Some(&logical))?);
    let scale = PhysicalParam::new(scale.map(|(shape,dtype)|factory.array(names[1], &shape, dtype, None)).transpose()?);
    let bias = PhysicalParam::new(affine_bias.map(|(shape,dtype)|factory.array(names[2], &shape, dtype, None)).transpose()?);
    Ok((weight, scale, bias))
}

/// Actual shared projection and physical-field call/return controls; native
/// descriptor storage is supplied by the explicit parameter factory.
pub(crate) fn parameter_factory_control_bytes()->Option<usize> {
    use std::mem::{size_of,size_of_val};
    let frames=[size_of::<[i32;3]>(),size_of::<([i32;3],Dtype,Option<([i32;3],Dtype)>,Option<([i32;3],Dtype)>)>(),
        size_of::<gated_product::GroupProjectionParams>(),
        size_of::<([&str;3],i32,i32,i32,Option<WeightQuantization>,Option<WeightQuantization>,Dtype,
            Option<(eredu_checkpoint::BlockFp8Format,eredu_nn::LinearRowLayout)>)>(),
        size_of::<(PhysicalParam<Array>,PhysicalParam<Option<Array>>,PhysicalParam<Option<Array>>)>() ,
        size_of::<Result<Array,eredu_nn::Error>>(),size_of::<Result<Option<Array>,eredu_nn::Error>>(),
        size_of::<Result<gated_product::GroupProjectionParams,eredu_nn::Error>>(),
        size_of::<(&str,&[i32],Dtype,Option<&[i32]>)>()];
    frames.into_iter().try_fold(size_of_val(&frames),usize::checked_add)
}
