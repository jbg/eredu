//! CPU spectral kernels over row-major Candle tensors.

use eredu_nn::{
    operation_geometry::{InverseRealFftGeometry, RealFftGeometry},
    Error, Tensor,
};
use rustfft::{num_complex::Complex32, FftPlanner};

use crate::{CandleContext, CandleTensor};

pub(crate) fn forward(
    input: &CandleTensor,
    length: i32,
    axis: i32,
    context: &CandleContext,
) -> Result<(CandleTensor, CandleTensor), Error> {
    let geometry = RealFftGeometry::new(input.shape(), length, axis)?;
    let axis = geometry.axis();
    let shape = input.shape();
    let last = shape.len() as i32 - 1;
    let input = f32_lanes(input, axis, last, context)?;
    let length = length as usize;
    let bins = length / 2 + 1;
    let plan = FftPlanner::new().plan_fft_forward(length);
    let mut scratch = vec![Complex32::default(); plan.get_inplace_scratch_len()];
    let mut lane = vec![Complex32::default(); length];
    let count = input.len() / length * bins;
    let (mut real, mut imaginary) = (Vec::with_capacity(count), Vec::with_capacity(count));
    for row in input.chunks_exact(length) {
        for (value, &sample) in lane.iter_mut().zip(row) {
            *value = Complex32::new(sample, 0.0);
        }
        plan.process_with_scratch(&mut lane, &mut scratch);
        real.extend(lane[..bins].iter().map(|value| value.re));
        imaginary.extend(lane[..bins].iter().map(|value| value.im));
    }
    Ok((
        restore_axis(&real, shape, axis, last, bins as i32, context)?,
        restore_axis(&imaginary, shape, axis, last, bins as i32, context)?,
    ))
}

pub(crate) fn inverse(
    real: &CandleTensor,
    imaginary: &CandleTensor,
    length: i32,
    axis: i32,
    context: &CandleContext,
) -> Result<CandleTensor, Error> {
    let geometry = InverseRealFftGeometry::new(real.shape(), imaginary.shape(), length, axis)?;
    let axis = geometry.axis();
    let shape = real.shape();
    let last = shape.len() as i32 - 1;
    let real = f32_lanes(real, axis, last, context)?;
    let imaginary = f32_lanes(imaginary, axis, last, context)?;
    let length = length as usize;
    let bins = length / 2 + 1;
    let plan = FftPlanner::new().plan_fft_inverse(length);
    let mut scratch = vec![Complex32::default(); plan.get_inplace_scratch_len()];
    let mut lane = vec![Complex32::default(); length];
    let mut output = Vec::with_capacity(real.len() / bins * length);
    for (real, imaginary) in real.chunks_exact(bins).zip(imaginary.chunks_exact(bins)) {
        lane[0] = Complex32::new(real[0], 0.0);
        for bin in 1..bins {
            if length - bin == bin {
                lane[bin] = Complex32::new(real[bin], 0.0);
            } else {
                let value = Complex32::new(real[bin], imaginary[bin]);
                lane[bin] = value;
                lane[length - bin] = value.conj();
            }
        }
        plan.process_with_scratch(&mut lane, &mut scratch);
        output.extend(lane.iter().map(|value| value.re / length as f32));
    }
    restore_axis(&output, shape, axis, last, length as i32, context)
}

fn f32_lanes(
    tensor: &CandleTensor,
    axis: i32,
    last: i32,
    context: &CandleContext,
) -> Result<Vec<f32>, Error> {
    if tensor.value.dtype() != candle_core::DType::F32 {
        return Err(Error::backend("real FFT requires F32 tensors"));
    }
    context
        .validate_tensor_device(tensor)
        .map_err(Error::backend_source)?;
    tensor.swap_axes(axis, last, context)?.to_f32_vec(context)
}

fn restore_axis(
    values: &[f32],
    shape: &[i32],
    axis: i32,
    last: i32,
    width: i32,
    context: &CandleContext,
) -> Result<CandleTensor, Error> {
    let mut shape = shape.to_vec();
    shape.swap(axis as usize, last as usize);
    shape[last as usize] = width;
    CandleTensor::from_f32_slice(values, &shape, context)?.swap_axes(axis, last, context)
}
