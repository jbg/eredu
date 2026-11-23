//! Native recurrent operator dispatch and CPU kernels.

use candle_core::{DType, Tensor as NativeTensor};
use eredu_nn::{Error, GatedDeltaScanInput, GatedDeltaScanOutput};
use rayon::prelude::*;

use crate::CandleTensor;

/// Per-channel reduction over an already padded causal input.
pub(crate) fn depthwise_convolution(
    input: &NativeTensor,
    weight: &NativeTensor,
    dilation: usize,
) -> candle_core::Result<NativeTensor> {
    let (batch, sequence, channels) = input.dims3()?;
    let kernel = weight.dim(1)?;
    // The ARM CPU reduction rounds products in groups of four before adding
    // them in tap order, then uses fused multiply-add for the remainder.
    let separate_products = if cfg!(target_arch = "aarch64") {
        kernel / 4 * 4
    } else {
        0
    };
    let output_length = sequence - dilation * (kernel - 1);
    let source = input.flatten_all()?.to_vec1::<f32>()?;
    let weights = weight.flatten_all()?.to_vec1::<f32>()?;
    let mut output = vec![0.; batch * output_length * channels];
    output
        .par_iter_mut()
        .with_min_len(512)
        .enumerate()
        .for_each(|(index, value)| {
            let channel = index % channels;
            let row = index / channels;
            let start = ((row / output_length) * sequence + row % output_length) * channels;
            let taps = &weights[channel * kernel..][..kernel];
            *value = taps.iter().enumerate().fold(0.0, |sum, (tap, weight)| {
                let input = source[start + tap * dilation * channels + channel];
                if tap < separate_products {
                    sum + input * weight
                } else {
                    input.mul_add(*weight, sum)
                }
            });
        });
    NativeTensor::from_vec(output, (batch, output_length, channels), input.device())
}

pub(crate) fn gated_delta_scan(
    input: GatedDeltaScanInput<'_, CandleTensor>,
) -> Result<GatedDeltaScanOutput<CandleTensor>, Error> {
    let result = (|| {
        let q = &input.query.value;
        let k = &input.key.value;
        let v = &input.value.value;
        let decay = &input.log_decay.value;
        let beta = &input.beta.value;
        let (batch, sequence, heads, keys) = q.dims4()?;
        let values = v.dim(candle_core::D::Minus1)?;
        let groups = [batch, sequence, heads];
        let vector_decay = decay.dims() == q.dims();
        let state_shape = (batch, heads, keys, values);
        if [batch, sequence, heads, keys, values].contains(&0)
            || k.dims() != q.dims()
            || v.dims() != [batch, sequence, heads, values]
            || beta.dims() != groups
            || (!vector_decay && decay.dims() != groups)
            || [q, k, v, decay, beta].into_iter().any(|tensor| {
                !tensor.device().same_device(q.device())
                    || !matches!(tensor.dtype(), DType::F32 | DType::F16 | DType::BF16)
            })
        {
            candle_core::bail!("invalid gated-delta activation geometry");
        }
        let initial = input.initial_state.map(|state| &state.value);
        if initial.is_some_and(|state| {
            state.dims() != [batch, heads, keys, values]
                || state.dtype() != DType::F32
                || !state.device().same_device(q.device())
        }) {
            candle_core::bail!("invalid gated-delta initial state");
        }
        #[cfg(all(feature = "metal", target_os = "macos"))]
        if q.device().is_metal() {
            return crate::metal_f32::gated_delta_scan(q, k, v, decay, beta, initial);
        }
        if !q.device().is_cpu() {
            candle_core::bail!("gated-delta scan is unavailable on this device");
        }
        let read = |x: &NativeTensor| x.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>();
        let query = read(q)?;
        let key = read(k)?;
        let value = read(v)?;
        let decay = read(decay)?;
        let beta = read(beta)?;
        let state = initial.map(read).transpose()?;
        let state_width = keys.checked_mul(values).ok_or_else(|| {
            candle_core::Error::Msg("gated-delta state geometry overflows".into())
        })?;
        let state_length = (batch * heads).checked_mul(state_width).ok_or_else(|| {
            candle_core::Error::Msg("gated-delta state geometry overflows".into())
        })?;
        let outputs: Vec<_> = (0..batch * heads)
            .into_par_iter()
            .map(|head| {
                let mut state_rows = vec![0.; state_width];
                if let Some(state) = &state {
                    let previous = &state[head * state_width..][..state_width];
                    for j in 0..values {
                        for i in 0..keys {
                            state_rows[j * keys + i] = previous[i * values + j];
                        }
                    }
                }
                let mut output = vec![0.; sequence * values];
                let mut factors = vec![0.; keys];
                for token in 0..sequence {
                    let group = (head / heads * sequence + token) * heads + head % heads;
                    let query = &query[group * keys..][..keys];
                    let key = &key[group * keys..][..keys];
                    let value = &value[group * values..][..values];
                    let output = &mut output[token * values..][..values];
                    if vector_decay {
                        for (factor, &x) in factors.iter_mut().zip(&decay[group * keys..][..keys]) {
                            *factor = x.exp();
                        }
                    } else {
                        factors.fill(decay[group].exp());
                    }
                    for ((row, &value), out) in
                        state_rows.chunks_exact_mut(keys).zip(value).zip(output)
                    {
                        for (state, &factor) in row.iter_mut().zip(&factors) {
                            *state *= factor;
                        }
                        let delta = (value
                            - eredu_nn::arithmetic::ordered_f32::unpacked_dot(row, key))
                            * beta[group];
                        for (state, &key) in row.iter_mut().zip(key) {
                            *state = key.mul_add(delta, *state);
                        }
                        *out = eredu_nn::arithmetic::ordered_f32::unpacked_dot(row, query);
                    }
                }
                (state_rows, output)
            })
            .collect();
        let mut state = vec![0.; state_length];
        let mut output = vec![0.; v.elem_count()];
        for (head, (rows, tokens)) in outputs.into_iter().enumerate() {
            for j in 0..values {
                for i in 0..keys {
                    state[head * state_width + i * values + j] = rows[j * keys + i];
                }
            }
            for token in 0..sequence {
                let group = (head / heads * sequence + token) * heads + head % heads;
                output[group * values..][..values]
                    .copy_from_slice(&tokens[token * values..][..values]);
            }
        }
        Ok((
            NativeTensor::from_vec(state, state_shape, q.device())?,
            NativeTensor::from_vec(output, v.shape(), q.device())?.to_dtype(q.dtype())?,
        ))
    })();
    let (state, output) = result.map_err(Error::backend_source)?;
    Ok(GatedDeltaScanOutput {
        state: CandleTensor::wrap(Ok(state))?,
        output: CandleTensor::wrap(Ok(output))?,
    })
}

pub(crate) fn selective_state_space_scan(
    input: eredu_nn::SelectiveStateSpaceScanInput<'_, CandleTensor>,
    context: &crate::CandleContext,
) -> Result<eredu_nn::SelectiveStateSpaceScanOutput<CandleTensor>, Error> {
    use eredu_nn::{SelectiveStateSpaceScanOutput, Tensor, TensorElementType};

    if input.chunk_size == 0 {
        return Err(Error::backend(
            "selective state-space scan requires a nonzero chunk size",
        ));
    }
    let element = input
        .values
        .element_type()
        .filter(|element| {
            matches!(
                element,
                TensorElementType::F16 | TensorElementType::Bf16 | TensorElementType::F32
            )
        })
        .ok_or_else(|| {
            Error::backend("selective state-space scan requires floating activations")
        })?;
    input.memory_invocation(element)?;
    let [batch, sequence, heads, head_width] = *input.values.shape() else {
        return Err(Error::backend(
            "selective state-space scan requires rank-four activations",
        ));
    };
    let state_width = input.input_state.dim(3);
    let read = |value: &CandleTensor| {
        context
            .validate_tensor_device(value)
            .map_err(Error::backend_source)?;
        value.to_f32_vec(context)
    };
    let initial = input.initial_state.map(read).transpose()?;
    let (state, output) = eredu_nn::reference_selective_state_space_scan(
        batch as usize,
        sequence as usize,
        heads as usize,
        head_width as usize,
        state_width as usize,
        &read(input.values)?,
        &read(input.input_state)?,
        &read(input.output_state)?,
        &read(input.time_step)?,
        &read(input.time_step_bias)?,
        &read(input.transition_log)?,
        &read(input.skip)?,
        input.time_step_floor,
        initial.as_deref(),
    )?;
    Ok(SelectiveStateSpaceScanOutput {
        state: CandleTensor::from_f32_slice(
            &state,
            &[batch, heads, head_width, state_width],
            context,
        )?,
        output: CandleTensor::from_f32_slice(&output, input.values.shape(), context)?
            .cast_float(element, context)?,
    })
}
