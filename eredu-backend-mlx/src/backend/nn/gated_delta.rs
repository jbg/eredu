//! Architecture-neutral gated-delta recurrent scans.
//!
//! The state layout is `[batch, heads, key_dim, value_dim]`. Queries, keys,
//! values, log decays, and beta values are sequence-major:
//! `[batch, sequence, heads, ...]`. Both scalar log decay
//! `[batch, sequence, heads]` and per-key-channel log decay
//! `[batch, sequence, heads, key_dim]` are supported.

#[cfg(not(feature = "cuda"))]
use std::cell::RefCell;

#[cfg(not(feature = "cuda"))]
use safemlx::fast::{CustomKernelConfig, RecurrentScanKernel, StatefulMetalKernel};
#[cfg(not(feature = "cuda"))]
use safemlx::DeviceType;
use safemlx::{
    error::Exception,
    ops::{
        concatenate_axis, exp,
        indexing::{NewAxis, TryIndexOp},
        sum_axis, zeros,
    },
    Array, Dtype, Stream,
};

#[cfg(not(feature = "cuda"))]
thread_local! {
    static SCALAR_KERNELS: RefCell<Option<RecurrentScanKernel>> = const { RefCell::new(None) };
    static VECTOR_KERNELS: RefCell<Option<RecurrentScanKernel>> = const { RefCell::new(None) };
}

#[cfg(not(feature = "cuda"))]
const PREFILL_SHORT_SCAN_TOKENS: i32 = 64;
#[cfg(not(feature = "cuda"))]
const PREFILL_MEDIUM_SCAN_TOKENS: i32 = 16;
#[cfg(not(feature = "cuda"))]
const PREFILL_LONG_SCAN_TOKENS: i32 = 32;

#[cfg(not(feature = "cuda"))]
fn metal_kernels(vector_decay: bool) -> Result<RecurrentScanKernel, Exception> {
    let decay_index = if vector_decay {
        "float gate = metal::exp(g[group * KD + kd]);"
    } else {
        "float gate = metal::exp(g[group]);"
    };
    let prefill_decay_index = if vector_decay {
        "float gate = metal::exp(g[gh_idx * KD + kd]);"
    } else {
        "float gate = metal::exp(g[gh_idx]);"
    };
    let decode_source = format!(
        concat!(
            "uint elem = thread_position_in_grid.x;",
            "uint vd = elem % VD;",
            "uint group = elem / VD;",
            "uint state_base = group * KD * VD;",
            "uint vec_base = group * KD;",
            "uint value_base = group * VD;",
            "float kv_mem = 0.0f;",
            "for (uint kd = 0; kd < KD; ++kd) {{",
            "  uint state_idx = state_base + kd * VD + vd;",
            "  {decay_index}",
            "  kv_mem += float(state[state_idx]) * gate * float(key[vec_base + kd]);",
            "}}",
            "float delta = (float(value[value_base + vd]) - kv_mem) * float(beta[group]);",
            "float acc = 0.0f;",
            "for (uint kd = 0; kd < KD; ++kd) {{",
            "  uint state_idx = state_base + kd * VD + vd;",
            "  {decay_index}",
            "  float updated = float(state[state_idx]) * gate + float(key[vec_base + kd]) * delta;",
            "  state_out[state_idx] = updated;",
            "  acc += updated * float(query[vec_base + kd]);",
            "}}",
            "out[value_base + vd] = acc;"
        ),
        decay_index = decay_index
    );
    let prefill_source = format!(
        concat!(
            "uint elem = thread_position_in_grid.x;",
            "uint vd = elem % VD;",
            "uint group = elem / VD;",
            "uint h = group % H;",
            "uint b = group / H;",
            "uint state_base = group * KD * VD;",
            "for (uint t = 0; t < L; ++t) {{",
            "  uint gh_idx = (b * L + t) * H + h;",
            "  uint vec_base = gh_idx * KD;",
            "  uint value_base = gh_idx * VD;",
            "  float kv_mem = 0.0f;",
            "  for (uint kd = 0; kd < KD; ++kd) {{",
            "    uint state_idx = state_base + kd * VD + vd;",
            "    float prev = (t == 0) ? float(state[state_idx]) : float(state_out[state_idx]);",
            "    {prefill_decay_index}",
            "    kv_mem += prev * gate * float(key[vec_base + kd]);",
            "  }}",
            "  float delta = (float(value[value_base + vd]) - kv_mem) * float(beta[gh_idx]);",
            "  float acc = 0.0f;",
            "  for (uint kd = 0; kd < KD; ++kd) {{",
            "    uint state_idx = state_base + kd * VD + vd;",
            "    float prev = (t == 0) ? float(state[state_idx]) : float(state_out[state_idx]);",
            "    {prefill_decay_index}",
            "    float updated = prev * gate + float(key[vec_base + kd]) * delta;",
            "    state_out[state_idx] = updated;",
            "    acc += updated * float(query[vec_base + kd]);",
            "  }}",
            "  out[value_base + vd] = acc;",
            "}}"
        ),
        prefill_decay_index = prefill_decay_index
    );
    let suffix = if vector_decay { "vector" } else { "scalar" };
    Ok(RecurrentScanKernel::new(
        StatefulMetalKernel::new(
            format!("gated_delta_decode_{suffix}"),
            ["state", "query", "key", "value", "g", "beta"],
            ["out", "state_out"],
            &decode_source,
            "",
            true,
            false,
        )?,
        StatefulMetalKernel::new(
            format!("gated_delta_prefill_{suffix}"),
            ["state", "query", "key", "value", "g", "beta"],
            ["out", "state_out"],
            &prefill_source,
            "",
            true,
            false,
        )?,
    ))
}

fn recurrent_step(
    state: &Array,
    query: &Array,
    key: &Array,
    value: &Array,
    log_decay: &Array,
    beta: &Array,
    stream: &Stream,
) -> Result<(Array, Array), Exception> {
    let decay = match log_decay.ndim() {
        2 => exp(log_decay, stream)?.try_index_device((.., .., NewAxis, NewAxis), stream)?,
        3 => exp(log_decay, stream)?.try_index_device((.., .., .., NewAxis), stream)?,
        rank => {
            return Err(Exception::custom(format!(
                "gated-delta step expects rank-2 or rank-3 log decay, got rank {rank}"
            )))
        }
    };
    let state = state.as_dtype(Dtype::Float32, stream)?;
    let query = query.as_dtype(Dtype::Float32, stream)?;
    let key = key.as_dtype(Dtype::Float32, stream)?;
    let value = value.as_dtype(Dtype::Float32, stream)?;
    let beta = beta.as_dtype(Dtype::Float32, stream)?;
    let gated_state = state.multiply(decay.as_dtype(Dtype::Float32, stream)?, stream)?;
    let key_column = key.try_index_device((.., .., .., NewAxis), stream)?;
    let memory = sum_axis(
        gated_state.multiply(&key_column, stream)?,
        -2,
        false,
        stream,
    )?;
    let delta = value
        .subtract(memory, stream)?
        .multiply(beta.try_index_device((.., .., NewAxis), stream)?, stream)?
        .try_index_device((.., .., NewAxis, ..), stream)?;
    let state = gated_state.add(key_column.multiply(delta, stream)?, stream)?;
    let output = sum_axis(
        state.multiply(
            query.try_index_device((.., .., .., NewAxis), stream)?,
            stream,
        )?,
        -2,
        false,
        stream,
    )?;
    Ok((state, output))
}

#[cfg(not(feature = "cuda"))]
#[allow(clippy::too_many_arguments)]
fn metal_scan_chunk(
    state: &Array,
    query: &Array,
    key: &Array,
    value: &Array,
    log_decay: &Array,
    beta: &Array,
    decode: bool,
    stream: &Stream,
) -> Result<(Array, Array), Exception> {
    let shape = query.shape();
    let batch = shape[0];
    let sequence = shape[1];
    let heads = shape[2];
    let key_dim = shape[3];
    let value_dim = value.dim(-1);
    let vector_decay = log_decay.ndim() == 4;
    let state = state.as_dtype(Dtype::Float32, stream)?;
    let query = query.as_dtype(Dtype::Float32, stream)?;
    let key = key.as_dtype(Dtype::Float32, stream)?;
    let value = value.as_dtype(Dtype::Float32, stream)?;
    let log_decay = log_decay.as_dtype(Dtype::Float32, stream)?;
    let beta = beta.as_dtype(Dtype::Float32, stream)?;
    let run = |cell: &RefCell<Option<RecurrentScanKernel>>| -> Result<_, Exception> {
        if cell.borrow().is_none() {
            *cell.borrow_mut() = Some(metal_kernels(vector_decay)?);
        }
        let config = CustomKernelConfig::new()
            .with_template_arg_int("KD", key_dim)
            .with_template_arg_int("VD", value_dim)
            .with_grid([batch * heads * value_dim, 1, 1])
            .with_thread_group([256, 1, 1])
            .with_output_arg([batch, sequence, heads, value_dim], Dtype::Float32)
            .with_output_arg([batch, heads, key_dim, value_dim], Dtype::Float32);
        let kernel = cell.borrow();
        let kernel = kernel.as_ref().expect("gated-delta kernels initialized");
        if decode {
            kernel.decode_device(
                [&state, &query, &key, &value, &log_decay, &beta],
                &config,
                stream,
            )
        } else {
            let config = config
                .with_template_arg_int("L", sequence)
                .with_template_arg_int("H", heads);
            kernel.prefill_device(
                [&state, &query, &key, &value, &log_decay, &beta],
                &config,
                stream,
            )
        }
    };
    let output = if vector_decay {
        VECTOR_KERNELS.with(run)?
    } else {
        SCALAR_KERNELS.with(run)?
    };
    let (output, state) = output.into_tuple();
    Ok((state, output))
}

#[cfg(not(feature = "cuda"))]
fn metal_scan(
    mut state: Array,
    query: &Array,
    key: &Array,
    value: &Array,
    log_decay: &Array,
    beta: &Array,
    stream: &Stream,
) -> Result<(Array, Array), Exception> {
    let length = query.dim(1);
    if length == 1 {
        return metal_scan_chunk(&state, query, key, value, log_decay, beta, true, stream);
    }
    let chunk_tokens = if length <= PREFILL_SHORT_SCAN_TOKENS {
        PREFILL_SHORT_SCAN_TOKENS
    } else if length <= 256 {
        PREFILL_MEDIUM_SCAN_TOKENS
    } else {
        PREFILL_LONG_SCAN_TOKENS
    };
    let mut outputs = Vec::with_capacity(((length + chunk_tokens - 1) / chunk_tokens) as usize);
    let mut start = 0;
    while start < length {
        let end = (start + chunk_tokens).min(length);
        let query = query.try_index_device((.., start..end, .., ..), stream)?;
        let key = key.try_index_device((.., start..end, .., ..), stream)?;
        let value = value.try_index_device((.., start..end, .., ..), stream)?;
        let log_decay = if log_decay.ndim() == 4 {
            log_decay.try_index_device((.., start..end, .., ..), stream)?
        } else {
            log_decay.try_index_device((.., start..end, ..), stream)?
        };
        let beta = beta.try_index_device((.., start..end, ..), stream)?;
        let (next, output) = metal_scan_chunk(
            &state, &query, &key, &value, &log_decay, &beta, false, stream,
        )?;
        state = next;
        outputs.push(output);
        start = end;
    }
    Ok((state, concatenate_axis(&outputs, 1, stream)?))
}

/// Runs a gated-delta scan and returns `(final_state, sequence_output)`.
///
/// State is always accumulated in F32. The returned output uses the query
/// dtype, matching the surrounding model activations.
pub fn gated_delta_scan(
    query: &Array,
    key: &Array,
    value: &Array,
    log_decay: &Array,
    beta: &Array,
    initial_state: Option<Array>,
    stream: &Stream,
) -> Result<(Array, Array), Exception> {
    if query.ndim() != 4 || key.ndim() != 4 || value.ndim() != 4 {
        return Err(Exception::custom(
            "gated-delta query, key, and value tensors must be rank 4",
        ));
    }
    if query.shape()[..3] != key.shape()[..3] || query.shape()[..3] != value.shape()[..3] {
        return Err(Exception::custom(
            "gated-delta query, key, and value batch/sequence/head dimensions must match",
        ));
    }
    let shape = query.shape();
    let batch = shape[0];
    let sequence = shape[1];
    let heads = shape[2];
    let key_dim = shape[3];
    let value_dim = value.dim(-1);
    let expected_decay_rank = if log_decay.ndim() == 3 { 3 } else { 4 };
    if !matches!(log_decay.ndim(), 3 | 4)
        || log_decay.shape()[..3] != shape[..3]
        || (expected_decay_rank == 4 && log_decay.dim(-1) != key_dim)
    {
        return Err(Exception::custom(format!(
            "gated-delta log decay shape {:?} is incompatible with query shape {:?}",
            log_decay.shape(),
            query.shape()
        )));
    }
    if beta.shape() != &shape[..3] {
        return Err(Exception::custom(format!(
            "gated-delta beta shape {:?} must equal {:?}",
            beta.shape(),
            &shape[..3]
        )));
    }

    let mut state =
        initial_state.unwrap_or(zeros::<f32>(&[batch, heads, key_dim, value_dim], stream)?);
    if state.shape() != [batch, heads, key_dim, value_dim] {
        return Err(Exception::custom(format!(
            "gated-delta state shape {:?} must equal [{batch}, {heads}, {key_dim}, {value_dim}]",
            state.shape()
        )));
    }

    #[cfg(not(feature = "cuda"))]
    if stream.get_device()?.get_type()? == DeviceType::Gpu {
        let output_dtype = query.dtype();
        let (state, output) = metal_scan(state, query, key, value, log_decay, beta, stream)?;
        return Ok((state, output.as_dtype(output_dtype, stream)?));
    }

    let output_dtype = query.dtype();
    let mut outputs = Vec::with_capacity(sequence as usize);
    for token in 0..sequence {
        let q = query.try_index_device((.., token, .., ..), stream)?;
        let k = key.try_index_device((.., token, .., ..), stream)?;
        let v = value.try_index_device((.., token, .., ..), stream)?;
        let g = if log_decay.ndim() == 3 {
            log_decay.try_index_device((.., token, ..), stream)?
        } else {
            log_decay.try_index_device((.., token, .., ..), stream)?
        };
        let b = beta.try_index_device((.., token, ..), stream)?;
        let (next, output) = recurrent_step(&state, &q, &k, &v, &g, &b, stream)?;
        state = next;
        outputs.push(
            output
                .as_dtype(output_dtype, stream)?
                .try_index_device((.., NewAxis, .., ..), stream)?,
        );
    }
    Ok((state, concatenate_axis(&outputs, 1, stream)?))
}

#[cfg(test)]
mod tests {
    use safemlx::{
        ops::{concatenate_axis, indexing::TryIndexOp},
        Array, Device, DeviceType, ExecutionContext,
    };

    use super::gated_delta_scan;

    #[test]
    fn vector_decay_matches_scalar_when_channels_are_equal() {
        let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = context.stream();
        let query = Array::from_slice(&[0.5f32, -0.25, 0.1, 0.2], &[1, 2, 1, 2]);
        let key = Array::from_slice(&[0.3f32, 0.4, -0.2, 0.7], &[1, 2, 1, 2]);
        let value = Array::from_slice(&[1.0f32, -0.5, 0.25, 0.75], &[1, 2, 1, 2]);
        let beta = Array::from_slice(&[0.8f32, 0.6], &[1, 2, 1]);
        let scalar = Array::from_slice(&[-0.2f32, -0.4], &[1, 2, 1]);
        let vector = Array::from_slice(&[-0.2f32, -0.2, -0.4, -0.4], &[1, 2, 1, 2]);
        let (_, scalar_output) =
            gated_delta_scan(&query, &key, &value, &scalar, &beta, None, stream).unwrap();
        let (_, vector_output) =
            gated_delta_scan(&query, &key, &value, &vector, &beta, None, stream).unwrap();
        let scalar_output = scalar_output.evaluated().unwrap();
        let vector_output = vector_output.evaluated().unwrap();
        let scalar = scalar_output.as_slice::<f32>();
        let vector = vector_output.as_slice::<f32>();
        assert!(scalar
            .iter()
            .zip(vector)
            .all(|(left, right)| (left - right).abs() < 1e-6));
    }

    #[test]
    fn cached_chunks_match_one_scan() {
        let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = context.stream();
        let query = Array::from_slice(&[0.5f32, -0.25, 0.1, 0.2, -0.4, 0.8], &[1, 3, 1, 2]);
        let key = Array::from_slice(&[0.3f32, 0.4, -0.2, 0.7, 0.6, -0.1], &[1, 3, 1, 2]);
        let value = Array::from_slice(&[1.0f32, -0.5, 0.25, 0.75, -0.3, 0.9], &[1, 3, 1, 2]);
        let beta = Array::from_slice(&[0.8f32, 0.6, 0.4], &[1, 3, 1]);
        let decay = Array::from_slice(&[-0.2f32, -0.1, -0.4, -0.3, -0.5, -0.25], &[1, 3, 1, 2]);
        let (expected_state, expected) =
            gated_delta_scan(&query, &key, &value, &decay, &beta, None, stream).unwrap();
        let (state, first) = gated_delta_scan(
            &query.try_index_device((.., ..2, .., ..), stream).unwrap(),
            &key.try_index_device((.., ..2, .., ..), stream).unwrap(),
            &value.try_index_device((.., ..2, .., ..), stream).unwrap(),
            &decay.try_index_device((.., ..2, .., ..), stream).unwrap(),
            &beta.try_index_device((.., ..2, ..), stream).unwrap(),
            None,
            stream,
        )
        .unwrap();
        let (actual_state, second) = gated_delta_scan(
            &query.try_index_device((.., 2.., .., ..), stream).unwrap(),
            &key.try_index_device((.., 2.., .., ..), stream).unwrap(),
            &value.try_index_device((.., 2.., .., ..), stream).unwrap(),
            &decay.try_index_device((.., 2.., .., ..), stream).unwrap(),
            &beta.try_index_device((.., 2.., ..), stream).unwrap(),
            Some(state),
            stream,
        )
        .unwrap();
        let actual = concatenate_axis(&[first, second], 1, stream).unwrap();
        let expected = expected.evaluated().unwrap();
        let actual = actual.evaluated().unwrap();
        assert!(expected
            .as_slice::<f32>()
            .iter()
            .zip(actual.as_slice::<f32>())
            .all(|(left, right)| (left - right).abs() < 1e-6));
        let expected_state = expected_state.evaluated().unwrap();
        let actual_state = actual_state.evaluated().unwrap();
        assert!(expected_state
            .as_slice::<f32>()
            .iter()
            .zip(actual_state.as_slice::<f32>())
            .all(|(left, right)| (left - right).abs() < 1e-6));
    }

    #[test]
    #[ignore = "requires an MLX Metal device"]
    fn metal_vector_decay_matches_cpu() {
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let values = |stream: &safemlx::Stream| {
            let query = Array::from_slice(&[0.5f32, -0.25, 0.1, 0.2], &[1, 2, 1, 2]);
            let key = Array::from_slice(&[0.3f32, 0.4, -0.2, 0.7], &[1, 2, 1, 2]);
            let value = Array::from_slice(&[1.0f32, -0.5, 0.25, 0.75], &[1, 2, 1, 2]);
            let beta = Array::from_slice(&[0.8f32, 0.6], &[1, 2, 1]);
            let decay = Array::from_slice(&[-0.2f32, -0.1, -0.4, -0.3], &[1, 2, 1, 2]);
            gated_delta_scan(&query, &key, &value, &decay, &beta, None, stream).unwrap()
        };
        let (cpu_state, cpu_output) = values(cpu.stream());
        let (gpu_state, gpu_output) = values(gpu.stream());
        let cpu_output = cpu_output.evaluated().unwrap();
        let gpu_output = gpu_output.evaluated().unwrap();
        assert!(cpu_output
            .as_slice::<f32>()
            .iter()
            .zip(gpu_output.as_slice::<f32>())
            .all(|(left, right)| (left - right).abs() < 1e-5));
        let cpu_state = cpu_state.evaluated().unwrap();
        let gpu_state = gpu_state.evaluated().unwrap();
        assert!(cpu_state
            .as_slice::<f32>()
            .iter()
            .zip(gpu_state.as_slice::<f32>())
            .all(|(left, right)| (left - right).abs() < 1e-5));
    }
}
