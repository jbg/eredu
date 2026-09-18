//! Architecture-neutral gated-delta recurrent scans.
//!
//! The state layout is `[batch, heads, key_dim, value_dim]`. Queries, keys,
//! values, log decays, and beta values are sequence-major:
//! `[batch, sequence, heads, ...]`. Both scalar log decay
//! `[batch, sequence, heads]` and per-key-channel log decay
//! `[batch, sequence, heads, key_dim]` are supported.

#[cfg(not(feature = "cuda"))]
use safemlx::DeviceType;
use safemlx::{
    Array, Dtype, Stream,
    error::Exception,
    ops::{
        concatenate_axis, exp,
        indexing::{NewAxis, TryIndexOp},
        sum_axis, zeros,
    },
};

#[cfg(not(feature = "cuda"))]
const PREFILL_SHORT_SCAN_TOKENS: i32 = 64;
#[cfg(not(feature = "cuda"))]
const PREFILL_MEDIUM_SCAN_TOKENS: i32 = 16;
#[cfg(not(feature = "cuda"))]
const PREFILL_LONG_SCAN_TOKENS: i32 = 32;

#[cfg(not(feature = "cuda"))]
pub(super) fn metal_scan_chunk_tokens(length: i32) -> i32 {
    if length <= PREFILL_SHORT_SCAN_TOKENS {
        PREFILL_SHORT_SCAN_TOKENS
    } else if length <= 256 {
        PREFILL_MEDIUM_SCAN_TOKENS
    } else {
        PREFILL_LONG_SCAN_TOKENS
    }
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
            )));
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
    use crate::backend::managed_memory::recurrent_kernel::{self, ScanKernel};
    use safemlx::fast::BorrowedKernelOutput;
    let kind = ScanKernel::select(decode, vector_decay);
    // Authenticate the prepared source before constructing any native cast.
    recurrent_kernel::validate_call(kind)?;
    let grid_width = batch
        .checked_mul(heads)
        .and_then(|n| n.checked_mul(value_dim))
        .ok_or_else(|| match safemlx::OriginalScopeObserver::try_current() {
            Ok(Some(observer)) => observer.capacity_error(),
            Err(error) => error,
            Ok(None) => Exception::custom("gated-delta dispatch dimensions overflow"),
        })?;
    let state = state.as_dtype(Dtype::Float32, stream)?;
    let query = query.as_dtype(Dtype::Float32, stream)?;
    let key = key.as_dtype(Dtype::Float32, stream)?;
    let value = value.as_dtype(Dtype::Float32, stream)?;
    let log_decay = log_decay.as_dtype(Dtype::Float32, stream)?;
    let beta = beta.as_dtype(Dtype::Float32, stream)?;
    let sequence_shape = [batch, sequence, heads, value_dim];
    let state_shape = [batch, heads, key_dim, value_dim];
    let inputs = [&state, &query, &key, &value, &log_decay, &beta];
    let outputs = [
        BorrowedKernelOutput {
            shape: &sequence_shape,
            dtype: Dtype::Float32,
        },
        BorrowedKernelOutput {
            shape: &state_shape,
            dtype: Dtype::Float32,
        },
    ];
    let grid = [grid_width, 1, 1];
    let [output, state] = recurrent_kernel::apply(kind, inputs, outputs, grid, stream)?;
    Ok((state, output))
}

// Every scan slice retains all axes and uses checked nonnegative bounds. The
// fixed Slice bridge avoids the general indexing worker's temporary index Vec.
#[cfg(not(feature = "cuda"))]
fn metal_sequence_slice(
    input: &Array,
    start: i32,
    end: i32,
    stream: &Stream,
) -> Result<Array, Exception> {
    match input.shape() {
        &[batch, _, heads, width] => input.try_slice(
            &[0, start, 0, 0],
            &[batch, end, heads, width],
            &[1; 4],
            stream,
        ),
        &[batch, _, heads] => {
            input.try_slice(&[0, start, 0], &[batch, end, heads], &[1; 3], stream)
        }
        _ => Err(Exception::custom(
            "gated-delta sequence slice requires rank 3 or 4",
        )),
    }
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
    let chunk_tokens = metal_scan_chunk_tokens(length);
    let mut outputs = crate::backend::nn::tensor::GroupedChunkOutputs::prepare(
        (length as usize).div_ceil(chunk_tokens as usize),
    )?;
    let mut start = 0;
    while start < length {
        let end = start + (length - start).min(chunk_tokens);
        let query = metal_sequence_slice(query, start, end, stream)?;
        let key = metal_sequence_slice(key, start, end, stream)?;
        let value = metal_sequence_slice(value, start, end, stream)?;
        let log_decay = metal_sequence_slice(log_decay, start, end, stream)?;
        let beta = metal_sequence_slice(beta, start, end, stream)?;
        let (next, output) = metal_scan_chunk(
            &state, &query, &key, &value, &log_decay, &beta, false, stream,
        )?;
        state = next;
        outputs.push(output)?;
        start = end;
    }
    Ok((state, concatenate_axis(outputs.as_slice(), 1, stream)?))
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
    if key.dim(-1) != key_dim {
        return Err(Exception::custom(
            "gated-delta query and key widths must match",
        ));
    }
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

    let mut state = match initial_state {
        Some(state) => state,
        None => zeros::<f32>(&[batch, heads, key_dim, value_dim], stream)?,
    };
    if state.shape() != [batch, heads, key_dim, value_dim] {
        return Err(Exception::custom(format!(
            "gated-delta state shape {:?} must equal [{batch}, {heads}, {key_dim}, {value_dim}]",
            state.shape()
        )));
    }

    #[cfg(not(feature = "cuda"))]
    if stream.device_type()? == DeviceType::Gpu {
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
mod tests;
