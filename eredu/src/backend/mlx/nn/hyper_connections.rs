//! Architecture-neutral hyper-connection residual mixing.
//!
//! Multi-stream residual blocks use a small doubly-stochastic matrix to mix
//! streams.  Keeping the numerical definition here avoids embedding a second
//! implementation in target and speculative-decoder code.

use safemlx::{
    error::Exception,
    macros::ModuleParameters,
    module::Param,
    ops::{einsum, indexing::TryIndexOp, matmul, mean_axis, rsqrt, sigmoid, softmax_axis},
    Array, Dtype, Stream,
};

/// The three tensors produced by one hyper-connection split.
#[derive(Debug, Clone)]
pub struct HyperConnectionSplit {
    /// Weights reducing the residual streams before a sublayer.
    pub pre: Array,
    /// Weights injecting the sublayer result into each residual stream.
    pub post: Array,
    /// Doubly-stochastic residual-stream mixing matrix.
    pub combination: Array,
}

/// Trainable multi-stream hyper-connection.
#[derive(Debug, Clone, ModuleParameters)]
pub struct HyperConnection {
    /// Number of residual streams.
    pub streams: i32,
    /// Hidden width of each residual stream.
    pub hidden_size: i32,
    /// Sinkhorn iterations.
    pub iterations: usize,
    /// Numerical epsilon used by RMS and Sinkhorn normalization.
    pub epsilon: f32,
    #[param]
    /// Projection producing pre/post/combination logits.
    pub function: Param<Array>,
    #[param]
    /// Additive mixing base, kept in FP32.
    pub base: Param<Array>,
    #[param]
    /// Three learned mixing scales, kept in FP32.
    pub scale: Param<Array>,
}

impl HyperConnection {
    /// Creates an unloaded hyper-connection.
    pub fn unloaded(
        streams: i32,
        hidden_size: i32,
        iterations: usize,
        epsilon: f32,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        if streams <= 0 || hidden_size <= 0 {
            return Err(Exception::custom(
                "hyper-connection streams and hidden size must be positive",
            ));
        }
        let width = (2 + streams) * streams;
        Ok(Self {
            streams,
            hidden_size,
            iterations,
            epsilon,
            function: Param::unloaded(&[width, streams * hidden_size], Dtype::Float32, stream)?,
            base: Param::unloaded(&[width], Dtype::Float32, stream)?,
            scale: Param::unloaded(&[3], Dtype::Float32, stream)?,
        })
    }

    /// Collapses residual streams and returns the coefficients needed to expand them.
    pub fn collapse(
        &mut self,
        residual: &Array,
        norm_epsilon: f32,
        stream: &Stream,
    ) -> Result<(Array, Array, Array), Exception> {
        let (collapsed, split) = self.collapse_split(residual, norm_epsilon, stream)?;
        Ok((collapsed, split.post, split.combination))
    }

    /// Collapses residual streams while retaining all typed mixing coefficients.
    pub fn collapse_split(
        &mut self,
        residual: &Array,
        norm_epsilon: f32,
        stream: &Stream,
    ) -> Result<(Array, HyperConnectionSplit), Exception> {
        if residual.ndim() != 4
            || residual.dim(2) != self.streams
            || residual.dim(3) != self.hidden_size
        {
            return Err(Exception::custom(format!(
                "hyper-connection requires [batch, tokens, {}, {}], got {:?}",
                self.streams,
                self.hidden_size,
                residual.shape()
            )));
        }
        let residual_fp32 = residual.as_dtype(Dtype::Float32, stream)?;
        let flat = residual_fp32.reshape(
            &[
                residual.dim(0),
                residual.dim(1),
                self.streams * self.hidden_size,
            ],
            stream,
        )?;
        let normalized = weightless_rms_norm(&flat, norm_epsilon, stream)?;
        let mixes = matmul(
            &normalized,
            &self.function.as_ref().transpose(stream)?,
            stream,
        )?;
        let split = split_sinkhorn(
            &mixes,
            self.scale.as_ref(),
            self.base.as_ref(),
            self.streams,
            self.iterations,
            self.epsilon,
            stream,
        )?;
        let collapsed = einsum("blh,blhd->bld", [&split.pre, &residual_fp32], stream)?
            .as_dtype(residual.dtype(), stream)?;
        Ok((collapsed, split))
    }
}

/// Injects a sublayer result and mixes the previous residual streams.
pub fn expand(
    sublayer: &Array,
    residual: &Array,
    post: &Array,
    combination: &Array,
    stream: &Stream,
) -> Result<Array, Exception> {
    let dtype = sublayer.dtype();
    let injected = post
        .try_index_device((.., .., .., safemlx::ops::indexing::NewAxis), stream)?
        .multiply(
            sublayer
                .as_dtype(Dtype::Float32, stream)?
                .try_index_device((.., .., safemlx::ops::indexing::NewAxis, ..), stream)?,
            stream,
        )?;
    let mixed = einsum(
        "blji,bljd->blid",
        [combination, &residual.as_dtype(Dtype::Float32, stream)?],
        stream,
    )?;
    injected.add(mixed, stream)?.as_dtype(dtype, stream)
}

/// Final learned collapse from multiple residual streams to one hidden state.
#[derive(Debug, Clone, ModuleParameters)]
pub struct HyperHead {
    /// Number of residual streams.
    pub streams: i32,
    /// Hidden width per stream.
    pub hidden_size: i32,
    /// RMS normalization epsilon.
    pub norm_epsilon: f32,
    /// Mixing epsilon.
    pub epsilon: f32,
    #[param]
    /// Projection producing per-stream collapse logits.
    pub function: Param<Array>,
    #[param]
    /// Per-stream additive base.
    pub base: Param<Array>,
    #[param]
    /// Learned collapse scale.
    pub scale: Param<Array>,
}

impl HyperHead {
    /// Creates an unloaded final hyper-head.
    pub fn unloaded(
        streams: i32,
        hidden_size: i32,
        norm_epsilon: f32,
        epsilon: f32,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        Ok(Self {
            streams,
            hidden_size,
            norm_epsilon,
            epsilon,
            function: Param::unloaded(&[streams, streams * hidden_size], Dtype::Float32, stream)?,
            base: Param::unloaded(&[streams], Dtype::Float32, stream)?,
            scale: Param::unloaded(&[1], Dtype::Float32, stream)?,
        })
    }

    /// Collapses `[batch, tokens, streams, hidden]` into `[batch, tokens, hidden]`.
    pub fn forward(&mut self, residual: &Array, stream: &Stream) -> Result<Array, Exception> {
        let dtype = residual.dtype();
        let fp32 = residual.as_dtype(Dtype::Float32, stream)?;
        let flat = fp32.reshape(
            &[
                residual.dim(0),
                residual.dim(1),
                self.streams * self.hidden_size,
            ],
            stream,
        )?;
        let normalized = weightless_rms_norm(&flat, self.norm_epsilon, stream)?;
        let logits = matmul(
            &normalized,
            &self.function.as_ref().transpose(stream)?,
            stream,
        )?;
        let pre = sigmoid(
            logits
                .multiply(self.scale.as_ref(), stream)?
                .add(self.base.as_ref(), stream)?,
            stream,
        )?
        .add(Array::from_f32(self.epsilon), stream)?;
        pre.try_index_device((.., .., .., safemlx::ops::indexing::NewAxis), stream)?
            .multiply(&fp32, stream)?
            .sum_axis(2, false, stream)?
            .as_dtype(dtype, stream)
    }
}

fn weightless_rms_norm(value: &Array, epsilon: f32, stream: &Stream) -> Result<Array, Exception> {
    let variance = mean_axis(&value.square(stream)?, -1, true, stream)?;
    value.multiply(
        rsqrt(variance.add(Array::from_f32(epsilon), stream)?, stream)?,
        stream,
    )
}

/// Applies FP32 hyper-connection splitting and Sinkhorn normalization.
///
/// `mixes` has shape `[..., (2 + streams) * streams]`, `scale` has shape
/// `[3]`, and `base` has the same final dimension as `mixes`.  The returned
/// tensors have shapes `[..., streams]`, `[..., streams]`, and
/// `[..., streams, streams]` respectively.
pub fn split_sinkhorn(
    mixes: &Array,
    scale: &Array,
    base: &Array,
    streams: i32,
    iterations: usize,
    epsilon: f32,
    stream: &Stream,
) -> Result<HyperConnectionSplit, Exception> {
    if streams <= 0 {
        return Err(Exception::custom(
            "hyper-connection stream count must be positive",
        ));
    }
    if iterations == 0 {
        return Err(Exception::custom(
            "hyper-connection Sinkhorn iteration count must be positive",
        ));
    }
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(Exception::custom(
            "hyper-connection epsilon must be finite and positive",
        ));
    }
    let mixed_width = (2 + streams)
        .checked_mul(streams)
        .ok_or_else(|| Exception::custom("hyper-connection width overflowed"))?;
    if mixes.ndim() == 0 || mixes.dim(-1) != mixed_width {
        return Err(Exception::custom(format!(
            "hyper-connection mixes require final dimension {mixed_width}, got {:?}",
            mixes.shape()
        )));
    }
    if scale.shape() != [3] || base.shape() != [mixed_width] {
        return Err(Exception::custom(format!(
            "hyper-connection scale/base shapes must be [3] and [{mixed_width}], got {:?} and {:?}",
            scale.shape(),
            base.shape()
        )));
    }

    let output_prefix = mixes.shape()[..mixes.ndim() - 1].to_vec();
    // Work in two dimensions so the implementation is independent of the
    // number of leading batch/token axes.  This also keeps all slicing on the
    // final feature dimension; tuple indexing would otherwise slice the
    // second axis for rank-three decoder activations.
    let mixes = mixes
        .as_dtype(Dtype::Float32, stream)?
        .reshape(&[-1, mixed_width], stream)?;
    let scale = scale.as_dtype(Dtype::Float32, stream)?;
    let base = base.as_dtype(Dtype::Float32, stream)?;

    let pre_logits = mixes.try_index_device((.., ..streams), stream)?;
    let pre_base = base.try_index_device(..streams, stream)?;
    let pre_scale = scale.try_index_device(0, stream)?;
    let pre = sigmoid(
        pre_logits
            .multiply(pre_scale, stream)?
            .add(pre_base, stream)?,
        stream,
    )?
    .add(Array::from_f32(epsilon), stream)?;

    let post_logits = mixes.try_index_device((.., streams..2 * streams), stream)?;
    let post_base = base.try_index_device(streams..2 * streams, stream)?;
    let post_scale = scale.try_index_device(1, stream)?;
    let post = sigmoid(
        post_logits
            .multiply(post_scale, stream)?
            .add(post_base, stream)?,
        stream,
    )?
    .multiply(Array::from_f32(2.0), stream)?;

    let combination_width = streams * streams;
    let combination_logits = mixes
        .try_index_device((.., 2 * streams..mixed_width), stream)?
        .multiply(scale.try_index_device(2, stream)?, stream)?
        .add(
            base.try_index_device(2 * streams..mixed_width, stream)?,
            stream,
        )?;
    let combination_shape = vec![-1, streams, streams];
    debug_assert_eq!(combination_width, mixed_width - 2 * streams);
    let mut combination = softmax_axis(
        combination_logits.reshape(&combination_shape, stream)?,
        -1,
        true,
        stream,
    )?
    .add(Array::from_f32(epsilon), stream)?;
    combination = normalize_axis(&combination, -2, epsilon, stream)?;
    for _ in 1..iterations {
        combination = normalize_axis(&combination, -1, epsilon, stream)?;
        combination = normalize_axis(&combination, -2, epsilon, stream)?;
    }

    let mut vector_shape = output_prefix.clone();
    vector_shape.push(streams);
    let mut matrix_shape = output_prefix;
    matrix_shape.extend([streams, streams]);
    Ok(HyperConnectionSplit {
        pre: pre.reshape(&vector_shape, stream)?,
        post: post.reshape(&vector_shape, stream)?,
        combination: combination.reshape(&matrix_shape, stream)?,
    })
}

fn normalize_axis(
    value: &Array,
    axis: i32,
    epsilon: f32,
    stream: &Stream,
) -> Result<Array, Exception> {
    value.divide(
        value
            .sum_axis(axis, true, stream)?
            .add(Array::from_f32(epsilon), stream)?,
        stream,
    )
}

#[cfg(test)]
mod tests {
    use super::split_sinkhorn;
    use safemlx::{Array, Device, DeviceType, ExecutionContext};

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn preserves_arbitrary_leading_dimensions_and_normalizes_columns() {
        let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = context.stream();
        let mixes = Array::from_slice(&[0.0f32; 48], &[2, 3, 8]);
        let scale = Array::from_slice(&[1.0f32, 1.0, 1.0], &[3]);
        let base = Array::from_slice(&[0.0f32; 8], &[8]);
        let split = split_sinkhorn(&mixes, &scale, &base, 2, 20, 1e-6, stream).unwrap();

        assert_eq!(split.pre.shape(), [2, 3, 2]);
        assert_eq!(split.post.shape(), [2, 3, 2]);
        assert_eq!(split.combination.shape(), [2, 3, 2, 2]);
        let columns = split.combination.sum_axis(-2, false, stream).unwrap();
        assert!(columns
            .all_close(
                Array::ones::<f32>(&[2, 3, 2], stream).unwrap(),
                1e-4,
                1e-4,
                None,
                stream,
            )
            .unwrap()
            .item::<bool>(stream));
    }
}
