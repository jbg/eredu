//! Architecture-neutral hyper-connection residual mixing.
//!
//! Multi-stream residual blocks use a small doubly-stochastic matrix to mix
//! streams.  Keeping the numerical definition here avoids embedding a second
//! implementation in target and speculative-decoder code.

use eredu_backend_mlx_macros::PhysicalParameters;
use safemlx::{Array, Dtype, Stream, error::Exception};
mod native;
pub(crate) mod worker;
pub(crate) use native::control_bytes;

use crate::module::PhysicalParam;

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
#[derive(Debug, Clone, PhysicalParameters)]
#[module(root = crate)]
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
    pub function: PhysicalParam<Array>,
    #[param]
    /// Additive mixing base, kept in FP32.
    pub base: PhysicalParam<Array>,
    #[param]
    /// Three learned mixing scales, kept in FP32.
    pub scale: PhysicalParam<Array>,
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
        let width = validate_split(streams, iterations, epsilon)?;
        let columns = streams.checked_mul(hidden_size).ok_or_else(|| {
            native::invalid("hyper projection width overflow", || {
                "hyper projection width overflow".into()
            })
        })?;
        Ok(Self {
            streams,
            hidden_size,
            iterations,
            epsilon,
            function: PhysicalParam::unloaded(&[width, columns], Dtype::Float32, stream)?,
            base: PhysicalParam::unloaded(&[width], Dtype::Float32, stream)?,
            scale: PhysicalParam::unloaded(&[3], Dtype::Float32, stream)?,
        })
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
            return Err(native::invalid("hyper residual geometry", || {
                format!(
                    "hyper-connection requires [batch, tokens, {}, {}], got {:?}",
                    self.streams,
                    self.hidden_size,
                    residual.shape()
                )
            }));
        }
        let dimensions = native::dimensions(residual)?;
        let width = validate_split(self.streams, self.iterations, self.epsilon)?;
        native::parameters(
            dimensions,
            width,
            3,
            self.function.as_ref(),
            self.base.as_ref(),
            self.scale.as_ref(),
        )?;
        native::profile(
            &[
                residual,
                self.function.as_ref(),
                self.base.as_ref(),
                self.scale.as_ref(),
            ],
            2,
            stream,
        )?;
        let (collapsed, split) = worker::collapse(
            &mut native::Native(stream),
            residual,
            self.function.as_ref(),
            self.base.as_ref(),
            self.scale.as_ref(),
            dimensions,
            self.iterations,
            self.epsilon,
            norm_epsilon,
        )?;
        Ok((
            collapsed,
            HyperConnectionSplit {
                pre: split.pre,
                post: split.post,
                combination: split.combination,
            },
        ))
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
    let dimensions = native::dimensions(residual)?;
    let [b, t, s, h] = dimensions;
    if sublayer.shape() != [b, t, h]
        || post.shape() != [b, t, s]
        || combination.shape() != [b, t, s, s]
    {
        return Err(native::invalid("hyper expansion geometry", || {
            "hyper expansion differs from residual geometry".into()
        }));
    }
    native::profile(&[sublayer, residual, post, combination], 2, stream)?;
    worker::expand(
        &mut native::Native(stream),
        sublayer,
        residual,
        post,
        combination,
        dimensions,
    )
}

/// Final learned collapse from multiple residual streams to one hidden state.
#[derive(Debug, Clone, PhysicalParameters)]
#[module(root = crate)]
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
    pub function: PhysicalParam<Array>,
    #[param]
    /// Per-stream additive base.
    pub base: PhysicalParam<Array>,
    #[param]
    /// Learned collapse scale.
    pub scale: PhysicalParam<Array>,
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
        let columns = streams
            .checked_mul(hidden_size)
            .filter(|_| streams > 0 && hidden_size > 0)
            .ok_or_else(|| {
                native::invalid("hyper head projection geometry", || {
                    "hyper head projection geometry overflow".into()
                })
            })?;
        Ok(Self {
            streams,
            hidden_size,
            norm_epsilon,
            epsilon,
            function: PhysicalParam::unloaded(&[streams, columns], Dtype::Float32, stream)?,
            base: PhysicalParam::unloaded(&[streams], Dtype::Float32, stream)?,
            scale: PhysicalParam::unloaded(&[1], Dtype::Float32, stream)?,
        })
    }

    /// Collapses `[batch, tokens, streams, hidden]` into `[batch, tokens, hidden]`.
    pub fn forward(&mut self, residual: &Array, stream: &Stream) -> Result<Array, Exception> {
        self.forward_with_coefficients_observer(residual, stream, None)
    }

    /// Borrows the coefficients already consumed by the ordinary final sum.
    pub fn forward_with_coefficients_observer(
        &mut self,
        residual: &Array,
        stream: &Stream,
        observer: Option<&mut dyn FnMut(&Array) -> Result<(), Exception>>,
    ) -> Result<Array, Exception> {
        let dimensions = native::dimensions(residual)?;
        if dimensions[2] != self.streams || dimensions[3] != self.hidden_size {
            return Err(native::invalid("hyper head geometry", || {
                "hyper head differs from residual geometry".into()
            }));
        }
        native::profile(
            &[
                residual,
                self.function.as_ref(),
                self.base.as_ref(),
                self.scale.as_ref(),
            ],
            2,
            stream,
        )?;
        native::parameters(
            dimensions,
            self.streams,
            1,
            self.function.as_ref(),
            self.base.as_ref(),
            self.scale.as_ref(),
        )?;
        let mut worker = native::Native(stream);
        let (fp32, pre) = worker::head_coefficients(
            &mut worker,
            residual,
            self.function.as_ref(),
            self.base.as_ref(),
            self.scale.as_ref(),
            dimensions,
            self.norm_epsilon,
            self.epsilon,
        )?;
        if let Some(observer) = observer {
            observer(&pre)?;
        }
        // Retain this exact widened source through the borrowed callback and
        // consume it in the ordinary final sum; no repeated source conversion.
        worker::head_sum(&mut worker, &fp32, &pre, residual, dimensions)
    }
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
    let mixed_width = validate_split(streams, iterations, epsilon)?;
    if mixes.ndim() == 0 || mixes.dim(-1) != mixed_width {
        return Err(native::invalid("hyper mixes width", || {
            format!(
                "hyper-connection mixes require final dimension {mixed_width}, got {:?}",
                mixes.shape()
            )
        }));
    }
    if scale.shape() != [3] || base.shape() != [mixed_width] {
        return Err(native::invalid("hyper scale/base geometry", || {
            format!(
                "hyper-connection scale/base shapes must be [3] and [{mixed_width}], got {:?} and {:?}",
                scale.shape(),
                base.shape()
            )
        }));
    }
    native::profile(&[mixes, scale, base], mixes.ndim() - 1, stream)?;
    let split = worker::split(
        &mut native::Native(stream),
        mixes,
        scale,
        base,
        streams,
        iterations,
        epsilon,
    )?;
    Ok(HyperConnectionSplit {
        pre: split.pre,
        post: split.post,
        combination: split.combination,
    })
}

fn validate_split(streams: i32, iterations: usize, epsilon: f32) -> Result<i32, Exception> {
    for (valid, message) in [
        (
            streams > 0,
            "hyper-connection stream count must be positive",
        ),
        (
            iterations > 0,
            "hyper-connection Sinkhorn iteration count must be positive",
        ),
        (
            epsilon.is_finite() && epsilon > 0.0,
            "hyper-connection epsilon must be finite and positive",
        ),
    ] {
        if !valid {
            return Err(native::invalid(message, || message.into()));
        }
    }
    streams
        .checked_add(2)
        .and_then(|n| n.checked_mul(streams))
        .ok_or_else(|| {
            native::invalid("hyper-connection width overflowed", || {
                "hyper-connection width overflowed".into()
            })
        })
}

#[cfg(test)]
mod tests;
