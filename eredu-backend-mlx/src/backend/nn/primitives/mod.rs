//! MLX neural-network primitives used by the backend implementation.

mod embedding;
mod normalization;
mod positional_encoding;
mod quantized;

pub(crate) use crate::backend::nn::layers::{elu, gelu, gelu_approximate, log_sigmoid, softplus};
pub(crate) use embedding::Embedding;
pub(crate) use normalization::RmsNorm;
pub(crate) use positional_encoding::{
    RopeInput, RopeInputBuilder, RotaryPositionalEncoding, RotaryPositionalEncodingBuilder,
};
pub(crate) use quantized::QuantizedEmbedding;
