//! Backend-neutral neural audio codec architectures.
//!
//! This crate keeps codec implementations optional and separate from
//! `eredu`. Realtime language models can operate on discrete codec tokens,
//! while applications that need audio encode/decode can depend on this crate.

#![warn(missing_docs)]

/// Mimi neural audio tokenizer support.
pub mod mimi;

use eredu_nn::Tensor;

/// Common interface for neural audio tokenizers.
pub trait AudioTokenizer {
    /// Backend-native tensor handle.
    type Tensor: Tensor;

    /// Codec configuration.
    fn config(&self) -> AudioTokenizerConfig;

    /// Encodes mono PCM shaped `[batch, channels, samples]` into codec tokens.
    fn encode(
        &mut self,
        pcm: &Self::Tensor,
        context: &<Self::Tensor as Tensor>::Context,
    ) -> Result<Self::Tensor, Error>;

    /// Decodes codec tokens shaped `[batch, codebooks, frames]` into PCM.
    fn decode(
        &mut self,
        codes: &Self::Tensor,
        context: &<Self::Tensor as Tensor>::Context,
    ) -> Result<Self::Tensor, Error>;
}

/// Static metadata for pairing an audio tokenizer with a realtime model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AudioTokenizerConfig {
    /// Audio sample rate in Hz.
    pub sample_rate: f64,
    /// Codec frame rate in Hz.
    pub frame_rate: f64,
    /// Number of audio channels supported by the codec.
    pub channels: i32,
    /// Number of active codebooks used for encode/decode.
    pub codebooks: i32,
    /// Codebook cardinality.
    pub cardinality: i32,
}

/// Errors returned by codec construction and tokenization operations.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Invalid input or checkpoint shape.
    #[error("{0}")]
    InvalidShape(String),

    /// Underlying neural-compute backend error.
    #[error(transparent)]
    Compute(#[from] eredu_nn::Error),
}
