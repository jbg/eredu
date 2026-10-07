//! Backend-neutral neural audio codec architectures.
//!
//! This crate keeps codec implementations optional and separate from
//! `eredu`. Realtime language models can operate on discrete codec tokens,
//! while applications that need audio encode/decode can depend on this crate.

#![warn(missing_docs)]

/// Mimi neural audio tokenizer support.
pub mod mimi;

/// EAR residual-code waveform decoding and transactional streaming composition.
pub mod ear;

/// Coordinated waveform composition over stage-owned decoder carry.
pub mod ear_pipeline;

/// Fixed-mean probabilistic residual vector quantization.
pub mod prvq;
pub mod prvq_layered;

/// Cold exact bindings for independently leased PRVQ codebooks.
pub mod prvq_preparation;

/// EAR codec causal ConvNeXt latent-to-spectrogram decoder.
pub mod ear_decoder;

/// Cold admission of exact decoder pipeline communication and sources.
pub mod ear_decoder_distribution;
/// EAR neural decoder execution over shared resident and bounded parameter providers.
pub mod ear_decoder_layered;
/// Streaming decoder pipeline ownership, state and changing-cadence transport.
pub mod ear_decoder_partition;
/// Cold exact parameter bindings for independently leased EAR decoder units.
pub mod ear_decoder_preparation;

/// EAR codec finite spectrogram-to-latent encoder.
pub mod ear_encoder;
/// Cold admission of exact finite encoder pipeline communication and sources.
pub mod ear_encoder_distribution;
pub mod ear_encoder_layered;
/// Exact finite encoder pipeline ownership and changing-cadence transport.
pub mod ear_encoder_partition;
/// Cold exact binding plans for EAR encoder execution units.
pub mod ear_encoder_preparation;

/// Shared bounded streaming convolutions and finite-stream lifecycle.
pub mod streaming;

/// Periodic-Hann STFT, inverse STFT and overlap-add mechanisms.
pub mod spectral;

/// EAR codec spectral reconstruction and bounded complex history.
pub mod ear_spectral;

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
    /// A completed streaming layer requires reset before accepting more input.
    #[error("streaming codec layer has finished; reset before submitting input")]
    StreamFinished,

    /// Invalid input or checkpoint shape.
    #[error("{0}")]
    InvalidShape(String),

    /// Portable layered execution graph failure.
    #[error(transparent)]
    Graph(#[from] eredu_runtime::ExecutionGraphError),

    /// Portable layered execution unit layout failure.
    #[error(transparent)]
    UnitLayout(#[from] eredu_runtime::ExecutionUnitLayoutError),

    /// Portable layered execution state failure.
    #[error(transparent)]
    State(#[from] eredu_runtime::StateError),

    /// Portable layered execution schedule failure.
    #[error(transparent)]
    Schedule(#[from] eredu_core::LayerScheduleError),

    /// Portable layered execution parameters failure.
    #[error(transparent)]
    Parameters(#[from] eredu_runtime::ParallelPlanError),

    /// Portable layered execution ownership failure.
    #[error(transparent)]
    Ownership(#[from] eredu_runtime::ArchitectureParameterError),

    /// Portable layered execution identity failure.
    #[error(transparent)]
    Identity(#[from] eredu_core::cache::PromptCacheError),

    /// Portable layered execution backend failure.
    #[error(transparent)]
    Backend(#[from] eredu_core::BackendFailure),

    /// Underlying neural-compute backend error.
    #[error(transparent)]
    Compute(#[from] eredu_nn::Error),
}
