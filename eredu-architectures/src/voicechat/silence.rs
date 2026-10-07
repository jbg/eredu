//! Fixed one-second perception conditioning for asynchronous tool-channel steps.
//!
//! This preparation is independent of live PCM, RNNT and conversational clocks.
//! The released wrapper encodes an entire second of silence with fresh perception
//! state and retains only its first projected frame. Streaming startup uses a
//! different mel window and cannot substitute for this conditioning.

use eredu_media::audio::{
    streaming::{CenteredSlaneyLogMelConfig, MelWindowBoundary, StreamingLogMelWindow},
    LogMelFeatures,
};
use eredu_nn::{Index, NeuralBackend, Tensor, TensorElementType};

use super::{
    artifact::RetainedVoiceChatArtifacts,
    config::PreprocessingConfig,
    frontend::{preprocessing_geometry, FrontendError, INPUT_SAMPLE_RATE},
    perception::{PerceptionError, PerceptionState},
    perception_layered::PerceptionExecution,
};

/// Invalid retained preprocessing, incompatible execution or failed neural work.
#[derive(Debug, thiserror::Error)]
pub enum SilenceConditioningError {
    /// Portable host preprocessing failed before publishing any conditioning.
    #[error(transparent)]
    Frontend(#[from] FrontendError),
    /// Requested precision or perception geometry is incompatible.
    #[error("{0}")]
    Geometry(&'static str),
    /// Learned perception failed; no live state was borrowed or changed.
    #[error(transparent)]
    Perception(#[from] PerceptionError),
    /// Tensor mechanism failed with its original source retained.
    #[error(transparent)]
    Neural(#[from] eredu_nn::Error),
}

/// Exact host coefficients and geometry; cold preparation requires no FFT support.
///
/// Extraction and neural work occur only in `mel_features` and `execute`.
#[derive(Debug, Clone)]
pub struct PreparedSilenceConditioning {
    geometry: CenteredSlaneyLogMelConfig,
    window: Vec<f32>,
    filters: Vec<f32>,
}

impl PreparedSilenceConditioning {
    /// Uses only already-retained preprocessing buffers and normalized policy.
    pub fn from_retained(
        artifacts: &RetainedVoiceChatArtifacts,
    ) -> Result<Self, SilenceConditioningError> {
        let preprocessing = artifacts
            .preprocessing()
            .ok_or(FrontendError::MissingPreprocessing)?;
        Self::with_coefficients(
            &artifacts.plan().config().perception.preprocessor,
            preprocessing.window().to_vec(),
            preprocessing.filters().to_vec(),
        )
    }

    /// Explicit canonical coefficients for independently prepared inputs and fixtures.
    /// No coefficients are regenerated and no audio processor is allocated here.
    pub fn with_coefficients(
        config: &PreprocessingConfig,
        window: Vec<f32>,
        filters: Vec<f32>,
    ) -> Result<Self, SilenceConditioningError> {
        let geometry = preprocessing_geometry(config, INPUT_SAMPLE_RATE as usize)?;
        if window.len() != geometry.window_length
            || filters.len() != geometry.mel_bins * (geometry.fft_length / 2 + 1)
            || window.iter().any(|v| !v.is_finite())
            || filters.iter().any(|v| !v.is_finite() || *v < 0.0)
        {
            return Err(SilenceConditioningError::Geometry(
                "invalid retained silence preprocessing coefficients",
            ));
        }
        Ok(Self {
            geometry,
            window,
            filters,
        })
    }

    /// Rejects non-floating conditioning precision before native submission opens.
    pub fn validate_dtype(output_dtype: TensorElementType) -> Result<(), SilenceConditioningError> {
        if !matches!(
            output_dtype,
            TensorElementType::F16
                | TensorElementType::Bf16
                | TensorElementType::F32
                | TensorElementType::F64
        ) {
            return Err(SilenceConditioningError::Geometry(
                "silence conditioning requires floating precision",
            ));
        }
        Ok(())
    }

    /// Extracts all 101 centered mel rows of the released 16,000-sample silence.
    /// Dither and padding-to-multiple are disabled as in the inference wrapper.
    pub fn mel_features(&self) -> Result<LogMelFeatures, SilenceConditioningError> {
        let mut window = StreamingLogMelWindow::with_coefficients(
            self.geometry.clone(),
            self.window.clone(),
            self.filters.clone(),
        )
        .map_err(FrontendError::from)?;
        window
            .append(&vec![0.0; INPUT_SAMPLE_RATE as usize], INPUT_SAMPLE_RATE)
            .map_err(FrontendError::from)?;
        let frames = window.frame_count();
        Ok(window
            .extract(0..frames, MelWindowBoundary::Reflect)
            .map_err(FrontendError::from)?)
    }

    /// Runs full causal perception once and retains a compact `[1, 1, H]` frame.
    ///
    /// Parameters use the ordinary Float32 perception equations. Only the final
    /// conditioning is cast to the explicitly selected language-model precision.
    /// The caller establishes native completion before publishing the returned
    /// tensor; this method neither starts RNNT nor installs perception caches.
    pub fn execute<B: NeuralBackend, P: PerceptionExecution<B>>(
        &self,
        perception: &mut P,
        output_dtype: TensorElementType,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, SilenceConditioningError> {
        Self::validate_dtype(output_dtype)?;
        if perception.dimensions().subsampling.features != self.geometry.mel_bins as i32 {
            return Err(SilenceConditioningError::Geometry(
                "silence preprocessing and perception mel widths differ",
            ));
        }
        let mel = self.mel_features()?;
        let input = B::Tensor::from_f32_slice(
            &mel.values,
            &[1, mel.frames as i32, mel.mel_bins as i32],
            context,
        )?;
        let output = perception.forward(&input, 0, &PerceptionState::default(), context)?;
        let first = output
            .projected
            .index(&[Index::Full, Index::Range(0, 1), Index::Full], context)?
            .compact(context)?;
        Ok(first.cast_float(output_dtype, context)?)
    }
}
