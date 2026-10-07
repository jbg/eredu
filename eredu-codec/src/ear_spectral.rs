//! EAR codec magnitude/phase reconstruction and bounded streaming spectral history.
//!
//! Streaming prepends two half-padding intervals of retained spectra, then crops
//! one interval at each waveform end. Flushing appends one zero spectral interval.
//! This preserves the reference startup delay and finite-edge envelope; ordinary
//! finite reconstruction and streamed output have different endpoint normalization.

use eredu_core::{execution_control::SnapshotEstimate, BackendFailure};
use eredu_runtime::execution_control::{
    snapshot_value_estimate, SnapshotReservation, SnapshotValue,
};

use eredu_nn::RetainedTensorRoots;

use crate::{spectral::PeriodicHannIstft, streaming::StreamPhase, Error};
use eredu_nn::{Index, NeuralBackend, NeuralOperatorCapabilities, Tensor, TensorElementType};

/// Released spectral synthesis geometry, independent of latent upsampling rates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpectralConfig {
    /// Even inverse FFT length (released: 16).
    pub fft_size: i32,
    /// Samples between adjacent spectra (released: 4).
    pub hop_length: i32,
}
impl SpectralConfig {
    /// Validates FFT, hop and bounded integer crop geometry without allocation.
    /// Symmetric integer endpoint crops require an even FFT-minus-hop: an odd
    /// difference adds one sample per invocation and violates fixed PCM cadence.
    pub fn validate(self) -> Result<(), Error> {
        if self.fft_size < 4
            || self.fft_size > i32::MAX - 2
            || self.fft_size % 2 != 0
            || self.hop_length <= 0
            || self.hop_length > self.fft_size - 2
        {
            return Err(invalid("invalid EAR spectral geometry"));
        }
        if (self.fft_size - self.hop_length) % 2 != 0 {
            return Err(invalid(
                "EAR fixed cadence requires an even FFT-minus-hop for symmetric endpoint crops",
            ));
        }
        let pad = (self.fft_size - self.hop_length) / 2;
        let half = pad / self.hop_length + i32::from(pad % self.hop_length != 0);
        half.checked_mul(self.hop_length)
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(|| invalid("spectral waveform crop exceeds i32"))?;
        Ok(())
    }
    /// Numerical capabilities needed before constructing any native constants.
    pub const fn required_capabilities(self) -> NeuralOperatorCapabilities {
        NeuralOperatorCapabilities::IRFFT
            .union(NeuralOperatorCapabilities::SIN_COS)
            .union(NeuralOperatorCapabilities::FULL_F32)
            .union(NeuralOperatorCapabilities::CAST_FLOAT)
            .union(NeuralOperatorCapabilities::SOFTPLUS)
            .union(NeuralOperatorCapabilities::EXP)
            .union(NeuralOperatorCapabilities::CLIP)
            .union(NeuralOperatorCapabilities::BROADCAST_TO)
    }
    fn half_padding(self) -> Result<i32, Error> {
        self.validate()?;
        let pad = (self.fft_size - self.hop_length) / 2;
        Ok(pad / self.hop_length + i32::from(pad % self.hop_length != 0))
    }
    /// Spectral startup latency and nonempty drain length, without native allocation.
    pub fn delay_samples(self) -> Result<i32, Error> {
        // Validation bounds both waveform crops before this multiplication.
        Ok(self.half_padding()? * self.hop_length)
    }
}

/// Mutable spectral carry only; cloning does not retain learned parameters.
#[derive(Debug, Clone)]
pub struct SpectralState<T: Tensor> {
    config: SpectralConfig,
    batch: i32,
    real: Option<T>,
    imaginary: Option<T>,
    frames_seen: u64,
    samples_emitted: u64,
    phase: StreamPhase,
}

impl<T: Tensor + SnapshotValue> SnapshotValue for SpectralState<T> {
    fn estimate_snapshot_value(&self) -> Result<SnapshotEstimate, BackendFailure> {
        snapshot_value_estimate::<Self>([
            self.real.estimate_snapshot_value(),
            self.imaginary.estimate_snapshot_value(),
        ])
    }
    fn copy_snapshot_value(&self, retention: &SnapshotReservation) -> Result<Self, BackendFailure> {
        Ok(Self {
            real: self.real.copy_snapshot_value(retention)?,
            imaginary: self.imaginary.copy_snapshot_value(retention)?,
            config: self.config,
            batch: self.batch,
            frames_seen: self.frames_seen,
            samples_emitted: self.samples_emitted,
            phase: self.phase,
        })
    }
}

impl<T: Tensor> RetainedTensorRoots<T> for SpectralState<T> {
    fn visit_retained_tensors<'a>(&'a self, visitor: &mut dyn FnMut(&'a T)) {
        if let Some(real) = &self.real {
            visitor(real);
        }
        if let Some(imaginary) = &self.imaginary {
            visitor(imaginary);
        }
    }
}
impl<T: Tensor> SpectralState<T> {
    /// Whether finite spectral drain has completed.
    pub fn is_finished(&self) -> bool {
        self.phase == StreamPhase::Finished
    }
    /// Input spectral columns consumed, excluding startup and flush padding.
    pub fn frames_seen(&self) -> u64 {
        self.frames_seen
    }
    /// Committed-candidate output samples, including declared startup and flush.
    pub fn samples_emitted(&self) -> u64 {
        self.samples_emitted
    }
    /// Batch geometry retained independently of any tensor allocation.
    pub fn batch(&self) -> i32 {
        self.batch
    }
    /// Bounded complex-history length represented by split real tensors.
    pub fn retained_frames(&self) -> i32 {
        self.real.as_ref().map_or(0, |t| t.dim(2))
    }
    /// Split complex history, useful for snapshot and residency diagnostics.
    pub fn history(&self) -> Option<(&T, &T)> {
        self.real.as_ref().zip(self.imaginary.as_ref())
    }
}

/// Parameter-free EAR spectral equations over general neural mechanisms.
pub struct EarSpectralDecoder<B: NeuralBackend> {
    config: SpectralConfig,
    half_padding: i32,
    istft: PeriodicHannIstft<B::Tensor>,
}
impl<B: NeuralBackend> EarSpectralDecoder<B> {
    /// Constructs bounded window/routing constants, without checkpoint access.
    pub fn new(
        config: SpectralConfig,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let half_padding = config.half_padding()?;
        B::require_operator_capabilities("EAR spectral synthesis", config.required_capabilities())?;
        let istft = PeriodicHannIstft::new(config.fft_size, config.hop_length, context)?;
        Ok(Self {
            config,
            half_padding,
            istft,
        })
    }
    /// Exact spectral geometry.
    pub fn config(&self) -> SpectralConfig {
        self.config
    }
    /// Spectral-stream startup delay and additional final drain samples.
    pub fn delay_samples(&self) -> i32 {
        self.half_padding * self.config.hop_length
    }
    /// Creates empty state with explicit batch geometry and no native allocation.
    pub fn initial_state(&self, batch: i32) -> Result<SpectralState<B::Tensor>, Error> {
        if batch <= 0 {
            return Err(invalid("spectral batch must be positive"));
        }
        Ok(SpectralState {
            config: self.config,
            batch,
            real: None,
            imaginary: None,
            frames_seen: 0,
            samples_emitted: 0,
            phase: StreamPhase::AcceptingInput,
        })
    }
    fn validate_raw(&self, raw: &B::Tensor) -> Result<i32, Error> {
        if raw.shape().len() != 3
            || raw.dim(0) <= 0
            || raw.dim(1) != self.config.fft_size + 2
            || raw.dim(2) < 0
            || !matches!(
                raw.element_type(),
                Some(TensorElementType::F32 | TensorElementType::F16 | TensorElementType::Bf16)
            )
        {
            return Err(invalid(
                "EAR spectral projection must be floating [batch,FFT+2,frames]",
            ));
        }
        Ok(raw.dim(2))
    }
    fn spectrum(
        &self,
        raw: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(B::Tensor, B::Tensor), Error> {
        let raw = raw.cast_float(TensorElementType::F32, context)?;
        let bins = self.config.fft_size / 2 + 1;
        let magnitude = raw.index(&[Index::Full, Index::Range(0, bins), Index::Full], context)?;
        let phase = raw.index(
            &[Index::Full, Index::Range(bins, 2 * bins), Index::Full],
            context,
        )?;
        let log_max = B::Tensor::full_f32(100.0f64.ln() as f32, &[], context)?;
        let magnitude = B::exp(
            B::softplus(log_max.subtract(&magnitude, context)?, 1.0, context)?
                .multiply_scalar(-1.0, context)?,
            context,
        )?
        .multiply_scalar(100.0, context)?;
        let (sine, cosine) = phase.sin_cos(context)?;
        let real = magnitude.multiply(&cosine, context)?;
        let mid = magnitude
            .index(
                &[Index::Full, Index::Range(1, bins - 1), Index::Full],
                context,
            )?
            .multiply(
                &sine.index(
                    &[Index::Full, Index::Range(1, bins - 1), Index::Full],
                    context,
                )?,
                context,
            )?;
        let zeros = B::Tensor::full_f32(0.0, &[raw.dim(0), 1, raw.dim(2)], context)?;
        let imaginary = B::Tensor::concatenate(&[zeros.clone(), mid, zeros], 1, context)?;
        Ok((real, imaginary))
    }
    /// Ordinary finite reconstruction without streaming delay or spectral carry.
    pub fn ordinary(
        &self,
        raw: &B::Tensor,
        constrain: bool,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let frames = self.validate_raw(raw)?;
        if frames == 0 {
            return Ok(B::Tensor::full_f32(0.0, &[raw.dim(0), 1, 0], context)?);
        }
        let (real, imaginary) = self.spectrum(raw, context)?;
        self.istft
            .reconstruct(&real, &imaginary, constrain, context)
    }
    /// Proposes PCM and new carry without mutating the input state. The caller's
    /// existing completion transaction owns evaluation and state publication.
    /// Empty chunks are no-ops; flushing an empty invocation emits no invented audio.
    pub fn step(
        &self,
        raw: &B::Tensor,
        state: &SpectralState<B::Tensor>,
        flush: bool,
        constrain: bool,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(B::Tensor, SpectralState<B::Tensor>), Error> {
        if state.is_finished() {
            return Err(Error::StreamFinished);
        }
        let frames = self.validate_raw(raw)?;
        if state.config != self.config || state.batch != raw.dim(0) {
            return Err(invalid(
                "spectral state belongs to a different geometry or batch",
            ));
        }
        let mut next = state.clone();
        if frames == 0 && (!flush || state.frames_seen == 0) {
            if flush {
                next.phase = StreamPhase::Finished;
            }
            return Ok((
                B::Tensor::full_f32(0.0, &[state.batch, 1, 0], context)?,
                next,
            ));
        }
        let next_frames = state
            .frames_seen
            .checked_add(frames as u64)
            .ok_or_else(|| invalid("spectral frame frontier exhausted"))?;
        let padding = self
            .half_padding
            .checked_mul(2)
            .ok_or_else(|| invalid("spectral history extent overflows"))?;
        let joined_frames = frames
            .checked_add(padding)
            .ok_or_else(|| invalid("spectral frame extent exceeds i32"))?;
        let synthesis_frames = joined_frames
            .checked_add(if flush { self.half_padding } else { 0 })
            .ok_or_else(|| invalid("spectral flush extent exceeds i32"))?;
        let output_samples = self
            .istft
            .output_samples(synthesis_frames)?
            .checked_sub(self.delay_samples() * 2)
            .ok_or_else(|| invalid("spectral crop exceeds waveform"))?;
        let next_samples = state
            .samples_emitted
            .checked_add(output_samples as u64)
            .ok_or_else(|| invalid("spectral sample frontier exhausted"))?;
        let bins = self.config.fft_size / 2 + 1;
        let (real, imaginary) = if frames == 0 {
            let empty = B::Tensor::full_f32(0.0, &[state.batch, bins, 0], context)?;
            (empty.clone(), empty)
        } else {
            self.spectrum(raw, context)?
        };
        let prefix = |retained: &Option<B::Tensor>| -> Result<B::Tensor, Error> {
            Ok(match retained {
                Some(value) => value.clone(),
                None => B::Tensor::full_f32(0.0, &[state.batch, bins, padding], context)?,
            })
        };
        let mut real = B::Tensor::concatenate(&[prefix(&state.real)?, real], 2, context)?;
        let mut imaginary =
            B::Tensor::concatenate(&[prefix(&state.imaginary)?, imaginary], 2, context)?;
        if flush {
            let tail = B::Tensor::full_f32(0.0, &[state.batch, bins, self.half_padding], context)?;
            real = B::Tensor::concatenate(&[real, tail.clone()], 2, context)?;
            imaginary = B::Tensor::concatenate(&[imaginary, tail], 2, context)?;
            next.real = None;
            next.imaginary = None;
            next.phase = StreamPhase::Finished;
        } else {
            let slice = [
                Index::Full,
                Index::Full,
                Index::Range(joined_frames - padding, joined_frames),
            ];
            next.real = Some(real.index(&slice, context)?.compact(context)?);
            next.imaginary = Some(imaginary.index(&slice, context)?.compact(context)?);
        }
        let reconstructed = self
            .istft
            .reconstruct(&real, &imaginary, constrain, context)?;
        let delay = self.delay_samples();
        let output = reconstructed.index(
            &[
                Index::Full,
                Index::Full,
                Index::Range(delay, reconstructed.dim(2) - delay),
            ],
            context,
        )?;
        next.frames_seen = next_frames;
        next.samples_emitted = next_samples;
        Ok((output, next))
    }
    /// Completes only spectral overlap; neural convolution carries have their own owner.
    pub fn finish(
        &self,
        state: &SpectralState<B::Tensor>,
        constrain: bool,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(B::Tensor, SpectralState<B::Tensor>), Error> {
        if state.is_finished() {
            return Err(Error::StreamFinished);
        }
        let empty = B::Tensor::full_f32(0.0, &[state.batch, self.config.fft_size + 2, 0], context)?;
        self.step(&empty, state, true, constrain, context)
    }
}
fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidShape(message.into())
}
