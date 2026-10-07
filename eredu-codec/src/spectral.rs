//! Portable periodic-Hann STFT and inverse STFT with explicit endpoint policy.

use crate::Error;
use eredu_nn::{Index, NeuralOperatorCapabilities, PadMode, Tensor, TensorElementType};

/// Parameter-free finite STFT with zero endpoint padding. Input is mono
/// `[batch, 1, samples]`; output components use `[batch, bins, frames]`.
/// The periodic window covers the complete FFT; the transform is unnormalized.
#[derive(Debug, Clone)]
pub struct PeriodicHannStft<T: Tensor> {
    fft_size: i32,
    hop_length: i32,
    window: T,
}
impl<T: Tensor> PeriodicHannStft<T> {
    /// Constructs a periodic Hann window; no waveform or transform state is retained.
    pub fn new(fft_size: i32, hop_length: i32, context: &T::Context) -> Result<Self, Error> {
        if fft_size < 2 || hop_length <= 0 || hop_length > fft_size {
            return Err(invalid("forward STFT requires FFT>=2 and 0<hop<=FFT"));
        }
        Ok(Self {
            fft_size,
            hop_length,
            window: periodic_hann::<T>(fft_size, context)?.reshape(&[1, 1, fft_size], context)?,
        })
    }
    /// Optional numerical capabilities required before native construction.
    pub const fn required_capabilities() -> NeuralOperatorCapabilities {
        NeuralOperatorCapabilities::RFFT
            .union(NeuralOperatorCapabilities::SIN_COS)
            .union(NeuralOperatorCapabilities::CAST_FLOAT)
            .union(NeuralOperatorCapabilities::FULL_F32)
            .union(NeuralOperatorCapabilities::FROM_I32_SLICE)
    }
    /// Number of complete analysis windows after the explicit zero padding.
    pub fn output_frames(&self, samples: i32) -> Result<i32, Error> {
        if samples < self.hop_length {
            return Err(invalid("forward STFT requires at least one hop of samples"));
        }
        samples
            .checked_add(self.fft_size - self.hop_length)
            .ok_or_else(|| invalid("forward STFT padded input exceeds i32"))?;
        Ok(samples / self.hop_length)
    }
    /// Analyzes a finite waveform in F32. Left padding is
    /// `floor((FFT-hop)/2)` and right padding receives the remaining samples.
    /// The returned tensors are Cartesian components, not magnitude and phase.
    pub fn analyze(&self, pcm: &T, context: &T::Context) -> Result<(T, T), Error> {
        if pcm.shape().len() != 3 || pcm.dim(0) <= 0 || pcm.dim(1) != 1 {
            return Err(invalid(
                "forward STFT requires mono [batch,1,samples] input",
            ));
        }
        let frames = self.output_frames(pcm.dim(2))?;
        let count = frames
            .checked_mul(self.fft_size)
            .ok_or_else(|| invalid("forward STFT frame extent exceeds i32"))?;
        let mut indices = Vec::new();
        indices
            .try_reserve_exact(count as usize)
            .map_err(|e| invalid(e.to_string()))?;
        for frame in 0..frames {
            let offset = frame * self.hop_length;
            indices.extend((0..self.fft_size).map(|tap| offset + tap));
        }
        let left = (self.fft_size - self.hop_length) / 2;
        let right = self.fft_size - self.hop_length - left;
        let input = pcm
            .cast_float(TensorElementType::F32, context)?
            .squeeze_axes(&[1], context)?;
        let padded = T::pad(&input, &[(0, 0), (left, right)], PadMode::Constant, context)?;
        let framed = padded
            .take_axis(&T::from_i32_slice(&indices, &[count], context)?, 1, context)?
            .reshape(&[pcm.dim(0), frames, self.fft_size], context)?
            .multiply(&self.window, context)?;
        let (real, imaginary) = framed.rfft(self.fft_size, 2, context)?;
        Ok((
            real.transpose_axes(&[0, 2, 1], context)?,
            imaginary.transpose_axes(&[0, 2, 1], context)?,
        ))
    }
}

/// Parameter-free inverse STFT constants. Spectra use `[batch, bins, frames]`;
/// output uses `[batch, 1, samples]`. Real and imaginary components remain
/// separate portable F32 tensors until the backend inverse FFT operation.
#[derive(Debug, Clone)]
pub struct PeriodicHannIstft<T: Tensor> {
    fft_size: i32,
    hop_length: i32,
    pad: i32,
    window: T,
    fold_kernel: T,
}
impl<T: Tensor> PeriodicHannIstft<T> {
    /// Constructs the periodic window and fixed overlap-add routing kernel.
    /// Trims `(fft_size-hop_length)/2` samples from each reconstructed endpoint.
    pub fn new(fft_size: i32, hop_length: i32, context: &T::Context) -> Result<Self, Error> {
        if fft_size < 4 || fft_size % 2 != 0 || hop_length <= 0 || hop_length > fft_size - 2 {
            return Err(invalid(
                "inverse STFT requires even FFT>=4 and 0<hop<=FFT-2",
            ));
        }
        let pad = (fft_size - hop_length) / 2;
        // Every interior envelope is bounded below by this two-frame envelope.
        // Verify the geometric nonzero-overlap condition before native allocation.
        let mut envelope = vec![
            0.0f64;
            (fft_size as usize)
                .checked_add(hop_length as usize)
                .ok_or_else(|| invalid("window envelope extent overflows"))?
        ];
        for i in 0..fft_size {
            let w = 0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / fft_size as f64).cos();
            envelope[i as usize] += w * w;
            envelope[(i + hop_length) as usize] += w * w;
        }
        if envelope[pad as usize..envelope.len() - pad as usize]
            .iter()
            .any(|&n| n <= 1e-11)
        {
            return Err(invalid(
                "inverse STFT window envelope is not strictly positive after trimming",
            ));
        }
        let window = periodic_hann::<T>(fft_size, context)?.reshape(&[1, fft_size, 1], context)?;
        let count = (fft_size as usize)
            .checked_mul(fft_size as usize)
            .ok_or_else(|| invalid("overlap routing extent overflows"))?;
        let mut identity = Vec::new();
        identity
            .try_reserve_exact(count)
            .map_err(|e| invalid(e.to_string()))?;
        identity.resize(count, 0.0);
        for i in 0..fft_size as usize {
            identity[i * fft_size as usize + i] = 1.0;
        }
        let fold_kernel = T::from_f32_slice(&identity, &[1, fft_size, fft_size], context)?;
        Ok(Self {
            fft_size,
            hop_length,
            pad,
            window,
            fold_kernel,
        })
    }
    /// Reconstructs an ordinary finite spectrogram. Clipping, when selected,
    /// bounds each inverse-FFT tap to ±window before windowing and overlap-add.
    pub fn reconstruct(
        &self,
        real: &T,
        imaginary: &T,
        constrain: bool,
        context: &T::Context,
    ) -> Result<T, Error> {
        if real.shape().len() != 3
            || real.shape() != imaginary.shape()
            || real.dim(0) <= 0
            || real.dim(1) != self.fft_size / 2 + 1
            || real.dim(2) <= 0
            || real.element_type() != Some(TensorElementType::F32)
            || imaginary.element_type() != Some(TensorElementType::F32)
        {
            return Err(invalid(
                "inverse STFT requires matching nonempty F32 [batch,bins,frames] components",
            ));
        }
        let frames = real.dim(2);
        let output_length = (frames - 1)
            .checked_mul(self.hop_length)
            .and_then(|n| n.checked_add(self.fft_size))
            .ok_or_else(|| invalid("inverse STFT output exceeds i32"))?;
        let mut inverse = T::irfft(real, imaginary, self.fft_size, 1, context)?;
        if constrain {
            inverse = inverse.clip(
                &self.window.multiply_scalar(-1.0, context)?,
                &self.window,
                context,
            )?;
        }
        let windowed = inverse
            .multiply(&self.window, context)?
            .transpose_axes(&[0, 2, 1], context)?;
        let fold = |input: &T| {
            T::conv_transpose1d(
                input,
                &self.fold_kernel,
                self.hop_length,
                0,
                1,
                0,
                1,
                context,
            )
        };
        let wave = fold(&windowed)?;
        let envelope_frames = self
            .window
            .square(context)?
            .transpose_axes(&[0, 2, 1], context)?
            .broadcast_to(&[1, frames, self.fft_size], context)?;
        let envelope = fold(&envelope_frames)?;
        let slice = [
            Index::Full,
            Index::Range(self.pad, output_length - self.pad),
            Index::Full,
        ];
        Ok(wave
            .index(&slice, context)?
            .divide(&envelope.index(&slice, context)?, context)?
            .transpose_axes(&[0, 2, 1], context)?)
    }
    /// Number of samples returned by an ordinary finite reconstruction.
    pub fn output_samples(&self, frames: i32) -> Result<i32, Error> {
        if frames <= 0 {
            return Err(invalid("inverse STFT requires at least one frame"));
        }
        (frames - 1)
            .checked_mul(self.hop_length)
            .and_then(|n| n.checked_add(self.fft_size - 2 * self.pad))
            .ok_or_else(|| invalid("inverse STFT output exceeds i32"))
    }
}
fn periodic_hann<T: Tensor>(fft_size: i32, context: &T::Context) -> Result<T, Error> {
    let factor = (std::f64::consts::TAU / fft_size as f64) as f32;
    let angles = (0..fft_size).map(|i| i as f32 * factor).collect::<Vec<_>>();
    let (_, cosine) = T::from_f32_slice(&angles, &[fft_size], context)?.sin_cos(context)?;
    Ok(cosine
        .multiply_scalar(-0.5, context)?
        .add(&T::full_f32(0.5, &[], context)?, context)?)
}
fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidShape(message.into())
}
