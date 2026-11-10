//! Portable periodic-Hann STFT and inverse STFT with explicit endpoint policy.

use crate::Error;
use eredu_nn::{Index, NeuralOperatorCapabilities, PadMode, Tensor, TensorElementType};

mod window;

/// Periodic Hann coefficient arithmetic, independent of endpoint padding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowArithmetic {
    /// Backend F32 sine/cosine construction.
    Native,
    /// Fixed host F32 range reduction and fused cosine polynomial.
    Float32Polynomial,
}
/// Explicit mechanism for the forward spectral transform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnalysisExecution {
    /// Use the supplied execution context's native F32 FFT.
    Native,
    /// Use host F32 FFT arithmetic, including required device transfers.
    HostFloat32,
}

/// Explicit finite-analysis endpoint extension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnalysisEndpoints {
    /// Extend both endpoints with zero samples.
    Zero,
    /// Reflect around each endpoint without repeating its edge sample.
    Reflect,
}

/// Parameter-free finite STFT with explicit endpoint padding. Input is mono
/// `[batch, 1, samples]`; output components use `[batch, bins, frames]`.
/// The periodic window covers the complete FFT; the transform is unnormalized.
#[derive(Debug, Clone)]
pub struct PeriodicHannStft<T: Tensor> {
    fft_size: i32,
    hop_length: i32,
    window: T,
    endpoints: AnalysisEndpoints,
    execution: AnalysisExecution,
}
impl<T: Tensor> PeriodicHannStft<T> {
    /// Constructs a periodic Hann window; no waveform or transform state is retained.
    pub fn new(fft_size: i32, hop_length: i32, context: &T::Context) -> Result<Self, Error> {
        Self::with_endpoints(fft_size, hop_length, AnalysisEndpoints::Zero, context)
    }
    /// Constructs the same periodic-Hann transform with explicit endpoint policy.
    pub fn with_endpoints(
        fft_size: i32,
        hop_length: i32,
        endpoints: AnalysisEndpoints,
        context: &T::Context,
    ) -> Result<Self, Error> {
        Self::with_policy(
            fft_size,
            hop_length,
            endpoints,
            WindowArithmetic::Native,
            AnalysisExecution::Native,
            context,
        )
    }
    /// Constructs explicit window and transform arithmetic without selecting a family.
    pub fn with_policy(
        fft_size: i32,
        hop_length: i32,
        endpoints: AnalysisEndpoints,
        window: WindowArithmetic,
        execution: AnalysisExecution,
        context: &T::Context,
    ) -> Result<Self, Error> {
        if fft_size < 2 || hop_length <= 0 || hop_length > fft_size {
            return Err(invalid("forward STFT requires FFT>=2 and 0<hop<=FFT"));
        }
        Ok(Self {
            fft_size,
            hop_length,
            window: match window {
                WindowArithmetic::Native => periodic_hann::<T>(fft_size, context)?,
                WindowArithmetic::Float32Polynomial => {
                    T::from_f32_slice(&window::hann(fft_size), &[fft_size], context)?
                }
            }
            .reshape(&[1, 1, fft_size], context)?,
            endpoints,
            execution,
        })
    }
    /// Optional numerical capabilities required before native construction.
    pub const fn required_capabilities() -> NeuralOperatorCapabilities {
        Self::required_capabilities_for(WindowArithmetic::Native, AnalysisExecution::Native)
    }
    /// Exact optional mechanisms for one retained arithmetic choice.
    pub const fn required_capabilities_for(
        window: WindowArithmetic,
        execution: AnalysisExecution,
    ) -> NeuralOperatorCapabilities {
        let fft = match execution {
            AnalysisExecution::Native => NeuralOperatorCapabilities::RFFT,
            AnalysisExecution::HostFloat32 => NeuralOperatorCapabilities::HOST_RFFT,
        };
        let required = fft
            .union(NeuralOperatorCapabilities::CAST_FLOAT)
            .union(NeuralOperatorCapabilities::FROM_I32_SLICE);
        match window {
            WindowArithmetic::Native => required
                .union(NeuralOperatorCapabilities::SIN_COS)
                .union(NeuralOperatorCapabilities::FULL_F32),
            WindowArithmetic::Float32Polynomial => required,
        }
    }
    /// Number of complete analysis windows after the explicit endpoint padding.
    pub fn output_frames(&self, samples: i32) -> Result<i32, Error> {
        if samples < self.hop_length {
            return Err(invalid("forward STFT requires at least one hop of samples"));
        }
        samples
            .checked_add(self.fft_size - self.hop_length)
            .ok_or_else(|| invalid("forward STFT padded input exceeds i32"))?;
        Ok(samples / self.hop_length)
    }
    /// Analyzes a finite waveform in F32. Reflective endpoints require each
    /// padding extent to be smaller than the input. Left padding is
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
        let padded = match self.endpoints {
            AnalysisEndpoints::Zero => {
                T::pad(&input, &[(0, 0), (left, right)], PadMode::Constant, context)?
            }
            AnalysisEndpoints::Reflect => {
                let samples = pcm.dim(2);
                if left >= samples || right >= samples {
                    return Err(invalid(
                        "reflective STFT padding must be smaller than the waveform",
                    ));
                }
                let indices = (-left..samples + right)
                    .map(|i| {
                        if i < 0 {
                            -i
                        } else if i >= samples {
                            samples - 2 - (i - samples)
                        } else {
                            i
                        }
                    })
                    .collect::<Vec<_>>();
                input.take_axis(
                    &T::from_i32_slice(&indices, &[samples + left + right], context)?,
                    1,
                    context,
                )?
            }
        };
        let framed = padded
            .take_axis(&T::from_i32_slice(&indices, &[count], context)?, 1, context)?
            .reshape(&[pcm.dim(0), frames, self.fft_size], context)?
            .multiply(&self.window, context)?;
        let (real, imaginary) = match self.execution {
            AnalysisExecution::Native => framed.rfft(self.fft_size, 2, context)?,
            AnalysisExecution::HostFloat32 => framed.host_rfft(self.fft_size, 2, context)?,
        };
        Ok((
            real.transpose_axes(&[0, 2, 1], context)?,
            imaginary.transpose_axes(&[0, 2, 1], context)?,
        ))
    }
}

/// Unnormalized inverse-FFT overlap-add values before endpoint trimming.
#[derive(Debug, Clone)]
pub struct OverlapAddOutput<T> {
    /// Windowed sample sums shaped `[batch, samples, 1]`.
    pub audio: T,
    /// Window-square sums shaped `[1, samples, 1]`, shared across the batch.
    pub envelope: T,
}

/// Bounded sample/envelope continuation for finite spectral chunks. Only the
/// unfinished `FFT-hop` tail and the unconsumed initial trim are retained.
#[derive(Debug, Clone)]
pub struct OverlapAddStream<T: Tensor> {
    batch: i32,
    fft_size: i32,
    hop_length: i32,
    left_trim: i32,
    carry: Option<OverlapAddOutput<T>>,
    phase: crate::streaming::StreamPhase,
}

impl<T: Tensor + eredu_runtime::execution_control::SnapshotValue>
    eredu_runtime::execution_control::SnapshotValue for OverlapAddStream<T>
{
    fn estimate_snapshot_value(
        &self,
    ) -> Result<eredu_core::execution_control::SnapshotEstimate, eredu_core::BackendFailure> {
        eredu_runtime::execution_control::snapshot_value_estimate::<Self>(
            self.retained_values().map(T::estimate_snapshot_value),
        )
    }

    fn copy_snapshot_value(
        &self,
        retention: &eredu_runtime::execution_control::SnapshotReservation,
    ) -> Result<Self, eredu_core::BackendFailure> {
        let carry = self
            .carry
            .as_ref()
            .map(|carry| {
                Ok::<_, eredu_core::BackendFailure>(OverlapAddOutput {
                    audio: carry.audio.copy_snapshot_value(retention)?,
                    envelope: carry.envelope.copy_snapshot_value(retention)?,
                })
            })
            .transpose()?;
        Ok(Self { carry, ..*self })
    }
}

impl<T: Tensor> OverlapAddStream<T> {
    /// Starts one finite stream. Even symmetric endpoint trimming is explicit;
    /// batch geometry cannot change while its sample/envelope carry is live.
    pub fn new(batch: i32, fft_size: i32, hop_length: i32) -> Result<Self, Error> {
        if batch <= 0
            || fft_size < 4
            || hop_length <= 0
            || hop_length > fft_size - 2
            || (fft_size - hop_length) % 2 != 0
        {
            return Err(invalid("invalid spectral streaming geometry"));
        }
        Ok(Self {
            batch,
            fft_size,
            hop_length,
            left_trim: (fft_size - hop_length) / 2,
            carry: None,
            phase: crate::streaming::StreamPhase::AcceptingInput,
        })
    }
    /// Finite-input lifecycle; a finalized stream rejects subsequent input.
    pub fn phase(&self) -> crate::streaming::StreamPhase {
        self.phase
    }
    /// Exact unfinished roots retained for a snapshot or transport owner.
    pub fn retained_values(&self) -> impl Iterator<Item = &T> {
        self.carry.iter().flat_map(|c| [&c.audio, &c.envelope])
    }
    /// Consumes raw overlap-add sums. `None` supplies no new spectral frames;
    /// finalizing with `None` flushes the held tail. Output is normalized F32
    /// `[batch,1,samples]`, possibly empty. State publication follows successful
    /// construction of all new roots; submission owners still establish native
    /// completion before exposing samples or persistent snapshots.
    pub fn push(
        &mut self,
        output: Option<OverlapAddOutput<T>>,
        finalize: bool,
        context: &T::Context,
    ) -> Result<T, Error> {
        use crate::streaming::StreamPhase;
        if self.phase == StreamPhase::Finished {
            return Err(Error::StreamFinished);
        }
        let tail = self.fft_size - self.hop_length;
        let trim = tail / 2;
        let flush = output.is_none();
        let Some(mut output) = output.or_else(|| if finalize { self.carry.clone() } else { None })
        else {
            let empty = T::full_f32(0.0, &[self.batch, 1, 0], context)?;
            if finalize {
                self.phase = StreamPhase::Finished;
            }
            return Ok(empty);
        };
        // A flush already owns the accumulated tail. New chunks include their
        // full FFT extent, and overlap that tail at the first sample.
        let length = output.audio.shape().get(1).copied().unwrap_or(0);
        if output.audio.shape() != [self.batch, length, 1]
            || output.envelope.shape() != [1, length, 1]
            || output.audio.element_type() != Some(TensorElementType::F32)
            || output.envelope.element_type() != Some(TensorElementType::F32)
            || (!flush
                && (length < self.fft_size || (length - self.fft_size) % self.hop_length != 0))
            || (flush && (!finalize || length != tail))
        {
            return Err(invalid(
                "spectral stream requires complete matching F32 sample/envelope sums",
            ));
        }
        let slice = |value: &T, start, end| {
            value.index(
                &[Index::Full, Index::Range(start, end), Index::Full],
                context,
            )
        };
        if !flush {
            if let Some(carry) = &self.carry {
                output.audio = T::concatenate(
                    &[
                        slice(&output.audio, 0, tail)?.add(&carry.audio, context)?,
                        slice(&output.audio, tail, length)?,
                    ],
                    1,
                    context,
                )?;
                output.envelope = T::concatenate(
                    &[
                        slice(&output.envelope, 0, tail)?.add(&carry.envelope, context)?,
                        slice(&output.envelope, tail, length)?,
                    ],
                    1,
                    context,
                )?;
            }
        }
        let stable = length - if finalize { trim } else { tail };
        let skip = self.left_trim.min(stable);
        let samples = slice(&output.audio, skip, stable)?
            .divide(&slice(&output.envelope, skip, stable)?, context)?
            .transpose_axes(&[0, 2, 1], context)?;
        let carry = if finalize {
            None
        } else {
            Some(OverlapAddOutput {
                audio: slice(&output.audio, stable, length)?.compact(context)?,
                envelope: slice(&output.envelope, stable, length)?.compact(context)?,
            })
        };
        self.carry = carry;
        self.left_trim -= skip;
        if finalize {
            self.phase = StreamPhase::Finished;
        }
        Ok(samples)
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
    /// Replaces the synthesis coefficients with an explicit retained F32 window.
    /// The caller's window must satisfy nonzero overlap after endpoint trimming.
    /// The view shares its source; the fixed overlap routing kernel is reused.
    pub fn set_window(&mut self, window: &T, context: &T::Context) -> Result<(), Error> {
        if window.shape() != [self.fft_size]
            || window.element_type() != Some(TensorElementType::F32)
        {
            return Err(invalid(
                "synthesis window must be F32 with one coefficient per FFT tap",
            ));
        }
        window.require_finite(context)?;
        self.window = window.reshape(&[1, self.fft_size, 1], context)?;
        Ok(())
    }
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
        let output = self.overlap_add(real, imaginary, constrain, context)?;
        let output_length = output.audio.dim(1);
        let slice = [
            Index::Full,
            Index::Range(self.pad, output_length - self.pad),
            Index::Full,
        ];
        Ok(output
            .audio
            .index(&slice, context)?
            .divide(&output.envelope.index(&slice, context)?, context)?
            .transpose_axes(&[0, 2, 1], context)?)
    }
    /// Reconstructs unnormalized sample and envelope sums for a caller-owned
    /// finite or streaming endpoint policy. Arithmetic matches `reconstruct`;
    /// neither endpoint is trimmed and no division occurs here.
    pub fn overlap_add(
        &self,
        real: &T,
        imaginary: &T,
        constrain: bool,
        context: &T::Context,
    ) -> Result<OverlapAddOutput<T>, Error> {
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
        if wave.shape() != [real.dim(0), output_length, 1]
            || envelope.shape() != [1, output_length, 1]
        {
            return Err(invalid(
                "inverse STFT overlap-add mechanism returned invalid geometry",
            ));
        }
        Ok(OverlapAddOutput {
            audio: wave,
            envelope,
        })
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
