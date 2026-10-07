//! EAR waveform/code composition over independently materializable components.
//!
//! Codes use `[batch, frames, depth]`, including nonempty depth prefixes;
//! PCM uses `[batch, 1, samples]`.
//! Streaming transitions return unpublished state. The caller commits it only
//! after its ordinary execution/completion transaction succeeds.

use eredu_core::{execution_control::SnapshotEstimate, BackendFailure};
use eredu_runtime::execution_control::{
    snapshot_value_estimate, SnapshotReservation, SnapshotValue,
};

use eredu_nn::RetainedTensorRoots;

use eredu_nn::{NeuralBackend, Parameterized, Tensor};

use crate::{
    ear_decoder::{DecoderConfig, DecoderState, NeuralDecoder},
    ear_decoder_layered::DecoderExecution,
    ear_encoder::{EncoderConfig, NeuralEncoder},
    ear_encoder_layered::EncoderExecution,
    ear_spectral::{EarSpectralDecoder, SpectralConfig, SpectralState},
    prvq::{Geometry, ResidualQuantizer},
    prvq_layered::QuantizerExecution,
    spectral::PeriodicHannStft,
    Error,
};

/// Finite waveform encoder geometry, with explicitly shared residual codebooks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioEncoderConfig {
    /// Encoder-order learned stages.
    pub neural: EncoderConfig,
    /// Forward spectral geometry.
    pub spectral: SpectralConfig,
    /// Required geometry of the borrowed codec quantizer.
    pub quantizer: Geometry,
}
impl AudioEncoderConfig {
    /// Validates component geometry before construction or waveform allocation.
    pub fn validate(&self) -> Result<(), Error> {
        self.neural.validate()?;
        self.spectral.validate()?;
        if self.quantizer.channels != self.neural.latent_size
            || self.quantizer.depth == 0
            || i32::try_from(self.quantizer.depth).is_err()
            || self.quantizer.cardinality <= 0
            || self
                .quantizer
                .channels
                .checked_mul(self.quantizer.cardinality)
                .is_none()
            || self.spectral.fft_size != self.neural.fft_size
        {
            return Err(invalid("inconsistent EAR encoder component geometry"));
        }
        self.samples_per_frame()?;
        Ok(())
    }
    /// Required input samples per encoded frame (released: 1764).
    pub fn samples_per_frame(&self) -> Result<i32, Error> {
        self.neural
            .frame_reduction()?
            .checked_mul(self.spectral.hop_length)
            .filter(|n| *n > 0)
            .ok_or_else(|| invalid("EAR encoder sample cadence overflows"))
    }
}

/// Finite mono waveform encoding through shared STFT, ConvNeXt and PRVQ equations.
///
/// Quantization borrows the same codec codebooks used by decoding. This component
/// declares no second learned copy, streaming cache or implicit padding policy.
#[derive(Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct AudioEncoder<B: NeuralBackend, E = NeuralEncoder<B>> {
    /// Independently materializable learned encoder units.
    pub neural: E,
    #[parameter(skip)]
    spectral: PeriodicHannStft<B::Tensor>,
    #[parameter(skip)]
    config: AudioEncoderConfig,
}
impl<B: NeuralBackend> AudioEncoder<B> {
    /// Declares encoder parameters and derived spectral constants. `prefix` is
    /// the codec namespace, e.g. `tts_model.audio_codec`.
    pub fn new(
        config: AudioEncoderConfig,
        prefix: &str,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        config.validate()?;
        B::require_operator_capabilities(
            "EAR waveform encoder",
            PeriodicHannStft::<B::Tensor>::required_capabilities(),
        )?;
        if prefix.is_empty() {
            return Err(invalid("empty EAR codec parameter prefix"));
        }
        let neural =
            NeuralEncoder::<B>::new(config.neural.clone(), &format!("{prefix}.encoder"), context)?;
        Self::from_parts(config, neural, context)
    }
}

impl<B: NeuralBackend, E: EncoderExecution<B>> AudioEncoder<B, E> {
    /// Consumes the learned encoder and creates only derived spectral constants.
    /// Geometry is validated before allocating those constants.
    pub fn from_parts(
        config: AudioEncoderConfig,
        neural: E,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        config.validate()?;
        if neural.config() != &config.neural {
            return Err(invalid("EAR encoder units disagree with retained geometry"));
        }
        B::require_operator_capabilities(
            "EAR waveform encoder",
            PeriodicHannStft::<B::Tensor>::required_capabilities(),
        )?;
        Ok(Self {
            neural,
            spectral: PeriodicHannStft::new(
                config.spectral.fft_size,
                config.spectral.hop_length,
                context,
            )?,
            config,
        })
    }
    /// Retained normalized encoder geometry.
    pub fn config(&self) -> &AudioEncoderConfig {
        &self.config
    }

    /// Encodes a complete finite mono waveform into unquantized latent vectors.
    /// Input samples must be a positive multiple of the codec cadence; callers
    /// make any final padding explicit before this operation.
    pub fn latent(
        &mut self,
        pcm: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        if self.neural.config() != &self.config.neural {
            return Err(invalid("EAR encoder units disagree with retained geometry"));
        }
        if !matches!(pcm.shape(), [batch, 1, samples] if *batch > 0 && *samples > 0 && *samples % self.config.samples_per_frame()? == 0)
        {
            return Err(invalid(
                "EAR encoder requires mono PCM with a positive whole-frame sample count",
            ));
        }
        let (real, imaginary) = self.spectral.analyze(pcm, context)?;
        let spectrum = B::Tensor::concatenate(&[real, imaginary], 1, context)?;
        self.neural.forward(&spectrum, context)
    }

    /// Produces `[batch, frames, depth]` codes using borrowed fixed-mean tables.
    /// The finite encoder has no mutable state and a failed invocation can be
    /// resubmitted after the execution owner establishes native completion.
    pub fn encode<Q: QuantizerExecution<B>>(
        &mut self,
        pcm: &B::Tensor,
        quantizer: &mut Q,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        if quantizer.geometry() != self.config.quantizer {
            return Err(invalid(
                "EAR encoder quantizer disagrees with retained geometry",
            ));
        }
        Ok(quantizer
            .quantize(&self.latent(pcm, context)?, context)?
            .codes)
    }
}

/// Normalized geometry for the released residual-quantizer waveform decoder.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioDecoderConfig {
    /// Fixed-mean residual codebook geometry.
    pub quantizer: Geometry,
    /// Neural stages in decoder order, from latent vectors to spectra.
    pub neural: DecoderConfig,
    /// Inverse-transform and overlap geometry.
    pub spectral: SpectralConfig,
}

impl SnapshotValue for AudioDecoderConfig {
    fn estimate_snapshot_value(&self) -> Result<SnapshotEstimate, BackendFailure> {
        snapshot_value_estimate::<Self>([self.neural.estimate_snapshot_value()])
    }
    fn copy_snapshot_value(&self, retention: &SnapshotReservation) -> Result<Self, BackendFailure> {
        Ok(Self {
            quantizer: self.quantizer,
            spectral: self.spectral,
            neural: self.neural.copy_snapshot_value(retention)?,
        })
    }
}

impl AudioDecoderConfig {
    /// Validates cross-component geometry without allocating native resources.
    pub fn validate(&self) -> Result<(), Error> {
        self.neural.validate()?;
        self.spectral.validate()?;
        if self.quantizer.depth == 0
            || i32::try_from(self.quantizer.depth).is_err()
            || self.quantizer.cardinality <= 0
            || self.quantizer.channels != self.neural.latent_size
            || self
                .quantizer
                .cardinality
                .checked_mul(self.quantizer.channels)
                .is_none()
            || self.spectral.fft_size != self.neural.fft_size
        {
            return Err(invalid("inconsistent EAR decoder component geometry"));
        }
        self.samples_per_frame()?;
        Ok(())
    }

    /// PCM cadence before the separately declared startup delay and flush tail.
    pub fn samples_per_frame(&self) -> Result<i32, Error> {
        self.neural
            .rates
            .iter()
            .try_fold(self.spectral.hop_length, |n, rate| {
                n.checked_mul(*rate)
                    .filter(|n| *n > 0)
                    .ok_or_else(|| invalid("EAR decoder sample cadence exceeds tensor geometry"))
            })
    }
}

/// Parameter-free neural and spectral carry for one batch of finite streams.
#[derive(Debug, Clone)]
pub struct AudioDecoderState<T: Tensor, N = DecoderState<T>> {
    config: AudioDecoderConfig,
    neural: N,
    spectral: SpectralState<T>,
    batch: i32,
    frames: u64,
}

impl<T: Tensor + SnapshotValue, N: SnapshotValue> SnapshotValue for AudioDecoderState<T, N> {
    fn estimate_snapshot_value(&self) -> Result<SnapshotEstimate, BackendFailure> {
        snapshot_value_estimate::<Self>([
            self.config.estimate_snapshot_value(),
            self.neural.estimate_snapshot_value(),
            self.spectral.estimate_snapshot_value(),
        ])
    }
    fn copy_snapshot_value(&self, retention: &SnapshotReservation) -> Result<Self, BackendFailure> {
        Ok(Self {
            config: self.config.copy_snapshot_value(retention)?,
            neural: self.neural.copy_snapshot_value(retention)?,
            spectral: self.spectral.copy_snapshot_value(retention)?,
            batch: self.batch,
            frames: self.frames,
        })
    }
}

impl<T: Tensor, N: RetainedTensorRoots<T>> RetainedTensorRoots<T> for AudioDecoderState<T, N> {
    fn visit_retained_tensors<'a>(&'a self, visitor: &mut dyn FnMut(&'a T)) {
        self.neural.visit_retained_tensors(visitor);
        self.spectral.visit_retained_tensors(visitor);
    }
}

impl<T: Tensor, N> AudioDecoderState<T, N> {
    /// Changes only the selected neural-state representation, preserving waveform clocks
    /// and spectral carry. The selected execution provider validates neural ownership.
    pub fn map_neural<M>(self, map: impl FnOnce(N) -> M) -> AudioDecoderState<T, M> {
        AudioDecoderState {
            config: self.config,
            neural: map(self.neural),
            spectral: self.spectral,
            batch: self.batch,
            frames: self.frames,
        }
    }
    /// Fixed number of independently decoded streams.
    pub const fn batch(&self) -> i32 {
        self.batch
    }
    /// Number of accepted latent/code frames per batch row.
    pub const fn frames_seen(&self) -> u64 {
        self.frames
    }
    /// Number of committed-candidate output samples, including flush samples.
    pub fn samples_emitted(&self) -> u64 {
        self.spectral.samples_emitted()
    }
    /// Whether the spectral tail has been drained.
    pub fn is_finished(&self) -> bool {
        self.spectral.is_finished()
    }
    /// Independently inspectable convolution carry; no learned parameters are retained.
    pub const fn neural(&self) -> &N {
        &self.neural
    }
    /// Independently inspectable compact spectral history.
    pub const fn spectral(&self) -> &SpectralState<T> {
        &self.spectral
    }
}

/// Selected waveform execution over ordinary or stage-local unpublished neural carry.
pub trait AudioDecoderExecution<B: NeuralBackend> {
    /// Only the neural state owned by the selected execution provider.
    type NeuralState: Clone + RetainedTensorRoots<B::Tensor>;
    /// Complete waveform geometry.
    fn config(&self) -> &AudioDecoderConfig;
    /// Empty fixed-batch carry; no learned parameters are copied.
    fn initial_state(
        &self,
        batch: i32,
    ) -> Result<AudioDecoderState<B::Tensor, Self::NeuralState>, Error>;
    /// Proposes one code chunk without adopting its neural or spectral carry.
    fn step(
        &mut self,
        codes: &B::Tensor,
        state: &AudioDecoderState<B::Tensor, Self::NeuralState>,
        constrain: bool,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(B::Tensor, AudioDecoderState<B::Tensor, Self::NeuralState>), Error>;
    /// Drains retained overlap without consuming a code frame.
    fn finish(
        &mut self,
        state: &AudioDecoderState<B::Tensor, Self::NeuralState>,
        constrain: bool,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<
        (
            Option<B::Tensor>,
            AudioDecoderState<B::Tensor, Self::NeuralState>,
        ),
        Error,
    >;
}

/// Shared spectral equations and geometry for independently selected neural execution.
/// Derived constants are retained once and contain no learned checkpoint parameters.
pub struct AudioDecoderKernel<B: NeuralBackend> {
    spectral: EarSpectralDecoder<B>,
    config: AudioDecoderConfig,
}
impl<B: NeuralBackend> AudioDecoderKernel<B> {
    /// Validates normalized geometry before creating derived spectral constants.
    pub fn new(
        config: AudioDecoderConfig,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        config.validate()?;
        B::require_operator_capabilities(
            "EAR waveform decoder",
            config.spectral.required_capabilities(),
        )?;
        Ok(Self {
            spectral: EarSpectralDecoder::new(config.spectral, context)?,
            config,
        })
    }
    /// Exact cross-component geometry.
    pub fn config(&self) -> &AudioDecoderConfig {
        &self.config
    }
    /// Streaming startup delay, emitted as a tail during drain.
    pub fn delay_samples(&self) -> i32 {
        self.spectral.delay_samples()
    }
    pub(crate) fn validate_components(
        &self,
        neural: &DecoderConfig,
        quantizer: Geometry,
    ) -> Result<(), Error> {
        if neural != &self.config.neural || quantizer != self.config.quantizer {
            return Err(invalid(
                "EAR decoder components disagree with retained geometry",
            ));
        }
        Ok(())
    }
    pub(crate) fn initial_state<N>(
        &self,
        neural: N,
        batch: i32,
    ) -> Result<AudioDecoderState<B::Tensor, N>, Error> {
        Ok(AudioDecoderState {
            config: self.config.clone(),
            neural,
            spectral: self.spectral.initial_state(batch)?,
            batch,
            frames: 0,
        })
    }
    pub(crate) fn validate_state<N>(
        &self,
        state: &AudioDecoderState<B::Tensor, N>,
    ) -> Result<(), Error> {
        if state.config != self.config {
            return Err(invalid(
                "EAR decoder state belongs to different component geometry",
            ));
        }
        Ok(())
    }
    pub(crate) fn prepare_step<N>(
        &self,
        codes: &B::Tensor,
        state: &AudioDecoderState<B::Tensor, N>,
    ) -> Result<u64, Error> {
        self.validate_state(state)?;
        if state.is_finished() {
            return Err(Error::StreamFinished);
        }
        if codes.shape().len() != 3 || codes.dim(0) != state.batch || codes.dim(1) <= 0 {
            return Err(invalid(
                "EAR codes require a nonempty [batch, frames, depth] chunk",
            ));
        }
        state
            .frames
            .checked_add(codes.dim(1) as u64)
            .ok_or_else(|| invalid("EAR code frame clock exhausted"))
    }
    pub(crate) fn render<N>(
        &self,
        raw: &B::Tensor,
        neural: N,
        state: &AudioDecoderState<B::Tensor, N>,
        frames: u64,
        constrain: bool,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(B::Tensor, AudioDecoderState<B::Tensor, N>), Error> {
        let (pcm, spectral) =
            self.spectral
                .step(raw, &state.spectral, false, constrain, context)?;
        Ok((
            pcm,
            AudioDecoderState {
                config: state.config.clone(),
                neural,
                spectral,
                batch: state.batch,
                frames,
            },
        ))
    }
    pub(crate) fn drain<N>(
        &self,
        neural: N,
        state: &AudioDecoderState<B::Tensor, N>,
        constrain: bool,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(Option<B::Tensor>, AudioDecoderState<B::Tensor, N>), Error> {
        let (pcm, spectral) = self.spectral.finish(&state.spectral, constrain, context)?;
        let output = (pcm.dim(2) != 0).then_some(pcm);
        Ok((
            output,
            AudioDecoderState {
                config: state.config.clone(),
                neural,
                spectral,
                batch: state.batch,
                frames: state.frames,
            },
        ))
    }
}

/// Codebooks, neural units and spectral reconstruction composed without a scheduler.
#[derive(Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct AudioDecoder<B: NeuralBackend, D = NeuralDecoder<B>, Q = ResidualQuantizer<B>> {
    /// Selected codebook execution, shared with finite waveform encoding.
    pub quantizer: Q,
    /// Selected execution of upsamplers, ConvNeXt blocks and spectral projection.
    pub neural: D,
    #[parameter(skip)]
    kernel: AudioDecoderKernel<B>,
}
impl<B: NeuralBackend> AudioDecoder<B> {
    /// Declares unloaded learned parameters and derived spectral constants.
    pub fn new(
        config: AudioDecoderConfig,
        prefix: &str,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        config.validate()?;
        B::require_operator_capabilities(
            "EAR waveform decoder",
            config.spectral.required_capabilities(),
        )?;
        if prefix.is_empty() {
            return Err(invalid("empty EAR codec parameter prefix"));
        }
        let quantizer =
            ResidualQuantizer::<B>::new(config.quantizer, &format!("{prefix}.prvq"), context)?;
        let neural =
            NeuralDecoder::<B>::new(config.neural.clone(), &format!("{prefix}.decoder"), context)?;
        Self::from_parts(config, quantizer, neural, context)
    }
}
impl<B: NeuralBackend, D: DecoderExecution<B>, Q: QuantizerExecution<B>> AudioDecoder<B, D, Q> {
    /// Consumes selected learned providers and creates only derived spectral constants.
    pub fn from_parts(
        config: AudioDecoderConfig,
        quantizer: Q,
        neural: D,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        config.validate()?;
        if neural.config() != &config.neural || quantizer.geometry() != config.quantizer {
            return Err(invalid(
                "EAR decoder components disagree with retained geometry",
            ));
        }
        Ok(Self {
            quantizer,
            neural,
            kernel: AudioDecoderKernel::new(config, context)?,
        })
    }
    /// Streaming startup delay in PCM samples; draining emits this many tail samples.
    pub fn delay_samples(&self) -> i32 {
        self.kernel.delay_samples()
    }
    /// Evaluates a finite sequence using the reference offline edge policy.
    /// Its boundary normalization differs from a drained streaming sequence.
    pub fn forward(
        &mut self,
        codes: &B::Tensor,
        constrain: bool,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.kernel
            .validate_components(self.neural.config(), self.quantizer.geometry())?;
        let latent = self.quantizer.decode(codes, context)?;
        let raw = self.neural.forward(&latent, context)?;
        self.kernel.spectral.ordinary(&raw, constrain, context)
    }
}
impl<B: NeuralBackend, D: DecoderExecution<B>, Q: QuantizerExecution<B>> AudioDecoderExecution<B>
    for AudioDecoder<B, D, Q>
{
    type NeuralState = DecoderState<B::Tensor>;
    fn config(&self) -> &AudioDecoderConfig {
        self.kernel.config()
    }
    fn initial_state(&self, batch: i32) -> Result<AudioDecoderState<B::Tensor>, Error> {
        self.kernel
            .validate_components(self.neural.config(), self.quantizer.geometry())?;
        self.kernel
            .initial_state(self.neural.initial_state()?, batch)
    }
    fn step(
        &mut self,
        codes: &B::Tensor,
        state: &AudioDecoderState<B::Tensor>,
        constrain: bool,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(B::Tensor, AudioDecoderState<B::Tensor>), Error> {
        self.kernel
            .validate_components(self.neural.config(), self.quantizer.geometry())?;
        let frames = self.kernel.prepare_step(codes, state)?;
        let latent = self.quantizer.decode(codes, context)?;
        let (raw, neural) = self.neural.step(&latent, &state.neural, context)?;
        self.kernel
            .render(&raw, neural, state, frames, constrain, context)
    }
    fn finish(
        &mut self,
        state: &AudioDecoderState<B::Tensor>,
        constrain: bool,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(Option<B::Tensor>, AudioDecoderState<B::Tensor>), Error> {
        self.kernel
            .validate_components(self.neural.config(), self.quantizer.geometry())?;
        self.kernel.validate_state(state)?;
        if state.is_finished() {
            return Ok((None, state.clone()));
        }
        let neural = self.neural.finish(&state.neural)?;
        self.kernel.drain(neural, state, constrain, context)
    }
}

fn invalid(message: &str) -> Error {
    Error::InvalidShape(message.into())
}
