//! Coordinated waveform composition over stage-owned neural decoder carry.
//!
//! The wrapper uses the ordinary waveform kernel and the existing component
//! pipeline. It owns no scheduler, parameters, committed state or epoch allocator.
use crate::{
    ear::{AudioDecoderConfig, AudioDecoderExecution, AudioDecoderKernel, AudioDecoderState},
    ear_decoder::DecoderConfig,
    prvq_layered::QuantizerExecution,
    Error,
};
use eredu_core::{BackendFailure, DistributedCommitEpoch, DistributedCommitOutcome};
use eredu_nn::{NeuralBackend, RetainedTensorRoots, Tensor, TensorElementType};
use eredu_runtime::replicated_execution::ReplicatedCoordinator;

const PROTOCOL: u64 = 0x45415250434d0001;

/// Selected neural stage over the existing distributed execution mechanism.
pub trait DecoderPipelineExecution<B: NeuralBackend> {
    /// Only the mutable convolution carry owned by this stage.
    type State: Clone + RetainedTensorRoots<B::Tensor>;
    /// Exact global neural geometry.
    fn config(&self) -> &DecoderConfig;
    /// Validates local ownership and lifecycle without executing learned operations.
    fn validate_state(&self, parent: &Self::State, batch: i32, finished: bool)
        -> Result<(), Error>;
    /// Empty stage-local convolution carry.
    fn initial_state(&self) -> Result<Self::State, Error>;
    /// Releases local carry without acquiring parameters or producing a spectral tail.
    fn finish(&self, parent: &Self::State) -> Result<Self::State, Error>;
    /// Runs a candidate through shared transport, completion and commit agreement.
    fn forward(
        &mut self,
        latent: &B::Tensor,
        state: &mut Self::State,
        descriptor: [u64; 8],
        epoch: DistributedCommitEpoch,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(B::Tensor, DistributedCommitOutcome), BackendFailure>;
}

/// Borrowed authority for one waveform step or drain attempt.
///
/// Every participant supplies matching recorded identity and a fresh epoch.
/// Admission consumes that epoch even if subsequent work fails. Retries, replay
/// and forks require fresh authority; waveform snapshots never refund it.
/// Quantizer providers that use collectives remain responsible for agreeing
/// their own acquisition failures before entering those collectives.
pub struct PipelineAudioDecoder<'a, B: NeuralBackend, P, Q, C> {
    kernel: &'a AudioDecoderKernel<B>,
    neural: &'a mut P,
    quantizer: &'a mut Q,
    coordinator: &'a mut C,
    identity: [u64; 3],
    epoch: Option<DistributedCommitEpoch>,
}
impl<'a, B: NeuralBackend, P, Q, C> PipelineAudioDecoder<'a, B, P, Q, C> {
    /// Borrows retained mechanisms and explicit authority without native execution.
    pub fn new(
        kernel: &'a AudioDecoderKernel<B>,
        neural: &'a mut P,
        quantizer: &'a mut Q,
        coordinator: &'a mut C,
        identity: [u64; 3],
        epoch: DistributedCommitEpoch,
    ) -> Self {
        Self {
            kernel,
            neural,
            quantizer,
            coordinator,
            identity,
            epoch: Some(epoch),
        }
    }
    /// Whether this wrapper has admitted a step or drain attempt.
    pub fn is_consumed(&self) -> bool {
        self.epoch.is_none()
    }
}
fn agreed<T, B, C>(
    coordinator: &mut C,
    descriptor: [u64; 8],
    local: Result<T, Error>,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<T, Error>
where
    B: NeuralBackend,
    C: ReplicatedCoordinator<B::Tensor, <B::Tensor as Tensor>::Context>,
{
    if !coordinator.agree(descriptor, local.is_ok(), context)? {
        return Err(local.err().unwrap_or_else(|| {
            Error::InvalidShape(
                "waveform participants differ in identity, input or execution success".into(),
            )
        }));
    }
    local
}
impl<B, P, Q, C> PipelineAudioDecoder<'_, B, P, Q, C>
where
    B: NeuralBackend,
    P: DecoderPipelineExecution<B>,
    Q: QuantizerExecution<B>,
    C: ReplicatedCoordinator<B::Tensor, <B::Tensor as Tensor>::Context>,
{
    fn admit(
        &mut self,
        state: &AudioDecoderState<B::Tensor, P::State>,
        constrain: bool,
        drain: bool,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(DistributedCommitEpoch, [u64; 8]), Error> {
        let descriptor = [
            PROTOCOL,
            self.identity[0],
            self.identity[1],
            self.identity[2],
            self.epoch.map_or(0, DistributedCommitEpoch::value),
            state.frames_seen(),
            state.samples_emitted(),
            u64::from(drain),
        ];
        let preflight = (|| {
            if self.epoch.is_none() {
                return Err(Error::InvalidShape(
                    "waveform invocation has consumed its epoch".into(),
                ));
            }
            self.kernel
                .validate_components(self.neural.config(), self.quantizer.geometry())?;
            self.quantizer.validate_coordinated_execution()?;
            self.kernel.validate_state(state)?;
            self.neural
                .validate_state(state.neural(), state.batch(), state.is_finished())
        })();
        agreed::<_, B, C>(self.coordinator, descriptor, preflight, context)?;
        let epoch = self.epoch.take().expect("agreed unused waveform epoch");
        let config = self.kernel.config();
        agreed::<_, B, C>(
            self.coordinator,
            [
                PROTOCOL + 1,
                u64::from(constrain),
                u64::from(state.is_finished()),
                config.quantizer.depth as u64,
                config.quantizer.cardinality as u64,
                config.quantizer.channels as u64,
                state.batch() as u64,
                0,
            ],
            Ok(()),
            context,
        )?;
        let neural = &config.neural;
        agreed::<_, B, C>(
            self.coordinator,
            [
                PROTOCOL + 10,
                neural.latent_size as u64,
                neural.base_hidden_size as u64,
                neural.blocks_per_stage as u64,
                neural.kernel_size as u64,
                neural.fft_size as u64,
                neural.groups as u64,
                neural.rates.len() as u64,
            ],
            Ok(()),
            context,
        )?;
        agreed::<_, B, C>(
            self.coordinator,
            [
                PROTOCOL + 11,
                config.spectral.fft_size as u64,
                config.spectral.hop_length as u64,
                neural.channel_mult.len() as u64,
                0,
                0,
                0,
                0,
            ],
            Ok(()),
            context,
        )?;
        for (index, (&rate, &multiplier)) in
            neural.rates.iter().zip(&neural.channel_mult).enumerate()
        {
            agreed::<_, B, C>(
                self.coordinator,
                [
                    PROTOCOL + 12,
                    index as u64,
                    rate as u64,
                    multiplier as u64,
                    0,
                    0,
                    0,
                    0,
                ],
                Ok(()),
                context,
            )?;
        }
        Ok((epoch, descriptor))
    }
}
impl<B, P, Q, C> AudioDecoderExecution<B> for PipelineAudioDecoder<'_, B, P, Q, C>
where
    B: NeuralBackend,
    P: DecoderPipelineExecution<B>,
    Q: QuantizerExecution<B>,
    C: ReplicatedCoordinator<B::Tensor, <B::Tensor as Tensor>::Context>,
{
    type NeuralState = P::State;
    fn config(&self) -> &AudioDecoderConfig {
        self.kernel.config()
    }
    fn initial_state(&self, batch: i32) -> Result<AudioDecoderState<B::Tensor, P::State>, Error> {
        self.kernel
            .validate_components(self.neural.config(), self.quantizer.geometry())?;
        self.kernel
            .initial_state(self.neural.initial_state()?, batch)
    }
    fn step(
        &mut self,
        codes: &B::Tensor,
        state: &AudioDecoderState<B::Tensor, P::State>,
        constrain: bool,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(B::Tensor, AudioDecoderState<B::Tensor, P::State>), Error> {
        let (epoch, mut descriptor) = self.admit(state, constrain, false, context)?;
        let preflight = (|| {
            let frames = self.kernel.prepare_step(codes, state)?;
            if codes.element_type() != Some(TensorElementType::I32)
                || codes.dim(2) <= 0
                || codes.dim(2) as usize > self.config().quantizer.depth
            {
                return Err(Error::InvalidShape(
                    "waveform pipeline requires I32 codes with admitted depth".into(),
                ));
            }
            Ok(frames)
        })();
        descriptor[0] = PROTOCOL + 2;
        let frames = agreed::<_, B, C>(self.coordinator, descriptor, preflight, context)?;
        agreed::<_, B, C>(
            self.coordinator,
            [
                PROTOCOL + 3,
                codes.dim(0) as u64,
                codes.dim(1) as u64,
                codes.dim(2) as u64,
                frames,
                0,
                0,
                0,
            ],
            Ok(()),
            context,
        )?;
        // Code observation covers exactly the supplied chunk geometry. Each content
        // agreement spends the shared communication budget; exact values prevent
        // independent rank decisions.
        let values = codes
            .to_i32_vec(context)
            .map_err(Error::from)
            .and_then(|values| {
                let expected = codes
                    .shape()
                    .iter()
                    .try_fold(1usize, |n, &d| n.checked_mul(d as usize));
                if expected != Some(values.len())
                    || values
                        .iter()
                        .any(|&value| value < 0 || value >= self.config().quantizer.cardinality)
                {
                    return Err(Error::InvalidShape(
                        "waveform codes are outside the admitted codebook domain".into(),
                    ));
                }
                Ok(values)
            });
        descriptor[0] = PROTOCOL + 4;
        let values = agreed::<_, B, C>(self.coordinator, descriptor, values, context)?;
        for (index, chunk) in values.chunks(6).enumerate() {
            let mut content = [0u64; 8];
            content[0] = PROTOCOL + 5;
            content[1] = index as u64;
            for (slot, value) in content[2..].iter_mut().zip(chunk) {
                *slot = *value as u64;
            }
            agreed::<_, B, C>(self.coordinator, content, Ok(()), context)?;
        }
        let latent = self.quantizer.decode(codes, context);
        descriptor[0] = PROTOCOL + 6;
        let latent = agreed::<_, B, C>(self.coordinator, descriptor, latent, context)?;
        let mut neural = state.neural().clone();
        descriptor[0] = PROTOCOL + 7;
        let raw = self
            .neural
            .forward(&latent, &mut neural, descriptor, epoch, context)
            .map_err(Error::from)
            .and_then(|(raw, outcome)| {
                if outcome != DistributedCommitOutcome::Committed(epoch) {
                    return Err(Error::InvalidShape(
                        "waveform neural candidate was not committed".into(),
                    ));
                }
                Ok(raw)
            });
        let raw = agreed::<_, B, C>(self.coordinator, descriptor, raw, context)?;
        let rendered = self
            .kernel
            .render(&raw, neural, state, frames, constrain, context);
        descriptor[0] = PROTOCOL + 8;
        agreed::<_, B, C>(self.coordinator, descriptor, rendered, context)
    }
    fn finish(
        &mut self,
        state: &AudioDecoderState<B::Tensor, P::State>,
        constrain: bool,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(Option<B::Tensor>, AudioDecoderState<B::Tensor, P::State>), Error> {
        let (_, mut descriptor) = self.admit(state, constrain, true, context)?;
        if state.is_finished() {
            return Ok((None, state.clone()));
        }
        let drained = self
            .neural
            .finish(state.neural())
            .and_then(|neural| self.kernel.drain(neural, state, constrain, context));
        descriptor[0] = PROTOCOL + 9;
        agreed::<_, B, C>(self.coordinator, descriptor, drained, context)
    }
}
