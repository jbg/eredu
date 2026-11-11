//! Streaming synthesis over independently acquired LiquidCodec decoder weights.

use super::*;
use eredu_core::BackendFailure;
use eredu_runtime::{
    ExecutionGraph, ExecutionGroupSpec, ExecutionUnitAddressMap, ExecutionUnitLayout,
    LayerwiseAcquireError, LayerwisePolicy, LayerwisePolicyForward, RuntimeLayerState,
};
use std::iter;

/// One learned synthesis resource.
#[derive(Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct DecoderModule<B: NeuralBackend>(Module<B>);

#[derive(Parameterized)]
#[parameterized(tensor = "B::Tensor")]
enum Module<B: NeuralBackend> {
    Quantizer(ScalarDecoder<B>),
    Input(B::Linear),
    Block(BackboneBlock<B>),
    Output(DecoderOutput<B>),
}

#[derive(Parameterized)]
#[parameterized(tensor = "B::Tensor")]
struct DecoderOutput<B: NeuralBackend> {
    norm: B::Normalization,
    polar: B::Linear,
    window: Parameter<B::Tensor>,
    #[parameter(skip)]
    synthesis: Synthesis<B>,
}

fn build<B: NeuralBackend>(
    d: &Declaration,
    unit: DecoderUnit,
    ctx: &<B::Tensor as Tensor>::Context,
) -> Result<DecoderModule<B>, Error> {
    Ok(DecoderModule(match unit {
        DecoderUnit::QuantizerOutput => Module::Quantizer(ScalarDecoder::new(&d.quantizer, ctx)?),
        DecoderUnit::Input => Module::Input(B::linear(d.from_quantizer.clone(), ctx)?),
        DecoderUnit::Block(i) => Module::Block(BackboneBlock::from_spec(
            d.decoder[i].clone(),
            Some(d.config.decoder_window),
            ctx,
        )?),
        DecoderUnit::Output => Module::Output(DecoderOutput {
            norm: B::normalization(d.decoder_norm.clone(), ctx)?,
            polar: B::linear(d.polar.clone(), ctx)?,
            window: Parameter::unloaded(d.window.clone(), &[d.config.stft_n_fft], ctx)?,
            synthesis: Synthesis::new(&d.config, ctx)?,
        }),
    }))
}

impl Declaration {
    /// Decoder invocation order and distinct learned storage owners.
    pub fn decoder_invocations(&self) -> Result<ExecutionUnitAddressMap, Error> {
        let group = "liquidcodec.decoder";
        let graph = ExecutionGraph::new(vec![ExecutionGroupSpec::root(group)], group)
            .map_err(BackendFailure::from_error)?;
        let layout = ExecutionUnitLayout::new(&graph, [self.decoder_units().count()])
            .map_err(BackendFailure::from_error)?;
        Ok(ExecutionUnitAddressMap::identity(&layout))
    }
}

/// Shared streaming decoder interface over caller-owned neural and spectral state.
pub trait DecoderExecution<B: NeuralBackend, S> {
    /// Attaches empty neural state and a new spectral continuation.
    fn decoder_state(&self, neural: S, batch: i32) -> Result<DecoderState<S, B::Tensor>, Error>;
    /// Advances complete codebooks or flushes pending overlap.
    fn decode_chunk(
        &mut self,
        codes: Option<&B::Tensor>,
        state: &mut DecoderState<S, B::Tensor>,
        finalize: bool,
        ctx: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>;
}

/// LiquidCodec streaming synthesis over a shared parameter policy.
pub struct PolicyDecoder<B: NeuralBackend, P> {
    declaration: Declaration,
    invocations: ExecutionUnitAddressMap,
    policy: P,
    unweighted_quantizer: Option<ScalarDecoder<B>>,
}

impl<B: NeuralBackend, P> PolicyDecoder<B, P> {
    /// Constructs one unloaded native module per declared owner.
    pub fn resident_units(
        d: &Declaration,
        ctx: &<B::Tensor as Tensor>::Context,
    ) -> Result<Vec<DecoderModule<B>>, Error> {
        d.decoder_units().map(|unit| build(d, unit, ctx)).collect()
    }

    /// Attaches the retained parameter policy to the architecture traversal.
    pub fn new(
        declaration: Declaration,
        invocations: ExecutionUnitAddressMap,
        policy: P,
        ctx: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        B::require_operator_capabilities(
            "LiquidCodec neural decoder",
            eredu_nn::NeuralOperatorCapabilities::SIGMOID,
        )?;
        if invocations != declaration.decoder_invocations()? {
            return Err(invalid(
                "LiquidCodec invocations differ from the retained decoder",
            ));
        }
        let unweighted_quantizer = if declaration.quantizer.output.is_none() {
            Some(ScalarDecoder::new(&declaration.quantizer, ctx)?)
        } else {
            None
        };
        Ok(Self {
            declaration,
            invocations,
            policy,
            unweighted_quantizer,
        })
    }
}

impl<B, S, P> DecoderExecution<B, S> for PolicyDecoder<B, P>
where
    B: NeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
    P: LayerwisePolicy<B, DecoderModule<B>>,
    P::Error: std::error::Error + Send + Sync + 'static,
{
    fn decoder_state(&self, neural: S, batch: i32) -> Result<DecoderState<S, B::Tensor>, Error> {
        decoder_state::<B, S>(
            &self.declaration.config,
            &self.declaration.decoder_state,
            neural,
            batch,
        )
    }

    fn decode_chunk(
        &mut self,
        codes: Option<&B::Tensor>,
        state: &mut DecoderState<S, B::Tensor>,
        finalize: bool,
        ctx: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let d = &self.declaration;
        advance_chunk(&d.config, codes, state, finalize, ctx, |codes, state| {
            let time = spectrum_time::<B, S>(&d.config, &d.decoder_state, codes, state)?;
            let mut execution = LayerwisePolicyForward::begin(&mut self.policy, codes, ctx)
                .map_err(BackendFailure::from_error)?;
            let mut hidden = match &mut self.unweighted_quantizer {
                Some(quantizer) => quantizer
                    .decode(codes, ctx)?
                    .transpose_axes(&[0, 2, 1], ctx)?,
                None => codes.clone(),
            };
            for (ordinal, (unit, address)) in d
                .decoder_units()
                .zip(self.invocations.logical_addresses())
                .enumerate()
            {
                let module = execution
                    .acquire(ordinal, address, |ctx| build(d, unit, ctx))
                    .map_err(|error| match error {
                        LayerwiseAcquireError::Architecture(error) => error,
                        LayerwiseAcquireError::Policy(error) => {
                            BackendFailure::from_error(error).into()
                        }
                    })?;
                let output = match (&mut module.0, unit) {
                    (Module::Quantizer(quantizer), DecoderUnit::QuantizerOutput) => quantizer
                        .decode(&hidden, ctx)?
                        .transpose_axes(&[0, 2, 1], ctx)?,
                    (Module::Input(input), DecoderUnit::Input) => expand_frames(
                        &input.forward(&hidden, ctx)?,
                        d.config.stft_frame_stack,
                        time,
                        ctx,
                    )?,
                    (Module::Block(block), DecoderUnit::Block(i)) => {
                        let layer = state.layer(i)?;
                        let output = block.forward(&hidden, layer, ctx)?;
                        execution
                            .complete(&output, layer.retained_values(), iter::once(&hidden))
                            .map_err(BackendFailure::from_error)?;
                        hidden = output;
                        continue;
                    }
                    (Module::Output(module), DecoderUnit::Output) => {
                        let norm = module.norm.forward(&hidden, ctx)?;
                        let polar = module
                            .polar
                            .forward(&norm, ctx)?
                            .transpose_axes(&[0, 2, 1], ctx)?;
                        let (magnitude, angle) = polar_parts(&polar, d.config.stft_n_fft, ctx)?;
                        module.synthesis.set_window(module.window.as_ref(), ctx)?;
                        let raw = module.synthesis.overlap_add(&magnitude, &angle, ctx)?;
                        execution
                            .complete(
                                &raw.audio,
                                iter::empty(),
                                [&hidden, &raw.envelope].into_iter(),
                            )
                            .map_err(BackendFailure::from_error)?;
                        execution
                            .finish(&raw.audio)
                            .map_err(BackendFailure::from_error)?;
                        return Ok(raw);
                    }
                    _ => return Err(invalid("LiquidCodec module differs from its invocation")),
                };
                execution
                    .complete(&output, iter::empty(), iter::once(&hidden))
                    .map_err(BackendFailure::from_error)?;
                hidden = output;
            }
            Err(invalid("LiquidCodec decoder has no synthesis output"))
        })
    }
}
