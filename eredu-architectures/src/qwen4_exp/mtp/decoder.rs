//! Layer-sized prediction owners using the target's mixer and feed-forward equations.
use super::*;
use crate::{
    decoder::{ComponentInstrumentation, DecoderBoundary},
    qwen4_exp::{
        config::Config,
        feed_forward::{FeedForwardSublayer, FeedForwardSublayerSpec},
        indexed::IndexedSublayerInput,
        qsa::QsaExecutionLimits,
        residual::{mixer_spec, ResidualBoundary},
        target::{Mixer, MixerSpec},
    },
};
use eredu_nn::{
    AttentionCache, DeclareLogicalParameters, DistributedNeuralBackend, GroupedNeuralBackend,
    LinearFormatSpec, RotaryPosition, TensorElementType,
};
use eredu_runtime::{
    ParameterProvider, RoutedBankId, RoutedObservationPoints, RuntimeAppendStreams,
    RuntimeStateComponents, StateLayout,
};

/// Finite prediction invocation and QSA scratch bounds, separate from target state.
#[derive(Debug, Clone, Copy)]
pub struct PredictionLimits {
    /// Admitted batch, chunk and coordinator scratch.
    pub qsa: QsaExecutionLimits,
    /// Summaries per bounded score tile.
    pub tile_blocks: i32,
    /// Scratch for one tiled selection.
    pub selection_workspace: u64,
    /// Activation/cache representation.
    pub element: TensorElementType,
}
/// Parameter slots shared by prediction invocations, without state or execution bounds.
#[derive(Debug, Clone)]
pub struct PredictionSharedSpec {
    /// Full-residual fusion with the next-token embedding.
    pub fusion: PredictionFusionSpec,
    /// Prediction-specific gated collapse, without target readout parameters.
    pub readout: eredu_nn::residual_streams::GatedResidualSpec,
}
impl PredictionSharedSpec {
    /// Builds the shared parameter namespace independently of decoder invocation limits.
    pub fn from_config(
        config: &Config,
        mut format: impl FnMut(&str) -> Result<LinearFormatSpec, Error>,
    ) -> Result<Self, Error> {
        let spec = Self {
            fusion: PredictionFusionSpec::from_config(config, &mut format)?,
            readout: mixer_spec(config, "mtp.hyper_connection_mixer", false, format)?,
        };
        spec.validate()?;
        Ok(spec)
    }
    /// Validates geometry and complete shared-slot identity before allocation.
    pub fn validate(&self) -> Result<(), Error> {
        self.logical_parameters().map(|_| ())
    }
}
impl DeclareLogicalParameters for PredictionSharedSpec {
    fn logical_parameters(&self) -> Result<Vec<eredu_nn::LogicalParameterDeclaration<'_>>, Error> {
        self.fusion.validate()?;
        self.readout.validate()?;
        if self.fusion.geometry != self.readout.geometry || self.readout.injection.is_some() {
            return Err(Error::backend(
                "prediction fusion/readout residual geometry mismatch",
            ));
        }
        let mut parameters = self.fusion.logical_parameters()?;
        parameters.extend(self.readout.logical_parameters()?);
        eredu_nn::validate_logical_parameter_topology(&parameters)
            .map_err(Error::backend_source)?;
        Ok(parameters)
    }
}

/// One independently materialized prediction depth and separately owned state slot.
#[derive(Debug, Clone)]
pub struct PredictionUnitSpec {
    /// Physical depth beneath `mtp.layers`.
    pub depth: usize,
    /// Same recurrent/indexed equations as the target, with prediction rotary policy.
    pub mixer: MixerSpec,
    /// Shared/routed branch in the independent prediction bank namespace.
    pub feed_forward: FeedForwardSublayerSpec,
}
/// Prediction-only modules; embedding and vocabulary projection stay on the target.
#[derive(Debug, Clone)]
pub struct PredictionSpec {
    /// Shared fusion and readout slots, independent of decoder state ownership.
    pub shared: PredictionSharedSpec,
    /// One bounded owner per configured prediction depth.
    pub units: Vec<PredictionUnitSpec>,
    /// Explicit finite invocation limits.
    pub limits: PredictionLimits,
}
impl PredictionSpec {
    /// Optional mechanisms required before allocating the prediction owners.
    pub fn required_operators(&self) -> eredu_nn::NeuralOperatorCapabilities {
        use eredu_nn::NeuralOperatorCapabilities as C;
        self.units.iter().fold(
            C::BROADCAST_TO.union(C::SIGMOID).union(C::CAST_FLOAT),
            |required, unit| required.union(unit.mixer.required_operators()),
        )
    }
    /// Attaches independent state, invocation limits and bank binding to retained slots.
    pub(crate) fn from_parameters(
        config: &Config,
        limits: PredictionLimits,
        bank: RoutedBankId,
        parameters: super::declarations::PredictionParameters,
    ) -> Result<Self, Error> {
        use crate::qwen4_exp::target::declarations::UnitParameters;
        let prediction = config
            .prediction
            .as_ref()
            .ok_or_else(|| Error::backend("artifact has no embedded prediction schedule"))?;
        if prediction.layers.is_empty()
            || limits.qsa.batch <= 0
            || limits.qsa.tokens <= 0
            || parameters.units.len() != prediction.layers.len()
        {
            return Err(Error::backend(
                "invalid prediction schedule or invocation bounds",
            ));
        }
        parameters.logical_parameters()?;
        for (unit, kind) in parameters.units.iter().zip(prediction.layers.iter()) {
            use crate::qwen4_exp::{config::LayerKind, target::declarations::MixerParameters};
            let valid = matches!(
                (unit, kind),
                (
                    UnitParameters::Decoder {
                        mixer: MixerParameters::Recurrent(_),
                        ..
                    },
                    LayerKind::Recurrent
                ) | (
                    UnitParameters::Decoder {
                        mixer: MixerParameters::Indexed { .. },
                        ..
                    },
                    LayerKind::Indexed
                )
            );
            if !valid {
                return Err(Error::backend(
                    "prediction mixer differs from the retained invocation schedule",
                ));
            }
        }
        let mut config = config.clone();
        config.attention.rotary.base = prediction.rope_theta;
        let units = parameters
            .units
            .into_iter()
            .map(|unit| {
                let UnitParameters::Decoder {
                    layer: depth,
                    mixer,
                    mut feed_forward,
                } = unit
                else {
                    return Err(Error::backend("prediction cannot own a lexical invocation"));
                };
                feed_forward.feed_forward.bank = bank;
                Ok(PredictionUnitSpec {
                    depth,
                    mixer: mixer.into_spec(
                        &config,
                        limits.tile_blocks,
                        limits.selection_workspace,
                        limits.qsa,
                        limits.element,
                    )?,
                    feed_forward,
                })
            })
            .collect::<Result<Vec<_>, Error>>()?;
        let spec = Self {
            shared: parameters.shared,
            units,
            limits,
        };
        spec.state_layout()?;
        Ok(spec)
    }
    /// Projects parameter scopes from the retained executable declaration.
    pub(crate) fn parameter_declaration(&self) -> super::declarations::PredictionParameters {
        use crate::qwen4_exp::target::declarations::{MixerParameters, UnitParameters};
        super::declarations::PredictionParameters {
            shared: self.shared.clone(),
            units: self
                .units
                .iter()
                .map(|unit| UnitParameters::Decoder {
                    layer: unit.depth,
                    mixer: match &unit.mixer {
                        MixerSpec::Recurrent(spec) => MixerParameters::Recurrent(spec.clone()),
                        MixerSpec::Indexed(spec) => MixerParameters::Indexed {
                            residual: spec.residual.clone(),
                            parameters: spec.parameters.clone(),
                        },
                    },
                    feed_forward: unit.feed_forward.clone(),
                })
                .collect(),
        }
    }

    /// Prediction state never aliases the target's lexical, recurrent or QSA slots.
    pub fn state_layout(&self) -> Result<StateLayout, Error> {
        let layout = self.state_geometry()?;
        self.declarations()?;
        Ok(layout)
    }
    fn declarations(&self) -> Result<Vec<eredu_nn::LogicalParameterDeclaration<'_>>, Error> {
        let mut parameters = self.shared.logical_parameters()?;
        for unit in &self.units {
            parameters.extend(unit.logical_parameters()?);
        }
        eredu_nn::validate_logical_parameter_topology(&parameters)
            .map_err(Error::backend_source)?;
        Ok(parameters)
    }
    fn state_geometry(&self) -> Result<StateLayout, Error> {
        self.shared.validate()?;
        if self.units.is_empty() || self.limits.qsa.batch <= 0 || self.limits.qsa.tokens <= 0 {
            return Err(Error::backend(
                "prediction shared geometry, schedule or bounds are invalid",
            ));
        }
        for (depth, unit) in self.units.iter().enumerate() {
            let geometry = match &unit.mixer {
                MixerSpec::Recurrent(mixer) => mixer.residual.geometry,
                MixerSpec::Indexed(mixer) => mixer.residual.geometry,
            };
            unit.validate()?;
            if unit.depth != depth
                || unit.feed_forward.feed_forward.bank
                    != self.units[0].feed_forward.feed_forward.bank
                || geometry != self.shared.fusion.geometry
                || unit.feed_forward.residual.geometry != geometry
            {
                return Err(Error::backend(
                    "prediction unit schedule or residual geometry mismatch",
                ));
            }
        }
        let policies = self
            .units
            .iter()
            .map(|u| u.mixer.state_policy())
            .collect::<Result<Vec<_>, _>>()?;
        StateLayout::new(
            eredu_core::LayerSchedule::new(policies.len(), policies).map_err(Error::backend)?,
        )
        .map_err(Error::backend)
    }
}
impl PredictionUnitSpec {
    /// Validates one complete decoder depth independently of backend resources.
    pub fn validate(&self) -> Result<(), Error> {
        self.logical_parameters().map(|_| ())
    }
}
impl DeclareLogicalParameters for PredictionUnitSpec {
    fn logical_parameters(&self) -> Result<Vec<eredu_nn::LogicalParameterDeclaration<'_>>, Error> {
        let geometry = match &self.mixer {
            MixerSpec::Recurrent(spec) => spec.residual.geometry,
            MixerSpec::Indexed(spec) => spec.residual.geometry,
        };
        if geometry != self.feed_forward.residual.geometry
            || self.feed_forward.feed_forward.layer != self.depth
        {
            return Err(Error::backend(
                "prediction mixer/feed-forward depth or residual geometry mismatch",
            ));
        }
        let mut parameters = self.mixer.logical_parameters()?;
        parameters.extend(self.feed_forward.logical_parameters()?);
        eredu_nn::validate_logical_parameter_topology(&parameters)
            .map_err(Error::backend_source)?;
        Ok(parameters)
    }
}
impl DeclareLogicalParameters for PredictionSpec {
    fn logical_parameters(&self) -> Result<Vec<eredu_nn::LogicalParameterDeclaration<'_>>, Error> {
        self.state_geometry()?;
        self.declarations()
    }
}

/// Shared weights remain pinned while one prediction depth is acquired.
#[derive(Debug, eredu_nn::Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct PredictionShared<B: NeuralBackend> {
    fusion: PredictionFusion<B>,
    readout: ResidualBoundary<B>,
    #[parameter(skip)]
    limits: PredictionLimits,
}
impl<B: NeuralBackend> PredictionShared<B> {
    pub(crate) fn geometry(&self) -> ResidualStreamGeometry {
        self.fusion.geometry
    }

    /// Allocates prediction fusion/readout only; no duplicate target vocabulary weights.
    pub fn new(
        spec: &PredictionSpec,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        spec.state_layout()?;
        B::require_operator_capabilities("qwen4_exp prediction", spec.required_operators())?;
        Ok(Self {
            fusion: PredictionFusion::new(spec.shared.fusion.clone(), context)?,
            readout: ResidualBoundary::new(spec.shared.readout.clone(), context)?,
            limits: spec.limits,
        })
    }
}
/// Materializable prediction block, consumed by the shared parameter-provider contract.
#[derive(Debug, eredu_nn::Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct PredictionUnit<B: GroupedNeuralBackend + DistributedNeuralBackend> {
    mixer: Mixer<B>,
    feed_forward: FeedForwardSublayer<B>,
    #[parameter(skip)]
    spec: PredictionUnitSpec,
    #[parameter(skip)]
    tensor_partition: Option<(usize, usize)>,
}
/// Current token embeddings and pre-collapse target/draft residual streams.
pub struct PredictionInput<'a, T> {
    /// Ordinary `[batch,tokens,hidden]` shared token/media embeddings.
    pub embeddings: &'a T,
    /// Complete `[batch,tokens,streams,hidden]` retained residual.
    pub residual: &'a T,
    /// Exact causal visibility for the current chunk.
    pub visible: Option<&'a [bool]>,
    /// Current rotary products or prediction cache offset.
    pub rotary: Option<RotaryPosition<'a, T>>,
}
/// Separate prediction capture and input to the target-owned vocabulary projection.
pub struct PredictionForward<T> {
    /// Pre-collapse streams conditioning subsequent draft steps.
    pub capture: T,
    /// Prediction-specific gated collapse, in ordinary hidden width.
    pub hidden: T,
}
impl<B: GroupedNeuralBackend + DistributedNeuralBackend> PredictionUnit<B> {
    pub(crate) fn specification(&self) -> &PredictionUnitSpec {
        &self.spec
    }

    /// Constructs a single independently streamable depth without acquiring experts.
    pub fn new(
        spec: PredictionUnitSpec,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        spec.validate()?;
        B::require_operator_capabilities(
            "qwen4_exp prediction mixer",
            spec.mixer.required_operators(),
        )?;
        Ok(Self {
            mixer: Mixer::new(spec.mixer.clone(), context)?,
            feed_forward: FeedForwardSublayer::new(spec.feed_forward.clone(), context)?,
            spec,
            tensor_partition: None,
        })
    }

    pub(crate) fn new_partitioned(
        spec: PredictionUnitSpec,
        rank: usize,
        ranks: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        if ranks == 0 || rank >= ranks {
            return Err(Error::backend("invalid prediction tensor rank or size"));
        }
        let mut unit = Self::new(spec, context)?;
        unit.tensor_partition = Some((rank, ranks));
        Ok(unit)
    }
    /// Runs fusion, the decoder and its own readout, retaining the uncollapsed result.
    /// The enclosing prediction driver owns checkpoint/rollback and completion.
    pub fn forward<S, P>(
        &mut self,
        shared: &mut PredictionShared<B>,
        input: PredictionInput<'_, B::Tensor>,
        state: &mut S,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<PredictionForward<B::Tensor>, Error>
    where
        S: AttentionCache<B::Tensor> + RuntimeStateComponents<B> + RuntimeAppendStreams<B>,
        P: ParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        if self.tensor_partition.is_some_and(|(_, ranks)| ranks > 1) {
            return Err(Error::backend(
                "rank-local prediction requires parallel execution",
            ));
        }
        self.forward_with(
            shared,
            input,
            state,
            provider,
            None,
            context,
            instrumentation,
            |feed_forward, hidden, points, provider, instrumentation| {
                feed_forward.forward(hidden, points, provider, context, instrumentation)
            },
        )
    }

    /// Runs localized mixer and feed-forward projections with their ordinary
    /// reductions, retaining replicated fusion, residual streams and readout.
    /// Expert exchange, when selected, belongs to the supplied provider.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_parallel<S, P>(
        &mut self,
        shared: &mut PredictionShared<B>,
        input: PredictionInput<'_, B::Tensor>,
        state: &mut S,
        provider: &mut P,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<PredictionForward<B::Tensor>, Error>
    where
        S: AttentionCache<B::Tensor> + RuntimeStateComponents<B> + RuntimeAppendStreams<B>,
        P: eredu_runtime::TensorParallelParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        let (rank, ranks) = self.tensor_partition.ok_or_else(|| {
            Error::backend("parallel prediction requires a retained tensor partition")
        })?;
        if B::parallel_rank(parallel) != rank || B::parallel_size(parallel) != ranks {
            return Err(Error::backend(
                "prediction tensor partition differs from collective rank or size",
            ));
        }
        self.forward_with(
            shared,
            input,
            state,
            provider,
            Some(parallel),
            context,
            instrumentation,
            |feed_forward, hidden, points, provider, instrumentation| {
                feed_forward.forward_parallel(
                    hidden,
                    points,
                    provider,
                    parallel,
                    context,
                    instrumentation,
                )
            },
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_with<S, P>(
        &mut self,
        shared: &mut PredictionShared<B>,
        input: PredictionInput<'_, B::Tensor>,
        state: &mut S,
        provider: &mut P,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
        forward_feed_forward: impl FnOnce(
            &mut FeedForwardSublayer<B>,
            &B::Tensor,
            RoutedObservationPoints,
            &mut P,
            &mut ComponentInstrumentation<'_, B::Tensor>,
        ) -> Result<B::Tensor, Error>,
    ) -> Result<PredictionForward<B::Tensor>, Error>
    where
        S: AttentionCache<B::Tensor> + RuntimeStateComponents<B> + RuntimeAppendStreams<B>,
        P: ParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        let shape = input.residual.shape();
        shared.fusion.geometry.validate_streams(shape)?;
        if shared.fusion.geometry != self.spec.feed_forward.residual.geometry
            || shape[0] > shared.limits.qsa.batch
            || shape[1] > shared.limits.qsa.tokens
        {
            return Err(Error::backend(
                "prediction invocation exceeds declared geometry or bounds",
            ));
        }
        if input
            .visible
            .is_some_and(|v| v.len() != (shape[0] as usize) * (shape[1] as usize))
        {
            return Err(Error::backend("prediction visibility geometry mismatch"));
        }
        let residual = instrumentation.apply("input", input.residual.clone())?;
        let embeddings = instrumentation.apply("embedding", input.embeddings.clone())?;
        let fused = shared.fusion.forward(&embeddings, &residual, context)?;
        let fused = instrumentation.apply("fusion", fused)?;
        let hidden = match &mut self.mixer {
            Mixer::Indexed(mixer) => mixer
                .forward(
                    IndexedSublayerInput {
                        residual: &fused,
                        visible: input.visible,
                        rotary: input.rotary,
                    },
                    state,
                    parallel,
                    context,
                    instrumentation,
                )
                .map_err(Error::backend_source)?,
            Mixer::Recurrent(mixer) => {
                let padding = input
                    .visible
                    .map(|visible| {
                        B::Tensor::from_f32_slice(
                            &visible
                                .iter()
                                .map(|v| u8::from(*v) as f32)
                                .collect::<Vec<_>>(),
                            &shape[..2],
                            context,
                        )?
                        .cast_float(
                            fused
                                .element_type()
                                .ok_or_else(|| Error::backend("missing prediction scalar type"))?,
                            context,
                        )
                    })
                    .transpose()?;
                match parallel {
                    Some(parallel) => mixer.forward_parallel(
                        &fused,
                        padding.as_ref(),
                        state,
                        parallel,
                        context,
                        instrumentation,
                    )?,
                    None => {
                        mixer.forward(&fused, padding.as_ref(), state, context, instrumentation)?
                    }
                }
            }
        };
        let ff = &self.spec.feed_forward.feed_forward;
        let capture = forward_feed_forward(
            &mut self.feed_forward,
            &hidden,
            RoutedObservationPoints::new(
                ff.bank,
                format!("mtp.layers.{}.mlp", self.spec.depth),
                ff.parameters.routed.selector.selection().group_count(),
            ),
            provider,
            instrumentation,
        )?;
        let capture = instrumentation.apply("capture", capture)?;
        let hidden = shared
            .readout
            .collapse(&capture, context, instrumentation)?;
        Ok(PredictionForward { capture, hidden })
    }
}
