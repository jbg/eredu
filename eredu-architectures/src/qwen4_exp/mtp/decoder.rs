//! Layer-sized prediction owners using the target's mixer and feed-forward equations.
use super::*;
use crate::{
    decoder::{ComponentInstrumentation, DecoderBoundary},
    qwen4_exp::{
        config::{Config, LayerKind},
        feed_forward::{FeedForwardSublayer, FeedForwardSublayerSpec},
        indexed::{IndexedSublayerInput, IndexedSublayerSpec},
        qsa::QsaExecutionLimits,
        recurrent::RecurrentSublayerSpec,
        residual::{mixer_spec, ResidualBoundary},
        target::{Mixer, MixerSpec},
    },
};
use eredu_nn::{
    AttentionCache, DistributedNeuralBackend, GroupedGatedProductSpec, GroupedNeuralBackend,
    LinearFormatSpec, NormalizationScale, ParameterSpec, RotaryPosition, TensorElementType,
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
    /// Shared full-residual fusion.
    pub fusion: PredictionFusionSpec,
    /// Prediction-specific gated readout, without the target readout or extra RMSNorm.
    pub readout: eredu_nn::residual_streams::GatedResidualSpec,
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
    /// Constructs exact released names from retained formats and expert declarations.
    pub fn from_prepared(
        config: &Config,
        limits: PredictionLimits,
        bank: RoutedBankId,
        mut experts: impl FnMut(usize) -> Result<GroupedGatedProductSpec, Error>,
        mut format: impl FnMut(&str) -> Result<LinearFormatSpec, Error>,
    ) -> Result<Self, Error> {
        let prediction = config
            .prediction
            .as_ref()
            .ok_or_else(|| Error::backend("artifact has no embedded prediction schedule"))?;
        if prediction.layers.is_empty() || limits.qsa.batch <= 0 || limits.qsa.tokens <= 0 {
            return Err(Error::backend(
                "invalid prediction schedule or invocation bounds",
            ));
        }
        let mut config = config.clone();
        config.attention.rotary.base = prediction.rope_theta;
        let geometry = ResidualStreamGeometry::new(config.residual.streams, config.hidden_size)?;
        let norm = |name: &str, dimensions| -> Result<_, Error> {
            Ok(NormalizationConstructionSpec {
                dimensions,
                groups: None,
                epsilon: config.norm_epsilon,
                scale: NormalizationScale::LearnedOffset {
                    weight: ParameterSpec::trainable(name).map_err(Error::backend)?,
                    offset: 1.,
                },
            })
        };
        let mut projection = |name: &str| -> Result<_, Error> {
            Ok(LinearSpec {
                input: config.hidden_size,
                output: config.hidden_size,
                weight: ParameterSpec::trainable(name).map_err(Error::backend)?,
                bias: None,
                format: format(name)?,
            })
        };
        let fusion = PredictionFusionSpec {
            geometry,
            embedding_norm: norm("mtp.pre_fc_norm_embedding.weight", config.hidden_size)?,
            hidden_norm: norm("mtp.pre_fc_norm_hidden.weight", geometry.flattened_width())?,
            embedding_projection: projection("mtp.fc_embedding.weight")?,
            hidden_projection: projection("mtp.fc_hidden.weight")?,
        };
        fusion.validate()?;
        let readout = mixer_spec(&config, "mtp.hyper_connection_mixer", false, &mut format)?;
        let units = prediction
            .layers
            .iter()
            .enumerate()
            .map(|(depth, kind)| {
                let root = format!("mtp.layers.{depth}");
                let mixer = match kind {
                    LayerKind::Recurrent => MixerSpec::Recurrent(
                        RecurrentSublayerSpec::from_config(&config, &root, &mut format)?,
                    ),
                    LayerKind::Indexed => MixerSpec::Indexed(
                        IndexedSublayerSpec::from_config(
                            &config,
                            &root,
                            limits.tile_blocks,
                            limits.selection_workspace,
                            limits.qsa,
                            limits.element,
                            TensorElementType::F32,
                            &mut format,
                        )
                        .map_err(Error::backend_source)?,
                    ),
                };
                let mut feed_forward = FeedForwardSublayerSpec::from_config(
                    &config,
                    depth,
                    &root,
                    experts(depth)?,
                    &mut format,
                )?;
                feed_forward.feed_forward.bank = bank;
                Ok(PredictionUnitSpec {
                    depth,
                    mixer,
                    feed_forward,
                })
            })
            .collect::<Result<Vec<_>, Error>>()?;
        let spec = Self {
            fusion,
            readout,
            units,
            limits,
        };
        spec.state_layout()?;
        Ok(spec)
    }
    /// Prediction state never aliases the target's lexical, recurrent or QSA slots.
    pub fn state_layout(&self) -> Result<StateLayout, Error> {
        self.fusion.validate()?;
        self.readout.validate()?;
        if self.fusion.geometry != self.readout.geometry
            || self.readout.injection.is_some()
            || self.units.is_empty()
            || self.limits.qsa.batch <= 0
            || self.limits.qsa.tokens <= 0
        {
            return Err(Error::backend(
                "prediction shared geometry, schedule or bounds are invalid",
            ));
        }
        for (depth, unit) in self.units.iter().enumerate() {
            let geometry = match &unit.mixer {
                MixerSpec::Recurrent(mixer) => {
                    mixer.validate()?;
                    mixer.residual.geometry
                }
                MixerSpec::Indexed(mixer) => {
                    mixer.validate().map_err(Error::backend_source)?;
                    mixer.residual.geometry
                }
            };
            unit.feed_forward.validate()?;
            if unit.depth != depth
                || unit.feed_forward.feed_forward.layer != depth
                || geometry != self.fusion.geometry
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
        if spec.fusion.geometry != spec.readout.geometry {
            return Err(Error::backend(
                "prediction fusion/readout residual geometry mismatch",
            ));
        }
        Ok(Self {
            fusion: PredictionFusion::new(spec.fusion.clone(), context)?,
            readout: ResidualBoundary::new(spec.readout.clone(), context)?,
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
        spec.feed_forward.validate()?;
        B::require_operator_capabilities(
            "qwen4_exp prediction mixer",
            spec.mixer.required_operators(),
        )?;
        let geometry = match &spec.mixer {
            MixerSpec::Recurrent(s) => s.residual.geometry,
            MixerSpec::Indexed(s) => s.residual.geometry,
        };
        if geometry != spec.feed_forward.residual.geometry
            || spec.feed_forward.feed_forward.layer != spec.depth
        {
            return Err(Error::backend(
                "prediction mixer/feed-forward owner or geometry mismatch",
            ));
        }
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
                ff.router.selection().group_count(),
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
