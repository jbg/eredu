//! Layer-sized target construction for shared ordinary and bounded traversal.
use super::{
    config::{Config, LayerKind},
    feed_forward::{FeedForwardSublayer, FeedForwardSublayerSpec},
    indexed::{IndexedSublayer, IndexedSublayerInput, IndexedSublayerSpec},
    input::{RequestBoundarySchema, RequestContext},
    ngram::NGramHashSpec,
    ple::{LexicalInjectionSpec, LexicalSublayer, LexicalSublayerInput},
    qsa::QsaExecutionLimits,
    recurrent::{RecurrentSublayer, RecurrentSublayerSpec},
    residual::{mixer_spec, ResidualBoundary},
};
use crate::{
    decoder::{ComponentInstrumentation, StaticModules},
    hybrid_decoder::HybridDecoder,
};
use eredu_core::{cache::LayerCachePolicy, LayerSchedule};
use eredu_nn::{
    AttentionCache, DistributedNeuralBackend, EmbeddingSpec, Error, GroupedGatedProductSpec,
    GroupedNeuralBackend, LinearFormatSpec, LinearSpec, ParameterSpec, Tensor, TensorElementType,
};
use eredu_runtime::{
    ParameterBankAccess, ParameterProvider, RoutedBankId, RoutedObservationPoints,
    RuntimeAppendStreams, RuntimeStateComponents, StateLayout,
};
use std::collections::BTreeMap;

/// Bounds retained by target construction, independently of table/cache residency.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TargetLimits {
    /// Coordinator batch/token/scratch admission.
    pub qsa: QsaExecutionLimits,
    /// Maximum cumulative tokens per lane, including prefill and cached decode.
    pub history_tokens: i32,
    /// Complete summaries per score tile.
    pub tile_blocks: i32,
    /// Scratch for a single tiled block-selection query.
    pub selection_workspace: u64,
    /// Maximum total original IDs in one target invocation (batch times chunk tokens).
    pub invocation_tokens: usize,
    /// Maximum lookup rows in one injection invocation.
    pub lookup_rows: usize,
    /// Activation/cache scalar representation selected during preparation.
    pub element: TensorElementType,
}
impl TargetLimits {
    /// Checks invocation geometry before source literals or native allocation.
    pub(crate) fn validate_for_config(&self, config: &Config) -> Result<(), Error> {
        if self.qsa.batch <= 0
            || self.qsa.tokens <= 0
            || self.invocation_tokens == 0
            || self.history_tokens < self.qsa.tokens
            || self.history_tokens > config.max_positions
            || (self.qsa.batch as usize)
                .checked_mul(self.qsa.tokens as usize)
                .is_none_or(|n| n < self.invocation_tokens)
            || !matches!(
                self.element,
                TensorElementType::F16 | TensorElementType::Bf16 | TensorElementType::F32
            )
        {
            return Err(Error::backend("target invocation bounds are inconsistent"));
        }
        if !config.ngram.layers.is_empty() {
            let rows = config
                .ngram
                .order
                .checked_sub(1)
                .and_then(|n| n.checked_mul(config.ngram.heads))
                .filter(|n| *n > 0)
                .and_then(|n| (n as usize).checked_mul(self.invocation_tokens));
            if self.lookup_rows > i32::MAX as usize || rows.is_none_or(|n| n > self.lookup_rows) {
                return Err(Error::backend(
                    "lexical lookup cannot cover admitted invocation tokens",
                ));
            }
        }
        if config.layers.iter().any(|kind| *kind == LayerKind::Indexed) {
            super::qsa::QsaSelectionSpec {
                heads: config.attention.index_heads,
                dimensions: config.attention.index_head_dim,
                ratio: config.attention.ratio,
                token_budget: config.attention.budget,
                tile_blocks: self.tile_blocks,
                workspace_bytes: self.selection_workspace,
            }
            .validate()
            .map_err(Error::backend_source)?;
        }
        Ok(())
    }
}
/// A decoder mixer selected by the released schedule, retaining exact prepared specs.
#[derive(Debug, Clone)]
pub enum MixerSpec {
    /// Fixed recurrence and convolution state.
    Recurrent(RecurrentSublayerSpec),
    /// Full K/V, named summaries and fixed partial state.
    Indexed(IndexedSublayerSpec),
}
impl MixerSpec {
    /// Optional tensor mechanisms used by either target or prediction execution.
    pub fn required_operators(&self) -> eredu_nn::NeuralOperatorCapabilities {
        use eredu_nn::NeuralOperatorCapabilities as C;
        match self {
            Self::Recurrent(_) => crate::gated_delta::REQUIRED_OPERATORS,
            Self::Indexed(_) => C::INDEXED_ATTENTION
                .union(C::TO_F32_VEC)
                .union(C::FROM_I32_SLICE)
                .union(C::TO_I32_VEC)
                .union(C::FULL_F32)
                .union(C::CAST_FLOAT)
                .union(C::BROADCAST_TO)
                .union(C::SIGMOID),
        }
    }
    /// State owned by one target or prediction mixer invocation.
    pub fn state_policy(&self) -> Result<LayerCachePolicy, Error> {
        match self {
            Self::Recurrent(spec) => spec.mixer.state_policy(),
            Self::Indexed(spec) => spec
                .state
                .cache_policy(spec.kv_heads, spec.head_dim)
                .map_err(Error::backend_source),
        }
    }
}
/// One bounded owner; an injection can be cut independently from its decoder block.
#[derive(Debug, Clone)]
pub enum UnitSpec {
    /// Table acquisition and lexical histories belong only to this invocation.
    Lexical {
        /// Original checkpoint layer.
        layer: usize,
        /// Exact injection parameters and histories.
        spec: LexicalInjectionSpec,
    },
    /// One mixer plus its shared/routed feed-forward sublayer.
    Decoder {
        /// Original checkpoint layer.
        layer: usize,
        /// Selected attention or recurrent construction.
        mixer: MixerSpec,
        /// Shared/routed feed-forward construction.
        feed_forward: FeedForwardSublayerSpec,
    },
}
impl UnitSpec {
    /// Original checkpoint layer, distinct from the execution/state ordinal.
    pub fn layer(&self) -> usize {
        match self {
            Self::Lexical { layer, .. } | Self::Decoder { layer, .. } => *layer,
        }
    }
    /// Semantic parameter/observation path for this unit.
    pub fn path(&self) -> String {
        match self {
            Self::Lexical { layer, .. } => format!("model.layers.{layer}.ple"),
            Self::Decoder { layer, .. } => format!("model.layers.{layer}"),
        }
    }
    /// Exact unit-local mutable state geometry.
    pub fn state_policy(&self) -> Result<LayerCachePolicy, Error> {
        match self {
            Self::Lexical { spec, .. } => {
                LayerCachePolicy::fixed_only(spec.state_policies()?).map_err(Error::backend)
            }
            Self::Decoder { mixer, .. } => mixer.state_policy(),
        }
    }
}
/// Fully specified target parameters. Table/expert payloads remain with providers.
#[derive(Debug, Clone)]
pub struct TargetSpec {
    pub(crate) config: Config,
    /// Bounded request/coordinator contract.
    pub limits: TargetLimits,
    /// Shared token lookup.
    pub embedding: EmbeddingSpec,
    /// Independent output projection; `None` shares the embedding parameter.
    pub head: Option<LinearSpec>,
    /// Released final gated collapse, without another RMSNorm.
    pub boundary: eredu_nn::residual_streams::GatedResidualSpec,
    /// Ordered execution owners; each owns one matching state slot.
    pub units: Vec<UnitSpec>,
}
impl TargetSpec {
    /// Exact family geometry retained by this header-derived target specification.
    pub fn configuration(&self) -> &Config {
        &self.config
    }
    /// Builds global target geometry from metadata-only row contracts and independent experts.
    /// The expert callback receives checkpoint layer and execution owner separately.
    pub fn from_headers(
        config: Config,
        tables: &BTreeMap<usize, eredu_runtime::RowLookupSpec>,
        limits: TargetLimits,
        mut experts: impl FnMut(usize, usize) -> Result<GroupedGatedProductSpec, Error>,
        mut format: impl FnMut(&str) -> Result<LinearFormatSpec, Error>,
    ) -> Result<Self, Error> {
        limits.validate_for_config(&config)?;
        if tables.len() != config.ngram.layers.len()
            || tables.keys().any(|l| !config.ngram.layers.contains(l))
        {
            return Err(Error::backend(
                "target prepared tables or request bounds differ from configuration",
            ));
        }
        let parameter = |name: &str| ParameterSpec::trainable(name).map_err(Error::backend);
        let embedding = EmbeddingSpec {
            vocabulary: config.vocabulary,
            dimensions: config.hidden_size,
            weight: parameter("model.embed_tokens.weight")?,
            format: format("model.embed_tokens.weight")?,
        };
        let head = if config.tied_embeddings {
            None
        } else {
            Some(LinearSpec {
                input: config.hidden_size,
                output: config.vocabulary,
                weight: parameter("lm_head.weight")?,
                bias: None,
                format: format("lm_head.weight")?,
            })
        };
        let boundary = mixer_spec(&config, "model.hyper_connection_mixer", false, &mut format)?;
        let mut units = Vec::new();
        for layer in 0..config.layers.len() {
            let root = format!("model.layers.{layer}");
            if let Some(table) = tables.get(&layer) {
                if table.unit != units.len() {
                    return Err(Error::backend(
                        "prepared table owner differs from lexical execution ordinal",
                    ));
                }
                let spec = LexicalInjectionSpec::from_header(
                    &config,
                    layer,
                    &root,
                    table,
                    limits.lookup_rows,
                    &mut format,
                )
                .map_err(Error::backend_source)?;
                if spec.embedding.max_tokens() < limits.invocation_tokens {
                    return Err(Error::backend(
                        "lexical lookup cannot cover admitted invocation tokens",
                    ));
                }
                units.push(UnitSpec::Lexical { layer, spec });
            }
            let mixer =
                match config
                    .layers
                    .get(layer)
                    .ok_or_else(|| Error::backend("missing target layer kind"))?
                {
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
            let owner = units.len();
            let feed_forward = FeedForwardSublayerSpec::from_config(
                &config,
                owner,
                &root,
                experts(layer, owner)?,
                &mut format,
            )?;
            units.push(UnitSpec::Decoder {
                layer,
                mixer,
                feed_forward,
            });
        }
        let spec = Self {
            config,
            limits,
            embedding,
            head,
            boundary,
            units,
        };
        spec.state_layout()?;
        Ok(spec)
    }
    /// Optional mechanisms required by the complete target, before allocation.
    pub fn required_operators(&self) -> eredu_nn::NeuralOperatorCapabilities {
        use eredu_nn::NeuralOperatorCapabilities as C;
        let mut required = C::FROM_I32_SLICE
            .union(C::TO_I32_VEC)
            .union(C::FULL_F32)
            .union(C::CAST_FLOAT)
            .union(C::BROADCAST_TO)
            .union(C::SIGMOID);
        for unit in &self.units {
            required = required.union(match unit {
                UnitSpec::Lexical { .. } => {
                    C::ABS.union(C::SIGN).union(C::SQRT).union(C::TO_F32_VEC)
                }
                UnitSpec::Decoder { mixer, .. } => mixer.required_operators(),
            });
        }
        required
    }

    /// Header-derived identity for cold geometry, independent of hash literals.
    pub fn geometry_fingerprint(&self) -> String {
        eredu_core::cache::derive_prompt_cache_architecture_fingerprint(
            "qwen4_exp.target.geometry.v1",
            [
                ("embedding", format!("{:?}", self.embedding)),
                ("head", format!("{:?}", self.head)),
                ("readout", format!("{:?}", self.boundary)),
                ("eos", format!("{:?}", self.config.eos)),
                ("history_tokens", self.limits.history_tokens.to_string()),
                ("units", format!("{:?}", self.units)),
            ],
        )
    }

    /// Authoritative state ownership matches bounded execution units one-for-one.
    pub fn state_layout(&self) -> Result<StateLayout, Error> {
        self.boundary.validate()?;
        let mut ordinal = 0;
        for (layer, kind) in self.config.layers.iter().enumerate() {
            if self.config.ngram.layers.contains(&layer) {
                match self.units.get(ordinal) {
                    Some(UnitSpec::Lexical { layer: owner, spec })
                        if *owner == layer && spec.geometry == self.boundary.geometry => {}
                    _ => {
                        return Err(Error::backend(
                            "target lexical unit schedule or geometry mismatch",
                        ));
                    }
                }
                ordinal += 1;
            }
            match self.units.get(ordinal) {
                Some(UnitSpec::Decoder {
                    layer: owner,
                    mixer,
                    feed_forward,
                }) if *owner == layer
                    && feed_forward.residual.geometry == self.boundary.geometry =>
                {
                    let valid = match (kind, mixer) {
                        (LayerKind::Recurrent, MixerSpec::Recurrent(s)) => {
                            s.residual.geometry == self.boundary.geometry
                        }
                        (LayerKind::Indexed, MixerSpec::Indexed(s)) => {
                            s.residual.geometry == self.boundary.geometry
                        }
                        _ => false,
                    };
                    if !valid {
                        return Err(Error::backend(
                            "target mixer schedule or residual geometry mismatch",
                        ));
                    }
                }
                _ => return Err(Error::backend("target decoder unit schedule mismatch")),
            }
            ordinal += 1;
        }
        if ordinal != self.units.len() {
            return Err(Error::backend("target has undeclared execution units"));
        }
        let policies = self
            .units
            .iter()
            .enumerate()
            .map(|(index, unit)| {
                let policy = unit.state_policy()?;
                if index == 0 {
                    super::position::policy(policy)
                } else {
                    Ok(policy)
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        StateLayout::new(LayerSchedule::new(policies.len(), policies).map_err(Error::backend)?)
            .map_err(Error::backend)
    }
    /// Complete residual streams and original request provenance cross every cut.
    pub fn boundary_schema(&self) -> Result<RequestBoundarySchema, Error> {
        RequestBoundarySchema::new(
            self.boundary.geometry,
            self.config.attention.rotary.dimensions,
        )
    }
    /// Retains ordinary parameter recipes for exactly one bounded owner. Experts
    /// and table rows remain independently addressable through their providers.
    pub fn unit_parameter_recipes(
        &self,
        index: usize,
        source: &dyn eredu_checkpoint::store::CheckpointSource,
    ) -> Result<BTreeMap<String, eredu_checkpoint::recipe::DerivedWeightRecipe>, String> {
        use super::checkpoint::recipes::{parameter_recipes, ParameterScope};
        let scope = match self
            .units
            .get(index)
            .ok_or("target unit ordinal out of bounds")?
        {
            UnitSpec::Lexical { layer, .. } => ParameterScope::Lexical(*layer),
            UnitSpec::Decoder { layer, .. } => ParameterScope::Target(*layer),
        };
        parameter_recipes(source.source_keys(), &self.config, scope)
    }
}
/// Concrete mixer module, never selected by a native backend.
#[derive(Debug, eredu_nn::Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub enum Mixer<B: GroupedNeuralBackend + DistributedNeuralBackend> {
    /// Recurrent update with fixed components.
    Recurrent(RecurrentSublayer<B>),
    /// Indexed update with full K/V and auxiliary streams.
    Indexed(IndexedSublayer<B>),
}
impl<B: GroupedNeuralBackend + DistributedNeuralBackend> Mixer<B> {
    /// Constructs the same mixer equations for a target or prediction namespace.
    pub fn new(spec: MixerSpec, context: &<B::Tensor as Tensor>::Context) -> Result<Self, Error> {
        match spec {
            MixerSpec::Recurrent(s) => Ok(Self::Recurrent(RecurrentSublayer::new(s, context)?)),
            MixerSpec::Indexed(s) => Ok(Self::Indexed(
                IndexedSublayer::new(s, context).map_err(Error::backend_source)?,
            )),
        }
    }
}
/// Concrete layer-sized unit consumed by the shared residency policy.
#[derive(Debug, eredu_nn::Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub enum Unit<B: GroupedNeuralBackend + DistributedNeuralBackend> {
    /// Independent lexical state and table lookup.
    Lexical(LexicalSublayer<B>),
    /// A single decoder block.
    Decoder {
        /// Selected stateful mixer.
        mixer: Mixer<B>,
        /// Stateless shared/routed update after the mixer.
        feed_forward: FeedForwardSublayer<B>,
    },
}
/// Target geometry bound to exact per-injection integer controls. A header-only
/// specification cannot construct an executable or authorize persisted state.
#[derive(Debug, Clone)]
pub struct BoundTargetSpec {
    spec: TargetSpec,
    hashes: BTreeMap<usize, NGramHashSpec>,
    state_fingerprint: String,
}
impl BoundTargetSpec {
    /// Validates every control owner and literal geometry without backend resources.
    pub fn new(spec: TargetSpec, hashes: BTreeMap<usize, NGramHashSpec>) -> Result<Self, Error> {
        spec.limits.validate_for_config(&spec.config)?;
        spec.state_layout()?;
        let layers: Vec<_> = spec
            .units
            .iter()
            .filter_map(|unit| match unit {
                UnitSpec::Lexical { layer, .. } => Some(*layer),
                _ => None,
            })
            .collect();
        if hashes.keys().copied().collect::<Vec<_>>() != layers {
            return Err(Error::backend(
                "bound target hash owners differ from lexical units",
            ));
        }
        for unit in &spec.units {
            if let UnitSpec::Lexical {
                layer,
                spec: lexical,
            } = unit
            {
                let hash = &hashes[layer];
                lexical
                    .embedding
                    .validate_hash(hash)
                    .map_err(Error::backend_source)?;
                if spec.config.eos.first().is_none_or(|&eos| {
                    !hash.matches_geometry(
                        spec.config.vocabulary as u64,
                        eos as u64,
                        spec.config.ngram.order as usize,
                        spec.config.ngram.heads as usize,
                    )
                }) {
                    return Err(Error::backend_source(super::ngram::NGramError::Constants));
                }
            }
        }
        let state_fingerprint = eredu_core::cache::derive_prompt_cache_architecture_fingerprint(
            "qwen4_exp.target.bound.v1",
            [
                ("geometry", spec.geometry_fingerprint()),
                ("hash_controls", format!("{hashes:?}")),
            ],
        );
        Ok(Self {
            spec,
            hashes,
            state_fingerprint,
        })
    }
    /// Immutable geometry paired with the controls validated by this value.
    pub fn geometry(&self) -> &TargetSpec {
        &self.spec
    }
    /// Literal-aware identity used for model state and prompt-cache compatibility.
    pub fn state_fingerprint(&self) -> &str {
        &self.state_fingerprint
    }
}
/// Target-only layered composition. Prediction and media encoders attach in total
/// prepared construction; this type does not register a family or facade load path.
pub struct TargetModel<B: GroupedNeuralBackend + DistributedNeuralBackend> {
    pub(crate) spec: TargetSpec,
    hashes: BTreeMap<usize, NGramHashSpec>,
    state_fingerprint: String,
    tensor_partition: Option<TargetTensorPartition>,
    partition_state: eredu_runtime::PartitionState,
    global_parameters: Option<std::sync::Arc<eredu_runtime::ArchitectureParameterDescription>>,
    pub(crate) decoder: HybridDecoder<B, (), ResidualBoundary<B>>,
}
impl<B: GroupedNeuralBackend + DistributedNeuralBackend> Clone for TargetModel<B> {
    fn clone(&self) -> Self {
        Self {
            spec: self.spec.clone(),
            hashes: self.hashes.clone(),
            state_fingerprint: self.state_fingerprint.clone(),
            tensor_partition: self.tensor_partition.clone(),
            partition_state: self.partition_state.clone(),
            global_parameters: self.global_parameters.clone(),
            decoder: self.decoder.clone(),
        }
    }
}
impl<B: GroupedNeuralBackend + DistributedNeuralBackend> TargetModel<B> {
    /// Allocates pinned parameters only; layer residency constructs units on demand.
    pub fn new(
        bound: BoundTargetSpec,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let BoundTargetSpec {
            spec,
            hashes,
            state_fingerprint,
        } = bound;
        spec.state_layout()?;
        spec.limits.validate_for_config(&spec.config)?;
        for (ordinal, unit) in spec.units.iter().enumerate() {
            match unit {
                UnitSpec::Lexical { spec: lexical, .. } => lexical.validate()?,
                UnitSpec::Decoder {
                    mixer,
                    feed_forward,
                    ..
                } => {
                    feed_forward.validate()?;
                    if feed_forward.feed_forward.layer != ordinal {
                        return Err(Error::backend("expert owner differs from target unit"));
                    }
                    match mixer {
                        MixerSpec::Recurrent(r) => r.validate()?,
                        MixerSpec::Indexed(a) => {
                            a.validate().map_err(Error::backend_source)?;
                            a.limits
                                .validate::<B::Tensor>(a.state)
                                .map_err(Error::backend_source)?;
                        }
                    }
                }
            }
        }
        B::require_operator_capabilities("qwen4_exp target", spec.required_operators())?;
        let boundary = ResidualBoundary::new(spec.boundary.clone(), context)?;
        let modules = StaticModules::from_boundary(
            spec.embedding.clone(),
            spec.head.clone(),
            boundary,
            context,
        )?;
        let decoder = HybridDecoder::new(modules, "model.units", spec.units.len())?;
        let partition_state =
            eredu_runtime::PartitionState::new(spec.state_layout()?, 0).map_err(Error::backend)?;
        Ok(Self {
            spec,
            hashes,
            state_fingerprint,
            tensor_partition: None,
            partition_state,
            global_parameters: None,
            decoder,
        })
    }
    /// Constructs rank-local units while retaining replicated vocabulary and residual mixers.
    /// Retains the same partition that authored source transformations, validating it against
    /// the global bound specification. No partition is reconstructed during native binding.
    /// Execution validates the retained rank against the actual collective member.
    pub fn new_tensor_parallel(
        bound: BoundTargetSpec,
        partition: TargetTensorPartition,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let local = bound.tensor_partition(&partition)?;
        let mut model = Self::new(local, context)?;
        model.tensor_partition = Some(partition);
        Ok(model)
    }

    /// Exact local geometry and source selections; absent on an ordinary target.
    pub fn tensor_partition(&self) -> Option<&TargetTensorPartition> {
        self.tensor_partition.as_ref()
    }

    pub(crate) fn retain_global_parameters(
        &mut self,
        parameters: eredu_runtime::ArchitectureParameterDescription,
    ) {
        self.global_parameters = Some(std::sync::Arc::new(parameters));
    }

    /// Literal-aware state identity retained at construction.
    pub fn state_fingerprint(&self) -> &str {
        &self.state_fingerprint
    }

    /// Constructs one owner without reading expert or table payloads.
    pub fn construct_unit(
        &self,
        index: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Unit<B>, Error> {
        match self
            .spec
            .units
            .get(index)
            .ok_or_else(|| Error::backend("target unit ordinal out of bounds"))?
        {
            UnitSpec::Lexical { layer, spec } => Ok(Unit::Lexical(LexicalSublayer::new(
                spec.clone(),
                self.hashes[layer].clone(),
                context,
            )?)),
            UnitSpec::Decoder {
                mixer,
                feed_forward,
                ..
            } => Ok(Unit::Decoder {
                mixer: Mixer::new(mixer.clone(), context)?,
                feed_forward: FeedForwardSublayer::new(feed_forward.clone(), context)?,
            }),
        }
    }
    pub(crate) fn execute<P, S>(
        &self,
        index: usize,
        unit: &mut Unit<B>,
        hidden: &B::Tensor,
        state: &mut S,
        request: &RequestContext<B::Tensor>,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error>
    where
        P: ParameterProvider<B>,
        P::Error: std::fmt::Display,
        S: AttentionCache<B::Tensor> + RuntimeStateComponents<B> + RuntimeAppendStreams<B>,
    {
        if self
            .tensor_partition
            .as_ref()
            .is_some_and(|partition| partition.ranks() > 1)
        {
            return Err(Error::backend(
                "rank-local target requires parallel unit execution",
            ));
        }
        self.execute_with(
            index,
            unit,
            hidden,
            state,
            request,
            provider,
            None,
            context,
            instrumentation,
            |feed_forward, hidden, points, provider, instrumentation| {
                feed_forward.forward(hidden, points, provider, context, instrumentation)
            },
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_with<P, S>(
        &self,
        index: usize,
        unit: &mut Unit<B>,
        hidden: &B::Tensor,
        state: &mut S,
        request: &RequestContext<B::Tensor>,
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
    ) -> Result<B::Tensor, Error>
    where
        P: ParameterProvider<B>,
        P::Error: std::fmt::Display,
        S: AttentionCache<B::Tensor> + RuntimeStateComponents<B> + RuntimeAppendStreams<B>,
    {
        if state.position() != request.offset() {
            return Err(Error::backend("target unit request/state prefix mismatch"));
        }
        if index == 0 {
            super::position::resolve::<B, _>(
                state,
                request.offset(),
                Some(request.position_delta()),
                context,
            )
            .map_err(Error::backend_source)?;
        }
        let spec = self
            .spec
            .units
            .get(index)
            .ok_or_else(|| Error::backend("target unit ordinal out of bounds"))?;
        let padding = request.padding(hidden, context)?;
        let result = match (spec, unit) {
            (UnitSpec::Lexical { .. }, Unit::Lexical(unit)) => unit
                .forward(
                    LexicalSublayerInput {
                        residual: hidden,
                        ids: Some(request.ids()),
                        padding: Some(&padding),
                        offset: request.offset(),
                    },
                    state,
                    provider,
                    if hidden.dim(1) == 1 {
                        ParameterBankAccess::Incremental
                    } else {
                        ParameterBankAccess::Bulk
                    },
                    context,
                    instrumentation,
                )
                .map_err(Error::backend_source)?,
            (
                UnitSpec::Decoder {
                    layer,
                    feed_forward: ff,
                    mixer: declared,
                },
                Unit::Decoder {
                    mixer,
                    feed_forward,
                },
            ) => {
                let hidden = match (declared, mixer) {
                    (MixerSpec::Recurrent(_), Mixer::Recurrent(m)) => match parallel {
                        Some(parallel) => m.forward_parallel(
                            hidden,
                            Some(&padding),
                            state,
                            parallel,
                            context,
                            instrumentation,
                        )?,
                        None => {
                            m.forward(hidden, Some(&padding), state, context, instrumentation)?
                        }
                    },
                    (MixerSpec::Indexed(_), Mixer::Indexed(m)) => m
                        .forward(
                            IndexedSublayerInput {
                                residual: hidden,
                                visible: Some(request.visible()),
                                rotary: Some(request.rotary()),
                            },
                            state,
                            parallel,
                            context,
                            instrumentation,
                        )
                        .map_err(Error::backend_source)?,
                    _ => return Err(Error::backend("target mixer does not match declared unit")),
                };
                forward_feed_forward(
                    feed_forward,
                    &hidden,
                    RoutedObservationPoints::new(
                        RoutedBankId::new(0),
                        format!("model.layers.{layer}.mlp"),
                        ff.feed_forward.router.selection().group_count(),
                    ),
                    provider,
                    instrumentation,
                )?
            }
            _ => return Err(Error::backend("target module does not match declared unit")),
        };
        if index == 0 {
            super::position::commit::<B, _>(state, request.position_delta(), context)
                .map_err(Error::backend_source)?;
        }
        Ok(result)
    }
}

pub(crate) mod parallel_geometry;
pub(crate) use parallel_geometry::parallel_layout::partition_unit_observations;
pub use parallel_geometry::TargetTensorPartition;
mod layered;
pub use layered::{TargetForward, TargetInput};

mod parallel;
mod partitioned;
