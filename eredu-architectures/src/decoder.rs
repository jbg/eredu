//! Shared backend-neutral decoder mechanics.
//!
//! Architecture families retain configuration, checkpoint naming, identity, and
//! policy while reusing these statically dispatched decoder operations.

use eredu_checkpoint::{LinearFormat, WeightQuantization};
use eredu_core::cache::LayerCachePolicy;
use eredu_core::{AttentionPolicy, LayerSchedule};
use eredu_nn::{
    AttentionCache, AttentionRequest, EmbeddingLookupPolicy, EmbeddingOperator, EmbeddingSpec,
    Error, FusedProjectionLayout, FusedProjectionSegment, GatedProductPolicy, GroupedNeuralBackend,
    Index, LinearOperator, LinearSpec, NeuralBackend, NormalizationConstructionSpec,
    NormalizationOperator, Parameter, ParameterSpec, RotaryOperator, RotaryPosition, RotarySpec,
    RotarySubspace, Tensor, VocabularyParallelRange,
};
use eredu_runtime::{
    aligned_partition_units, module_parameter_group, partitioned_module_parameter_group,
    partitioned_projection_group, segmented_projection_group, ArchitectureParameterDescription,
    ExecutionGraph, ExecutionUnitLayout, LayerRuntimeState, LayeredArchitecture,
    LayeredForwardState, LayeredPartitionInput, LocalModelLayout, MemberSharding,
    OwnedParameterGroupSpec, ParallelLayeredArchitecture, ParallelPlanError, ParameterGroupOwner,
    ParameterGroupSpec, ParameterRole, PartitionedLayeredArchitecture, ProjectionSharding,
    StateLayout, TensorPlacement,
};

/// Stable identity of the shared decoder target execution group.
pub const TARGET_EXECUTION_GROUP: &str = "target";
/// Stable identity of an ordinary one-group text decoder.
pub const TEXT_DECODER_EXECUTION_GROUP: &str = "text_decoder";

/// Canonical field segments used by one shared decoder block.
///
/// Architecture families can replace checkpoint vocabulary without replacing
/// the shared attention, residual, feed-forward, or parallel-placement logic.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct BlockParameterFields<'a> {
    /// Self-attention module below one layer.
    pub attention: &'a str,
    /// Query projection below the attention module.
    pub attention_query: &'a str,
    /// Key projection below the attention module.
    pub attention_key: &'a str,
    /// Value projection below the attention module.
    pub attention_value: &'a str,
    /// Output projection below the attention module.
    pub attention_output: &'a str,
    /// Optional learned attention-sink parameter.
    pub attention_sinks: &'a str,
    /// Optional query normalization below the attention module.
    pub attention_query_norm: &'a str,
    /// Optional key normalization below the attention module.
    pub attention_key_norm: &'a str,
    /// Feed-forward module below one layer.
    pub feed_forward: &'a str,
    /// Gate projection below a split feed-forward module.
    pub feed_forward_gate: &'a str,
    /// Up projection below a split feed-forward module.
    pub feed_forward_up: &'a str,
    /// Output projection below the feed-forward module.
    pub feed_forward_output: &'a str,
    /// Pre-attention normalization below one layer.
    pub input_norm: &'a str,
    /// Pre-feed-forward normalization below one layer.
    pub post_attention_norm: &'a str,
}

impl Default for BlockParameterFields<'_> {
    fn default() -> Self {
        Self {
            attention: "self_attn",
            attention_query: "q_proj",
            attention_key: "k_proj",
            attention_value: "v_proj",
            attention_output: "o_proj",
            attention_sinks: "sinks",
            attention_query_norm: "q_norm",
            attention_key_norm: "k_norm",
            feed_forward: "mlp",
            feed_forward_gate: "gate_proj",
            feed_forward_up: "up_proj",
            feed_forward_output: "down_proj",
            input_norm: "input_layernorm",
            post_attention_norm: "post_attention_layernorm",
        }
    }
}

impl BlockParameterFields<'_> {
    fn validate(self) -> Result<Self, Error> {
        for (role, field) in [
            ("attention module", self.attention),
            ("attention query projection", self.attention_query),
            ("attention key projection", self.attention_key),
            ("attention value projection", self.attention_value),
            ("attention output projection", self.attention_output),
            ("attention sinks", self.attention_sinks),
            ("attention query norm", self.attention_query_norm),
            ("attention key norm", self.attention_key_norm),
            ("feed-forward module", self.feed_forward),
            ("feed-forward gate projection", self.feed_forward_gate),
            ("feed-forward up projection", self.feed_forward_up),
            ("feed-forward output projection", self.feed_forward_output),
            ("input norm", self.input_norm),
            ("post-attention norm", self.post_attention_norm),
        ] {
            if field.trim().is_empty() {
                return Err(Error::backend(format!(
                    "decoder block {role} field must not be empty"
                )));
            }
        }
        Ok(self)
    }
}

/// Geometry and policy required by the shared decoder mechanics.
pub trait Config: 'static {
    /// Stable architecture family used by persistence identity.
    fn model_family(&self) -> &'static str;
    /// Stable normalized model identity.
    fn model_identity(&self) -> &str;
    /// Stable identity of the complete normalized architecture policy.
    ///
    /// Implementations must bind every construction, equation, state, and
    /// encoding policy that can affect decoder or cache compatibility.
    fn architecture_fingerprint(&self) -> String;
    /// Canonical parameter namespace for this decoder body.
    fn parameter_root(&self) -> &str {
        "model"
    }
    /// Canonical parameter fields used within each shared decoder block.
    fn block_parameter_fields(&self) -> BlockParameterFields<'_> {
        BlockParameterFields::default()
    }
    /// Returns the canonical routed-observation point for one decoder layer.
    fn routed_observation_point(
        &self,
        _unit_path: &str,
        _layer: usize,
    ) -> Option<eredu_runtime::RoutedObservationPoint> {
        None
    }
    /// Validates architecture-owned configuration policy.
    fn validate_config(&self) -> Result<(), Error>;
    /// Transformer hidden size.
    fn hidden_size(&self) -> i32;
    /// Number of decoder layers.
    fn num_hidden_layers(&self) -> i32;
    /// SwiGLU intermediate width.
    fn intermediate_size(&self) -> i32;
    /// Number of query heads.
    fn num_attention_heads(&self) -> i32;
    /// Number of key/value heads.
    fn num_key_value_heads(&self) -> i32;
    /// Per-head width.
    fn head_dim(&self) -> i32;
    /// RMSNorm epsilon.
    fn rms_norm_epsilon(&self) -> f32;
    /// Vocabulary size.
    fn vocabulary_size(&self) -> i32;
    /// Whether one attention projection owns a learned bias.
    fn attention_bias(&self, projection: AttentionProjection) -> bool;
    /// Physical construction of the query/key/value input projection.
    fn attention_projection_layout(&self) -> AttentionProjectionLayout<'_> {
        AttentionProjectionLayout::Split
    }
    /// Whether each attention layer owns one learned logit per query head.
    fn learned_attention_sinks(&self) -> bool {
        false
    }
    /// Optional per-head Q/K RMS-normalization epsilon.
    fn query_key_norm_epsilon(&self) -> Option<f32> {
        None
    }
    /// Whether projections own MLP biases.
    fn mlp_bias(&self) -> bool;
    /// Physical construction of the gate/up input projection.
    fn gated_projection_layout(&self) -> GatedProjectionLayout<'_> {
        GatedProjectionLayout::Split
    }
    /// Optional equation policy applied by each dense gated product.
    fn gated_product_policy(&self) -> Option<GatedProductPolicy> {
        None
    }
    /// Whether the language-model head is tied to input embeddings.
    fn tie_word_embeddings(&self) -> bool;
    /// Exact per-layer attention policy.
    fn attention_schedule(&self) -> &LayerSchedule<AttentionPolicy>;
    /// Physical encoding selected for one canonical checkpoint parameter.
    fn weight_quantization(&self, name: &str) -> Option<WeightQuantization>;
    /// Complete rotary-position construction specification.
    fn rotary_spec(&self, dimensions: i32) -> RotarySpec;
    /// Whether this decoder stack applies rotary position encoding.
    fn rotary_enabled(&self) -> bool {
        true
    }
}

/// Declares cache identity shared by ordinary layered decoder families.
pub fn state_identity<C: Config>(
    args: &C,
    layout: &StateLayout,
    global_layer_start: usize,
    topology: eredu_core::cache::PromptCacheTopology,
) -> Result<eredu_runtime::ModelStateIdentity, Error> {
    args.validate_config()?;
    topology.validate().map_err(Error::backend)?;
    let layer_count = usize::try_from(args.num_hidden_layers()).map_err(Error::backend)?;
    let global_layer_end = global_layer_start
        .checked_add(layout.len())
        .ok_or_else(|| Error::backend("decoder owned state range overflowed"))?;
    if global_layer_end > layer_count {
        return Err(Error::backend(format!(
            "{} owns state layers {global_layer_start}..{global_layer_end}, outside {layer_count} layers",
            args.model_family()
        )));
    }
    eredu_runtime::ModelStateIdentity::new(
        args.model_family(),
        args.model_identity(),
        args.architecture_fingerprint(),
        layer_count,
        global_layer_start,
        0,
        topology,
    )
    .map_err(Error::backend)
}

/// Semantic attention projection selected by architecture policy.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum AttentionProjection {
    /// Query projection.
    Query,
    /// Key projection.
    Key,
    /// Value projection.
    Value,
    /// Output projection.
    Output,
}

/// Architecture-selected physical query/key/value projection layout.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum AttentionProjectionLayout<'a> {
    /// Independent query, key, and value affine projections.
    Split,
    /// One component-major query/key/value affine projection.
    Fused {
        /// Canonical projection field below the attention module.
        field: &'a str,
    },
}

/// Architecture-selected physical gate/up projection layout.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum GatedProjectionLayout<'a> {
    /// Independent gate and up affine projections.
    Split,
    /// One component-major gate/up affine projection.
    Fused {
        /// Canonical projection field below the MLP module.
        field: &'a str,
    },
}

/// Construction policy for one named table in a deterministic embedding sum.
#[derive(Debug, Clone)]
pub struct NamedEmbeddingSpec {
    /// Stable semantic stream name used for validation diagnostics.
    pub name: String,
    /// Canonical embedding parameter and physical format.
    pub embedding: EmbeddingSpec,
    /// Strict or diagnostic zero-sentinel lookup behavior.
    pub lookup: EmbeddingLookupPolicy,
}

/// One backend-native named table participating in an embedding sum.
#[derive(Debug, Clone, eredu_nn::Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct NamedEmbedding<B: NeuralBackend> {
    /// Backend-native embedding operator.
    pub embedding: B::Embedding,
    #[parameter(skip)]
    name: String,
    #[parameter(skip)]
    lookup: EmbeddingLookupPolicy,
}

/// Ordered multi-stream embedding lookup and deterministic sum.
#[derive(Debug, Clone, eredu_nn::Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct MultiTableEmbedding<B: NeuralBackend> {
    /// Tables in semantic stream order.
    pub tables: Vec<NamedEmbedding<B>>,
}

impl<B: NeuralBackend> MultiTableEmbedding<B> {
    /// Builds validated named tables without materializing checkpoint values.
    pub fn new(
        specs: impl IntoIterator<Item = NamedEmbeddingSpec>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let specs = specs.into_iter().collect::<Vec<_>>();
        if specs.is_empty() {
            return Err(Error::backend(
                "multi-table embedding sum requires at least one table",
            ));
        }
        let mut names = std::collections::BTreeSet::new();
        let mut dimensions = None;
        let mut tables = Vec::with_capacity(specs.len());
        for spec in specs {
            if spec.name.trim().is_empty() || !names.insert(spec.name.clone()) {
                return Err(Error::backend(format!(
                    "multi-table embedding name {:?} is empty or duplicated",
                    spec.name
                )));
            }
            spec.lookup.validate()?;
            if spec.embedding.vocabulary <= 0 || spec.embedding.dimensions <= 0 {
                return Err(Error::backend(format!(
                    "embedding table {:?} has invalid geometry vocabulary={} dimensions={}",
                    spec.name, spec.embedding.vocabulary, spec.embedding.dimensions
                )));
            }
            if dimensions
                .replace(spec.embedding.dimensions)
                .is_some_and(|expected| expected != spec.embedding.dimensions)
            {
                return Err(Error::backend(format!(
                    "embedding table {:?} width {} differs from preceding width {:?}",
                    spec.name, spec.embedding.dimensions, dimensions
                )));
            }
            tables.push(NamedEmbedding {
                embedding: B::embedding(spec.embedding, context)?,
                name: spec.name,
                lookup: spec.lookup,
            });
        }
        Ok(Self { tables })
    }

    /// Builds rank-local vocabulary tables with validated global ownership.
    pub fn new_vocabulary_parallel(
        specs: impl IntoIterator<Item = NamedEmbeddingSpec>,
        ranges: impl IntoIterator<Item = eredu_nn::VocabularyParallelRange>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error>
    where
        B: eredu_nn::DistributedNeuralBackend,
    {
        let specs = specs.into_iter().collect::<Vec<_>>();
        let ranges = ranges.into_iter().collect::<Vec<_>>();
        if specs.is_empty() || specs.len() != ranges.len() {
            return Err(Error::backend(
                "parallel multi-table embeddings require one range per table",
            ));
        }
        let mut names = std::collections::BTreeSet::new();
        let mut dimensions = None;
        let mut tables = Vec::with_capacity(specs.len());
        for (spec, range) in specs.into_iter().zip(ranges) {
            if spec.name.trim().is_empty() || !names.insert(spec.name.clone()) {
                return Err(Error::backend(
                    "parallel embedding name is empty or duplicated",
                ));
            }
            spec.lookup.validate()?;
            if dimensions
                .replace(spec.embedding.dimensions)
                .is_some_and(|expected| expected != spec.embedding.dimensions)
            {
                return Err(Error::backend("parallel embedding widths differ"));
            }
            tables.push(NamedEmbedding {
                embedding: B::vocabulary_parallel_embedding(spec.embedding, range, context)?,
                name: spec.name,
                lookup: spec.lookup,
            });
        }
        Ok(Self { tables })
    }

    /// Looks up every global-token stream and reduces rank-local contributions.
    pub fn forward_parallel(
        &mut self,
        inputs: &[&B::Tensor],
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        B: eredu_nn::DistributedNeuralBackend,
    {
        if inputs.len() != self.tables.len() {
            return Err(Error::backend("parallel embedding input count drifted"));
        }
        let mut output: Option<B::Tensor> = None;
        for (table, input) in self.tables.iter_mut().zip(inputs) {
            let value = B::vocabulary_parallel_lookup(
                &mut table.embedding,
                input,
                table.lookup,
                parallel,
                context,
            )?;
            output = Some(match output {
                Some(output) => output.add(&value, context)?,
                None => value,
            });
        }
        output.ok_or_else(|| Error::backend("parallel embedding sum is empty"))
    }

    /// Looks up one token tensor per table and sums in declared stream order.
    pub fn forward(
        &mut self,
        tokens: &[&B::Tensor],
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        if tokens.len() != self.tables.len() {
            return Err(Error::backend(format!(
                "multi-table embedding received {} token streams, expected {}",
                tokens.len(),
                self.tables.len()
            )));
        }
        let expected_shape = tokens
            .first()
            .map(|tokens| tokens.shape().to_vec())
            .ok_or_else(|| Error::backend("multi-table embedding received no token streams"))?;
        let mut sum: Option<B::Tensor> = None;
        for (table, tokens) in self.tables.iter_mut().zip(tokens.iter().copied()) {
            if tokens.shape() != expected_shape {
                return Err(Error::backend(format!(
                    "embedding stream {:?} has token shape {:?}, expected {:?}",
                    table.name,
                    tokens.shape(),
                    expected_shape
                )));
            }
            let embedded = table.embedding.lookup(tokens, table.lookup, context)?;
            sum = Some(match sum {
                Some(sum) => sum.add(&embedded, context)?,
                None => embedded,
            });
        }
        sum.ok_or_else(|| Error::backend("multi-table embedding received no tables"))
    }

    /// Returns stable table names in deterministic stream order.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tables.iter().map(|table| table.name.as_str())
    }
}

/// Derives the canonical backend-neutral cache layout for this decoder.
pub fn cache_layout<C: Config>(config: &C) -> Result<LayerSchedule<LayerCachePolicy>, Error> {
    cache_layout_with_key_value_heads(
        config,
        std::iter::repeat_n(
            config.num_key_value_heads(),
            config.attention_schedule().len(),
        ),
    )
}

/// Declares the complete mutable-state geometry consumed by either resident
/// or bounded-residency execution.
pub fn state_layout<C: Config>(config: &C) -> Result<StateLayout, Error> {
    StateLayout::new(cache_layout(config)?).map_err(Error::backend)
}

/// Complete planner-derived construction geometry for one shared decoder rank.
///
/// The value is backend-neutral and is the single source of truth for local
/// unit construction, vocabulary ownership, and mutable cache geometry.
#[derive(Debug, Clone)]
pub struct LocalGeometry<C> {
    blocks: Vec<C>,
    embedding_range: VocabularyParallelRange,
    output_range: Option<VocabularyParallelRange>,
    state_layout: StateLayout,
    architecture_fingerprint: String,
}

impl<C: Config> LocalGeometry<C> {
    /// Returns the rank-local configuration of one decoder block.
    pub fn block(&self, index: usize) -> Option<&C> {
        self.blocks.get(index)
    }

    /// Returns local decoder blocks in global execution order.
    pub fn blocks(&self) -> &[C] {
        &self.blocks
    }

    /// Returns this rank's input-embedding vocabulary ownership.
    pub const fn embedding_range(&self) -> &VocabularyParallelRange {
        &self.embedding_range
    }

    /// Returns this rank's untied output-head vocabulary ownership.
    pub const fn output_range(&self) -> Option<&VocabularyParallelRange> {
        self.output_range.as_ref()
    }

    /// Returns the cache layout derived from the same local block geometry.
    pub const fn state_layout(&self) -> &StateLayout {
        &self.state_layout
    }

    /// Validates that this geometry was derived from this exact normalized
    /// model configuration and that its state/vocabulary views have not
    /// drifted from its local block geometry.
    pub fn validate_for(&self, config: &C) -> Result<(), ParallelPlanError> {
        let layers = usize::try_from(config.num_hidden_layers()).map_err(|_| {
            ParallelPlanError::InvalidGroup("decoder layer count exceeds usize".into())
        })?;
        if self.architecture_fingerprint != config.architecture_fingerprint()
            || self.blocks.len() != layers
        {
            return Err(ParallelPlanError::InvalidGroup(
                "rank-local decoder geometry belongs to a different normalized configuration"
                    .into(),
            ));
        }
        self.embedding_range
            .validate_global_rows(config.vocabulary_size())
            .map_err(|error| ParallelPlanError::InvalidTensor(error.to_string()))?;
        match (config.tie_word_embeddings(), &self.output_range) {
            (true, None) => {}
            (false, Some(range)) => range
                .validate_global_rows(config.vocabulary_size())
                .map_err(|error| ParallelPlanError::InvalidTensor(error.to_string()))?,
            (true, Some(_)) => {
                return Err(ParallelPlanError::InvalidTensor(
                    "tied decoder output has a separate vocabulary range".into(),
                ))
            }
            (false, None) => {
                return Err(ParallelPlanError::InvalidTensor(
                    "untied decoder output has no vocabulary range".into(),
                ))
            }
        }
        let expected = StateLayout::new(
            cache_layout_with_key_value_heads(
                config,
                self.blocks.iter().map(Config::num_key_value_heads),
            )
            .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?,
        )
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
        if expected != self.state_layout {
            return Err(ParallelPlanError::InvalidGroup(
                "rank-local decoder state layout drifted from block geometry".into(),
            ));
        }
        Ok(())
    }
}

/// Derives one shared decoder's complete rank-local geometry from a typed plan.
pub fn local_geometry<C, F>(
    config: &C,
    layout: &LocalModelLayout,
    mut local_block: F,
) -> Result<LocalGeometry<C>, ParallelPlanError>
where
    C: Config,
    F: FnMut(&C, usize, &LocalModelLayout) -> Result<C, ParallelPlanError>,
{
    let layers = usize::try_from(config.num_hidden_layers())
        .map_err(|_| ParallelPlanError::InvalidGroup("decoder layer count exceeds usize".into()))?;
    let blocks = (0..layers)
        .map(|index| local_block(config, index, layout))
        .collect::<Result<Vec<_>, _>>()?;
    let key_value_heads = blocks.iter().map(Config::num_key_value_heads);
    let state_layout = StateLayout::new(
        cache_layout_with_key_value_heads(config, key_value_heads)
            .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?,
    )
    .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    let vocabulary = usize::try_from(config.vocabulary_size()).map_err(|_| {
        ParallelPlanError::InvalidGroup("decoder vocabulary size exceeds usize".into())
    })?;
    let embedding_name = format!("{}.embed_tokens", config.parameter_root());
    let embedding_range = vocabulary_range(layout, &embedding_name, vocabulary)?;
    let output_range = if config.tie_word_embeddings() {
        None
    } else {
        Some(vocabulary_range(layout, "lm_head", vocabulary)?)
    };
    let geometry = LocalGeometry {
        blocks,
        embedding_range,
        output_range,
        state_layout,
        architecture_fingerprint: config.architecture_fingerprint(),
    };
    geometry.validate_for(config)?;
    Ok(geometry)
}

fn vocabulary_range(
    layout: &LocalModelLayout,
    logical_name: &str,
    global_vocabulary: usize,
) -> Result<VocabularyParallelRange, ParallelPlanError> {
    let mut selected: Option<std::ops::Range<usize>> = None;
    let mut found = false;
    for (target, tensor) in layout
        .tensors()
        .filter(|(_, tensor)| tensor.logical_name() == logical_name)
    {
        found = true;
        let range = match tensor.placement() {
            TensorPlacement::Range {
                axis: 0,
                start,
                end,
            } => *start..*end,
            TensorPlacement::Replicated => 0..global_vocabulary,
            placement => {
                return Err(ParallelPlanError::InvalidTensor(format!(
                    "vocabulary member {target} has non-row placement {placement:?}"
                )))
            }
        };
        if tensor.global_shape().first().copied() != Some(global_vocabulary) {
            return Err(ParallelPlanError::InvalidTensor(format!(
                "vocabulary member {target} has global shape {:?}, expected {global_vocabulary} rows",
                tensor.global_shape()
            )));
        }
        if selected.as_ref().is_some_and(|selected| selected != &range) {
            return Err(ParallelPlanError::InvalidTensor(format!(
                "vocabulary group {logical_name} has inconsistent companion selections"
            )));
        }
        selected = Some(range);
    }
    if !found {
        return Err(ParallelPlanError::InvalidTensor(format!(
            "missing local vocabulary layout for {logical_name}"
        )));
    }
    let range = VocabularyParallelRange {
        global_vocabulary,
        local: selected.expect("a found vocabulary member supplies a selection"),
    };
    range
        .validate()
        .map_err(|error| ParallelPlanError::InvalidTensor(error.to_string()))?;
    Ok(range)
}

/// Declares the complete mutable-state geometry consumed by resident or bounded execution.
pub fn cache_layout_with_key_value_heads<C: Config>(
    config: &C,
    key_value_heads: impl IntoIterator<Item = i32>,
) -> Result<LayerSchedule<LayerCachePolicy>, Error> {
    let layers = usize::try_from(config.num_hidden_layers()).map_err(Error::backend)?;
    let key_value_heads = key_value_heads.into_iter().collect::<Vec<_>>();
    if key_value_heads.len() != layers {
        return Err(Error::backend(format!(
            "decoder cache geometry has {} layers, expected {layers}",
            key_value_heads.len()
        )));
    }
    let policies = config
        .attention_schedule()
        .iter()
        .zip(key_value_heads)
        .map(|(attention, key_value_heads)| {
            LayerCachePolicy::key_value(*attention, key_value_heads, config.head_dim())
                .map_err(Error::backend)
        })
        .collect::<Result<Vec<_>, _>>()?;
    LayerSchedule::new(layers, policies).map_err(Error::backend)
}

/// Creates one concrete backend cache per decoder layer from the neutral policy.
///
/// Cache construction is outside inference. The closure is monomorphized and
/// returns the backend's native cache type without boxing or tensor conversion.
pub fn create_caches<C: Config, K>(
    config: &C,
    mut create: impl FnMut(usize, Option<i32>) -> K,
) -> Result<Vec<Option<K>>, Error> {
    validate_schedule(config)?;
    config
        .attention_schedule()
        .iter()
        .enumerate()
        .map(|(layer, policy)| {
            let window = policy
                .window()
                .map(|window| i32::try_from(window.get()))
                .transpose()
                .map_err(Error::backend)?;
            Ok(Some(create(layer, window)))
        })
        .collect()
}

/// Validates that concrete backend caches implement the architecture's policy.
pub fn validate_caches<B, C, K>(config: &C, caches: &[Option<K>]) -> Result<(), Error>
where
    B: NeuralBackend,
    C: Config,
    K: AttentionCache<B::Tensor>,
{
    validate_schedule(config)?;
    if caches.len() != config.attention_schedule().len() {
        return Err(Error::backend(format!(
            "decoder cache has {} layers, expected {}",
            caches.len(),
            config.attention_schedule().len()
        )));
    }
    for (layer, (cache, policy)) in caches
        .iter()
        .zip(config.attention_schedule().iter())
        .enumerate()
    {
        let cache = cache
            .as_ref()
            .ok_or_else(|| Error::backend(format!("decoder cache is missing layer {layer}")))?;
        let expected = policy
            .window()
            .map(|window| i32::try_from(window.get()))
            .transpose()
            .map_err(Error::backend)?;
        if cache.max_size() != expected {
            return Err(Error::backend(format!(
                "decoder cache policy mismatch at layer {layer}: expected {policy:?}, cache window is {:?}",
                cache.max_size()
            )));
        }
    }
    Ok(())
}

fn validate_schedule<C: Config>(config: &C) -> Result<(), Error> {
    let layers = usize::try_from(config.num_hidden_layers()).map_err(Error::backend)?;
    if config.attention_schedule().len() != layers {
        return Err(Error::backend(format!(
            "decoder attention schedule has {} layers, expected {layers}",
            config.attention_schedule().len()
        )));
    }
    Ok(())
}

/// Hidden-state input for one decoder block.
pub struct AttentionInput<'a, T, C> {
    /// Hidden states shaped `[batch, sequence, hidden]`.
    pub hidden: &'a T,
    /// Optional additive or boolean attention mask.
    pub mask: Option<&'a T>,
    /// Optional mutable layer cache.
    pub cache: Option<&'a mut C>,
    /// Whether the block may select its mask-free sliding prefill kernel.
    pub allow_sliding_prefill: bool,
    /// Optional caller-provided explicit rotary position data.
    pub rotary_position: Option<RotaryPosition<'a, T>>,
}

/// Shared grouped-query self attention.
#[derive(Debug, Clone, eredu_nn::Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct FusedAttentionProjection<B: NeuralBackend> {
    /// Component-major affine projection.
    pub projection: B::Linear,
    /// Validated query/key/value component geometry.
    #[parameter(skip)]
    pub layout: FusedProjectionLayout,
}

/// Split or fused physical query/key/value operators feeding one attention path.
#[derive(Debug, Clone, eredu_nn::Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub enum AttentionInputProjection<B: NeuralBackend> {
    /// Independent projections used by conventional decoder checkpoints.
    Split {
        /// Query projection.
        query: B::Linear,
        /// Key projection.
        key: B::Linear,
        /// Value projection.
        value: B::Linear,
    },
    /// One component-major fused projection.
    Fused(FusedAttentionProjection<B>),
}

/// Shared grouped-query self attention.
#[derive(Debug, Clone, eredu_nn::Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct Attention<B: NeuralBackend> {
    /// Number of query heads.
    #[parameter(skip)]
    pub query_heads: i32,
    /// Number of key/value heads.
    #[parameter(skip)]
    pub key_value_heads: i32,
    /// Inverse square-root head scaling.
    #[parameter(skip)]
    pub scale: f32,
    /// Split or fused query/key/value projections.
    pub input_projection: AttentionInputProjection<B>,
    /// Output projection.
    pub output: B::Linear,
    /// Optional learned logit participating in attention softmax for each query head.
    pub sinks: Option<Parameter<B::Tensor>>,
    /// Optional per-head query normalization.
    pub query_norm: Option<B::Normalization>,
    /// Optional per-head key normalization.
    pub key_norm: Option<B::Normalization>,
    /// Optional rotary-position operator for positioned attention families.
    pub rotary: Option<B::Rotary>,
    /// Layer-local sliding window.
    #[parameter(skip)]
    pub sliding_window: Option<i32>,
    /// Whether the query projection's second half gates attended values.
    #[parameter(skip)]
    pub query_output_gate: bool,
}

struct ProjectedAttention<T> {
    queries: T,
    keys: T,
    values: T,
    output_gate: Option<T>,
    batch: i32,
    sequence: i32,
}

impl<B: NeuralBackend> Attention<B> {
    /// Assembles grouped-query attention from architecture-named operators.
    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        query_heads: i32,
        key_value_heads: i32,
        head_dim: i32,
        query: B::Linear,
        key: B::Linear,
        value: B::Linear,
        output: B::Linear,
        query_norm: Option<B::Normalization>,
        key_norm: Option<B::Normalization>,
        rotary: Option<B::Rotary>,
        sliding_window: Option<i32>,
    ) -> Result<Self, Error> {
        Self::from_parts_with_query_gate(
            query_heads,
            key_value_heads,
            head_dim,
            query,
            key,
            value,
            output,
            query_norm,
            key_norm,
            rotary,
            sliding_window,
            false,
        )
    }

    /// Assembles grouped-query attention whose fused query projection carries
    /// an equally sized output gate in its second half.
    #[allow(clippy::too_many_arguments)]
    pub fn from_gated_parts(
        query_heads: i32,
        key_value_heads: i32,
        head_dim: i32,
        query: B::Linear,
        key: B::Linear,
        value: B::Linear,
        output: B::Linear,
        query_norm: Option<B::Normalization>,
        key_norm: Option<B::Normalization>,
        rotary: Option<B::Rotary>,
        sliding_window: Option<i32>,
    ) -> Result<Self, Error> {
        Self::from_parts_with_query_gate(
            query_heads,
            key_value_heads,
            head_dim,
            query,
            key,
            value,
            output,
            query_norm,
            key_norm,
            rotary,
            sliding_window,
            true,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn from_parts_with_query_gate(
        query_heads: i32,
        key_value_heads: i32,
        head_dim: i32,
        query: B::Linear,
        key: B::Linear,
        value: B::Linear,
        output: B::Linear,
        query_norm: Option<B::Normalization>,
        key_norm: Option<B::Normalization>,
        rotary: Option<B::Rotary>,
        sliding_window: Option<i32>,
        query_output_gate: bool,
    ) -> Result<Self, Error> {
        if query_heads <= 0
            || key_value_heads <= 0
            || head_dim <= 0
            || query_heads % key_value_heads != 0
            || sliding_window.is_some_and(|window| window <= 0)
        {
            return Err(Error::backend(format!(
                "invalid grouped-query attention geometry q={query_heads} kv={key_value_heads} dim={head_dim} window={sliding_window:?}"
            )));
        }
        Ok(Self {
            query_heads,
            key_value_heads,
            scale: (head_dim as f32).sqrt().recip(),
            input_projection: AttentionInputProjection::Split { query, key, value },
            output,
            sinks: None,
            query_norm,
            key_norm,
            rotary,
            sliding_window,
            query_output_gate,
        })
    }

    /// Builds unloaded grouped-query attention for one global layer.
    pub fn new<C: Config>(
        config: &C,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        if config.learned_attention_sinks() {
            crate::operator_requirements::require::<B>(
                "shared decoder attention sinks",
                eredu_nn::NeuralOperatorCapabilities::ATTENTION_SINKS,
            )?;
        }
        let fields = config.block_parameter_fields().validate()?;
        let prefix = format!(
            "{}.layers.{layer}.{}",
            config.parameter_root(),
            fields.attention
        );
        let hidden = config.hidden_size();
        let head = config.head_dim();
        let query_heads = config.num_attention_heads();
        let key_value_heads = config.num_key_value_heads();
        let linear = |field: &str, input, output, bias: bool| {
            let weight_name = format!("{prefix}.{field}.weight");
            let bias = bias
                .then(|| ParameterSpec::trainable(format!("{prefix}.{field}.bias")))
                .transpose()
                .map_err(Error::backend)?;
            B::linear(
                LinearSpec {
                    input,
                    output,
                    weight: ParameterSpec::trainable(&weight_name).map_err(Error::backend)?,
                    bias,
                    format: crate::linear_format::standard_linear_format(
                        &weight_name,
                        config.weight_quantization(&weight_name).into(),
                    )?,
                },
                context,
            )
        };
        let policy = config.attention_schedule().get(layer).ok_or_else(|| {
            Error::backend(format!(
                "decoder attention schedule has no policy for layer {layer}"
            ))
        })?;
        let query_width = query_heads
            .checked_mul(head)
            .ok_or_else(|| Error::backend("decoder query projection width overflowed"))?;
        let key_value_width = key_value_heads
            .checked_mul(head)
            .ok_or_else(|| Error::backend("decoder key/value projection width overflowed"))?;
        let input_projection = match config.attention_projection_layout() {
            AttentionProjectionLayout::Split => AttentionInputProjection::Split {
                query: linear(
                    fields.attention_query,
                    hidden,
                    query_width,
                    config.attention_bias(AttentionProjection::Query),
                )?,
                key: linear(
                    fields.attention_key,
                    hidden,
                    key_value_width,
                    config.attention_bias(AttentionProjection::Key),
                )?,
                value: linear(
                    fields.attention_value,
                    hidden,
                    key_value_width,
                    config.attention_bias(AttentionProjection::Value),
                )?,
            },
            AttentionProjectionLayout::Fused { field } => {
                if field.trim().is_empty() {
                    return Err(Error::backend(
                        "fused QKV projection field must not be empty",
                    ));
                }
                let biases = [
                    config.attention_bias(AttentionProjection::Query),
                    config.attention_bias(AttentionProjection::Key),
                    config.attention_bias(AttentionProjection::Value),
                ];
                if biases.iter().any(|bias| *bias != biases[0]) {
                    return Err(Error::backend(
                        "fused QKV projection requires identical query/key/value bias policy",
                    ));
                }
                let layout = FusedProjectionLayout::new([
                    FusedProjectionSegment::new("query", query_width)?,
                    FusedProjectionSegment::new("key", key_value_width)?,
                    FusedProjectionSegment::new("value", key_value_width)?,
                ])?;
                let projection = linear(field, hidden, layout.output_width(), biases[0])?;
                AttentionInputProjection::Fused(FusedAttentionProjection { projection, layout })
            }
        };
        Ok(Self {
            query_heads,
            key_value_heads,
            scale: (head as f32).sqrt().recip(),
            input_projection,
            output: linear(
                fields.attention_output,
                query_width,
                hidden,
                config.attention_bias(AttentionProjection::Output),
            )?,
            sinks: config
                .learned_attention_sinks()
                .then(|| {
                    Parameter::unloaded(
                        ParameterSpec::trainable(format!("{prefix}.{}", fields.attention_sinks))
                            .map_err(Error::backend)?,
                        &[query_heads],
                        context,
                    )
                })
                .transpose()?,
            query_norm: config
                .query_key_norm_epsilon()
                .map(|epsilon| {
                    B::normalization(
                        NormalizationConstructionSpec::learned(
                            head,
                            epsilon,
                            ParameterSpec::trainable(format!(
                                "{prefix}.{}.weight",
                                fields.attention_query_norm
                            ))
                            .map_err(Error::backend)?,
                        ),
                        context,
                    )
                })
                .transpose()?,
            key_norm: config
                .query_key_norm_epsilon()
                .map(|epsilon| {
                    B::normalization(
                        NormalizationConstructionSpec::learned(
                            head,
                            epsilon,
                            ParameterSpec::trainable(format!(
                                "{prefix}.{}.weight",
                                fields.attention_key_norm
                            ))
                            .map_err(Error::backend)?,
                        ),
                        context,
                    )
                })
                .transpose()?,
            rotary: config
                .rotary_enabled()
                .then(|| B::rotary(config.rotary_spec(head), context))
                .transpose()?,
            sliding_window: policy
                .window()
                .map(|window| i32::try_from(window.get()))
                .transpose()
                .map_err(Error::backend)?,
            query_output_gate: false,
        })
    }

    fn projections(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<ProjectedAttention<B::Tensor>, Error> {
        let batch = hidden.dim(0);
        let sequence = hidden.dim(1);
        let reshape = |tensor: B::Tensor, heads| {
            tensor
                .reshape(&[batch, sequence, heads, -1], context)?
                .transpose_axes(&[0, 2, 1, 3], context)
        };
        let (query, key, value) = match &mut self.input_projection {
            AttentionInputProjection::Split { query, key, value } => (
                query.forward(hidden, context)?,
                key.forward(hidden, context)?,
                value.forward(hidden, context)?,
            ),
            AttentionInputProjection::Fused(fused) => {
                let projected = fused.projection.forward(hidden, context)?;
                let mut components = fused.layout.split(&projected, context)?.into_iter();
                let query = components.next().ok_or_else(|| {
                    Error::backend("fused QKV projection is missing its query component")
                })?;
                let key = components.next().ok_or_else(|| {
                    Error::backend("fused QKV projection is missing its key component")
                })?;
                let value = components.next().ok_or_else(|| {
                    Error::backend("fused QKV projection is missing its value component")
                })?;
                if components.next().is_some() {
                    return Err(Error::backend(
                        "fused QKV projection produced unexpected extra components",
                    ));
                }
                (query, key, value)
            }
        };
        let query = query.reshape(&[batch, sequence, self.query_heads, -1], context)?;
        let (query, output_gate) = if self.query_output_gate {
            let projected = query.dim(3);
            if projected <= 0 || projected % 2 != 0 {
                return Err(Error::backend(format!(
                    "gated query projection has invalid final width {projected}"
                )));
            }
            let head = projected / 2;
            (
                query.index(
                    &[Index::Full, Index::Full, Index::Full, Index::Range(0, head)],
                    context,
                )?,
                Some(
                    query
                        .index(
                            &[
                                Index::Full,
                                Index::Full,
                                Index::Full,
                                Index::Range(head, projected),
                            ],
                            context,
                        )?
                        .reshape(&[batch, sequence, self.query_heads * head], context)?,
                ),
            )
        } else {
            (query, None)
        };
        let mut queries = query.transpose_axes(&[0, 2, 1, 3], context)?;
        if let Some(norm) = &mut self.query_norm {
            queries = norm.forward(&queries, context)?;
        }
        let mut keys = reshape(key, self.key_value_heads)?;
        if let Some(norm) = &mut self.key_norm {
            keys = norm.forward(&keys, context)?;
        }
        let values = reshape(value, self.key_value_heads)?;
        Ok(ProjectedAttention {
            queries,
            keys,
            values,
            output_gate,
            batch,
            sequence,
        })
    }

    fn attend<C: AttentionCache<B::Tensor>>(
        &mut self,
        hidden: &B::Tensor,
        mask: Option<&B::Tensor>,
        mut cache: Option<&mut C>,
        allow_sliding_prefill: bool,
        rotary_position: Option<RotaryPosition<'_, B::Tensor>>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let ProjectedAttention {
            queries,
            keys,
            values,
            output_gate,
            batch,
            sequence,
        } = self.projections(hidden, context)?;
        let offset = cache.as_ref().map_or(0, |cache| cache.offset());
        let (queries, keys) = match &mut self.rotary {
            Some(rotary) => {
                let position = rotary_position.unwrap_or(RotaryPosition::Offset(offset));
                (
                    rotary.forward_subspace(&queries, RotarySubspace::Full, position, context)?,
                    rotary.forward_subspace(&keys, RotarySubspace::Full, position, context)?,
                )
            }
            None => (queries, keys),
        };
        let (keys, values) = match cache.as_mut() {
            Some(cache) => cache.update_for_attention(keys, values, context)?,
            None => (keys, values),
        };
        let sinks = self.sinks.as_ref().map(Parameter::as_ref);
        if let Some(window) = self
            .sliding_window
            .filter(|_| allow_sliding_prefill && sequence > 1)
        {
            return B::sliding_window_attention_with_sinks(
                AttentionRequest {
                    queries,
                    keys,
                    values,
                    scale: self.scale,
                    mask,
                    sinks,
                },
                window,
                offset,
                context,
            );
        }
        let request = AttentionRequest {
            queries,
            keys,
            values,
            scale: self.scale,
            mask,
            sinks,
        };
        let attended = if let Some(cache) = cache {
            cache.attention(request, context)?
        } else {
            B::attention_with_sinks(request, context)?
        };
        let attended = attended
            .transpose_axes(&[0, 2, 1, 3], context)?
            .reshape(&[batch, sequence, -1], context)?;
        match output_gate {
            Some(gate) => attended.multiply(&B::sigmoid(gate, context)?, context),
            None => Ok(attended),
        }
    }

    /// Executes grouped-query attention and its output projection.
    pub fn forward<C: AttentionCache<B::Tensor>>(
        &mut self,
        input: AttentionInput<'_, B::Tensor, C>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let attended = self.attend(
            input.hidden,
            input.mask,
            input.cache,
            input.allow_sliding_prefill,
            input.rotary_position,
            context,
        )?;
        self.output.forward(&attended, context)
    }

    /// Executes grouped-query attention with a row-parallel output projection.
    pub fn forward_parallel<C: AttentionCache<B::Tensor>>(
        &mut self,
        input: AttentionInput<'_, B::Tensor, C>,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let attended = self.attend(
            input.hidden,
            input.mask,
            input.cache,
            input.allow_sliding_prefill,
            input.rotary_position,
            context,
        )?;
        B::row_parallel_linear(&mut self.output, &attended, parallel, context)
    }
}

/// One component-major fused gate/up affine projection.
#[derive(Debug, Clone, eredu_nn::Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct FusedGatedProjection<B: NeuralBackend> {
    /// Fused affine operator.
    pub projection: B::Linear,
    /// Validated gate/up component geometry.
    #[parameter(skip)]
    pub layout: FusedProjectionLayout,
}

/// Split or fused physical gate/up operators feeding one gated product.
#[derive(Debug, Clone, eredu_nn::Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub enum GatedInputProjection<B: NeuralBackend> {
    /// Independent gate and up projections.
    Split {
        /// Gate projection.
        gate: B::Linear,
        /// Up projection.
        up: B::Linear,
    },
    /// One component-major fused projection.
    Fused(FusedGatedProjection<B>),
}

/// Shared dense SwiGLU feed-forward network.
#[derive(Debug, Clone, eredu_nn::Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct Mlp<B: NeuralBackend> {
    /// Split or fused gate/up input projection.
    pub input_projection: GatedInputProjection<B>,
    /// Down projection.
    pub down: B::Linear,
    /// Optional shared pre-activation bound.
    #[parameter(skip)]
    pub limit: Option<GatedProductPolicy>,
}

impl<B: NeuralBackend> Mlp<B> {
    /// Assembles a dense gated network from independent gate/up projections.
    pub fn from_parts(
        gate: B::Linear,
        up: B::Linear,
        down: B::Linear,
        policy: Option<GatedProductPolicy>,
    ) -> Self {
        Self {
            input_projection: GatedInputProjection::Split { gate, up },
            down,
            limit: policy,
        }
    }

    /// Builds an unloaded dense SwiGLU network for one global layer.
    pub fn new<C: Config>(
        config: &C,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let fields = config.block_parameter_fields().validate()?;
        let prefix = format!(
            "{}.layers.{layer}.{}",
            config.parameter_root(),
            fields.feed_forward
        );
        let build = |field: &str, input, output| {
            let weight_name = format!("{prefix}.{field}.weight");
            let bias = config
                .mlp_bias()
                .then(|| ParameterSpec::trainable(format!("{prefix}.{field}.bias")))
                .transpose()
                .map_err(Error::backend)?;
            B::linear(
                LinearSpec {
                    input,
                    output,
                    weight: ParameterSpec::trainable(&weight_name).map_err(Error::backend)?,
                    bias,
                    format: crate::linear_format::standard_linear_format(
                        &weight_name,
                        config.weight_quantization(&weight_name).into(),
                    )?,
                },
                context,
            )
        };
        let input_projection = match config.gated_projection_layout() {
            GatedProjectionLayout::Split => GatedInputProjection::Split {
                gate: build(
                    fields.feed_forward_gate,
                    config.hidden_size(),
                    config.intermediate_size(),
                )?,
                up: build(
                    fields.feed_forward_up,
                    config.hidden_size(),
                    config.intermediate_size(),
                )?,
            },
            GatedProjectionLayout::Fused { field } => {
                if field.trim().is_empty() {
                    return Err(Error::backend(
                        "fused gate/up projection field must not be empty",
                    ));
                }
                let layout = FusedProjectionLayout::new([
                    FusedProjectionSegment::new("gate", config.intermediate_size())?,
                    FusedProjectionSegment::new("up", config.intermediate_size())?,
                ])?;
                GatedInputProjection::Fused(FusedGatedProjection {
                    projection: build(field, config.hidden_size(), layout.output_width())?,
                    layout,
                })
            }
        };
        Ok(Self {
            input_projection,
            down: build(
                fields.feed_forward_output,
                config.intermediate_size(),
                config.hidden_size(),
            )?,
            limit: config.gated_product_policy(),
        })
    }

    fn hidden(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let (gate, up) = match &mut self.input_projection {
            GatedInputProjection::Split { gate, up } => {
                (gate.forward(input, context)?, up.forward(input, context)?)
            }
            GatedInputProjection::Fused(fused) => {
                let projected = fused.projection.forward(input, context)?;
                let mut components = fused.layout.split(&projected, context)?.into_iter();
                let gate = components.next().ok_or_else(|| {
                    Error::backend("fused gate/up projection is missing its gate component")
                })?;
                let up = components.next().ok_or_else(|| {
                    Error::backend("fused gate/up projection is missing its up component")
                })?;
                if components.next().is_some() {
                    return Err(Error::backend(
                        "fused gate/up projection produced unexpected extra components",
                    ));
                }
                (gate, up)
            }
        };
        B::gated_product(gate, up, self.limit.unwrap_or_default(), context)
    }

    fn forward(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let hidden = self.hidden(input, context)?;
        self.down.forward(&hidden, context)
    }

    fn forward_parallel(
        &mut self,
        input: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let hidden = self.hidden(input, context)?;
        B::row_parallel_linear(&mut self.down, &hidden, parallel, context)
    }
}

/// Feed-forward policy executed by the shared residual decoder block.
pub trait FeedForwardOperator<B: NeuralBackend>: eredu_nn::Parameterized<B::Tensor> {
    /// Executes replicated feed-forward computation.
    fn forward_feed_forward(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>;
}

/// Additive feed-forward execution for tensor-parallel realizations.
pub trait TensorParallelFeedForwardOperator<B: NeuralBackend>: FeedForwardOperator<B> {
    /// Executes tensor-parallel feed-forward computation.
    fn forward_feed_forward_parallel(
        &mut self,
        input: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>;
}

impl<B: NeuralBackend> FeedForwardOperator<B> for Mlp<B> {
    fn forward_feed_forward(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.forward(input, context)
    }
}

impl<B: NeuralBackend> TensorParallelFeedForwardOperator<B> for Mlp<B> {
    fn forward_feed_forward_parallel(
        &mut self,
        input: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.forward_parallel(input, parallel, context)
    }
}

/// One RMS-pre-norm residual decoder block.
#[derive(Debug, Clone, eredu_nn::Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct TransformerBlock<B: NeuralBackend, F = Mlp<B>> {
    /// Self-attention operator.
    pub self_attention: Attention<B>,
    /// Feed-forward operator.
    pub mlp: F,
    /// Pre-attention RMSNorm.
    pub input_norm: B::Normalization,
    /// Pre-MLP RMSNorm.
    pub post_attention_norm: B::Normalization,
}

impl<B: NeuralBackend> TransformerBlock<B, Mlp<B>> {
    /// Builds an unloaded block for one global layer index.
    pub fn new<C: Config>(
        config: &C,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let fields = config.block_parameter_fields().validate()?;
        Ok(Self {
            self_attention: Attention::new(config, layer, context)?,
            mlp: Mlp::new(config, layer, context)?,
            input_norm: B::normalization(
                NormalizationConstructionSpec::learned(
                    config.hidden_size(),
                    config.rms_norm_epsilon(),
                    ParameterSpec::trainable(format!(
                        "{}.layers.{layer}.{}.weight",
                        config.parameter_root(),
                        fields.input_norm
                    ))
                    .map_err(Error::backend)?,
                ),
                context,
            )?,
            post_attention_norm: B::normalization(
                NormalizationConstructionSpec::learned(
                    config.hidden_size(),
                    config.rms_norm_epsilon(),
                    ParameterSpec::trainable(format!(
                        "{}.layers.{layer}.{}.weight",
                        config.parameter_root(),
                        fields.post_attention_norm
                    ))
                    .map_err(Error::backend)?,
                ),
                context,
            )?,
        })
    }
}

impl<B, F> TransformerBlock<B, F>
where
    B: NeuralBackend,
    F: FeedForwardOperator<B>,
{
    /// Executes this block with replicated projections.
    pub fn forward<C: AttentionCache<B::Tensor>>(
        &mut self,
        input: AttentionInput<'_, B::Tensor, C>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let normalized = self.input_norm.forward(input.hidden, context)?;
        let attention = self.self_attention.forward(
            AttentionInput {
                hidden: &normalized,
                mask: input.mask,
                cache: input.cache,
                allow_sliding_prefill: input.allow_sliding_prefill,
                rotary_position: input.rotary_position,
            },
            context,
        )?;
        let hidden = input.hidden.add(&attention, context)?;
        let normalized = self.post_attention_norm.forward(&hidden, context)?;
        let mlp = self.mlp.forward_feed_forward(&normalized, context)?;
        hidden.add(&mlp, context)
    }

    /// Executes attention and residuals while delegating feed-forward execution.
    pub fn forward_with_feed_forward<C, H>(
        &mut self,
        input: AttentionInput<'_, B::Tensor, C>,
        context: &<B::Tensor as Tensor>::Context,
        feed_forward: H,
    ) -> Result<B::Tensor, Error>
    where
        C: AttentionCache<B::Tensor>,
        H: FnOnce(&mut F, &B::Tensor, &<B::Tensor as Tensor>::Context) -> Result<B::Tensor, Error>,
    {
        let normalized = self.input_norm.forward(input.hidden, context)?;
        let attention = self.self_attention.forward(
            AttentionInput {
                hidden: &normalized,
                mask: input.mask,
                cache: input.cache,
                allow_sliding_prefill: input.allow_sliding_prefill,
                rotary_position: input.rotary_position,
            },
            context,
        )?;
        let hidden = input.hidden.add(&attention, context)?;
        let normalized = self.post_attention_norm.forward(&hidden, context)?;
        let mlp = feed_forward(&mut self.mlp, &normalized, context)?;
        hidden.add(&mlp, context)
    }

    /// Executes a block with rank-local column projections and reduced row projections.
    pub fn forward_tensor_parallel<C: AttentionCache<B::Tensor>>(
        &mut self,
        input: AttentionInput<'_, B::Tensor, C>,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        F: TensorParallelFeedForwardOperator<B>,
    {
        let normalized = self.input_norm.forward(input.hidden, context)?;
        let attention = self.self_attention.forward_parallel(
            AttentionInput {
                hidden: &normalized,
                mask: input.mask,
                cache: input.cache,
                allow_sliding_prefill: input.allow_sliding_prefill,
                rotary_position: input.rotary_position,
            },
            parallel,
            context,
        )?;
        let hidden = input.hidden.add(&attention, context)?;
        let normalized = self.post_attention_norm.forward(&hidden, context)?;
        let mlp = self
            .mlp
            .forward_feed_forward_parallel(&normalized, parallel, context)?;
        hidden.add(&mlp, context)
    }

    /// Executes tensor-parallel attention and residuals with delegated feed-forward execution.
    pub fn forward_tensor_parallel_with_feed_forward<C, H>(
        &mut self,
        input: AttentionInput<'_, B::Tensor, C>,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        feed_forward: H,
    ) -> Result<B::Tensor, Error>
    where
        C: AttentionCache<B::Tensor>,
        H: FnOnce(&mut F, &B::Tensor, &<B::Tensor as Tensor>::Context) -> Result<B::Tensor, Error>,
    {
        let normalized = self.input_norm.forward(input.hidden, context)?;
        let attention = self.self_attention.forward_parallel(
            AttentionInput {
                hidden: &normalized,
                mask: input.mask,
                cache: input.cache,
                allow_sliding_prefill: input.allow_sliding_prefill,
                rotary_position: input.rotary_position,
            },
            parallel,
            context,
        )?;
        let hidden = input.hidden.add(&attention, context)?;
        let normalized = self.post_attention_norm.forward(&hidden, context)?;
        let mlp = feed_forward(&mut self.mlp, &normalized, context)?;
        hidden.add(&mlp, context)
    }
}

/// Declares attention and normalization groups shared by dense and routed blocks.
pub fn block_common_parallel_parameter_groups<B: NeuralBackend, F>(
    block: &TransformerBlock<B, F>,
    config: &impl Config,
    layer: usize,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    let prefix = format!("{}.layers.{layer}", config.parameter_root());
    let fields = config
        .block_parameter_fields()
        .validate()
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    let attention_prefix = format!("{prefix}.{}", fields.attention);
    let query_heads = usize::try_from(config.num_attention_heads()).map_err(|_| {
        ParallelPlanError::InvalidGroup("decoder query-head count exceeds usize".into())
    })?;
    let key_value_heads = usize::try_from(config.num_key_value_heads()).map_err(|_| {
        ParallelPlanError::InvalidGroup("decoder key/value-head count exceeds usize".into())
    })?;
    let head_dimension = usize::try_from(config.head_dim()).map_err(|_| {
        ParallelPlanError::InvalidGroup("decoder head dimension exceeds usize".into())
    })?;
    if head_dimension == 0 || key_value_heads == 0 || !query_heads.is_multiple_of(key_value_heads) {
        return Err(ParallelPlanError::InvalidGroup(format!(
            "decoder attention geometry q={query_heads}, kv={key_value_heads}, dim={head_dimension} does not form positive integral GQA groups"
        )));
    }
    let group_width = (query_heads / key_value_heads)
        .checked_mul(head_dimension)
        .ok_or_else(|| {
            ParallelPlanError::InvalidGroup("decoder GQA group width overflowed".into())
        })?;
    let attention_alignment = config
        .weight_quantization(&format!(
            "{attention_prefix}.{}.weight",
            fields.attention_output
        ))
        .map_or(Ok(1), |quantization| {
            usize::try_from(quantization.group_size()).map_err(|_| {
                ParallelPlanError::InvalidGroup(
                    "decoder output-projection quantization group exceeds usize".into(),
                )
            })
        })?;
    let attention_units = aligned_partition_units(
        &attention_prefix,
        key_value_heads,
        group_width,
        attention_alignment,
    )?;
    let attention = match &block.self_attention.input_projection {
        AttentionInputProjection::Split { query, key, value } => {
            partitioned_projection_group::<B::Tensor, B::Linear>(
                format!("{attention_prefix}.projections"),
                ParameterRole::AttentionHeads,
                &[
                    (query, ProjectionSharding::Column),
                    (key, ProjectionSharding::Column),
                    (value, ProjectionSharding::Column),
                    (&block.self_attention.output, ProjectionSharding::Row),
                ],
                attention_units,
            )?
        }
        AttentionInputProjection::Fused(fused) => {
            segmented_projection_group::<B::Tensor, B::Linear>(
                format!("{attention_prefix}.projections"),
                ParameterRole::AttentionHeads,
                &fused.projection,
                &block.self_attention.output,
                fused_projection_ranges(&fused.layout)?,
                attention_units,
            )?
        }
    };

    let input_norm = module_parameter_group::<B::Tensor, _>(
        format!("{prefix}.{}", fields.input_norm),
        ParameterRole::Replicated,
        &block.input_norm,
        |_, _| Ok(MemberSharding::Replicated),
    )?;
    let post_attention_norm = module_parameter_group::<B::Tensor, _>(
        format!("{prefix}.{}", fields.post_attention_norm),
        ParameterRole::Replicated,
        &block.post_attention_norm,
        |_, _| Ok(MemberSharding::Replicated),
    )?;
    let mut groups = vec![attention];
    if let Some(sinks) = &block.self_attention.sinks {
        groups.push(partitioned_module_parameter_group::<B::Tensor, _>(
            format!("{attention_prefix}.{}", fields.attention_sinks),
            ParameterRole::AttentionHeads,
            query_heads,
            sinks,
            |_, shape| {
                if shape != [query_heads] {
                    return Err(ParallelPlanError::InvalidTensor(format!(
                        "decoder attention sinks have shape {shape:?}, expected [{query_heads}]"
                    )));
                }
                Ok(MemberSharding::Partitioned { axis: 0 })
            },
        )?);
    }
    if let Some(norm) = &block.self_attention.query_norm {
        groups.push(module_parameter_group::<B::Tensor, _>(
            format!("{attention_prefix}.{}", fields.attention_query_norm),
            ParameterRole::Replicated,
            norm,
            |_, _| Ok(MemberSharding::Replicated),
        )?);
    }
    if let Some(norm) = &block.self_attention.key_norm {
        groups.push(module_parameter_group::<B::Tensor, _>(
            format!("{attention_prefix}.{}", fields.attention_key_norm),
            ParameterRole::Replicated,
            norm,
            |_, _| Ok(MemberSharding::Replicated),
        )?);
    }
    groups.extend([input_norm, post_attention_norm]);
    Ok(groups)
}

/// Declares the dense SwiGLU placement group shared by dense decoder families.
pub fn dense_mlp_parallel_parameter_group<B: NeuralBackend>(
    mlp: &Mlp<B>,
    config: &impl Config,
    layer: usize,
) -> Result<ParameterGroupSpec, ParallelPlanError> {
    let fields = config
        .block_parameter_fields()
        .validate()
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    let prefix = format!(
        "{}.layers.{layer}.{}",
        config.parameter_root(),
        fields.feed_forward
    );
    let intermediate = usize::try_from(config.intermediate_size()).map_err(|_| {
        ParallelPlanError::InvalidGroup("decoder feed-forward width exceeds usize".into())
    })?;
    let alignment = config
        .weight_quantization(&format!("{prefix}.{}.weight", fields.feed_forward_output))
        .map_or(Ok(1), |quantization| {
            usize::try_from(quantization.group_size()).map_err(|_| {
                ParallelPlanError::InvalidGroup(
                    "decoder down-projection quantization group exceeds usize".into(),
                )
            })
        })?;
    let units = aligned_partition_units(&prefix, intermediate, 1, alignment)?;
    match &mlp.input_projection {
        GatedInputProjection::Split { gate, up } => {
            partitioned_projection_group::<B::Tensor, B::Linear>(
                format!("{prefix}.projections"),
                ParameterRole::FeedForwardIntermediate,
                &[
                    (gate, ProjectionSharding::Column),
                    (up, ProjectionSharding::Column),
                    (&mlp.down, ProjectionSharding::Row),
                ],
                units,
            )
        }
        GatedInputProjection::Fused(fused) => segmented_projection_group::<B::Tensor, B::Linear>(
            format!("{prefix}.projections"),
            ParameterRole::FeedForwardIntermediate,
            &fused.projection,
            &mlp.down,
            fused_projection_ranges(&fused.layout)?,
            units,
        ),
    }
}

fn fused_projection_ranges(
    layout: &FusedProjectionLayout,
) -> Result<Vec<std::ops::Range<usize>>, ParallelPlanError> {
    let mut start = 0usize;
    layout
        .segments()
        .iter()
        .map(|segment| {
            let width = usize::try_from(segment.width()).map_err(|_| {
                ParallelPlanError::InvalidTensor(format!(
                    "fused projection segment {} exceeds usize",
                    segment.name()
                ))
            })?;
            let end = start.checked_add(width).ok_or_else(|| {
                ParallelPlanError::InvalidTensor(
                    "fused projection segment ranges overflowed usize".into(),
                )
            })?;
            let range = start..end;
            start = end;
            Ok(range)
        })
        .collect()
}

/// Declares every rank-local placement group for one dense shared decoder block.
pub fn layer_parallel_parameter_groups<B: NeuralBackend>(
    block: &TransformerBlock<B>,
    config: &impl Config,
    layer: usize,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    let mut groups = block_common_parallel_parameter_groups(block, config, layer)?;
    groups.push(dense_mlp_parallel_parameter_group(
        &block.mlp, config, layer,
    )?);
    Ok(groups)
}

/// Derives the rank-local construction geometry of one tensor-parallel block
/// from the neutral placement layout.
pub fn static_parallel_parameter_groups<B: NeuralBackend>(
    embeddings: &B::Embedding,
    norm: &B::Normalization,
    head: Option<&B::Linear>,
    parameter_root: &str,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    let mut groups = vec![
        module_parameter_group::<B::Tensor, _>(
            format!("{parameter_root}.embed_tokens"),
            ParameterRole::Vocabulary,
            embeddings,
            |_, shape| {
                if shape.is_empty() {
                    Err(ParallelPlanError::InvalidTensor(
                        "decoder embedding parameter is scalar".into(),
                    ))
                } else {
                    Ok(MemberSharding::Balanced { axis: 0 })
                }
            },
        )?,
        module_parameter_group::<B::Tensor, _>(
            format!("{parameter_root}.norm"),
            ParameterRole::Replicated,
            norm,
            |_, _| Ok(MemberSharding::Replicated),
        )?,
    ];
    if let Some(head) = head {
        groups.push(module_parameter_group::<B::Tensor, _>(
            "lm_head",
            ParameterRole::Vocabulary,
            head,
            |_, shape| {
                if shape.is_empty() {
                    Err(ParallelPlanError::InvalidTensor(
                        "decoder language-model head parameter is scalar".into(),
                    ))
                } else {
                    Ok(MemberSharding::Balanced { axis: 0 })
                }
            },
        )?);
    }
    Ok(groups)
}

/// Pinned modules shared by resident and bounded-residency execution.
#[derive(Debug, eredu_nn::Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct StaticModules<B: NeuralBackend> {
    /// Token embedding table.
    pub embeddings: B::Embedding,
    /// Final RMSNorm.
    pub norm: B::Normalization,
    /// Optional untied vocabulary projection.
    pub lm_head: Option<B::Linear>,
}

impl<B: NeuralBackend> Clone for StaticModules<B> {
    fn clone(&self) -> Self {
        Self {
            embeddings: self.embeddings.clone(),
            norm: self.norm.clone(),
            lm_head: self.lm_head.clone(),
        }
    }
}

/// Architecture-supplied identities and geometry for the shared pinned text
/// modules.
#[derive(Debug, Clone)]
pub struct StaticModuleSpec {
    /// Token embedding parameter identity.
    pub embedding_weight: String,
    /// Final normalization parameter identity.
    pub normalization_weight: String,
    /// Untied output-head parameter identity.
    pub head_weight: String,
    /// Vocabulary row count.
    pub vocabulary: i32,
    /// Hidden width.
    pub hidden_size: i32,
    /// Final RMS normalization epsilon.
    pub normalization_epsilon: f32,
    /// Fixed scalar added to the learned final-normalization scale.
    pub normalization_offset: f32,
    /// Packed embedding format, when supported by the general embedding operator.
    pub embedding_quantization: Option<WeightQuantization>,
    /// Complete output-head physical encoding.
    pub head_format: LinearFormat,
    /// Whether output logits reuse the embedding table.
    pub tied_head: bool,
}

impl<B: NeuralBackend> StaticModules<B> {
    /// Builds unloaded pinned modules from architecture-owned parameter
    /// identities and physical formats.
    pub fn from_spec(
        spec: StaticModuleSpec,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let embeddings = B::embedding(
            EmbeddingSpec {
                vocabulary: spec.vocabulary,
                dimensions: spec.hidden_size,
                weight: ParameterSpec::trainable(&spec.embedding_weight).map_err(Error::backend)?,
                format: crate::linear_format::standard_linear_format(
                    &spec.embedding_weight,
                    spec.embedding_quantization.into(),
                )?,
            },
            context,
        )?;
        let normalization_weight =
            ParameterSpec::trainable(&spec.normalization_weight).map_err(Error::backend)?;
        let norm = B::normalization(
            eredu_nn::NormalizationConstructionSpec {
                dimensions: spec.hidden_size,
                epsilon: spec.normalization_epsilon,
                scale: if spec.normalization_offset == 0.0 {
                    eredu_nn::NormalizationScale::Learned(normalization_weight)
                } else {
                    eredu_nn::NormalizationScale::LearnedOffset {
                        weight: normalization_weight,
                        offset: spec.normalization_offset,
                    }
                },
            },
            context,
        )?;
        let lm_head = if spec.tied_head {
            None
        } else {
            Some(B::linear(
                LinearSpec {
                    input: spec.hidden_size,
                    output: spec.vocabulary,
                    weight: ParameterSpec::trainable(&spec.head_weight).map_err(Error::backend)?,
                    bias: None,
                    format: crate::linear_format::standard_linear_format(
                        &spec.head_weight,
                        spec.head_format,
                    )?,
                },
                context,
            )?)
        };
        Ok(Self {
            embeddings,
            norm,
            lm_head,
        })
    }

    /// Builds the same pinned modules with planner-derived vocabulary ownership.
    pub fn from_parallel_spec(
        spec: StaticModuleSpec,
        embedding_range: VocabularyParallelRange,
        output_range: Option<VocabularyParallelRange>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error>
    where
        B: eredu_nn::DistributedNeuralBackend,
    {
        embedding_range.validate_global_rows(spec.vocabulary)?;
        let embeddings = B::vocabulary_parallel_embedding(
            EmbeddingSpec {
                vocabulary: spec.vocabulary,
                dimensions: spec.hidden_size,
                weight: ParameterSpec::trainable(&spec.embedding_weight).map_err(Error::backend)?,
                format: crate::linear_format::standard_linear_format(
                    &spec.embedding_weight,
                    spec.embedding_quantization.into(),
                )?,
            },
            embedding_range,
            context,
        )?;
        let normalization_weight =
            ParameterSpec::trainable(&spec.normalization_weight).map_err(Error::backend)?;
        let norm = B::normalization(
            eredu_nn::NormalizationConstructionSpec {
                dimensions: spec.hidden_size,
                epsilon: spec.normalization_epsilon,
                scale: if spec.normalization_offset == 0.0 {
                    eredu_nn::NormalizationScale::Learned(normalization_weight)
                } else {
                    eredu_nn::NormalizationScale::LearnedOffset {
                        weight: normalization_weight,
                        offset: spec.normalization_offset,
                    }
                },
            },
            context,
        )?;
        let lm_head = match (spec.tied_head, output_range) {
            (true, None) => None,
            (true, Some(_)) => {
                return Err(Error::backend(
                    "tied decoder output must not declare separate vocabulary ownership",
                ))
            }
            (false, None) => {
                return Err(Error::backend(
                    "untied decoder output is missing vocabulary ownership",
                ))
            }
            (false, Some(range)) => {
                range.validate_global_rows(spec.vocabulary)?;
                Some(B::vocabulary_parallel_linear(
                    LinearSpec {
                        input: spec.hidden_size,
                        output: spec.vocabulary,
                        weight: ParameterSpec::trainable(&spec.head_weight)
                            .map_err(Error::backend)?,
                        bias: None,
                        format: crate::linear_format::standard_linear_format(
                            &spec.head_weight,
                            spec.head_format,
                        )?,
                    },
                    range,
                    context,
                )?)
            }
        };
        Ok(Self {
            embeddings,
            norm,
            lm_head,
        })
    }

    /// Builds unloaded pinned modules for a decoder family.
    pub fn new<C: Config>(
        config: &C,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let embedding_name = format!("{}.embed_tokens.weight", config.parameter_root());
        let norm_name = format!("{}.norm.weight", config.parameter_root());
        Self::from_spec(
            StaticModuleSpec {
                embedding_weight: embedding_name.clone(),
                normalization_weight: norm_name,
                head_weight: "lm_head.weight".into(),
                vocabulary: config.vocabulary_size(),
                hidden_size: config.hidden_size(),
                normalization_epsilon: config.rms_norm_epsilon(),
                normalization_offset: 0.0,
                embedding_quantization: config.weight_quantization(&embedding_name),
                head_format: config.weight_quantization("lm_head.weight").into(),
                tied_head: config.tie_word_embeddings(),
            },
            context,
        )
    }

    /// Builds rank-local pinned modules for a decoder family.
    pub fn new_parallel<C: Config>(
        config: &C,
        geometry: &LocalGeometry<C>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error>
    where
        B: eredu_nn::DistributedNeuralBackend,
    {
        let embedding_name = format!("{}.embed_tokens.weight", config.parameter_root());
        let norm_name = format!("{}.norm.weight", config.parameter_root());
        Self::from_parallel_spec(
            StaticModuleSpec {
                embedding_weight: embedding_name.clone(),
                normalization_weight: norm_name,
                head_weight: "lm_head.weight".into(),
                vocabulary: config.vocabulary_size(),
                hidden_size: config.hidden_size(),
                normalization_epsilon: config.rms_norm_epsilon(),
                normalization_offset: 0.0,
                embedding_quantization: config.weight_quantization(&embedding_name),
                head_format: config.weight_quantization("lm_head.weight").into(),
                tied_head: config.tie_word_embeddings(),
            },
            geometry.embedding_range().clone(),
            geometry.output_range().cloned(),
            context,
        )
    }
}

/// Borrowed token input for the shared layered lifecycle.
pub struct LayeredInput<'a, T> {
    /// Token ids shaped `[batch, sequence]`.
    pub tokens: &'a T,
    /// Optional caller-provided attention mask.
    pub mask: Option<&'a T>,
}

/// Shared declaration and validation for one ordered decoder execution group.
#[derive(Debug, Clone)]
pub struct SequentialGroup {
    name: &'static str,
    parameter_root: &'static str,
    units: usize,
}

/// Shared declaration for a target group followed by zero or more ordered
/// prediction groups.
#[derive(Debug, Clone)]
pub struct SequentialPredictionGroups {
    target: SequentialGroup,
    prediction_paths: Vec<Vec<String>>,
}

impl SequentialPredictionGroups {
    /// Creates the target group and `mtp.{depth}` prediction groups.
    pub fn new(
        target_parameter_root: &'static str,
        target_units: usize,
        prediction_roots: impl IntoIterator<Item = String>,
    ) -> Result<Self, Error> {
        Ok(Self {
            target: SequentialGroup::new(
                TARGET_EXECUTION_GROUP,
                target_parameter_root,
                target_units,
            )?,
            prediction_paths: prediction_roots
                .into_iter()
                .map(|root| vec![root])
                .collect(),
        })
    }

    /// Creates equally sized appended prediction groups over one physical namespace.
    pub fn new_pattern(
        target_parameter_root: &'static str,
        target_units: usize,
        prediction_parameter_root: &'static str,
        prediction_groups: usize,
        units_per_group: usize,
    ) -> Result<Self, Error> {
        if (prediction_groups != 0 && units_per_group == 0) || prediction_parameter_root.is_empty()
        {
            return Err(Error::backend(
                "prediction execution groups require non-empty names and units",
            ));
        }
        let prediction_paths = (0..prediction_groups)
            .map(|group| {
                let start = group
                    .checked_mul(units_per_group)
                    .ok_or_else(|| Error::backend("prediction physical index overflowed"))?;
                (0..units_per_group)
                    .map(|unit| {
                        start
                            .checked_add(unit)
                            .map(|physical| format!("{prediction_parameter_root}.{physical}"))
                            .ok_or_else(|| Error::backend("prediction physical index overflowed"))
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            target: SequentialGroup::new(
                TARGET_EXECUTION_GROUP,
                target_parameter_root,
                target_units,
            )?,
            prediction_paths,
        })
    }

    /// Builds one dependency chain from target through every prediction depth.
    pub fn execution_graph(&self) -> Result<eredu_runtime::ExecutionGraph, Error> {
        eredu_runtime::ExecutionGraph::chain(
            std::iter::once(TARGET_EXECUTION_GROUP.to_owned())
                .chain(self.prediction_execution_groups()),
        )
        .map_err(Error::backend)
    }

    /// Returns stable prediction-group identities in prediction-depth order.
    pub fn prediction_execution_groups(&self) -> Vec<String> {
        (0..self.prediction_paths.len())
            .map(|depth| format!("mtp.{depth}"))
            .collect()
    }

    /// Returns the number of units in one group.
    pub fn unit_count(&self, group: usize) -> Result<usize, Error> {
        if group == 0 {
            self.target.unit_count(0)
        } else if group <= self.prediction_paths.len() {
            Ok(self.prediction_paths[group - 1].len())
        } else {
            Err(Error::backend(format!(
                "execution group {group} is outside target plus {} prediction groups",
                self.prediction_paths.len()
            )))
        }
    }

    /// Returns one stable target or prediction unit path.
    pub fn unit_path(&self, group: usize, index: usize) -> Result<String, Error> {
        if group == 0 {
            return self.target.unit_path(0, index);
        }
        self.unit_count(group)?;
        self.prediction_paths[group - 1]
            .get(index)
            .cloned()
            .ok_or_else(|| {
                Error::backend(format!(
                    "unit {index} is outside {} units in prediction group {group}",
                    self.prediction_paths[group - 1].len()
                ))
            })
    }

    /// Selects the activation carried into a ready chain group.
    pub fn begin<T: Clone>(
        &self,
        group: usize,
        initial: &T,
        dependencies: &[&T],
    ) -> Result<T, Error> {
        self.unit_count(group)?;
        if group == 0 {
            return self.target.begin(0, initial, dependencies);
        }
        match dependencies {
            [dependency] => Ok((*dependency).clone()),
            _ => Err(Error::backend(format!(
                "prediction group {group} expected one dependency, received {}",
                dependencies.len()
            ))),
        }
    }

    /// Returns the number of appended prediction groups.
    pub fn prediction_count(&self) -> usize {
        self.prediction_paths.len()
    }
}

impl SequentialGroup {
    /// Creates one non-empty ordered group.
    pub fn new(
        name: &'static str,
        parameter_root: &'static str,
        units: usize,
    ) -> Result<Self, Error> {
        if name.is_empty() || parameter_root.is_empty() || units == 0 {
            return Err(Error::backend(
                "sequential decoder group requires non-empty names and units",
            ));
        }
        Ok(Self {
            name,
            parameter_root,
            units,
        })
    }

    /// Builds the corresponding one-group dependency graph.
    pub fn execution_graph(&self) -> Result<eredu_runtime::ExecutionGraph, Error> {
        eredu_runtime::ExecutionGraph::chain([self.name]).map_err(Error::backend)
    }

    /// Validates the group ordinal and returns its unit count.
    pub fn unit_count(&self, group: usize) -> Result<usize, Error> {
        if group != 0 {
            return Err(Error::backend(format!(
                "execution group {group} is outside {}",
                self.name
            )));
        }
        Ok(self.units)
    }

    /// Returns one validated stable unit path.
    pub fn unit_path(&self, group: usize, index: usize) -> Result<String, Error> {
        let count = self.unit_count(group)?;
        if index >= count {
            return Err(Error::backend(format!(
                "unit {index} is outside {count} {} units",
                self.name
            )));
        }
        Ok(format!("{}.{index}", self.parameter_root))
    }

    /// Starts the sole group from the initial activation.
    pub fn begin<T: Clone>(
        &self,
        group: usize,
        initial: &T,
        dependencies: &[&T],
    ) -> Result<T, Error> {
        self.unit_count(group)?;
        if !dependencies.is_empty() {
            return Err(Error::backend(format!(
                "{} received {} unexpected dependencies",
                self.name,
                dependencies.len()
            )));
        }
        Ok(initial.clone())
    }
}

/// Architecture-owned values retained across one layered forward pass.
pub struct ForwardContext<T> {
    mask: Option<T>,
    allow_sliding_prefill: bool,
    rotary_embeddings: Option<(T, T)>,
}

/// Statically dispatched construction policy for one decoder block family.
pub trait BlockFactory<B: NeuralBackend, C: Config>: 'static {
    /// Architecture-selected feed-forward policy inside the shared block.
    type FeedForward: FeedForwardOperator<B>;

    /// Validates configuration requirements specific to this block policy.
    fn validate(config: &C) -> Result<(), Error> {
        let _ = config;
        Ok(())
    }

    /// Builds one unloaded decoder block.
    fn build(
        config: &C,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<TransformerBlock<B, Self::FeedForward>, Error>;

    /// Declares the complete neutral parameter placement for one built block.
    fn parameter_groups(
        block: &TransformerBlock<B, Self::FeedForward>,
        config: &C,
        layer: usize,
    ) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError>;
}

/// Feed-forward policy that can delegate routed experts to runtime residency.
pub trait RoutedFeedForwardOperator<B: GroupedNeuralBackend>: FeedForwardOperator<B> {
    /// Executes replicated dense or provider-backed routed work.
    fn forward_with_provider<P>(
        &mut self,
        layer: usize,
        input: &B::Tensor,
        pass: eredu_runtime::ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        P: eredu_runtime::RoutedExpertProvider<B>,
        P::Error: std::fmt::Display;
}

/// Additive routed feed-forward execution for tensor-parallel realizations.
pub trait TensorParallelRoutedFeedForwardOperator<B: GroupedNeuralBackend>:
    RoutedFeedForwardOperator<B> + TensorParallelFeedForwardOperator<B>
{
    /// Executes tensor-parallel dense or provider-backed routed work.
    #[allow(clippy::too_many_arguments)]
    fn forward_parallel_with_provider<P>(
        &mut self,
        layer: usize,
        input: &B::Tensor,
        pass: eredu_runtime::ExpertPass,
        provider: &mut P,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        P: eredu_runtime::TensorParallelRoutedExpertProvider<B>,
        P::Error: std::fmt::Display;
}

/// Dense SwiGLU block factory used by Llama and other all-dense decoders.
pub struct DenseBlockFactory;

impl<B: NeuralBackend, C: Config> BlockFactory<B, C> for DenseBlockFactory {
    type FeedForward = Mlp<B>;

    fn build(
        config: &C,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<TransformerBlock<B, Self::FeedForward>, Error> {
        TransformerBlock::new(config, layer, context)
    }

    fn parameter_groups(
        block: &TransformerBlock<B, Self::FeedForward>,
        config: &C,
        layer: usize,
    ) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
        layer_parallel_parameter_groups(block, config, layer)
    }
}

/// Shared layered decoder lifecycle over architecture configuration and block policy.
pub struct LayeredModel<B: NeuralBackend, C: Config, P = DenseBlockFactory> {
    args: C,
    static_modules: StaticModules<B>,
    parallel_geometry: Option<std::sync::Arc<LocalGeometry<C>>>,
    block_factory: std::marker::PhantomData<fn() -> P>,
}

impl<B, C, P> LayeredModel<B, C, P>
where
    B: NeuralBackend,
    C: Config,
    P: BlockFactory<B, C>,
{
    /// Builds unloaded pinned modules from normalized architecture arguments.
    pub fn new(args: C, context: &<B::Tensor as Tensor>::Context) -> Result<Self, Error> {
        args.validate_config()?;
        P::validate(&args)?;
        if args.learned_attention_sinks() {
            crate::operator_requirements::require::<B>(
                "shared decoder attention sinks",
                eredu_nn::NeuralOperatorCapabilities::ATTENTION_SINKS,
            )?;
        }
        let static_modules = StaticModules::new(&args, context)?;
        Ok(Self {
            args,
            static_modules,
            parallel_geometry: None,
            block_factory: std::marker::PhantomData,
        })
    }

    /// Builds the same model lifecycle with planner-derived rank-local modules.
    pub fn new_parallel(
        args: C,
        geometry: LocalGeometry<C>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error>
    where
        B: eredu_nn::DistributedNeuralBackend,
    {
        args.validate_config()?;
        P::validate(&args)?;
        if args.learned_attention_sinks() {
            crate::operator_requirements::require::<B>(
                "shared decoder attention sinks",
                eredu_nn::NeuralOperatorCapabilities::ATTENTION_SINKS,
            )?;
        }
        geometry.validate_for(&args).map_err(Error::backend)?;
        let static_modules = StaticModules::new_parallel(&args, &geometry, context)?;
        Ok(Self {
            args,
            static_modules,
            parallel_geometry: Some(std::sync::Arc::new(geometry)),
            block_factory: std::marker::PhantomData,
        })
    }

    /// Returns the normalized architecture arguments.
    pub const fn args(&self) -> &C {
        &self.args
    }

    /// Borrows pinned modules for neutral checkpoint loading.
    pub const fn static_modules(&self) -> &StaticModules<B> {
        &self.static_modules
    }

    /// Mutably borrows pinned modules for neutral checkpoint binding.
    pub fn static_modules_mut(&mut self) -> &mut StaticModules<B> {
        &mut self.static_modules
    }

    /// Returns the replicated or rank-local mutable-state layout for this model.
    fn state_layout_impl(&self) -> Result<StateLayout, Error> {
        self.parallel_geometry
            .as_ref()
            .map(|geometry| geometry.state_layout().clone())
            .map_or_else(|| state_layout(&self.args), Ok)
    }

    /// Returns planner-derived geometry when this is a rank-local realization.
    pub fn parallel_geometry(&self) -> Option<&LocalGeometry<C>> {
        match self.parallel_geometry.as_ref() {
            Some(geometry) => Some(geometry.as_ref()),
            None => None,
        }
    }

    /// Shares the authoritative local geometry with a backend residency policy.
    pub fn shared_parallel_geometry(&self) -> Option<std::sync::Arc<LocalGeometry<C>>> {
        self.parallel_geometry.as_ref().map(std::sync::Arc::clone)
    }

    /// Constructs one canonical replicated or rank-local decoder unit.
    ///
    /// Residency and pipeline runtimes use this entry point so streamed units
    /// cannot drift from the model's planner-derived geometry.
    pub fn construct_unit(
        &self,
        index: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<TransformerBlock<B, P::FeedForward>, Error> {
        let count = usize::try_from(self.args.num_hidden_layers()).map_err(Error::backend)?;
        if index >= count {
            return Err(Error::backend(format!(
                "decoder unit {index} is outside {count} decoder layers"
            )));
        }
        let args = match &self.parallel_geometry {
            Some(geometry) => geometry.block(index).ok_or_else(|| {
                Error::backend(format!("decoder local geometry is missing block {index}"))
            })?,
            None => &self.args,
        };
        P::build(args, index, context)
    }

    /// Prepares architecture-owned mask state after an execution policy has
    /// produced embeddings, including vocabulary-parallel embeddings.
    pub fn begin_embedded<S>(
        &mut self,
        hidden: B::Tensor,
        supplied_mask: Option<&B::Tensor>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor>,
    {
        let expected = state_layout(&self.args)?;
        self.begin_embedded_with_layout(hidden, supplied_mask, state, &expected, context)
    }

    /// Prepares architecture-owned mask state against an explicitly realized
    /// state layout, such as the rank-local KV geometry produced by tensor
    /// parallel planning.
    pub fn begin_embedded_with_layout<S>(
        &mut self,
        hidden: B::Tensor,
        supplied_mask: Option<&B::Tensor>,
        state: &mut S,
        expected: &StateLayout,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor>,
    {
        self.begin_embedded_with_layout_and_rotary(
            hidden,
            supplied_mask,
            None,
            state,
            expected,
            context,
        )
    }

    /// Starts one replicated pipeline partition from tokens or upstream hidden
    /// state using the partition's authoritative local state layout.
    fn prepare_partition<S>(
        &mut self,
        input: LayeredPartitionInput<'_, B::Tensor>,
        supplied_mask: Option<&B::Tensor>,
        state: &mut S,
        expected: &StateLayout,
        first_state_ordinal: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor>,
    {
        let hidden = match input {
            LayeredPartitionInput::Tokens(tokens) => {
                self.static_modules.embeddings.forward(tokens, context)?
            }
            LayeredPartitionInput::Hidden { hidden, .. } => hidden,
        };
        self.begin_embedded_with_layout_at(
            hidden,
            supplied_mask,
            None,
            state,
            expected,
            first_state_ordinal,
            context,
        )
    }

    /// Starts one tensor-parallel pipeline partition through the same neutral
    /// entry point, including vocabulary-parallel embedding on the input rank.
    fn prepare_partition_parallel<S>(
        &mut self,
        input: LayeredPartitionInput<'_, B::Tensor>,
        supplied_mask: Option<&B::Tensor>,
        state: &mut S,
        expected: &StateLayout,
        first_state_ordinal: usize,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>, Error>
    where
        B: eredu_nn::DistributedNeuralBackend,
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor>,
    {
        let hidden = match input {
            LayeredPartitionInput::Tokens(tokens) => B::vocabulary_parallel_lookup(
                &mut self.static_modules.embeddings,
                tokens,
                EmbeddingLookupPolicy::Strict,
                parallel,
                context,
            )?,
            LayeredPartitionInput::Hidden { hidden, .. } => hidden,
        };
        self.begin_embedded_with_layout_at(
            hidden,
            supplied_mask,
            None,
            state,
            expected,
            first_state_ordinal,
            context,
        )
    }

    /// Prepares a layered pass with caller-provided explicit rotary embeddings.
    pub fn begin_embedded_with_layout_and_rotary<S>(
        &mut self,
        hidden: B::Tensor,
        supplied_mask: Option<&B::Tensor>,
        rotary_embeddings: Option<(&B::Tensor, &B::Tensor)>,
        state: &mut S,
        expected: &StateLayout,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor>,
    {
        self.begin_embedded_with_layout_at(
            hidden,
            supplied_mask,
            rotary_embeddings,
            state,
            expected,
            0,
            context,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn begin_embedded_with_layout_at<S>(
        &mut self,
        hidden: B::Tensor,
        supplied_mask: Option<&B::Tensor>,
        rotary_embeddings: Option<(&B::Tensor, &B::Tensor)>,
        state: &mut S,
        expected: &StateLayout,
        first_state_ordinal: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor>,
    {
        if state.layout() != expected {
            return Err(Error::backend(format!(
                "decoder runtime state layout {:?} does not match architecture layout {expected:?}",
                state.layout()
            )));
        }
        let sequence = hidden.dim(1);
        let mask = if let Some(mask) = supplied_mask {
            Some(mask.clone())
        } else if sequence > 1 {
            let cache = state.layer(first_state_ordinal).map_err(Error::backend)?;
            // The shared mask is consumed by full-attention layers. Sliding
            // layers use their typed window-aware attention path, so deriving
            // this mask from layer zero's retention policy would incorrectly
            // impose that window on later full-attention layers.
            Some(B::causal_mask(sequence, cache.offset(), None, context)?)
        } else {
            None
        };
        Ok(LayeredForwardState {
            hidden,
            context: ForwardContext {
                mask,
                allow_sliding_prefill: supplied_mask.is_none(),
                rotary_embeddings: rotary_embeddings
                    .map(|(cosine, sine)| (cosine.clone(), sine.clone())),
            },
        })
    }

    /// Executes one replicated block using architecture-owned forward state.
    pub fn forward_block<S>(
        &mut self,
        index: usize,
        block: &mut TransformerBlock<B, P::FeedForward>,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut ForwardContext<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor>,
    {
        let cache = state.layer(index).map_err(Error::backend)?;
        block.forward(
            AttentionInput {
                hidden,
                mask: forward.mask.as_ref(),
                cache: Some(cache),
                allow_sliding_prefill: forward.allow_sliding_prefill,
                rotary_position: forward
                    .rotary_embeddings
                    .as_ref()
                    .map(|(cosine, sine)| RotaryPosition::Embeddings { cosine, sine }),
            },
            context,
        )
    }

    /// Executes one replicated block while delegating its feed-forward policy
    /// to a composition-supplied executor.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_block_with_feed_forward<S, H>(
        &mut self,
        index: usize,
        block: &mut TransformerBlock<B, P::FeedForward>,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut ForwardContext<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
        feed_forward: H,
    ) -> Result<B::Tensor, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor>,
        H: FnOnce(
            &mut P::FeedForward,
            &B::Tensor,
            &<B::Tensor as Tensor>::Context,
        ) -> Result<B::Tensor, Error>,
    {
        let cache = state.layer(index).map_err(Error::backend)?;
        block.forward_with_feed_forward(
            AttentionInput {
                hidden,
                mask: forward.mask.as_ref(),
                cache: Some(cache),
                allow_sliding_prefill: forward.allow_sliding_prefill,
                rotary_position: forward
                    .rotary_embeddings
                    .as_ref()
                    .map(|(cosine, sine)| RotaryPosition::Embeddings { cosine, sine }),
            },
            context,
            feed_forward,
        )
    }

    /// Executes one tensor-parallel block using the same architecture-owned
    /// mask and state semantics as replicated execution.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_block_parallel<S>(
        &mut self,
        index: usize,
        block: &mut TransformerBlock<B, P::FeedForward>,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut ForwardContext<B::Tensor>,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor>,
        P::FeedForward: TensorParallelFeedForwardOperator<B>,
    {
        let cache = state.layer(index).map_err(Error::backend)?;
        block.forward_tensor_parallel(
            AttentionInput {
                hidden,
                mask: forward.mask.as_ref(),
                cache: Some(cache),
                allow_sliding_prefill: forward.allow_sliding_prefill,
                rotary_position: forward
                    .rotary_embeddings
                    .as_ref()
                    .map(|(cosine, sine)| RotaryPosition::Embeddings { cosine, sine }),
            },
            parallel,
            context,
        )
    }

    /// Executes one tensor-parallel block while delegating its feed-forward
    /// policy to a composition-supplied executor.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_block_parallel_with_feed_forward<S, H>(
        &mut self,
        index: usize,
        block: &mut TransformerBlock<B, P::FeedForward>,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut ForwardContext<B::Tensor>,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        feed_forward: H,
    ) -> Result<B::Tensor, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor>,
        H: FnOnce(
            &mut P::FeedForward,
            &B::Tensor,
            &<B::Tensor as Tensor>::Context,
        ) -> Result<B::Tensor, Error>,
    {
        let cache = state.layer(index).map_err(Error::backend)?;
        block.forward_tensor_parallel_with_feed_forward(
            AttentionInput {
                hidden,
                mask: forward.mask.as_ref(),
                cache: Some(cache),
                allow_sliding_prefill: forward.allow_sliding_prefill,
                rotary_position: forward
                    .rotary_embeddings
                    .as_ref()
                    .map(|(cosine, sine)| RotaryPosition::Embeddings { cosine, sine }),
            },
            parallel,
            context,
            feed_forward,
        )
    }

    /// Applies final normalization and the tied or untied output projection.
    pub fn finish_hidden(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let hidden = self.static_modules.norm.forward(hidden, context)?;
        match &mut self.static_modules.lm_head {
            Some(head) => head.forward(&hidden, context),
            None => self.static_modules.embeddings.as_linear(&hidden, context),
        }
    }

    /// Applies rank-local normalization and vocabulary-parallel projection for
    /// an output-owning pipeline partition.
    pub fn finish_hidden_parallel(
        &mut self,
        hidden: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        B: eredu_nn::DistributedNeuralBackend,
    {
        let hidden = self.static_modules.norm.forward(hidden, context)?;
        match &mut self.static_modules.lm_head {
            Some(head) => B::vocabulary_parallel_project(head, &hidden, parallel, context),
            None => B::vocabulary_parallel_embedding_project(
                &mut self.static_modules.embeddings,
                &hidden,
                parallel,
                context,
            ),
        }
    }
}

impl<B, C, P> LayeredModel<B, C, P>
where
    B: NeuralBackend,
    C: Config,
    P: BlockFactory<B, C>,
{
    /// Describes every shared-decoder parameter group with explicit static or
    /// architecture-global unit ownership.
    fn parameter_description_impl(
        &self,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<ArchitectureParameterDescription, Error> {
        let graph =
            ExecutionGraph::chain([TEXT_DECODER_EXECUTION_GROUP]).map_err(Error::backend)?;
        let count = usize::try_from(self.args.num_hidden_layers()).map_err(Error::backend)?;
        let layout = ExecutionUnitLayout::new(&graph, [count]).map_err(Error::backend)?;
        let static_groups = static_parallel_parameter_groups::<B>(
            &self.static_modules.embeddings,
            &self.static_modules.norm,
            self.static_modules.lm_head.as_ref(),
            self.args.parameter_root(),
        )
        .map_err(Error::backend)?;
        let mut expected = static_groups.clone();
        let mut owned = Vec::new();
        for (index, group) in static_groups.into_iter().enumerate() {
            let role = match index {
                0 => "embedding",
                1 => "norm",
                _ => "output",
            };
            let owner = if index == 0 && self.args.tie_word_embeddings() {
                ParameterGroupOwner::static_any_of(["embedding", "output"])
            } else {
                ParameterGroupOwner::static_role(role)
            };
            owned.push(OwnedParameterGroupSpec::new(owner, group));
        }
        let group_id = layout.group_id(0).expect("decoder layout group").clone();
        for index in 0..count {
            let unit = self.construct_unit(index, context)?;
            let groups = P::parameter_groups(&unit, &self.args, index).map_err(Error::backend)?;
            expected.extend(groups.iter().cloned());
            owned.extend(groups.into_iter().map(|group| {
                OwnedParameterGroupSpec::new(
                    ParameterGroupOwner::execution_unit(group_id.clone(), index),
                    group,
                )
            }));
        }
        ArchitectureParameterDescription::new(&graph, &layout, expected, owned)
            .map_err(Error::backend)
    }
}

impl<B, C, P> eredu_runtime::ArchitectureParameters<B> for LayeredModel<B, C, P>
where
    B: NeuralBackend,
    C: Config,
    P: BlockFactory<B, C>,
{
    type DefinitionError = Error;

    fn state_layout(&self) -> Result<StateLayout, Self::DefinitionError> {
        self.state_layout_impl()
    }

    fn state_identity(
        &self,
        state: &eredu_runtime::PartitionState,
        topology: eredu_core::cache::PromptCacheTopology,
    ) -> Result<eredu_runtime::ModelStateIdentity, Self::DefinitionError> {
        state_identity(
            &self.args,
            state.layout(),
            state.global_layer_offset(),
            topology,
        )
    }

    fn parameter_description(
        &self,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<ArchitectureParameterDescription, Self::DefinitionError> {
        self.parameter_description_impl(context)
    }

    fn visit_static_parameters<V>(&self, visitor: &mut V) -> Result<(), V::Error>
    where
        V: eredu_runtime::StaticParameterVisitor<B>,
    {
        visitor.visit("embedding", &self.static_modules.embeddings)?;
        visitor.visit("norm", &self.static_modules.norm)?;
        if let Some(head) = &self.static_modules.lm_head {
            visitor.visit("output", head)?;
        }
        Ok(())
    }

    fn visit_static_parameters_mut<V>(&mut self, visitor: &mut V) -> Result<(), V::Error>
    where
        V: eredu_runtime::StaticParameterVisitorMut<B>,
    {
        visitor.visit_mut("embedding", &mut self.static_modules.embeddings)?;
        visitor.visit_mut("norm", &mut self.static_modules.norm)?;
        if let Some(head) = &mut self.static_modules.lm_head {
            visitor.visit_mut("output", head)?;
        }
        Ok(())
    }
}

impl<B, C, P, S> LayeredArchitecture<B, S> for LayeredModel<B, C, P>
where
    B: NeuralBackend,
    C: Config,
    P: BlockFactory<B, C>,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor>,
{
    type Input<'a> = LayeredInput<'a, B::Tensor>;
    type StaticModules = StaticModules<B>;
    type Unit = TransformerBlock<B, P::FeedForward>;
    type ForwardContext = ForwardContext<B::Tensor>;
    type RetainedContextValues<'a>
        = std::option::Iter<'a, B::Tensor>
    where
        B::Tensor: 'a;
    type Error = Error;

    fn group_transport(&self, _group: usize) -> eredu_runtime::ArchitectureGroupTransport {
        crate::transport::decoder()
    }

    fn primary_execution_group(&self) -> &str {
        TEXT_DECODER_EXECUTION_GROUP
    }

    fn state_partition_plan(
        &self,
        layout: &eredu_runtime::StateLayout,
    ) -> eredu_runtime::ArchitectureStatePartitionPlan {
        crate::transport::pipeline_state(0, layout)
    }

    fn execution_graph(&self) -> Result<eredu_runtime::ExecutionGraph, Self::Error> {
        eredu_runtime::ExecutionGraph::chain([TEXT_DECODER_EXECUTION_GROUP]).map_err(Error::backend)
    }

    fn group_unit_count(&self, group: usize) -> Result<usize, Self::Error> {
        if group != 0 {
            return Err(Error::backend(format!(
                "decoder execution group {group} is outside the text decoder"
            )));
        }
        usize::try_from(self.args.num_hidden_layers()).map_err(Error::backend)
    }

    fn unit_path(&self, group: usize, index: usize) -> Result<String, Self::Error> {
        if group != 0 {
            return Err(Error::backend(format!(
                "decoder execution group {group} is outside the text decoder"
            )));
        }
        let count = usize::try_from(self.args.num_hidden_layers()).map_err(Error::backend)?;
        if index >= count {
            return Err(Error::backend(format!(
                "decoder unit {index} is outside {count} decoder layers"
            )));
        }
        Ok(format!("{}.layers.{index}", self.args.parameter_root()))
    }

    fn static_modules(&self) -> &Self::StaticModules {
        &self.static_modules
    }

    fn static_modules_mut(&mut self) -> &mut Self::StaticModules {
        &mut self.static_modules
    }

    fn build_unit(
        &self,
        group: usize,
        index: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Unit, Self::Error> {
        if group != 0 {
            return Err(Error::backend(format!(
                "decoder execution group {group} is outside the text decoder"
            )));
        }
        self.construct_unit(index, context)
    }

    fn begin_forward<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error> {
        let hidden = self
            .static_modules
            .embeddings
            .forward(input.tokens, context)?;
        self.begin_embedded(hidden, input.mask, state, context)
    }

    fn begin_execution_group(
        &mut self,
        group: usize,
        initial: &B::Tensor,
        dependencies: &[&B::Tensor],
        _state: &mut S,
        _forward: &mut Self::ForwardContext,
        _context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        if group != 0 || !dependencies.is_empty() {
            return Err(Error::backend(format!(
                "text decoder group {group} received {} dependencies",
                dependencies.len()
            )));
        }
        Ok(initial.clone())
    }

    fn forward_unit(
        &mut self,
        _group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.forward_block(index, unit, hidden, state, forward, context)
    }

    fn finish_forward(
        &mut self,
        hidden: &B::Tensor,
        _state: &mut S,
        _forward: &Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.finish_hidden(hidden, context)
    }

    fn retained_context_values<'a>(
        &'a self,
        forward: &'a Self::ForwardContext,
        _group: usize,
        _index: usize,
    ) -> Self::RetainedContextValues<'a> {
        forward.mask.iter()
    }
}

impl<B, C, P, S> eredu_runtime::ReplicatedTextArchitecture<B, S> for LayeredModel<B, C, P>
where
    B: NeuralBackend,
    C: Config,
    P: BlockFactory<B, C>,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor>,
{
    fn text_input<'a>(tokens: &'a B::Tensor, mask: Option<&'a B::Tensor>) -> Self::Input<'a> {
        LayeredInput { tokens, mask }
    }
}

impl<B, C, P, S> ParallelLayeredArchitecture<B, S> for LayeredModel<B, C, P>
where
    B: NeuralBackend + eredu_nn::DistributedNeuralBackend,
    C: Config,
    P: BlockFactory<B, C>,
    P::FeedForward: TensorParallelFeedForwardOperator<B>,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor>,
{
    fn begin_forward_parallel<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error> {
        let expected = self
            .parallel_geometry
            .as_ref()
            .ok_or_else(|| Error::backend("decoder model was not built with local geometry"))?
            .state_layout()
            .clone();
        let hidden = B::vocabulary_parallel_lookup(
            &mut self.static_modules.embeddings,
            input.tokens,
            EmbeddingLookupPolicy::Strict,
            parallel,
            context,
        )?;
        self.begin_embedded_with_layout(hidden, input.mask, state, &expected, context)
    }

    fn forward_unit_parallel(
        &mut self,
        _group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.forward_block_parallel(index, unit, hidden, state, forward, parallel, context)
    }

    fn finish_forward_parallel(
        &mut self,
        hidden: &B::Tensor,
        _state: &mut S,
        _forward: &Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.finish_hidden_parallel(hidden, parallel, context)
    }
}

impl<B, C, P, S> PartitionedLayeredArchitecture<B, S> for LayeredModel<B, C, P>
where
    B: NeuralBackend + eredu_nn::DistributedNeuralBackend,
    C: Config,
    P: BlockFactory<B, C>,
    P::FeedForward: TensorParallelFeedForwardOperator<B>,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor>,
{
    type Boundary = eredu_runtime::NoAuxiliaryBoundarySchema;

    fn boundary_schema(&self) -> Result<Self::Boundary, Self::Error> {
        Ok(eredu_runtime::NoAuxiliaryBoundarySchema::new(
            self.args().hidden_size(),
        ))
    }

    fn begin_partition<'a>(
        &mut self,
        input: LayeredPartitionInput<'a, B::Tensor>,
        mask: Option<&B::Tensor>,
        state: &mut S,
        expected: &StateLayout,
        first_state_ordinal: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error> {
        LayeredModel::prepare_partition(
            self,
            input,
            mask,
            state,
            expected,
            first_state_ordinal,
            context,
        )
    }

    fn begin_partition_parallel<'a>(
        &mut self,
        input: LayeredPartitionInput<'a, B::Tensor>,
        mask: Option<&B::Tensor>,
        state: &mut S,
        expected: &StateLayout,
        first_state_ordinal: usize,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error> {
        LayeredModel::prepare_partition_parallel(
            self,
            input,
            mask,
            state,
            expected,
            first_state_ordinal,
            parallel,
            context,
        )
    }

    fn finish_partition(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &Self::ForwardContext,
        owns_output: bool,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<eredu_runtime::LayeredPartitionOutput<B::Tensor>, Self::Error> {
        if owns_output {
            let output = match parallel {
                Some(parallel) => {
                    self.finish_forward_parallel(hidden, state, forward, parallel, context)?
                }
                None => self.finish_forward(hidden, state, forward, context)?,
            };
            Ok(eredu_runtime::LayeredPartitionOutput::Final {
                output,
                retained: None,
            })
        } else {
            Ok(eredu_runtime::LayeredPartitionOutput::Boundary {
                hidden: hidden.clone(),
                auxiliary: eredu_runtime::NoAuxiliaryBoundary,
            })
        }
    }
}

impl<B, C, P, S> eredu_runtime::RoutedLayeredArchitecture<B, S> for LayeredModel<B, C, P>
where
    B: GroupedNeuralBackend,
    C: Config,
    P: BlockFactory<B, C>,
    P::FeedForward: RoutedFeedForwardOperator<B>,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor>,
{
    fn routed_observation_point(
        &self,
        group: usize,
        index: usize,
    ) -> Result<Option<eredu_runtime::RoutedObservationPoint>, Self::Error> {
        let unit_path = <Self as LayeredArchitecture<B, S>>::unit_path(self, group, index)?;
        Ok(self.args.routed_observation_point(&unit_path, index))
    }

    fn forward_unit_with_provider<R>(
        &mut self,
        _group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        pass: eredu_runtime::ExpertPass,
        provider: &mut R,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error>
    where
        R: eredu_runtime::RoutedExpertProvider<B>,
        R::Error: std::fmt::Display,
    {
        self.forward_block_with_feed_forward(
            index,
            unit,
            hidden,
            state,
            forward,
            context,
            |policy, normalized, context| {
                policy.forward_with_provider(index, normalized, pass, provider, context)
            },
        )
    }
}

impl<B, C, P, S> eredu_runtime::ParallelRoutedLayeredArchitecture<B, S> for LayeredModel<B, C, P>
where
    B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    C: Config,
    P: BlockFactory<B, C>,
    P::FeedForward: TensorParallelRoutedFeedForwardOperator<B>,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor>,
{
    fn forward_unit_parallel_with_provider<R>(
        &mut self,
        _group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        pass: eredu_runtime::ExpertPass,
        provider: &mut R,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error>
    where
        R: eredu_runtime::TensorParallelRoutedExpertProvider<B>,
        R::Error: std::fmt::Display,
    {
        self.forward_block_parallel_with_feed_forward(
            index,
            unit,
            hidden,
            state,
            forward,
            parallel,
            context,
            |policy, normalized, context| {
                policy.forward_parallel_with_provider(
                    index, normalized, pass, provider, parallel, context,
                )
            },
        )
    }
}
