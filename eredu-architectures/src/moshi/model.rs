//! Portable temporal-plus-depth Moshi-family execution graph.

use crate::decoder::{self, Config as DecoderConfig, MultiTableEmbedding, NamedEmbeddingSpec};
use eredu_core::LayerSchedule;
use eredu_nn::{
    AttentionCache, EmbeddingLookupPolicy, EmbeddingSpec, Error, LinearOperator, LinearSpec,
    NeuralBackend, NormalizationConstructionSpec, NormalizationOperator, ParameterSpec, Tensor,
    VocabularyParallelRange,
};
use eredu_runtime::{
    ArchitectureParameterDescription, ExecutionGraph, LayerRuntimeState, LayeredArchitecture,
    LayeredForwardState, LayeredTraversalPoint, ParallelLayeredArchitecture,
    ResettableRuntimeState, SamplingBackend, SequentialDecisionBoundary, SequentialDecisionError,
    StateLayout, StateSegmentId, StateSegmentLifetime, StateSegmentSpec, TokenDomain,
};

use super::{block, depth::DepthSlice, MoshiConfig};

/// Persistent temporal-cache segment identity.
pub const TEMPORAL_STATE_SEGMENT: &str = "temporal";
/// Frame-local reusable depth-cache segment identity.
pub const DEPTH_STATE_SEGMENT: &str = "depth";

/// Stable architecture-owned activation boundary used by parity fixtures.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum ObservationPoint {
    /// Sum of text and audio embeddings before the temporal transformer.
    TemporalInput,
    /// Output of one temporal decoder block.
    TemporalLayer {
        /// Zero-based temporal decoder layer index.
        layer: usize,
    },
    /// Complete text-vocabulary logits after temporal normalization.
    TextLogits,
    /// Complete audio-vocabulary logits produced by one ordered depth slice.
    DepthSliceLogits {
        /// Zero-based ordered depth-slice index.
        slice: usize,
    },
}

impl ObservationPoint {
    /// Returns the canonical parameter-aligned observation path.
    pub fn path(self) -> String {
        match self {
            Self::TemporalInput => "temporal.input".into(),
            Self::TemporalLayer { layer } => format!("transformer.layers.{layer}.output"),
            Self::TextLogits => "text_linear.logits".into(),
            Self::DepthSliceLogits { slice } => {
                format!("depformer.slices.{slice}.logits")
            }
        }
    }
}

/// Returns every frozen observation point in production traversal order.
pub fn observation_points(config: &MoshiConfig) -> Vec<ObservationPoint> {
    let temporal_layers = usize::try_from(config.temporal().num_hidden_layers())
        .expect("validated Moshi temporal layer count fits usize");
    std::iter::once(ObservationPoint::TemporalInput)
        .chain((0..temporal_layers).map(|layer| ObservationPoint::TemporalLayer { layer }))
        .chain(std::iter::once(ObservationPoint::TextLogits))
        .chain(
            (0..config.frame_schedule().depth_audio_codebooks())
                .map(|slice| ObservationPoint::DepthSliceLogits { slice }),
        )
        .collect()
}

/// Pinned parameters that are not residency execution units.
#[derive(Debug, Clone, eredu_nn::Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct StaticModules<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> {
    /// Text followed by temporal audio-codebook embedding tables.
    pub embeddings: MultiTableEmbedding<B>,
    /// Final temporal normalization.
    pub output_norm: B::Normalization,
    /// Temporal text-vocabulary projection.
    pub text_output: B::Linear,
}

impl<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> StaticModules<B> {
    fn embedding_specs(config: &MoshiConfig) -> Result<Vec<NamedEmbeddingSpec>, Error> {
        let text_input_vocabulary = config
            .text_vocabulary_size()
            .checked_add(1)
            .ok_or_else(|| Error::backend("Moshi text input vocabulary overflowed"))?;
        let audio_input_vocabulary = config
            .audio_vocabulary_size()
            .checked_add(1)
            .ok_or_else(|| Error::backend("Moshi audio input vocabulary overflowed"))?;
        let mut embeddings = Vec::with_capacity(
            config
                .frame_schedule()
                .total_audio_codebooks()
                .saturating_add(1),
        );
        embeddings.push(NamedEmbeddingSpec {
            name: "text".into(),
            embedding: EmbeddingSpec {
                vocabulary: text_input_vocabulary,
                dimensions: config.temporal().hidden_size(),
                weight: ParameterSpec::trainable("text_emb.weight").map_err(Error::backend)?,
                format: crate::linear_format::standard_linear_format(
                    "text_emb.weight",
                    config.native_quantization().into(),
                )?,
            },
            lookup: EmbeddingLookupPolicy::ZeroSentinel(-1),
        });
        for codebook in 0..config.frame_schedule().total_audio_codebooks() {
            let name = format!("audio_embs.{codebook}.weight");
            embeddings.push(NamedEmbeddingSpec {
                name: format!("audio_{codebook}"),
                embedding: EmbeddingSpec {
                    vocabulary: audio_input_vocabulary,
                    dimensions: config.temporal().hidden_size(),
                    weight: ParameterSpec::trainable(&name).map_err(Error::backend)?,
                    format: crate::linear_format::standard_linear_format(
                        &name,
                        config.native_quantization().into(),
                    )?,
                },
                lookup: EmbeddingLookupPolicy::ZeroSentinel(-1),
            });
        }
        Ok(embeddings)
    }

    fn new(config: &MoshiConfig, context: &<B::Tensor as Tensor>::Context) -> Result<Self, Error> {
        let embeddings = MultiTableEmbedding::new(Self::embedding_specs(config)?, context)?;
        let output_norm = B::normalization(
            NormalizationConstructionSpec::learned(
                config.temporal().hidden_size(),
                config.temporal().rms_norm_epsilon(),
                ParameterSpec::trainable("out_norm.weight").map_err(Error::backend)?,
            ),
            context,
        )?;
        let text_output = B::linear(
            LinearSpec {
                input: config.temporal().hidden_size(),
                output: config.text_vocabulary_size(),
                weight: ParameterSpec::trainable("text_linear.weight").map_err(Error::backend)?,
                bias: None,
                format: crate::linear_format::standard_linear_format(
                    "text_linear.weight",
                    config.native_quantization().into(),
                )?,
            },
            context,
        )?;
        Ok(Self {
            embeddings,
            output_norm,
            text_output,
        })
    }

    fn new_parallel(
        config: &MoshiConfig,
        geometry: &super::LocalGeometry,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let specs = Self::embedding_specs(config)?;
        let ranges = std::iter::once("text_emb.weight".to_owned())
            .chain(
                (0..config.frame_schedule().total_audio_codebooks())
                    .map(|index| format!("audio_embs.{index}.weight")),
            )
            .map(|target| {
                let local = geometry
                    .vocabulary_range(&target)
                    .cloned()
                    .ok_or_else(|| Error::backend(format!("missing local range for {target}")))?;
                let global_vocabulary = specs
                    .iter()
                    .find(|spec| spec.embedding.weight.id.as_str() == target)
                    .map(|spec| spec.embedding.vocabulary as usize)
                    .ok_or_else(|| {
                        Error::backend(format!("missing embedding spec for {target}"))
                    })?;
                Ok(VocabularyParallelRange {
                    global_vocabulary,
                    local,
                })
            })
            .collect::<Result<Vec<_>, Error>>()?;
        let embeddings = MultiTableEmbedding::new_vocabulary_parallel(specs, ranges, context)?;
        let output_norm = B::normalization(
            NormalizationConstructionSpec::learned(
                config.temporal().hidden_size(),
                config.temporal().rms_norm_epsilon(),
                ParameterSpec::trainable("out_norm.weight").map_err(Error::backend)?,
            ),
            context,
        )?;
        let local = geometry
            .vocabulary_range("text_linear.weight")
            .cloned()
            .ok_or_else(|| Error::backend("missing local text output range"))?;
        let text_output = B::vocabulary_parallel_linear(
            LinearSpec {
                input: config.temporal().hidden_size(),
                output: config.text_vocabulary_size(),
                weight: ParameterSpec::trainable("text_linear.weight").map_err(Error::backend)?,
                bias: None,
                format: crate::linear_format::standard_linear_format(
                    "text_linear.weight",
                    config.native_quantization().into(),
                )?,
            },
            VocabularyParallelRange {
                global_vocabulary: config.text_vocabulary_size() as usize,
                local,
            },
            context,
        )?;
        Ok(Self {
            embeddings,
            output_norm,
            text_output,
        })
    }
}

/// One resident or bounded execution unit in the two-group graph.
#[derive(Debug, Clone, eredu_nn::Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub enum Unit<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> {
    /// One temporal shared decoder block.
    Temporal(block::Block<B>),
    /// One complete ordered depth-codebook slice.
    Depth(DepthSlice<B>),
}

/// Prepared text and per-codebook temporal input tokens.
pub struct Input<'a, T> {
    /// Text tokens shaped like each audio-codebook token tensor.
    pub text: &'a T,
    /// One token tensor per temporal audio codebook.
    pub audio: &'a [&'a T],
    /// Optional caller-supplied temporal attention mask.
    pub mask: Option<&'a T>,
}

/// Architecture values retained through one temporal-plus-depth pass.
pub struct ForwardContext<T> {
    temporal_mask: Option<T>,
    allow_sliding_prefill: bool,
    temporal_output: Option<T>,
    text_logits: Option<T>,
    previous_depth_token: Option<T>,
    return_demand: eredu_core::OutputDemand,
    text_demand: eredu_core::OutputDemand,
    depth_demand: eredu_core::OutputDemand,
}

impl<T> ForwardContext<T> {
    /// Borrows the shared temporal attention mask.
    pub const fn temporal_mask(&self) -> Option<&T> {
        self.temporal_mask.as_ref()
    }

    /// Whether temporal sliding attention may use its optimized prefill path.
    pub const fn allow_sliding_prefill(&self) -> bool {
        self.allow_sliding_prefill
    }

    /// Borrows normalized temporal hidden state after the temporal group.
    pub const fn temporal_output(&self) -> Option<&T> {
        self.temporal_output.as_ref()
    }

    /// Borrows text logits after the temporal group.
    pub const fn text_logits(&self) -> Option<&T> {
        self.text_logits.as_ref()
    }

    /// Whether the current depth unit produced actual vocabulary scores.
    pub const fn has_depth_logits(&self) -> bool {
        !matches!(self.depth_demand, eredu_core::OutputDemand::StateOnly)
    }

    /// Borrows the token feeding the next depth slice.
    pub const fn previous_depth_token(&self) -> Option<&T> {
        self.previous_depth_token.as_ref()
    }

    /// Records normalized temporal state and complete text logits.
    pub fn set_temporal_outputs(&mut self, temporal: T, text_logits: T) {
        self.temporal_output = Some(temporal);
        self.text_logits = Some(text_logits);
    }

    /// Replaces complete text logits at the declared decision boundary.
    ///
    /// Realtime observation uses this before the sequential decision driver,
    /// so the intervened value is the one actually consumed by sampling and
    /// retained through completion.
    pub fn replace_text_logits(&mut self, text_logits: T) {
        self.text_logits = Some(text_logits);
    }
}

/// Builds the exact persistent-temporal plus frame-local-depth state layout.
pub fn state_layout(config: &MoshiConfig) -> Result<StateLayout, Error> {
    state_layout_destination(config, &crate::state_geometry::Ordinary(Error::backend_message))
}

fn state_layout_destination<D: crate::state_geometry::Destination>(
    config: &MoshiConfig,
    destination: &D,
) -> Result<StateLayout, D::Error> {
    destination.controls::<(&MoshiConfig, StateLayout,
        [LayerSchedule<eredu_core::cache::LayerCachePolicy>; 2],
        Vec<eredu_core::cache::LayerCachePolicy>, Vec<StateSegmentSpec>, usize, usize)>()?;
    let temporal = decoder::cache_layout_destination(config.temporal(),
        std::iter::repeat_n(config.temporal().num_key_value_heads(),
            config.temporal().attention_schedule().len()), destination)?;
    let depth = decoder::cache_layout_destination(config.depth_template(),
        std::iter::repeat_n(config.depth_template().num_key_value_heads(),
            config.depth_template().attention_schedule().len()), destination)?;
    let temporal_count = temporal.len();
    let total = temporal_count.checked_add(depth.len())
        .ok_or_else(|| destination.error(format_args!("Moshi state layer count overflowed")))?;
    // Both decoder producers emit key/value policies, whose clone owns no nested allocation.
    let mut policies = destination.vector(total)?;
    policies.extend(temporal.iter().cloned());
    policies.extend(depth.iter().cloned());
    let schedule = destination.schedule(total, policies)?;
    let mut segments = destination.vector(2)?;
    segments.push(destination.segment(TEMPORAL_STATE_SEGMENT, 0..temporal_count,
        StateSegmentLifetime::Persistent, 0)?);
    segments.push(destination.segment(DEPTH_STATE_SEGMENT, temporal_count..total,
        StateSegmentLifetime::FrameLocal, 0)?);
    destination.segmented(schedule, segments)
}

/// One portable Moshi-family model shared by resident and bounded runtimes.
pub struct LayeredModel<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> {
    source: MoshiRealtimeModelSource,
    static_modules: StaticModules<B>,
}

#[derive(Debug)]
struct MoshiRealtimeModelDefinition {
    config: MoshiConfig,
    parallel_geometry: Option<super::LocalGeometry>,
    parallel_execution_identity: Option<String>,
    graph: ExecutionGraph,
}

/// Exact immutable architecture selection shared by the retained executable and
/// its metadata equations. This contains no backend tensor, device or authority.
#[derive(Debug,Clone)]
pub struct MoshiRealtimeModelSource(std::sync::Arc<MoshiRealtimeModelDefinition>);
impl MoshiRealtimeModelSource {
    /// Matches the retained construction owner, not equal configuration values.
    pub fn same_source(&self,other:&Self)->bool {std::sync::Arc::ptr_eq(&self.0,&other.0)}

    /// Whether this actual source requires a partition traversal and collectives.
    pub fn is_partitioned(&self)->bool {self.0.parallel_geometry.is_some()}

    /// Borrows the dependency graph retained by the same ordinary constructor.
    pub fn execution_graph(&self) -> &ExecutionGraph {
        &self.0.graph
    }

    /// Builds unloaded metadata modules from this exact selected architecture.
    /// Actual static/unit bindings and projected state remain separate inputs.
    pub fn workspace_model(&self,context:&eredu_nn::workspace::WorkspaceContext)
        ->Result<LayeredModel<eredu_nn::workspace::WorkspaceBackend>,Error> {
        context.charge_metadata(std::mem::size_of::<(Self,&Self,&eredu_nn::workspace::WorkspaceContext,
            LayeredModel<eredu_nn::workspace::WorkspaceBackend>,
            Result<LayeredModel<eredu_nn::workspace::WorkspaceBackend>,Error>)>())?;
        LayeredModel::from_source(self.clone(),context)
    }
}

impl<B: NeuralBackend + eredu_nn::DistributedNeuralBackend>
    eredu_runtime::RealtimeArchitectureIdentity for LayeredModel<B>
{
    fn realtime_construction_identity(
        &self,
    ) -> Result<eredu_runtime::RealtimeArchitectureConstructionIdentity, String> {
        let fingerprint = self.source.0.config.architecture_fingerprint();
        let identity = |value: String| {
            eredu_runtime::RealtimeIdentity::new(value).map_err(|error| error.to_string())
        };
        Ok(
            eredu_runtime::RealtimeArchitectureConstructionIdentity::new(
                identity(format!("moshi.realtime:{fingerprint}"))?,
                identity(format!("moshi.schedule:{fingerprint}"))?,
                identity(match &self.source.0.parallel_execution_identity {
                    Some(execution) => format!("moshi.state:{execution}"),
                    None if self.source.0.parallel_geometry.is_none() => {
                        format!("moshi.state:{fingerprint};replicated")
                    }
                    None => {
                        return Err(
                            "Moshi tensor-parallel architecture has no selected execution identity"
                                .into(),
                        );
                    }
                })?,
            ),
        )
    }
}

impl<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> eredu_runtime::ArchitectureParameters<B>
    for LayeredModel<B>
{
    type DefinitionError = Error;

    fn state_layout(&self, context: Option<&eredu_nn::workspace::WorkspaceContext>)
        -> Result<StateLayout, Self::DefinitionError> {
        self.state_layout_impl(context)
    }

    fn state_identity(
        &self,
        state: &eredu_runtime::PartitionState,
        topology: eredu_core::cache::PromptCacheTopology,
        context: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<eredu_runtime::ModelStateIdentity, Self::DefinitionError> {
        let metadata = decoder::identity::Metadata::new(context);
        metadata.controls::<(&Self, &eredu_runtime::PartitionState,
            eredu_core::cache::PromptCacheTopology, eredu_runtime::ModelStateIdentity,
            usize, usize)>()?;
        topology.validate_with_diagnostic(|message| metadata.prompt_error(message))?;
        // Configuration and selected local geometry were validated at source birth.
        // Identity needs their exact cardinality, not another allocated state layout.
        let layer_count = match &self.source.0.parallel_geometry {
            Some(geometry) => geometry.state_layout().len(),
            None => self.source.0.config.temporal().attention_schedule().len()
                .checked_add(self.source.0.config.depth_template().attention_schedule().len())
                .ok_or_else(|| metadata.error(format_args!("Moshi state layer count overflowed")))?,
        };
        let global_layer_end = state.global_layer_offset().checked_add(state.layout().len())
            .ok_or_else(|| metadata.error(format_args!("Moshi owned state range overflowed")))?;
        if global_layer_end > layer_count {
            return Err(metadata.error(format_args!(
                "Moshi owns state layers {}..{global_layer_end}, outside {layer_count} layers",
                state.global_layer_offset())));
        }
        eredu_runtime::ModelStateIdentity::new_with_diagnostic(
            metadata.text(self.source.0.config.family())?,
            metadata.text(self.source.0.config.effective_model_type().as_str())?,
            metadata.text(self.source.0.config.architecture_fingerprint())?,
            layer_count, state.global_layer_offset(), 0, topology,
            |message| metadata.prompt_error(message))
    }

    fn parameter_description(
        &self,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<std::borrow::Cow<'_, ArchitectureParameterDescription>, Self::DefinitionError> {
        let metadata = B::construction_metadata(context);
        decoder::identity::Metadata::new(metadata).controls::<(&Self,
            &<B::Tensor as Tensor>::Context,
            std::borrow::Cow<'_, ArchitectureParameterDescription>)>()?;
        super::parallel::declaration::description(&self.source.0.config, metadata)
            .map(std::borrow::Cow::Owned)
    }

    fn static_parameter_recipes(
        &self,
        source: &dyn eredu_checkpoint::store::CheckpointSource,
    ) -> Result<
        std::collections::BTreeMap<String, eredu_checkpoint::recipe::DerivedWeightRecipe>,
        String,
    > {
        let recipes = super::checkpoint::canonical_recipes(&self.source.0.config, source)?.into_outputs();
        crate::static_parameters::module_recipes(&self.static_modules, recipes)
    }

    fn retained_static_value_slot_bound(&self) -> Option<usize> {
        eredu_nn::Parameterized::retained_value_slot_bound(&self.static_modules)
    }

    fn visit_retained_static_values(&self, visitor: &mut dyn FnMut(&B::Tensor)) -> bool {
        eredu_nn::Parameterized::visit_retained_values(&self.static_modules, visitor)
    }

    fn visit_static_parameters<V>(&self, visitor: &mut V) -> Result<(), V::Error>
    where
        V: eredu_runtime::StaticParameterVisitor<B>,
    {
        visitor.visit("embedding", &self.static_modules.embeddings)?;
        visitor.visit("norm", &self.static_modules.output_norm)?;
        visitor.visit("output", &self.static_modules.text_output)
    }

    fn visit_static_parameters_mut<V>(&mut self, visitor: &mut V) -> Result<(), V::Error>
    where
        V: eredu_runtime::StaticParameterVisitorMut<B>,
    {
        visitor.visit_mut("embedding", &mut self.static_modules.embeddings)?;
        visitor.visit_mut("norm", &mut self.static_modules.output_norm)?;
        visitor.visit_mut("output", &mut self.static_modules.text_output)
    }
}

impl<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> LayeredModel<B> {
    /// Builds unloaded pinned modules from one normalized configuration.
    pub fn new(config:MoshiConfig,context:&<B::Tensor as Tensor>::Context)->Result<Self,Error> {
        Self::new_source(config,None,None,context)
    }

    /// Builds the same lifecycle with planner-derived rank-local modules.
    pub fn new_parallel(config:MoshiConfig,geometry:super::LocalGeometry,
        context:&<B::Tensor as Tensor>::Context)->Result<Self,Error> {
        Self::new_source(config,Some(geometry),None,context)
    }

    /// Builds rank-local modules paired with the actual selected execution ID.
    pub fn new_selected_parallel(config:MoshiConfig,geometry:super::LocalGeometry,
        execution_identity:String,context:&<B::Tensor as Tensor>::Context)->Result<Self,Error> {
        Self::new_source(config,Some(geometry),Some(execution_identity),context)
    }

    fn new_source(config:MoshiConfig,parallel_geometry:Option<super::LocalGeometry>,
        parallel_execution_identity:Option<String>,context:&<B::Tensor as Tensor>::Context)
        ->Result<Self,Error> {
        config.temporal().validate_config()?;
        config.depth_template().validate_config()?;
        if let Some(geometry)=&parallel_geometry {geometry.validate_for(&config).map_err(Error::backend)?;}
        let graph=match B::construction_metadata(context) {
            Some(metadata)=>{
                let mut groups=metadata.metadata_vec(2)?;
                let first=metadata.metadata_string(format_args!("temporal_transformer"))?;
                groups.push(eredu_runtime::ExecutionGroupSpec::root(first));
                let mut dependencies=metadata.metadata_vec(1)?;
                dependencies.push(metadata.metadata_string(format_args!("temporal_transformer"))?);
                groups.push(eredu_runtime::ExecutionGroupSpec::from_parts(
                    metadata.metadata_string(format_args!("depth_codebook_slices"))?,dependencies));
                ExecutionGraph::new_with_metadata(groups,"depth_codebook_slices",metadata)?
            }
            None=>ExecutionGraph::chain(["temporal_transformer","depth_codebook_slices"])
                .map_err(Error::backend)?,
        };
        let definition=MoshiRealtimeModelDefinition{config,parallel_geometry,parallel_execution_identity,graph};
        let source=MoshiRealtimeModelSource(match B::construction_metadata(context) {
            Some(metadata)=>metadata.metadata_arc(definition)?,
            None=>std::sync::Arc::new(definition),
        });
        Self::from_source(source,context)
    }

    fn from_source(source:MoshiRealtimeModelSource,context:&<B::Tensor as Tensor>::Context)
        ->Result<Self,Error> {
        let config=&source.0.config;
        let static_modules=match &source.0.parallel_geometry {
            Some(geometry)=>StaticModules::new_parallel(config,geometry,context)?,
            None=>StaticModules::new(config,context)?,
        };
        Ok(Self{source,static_modules})
    }

    /// Shares the actual immutable source without cloning its configuration,
    /// local geometry, state layout or execution identity.
    pub fn realtime_workspace_source(&self)->MoshiRealtimeModelSource {self.source.clone()}

    /// Borrows normalized architecture policy.
    pub fn config(&self) -> &MoshiConfig {
        &self.source.0.config
    }

    /// Borrows pinned model parameters.
    pub const fn static_modules(&self) -> &StaticModules<B> {
        &self.static_modules
    }

    /// Mutably borrows pinned model parameters.
    pub fn static_modules_mut(&mut self) -> &mut StaticModules<B> {
        &mut self.static_modules
    }

    /// Ordered text plus audio prediction count.
    pub fn decision_count(&self) -> usize {
        self.source.0.config
            .frame_schedule()
            .depth_audio_codebooks()
            .saturating_add(1)
    }

    /// State layout for this model's replicated or rank-local construction.
    fn state_layout_impl(&self, context: Option<&eredu_nn::workspace::WorkspaceContext>)
        -> Result<StateLayout, Error> {
        let metadata = decoder::identity::Metadata::new(context);
        metadata.controls::<(&Self, StateLayout)>()?;
        match (&self.source.0.parallel_geometry, metadata.context()) {
            (Some(geometry), Some(context)) => geometry.state_layout().clone_workspace(context),
            (Some(geometry), None) => Ok(geometry.state_layout().clone()),
            (None, Some(context)) => state_layout_destination(&self.source.0.config,
                &crate::state_geometry::Counted::new(context, Error::backend_message)),
            (None, None) => state_layout(&self.source.0.config),
        }
    }

    fn build_unit_impl(
        &self,
        group: usize,
        index: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Unit<B>, Error> {
        let count = match group {
            0 => usize::try_from(self.source.0.config.temporal().num_hidden_layers())
                .map_err(Error::backend)?,
            1 => self.source.0.config.frame_schedule().depth_audio_codebooks(),
            _ => {
                return Err(Error::backend(format!(
                    "Moshi execution group {group} is outside 0..2"
                )));
            }
        };
        if index >= count {
            return Err(Error::backend(format!(
                "Moshi unit {index} is outside execution group {group}"
            )));
        }
        if let Some(geometry) = &self.source.0.parallel_geometry {
            return geometry.build_unit(&self.source.0.config, group, index, context);
        }
        match group {
            0 => Ok(Unit::Temporal(block::build(
                self.source.0.config.temporal(),
                index,
                context,
            )?)),
            1 => Ok(Unit::Depth(DepthSlice::new(&self.source.0.config, index, context)?)),
            _ => unreachable!("group was validated"),
        }
    }

    /// Starts a pass from a caller-produced embedding against an explicit
    /// state layout, preserving the canonical depth-reset and mask lifecycle.
    pub fn begin_embedded_with_layout<S>(
        &mut self,
        hidden: B::Tensor,
        supplied_mask: Option<&B::Tensor>,
        state: &mut S,
        expected: &StateLayout,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>, Error>
    where
        S: LayerRuntimeState<B> + ResettableRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor>,
    {
        let metadata = crate::decoder::ModuleMetadata::new::<B>(context);
        metadata.controls::<(&Self, B::Tensor, Option<&B::Tensor>, &mut S,
            &StateLayout, &<B::Tensor as Tensor>::Context, StateSegmentId,
            Option<B::Tensor>, LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>)>()?;
        if state.layout() != expected {
            return Err(metadata.error(format_args!("Moshi runtime state layout mismatch")));
        }
        let depth = StateSegmentId::new(metadata.text(format_args!("{DEPTH_STATE_SEGMENT}"))?)
            .map_err(|cause| metadata.source(cause))?;
        state.reset_segment(&depth).map_err(|cause| metadata.source(cause))?;
        let temporal_mask = if let Some(mask) = supplied_mask {
            Some(mask.clone())
        } else if hidden.dim(1) > 1 {
            let offset = state.layer(0).map_err(|cause| metadata.source(cause))?.offset();
            Some(B::causal_mask(hidden.dim(1), offset, None, context)?)
        } else {
            None
        };
        Ok(LayeredForwardState {
            hidden,
            context: ForwardContext {
                temporal_mask,
                allow_sliding_prefill: supplied_mask.is_none(),
                temporal_output: None,
                text_logits: None,
                previous_depth_token: None,
                return_demand: eredu_core::OutputDemand::Sequence,
                text_demand: eredu_core::OutputDemand::Sequence,
                depth_demand: eredu_core::OutputDemand::Sequence,
            },
        })
    }
}

fn group_transport(group: usize) -> eredu_runtime::ArchitectureGroupTransport {
    match group {
        0 => crate::transport::decoder(),
        _ => crate::transport::prediction(),
    }
}

impl<B, S> LayeredArchitecture<B, S> for LayeredModel<B>
where
    B: NeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B> + ResettableRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor>,
{
    type Input<'a> = Input<'a, B::Tensor>;

    fn inference_input_shape(input: &Self::Input<'_>) -> Result<Option<[u64; 2]>, Self::Error> {
        crate::prefill::token_shape(input.text).map(Some)
    }

    type StaticModules = StaticModules<B>;
    type Unit = Unit<B>;
    type ForwardContext = ForwardContext<B::Tensor>;
    type RetainedContextValues<'a>
        = std::iter::Flatten<std::array::IntoIter<Option<&'a B::Tensor>, 3>>
    where
        B::Tensor: 'a;
    type Error = Error;

    fn group_transport(&self, group: usize) -> eredu_runtime::ArchitectureGroupTransport {
        group_transport(group)
    }

    fn primary_execution_group(&self) -> &str {
        "temporal_transformer"
    }

    fn prediction_execution_groups(&self) -> Vec<String> {
        vec!["depth_codebook_slices".to_owned()]
    }

    fn state_partition_plan(
        &self,
        layout: &eredu_runtime::StateLayout,
    ) -> eredu_runtime::ArchitectureStatePartitionPlan {
        let temporal_layers = layout
            .segments()
            .first()
            .map(|segment| segment.layers().end)
            .unwrap_or(0);
        crate::transport::pipeline_with_output_state(0, temporal_layers, layout)
    }

    fn execution_graph(&self) -> Result<eredu_runtime::ArchitectureExecutionGraph<'_>, Self::Error> {
        Ok(eredu_runtime::ArchitectureExecutionGraph::borrowed(
            &self.source.0.graph,
        ))
    }

    fn group_unit_count(&self, group: usize, metadata_context: Option<&eredu_nn::workspace::WorkspaceContext>) -> Result<usize, Self::Error> {
        let metadata = crate::decoder::ModuleMetadata::destination(metadata_context);
        metadata.controls::<(&Self, usize, Option<&eredu_nn::workspace::WorkspaceContext>)>()?;

        match group {
            0 => {
                usize::try_from(self.source.0.config.temporal().num_hidden_layers()).map_err(|cause| metadata.source(cause))
            }
            1 => Ok(self.source.0.config.frame_schedule().depth_audio_codebooks()),
            _ => Err(metadata.error(format_args!(
                "Moshi execution group {group} is outside 0..2"
            ))),
        }
    }

    fn unit_path(&self, group: usize, index: usize, metadata_context: Option<&eredu_nn::workspace::WorkspaceContext>) -> Result<String, Self::Error> {
        let metadata = crate::decoder::ModuleMetadata::destination(metadata_context);
        metadata.controls::<(&Self, usize, usize, Option<&eredu_nn::workspace::WorkspaceContext>)>()?;

        let count = match group {
            0 => usize::try_from(self.source.0.config.temporal().num_hidden_layers())
                .map_err(|cause| metadata.source(cause))?,
            1 => self.source.0.config.frame_schedule().depth_audio_codebooks(),
            _ => {
                return Err(metadata.error(format_args!(
                    "Moshi execution group {group} is outside 0..2"
                )));
            }
        };
        if index >= count {
            return Err(metadata.error(format_args!(
                "Moshi unit {index} is outside execution group {group}"
            )));
        }
        match group {
            0 => metadata.text(format_args!("transformer.layers.{index}")),
            1 => metadata.text(format_args!("depformer.slices.{index}")),
            _ => unreachable!("group was validated"),
        }
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
        self.build_unit_impl(group, index, context)
    }

    fn begin_forward<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error> {
        let metadata = crate::decoder::ModuleMetadata::new::<B>(context);
        metadata.controls::<(&mut Self, Self::Input<'_>, &mut S,
            &<B::Tensor as Tensor>::Context, Vec<&B::Tensor>, B::Tensor, StateLayout,
            LayeredForwardState<B::Tensor, Self::ForwardContext>)>()?;
        if input.audio.len() != self.source.0.config.frame_schedule().total_audio_codebooks() {
            return Err(metadata.error(format_args!(
                "Moshi temporal input has {} audio codebooks, expected {}",
                input.audio.len(), self.source.0.config.frame_schedule().total_audio_codebooks())));
        }
        let count = input.audio.len().checked_add(1)
            .ok_or_else(|| metadata.error(format_args!("Moshi token input count overflowed")))?;
        let mut tokens = metadata.vector(count)?;
        tokens.push(input.text);
        tokens.extend(input.audio.iter().copied());
        let hidden = self.static_modules.embeddings.forward(&tokens, context)?;
        let expected = match B::construction_metadata(context).filter(|context| context.uses_checked_metadata()) {
            Some(context) => state_layout_destination(&self.source.0.config,
                &crate::state_geometry::Counted::new(context, Error::backend_message))?,
            None => state_layout(&self.source.0.config)?,
        };
        self.begin_embedded_with_layout(hidden, input.mask, state, &expected, context)
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
        match (group, dependencies) {
            (0, []) => Ok(initial.clone()),
            (1, [temporal]) => Ok((*temporal).clone()),
            _ => Err(Error::backend(format!(
                "Moshi group {group} received {} dependencies",
                dependencies.len()
            ))),
        }
    }

    fn state_ordinal(&self, group: usize, index: usize, ordinal: usize) -> usize {
        if group == 1 {
            self.source.0.config.temporal().num_hidden_layers() as usize
        } else {
            debug_assert_eq!(ordinal, index);
            ordinal
        }
    }

    fn retained_state_ordinals(
        &self,
        group: usize,
        index: usize,
        ordinal: usize,
    ) -> std::ops::Range<usize> {
        if group == 1 {
            let start = self.source.0.config.temporal().num_hidden_layers() as usize;
            start..start + self.source.0.config.depth_template().num_hidden_layers() as usize
        } else {
            debug_assert_eq!(ordinal, index);
            ordinal..ordinal + 1
        }
    }

    fn forward_unit(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        match (group, unit) {
            (0, Unit::Temporal(block)) => block::forward(
                block,
                index,
                hidden,
                forward.temporal_mask.as_ref(),
                forward.allow_sliding_prefill,
                state,
                context,
            ),
            (1, Unit::Depth(slice)) => {
                let temporal = forward
                    .temporal_output
                    .as_ref()
                    .ok_or_else(|| Error::backend("Moshi temporal output is unavailable"))?;
                let previous = forward.previous_depth_token.as_ref().ok_or_else(|| {
                    Error::backend("Moshi depth slice requires an accepted prior decision")
                })?;
                let metadata=crate::decoder::ModuleMetadata::new::<B>(context);
                let transformer=self.source.0.config.depth_template_for(index,
                    |message|metadata.error(message))?;
                slice.forward_with_readout(
                    transformer,
                    temporal,
                    previous,
                    self.source.0.config.temporal().num_hidden_layers() as usize,
                    state,
                    forward.depth_demand,
                    context,
                )
            }
            _ => Err(Error::backend(
                "Moshi execution unit does not match its group",
            )),
        }
    }

    fn complete_execution_group(
        &mut self,
        group: usize,
        hidden: &B::Tensor,
        _state: &mut S,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        if group == 0 {
            let temporal = self.static_modules.output_norm.forward(hidden, context)?;
            let logits = crate::readout::select_readout_positions(
                &temporal,
                forward.text_demand,
                1,
                context,
            )?
            .map(|selected| self.static_modules.text_output.forward(&selected, context))
            .transpose()?;
            forward.temporal_output = Some(temporal.clone());
            forward.text_logits = logits;
            Ok(temporal)
        } else {
            Ok(hidden.clone())
        }
    }

    fn set_readout_demand(
        &self,
        forward: &mut Self::ForwardContext,
        demand: eredu_core::OutputDemand,
    ) {
        forward.return_demand = demand;
        forward.text_demand = demand;
        forward.depth_demand = demand;
    }

    fn select_readout_positions(
        &self,
        _hidden: &B::Tensor,
        forward: &Self::ForwardContext,
        demand: eredu_core::OutputDemand,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Option<B::Tensor>, Self::Error> {
        let temporal = forward
            .temporal_output
            .as_ref()
            .ok_or_else(|| Error::backend("Moshi temporal output is unavailable"))?;
        crate::readout::select_readout_positions(temporal, demand, 1, context)
    }

    fn finish_forward(
        &mut self,
        _hidden: &B::Tensor,
        _state: &mut S,
        forward: &Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        let logits = forward
            .text_logits
            .as_ref()
            .ok_or_else(|| Error::backend("Moshi text logits are unavailable"))?;
        crate::readout::select_readout_positions(logits, forward.return_demand, 1, context)?
            .ok_or_else(|| Error::backend("Moshi state-only readout has no scores"))
    }

    fn retained_context_values<'a>(
        &'a self,
        forward: &'a Self::ForwardContext,
        _group: usize,
        _index: usize,
    ) -> Self::RetainedContextValues<'a> {
        [
            forward.temporal_mask.as_ref(),
            forward.temporal_output.as_ref(),
            forward.previous_depth_token.as_ref(),
        ]
        .into_iter()
        .flatten()
    }
}

#[cfg(test)]
mod transport_tests {
    use super::group_transport;
    use eredu_runtime::{
        ArchitectureGroupKind, ArchitectureGroupPlacement, ArchitectureMergeDestination,
    };

    #[test]
    fn temporal_and_depth_groups_declare_distinct_placement() {
        let temporal = group_transport(0);
        assert_eq!(temporal.kind, ArchitectureGroupKind::Decoder);
        assert_eq!(temporal.placement, ArchitectureGroupPlacement::Pipeline);
        assert_eq!(temporal.first_owner_static_roles, ["embedding"]);
        assert_eq!(temporal.last_owner_static_roles, ["norm", "output"]);
        assert_eq!(
            temporal.merge_destination,
            ArchitectureMergeDestination::LastOwner
        );

        let depth = group_transport(1);
        assert_eq!(depth.kind, ArchitectureGroupKind::Prediction);
        assert_eq!(depth.placement, ArchitectureGroupPlacement::OutputOwner);
        assert!(depth.first_owner_static_roles.is_empty());
        assert!(depth.last_owner_static_roles.is_empty());
        assert_eq!(
            depth.merge_destination,
            ArchitectureMergeDestination::OutputOwner
        );
    }
}

impl<B, S> ParallelLayeredArchitecture<B, S> for LayeredModel<B>
where
    B: NeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B> + ResettableRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor>,
{
    fn begin_forward_parallel<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error> {
        let metadata = crate::decoder::ModuleMetadata::new::<B>(context);
        metadata.controls::<(&mut Self, Self::Input<'_>, &mut S, &B::ParallelContext,
            &<B::Tensor as Tensor>::Context, Vec<&B::Tensor>, B::Tensor, StateLayout,
            LayeredForwardState<B::Tensor, Self::ForwardContext>)>()?;
        if self.source.0.parallel_geometry.is_none() {
            return Err(metadata.error(format_args!("Moshi model was not built with local geometry")));
        }
        if input.audio.len() != self.source.0.config.frame_schedule().total_audio_codebooks() {
            return Err(metadata.error(format_args!("Moshi parallel audio input count drifted")));
        }
        let count = input.audio.len().checked_add(1)
            .ok_or_else(|| metadata.error(format_args!("Moshi token input count overflowed")))?;
        let mut tokens = metadata.vector(count)?;
        tokens.push(input.text);
        tokens.extend(input.audio.iter().copied());
        let hidden = self.static_modules.embeddings.forward_parallel(&tokens, parallel, context)?;
        let expected = self.state_layout_impl(B::construction_metadata(context))?;
        self.begin_embedded_with_layout(hidden, input.mask, state, &expected, context)
    }

    fn forward_unit_parallel(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        match (group, unit) {
            (0, Unit::Temporal(block)) => super::forward_temporal_block_parallel(
                block, index, hidden, state, forward, parallel, context,
            ),
            (1, Unit::Depth(slice)) => {
                let temporal = forward
                    .temporal_output
                    .as_ref()
                    .ok_or_else(|| Error::backend("Moshi temporal output is unavailable"))?;
                let previous = forward.previous_depth_token.as_ref().ok_or_else(|| {
                    Error::backend("Moshi depth slice requires an accepted prior decision")
                })?;
                let metadata=crate::decoder::ModuleMetadata::new::<B>(context);
                let transformer=self.source.0.config.depth_template_for(index,
                    |message|metadata.error(message))?;
                slice.forward_parallel_with_readout(
                    transformer,
                    temporal,
                    previous,
                    self.source.0.config.temporal().num_hidden_layers() as usize,
                    state,
                    forward.depth_demand,
                    parallel,
                    context,
                )
            }
            _ => Err(Error::backend("Moshi parallel unit/group mismatch")),
        }
    }

    fn complete_execution_group_parallel(
        &mut self,
        group: usize,
        hidden: &B::Tensor,
        _state: &mut S,
        forward: &mut Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        if group == 0 {
            let temporal = self.static_modules.output_norm.forward(hidden, context)?;
            let logits = crate::readout::select_readout_positions(
                &temporal,
                forward.text_demand,
                1,
                context,
            )?
            .map(|selected| {
                B::vocabulary_parallel_project(
                    &mut self.static_modules.text_output,
                    &selected,
                    parallel,
                    context,
                )
            })
            .transpose()?;
            forward.temporal_output = Some(temporal.clone());
            forward.text_logits = logits;
            Ok(temporal)
        } else {
            Ok(hidden.clone())
        }
    }

    fn finish_forward_parallel(
        &mut self,
        _hidden: &B::Tensor,
        _state: &mut S,
        forward: &Self::ForwardContext,
        _parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        let logits = forward
            .text_logits
            .as_ref()
            .ok_or_else(|| Error::backend("Moshi text logits are unavailable"))?;
        crate::readout::select_readout_positions(logits, forward.return_demand, 1, context)?
            .ok_or_else(|| Error::backend("Moshi state-only readout has no scores"))
    }
}

/// Stateless mapping from the two-group graph to ordered runtime decisions.
#[derive(Debug, Clone, Copy)]
pub struct DecisionBoundary {
    text: TokenDomain,
    audio: TokenDomain,
}

impl DecisionBoundary {
    /// Creates target domains including the released padding embedding row.
    pub fn new(config: &MoshiConfig) -> Result<Self, Error> {
        let text = usize::try_from(
            config
                .text_vocabulary_size()
                .checked_add(1)
                .ok_or_else(|| Error::backend("Moshi text decision domain overflowed"))?,
        )
        .map_err(Error::backend)?;
        let audio = usize::try_from(
            config
                .audio_vocabulary_size()
                .checked_add(1)
                .ok_or_else(|| Error::backend("Moshi audio decision domain overflowed"))?,
        )
        .map_err(Error::backend)?;
        Ok(Self {
            text: TokenDomain::new(text),
            audio: TokenDomain::new(audio),
        })
    }

    /// Accepted text target IDs, including the released padding row.
    pub const fn text_token_domain(&self) -> TokenDomain {
        self.text
    }

    /// Accepted audio target IDs, including the released padding row.
    pub const fn audio_token_domain(&self) -> TokenDomain {
        self.audio
    }
}

impl<SB, T, E> SequentialDecisionBoundary<SB, ForwardContext<T>, E> for DecisionBoundary
where
    SB: SamplingBackend<Logits = T, Token = T>,
    T: Clone,
    SB::Error: std::fmt::Display,
    E: From<Error>,
{
    fn prediction_at(
        &self,
        point: LayeredTraversalPoint,
        _forward: &ForwardContext<T>,
    ) -> Option<usize> {
        match point {
            LayeredTraversalPoint::Group { group: 0 } => Some(0),
            LayeredTraversalPoint::Unit { group: 1, index } => Some(index + 1),
            _ => None,
        }
    }

    fn prepare_logits(
        &mut self,
        prediction: usize,
        _point: LayeredTraversalPoint,
        demand: eredu_core::OutputDemand,
        forward: &mut ForwardContext<T>,
        _context: &SB::Context,
    ) -> Result<(), E> {
        if prediction == 0 {
            forward.text_demand = eredu_runtime::merge_output_demand(forward.return_demand, demand);
        } else {
            forward.depth_demand = demand;
        }
        Ok(())
    }

    fn optional_logits(
        &mut self,
        prediction: usize,
        _point: LayeredTraversalPoint,
        value: &T,
        forward: &mut ForwardContext<T>,
        _context: &SB::Context,
    ) -> Result<Option<T>, E> {
        Ok(if prediction == 0 {
            forward.text_logits.clone()
        } else if forward.has_depth_logits() {
            Some(value.clone())
        } else {
            None
        })
    }

    fn logits(
        &mut self,
        prediction: usize,
        _point: LayeredTraversalPoint,
        value: &T,
        forward: &mut ForwardContext<T>,
        _context: &SB::Context,
    ) -> Result<T, E> {
        if prediction == 0 {
            forward
                .text_logits
                .clone()
                .ok_or_else(|| Error::backend("Moshi text decision has no logits").into())
        } else {
            Ok(value.clone())
        }
    }

    fn token_domain(
        &mut self,
        prediction: usize,
        _point: LayeredTraversalPoint,
        _forward: &ForwardContext<T>,
    ) -> Result<TokenDomain, E> {
        Ok(if prediction == 0 {
            self.text
        } else {
            self.audio
        })
    }

    fn accept(
        &mut self,
        _prediction: usize,
        _point: LayeredTraversalPoint,
        token: &T,
        forward: &mut ForwardContext<T>,
        _context: &SB::Context,
    ) -> Result<(), E> {
        forward.previous_depth_token = Some(token.clone());
        Ok(())
    }

    fn decision_error(&mut self, error: SequentialDecisionError<SB::Error>) -> E {
        Error::backend(error.to_string()).into()
    }
}
