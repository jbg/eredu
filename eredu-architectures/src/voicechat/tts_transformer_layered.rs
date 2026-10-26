//! Shared layered traversal for causal speech and bidirectional character stacks.

use super::tts_transformer::{encoder_mask, NoTransformerCache, TransformerConfig};
use crate::decoder::{self, AttentionInput, Config, PartitionedConfig, TransformerBlock};
use eredu_core::{cache::LayerCachePolicy, BackendFailure, LayerSchedule};
use eredu_nn::{
    AttentionCache, NeuralBackend, NormalizationOperator, Parameterized, Tensor, TensorElementType,
};
use eredu_runtime::{
    ArchitectureGroupKind, ArchitectureGroupPlacement, ArchitectureGroupTransport,
    ArchitectureMergeDestination, ArchitectureParameterDescription, ArchitectureStatePartitionPlan,
    ArchitectureStatePartitionRule, ComponentAdapter, ComponentDefinition, ComponentEquations,
    ComponentKernel, ComponentUnit, ExecutionGraph, ExecutionGroupSpec, ExecutionUnitAddress,
    ExecutionUnitLayout, LayeredForwardState, LayerwisePolicy, LayerwiseRuntime, LocalModelLayout,
    OwnedParameterGroupSpec, ParallelComponentKernel, ParameterGroupOwner, RuntimeLayerState,
    RuntimeState, StateError, StateLayout, SubmissionBackend,
};
use std::{marker::PhantomData, ops::DerefMut};

#[path = "tts_transformer_partition.rs"]
mod partition;
pub use partition::{TransformerBoundary, TransformerBoundarySchema};
#[path = "tts_transformer_pipeline.rs"]
mod pipeline;

/// Invalid portable transformer policy, state, or native execution.
#[derive(Debug, thiserror::Error)]
pub enum TransformerError {
    /// Shared component declaration or selected ownership is invalid.
    #[error(transparent)]
    Component(#[from] eredu_runtime::ComponentDefinitionError),
    /// Invalid supplied geometry, cache frontier or request mode.
    #[error("{0}")]
    Configuration(&'static str),
    /// Shared decoder construction or numerical mechanism failed.
    #[error(transparent)]
    Compute(#[from] eredu_nn::Error),
    /// Portable graph declarations are invalid.
    #[error(transparent)]
    Graph(#[from] eredu_runtime::ExecutionGraphError),
    /// Unit ownership cannot map to the graph.
    #[error(transparent)]
    UnitLayout(#[from] eredu_runtime::ExecutionUnitLayoutError),
    /// Mutable state layout or indexing is invalid.
    #[error(transparent)]
    State(#[from] StateError),
    /// Layer cache policy is invalid.
    #[error(transparent)]
    Schedule(#[from] eredu_core::LayerScheduleError),
    /// Symbolic parameter topology is invalid.
    #[error(transparent)]
    Parameters(#[from] eredu_runtime::ParallelPlanError),
    /// Parameter ownership is invalid.
    #[error(transparent)]
    Ownership(#[from] eredu_runtime::ArchitectureParameterError),
    /// Runtime state identity is invalid.
    #[error(transparent)]
    Identity(#[from] eredu_core::cache::PromptCacheError),
    /// Provider or submission failed, preserving its original source.
    #[error(transparent)]
    Backend(#[from] BackendFailure),
}

/// One independently leased transformer block or final hidden normalization.
#[derive(Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub enum TransformerUnit<B: NeuralBackend> {
    /// Shared decoder block with exact family configuration.
    Block(TransformerBlock<B>),
    /// Offset RMS normalization after all blocks.
    Norm(B::Normalization),
}
/// Resident and bounded stacks expose one execution contract over borrowed caches.
pub trait TransformerExecution<B: NeuralBackend> {
    /// Identity certifying immutable parameters and deterministic encoder results.
    /// Replacing parameters, equations or execution resources changes the identity.
    /// Mutable and collective providers return `None`; cache hits skip their calls.
    fn immutable_parameters_identity(
        &self,
        _context: &<B::Tensor as Tensor>::Context,
    ) -> Option<&eredu_core::resources::ResourceIdentity> {
        None
    }

    /// Exact retained equations and physical parameter formats.
    fn config(&self) -> &TransformerConfig;
    /// Runs independent character rows with boolean key-padding visibility.
    fn forward_encoder(
        &mut self,
        embedded: &B::Tensor,
        key_mask: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, TransformerError>;
}

/// Causal execution over the state selected by the provider. Validation observes
/// the logical frontier even when a pipeline stage owns no attention slots.
pub trait CausalTransformerExecution<B: NeuralBackend, S: ?Sized>: TransformerExecution<B> {
    /// Validates selected ownership, cache geometry and the common logical frontier.
    fn causal_position(&self, state: &S) -> Result<i32, TransformerError>;
    /// Advances caller-owned unpublished state through one selected traversal.
    fn forward_causal(
        &mut self,
        embedded: &B::Tensor,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, TransformerError>;
}

/// Checks the complete ordinary causal cache state before parameter access.
pub fn validate_causal_caches<B: NeuralBackend, C: AttentionCache<B::Tensor>>(
    config: &TransformerConfig,
    caches: &[C],
) -> Result<i32, TransformerError> {
    if !config.geometry().causal || caches.len() != config.geometry().num_hidden_layers {
        return Err(TransformerError::Configuration(
            "causal TTS cache ownership differs",
        ));
    }
    let position = caches
        .first()
        .ok_or(TransformerError::Configuration("missing causal TTS caches"))?
        .offset();
    if position < 0
        || position as usize > config.geometry().max_position_embeddings
        || caches
            .iter()
            .zip(config.attention_schedule().iter())
            .any(|(cache, policy)| {
                cache.offset() != position
                    || cache.max_size() != policy.window().map(|n| n.get() as i32)
            })
    {
        return Err(TransformerError::Configuration(
            "causal TTS cache frontier or retention differs",
        ));
    }
    Ok(position)
}

/// Caller-owned cache state over either an owned vector or a transient mutable slice.
/// Storage ownership changes neither the equations nor the publication boundary.
pub struct TransformerRuntimeState<
    'a,
    B: NeuralBackend,
    C: RuntimeLayerState<B>,
    D: DerefMut<Target = [C]> = &'a mut [C],
> {
    layout: StateLayout,
    caches: D,
    causal: bool,
    cache_base: usize,
    position: i32,
    identity: String,
    backend: PhantomData<(fn() -> B, &'a mut [C])>,
}
/// Owned candidate cache state for persistent distributed component runtimes.
pub type OwnedTransformerRuntimeState<B, C> = TransformerRuntimeState<'static, B, C, Vec<C>>;
/// State type selecting runtime storage; each invocation supplies its actual cache borrow.
pub type TransformerStateMarker<B> = TransformerRuntimeState<'static, B, NoTransformerCache<B>>;
impl<'a, B: NeuralBackend, C: RuntimeLayerState<B>, D: DerefMut<Target = [C]>>
    TransformerRuntimeState<'a, B, C, D>
{
    /// Cache slots in local partition order; this does not publish their state.
    pub fn caches(&self) -> &[C] {
        &self.caches
    }
    /// Mutably accesses the caller's candidate cache slots.
    pub fn caches_mut(&mut self) -> &mut [C] {
        &mut self.caches
    }
    /// Returns the original storage after the enclosing invocation retires.
    pub fn into_caches(self) -> D {
        self.caches
    }
    /// Copies cache storage through the caller's isolation mechanism while
    /// preserving selected ownership and the logical frontier. The callback must
    /// return independent mutable caches; execution validates their geometry.
    pub fn try_map_caches<C2, E>(
        &self,
        copy: impl FnOnce(&[C]) -> Result<Vec<C2>, E>,
    ) -> Result<OwnedTransformerRuntimeState<B, C2>, E>
    where
        C2: RuntimeLayerState<B> + 'static,
    {
        Ok(TransformerRuntimeState {
            layout: self.layout.clone(),
            caches: copy(&self.caches)?,
            causal: self.causal,
            cache_base: self.cache_base,
            position: self.position,
            identity: self.identity.clone(),
            backend: PhantomData,
        })
    }

    /// Borrows caches under the architecture's exact selected rank-local layout.
    pub fn for_architecture(
        architecture: &TransformerKernel<B>,
        caches: D,
    ) -> Result<Self, TransformerError>
    where
        C: AttentionCache<B::Tensor>,
    {
        if architecture.partition.is_some() {
            return Err(TransformerError::Configuration(
                "partitioned TTS state requires for_partition",
            ));
        }
        Ok(Self {
            layout: architecture.state_layout.clone(),
            position: caches.first().map_or(0, |cache| cache.offset()),
            identity: architecture.config.architecture_fingerprint(),
            caches,
            causal: architecture.config.geometry().causal,
            cache_base: 0,
            backend: PhantomData,
        })
    }
    /// Wraps an unpublished branch without copying native cache handles.
    pub fn new(config: &TransformerConfig, caches: D) -> Result<Self, TransformerError>
    where
        C: AttentionCache<B::Tensor>,
    {
        Ok(Self {
            layout: transformer_state_layout(config)?,
            position: caches.first().map_or(0, |cache| cache.offset()),
            identity: config.architecture_fingerprint(),
            caches,
            causal: config.geometry().causal,
            cache_base: 0,
            backend: PhantomData,
        })
    }
}
impl<
        B: NeuralBackend,
        C: AttentionCache<B::Tensor> + RuntimeLayerState<B>,
        D: DerefMut<Target = [C]>,
    > TransformerRuntimeState<'_, B, C, D>
{
    /// Retained logical causal position, including stages without attention slots.
    pub fn causal_position(&self) -> Result<i32, TransformerError> {
        if self.position < 0
            || self
                .caches
                .iter()
                .any(|cache| cache.offset() != self.position)
        {
            return Err(TransformerError::Configuration(
                "TTS state cache and logical frontiers differ",
            ));
        }
        Ok(self.position)
    }
}

impl<B: NeuralBackend, C: RuntimeLayerState<B>, D: DerefMut<Target = [C]>> RuntimeState<B>
    for TransformerRuntimeState<'_, B, C, D>
{
    type RetainedValues<'a>
        = C::RetainedValues<'a>
    where
        Self: 'a,
        B::Tensor: 'a;
    fn layout(&self) -> &StateLayout {
        &self.layout
    }
    fn optional_layout(&self) -> Option<&StateLayout> {
        self.causal.then_some(&self.layout)
    }
    fn retained_values(
        &self,
        ordinal: usize,
        _: ExecutionUnitAddress,
    ) -> Result<Self::RetainedValues<'_>, StateError> {
        Ok(self
            .caches
            .get(ordinal)
            .ok_or(StateError::UnknownLayer {
                layer: ordinal,
                count: self.caches.len(),
            })?
            .retained_values())
    }
}

/// Canonical embedded-input transformer operations and cache references.
pub fn tts_transformer_component_definition(
    config: &TransformerConfig,
) -> Result<ComponentDefinition, TransformerError> {
    transformer_component_definition_with_state(config, transformer_state_layout(config)?)
}
fn transformer_component_definition_with_state(
    config: &TransformerConfig,
    state: StateLayout,
) -> Result<ComponentDefinition, TransformerError> {
    let parameters = tts_transformer_parameter_description(config)?;
    let units = (0..parameters.unit_layout().len())
        .map(|ordinal| {
            let address = parameters
                .unit_layout()
                .address(ordinal)
                .expect("canonical unit");
            ComponentUnit {
                path: if address.group() == 0 {
                    format!("{}.layers.{}", config.parameter_root(), address.index())
                } else {
                    format!("{}.norm", config.parameter_root())
                },
                state: if address.group() == 0 && config.geometry().causal {
                    address.index()..address.index() + 1
                } else {
                    0..0
                },
            }
        })
        .collect();
    let transport = (0..2)
        .map(|group| ArchitectureGroupTransport {
            placement: ArchitectureGroupPlacement::Pipeline,
            kind: if group == 1 {
                ArchitectureGroupKind::Projector
            } else if config.geometry().causal {
                ArchitectureGroupKind::Decoder
            } else {
                ArchitectureGroupKind::AudioEncoder
            },
            first_owner_static_roles: vec![],
            last_owner_static_roles: vec![],
            merge_destination: ArchitectureMergeDestination::LastOwner,
            parallel_subgroup: None,
            request_optional: false,
        })
        .collect();
    Ok(ComponentDefinition::new(
        config.parameter_root(),
        parameters,
        units,
        transport,
        config.parameter_root(),
        vec![],
        state,
        ArchitectureStatePartitionPlan::new([ArchitectureStatePartitionRule::group_units(
            0,
            0..config.geometry().num_hidden_layers,
        )]),
    )?)
}

/// Embedded input for either transformer invocation, with explicit key visibility.
pub struct TransformerInput<'a, T> {
    /// Embedded batch, sequence and hidden features.
    pub embedded: &'a T,
    /// Present only for the bidirectional character encoder.
    pub key_mask: Option<&'a T>,
}
/// Invocation-local masks and the initial common cache frontier.
pub struct TransformerForward<T> {
    keys: Option<T>,
    mask: Option<T>,
    sequence: i32,
    offset: i32,
    dtype: TensorElementType,
    normalized: bool,
}

/// Exact transformer graph with separately materializable final normalization.
pub struct TransformerKernel<B: NeuralBackend> {
    config: TransformerConfig,
    local_blocks: Option<Vec<TransformerConfig>>,
    state_layout: StateLayout,
    partition: Option<partition::TransformerOwnership>,
    backend: PhantomData<fn() -> B>,
}
impl<B: NeuralBackend> TransformerKernel<B> {
    /// Validates normalized policy without allocating native parameters.
    pub fn new(config: TransformerConfig) -> Result<Self, TransformerError> {
        config.validate_config()?;
        let state_layout = transformer_state_layout(&config)?;
        Ok(Self {
            config,
            local_blocks: None,
            state_layout,
            partition: None,
            backend: PhantomData,
        })
    }
    /// Constructs rank-local modules from an already selected physical layout.
    /// Global parameter identities remain unchanged; only projection widths and
    /// cache head counts are local. No tensor or parameter payload is allocated.
    pub fn new_parallel(
        config: TransformerConfig,
        layout: &LocalModelLayout,
    ) -> Result<Self, TransformerError> {
        let blocks = transformer_local_configs(&config, layout)?;
        let state_layout = transformer_local_state_layout(&config, &blocks)?;
        Ok(Self {
            config,
            local_blocks: Some(blocks),
            state_layout,
            partition: None,
            backend: PhantomData,
        })
    }
    /// Couples the selected geometry with shared typed execution contracts.
    pub fn into_component(
        self,
    ) -> Result<ComponentAdapter<ComponentDefinition, Self>, TransformerError> {
        Ok(ComponentAdapter::new(
            transformer_component_definition_with_state(&self.config, self.state_layout.clone())?,
            self,
        )?)
    }
    /// Whether parameter construction and execution require tensor collectives.
    pub fn is_parallel(&self) -> bool {
        self.local_blocks.is_some()
    }
    /// Retained exact construction policy.
    pub fn config(&self) -> &TransformerConfig {
        &self.config
    }
}
impl<B: NeuralBackend> TransformerKernel<B> {
    /// Validates exact retained ownership and frontier without learned execution.
    pub fn validate_causal_state<C, D>(
        &self,
        state: &TransformerRuntimeState<'_, B, C, D>,
    ) -> Result<i32, TransformerError>
    where
        C: AttentionCache<B::Tensor> + RuntimeLayerState<B>,
        D: DerefMut<Target = [C]>,
    {
        if !self.config.geometry().causal
            || state.identity != self.config.architecture_fingerprint()
        {
            return Err(TransformerError::Configuration(
                "TTS state belongs to different causal equations",
            ));
        }
        let (start, count, layout, retained) = match &self.partition {
            Some(selected) => (
                selected.blocks.start,
                selected.blocks.len(),
                selected.local_state.as_ref().unwrap_or(&self.state_layout),
                !selected.blocks.is_empty(),
            ),
            None => (
                0,
                self.config.geometry().num_hidden_layers,
                &self.state_layout,
                true,
            ),
        };
        if state.cache_base != start
            || state.caches.len() != count
            || &state.layout != layout
            || state.causal != retained
        {
            return Err(TransformerError::Configuration(
                "TTS state ownership differs from selected architecture",
            ));
        }
        let position = state.causal_position()?;
        if position as usize > self.config.geometry().max_position_embeddings
            || state.caches.iter().enumerate().any(|(index, cache)| {
                let policy = self
                    .config
                    .attention_schedule()
                    .get(start + index)
                    .expect("validated layer range");
                cache.max_size() != policy.window().map(|n| n.get() as i32)
            })
        {
            return Err(TransformerError::Configuration(
                "TTS state capacity or retention differs",
            ));
        }
        Ok(position)
    }

    fn begin_input<
        C: AttentionCache<B::Tensor> + RuntimeLayerState<B>,
        D: DerefMut<Target = [C]>,
    >(
        &mut self,
        input: TransformerInput<'_, B::Tensor>,
        state: &mut TransformerRuntimeState<'_, B, C, D>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, TransformerForward<B::Tensor>>, TransformerError>
    {
        let sequence = validate_input(&self.config, input.embedded)?;
        if state.layout != self.state_layout.clone() {
            return Err(TransformerError::Configuration(
                "TTS runtime state layout mismatch",
            ));
        }
        let offset;

        let hidden = if self.config.geometry().causal {
            if input.key_mask.is_some()
                || state.caches.len() != self.config.geometry().num_hidden_layers
            {
                return Err(TransformerError::Configuration(
                    "causal TTS execution requires one cache per causal layer",
                ));
            }
            offset = self.validate_causal_state(state)?;
            if offset < 0
                || offset
                    .checked_add(sequence)
                    .is_none_or(|n| n as usize > self.config.geometry().max_position_embeddings)
            {
                return Err(TransformerError::Configuration(
                    "TTS positional capacity exhausted",
                ));
            }
            for (cache, policy) in state
                .caches
                .iter()
                .zip(self.config.attention_schedule().iter())
            {
                if cache.offset() != offset
                    || cache.max_size() != policy.window().map(|n| n.get() as i32)
                {
                    return Err(TransformerError::Configuration(
                        "TTS cache frontier or retention policy mismatch",
                    ));
                }
            }
            input.embedded.clone()
        } else {
            let keys = input.key_mask.ok_or(TransformerError::Configuration(
                "character encoder requires key mask",
            ))?;
            if !state.caches.is_empty()
                || keys.shape() != &input.embedded.shape()[..2]
                || keys.element_type() != Some(TensorElementType::Bool)
            {
                return Err(TransformerError::Configuration(
                    "character encoder requires boolean mask matching embedded batch",
                ));
            }
            offset = 0;
            let scale =
                B::Tensor::full_f32((self.config.hidden_size() as f32).sqrt(), &[], context)?
                    .cast_float(
                        input.embedded.element_type().expect("validated precision"),
                        context,
                    )?;
            input.embedded.multiply(&scale, context)?
        };
        Ok(LayeredForwardState {
            hidden,
            context: TransformerForward {
                keys: input.key_mask.cloned(),
                mask: None,
                sequence,
                offset,
                dtype: input.embedded.element_type().expect("validated precision"),
                normalized: false,
            },
        })
    }
}

/// Validates complete physical members before deriving rank-local block geometry.
pub(super) fn transformer_local_configs(
    config: &TransformerConfig,
    layout: &LocalModelLayout,
) -> Result<Vec<TransformerConfig>, TransformerError> {
    let global = tts_transformer_parameter_description(config)?;
    let expected = global
        .groups()
        .iter()
        .flat_map(|group| group.members())
        .count();
    if layout.len() != expected {
        return Err(TransformerError::Configuration(
            "TTS local layout has different parameter coverage",
        ));
    }
    for member in global.groups().iter().flat_map(|group| group.members()) {
        let actual = layout
            .tensor(member.target())
            .ok_or(TransformerError::Configuration(
                "TTS local layout is missing a parameter",
            ))?;
        if actual.global_shape() != member.global_shape()
            || actual.local_shape().len() != member.global_shape().len()
            || actual.local_shape().contains(&0)
            || !actual.additional_placements().is_empty()
        {
            return Err(TransformerError::Configuration(
                "TTS local layout has different or empty physical geometry",
            ));
        }
    }
    let blocks = (0..config.geometry().num_hidden_layers)
        .map(|index| config.local_block_config(index, layout))
        .collect::<Result<Vec<_>, _>>()?;
    let local_descriptions = blocks
        .iter()
        .map(decoder::embedded_parameter_description)
        .collect::<Result<Vec<_>, _>>()?;
    let global_decoder = decoder::embedded_parameter_description(config)?;
    for group in global.groups() {
        let local = match group.owner() {
            ParameterGroupOwner::ExecutionUnit {
                group: id,
                global_unit,
                ..
            } if id.as_str() == config.parameter_root() => &local_descriptions[*global_unit],
            _ => &global_decoder,
        };
        for member in group.members() {
            let expected = local
                .groups()
                .iter()
                .flat_map(|group| group.members())
                .find(|candidate| candidate.target() == member.target())
                .ok_or(TransformerError::Configuration(
                    "TTS local parameter declaration differs",
                ))?;
            if layout
                .tensor(member.target())
                .expect("validated complete layout")
                .local_shape()
                != expected.global_shape()
            {
                return Err(TransformerError::Configuration(
                    "TTS local projection or normalization geometry differs",
                ));
            }
        }
    }
    Ok(blocks)
}

pub(super) fn transformer_local_state_layout(
    config: &TransformerConfig,
    blocks: &[TransformerConfig],
) -> Result<StateLayout, TransformerError> {
    if config.geometry().causal {
        Ok(StateLayout::new(
            decoder::cache_layout_with_key_value_heads(
                config,
                blocks.iter().map(Config::num_key_value_heads),
            )?,
        )?)
    } else {
        transformer_state_layout(config)
    }
}

pub(super) fn validate_transformer_partition<G>(
    definition: &ComponentDefinition,
    config: &TransformerConfig,
    layout: Option<&LocalModelLayout>,
    selected: &eredu_runtime::ArchitecturePartition<G, TransformerBoundarySchema>,
) -> Result<(), TransformerError> {
    if let Some(layout) = layout {
        transformer_local_configs(config, layout)?;
    }
    partition::validate_partition(definition, config, selected).map(|_| ())
}

fn transformer_graph(config: &TransformerConfig) -> Result<ExecutionGraph, TransformerError> {
    let blocks = config.parameter_root().to_owned();
    let norm = format!("{}.normalization", config.parameter_root());
    Ok(ExecutionGraph::new(
        vec![
            ExecutionGroupSpec::root(&blocks),
            ExecutionGroupSpec::with_dependencies(&norm, [blocks.as_str()]),
        ],
        &norm,
    )?)
}
fn transformer_state_layout(config: &TransformerConfig) -> Result<StateLayout, TransformerError> {
    if config.geometry().causal {
        Ok(decoder::state_layout(config)?)
    } else {
        Ok(StateLayout::new(LayerSchedule::new(
            config.geometry().num_hidden_layers,
            vec![LayerCachePolicy::NoState; config.geometry().num_hidden_layers],
        )?)?)
    }
}
/// Cold shared-decoder parameter shapes and format companions, with speech unit ownership.
pub fn tts_transformer_parameter_description(
    config: &TransformerConfig,
) -> Result<ArchitectureParameterDescription, TransformerError> {
    let description = decoder::embedded_parameter_description(config)?;
    let graph = transformer_graph(config)?;
    let layout = ExecutionUnitLayout::new(&graph, [config.geometry().num_hidden_layers, 1])?;
    let mut owned = Vec::new();
    for old in description.groups() {
        let (group, index) = match old.owner() {
            ParameterGroupOwner::ExecutionUnit { global_unit, .. } => (0, *global_unit),
            ParameterGroupOwner::StaticRole(role) if role == "norm" => (1, 0),
            _ => {
                return Err(TransformerError::Configuration(
                    "unexpected embedded transformer parameter owner",
                ));
            }
        };
        let spec = old.group().clone();
        owned.push(OwnedParameterGroupSpec::new(
            ParameterGroupOwner::execution_unit(
                layout.group_id(group).expect("declared group").clone(),
                index,
            ),
            spec,
        ));
    }
    Ok(ArchitectureParameterDescription::from_owned(&graph, &layout, owned)?)
}
impl<B: NeuralBackend> ComponentKernel<B> for TransformerKernel<B> {
    type Input<'a> = TransformerInput<'a, B::Tensor>;
    type Unit = TransformerUnit<B>;
    type ForwardContext = TransformerForward<B::Tensor>;
    type RetainedContextValues<'a>
        = std::iter::Chain<std::option::Iter<'a, B::Tensor>, std::option::Iter<'a, B::Tensor>>
    where
        B::Tensor: 'a;
    type Error = TransformerError;
    fn state_identity(
        &self,
        state: &eredu_runtime::PartitionState,
        topology: eredu_core::cache::PromptCacheTopology,
    ) -> Result<eredu_runtime::ModelStateIdentity, TransformerError> {
        let global = self.state_layout.clone();
        let begin = state.global_layer_offset();
        let end = begin
            .checked_add(state.layout().len())
            .filter(|&n| n <= global.len())
            .ok_or(TransformerError::Configuration(
                "TTS state partition exceeds stack",
            ))?;
        if global.slice(begin..end)? != *state.layout() {
            return Err(TransformerError::Configuration(
                "TTS partition state differs from selected layout",
            ));
        }
        Ok(eredu_runtime::ModelStateIdentity::new(
            "nemotron_voicechat",
            self.config.parameter_root(),
            self.config.architecture_fingerprint(),
            global.len(),
            begin,
            0,
            topology,
        )?)
    }
    fn build_unit(
        &self,
        address: ExecutionUnitAddress,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<TransformerUnit<B>, TransformerError> {
        let group = address.group();
        let index = address.index();
        Ok(if group == 0 {
            TransformerUnit::Block(
                self.local_blocks
                    .as_ref()
                    .map_or(&self.config, |blocks| &blocks[index])
                    .build_block(index, context)?,
            )
        } else {
            TransformerUnit::Norm(self.config.build_final_norm::<B>(context)?)
        })
    }
    fn executes_group(&self, _: usize, _: &Self::ForwardContext) -> bool {
        true
    }
    fn retained_values<'a>(
        &'a self,
        forward: &'a Self::ForwardContext,
        _: ExecutionUnitAddress,
    ) -> Self::RetainedContextValues<'a> {
        forward.keys.iter().chain(forward.mask.iter())
    }
}
impl<B, C, D: DerefMut<Target = [C]>> ComponentEquations<B, TransformerRuntimeState<'_, B, C, D>>
    for TransformerKernel<B>
where
    B: NeuralBackend,
    C: AttentionCache<B::Tensor> + RuntimeLayerState<B>,
{
    fn begin<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut TransformerRuntimeState<'_, B, C, D>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, TransformerError> {
        if self.is_parallel() || self.partition.is_some() {
            return Err(TransformerError::Configuration(
                "selected TTS transformer requires its parallel or partition invocation",
            ));
        }
        self.begin_input(input, state, context)
    }
    fn begin_group(
        &mut self,
        _: usize,
        initial: &B::Tensor,
        dependencies: &[&B::Tensor],
        _: &mut TransformerRuntimeState<'_, B, C, D>,
        _: &mut Self::ForwardContext,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, TransformerError> {
        Ok(dependencies.first().copied().unwrap_or(initial).clone())
    }
    fn execute(
        &mut self,
        address: ExecutionUnitAddress,
        unit: &mut TransformerUnit<B>,
        hidden: &B::Tensor,
        state: &mut TransformerRuntimeState<'_, B, C, D>,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, TransformerError> {
        if self.is_parallel() {
            return Err(TransformerError::Configuration(
                "rank-local TTS transformer requires parallel execution",
            ));
        }
        self.execute_unit(
            address.group(),
            address.index(),
            unit,
            hidden,
            state,
            forward,
            None,
            context,
        )
    }
    fn finish(
        &mut self,
        hidden: &B::Tensor,
        state: &mut TransformerRuntimeState<'_, B, C, D>,
        forward: &Self::ForwardContext,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, TransformerError> {
        if self.config.geometry().causal {
            state.position = forward.offset + forward.sequence;
        }
        Ok(hidden.clone())
    }
}
impl<B: NeuralBackend> TransformerKernel<B> {
    #[allow(clippy::too_many_arguments)]
    fn execute_unit<
        C: AttentionCache<B::Tensor> + RuntimeLayerState<B>,
        D: DerefMut<Target = [C]>,
    >(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut TransformerUnit<B>,
        hidden: &B::Tensor,
        state: &mut TransformerRuntimeState<'_, B, C, D>,
        forward: &mut TransformerForward<B::Tensor>,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, TransformerError> {
        match (group, unit) {
            (0, TransformerUnit::Block(block)) => {
                if self.config.geometry().causal {
                    let cache = index
                        .checked_sub(state.cache_base)
                        .and_then(|local| state.caches.get_mut(local))
                        .ok_or(TransformerError::Configuration(
                            "TTS partition does not own this cache",
                        ))?;
                    let window = cache.max_size();
                    let past = if cache.uses_blockwise_attention() {
                        forward.offset
                    } else {
                        window.map_or(forward.offset, |w| forward.offset.min(w - 1))
                    };
                    forward.mask = Some(B::causal_mask(
                        forward.sequence,
                        past,
                        window.map(|w| w - 1),
                        context,
                    )?);
                    Ok(forward_block::<B, C>(
                        block,
                        AttentionInput {
                            hidden,
                            selected_positions: None,
                            mask: forward.mask.as_ref(),
                            cache: Some(cache),
                            allow_sliding_prefill: false,
                            rotary_position: None,
                        },
                        parallel,
                        context,
                    )?)
                } else {
                    let window = self
                        .config
                        .attention_schedule()
                        .get(index)
                        .expect("validated block")
                        .window()
                        .map(|n| n.get() as i32);
                    forward.mask = Some(encoder_mask::<B::Tensor>(
                        forward.keys.as_ref().expect("validated encoder keys"),
                        forward.sequence,
                        window,
                        forward.dtype,
                        context,
                    )?);
                    Ok(forward_block::<B, NoTransformerCache<B>>(
                        block,
                        AttentionInput {
                            hidden,
                            selected_positions: None,
                            mask: forward.mask.as_ref(),
                            cache: None,
                            allow_sliding_prefill: false,
                            rotary_position: None,
                        },
                        parallel,
                        context,
                    )?)
                }
            }
            (1, TransformerUnit::Norm(norm)) => {
                if forward.normalized {
                    return Err(TransformerError::Configuration(
                        "TTS normalization already executed",
                    ));
                }
                let output = norm.forward(hidden, context)?;
                forward.normalized = true;
                Ok(output)
            }
            _ => Err(TransformerError::Configuration(
                "leased TTS transformer unit differs from graph address",
            )),
        }
    }
}

fn forward_block<B: NeuralBackend, C: AttentionCache<B::Tensor>>(
    block: &mut TransformerBlock<B>,
    input: AttentionInput<'_, B::Tensor, C>,
    parallel: Option<&B::ParallelContext>,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<B::Tensor, eredu_nn::Error> {
    match parallel {
        Some(parallel) => {
            block
                .body
                .forward_tensor_parallel(block.output_norm.as_mut(), input, parallel, context)
        }
        None => block
            .body
            .forward(block.output_norm.as_mut(), input, context),
    }
}

impl<B, C, D: DerefMut<Target = [C]>>
    ParallelComponentKernel<B, TransformerRuntimeState<'_, B, C, D>> for TransformerKernel<B>
where
    B: NeuralBackend,
    C: AttentionCache<B::Tensor> + RuntimeLayerState<B>,
{
    fn begin_parallel<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut TransformerRuntimeState<'_, B, C, D>,
        _: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, TransformerError> {
        if !self.is_parallel() || self.partition.is_some() {
            return Err(TransformerError::Configuration(
                "parallel TTS execution requires complete rank-local construction",
            ));
        }
        self.begin_input(input, state, context)
    }
    fn execute_parallel(
        &mut self,
        address: ExecutionUnitAddress,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut TransformerRuntimeState<'_, B, C, D>,
        forward: &mut Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, TransformerError> {
        self.execute_unit(
            address.group(),
            address.index(),
            unit,
            hidden,
            state,
            forward,
            Some(parallel),
            context,
        )
    }
    fn finish_parallel(
        &mut self,
        hidden: &B::Tensor,
        state: &mut TransformerRuntimeState<'_, B, C, D>,
        forward: &Self::ForwardContext,
        _: &B::ParallelContext,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, TransformerError> {
        if self.config.geometry().causal {
            state.position = forward.offset + forward.sequence;
        }
        Ok(hidden.clone())
    }
}

fn validate_input<T: Tensor>(
    config: &TransformerConfig,
    hidden: &T,
) -> Result<i32, TransformerError> {
    let shape = hidden.shape();
    if shape.len() != 3
        || shape[0] <= 0
        || shape[1] <= 0
        || shape[2] != config.hidden_size()
        || shape[1] as usize > config.geometry().max_position_embeddings
        || !matches!(
            hidden.element_type(),
            Some(TensorElementType::F32 | TensorElementType::F16 | TensorElementType::Bf16)
        )
    {
        return Err(TransformerError::Configuration(
            "invalid TTS embedded input or positional capacity",
        ));
    }
    Ok(shape[1])
}

/// Existing bounded provider and traversal for either transformer stack.
pub struct LayerwiseTransformer<B, P>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    P: LayerwisePolicy<B, TransformerUnit<B>>,
{
    runtime: LayerwiseRuntime<
        ComponentAdapter<ComponentDefinition, TransformerKernel<B>>,
        B,
        TransformerStateMarker<B>,
        P,
    >,
}
impl<B, P> LayerwiseTransformer<B, P>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    P: LayerwisePolicy<B, TransformerUnit<B>>,
    P::Error: std::fmt::Display,
{
    /// Adopts an exact already-prepared unit policy without allocating duplicate parameters.
    pub fn new(
        architecture: ComponentAdapter<ComponentDefinition, TransformerKernel<B>>,
        policy: P,
    ) -> Self {
        Self {
            runtime: LayerwiseRuntime::new(architecture, policy),
        }
    }
    /// Existing shared runtime for diagnostics and parameter inspection.
    pub fn runtime(
        &self,
    ) -> &LayerwiseRuntime<
        ComponentAdapter<ComponentDefinition, TransformerKernel<B>>,
        B,
        TransformerStateMarker<B>,
        P,
    > {
        &self.runtime
    }
    /// Existing shared runtime for bounded parameter operations.
    pub fn runtime_mut(
        &mut self,
    ) -> &mut LayerwiseRuntime<
        ComponentAdapter<ComponentDefinition, TransformerKernel<B>>,
        B,
        TransformerStateMarker<B>,
        P,
    > {
        &mut self.runtime
    }
}
fn runtime_error<P: std::error::Error + Send + Sync + 'static>(
    error: eredu_runtime::LayerwiseRuntimeError<TransformerError, P>,
) -> TransformerError {
    match error {
        eredu_runtime::LayerwiseRuntimeError::Architecture(error) => error,
        eredu_runtime::LayerwiseRuntimeError::State(error) => error.into(),
        eredu_runtime::LayerwiseRuntimeError::Layout(error) => error.into(),
        eredu_runtime::LayerwiseRuntimeError::Policy(error) => {
            BackendFailure::from_error(error).into()
        }
        error @ eredu_runtime::LayerwiseRuntimeError::Submission(_) => {
            BackendFailure::from_error(error).into()
        }
    }
}
impl<B, P> TransformerExecution<B> for LayerwiseTransformer<B, P>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    P: LayerwisePolicy<B, TransformerUnit<B>>,
    P::Error: std::error::Error + Send + Sync + 'static,
{
    fn config(&self) -> &TransformerConfig {
        self.runtime.architecture().kernel().config()
    }
    fn forward_encoder(
        &mut self,
        embedded: &B::Tensor,
        key_mask: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, TransformerError> {
        let mut empty = [];
        let mut state =
            TransformerRuntimeState::<B, NoTransformerCache<B>>::new(self.config(), &mut empty)?;
        self.runtime
            .forward_with_state(
                TransformerInput {
                    embedded,
                    key_mask: Some(key_mask),
                },
                &mut state,
                context,
            )
            .map_err(runtime_error)
    }
}

impl<B, P, C> CausalTransformerExecution<B, [C]> for LayerwiseTransformer<B, P>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    P: LayerwisePolicy<B, TransformerUnit<B>>,
    P::Error: std::error::Error + Send + Sync + 'static,
    C: AttentionCache<B::Tensor> + RuntimeLayerState<B>,
{
    fn causal_position(&self, caches: &[C]) -> Result<i32, TransformerError> {
        validate_causal_caches::<B, C>(self.config(), caches)
    }
    fn forward_causal(
        &mut self,
        embedded: &B::Tensor,
        caches: &mut [C],
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, TransformerError> {
        let mut state = TransformerRuntimeState::<B, C>::new(self.config(), caches)?;
        self.runtime
            .forward_with_state(
                TransformerInput {
                    embedded,
                    key_mask: None,
                },
                &mut state,
                context,
            )
            .map_err(runtime_error)
    }
}

impl<B, P> LayerwiseTransformer<B, P>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    P: LayerwisePolicy<B, TransformerUnit<B>>,
    P::Error: std::error::Error + Send + Sync + 'static,
{
    /// Runs the selected rank-local causal stack through shared bounded traversal.
    pub fn forward_causal_parallel<C: AttentionCache<B::Tensor> + RuntimeLayerState<B>>(
        &mut self,
        embedded: &B::Tensor,
        caches: &mut [C],
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, TransformerError> {
        let mut state = TransformerRuntimeState::<B, C>::for_architecture(
            self.runtime.architecture().kernel(),
            caches,
        )?;
        self.runtime
            .forward_parallel_with_state(
                TransformerInput {
                    embedded,
                    key_mask: None,
                },
                &mut state,
                parallel,
                context,
            )
            .map_err(runtime_error)
    }
    /// Runs rank-local character attention with the same padding and local masks.
    pub fn forward_encoder_parallel(
        &mut self,
        embedded: &B::Tensor,
        key_mask: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, TransformerError> {
        let mut empty = [];
        let mut state = TransformerRuntimeState::<B, NoTransformerCache<B>>::for_architecture(
            self.runtime.architecture().kernel(),
            &mut empty,
        )?;
        self.runtime
            .forward_parallel_with_state(
                TransformerInput {
                    embedded,
                    key_mask: Some(key_mask),
                },
                &mut state,
                parallel,
                context,
            )
            .map_err(runtime_error)
    }
}
