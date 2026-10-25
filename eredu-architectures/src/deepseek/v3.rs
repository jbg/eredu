//! Thin DeepSeek-V3/R1 architecture policy.

use eredu_core::{cache::LayerCachePolicy, AttentionPolicy, LayerSchedule};
use eredu_nn::{
    BlockwiseAttentionBackend, CompressedAttentionCache, EmbeddingLookupPolicy, EmbeddingOperator,
    Error, GroupedGatedProductOperator, GroupedNeuralBackend, NormalizationOperator, Parameterized,
    Tensor,
};
use eredu_runtime::{
    ComponentAdapter, ComponentEquations, ComponentKernel, ComponentPartitionContext,
    ComponentPinnedResources, ExecutionUnitAddress, KernelParameterModules,
    ParallelComponentKernel, ParallelRoutedComponentKernel, PartitionComponentKernel,
    PipelineComponentKernel, RoutedComponentKernel, TextComponentKernel,
};
use eredu_runtime::{
    LayerRuntimeState, LayeredForwardState, LayeredPartitionInput, LayeredPartitionOutput,
    ModelStateIdentity, ParameterProvider, RuntimeStateComponents, StateLayout,
    StateSegmentLifetime, StateSegmentSpec,
};
use std::sync::Arc;

use crate::decoder::{DecoderStaticResources, PartitionStaticModules, StaticModuleSpec};

use super::{
    block::V3Block,
    moe::MoePolicy,
    mtp::{EmbeddedInput, ForwardMode, RetainedValues, V3PredictionLayer},
    LayerPolicy, V3Args,
};

/// Shared runtime adapter over V3 equations and their validated semantic definition.
pub type Model<B> =
    ComponentAdapter<Arc<eredu_runtime::ComponentDefinition>, V3Kernel<B>, KernelParameterModules>;

/// Declares V3 cache identity independently of concrete state storage.
pub fn state_identity(
    args: &V3Args,
    layout: &StateLayout,
    global_layer_start: usize,
    topology: eredu_core::cache::PromptCacheTopology,
) -> Result<ModelStateIdentity, Error> {
    args.validate().map_err(Error::backend)?;
    topology.validate().map_err(Error::backend)?;
    let layer_count = usize::try_from(
        args.num_hidden_layers
            .checked_add(args.num_nextn_predict_layers)
            .ok_or_else(|| Error::backend("V3 state layer count overflowed"))?,
    )
    .map_err(Error::backend)?;
    let global_layer_end = global_layer_start
        .checked_add(layout.len())
        .ok_or_else(|| Error::backend("V3 owned state range overflowed"))?;
    if global_layer_end > layer_count {
        return Err(Error::backend(format!(
            "V3 owns state layers {global_layer_start}..{global_layer_end}, outside {layer_count} layers"
        )));
    }
    eredu_runtime::ModelStateIdentity::new(
        "deepseek_v3",
        args.model_type.clone(),
        super::v3_architecture_fingerprint(args),
        layer_count,
        global_layer_start,
        0,
        topology,
    )
    .map_err(Error::backend)
}

/// One target or appended prediction execution unit.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub enum Unit<
    B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend + BlockwiseAttentionBackend,
> {
    /// Ordinary target decoder block.
    Target(V3Block<B>),
    /// One embedded prediction depth.
    Prediction(V3PredictionLayer<B>),
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend + BlockwiseAttentionBackend>
    Unit<B>
{
    /// Returns the exact routed bank specification retained by this realized unit.
    pub fn expert_bank_spec(&self) -> Option<&eredu_nn::GroupedGatedProductSpec> {
        let feed_forward = match self {
            Self::Target(block) => &block.feed_forward,
            Self::Prediction(prediction) => &prediction.decoder.feed_forward,
        };
        match feed_forward {
            super::block::V3FeedForward::Dense(_) => None,
            super::block::V3FeedForward::Routed(moe) => Some(moe.experts.spec()),
        }
    }
}

/// V3 values retained for one target-model forward.
pub struct ForwardContext<T> {
    tokens: T,
    embedded: T,
    mask: Option<T>,
    mode: ForwardMode,
    target_capture: Option<T>,
    draft_logits: Option<T>,
    draft_hidden: Option<T>,
}

impl<T> ForwardContext<T> {
    /// Borrows the token identity retained on every target pipeline rank.
    pub const fn tokens(&self) -> &T {
        &self.tokens
    }

    /// Borrows the architecture-prepared target mask.
    pub const fn mask(&self) -> Option<&T> {
        self.mask.as_ref()
    }

    /// Borrows the final target hidden state when this was a target pass.
    pub const fn target_capture(&self) -> Option<&T> {
        self.target_capture.as_ref()
    }

    /// Borrows prediction logits when this was a draft pass.
    pub const fn draft_logits(&self) -> Option<&T> {
        self.draft_logits.as_ref()
    }

    /// Borrows the hidden state emitted by the selected prediction group.
    pub const fn draft_hidden(&self) -> Option<&T> {
        self.draft_hidden.as_ref()
    }
}

/// Immutable target-pass values transported across pipeline partitions.
#[derive(Debug, Clone)]
pub struct TargetBoundary<T> {
    tokens: T,
    embedded: T,
}

/// Family-owned wire schema for immutable V3 target context.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct TargetBoundarySchema {
    hidden_size: i32,
}

impl TargetBoundarySchema {
    /// Derives the boundary schema from the normalized target configuration.
    pub const fn from_args(args: &V3Args) -> Self {
        Self {
            hidden_size: args.hidden_size,
        }
    }
}

impl<T> TargetBoundary<T> {
    /// Creates the typed target boundary in canonical wire order.
    pub const fn new(tokens: T, embedded: T) -> Self {
        Self { tokens, embedded }
    }

    /// Borrows original target token ids.
    pub const fn tokens(&self) -> &T {
        &self.tokens
    }

    /// Borrows the input-rank embeddings retained for prediction groups.
    pub const fn embedded(&self) -> &T {
        &self.embedded
    }

    /// Decomposes the boundary without cloning backend tensors.
    pub fn into_parts(self) -> (T, T) {
        (self.tokens, self.embedded)
    }
}

impl eredu_runtime::ArchitectureBoundary for TargetBoundarySchema {
    type Boundary<T> = TargetBoundary<T>;

    const IDENTITY: &'static str = "deepseek_v3.target";

    fn primary_tensor_spec(&self) -> eredu_runtime::BoundaryTensorSpec {
        eredu_runtime::BoundaryTensorSpec::primary_activation(self.hidden_size)
    }

    fn auxiliary_tensor_specs(&self) -> Vec<eredu_runtime::BoundaryTensorSpec> {
        use eredu_runtime::{BoundaryTensorDimension as Dim, BoundaryTensorDtype as Dtype};
        vec![
            eredu_runtime::BoundaryTensorSpec::new(
                "tokens",
                [Dim::Batch, Dim::Sequence],
                Dtype::Uint32,
            ),
            eredu_runtime::BoundaryTensorSpec::new(
                "embedded",
                [Dim::Batch, Dim::Sequence, Dim::Fixed(self.hidden_size)],
                Dtype::Activation,
            ),
        ]
    }

    fn encode<T>(
        &self,
        boundary: Self::Boundary<T>,
    ) -> Result<
        Vec<eredu_runtime::ArchitectureBoundaryValue<T>>,
        eredu_runtime::ArchitectureBoundaryError,
    > {
        Ok(vec![
            eredu_runtime::ArchitectureBoundaryValue::new("tokens", boundary.tokens)?,
            eredu_runtime::ArchitectureBoundaryValue::new("embedded", boundary.embedded)?,
        ])
    }

    fn decode<T>(
        &self,
        tensors: Vec<T>,
    ) -> Result<Self::Boundary<T>, eredu_runtime::ArchitectureBoundaryError> {
        eredu_runtime::validate_boundary_tensor_count(self, &tensors)?;
        let mut tensors = tensors.into_iter();
        Ok(TargetBoundary {
            tokens: tensors.next().expect("validated target tokens"),
            embedded: tensors.next().expect("validated target embeddings"),
        })
    }
}

/// Target input owned by either the first or a downstream pipeline rank.
pub enum TargetPartitionInput<'a, T> {
    /// Token ids embedded by the input-owning partition.
    Tokens(&'a T),
    /// Hidden state plus immutable input-rank context on later partitions.
    Hidden {
        /// Upstream decoder activation.
        hidden: T,
        /// Original token ids and embeddings.
        boundary: TargetBoundary<T>,
    },
}

/// Validated V3 unit construction without pinned embedding or readout resources.
/// Complete models and independently leased prediction units share this factory.
pub struct V3UnitFactory {
    args: V3Args,
    definition: Arc<eredu_runtime::ComponentDefinition>,
    parallel_geometry: Option<Arc<super::parallel::V3LocalGeometry>>,
    expert_realization: Option<crate::ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>>,
}

impl V3UnitFactory {
    /// Retains the semantic definition and validates global or selected local geometry.
    /// This step allocates no native modules.
    pub fn from_definition(
        args: V3Args,
        definition: Arc<eredu_runtime::ComponentDefinition>,
        geometry: Option<super::parallel::V3LocalGeometry>,
    ) -> Result<Self, Error> {
        let definition = select_definition(&args, definition, geometry.as_ref())?;
        Ok(Self {
            args,
            definition,
            parallel_geometry: geometry.map(Arc::new),
            expert_realization: None,
        })
    }

    fn require_operators<
        B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend + BlockwiseAttentionBackend,
    >(
        &self,
    ) -> Result<(), Error> {
        crate::operator_requirements::require::<B>(
            "DeepSeek-V3",
            if self.parallel_geometry.is_some() {
                crate::operator_requirements::DEEPSEEK_V3
                    .union(eredu_nn::NeuralOperatorCapabilities::SUM_PARALLEL)
            } else {
                crate::operator_requirements::DEEPSEEK_V3
            },
        )
    }

    /// Constructs only the addressed target or prediction resource.
    pub fn construct_unit<B>(
        &self,
        group: usize,
        index: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Unit<B>, Error>
    where
        B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend + BlockwiseAttentionBackend,
    {
        self.require_operators::<B>()?;
        self.definition.unit(group, index)?;
        let args = self
            .parallel_geometry
            .as_ref()
            .map_or(&self.args, |geometry| geometry.args());
        let owner_group = if group == 0 {
            "target".to_owned()
        } else {
            format!("mtp.{}", group - 1)
        };
        let selected_spec = self
            .expert_realization
            .as_ref()
            .and_then(|realization| realization.unit_spec(&owner_group, index))
            .cloned();
        if group == 0
            && self.expert_realization.is_some()
            && args.layer_schedule.get(index) == Some(&LayerPolicy::SparseMoe)
            && selected_spec.is_none()
        {
            return Err(Error::backend(format!(
                "V3 expert realization has no bank for {owner_group}.{index}"
            )));
        }
        let mut unit = if group == 0 {
            let block = match selected_spec.clone() {
                Some(spec) => V3Block::new_with_expert_spec(args, index, spec, context)?,
                None => V3Block::new(args, index, context)?,
            };
            Unit::Target(block)
        } else {
            Unit::Prediction(V3PredictionLayer::new(args, group - 1, context)?)
        };
        if group != 0 {
            if let Some(realization) = &self.expert_realization {
                let spec = realization.unit_spec(&owner_group, index);
                let feed_forward = match &mut unit {
                    Unit::Target(block) => &mut block.feed_forward,
                    Unit::Prediction(prediction) => &mut prediction.decoder.feed_forward,
                };
                match (feed_forward, spec) {
                    (super::block::V3FeedForward::Routed(moe), Some(spec)) => {
                        moe.experts = B::grouped_gated_product(spec.clone(), context)?;
                    }
                    (super::block::V3FeedForward::Routed(_), None) => {
                        return Err(Error::backend(format!(
                            "V3 expert realization has no bank for {owner_group}.{index}"
                        )));
                    }
                    (super::block::V3FeedForward::Dense(_), Some(_)) => {
                        return Err(Error::backend(format!(
                            "V3 expert realization names dense unit {owner_group}.{index}"
                        )));
                    }
                    (super::block::V3FeedForward::Dense(_), None) => {}
                }
            }
        }
        Ok(unit)
    }
}

/// Thin target-model architecture over the shared V3 block and generic
/// resident/layerwise execution lifecycle.
pub struct V3Kernel<
    B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend + BlockwiseAttentionBackend,
> {
    construction: V3UnitFactory,
    static_modules: PartitionStaticModules<B>,
    partition_target_start: usize,
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend + BlockwiseAttentionBackend>
    V3Kernel<B>
{
    /// Materializes ordinary modules from the cold retained declaration.
    pub fn from_definition(
        args: V3Args,
        definition: Arc<eredu_runtime::ComponentDefinition>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Self::materialize(args, definition, None, |_| Ok(None), context)
    }
    /// Materializes modules using the retained definition and authoritative local geometry.
    pub fn from_parallel_definition(
        args: V3Args,
        definition: Arc<eredu_runtime::ComponentDefinition>,
        geometry: super::parallel::V3LocalGeometry,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Self::materialize(args, definition, Some(geometry), |_| Ok(None), context)
    }
    /// Validates selected ownership before allocating its pinned resources.
    pub fn from_partition_definition<G, A>(
        args: V3Args,
        definition: Arc<eredu_runtime::ComponentDefinition>,
        geometry: super::parallel::V3LocalGeometry,
        partition: &eredu_runtime::ArchitecturePartition<G, A>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Model<B>, Error> {
        Self::materialize(
            args,
            definition,
            Some(geometry),
            |definition| {
                definition
                    .select_partition(partition)
                    .map(Some)
                    .map_err(Error::from)
            },
            context,
        )?
        .into_partition_model(partition)
    }

    fn materialize(
        args: V3Args,
        definition: Arc<eredu_runtime::ComponentDefinition>,
        geometry: Option<super::parallel::V3LocalGeometry>,
        select: impl FnOnce(
            &eredu_runtime::ComponentDefinition,
        ) -> Result<Option<eredu_runtime::ComponentSelection>, Error>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let construction = V3UnitFactory::from_definition(args, definition, geometry)?;
        construction.require_operators::<B>()?;
        let selection = select(&construction.definition)?;
        let partition_target_start = selection
            .as_ref()
            .and_then(|selection| selection.state())
            .map_or(0, |state| state.global_layer_offset());
        let static_modules = PartitionStaticModules::from_selected_spec(
            static_spec(&construction.args),
            construction.parallel_geometry.as_ref().map(|geometry| {
                (
                    geometry.embedding_range().clone(),
                    Some(geometry.output_range().clone()),
                )
            }),
            |role| {
                selection.as_ref().is_none_or(|selection| {
                    selection
                        .owns_storage(&eredu_runtime::ParameterStorageId::StaticRole(role.into()))
                })
            },
            context,
        )?;
        Ok(Self {
            construction,
            static_modules,
            partition_target_start,
        })
    }

    /// Returns the state layout matching the modules this instance builds.
    fn state_layout_impl(&self) -> Result<StateLayout, Error> {
        Ok(self.construction.definition.state_layout().clone())
    }

    /// Borrows the validated declaration retained before native construction.
    pub fn definition(&self) -> &eredu_runtime::ComponentDefinition {
        &self.construction.definition
    }

    /// Borrows the shared rank-local geometry used by unit factories.
    pub fn shared_parallel_geometry(&self) -> Option<Arc<super::parallel::V3LocalGeometry>> {
        self.construction.parallel_geometry.clone()
    }

    /// Installs the architecture-derived expert realization used by every unit factory.
    pub fn install_expert_realization(
        &mut self,
        realization: crate::ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>,
    ) {
        self.construction.expert_realization = Some(realization);
    }

    fn embedding_mut(&mut self) -> Result<&mut B::Embedding, Error> {
        self.static_modules
            .embeddings
            .as_mut()
            .ok_or_else(|| Error::backend("V3 does not own embedding storage"))
    }
    fn normalization_mut(&mut self) -> Result<&mut B::Normalization, Error> {
        self.static_modules
            .norm
            .as_mut()
            .ok_or_else(|| Error::backend("V3 does not own readout normalization"))
    }
    fn begin_input<S>(
        &mut self,
        input: EmbeddedInput<'_, B::Tensor>,
        state: &mut S,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: CompressedAttentionCache<B::Tensor>,
    {
        if parallel.is_some() && self.construction.parallel_geometry.is_none() {
            return Err(Error::backend("V3 model was not built with local geometry"));
        }
        let expected = self.state_layout_impl()?;
        if state.layout() != &expected {
            return Err(Error::backend(format!(
                "V3 runtime state layout {:?} does not match architecture layout {expected:?}",
                state.layout()
            )));
        }
        let (tokens, supplied_hidden, supplied_mask, mode) = match input {
            EmbeddedInput::Target { tokens, mask } => (tokens, None, mask, ForwardMode::Target),
            EmbeddedInput::Draft {
                tokens,
                hidden,
                depth,
            } => {
                if depth >= self.construction.definition.prediction_groups().len() {
                    return Err(Error::backend(format!(
                        "V3 prediction depth {depth} is outside {} groups",
                        self.construction.definition.prediction_groups().len()
                    )));
                }
                (tokens, Some(hidden), None, ForwardMode::Draft(depth))
            }
            EmbeddedInput::DsparkContext { .. } | EmbeddedInput::DsparkProposal { .. } => {
                return Err(Error::backend(
                    "DSpark input is only supported by DeepSeek-V4",
                ));
            }
        };
        let embedded = match parallel {
            Some(parallel) => B::vocabulary_parallel_lookup(
                self.embedding_mut()?,
                tokens,
                EmbeddingLookupPolicy::Strict,
                parallel,
                context,
            )?,
            None => self.embedding_mut()?.forward(tokens, context)?,
        };
        let hidden = supplied_hidden.cloned().unwrap_or_else(|| embedded.clone());
        let sequence = embedded.dim(1);
        let mask = if let Some(mask) = supplied_mask {
            Some(mask.clone())
        } else if matches!(mode, ForwardMode::Target) && sequence > 1 {
            let offset = state.layer(0).map_err(Error::backend)?.offset();
            Some(B::causal_mask(sequence, offset, None, context)?)
        } else {
            None
        };
        Ok(LayeredForwardState {
            hidden,
            context: ForwardContext {
                tokens: tokens.clone(),
                embedded,
                mask,
                mode,
                target_capture: None,
                draft_logits: None,
                draft_hidden: None,
            },
        })
    }

    /// Consumes the complete declaration, equations, and pinned resource aggregate.
    pub fn into_model(self) -> Result<Model<B>, Error> {
        ComponentAdapter::with_pinned_resources::<B>(self.construction.definition.clone(), self)
            .map_err(Error::from)
    }
    /// Consumes exact partition ownership, retaining canonical resources and local state origin.
    pub fn into_partition_model<G, A>(
        mut self,
        partition: &eredu_runtime::ArchitecturePartition<G, A>,
    ) -> Result<Model<B>, Error> {
        let selection = self.construction.definition.select_partition(partition)?;
        self.partition_target_start = selection
            .state()
            .map_or(0, |state| state.global_layer_offset());
        if !selection.owns_storage(&eredu_runtime::ParameterStorageId::StaticRole(
            "embedding".into(),
        )) {
            self.static_modules.embeddings = None;
        }
        if !selection.owns_storage(&eredu_runtime::ParameterStorageId::StaticRole(
            "norm".into(),
        )) {
            self.static_modules.norm = None;
        }
        if !selection.owns_storage(&eredu_runtime::ParameterStorageId::StaticRole(
            "output".into(),
        )) {
            self.static_modules.lm_head = None;
        }
        ComponentAdapter::with_partition_resources::<B, _, _>(
            self.construction.definition.clone(),
            self,
            partition,
        )
        .map_err(Error::from)
    }

    fn target_state_ordinal(&self, index: usize) -> Result<usize, Error> {
        index
            .checked_sub(self.partition_target_start)
            .ok_or_else(|| {
                Error::backend("V3 target unit precedes the selected partition state slice")
            })
    }

    /// Returns the normalized V3 arguments.
    pub const fn args(&self) -> &V3Args {
        &self.construction.args
    }

    /// Returns the prediction depth count declared by the execution graph.
    pub fn mtp_len(&self) -> usize {
        self.construction.definition.prediction_groups().len()
    }

    /// Borrows pinned modules for neutral checkpoint binding.
    pub const fn static_modules(&self) -> &PartitionStaticModules<B> {
        &self.static_modules
    }

    /// Mutably borrows pinned modules for neutral checkpoint binding.
    pub fn static_modules_mut(&mut self) -> &mut PartitionStaticModules<B> {
        &mut self.static_modules
    }

    /// Constructs one target or prediction unit from this model's authoritative
    /// global or rank-local geometry.
    pub fn construct_unit(
        &self,
        group: usize,
        index: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Unit<B>, Error> {
        self.construction.construct_unit::<B>(group, index, context)
    }

    /// Starts a replicated target partition and returns its typed immutable
    /// cross-partition boundary.
    #[allow(clippy::too_many_arguments)]
    pub fn begin_partition_target<S>(
        &mut self,
        input: TargetPartitionInput<'_, B::Tensor>,
        supplied_mask: Option<&B::Tensor>,
        state: &mut S,
        expected: &StateLayout,
        first_state_ordinal: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<
        (
            LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>,
            TargetBoundary<B::Tensor>,
        ),
        Error,
    >
    where
        S: LayerRuntimeState<B>,
        S::LayerState: RuntimeStateComponents<B>,
    {
        let (tokens, embedded, hidden) = match input {
            TargetPartitionInput::Tokens(tokens) => {
                let embedded = self.embedding_mut()?.forward(tokens, context)?;
                (tokens.clone(), embedded.clone(), embedded)
            }
            TargetPartitionInput::Hidden { hidden, boundary } => {
                let (tokens, embedded) = boundary.into_parts();
                (tokens, embedded, hidden)
            }
        };
        self.begin_partition_target_embedded(
            tokens,
            embedded,
            hidden,
            supplied_mask,
            state,
            expected,
            first_state_ordinal,
            context,
        )
    }

    /// Starts a tensor-parallel target partition with input-rank vocabulary
    /// lookup and no downstream re-embedding.
    #[allow(clippy::too_many_arguments)]
    pub fn begin_partition_target_parallel<S>(
        &mut self,
        input: TargetPartitionInput<'_, B::Tensor>,
        supplied_mask: Option<&B::Tensor>,
        state: &mut S,
        expected: &StateLayout,
        first_state_ordinal: usize,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<
        (
            LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>,
            TargetBoundary<B::Tensor>,
        ),
        Error,
    >
    where
        S: LayerRuntimeState<B>,
        S::LayerState: RuntimeStateComponents<B>,
    {
        let (tokens, embedded, hidden) = match input {
            TargetPartitionInput::Tokens(tokens) => {
                let embedded = B::vocabulary_parallel_lookup(
                    self.embedding_mut()?,
                    tokens,
                    EmbeddingLookupPolicy::Strict,
                    parallel,
                    context,
                )?;
                (tokens.clone(), embedded.clone(), embedded)
            }
            TargetPartitionInput::Hidden { hidden, boundary } => {
                let (tokens, embedded) = boundary.into_parts();
                (tokens, embedded, hidden)
            }
        };
        self.begin_partition_target_embedded(
            tokens,
            embedded,
            hidden,
            supplied_mask,
            state,
            expected,
            first_state_ordinal,
            context,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn begin_partition_target_embedded<S>(
        &self,
        tokens: B::Tensor,
        embedded: B::Tensor,
        hidden: B::Tensor,
        supplied_mask: Option<&B::Tensor>,
        state: &mut S,
        expected: &StateLayout,
        first_state_ordinal: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<
        (
            LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>,
            TargetBoundary<B::Tensor>,
        ),
        Error,
    >
    where
        S: LayerRuntimeState<B>,
        S::LayerState: RuntimeStateComponents<B>,
    {
        if state.layout() != expected {
            return Err(Error::backend(format!(
                "V3 partition state layout {:?} does not match architecture partition layout {expected:?}",
                state.layout()
            )));
        }
        let sequence = embedded.dim(1);
        let mask = if let Some(mask) = supplied_mask {
            Some(mask.clone())
        } else if sequence > 1 {
            let offset = state
                .layer(first_state_ordinal)
                .map_err(Error::backend)?
                .position();
            Some(B::causal_mask(sequence, offset, None, context)?)
        } else {
            None
        };
        let boundary = TargetBoundary::new(tokens.clone(), embedded.clone());
        Ok((
            LayeredForwardState {
                hidden,
                context: ForwardContext {
                    tokens,
                    embedded,
                    mask,
                    mode: ForwardMode::Target,
                    target_capture: None,
                    draft_logits: None,
                    draft_hidden: None,
                },
            },
            boundary,
        ))
    }

    /// Applies the shared target norm and vocabulary head on the final stage.
    pub fn pipeline_finish(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.finish_target_instrumented(
            hidden,
            None,
            context,
            &mut crate::decoder::ComponentInstrumentation::disabled(),
        )
    }

    fn finish_target_instrumented(
        &mut self,
        hidden: &B::Tensor,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut crate::decoder::ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        if self.static_modules.lm_head.is_none() {
            return Err(Error::backend("V3 does not own vocabulary output storage"));
        }
        self.static_modules.decoder_resources_mut().finish_parallel(
            hidden,
            parallel,
            context,
            instrumentation,
        )
    }

    /// Applies the final target normalization without projecting vocabulary
    /// logits, allowing a distributed composition to own a sharded head.
    pub fn pipeline_finish_hidden(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.normalization_mut()?.forward(hidden, context)
    }

    /// Applies rank-local normalization and complete vocabulary projection.
    pub fn pipeline_finish_parallel(
        &mut self,
        hidden: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.finish_target_instrumented(
            hidden,
            Some(parallel),
            context,
            &mut crate::decoder::ComponentInstrumentation::disabled(),
        )
    }

    /// Finishes the serial target partition through the architecture output boundary.
    pub fn finish_partition_target(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.pipeline_finish(hidden, context)
    }

    /// Finishes the tensor-parallel target partition through the architecture output boundary.
    pub fn finish_partition_target_parallel(
        &mut self,
        hidden: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.pipeline_finish_parallel(hidden, parallel, context)
    }

    /// Executes one sequential embedded-prediction unit owned by the final
    /// pipeline stage.
    pub fn pipeline_forward_prediction<C>(
        &mut self,
        unit: &mut Unit<B>,
        hidden: &B::Tensor,
        tokens: &B::Tensor,
        cache: &mut C,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<super::mtp::PredictionOutput<B::Tensor>, Error>
    where
        C: CompressedAttentionCache<B::Tensor>,
    {
        let embedded = self.embedding_mut()?.forward(tokens, context)?;
        match unit {
            Unit::Prediction(unit) => unit.forward(hidden, &embedded, tokens, cache, context),
            Unit::Target(_) => Err(Error::backend(
                "a V3 target unit cannot execute as an embedded predictor",
            )),
        }
    }

    /// Executes one sequential embedded-prediction unit with runtime-supplied
    /// routed experts.
    #[allow(clippy::too_many_arguments)]
    pub fn pipeline_forward_prediction_with_provider<C, P>(
        &mut self,
        unit: &mut Unit<B>,
        hidden: &B::Tensor,
        tokens: &B::Tensor,
        cache: &mut C,
        pass: eredu_runtime::ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<super::mtp::PredictionOutput<B::Tensor>, Error>
    where
        C: CompressedAttentionCache<B::Tensor>,
        P: eredu_runtime::ParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        let embedded = self.embedding_mut()?.forward(tokens, context)?;
        match unit {
            Unit::Prediction(unit) => unit
                .forward_with_provider(hidden, &embedded, tokens, cache, pass, provider, context),
            Unit::Target(_) => Err(Error::backend(
                "a V3 target unit cannot execute as an embedded predictor",
            )),
        }
    }

    /// Executes a tensor-parallel predictor with architecture-owned embedding
    /// and collective semantics.
    pub fn pipeline_forward_prediction_neutral_parallel<C>(
        &mut self,
        unit: &mut Unit<B>,
        hidden: &B::Tensor,
        tokens: &B::Tensor,
        cache: &mut C,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<super::mtp::PredictionOutput<B::Tensor>, Error>
    where
        C: CompressedAttentionCache<B::Tensor>,
    {
        let embedded = B::vocabulary_parallel_lookup(
            self.embedding_mut()?,
            tokens,
            eredu_nn::EmbeddingLookupPolicy::Strict,
            parallel,
            context,
        )?;
        match unit {
            Unit::Prediction(unit) => unit.forward_parallel(
                hidden,
                &embedded,
                tokens,
                cache,
                context,
                |value, context| B::sum_parallel(value, parallel, context),
            ),
            Unit::Target(_) => Err(Error::backend(
                "a V3 target unit cannot execute as an embedded predictor",
            )),
        }
    }

    /// Executes a tensor-parallel predictor with architecture-owned embedding,
    /// collectives, and runtime-supplied routed experts.
    #[allow(clippy::too_many_arguments)]
    pub fn pipeline_forward_prediction_neutral_parallel_with_provider<C, P>(
        &mut self,
        unit: &mut Unit<B>,
        hidden: &B::Tensor,
        tokens: &B::Tensor,
        cache: &mut C,
        pass: eredu_runtime::ExpertPass,
        provider: &mut P,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<super::mtp::PredictionOutput<B::Tensor>, Error>
    where
        C: CompressedAttentionCache<B::Tensor>,
        P: eredu_runtime::TensorParallelParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        let embedded = B::vocabulary_parallel_lookup(
            self.embedding_mut()?,
            tokens,
            eredu_nn::EmbeddingLookupPolicy::Strict,
            parallel,
            context,
        )?;
        match unit {
            Unit::Prediction(unit) => unit.forward_parallel_with_provider(
                hidden,
                &embedded,
                tokens,
                cache,
                pass,
                provider,
                context,
                |value, context| B::sum_parallel(value, parallel, context),
            ),
            Unit::Target(_) => Err(Error::backend(
                "a V3 target unit cannot execute as an embedded predictor",
            )),
        }
    }

    /// Observes one typed prediction extension, including its outer invocation
    /// boundaries. The enclosing executor owns admission, completion and rollback.
    #[allow(clippy::too_many_arguments)]
    pub fn pipeline_forward_prediction_observed<C, O>(
        &mut self,
        unit: &mut Unit<B>,
        depth: usize,
        hidden: &B::Tensor,
        tokens: &B::Tensor,
        cache: &mut C,
        pass: eredu_runtime::ExpertPass,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<super::mtp::PredictionOutput<B::Tensor>, Error>
    where
        C: CompressedAttentionCache<B::Tensor>,
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        self.pipeline_prediction_observation(
            unit,
            depth,
            hidden,
            tokens,
            parallel,
            context,
            observer,
            |unit, path, hidden, embedded, observer| match parallel {
                Some(parallel) => unit.forward_parallel_observed(
                    path,
                    hidden,
                    embedded,
                    tokens,
                    cache,
                    context,
                    observer,
                    |value, context| B::sum_parallel(value, parallel, context),
                ),
                None => unit.forward_observed_with_provider(
                    path,
                    hidden,
                    embedded,
                    tokens,
                    cache,
                    pass,
                    &mut eredu_runtime::ResidentExpertProvider,
                    context,
                    observer,
                ),
            },
        )
    }

    /// Provider-aware prediction-extension observations use the same ordinary
    /// embedding, decoder reductions and head equations as unobserved execution.
    #[allow(clippy::too_many_arguments)]
    pub fn pipeline_forward_prediction_observed_with_provider<C, P, O>(
        &mut self,
        unit: &mut Unit<B>,
        depth: usize,
        hidden: &B::Tensor,
        tokens: &B::Tensor,
        cache: &mut C,
        pass: eredu_runtime::ExpertPass,
        provider: &mut P,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<super::mtp::PredictionOutput<B::Tensor>, Error>
    where
        C: CompressedAttentionCache<B::Tensor>,
        P: eredu_runtime::TensorParallelParameterProvider<B>,
        P::Error: std::fmt::Display,
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        self.pipeline_prediction_observation(
            unit,
            depth,
            hidden,
            tokens,
            parallel,
            context,
            observer,
            |unit, path, hidden, embedded, observer| match parallel {
                Some(parallel) => unit.forward_parallel_observed_with_provider(
                    path,
                    hidden,
                    embedded,
                    tokens,
                    cache,
                    pass,
                    provider,
                    context,
                    observer,
                    |value, context| B::sum_parallel(value, parallel, context),
                ),
                None => unit.forward_observed_with_provider(
                    path, hidden, embedded, tokens, cache, pass, provider, context, observer,
                ),
            },
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn pipeline_prediction_observation<O, F>(
        &mut self,
        unit: &mut Unit<B>,
        depth: usize,
        hidden: &B::Tensor,
        tokens: &B::Tensor,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
        forward: F,
    ) -> Result<super::mtp::PredictionOutput<B::Tensor>, Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
        F: FnOnce(
            &mut super::mtp::V3PredictionLayer<B>,
            &str,
            &B::Tensor,
            &B::Tensor,
            &mut O,
        ) -> Result<super::mtp::PredictionOutput<B::Tensor>, Error>,
    {
        let Unit::Prediction(unit) = unit else {
            return Err(Error::backend(
                "a V3 target unit cannot execute as an embedded predictor",
            ));
        };
        // A prepared target can omit prediction units from its own groups; the
        // typed extension supplies the admitted depth and prediction module.
        let layer = (self.construction.args.num_hidden_layers as usize)
            .checked_add(depth)
            .ok_or_else(|| Error::backend("V3 prediction invocation index overflow"))?;
        let path = format!("model.layers.{layer}");
        let hidden =
            eredu_runtime::observe_and_intervene(observer, &format!("{path}.input"), hidden)?;
        observer.observe(&format!("mtp.{depth}.capture"), &hidden)?;
        let embedded = match parallel {
            Some(parallel) => B::vocabulary_parallel_lookup(
                self.embedding_mut()?,
                tokens,
                eredu_nn::EmbeddingLookupPolicy::Strict,
                parallel,
                context,
            )?,
            None => self.embedding_mut()?.forward(tokens, context)?,
        };
        let mut output = forward(unit, &path, &hidden, &embedded, observer)?;
        output.hidden = eredu_runtime::observe_and_intervene(
            observer,
            &format!("mtp.{depth}.output"),
            &output.hidden,
        )?;
        output.hidden = eredu_runtime::observe_and_intervene(
            observer,
            &format!("{path}.output"),
            &output.hidden,
        )?;
        Ok(output)
    }

    /// Executes one target or prediction unit with runtime-supplied experts.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_unit_with_provider<S, P>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Unit<B>,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut ForwardContext<B::Tensor>,
        pass: eredu_runtime::ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: CompressedAttentionCache<B::Tensor>,
        P: eredu_runtime::ParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        self.construction.definition.unit(group, index)?;
        match unit {
            Unit::Target(unit) if group == 0 => unit.forward_with_provider(
                hidden,
                forward.mask.as_ref(),
                Some(
                    state
                        .layer(self.target_state_ordinal(index)?)
                        .map_err(Error::backend)?,
                ),
                pass,
                provider,
                context,
            ),
            Unit::Prediction(unit) if group > 0 => {
                let layer = usize::try_from(self.construction.args.num_hidden_layers)
                    .map_err(Error::backend)?
                    + group
                    - 1;
                let output = unit.forward_with_provider(
                    hidden,
                    &forward.embedded,
                    &forward.tokens,
                    state
                        .layer(self.target_state_ordinal(layer)?)
                        .map_err(Error::backend)?,
                    pass,
                    provider,
                    context,
                )?;
                forward.draft_logits = Some(output.logits);
                Ok(output.hidden)
            }
            _ => Err(Error::backend(format!(
                "V3 execution unit does not match group {group}"
            ))),
        }
    }

    /// Executes one tensor-partitioned target or prediction unit with
    /// runtime-supplied experts. All row-parallel partials are reduced on the
    /// supplied tensor-parallel context, while routed expert execution stays
    /// delegated to `provider` (and may therefore use a distinct EP group).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_unit_parallel_with_provider<S, P>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Unit<B>,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut ForwardContext<B::Tensor>,
        pass: eredu_runtime::ExpertPass,
        parallel: &B::ParallelContext,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: CompressedAttentionCache<B::Tensor>,
        P: eredu_runtime::TensorParallelParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        self.construction.definition.unit(group, index)?;
        match unit {
            Unit::Target(unit) if group == 0 => unit.forward_parallel_with_provider(
                hidden,
                forward.mask.as_ref(),
                Some(
                    state
                        .layer(self.target_state_ordinal(index)?)
                        .map_err(Error::backend)?,
                ),
                pass,
                provider,
                context,
                |value, context| B::sum_parallel(value, parallel, context),
            ),
            Unit::Prediction(unit) if group > 0 => {
                let layer = usize::try_from(self.construction.args.num_hidden_layers)
                    .map_err(Error::backend)?
                    + group
                    - 1;
                let output = unit.forward_parallel_with_provider(
                    hidden,
                    &forward.embedded,
                    &forward.tokens,
                    state
                        .layer(self.target_state_ordinal(layer)?)
                        .map_err(Error::backend)?,
                    pass,
                    provider,
                    context,
                    |value, context| B::sum_parallel(value, parallel, context),
                )?;
                forward.draft_logits = Some(output.logits);
                Ok(output.hidden)
            }
            _ => Err(Error::backend(format!(
                "V3 parallel execution unit does not match group {group}"
            ))),
        }
    }

    /// Executes one graph unit with stable target/MTP observation and
    /// intervention points.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_unit_observed<S, O>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Unit<B>,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut ForwardContext<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: CompressedAttentionCache<B::Tensor>,
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        self.forward_unit_observed_with_provider(
            group,
            index,
            unit,
            hidden,
            state,
            forward,
            eredu_runtime::ExpertPass::Decode,
            &mut eredu_runtime::ResidentExpertProvider,
            context,
            observer,
        )
    }

    /// Executes one observed graph unit with runtime-supplied experts.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_unit_observed_with_provider<S, O, P>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Unit<B>,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut ForwardContext<B::Tensor>,
        pass: eredu_runtime::ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: CompressedAttentionCache<B::Tensor>,
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
        P: eredu_runtime::ParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        match unit {
            Unit::Target(unit) if group == 0 => unit.forward_internal_observed_with_provider(
                &format!("model.layers.{index}"),
                hidden,
                forward.mask.as_ref(),
                Some(
                    state
                        .layer(self.target_state_ordinal(index)?)
                        .map_err(Error::backend)?,
                ),
                pass,
                provider,
                context,
                observer,
            ),
            Unit::Prediction(unit) if group > 0 => {
                observer.observe(&format!("mtp.{}.capture", group - 1), hidden)?;
                let layer = usize::try_from(self.construction.args.num_hidden_layers)
                    .map_err(Error::backend)?
                    + group
                    - 1;
                let output = unit.forward_observed_with_provider(
                    &format!("model.layers.{layer}"),
                    hidden,
                    &forward.embedded,
                    &forward.tokens,
                    state
                        .layer(self.target_state_ordinal(layer)?)
                        .map_err(Error::backend)?,
                    pass,
                    provider,
                    context,
                    observer,
                )?;
                forward.draft_logits = Some(output.logits);
                eredu_runtime::observe_and_intervene(
                    observer,
                    &format!("mtp.{}.output", group - 1),
                    &output.hidden,
                )
            }
            _ => Err(Error::backend(format!(
                "V3 observed execution unit does not match group {group}"
            ))),
        }
    }
}

pub(crate) fn static_spec(args: &V3Args) -> StaticModuleSpec {
    StaticModuleSpec {
        normalization_groups: None,
        embedding_weight: "model.embed_tokens.weight".into(),
        normalization_weight: "model.norm.weight".into(),
        head_weight: "lm_head.weight".into(),
        vocabulary: args.vocab_size,
        hidden_size: args.hidden_size,
        normalization_epsilon: args.rms_norm_eps,
        normalization_offset: 0.0,
        embedding_quantization: args
            .linear_formats
            .get("model.embed_tokens.weight")
            .and_then(|format| format.weight_quantization()),
        // Published V3/R1 checkpoints keep the output head dense, including
        // otherwise block-FP8 models. An explicit selected format also covers
        // GGUF storage and admitted load-time transformations of the head.
        head_format: args
            .linear_formats
            .get("lm_head.weight")
            .copied()
            .unwrap_or(eredu_checkpoint::LinearFormat::Dense),
        tied_head: false,
    }
}

/// Builds the shared MoE assembly policy for one sparse target layer.
pub fn moe_policy(args: &V3Args, layer: usize) -> Result<MoePolicy, Error> {
    if args.layer_schedule.get(layer) != Some(&LayerPolicy::SparseMoe) {
        return Err(Error::backend(format!("V3 layer {layer} is not sparse")));
    }
    moe_policy_for_layer(args, layer)
}

/// Returns the architecture-owned routed bank specification for a target or MTP layer.
pub fn expert_bank_spec(
    args: &V3Args,
    layer: usize,
) -> Result<eredu_nn::GroupedGatedProductSpec, Error> {
    let policy = if layer < usize::try_from(args.num_hidden_layers).map_err(Error::backend)? {
        moe_policy(args, layer)?
    } else {
        prediction_moe_policy(args, layer)?
    };
    crate::deepseek::moe::expert_bank_spec(&policy)
}

pub(crate) fn localized_expert_bank_spec(
    args: &V3Args,
    layer: usize,
    expert_count: i32,
    intermediate_dimensions: i32,
) -> Result<eredu_nn::GroupedGatedProductSpec, Error> {
    expert_bank_spec(args, layer)?.with_group_geometry(expert_count, intermediate_dimensions)
}

pub(crate) fn prediction_moe_policy(args: &V3Args, layer: usize) -> Result<MoePolicy, Error> {
    let start = usize::try_from(args.num_hidden_layers).map_err(Error::backend)?;
    let end = usize::try_from(args.num_hidden_layers + args.num_nextn_predict_layers)
        .map_err(Error::backend)?;
    if !(start..end).contains(&layer) {
        return Err(Error::backend(format!(
            "V3 prediction layer {layer} is outside {start}..{end}"
        )));
    }
    moe_policy_for_layer(args, layer)
}

fn moe_policy_for_layer(args: &V3Args, layer: usize) -> Result<MoePolicy, Error> {
    let root = format!("model.layers.{layer}.mlp");
    Ok(MoePolicy {
        layer,
        hidden: args.hidden_size,
        expert_count: args.n_routed_experts,
        routes_per_token: args.num_experts_per_tok,
        expert_width: args.moe_intermediate_size,
        shared_width: args
            .moe_intermediate_size
            .checked_mul(args.n_shared_experts)
            .ok_or_else(|| Error::backend("V3 shared expert width overflowed"))?,
        scoring: eredu_nn::GroupScoring::Sigmoid,
        normalize_routes: args.norm_topk_prob,
        normalization_epsilon: 1e-20,
        routed_scaling: args.routed_scaling_factor,
        expert_groups: args.n_group,
        selected_groups: args.topk_group,
        router_weight: format!("{root}.gate.weight"),
        // Released routers stay dense unless exact selected encoding facts
        // explicitly request a different realization (including load transforms).
        router_format: args
            .linear_formats
            .get(&format!("{root}.gate.weight"))
            .copied()
            .unwrap_or(eredu_checkpoint::LinearFormat::Dense),
        correction_bias: Some(format!("{root}.gate.e_score_correction_bias")),
        expert_gate_up: format!("{root}.experts.gate_up_proj"),
        expert_down: format!("{root}.experts.down_proj"),
        shared_gate: format!("{root}.shared_experts.gate_proj.weight"),
        shared_up: format!("{root}.shared_experts.up_proj.weight"),
        shared_down: format!("{root}.shared_experts.down_proj.weight"),
        shared_gate_format: args
            .linear_format_for(&format!("{root}.shared_experts.gate_proj.weight")),
        shared_up_format: args.linear_format_for(&format!("{root}.shared_experts.up_proj.weight")),
        shared_down_format: args
            .linear_format_for(&format!("{root}.shared_experts.down_proj.weight")),
        expert_gate_up_format: args.linear_format_for(&format!("{root}.experts.gate_up_proj")),
        expert_down_format: args.linear_format_for(&format!("{root}.experts.down_proj")),
        shared_limit: None,
        limit: None,
    })
}

/// Declares compressed latent plus rotary state for target and embedded MTP
/// layers without exposing a backend cache implementation.
pub fn state_layout(args: &V3Args) -> Result<StateLayout, Error> {
    args.validate().map_err(Error::backend)?;
    let layers = usize::try_from(
        args.num_hidden_layers
            .checked_add(args.num_nextn_predict_layers)
            .ok_or_else(|| Error::backend("V3 total state layer count overflowed"))?,
    )
    .map_err(Error::backend)?;
    let policies = (0..layers)
        .map(|_| {
            LayerCachePolicy::compressed_latent_rotary(
                AttentionPolicy::Full,
                args.kv_lora_rank,
                args.qk_rope_head_dim,
            )
            .map_err(Error::backend)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let target_layers = usize::try_from(args.num_hidden_layers).map_err(Error::backend)?;
    let mut segments = vec![StateSegmentSpec::new(
        super::TARGET_STATE_SEGMENT,
        0..target_layers,
        StateSegmentLifetime::Persistent,
        0,
    )
    .map_err(Error::backend)?];
    if layers > target_layers {
        segments.push(
            StateSegmentSpec::new(
                super::PREDICTION_STATE_SEGMENT,
                target_layers..layers,
                StateSegmentLifetime::Persistent,
                -1,
            )
            .map_err(Error::backend)?,
        );
    }
    StateLayout::segmented(
        LayerSchedule::new(layers, policies).map_err(Error::backend)?,
        segments,
    )
    .map_err(Error::backend)
}

/// Validates source-free V3 invocation, storage, state and transport ownership.
pub fn component_definition(
    args: &V3Args,
    geometry: Option<&super::parallel::V3LocalGeometry>,
) -> Result<eredu_runtime::ComponentDefinition, Error> {
    args.validate().map_err(Error::backend_source)?;
    let state = component_state(args, geometry)?;
    let parameters = super::parallel::v3_parameters(args).map_err(Error::backend_source)?;
    super::component::definition(
        "deepseek_v3",
        parameters,
        state,
        crate::transport::decoder(),
        |_address, ordinal| format!("model.layers.{ordinal}"),
    )
}

fn component_state(
    args: &V3Args,
    geometry: Option<&super::parallel::V3LocalGeometry>,
) -> Result<StateLayout, Error> {
    match geometry {
        Some(geometry) => {
            geometry.validate_for(args).map_err(Error::backend_source)?;
            Ok(geometry.state_layout().clone())
        }
        None => state_layout(args),
    }
}
/// Selects executable formats while preserving logical consumers and observation seams.
pub fn project_parameters(
    args: &V3Args,
    definition: &Arc<eredu_runtime::ComponentDefinition>,
) -> Result<Arc<eredu_runtime::ComponentDefinition>, Error> {
    args.validate().map_err(Error::backend_source)?;
    super::component::project_parameters(
        "deepseek_v3",
        super::parallel::v3_parameters(args).map_err(Error::backend_source)?,
        definition,
    )
}
fn invalid_definition(
    definition: &eredu_runtime::ComponentDefinition,
    resource: impl Into<String>,
) -> Error {
    eredu_runtime::ComponentDefinitionError::Invalid {
        component: definition.id().into(),
        resource: resource.into(),
    }
    .into()
}
fn select_definition(
    args: &V3Args,
    definition: Arc<eredu_runtime::ComponentDefinition>,
    geometry: Option<&super::parallel::V3LocalGeometry>,
) -> Result<Arc<eredu_runtime::ComponentDefinition>, Error> {
    args.validate().map_err(Error::backend_source)?;
    if definition.id() != "deepseek_v3" {
        return Err(invalid_definition(
            &definition,
            "DeepSeek-V3 component identity",
        ));
    }
    definition.validate_parameter_topology(
        &super::parallel::v3_parameters(args).map_err(Error::backend_source)?,
    )?;
    let state = component_state(args, geometry)?;
    let global = state_layout(args)?;
    if definition.state_layout() != &global && definition.state_layout() != &state {
        return Err(invalid_definition(
            &definition,
            "V3 retained state differs from global or selected geometry",
        ));
    }
    if let Some((_, layout)) = definition.parameters().partition_layout() {
        let retained =
            super::parallel::v3_local_geometry(args, layout).map_err(Error::backend_source)?;
        let full = eredu_nn::VocabularyParallelRange {
            global_vocabulary: args.vocab_size as usize,
            local: 0..args.vocab_size as usize,
        };
        let selected = geometry.map_or(args, super::parallel::V3LocalGeometry::args);
        let actual = retained.args();
        if retained.state_layout() != &state
            || retained.embedding_range()
                != geometry.map_or(&full, super::parallel::V3LocalGeometry::embedding_range)
            || retained.output_range()
                != geometry.map_or(&full, super::parallel::V3LocalGeometry::output_range)
            || actual.num_attention_heads != selected.num_attention_heads
            || actual.intermediate_size != selected.intermediate_size
            || actual.moe_intermediate_size != selected.moe_intermediate_size
            || actual.q_lora_rank != selected.q_lora_rank
            || actual.kv_lora_rank != selected.kv_lora_rank
            || actual.qk_nope_head_dim != selected.qk_nope_head_dim
            || actual.qk_rope_head_dim != selected.qk_rope_head_dim
            || actual.v_head_dim != selected.v_head_dim
        {
            return Err(invalid_definition(
                &definition,
                "V3 retained placement differs from selected geometry",
            ));
        }
    }
    super::component::validate(
        &definition,
        crate::transport::decoder(),
        |_, ordinal| format!("model.layers.{ordinal}"),
        |_, _, boundaries| !boundaries.is_kernel_owned(),
    )?;
    if definition.state_layout() == &state {
        Ok(definition)
    } else {
        definition
            .as_ref()
            .clone()
            .with_state_layout(state)
            .map(Arc::new)
            .map_err(Error::from)
    }
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend + BlockwiseAttentionBackend>
    ComponentKernel<B> for V3Kernel<B>
{
    type Input<'a>
        = EmbeddedInput<'a, B::Tensor>
    where
        Self: 'a;
    type Unit = Unit<B>;
    type ForwardContext = ForwardContext<B::Tensor>;
    type RetainedContextValues<'a>
        = RetainedValues<'a, B::Tensor>
    where
        Self: 'a,
        B::Tensor: 'a;
    type Error = Error;
    fn state_identity(
        &self,
        state: &eredu_runtime::PartitionState,
        topology: eredu_core::cache::PromptCacheTopology,
    ) -> Result<ModelStateIdentity, Self::Error> {
        state_identity(
            &self.construction.args,
            state.layout(),
            state.global_layer_offset(),
            topology,
        )
    }
    fn build_unit(
        &self,
        address: ExecutionUnitAddress,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Unit, Self::Error> {
        let group = address.group();
        let index = address.index();
        self.construct_unit(group, index, context)
    }
    fn executes_group(&self, group: usize, forward: &Self::ForwardContext) -> bool {
        match forward.mode {
            ForwardMode::Target => group == 0,
            ForwardMode::Draft(depth) => group == depth + 1,
            ForwardMode::DsparkContext | ForwardMode::DsparkProposal => false,
        }
    }
    fn retained_values<'a>(
        &'a self,
        forward: &'a Self::ForwardContext,
        _address: ExecutionUnitAddress,
    ) -> Self::RetainedContextValues<'a> {
        RetainedValues::new([
            Some(&forward.tokens),
            Some(&forward.embedded),
            forward.mask.as_ref(),
            forward.target_capture.as_ref(),
            forward.draft_logits.as_ref(),
            forward.draft_hidden.as_ref(),
        ])
    }
    fn prediction_target_capture(context: &Self::ForwardContext) -> Option<&B::Tensor> {
        context.target_capture()
    }
    fn prediction_target_placeholder_shape(
        &self,
        forward: &Self::ForwardContext,
    ) -> Result<Option<Vec<i32>>, Self::Error> {
        Ok(Some(vec![
            forward.tokens().dim(0),
            forward.tokens().dim(1),
            self.construction.args.hidden_size,
        ]))
    }
}
impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend + BlockwiseAttentionBackend>
    ComponentPinnedResources<B> for V3Kernel<B>
{
    type Modules = PartitionStaticModules<B>;
    fn pinned_modules(&self) -> &Self::Modules {
        &self.static_modules
    }
    fn pinned_modules_mut(&mut self) -> &mut Self::Modules {
        &mut self.static_modules
    }
    fn visit_pinned<V: eredu_runtime::StaticParameterVisitor<B>>(
        &self,
        visitor: &mut V,
    ) -> Result<(), V::Error> {
        self.static_modules.decoder_resources().visit(visitor)
    }
    fn visit_pinned_mut<V: eredu_runtime::StaticParameterVisitorMut<B>>(
        &mut self,
        visitor: &mut V,
    ) -> Result<(), V::Error> {
        self.static_modules.decoder_resources_mut().visit(visitor)
    }
}
impl<B, S> ComponentEquations<B, S> for V3Kernel<B>
where
    B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend + BlockwiseAttentionBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: CompressedAttentionCache<B::Tensor>,
{
    fn observation_hooks(&self) -> eredu_runtime::inspection::ObservationHookSupport {
        let target_only = self.construction.definition.prediction_groups().is_empty();
        eredu_runtime::inspection::ObservationHookSupport::internal(
            target_only,
            target_only,
            target_only,
        )
    }
    fn begin<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error> {
        self.begin_input(input, state, None, context)
    }
    fn begin_observed<'a, O>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        let mut forward = self.begin(input, state, context)?;
        if matches!(forward.context.mode, ForwardMode::Target) {
            let mut borrowed = eredu_runtime::BorrowedActivationObserver(observer);
            forward.hidden =
                crate::decoder::ComponentInstrumentation::new("readout", &mut borrowed)
                    .apply("embedding", forward.hidden)?;
            forward.context.embedded = forward.hidden.clone();
        }
        Ok(forward)
    }
    fn begin_group(
        &mut self,
        group: usize,
        initial: &B::Tensor,
        dependencies: &[&B::Tensor],
        _state: &mut S,
        _forward: &mut Self::ForwardContext,
        _context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        super::component::begin_group(&self.construction.definition, group, initial, dependencies)
    }
    fn execute(
        &mut self,
        address: ExecutionUnitAddress,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        let group = address.group();
        let index = address.index();
        self.construction.definition.unit(group, index)?;
        match unit {
            Unit::Target(unit) if group == 0 => {
                let cache = state
                    .layer(self.target_state_ordinal(index)?)
                    .map_err(Error::backend)?;
                unit.forward(hidden, forward.mask.as_ref(), Some(cache), context)
            }
            Unit::Prediction(unit) if group > 0 => {
                let layer = usize::try_from(self.construction.args.num_hidden_layers)
                    .map_err(Error::backend)?
                    + group
                    - 1;
                let output = unit.forward(
                    hidden,
                    &forward.embedded,
                    &forward.tokens,
                    state
                        .layer(self.target_state_ordinal(layer)?)
                        .map_err(Error::backend)?,
                    context,
                )?;
                forward.draft_logits = Some(output.logits);
                Ok(output.hidden)
            }
            _ => Err(Error::backend(format!(
                "V3 execution unit does not match group {group}"
            ))),
        }
    }
    fn execute_observed<O>(
        &mut self,
        address: ExecutionUnitAddress,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        let group = address.group();
        let index = address.index();
        V3Kernel::forward_unit_observed(
            self, group, index, unit, hidden, state, forward, context, observer,
        )
    }
    fn complete_group(
        &mut self,
        group: usize,
        hidden: &B::Tensor,
        _state: &mut S,
        forward: &mut Self::ForwardContext,
        _context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        if group == 0 && matches!(forward.mode, ForwardMode::Target) {
            forward.target_capture = Some(hidden.clone());
        } else if matches!(forward.mode, ForwardMode::Draft(_)) {
            forward.draft_hidden = Some(hidden.clone());
        }
        Ok(hidden.clone())
    }
    fn finish(
        &mut self,
        hidden: &B::Tensor,
        _state: &mut S,
        forward: &Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        match forward.mode {
            ForwardMode::Target => self.pipeline_finish(hidden, context),
            ForwardMode::Draft(_) => forward
                .draft_logits
                .clone()
                .ok_or_else(|| Error::backend("V3 draft group produced no logits")),
            ForwardMode::DsparkContext | ForwardMode::DsparkProposal => {
                Err(Error::backend("DSpark mode reached the V3 output path"))
            }
        }
    }
    fn finish_observed<O>(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        if matches!(forward.mode, ForwardMode::Target) {
            let mut borrowed = eredu_runtime::BorrowedActivationObserver(observer);
            self.finish_target_instrumented(
                hidden,
                None,
                context,
                &mut crate::decoder::ComponentInstrumentation::new("readout", &mut borrowed),
            )
        } else {
            self.finish(hidden, state, forward, context)
        }
    }
}
impl<B, S> ParallelComponentKernel<B, S> for V3Kernel<B>
where
    B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend + BlockwiseAttentionBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: CompressedAttentionCache<B::Tensor>,
{
    fn parallel_observation_hooks(&self) -> eredu_runtime::inspection::ObservationHookSupport {
        let target_only = self.construction.definition.prediction_groups().is_empty();
        eredu_runtime::inspection::ObservationHookSupport::internal(
            target_only,
            target_only,
            target_only,
        )
    }
    fn begin_parallel<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error> {
        self.begin_input(input, state, Some(parallel), context)
    }
    fn begin_parallel_observed<'a, O>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        let mut forward = self.begin_parallel(input, state, parallel, context)?;
        if matches!(forward.context.mode, ForwardMode::Target) {
            let mut borrowed = eredu_runtime::BorrowedActivationObserver(observer);
            forward.hidden =
                crate::decoder::ComponentInstrumentation::new("readout", &mut borrowed)
                    .apply("embedding", forward.hidden)?;
            forward.context.embedded = forward.hidden.clone();
        }
        Ok(forward)
    }
    fn execute_parallel(
        &mut self,
        address: ExecutionUnitAddress,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        let group = address.group();
        let index = address.index();
        self.construction.definition.unit(group, index)?;
        match unit {
            Unit::Target(unit) if group == 0 => unit.forward_parallel(
                hidden,
                forward.mask.as_ref(),
                Some(
                    state
                        .layer(self.target_state_ordinal(index)?)
                        .map_err(Error::backend)?,
                ),
                context,
                |value, context| B::sum_parallel(value, parallel, context),
            ),
            Unit::Prediction(unit) if group > 0 => {
                let layer = usize::try_from(self.construction.args.num_hidden_layers)
                    .map_err(Error::backend)?
                    + group
                    - 1;
                let output = unit.forward_parallel(
                    hidden,
                    &forward.embedded,
                    &forward.tokens,
                    state
                        .layer(self.target_state_ordinal(layer)?)
                        .map_err(Error::backend)?,
                    context,
                    |value, context| B::sum_parallel(value, parallel, context),
                )?;
                forward.draft_logits = Some(output.logits);
                Ok(output.hidden)
            }
            _ => Err(Error::backend(format!(
                "V3 parallel execution unit does not match group {group}"
            ))),
        }
    }
    fn execute_parallel_observed<O>(
        &mut self,
        address: ExecutionUnitAddress,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        let group = address.group();
        let index = address.index();
        match unit {
            Unit::Target(unit) if group == 0 => unit.forward_parallel_observed(
                &format!("model.layers.{index}"),
                hidden,
                forward.mask.as_ref(),
                Some(
                    state
                        .layer(self.target_state_ordinal(index)?)
                        .map_err(Error::backend)?,
                ),
                context,
                observer,
                |value, context| B::sum_parallel(value, parallel, context),
            ),
            Unit::Prediction(unit) if group > 0 => {
                observer.observe(&format!("mtp.{}.capture", group - 1), hidden)?;
                let layer = usize::try_from(self.construction.args.num_hidden_layers)
                    .map_err(Error::backend)?
                    + group
                    - 1;
                let output = unit.forward_parallel_observed(
                    &format!("model.layers.{layer}"),
                    hidden,
                    &forward.embedded,
                    &forward.tokens,
                    state
                        .layer(self.target_state_ordinal(layer)?)
                        .map_err(Error::backend)?,
                    context,
                    observer,
                    |value, context| B::sum_parallel(value, parallel, context),
                )?;
                forward.draft_logits = Some(output.logits);
                eredu_runtime::observe_and_intervene(
                    observer,
                    &format!("mtp.{}.output", group - 1),
                    &output.hidden,
                )
            }
            _ => self.execute_parallel(address, unit, hidden, state, forward, parallel, context),
        }
    }
    fn finish_parallel(
        &mut self,
        hidden: &B::Tensor,
        _state: &mut S,
        forward: &Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        match forward.mode {
            ForwardMode::Target => self.pipeline_finish_parallel(hidden, parallel, context),
            ForwardMode::Draft(_) => forward
                .draft_logits
                .clone()
                .ok_or_else(|| Error::backend("V3 draft group produced no logits")),
            ForwardMode::DsparkContext | ForwardMode::DsparkProposal => {
                Err(Error::backend("DSpark mode reached the V3 output path"))
            }
        }
    }
    fn finish_parallel_observed<O>(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        if matches!(forward.mode, ForwardMode::Target) {
            let mut borrowed = eredu_runtime::BorrowedActivationObserver(observer);
            self.finish_target_instrumented(
                hidden,
                Some(parallel),
                context,
                &mut crate::decoder::ComponentInstrumentation::new("readout", &mut borrowed),
            )
        } else {
            self.finish_parallel(hidden, state, forward, parallel, context)
        }
    }
}
impl<B, S> RoutedComponentKernel<B, S> for V3Kernel<B>
where
    B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend + BlockwiseAttentionBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: CompressedAttentionCache<B::Tensor>,
{
    fn routed_unit_observations(&self) -> bool {
        self.construction.definition.prediction_groups().is_empty()
    }
    fn routed_sparse_observations(&self) -> bool {
        self.construction.definition.prediction_groups().is_empty()
    }
    fn execute_routed<P>(
        &mut self,
        address: ExecutionUnitAddress,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        pass: eredu_runtime::ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error>
    where
        P: ParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        let group = address.group();
        let index = address.index();
        V3Kernel::forward_unit_with_provider(
            self, group, index, unit, hidden, state, forward, pass, provider, context,
        )
    }
    fn execute_routed_observed<P, O>(
        &mut self,
        address: ExecutionUnitAddress,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        pass: eredu_runtime::ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Self::Error>
    where
        P: ParameterProvider<B>,
        P::Error: std::fmt::Display,
        O: eredu_runtime::ActivationObserver<B::Tensor, Self::Error> + ?Sized,
    {
        let group = address.group();
        let index = address.index();
        V3Kernel::forward_unit_observed_with_provider(
            self, group, index, unit, hidden, state, forward, pass, provider, context, observer,
        )
    }
}
impl<B, S> ParallelRoutedComponentKernel<B, S> for V3Kernel<B>
where
    B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend + BlockwiseAttentionBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: CompressedAttentionCache<B::Tensor>,
{
    fn parallel_routed_unit_observations(&self) -> bool {
        self.construction.definition.prediction_groups().is_empty()
    }
    fn parallel_routed_sparse_observations(&self) -> bool {
        self.construction.definition.prediction_groups().is_empty()
    }
    fn execute_parallel_routed<P>(
        &mut self,
        address: ExecutionUnitAddress,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        pass: eredu_runtime::ExpertPass,
        provider: &mut P,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error>
    where
        P: eredu_runtime::TensorParallelParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        let group = address.group();
        let index = address.index();
        V3Kernel::forward_unit_parallel_with_provider(
            self, group, index, unit, hidden, state, forward, pass, parallel, provider, context,
        )
    }
    fn execute_parallel_routed_observed<P, O>(
        &mut self,
        address: ExecutionUnitAddress,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        pass: eredu_runtime::ExpertPass,
        provider: &mut P,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Self::Error>
    where
        P: eredu_runtime::TensorParallelParameterProvider<B>,
        P::Error: std::fmt::Display,
        O: eredu_runtime::ActivationObserver<B::Tensor, Self::Error> + ?Sized,
    {
        let group = address.group();
        let index = address.index();
        match unit {
            Unit::Target(unit) if group == 0 => unit.forward_parallel_observed_with_provider(
                &format!("model.layers.{index}"),
                hidden,
                forward.mask.as_ref(),
                Some(
                    state
                        .layer(self.target_state_ordinal(index)?)
                        .map_err(Error::backend)?,
                ),
                pass,
                provider,
                context,
                observer,
                |value, context| B::sum_parallel(value, parallel, context),
            ),
            Unit::Prediction(unit) if group > 0 => {
                observer.observe(&format!("mtp.{}.capture", group - 1), hidden)?;
                let layer = usize::try_from(self.construction.args.num_hidden_layers)
                    .map_err(Error::backend)?
                    + group
                    - 1;
                let output = unit.forward_parallel_observed_with_provider(
                    &format!("model.layers.{layer}"),
                    hidden,
                    &forward.embedded,
                    &forward.tokens,
                    state
                        .layer(self.target_state_ordinal(layer)?)
                        .map_err(Error::backend)?,
                    pass,
                    provider,
                    context,
                    observer,
                    |value, context| B::sum_parallel(value, parallel, context),
                )?;
                forward.draft_logits = Some(output.logits);
                eredu_runtime::observe_and_intervene(
                    observer,
                    &format!("mtp.{}.output", group - 1),
                    &output.hidden,
                )
            }
            _ => V3Kernel::forward_unit_parallel_with_provider(
                self, group, index, unit, hidden, state, forward, pass, parallel, provider, context,
            ),
        }
    }
}
impl<B, S> PartitionComponentKernel<B, S> for V3Kernel<B>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::DistributedNeuralBackend
        + BlockwiseAttentionBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: CompressedAttentionCache<B::Tensor> + RuntimeStateComponents<B>,
{
    type Boundary = TargetBoundarySchema;
    fn partition_observation_hooks(
        &self,
        _tensor_parallel: bool,
    ) -> eredu_runtime::inspection::ObservationHookSupport {
        let target_only = self.construction.definition.prediction_groups().is_empty();
        eredu_runtime::inspection::ObservationHookSupport::internal(
            target_only,
            target_only,
            target_only,
        )
    }
    fn boundary_schema(&self) -> Result<Self::Boundary, Self::Error> {
        Ok(TargetBoundarySchema::from_args(self.args()))
    }
    fn finish_partition_observed<O>(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &Self::ForwardContext,
        owns_output: bool,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<LayeredPartitionOutput<B::Tensor, TargetBoundary<B::Tensor>>, Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        if !owns_output {
            return self.finish_partition(hidden, state, forward, false, parallel, context);
        }
        let mut borrowed = eredu_runtime::BorrowedActivationObserver(observer);
        let output = self.finish_target_instrumented(
            hidden,
            parallel,
            context,
            &mut crate::decoder::ComponentInstrumentation::new("readout", &mut borrowed),
        )?;
        Ok(LayeredPartitionOutput::Final {
            output,
            retained: Some(hidden.clone()),
        })
    }
    fn finish_partition(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &Self::ForwardContext,
        owns_output: bool,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredPartitionOutput<B::Tensor, TargetBoundary<B::Tensor>>, Self::Error> {
        if owns_output {
            let output = match parallel {
                Some(parallel) => {
                    self.finish_parallel(hidden, state, forward, parallel, context)?
                }
                None => self.finish(hidden, state, forward, context)?,
            };
            Ok(LayeredPartitionOutput::Final {
                output,
                retained: Some(hidden.clone()),
            })
        } else {
            Ok(LayeredPartitionOutput::Boundary {
                hidden: hidden.clone(),
                auxiliary: TargetBoundary::new(forward.tokens.clone(), forward.embedded.clone()),
            })
        }
    }
    fn begin_partition(
        &mut self,
        input: LayeredPartitionInput<'_, B::Tensor, TargetBoundary<B::Tensor>>,
        mask: Option<&B::Tensor>,
        state: &mut S,
        selected: ComponentPartitionContext<'_, B>,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error> {
        let expected = selected
            .expected
            .ok_or_else(|| Error::backend("V3 pipeline requires selected state geometry"))?;
        let input = match input {
            LayeredPartitionInput::Tokens(tokens) => TargetPartitionInput::Tokens(tokens),
            LayeredPartitionInput::Hidden { hidden, auxiliary } => TargetPartitionInput::Hidden {
                hidden,
                boundary: auxiliary,
            },
        };
        match selected.parallel {
            Some(parallel) => self.begin_partition_target_parallel(
                input,
                mask,
                state,
                expected,
                selected.first_state_ordinal,
                parallel,
                selected.native,
            ),
            None => self.begin_partition_target(
                input,
                mask,
                state,
                expected,
                selected.first_state_ordinal,
                selected.native,
            ),
        }
        .map(|(forward, _)| forward)
    }
    fn begin_partition_observed<O>(
        &mut self,
        input: LayeredPartitionInput<'_, B::Tensor, TargetBoundary<B::Tensor>>,
        mask: Option<&B::Tensor>,
        state: &mut S,
        selected: ComponentPartitionContext<'_, B>,
        observer: &mut O,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        let embedded = matches!(&input, LayeredPartitionInput::Tokens(_));
        let mut forward = self.begin_partition(input, mask, state, selected)?;
        if embedded {
            let mut borrowed = eredu_runtime::BorrowedActivationObserver(observer);
            forward.hidden =
                crate::decoder::ComponentInstrumentation::new("readout", &mut borrowed)
                    .apply("embedding", forward.hidden)?;
            forward.context.embedded = forward.hidden.clone();
        }
        Ok(forward)
    }
    fn enter_partition_group(
        &mut self,
        _group: usize,
        initial: &B::Tensor,
        _state: &mut S,
        _forward: &mut Self::ForwardContext,
        _parallel: Option<&B::ParallelContext>,
        _context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        Ok(initial.clone())
    }
}
impl<B, S> PipelineComponentKernel<B, S> for V3Kernel<B>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::DistributedNeuralBackend
        + BlockwiseAttentionBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: CompressedAttentionCache<B::Tensor> + RuntimeStateComponents<B>,
{
    fn input_dimensions(&self, input: &Self::Input<'_>) -> Result<(i32, i32), Self::Error> {
        match input {
            EmbeddedInput::Target { tokens, .. } => Ok((tokens.dim(0), tokens.dim(1))),
            _ => Err(eredu_nn::Error::backend(
                "target partition received prediction input",
            )),
        }
    }
    fn tensor_vocabulary_sharded(&self) -> bool {
        true
    }
    fn output_width(&self) -> i32 {
        self.construction.args.vocab_size
    }
    fn begin_pipeline<'a, O>(
        &mut self,
        input: Self::Input<'a>,
        incoming: Option<(B::Tensor, TargetBoundary<B::Tensor>)>,
        state: &mut S,
        selected: ComponentPartitionContext<'_, B>,
        observer: &mut O,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        let (tokens, mask) = match input {
            EmbeddedInput::Target { tokens, mask } => (tokens, mask),
            _ => return Err(Error::backend("target partition received prediction input")),
        };
        let input = match incoming {
            Some((hidden, auxiliary)) => LayeredPartitionInput::Hidden { hidden, auxiliary },
            None => LayeredPartitionInput::Tokens(tokens),
        };
        self.begin_partition_observed(input, mask, state, selected, observer)
    }
}
impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend + BlockwiseAttentionBackend>
    TextComponentKernel<B> for V3Kernel<B>
{
    fn text_input<'a>(tokens: &'a B::Tensor, mask: Option<&'a B::Tensor>) -> Self::Input<'a> {
        EmbeddedInput::Target { tokens, mask }
    }
}
