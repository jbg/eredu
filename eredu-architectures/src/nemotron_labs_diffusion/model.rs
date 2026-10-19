//! One parameter catalog serving causal execution and transient block refinement.

use eredu_core::decoding::{AdapterScope, AttentionInvocation, CacheWritePolicy};
use eredu_nn::{Error, NeuralBackend, ReadOnlyAttentionCache, ReadOnlyCache, Tensor};
use eredu_runtime::{
    ArchitectureParameters, LayerRuntimeState, LayeredArchitecture, LayeredForwardState,
};

use super::FamilyConfig;
use crate::decoder::{
    self, AttentionInput, ComponentInstrumentation, Config, LayeredModel, StaticModules,
    TransformerBlock,
};

/// Borrowed token span with immutable invocation semantics.
#[derive(Debug, Clone, Copy)]
pub struct TextInput<'a, T> {
    /// Complete supplied token span.
    pub tokens: &'a T,
    /// Optional architecture-prepared position visibility.
    pub mask: Option<&'a T>,
    /// Request-local attention, cache and adapter policy.
    pub invocation: AttentionInvocation,
    /// Borrowed feedback is rejected because masked refinement has no feedback equation.
    pub feedback: Option<&'a T>,
}

/// Context retained for a single layered invocation, never on shared modules.
pub struct ForwardContext<T> {
    /// Complete current position visibility, including the retained prefix.
    pub mask: Option<T>,
    /// Explicit cache-write and score policy.
    pub invocation: AttentionInvocation,
}

/// Shared dense transformer whose static modules and independently addressed
/// layers are bound once and reused by every invocation.
#[derive(Clone)]
pub struct TextModel<B: NeuralBackend> {
    base: LayeredModel<B, FamilyConfig>,
}

impl<B: NeuralBackend> eredu_nn::Parameterized<B::Tensor> for TextModel<B> {
    fn set_trainable(&mut self, trainable: bool) {
        self.static_modules_mut().set_trainable(trainable);
    }
    fn visit_parameters<'a, V: eredu_nn::ParameterVisitor<'a, B::Tensor>>(
        &'a self,
        visitor: &mut V,
    ) {
        self.static_modules().visit_parameters(visitor);
    }
    fn visit_parameters_mut<'a, V: eredu_nn::ParameterVisitorMut<'a, B::Tensor>>(
        &'a mut self,
        visitor: &mut V,
    ) {
        self.static_modules_mut().visit_parameters_mut(visitor);
    }
}

/// One shared base layer and optional separately bound low-rank draft parameters.
#[derive(Debug, Clone, eredu_nn::Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct TextLayer<B: NeuralBackend> {
    /// Singular owner of base projection storage.
    pub base: TransformerBlock<B>,
    /// Draft-only delta; an invocation decides whether to borrow it.
    pub draft_adapter: Option<eredu_nn::low_rank::LowRankProjection<B>>,
}
impl<B: NeuralBackend> TextLayer<B> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward<C: ReadOnlyAttentionCache<B::Tensor>>(
        &mut self,
        hidden: &B::Tensor,
        mask: Option<&B::Tensor>,
        invocation: AttentionInvocation,
        cache: &mut C,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        let adapter = if invocation.adapters == AdapterScope::Draft {
            self.draft_adapter.as_mut()
        } else {
            None
        };
        if invocation.cache_write == CacheWritePolicy::Transient {
            let mut view = ReadOnlyCache::new(cache);
            self.base.forward_with_output_adapter_observed(
                AttentionInput {
                    hidden,
                    mask,
                    cache: Some(&mut view),
                    allow_sliding_prefill: false,
                    rotary_position: None,
                    selected_positions: None,
                },
                adapter,
                parallel,
                context,
                instrumentation,
            )
        } else {
            self.base.forward_with_output_adapter_observed(
                AttentionInput {
                    hidden,
                    mask,
                    cache: Some(cache),
                    allow_sliding_prefill: false,
                    rotary_position: None,
                    selected_positions: None,
                },
                adapter,
                parallel,
                context,
                instrumentation,
            )
        }
    }
}

impl<B: NeuralBackend> TextModel<B> {
    /// Declares one shared base without allocating any second invocation stack.
    pub fn new(
        config: FamilyConfig,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Ok(Self {
            base: LayeredModel::new(config, context)?,
        })
    }
    /// Declares the same base with exact tensor-parallel local geometry.
    pub fn new_parallel(
        config: FamilyConfig,
        geometry: decoder::LocalGeometry<FamilyConfig>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error>
    where
        B: eredu_nn::DistributedNeuralBackend,
    {
        Ok(Self {
            base: LayeredModel::new_parallel(config, geometry, context)?,
        })
    }
    /// Declares optional low-rank modules before any residency or state is built.
    pub fn new_with_draft_adapter(
        config: FamilyConfig,
        adapter: super::adapter::DraftAdapterConfig,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Self::new(config.with_draft_adapter(adapter), context)
    }
    /// Declares the same optional adapter with input-sharded down parameters.
    pub fn new_parallel_with_draft_adapter(
        config: FamilyConfig,
        geometry: decoder::LocalGeometry<FamilyConfig>,
        adapter: super::adapter::DraftAdapterConfig,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error>
    where
        B: eredu_nn::DistributedNeuralBackend,
    {
        Self::new_parallel(config.with_draft_adapter(adapter), geometry, context)
    }
    /// Complete family geometry shared by all phases.
    pub fn config(&self) -> &FamilyConfig {
        self.base.args()
    }
    /// Borrowed pinned parameter modules, with a singular storage owner.
    pub fn static_modules(&self) -> &StaticModules<B> {
        self.base.static_modules()
    }
    /// Mutable binding access to that same pinned parameter catalog.
    pub fn static_modules_mut(&mut self) -> &mut StaticModules<B> {
        self.base.static_modules_mut()
    }
    /// Starts a text invocation from already assembled media/text embeddings.
    /// Cache visibility and write authority are identical to token ingress.
    pub(crate) fn begin_embedded<S>(
        &mut self,
        hidden: B::Tensor,
        mask: Option<&B::Tensor>,
        invocation: AttentionInvocation,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: ReadOnlyAttentionCache<B::Tensor>,
    {
        if ![
            AttentionInvocation::CAUSAL,
            AttentionInvocation::DIFFUSION,
            AttentionInvocation::DRAFT,
        ]
        .contains(&invocation)
        {
            return Err(Error::backend(
                "invalid shared-backbone invocation semantics",
            ));
        }
        if hidden.shape().len() != 3
            || hidden.dim(0) <= 0
            || hidden.dim(1) <= 0
            || hidden.dim(2) != self.config().hidden_size()
        {
            return Err(Error::backend("invalid shared-backbone embedding geometry"));
        }
        let mut supplied = mask.cloned();
        if invocation.cache_write == CacheWritePolicy::Transient && supplied.is_none() {
            let offset = eredu_nn::AttentionCache::offset(state.layer(0).map_err(Error::backend)?);
            let length = hidden.dim(1);
            supplied = Some(B::Tensor::full_f32(
                0.0,
                &[
                    length,
                    offset
                        .checked_add(length)
                        .ok_or_else(|| Error::backend("block visibility overflow"))?,
                ],
                context,
            )?);
        }
        let layout = state.layout().clone();
        let forward = self.base.begin_embedded_with_layout(
            hidden,
            supplied.as_ref(),
            state,
            &layout,
            context,
        )?;
        Ok(LayeredForwardState {
            hidden: forward.hidden,
            context: ForwardContext {
                mask: forward.context.visibility_mask().cloned(),
                invocation,
            },
        })
    }

    /// Constructs one independently addressable shared layer.
    pub fn construct_unit(
        &self,
        index: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<TextLayer<B>, Error> {
        let base = self.base.construct_unit(index, context)?;
        let draft_adapter = self
            .config()
            .draft_adapter()
            .map(|adapter| {
                adapter.layer(
                    index,
                    base.self_attention.query_heads * self.config().head_dim(),
                    self.config().hidden_size(),
                    context,
                )
            })
            .transpose()?;
        Ok(TextLayer {
            base,
            draft_adapter,
        })
    }
}

impl<B: NeuralBackend> ArchitectureParameters<B> for TextModel<B> {
    type DefinitionError = Error;
    fn state_layout(&self) -> Result<eredu_runtime::StateLayout, Error> {
        self.base.state_layout()
    }
    fn state_identity(
        &self,
        state: &eredu_runtime::PartitionState,
        topology: eredu_core::cache::PromptCacheTopology,
    ) -> Result<eredu_runtime::ModelStateIdentity, Error> {
        self.base.state_identity(state, topology)
    }
    fn parameter_description(
        &self,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<eredu_runtime::ArchitectureParameterDescription, Error> {
        super::parallel::parameter_description(self.config()).map_err(Error::backend)
    }
    fn static_parameter_recipes(
        &self,
        source: &dyn eredu_checkpoint::store::CheckpointSource,
    ) -> Result<
        std::collections::BTreeMap<String, eredu_checkpoint::recipe::DerivedWeightRecipe>,
        String,
    > {
        super::checkpoint::module_source_recipes(self.base.static_modules(), source)
    }
    fn visit_static_parameters<V: eredu_runtime::StaticParameterVisitor<B>>(
        &self,
        visitor: &mut V,
    ) -> Result<(), V::Error> {
        self.base.visit_static_parameters(visitor)
    }
    fn visit_static_parameters_mut<V: eredu_runtime::StaticParameterVisitorMut<B>>(
        &mut self,
        visitor: &mut V,
    ) -> Result<(), V::Error> {
        self.base.visit_static_parameters_mut(visitor)
    }
}

impl<B, S> LayeredArchitecture<B, S> for TextModel<B>
where
    B: NeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: ReadOnlyAttentionCache<B::Tensor>,
{
    type Input<'a> = TextInput<'a, B::Tensor>;
    type StaticModules = StaticModules<B>;
    type Unit = TextLayer<B>;
    type ForwardContext = ForwardContext<B::Tensor>;
    type RetainedContextValues<'a>
        = std::option::Iter<'a, B::Tensor>
    where
        B::Tensor: 'a;
    type Error = Error;

    fn observation_hooks(&self) -> eredu_runtime::inspection::ObservationHookSupport {
        eredu_runtime::inspection::ObservationHookSupport::internal(true, true, true)
    }
    fn group_transport(&self, group: usize) -> eredu_runtime::ArchitectureGroupTransport {
        <LayeredModel<B, FamilyConfig> as LayeredArchitecture<B, S>>::group_transport(
            &self.base, group,
        )
    }
    fn primary_execution_group(&self) -> &str {
        decoder::TEXT_DECODER_EXECUTION_GROUP
    }
    fn state_partition_plan(
        &self,
        layout: &eredu_runtime::StateLayout,
    ) -> eredu_runtime::ArchitectureStatePartitionPlan {
        <LayeredModel<B, FamilyConfig> as LayeredArchitecture<B, S>>::state_partition_plan(
            &self.base, layout,
        )
    }
    fn execution_graph(&self) -> Result<eredu_runtime::ExecutionGraph, Error> {
        <LayeredModel<B, FamilyConfig> as LayeredArchitecture<B, S>>::execution_graph(&self.base)
    }
    fn group_unit_count(&self, group: usize) -> Result<usize, Error> {
        <LayeredModel<B, FamilyConfig> as LayeredArchitecture<B, S>>::group_unit_count(
            &self.base, group,
        )
    }
    fn unit_path(&self, group: usize, index: usize) -> Result<String, Error> {
        <LayeredModel<B, FamilyConfig> as LayeredArchitecture<B, S>>::unit_path(
            &self.base, group, index,
        )
    }
    fn static_modules(&self) -> &Self::StaticModules {
        self.base.static_modules()
    }
    fn static_modules_mut(&mut self) -> &mut Self::StaticModules {
        self.base.static_modules_mut()
    }
    fn build_unit(
        &self,
        group: usize,
        index: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Unit, Error> {
        <Self as LayeredArchitecture<B, S>>::unit_path(self, group, index)?;
        self.construct_unit(index, context)
    }

    fn begin_forward<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error> {
        if ![
            AttentionInvocation::CAUSAL,
            AttentionInvocation::DIFFUSION,
            AttentionInvocation::DRAFT,
        ]
        .contains(&input.invocation)
        {
            return Err(Error::backend(
                "invalid shared-backbone invocation semantics",
            ));
        }
        if input.feedback.is_some() {
            return Err(Error::backend(
                "masked refinement does not consume feedback embeddings",
            ));
        }
        if input.tokens.shape().len() != 2 || input.tokens.dim(0) <= 0 || input.tokens.dim(1) <= 0 {
            return Err(Error::backend(
                "shared-backbone input requires nonempty batched token IDs",
            ));
        }
        // Draft scope is inert when no optional adapter is installed. It never
        // changes shared projections or any ordinary causal invocation.
        let hidden = self
            .base
            .static_modules_mut()
            .embed(input.tokens, context)?;
        self.begin_embedded(hidden, input.mask, input.invocation, state, context)
    }

    fn begin_forward_observed<'a, O>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        let mut forward = self.begin_forward(input, state, context)?;
        let mut observer = eredu_runtime::BorrowedActivationObserver(observer);
        forward.hidden = ComponentInstrumentation::new("readout", &mut observer)
            .apply("embedding", forward.hidden)?;
        Ok(forward)
    }

    fn begin_execution_group(
        &mut self,
        group: usize,
        initial: &B::Tensor,
        dependencies: &[&B::Tensor],
        _: &mut S,
        _: &mut Self::ForwardContext,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        if group != 0 || !dependencies.is_empty() {
            return Err(Error::backend("invalid shared text execution group"));
        }
        Ok(initial.clone())
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
    ) -> Result<B::Tensor, Error> {
        self.forward_unit_observed(
            group,
            index,
            unit,
            hidden,
            state,
            forward,
            context,
            &mut eredu_runtime::NoopObserver,
        )
    }

    fn forward_unit_observed<O>(
        &mut self,
        group: usize,
        index: usize,
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
        let path = <Self as LayeredArchitecture<B, S>>::unit_path(self, group, index)?;
        let cache = state.layer(index).map_err(Error::backend)?;
        let mut observer = eredu_runtime::BorrowedActivationObserver(observer);
        let mut instrumentation = ComponentInstrumentation::new(&path, &mut observer);
        unit.forward(
            hidden,
            forward.mask.as_ref(),
            forward.invocation,
            cache,
            None,
            context,
            &mut instrumentation,
        )
    }

    fn finish_forward(
        &mut self,
        hidden: &B::Tensor,
        _: &mut S,
        _: &Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.base.finish_hidden(hidden, context)
    }
    fn finish_forward_observed<O>(
        &mut self,
        hidden: &B::Tensor,
        _: &mut S,
        _: &Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        let mut observer = eredu_runtime::BorrowedActivationObserver(observer);
        self.base.static_modules_mut().finish_instrumented(
            hidden,
            context,
            &mut ComponentInstrumentation::new("readout", &mut observer),
        )
    }
    fn projects_final_text_position() -> bool {
        true
    }
    fn finish_text_forward(
        &mut self,
        hidden: &B::Tensor,
        _: &mut S,
        _: &Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let position = hidden.dim(1) - 1;
        let last = hidden.index(
            &[
                eredu_nn::Index::Full,
                eredu_nn::Index::Range(position, position + 1),
                eredu_nn::Index::Full,
            ],
            context,
        )?;
        self.base.finish_hidden(&last, context)
    }
    fn retained_context_values<'a>(
        &'a self,
        forward: &'a Self::ForwardContext,
        _: usize,
        _: usize,
    ) -> Self::RetainedContextValues<'a> {
        forward.mask.iter()
    }
}

impl<B, S> eredu_runtime::ReplicatedTextArchitecture<B, S> for TextModel<B>
where
    B: NeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: ReadOnlyAttentionCache<B::Tensor>,
{
    fn supports_chunked_prefill() -> bool {
        true
    }
    fn text_input<'a>(tokens: &'a B::Tensor, mask: Option<&'a B::Tensor>) -> Self::Input<'a> {
        TextInput {
            tokens,
            mask,
            invocation: AttentionInvocation::CAUSAL,
            feedback: None,
        }
    }
    fn block_refinement_geometry(
        &self,
    ) -> Option<eredu_runtime::generation::refinement::architecture::BlockRefinementGeometry> {
        Some(<Self as eredu_runtime::generation::refinement::architecture::BlockRefinementArchitecture<B, S>>::refinement_geometry(self))
    }
    fn invocation_input<'a>(
        tokens: &'a B::Tensor,
        feedback: Option<&'a B::Tensor>,
        mask: Option<&'a B::Tensor>,
        invocation: AttentionInvocation,
    ) -> Option<Self::Input<'a>> {
        if ![
            AttentionInvocation::CAUSAL,
            AttentionInvocation::DIFFUSION,
            AttentionInvocation::DRAFT,
        ]
        .contains(&invocation)
            || feedback.is_some()
        {
            return None;
        }
        Some(TextInput {
            tokens,
            mask,
            invocation,
            feedback,
        })
    }
}

impl<B, S> eredu_runtime::ParallelLayeredArchitecture<B, S> for TextModel<B>
where
    B: NeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: ReadOnlyAttentionCache<B::Tensor>,
{
    fn parallel_observation_hooks(&self) -> eredu_runtime::inspection::ObservationHookSupport {
        <Self as LayeredArchitecture<B, S>>::observation_hooks(self)
    }
    fn begin_forward_parallel<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error> {
        if ![
            AttentionInvocation::CAUSAL,
            AttentionInvocation::DIFFUSION,
            AttentionInvocation::DRAFT,
        ]
        .contains(&input.invocation)
            || input.feedback.is_some()
        {
            return Err(Error::backend(
                "invalid shared-backbone parallel invocation semantics",
            ));
        }
        if input.tokens.shape().len() != 2 || input.tokens.dim(0) <= 0 || input.tokens.dim(1) <= 0 {
            return Err(Error::backend(
                "shared-backbone input requires nonempty batched token IDs",
            ));
        }
        let hidden = self.base.static_modules_mut().embed_parallel(
            input.tokens,
            eredu_nn::EmbeddingLookupPolicy::Strict,
            parallel,
            context,
        )?;
        self.begin_embedded(hidden, input.mask, input.invocation, state, context)
    }

    fn begin_forward_parallel_observed<'a, O>(
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
        let mut forward = self.begin_forward_parallel(input, state, parallel, context)?;
        let mut observer = eredu_runtime::BorrowedActivationObserver(observer);
        forward.hidden = ComponentInstrumentation::new("readout", &mut observer)
            .apply("embedding", forward.hidden)?;
        Ok(forward)
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
    ) -> Result<B::Tensor, Error> {
        self.forward_unit_parallel_observed(
            group,
            index,
            unit,
            hidden,
            state,
            forward,
            parallel,
            context,
            &mut eredu_runtime::NoopObserver,
        )
    }
    fn forward_unit_parallel_observed<O>(
        &mut self,
        group: usize,
        index: usize,
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
        let path = <Self as LayeredArchitecture<B, S>>::unit_path(self, group, index)?;
        let cache = state.layer(index).map_err(Error::backend)?;
        let mut observer = eredu_runtime::BorrowedActivationObserver(observer);
        let mut instrumentation = ComponentInstrumentation::new(&path, &mut observer);
        unit.forward(
            hidden,
            forward.mask.as_ref(),
            forward.invocation,
            cache,
            Some(parallel),
            context,
            &mut instrumentation,
        )
    }
    fn finish_forward_parallel(
        &mut self,
        hidden: &B::Tensor,
        _: &mut S,
        _: &Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.base.finish_hidden_parallel(hidden, parallel, context)
    }
    fn finish_forward_parallel_observed<O>(
        &mut self,
        hidden: &B::Tensor,
        _: &mut S,
        _: &Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        let mut observer = eredu_runtime::BorrowedActivationObserver(observer);
        self.base.static_modules_mut().finish_parallel_instrumented(
            hidden,
            parallel,
            context,
            &mut ComponentInstrumentation::new("readout", &mut observer),
        )
    }
}

impl<B, S> eredu_runtime::generation::refinement::architecture::BlockRefinementArchitecture<B, S>
    for TextModel<B>
where
    B: NeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: ReadOnlyAttentionCache<B::Tensor>,
{
    fn refinement_geometry(
        &self,
    ) -> eredu_runtime::generation::refinement::architecture::BlockRefinementGeometry {
        eredu_runtime::generation::refinement::architecture::BlockRefinementGeometry {
            canvas_length: self.config().block_size,
            vocabulary_size: std::num::NonZeroU32::new(self.config().vocabulary_size() as u32)
                .expect("validated vocabulary"),
        }
    }
    fn context_input<'a>(tokens: &'a B::Tensor, mask: Option<&'a B::Tensor>) -> Self::Input<'a> {
        TextInput {
            tokens,
            mask,
            invocation: AttentionInvocation::CAUSAL,
            feedback: None,
        }
    }
    fn refinement_input<'a>(
        tokens: &'a B::Tensor,
        feedback: Option<&'a B::Tensor>,
        mask: Option<&'a B::Tensor>,
    ) -> Self::Input<'a> {
        // Masked-token refinement owns no soft-embedding feedback.
        TextInput {
            tokens,
            mask,
            invocation: AttentionInvocation::DIFFUSION,
            feedback,
        }
    }
}
