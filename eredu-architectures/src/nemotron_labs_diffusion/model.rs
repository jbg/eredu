//! One parameter catalog serving causal execution and transient block refinement.

use eredu_core::decoding::{AdapterScope, AttentionInvocation, CacheWritePolicy};
use eredu_nn::{
    AttentionCache, Error, NeuralBackend, ReadOnlyAttentionCache, ReadOnlyCache, Tensor,
};
use eredu_runtime::{
    ComponentAdapter, ComponentDefinition, ComponentEquations, ComponentKernel,
    ComponentPinnedResources, ExecutionUnitAddress, KernelParameterModules, LayerRuntimeState,
    LayeredForwardState, ParallelComponentKernel, RefinementComponentKernel, TextComponentKernel,
};

use super::FamilyConfig;
use crate::decoder::{
    self, AttentionInput, ComponentInstrumentation, Config, StaticModules, TransformerBlock,
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
    /// Canonical prefix frontier transported to the next owner.
    pub(crate) position: i32,
}

/// Complete text execution through the shared validated runtime adapter.
pub type TextModel<B> =
    ComponentAdapter<std::sync::Arc<ComponentDefinition>, TextKernel<B>, KernelParameterModules>;

/// Family invocation equations over the shared decoder's physical resources.
#[derive(Clone)]
pub struct TextKernel<
    B: NeuralBackend,
    M: decoder::DecoderStaticResources<B> = StaticModules<B>,
    G: decoder::DecoderGeometry<FamilyConfig> = decoder::CompleteDecoderGeometry<FamilyConfig>,
> {
    pub(super) definition: std::sync::Arc<ComponentDefinition>,
    pub(super) base: decoder::DecoderKernel<B, FamilyConfig, decoder::DenseBlockFactory, M, G>,
}

impl<B, M, G> eredu_nn::Parameterized<B::Tensor> for TextKernel<B, M, G>
where
    B: NeuralBackend,
    M: decoder::DecoderStaticResources<B>,
    G: decoder::DecoderGeometry<FamilyConfig>,
{
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
            self.base.body.forward_with_output_adapter_observed(
                self.base.output_norm.as_mut(),
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
            self.base.body.forward_with_output_adapter_observed(
                self.base.output_norm.as_mut(),
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

impl<B, M, G> TextKernel<B, M, G>
where
    B: NeuralBackend,
    M: decoder::DecoderStaticResources<B>,
    G: decoder::DecoderGeometry<FamilyConfig>,
{
    /// Retained semantic declaration used by execution and inspection.
    pub fn definition(&self) -> &eredu_runtime::ComponentDefinition {
        &self.definition
    }

    /// Complete family geometry shared by all phases.
    pub fn config(&self) -> &FamilyConfig {
        self.base.args()
    }
    /// Borrowed pinned parameter modules, with a singular storage owner.
    pub fn static_modules(&self) -> &M {
        self.base.static_modules()
    }
    /// Mutable binding access to that same pinned parameter catalog.
    pub fn static_modules_mut(&mut self) -> &mut M {
        self.base.static_modules_mut()
    }
    /// Borrows the selected embedding owner for composite token assembly.
    pub(super) fn embed_tokens(
        &mut self,
        tokens: &B::Tensor,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        B: eredu_nn::DistributedNeuralBackend,
    {
        let embedding = self
            .base
            .static_modules_mut()
            .decoder_resources_mut()
            .embedding
            .ok_or_else(|| Error::backend("decoder does not own input embedding"))?;
        match parallel {
            Some(parallel) => B::vocabulary_parallel_lookup(
                embedding,
                tokens,
                eredu_nn::EmbeddingLookupPolicy::Strict,
                parallel,
                context,
            ),
            None => eredu_nn::EmbeddingOperator::forward(embedding, tokens, context),
        }
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
        let position = eredu_nn::AttentionCache::offset(state.layer(0).map_err(Error::backend)?);
        let supplied = self.prepare_mask(mask, hidden.dim(1), invocation, state, context)?;
        let layout = state.layout().clone();
        let forward = self.base.begin_embedded_with_layout(
            hidden,
            supplied.as_ref(),
            state,
            &layout,
            context,
        )?;
        Ok(Self::invocation_forward(forward, invocation, position))
    }

    /// Constructs one independently addressable shared layer.
    pub fn construct_unit(
        &self,
        index: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<TextLayer<B>, Error> {
        let base = self.base.construct_block(index, context)?;
        let draft_adapter = self
            .config()
            .draft_adapter()
            .map(|adapter| {
                adapter.layer(
                    index,
                    base.body.self_attention.query_heads * self.config().head_dim(),
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
    pub(super) fn prepare_mask<S: LayerRuntimeState<B>>(
        &self,
        mask: Option<&B::Tensor>,
        length: i32,
        invocation: AttentionInvocation,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Option<B::Tensor>, Error>
    where
        S::LayerState: AttentionCache<B::Tensor>,
    {
        if let Some(mask) = mask {
            return Ok(Some(mask.clone()));
        }
        if invocation.cache_write == CacheWritePolicy::Transient {
            let frontier = state.layer(0).map_err(Error::backend)?.offset();
            return Ok(Some(B::Tensor::full_f32(
                0.0,
                &[
                    length,
                    frontier
                        .checked_add(length)
                        .ok_or_else(|| Error::backend("partition canvas visibility overflow"))?,
                ],
                context,
            )?));
        }
        Ok(None)
    }
    pub(super) fn invocation_forward(
        forward: LayeredForwardState<B::Tensor, decoder::ForwardContext<B::Tensor>>,
        invocation: AttentionInvocation,
        position: i32,
    ) -> LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>> {
        LayeredForwardState {
            hidden: forward.hidden,
            context: ForwardContext {
                mask: forward.context.visibility_mask().cloned(),
                invocation,
                position,
            },
        }
    }
    fn validate_input(input: &TextInput<'_, B::Tensor>) -> Result<(), Error> {
        if Self::invocation_input(input.tokens, input.feedback, input.mask, input.invocation)
            .is_none()
        {
            return Err(Error::backend(
                "invalid shared-backbone invocation semantics",
            ));
        }
        if input.tokens.shape().len() != 2 || input.tokens.dim(0) <= 0 || input.tokens.dim(1) <= 0 {
            return Err(Error::backend(
                "shared-backbone input requires nonempty batched token IDs",
            ));
        }
        Ok(())
    }
    pub(super) fn run_unit<S, O>(
        &mut self,
        address: ExecutionUnitAddress,
        unit: &mut TextLayer<B>,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut ForwardContext<B::Tensor>,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: ReadOnlyAttentionCache<B::Tensor>,
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        let path = &self.definition.unit(address.group(), address.index())?.path;
        let cache = state
            .layer(self.base.local_geometry().state_index(address.index())?)
            .map_err(Error::backend)?;
        let mut observer = eredu_runtime::BorrowedActivationObserver(observer);
        let mut instrumentation = ComponentInstrumentation::new(path, &mut observer);
        unit.forward(
            hidden,
            forward.mask.as_ref(),
            forward.invocation,
            cache,
            parallel,
            context,
            &mut instrumentation,
        )
    }
}

impl<B, M, G> ComponentPinnedResources<B> for TextKernel<B, M, G>
where
    B: NeuralBackend,
    M: decoder::DecoderStaticResources<B>,
    G: decoder::DecoderGeometry<FamilyConfig>,
{
    type Modules = M;
    fn pinned_modules(&self) -> &Self::Modules {
        self.base.static_modules()
    }
    fn pinned_modules_mut(&mut self) -> &mut Self::Modules {
        self.base.static_modules_mut()
    }
    fn visit_pinned<V: eredu_runtime::StaticParameterVisitor<B>>(
        &self,
        visitor: &mut V,
    ) -> Result<(), V::Error> {
        self.base.visit_pinned(visitor)
    }
    fn visit_pinned_mut<V: eredu_runtime::StaticParameterVisitorMut<B>>(
        &mut self,
        visitor: &mut V,
    ) -> Result<(), V::Error> {
        self.base.visit_pinned_mut(visitor)
    }
}
impl<B, M, G> ComponentKernel<B> for TextKernel<B, M, G>
where
    B: NeuralBackend,
    M: decoder::DecoderStaticResources<B>,
    G: decoder::DecoderGeometry<FamilyConfig>,
{
    type Input<'a> = TextInput<'a, B::Tensor>;
    type Unit = TextLayer<B>;
    type ForwardContext = ForwardContext<B::Tensor>;
    type RetainedContextValues<'a>
        = std::option::Iter<'a, B::Tensor>
    where
        B::Tensor: 'a;
    type Error = Error;
    fn state_identity(
        &self,
        state: &eredu_runtime::PartitionState,
        topology: eredu_core::cache::PromptCacheTopology,
    ) -> Result<eredu_runtime::ModelStateIdentity, Error> {
        self.base.state_identity(state, topology)
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
    fn build_unit(
        &self,
        address: ExecutionUnitAddress,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Unit, Error> {
        self.construct_unit(address.index(), context)
    }
    fn executes_group(&self, _: usize, _: &Self::ForwardContext) -> bool {
        true
    }
    fn retained_values<'a>(
        &'a self,
        forward: &'a Self::ForwardContext,
        _: ExecutionUnitAddress,
    ) -> Self::RetainedContextValues<'a> {
        forward.mask.iter()
    }
}
impl<B, S, M, G> ComponentEquations<B, S> for TextKernel<B, M, G>
where
    B: NeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: ReadOnlyAttentionCache<B::Tensor>,
    M: decoder::DecoderStaticResources<B>,
    G: decoder::DecoderGeometry<FamilyConfig>,
{
    fn observation_hooks(&self) -> eredu_runtime::inspection::ObservationHookSupport {
        eredu_runtime::inspection::ObservationHookSupport::internal(true, true, true)
    }
    fn begin<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error> {
        Self::validate_input(&input)?;
        let position = eredu_nn::AttentionCache::offset(state.layer(0).map_err(Error::backend)?);
        let mask = self.prepare_mask(
            input.mask,
            input.tokens.dim(1),
            input.invocation,
            state,
            context,
        )?;
        let expected = state.layout().clone();
        let forward = self.base.prepare_partition(
            eredu_runtime::LayeredPartitionInput::Tokens(input.tokens),
            mask.as_ref(),
            state,
            &expected,
            0,
            context,
        )?;
        Ok(Self::invocation_forward(
            forward,
            input.invocation,
            position,
        ))
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
        let mut observer = eredu_runtime::BorrowedActivationObserver(observer);
        forward.hidden = ComponentInstrumentation::new("readout", &mut observer)
            .apply("embedding", forward.hidden)?;
        Ok(forward)
    }
    fn finish(
        &mut self,
        hidden: &B::Tensor,
        _: &mut S,
        _: &Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.base.finish_hidden(hidden, context)
    }
    fn finish_observed<O>(
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
        self.base.finish_hidden_instrumented(
            hidden,
            context,
            &mut ComponentInstrumentation::new("readout", &mut observer),
        )
    }
    fn projects_final_text_position() -> bool {
        G::FINAL_TEXT_POSITION
    }
    fn finish_text(
        &mut self,
        hidden: &B::Tensor,
        _: &mut S,
        _: &Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        if !G::FINAL_TEXT_POSITION {
            return self.base.finish_hidden(hidden, context);
        }
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
    fn begin_group(
        &mut self,
        _: usize,
        initial: &B::Tensor,
        _: &[&B::Tensor],
        _: &mut S,
        _: &mut Self::ForwardContext,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        Ok(initial.clone())
    }
    fn execute(
        &mut self,
        address: ExecutionUnitAddress,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.run_unit(
            address,
            unit,
            hidden,
            state,
            forward,
            None,
            context,
            &mut eredu_runtime::NoopObserver,
        )
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
        self.run_unit(
            address, unit, hidden, state, forward, None, context, observer,
        )
    }
}
impl<B, S, M, G> ParallelComponentKernel<B, S> for TextKernel<B, M, G>
where
    B: eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: ReadOnlyAttentionCache<B::Tensor>,
    M: decoder::DecoderStaticResources<B>,
    G: decoder::DecoderGeometry<FamilyConfig>,
{
    fn parallel_observation_hooks(&self) -> eredu_runtime::inspection::ObservationHookSupport {
        <Self as ComponentEquations<B, S>>::observation_hooks(self)
    }
    fn begin_parallel<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error> {
        Self::validate_input(&input)?;
        let position = eredu_nn::AttentionCache::offset(state.layer(0).map_err(Error::backend)?);
        let mask = self.prepare_mask(
            input.mask,
            input.tokens.dim(1),
            input.invocation,
            state,
            context,
        )?;
        let expected = state.layout().clone();
        let forward = self.base.prepare_partition_parallel(
            eredu_runtime::LayeredPartitionInput::Tokens(input.tokens),
            mask.as_ref(),
            state,
            &expected,
            0,
            parallel,
            context,
        )?;
        Ok(Self::invocation_forward(
            forward,
            input.invocation,
            position,
        ))
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
        let mut observer = eredu_runtime::BorrowedActivationObserver(observer);
        forward.hidden = ComponentInstrumentation::new("readout", &mut observer)
            .apply("embedding", forward.hidden)?;
        Ok(forward)
    }
    fn finish_parallel(
        &mut self,
        hidden: &B::Tensor,
        _: &mut S,
        _: &Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.base.finish_hidden_parallel(hidden, parallel, context)
    }
    fn finish_parallel_observed<O>(
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
        self.base.finish_local_instrumented(
            hidden,
            Some(parallel),
            context,
            &mut ComponentInstrumentation::new("readout", &mut observer),
        )
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
    ) -> Result<B::Tensor, Error> {
        self.run_unit(
            address,
            unit,
            hidden,
            state,
            forward,
            Some(parallel),
            context,
            &mut eredu_runtime::NoopObserver,
        )
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
        self.run_unit(
            address,
            unit,
            hidden,
            state,
            forward,
            Some(parallel),
            context,
            observer,
        )
    }
}
impl<B, M, G> TextComponentKernel<B> for TextKernel<B, M, G>
where
    B: NeuralBackend,
    M: decoder::DecoderStaticResources<B>,
    G: decoder::DecoderGeometry<FamilyConfig>,
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
        Some(self.config().generation_geometry())
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
impl<B: NeuralBackend> RefinementComponentKernel<B> for TextKernel<B> {
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

impl<B: NeuralBackend> TextKernel<B> {
    /// Materializes the complete declared text resources.
    pub fn new(
        config: FamilyConfig,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<TextModel<B>, Error> {
        let definition = std::sync::Arc::new(super::definition::text(&config, None)?);
        let base = decoder::CompleteDecoderKernel::new(config, context)?
            .into_parts()
            .1;
        let kernel = Self {
            definition: definition.clone(),
            base,
        };
        ComponentAdapter::with_pinned_resources::<B>(definition, kernel).map_err(Error::from)
    }
    /// Materializes exact tensor-local complete text resources.
    pub fn new_parallel(
        config: FamilyConfig,
        geometry: decoder::LocalGeometry<FamilyConfig>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<TextModel<B>, Error>
    where
        B: eredu_nn::DistributedNeuralBackend,
    {
        let definition = std::sync::Arc::new(super::definition::text(&config, Some(&geometry))?);
        let base = decoder::CompleteDecoderKernel::new_parallel(config, geometry, context)?
            .into_parts()
            .1;
        let kernel = Self {
            definition: definition.clone(),
            base,
        };
        ComponentAdapter::with_pinned_resources::<B>(definition, kernel).map_err(Error::from)
    }
}
