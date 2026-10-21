//! Typed target equations and request resources for the shared runtime adapter.
use super::*;
use eredu_nn::{EmbeddingOperator, RotaryPosition};
use eredu_runtime::{
    ComponentEquations, ComponentKernel, ComponentPinnedResources, LayerRuntimeState,
    LayeredForwardState, ResidentExpertProvider, RoutedComponentKernel, StaticParameterVisitor,
    StaticParameterVisitorMut, TextComponentKernel,
};

/// Original IDs plus optional already-replaced media embeddings for one target pass.
pub struct TargetInput<'a, T> {
    /// Initial media offset; continuations must agree with committed state.
    /// `None` uses the exact persisted offset (zero for a fresh text request).
    pub position_delta: Option<i32>,
    /// Original IDs before media embedding replacement; required for every pass.
    pub ids: Option<eredu_runtime::OriginalTokenIds<'a, T>>,
    /// Positive batch lanes.
    pub batch: i32,
    /// Current tokens per lane.
    pub tokens: i32,
    /// Optional prepared embeddings in `[batch,tokens,hidden]` order.
    pub embeddings: Option<&'a T>,
    /// Optional row-major padding visibility.
    pub visible: Option<eredu_runtime::TokenVisibility<'a, T>>,
    /// Text offsets or prepared media products for current tokens.
    pub rotary: Option<RotaryPosition<'a, T>>,
}
/// The full pre-collapse target value stays available to the prediction driver.
pub struct TargetForward<T> {
    /// Retained original IDs and exact transport metadata.
    pub request: RequestContext<T>,
    target_hidden: Option<T>,
}
impl<B: GroupedNeuralBackend + DistributedNeuralBackend> TargetKernel<B> {
    pub(crate) fn begin_target_input<S>(
        &mut self,
        input: TargetInput<'_, B::Tensor>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, TargetForward<B::Tensor>>, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState:
            AttentionCache<B::Tensor> + RuntimeStateComponents<B> + RuntimeAppendStreams<B>,
    {
        if self.partition_state.global_layer_offset() != 0 {
            return Err(Error::backend(
                "target input requires the first pipeline owner",
            ));
        }
        self.validate_partition_state(state, state.layout())?;
        if input.batch > self.spec.limits.qsa.batch || input.tokens > self.spec.limits.qsa.tokens {
            return Err(Error::backend(
                "target request exceeds admitted invocation bounds",
            ));
        }
        let offset = self.partition_offset(state)?;
        self.spec
            .limits
            .validate_history(offset, input.tokens)
            .map_err(Error::backend_source)?;
        let delta = super::super::position::resolve::<B, _>(
            state.layer(0).map_err(Error::backend)?,
            offset,
            input.position_delta,
            context,
        )
        .map_err(Error::backend_source)?;
        let rotary = match input.rotary {
            Some(rotary) => rotary,
            None => RotaryPosition::Offset(
                super::super::position::rotary_offset(offset, delta)
                    .map_err(Error::backend_source)?,
            ),
        };
        let request = RequestContext::new(
            input.ids,
            input.visible,
            input.batch,
            input.tokens,
            offset,
            self.spec.config.vocabulary,
            self.spec.limits.invocation_tokens,
            self.spec.config.attention.rotary.dimensions,
            Some(rotary),
            delta,
            context,
        )
        .map_err(Error::backend_source)?;
        let embeddings = match input.embeddings {
            Some(embeddings)
                if embeddings.shape()
                    == [input.batch, input.tokens, self.spec.config.hidden_size] =>
            {
                embeddings.clone()
            }
            Some(_) => {
                return Err(Error::backend(
                    "target supplied embedding geometry mismatch",
                ));
            }
            None => self
                .statics
                .embedding_mut()?
                .forward(&request.boundary().ids, context)?,
        };
        let hidden = self.spec.boundary.geometry.expand(&embeddings, context)?;
        Ok(LayeredForwardState {
            hidden,
            context: TargetForward {
                request,
                target_hidden: None,
            },
        })
    }
    /// Validates a transported request without re-embedding or reconstructing IDs
    /// from residual values. Partition construction supplies the next unit's prefix.
    pub fn resume_request(
        &self,
        hidden: B::Tensor,
        boundary: super::super::input::RequestBoundary<B::Tensor>,
        offset: i32,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, TargetForward<B::Tensor>>, Error> {
        self.spec
            .boundary
            .geometry
            .validate_streams(hidden.shape())?;
        if hidden.dim(0) > self.spec.limits.qsa.batch || hidden.dim(1) > self.spec.limits.qsa.tokens
        {
            return Err(Error::backend(
                "target boundary exceeds admitted invocation bounds",
            ));
        }
        self.spec
            .limits
            .validate_history(offset, hidden.dim(1))
            .map_err(Error::backend_source)?;
        let request = RequestContext::from_boundary(
            boundary,
            hidden.dim(0),
            hidden.dim(1),
            offset,
            self.spec.config.vocabulary,
            self.spec.limits.invocation_tokens,
            self.spec.config.attention.rotary.dimensions,
            context,
        )
        .map_err(Error::backend_source)?;
        Ok(LayeredForwardState {
            hidden,
            context: TargetForward {
                request,
                target_hidden: None,
            },
        })
    }
}
impl<B: GroupedNeuralBackend + DistributedNeuralBackend> ComponentPinnedResources<B>
    for TargetKernel<B>
{
    type Modules = BoundaryModules<B, ResidualBoundary<B>>;
    fn pinned_modules(&self) -> &Self::Modules {
        &self.statics
    }
    fn pinned_modules_mut(&mut self) -> &mut Self::Modules {
        &mut self.statics
    }
    fn visit_pinned<V: StaticParameterVisitor<B>>(&self, v: &mut V) -> Result<(), V::Error> {
        self.statics.visit(v)
    }
    fn visit_pinned_mut<V: StaticParameterVisitorMut<B>>(
        &mut self,
        v: &mut V,
    ) -> Result<(), V::Error> {
        self.statics.visit_mut(v)
    }
}
impl<B: GroupedNeuralBackend + DistributedNeuralBackend> ComponentKernel<B> for TargetKernel<B> {
    type Input<'a>
        = TargetInput<'a, B::Tensor>
    where
        Self: 'a;
    type Unit = Unit<B>;
    type ForwardContext = TargetForward<B::Tensor>;
    type RetainedContextValues<'a>
        = std::iter::Chain<std::array::IntoIter<&'a B::Tensor, 5>, std::option::Iter<'a, B::Tensor>>
    where
        Self: 'a,
        B::Tensor: 'a;
    type Error = Error;
    fn state_identity(
        &self,
        state: &eredu_runtime::PartitionState,
        topology: eredu_core::cache::PromptCacheTopology,
    ) -> Result<eredu_runtime::ModelStateIdentity, Error> {
        let global = self.definition.state_layout();
        let begin = state.global_layer_offset();
        if begin
            .checked_add(state.layout().len())
            .is_none_or(|end| end > global.len())
        {
            return Err(Error::backend(
                "target state partition outside declared units",
            ));
        }
        if &global
            .slice(begin..begin + state.layout().len())
            .map_err(Error::backend)?
            != state.layout()
        {
            return Err(Error::backend(
                "target partition state differs from declared geometry",
            ));
        }
        let fingerprint = self.state_fingerprint().to_owned();
        eredu_runtime::ModelStateIdentity::new(
            "qwen4_exp",
            "qwen4_exp_text",
            fingerprint,
            global.len(),
            begin,
            0,
            topology,
        )
        .map_err(Error::backend)
    }
    fn static_parameter_recipes(
        &self,
        source: &dyn eredu_checkpoint::store::CheckpointSource,
    ) -> Result<BTreeMap<String, eredu_checkpoint::recipe::DerivedWeightRecipe>, String> {
        crate::static_parameters::module_recipes(
            &self.statics,
            super::super::checkpoint::recipes::parameter_recipes(
                source.source_keys(),
                &self.spec.config,
                super::super::checkpoint::recipes::ParameterScope::Static,
            )?,
        )
    }
    fn prediction_target_capture(forward: &Self::ForwardContext) -> Option<&B::Tensor> {
        forward.target_hidden.as_ref()
    }
    fn build_unit(
        &self,
        address: eredu_runtime::ExecutionUnitAddress,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Unit<B>, Error> {
        self.construct_unit(address.index(), context)
    }
    fn executes_group(&self, _: usize, _: &Self::ForwardContext) -> bool {
        true
    }
    fn retained_values<'a>(
        &'a self,
        forward: &'a Self::ForwardContext,
        _: eredu_runtime::ExecutionUnitAddress,
    ) -> Self::RetainedContextValues<'a> {
        forward
            .request
            .retained()
            .chain(forward.target_hidden.iter())
    }
}
impl<B, S> ComponentEquations<B, S> for TargetKernel<B>
where
    B: GroupedNeuralBackend + DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B> + RuntimeAppendStreams<B>,
{
    fn observation_hooks(&self) -> eredu_runtime::inspection::ObservationHookSupport {
        eredu_runtime::inspection::ObservationHookSupport::internal(true, true, true)
            .with_routed_units(true)
    }
    fn begin<'a>(
        &mut self,
        input: TargetInput<'a, B::Tensor>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error> {
        self.begin_target_input(input, state, context)
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
        let mut f = self.begin(input, state, context)?;
        let mut borrowed = eredu_runtime::BorrowedActivationObserver(observer);
        f.hidden =
            ComponentInstrumentation::new("readout", &mut borrowed).apply("embedding", f.hidden)?;
        Ok(f)
    }
    fn begin_group(
        &mut self,
        group: usize,
        initial: &B::Tensor,
        dependencies: &[&B::Tensor],
        _: &mut S,
        _: &mut Self::ForwardContext,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let _ = (group, dependencies);
        Ok(initial.clone())
    }
    fn execute(
        &mut self,
        address: eredu_runtime::ExecutionUnitAddress,
        unit: &mut Unit<B>,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let index = address.index();
        self.run_equation(
            index,
            unit,
            hidden,
            state
                .layer(self.state_index(index)?)
                .map_err(Error::backend)?,
            &forward.request,
            &mut ResidentExpertProvider,
            context,
            &mut ComponentInstrumentation::disabled(),
        )
    }
    fn execute_observed<O>(
        &mut self,
        address: eredu_runtime::ExecutionUnitAddress,
        unit: &mut Unit<B>,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        self.execute_routed_observed(
            address,
            unit,
            hidden,
            state,
            forward,
            eredu_runtime::ExpertPass::Prefill,
            &mut ResidentExpertProvider,
            context,
            observer,
        )
    }
    fn complete_group(
        &mut self,
        group: usize,
        hidden: &B::Tensor,
        _: &mut S,
        forward: &mut Self::ForwardContext,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let _ = group;
        forward.target_hidden = Some(hidden.clone());
        Ok(hidden.clone())
    }
    fn finish(
        &mut self,
        hidden: &B::Tensor,
        _: &mut S,
        _: &Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.statics
            .finish_instrumented(hidden, context, &mut ComponentInstrumentation::disabled())
    }
    fn projects_final_text_position() -> bool {
        true
    }
    fn finish_text(
        &mut self,
        hidden: &B::Tensor,
        _: &mut S,
        _: &Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.statics.finish_instrumented(
            &crate::decoder::final_hidden_position(hidden, context)?,
            context,
            &mut ComponentInstrumentation::disabled(),
        )
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
        let mut b = eredu_runtime::BorrowedActivationObserver(observer);
        self.statics.finish_instrumented(
            hidden,
            context,
            &mut ComponentInstrumentation::new("readout", &mut b),
        )
    }
}
impl<B, S> RoutedComponentKernel<B, S> for TargetKernel<B>
where
    B: GroupedNeuralBackend + DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B> + RuntimeAppendStreams<B>,
{
    fn routed_unit_observations(&self) -> bool {
        true
    }
    fn routed_sparse_observations(&self) -> bool {
        true
    }
    fn execute_routed<P>(
        &mut self,
        address: eredu_runtime::ExecutionUnitAddress,
        unit: &mut Unit<B>,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut TargetForward<B::Tensor>,
        _: eredu_runtime::ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        P: ParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        let index = address.index();
        self.run_equation(
            index,
            unit,
            hidden,
            state
                .layer(self.state_index(index)?)
                .map_err(Error::backend)?,
            &forward.request,
            provider,
            context,
            &mut ComponentInstrumentation::disabled(),
        )
    }
    fn execute_routed_observed<P, O>(
        &mut self,
        address: eredu_runtime::ExecutionUnitAddress,
        unit: &mut Unit<B>,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut TargetForward<B::Tensor>,
        _: eredu_runtime::ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Error>
    where
        P: ParameterProvider<B>,
        P::Error: std::fmt::Display,
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        let index = address.index();
        let mut b = eredu_runtime::BorrowedActivationObserver(observer);
        // Lexical component observations use the original decoder layer prefix.
        let path = format!("model.layers.{}", self.spec.units[index].layer());
        self.run_equation(
            index,
            unit,
            hidden,
            state
                .layer(self.state_index(index)?)
                .map_err(Error::backend)?,
            &forward.request,
            provider,
            context,
            &mut ComponentInstrumentation::new(&path, &mut b),
        )
    }
}
impl<B: GroupedNeuralBackend + DistributedNeuralBackend> TextComponentKernel<B>
    for TargetKernel<B>
{
    fn text_input<'a>(tokens: &'a B::Tensor, mask: Option<&'a B::Tensor>) -> Self::Input<'a> {
        let (batch, sequence) = match tokens.shape() {
            [batch, sequence] => (*batch, *sequence),
            _ => (0, 0),
        };
        TargetInput {
            position_delta: None,
            ids: Some(eredu_runtime::OriginalTokenIds::Tensor(tokens)),
            batch,
            tokens: sequence,
            embeddings: None,
            visible: mask.map(eredu_runtime::TokenVisibility::Tensor),
            rotary: None,
        }
    }
    fn supports_chunked_prefill() -> bool {
        true
    }
}
