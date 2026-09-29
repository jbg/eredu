//! Shared traversal adapter; no family-specific generation or residency engine.
use super::*;
use eredu_nn::{Parameterized, RotaryPosition};
use eredu_runtime::{
    module_parameter_group, ArchitectureParameterDescription, ArchitectureParameters,
    ExecutionUnitLayout, LayerRuntimeState, LayeredArchitecture, LayeredForwardState,
    MemberSharding, OwnedParameterGroupSpec, ParameterGroupOwner, ParameterRole,
    ResidentExpertProvider, RoutedLayeredArchitecture, StaticParameterVisitor,
    StaticParameterVisitorMut,
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
impl<B: GroupedNeuralBackend + DistributedNeuralBackend> TargetModel<B> {
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
        let hidden = match input.embeddings {
            Some(embeddings)
                if embeddings.shape()
                    == [input.batch, input.tokens, self.spec.config.hidden_size] =>
            {
                self.decoder
                    .static_modules()
                    .expand_embeddings(embeddings, context)?
            }
            Some(_) => {
                return Err(Error::backend(
                    "target supplied embedding geometry mismatch",
                ));
            }
            None => self
                .decoder
                .static_modules_mut()
                .embed(&request.boundary().ids, context)?,
        };
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
impl<B: GroupedNeuralBackend + DistributedNeuralBackend> ArchitectureParameters<B>
    for TargetModel<B>
{
    type DefinitionError = Error;
    fn state_layout(&self) -> Result<StateLayout, Error> {
        self.spec.state_layout()
    }
    fn state_identity(
        &self,
        state: &eredu_runtime::PartitionState,
        topology: eredu_core::cache::PromptCacheTopology,
    ) -> Result<eredu_runtime::ModelStateIdentity, Error> {
        let global = self.spec.state_layout()?;
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
    fn parameter_description(
        &self,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<ArchitectureParameterDescription, Error> {
        if let Some(parameters) = &self.global_parameters {
            return Ok((**parameters).clone());
        }
        let graph = self.decoder.execution_graph()?;
        let layout =
            ExecutionUnitLayout::new(&graph, [self.spec.units.len()]).map_err(Error::backend)?;
        let mut expected = Vec::new();
        let mut owned = Vec::new();
        fn add<T: Tensor, M: Parameterized<T>>(
            name: &str,
            role: ParameterRole,
            owner: ParameterGroupOwner,
            module: &M,
            expected: &mut Vec<eredu_runtime::ParameterGroupSpec>,
            owned: &mut Vec<OwnedParameterGroupSpec>,
        ) -> Result<(), Error> {
            let group =
                module_parameter_group(name, role, module, |_, _| Ok(MemberSharding::Replicated))
                    .map_err(Error::backend)?;
            expected.push(group.clone());
            owned.push(OwnedParameterGroupSpec::new(owner, group));
            Ok(())
        }
        let modules = self.decoder.static_modules();
        add(
            "embedding",
            ParameterRole::Vocabulary,
            if self.spec.head.is_none() {
                ParameterGroupOwner::static_any_of(["embedding", "output"])
            } else {
                ParameterGroupOwner::static_role("embedding")
            },
            &modules.embeddings,
            &mut expected,
            &mut owned,
        )?;
        add(
            "readout",
            ParameterRole::Replicated,
            ParameterGroupOwner::static_role("norm"),
            &modules.boundary,
            &mut expected,
            &mut owned,
        )?;
        if let Some(head) = &modules.lm_head {
            add(
                "output",
                ParameterRole::Vocabulary,
                ParameterGroupOwner::static_role("output"),
                head,
                &mut expected,
                &mut owned,
            )?;
        }
        for (index, spec) in self.spec.units.iter().enumerate() {
            add(
                &spec.path(),
                ParameterRole::Replicated,
                ParameterGroupOwner::execution_unit(layout.group_id(0).unwrap().clone(), index),
                &self.construct_unit(index, context)?,
                &mut expected,
                &mut owned,
            )?;
        }
        ArchitectureParameterDescription::new(&graph, &layout, expected, owned)
            .map_err(Error::backend)
    }
    fn static_parameter_recipes(
        &self,
        source: &dyn eredu_checkpoint::store::CheckpointSource,
    ) -> Result<BTreeMap<String, eredu_checkpoint::recipe::DerivedWeightRecipe>, String> {
        crate::static_parameters::module_recipes(
            self.decoder.static_modules(),
            super::super::checkpoint::recipes::parameter_recipes(
                source.source_keys(),
                &self.spec.config,
                super::super::checkpoint::recipes::ParameterScope::Static,
            )?,
        )
    }
    fn visit_static_parameters<V: StaticParameterVisitor<B>>(
        &self,
        v: &mut V,
    ) -> Result<(), V::Error> {
        let m = self.decoder.static_modules();
        v.visit("embedding", &m.embeddings)?;
        v.visit("norm", &m.boundary)?;
        if let Some(head) = &m.lm_head {
            v.visit("output", head)?;
        }
        Ok(())
    }
    fn visit_static_parameters_mut<V: StaticParameterVisitorMut<B>>(
        &mut self,
        v: &mut V,
    ) -> Result<(), V::Error> {
        let m = self.decoder.static_modules_mut();
        v.visit_mut("embedding", &mut m.embeddings)?;
        v.visit_mut("norm", &mut m.boundary)?;
        if let Some(head) = &mut m.lm_head {
            v.visit_mut("output", head)?;
        }
        Ok(())
    }
}
impl<B, S> LayeredArchitecture<B, S> for TargetModel<B>
where
    B: GroupedNeuralBackend + DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B> + RuntimeAppendStreams<B>,
{
    type Input<'a> = TargetInput<'a, B::Tensor>;
    type StaticModules = StaticModules<B, ResidualBoundary<B>>;
    type Unit = Unit<B>;
    type ForwardContext = TargetForward<B::Tensor>;
    type RetainedContextValues<'a>
        = std::iter::Chain<std::array::IntoIter<&'a B::Tensor, 5>, std::option::Iter<'a, B::Tensor>>
    where
        B::Tensor: 'a;
    type Error = Error;
    fn observation_hooks(&self) -> eredu_runtime::inspection::ObservationHookSupport {
        eredu_runtime::inspection::ObservationHookSupport::internal(true, true, true)
            .with_routed_units(true)
    }
    fn group_transport(&self, _: usize) -> eredu_runtime::ArchitectureGroupTransport {
        crate::transport::decoder()
    }
    fn primary_execution_group(&self) -> &str {
        crate::decoder::TARGET_EXECUTION_GROUP
    }
    fn state_partition_plan(
        &self,
        layout: &StateLayout,
    ) -> eredu_runtime::ArchitectureStatePartitionPlan {
        crate::transport::pipeline_state(0, layout)
    }
    fn execution_graph(&self) -> Result<eredu_runtime::ExecutionGraph, Error> {
        self.decoder.execution_graph()
    }
    fn group_unit_count(&self, group: usize) -> Result<usize, Error> {
        self.decoder.group_unit_count(group)
    }
    fn unit_path(&self, group: usize, index: usize) -> Result<String, Error> {
        self.decoder.unit_path(group, index)?;
        Ok(self.spec.units[index].path())
    }
    fn static_modules(&self) -> &Self::StaticModules {
        self.decoder.static_modules()
    }
    fn static_modules_mut(&mut self) -> &mut Self::StaticModules {
        self.decoder.static_modules_mut()
    }
    fn build_unit(
        &self,
        group: usize,
        index: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Unit<B>, Error> {
        self.decoder.unit_path(group, index)?;
        self.construct_unit(index, context)
    }
    fn begin_forward<'a>(
        &mut self,
        input: TargetInput<'a, B::Tensor>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error> {
        self.begin_target_input(input, state, context)
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
        let mut f = self.begin_forward(input, state, context)?;
        let mut borrowed = eredu_runtime::BorrowedActivationObserver(observer);
        f.hidden =
            ComponentInstrumentation::new("readout", &mut borrowed).apply("embedding", f.hidden)?;
        Ok(f)
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
        self.decoder.begin_group(group, initial, dependencies)
    }
    fn forward_unit(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Unit<B>,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.decoder.unit_path(group, index)?;
        self.execute(
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
    fn forward_unit_observed<O>(
        &mut self,
        group: usize,
        index: usize,
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
        self.forward_unit_observed_with_provider(
            group,
            index,
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
    fn complete_execution_group(
        &mut self,
        group: usize,
        hidden: &B::Tensor,
        _: &mut S,
        forward: &mut Self::ForwardContext,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.decoder.group_unit_count(group)?;
        forward.target_hidden = Some(hidden.clone());
        Ok(hidden.clone())
    }
    fn prediction_target_capture(forward: &Self::ForwardContext) -> Option<&B::Tensor> {
        forward.target_hidden.as_ref()
    }
    fn finish_forward(
        &mut self,
        hidden: &B::Tensor,
        _: &mut S,
        _: &Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.decoder.finish_logits(hidden, context)
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
        self.decoder.finish_text_logits(hidden, context)
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
        let mut b = eredu_runtime::BorrowedActivationObserver(observer);
        self.decoder.finish_logits_instrumented(
            hidden,
            context,
            &mut ComponentInstrumentation::new("readout", &mut b),
        )
    }
    fn retained_context_values<'a>(
        &'a self,
        forward: &'a Self::ForwardContext,
        _: usize,
        _: usize,
    ) -> Self::RetainedContextValues<'a> {
        forward
            .request
            .retained()
            .chain(forward.target_hidden.iter())
    }
}
impl<B, S> RoutedLayeredArchitecture<B, S> for TargetModel<B>
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
    fn forward_unit_with_provider<P>(
        &mut self,
        group: usize,
        index: usize,
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
        self.decoder.unit_path(group, index)?;
        self.execute(
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
    fn forward_unit_observed_with_provider<P, O>(
        &mut self,
        group: usize,
        index: usize,
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
        self.decoder.unit_path(group, index)?;
        let mut b = eredu_runtime::BorrowedActivationObserver(observer);
        // Lexical component observations use the original decoder layer prefix.
        let path = format!("model.layers.{}", self.spec.units[index].layer());
        self.execute(
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

impl<B, S> eredu_runtime::ReplicatedTextArchitecture<B, S> for TargetModel<B>
where
    B: GroupedNeuralBackend + DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B> + RuntimeAppendStreams<B>,
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
