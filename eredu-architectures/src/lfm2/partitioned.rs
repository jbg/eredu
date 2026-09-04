//! Exact resident TP/PP construction for dense prediction-free LFM2.

use eredu_nn::{
    AttentionCache, EmbeddingLookupPolicy, EmbeddingOperator, EmbeddingSpec, Error, LinearOperator,
    LinearSpec, NormalizationConstructionSpec, NormalizationOperator, ParameterSpec, Tensor,
};
use eredu_runtime::{
    ArchitectureParameterDescription, ExecutionGraph, LayerRuntimeState, LayeredArchitecture,
    LayeredForwardState, LayeredPartitionInput, ParallelLayeredArchitecture,
    ParallelRoutedLayeredArchitecture, PartitionedLayeredArchitecture, RoutedExpertProvider,
    RoutedLayeredArchitecture, RuntimeStateComponents, StateLayout,
};

use super::{ForwardContext, ModelArgs, PartitionLocalGeometry};
use crate::decoder::{LayeredInput, PartitionStaticModules};

/// Dense LFM2 whose units and pinned modules are limited to one admitted rank.
pub struct PartitionedLayeredModel<
    B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
> {
    args: ModelArgs,
    static_modules: PartitionStaticModules<B>,
    geometry: PartitionLocalGeometry,
    parameters: ArchitectureParameterDescription,
}

fn partition_static_modules<B, A>(
    args: &ModelArgs,
    geometry: &PartitionLocalGeometry,
    partition: &eredu_runtime::ArchitecturePartition<PartitionLocalGeometry, A>,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<PartitionStaticModules<B>, Error>
where
    B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
{
    let ownership = partition.ownership();
    let embedding_name = "model.embed_tokens.weight";
    let embeddings = (ownership.owns_input()
        || (ownership.owns_output() && args.tie_word_embeddings))
        .then(|| {
            B::vocabulary_parallel_embedding(
                EmbeddingSpec {
                    vocabulary: args.vocab_size,
                    dimensions: args.hidden_size,
                    weight: ParameterSpec::trainable(embedding_name).map_err(Error::backend)?,
                    format: crate::linear_format::standard_linear_format(
                        embedding_name,
                        args.weight_quantization_for(embedding_name).into(),
                    )?,
                },
                geometry.embedding_range().clone(),
                context,
            )
        })
        .transpose()?;
    let norm = ownership
        .owns_output()
        .then(|| {
            B::normalization(
                NormalizationConstructionSpec::learned(
                    args.hidden_size,
                    args.norm_eps,
                    ParameterSpec::trainable("model.embedding_norm.weight")
                        .map_err(Error::backend)?,
                ),
                context,
            )
        })
        .transpose()?;
    let lm_head = (ownership.owns_output() && !args.tie_word_embeddings)
        .then(|| {
            let name = "lm_head.weight";
            B::vocabulary_parallel_linear(
                LinearSpec {
                    input: args.hidden_size,
                    output: args.vocab_size,
                    weight: ParameterSpec::trainable(name).map_err(Error::backend)?,
                    bias: None,
                    format: crate::linear_format::standard_linear_format(
                        name,
                        args.weight_quantization_for(name).into(),
                    )?,
                },
                geometry.output_range().cloned().ok_or_else(|| {
                    Error::backend("untied LFM2 output owner has no vocabulary range")
                })?,
                context,
            )
        })
        .transpose()?;
    Ok(PartitionStaticModules {
        embeddings,
        norm,
        lm_head,
    })
}

impl<B> PartitionedLayeredModel<B>
where
    B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
{
    /// Constructs only modules already selected by one exact dense LFM2 partition.
    pub fn from_partition<A>(
        args: ModelArgs,
        parameters: &ArchitectureParameterDescription,
        partition: &eredu_runtime::ArchitecturePartition<PartitionLocalGeometry, A>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        args.validate().map_err(Error::backend)?;
        if parameters.graph() != partition.graph()
            || parameters.unit_layout() != partition.unit_layout()
        {
            return Err(Error::backend(
                "LFM2 partition belongs to a different parameter topology",
            ));
        }
        partition
            .local_geometry()
            .validate_for(&args)
            .map_err(Error::backend)?;
        let [group] = partition.groups() else {
            return Err(Error::backend(
                "partitioned LFM2 must own exactly one execution group",
            ));
        };
        let owned = partition.local_geometry().owned_units();
        if group.group().as_str() != crate::decoder::TARGET_EXECUTION_GROUP
            || group.global_units() != owned
        {
            return Err(Error::backend(
                "LFM2 construction range differs from selected partition",
            ));
        }
        let state = partition
            .state()
            .ok_or_else(|| Error::backend("partitioned LFM2 has no selected state"))?;
        let expected_state = partition
            .local_geometry()
            .complete_state_layout()
            .slice(owned.clone())
            .map_err(Error::backend)?;
        if state.global_layer_offset() != owned.start || state.layout() != &expected_state {
            return Err(Error::backend(
                "LFM2 partition state does not match its global unit range",
            ));
        }
        let geometry = partition.local_geometry().clone();
        let static_modules = partition_static_modules(&args, &geometry, partition, context)?;
        Ok(Self {
            args,
            static_modules,
            geometry,
            parameters: parameters.clone(),
        })
    }

    /// Normalized dense LFM2 policy.
    pub const fn args(&self) -> &ModelArgs {
        &self.args
    }

    /// Exact local TP/PP geometry.
    pub const fn local_geometry(&self) -> &PartitionLocalGeometry {
        &self.geometry
    }

    /// Pinned modules physically allocated on this pipeline partition.
    pub const fn static_modules(&self) -> &PartitionStaticModules<B> {
        &self.static_modules
    }

    /// Constructs one owned block and rejects every other global index.
    pub fn construct_unit(
        &self,
        global_unit: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<super::Block<B>, Error> {
        let geometry = self.geometry.block(global_unit).ok_or_else(|| {
            Error::backend(format!(
                "LFM2 unit {global_unit} is not owned by local range {:?}",
                self.geometry.owned_units()
            ))
        })?;
        let routed_spec = self
            .geometry
            .expert_realization()
            .and_then(|plan| plan.unit_spec(crate::decoder::TARGET_EXECUTION_GROUP, global_unit))
            .cloned();
        super::Block::new_with_geometry_and_routed_spec(
            &self.args,
            global_unit,
            *geometry,
            routed_spec,
            context,
        )
    }

    fn begin_hidden<S>(
        &self,
        hidden: B::Tensor,
        mask: Option<&B::Tensor>,
        state: &mut S,
        expected: &StateLayout,
        first_state_ordinal: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: RuntimeStateComponents<B>,
    {
        if state.layout() != expected || first_state_ordinal >= expected.len() {
            return Err(Error::backend(
                "LFM2 runtime state does not match its partition-local layout",
            ));
        }
        let sequence = hidden.dim(1);
        let mask =
            if let Some(mask) = mask {
                Some(mask.clone())
            } else if sequence > 1
                && self.args.layer_schedule.iter().any(|policy| {
                    matches!(policy.operator, super::OperatorPolicy::SelfAttention(_))
                })
            {
                Some(B::causal_mask(
                    sequence,
                    state
                        .layer(first_state_ordinal)
                        .map_err(Error::backend)?
                        .position(),
                    None,
                    context,
                )?)
            } else {
                None
            };
        Ok(LayeredForwardState {
            hidden,
            context: ForwardContext { mask },
        })
    }

    fn finish_local(
        &mut self,
        hidden: &B::Tensor,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let norm = self
            .static_modules
            .norm
            .as_mut()
            .ok_or_else(|| Error::backend("LFM2 partition does not own final normalization"))?;
        let hidden = norm.forward(hidden, context)?;
        match (parallel, self.static_modules.lm_head.as_mut()) {
            (Some(parallel), Some(head)) => {
                B::vocabulary_parallel_project(head, &hidden, parallel, context)
            }
            (None, Some(head)) => head.forward(&hidden, context),
            (Some(parallel), None) => B::vocabulary_parallel_embedding_project(
                self.static_modules
                    .embeddings
                    .as_mut()
                    .ok_or_else(|| Error::backend("tied LFM2 output partition has no embedding"))?,
                &hidden,
                parallel,
                context,
            ),
            (None, None) => self
                .static_modules
                .embeddings
                .as_mut()
                .ok_or_else(|| Error::backend("tied LFM2 output partition has no embedding"))?
                .as_linear(&hidden, context),
        }
    }

    fn local_state_ordinal(&self, global_unit: usize) -> Result<usize, Error> {
        let owned = self.geometry.owned_units();
        owned
            .contains(&global_unit)
            .then_some(global_unit - owned.start)
            .ok_or_else(|| Error::backend("LFM2 attempted state access for an unowned unit"))
    }
}

impl<B> eredu_runtime::ArchitectureParameters<B> for PartitionedLayeredModel<B>
where
    B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
{
    type DefinitionError = Error;

    fn state_layout(&self) -> Result<StateLayout, Self::DefinitionError> {
        Ok(self.geometry.complete_state_layout().clone())
    }

    fn state_identity(
        &self,
        state: &eredu_runtime::PartitionState,
        topology: eredu_core::cache::PromptCacheTopology,
    ) -> Result<eredu_runtime::ModelStateIdentity, Self::DefinitionError> {
        super::state_identity(
            &self.args,
            state.layout(),
            state.global_layer_offset(),
            topology,
        )
    }

    fn parameter_description(
        &self,
        _context: &<B::Tensor as Tensor>::Context,
    ) -> Result<ArchitectureParameterDescription, Self::DefinitionError> {
        Ok(self.parameters.clone())
    }

    fn visit_static_parameters<V>(&self, visitor: &mut V) -> Result<(), V::Error>
    where
        V: eredu_runtime::StaticParameterVisitor<B>,
    {
        if let Some(embedding) = &self.static_modules.embeddings {
            visitor.visit("embedding", embedding)?;
        }
        if let Some(norm) = &self.static_modules.norm {
            visitor.visit("norm", norm)?;
        }
        if let Some(head) = &self.static_modules.lm_head {
            visitor.visit("output", head)?;
        }
        Ok(())
    }

    fn visit_static_parameters_mut<V>(&mut self, visitor: &mut V) -> Result<(), V::Error>
    where
        V: eredu_runtime::StaticParameterVisitorMut<B>,
    {
        if let Some(embedding) = &mut self.static_modules.embeddings {
            visitor.visit_mut("embedding", embedding)?;
        }
        if let Some(norm) = &mut self.static_modules.norm {
            visitor.visit_mut("norm", norm)?;
        }
        if let Some(head) = &mut self.static_modules.lm_head {
            visitor.visit_mut("output", head)?;
        }
        Ok(())
    }
}

impl<B, S> LayeredArchitecture<B, S> for PartitionedLayeredModel<B>
where
    B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
{
    type Input<'a> = LayeredInput<'a, B::Tensor>;
    type StaticModules = PartitionStaticModules<B>;
    type Unit = super::Block<B>;
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
        crate::decoder::TARGET_EXECUTION_GROUP
    }

    fn state_partition_plan(
        &self,
        layout: &StateLayout,
    ) -> eredu_runtime::ArchitectureStatePartitionPlan {
        crate::transport::pipeline_state(0, layout)
    }

    fn execution_graph(&self) -> Result<ExecutionGraph, Self::Error> {
        Ok(self.parameters.graph().clone())
    }

    fn group_unit_count(&self, group: usize) -> Result<usize, Self::Error> {
        if group != 0 {
            return Err(Error::backend("LFM2 group is outside the target decoder"));
        }
        usize::try_from(self.args.num_hidden_layers).map_err(Error::backend)
    }

    fn unit_path(&self, group: usize, index: usize) -> Result<String, Self::Error> {
        if group != 0
            || index >= usize::try_from(self.args.num_hidden_layers).map_err(Error::backend)?
        {
            return Err(Error::backend("LFM2 unit is outside the target decoder"));
        }
        Ok(format!("model.layers.{index}"))
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
            return Err(Error::backend("LFM2 group is outside the target decoder"));
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
            .as_mut()
            .ok_or_else(|| Error::backend("LFM2 partition does not own input embedding"))?
            .forward(input.tokens, context)?;
        self.begin_hidden(
            hidden,
            input.mask,
            state,
            self.geometry.complete_state_layout(),
            0,
            context,
        )
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
            return Err(Error::backend("LFM2 target group has invalid dependencies"));
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
    ) -> Result<B::Tensor, Self::Error> {
        if group != 0 {
            return Err(Error::backend("LFM2 group is outside the target decoder"));
        }
        let ordinal = self.local_state_ordinal(index)?;
        unit.forward(
            hidden,
            forward.mask.as_ref(),
            state.layer(ordinal).map_err(Error::backend)?,
            context,
        )
    }

    fn finish_forward(
        &mut self,
        hidden: &B::Tensor,
        _state: &mut S,
        _forward: &Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.finish_local(hidden, None, context)
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

impl<B, S> ParallelLayeredArchitecture<B, S> for PartitionedLayeredModel<B>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
{
    fn begin_forward_parallel<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error> {
        let hidden = B::vocabulary_parallel_lookup(
            self.static_modules
                .embeddings
                .as_mut()
                .ok_or_else(|| Error::backend("LFM2 partition does not own input embedding"))?,
            input.tokens,
            EmbeddingLookupPolicy::Strict,
            parallel,
            context,
        )?;
        self.begin_hidden(
            hidden,
            input.mask,
            state,
            self.geometry.complete_state_layout(),
            0,
            context,
        )
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
        if group != 0 {
            return Err(Error::backend("LFM2 group is outside the target decoder"));
        }
        let ordinal = self.local_state_ordinal(index)?;
        unit.forward_parallel(
            hidden,
            forward.mask.as_ref(),
            state.layer(ordinal).map_err(Error::backend)?,
            parallel,
            context,
        )
    }

    fn finish_forward_parallel(
        &mut self,
        hidden: &B::Tensor,
        _state: &mut S,
        _forward: &Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.finish_local(hidden, Some(parallel), context)
    }
}

impl<B, S> RoutedLayeredArchitecture<B, S> for PartitionedLayeredModel<B>
where
    B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
{
    fn forward_unit_with_provider<P>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        pass: eredu_runtime::ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error>
    where
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        if group != 0 {
            return Err(Error::backend("LFM2 group is outside the target decoder"));
        }
        let ordinal = self.local_state_ordinal(index)?;
        unit.forward_with_feed_forward(
            hidden,
            forward.mask.as_ref(),
            state.layer(ordinal).map_err(Error::backend)?,
            context,
            |policy, normalized, context| {
                policy.forward_with_provider(normalized, pass, context, provider)
            },
        )
    }
}

impl<B, S> ParallelRoutedLayeredArchitecture<B, S> for PartitionedLayeredModel<B>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
{
    fn forward_unit_parallel_with_provider<P>(
        &mut self,
        group: usize,
        index: usize,
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
        P: eredu_runtime::TensorParallelRoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        if group != 0 {
            return Err(Error::backend("LFM2 group is outside the target decoder"));
        }
        let ordinal = self.local_state_ordinal(index)?;
        unit.forward_parallel_with_feed_forward(
            hidden,
            forward.mask.as_ref(),
            state.layer(ordinal).map_err(Error::backend)?,
            parallel,
            context,
            |policy, normalized, parallel, context| {
                policy.forward_parallel_with_provider(normalized, pass, parallel, context, provider)
            },
        )
    }
}

impl<B, S> PartitionedLayeredArchitecture<B, S> for PartitionedLayeredModel<B>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
{
    type Boundary = eredu_runtime::NoAuxiliaryBoundarySchema;

    fn boundary_schema(&self) -> Result<Self::Boundary, Self::Error> {
        Ok(eredu_runtime::NoAuxiliaryBoundarySchema::new(
            self.args.hidden_size,
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
        let hidden = match input {
            LayeredPartitionInput::Tokens(tokens) => self
                .static_modules
                .embeddings
                .as_mut()
                .ok_or_else(|| Error::backend("LFM2 partition does not own input embedding"))?
                .forward(tokens, context)?,
            LayeredPartitionInput::Hidden { hidden, .. } => hidden,
        };
        self.begin_hidden(hidden, mask, state, expected, first_state_ordinal, context)
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
        let hidden = match input {
            LayeredPartitionInput::Tokens(tokens) => B::vocabulary_parallel_lookup(
                self.static_modules
                    .embeddings
                    .as_mut()
                    .ok_or_else(|| Error::backend("LFM2 partition does not own input embedding"))?,
                tokens,
                EmbeddingLookupPolicy::Strict,
                parallel,
                context,
            )?,
            LayeredPartitionInput::Hidden { hidden, .. } => hidden,
        };
        self.begin_hidden(hidden, mask, state, expected, first_state_ordinal, context)
    }

    fn finish_partition(
        &mut self,
        hidden: &B::Tensor,
        _state: &mut S,
        _forward: &Self::ForwardContext,
        owns_output: bool,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<eredu_runtime::LayeredPartitionOutput<B::Tensor>, Self::Error> {
        if owns_output {
            Ok(eredu_runtime::LayeredPartitionOutput::Final {
                output: self.finish_local(hidden, parallel, context)?,
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

impl<B, S> eredu_runtime::ReplicatedTextArchitecture<B, S> for PartitionedLayeredModel<B>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
{
    fn text_input<'a>(tokens: &'a B::Tensor, mask: Option<&'a B::Tensor>) -> Self::Input<'a> {
        LayeredInput { tokens, mask }
    }
}

impl<B, S> crate::partitioned_execution::TextPartitionArchitecture<B, S>
    for PartitionedLayeredModel<B>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
{
    fn partition_text_input<'a>(input: Self::Input<'a>) -> (&'a B::Tensor, Option<&'a B::Tensor>) {
        (input.tokens, input.mask)
    }

    fn partition_output_width(&self) -> i32 {
        self.args.vocab_size
    }
}
