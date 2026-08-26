//! Unified resident and bounded-residency Nemotron-H lifecycle.

use eredu_nn::{
    AttentionCache, EmbeddingLookupPolicy, EmbeddingOperator, Error, NormalizationOperator,
    Parameterized, RoutedNeuralBackend, Tensor,
};
use eredu_runtime::{
    ArchitectureParameterDescription, ExecutionUnitLayout, LayerRuntimeState, LayeredArchitecture,
    LayeredForwardState, LayeredPartitionInput, LayeredPartitionOutput, OwnedParameterGroupSpec,
    ParallelLayeredArchitecture, ParallelRoutedLayeredArchitecture, ParameterGroupOwner,
    PartitionedLayeredArchitecture, RoutedExpertProvider, RoutedLayeredArchitecture,
    RuntimeStateComponents, StateLayout,
};

use super::{
    state_layout, static_parallel_parameter_groups, unit_parallel_parameter_groups, Block,
    EmbeddedInput, ForwardMode, LocalGeometry, ModelArgs, PredictionUnit, RetainedValues,
};
use crate::decoder::{SequentialPredictionGroups, StaticModuleSpec, StaticModules};

/// One target or appended prediction physical unit.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub enum Unit<B: RoutedNeuralBackend> {
    /// Ordinary target-model unit.
    Target(Block<B>),
    /// One physical unit in an MTP prediction group.
    Prediction(PredictionUnit<B>),
}

impl<B, S> RoutedLayeredArchitecture<B, S> for LayeredModel<B>
where
    B: RoutedNeuralBackend,
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
        _pass: eredu_runtime::ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error>
    where
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        LayeredModel::forward_unit_with_provider(
            self, group, index, unit, hidden, state, forward, provider, context,
        )
    }
}

impl<B, S> ParallelRoutedLayeredArchitecture<B, S> for LayeredModel<B>
where
    B: RoutedNeuralBackend,
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
        _pass: eredu_runtime::ExpertPass,
        provider: &mut P,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error>
    where
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        LayeredModel::forward_unit_parallel_with_provider(
            self, group, index, unit, hidden, state, forward, parallel, provider, context,
        )
    }
}

/// Request-local values retained through one target or draft pass.
pub struct ForwardContext<T> {
    tokens: T,
    embedded: T,
    mask: Option<T>,
    mode: ForwardMode,
    target_capture: Option<T>,
    draft_logits: Option<T>,
}

impl<T> ForwardContext<T> {
    /// Borrows the architecture-prepared target or prediction mask.
    pub const fn mask(&self) -> Option<&T> {
        self.mask.as_ref()
    }
}

/// Immutable target-pass values transported across pipeline partitions.
#[derive(Debug, Clone)]
pub struct TargetBoundary<T> {
    tokens: T,
    embedded: T,
}

/// Family-owned wire schema for immutable target context.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct TargetBoundarySchema {
    hidden_size: i32,
}

impl TargetBoundarySchema {
    /// Derives the boundary schema from the normalized target configuration.
    pub const fn from_args(args: &ModelArgs) -> Self {
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

    /// Borrows the input-rank token embeddings required by prediction groups.
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

    const IDENTITY: &'static str = "nemotron_h.target";

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
    ) -> Result<Vec<T>, eredu_runtime::ArchitectureBoundaryError> {
        Ok(vec![boundary.tokens, boundary.embedded])
    }

    fn decode<T>(
        &self,
        tensors: Vec<T>,
    ) -> Result<Self::Boundary<T>, eredu_runtime::ArchitectureBoundaryError> {
        eredu_runtime::validate_boundary_tensor_count(self, &tensors)?;
        let mut tensors = tensors.into_iter();
        let tokens = tensors.next().expect("validated target tokens");
        let embedded = tensors.next().expect("validated target embeddings");
        Ok(TargetBoundary { tokens, embedded })
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
        /// Original tokens and embeddings.
        boundary: TargetBoundary<T>,
    },
}

impl<T> ForwardContext<T> {
    /// Returns whether this invocation executes the target or one draft depth.
    pub const fn mode(&self) -> ForwardMode {
        self.mode
    }
    /// Borrows the target hidden state captured before final normalization.
    pub const fn target_capture(&self) -> Option<&T> {
        self.target_capture.as_ref()
    }
    /// Borrows logits emitted by an MTP draft pass.
    pub const fn draft_logits(&self) -> Option<&T> {
        self.draft_logits.as_ref()
    }
}

/// Shared layered Nemotron-H model including graph-visible MTP groups.
pub struct LayeredModel<B: RoutedNeuralBackend> {
    args: ModelArgs,
    static_modules: StaticModules<B>,
    groups: SequentialPredictionGroups,
    target_units: usize,
    prediction_steps: usize,
    prediction_pattern: usize,
    parallel_geometry: Option<std::sync::Arc<LocalGeometry>>,
}

impl<B: RoutedNeuralBackend> eredu_runtime::ArchitectureParameters<B> for LayeredModel<B> {
    type DefinitionError = Error;

    fn state_layout(&self) -> Result<StateLayout, Self::DefinitionError> {
        self.state_layout_impl()
    }

    fn parameter_description(
        &self,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<ArchitectureParameterDescription, Self::DefinitionError> {
        self.parameter_description_impl(context)
    }

    fn static_parameter_recipes(
        &self,
        source: &dyn eredu_checkpoint::store::CheckpointSource,
    ) -> Result<
        std::collections::BTreeMap<String, eredu_checkpoint::recipe::DerivedWeightRecipe>,
        String,
    > {
        super::static_recipes(source, &self.args, None)
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

impl<B: RoutedNeuralBackend> LayeredModel<B> {
    /// Builds unloaded static modules and validates target plus MTP schedules.
    pub fn new(args: ModelArgs, context: &<B::Tensor as Tensor>::Context) -> Result<Self, Error> {
        crate::operator_requirements::require::<B>(
            "Nemotron-H",
            crate::operator_requirements::NEMOTRON_H,
        )?;
        args.validate().map_err(Error::backend)?;
        let (target_units, prediction_steps, prediction_pattern) = Self::schedule(&args)?;
        let static_modules = StaticModules::from_spec(Self::static_spec(&args), context)?;
        let groups = SequentialPredictionGroups::new_pattern(
            "model.layers",
            target_units,
            "model.mtp.layers",
            prediction_steps,
            prediction_pattern,
        )?;
        Ok(Self {
            args,
            static_modules,
            groups,
            target_units,
            prediction_steps,
            prediction_pattern,
            parallel_geometry: None,
        })
    }

    /// Builds the same target-plus-MTP lifecycle with rank-local geometry.
    pub fn new_parallel(
        args: ModelArgs,
        geometry: LocalGeometry,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        crate::operator_requirements::require::<B>(
            "Nemotron-H",
            crate::operator_requirements::NEMOTRON_H,
        )?;
        args.validate().map_err(Error::backend)?;
        geometry.validate_for(&args).map_err(Error::backend)?;
        let (target_units, prediction_steps, prediction_pattern) = Self::schedule(&args)?;
        let static_modules = StaticModules::from_parallel_spec(
            Self::static_spec(&args),
            geometry.embedding_range().clone(),
            geometry.output_range().cloned(),
            context,
        )?;
        let groups = SequentialPredictionGroups::new_pattern(
            "model.layers",
            target_units,
            "model.mtp.layers",
            prediction_steps,
            prediction_pattern,
        )?;
        Ok(Self {
            args,
            static_modules,
            groups,
            target_units,
            prediction_steps,
            prediction_pattern,
            parallel_geometry: Some(std::sync::Arc::new(geometry)),
        })
    }

    fn schedule(args: &ModelArgs) -> Result<(usize, usize, usize), Error> {
        let target_units = usize::try_from(args.num_hidden_layers).map_err(Error::backend)?;
        let prediction_steps =
            usize::try_from(args.num_nextn_predict_layers).map_err(Error::backend)?;
        let prediction_units = args.mtp_policies().map_err(Error::backend)?.len();
        let prediction_pattern = if prediction_steps == 0 {
            0
        } else {
            prediction_units
                .checked_div(prediction_steps)
                .filter(|n| *n > 0)
                .ok_or_else(|| Error::backend("Nemotron-H MTP pattern is empty"))?
        };
        Ok((target_units, prediction_steps, prediction_pattern))
    }

    fn static_spec(args: &ModelArgs) -> StaticModuleSpec {
        let embedding_name = "model.embeddings.weight";
        StaticModuleSpec {
            embedding_weight: embedding_name.into(),
            normalization_weight: "model.norm_f.weight".into(),
            head_weight: "lm_head.weight".into(),
            vocabulary: args.vocab_size,
            hidden_size: args.hidden_size,
            normalization_epsilon: args.layer_norm_epsilon,
            normalization_offset: 0.0,
            embedding_quantization: args.weight_quantization_for(embedding_name),
            head_format: args.weight_quantization_for("lm_head.weight").into(),
            tied_head: args.tie_word_embeddings,
        }
    }

    /// Returns normalized architecture policy.
    pub const fn args(&self) -> &ModelArgs {
        &self.args
    }

    /// Returns the prediction depth count declared by the execution graph.
    pub fn mtp_len(&self) -> usize {
        self.groups.prediction_count()
    }

    /// Describes target and patterned prediction parameters with explicit
    /// canonical execution-unit ownership.
    fn parameter_description_impl(
        &self,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<ArchitectureParameterDescription, Error> {
        let graph = self.groups.execution_graph()?;
        let counts = std::iter::once(self.target_units)
            .chain(std::iter::repeat_n(
                self.prediction_pattern,
                self.prediction_steps,
            ))
            .collect::<Vec<_>>();
        let layout = ExecutionUnitLayout::new(&graph, counts.clone()).map_err(Error::backend)?;
        let static_groups =
            static_parallel_parameter_groups(&self.static_modules).map_err(Error::backend)?;
        let mut expected = static_groups.clone();
        let mut owned = static_groups
            .into_iter()
            .enumerate()
            .map(|(index, group)| {
                OwnedParameterGroupSpec::new(
                    if index == 0 {
                        let mut roles = vec!["embedding", "mtp"];
                        if self.args.tie_word_embeddings {
                            roles.push("output");
                        }
                        ParameterGroupOwner::static_any_of(roles)
                    } else {
                        ParameterGroupOwner::static_role(match index {
                            0 => "embedding",
                            1 => "norm",
                            _ => "output",
                        })
                    },
                    group,
                )
            })
            .collect::<Vec<_>>();
        for (group_index, &count) in counts.iter().enumerate() {
            let owner_group = layout
                .group_id(group_index)
                .expect("Nemotron-H layout group")
                .clone();
            for index in 0..count {
                let unit = self.construct_unit(group_index, index, context)?;
                let flat = if group_index == 0 {
                    index
                } else {
                    self.target_units + (group_index - 1) * self.prediction_pattern + index
                };
                let groups = unit_parallel_parameter_groups(&unit, &self.args, flat)
                    .map_err(Error::backend)?;
                expected.extend(groups.iter().cloned());
                owned.extend(groups.into_iter().map(|group| {
                    OwnedParameterGroupSpec::new(
                        ParameterGroupOwner::execution_unit(owner_group.clone(), index),
                        group,
                    )
                }));
            }
        }
        ArchitectureParameterDescription::new(&graph, &layout, expected, owned)
            .map_err(Error::backend)
    }
    /// Borrows pinned static modules.
    pub const fn static_modules(&self) -> &StaticModules<B> {
        &self.static_modules
    }
    /// Mutably borrows pinned static modules for checkpoint binding.
    pub fn static_modules_mut(&mut self) -> &mut StaticModules<B> {
        &mut self.static_modules
    }
    /// Returns the authoritative target plus MTP state layout.
    pub fn state_layout(&self) -> Result<StateLayout, Error> {
        state_layout(&self.args).map_err(Error::backend)
    }

    /// Returns the replicated or planner-derived heterogeneous state layout.
    fn state_layout_impl(&self) -> Result<StateLayout, Error> {
        self.parallel_geometry
            .as_ref()
            .map(|geometry| geometry.state_layout().clone())
            .map_or_else(|| self.state_layout(), Ok)
    }

    /// Returns planner-derived geometry for a rank-local realization.
    pub fn parallel_geometry(&self) -> Option<&LocalGeometry> {
        self.parallel_geometry.as_deref()
    }

    /// Shares authoritative local geometry with a backend residency policy.
    pub fn shared_parallel_geometry(&self) -> Option<std::sync::Arc<LocalGeometry>> {
        self.parallel_geometry.as_ref().map(std::sync::Arc::clone)
    }

    /// Constructs one canonical target or MTP unit from model-owned geometry.
    pub fn construct_unit(
        &self,
        group: usize,
        index: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Unit<B>, Error> {
        self.unit_path_inner(group, index)?;
        if group == 0 {
            let block = match &self.parallel_geometry {
                Some(geometry) => Block::new_with_geometry(
                    &self.args,
                    index,
                    *geometry.target_unit(index).ok_or_else(|| {
                        Error::backend(format!(
                            "rank-local Nemotron-H geometry is missing target unit {index}"
                        ))
                    })?,
                    context,
                )?,
                None => Block::new(&self.args, index, context)?,
            };
            Ok(Unit::Target(block))
        } else {
            let depth = group - 1;
            let unit = match &self.parallel_geometry {
                Some(geometry) => {
                    let physical = depth
                        .checked_mul(self.prediction_pattern)
                        .and_then(|start| start.checked_add(index))
                        .ok_or_else(|| {
                            Error::backend("Nemotron-H MTP physical index overflowed")
                        })?;
                    let policies = self.args.mtp_policies().map_err(Error::backend)?;
                    PredictionUnit::new_with_geometry(
                        &self.args,
                        depth,
                        index,
                        *policies.get(physical).ok_or_else(|| {
                            Error::backend(format!(
                                "Nemotron-H MTP policy is missing physical unit {physical}"
                            ))
                        })?,
                        *geometry.prediction_unit(physical).ok_or_else(|| {
                            Error::backend(format!(
                                "rank-local Nemotron-H geometry is missing MTP unit {physical}"
                            ))
                        })?,
                        context,
                    )?
                }
                None => PredictionUnit::new(&self.args, depth, index, context)?,
            };
            Ok(Unit::Prediction(unit))
        }
    }

    /// Begins a target pass from a placement-supplied token embedding.
    pub fn begin_embedded_target<S>(
        &self,
        tokens: &B::Tensor,
        hidden: B::Tensor,
        mask: Option<&B::Tensor>,
        state: &mut S,
        expected: &StateLayout,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: RuntimeStateComponents<B>,
    {
        self.begin_embedded_target_at(tokens, hidden, mask, state, expected, 0, context)
    }

    /// Starts a replicated target partition without re-embedding on downstream
    /// ranks and returns the typed immutable boundary to relay unchanged.
    #[allow(clippy::too_many_arguments)]
    pub fn begin_partition_target<S>(
        &mut self,
        input: TargetPartitionInput<'_, B::Tensor>,
        mask: Option<&B::Tensor>,
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
                let embedded = self.static_modules.embeddings.forward(tokens, context)?;
                (tokens.clone(), embedded.clone(), embedded)
            }
            TargetPartitionInput::Hidden { hidden, boundary } => {
                let (tokens, embedded) = boundary.into_parts();
                (tokens, embedded, hidden)
            }
        };
        let forward = self.begin_embedded_target_at(
            &tokens,
            hidden,
            mask,
            state,
            expected,
            first_state_ordinal,
            context,
        )?;
        Ok((forward, TargetBoundary::new(tokens, embedded)))
    }

    /// Tensor-parallel target-partition entry with input-rank vocabulary lookup.
    #[allow(clippy::too_many_arguments)]
    pub fn begin_partition_target_parallel<S>(
        &mut self,
        input: TargetPartitionInput<'_, B::Tensor>,
        mask: Option<&B::Tensor>,
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
                    &mut self.static_modules.embeddings,
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
        let forward = self.begin_embedded_target_at(
            &tokens,
            hidden,
            mask,
            state,
            expected,
            first_state_ordinal,
            context,
        )?;
        Ok((forward, TargetBoundary::new(tokens, embedded)))
    }

    #[allow(clippy::too_many_arguments)]
    fn begin_embedded_target_at<S>(
        &self,
        tokens: &B::Tensor,
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
        if state.layout() != expected {
            return Err(Error::backend(format!(
                "Nemotron-H runtime state layout {:?} does not match architecture layout {expected:?}",
                state.layout()
            )));
        }
        let mask = if let Some(mask) = mask {
            Some(mask.clone())
        } else if hidden.dim(1) > 1 {
            Some(B::causal_mask(
                hidden.dim(1),
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
            hidden: hidden.clone(),
            context: ForwardContext {
                tokens: tokens.clone(),
                embedded: hidden,
                mask,
                mode: ForwardMode::Target,
                target_capture: None,
                draft_logits: None,
            },
        })
    }

    /// Begins one MTP draft pass from a placement-supplied token embedding.
    #[allow(clippy::too_many_arguments)]
    pub fn begin_embedded_draft<S>(
        &self,
        tokens: &B::Tensor,
        embedded: B::Tensor,
        hidden: &B::Tensor,
        depth: usize,
        state: &mut S,
        expected: &StateLayout,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: RuntimeStateComponents<B>,
    {
        if state.layout() != expected {
            return Err(Error::backend(format!(
                "Nemotron-H runtime state layout {:?} does not match architecture layout {expected:?}",
                state.layout()
            )));
        }
        if depth >= self.prediction_steps {
            return Err(Error::backend(format!(
                "Nemotron-H MTP depth {depth} is outside {} groups",
                self.prediction_steps
            )));
        }
        let position_layer = self
            .target_units
            .checked_add(depth * self.prediction_pattern)
            .ok_or_else(|| Error::backend("Nemotron-H MTP mask state index overflowed"))?;
        let mask = if embedded.dim(1) > 1 {
            Some(B::causal_mask(
                embedded.dim(1),
                state
                    .layer(position_layer)
                    .map_err(Error::backend)?
                    .position(),
                None,
                context,
            )?)
        } else {
            None
        };
        Ok(LayeredForwardState {
            hidden: hidden.clone(),
            context: ForwardContext {
                tokens: tokens.clone(),
                embedded,
                mask,
                mode: ForwardMode::Draft(depth),
                target_capture: None,
                draft_logits: None,
            },
        })
    }

    /// Applies the replicated target output boundary.
    pub fn finish_partition_target(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let hidden = self.static_modules.norm.forward(hidden, context)?;
        match &mut self.static_modules.lm_head {
            Some(head) => eredu_nn::LinearOperator::forward(head, &hidden, context),
            None => self.static_modules.embeddings.as_linear(&hidden, context),
        }
    }

    /// Applies the tensor-parallel target output boundary.
    pub fn finish_partition_target_parallel(
        &mut self,
        hidden: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
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

    /// Applies tensor-parallel vocabulary lookup for a prediction boundary.
    pub fn embed_partition_parallel(
        &mut self,
        tokens: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        B::vocabulary_parallel_lookup(
            &mut self.static_modules.embeddings,
            tokens,
            EmbeddingLookupPolicy::Strict,
            parallel,
            context,
        )
    }

    /// Projects prediction hidden state through the tensor-parallel vocabulary
    /// boundary without target-only final normalization.
    pub fn project_partition_parallel(
        &mut self,
        hidden: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        match &mut self.static_modules.lm_head {
            Some(head) => B::vocabulary_parallel_project(head, hidden, parallel, context),
            None => B::vocabulary_parallel_embedding_project(
                &mut self.static_modules.embeddings,
                hidden,
                parallel,
                context,
            ),
        }
    }

    fn validate_group(&self, group: usize) -> Result<usize, Error> {
        self.groups.unit_count(group)
    }

    fn unit_path_inner(&self, group: usize, index: usize) -> Result<String, Error> {
        self.groups.unit_path(group, index)
    }

    fn state_index(&self, group: usize, index: usize) -> Result<usize, Error> {
        if group == 0 {
            return Ok(index);
        }
        self.target_units
            .checked_add(
                (group - 1)
                    .checked_mul(self.prediction_pattern)
                    .and_then(|start| start.checked_add(index))
                    .ok_or_else(|| Error::backend("Nemotron-H MTP state index overflowed"))?,
            )
            .ok_or_else(|| Error::backend("Nemotron-H total state index overflowed"))
    }

    /// Executes a target or prediction unit through a runtime expert provider.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_unit_with_provider<S, P>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Unit<B>,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut ForwardContext<B::Tensor>,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        self.unit_path_inner(group, index)?;
        let state_index = self.state_index(group, index)?;
        match unit {
            Unit::Target(block) if group == 0 => block.forward_with_provider(
                hidden,
                forward.mask.as_ref(),
                state.layer(state_index).map_err(Error::backend)?,
                context,
                provider,
            ),
            Unit::Prediction(unit) if group > 0 => unit.forward_with_provider(
                hidden,
                &forward.embedded,
                forward.mask.as_ref(),
                state.layer(state_index).map_err(Error::backend)?,
                context,
                provider,
            ),
            _ => Err(Error::backend(format!(
                "Nemotron-H execution unit does not match group {group}"
            ))),
        }
    }

    /// Executes a target unit with stable activation and routing observations.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_unit_observed_with_provider<S, O, P>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Unit<B>,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut ForwardContext<B::Tensor>,
        observer: &mut O,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
        O: eredu_runtime::ActivationObserver<B::Tensor, Error>,
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        let path = self.unit_path_inner(group, index)?;
        let state_index = self.state_index(group, index)?;
        match unit {
            Unit::Target(block) if group == 0 => block.forward_observed_with_provider(
                &path,
                self.args.n_routed_experts,
                hidden,
                forward.mask.as_ref(),
                state.layer(state_index).map_err(Error::backend)?,
                context,
                observer,
                provider,
            ),
            _ => Err(Error::backend(
                "observed Nemotron-H execution currently requires a target unit",
            )),
        }
    }

    /// Executes one placement-resolved unit with tensor collectives.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_unit_parallel_with_provider<S, P>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Unit<B>,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut ForwardContext<B::Tensor>,
        parallel: &B::ParallelContext,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        self.unit_path_inner(group, index)?;
        let state_index = self.state_index(group, index)?;
        match unit {
            Unit::Target(block) if group == 0 => block.forward_parallel(
                hidden,
                forward.mask.as_ref(),
                state.layer(state_index).map_err(Error::backend)?,
                parallel,
                context,
                provider,
            ),
            Unit::Prediction(unit) if group > 0 => unit.forward_parallel_with_provider(
                hidden,
                &forward.embedded,
                forward.mask.as_ref(),
                state.layer(state_index).map_err(Error::backend)?,
                parallel,
                context,
                provider,
            ),
            _ => Err(Error::backend(format!(
                "Nemotron-H execution unit does not match group {group}"
            ))),
        }
    }
}

impl<B, S> LayeredArchitecture<B, S> for LayeredModel<B>
where
    B: RoutedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
{
    type Input<'a> = EmbeddedInput<'a, B::Tensor>;
    type StaticModules = StaticModules<B>;
    type Unit = Unit<B>;
    type ForwardContext = ForwardContext<B::Tensor>;
    type RetainedContextValues<'a>
        = RetainedValues<'a, B::Tensor>
    where
        B::Tensor: 'a;
    type Error = Error;

    fn group_transport(&self, group: usize) -> eredu_runtime::ArchitectureGroupTransport {
        if group == 0 {
            crate::transport::decoder()
        } else {
            prediction_group_transport(group)
        }
    }

    fn model_identity(&self) -> &str {
        &self.args.model_type
    }
    fn execution_graph(&self) -> Result<eredu_runtime::ExecutionGraph, Self::Error> {
        self.groups.execution_graph()
    }
    fn group_unit_count(&self, group: usize) -> Result<usize, Self::Error> {
        self.validate_group(group)
    }
    fn unit_path(&self, group: usize, index: usize) -> Result<String, Self::Error> {
        self.unit_path_inner(group, index)
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
        self.construct_unit(group, index, context)
    }

    fn begin_forward<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error> {
        let expected = self.state_layout()?;
        if state.layout() != &expected {
            return Err(Error::backend(format!("Nemotron-H runtime state layout {:?} does not match architecture layout {expected:?}", state.layout())));
        }
        let (tokens, supplied_hidden, supplied_mask, mode) = match input {
            EmbeddedInput::Target { tokens, mask } => (tokens, None, mask, ForwardMode::Target),
            EmbeddedInput::Draft {
                tokens,
                hidden,
                depth,
            } => {
                if depth >= self.prediction_steps {
                    return Err(Error::backend(format!(
                        "Nemotron-H MTP depth {depth} is outside {} groups",
                        self.prediction_steps
                    )));
                }
                (tokens, Some(hidden), None, ForwardMode::Draft(depth))
            }
        };
        let embedded = self.static_modules.embeddings.forward(tokens, context)?;
        let hidden = supplied_hidden.cloned().unwrap_or_else(|| embedded.clone());
        let position_layer = match mode {
            ForwardMode::Target => 0,
            ForwardMode::Draft(depth) => self
                .target_units
                .checked_add(depth * self.prediction_pattern)
                .ok_or_else(|| Error::backend("Nemotron-H MTP mask state index overflowed"))?,
        };
        let mask = if let Some(mask) = supplied_mask {
            Some(mask.clone())
        } else if embedded.dim(1) > 1 {
            Some(B::causal_mask(
                embedded.dim(1),
                state
                    .layer(position_layer)
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
            context: ForwardContext {
                tokens: tokens.clone(),
                embedded,
                mask,
                mode,
                target_capture: None,
                draft_logits: None,
            },
        })
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
        self.groups.begin(group, initial, dependencies)
    }

    fn should_execute_group(&self, group: usize, forward: &Self::ForwardContext) -> bool {
        match forward.mode {
            ForwardMode::Target => group == 0,
            ForwardMode::Draft(depth) => group == depth + 1,
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
        self.unit_path_inner(group, index)?;
        let state_index = self.state_index(group, index)?;
        match unit {
            Unit::Target(block) if group == 0 => block.forward(
                hidden,
                forward.mask.as_ref(),
                state.layer(state_index).map_err(Error::backend)?,
                context,
            ),
            Unit::Prediction(unit) if group > 0 => unit.forward(
                hidden,
                &forward.embedded,
                forward.mask.as_ref(),
                state.layer(state_index).map_err(Error::backend)?,
                context,
            ),
            _ => Err(Error::backend(format!(
                "Nemotron-H execution unit does not match group {group}"
            ))),
        }
    }

    fn complete_execution_group(
        &mut self,
        group: usize,
        hidden: &B::Tensor,
        _state: &mut S,
        forward: &mut Self::ForwardContext,
        _context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        if (group == 0 && matches!(forward.mode, ForwardMode::Target))
            || matches!(forward.mode, ForwardMode::Draft(depth) if group == depth + 1)
        {
            forward.target_capture = Some(hidden.clone());
        }
        Ok(hidden.clone())
    }

    fn finish_forward(
        &mut self,
        hidden: &B::Tensor,
        _state: &mut S,
        forward: &Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        match forward.mode {
            ForwardMode::Target => {
                let hidden = self.static_modules.norm.forward(hidden, context)?;
                match &mut self.static_modules.lm_head {
                    Some(head) => eredu_nn::LinearOperator::forward(head, &hidden, context),
                    None => self.static_modules.embeddings.as_linear(&hidden, context),
                }
            }
            ForwardMode::Draft(_) => match &mut self.static_modules.lm_head {
                Some(head) => eredu_nn::LinearOperator::forward(head, hidden, context),
                None => self.static_modules.embeddings.as_linear(hidden, context),
            },
        }
    }

    fn retained_context_values<'a>(
        &'a self,
        forward: &'a Self::ForwardContext,
        _group: usize,
        _index: usize,
    ) -> Self::RetainedContextValues<'a> {
        RetainedValues::new([
            Some(&forward.tokens),
            Some(&forward.embedded),
            forward.mask.as_ref(),
            forward.target_capture.as_ref(),
            forward.draft_logits.as_ref(),
        ])
    }
}

fn prediction_group_transport(group: usize) -> eredu_runtime::ArchitectureGroupTransport {
    let mut transport = crate::transport::prediction();
    if group == 1 {
        transport.first_owner_static_roles.push("mtp".into());
    }
    transport
}

#[cfg(test)]
#[allow(
    clippy::items_after_test_module,
    reason = "the transport contract tests stay adjacent to the transport declaration"
)]
mod transport_tests {
    use super::{prediction_group_transport, TargetBoundarySchema};
    use eredu_runtime::{ArchitectureBoundary, BoundaryTensorDtype};

    #[test]
    fn target_boundary_declares_tokens_and_retained_embeddings() {
        let schema = TargetBoundarySchema { hidden_size: 32 };
        let tensors = schema.wire_schema().unwrap().resolve(2, 7).unwrap();
        assert_eq!(tensors.primary().shape(), [2, 7, 32]);
        assert_eq!(tensors.auxiliary().len(), 2);
        assert_eq!(tensors.auxiliary()[0].dtype(), BoundaryTensorDtype::Uint32);
        assert_eq!(tensors.auxiliary()[1].shape(), [2, 7, 32]);
    }

    #[test]
    fn first_prediction_group_owns_shared_mtp_embedding_role_once() {
        assert_eq!(
            prediction_group_transport(1).first_owner_static_roles,
            ["mtp"]
        );
        assert!(prediction_group_transport(2)
            .first_owner_static_roles
            .is_empty());
    }
}

impl<B, S> ParallelLayeredArchitecture<B, S> for LayeredModel<B>
where
    B: RoutedNeuralBackend,
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
        let expected = self
            .parallel_geometry
            .as_ref()
            .ok_or_else(|| {
                Error::backend("Nemotron-H model was not built with rank-local geometry")
            })?
            .state_layout()
            .clone();
        match input {
            EmbeddedInput::Target { tokens, mask } => {
                let embedded = B::vocabulary_parallel_lookup(
                    &mut self.static_modules.embeddings,
                    tokens,
                    EmbeddingLookupPolicy::Strict,
                    parallel,
                    context,
                )?;
                self.begin_embedded_target(tokens, embedded, mask, state, &expected, context)
            }
            EmbeddedInput::Draft {
                tokens,
                hidden,
                depth,
            } => {
                let embedded = B::vocabulary_parallel_lookup(
                    &mut self.static_modules.embeddings,
                    tokens,
                    EmbeddingLookupPolicy::Strict,
                    parallel,
                    context,
                )?;
                self.begin_embedded_draft(
                    tokens, embedded, hidden, depth, state, &expected, context,
                )
            }
        }
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
        self.forward_unit_parallel_with_provider(
            group,
            index,
            unit,
            hidden,
            state,
            forward,
            parallel,
            &mut eredu_runtime::ResidentExpertProvider,
            context,
        )
    }

    fn finish_forward_parallel(
        &mut self,
        hidden: &B::Tensor,
        _state: &mut S,
        forward: &Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        if self.parallel_geometry.is_none() {
            return Err(Error::backend(
                "Nemotron-H model was not built with rank-local geometry",
            ));
        }
        let hidden = match forward.mode {
            ForwardMode::Target => self.static_modules.norm.forward(hidden, context)?,
            ForwardMode::Draft(_) => hidden.clone(),
        };
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

impl<B, S> PartitionedLayeredArchitecture<B, S> for LayeredModel<B>
where
    B: RoutedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
{
    type Boundary = TargetBoundarySchema;

    fn boundary_schema(&self) -> Result<Self::Boundary, Self::Error> {
        Ok(TargetBoundarySchema::from_args(self.args()))
    }

    fn begin_partition<'a>(
        &mut self,
        input: LayeredPartitionInput<'a, B::Tensor, TargetBoundary<B::Tensor>>,
        mask: Option<&B::Tensor>,
        state: &mut S,
        expected: &StateLayout,
        first_state_ordinal: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error> {
        let input = match input {
            LayeredPartitionInput::Tokens(tokens) => TargetPartitionInput::Tokens(tokens),
            LayeredPartitionInput::Hidden { hidden, auxiliary } => TargetPartitionInput::Hidden {
                hidden,
                boundary: auxiliary,
            },
        };
        self.begin_partition_target(input, mask, state, expected, first_state_ordinal, context)
            .map(|(forward, _)| forward)
    }

    fn begin_partition_parallel<'a>(
        &mut self,
        input: LayeredPartitionInput<'a, B::Tensor, TargetBoundary<B::Tensor>>,
        mask: Option<&B::Tensor>,
        state: &mut S,
        expected: &StateLayout,
        first_state_ordinal: usize,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error> {
        let input = match input {
            LayeredPartitionInput::Tokens(tokens) => TargetPartitionInput::Tokens(tokens),
            LayeredPartitionInput::Hidden { hidden, auxiliary } => TargetPartitionInput::Hidden {
                hidden,
                boundary: auxiliary,
            },
        };
        self.begin_partition_target_parallel(
            input,
            mask,
            state,
            expected,
            first_state_ordinal,
            parallel,
            context,
        )
        .map(|(forward, _)| forward)
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
                    self.finish_forward_parallel(hidden, state, forward, parallel, context)?
                }
                None => self.finish_forward(hidden, state, forward, context)?,
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
}
