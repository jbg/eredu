//! Unified resident and bounded-residency Nemotron-H lifecycle.

use eredu_nn::{
    AttentionCache, EmbeddingLookupPolicy, EmbeddingOperator, Error, GroupedNeuralBackend,
    Parameterized, Tensor,
};
use eredu_runtime::{
    LayerRuntimeState, LayeredForwardState, LayeredPartitionInput, LayeredPartitionOutput,
    ParameterProvider, RuntimeStateComponents, StateLayout,
};

use super::{
    state_layout, Block, EmbeddedInput, ForwardMode, LocalGeometry, ModelArgs, Operator,
    PredictionUnit, RetainedValues, TargetReadout,
};
use crate::decoder::{ComponentInstrumentation, DecoderStaticResources, StaticModules};
use eredu_runtime::{ComponentEquations as _, ParallelComponentKernel as _};

/// One target or appended prediction physical unit.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub enum Unit<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> {
    /// Ordinary target-model unit.
    Target(Block<B>),
    /// One physical unit in an MTP prediction group.
    Prediction(PredictionUnit<B>),
}

/// Request-local values retained through one target or draft pass.
pub struct ForwardContext<T> {
    tokens: T,
    embedded: T,
    mask: Option<T>,
    mode: ForwardMode,
    readout: TargetReadout,
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
    readout: T,
}

/// Family-owned wire schema for immutable target context.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct TargetBoundarySchema {
    hidden_size: i32,
    signed_tokens: bool,
}

impl TargetBoundarySchema {
    /// Derives the boundary schema from the normalized target configuration.
    pub const fn from_args(args: &ModelArgs) -> Self {
        Self {
            hidden_size: args.hidden_size,
            signed_tokens: false,
        }
    }

    /// Selects signed token storage for supplied-embedding producers.
    ///
    /// The exact wire schema includes this dtype and rejects disagreement between
    /// ranks; ordinary tokenizer-driven pipelines retain unsigned token storage.
    pub const fn with_signed_tokens(mut self) -> Self {
        self.signed_tokens = true;
        self
    }

    /// Width of both the activation and retained input embedding.
    pub const fn hidden_size(&self) -> i32 {
        self.hidden_size
    }

    /// Validates explicit ingress against its selected wire geometry and token dtype.
    pub fn validate_input<T: Tensor>(
        &self,
        tokens: &T,
        embedded: &T,
    ) -> Result<(), TargetBoundaryInputError> {
        let shape = tokens.shape();
        if shape.len() != 2
            || shape.iter().any(|&n| n <= 0)
            || embedded.shape() != [shape[0], shape[1], self.hidden_size]
        {
            return Err(TargetBoundaryInputError::Geometry);
        }
        let expected = if self.signed_tokens {
            eredu_nn::TensorElementType::I32
        } else {
            eredu_nn::TensorElementType::U32
        };
        if tokens.element_type() != Some(expected) {
            return Err(TargetBoundaryInputError::TokenType {
                expected,
                actual: tokens.element_type(),
            });
        }
        Ok(())
    }
}

/// Supplied pipeline ingress does not match the exact retained target wire schema.
#[derive(Debug, thiserror::Error)]
pub enum TargetBoundaryInputError {
    /// Token or embedding geometry differs from the selected boundary.
    #[error("Nemotron-H boundary requires matching nonempty [batch, sequence] tokens and [batch, sequence, hidden] embeddings")]
    Geometry,
    /// Token storage differs from the wire schema's integer representation.
    #[error("Nemotron-H boundary token dtype {actual:?} differs from {expected:?}")]
    TokenType {
        /// Exact integer dtype declared by this boundary.
        expected: eredu_nn::TensorElementType,
        /// Tensor-reported dtype, absent if the backend cannot establish it.
        actual: Option<eredu_nn::TensorElementType>,
    },
}

impl<T> TargetBoundary<T> {
    /// Creates the typed target boundary in canonical wire order.
    pub const fn new(tokens: T, embedded: T, readout: T) -> Self {
        Self {
            tokens,
            embedded,
            readout,
        }
    }

    /// Borrows original target token ids.
    pub const fn tokens(&self) -> &T {
        &self.tokens
    }

    /// Borrows the input-rank token embeddings required by prediction groups.
    pub const fn embedded(&self) -> &T {
        &self.embedded
    }

    /// Exact I32[1] target readout control transported with this invocation.
    pub const fn readout(&self) -> &T {
        &self.readout
    }

    /// Decomposes the boundary without cloning backend tensors.
    pub fn into_parts(self) -> (T, T, T) {
        (self.tokens, self.embedded, self.readout)
    }
}

impl eredu_runtime::ArchitectureBoundary for TargetBoundarySchema {
    type Boundary<T> = TargetBoundary<T>;

    const IDENTITY: &'static str = "nemotron_h.target.v2";

    fn primary_tensor_spec(&self) -> eredu_runtime::BoundaryTensorSpec {
        eredu_runtime::BoundaryTensorSpec::primary_activation(self.hidden_size)
    }

    fn auxiliary_tensor_specs(&self) -> Vec<eredu_runtime::BoundaryTensorSpec> {
        use eredu_runtime::{BoundaryTensorDimension as Dim, BoundaryTensorDtype as Dtype};
        vec![
            eredu_runtime::BoundaryTensorSpec::new(
                "tokens",
                [Dim::Batch, Dim::Sequence],
                if self.signed_tokens {
                    Dtype::Int32
                } else {
                    Dtype::Uint32
                },
            ),
            eredu_runtime::BoundaryTensorSpec::new(
                "embedded",
                [Dim::Batch, Dim::Sequence, Dim::Fixed(self.hidden_size)],
                Dtype::Activation,
            ),
            eredu_runtime::BoundaryTensorSpec::new("readout", [Dim::Fixed(1)], Dtype::Int32),
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
            eredu_runtime::ArchitectureBoundaryValue::new("readout", boundary.readout)?,
        ])
    }

    fn decode<T>(
        &self,
        tensors: Vec<T>,
    ) -> Result<Self::Boundary<T>, eredu_runtime::ArchitectureBoundaryError> {
        eredu_runtime::validate_boundary_tensor_count(self, &tensors)?;
        let mut tensors = tensors.into_iter();
        let tokens = tensors.next().expect("validated target tokens");
        let embedded = tensors.next().expect("validated target embeddings");
        let readout = tensors.next().expect("validated target readout");
        Ok(TargetBoundary {
            tokens,
            embedded,
            readout,
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
pub type LayeredModel<B> = eredu_runtime::ComponentAdapter<
    std::sync::Arc<eredu_runtime::ComponentDefinition>,
    CompleteNemotronKernel<B>,
    eredu_runtime::KernelParameterModules,
>;

/// Complete target and embedded-prediction equations over one retained definition.
pub struct CompleteNemotronKernel<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> {
    args: ModelArgs,
    static_modules: StaticModules<B>,
    definition: std::sync::Arc<eredu_runtime::ComponentDefinition>,
    target_units: usize,
    prediction_steps: usize,
    prediction_pattern: usize,
    parallel_geometry: Option<std::sync::Arc<LocalGeometry>>,
    expert_realization: Option<crate::ExpertRealizationPlan<eredu_nn::GroupedRelu2Spec>>,
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> CompleteNemotronKernel<B> {
    /// Materializes serial resources from the cold retained declaration.
    pub fn from_definition(
        args: ModelArgs,
        definition: std::sync::Arc<eredu_runtime::ComponentDefinition>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredModel<B>, Error> {
        Self::materialize(args, definition, None, context)
    }
    /// Materializes the same target/MTP equations with admitted rank-local geometry.
    pub fn from_parallel_definition(
        args: ModelArgs,
        definition: std::sync::Arc<eredu_runtime::ComponentDefinition>,
        geometry: LocalGeometry,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredModel<B>, Error> {
        Self::materialize(args, definition, Some(geometry), context)
    }
    fn materialize(
        args: ModelArgs,
        definition: std::sync::Arc<eredu_runtime::ComponentDefinition>,
        geometry: Option<LocalGeometry>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredModel<B>, Error> {
        crate::operator_requirements::require::<B>(
            "Nemotron-H",
            crate::operator_requirements::NEMOTRON_H,
        )?;
        args.validate().map_err(Error::backend)?;
        let definition = match &geometry {
            Some(geometry) => {
                geometry.validate_for(&args).map_err(Error::backend)?;
                super::parameters::select_definition(
                    &args,
                    definition,
                    geometry.state_layout(),
                    (
                        geometry.embedding_range().clone(),
                        geometry.output_range().cloned(),
                    ),
                    geometry
                        .target_units()
                        .iter()
                        .chain(geometry.prediction_units())
                        .copied()
                        .enumerate(),
                )?
            }
            None => {
                let state = state_layout(&args).map_err(Error::backend)?;
                let full = eredu_nn::VocabularyParallelRange {
                    global_vocabulary: args.vocab_size as usize,
                    local: 0..args.vocab_size as usize,
                };
                let targets = super::parallel::global_target_geometry(&args)
                    .map_err(Error::backend_source)?;
                let predictions = super::parallel::global_prediction_geometry(&args)
                    .map_err(Error::backend_source)?;
                super::parameters::select_definition(
                    &args,
                    definition,
                    &state,
                    (full.clone(), (!args.tie_word_embeddings).then_some(full)),
                    targets.into_iter().chain(predictions).enumerate(),
                )?
            }
        };
        let static_modules = match &geometry {
            Some(geometry) => StaticModules::from_parallel_spec(
                super::static_spec(&args, super::FinalNormalization::Float32),
                geometry.embedding_range().clone(),
                geometry.output_range().cloned(),
                context,
            )?,
            None => StaticModules::from_spec(
                super::static_spec(&args, super::FinalNormalization::Float32),
                context,
            )?,
        };
        Self::with_selected_resources(args, definition, static_modules, geometry)
    }

    fn with_selected_resources(
        args: ModelArgs,
        definition: std::sync::Arc<eredu_runtime::ComponentDefinition>,
        static_modules: StaticModules<B>,
        geometry: Option<LocalGeometry>,
    ) -> Result<LayeredModel<B>, Error> {
        let target_units = definition.unit_count(0)?;
        let prediction_steps = definition.prediction_groups().len();
        let prediction_pattern = if prediction_steps == 0 {
            0
        } else {
            definition.unit_count(1)?
        };
        let kernel = Self {
            args,
            static_modules,
            definition: std::sync::Arc::clone(&definition),
            target_units,
            prediction_steps,
            prediction_pattern,
            parallel_geometry: geometry.map(std::sync::Arc::new),
            expert_realization: None,
        };
        eredu_runtime::ComponentAdapter::with_pinned_resources::<B>(definition, kernel)
            .map_err(Error::from)
    }

    /// Returns normalized architecture policy.
    pub const fn args(&self) -> &ModelArgs {
        &self.args
    }

    /// Returns the prediction depth count declared by the execution graph.
    pub fn mtp_len(&self) -> usize {
        self.definition.prediction_groups().len()
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

    /// Returns planner-derived geometry for a rank-local realization.
    pub fn parallel_geometry(&self) -> Option<&LocalGeometry> {
        self.parallel_geometry.as_deref()
    }

    /// Shares authoritative local geometry with a backend residency policy.
    pub fn shared_parallel_geometry(&self) -> Option<std::sync::Arc<LocalGeometry>> {
        self.parallel_geometry.as_ref().map(std::sync::Arc::clone)
    }

    /// Installs the architecture-derived expert realization used by every unit factory.
    pub fn install_expert_realization(
        &mut self,
        realization: crate::ExpertRealizationPlan<eredu_nn::GroupedRelu2Spec>,
    ) {
        self.expert_realization = Some(realization);
    }

    /// Borrows the architecture-derived expert realization used by unit factories.
    pub fn expert_realization_plan(
        &self,
    ) -> Option<&crate::ExpertRealizationPlan<eredu_nn::GroupedRelu2Spec>> {
        self.expert_realization.as_ref()
    }

    /// Projects the installed realization into an opaque backend provider descriptor.
    pub fn expert_provider_descriptor(
        &self,
    ) -> Result<Option<super::ExpertProviderDescriptor>, Error> {
        self.expert_realization
            .as_ref()
            .map(|plan| super::expert_provider_descriptor(plan, &self.args))
            .transpose()
    }

    /// Selects, installs, and erases the rank-local expert realization for a
    /// backend provider without exposing the realization plan.
    pub fn prepare_expert_provider_descriptor(
        &mut self,
        topology: eredu_core::ParallelRankTopology,
    ) -> Result<Option<super::ExpertProviderDescriptor>, Error> {
        let realization =
            super::expert_realization_plan(&self.args, self.parallel_geometry(), topology)?;
        if let Some(realization) = realization {
            self.install_expert_realization(realization);
        }
        self.expert_provider_descriptor()
    }

    /// Constructs one canonical target or MTP unit from model-owned geometry.
    pub fn construct_unit(
        &self,
        group: usize,
        index: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Unit<B>, Error> {
        self.unit_path_inner(group, index)?;
        let mut unit = if group == 0 {
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
            Unit::Target(block)
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
            Unit::Prediction(unit)
        };
        if let Some(realization) = &self.expert_realization {
            let owner_group = if group == 0 {
                "target".to_owned()
            } else {
                format!("mtp.{}", group - 1)
            };
            if let Some(spec) = realization.unit_spec(&owner_group, index) {
                let operator = match &mut unit {
                    Unit::Target(block) => &mut block.operator,
                    Unit::Prediction(unit) => &mut unit.block.operator,
                };
                let Operator::Sparse(moe) = operator else {
                    return Err(Error::backend(format!(
                        "Nemotron-H realization names non-sparse unit {owner_group}.{index}"
                    )));
                };
                moe.experts = B::grouped_relu2(spec.clone(), context)?;
            }
        }
        Ok(unit)
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
        let (tokens, embedded, hidden, readout) = match input {
            TargetPartitionInput::Tokens(tokens) => {
                let embedded = self.static_modules.embeddings.forward(tokens, context)?;
                (
                    tokens.clone(),
                    embedded.clone(),
                    embedded,
                    TargetReadout::Logits.encode::<B::Tensor>(context)?,
                )
            }
            TargetPartitionInput::Hidden { hidden, boundary } => {
                let (tokens, embedded, readout) = boundary.into_parts();
                (tokens, embedded, hidden, readout)
            }
        };
        let selected_readout = TargetReadout::decode(&readout, context)?;
        let mut forward = self.begin_embedded_target_at(
            &tokens,
            hidden,
            mask,
            state,
            expected,
            first_state_ordinal,
            context,
        )?;
        forward.context.readout = selected_readout;
        Ok((forward, TargetBoundary::new(tokens, embedded, readout)))
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
        let (tokens, embedded, hidden, readout) = match input {
            TargetPartitionInput::Tokens(tokens) => {
                let embedded = B::vocabulary_parallel_lookup(
                    &mut self.static_modules.embeddings,
                    tokens,
                    EmbeddingLookupPolicy::Strict,
                    parallel,
                    context,
                )?;
                (
                    tokens.clone(),
                    embedded.clone(),
                    embedded,
                    TargetReadout::Logits.encode::<B::Tensor>(context)?,
                )
            }
            TargetPartitionInput::Hidden { hidden, boundary } => {
                let (tokens, embedded, readout) = boundary.into_parts();
                (tokens, embedded, hidden, readout)
            }
        };
        let selected_readout = TargetReadout::decode(&readout, context)?;
        let mut forward = self.begin_embedded_target_at(
            &tokens,
            hidden,
            mask,
            state,
            expected,
            first_state_ordinal,
            context,
        )?;
        forward.context.readout = selected_readout;
        Ok((forward, TargetBoundary::new(tokens, embedded, readout)))
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
        super::mtp::validate_target_embeddings(tokens, &hidden, self.args.hidden_size)?;
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
                readout: TargetReadout::Logits,
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
                readout: TargetReadout::Logits,
                target_capture: None,
                draft_logits: None,
            },
        })
    }

    fn finish_readout(
        &mut self,
        hidden: &B::Tensor,
        mode: ForwardMode,
        readout: TargetReadout,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        let resources = self.static_modules.decoder_resources_mut();
        match mode {
            ForwardMode::Target => {
                readout.apply(resources, hidden, parallel, context, instrumentation)
            }
            // Prediction units own their final norm; target observation seams do not apply.
            ForwardMode::Draft(_) => resources.project(
                hidden,
                parallel,
                context,
                &mut ComponentInstrumentation::disabled(),
            ),
        }
    }

    /// Applies the replicated target output boundary.
    pub fn finish_partition_target(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.finish_readout(
            hidden,
            ForwardMode::Target,
            TargetReadout::Logits,
            None,
            context,
            &mut ComponentInstrumentation::disabled(),
        )
    }

    /// Applies the tensor-parallel target output boundary.
    pub fn finish_partition_target_parallel(
        &mut self,
        hidden: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.finish_readout(
            hidden,
            ForwardMode::Target,
            TargetReadout::Logits,
            Some(parallel),
            context,
            &mut ComponentInstrumentation::disabled(),
        )
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
        self.static_modules.decoder_resources_mut().project(
            hidden,
            Some(parallel),
            context,
            &mut ComponentInstrumentation::disabled(),
        )
    }

    fn unit_path_inner(&self, group: usize, index: usize) -> Result<String, Error> {
        Ok(self.definition.unit(group, index)?.path.clone())
    }

    fn state_index(&self, group: usize, index: usize) -> Result<usize, Error> {
        Ok(self.definition.unit(group, index)?.state.start)
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
        P: ParameterProvider<B>,
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
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
        P: ParameterProvider<B>,
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
        P: eredu_runtime::TensorParallelParameterProvider<B>,
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

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> eredu_runtime::ComponentKernel<B>
    for CompleteNemotronKernel<B>
{
    type Input<'a> = EmbeddedInput<'a, B::Tensor>;
    type Unit = Unit<B>;
    type ForwardContext = ForwardContext<B::Tensor>;
    type RetainedContextValues<'a>
        = RetainedValues<'a, B::Tensor>
    where
        B::Tensor: 'a;
    type Error = Error;
    fn state_identity(
        &self,
        state: &eredu_runtime::PartitionState,
        topology: eredu_core::cache::PromptCacheTopology,
    ) -> Result<eredu_runtime::ModelStateIdentity, Self::Error> {
        super::state_identity(
            &self.args,
            state.layout(),
            state.global_layer_offset(),
            topology,
        )
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
    fn prediction_target_capture(forward: &Self::ForwardContext) -> Option<&B::Tensor> {
        forward.target_capture.as_ref()
    }
    fn retained_values<'a>(
        &'a self,
        forward: &'a Self::ForwardContext,
        _address: eredu_runtime::ExecutionUnitAddress,
    ) -> Self::RetainedContextValues<'a> {
        RetainedValues::new([
            Some(&forward.tokens),
            Some(&forward.embedded),
            forward.mask.as_ref(),
            forward.target_capture.as_ref(),
            forward.draft_logits.as_ref(),
        ])
    }
    fn executes_group(&self, group: usize, forward: &Self::ForwardContext) -> bool {
        match forward.mode {
            ForwardMode::Target => group == 0,
            ForwardMode::Draft(depth) => group == depth + 1,
        }
    }
    fn build_unit(
        &self,
        address: eredu_runtime::ExecutionUnitAddress,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Unit, Self::Error> {
        self.construct_unit(address.group(), address.index(), context)
    }
}
impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend>
    eredu_runtime::ComponentPinnedResources<B> for CompleteNemotronKernel<B>
{
    type Modules = StaticModules<B>;
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
impl<B, S> eredu_runtime::ComponentEquations<B, S> for CompleteNemotronKernel<B>
where
    B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
{
    fn begin<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error> {
        let expected = self.state_layout()?;
        if state.layout() != &expected {
            return Err(Error::backend(format!("Nemotron-H runtime state layout {:?} does not match architecture layout {expected:?}", state.layout())));
        }
        let input = match input {
            EmbeddedInput::TargetEmbedded {
                tokens,
                embedded,
                mask,
                readout,
            } => {
                let mut forward = self.begin_embedded_target(
                    tokens,
                    embedded.clone(),
                    mask,
                    state,
                    &expected,
                    context,
                )?;
                forward.context.readout = readout;
                return Ok(forward);
            }
            input => input,
        };
        let (tokens, supplied_hidden, supplied_mask, mode) = match input {
            EmbeddedInput::Target { tokens, mask } => (tokens, None, mask, ForwardMode::Target),
            EmbeddedInput::TargetEmbedded { .. } => unreachable!("supplied target handled above"),
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
                readout: TargetReadout::Logits,
                target_capture: None,
                draft_logits: None,
            },
        })
    }
    fn begin_observed<'a, O>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Self::Error> + ?Sized,
    {
        let mut forward = self.begin(input, state, context)?;
        if matches!(forward.context.mode, ForwardMode::Target) {
            forward = crate::decoder::ComponentInstrumentation::apply_embedding(forward, observer)?;
            forward.context.embedded = forward.hidden.clone();
        }
        Ok(forward)
    }
    fn begin_group(
        &mut self,
        _group: usize,
        initial: &B::Tensor,
        dependencies: &[&B::Tensor],
        _state: &mut S,
        _forward: &mut Self::ForwardContext,
        _context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        Ok(dependencies.first().copied().unwrap_or(initial).clone())
    }
    fn execute(
        &mut self,
        address: eredu_runtime::ExecutionUnitAddress,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        let (group, index) = (address.group(), address.index());
        self.forward_unit_with_provider(
            group,
            index,
            unit,
            hidden,
            state,
            forward,
            &mut eredu_runtime::ResidentExpertProvider,
            context,
        )
    }
    fn execute_observed<O>(
        &mut self,
        address: eredu_runtime::ExecutionUnitAddress,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Self::Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Self::Error> + ?Sized,
    {
        let (group, index) = (address.group(), address.index());
        self.forward_unit_observed_with_provider(
            group,
            index,
            unit,
            hidden,
            state,
            forward,
            observer,
            &mut eredu_runtime::ResidentExpertProvider,
            context,
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
        if (group == 0 && matches!(forward.mode, ForwardMode::Target))
            || matches!(forward.mode, ForwardMode::Draft(depth) if group == depth + 1)
        {
            forward.target_capture = Some(hidden.clone());
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
        self.finish_readout(
            hidden,
            forward.mode,
            forward.readout,
            None,
            context,
            &mut ComponentInstrumentation::disabled(),
        )
    }
    fn finish_observed<O>(
        &mut self,
        hidden: &B::Tensor,
        _state: &mut S,
        forward: &Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Self::Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Self::Error> + ?Sized,
    {
        let mut observer = eredu_runtime::BorrowedActivationObserver(observer);
        self.finish_readout(
            hidden,
            forward.mode,
            forward.readout,
            None,
            context,
            &mut ComponentInstrumentation::new("readout", &mut observer),
        )
    }
}
impl<B, S> eredu_runtime::ParallelComponentKernel<B, S> for CompleteNemotronKernel<B>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
{
    fn begin_parallel<'a>(
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
            EmbeddedInput::TargetEmbedded {
                tokens,
                embedded,
                mask,
                readout,
            } => {
                let mut forward = self.begin_embedded_target(
                    tokens,
                    embedded.clone(),
                    mask,
                    state,
                    &expected,
                    context,
                )?;
                forward.context.readout = readout;
                Ok(forward)
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
    fn execute_parallel(
        &mut self,
        address: eredu_runtime::ExecutionUnitAddress,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        let (group, index) = (address.group(), address.index());
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
    fn finish_parallel(
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
        self.finish_readout(
            hidden,
            forward.mode,
            forward.readout,
            Some(parallel),
            context,
            &mut ComponentInstrumentation::disabled(),
        )
    }
    fn finish_parallel_observed<O>(
        &mut self,
        hidden: &B::Tensor,
        _state: &mut S,
        forward: &Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Self::Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Self::Error> + ?Sized,
    {
        if self.parallel_geometry.is_none() {
            return Err(Error::backend(
                "Nemotron-H model was not built with rank-local geometry",
            ));
        }
        let mut observer = eredu_runtime::BorrowedActivationObserver(observer);
        self.finish_readout(
            hidden,
            forward.mode,
            forward.readout,
            Some(parallel),
            context,
            &mut ComponentInstrumentation::new("readout", &mut observer),
        )
    }
}
impl<B, S> eredu_runtime::RoutedComponentKernel<B, S> for CompleteNemotronKernel<B>
where
    B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
{
    fn routed_sparse_observations(&self) -> bool {
        true
    }
    fn execute_routed<P>(
        &mut self,
        address: eredu_runtime::ExecutionUnitAddress,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        _pass: eredu_runtime::ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error>
    where
        P: ParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        let (group, index) = (address.group(), address.index());
        CompleteNemotronKernel::forward_unit_with_provider(
            self, group, index, unit, hidden, state, forward, provider, context,
        )
    }
    fn execute_routed_observed<P, O>(
        &mut self,
        address: eredu_runtime::ExecutionUnitAddress,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        _pass: eredu_runtime::ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Self::Error>
    where
        P: ParameterProvider<B>,
        P::Error: std::fmt::Display,
        O: eredu_runtime::ActivationObserver<B::Tensor, Self::Error> + ?Sized,
    {
        let (group, index) = (address.group(), address.index());
        CompleteNemotronKernel::forward_unit_observed_with_provider(
            self, group, index, unit, hidden, state, forward, observer, provider, context,
        )
    }
}
impl<B, S> eredu_runtime::ParallelRoutedComponentKernel<B, S> for CompleteNemotronKernel<B>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
{
    fn execute_parallel_routed<P>(
        &mut self,
        address: eredu_runtime::ExecutionUnitAddress,
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
        P: eredu_runtime::TensorParallelParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        let (group, index) = (address.group(), address.index());
        CompleteNemotronKernel::forward_unit_parallel_with_provider(
            self, group, index, unit, hidden, state, forward, parallel, provider, context,
        )
    }
}
impl<B, S> eredu_runtime::PartitionComponentKernel<B, S> for CompleteNemotronKernel<B>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
{
    type Boundary = TargetBoundarySchema;
    fn boundary_schema(&self) -> Result<Self::Boundary, Self::Error> {
        Ok(TargetBoundarySchema::from_args(self.args()))
    }
    fn begin_partition(
        &mut self,
        input: LayeredPartitionInput<'_, B::Tensor, TargetBoundary<B::Tensor>>,
        mask: Option<&B::Tensor>,
        state: &mut S,
        selected: eredu_runtime::ComponentPartitionContext<'_, B>,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error> {
        let expected = selected.expected.ok_or_else(|| {
            Error::backend("Nemotron-H partition requires selected state geometry")
        })?;
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
    fn enter_partition_group(
        &mut self,
        group: usize,
        initial: &B::Tensor,
        _state: &mut S,
        _forward: &mut Self::ForwardContext,
        _parallel: Option<&B::ParallelContext>,
        _context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        if group != 0 {
            return Err(Error::backend("Nemotron-H target partition cannot enter a prediction dependency without its producer"));
        }
        Ok(initial.clone())
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
                auxiliary: TargetBoundary::new(
                    forward.tokens.clone(),
                    forward.embedded.clone(),
                    forward.readout.encode::<B::Tensor>(context)?,
                ),
            })
        }
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
    ) -> Result<LayeredPartitionOutput<B::Tensor, TargetBoundary<B::Tensor>>, Self::Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Self::Error> + ?Sized,
    {
        if !owns_output {
            return self.finish_partition(hidden, state, forward, owns_output, parallel, context);
        }
        let output = match parallel {
            Some(parallel) => {
                self.finish_parallel_observed(hidden, state, forward, parallel, context, observer)?
            }
            None => self.finish_observed(hidden, state, forward, context, observer)?,
        };
        Ok(LayeredPartitionOutput::Final {
            output,
            retained: Some(hidden.clone()),
        })
    }
}
impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend>
    eredu_runtime::TextComponentKernel<B> for CompleteNemotronKernel<B>
{
    fn text_input<'a>(tokens: &'a B::Tensor, mask: Option<&'a B::Tensor>) -> Self::Input<'a> {
        EmbeddedInput::target(tokens, mask)
    }
}
