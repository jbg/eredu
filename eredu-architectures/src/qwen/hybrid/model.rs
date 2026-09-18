//! Unified layered lifecycle for hybrid target and embedded-MTP execution.

use eredu_core::cache::PromptCacheTopology;
use eredu_nn::{
    AttentionCache, EmbeddingLookupPolicy, EmbeddingOperator, Error, GroupedNeuralBackend,
    LinearOperator, NormalizationOperator, Parameterized, Tensor,
};
use eredu_runtime::{
    ArchitectureParameterDescription, ExecutionUnitLayout, LayerRuntimeState, LayeredArchitecture,
    LayeredForwardState, LayeredPartitionInput, LayeredPartitionOutput, ModelStateIdentity,
    OwnedParameterGroupSpec, ParallelLayeredArchitecture, ParallelRoutedLayeredArchitecture,
    ParameterGroupOwner, PartitionedLayeredArchitecture, ResidentExpertProvider,
    RoutedExpertProvider, RoutedLayeredArchitecture, RuntimeStateComponents, StateLayout,
};

use crate::{
    decoder::{static_parallel_parameter_groups, StaticModuleSpec, StaticModules},
    hybrid_decoder::HybridDecoder,
};

use super::{
    state_layout, unit_parallel_parameter_groups, Block,
    EmbeddedInput, ForwardMode, HybridConfig, LocalGeometry, PredictionUnit,
};

/// One target or configured prediction execution unit.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub enum Unit<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> {
    /// Ordinary hybrid decoder block.
    Target(Block<B>),
    /// One embedded prediction depth.
    Prediction(PredictionUnit<B>),
}

impl<B, S> RoutedLayeredArchitecture<B, S> for LayeredModel<B>
where
    B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
{
    fn routed_unit_observations(&self) -> bool {
        self.prediction_steps == 0
    }

    fn routed_sparse_observations(&self) -> bool {
        self.prediction_steps == 0
    }

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

    fn forward_unit_observed_with_provider<P, O>(
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
        observer: &mut O,
    ) -> Result<B::Tensor, Self::Error>
    where
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
        O: eredu_runtime::ActivationObserver<B::Tensor, Self::Error> + ?Sized,
    {
        let path = self.decoder.unit_path(group, index, None)?;
        let lane = state
            .layer(self.state_index(group, index)?)
            .map_err(Error::backend)?;
        let points = eredu_runtime::RoutedObservationPoints::new(
            eredu_runtime::RoutedBankId::new(0),
            format_args!("{path}.mlp"),
            self.config.num_experts, None)?;
        match unit {
            Unit::Target(block) if group == 0 => block.forward_components_with_provider(
                &path,
                points,
                hidden,
                forward.mask.as_ref(),
                lane,
                context,
                provider,
                observer,
            ),
            Unit::Prediction(unit) if group > 0 => unit.forward_observed_with_provider(
                self.prediction_shared_mut()?,
                &path,
                points,
                hidden,
                &forward.embedded,
                forward.mask.as_ref(),
                lane,
                context,
                provider,
                observer,
            ),
            _ => Err(Error::backend(format!(
                "Qwen hybrid unit does not match execution group {group}"
            ))),
        }
    }
}

impl<B, S> ParallelRoutedLayeredArchitecture<B, S> for LayeredModel<B>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
{
    fn parallel_routed_unit_observations(&self) -> bool {
        self.prediction_steps == 0
    }

    fn parallel_routed_sparse_observations(&self) -> bool {
        self.prediction_steps == 0
    }

    fn forward_unit_parallel_observed_with_provider<P, O>(
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
        observer: &mut O,
    ) -> Result<B::Tensor, Error>
    where
        P: eredu_runtime::TensorParallelRoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        let path = self.decoder.unit_path(group, index, None)?;
        let lane = state
            .layer(self.state_index(group, index)?)
            .map_err(Error::backend)?;
        let points = eredu_runtime::RoutedObservationPoints::new(
            eredu_runtime::RoutedBankId::new(0),
            format_args!("{path}.mlp"),
            self.config.num_experts, None)?;
        match unit {
            Unit::Target(block) if group == 0 => block.forward_components_parallel_with_provider(
                &path,
                points,
                hidden,
                forward.mask.as_ref(),
                lane,
                parallel,
                context,
                provider,
                observer,
            ),
            Unit::Prediction(unit) if group > 0 => unit.forward_parallel_observed_with_provider(
                self.prediction_shared_mut()?,
                &path,
                points,
                hidden,
                &forward.embedded,
                forward.mask.as_ref(),
                lane,
                parallel,
                context,
                provider,
                observer,
            ),
            _ => Err(Error::backend(format!(
                "Qwen hybrid unit does not match execution group {group}"
            ))),
        }
    }

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
        P: eredu_runtime::TensorParallelRoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        LayeredModel::forward_unit_with_provider_parallel(
            self, group, index, unit, hidden, state, forward, provider, parallel, context,
        )
    }
}

/// Request-local values retained through one target or prediction pass.
pub struct ForwardContext<T> {
    tokens: T,
    embedded: T,
    mask: Option<T>,
    mode: ForwardMode,
    target_hidden: Option<T>,
}

/// Typed target input for one pipeline partition.
pub enum TargetPartitionInput<'a, T> {
    /// Token identities owned by the first target partition.
    Tokens(&'a T),
    /// Embedded activation received from an upstream target partition.
    Hidden(T),
}

impl<T> ForwardContext<T> {
    /// Current target or prediction selection.
    pub const fn mode(&self) -> ForwardMode {
        self.mode
    }

    /// Hidden state captured at the selected execution-group boundary.
    pub const fn target_hidden(&self) -> Option<&T> {
        self.target_hidden.as_ref()
    }
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> LayeredModel<B> {
    /// Resumes an already embedded target partition in the canonical layered context.
    pub fn resume_target_partition(
        &self,
        hidden: B::Tensor,
        mask: Option<B::Tensor>,
    ) -> LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>> {
        LayeredForwardState {
            context: ForwardContext {
                tokens: hidden.clone(),
                embedded: hidden.clone(),
                mask,
                mode: ForwardMode::Target,
                target_hidden: None,
            },
            hidden,
        }
    }

    /// Enters or resumes one routed target partition with architecture-owned
    /// embedding and causal-mask construction.
    pub fn begin_routed_target_partition(
        &mut self,
        input: TargetPartitionInput<'_, B::Tensor>,
        explicit_mask: Option<&B::Tensor>,
        sequence: i32,
        offset: i32,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>, Error> {
        let hidden = match input {
            TargetPartitionInput::Tokens(tokens) => match parallel {
                Some(parallel) => {
                    self.begin_partition_embedding_parallel(tokens, parallel, context)?
                }
                None => self.begin_partition_embedding(tokens, context)?,
            },
            TargetPartitionInput::Hidden(hidden) => hidden,
        };
        let mask = match explicit_mask {
            Some(mask) => Some(mask.clone()),
            None if sequence > 1 => Some(B::causal_mask(sequence, offset, None, context)?),
            None => None,
        };
        Ok(self.resume_target_partition(hidden, mask))
    }
}

/// One neutral layered model for Qwen3-Next and every Qwen3.5 text policy.
pub struct LayeredModel<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> {
    config: crate::replicated_text::config_source::ConfigOwner<HybridConfig>,
    construction_parameters: Option<crate::routed_text::RetainedRoutedDescription>,
    construction_units: Option<crate::routed_text::RetainedRoutedUnits>,
    decoder: HybridDecoder<B, Option<super::PredictionShared<B>>>,
    target_layers: usize,
    prediction_steps: usize,
    parallel_geometry: Option<std::sync::Arc<LocalGeometry>>,
    partition_target_start: usize,
    global_parameters: Option<std::sync::Arc<ArchitectureParameterDescription>>,
    expert_realization: Option<crate::ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>>,
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend>
    crate::routed_text::RoutedConstructionParameters<B> for LayeredModel<B>
{
    fn install_construction_parameters(&mut self, source: crate::routed_text::RetainedRoutedDescription) {
        self.construction_parameters = Some(source);
    }
    fn prepare_construction_units(
        &self, banks: Option<&crate::routed_text::RetainedRoutedBanks>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Option<crate::routed_text::RetainedRoutedUnits>, Error> {
        super::block::construction::require_source_compiler::<B>(context)?;
        let selected = banks.map(|banks| banks.get(&eredu_runtime::RoutedBankId::new(0))
            .and_then(|bank| bank.plan().gated())
            .ok_or_else(|| Error::from(eredu_nn::workspace::WorkspaceMetadataError::Unqualified)))
            .transpose()?;
        let mut specs = Vec::with_capacity(self.target_layers);
        for index in 0..self.target_layers {
            let config = self.target_unit_config(index);
            let realization = selected.or(self.expert_realization.as_ref());
            let spec = realization.map(|plan| plan.unit_spec(crate::decoder::TARGET_EXECUTION_GROUP, index)
                .cloned().ok_or_else(|| Error::backend("Qwen partition has no selected expert unit")))
                .transpose()?;
            specs.push(super::block::TargetBlockSpec::new(config, index, spec)?);
        }
        Ok(Some(crate::routed_text::RetainedRoutedUnits::qwen_source(specs)))
    }
    fn install_construction_units(&mut self, source: Option<crate::routed_text::RetainedRoutedUnits>) -> Result<(), Error> {
        if let Some(source) = &source {
            if source.qwen()?.len() != self.target_layers {
                return Err(eredu_nn::workspace::WorkspaceMetadataError::Unqualified.into());
            }
        }
        self.construction_units = source;
        Ok(())
    }
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend>
    eredu_runtime::ArchitectureParameters<B> for LayeredModel<B>
{
    type DefinitionError = Error;


    fn state_layout(&self, context: Option<&eredu_nn::workspace::WorkspaceContext>)
        -> Result<StateLayout, Self::DefinitionError> {
match context { Some(context) => {
        match &self.parallel_geometry {
            Some(geometry) => geometry.state_layout().clone_workspace(context),
            None => super::state_layout_with_metadata(&self.config, context),
        }
    }, None => {
        self.state_layout_impl()
    } }
}
    fn state_identity(&self, state: &eredu_runtime::PartitionState,
        topology: eredu_core::cache::PromptCacheTopology,
        context: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<ModelStateIdentity, Self::DefinitionError> {
match context { Some(context) => {
        state_identity_with_metadata(&self.config, state.layout(), state.global_layer_offset(), topology, context)
    }, None => {
        state_identity(
            &self.config,
            state.layout(),
            state.global_layer_offset(),
            topology,
        )
    } }
}



    fn static_parameter_recipes(
        &self,
        source: &dyn eredu_checkpoint::store::CheckpointSource,
    ) -> Result<
        std::collections::BTreeMap<String, eredu_checkpoint::recipe::DerivedWeightRecipe>,
        String,
    > {
        let recipes = super::static_recipes(source)?;
        crate::static_parameters::module_recipes(self.decoder.extended_static_modules(), recipes)
    }

    fn parameter_description(&self,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<std::borrow::Cow<'_, eredu_runtime::ArchitectureParameterDescription>, Self::DefinitionError> {
        match &self.construction_parameters {
            Some(source) => {
                crate::decoder::ModuleMetadata::new::<B>(context)
                    .controls::<std::borrow::Cow<'_, eredu_runtime::ArchitectureParameterDescription>>()?;
                Ok(std::borrow::Cow::Borrowed(source))
            }
            None if B::construction_metadata(context).is_some_and(|metadata| metadata.uses_checked_metadata()) => {
                Err(eredu_nn::workspace::WorkspaceMetadataError::Unqualified.into())
            }
            None => (|| {
        self.parameter_description_impl(context)
    })().map(std::borrow::Cow::Owned),
        }
    }

    fn retained_static_value_slot_bound(&self) -> Option<usize> {
        eredu_nn::Parameterized::retained_value_slot_bound(self.decoder.extended_static_modules())
    }

    fn visit_retained_static_values(&self, visitor: &mut dyn FnMut(&B::Tensor)) -> bool {
        eredu_nn::Parameterized::visit_retained_values(
            self.decoder.extended_static_modules(),
            visitor,
        )
    }

    fn visit_static_parameters<V>(&self, visitor: &mut V) -> Result<(), V::Error>
    where
        V: eredu_runtime::StaticParameterVisitor<B>,
    {
        if let Some(shared) = &self.decoder.extended_static_modules().extension {
            visitor.visit("mtp", shared)?;
        }
        let modules = self.decoder.static_modules();
        visitor.visit("embedding", &modules.embeddings)?;
        visitor.visit("norm", &modules.norm)?;
        if let Some(head) = &modules.lm_head {
            visitor.visit("output", head)?;
        }
        Ok(())
    }

    fn visit_static_parameters_mut<V>(&mut self, visitor: &mut V) -> Result<(), V::Error>
    where
        V: eredu_runtime::StaticParameterVisitorMut<B>,
    {
        if let Some(shared) = &mut self.decoder.extended_static_modules_mut().extension {
            visitor.visit_mut("mtp", shared)?;
        }
        let modules = self.decoder.static_modules_mut();
        visitor.visit_mut("embedding", &mut modules.embeddings)?;
        visitor.visit_mut("norm", &mut modules.norm)?;
        if let Some(head) = &mut modules.lm_head {
            visitor.visit_mut("output", head)?;
        }
        Ok(())
    }
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> LayeredModel<B> {
    /// Builds unloaded pinned modules and the exact configured graph.
    pub fn new(config: HybridConfig, context: &<B::Tensor as Tensor>::Context) -> Result<Self, Error> {
        Self::new_with_config(config.into(),context)
    }
    pub(crate) fn new_with_config(config:crate::replicated_text::config_source::ConfigOwner<HybridConfig>,
        context:&<B::Tensor as Tensor>::Context)->Result<Self,Error>{
        crate::decoder::ModuleMetadata::new::<B>(context).controls::<Self>()?;
        let (decoder, target_layers, prediction_steps) = Self::prepare_decoder(&config, context)?;
        Ok(Self {
            config,
            decoder,
            target_layers,
            prediction_steps,
            construction_parameters: None,
            construction_units: None,
            parallel_geometry: None,
            partition_target_start: 0,
            global_parameters: None,
            expert_realization: None,
        })
    }

    /// Builds the same validated pinned decoder from the caller's immutable
    /// configuration, then moves only its modules to composite construction.
    pub(crate) fn prepare_static_modules(
        config: &HybridConfig,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<crate::hybrid_decoder::HybridStaticModules<B, Option<super::PredictionShared<B>>>, Error>
    {
        crate::decoder::ModuleMetadata::new::<B>(context).controls::<
            crate::hybrid_decoder::HybridStaticModules<B, Option<super::PredictionShared<B>>>,
        >()?;
        let (decoder, _, _) = Self::prepare_decoder(config, context)?;
        Ok(decoder.into_extended_static_modules())
    }

    fn prepare_decoder(
        config: &HybridConfig,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<
        (
            HybridDecoder<B, Option<super::PredictionShared<B>>>,
            usize,
            usize,
        ),
        Error,
    > {
        let module_metadata = crate::decoder::ModuleMetadata::new::<B>(context);
        let metadata = crate::decoder::identity::Metadata::new(B::construction_metadata(context));
        module_metadata.controls::<(
            HybridDecoder<B, Option<super::PredictionShared<B>>>,
            usize,
            usize,
            StaticModuleSpec,
        )>()?;
        module_metadata.require::<B>("Qwen hybrid", crate::operator_requirements::QWEN_HYBRID)?;
        match metadata.context() {
            Some(context) => {
                config.validate_with_diagnostic(|message| context.metadata_error(message))?
            }
            None => config.validate().map_err(Error::backend)?,
        }
        let target_layers =
            usize::try_from(config.num_hidden_layers).map_err(|cause| metadata.source(cause))?;
        let prediction_steps =
            usize::try_from(config.mtp_num_hidden_layers).map_err(|cause| metadata.source(cause))?;
        let embedding_name = "model.embed_tokens.weight";
        let decoder = HybridDecoder::new_with_prediction_groups(
            StaticModuleSpec {
                normalization_groups: None,
                embedding_weight: metadata.text(embedding_name)?,
                normalization_weight: metadata.text("model.norm.weight")?,
                head_weight: metadata.text("lm_head.weight")?,
                vocabulary: config.vocab_size,
                hidden_size: config.hidden_size,
                normalization_epsilon: config.rms_norm_eps,
                normalization_offset: 1.0,
                embedding_quantization: config
                    .linear_format("model.embed_tokens.weight")
                    .weight_quantization(),
                head_format: config.linear_format("lm_head.weight"),
                tied_head: config.tie_word_embeddings,
            },
            "model.layers",
            target_layers,
            "mtp.layers",
            prediction_steps,
            1,
            context,
        )?
        .with_static_extension(
            (prediction_steps > 0)
                .then(|| super::PredictionShared::new(config, context))
                .transpose()?,
        );
        Ok((decoder, target_layers, prediction_steps))
    }

    /// Applies the architecture-owned tensor-parallel token embedding boundary.
    pub fn pipeline_embed_parallel(
        &mut self,
        tokens: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        B::vocabulary_parallel_lookup(
            &mut self.decoder.static_modules_mut().embeddings,
            tokens,
            EmbeddingLookupPolicy::Strict,
            parallel,
            context,
        )
    }

    /// Applies the architecture-owned tensor-parallel output boundary.
    pub fn pipeline_finish_parallel(
        &mut self,
        hidden: &B::Tensor,
        normalize: bool,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let hidden = if normalize {
            self.decoder
                .static_modules_mut()
                .norm
                .forward(hidden, context)?
        } else {
            hidden.clone()
        };
        let modules = self.decoder.static_modules_mut();
        match &mut modules.lm_head {
            Some(head) => B::vocabulary_parallel_project(head, &hidden, parallel, context),
            None => B::vocabulary_parallel_embedding_project(
                &mut modules.embeddings,
                &hidden,
                parallel,
                context,
            ),
        }
    }

    /// Enters a serial target or prediction partition through token embedding.
    pub fn begin_partition_embedding(
        &mut self,
        tokens: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.decoder
            .static_modules_mut()
            .embeddings
            .forward(tokens, context)
    }

    /// Enters a tensor-parallel target or prediction partition through token embedding.
    pub fn begin_partition_embedding_parallel(
        &mut self,
        tokens: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.pipeline_embed_parallel(tokens, parallel, context)
    }

    fn finish_partition_output(
        &mut self,
        hidden: &B::Tensor,
        normalize: bool,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let hidden = if normalize {
            self.decoder
                .static_modules_mut()
                .norm
                .forward(hidden, context)?
        } else {
            hidden.clone()
        };
        let modules = self.decoder.static_modules_mut();
        match &mut modules.lm_head {
            Some(head) => head.forward(&hidden, context),
            None => modules.embeddings.as_linear(&hidden, context),
        }
    }

    /// Finishes a serial target partition with final normalization.
    pub fn finish_partition_target(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.finish_partition_output(hidden, true, context)
    }

    /// Finishes a tensor-parallel target partition with final normalization.
    pub fn finish_partition_target_parallel(
        &mut self,
        hidden: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.pipeline_finish_parallel(hidden, true, parallel, context)
    }

    /// Builds the canonical target/MTP graph with planner-derived local modules.
    pub fn new_parallel(
        config: HybridConfig,
        geometry: LocalGeometry,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        crate::operator_requirements::require::<B>(
            "Qwen hybrid tensor parallelism",
            eredu_nn::NeuralOperatorCapabilities::SUM_PARALLEL,
        )?;
        geometry.validate_for(&config).map_err(Error::backend)?;
        let mut model = Self::new(config.clone(), context)?;
        *model.decoder.static_modules_mut() = crate::decoder::StaticModules::from_parallel_spec(
            static_spec(&config)?,
            geometry.embedding_range().clone(),
            geometry.output_range().cloned(),
            context,
        )?;
        model.parallel_geometry = Some(std::sync::Arc::new(geometry));
        Ok(model)
    }

    /// Returns normalized hybrid policy.
    pub fn config(&self) -> &HybridConfig {
        &self.config
    }

    /// Returns the architecture-owned target and prediction traversal.
    pub fn unit_layout(&self) -> Result<ExecutionUnitLayout, Error> {
        let graph = self.decoder.execution_graph()?.into_owned();
        let counts = (0..graph.groups().len())
            .map(|group| self.decoder.group_unit_count(group, None))
            .collect::<Result<Vec<_>, _>>()?;
        ExecutionUnitLayout::new(&graph, counts).map_err(Error::backend)
    }

    /// Describes target and embedded-prediction parameters without deriving
    /// ownership from checkpoint-name substrings.
    fn parameter_description_impl(
        &self,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<ArchitectureParameterDescription, Error> {
        if let Some(parameters) = &self.global_parameters {
            return Ok(parameters.as_ref().clone());
        }
        if self.parallel_geometry.is_some() {
            return Self::new((*self.config).clone(), context)?.parameter_description_impl(context);
        }
        let graph = self.decoder.execution_graph()?.into_owned();
        let layout = self.unit_layout()?;
        let static_groups = static_parallel_parameter_groups::<B>(
            &self.decoder.static_modules().embeddings,
            &self.decoder.static_modules().norm,
            self.decoder.static_modules().lm_head.as_ref(),
            "model",
        )
        .map_err(Error::backend)?;
        let mut expected = static_groups.clone();
        let mut owned = static_groups
            .into_iter()
            .enumerate()
            .map(|(index, group)| {
                OwnedParameterGroupSpec::new(
                    if index == 0 {
                        let mut roles = vec!["embedding", "mtp"];
                        if self.config.tie_word_embeddings {
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
        if let Some(shared) = &self.decoder.extended_static_modules().extension {
            let groups =
                super::parallel::prediction_shared_parameter_groups(shared, self.prediction_steps)
                    .map_err(Error::backend)?;
            expected.extend(groups.iter().map(|group| group.group().clone()));
            owned.extend(groups);
        }
        for group_index in 0..graph.groups().len() {
            let group_id = layout
                .group_id(group_index)
                .expect("hybrid layout contains every graph group")
                .clone();
            let count = layout
                .group_range(group_index)
                .expect("hybrid layout contains every group range")
                .len();
            for index in 0..count {
                let unit = self.construct_unit(group_index, index, context)?;
                let groups =
                    unit_parallel_parameter_groups(&unit, &self.config, group_index, index)
                        .map_err(Error::backend)?;
                expected.extend(groups.iter().cloned());
                owned.extend(groups.into_iter().map(|group| {
                    OwnedParameterGroupSpec::new(
                        ParameterGroupOwner::execution_unit(group_id.clone(), index),
                        group,
                    )
                }));
            }
        }
        ArchitectureParameterDescription::new(&graph, &layout, expected, owned)
            .map_err(Error::backend)
    }

    /// Returns the pinned shared prediction modules and base text modules.
    pub fn into_extended_static_modules(
        self,
    ) -> crate::hybrid_decoder::HybridStaticModules<B, Option<super::PredictionShared<B>>> {
        self.decoder.into_extended_static_modules()
    }

    fn prediction_shared_mut(&mut self) -> Result<&mut super::PredictionShared<B>, Error> {
        self.decoder
            .extended_static_modules_mut()
            .extension
            .as_mut()
            .ok_or_else(|| Error::backend("Qwen prediction has no shared modules"))
    }

    /// Borrows pinned text modules.
    pub const fn static_modules(&self) -> &StaticModules<B> {
        self.decoder.static_modules()
    }

    /// Mutably borrows pinned text modules for checkpoint binding.
    pub fn static_modules_mut(&mut self) -> &mut StaticModules<B> {
        self.decoder.static_modules_mut()
    }

    /// Consumes the text graph and returns its pinned decoder modules.
    pub fn into_static_modules(self) -> StaticModules<B> {
        self.decoder.into_static_modules()
    }

    /// Returns the prediction depth count declared by the constructed execution graph.
    pub fn mtp_len(&self) -> usize {
        self.decoder.prediction_count()
    }

    /// Returns target plus configured prediction state policy.
    pub fn state_layout(&self) -> Result<StateLayout, Error> {
        state_layout(&self.config).map_err(Error::backend)
    }

    /// Returns replicated or planner-derived mutable state geometry.
    fn state_layout_impl(&self) -> Result<StateLayout, Error> {
        self.parallel_geometry
            .as_ref()
            .map(|geometry| geometry.state_layout().clone())
            .map_or_else(|| self.state_layout(), Ok)
    }

    /// Shares authoritative local geometry with placed residency factories.
    pub fn shared_parallel_geometry(&self) -> Option<std::sync::Arc<LocalGeometry>> {
        self.parallel_geometry.as_ref().map(std::sync::Arc::clone)
    }

    fn target_unit_config(&self, index: usize) -> &HybridConfig {
        self.parallel_geometry.as_ref().and_then(|geometry| geometry.target(index))
            .unwrap_or(&self.config)
    }

    /// Constructs one canonical target or prediction unit using this model's
    /// replicated or planner-derived local geometry.
    pub fn construct_unit(
        &self,
        group: usize,
        index: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Unit<B>, Error> {
        if group == 0 {
            if let Some(source) = &self.construction_units {
                let metadata = crate::decoder::ModuleMetadata::new::<B>(context);
                metadata.controls::<(&crate::routed_text::RetainedRoutedUnits, Unit<B>, usize)>()?;
                let spec = source.qwen()?.get(index).ok_or_else(|| metadata.error(format_args!(
                    "target unit {index} is outside the retained source")))?;
                return spec.instantiate::<B>(self.target_unit_config(index), context).map(Unit::Target);
            }
        }
        self.decoder.unit_path(group, index, None)?;
        if group == 0 {
            let config = self
                .parallel_geometry
                .as_ref()
                .and_then(|geometry| geometry.target(index))
                .unwrap_or(&self.config);
            let spec = self
                .expert_realization
                .as_ref()
                .map(|plan| {
                    plan.unit_spec(crate::decoder::TARGET_EXECUTION_GROUP, index)
                        .cloned()
                        .ok_or_else(|| Error::backend("Qwen partition has no selected expert unit"))
                })
                .transpose()?;
            Ok(Unit::Target(Block::new_with_routed_spec(
                config, index, spec, context,
            )?))
        } else {
            let config = self
                .parallel_geometry
                .as_ref()
                .and_then(|geometry| geometry.prediction(group - 1))
                .unwrap_or(&self.config);
            Ok(Unit::Prediction(PredictionUnit::new(
                config,
                group - 1,
                context,
            )?))
        }
    }

    pub(crate) fn retain_global_parameters(
        &mut self,
        parameters: ArchitectureParameterDescription,
    ) {
        self.global_parameters = Some(std::sync::Arc::new(parameters));
    }

    pub(crate) fn install_expert_realization(
        &mut self,
        realization: crate::ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>,
    ) {
        self.construction_units = None;
        self.construction_parameters = None;
        self.expert_realization = Some(realization);
    }

    /// Binds global target-unit addresses to this partition's local cache lanes.
    pub(crate) fn set_partition_target_start(&mut self, start: usize) {
        self.partition_target_start = start;
    }

    fn state_index(&self, group: usize, index: usize) -> Result<usize, Error> {
        if group == 0 {
            return index
                .checked_sub(self.partition_target_start)
                .ok_or_else(|| Error::backend("Qwen target unit precedes its partition state"));
        }
        if index != 0 || group > self.prediction_steps {
            return Err(Error::backend(format!(
                "Qwen hybrid execution address ({group}, {index}) is outside its graph"
            )));
        }
        self.target_layers
            .checked_add(group - 1)
            .ok_or_else(|| Error::backend("Qwen hybrid state index overflowed"))
    }

    /// Executes one graph unit through a runtime-owned routed-expert provider.
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
        P: eredu_runtime::RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        self.decoder.unit_path(group, index, None)?;
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
                self.prediction_shared_mut()?,
                hidden,
                &forward.embedded,
                forward.mask.as_ref(),
                state.layer(state_index).map_err(Error::backend)?,
                context,
                provider,
            ),
            _ => Err(Error::backend(format!(
                "Qwen hybrid unit does not match execution group {group}"
            ))),
        }
    }

    /// Executes one target or prediction unit with local projections and collectives.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_unit_with_provider_parallel<S, P>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Unit<B>,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut ForwardContext<B::Tensor>,
        provider: &mut P,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
        P: eredu_runtime::TensorParallelRoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        self.decoder.unit_path(group, index, None)?;
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
            Unit::Prediction(unit) if group > 0 => unit.forward_parallel(
                self.prediction_shared_mut()?,
                hidden,
                &forward.embedded,
                forward.mask.as_ref(),
                state.layer(state_index).map_err(Error::backend)?,
                parallel,
                context,
                provider,
            ),
            _ => Err(Error::backend(format!(
                "Qwen hybrid unit does not match execution group {group}"
            ))),
        }
    }
}

fn static_spec(config: &HybridConfig) -> Result<StaticModuleSpec, Error> {
    Ok(StaticModuleSpec {
        normalization_groups: None,
        embedding_weight: "model.embed_tokens.weight".into(),
        normalization_weight: "model.norm.weight".into(),
        head_weight: "lm_head.weight".into(),
        vocabulary: config.vocab_size,
        hidden_size: config.hidden_size,
        normalization_epsilon: config.rms_norm_eps,
        normalization_offset: 1.0,
        embedding_quantization: config
            .linear_format("model.embed_tokens.weight")
            .weight_quantization(),
        head_format: config.linear_format("lm_head.weight"),
        tied_head: config.tie_word_embeddings,
    })
}

impl<B, S> LayeredArchitecture<B, S> for LayeredModel<B>
where
    B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
{
    fn prefill_observation_declarations(
        &self, metadata_context: Option<&eredu_nn::workspace::WorkspaceContext>) -> Result<Vec<eredu_runtime::layered::PrefillObservationDeclaration>, Self::Error> {
        let metadata=crate::decoder::identity::Metadata::new(metadata_context);
        metadata.controls::<(&Self,Option<&eredu_nn::workspace::WorkspaceContext>,usize,usize,String,Vec<eredu_runtime::layered::PrefillObservationDeclaration>,std::ops::Range<usize>,Option<eredu_runtime::RoutedObservationPoints>,Result<Vec<eredu_runtime::layered::PrefillObservationDeclaration>,Error>)>()?;

        // Target recurrence carries its complete convolution/matrix state;
        // attention uses the same causal KV prefix and fixed rotary positions.
        // Dense/shared/routed FFNs are row-local. Prediction availability and
        // its separate hooks remain governed by observation_hooks below.
        let units = self.decoder.group_unit_count(0, metadata_context)?;
        let mut declarations = crate::decoder::ordinary_prefill_observation_declarations(
            (0..units).map(|index| self.decoder.unit_path(0, index, metadata_context)),
            true, metadata_context)?;
        // Same target bank invocation as observed execution; its expert equations are row-local.
        for index in 0..units {
            let path = self.decoder.unit_path(0, index, metadata_context)?;
            if self.config.num_experts > 0 {
                crate::decoder::append_routed_prefill_path(&mut declarations, &metadata.format(format_args!("{path}.mlp"))?, metadata_context)?;
            }
        }
        Ok(declarations)
    }

    fn observation_hooks(&self) -> eredu_runtime::inspection::ObservationHookSupport {
        let target_only = self.prediction_steps == 0;
        eredu_runtime::inspection::ObservationHookSupport::internal(
            target_only,
            target_only,
            target_only,
        )
    }

    type Input<'a> = EmbeddedInput<'a, B::Tensor>;

    fn inference_input_shape(input: &Self::Input<'_>) -> Result<Option<[u64; 2]>, Self::Error> {
        match input {
            EmbeddedInput::Target { tokens, .. } => crate::prefill::token_shape(*tokens).map(Some),
            _ => Ok(None),
        }
    }

    type StaticModules =
        crate::hybrid_decoder::HybridStaticModules<B, Option<super::PredictionShared<B>>>;
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

    fn primary_execution_group(&self) -> &str {
        crate::decoder::TARGET_EXECUTION_GROUP
    }

    fn prediction_execution_groups(&self) -> Vec<String> {
        self.decoder.prediction_execution_groups()
    }

    fn state_partition_plan(
        &self,
        layout: &eredu_runtime::StateLayout,
    ) -> eredu_runtime::ArchitectureStatePartitionPlan {
        crate::transport::pipeline_with_output_state(0, self.target_layers, layout)
    }

    fn execution_graph(&self) -> Result<eredu_runtime::ArchitectureExecutionGraph<'_>, Self::Error> {
        self.decoder.execution_graph()
    }


    fn group_unit_count(&self, group: usize, metadata_context: Option<&eredu_nn::workspace::WorkspaceContext>) -> Result<usize, Self::Error> {
        let metadata = crate::decoder::ModuleMetadata::destination(metadata_context);
        metadata.controls::<(&Self, usize, Option<&eredu_nn::workspace::WorkspaceContext>)>()?;

        self.decoder.group_unit_count(group, metadata_context)
    }

    fn unit_path(&self, group: usize, index: usize, metadata_context: Option<&eredu_nn::workspace::WorkspaceContext>) -> Result<String, Self::Error> {
        let metadata = crate::decoder::ModuleMetadata::destination(metadata_context);
        metadata.controls::<(&Self, usize, usize, Option<&eredu_nn::workspace::WorkspaceContext>)>()?;

        self.decoder.unit_path(group, index, metadata_context)
    }

    fn static_modules(&self) -> &Self::StaticModules {
        self.decoder.extended_static_modules()
    }

    fn static_modules_mut(&mut self) -> &mut Self::StaticModules {
        self.decoder.extended_static_modules_mut()
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
            return Err(Error::backend(format!(
                "Qwen hybrid runtime state layout {:?} does not match {expected:?}",
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
                if depth >= self.prediction_steps {
                    return Err(Error::backend(format!(
                        "Qwen hybrid MTP depth {depth} is outside {} configured groups",
                        self.prediction_steps
                    )));
                }
                (tokens, Some(hidden), None, ForwardMode::Draft(depth))
            }
        };
        let embedded = self
            .decoder
            .static_modules_mut()
            .embeddings
            .forward(tokens, context)?;
        let hidden = supplied_hidden.cloned().unwrap_or_else(|| embedded.clone());
        let position_layer = match mode {
            ForwardMode::Target => 0,
            ForwardMode::Draft(depth) => self.target_layers + depth,
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
                target_hidden: None,
            },
        })
    }

    fn begin_forward_observed<'a, O>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Self::Error> + ?Sized,
    {
        let mut forward = self.begin_forward(input, state, context)?;
        if matches!(forward.context.mode, ForwardMode::Target) {
            let mut borrowed = eredu_runtime::BorrowedActivationObserver(observer);
            forward.hidden =
                crate::decoder::ComponentInstrumentation::new("readout", &mut borrowed)
                    .apply("embedding", forward.hidden)?;
            forward.context.embedded = forward.hidden.clone();
        }
        Ok(forward)
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
        self.decoder.begin_group(group, initial, dependencies)
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
        self.forward_unit_with_provider(
            group,
            index,
            unit,
            hidden,
            state,
            forward,
            &mut ResidentExpertProvider,
            context,
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
    ) -> Result<B::Tensor, Self::Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Self::Error> + ?Sized,
    {
        <Self as RoutedLayeredArchitecture<B, S>>::forward_unit_observed_with_provider(
            self,
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
        _state: &mut S,
        forward: &mut Self::ForwardContext,
        _context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        if group == 0 || matches!(forward.mode, ForwardMode::Draft(depth) if group == depth + 1) {
            forward.target_hidden = Some(hidden.clone());
        }
        Ok(hidden.clone())
    }

    fn prediction_target_capture(forward: &Self::ForwardContext) -> Option<&B::Tensor> {
        matches!(forward.mode, ForwardMode::Target)
            .then_some(forward.target_hidden.as_ref())
            .flatten()
    }

    fn prediction_target_placeholder_shape(
        &self,
        forward: &Self::ForwardContext,
    ) -> Result<Option<Vec<i32>>, Self::Error> {
        Ok(Some(vec![
            forward.embedded.dim(0),
            forward.embedded.dim(1),
            self.config.hidden_size,
        ]))
    }

    fn select_readout_positions(
        &self,
        hidden: &B::Tensor,
        _forward: &Self::ForwardContext,
        demand: eredu_core::OutputDemand,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Option<B::Tensor>, Self::Error> {
        crate::readout::select_readout_positions(hidden, demand, 1, context)
    }

    fn finish_forward(
        &mut self,
        hidden: &B::Tensor,
        _state: &mut S,
        forward: &Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        match forward.mode {
            ForwardMode::Target => self.decoder.finish_logits(hidden, context),
            ForwardMode::Draft(_) => self.decoder.project_logits(hidden, context),
        }
    }

    fn finish_forward_observed<O>(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Self::Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Self::Error> + ?Sized,
    {
        if matches!(forward.mode, ForwardMode::Target) {
            let mut borrowed = eredu_runtime::BorrowedActivationObserver(observer);
            self.decoder.finish_logits_instrumented(
                hidden,
                context,
                &mut crate::decoder::ComponentInstrumentation::new("readout", &mut borrowed),
            )
        } else {
            self.finish_forward(hidden, state, forward, context)
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
            forward.target_hidden.as_ref(),
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
mod transport_tests {
    use super::prediction_group_transport;

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
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
{
    fn parallel_observation_hooks(&self) -> eredu_runtime::inspection::ObservationHookSupport {
        let target_only = self.prediction_steps == 0;
        eredu_runtime::inspection::ObservationHookSupport::internal(
            target_only,
            target_only,
            target_only,
        )
    }

    fn begin_forward_parallel_observed<'a, O>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Self::Error> + ?Sized,
    {
        let mut forward = self.begin_forward_parallel(input, state, parallel, context)?;
        if matches!(forward.context.mode, ForwardMode::Target) {
            let mut borrowed = eredu_runtime::BorrowedActivationObserver(observer);
            forward.hidden =
                crate::decoder::ComponentInstrumentation::new("readout", &mut borrowed)
                    .apply("embedding", forward.hidden)?;
            forward.context.embedded = forward.hidden.clone();
        }
        Ok(forward)
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
        <Self as ParallelRoutedLayeredArchitecture<B, S>>::forward_unit_parallel_observed_with_provider(
            self, group, index, unit, hidden, state, forward,
            if hidden.dim(1) == 1 { eredu_runtime::ExpertPass::Decode } else { eredu_runtime::ExpertPass::Prefill },
            &mut ResidentExpertProvider, parallel, context, observer,
        )
    }

    fn finish_forward_parallel_observed<O>(
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
        if self.parallel_geometry.is_none() {
            return Err(Error::backend("Qwen hybrid model has no local geometry"));
        }
        if matches!(forward.mode, ForwardMode::Target) {
            let mut borrowed = eredu_runtime::BorrowedActivationObserver(observer);
            self.decoder
                .static_modules_mut()
                .finish_parallel_instrumented(
                    hidden,
                    parallel,
                    context,
                    &mut crate::decoder::ComponentInstrumentation::new("readout", &mut borrowed),
                )
        } else {
            self.finish_forward_parallel(hidden, state, forward, parallel, context)
        }
    }

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
            .ok_or_else(|| Error::backend("Qwen hybrid model has no local geometry"))?
            .state_layout();
        if state.layout() != expected {
            return Err(Error::backend(
                "Qwen hybrid rank-local state layout mismatch",
            ));
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
                        "Qwen hybrid MTP depth {depth} is outside {} configured groups",
                        self.prediction_steps
                    )));
                }
                (tokens, Some(hidden), None, ForwardMode::Draft(depth))
            }
        };
        let embedded = B::vocabulary_parallel_lookup(
            &mut self.decoder.static_modules_mut().embeddings,
            tokens,
            EmbeddingLookupPolicy::Strict,
            parallel,
            context,
        )?;
        let hidden = supplied_hidden.cloned().unwrap_or_else(|| embedded.clone());
        let position_layer = match mode {
            ForwardMode::Target => 0,
            ForwardMode::Draft(depth) => self.target_layers + depth,
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
                target_hidden: None,
            },
        })
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
        self.forward_unit_with_provider_parallel(
            group,
            index,
            unit,
            hidden,
            state,
            forward,
            &mut ResidentExpertProvider,
            parallel,
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
            return Err(Error::backend("Qwen hybrid model has no local geometry"));
        }
        let hidden = match forward.mode {
            ForwardMode::Target => self
                .decoder
                .static_modules_mut()
                .norm
                .forward(hidden, context)?,
            ForwardMode::Draft(_) => hidden.clone(),
        };
        let modules = self.decoder.static_modules_mut();
        match &mut modules.lm_head {
            Some(head) => B::vocabulary_parallel_project(head, &hidden, parallel, context),
            None => B::vocabulary_parallel_embedding_project(
                &mut modules.embeddings,
                &hidden,
                parallel,
                context,
            ),
        }
    }
}

impl<B, S> PartitionedLayeredArchitecture<B, S> for LayeredModel<B>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
{
    fn partition_observation_hooks(
        &self,
        _tensor_parallel: bool,
    ) -> eredu_runtime::inspection::ObservationHookSupport {
        let target_only = self.prediction_steps == 0;
        eredu_runtime::inspection::ObservationHookSupport::internal(
            target_only,
            target_only,
            target_only,
        )
    }

    type Boundary = eredu_runtime::NoAuxiliaryBoundarySchema;

    fn boundary_schema(&self, metadata: Option<&eredu_nn::workspace::WorkspaceContext>) -> Result<Self::Boundary, Self::Error> {
        if let Some(metadata) = metadata {
            metadata.charge_metadata(std::mem::size_of::<(
                &Self, Option<&eredu_nn::workspace::WorkspaceContext>,
                Self::Boundary, Result<Self::Boundary, Self::Error>,
            )>())?;
        }

        Ok(eredu_runtime::NoAuxiliaryBoundarySchema::new(
            self.config().hidden_size,
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
        if state.layout() != expected {
            return Err(Error::backend(
                "Qwen hybrid partition state layout mismatch",
            ));
        }
        let (input, sequence) = match input {
            LayeredPartitionInput::Tokens(tokens) => {
                (TargetPartitionInput::Tokens(tokens), tokens.dim(1))
            }
            LayeredPartitionInput::Hidden { hidden, .. } => {
                let sequence = hidden.dim(1);
                (TargetPartitionInput::Hidden(hidden), sequence)
            }
        };
        let offset = state
            .layer(first_state_ordinal)
            .map_err(Error::backend)?
            .position();
        self.begin_routed_target_partition(input, mask, sequence, offset, None, context)
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
        if state.layout() != expected {
            return Err(Error::backend(
                "Qwen hybrid partition state layout mismatch",
            ));
        }
        let (input, sequence) = match input {
            LayeredPartitionInput::Tokens(tokens) => {
                (TargetPartitionInput::Tokens(tokens), tokens.dim(1))
            }
            LayeredPartitionInput::Hidden { hidden, .. } => {
                let sequence = hidden.dim(1);
                (TargetPartitionInput::Hidden(hidden), sequence)
            }
        };
        let offset = state
            .layer(first_state_ordinal)
            .map_err(Error::backend)?
            .position();
        self.begin_routed_target_partition(input, mask, sequence, offset, Some(parallel), context)
    }

    fn begin_partition_observed<'a, O>(
        &mut self,
        input: LayeredPartitionInput<'a, B::Tensor>,
        mask: Option<&B::Tensor>,
        state: &mut S,
        expected: &StateLayout,
        first_state_ordinal: usize,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        let embedded = matches!(&input, LayeredPartitionInput::Tokens(_));
        let mut forward = match parallel {
            Some(parallel) => self.begin_partition_parallel(
                input,
                mask,
                state,
                expected,
                first_state_ordinal,
                parallel,
                context,
            ),
            None => {
                self.begin_partition(input, mask, state, expected, first_state_ordinal, context)
            }
        }?;
        if embedded {
            let mut borrowed = eredu_runtime::BorrowedActivationObserver(observer);
            forward.hidden =
                crate::decoder::ComponentInstrumentation::new("readout", &mut borrowed)
                    .apply("embedding", forward.hidden)?;
            forward.context.embedded = forward.hidden.clone();
        }
        Ok(forward)
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
    ) -> Result<LayeredPartitionOutput<B::Tensor>, Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        if !owns_output {
            return self.finish_partition(hidden, state, forward, false, parallel, context);
        }
        let output = match parallel {
            Some(parallel) => self.finish_forward_parallel_observed(
                hidden, state, forward, parallel, context, observer,
            ),
            None => self.finish_forward_observed(hidden, state, forward, context, observer),
        }?;
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
    ) -> Result<LayeredPartitionOutput<B::Tensor>, Self::Error> {
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
                auxiliary: eredu_runtime::NoAuxiliaryBoundary,
            })
        }
    }
}

impl<B, S> crate::partitioned_execution::TextPartitionArchitecture<B, S> for LayeredModel<B>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
{
    fn partition_text_input<'a>(input: Self::Input<'a>) -> (&'a B::Tensor, Option<&'a B::Tensor>) {
        match input {
            EmbeddedInput::Target { tokens, mask } => (tokens, mask),
            EmbeddedInput::Draft { .. } => {
                unreachable!("target-only partition received prediction input")
            }
        }
    }

    fn partition_output_width(&self) -> i32 {
        self.config.vocab_size
    }
}

/// Allocation-free request-local retained tensor iterator.
pub struct RetainedValues<'a, T> {
    values: [Option<&'a T>; 4],
    next: usize,
}

impl<'a, T> RetainedValues<'a, T> {
    const fn new(values: [Option<&'a T>; 4]) -> Self {
        Self { values, next: 0 }
    }
}

impl<'a, T> Iterator for RetainedValues<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        while self.next < self.values.len() {
            let value = self.values[self.next];
            self.next += 1;
            if value.is_some() {
                return value;
            }
        }
        None
    }
}

/// Declares prompt identity independently of concrete state storage.
pub fn state_identity(
    config: &HybridConfig,
    layout: &StateLayout,
    global_layer_start: usize,
    topology: PromptCacheTopology,
) -> Result<ModelStateIdentity, Error> {
    state_identity_with(
        config,
        layout,
        global_layer_start,
        topology,
        crate::decoder::identity::Metadata::new(None),
    )
}

pub(crate) fn state_identity_with_metadata(
    config: &HybridConfig,
    layout: &eredu_runtime::StateLayout,
    global_layer_start: usize,
    topology: eredu_core::cache::PromptCacheTopology,
    context: &eredu_nn::workspace::WorkspaceContext,
) -> Result<eredu_runtime::ModelStateIdentity, eredu_nn::Error> {
    state_identity_with(
        config,
        layout,
        global_layer_start,
        topology,
        crate::decoder::identity::Metadata::new(Some(context)),
    )
}
fn state_identity_with(
    config: &HybridConfig,
    layout: &eredu_runtime::StateLayout,
    global_layer_start: usize,
    topology: eredu_core::cache::PromptCacheTopology,
    metadata: crate::decoder::identity::Metadata<'_>,
) -> Result<eredu_runtime::ModelStateIdentity, eredu_nn::Error> {
    metadata.controls::<eredu_runtime::ModelStateIdentity>()?;

    let target_layers =
        usize::try_from(config.num_hidden_layers).map_err(|cause| metadata.source(cause))?;
    let prediction_layers =
        usize::try_from(config.mtp_num_hidden_layers).map_err(|cause| metadata.source(cause))?;
    let layer_count = target_layers
        .checked_add(prediction_layers)
        .ok_or_else(|| metadata.error(format_args!("Qwen hybrid state layer count overflowed")))?;
    let global_layer_end = global_layer_start
        .checked_add(layout.len())
        .ok_or_else(|| metadata.error(format_args!("Qwen hybrid owned layer range overflowed")))?;
    if global_layer_end > layer_count {
        return Err(metadata.error(format_args!(
            "Qwen hybrid owns layers {global_layer_start}..{global_layer_end}, outside {layer_count} layers"
        )));
    }
    eredu_runtime::ModelStateIdentity::new_with_diagnostic(
        metadata.text(config.variant.model_kind().canonical_name())?,
        metadata.text(&config.model_type)?,
        super::config::prompt_cache_architecture_fingerprint_with_metadata(config, metadata)?,
        layer_count,
        global_layer_start,
        0,
        topology,
        |message| metadata.prompt_error(message),
    )
}

#[cfg(test)]
mod state_identity_tests {
    use super::state_identity;

    fn config(model_type: &str) -> serde_json::Value {
        serde_json::json!({
            "model_type": model_type,
            "vocab_size": 8,
            "hidden_size": 8,
            "num_hidden_layers": 2,
            "mtp_num_hidden_layers": 2,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "max_position_embeddings": 16,
            "intermediate_size": 16,
            "num_experts": 0,
            "tie_word_embeddings": true,
            "layer_types": ["full_attention", "full_attention"]
        })
    }

    #[test]
    fn embedded_prediction_layers_trail_the_prompt_frontier() {
        let parsed =
            crate::qwen::hybrid::model_args_from_config_value(&config("qwen3_5_text")).unwrap();
        let layout = crate::qwen::hybrid::state_layout(&parsed.text).unwrap();
        let identity = state_identity(
            &parsed.text,
            &layout,
            0,
            eredu_core::cache::PromptCacheTopology::default(),
        )
        .unwrap()
        .prompt_cache_identity(&layout)
        .unwrap();

        assert_eq!(identity.model_family(), "qwen3_5");
        assert_eq!(identity.layer_prefix_offsets(), [0, 0, -1, -1]);

        let prediction_layout = layout.slice(2..4).unwrap();
        let prediction_identity = state_identity(
            &parsed.text,
            &prediction_layout,
            2,
            eredu_core::cache::PromptCacheTopology::default(),
        )
        .unwrap()
        .prompt_cache_identity(&prediction_layout)
        .unwrap();
        assert_eq!(prediction_identity.layer_prefix_offsets(), [-1, -1]);
    }

    #[test]
    fn qwen3_next_prompt_cache_identity_uses_the_registry_family() {
        let parsed =
            crate::qwen::hybrid::model_args_from_config_value(&config("qwen3_next")).unwrap();
        let layout = crate::qwen::hybrid::state_layout(&parsed.text).unwrap();
        let identity = state_identity(
            &parsed.text,
            &layout,
            0,
            eredu_core::cache::PromptCacheTopology::default(),
        )
        .unwrap()
        .prompt_cache_identity(&layout)
        .unwrap();

        assert_eq!(identity.model_family(), "qwen3_next");
    }
}
