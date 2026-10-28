//! Unified layered lifecycle for hybrid target and embedded-MTP execution.

use eredu_core::cache::PromptCacheTopology;
use eredu_nn::{
    AttentionCache, EmbeddingLookupPolicy, EmbeddingOperator, Error, GroupedNeuralBackend,
    Parameterized, Tensor,
};
use eredu_runtime::{
    ExecutionUnitLayout, LayerRuntimeState, LayeredForwardState,
    LayeredPartitionInput, LayeredPartitionOutput, ModelStateIdentity, ParameterProvider,
    ResidentExpertProvider, RuntimeStateComponents, StateLayout,
};
use eredu_runtime::{
    ComponentEquations as _, ParallelComponentKernel as _, PartitionComponentKernel as _,
};

use crate::{
    decoder::{DecoderStaticResources, StaticModules},
    hybrid_decoder::HybridStaticModules,
};

use super::{
    prompt_cache_architecture_fingerprint, state_layout, Block, EmbeddedInput, ForwardMode,
    HybridConfig, LocalGeometry, PredictionUnit,
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

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> QwenKernel<B> {
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
pub type LayeredModel<B> = eredu_runtime::ComponentAdapter<
    std::sync::Arc<eredu_runtime::ComponentDefinition>,
    QwenKernel<B>,
    eredu_runtime::KernelParameterModules,
>;

/// Physical pinned resources for full or selected Qwen hybrid execution.
#[derive(Debug, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct QwenStaticModules<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> {
    /// Input or tied-output embedding on its declared owner.
    pub embeddings: Option<B::Embedding>,
    /// Final normalization on the output owner.
    pub boundary: Option<crate::decoder::NormalizedBoundary<B>>,
    /// Separate vocabulary head when untied and locally owned.
    pub lm_head: Option<B::Linear>,
    /// Canonical prediction fusion and normalization shared by local MTP consumers.
    pub prediction: Option<super::PredictionShared<B>>,
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> Clone for QwenStaticModules<B> {
    fn clone(&self) -> Self {
        Self {
            embeddings: self.embeddings.clone(),
            boundary: self.boundary.clone(),
            lm_head: self.lm_head.clone(),
            prediction: self.prediction.clone(),
        }
    }
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> DecoderStaticResources<B>
    for QwenStaticModules<B>
{
    fn decoder_resources(&self) -> crate::decoder::DecoderResources<'_, B> {
        crate::decoder::DecoderResources {
            embedding: self.embeddings.as_ref(),
            norm: self.boundary.as_ref().map(|boundary| &boundary.norm),
            head: self.lm_head.as_ref(),
        }
    }
    fn decoder_resources_mut(&mut self) -> crate::decoder::DecoderResourcesMut<'_, B> {
        crate::decoder::DecoderResourcesMut {
            embedding: self.embeddings.as_mut(),
            norm: self.boundary.as_mut().map(|boundary| &mut boundary.norm),
            head: self.lm_head.as_mut(),
        }
    }
}

/// Typed target, prediction and partition equations over the selected resource owners.
pub struct QwenKernel<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> {
    config: HybridConfig,
    static_modules: QwenStaticModules<B>,
    definition: std::sync::Arc<eredu_runtime::ComponentDefinition>,
    target_layers: usize,
    prediction_steps: usize,
    parallel_geometry: Option<std::sync::Arc<LocalGeometry>>,
    owned_target: std::ops::Range<usize>,
    expert_realization: Option<crate::ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>>,
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> QwenKernel<B> {
    /// Builds unloaded pinned modules and the exact configured graph.
    pub fn from_definition(
        config: HybridConfig,
        definition: std::sync::Arc<eredu_runtime::ComponentDefinition>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredModel<B>, Error> {
        crate::operator_requirements::require::<B>(
            "Qwen hybrid",
            crate::operator_requirements::QWEN_HYBRID,
        )?;
        Self::materialize(config, None, definition, context)
    }

    fn materialize(
        config: HybridConfig,
        geometry: Option<LocalGeometry>,
        definition: std::sync::Arc<eredu_runtime::ComponentDefinition>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredModel<B>, Error> {
        let definition = super::select_definition(&config, definition, geometry.as_ref())?;
        let target_layers = definition.unit_count(0)?;
        let prediction_steps = definition.prediction_groups().len();
        let spec = super::static_module_spec(&config);
        let base = match &geometry {
            Some(geometry) => StaticModules::from_parallel_spec(
                spec,
                geometry.embedding_range().clone(),
                geometry.output_range().cloned(),
                context,
            )?,
            None => StaticModules::from_spec(spec, context)?,
        };
        let extension = (prediction_steps > 0)
            .then(|| super::PredictionShared::new(&config, context))
            .transpose()?;
        let kernel = Self {
            config,
            static_modules: QwenStaticModules {
                embeddings: Some(base.embeddings),
                boundary: Some(base.boundary),
                lm_head: base.lm_head,
                prediction: extension,
            },
            definition: std::sync::Arc::clone(&definition),
            target_layers,
            prediction_steps,
            parallel_geometry: geometry.map(std::sync::Arc::new),
            owned_target: 0..target_layers,
            expert_realization: None,
        };
        eredu_runtime::ComponentAdapter::with_pinned_resources::<B>(definition, kernel)
            .map_err(Error::from)
    }

    fn embedding_mut(&mut self) -> Result<&mut B::Embedding, Error> {
        self.static_modules
            .embeddings
            .as_mut()
            .ok_or_else(|| Error::backend("Qwen component does not own input embedding"))
    }

    fn finish_readout(
        &mut self,
        hidden: &B::Tensor,
        normalize: bool,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut crate::decoder::ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        let mut resources = self.static_modules.decoder_resources_mut();
        let hidden = if normalize {
            resources.normalize(hidden, context, instrumentation)?
        } else {
            hidden.clone()
        };
        resources.project(&hidden, parallel, context, instrumentation)
    }

    /// Materializes only the static resources and units owned by a selected target partition.
    pub fn from_partition(
        config: HybridConfig,
        definition: std::sync::Arc<eredu_runtime::ComponentDefinition>,
        partition: &eredu_runtime::ArchitecturePartition<
            LocalGeometry,
            eredu_runtime::NoAuxiliaryBoundarySchema,
        >,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredModel<B>, Error> {
        crate::operator_requirements::require::<B>(
            "Qwen hybrid",
            crate::operator_requirements::QWEN_HYBRID,
        )?;
        crate::operator_requirements::require::<B>(
            "Qwen hybrid tensor parallelism",
            eredu_nn::NeuralOperatorCapabilities::SUM_PARALLEL,
        )?;
        if config.mtp_num_hidden_layers != 0 {
            return Err(Error::backend(
                "selected Qwen target construction requires prediction-free admission",
            ));
        }
        let geometry = partition.local_geometry();
        let definition = super::select_definition(&config, definition, Some(geometry))?;
        if partition.boundary_schema()
            != &eredu_runtime::NoAuxiliaryBoundarySchema::new(config.hidden_size)
        {
            return Err(Error::backend(
                "Qwen partition parameters or boundary differ from its configuration",
            ));
        }
        let [owned] = partition.groups() else {
            return Err(Error::backend(
                "Qwen target partition must own one execution group",
            ));
        };
        if owned.group().as_str() != definition.primary_group() {
            return Err(Error::backend(
                "Qwen target partition names another execution group",
            ));
        }
        let selection = definition.select_partition(partition)?;
        let range = owned.global_units();
        let text = crate::decoder::PartitionStaticModules::<B>::from_selected_spec(
            super::static_module_spec(&config),
            Some((
                geometry.embedding_range().clone(),
                geometry.output_range().cloned(),
            )),
            |role| {
                selection.owns_storage(&eredu_runtime::ParameterStorageId::StaticRole(role.into()))
            },
            context,
        )?;
        let boundary = text
            .norm
            .map(|norm| {
                crate::decoder::NormalizedBoundary::<B>::from_normalization(
                    config.hidden_size,
                    norm,
                )
            })
            .transpose()?;
        let kernel = Self {
            target_layers: definition.unit_count(0)?,
            prediction_steps: 0,
            config,
            static_modules: QwenStaticModules {
                embeddings: text.embeddings,
                boundary,
                lm_head: text.lm_head,
                prediction: None,
            },
            definition: std::sync::Arc::clone(&definition),
            parallel_geometry: Some(std::sync::Arc::new(geometry.clone())),
            owned_target: range,
            expert_realization: None,
        };
        eredu_runtime::ComponentAdapter::with_partition_resources::<B, _, _>(
            definition, kernel, partition,
        )
        .map_err(Error::from)
    }

    /// Applies the architecture-owned tensor-parallel token embedding boundary.
    pub fn pipeline_embed_parallel(
        &mut self,
        tokens: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        B::vocabulary_parallel_lookup(
            self.embedding_mut()?,
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
        self.finish_readout(
            hidden,
            normalize,
            Some(parallel),
            context,
            &mut crate::decoder::ComponentInstrumentation::disabled(),
        )
    }

    /// Enters a serial target or prediction partition through token embedding.
    pub fn begin_partition_embedding(
        &mut self,
        tokens: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.embedding_mut()?.forward(tokens, context)
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
        self.finish_readout(
            hidden,
            normalize,
            None,
            context,
            &mut crate::decoder::ComponentInstrumentation::disabled(),
        )
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
    pub fn from_parallel_definition(
        config: HybridConfig,
        definition: std::sync::Arc<eredu_runtime::ComponentDefinition>,
        geometry: LocalGeometry,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredModel<B>, Error> {
        crate::operator_requirements::require::<B>(
            "Qwen hybrid",
            crate::operator_requirements::QWEN_HYBRID,
        )?;
        crate::operator_requirements::require::<B>(
            "Qwen hybrid tensor parallelism",
            eredu_nn::NeuralOperatorCapabilities::SUM_PARALLEL,
        )?;
        Self::materialize(config, Some(geometry), definition, context)
    }

    /// Returns normalized hybrid policy.
    pub const fn config(&self) -> &HybridConfig {
        &self.config
    }

    /// Returns the architecture-owned target and prediction traversal.
    pub fn unit_layout(&self) -> Result<ExecutionUnitLayout, Error> {
        Ok(self.definition.parameters().unit_layout().clone())
    }

    /// Borrows the validated graph, parameter ownership and selected state geometry.
    pub fn definition(&self) -> &eredu_runtime::ComponentDefinition {
        &self.definition
    }

    /// Returns the pinned shared prediction modules and base text modules.
    pub fn into_extended_static_modules(
        self,
    ) -> Result<HybridStaticModules<B, Option<super::PredictionShared<B>>>, Error> {
        let modules = self.static_modules;
        if !self.config.tie_word_embeddings && modules.lm_head.is_none() {
            return Err(Error::backend(
                "Qwen component does not own its untied head",
            ));
        }
        Ok(HybridStaticModules {
            base: StaticModules {
                embeddings: modules
                    .embeddings
                    .ok_or_else(|| Error::backend("Qwen component does not own embedding"))?,
                boundary: modules
                    .boundary
                    .ok_or_else(|| Error::backend("Qwen component does not own readout"))?,
                lm_head: modules.lm_head,
            },
            extension: modules.prediction,
        })
    }

    fn prediction_shared_mut(&mut self) -> Result<&mut super::PredictionShared<B>, Error> {
        self.static_modules
            .prediction
            .as_mut()
            .ok_or_else(|| Error::backend("Qwen prediction has no shared modules"))
    }

    /// Borrows pinned text modules.
    pub const fn static_modules(&self) -> &QwenStaticModules<B> {
        &self.static_modules
    }

    /// Mutably borrows pinned text modules for checkpoint binding.
    pub fn static_modules_mut(&mut self) -> &mut QwenStaticModules<B> {
        &mut self.static_modules
    }

    /// Consumes the text graph and returns its pinned decoder modules.
    pub fn into_static_modules(self) -> Result<StaticModules<B>, Error> {
        self.into_extended_static_modules()
            .map(|modules| modules.base)
    }

    /// Returns the prediction depth count declared by the constructed execution graph.
    pub fn mtp_len(&self) -> usize {
        self.definition.prediction_groups().len()
    }

    /// Returns target plus configured prediction state policy.
    pub fn state_layout(&self) -> Result<StateLayout, Error> {
        state_layout(&self.config).map_err(Error::backend)
    }

    /// Shares authoritative local geometry with placed residency factories.
    pub fn shared_parallel_geometry(&self) -> Option<std::sync::Arc<LocalGeometry>> {
        self.parallel_geometry.as_ref().map(std::sync::Arc::clone)
    }

    /// Constructs one canonical target or prediction unit using this model's
    /// replicated or planner-derived local geometry.
    pub fn construct_unit(
        &self,
        group: usize,
        index: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Unit<B>, Error> {
        self.definition.unit(group, index)?;
        if group == 0 {
            self.state_index(group, index)?;
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

    pub(crate) fn install_expert_realization(
        &mut self,
        realization: crate::ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>,
    ) {
        self.expert_realization = Some(realization);
    }

    fn state_index(&self, group: usize, index: usize) -> Result<usize, Error> {
        let slot = self.definition.unit(group, index)?.state.start;
        if group == 0 {
            if !self.owned_target.contains(&index) {
                return Err(Error::backend(
                    "Qwen target unit is outside selected ownership",
                ));
            }
            Ok(slot - self.owned_target.start)
        } else {
            Ok(slot)
        }
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
        P: eredu_runtime::ParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        self.definition.unit(group, index)?;
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
        P: eredu_runtime::TensorParallelParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        self.definition.unit(group, index)?;
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
    let target_layers = usize::try_from(config.num_hidden_layers).map_err(Error::backend)?;
    let prediction_layers =
        usize::try_from(config.mtp_num_hidden_layers).map_err(Error::backend)?;
    let layer_count = target_layers
        .checked_add(prediction_layers)
        .ok_or_else(|| Error::backend("Qwen hybrid state layer count overflowed"))?;
    let global_layer_end = global_layer_start
        .checked_add(layout.len())
        .ok_or_else(|| Error::backend("Qwen hybrid owned layer range overflowed"))?;
    if global_layer_end > layer_count {
        return Err(Error::backend(format!(
            "Qwen hybrid owns layers {global_layer_start}..{global_layer_end}, outside {layer_count} layers"
        )));
    }
    eredu_runtime::ModelStateIdentity::new(
        config.variant.model_kind().canonical_name(),
        config.model_type.clone(),
        prompt_cache_architecture_fingerprint(config),
        layer_count,
        global_layer_start,
        0,
        topology,
    )
    .map_err(Error::backend)
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> eredu_runtime::ComponentKernel<B>
    for QwenKernel<B>
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
        topology: PromptCacheTopology,
    ) -> Result<ModelStateIdentity, Self::Error> {
        state_identity(
            &self.config,
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
        let recipes = super::static_recipes(source)?;
        crate::static_parameters::module_recipes(&self.static_modules, recipes)
    }
    fn build_unit(
        &self,
        address: eredu_runtime::ExecutionUnitAddress,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Unit, Self::Error> {
        let (group, index) = (address.group(), address.index());
        self.construct_unit(group, index, context)
    }
    fn executes_group(&self, group: usize, forward: &Self::ForwardContext) -> bool {
        match forward.mode {
            ForwardMode::Target => group == 0,
            ForwardMode::Draft(depth) => group == depth + 1,
        }
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
            forward.target_hidden.as_ref(),
        ])
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
}
impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend>
    eredu_runtime::ComponentPinnedResources<B> for QwenKernel<B>
{
    type Modules = QwenStaticModules<B>;
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
        if let Some(shared) = &self.static_modules.prediction {
            visitor.visit("mtp", shared)?;
        }
        self.static_modules.decoder_resources().visit(visitor)
    }
    fn visit_pinned_mut<V: eredu_runtime::StaticParameterVisitorMut<B>>(
        &mut self,
        visitor: &mut V,
    ) -> Result<(), V::Error> {
        if let Some(shared) = &mut self.static_modules.prediction {
            visitor.visit_mut("mtp", shared)?;
        }
        self.static_modules.decoder_resources_mut().visit(visitor)
    }
}
impl<B, S> eredu_runtime::ComponentEquations<B, S> for QwenKernel<B>
where
    B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
{
    fn observation_hooks(&self) -> eredu_runtime::inspection::ObservationHookSupport {
        let target_only = self.prediction_steps == 0;
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
        let embedded = self.embedding_mut()?.forward(tokens, context)?;
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
            forward =
                crate::decoder::ComponentInstrumentation::apply_embedding(forward, observer)?;
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
            &mut ResidentExpertProvider,
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
        <Self as eredu_runtime::RoutedComponentKernel<B, S>>::execute_routed_observed(
            self,
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
        _state: &mut S,
        forward: &mut Self::ForwardContext,
        _context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        if group == 0 || matches!(forward.mode, ForwardMode::Draft(depth) if group == depth + 1) {
            forward.target_hidden = Some(hidden.clone());
        }
        Ok(hidden.clone())
    }
    fn projects_final_text_position() -> bool {
        true
    }
    fn finish_text(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.finish(
            &crate::decoder::final_hidden_position(hidden, context)?,
            state,
            forward,
            context,
        )
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
            matches!(forward.mode, ForwardMode::Target),
            None,
            context,
            &mut crate::decoder::ComponentInstrumentation::disabled(),
        )
    }
    fn finish_observed<O>(
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
            self.finish_readout(
                hidden,
                true,
                None,
                context,
                &mut crate::decoder::ComponentInstrumentation::new("readout", &mut borrowed),
            )
        } else {
            self.finish(hidden, state, forward, context)
        }
    }
}
impl<B, S> eredu_runtime::ParallelComponentKernel<B, S> for QwenKernel<B>
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
    fn begin_parallel_observed<'a, O>(
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
        let mut forward = self.begin_parallel(input, state, parallel, context)?;
        if matches!(forward.context.mode, ForwardMode::Target) {
            forward = crate::decoder::ComponentInstrumentation::apply_embedding(forward, observer)?;
            forward.context.embedded = forward.hidden.clone();
        }
        Ok(forward)
    }
    fn execute_parallel_observed<O>(
        &mut self,
        address: eredu_runtime::ExecutionUnitAddress,
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
        <Self as eredu_runtime::ParallelRoutedComponentKernel<B, S>>::execute_parallel_routed_observed(
            self, address, unit, hidden, state, forward,
            if hidden.dim(1) == 1 { eredu_runtime::ExpertPass::Decode } else { eredu_runtime::ExpertPass::Prefill },
            &mut ResidentExpertProvider, parallel, context, observer,
        )
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
        if self.parallel_geometry.is_none() {
            return Err(Error::backend("Qwen hybrid model has no local geometry"));
        }
        if matches!(forward.mode, ForwardMode::Target) {
            let mut borrowed = eredu_runtime::BorrowedActivationObserver(observer);
            self.finish_readout(
                hidden,
                true,
                Some(parallel),
                context,
                &mut crate::decoder::ComponentInstrumentation::new("readout", &mut borrowed),
            )
        } else {
            self.finish_parallel(hidden, state, forward, parallel, context)
        }
    }
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
            self.embedding_mut()?,
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
    fn finish_parallel(
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
        self.finish_readout(
            hidden,
            matches!(forward.mode, ForwardMode::Target),
            Some(parallel),
            context,
            &mut crate::decoder::ComponentInstrumentation::disabled(),
        )
    }
}
impl<B, S> eredu_runtime::RoutedComponentKernel<B, S> for QwenKernel<B>
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
        QwenKernel::forward_unit_with_provider(
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
        let path = self.definition.unit(group, index)?.path.clone();
        let lane = state
            .layer(self.state_index(group, index)?)
            .map_err(Error::backend)?;
        let points = eredu_runtime::RoutedObservationPoints::new(
            eredu_runtime::RoutedBankId::new(0),
            format!("{path}.mlp"),
            self.config.num_experts,
        );
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
impl<B, S> eredu_runtime::ParallelRoutedComponentKernel<B, S> for QwenKernel<B>
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
        QwenKernel::forward_unit_with_provider_parallel(
            self, group, index, unit, hidden, state, forward, provider, parallel, context,
        )
    }
    fn execute_parallel_routed_observed<P, O>(
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
        observer: &mut O,
    ) -> Result<B::Tensor, Error>
    where
        P: eredu_runtime::TensorParallelParameterProvider<B>,
        P::Error: std::fmt::Display,
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        let (group, index) = (address.group(), address.index());
        let path = self.definition.unit(group, index)?.path.clone();
        let lane = state
            .layer(self.state_index(group, index)?)
            .map_err(Error::backend)?;
        let points = eredu_runtime::RoutedObservationPoints::new(
            eredu_runtime::RoutedBankId::new(0),
            format!("{path}.mlp"),
            self.config.num_experts,
        );
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
}
impl<B, S> eredu_runtime::PartitionComponentKernel<B, S> for QwenKernel<B>
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
    fn boundary_schema(&self) -> Result<Self::Boundary, Self::Error> {
        Ok(eredu_runtime::NoAuxiliaryBoundarySchema::new(
            self.config().hidden_size,
        ))
    }
    fn begin_partition<'a>(
        &mut self,
        input: LayeredPartitionInput<'a, B::Tensor>,
        mask: Option<&B::Tensor>,
        state: &mut S,
        selected: eredu_runtime::ComponentPartitionContext<'_, B>,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error> {
        let expected = selected.expected.ok_or_else(|| {
            Error::backend("Qwen hybrid partition requires selected state geometry")
        })?;
        let (first_state_ordinal, parallel, context) = (
            selected.first_state_ordinal,
            selected.parallel,
            selected.native,
        );
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
        self.begin_routed_target_partition(input, mask, sequence, offset, parallel, context)
    }
    fn begin_partition_observed<'a, O>(
        &mut self,
        input: LayeredPartitionInput<'a, B::Tensor>,
        mask: Option<&B::Tensor>,
        state: &mut S,
        selected: eredu_runtime::ComponentPartitionContext<'_, B>,
        observer: &mut O,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        let embedded = matches!(&input, LayeredPartitionInput::Tokens(_));
        let mut forward = self.begin_partition(input, mask, state, selected)?;
        if embedded {
            forward = crate::decoder::ComponentInstrumentation::apply_embedding(forward, observer)?;
            forward.context.embedded = forward.hidden.clone();
        }
        Ok(forward)
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
                auxiliary: eredu_runtime::NoAuxiliaryBoundary,
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
    ) -> Result<LayeredPartitionOutput<B::Tensor>, Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        if !owns_output {
            return self.finish_partition(hidden, state, forward, false, parallel, context);
        }
        let output = match parallel {
            Some(parallel) => {
                self.finish_parallel_observed(hidden, state, forward, parallel, context, observer)
            }
            None => self.finish_observed(hidden, state, forward, context, observer),
        }?;
        Ok(LayeredPartitionOutput::Final {
            output,
            retained: Some(hidden.clone()),
        })
    }
    fn enter_partition_group(
        &mut self,
        group: usize,
        initial: &B::Tensor,
        _state: &mut S,
        _forward: &mut Self::ForwardContext,
        _parallel: Option<&B::ParallelContext>,
        _context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        if group != 0 {
            return Err(Error::backend(
                "Qwen target partition cannot enter a prediction group without its producer",
            ));
        }
        Ok(initial.clone())
    }
}
impl<B, S> eredu_runtime::PipelineComponentKernel<B, S> for QwenKernel<B>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
{
    fn tensor_vocabulary_sharded(&self) -> bool {
        true
    }

    fn input_dimensions(&self, input: &Self::Input<'_>) -> Result<(i32, i32), Self::Error> {
        match input {
            EmbeddedInput::Target { tokens, .. } => Ok((tokens.dim(0), tokens.dim(1))),
            _ => Err(eredu_nn::Error::backend(
                "target partition received prediction input",
            )),
        }
    }
    fn begin_pipeline<'a, O>(
        &mut self,
        input: Self::Input<'a>,
        incoming: Option<(
            B::Tensor,
            <Self::Boundary as eredu_runtime::ArchitectureBoundary>::Boundary<B::Tensor>,
        )>,
        state: &mut S,
        selected: eredu_runtime::ComponentPartitionContext<'_, B>,
        observer: &mut O,
    ) -> Result<eredu_runtime::LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Self::Error> + ?Sized,
    {
        let (tokens, mask) = match input {
            EmbeddedInput::Target { tokens, mask } => (tokens, mask),
            _ => {
                return Err(eredu_nn::Error::backend(
                    "target partition received prediction input",
                ))
            }
        };
        let input = match incoming {
            Some((hidden, auxiliary)) => {
                eredu_runtime::LayeredPartitionInput::Hidden { hidden, auxiliary }
            }
            None => eredu_runtime::LayeredPartitionInput::Tokens(tokens),
        };
        self.begin_partition_observed(input, mask, state, selected, observer)
    }
    fn output_width(&self) -> i32 {
        self.config.vocab_size
    }
}
impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend>
    eredu_runtime::TextComponentKernel<B> for QwenKernel<B>
{
    fn supports_chunked_prefill() -> bool {
        super::CHUNKED_TEXT_PREFILL
    }
    fn text_input<'a>(tokens: &'a B::Tensor, mask: Option<&'a B::Tensor>) -> Self::Input<'a> {
        EmbeddedInput::target(tokens, mask)
    }
}
