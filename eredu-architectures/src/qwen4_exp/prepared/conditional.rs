//! Joint vision/target ownership and selection over the shared routed session driver.
use super::requirements::ParameterRequirements;
use super::*;
use crate::qwen4_exp::{
    conditional::{self, ConditionalModel},
    media::MediaIngress,
};
use eredu_runtime::{
    ExecutionUnitLayout, ReplicatedTextParameterOwner as Owner, ReplicatedTextRequirements,
};
fn invalid(e: impl std::fmt::Display) -> PreparationError {
    PreparationError::Contract(e.to_string())
}

/// Exact joint graph requirements, retaining the original target and projector sources.
#[derive(Clone)]
pub struct ConditionalExecutionPlan {
    plan: TargetExecutionPlan,
    ingress: MediaIngress,
}
/// Selected joint execution, without an alternate source or family dispatch in a backend.
#[derive(Clone)]
pub struct SelectedConditionalExecution {
    selected: SelectedTargetExecution,
    processor: crate::qwen4_exp::media::processor::SelectedMediaProcessor,
    ingress: MediaIngress,
}
impl ConditionalExecutionPlan {
    pub(super) fn into_parts(self) -> (TargetExecutionPlan, MediaIngress) {
        (self.plan, self.ingress)
    }
}
impl TargetExecutionPlan {
    /// Adds a retained optional vision group. Target unit identities, state slots,
    /// expert banks and row ownership are preserved; graph dependencies run vision first.
    pub fn with_vision(
        mut self,
        vision: PreparedVision,
    ) -> Result<ConditionalExecutionPlan, PreparationError> {
        let ingress = self.target.media_ingress(vision.clone()).map_err(invalid)?;
        let target_source = retain(
            self.target.artifact.clone(),
            self.target
                .artifact
                .source_keys()
                .into_iter()
                .filter(|key| !key.starts_with("model.visual."))
                .collect(),
        )?;
        let source: SharedCheckpointSource =
            Arc::new(eredu_checkpoint::store::CompositeCheckpointSource::new([
                target_source,
                vision.artifact().clone(),
            ])?);
        let physical = super::requirements::physical_sources(source.as_ref())?;
        let blocks = (0..vision.config().layer_count())
            .map(|index| vision.block(index).map(|block| block.recipes().clone()))
            .collect::<Result<Vec<_>, _>>()?;
        let requirements = joint_requirements(
            &self.requirements,
            &self.target.spec,
            vision.config(),
            vision.static_parameters().recipes(),
            &blocks,
            source.as_ref(),
            &physical,
            ingress.identity(),
        )?;
        self.requirements = requirements;
        self.target.artifact = source;
        self.capability = crate::capability::qwen4_exp_conditional(
            &self.target.spec,
            self.prediction
                .as_ref()
                .map(|(prediction, _)| &prediction.spec),
        )
        .map_err(invalid)?;
        Ok(ConditionalExecutionPlan {
            plan: self,
            ingress,
        })
    }
}
/// Shared header/bound graph authoring. This accepts metadata, never payload authority.
#[allow(clippy::too_many_arguments)]
pub(super) fn joint_requirements<C: eredu_checkpoint::recipe::RecipeCatalog + ?Sized>(
    target: &crate::routed_text::RoutedTextRequirements,
    spec: &TargetSpec,
    vision_config: &crate::qwen::vision::VisionConfig,
    vision_static: &BTreeMap<String, DerivedWeightRecipe>,
    vision_blocks: &[BTreeMap<String, DerivedWeightRecipe>],
    source: &C,
    physical: &BTreeMap<String, eredu_runtime::ReplicatedTextPhysicalSource>,
    media_identity: &str,
) -> Result<crate::routed_text::RoutedTextRequirements, PreparationError> {
    let mut formats = ParameterFormats(BTreeMap::new(), BTreeMap::new());
    for params in std::iter::once(vision_static).chain(vision_blocks.iter()) {
        for name in params.keys() {
            formats
                .0
                .insert(name.clone(), vision_config.linear_format(name));
        }
    }
    let mut declared = ParameterRequirements::new(source, physical, &formats);
    for (name, recipe) in vision_static {
        declared.add(name, recipe, Owner::StaticRole("vision".into()), false)?;
    }
    for index in 0..vision_config.layer_count() {
        for (name, recipe) in &vision_blocks[index] {
            declared.add(
                name,
                recipe,
                Owner::ExecutionUnit {
                    group: conditional::VISION_GROUP.into(),
                    unit: index,
                },
                false,
            )?;
        }
    }
    let old = target.text();
    let graph = conditional::graph()?;
    let layout = ExecutionUnitLayout::new(&graph, [spec.units.len(), vision_config.layer_count()])
        .map_err(invalid)?;
    let auxiliary_names = old
        .auxiliary_parameters()
        .iter()
        .map(|p| p.name())
        .collect::<BTreeSet<_>>();
    let mut recipes = old
        .derived_recipes()
        .iter()
        .filter(|(n, _)| !auxiliary_names.contains(n.as_str()))
        .map(|(n, r)| (n.clone(), r.clone()))
        .collect::<BTreeMap<_, _>>();
    let mut outputs = old
        .derived_recipe_outputs()
        .iter()
        .filter(|(n, _)| !auxiliary_names.contains(n.as_str()))
        .map(|(n, r)| (n.clone(), r.clone()))
        .collect::<BTreeMap<_, _>>();
    recipes.extend(declared.derived);
    outputs.extend(declared.outputs);
    let mut parameters = old.parameters().to_vec();
    parameters.extend(declared.parameters);
    let identity = eredu_core::cache::derive_prompt_cache_architecture_fingerprint(
        "qwen4_exp.conditional.v1",
        [
            ("target", old.architecture_identity().to_owned()),
            ("media", media_identity.to_owned()),
        ],
    );
    let mut text = ReplicatedTextRequirements::new(
        identity,
        old.operators()
            .union(crate::operator_requirements::QWEN_VISION),
        graph,
        layout,
        vec![crate::transport::decoder(), conditional::vision_transport()],
        old.state_layout().clone(),
        old.state_access(),
        parameters,
    )
    .map_err(invalid)?
    .with_chunked_prefill(old.supports_chunked_prefill())
    .with_execution_topology(old.execution_topology().cloned())
    .with_grouped_operations(old.grouped_operations().to_vec())
    .with_derived_recipes_and_shared_sources(recipes, outputs, old.shared_source_keys().clone())
    .map_err(invalid)?
    .with_append_streams(old.append_streams().to_vec())
    .map_err(invalid)?;
    if let Some(dtype) = old.floating_state_source() {
        text = text.with_floating_state_source(dtype.clone());
    }
    if !old.auxiliary_parameters().is_empty() {
        text = text
            .with_auxiliary_parameters(
                old.auxiliary_parameters().to_vec(),
                old.derived_recipes()
                    .iter()
                    .filter(|(n, _)| auxiliary_names.contains(n.as_str()))
                    .map(|(n, r)| (n.clone(), r.clone()))
                    .collect(),
                old.derived_recipe_outputs()
                    .iter()
                    .filter(|(n, _)| auxiliary_names.contains(n.as_str()))
                    .map(|(n, r)| (n.clone(), r.clone()))
                    .collect(),
            )
            .map_err(invalid)?;
    }
    text = text
        .with_replicated_static_roles(old.replicated_static_roles().to_vec())
        .map_err(invalid)?;
    let mut requirements =
        crate::routed_text::RoutedTextRequirements::new(text, target.banks().clone(), source)
            .map_err(invalid)?;
    if let Some(rows) = target.row_lookups() {
        requirements = requirements
            .with_row_lookups(rows.clone())
            .map_err(invalid)?;
    }
    Ok(requirements)
}
impl ConditionalExecutionPlan {
    pub(super) fn bind_selected(
        self,
        selected: crate::routed_text::SelectedRoutedTextRealization,
        processor: crate::qwen4_exp::media::processor::SelectedMediaProcessor,
        prediction_state: Option<eredu_runtime::SelectedStateRealization>,
    ) -> Result<SelectedConditionalExecution, TargetSelectionError> {
        Ok(SelectedConditionalExecution {
            selected: self.plan.bind_selected(selected, prediction_state)?,
            processor,
            ingress: self.ingress,
        })
    }

    /// Exact joint ordinary and bank requirements, including each encoder block.
    pub fn requirements(&self) -> &crate::routed_text::RoutedTextRequirements {
        self.plan.requirements()
    }
    /// Retained original-ID and media geometry policy.
    pub fn ingress(&self) -> &MediaIngress {
        &self.ingress
    }
    /// Exact independent prediction-state facts, when a prediction role was prepared.
    pub fn prediction_state_requirements(
        &self,
    ) -> Option<&eredu_runtime::StateRealizationRequirements> {
        self.plan.prediction_state_requirements()
    }
    /// Processor requirements from retained visual artifacts and prepared-input policy.
    pub fn processor_requirements(
        &self,
    ) -> Result<eredu_runtime::ProcessorExecutionRequirements, eredu_runtime::ProcessorSelectionError>
    {
        self.ingress.processor_requirements()
    }
    /// Selects every ordinary vision/target parameter using the same bounded policy.
    pub fn select(
        self,
        request: &crate::routed_text::RoutedTextSelectionRequest,
        mechanisms: &eredu_runtime::BackendMechanismCapabilities,
        prediction: Option<&eredu_runtime::StateMechanismCapabilities>,
        inputs: &eredu_runtime::ProcessorSelectionRequest,
        media: &eredu_runtime::MediaPrimitiveCapabilities,
    ) -> Result<SelectedConditionalExecution, TargetSelectionError> {
        let processor = self.ingress.select_processor(inputs, media)?;
        Ok(SelectedConditionalExecution {
            processor,
            selected: self.plan.select(request, mechanisms, prediction)?,
            ingress: self.ingress,
        })
    }
}
impl SelectedConditionalExecution {
    /// Independent predictor state retained by the joint conditional selection.
    pub fn prediction_state(&self) -> Option<&eredu_runtime::SelectedStateRealization> {
        self.selected.prediction_state()
    }

    fn selected_vision_config(&self) -> crate::qwen::vision::VisionConfig {
        let mut vision = self.ingress.vision().config().clone();
        vision.linear_formats.clear();
        for (name, format) in crate::replicated_text::selected_matrix_formats(
            self.selected.selected.text().requirements(),
            self.selected.selected.text(),
        ) {
            if name.starts_with("model.visual.") {
                vision.linear_formats.insert(name, format);
            }
        }
        vision
    }

    /// Reports the admitted request's shared vision invocations using the retained
    /// checkpoint formats and actual execution-context mechanism facts. The explicit
    /// activation representation is a conditional assumption, not inferred binding
    /// state. Missing promotion/storage/retirement facts prevent physical admission.
    pub fn encoder_memory<B: eredu_nn::NeuralBackend>(
        &self,
        input: &crate::qwen4_exp::media::AdmittedMediaInput<'_, B::Tensor>,
        element: eredu_nn::TensorElementType,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<crate::qwen::vision::VisionMemoryReport, crate::qwen4_exp::media::MediaInputError>
    {
        input.validate_policy(self.ingress.admission_config())?;
        let grids = input
            .admission()
            .parts()
            .iter()
            .flat_map(|part| match part {
                crate::media_plan::QwenInputPartPlan::Media { ingress, .. } => {
                    ingress.patch_grid.as_slice()
                }
                _ => &[],
            })
            .copied()
            .collect::<Vec<_>>();
        Ok(self.selected_vision_config().memory_report(
            &grids,
            element,
            |invocation| B::mechanism_memory_in_context(invocation, context),
            |invocation| B::segmented_attention_memory(invocation, context),
        )?)
    }

    /// Describes whole-request rotary preparation using retained position geometry
    /// and selected native context facts, independently from optional encoder work.
    /// Logical payload descriptions do not admit native scratch or physical pools.
    pub fn request_rotary_memory<B: eredu_nn::NeuralBackend>(
        &self,
        input: &crate::qwen4_exp::media::AdmittedMediaInput<'_, B::Tensor>,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<
        eredu_nn::mechanism_memory::MechanismMemoryContract,
        crate::qwen4_exp::media::MediaInputError,
    > {
        input.validate_policy(self.ingress.admission_config())?;
        let memory = B::mechanism_memory_in_context(&input.rotary_invocation()?, context)?;
        memory.validate()?;
        Ok(memory)
    }

    /// Retained source-independent processor and selected representation readiness.
    pub fn processor(&self) -> &crate::qwen4_exp::media::processor::SelectedMediaProcessor {
        &self.processor
    }
    /// Recipe/native materialization bounds for the combined ordinary graph.
    pub fn parameter_materialization_workspace(
        &self,
        mechanisms: &impl crate::PreparationMechanismProvider,
    ) -> Result<eredu_core::ParameterMaterializationWorkspace, String> {
        self.selected
            .parameter_materialization_workspace(mechanisms)
    }
    /// Persistent state estimate; image/video transient workspaces are separate.
    pub fn capability_estimate(&self) -> &crate::capability::CapabilityEstimate {
        self.selected.capability_estimate()
    }
    /// Selected state, weights, grouped experts and exact row mechanisms.
    pub fn realization(&self) -> &crate::routed_text::SelectedRoutedTextRealization {
        self.selected.realization()
    }
    /// Constructs the joint architecture through the ordinary prepared routed handoff.
    pub fn prepare<B, S>(
        self,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<
        (
            crate::routed_text::PreparedRoutedTextArchitecture<ConditionalModel<B>>,
            SharedCheckpointSource,
        ),
        PreparationError,
    >
    where
        B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend + Clone,
        S: eredu_runtime::LayerRuntimeState<B>,
        S::LayerState: eredu_nn::AttentionCache<B::Tensor>
            + eredu_runtime::RuntimeStateComponents<B>
            + eredu_runtime::RuntimeAppendStreams<B>,
    {
        self.prepare_with::<B, S, _>(context, |architecture| architecture)
    }

    /// Constructs the shared prepared-input adapter over the same selected modules,
    /// providers, source authority and retained processor selection.
    pub fn prepare_composite<B, S>(
        self,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<
        (
            crate::routed_text::PreparedRoutedTextArchitecture<
                crate::composite_execution::PreparedCompositeArchitecture<ConditionalModel<B>>,
            >,
            SharedCheckpointSource,
            crate::qwen4_exp::media::processor::SelectedMediaProcessor,
        ),
        PreparationError,
    >
    where
        B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend + Clone,
        S: eredu_runtime::LayerRuntimeState<B>,
        S::LayerState: eredu_nn::AttentionCache<B::Tensor>
            + eredu_runtime::RuntimeStateComponents<B>
            + eredu_runtime::RuntimeAppendStreams<B>,
    {
        let processor = self.processor.clone();
        let (prepared, source) = self.prepare_with::<B, S, _>(
            context,
            crate::composite_execution::PreparedCompositeArchitecture::new,
        )?;
        Ok((prepared, source, processor))
    }

    /// Constructs the shared routed composite handoff and invokes the ordinary
    /// family-neutral visitor. No artifact inspection or source reopening is needed.
    /// A retained prediction role requires the joint prediction driver and is
    /// rejected before construction starts by this target-only visitor.
    pub fn visit_composite<B, S, V>(
        self,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        mut visitor: V,
    ) -> Result<V::Output, crate::prepared_execution::PreparedExecutionError<V::Error>>
    where
        B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend + Clone + 'static,
        S: eredu_runtime::LayerRuntimeState<B>,
        S::LayerState: eredu_nn::AttentionCache<B::Tensor>
            + eredu_runtime::RuntimeStateComponents<B>
            + eredu_runtime::RuntimeAppendStreams<B>,
        V: crate::replicated_text::CompositeTextArchitectureVisitor<B, S>,
    {
        use crate::prepared_execution::PreparedExecutionError;
        use crate::replicated_text::PreparedRoutedCompositeTextArchitecture;
        if self.selected.prediction_state().is_some() {
            return Err(PreparedExecutionError::UnavailablePrediction);
        }
        let requirements = self.selected.plan.requirements.clone();
        let selected = self.selected.realization().clone();
        let processor_requirements = self
            .ingress
            .processor_requirements()
            .map_err(|error| PreparedExecutionError::Architecture(error.to_string()))?;
        let admission = self.ingress.admission_config().clone();
        visitor.construction_started();
        let (prepared, source, processor) = self
            .prepare_composite::<B, S>(context)
            .map_err(|error| PreparedExecutionError::Architecture(error.to_string()))?;
        let prepared = PreparedRoutedCompositeTextArchitecture::from_routed_declared(
            prepared,
            requirements,
            &selected,
            processor.realization().clone(),
            processor_requirements,
            admission,
        )
        .map_err(PreparedExecutionError::Architecture)?;
        visitor
            .visit_routed(prepared, source)
            .map_err(PreparedExecutionError::Backend)
    }

    /// Constructs the conditional target and independent predictor from one retained
    /// joint selection. The target receives primary authority while its shared bank
    /// provider retains the complete declared target/prediction parameter view.
    pub fn visit_composite_prediction<B, S, V>(
        self,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        mut visitor: V,
        binding: crate::prepared_execution::PredictionBinding,
    ) -> Result<V::Output, crate::prepared_execution::PreparedExecutionError<V::Error>>
    where
        B: eredu_nn::GroupedNeuralBackend
            + eredu_nn::DistributedNeuralBackend
            + eredu_nn::BlockwiseAttentionBackend
            + eredu_nn::HyperNeuralBackend
            + Clone
            + 'static,
        S: eredu_runtime::LayerRuntimeState<B>,
        S::LayerState: eredu_nn::AttentionCache<B::Tensor>
            + eredu_runtime::RuntimeStateComponents<B>
            + eredu_runtime::RuntimeAppendStreams<B>,
        V: crate::replicated_text::CompositeTextArchitectureVisitor<B, S>,
    {
        use crate::prepared_execution::PreparedExecutionError;
        use crate::replicated_text::PreparedRoutedCompositeTextArchitecture;
        if self.selected.prediction_state().is_none() {
            return Err(PreparedExecutionError::PredictionSourceMismatch);
        }
        let requirements = self.selected.plan.requirements.clone();
        let selected = self.selected.realization().clone();
        let processor_requirements = self
            .ingress
            .processor_requirements()
            .map_err(|error| PreparedExecutionError::Architecture(error.to_string()))?;
        let admission = self.ingress.admission_config().clone();
        let provider_source = crate::replicated_text::restrict_store_handoff(
            requirements.text(),
            self.selected.plan.target.artifact.clone(),
            crate::replicated_text::StoreHandoffScope::Complete,
        )
        .map_err(PreparedExecutionError::Architecture)?;
        visitor.construction_started();
        let prediction = self
            .selected
            .prepare_prediction_weights::<B>(context)
            .and_then(|weights| weights.with_discovery_binding(&binding).map_err(invalid))
            .map_err(|error| PreparedExecutionError::Architecture(error.to_string()))?;
        let (prepared, target_source, processor) = self
            .prepare_composite::<B, S>(context)
            .map_err(|error| PreparedExecutionError::Architecture(error.to_string()))?;
        let prepared = PreparedRoutedCompositeTextArchitecture::from_routed_declared(
            prepared,
            requirements,
            &selected,
            processor.realization().clone(),
            processor_requirements,
            admission,
        )
        .map_err(PreparedExecutionError::Architecture)?;
        visitor.visit_prediction(
            prepared,
            prediction,
            target_source,
            provider_source,
            binding,
        )
    }

    fn prepare_with<B, S, A>(
        self,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        adapt: impl Fn(ConditionalModel<B>) -> A,
    ) -> Result<
        (
            crate::routed_text::PreparedRoutedTextArchitecture<A>,
            SharedCheckpointSource,
        ),
        PreparationError,
    >
    where
        B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend + Clone,
        S: eredu_runtime::LayerRuntimeState<B>,
        S::LayerState: eredu_nn::AttentionCache<B::Tensor>
            + eredu_runtime::RuntimeStateComponents<B>
            + eredu_runtime::RuntimeAppendStreams<B>,
        A: eredu_runtime::LayeredArchitecture<B, S, Error = eredu_nn::Error>
            + eredu_runtime::RoutedLayeredArchitecture<B, S>,
        A::StaticModules: Clone,
    {
        let vision = self.selected_vision_config();
        let SelectedTargetExecution { plan, selected, .. } = self.selected;
        crate::routed_text::validate_selected_routed_handoff(&plan.requirements, &selected)
            .map_err(invalid)?;
        let execution_source = crate::replicated_text::restrict_store_handoff(
            plan.requirements.text(),
            plan.target.artifact.clone(),
            crate::replicated_text::StoreHandoffScope::Primary,
        )
        .map_err(invalid)?;
        let source_vision = self.ingress.vision().config().clone();
        let spec = plan
            .target
            .bind_spec(plan.target.selected_spec(&selected)?)?;
        let source_architecture = crate::replicated_text::selected_uses_transform(selected.text())
            .then(|| {
                ConditionalModel::<B>::new(
                    plan.target.bound_spec()?,
                    self.ingress.clone(),
                    source_vision,
                    context,
                )
            })
            .transpose()?;
        let architecture = ConditionalModel::<B>::new(spec, self.ingress, vision, context)?;
        let identity = architecture.state_fingerprint();
        let prepared = crate::routed_text::prepare_routed_architecture_handoff::<B, S, _>(
            adapt(architecture),
            source_architecture.map(adapt),
            plan.requirements,
            selected,
            (!plan.row_sources.entries().is_empty()).then_some(plan.row_sources),
            plan.capability,
            "qwen4_exp".into(),
            identity,
            context,
        )
        .map_err(invalid)?;
        Ok((prepared, execution_source))
    }
}
