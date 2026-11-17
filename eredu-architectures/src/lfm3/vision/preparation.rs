//! Retained cold selection and exact prepared bindings for the replicated VL graph.

use std::{collections::BTreeMap, sync::Arc};

use eredu_core::{ArtifactInspection, SessionCapabilities};
use eredu_nn::{DistributedNeuralBackend, Error, GroupedNeuralBackend, Tensor};
use eredu_runtime::{
    ComponentDefinition, PreparedComponent, ProcessorSelectionRequest,
    ReplicatedTextSelectionRequest, SelectedCompositeRealization, WeightLoweringKind,
};

use crate::{
    processor_execution::PreparedProcessor, processor_plan::ArtifactArchitecturePlan,
    replicated_text::lfm3_vl_requirements, PreparationMechanismProvider,
};

use super::{
    PreparedVisionLanguage, PreparedVlWeights, SelectedVlRankPlacement, SelectedVlRankWeights,
    VlLayeredModel, VlWeightPolicies,
};

/// Session facilities implemented by the retained VL generation route.
pub const VL_SESSION_CAPABILITIES: SessionCapabilities =
    SessionCapabilities::new(false, true, false);

/// A cold selection paired with the admitted inspection from which it was derived.
pub struct SelectedVlGraph {
    inspection: ArtifactInspection<ArtifactArchitecturePlan>,
    selected: SelectedCompositeRealization,
}

/// Exact component bindings and processor proof retained for a native visitor.
#[derive(Clone)]
pub struct PreparedVlGraph {
    artifact: PreparedVisionLanguage,
    component: PreparedComponent<Arc<ComponentDefinition>>,
    selected: SelectedCompositeRealization,
}

/// Cold VL selection paired with an exact decoder rank and its communication.
pub struct SelectedVlRankGraph {
    graph: SelectedVlGraph,
    placement: SelectedVlRankPlacement,
    policies: VlWeightPolicies,
}

impl SelectedVlRankGraph {
    /// Manifest available to the loader before binding any checkpoint source.
    pub fn communication_manifest(&self) -> &eredu_runtime::CommunicationManifest {
        self.placement.communication_manifest()
    }

    /// Binds the retained recipes, validates rank residency, and retains the cold placement.
    pub fn prepare(self) -> Result<PreparedVlRankGraph, Error> {
        self.graph
            .prepare()?
            .bind_rank(self.placement, self.policies)
    }
}

/// Portable construction and session geometry for a VL decoder rank.
pub struct PreparedVlDecoderStage<A> {
    /// Generic inputs to the native pipeline binder.
    pub pipeline: A,
    /// Maximum prefill width admitted for vocabulary publication.
    pub prefix: std::num::NonZeroUsize,
    /// Rank-local mutable state layout.
    pub state_layout: eredu_runtime::StateLayout,
    /// Global ordinal of the first state-owning layer.
    pub global_layer_start: usize,
    /// Output publication selected for this stage.
    pub publication: eredu_runtime::PartitionOutputPublication,
    /// Rank-local output authority.
    pub output: eredu_runtime::PartitionOutputAuthority,
}

impl<A> PreparedVlDecoderStage<A> {
    /// Binds the pipeline while retaining the portable session geometry.
    pub fn bind<D, E>(
        self,
        bind: impl FnOnce(A) -> Result<D, E>,
    ) -> Result<PreparedVlDecoderStage<D>, E> {
        Ok(PreparedVlDecoderStage {
            pipeline: bind(self.pipeline)?,
            prefix: self.prefix,
            state_layout: self.state_layout,
            global_layer_start: self.global_layer_start,
            publication: self.publication,
            output: self.output,
        })
    }
}

/// Constructs the language component and retains VL-specific session geometry.
pub fn prepare_vl_decoder_stage<B, S>(
    prepared: crate::lfm3::PreparedLanguagePartition,
    residency: eredu_runtime::LayerWeightResidency,
    communication: crate::lfm3::LanguagePipelineCommunication,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<
    PreparedVlDecoderStage<
        eredu_runtime::PreparedComponentPipeline<crate::lfm3::PartitionedLayeredModel<B>>,
    >,
    eredu_core::BackendFailure,
>
where
    B: GroupedNeuralBackend + DistributedNeuralBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
    S::LayerState: eredu_nn::AttentionCache<B::Tensor> + eredu_runtime::RuntimeStateComponents<B>,
{
    use eredu_core::BackendFailure;

    let contract = |message| BackendFailure::from_error(Error::backend(message));
    let prefix = prepared
        .vocabulary_prefix()
        .ok_or_else(|| contract("LFM3 VL pipeline requires a vocabulary prefix limit"))?;
    let state_layout = prepared
        .partition()
        .state()
        .ok_or_else(|| contract("LFM3 VL stage has no cache layout"))?
        .layout()
        .clone();
    let global_layer_start = prepared
        .rank()
        .layer_range(prepared.complete_layer_count())
        .map_err(BackendFailure::from_error)?
        .start;
    let publication = communication
        .plan
        .publication()
        .ok_or_else(|| contract("VL partition omitted output publication"))?;
    let output = communication
        .plan
        .publication_authority(&communication.manifest)
        .map_err(BackendFailure::from_error)?
        .ok_or_else(|| contract("VL partition omitted output authority"))?;
    let (architecture, source, invocations, pinned, units) =
        prepared.into_layerwise::<B, S>(context)?;
    let units = units
        .into_iter()
        .enumerate()
        .map(|(ordinal, bindings)| {
            let address = invocations
                .storage_layout()
                .address(ordinal)
                .ok_or_else(|| contract("VL decoder binding has no retained storage owner"))?;
            Ok((address, bindings))
        })
        .collect::<Result<_, BackendFailure>>()?;
    Ok(PreparedVlDecoderStage {
        pipeline: eredu_runtime::PreparedComponentPipeline {
            architecture,
            source,
            invocations,
            pinned,
            units,
            manifest: communication.manifest,
            plan: communication.plan,
            tensor_group: communication.tensor_group,
            activation_dtype: eredu_runtime::PipelineActivationDtype::Float32,
            residency,
        },
        prefix,
        state_layout,
        global_layer_start,
        publication,
        output,
    })
}

/// Exact rank bindings with the processor and request admission for the generation session.
pub struct PreparedVlRankGraph {
    weights: SelectedVlRankWeights,
    admission: super::VisionPromptAdmission,
    capabilities: crate::capability::CapabilityEstimate,
    selected: SelectedCompositeRealization,
    raw_processor: Option<PreparedProcessor>,
}

impl PreparedVlRankGraph {
    /// Prepared weights and communication for this rank.
    pub fn weights(&self) -> &SelectedVlRankWeights {
        &self.weights
    }

    pub(crate) fn processor(&self) -> Option<PreparedProcessor> {
        self.raw_processor.clone()
    }

    /// Constructs the selected session with caller-supplied component and state mechanisms.
    ///
    /// The portable graph owns admission and binding order. The mechanisms
    /// consume the retained resources and return the local session components.
    pub fn construct_session<B, S, V, D>(
        self,
        context: &<B::Tensor as Tensor>::Context,
        bind_pipeline: impl FnOnce(
            eredu_runtime::PreparedComponentPipeline<crate::lfm3::PartitionedLayeredModel<B>>,
        ) -> Result<D, eredu_core::BackendFailure>,
        bind_vision: impl FnOnce(
            super::PreparedVisionExecution,
            eredu_runtime::LayerWeightResidency,
        ) -> Result<V, eredu_core::BackendFailure>,
        allocate_state: impl FnOnce(
            eredu_runtime::StateLayout,
            usize,
        ) -> Result<S, eredu_core::BackendFailure>,
    ) -> Result<BoundVlRankSession<V, PreparedVlDecoderStage<D>, S>, eredu_core::BackendFailure>
    where
        B: GroupedNeuralBackend + DistributedNeuralBackend,
        S: eredu_runtime::LayerRuntimeState<B>,
        S::LayerState: eredu_nn::AttentionCache<B::Tensor> + eredu_runtime::RuntimeStateComponents<B>,
    {
        let decoder_residency = self.weights.decoder_residency();
        let (weights, admission, capabilities, processor, raw_processor, session, dtype) =
            self.into_parts();
        eredu_core::SessionAdmission::new(session)
            .validate(super::VL_SESSION_CAPABILITIES)
            .map_err(eredu_core::BackendFailure::from_error)?;
        let (vision, decoder) = weights.bind_components(
            |decoder, residency, communication| {
                prepare_vl_decoder_stage::<B, S>(decoder, residency, communication, context)?
                    .bind(bind_pipeline)
            },
            bind_vision,
        )?;
        let state = allocate_state(decoder.state_layout.clone(), decoder.global_layer_start)?;
        Ok(BoundVlRankSession {
            vision,
            decoder,
            state,
            admission,
            capabilities,
            processor,
            raw_processor,
            session,
            dtype,
            decoder_residency,
        })
    }

    /// Transfers the rank binding and the retained request and processor policies.
    pub fn into_parts(
        self,
    ) -> (
        SelectedVlRankWeights,
        super::VisionPromptAdmission,
        crate::capability::CapabilityEstimate,
        eredu_runtime::SelectedProcessorExecution,
        Option<PreparedProcessor>,
        SessionCapabilities,
        std::num::NonZeroU8,
    ) {
        let execution = self.selected.execution();
        let session = execution.session();
        let dtype = execution
            .state()
            .floating_dtype()
            .expect("admitted VL rank state dtype")
            .bytes();
        let (_, processor) = self.selected.into_parts();
        (
            self.weights,
            self.admission,
            self.capabilities,
            processor,
            self.raw_processor,
            session,
            dtype,
        )
    }
}

/// Bound components and retained admission for one selected VL rank.
pub struct BoundVlRankSession<V, D, S> {
    /// Replicated vision execution.
    pub vision: V,
    /// Rank-local language execution.
    pub decoder: D,
    /// Initial decoder state.
    pub state: S,
    /// VL prompt admission for the selected execution.
    pub admission: super::VisionPromptAdmission,
    /// Retained capability estimate.
    pub capabilities: crate::capability::CapabilityEstimate,
    /// Selected processor execution.
    pub processor: eredu_runtime::SelectedProcessorExecution,
    /// Prepared raw-media processor, when selected.
    pub raw_processor: Option<PreparedProcessor>,
    /// Admitted session capabilities.
    pub session: SessionCapabilities,
    /// Physical floating-state width in bytes.
    pub dtype: std::num::NonZeroU8,
    /// Selected rank-local decoder weight residency.
    pub decoder_residency: eredu_runtime::LayerWeightResidency,
}

impl SelectedVlGraph {
    /// Selects supported text execution and processor mechanisms without opening weight payloads.
    pub fn select(
        inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
        execution: &ReplicatedTextSelectionRequest,
        processor: &ProcessorSelectionRequest,
        mechanisms: &impl PreparationMechanismProvider,
    ) -> Result<Self, Error> {
        let requirements = lfm3_vl_requirements(inspection).map_err(Error::backend_source)?;
        let selected_execution = eredu_runtime::select_replicated_text_realization(
            &requirements.execution,
            execution,
            &mechanisms.replicated_text_capabilities(&requirements.execution, execution),
        )
        .map_err(Error::backend_source)?;
        let selected_processor = eredu_runtime::select_processor_execution(
            &requirements.processor,
            processor,
            &mechanisms.processor_capabilities(),
        )
        .map_err(Error::backend_source)?;
        if selected_execution.topology().world_size() != 1
            || selected_execution.uses_weight_transform()
        {
            return Err(Error::backend(
                "selected VL graph requires a replicated dense binding",
            ));
        }
        Ok(Self {
            inspection: inspection.clone(),
            selected: SelectedCompositeRealization::from_parts(
                selected_execution,
                selected_processor,
            ),
        })
    }

    pub(crate) fn into_realization(self) -> SelectedCompositeRealization {
        self.selected
    }

    /// Retains decoder geometry and communication for a rank before preparing sources.
    pub fn select_rank(
        self,
        rank: eredu_core::ParallelRankTopology,
        prefix: std::num::NonZeroUsize,
        policies: VlWeightPolicies,
    ) -> Result<SelectedVlRankGraph, Error> {
        validate_rank_state(self.selected.execution())?;
        let placement = self
            .inspection
            .architecture_plan()
            .lfm3_vl_declaration()
            .ok_or_else(|| Error::backend("selected graph omitted its VL declaration"))?
            .select_rank_placement(rank, prefix)?;
        Ok(SelectedVlRankGraph {
            graph: self,
            placement,
            policies,
        })
    }

    /// Binds the admitted source views and validates the exact selected lowering recipes.
    pub fn prepare(self) -> Result<PreparedVlGraph, Error> {
        let execution = self.selected.execution();
        let artifact = PreparedVisionLanguage::from_inspection(
            &self.inspection,
            execution.max_cached_shards(),
        )?;
        PreparedVlGraph::from_artifact(artifact, self.selected)
    }
}

impl PreparedVlGraph {
    pub(crate) fn from_source_graph(
        graph: &crate::prepared_sources::PreparedModelSourceGraph,
        selected: SelectedCompositeRealization,
    ) -> Result<Self, Error> {
        Self::from_artifact(PreparedVisionLanguage::from_source_graph(graph)?, selected)
    }

    fn from_artifact(
        artifact: PreparedVisionLanguage,
        selected: SelectedCompositeRealization,
    ) -> Result<Self, Error> {
        let execution = selected.execution();
        let source = artifact.component_source()?;
        eredu_runtime::validate_checkpoint_handoff(
            execution.requirements(),
            source.as_ref(),
            eredu_runtime::SourceHandoffScope::Primary,
        )
        .map_err(Error::backend_source)?;
        let definition = execution
            .requirements()
            .shared_component_definition()
            .ok_or_else(|| Error::backend("selected VL execution omitted the component"))?;
        let recipes = execution
            .materialization_tasks()
            .iter()
            .map(|task| {
                if !matches!(
                    task.lowering(),
                    WeightLoweringKind::Direct | WeightLoweringKind::Derived
                ) || !task.output_companions().is_empty()
                {
                    return Err(Error::backend(
                        "VL graph requires a direct dense parameter binding",
                    ));
                }
                let recipe = task.source_recipe().map_err(Error::backend_source)?;
                Ok((task.name(), recipe))
            })
            .collect::<Result<BTreeMap<_, _>, Error>>()?;
        let component = PreparedComponent::prepare(
            definition.id(),
            definition.clone(),
            source,
            |member| -> Result<_, eredu_runtime::ComponentPreparationError> {
                recipes.get(member.target()).cloned().ok_or_else(|| {
                    eredu_runtime::ComponentPreparationError::ArchitectureMismatch(format!(
                        "selected VL execution omitted parameter {:?}",
                        member.target(),
                    ))
                })
            },
        )
        .map_err(Error::backend_source)?;
        Ok(PreparedVlGraph {
            artifact,
            component,
            selected,
        })
    }

    pub(crate) fn processor(&self) -> Option<PreparedProcessor> {
        self.selected.processor().raw_media().then(|| {
            self.artifact
                .processor()
                .with_selected_modalities(self.selected.processor().clone())
        })
    }

    /// Exact residency, state and processor proofs retained by this graph.
    pub fn selected(&self) -> &SelectedCompositeRealization {
        &self.selected
    }

    /// Request admission derived from the same artifact as the selected graph.
    pub fn prompt_admission(&self) -> Result<super::VisionPromptAdmission, Error> {
        super::VisionPromptAdmission::new(&self.artifact)
    }

    /// Capability projection for the modalities authorized by the selected processor.
    pub fn capabilities(
        &self,
    ) -> Result<crate::capability::CapabilityEstimate, eredu_core::CapabilityError> {
        crate::capability::lfm3_vl(
            &self.artifact.language,
            self.selected.processor().modalities(),
        )
    }

    /// Projects the already selected lowering recipes onto a distributed decoder rank.
    pub fn select_rank(
        self,
        rank: eredu_core::ParallelRankTopology,
        prefix: std::num::NonZeroUsize,
        policies: VlWeightPolicies,
    ) -> Result<PreparedVlRankGraph, Error> {
        let placement =
            SelectedVlRankPlacement::select(self.artifact.language.clone(), rank, prefix)?;
        self.bind_rank(placement, policies)
    }

    /// Binds the selected rank to the retained lowering recipes without rebuilding its plan.
    pub fn bind_rank(
        self,
        placement: SelectedVlRankPlacement,
        policies: VlWeightPolicies,
    ) -> Result<PreparedVlRankGraph, Error> {
        self.bind_rank_with(placement, |weights| weights.select(policies))
    }

    pub(crate) fn bind_rank_with_load_budget(
        self,
        placement: SelectedVlRankPlacement,
    ) -> Result<PreparedVlRankGraph, Error> {
        let policy = self.selected.execution().residency();
        self.bind_rank_with(placement, |weights| weights.select_shared(policy))
    }

    fn bind_rank_with(
        self,
        placement: SelectedVlRankPlacement,
        select: impl FnOnce(super::PreparedVlRankWeights) -> Result<SelectedVlRankWeights, Error>,
    ) -> Result<PreparedVlRankGraph, Error> {
        validate_rank_state(self.selected.execution())?;
        let admission = self.prompt_admission()?;
        let capabilities = self.capabilities().map_err(Error::backend_source)?;
        let raw_processor = self.processor();
        let weights = PreparedVlWeights::from_component(&self.artifact, self.component)?
            .bind_rank(placement)?;
        let weights = select(weights)?;
        Ok(PreparedVlRankGraph {
            weights,
            admission,
            capabilities,
            raw_processor,
            selected: self.selected,
        })
    }

    /// Constructs typed equations and transfers the retained execution and processor proofs.
    pub fn into_layerwise<B>(
        self,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<
        (
            VlLayeredModel<B>,
            PreparedComponent<Arc<ComponentDefinition>>,
            SelectedCompositeRealization,
            Option<PreparedProcessor>,
        ),
        Error,
    >
    where
        B: GroupedNeuralBackend + DistributedNeuralBackend,
    {
        let architecture = self.artifact.layerwise_for::<B>(&self.component, context)?;
        let processor = self.processor();
        Ok((architecture, self.component, self.selected, processor))
    }
}

pub(crate) fn validate_rank_state(
    execution: &eredu_runtime::SelectedReplicatedTextRealization,
) -> Result<(), Error> {
    if execution.state().floating_dtype().is_none()
        || execution.state().policy() != &eredu_runtime::CacheResidencyPolicy::Device
        || execution.parameter_conversion_retention().is_some()
        || execution.prompt_cache()
    {
        return Err(Error::backend(
            "VL rank graph requires device cache and no conversion or prompt-cache retention",
        ));
    }
    Ok(())
}
