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
