//! Portable external-assistant selection and its inseparable construction handoff.

use std::num::NonZeroUsize;

use eredu_core::{
    ArtifactInspection, AutomaticPlanningError, DraftingPlan, ExecutionPlan, ExternalDraftArtifact,
    TokenizerCompatibilityProof,
};
use eredu_runtime::{
    execution_plan_quantization, select_speculative_realization, SelectedSpeculativeRealization,
    SpeculativeIdentity, SpeculativeMechanismCapabilities, SpeculativePlacementRequest,
    WeightLoweringDescriptor, WeightLoweringKind,
};

use super::{
    ExternalAssistantPreparation, ExternalAssistantPreparationVisitor,
    ExternalSpeculativeContractRequest, MaterializedExternalAssistant,
    MaterializedExternalAssistantVisitor, PreparedCompatibleExternalAssistant,
};
use crate::{
    composite_execution::ExternalPredictionCaptureRequest, processor_plan::ArtifactArchitecturePlan,
};

/// Exact source and speculative realization selected for one compatible assistant.
///
/// The source, target capture, placement, capacity, and tokenizer proof stay
/// paired through typed materialization. Backends cannot replace one part with
/// a separately prepared assistant or speculative selection.
pub struct PreparedExternalAssistantExecution {
    target: super::construction::AdmittedAssistantTarget,
    source: PreparedCompatibleExternalAssistant,
    selected: SelectedSpeculativeRealization,
    tokenizer: TokenizerCompatibilityProof,
    parameter_requirement: eredu_core::BoundedResidencyRequirement,
    source_placement: super::ExternalAssistantSourcePlacement,
}

impl PreparedExternalAssistantExecution {
    /// Exact target selection whose source placement and invocations are admitted.
    pub const fn target_selection(&self) -> &crate::SelectedPreparation {
        &self.target.selected
    }
    /// Exact local source placement selected against the retained target rank.
    pub const fn source_placement(&self) -> super::ExternalAssistantSourcePlacement {
        self.source_placement
    }
    /// Selected immutable parameter requirement, excluding state and workspace.
    pub const fn parameter_residency_requirement(&self) -> eredu_core::BoundedResidencyRequirement {
        self.parameter_requirement
    }
    /// Independently selected assistant parameter residency.
    pub fn residency(&self) -> eredu_runtime::LayerWeightResidency {
        self.source.residency()
    }
    /// Borrows the retained placement, proposal capacity, and mechanism selection.
    pub const fn selected(&self) -> &SelectedSpeculativeRealization {
        &self.selected
    }

    /// Materializes the exact prepared source while retaining its selection authority.
    pub fn materialize<V: ExternalAssistantPreparationVisitor>(
        self,
        visitor: V,
    ) -> Result<MaterializedExternalAssistantExecution<V>, V::Error> {
        let capture = self.source.capture().clone();
        let residency = self.source.residency();
        let assistant = self.source.visit(visitor)?;
        Ok(MaterializedExternalAssistantExecution {
            target: self.target,
            assistant,
            selected: self.selected,
            tokenizer: self.tokenizer,
            capture,
            residency,
            parameter_requirement: self.parameter_requirement,
            source_placement: self.source_placement,
        })
    }
}

/// A typed materialized assistant paired with its admitted execution contract.
pub struct MaterializedExternalAssistantExecution<V: ExternalAssistantPreparationVisitor> {
    target: super::construction::AdmittedAssistantTarget,
    assistant: MaterializedExternalAssistant<V>,
    selected: SelectedSpeculativeRealization,
    tokenizer: TokenizerCompatibilityProof,
    capture: ExternalPredictionCaptureRequest,
    residency: eredu_runtime::LayerWeightResidency,
    parameter_requirement: eredu_core::BoundedResidencyRequirement,
    source_placement: super::ExternalAssistantSourcePlacement,
}

impl<V: ExternalAssistantPreparationVisitor> MaterializedExternalAssistantExecution<V> {
    /// Target selection retained unchanged through assistant materialization.
    pub const fn target_selection(&self) -> &crate::SelectedPreparation {
        &self.target.selected
    }
    /// Source placement retained unchanged through native materialization.
    pub const fn source_placement(&self) -> super::ExternalAssistantSourcePlacement {
        self.source_placement
    }
    /// Cold parameter requirement retained without observing native allocations.
    pub const fn parameter_residency_requirement(&self) -> eredu_core::BoundedResidencyRequirement {
        self.parameter_requirement
    }
    /// Observes the selected assistant without acquiring execution mutation authority.
    pub fn visit_ref<W: super::MaterializedExternalAssistantRefVisitor<V>>(
        &self,
        visitor: W,
    ) -> W::Output {
        self.assistant.visit_ref(visitor)
    }

    /// The same assistant policy retained through native construction.
    pub const fn residency(&self) -> eredu_runtime::LayerWeightResidency {
        self.residency
    }
    /// Runs an architecture-generic visitor against the selected materialized assistant.
    pub fn visit<W: MaterializedExternalAssistantVisitor<V>>(&mut self, visitor: W) -> W::Output {
        self.assistant.visit(visitor)
    }

    /// Returns the exact realization retained before source and native construction.
    pub const fn selected(&self) -> &SelectedSpeculativeRealization {
        &self.selected
    }

    /// Returns the tokenizer proof retained by this construction.
    pub const fn tokenizer_compatibility(&self) -> TokenizerCompatibilityProof {
        self.tokenizer
    }

    /// Returns the target capture selected by architecture compatibility.
    pub const fn capture(&self) -> &ExternalPredictionCaptureRequest {
        &self.capture
    }
}

/// Selects an external assistant and prepares its source before native placement.
///
/// The retained target inspection and selection establish architecture compatibility
/// and exact rank ownership. A mismatched admission or topology fails before lowering. The
/// plan supplies portable transformation, reader-cache, placement, and capacity
/// policy; the backend supplies only exact lowering predicates and speculative
/// mechanism facts. Target weight residency does not change assistant residency.
pub fn prepare_execution_plan_assistant(
    plan: &ExecutionPlan,
    target: &ArtifactInspection<ArtifactArchitecturePlan>,
    target_selection: &crate::SelectedPreparation,
    artifact: ExternalDraftArtifact<ExternalAssistantPreparation>,
    lowering: impl Fn(&WeightLoweringDescriptor, bool) -> Option<WeightLoweringKind>,
    capabilities: &SpeculativeMechanismCapabilities,
) -> Result<ExternalDraftArtifact<PreparedExternalAssistantExecution>, AutomaticPlanningError> {
    let invalid =
        |error: &dyn std::fmt::Display| AutomaticPlanningError::Invalid(error.to_string());
    let rank_topology =
        crate::external_draft::external_draft_target_rank(plan, target, target_selection)?;
    let (placement, maximum_draft_tokens, residency) = match plan.drafting() {
        DraftingPlan::External {
            placement,
            max_draft_tokens,
            residency,
            ..
        } => (
            placement,
            NonZeroUsize::new(*max_draft_tokens).ok_or_else(|| {
                AutomaticPlanningError::Invalid("external draft capacity must be positive".into())
            })?,
            residency,
        ),
        _ => {
            return Err(AutomaticPlanningError::Invalid(
                "external assistant selection requires an external drafting plan".into(),
            ))
        }
    };
    let quantization = execution_plan_quantization(plan.weight_transformation())
        .map_err(|error| invalid(&error))?;
    let max_cached_sources = plan.max_cached_shards();
    let residency = eredu_runtime::execution_plan_layer_residency(
        residency,
        max_cached_sources,
        eredu_runtime::ResidencyDiagnostics::default(),
    )
    .map_err(|error| invalid(&error))?
    .layers();
    let preparation = artifact
        .preparation
        .select_materialization(quantization, max_cached_sources, lowering)
        .map_err(|message| AutomaticPlanningError::Backend {
            operation: "select_external_drafter",
            message,
        })?;
    let target_profile = target
        .architecture_plan()
        .external_assistant_target_profile()
        .ok_or_else(|| {
            AutomaticPlanningError::Invalid(
                "selected target does not admit an external assistant".into(),
            )
        })?;
    let mut preparation = preparation
        .prove_target_compatibility(&target_profile)
        .map_err(|error| {
            AutomaticPlanningError::Invalid(format!(
                "external assistant is incompatible with the selected target: {error}",
            ))
        })?
        .select_target_placement(target_selection)
        .map_err(|error| invalid(&error))?;
    preparation
        .admit_target_invocations(maximum_draft_tokens)
        .map_err(|error| invalid(&error))?;
    let placement_request =
        SpeculativePlacementRequest::from_topology(placement.execution_topology(plan.device()))
            .map_err(|error| invalid(&error))?;
    let processor = SpeculativeIdentity::new("prepared-chat/text-token-ids/v1")
        .map_err(|error| invalid(&error))?;
    let tokenizer = artifact.tokenizer_compatibility;
    let maximum_capture_sequence_length = target_selection
        .execution()
        .partitioned_invocation_limits()
        .map(|(_, sequence)| usize::try_from(sequence).map_err(|error| invalid(&error)))
        .transpose()?
        .and_then(NonZeroUsize::new)
        .unwrap_or(
            target_profile
                .maximum_capture_sequence_length()
                .map_err(|error| invalid(&error))?,
        );
    let contract = preparation
        .speculative_contract(ExternalSpeculativeContractRequest::new(
            rank_topology,
            processor,
            tokenizer,
            tokenizer.fingerprint(),
            maximum_draft_tokens,
            maximum_capture_sequence_length,
        ))
        .map_err(|error| {
            AutomaticPlanningError::Invalid(format!(
                "external speculative contract is invalid: {error}",
            ))
        })?;
    let selected = select_speculative_realization(
        contract.requirements(),
        &contract.selection_request(placement_request),
        capabilities,
    )
    .map_err(|error| invalid(&error))?;
    let target = preparation
        .take_admitted_target()
        .map_err(|error| invalid(&error))?;
    let source = preparation
        .prepare_source(max_cached_sources)
        .and_then(|source| source.with_residency(residency))
        .map(|source| {
            source.with_parameter_conversion_retention(plan.parameter_conversion_retention())
        })
        .map_err(|error| AutomaticPlanningError::Backend {
            operation: "prepare_external_drafter_source",
            message: error.to_string(),
        })?;
    let parameter_requirement = source
        .parameter_residency_requirement()
        .map_err(|error| invalid(&error))?;
    let source_placement = source.source_placement().ok_or_else(|| {
        AutomaticPlanningError::Invalid("assistant source omitted its selected placement".into())
    })?;
    Ok(ExternalDraftArtifact {
        preparation: PreparedExternalAssistantExecution {
            target,
            source,
            selected,
            tokenizer,
            parameter_requirement,
            source_placement,
        },
        tokenizer_compatibility: tokenizer,
    })
}
