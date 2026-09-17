//! Portable external-assistant selection and its inseparable construction handoff.

use std::num::NonZeroUsize;

use eredu_core::{
    ArtifactInspection, AutomaticPlanningError, DraftingPlan, ExecutionPlan, ExternalDraftArtifact,
    ParallelRankTopology, TokenizerCompatibilityProof,
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
    source: PreparedCompatibleExternalAssistant,
    selected: SelectedSpeculativeRealization,
    tokenizer: TokenizerCompatibilityProof,
}

impl PreparedExternalAssistantExecution {
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
        let assistant = self.source.visit(visitor)?;
        Ok(MaterializedExternalAssistantExecution {
            assistant,
            source: super::ExternalSelectionSource::new(self.selected,capture),
            tokenizer: self.tokenizer,
        })
    }
}

/// A typed materialized assistant paired with its admitted execution contract.
pub struct MaterializedExternalAssistantExecution<V: ExternalAssistantPreparationVisitor> {
    assistant: MaterializedExternalAssistant<V>,
    source: super::ExternalSelectionSource,
    tokenizer: TokenizerCompatibilityProof,
}

impl<V: ExternalAssistantPreparationVisitor> MaterializedExternalAssistantExecution<V> {
    /// Runs an architecture-generic visitor against the selected materialized assistant.
    pub fn visit<W: MaterializedExternalAssistantVisitor<V>>(&mut self, visitor: W) -> W::Output {
        self.assistant.visit(visitor)
    }

    /// Lends the actual retained selection and capture beside the concrete
    /// assistant. No clone, reconstruction or replacement of either authority
    /// occurs, and a visitor cannot extend their loan beyond this invocation.
    pub fn visit_selected<W: SelectedExternalAssistantVisitor<V>>(&mut self, visitor: W) -> W::Output {
        self.assistant.visit(SelectedVisit {
            visitor, source: &self.source,
        })
    }

    /// Concrete transport storage for `visit_selected`, with no payload/source
    /// clone. Callers with admitted metadata pay it before entering the visitor.
    pub fn selected_visit_control_bytes<W: SelectedExternalAssistantVisitor<V>>() -> Option<usize> {
        let parts = [
            std::mem::size_of::<SelectedVisit<'_, W>>(),
            std::mem::size_of::<W::Output>(),
            std::mem::size_of::<(&mut Self, W)>(),
            std::mem::size_of::<&super::ExternalSelectionSource>(),
        ];
        parts.into_iter().try_fold(std::mem::size_of_val(&parts), usize::checked_add)
    }

    /// Returns the exact realization retained before source and native construction.
    pub fn selected(&self) -> &SelectedSpeculativeRealization {
        self.source.selected()
    }

    /// Returns the tokenizer proof retained by this construction.
    pub const fn tokenizer_compatibility(&self) -> TokenizerCompatibilityProof {
        self.tokenizer
    }

    /// Returns the target capture selected by architecture compatibility.
    pub fn capture(&self) -> &ExternalPredictionCaptureRequest {
        self.source.capture()
    }
}


/// Family-blind continuation borrowing the inseparable assistant selection,
/// architecture capture request and actual typed materialized assistant.
pub trait SelectedExternalAssistantVisitor<V: ExternalAssistantPreparationVisitor> {
    /// Result after all selected-source loans end.
    type Output;
    /// Runs with the selection and capture that preceded this materialization.
    fn visit<A: super::ExternalAssistantArchitecture>(
        self,
        assistant: &mut V::Output<A>,
        selected: &SelectedSpeculativeRealization,
        capture: &ExternalPredictionCaptureRequest,
    ) -> Self::Output;
    /// Lends the actual immutable loaded declaration owner when the consumer
    /// needs to retain it beyond this synchronous operation loan.
    fn visit_source<A:super::ExternalAssistantArchitecture>(self,assistant:&mut V::Output<A>,source:&super::ExternalSelectionSource)->Self::Output where Self:Sized {
        self.visit::<A>(assistant,source.selected(),source.capture())
    }
}
struct SelectedVisit<'a, W> {
    visitor: W,
    source: &'a super::ExternalSelectionSource,
}
impl<V, W> MaterializedExternalAssistantVisitor<V> for SelectedVisit<'_, W>
where
    V: ExternalAssistantPreparationVisitor,
    W: SelectedExternalAssistantVisitor<V>,
{
    type Output = W::Output;
    fn visit<A: super::ExternalAssistantArchitecture>(self, assistant: &mut V::Output<A>) -> Self::Output {
        self.visitor.visit_source::<A>(assistant,self.source)
    }
}

/// Selects an external assistant and prepares its source before native placement.
///
/// The retained target inspection establishes architecture compatibility. The
/// plan supplies portable transformation, reader-cache, placement, and capacity
/// policy; the backend supplies only exact lowering predicates and speculative
/// mechanism facts. Target weight residency does not change assistant residency.
pub fn prepare_execution_plan_assistant(
    plan: &ExecutionPlan,
    target: &ArtifactInspection<ArtifactArchitecturePlan>,
    artifact: ExternalDraftArtifact<ExternalAssistantPreparation>,
    lowering: impl Fn(&WeightLoweringDescriptor, bool) -> Option<WeightLoweringKind>,
    capabilities: &SpeculativeMechanismCapabilities,
) -> Result<ExternalDraftArtifact<PreparedExternalAssistantExecution>, AutomaticPlanningError> {
    let invalid =
        |error: &dyn std::fmt::Display| AutomaticPlanningError::Invalid(error.to_string());
    plan.validate_structure().map_err(|error| invalid(&error))?;
    let (placement, maximum_draft_tokens) = match plan.drafting() {
        DraftingPlan::External {
            placement,
            max_draft_tokens,
            ..
        } => (
            placement,
            NonZeroUsize::new(*max_draft_tokens).ok_or_else(|| {
                AutomaticPlanningError::Invalid("external draft capacity must be positive".into())
            })?,
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
    let preparation = preparation
        .prove_target_compatibility(&target_profile)
        .map_err(|error| {
            AutomaticPlanningError::Invalid(format!(
                "external assistant is incompatible with the selected target: {error}",
            ))
        })?;
    let placement_request =
        SpeculativePlacementRequest::from_topology(placement.execution_topology(plan.device()))
            .map_err(|error| invalid(&error))?;
    let rank_topology =
        ParallelRankTopology::new(*plan.topology(), 0).map_err(|error| invalid(&error))?;
    let processor = SpeculativeIdentity::new("prepared-chat/text-token-ids/v1")
        .map_err(|error| invalid(&error))?;
    let tokenizer = artifact.tokenizer_compatibility;
    let contract = preparation
        .speculative_contract(ExternalSpeculativeContractRequest::new(
            rank_topology,
            processor,
            tokenizer,
            tokenizer.fingerprint(),
            maximum_draft_tokens,
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
    let source = preparation
        .prepare_source(max_cached_sources)
        .map_err(|error| AutomaticPlanningError::Backend {
            operation: "prepare_external_drafter_source",
            message: error.to_string(),
        })?;
    Ok(ExternalDraftArtifact {
        preparation: PreparedExternalAssistantExecution {
            source,
            selected,
            tokenizer,
        },
        tokenizer_compatibility: tokenizer,
    })
}
