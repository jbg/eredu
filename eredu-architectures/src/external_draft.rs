//! Cold selection of feature-conditioned assistants and ordinary autoregressive drafts.
use crate::{prepared_sources::PreparedModelSources, processor_plan::ArtifactArchitecturePlan};
use eredu_core::{
    artifact::ArtifactError, ArtifactInspection, AutomaticPlanningError, DraftingPlan,
    ExecutionPlan, ExternalDraftArtifact, ModelPreparationPlan, TokenizerCompatibilityProof,
};
use eredu_runtime::*;

/// Inspected external draft, before backend resources exist.
pub enum ExternalDraftPreparation {
    /// A publisher-trained assistant consuming target features.
    Assistant(crate::ExternalAssistantPreparation),
    /// An independently executable tokenizer-compatible language model.
    Autoregressive(ArtifactInspection<ArtifactArchitecturePlan>),
}

/// Ownership of the tokenizer used to interpret an external draft's global token IDs.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum ExternalDraftTokenizerSource {
    /// The assistant is trained in the proven target's vocabulary and borrows its tokenizer.
    Target,
    /// An independent tokenizer is reconstructed from the selected draft artifact.
    Artifact {
        /// Architecture-declared tokenizer reconstruction identity.
        model_type: &'static str,
    },
}

impl ExternalDraftPreparation {
    /// Declares whether preparation borrows the target tokenizer or proves an independent one.
    pub fn tokenizer_source(&self) -> ExternalDraftTokenizerSource {
        match self {
            Self::Assistant(p) => p.tokenizer_source(),
            Self::Autoregressive(p) => ExternalDraftTokenizerSource::Artifact {
                model_type: p.architecture_plan().model_kind().canonical_name(),
            },
        }
    }
    /// Tokenizer policy for the exact admitted artifact.
    pub fn tokenizer_model_type(&self) -> &str {
        match self {
            Self::Assistant(p) => p.tokenizer_model_type(),
            Self::Autoregressive(p) => p.architecture_plan().model_kind().canonical_name(),
        }
    }
}
/// Inspects an external draft using its architecture's complete artifact contract.
pub fn prepare_external_draft(
    path: impl AsRef<std::path::Path>,
) -> Result<ExternalDraftPreparation, ArtifactError> {
    let path = path.as_ref();
    match crate::prepare_external_assistant(path) {
        Ok(assistant) => Ok(ExternalDraftPreparation::Assistant(assistant)),
        Err(
            ArtifactError::UnsupportedModelType(_) | ArtifactError::UnsupportedGgufArchitecture(_),
        ) => crate::configuration::inspect_artifact(path)
            .map(ExternalDraftPreparation::Autoregressive),
        Err(error) => Err(error),
    }
}
/// Exact prepared sources paired with independent-draft selection.
pub struct PreparedAutoregressiveDraft {
    sources: PreparedModelSources,
    selected: SelectedSpeculativeRealization,
    tokenizer: TokenizerCompatibilityProof,
}
impl PreparedAutoregressiveDraft {
    /// Consumes the inseparable native materialization handoff.
    pub fn into_parts(
        self,
    ) -> (
        PreparedModelSources,
        SelectedSpeculativeRealization,
        TokenizerCompatibilityProof,
    ) {
        (self.sources, self.selected, self.tokenizer)
    }
}
/// Architecture-selected external execution mechanism.
pub enum PreparedExternalDraft {
    /// Target-feature-conditioned assistant.
    Assistant(crate::PreparedExternalAssistantExecution),
    /// Two ordinary decoders with isolated caches.
    Autoregressive(PreparedAutoregressiveDraft),
}

/// Resolves draft ownership from the target's exact retained cold selection.
pub(crate) fn external_draft_target_rank(
    plan: &ExecutionPlan,
    target: &ArtifactInspection<ArtifactArchitecturePlan>,
    selected: &crate::SelectedPreparation,
) -> Result<eredu_core::ParallelRankTopology, AutomaticPlanningError> {
    plan.validate_structure()
        .map_err(|e| AutomaticPlanningError::Invalid(e.to_string()))?;
    if !target
        .admission_token()
        .same_admission(&selected.admission_token())
    {
        return Err(AutomaticPlanningError::Invalid(
            "external draft target selection belongs to a different artifact admission".into(),
        ));
    }
    let rank = selected.execution().parallel_topology().unwrap_or_else(|| {
        eredu_core::ParallelRankTopology::new(
            eredu_core::ParallelTopology::new(1, 1, 1, 1).expect("singleton topology"),
            0,
        )
        .expect("singleton rank")
    });
    if rank.topology() != *plan.topology() {
        return Err(AutomaticPlanningError::Invalid(
            "external draft plan topology differs from the retained target selection".into(),
        ));
    }
    Ok(rank)
}

/// Selects sources, equations, placement and tokenizer compatibility before native loading.
pub fn prepare_execution_plan_draft(
    plan: &ExecutionPlan,
    target: &ArtifactInspection<ArtifactArchitecturePlan>,
    target_selection: &crate::SelectedPreparation,
    artifact: ExternalDraftArtifact<ExternalDraftPreparation>,
    mechanisms: &impl crate::PreparationMechanismProvider,
    lowering: impl Fn(&WeightLoweringDescriptor, bool) -> Option<WeightLoweringKind>,
) -> Result<ExternalDraftArtifact<PreparedExternalDraft>, AutomaticPlanningError> {
    let invalid = |e: &dyn std::fmt::Display| AutomaticPlanningError::Invalid(e.to_string());
    let target_rank = external_draft_target_rank(plan, target, target_selection)?;
    let tokenizer = artifact.tokenizer_compatibility;
    let inspection = match artifact.preparation {
        ExternalDraftPreparation::Assistant(preparation) => {
            let prepared = crate::prepare_execution_plan_assistant(
                plan,
                target,
                target_selection,
                ExternalDraftArtifact {
                    preparation,
                    tokenizer_compatibility: tokenizer,
                },
                lowering,
                &mechanisms.speculative_capabilities(),
            )?;
            return Ok(ExternalDraftArtifact {
                preparation: PreparedExternalDraft::Assistant(prepared.preparation),
                tokenizer_compatibility: tokenizer,
            });
        }
        ExternalDraftPreparation::Autoregressive(p) => p,
    };
    let (placement, capacity, residency) = match plan.drafting() {
        DraftingPlan::External {
            placement,
            max_draft_tokens,
            residency,
            ..
        } => (placement, *max_draft_tokens, residency),
        _ => {
            return Err(invalid(
                &"independent drafting requires an external drafting plan",
            ))
        }
    };
    let capacity = std::num::NonZeroUsize::new(capacity)
        .ok_or_else(|| invalid(&"draft capacity must be positive"))?;
    let draft_capabilities = match (
        inspection.architecture_plan().safetensors_architecture(),
        inspection.architecture_plan().gguf_plan(),
    ) {
        (Some(p), _) => {
            crate::preparation::prepared_safetensors_capabilities(p).map_err(|e| invalid(&e))?
        }
        (_, Some(p)) => crate::preparation::prepared_gguf_capabilities(p),
        _ => return Err(invalid(&"draft omitted its admitted architecture")),
    };
    let mut draft_plan = plan
        .clone()
        .with_drafting(DraftingPlan::Disabled)
        .with_residency(residency.clone());
    if !draft_capabilities.independently_addressable_experts() {
        draft_plan = draft_plan.with_expert_cache(None);
    }
    let request = NormalizedLoadRequest::from_execution_plan(
        &draft_plan,
        ResidencyDiagnostics::default(),
        None,
    )
    .map_err(|e| invalid(&e))?;
    let selection =
        crate::select_preparation(&inspection, &request, mechanisms).map_err(|e| invalid(&e))?;
    // Cold identities describe admitted equations and tensor encodings. Content
    // hashes remain deferred on the prepared sources, as for conditioned drafts.
    // Never format the deferred identity itself: its debug state changes after
    // resolution and contains filesystem locations.
    let admission_identity = format!(
        "admission/independent-autoregressive/{:?}/{:?}/{:?}",
        inspection.format(),
        inspection.architecture_plan(),
        inspection.tensors()
    );
    let prepared = ModelPreparationPlan::from_retained_admission(inspection, selection.admission())
        .map_err(|e| invalid(&e))?;
    let sources = crate::prepared_sources::prepare_model_sources(prepared, selection)
        .map_err(|e| invalid(&e))?;
    let identity = |s: String| SpeculativeIdentity::new(s).map_err(|e| invalid(&e));
    let target_identity = identity(format!("ordinary-target/{:?}", target.architecture_plan()))?;
    let draft_identity = identity(format!("ordinary-draft/{}", sources.execution_identity()))?;
    let strategy_identity = identity("independent-autoregressive/v1".into())?;
    let capture = SpeculativeCaptureSchema::independent(identity(
        "independent-autoregressive/no-target-features/v1".into(),
    )?);
    let topology = identity(format!("{:?}", plan.topology()))?;
    let cache = SpeculativeStateCacheIdentityIngredients::new(
        target_identity.clone(),
        strategy_identity.clone(),
        Some(draft_identity.clone()),
        Some(tokenizer.fingerprint()),
        identity(admission_identity)?,
        identity(format!(
            "{:?}/{}",
            sources.format(),
            sources.execution_identity()
        ))?,
        topology,
        target_rank.global_rank(),
        identity("prepared-chat/text-token-ids/v1".into())?,
        [
            "target.cache",
            "draft.cache",
            "draft.proposal",
            "target.rollback",
        ]
        .into_iter()
        .map(|s| identity(s.into()))
        .collect::<Result<Vec<_>, _>>()?,
    )
    .map_err(|e| invalid(&e))?;
    let requirements = SpeculativeRealizationRequirements::new(
        target_identity.clone(),
        SpeculativeStrategyRequirements::external(
            strategy_identity.clone(),
            capacity,
            tokenizer.fingerprint(),
        ),
        capture.clone(),
        SpeculativeMechanismRequirements::new([]),
        cache,
    )
    .map_err(|e| invalid(&e))?;
    let proof = SpeculativeArchitectureCompatibilityProof::new(
        target_identity,
        strategy_identity,
        capture.identity().clone(),
    );
    let selection = SpeculativeSelectionRequest::new(
        SpeculativePlacementRequest::from_topology(placement.execution_topology(plan.device()))
            .map_err(|e| invalid(&e))?,
        capture,
    )
    .with_architecture_proof(proof)
    .with_tokenizer_proof(tokenizer);
    let selected = select_speculative_realization(
        &requirements,
        &selection,
        &mechanisms.speculative_capabilities(),
    )
    .map_err(|e| invalid(&e))?;
    Ok(ExternalDraftArtifact {
        preparation: PreparedExternalDraft::Autoregressive(PreparedAutoregressiveDraft {
            sources,
            selected,
            tokenizer,
        }),
        tokenizer_compatibility: tokenizer,
    })
}
