//! Portable artifact inspection and retained text-session loading.

use std::path::Path;

use eredu_core::backend::ModelSessionInstaller;
use eredu_core::{
    DraftingPlan, ExecutionPlan, ExecutionPlanBackendFactory, ExternalDraftArtifact,
    ModelLoadingBackend, ModelRuntime, RealizedDrafting, SelectedExecutionPlanDrafting,
    SelectedExecutionPlanTargetFor, TextExecutionSession, TextGenerationBackend,
    TokenizerCompatibilityProof,
};
use eredu_text::tokenizer::Tokenizer;

use crate::api::{
    ArtifactError, ArtifactInspection, ArtifactSource, LoadedModel, LoadedModelLoadError,
    LoadedTextModelConfig, PlannedModel, PlannedModelLoadError,
};

/// Selects the installer used when composing this text backend.
pub trait TextSessionComposition: TextGenerationBackend {
    /// Loaded executable and resources consumed by this composition.
    type Model;

    /// Portable session selected for the prepared resource contract.
    type Installed: TextExecutionSession<Self>;

    /// Installer for the prepared executable and the installed session.
    type Installer: ModelSessionInstaller<Self, Model = Self::Model, Session = Self::Installed>;
}

pub(crate) type PreparedTextProduct<B, S> = eredu_core::backend::LoadedModelFor<
    B,
    eredu_architectures::processor_plan::ArtifactArchitecturePlan,
    S,
>;

pub(crate) fn inspect(source: &ArtifactSource) -> Result<ArtifactInspection, ArtifactError> {
    eredu_architectures::configuration::inspect_artifact_source(source)
}

pub(crate) fn inspect_path(path: &Path) -> Result<ArtifactInspection, ArtifactError> {
    eredu_architectures::configuration::inspect_artifact(path)
}

pub(crate) fn inspect_execution(
    backend: &impl eredu_runtime::BackendInspection,
    path: &Path,
    plan: &ExecutionPlan,
    parallel: Option<eredu_runtime::ParallelLoadRequest>,
) -> Result<eredu_architectures::ModelInspectionOutcome, eredu_core::BackendFailure> {
    use eredu_runtime::execution_plan::{execution_plan_world_rank, ExecutionPlanPreparation};

    let request = ExecutionPlanPreparation {
        diagnostics: backend.residency_diagnostics(),
        parallel,
        native_support: backend
            .plan_placement(plan.device(), execution_plan_world_rank(parallel))
            .map(|_| ()),
    }
    .normalize(plan)
    .map_err(|error| {
        eredu_core::BackendFailure::from_error(error).with_operation("model inspection")
    })?;
    Ok(eredu_architectures::inspect_model(
        path,
        &request,
        &backend.preparation_mechanisms(),
        backend.media_features(),
    ))
}

pub(crate) fn load_on<B, A>(
    backend: B,
    inspection: ArtifactInspection,
    options: eredu_core::ModelLoadOptions<A::Request, B::LoadPlacement>,
) -> Result<ModelRuntime<B, B::Installed>, LoadedModelLoadError>
where
    B: ModelLoadingBackend + TextSessionComposition,
    A: eredu_core::backend::DirectLoadFacts<Mechanisms = B::LoadMechanisms>
        + eredu_core::ModelLoadAdmission<
            eredu_architectures::processor_plan::ArtifactArchitecturePlan,
            SourcePreparation: eredu_core::ModelSourcePreparation<
                eredu_architectures::processor_plan::ArtifactArchitecturePlan,
                Prepared: eredu_core::BindModelSources<B, Model = B::Model>,
            >,
        >,
{
    let prepared = eredu_core::prepare_inspected_model::<B, _, A>(&backend, inspection, options)
        .map_err(map_model_load_error)?;
    ModelRuntime::install::<B::Installer>(backend, prepared).map_err(|error| {
        LoadedModelLoadError::Backend(eredu_core::BackendFailure::from_error(error))
    })
}

#[allow(clippy::type_complexity)]
pub(crate) fn select_text<F>(
    factory: &F,
    plan: &ExecutionPlan,
    artifact: ArtifactInspection,
    tokenizer: &Tokenizer,
    parallel: F::TargetOptions,
) -> Result<
    (
        SelectedExecutionPlanTargetFor<
            F,
            eredu_architectures::processor_plan::ArtifactArchitecturePlan,
        >,
        SelectedExecutionPlanDrafting<
            eredu_core::DraftSelectionFor<
                F,
                eredu_architectures::processor_plan::ArtifactArchitecturePlan,
            >,
        >,
    ),
    PlannedModelLoadError,
>
where
    F: ExecutionPlanBackendFactory<
        eredu_architectures::processor_plan::ArtifactArchitecturePlan,
        DraftFacts: eredu_core::ExecutionPlanDraftAdmission<
            eredu_architectures::processor_plan::ArtifactArchitecturePlan,
            F::Backend,
            Inspection = eredu_architectures::ExternalDraftPreparation,
        >,
    >,
{
    let external = match plan.drafting() {
        DraftingPlan::External { model, .. } => {
            let preparation = eredu_architectures::prepare_external_draft(model)
                .map_err(LoadedModelLoadError::Artifact)?;
            let target_fingerprint = eredu_text::tokenizer::vocabulary_fingerprint(tokenizer);
            let draft_fingerprint = match preparation.tokenizer_source() {
                eredu_architectures::external_draft::ExternalDraftTokenizerSource::Target => {
                    target_fingerprint
                }
                eredu_architectures::external_draft::ExternalDraftTokenizerSource::Artifact {
                    model_type,
                } => {
                    let draft_tokenizer = crate::api::tokenizer::load_tokenizer_for_model_type(
                        model_type,
                        Path::new(model),
                    )
                    .map_err(LoadedModelLoadError::Metadata)?;
                    eredu_text::tokenizer::vocabulary_fingerprint(&draft_tokenizer)
                }
            };
            let compatibility =
                TokenizerCompatibilityProof::prove(target_fingerprint, draft_fingerprint).map_err(
                    |error| eredu_core::AutomaticPlanningError::Invalid(error.to_string()),
                )?;
            Some(ExternalDraftArtifact {
                preparation,
                tokenizer_compatibility: compatibility,
            })
        }
        DraftingPlan::Disabled | DraftingPlan::Embedded { .. } => None,
        _ => return Err(LoadedModelLoadError::UnsupportedDraftingPlan.into()),
    };
    let target = eredu_core::select_execution_plan_target(factory, plan, artifact, parallel)?;
    let drafting = eredu_core::select_execution_plan_drafting(factory, plan, &target, external)?;
    Ok((target, drafting))
}

pub(crate) fn realize_text<F>(
    factory: &F,
    target: SelectedExecutionPlanTargetFor<
        F,
        eredu_architectures::processor_plan::ArtifactArchitecturePlan,
    >,
    drafting: SelectedExecutionPlanDrafting<
        eredu_core::DraftSelectionFor<
            F,
            eredu_architectures::processor_plan::ArtifactArchitecturePlan,
        >,
    >,
    tokenizer: Tokenizer,
    config: LoadedTextModelConfig,
) -> Result<
    PlannedModel<
        F::Backend,
        <F::Backend as TextSessionComposition>::Installed,
        RealizedDrafting<F::Drafter>,
    >,
    PlannedModelLoadError,
>
where
    F: ExecutionPlanBackendFactory<eredu_architectures::processor_plan::ArtifactArchitecturePlan>,
    F::Backend:
        TextSessionComposition<Model = PreparedTextProduct<F::Backend, F::SourcePreparation>>,
{
    let plan = target.execution_plan().clone();
    let realized = eredu_core::realize_execution_plan_target(factory, &plan, target)?;
    let runtime = realized
        .install::<<F::Backend as TextSessionComposition>::Installer>()
        .map_err(map_model_load_error)?;
    let model = LoadedModel::from_runtime(runtime, tokenizer, config);
    let drafting =
        eredu_core::realize_execution_plan_drafting(factory, &plan, model.runtime(), drafting)?;
    Ok(PlannedModel::new(model, drafting, plan.drafting().clone()))
}

fn map_model_load_error<E: std::error::Error + Send + Sync + 'static>(
    error: eredu_core::ModelLoadError<E>,
) -> LoadedModelLoadError {
    match error {
        eredu_core::ModelLoadError::Artifact(error) => LoadedModelLoadError::Artifact(error),
        eredu_core::ModelLoadError::Selection(error) => LoadedModelLoadError::Selection(error),
        eredu_core::ModelLoadError::SourcePreparation(error) => {
            LoadedModelLoadError::SourcePreparation(error)
        }
        eredu_core::ModelLoadError::Backend(error) => {
            LoadedModelLoadError::Backend(eredu_core::BackendFailure::from_error(error))
        }
        eredu_core::ModelLoadError::Construction(error) => LoadedModelLoadError::Backend(error),
        eredu_core::ModelLoadError::SessionCapability(error) => {
            LoadedModelLoadError::SessionCapability(error)
        }
        error => {
            LoadedModelLoadError::Artifact(eredu_core::artifact::ArtifactError::InvalidArtifact(
                format!("unsupported model-loading failure: {error}"),
            ))
        }
    }
}
