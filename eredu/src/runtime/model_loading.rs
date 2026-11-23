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

impl<B, N> TextSessionComposition for B
where
    B: eredu_core::backend::SharedTextNative<Native = N>,
    N: eredu_runtime::text_session_driver::TextSessionMechanisms<B>,
    eredu_runtime::text_session_driver::PreparedTextDriver<B, N>: TextExecutionSession<B>
        + ModelSessionInstaller<
            B,
            Session = eredu_runtime::text_session_driver::PreparedTextDriver<B, N>,
        >,
{
    type Model = <Self::Installer as ModelSessionInstaller<B>>::Model;
    type Installed = eredu_runtime::text_session_driver::PreparedTextDriver<B, N>;
    type Installer = Self::Installed;
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
            let mut preparation = eredu_architectures::prepare_external_draft(model)
                .map_err(LoadedModelLoadError::Artifact)?;
            let compatibility = prove_draft_tokenizer(tokenizer, &preparation)?;
            if let eredu_architectures::ExternalDraftPreparation::Autoregressive(artifact) =
                &mut preparation
            {
                // The exact vocabulary proof permits the same media token IDs
                // to be resolved for this independent artifact.
                bind_gguf_special_token_ids(artifact, tokenizer)
                    .map_err(LoadedModelLoadError::Metadata)?;
            }
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

pub(crate) fn bind_gguf_special_token_ids(
    inspection: &mut eredu_core::ArtifactInspection<
        eredu_architectures::processor_plan::ArtifactArchitecturePlan,
    >,
    tokenizer: &Tokenizer,
) -> Result<(), crate::api::tokenizer::TextMetadataError> {
    use crate::api::tokenizer::TextMetadataError;
    use eredu_architectures::processor_plan::{
        GgufSpecialTokenIds, GgufSpecialTokenKind, QwenMediaTokenIds,
    };

    let Some(kind) = inspection
        .architecture_plan()
        .required_gguf_special_tokens()
    else {
        return Ok(());
    };
    let required = |token: &str| {
        tokenizer.token_to_id(token).ok_or_else(|| {
            TextMetadataError::TokenizerConfiguration(format!(
                "GGUF tokenizer is missing required media token {token:?}"
            ))
        })
    };
    let ids = match kind {
        GgufSpecialTokenKind::Qwen => GgufSpecialTokenIds::Qwen(QwenMediaTokenIds {
            image_token_id: required("<|image_pad|>")?,
            video_token_id: required("<|video_pad|>")?,
            vision_start_token_id: required("<|vision_start|>")?,
            vision_end_token_id: required("<|vision_end|>")?,
        }),
        GgufSpecialTokenKind::Inkling => GgufSpecialTokenIds::Inkling {
            image_bos_token_id: required("<|content_image|>")?,
            audio_bos_token_id: required("<|content_audio_input|>")?,
        },
    };
    inspection
        .architecture_plan_mut()
        .bind_gguf_special_token_ids(ids)
        .map_err(|error| TextMetadataError::TokenizerConfiguration(error.to_string()))
}

fn prove_draft_tokenizer(
    tokenizer: &Tokenizer,
    preparation: &eredu_architectures::ExternalDraftPreparation,
) -> Result<TokenizerCompatibilityProof, PlannedModelLoadError> {
    use crate::api::tokenizer::{
        gguf_sidecar_dir, load_gguf_tokenizer_from_metadata, load_optional_gguf_tokenizer,
        load_optional_tokenizer_json, load_tokenizer_for_model_type,
    };
    use eredu_architectures::{
        external_assistant::ExternalAssistantCheckpoint,
        external_draft::ExternalDraftTokenizerSource, ExternalDraftPreparation,
    };
    use eredu_text::tokenizer::vocabulary_fingerprint;

    let (path, gguf) = match preparation {
        ExternalDraftPreparation::Autoregressive(artifact) => {
            (artifact.path(), artifact.gguf_checkpoint())
        }
        ExternalDraftPreparation::Assistant(assistant) => match assistant.checkpoint() {
            ExternalAssistantCheckpoint::SafeTensors { source, .. } => (source.as_path(), None),
            ExternalAssistantCheckpoint::Gguf { checkpoint, .. } => (
                checkpoint
                    .shards()
                    .first()
                    .expect("admitted GGUF has a shard")
                    .path(),
                Some(checkpoint),
            ),
        },
    };
    let draft_tokenizer = match (preparation.tokenizer_source(), gguf) {
        (ExternalDraftTokenizerSource::Target, None) => load_optional_tokenizer_json(path),
        (policy, Some(checkpoint)) => {
            let metadata = checkpoint
                .metadata()
                .iter()
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect();
            match policy {
                ExternalDraftTokenizerSource::Target => {
                    load_optional_gguf_tokenizer(gguf_sidecar_dir(path), &metadata)
                }
                ExternalDraftTokenizerSource::Artifact { .. } => {
                    load_gguf_tokenizer_from_metadata(path, &metadata)
                        .map(|loaded| Some(loaded.tokenizer))
                }
            }
        }
        (ExternalDraftTokenizerSource::Artifact { model_type }, None) => {
            load_tokenizer_for_model_type(model_type, path).map(Some)
        }
    }
    .map_err(LoadedModelLoadError::Metadata)?;
    let target = vocabulary_fingerprint(tokenizer);
    let draft = draft_tokenizer
        .as_ref()
        .map_or(target, vocabulary_fingerprint);
    TokenizerCompatibilityProof::prove(target, draft)
        .map_err(|error| eredu_core::AutomaticPlanningError::Invalid(error.to_string()).into())
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

pub(crate) fn map_model_load_error<E: std::error::Error + Send + Sync + 'static>(
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
