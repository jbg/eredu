//! Retained portable artifact, text, and execution admission.

use std::path::Path;

use eredu_core::{
    DraftingPlan, ExecutionPlanBackendFactory, ExternalDraftArtifact, ModelInspectionReport,
    ModelLoadingBackend, TextGenerationBackend, TokenizerCompatibilityProof,
};
use eredu_text::tokenizer::Tokenizer;

use super::{
    LoadedModel, LoadedModelLoadError, LoadedTextModelConfig, PlannedModel, PlannedModelLoadError,
    TextInspectionOptions, TextModelOptions,
};

pub use eredu_checkpoint::safetensors::{
    SafetensorsMetadataCatalog as SafetensorsSnapshot, SafetensorsShardError,
};
pub use eredu_core::artifact::{
    ArtifactError, ArtifactSource, CompanionSelection, ArtifactCompanionRole,
};

/// Admitted local artifact and architecture facts, without native resources.
pub type ArtifactInspection =
    eredu_core::ArtifactInspection<eredu_architectures::processor_plan::ArtifactArchitecturePlan>;

/// Admits a local artifact and its explicitly selected or discovered companions.
/// No tensor payloads, native devices, or execution streams are created.
pub fn inspect_artifact(source: &ArtifactSource) -> Result<ArtifactInspection, ArtifactError> {
    eredu_architectures::configuration::inspect_artifact_source(source)
}

/// Checks complete SafeTensors headers and exact index membership without MLX.
/// This establishes structural completeness, not payload checksum integrity,
/// architecture support, tokenizer readiness, or backend availability.
pub fn inspect_safetensors_snapshot(
    path: impl AsRef<Path>,
) -> Result<SafetensorsSnapshot, SafetensorsShardError> {
    SafetensorsSnapshot::discover(path)
}

/// Exact admitted artifact with resolved tokenizer, template and generation policy.
/// This value contains no native device, tensor, or stream. Loading consumes the
/// same text state that inspection probes, without reopening text sidecars.
pub struct PreparedTextModel {
    pub(super) artifact: ArtifactInspection,
    pub(super) tokenizer: Tokenizer,
    pub(super) config: LoadedTextModelConfig,
}

impl PreparedTextModel {
    /// Resolves text policy once from an admitted local artifact.
    /// Metadata-only inspections cannot grant local payload-loading authority.
    pub fn from_artifact(
        mut artifact: ArtifactInspection,
        options: TextModelOptions,
    ) -> Result<Self, LoadedModelLoadError> {
        if artifact.metadata_provenance().is_some() {
            return Err(ArtifactError::MetadataOnly.into());
        }
        let (tokenizer, config) = super::loaded::loaded_text_artifact(&artifact, options)?;
        super::loaded::bind_gguf_special_token_ids(&mut artifact, &tokenizer)?;
        Ok(Self {
            artifact,
            tokenizer,
            config,
        })
    }

    /// Exact retained artifact, including the selected projector and shard set.
    pub fn artifact(&self) -> &ArtifactInspection {
        &self.artifact
    }

    /// Resolved text policy used by inspection and eventual loading.
    pub fn text_config(&self) -> &LoadedTextModelConfig {
        &self.config
    }

    /// Validates one complete execution plan without realizing native resources.
    /// The resulting value retains the factory, plan, target selection and
    /// drafting selection; loading accepts no replacement options.
    pub fn select_execution_plan<'a, F, B>(
        self,
        factory: &'a F,
        plan: &eredu_core::ExecutionPlan,
    ) -> Result<SelectedTextModel<'a, F>, PlannedModelLoadError>
    where
        F: ExecutionPlanBackendFactory<
                Backend = B,
                DrafterPreparation = eredu_architectures::ExternalDraftPreparation,
            >,
        B: ModelLoadingBackend + TextGenerationBackend,
        B::ConfigurationResolver: eredu_core::ModelConfigurationResolver<
                ArtifactPlan = eredu_architectures::processor_plan::ArtifactArchitecturePlan,
            >,
    {
        let Self {
            artifact,
            tokenizer,
            config,
        } = self;
        let external = match plan.drafting() {
            DraftingPlan::External { model, .. } => {
                let preparation = eredu_architectures::prepare_external_draft(model)
                    .map_err(LoadedModelLoadError::Artifact)?;
                let target_fingerprint = eredu_text::tokenizer::vocabulary_fingerprint(&tokenizer);
                let draft_fingerprint = match preparation.tokenizer_source() {
                    eredu_architectures::external_draft::ExternalDraftTokenizerSource::Target => {
                        target_fingerprint
                    }
                    eredu_architectures::external_draft::ExternalDraftTokenizerSource::Artifact {
                        model_type,
                    } => {
                        let draft_tokenizer = super::tokenizer::load_tokenizer_for_model_type(
                            model_type,
                            Path::new(model),
                        )
                        .map_err(LoadedModelLoadError::Metadata)?;
                        eredu_text::tokenizer::vocabulary_fingerprint(&draft_tokenizer)
                    }
                };
                let compatibility = TokenizerCompatibilityProof::prove(
                    target_fingerprint,
                    draft_fingerprint,
                )
                .map_err(|error| eredu_core::AutomaticPlanningError::Invalid(error.to_string()))?;
                Some(ExternalDraftArtifact {
                    preparation,
                    tokenizer_compatibility: compatibility,
                })
            }
            DraftingPlan::Disabled | DraftingPlan::Embedded { .. } => None,
            _ => return Err(LoadedModelLoadError::UnsupportedDraftingPlan.into()),
        };
        let target = eredu_core::select_execution_plan_target(factory, plan, artifact)?;
        let drafting =
            eredu_core::select_execution_plan_drafting(factory, plan, &target, external)?;
        Ok(SelectedTextModel {
            factory,
            target,
            drafting,
            tokenizer,
            config,
        })
    }

    /// Loads the retained artifact and text state on an already created backend.
    /// Backend construction and placement are owned by the caller on this path.
    pub fn load_on<B>(
        self,
        backend: B,
        options: B::LoadOptions,
    ) -> Result<LoadedModel<B>, LoadedModelLoadError>
    where
        B: ModelLoadingBackend + TextGenerationBackend,
        B::ConfigurationResolver: eredu_core::ModelConfigurationResolver<
                ArtifactPlan = eredu_architectures::processor_plan::ArtifactArchitecturePlan,
            >,
    {
        LoadedModel::from_inspected(backend, self.artifact, options, self.tokenizer, self.config)
    }
}

/// Complete cold execution admission with the exact text state and factory.
/// Native realization consumes this value and cannot substitute another plan.
pub struct SelectedTextModel<'a, F: ExecutionPlanBackendFactory>
where
    F::Backend: TextGenerationBackend,
{
    factory: &'a F,
    target: eredu_core::SelectedExecutionPlanTarget<F::Backend>,
    drafting: eredu_core::SelectedExecutionPlanDrafting<F::SelectedDrafterPreparation>,
    tokenizer: Tokenizer,
    config: LoadedTextModelConfig,
}

impl<F: ExecutionPlanBackendFactory> SelectedTextModel<'_, F>
where
    F::Backend: TextGenerationBackend,
    <F::Backend as ModelLoadingBackend>::ConfigurationResolver:
        eredu_core::ModelConfigurationResolver<
                ArtifactPlan = eredu_architectures::processor_plan::ArtifactArchitecturePlan,
            >,
{
    /// Exact execution plan whose target and drafting admission are retained.
    pub fn execution_plan(&self) -> &eredu_core::ExecutionPlan {
        self.target.execution_plan()
    }

    /// Exact artifact retained by execution admission.
    pub fn artifact(&self) -> &ArtifactInspection {
        self.target.inspection()
    }

    /// Reports the retained target selection with probes of its actual text state.
    /// Optional chat probes describe the supplied request, not every future chat.
    pub fn inspect(&mut self, options: TextInspectionOptions) -> ModelInspectionReport {
        super::inspection::enrich_text_inspection(
            self.target.report().clone(),
            &mut self.tokenizer,
            &self.config,
            options,
        )
    }

    /// Realizes the already selected target and drafting mode without discovery,
    /// text-sidecar reads, policy normalization, or another cold selection.
    pub fn load(
        self,
    ) -> Result<
        PlannedModel<F::Backend, eredu_core::RealizedDrafting<F::Drafter>>,
        PlannedModelLoadError,
    > {
        let plan = self.target.execution_plan().clone();
        let realized = eredu_core::realize_execution_plan_target(self.factory, &plan, self.target)?;
        let runtime = realized
            .into_runtime()
            .map_err(super::loaded::map_model_load_error)?;
        let model = LoadedModel::from_runtime(runtime, self.tokenizer, self.config);
        let drafting = eredu_core::realize_execution_plan_drafting(
            self.factory,
            &plan,
            model.runtime(),
            self.drafting,
        )?;
        Ok(PlannedModel::new(model, drafting, plan.drafting().clone()))
    }
}
