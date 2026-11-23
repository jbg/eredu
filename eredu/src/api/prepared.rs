//! Retained portable artifact, text, and execution admission.

use std::path::Path;

use eredu_core::{
    ExecutionPlanBackendFactory, ModelInspectionReport, ModelLoadingBackend, TextGenerationBackend,
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
    ArtifactCompanionRole, ArtifactError, ArtifactSource, CompanionSelection,
    QuantizedTensorRepresentation,
};

/// Admitted local artifact and architecture facts, without native resources.
pub type ArtifactInspection =
    eredu_core::ArtifactInspection<eredu_architectures::processor_plan::ArtifactArchitecturePlan>;

/// Admits a local artifact and its explicitly selected or discovered companions.
/// No tensor payloads, native devices, or execution streams are created.
pub fn inspect_artifact(source: &ArtifactSource) -> Result<ArtifactInspection, ArtifactError> {
    crate::runtime::model_loading::inspect(source)
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
        crate::runtime::model_loading::bind_gguf_special_token_ids(&mut artifact, &tokenizer)?;
        Ok(Self {
            artifact,
            tokenizer,
            config,
        })
    }

    /// Prepares a chat using the admitted tokenizer and template, without loading
    /// weights or creating a native device. Uses the ordinary request validator.
    pub fn prepare_chat(
        &mut self,
        request: crate::runtime::chat::ChatTemplateRequest,
    ) -> Result<crate::runtime::chat::PreparedChat, super::TextModelError> {
        let template = self
            .config
            .chat_template
            .clone()
            .ok_or(super::TextModelError::MissingChatTemplate)?;
        let compiler = crate::runtime::chat::constraints::ConstraintCompiler::from_tokenizer(
            &self.tokenizer,
            &self.config.eos_token_ids,
        );
        super::request::prepare_chat_from_parts(
            &mut self.tokenizer,
            template,
            &self.config.model_id,
            &self.config.eos_token_ids,
            Some(&compiler),
            request,
        )
    }

    /// Exact retained artifact, including the selected projector and shard set.
    pub fn artifact(&self) -> &ArtifactInspection {
        &self.artifact
    }

    /// Resolved text policy used by inspection and eventual loading.
    pub fn text_config(&self) -> &LoadedTextModelConfig {
        &self.config
    }

    /// Validates the execution plan and per-rank options without opening native resources.
    /// The resulting value retains the factory, plan, target selection and
    /// drafting selection; loading accepts no replacement options.
    pub fn select_execution_plan<'a, F>(
        self,
        factory: &'a F,
        plan: &eredu_core::ExecutionPlan,
        parallel: F::TargetOptions,
    ) -> Result<SelectedTextModel<'a, F>, PlannedModelLoadError>
    where
        F: ExecutionPlanBackendFactory<
            eredu_architectures::processor_plan::ArtifactArchitecturePlan,
            DraftFacts: eredu_core::ExecutionPlanDraftAdmission<
                eredu_architectures::processor_plan::ArtifactArchitecturePlan,
                F::Backend,
                Inspection = eredu_architectures::ExternalDraftPreparation,
            >,
        >,
        F::Backend: TextGenerationBackend,
    {
        let Self {
            artifact,
            tokenizer,
            config,
        } = self;
        let (target, drafting) = crate::runtime::model_loading::select_text(
            factory, plan, artifact, &tokenizer, parallel,
        )?;
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
    pub fn load_on<B, A>(
        self,
        backend: B,
        options: eredu_core::ModelLoadOptions<A::Request, B::LoadPlacement>,
    ) -> Result<LoadedModel<B, B::Installed>, LoadedModelLoadError>
    where
        B: ModelLoadingBackend + super::TextSessionComposition,
        A: eredu_core::backend::DirectLoadFacts<Mechanisms = B::LoadMechanisms>
            + eredu_core::ModelLoadAdmission<
                eredu_architectures::processor_plan::ArtifactArchitecturePlan,
                SourcePreparation: eredu_core::ModelSourcePreparation<
                    eredu_architectures::processor_plan::ArtifactArchitecturePlan,
                    Prepared: eredu_core::BindModelSources<B, Model = B::Model>,
                >,
            >,
    {
        let runtime =
            crate::runtime::model_loading::load_on::<B, A>(backend, self.artifact, options)?;
        Ok(LoadedModel::from_runtime(
            runtime,
            self.tokenizer,
            self.config,
        ))
    }
}

/// Complete cold execution admission with the exact text state and factory.
/// Native realization consumes this value and cannot substitute another plan.
pub struct SelectedTextModel<
    'a,
    F: ExecutionPlanBackendFactory<eredu_architectures::processor_plan::ArtifactArchitecturePlan>,
> where
    F::Backend: TextGenerationBackend,
{
    factory: &'a F,
    target: eredu_core::SelectedExecutionPlanTargetFor<
        F,
        eredu_architectures::processor_plan::ArtifactArchitecturePlan,
    >,
    drafting: eredu_core::SelectedExecutionPlanDrafting<
        eredu_core::DraftSelectionFor<
            F,
            eredu_architectures::processor_plan::ArtifactArchitecturePlan,
        >,
    >,
    tokenizer: Tokenizer,
    config: LoadedTextModelConfig,
}

impl<
        F: ExecutionPlanBackendFactory<eredu_architectures::processor_plan::ArtifactArchitecturePlan>,
    > SelectedTextModel<'_, F>
where
    F::Backend: TextGenerationBackend,
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
        PlannedModel<
            F::Backend,
            <F::Backend as super::TextSessionComposition>::Installed,
            eredu_core::RealizedDrafting<F::Drafter>,
        >,
        PlannedModelLoadError,
    >
    where
        F::Backend: super::TextSessionComposition<
            Model = crate::runtime::model_loading::PreparedTextProduct<
                F::Backend,
                F::SourcePreparation,
            >,
        >,
    {
        crate::runtime::model_loading::realize_text(
            self.factory,
            self.target,
            self.drafting,
            self.tokenizer,
            self.config,
        )
    }
}
