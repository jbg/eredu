//! Architecture-aware total inspection over portable artifacts and mechanism facts.

use std::path::Path;

use eredu_core::{
    ArtifactFormat, ArtifactInspection, InspectionIssueCode, InspectionReadiness,
    InspectionSeverity, MediaFeatureAvailability, MediaProjectorRequirement, ModelInspectionReport,
};
use eredu_runtime::{NormalizedLoadRequest, NormalizedLoadRequestError};

use crate::{
    preparation::{
        gguf_composite_artifact_plan, prepared_gguf_capabilities,
        prepared_safetensors_capabilities, ArchitectureCapabilities, GgufArtifactComposition,
        GgufMediaProjectorRequirement,
    },
    preparation_selection::{
        select_preparation, PreparationMechanismProvider, PreparationSelectionError,
    },
    prepared_sources::{prepare_model_sources, PreparedModelSources, PreparedModelSourcesError},
    processor_plan::ArtifactArchitecturePlan,
    SelectedPreparation,
};

/// A structurally inspected artifact paired with its exact neutral preparation selection.
#[derive(Debug, Clone)]
pub struct SelectedModelInspection {
    inspection: ArtifactInspection<ArtifactArchitecturePlan>,
    preparation: SelectedPreparation,
    memory_materialization_workspace:
        eredu_core::Observed<eredu_core::ParameterMaterializationWorkspace>,
}

impl SelectedModelInspection {
    /// Returns the authoritative portable artifact inspection.
    pub const fn inspection(&self) -> &ArtifactInspection<ArtifactArchitecturePlan> {
        &self.inspection
    }

    /// Returns the exact selection retained from report generation.
    pub const fn preparation(&self) -> &SelectedPreparation {
        &self.preparation
    }

    /// Cold recipe and conversion buffers for request memory planning.
    pub const fn memory_materialization_workspace(
        &self,
    ) -> &eredu_core::Observed<eredu_core::ParameterMaterializationWorkspace> {
        &self.memory_materialization_workspace
    }

    /// Consumes both retained inputs for later source preparation and materialization.
    pub fn into_parts(
        self,
    ) -> (
        ArtifactInspection<ArtifactArchitecturePlan>,
        SelectedPreparation,
    ) {
        (self.inspection, self.preparation)
    }

    /// Consumes the exact retained inspection and admission into prepared
    /// sources without re-running request normalization or cold selection.
    pub fn prepare_sources(self) -> Result<PreparedModelSources, PreparedModelSourcesError> {
        let (inspection, preparation) = self.into_parts();
        let plan = eredu_core::ModelPreparationPlan::from_retained_admission(
            inspection,
            preparation.admission(),
        )?;
        prepare_model_sources(plan, preparation)
    }
}

/// Consumes the ready selection retained by a total inspection outcome.
///
/// Non-ready reports return `Ok(None)` and perform no source construction.
pub fn prepare_inspected_model_sources(
    outcome: ModelInspectionOutcome,
) -> Result<Option<PreparedModelSources>, PreparedModelSourcesError> {
    outcome
        .into_selected()
        .map(SelectedModelInspection::prepare_sources)
        .transpose()
}

/// Total inspection result for valid, unsupported, missing, and invalid artifacts.
#[derive(Debug, Clone)]
pub struct ModelInspectionOutcome {
    report: ModelInspectionReport,
    selected: Option<SelectedModelInspection>,
}

impl ModelInspectionOutcome {
    /// Returns the stable compatibility report for every inspection outcome.
    pub const fn report(&self) -> &ModelInspectionReport {
        &self.report
    }

    /// Returns the retained exact selection only when the requested load is ready.
    pub const fn selected(&self) -> Option<&SelectedModelInspection> {
        self.selected.as_ref()
    }

    /// Consumes the outcome into its stable report.
    pub fn into_report(self) -> ModelInspectionReport {
        self.report
    }

    /// Consumes the retained selection when inspection and cold selection succeeded.
    pub fn into_selected(self) -> Option<SelectedModelInspection> {
        self.selected
    }

    /// Consumes the report and optional retained selection together.
    pub fn into_parts(self) -> (ModelInspectionReport, Option<SelectedModelInspection>) {
        (self.report, self.selected)
    }
}

/// Inspects a local SafeTensors directory or GGUF checkpoint through one neutral driver.
///
/// Missing, unsupported, and invalid artifacts are represented in the returned
/// report instead of escaping through a backend-specific error path.
pub fn inspect_model<P>(
    path: impl AsRef<Path>,
    request: &NormalizedLoadRequest,
    mechanisms: &P,
    media: MediaFeatureAvailability,
) -> ModelInspectionOutcome
where
    P: PreparationMechanismProvider,
{
    let path = path.as_ref();
    if !path.exists() {
        let format = submitted_format(path);
        let mut report = ModelInspectionReport::unverified(path, format);
        reject_artifact(
            &mut report,
            path,
            format,
            &eredu_core::artifact::ArtifactError::MissingArtifact(path.to_owned()),
        );
        return ModelInspectionOutcome {
            report,
            selected: None,
        };
    }
    match crate::configuration::inspect_artifact(path) {
        Ok(inspection) => inspect_selected_model(inspection, request, mechanisms, media),
        Err(error) => {
            let format = submitted_format(path);
            let mut report = ModelInspectionReport::unverified(path, format);
            reject_artifact(&mut report, path, format, &error);
            ModelInspectionOutcome {
                report,
                selected: None,
            }
        }
    }
}

/// Inspects supplied immutable headers through the shared cold selection driver.
/// Primary artifact headers require no filesystem or payload access. Explicitly
/// requested companion sources receive separate header admission. No device or
/// stream is realized. Source preparation for a metadata-only primary returns
/// `ArtifactError::MetadataOnly`.
pub fn inspect_model_metadata<P: PreparationMechanismProvider>(
    metadata: &eredu_core::artifact::ArtifactMetadata,
    request: &NormalizedLoadRequest,
    mechanisms: &P,
    media: MediaFeatureAvailability,
) -> ModelInspectionOutcome {
    match eredu_core::artifact::inspect_artifact_metadata(
        metadata,
        &crate::configuration::MODEL_CONFIGURATIONS,
    ) {
        Ok(inspection) => inspect_selected_model(inspection, request, mechanisms, media),
        Err(error) => {
            let path = Path::new(&metadata.provenance.source);
            let format = metadata.format();
            let mut report = ModelInspectionReport::unverified(path, format);
            report.metadata_provenance = Some(metadata.provenance.clone());
            reject_artifact(&mut report, path, format, &error);
            ModelInspectionOutcome {
                report,
                selected: None,
            }
        }
    }
}

/// Applies one normalized request and mechanism report to an admitted artifact.
///
/// `select_preparation` is invoked exactly once. Its returned admission and
/// execution selection are used both for readiness reporting and later loading.
pub fn inspect_selected_model<P>(
    inspection: ArtifactInspection<ArtifactArchitecturePlan>,
    request: &NormalizedLoadRequest,
    mechanisms: &P,
    media: MediaFeatureAvailability,
) -> ModelInspectionOutcome
where
    P: PreparationMechanismProvider,
{
    let descriptor = inspection.architecture_plan().architecture_descriptor();
    let capabilities = architecture_capabilities(&inspection).and_then(|capabilities| {
        let capacity = inspection
            .architecture_plan()
            .prediction_target_projection()
            .map_err(|error| PreparationSelectionError::PredictionProjection(error.into()))?
            .map(|(_, extension)| {
                crate::prediction_extension::embedded_prediction_capacity(&extension)
            })
            .transpose()
            .map_err(|error| PreparationSelectionError::PredictionProjection(error.into()))?
            .map_or(
                capabilities.embedded_draft_layers().unwrap_or(0),
                |capacity| capacity.get(),
            );
        Ok((capabilities, capacity))
    });
    let (capabilities, embedded_capacity) = match capabilities {
        Ok(capabilities) => capabilities,
        Err(error) => {
            let mut report =
                ModelInspectionReport::unverified(inspection.path(), inspection.format());
            report.record_artifact_inspection(&inspection);
            invalidate_architecture(&mut report, error.to_string());
            record_media(&mut report, &inspection, media);
            record_discovery(&mut report, descriptor, None, mechanisms);
            return ModelInspectionOutcome {
                report,
                selected: None,
            };
        }
    };

    match select_preparation(&inspection, request, mechanisms) {
        Ok(preparation) => {
            let inspection = preparation.retained_architecture().map_or_else(
                || inspection.clone(),
                |architecture| {
                    inspection
                        .clone()
                        .map_architecture_plan(|_| architecture.clone())
                },
            );
            let descriptor = inspection.architecture_plan().architecture_descriptor();
            let mut report = eredu_core::assemble_portable_model_inspection(
                &inspection,
                preparation.admission(),
                if preparation.qwen4_construction().is_some() {
                    match preparation.execution().processor() {
                        Some(processor) => {
                            let modalities = processor.modalities();
                            eredu_core::InputModalities {
                                text: modalities.contains(&eredu_core::InputModality::Text),
                                image: modalities.contains(&eredu_core::InputModality::Image),
                                audio: modalities.contains(&eredu_core::InputModality::Audio),
                                video: modalities.contains(&eredu_core::InputModality::Video),
                            }
                        }
                        None => eredu_core::InputModalities::TEXT,
                    }
                } else {
                    inspection
                        .architecture_plan()
                        .muse_input_modalities()
                        .unwrap_or_else(|| capabilities.input_modalities())
                },
                capabilities.embedded_draft_layers(),
                safetensors_processor(&inspection, media),
            );
            report.resources.embedded_draft_capacity = eredu_core::Observed::exact(
                embedded_capacity,
                "normalized architecture prediction contract",
            );
            record_parameter_resources(&mut report.resources, preparation.execution());
            record_rank_resources(
                &mut report.resources,
                &inspection,
                preparation.execution(),
                mechanisms,
            );
            record_gguf_media(&mut report, &inspection, media);
            record_media_companions(&mut report, &inspection);
            record_discovery(&mut report, descriptor, Some(&preparation), mechanisms);
            let memory_materialization_workspace =
                if let Some(rank) = report.resources.selected_rank.value() {
                    rank.materialization_workspace.clone()
                } else if preparation.execution().parallel_topology().is_some() {
                    eredu_core::Observed::unavailable(
                        "rank-local conversion workspace is not projected for this architecture",
                    )
                } else {
                    let workspace = if let Some(target) = preparation.qwen4_construction() {
                        target.parameter_materialization_workspace(mechanisms)
                    } else {
                        crate::replicated_text::artifact::catalog(&inspection)
                            .map_err(|error| error.to_string())
                            .and_then(|source| {
                                preparation.execution().parameter_materialization_workspace(
                                    source.as_ref(),
                                    None,
                                    mechanisms,
                                )
                            })
                    };
                    match workspace {
                        Ok(value) => eredu_core::Observed::Available {
                            value,
                            kind: eredu_core::ObservationKind::Conservative,
                            source: "selected checkpoint recipes and backend materialization facts"
                                .into(),
                        },
                        Err(reason) => eredu_core::Observed::unavailable(reason),
                    }
                };
            ModelInspectionOutcome {
                report,
                selected: Some(SelectedModelInspection {
                    inspection,
                    preparation,
                    memory_materialization_workspace,
                }),
            }
        }
        Err(error) => {
            let mut report =
                ModelInspectionReport::unverified(inspection.path(), inspection.format());
            report.record_artifact_inspection(&inspection);
            report.record_architecture_capabilities(
                capabilities.input_modalities(),
                capabilities.embedded_draft_layers(),
            );
            report.resources.embedded_draft_capacity = eredu_core::Observed::exact(
                embedded_capacity,
                "normalized architecture prediction contract",
            );
            reject_selection(&mut report, error);
            record_media(&mut report, &inspection, media);
            record_discovery(&mut report, descriptor, None, mechanisms);
            ModelInspectionOutcome {
                report,
                selected: None,
            }
        }
    }
}

fn record_parameter_resources(
    report: &mut eredu_core::ModelResourceProfile,
    selected: &crate::SelectedExecution,
) {
    match selected.parameter_resources() {
        Ok(resources) => {
            let source = "unsharded selected executable tasks, including both ordinary and independently acquired parameters";
            report.materialized_parameter_bytes =
                eredu_core::Observed::exact(resources.parameter_bytes, source);
            report.pinned_parameter_bytes =
                eredu_core::Observed::exact(resources.pinned_bytes, source);
            report.largest_execution_group_bytes =
                eredu_core::Observed::exact(resources.largest_unit_bytes, source);
            report.largest_adjacent_execution_groups_bytes =
                eredu_core::Observed::exact(resources.largest_adjacent_units_bytes, source);
            report.expert_parameter_bytes =
                eredu_core::Observed::exact(resources.expert_bytes, source);
        }
        Err(error) => {
            let unavailable = eredu_core::Observed::unavailable(error.to_string());
            report.materialized_parameter_bytes = unavailable.clone();
            report.pinned_parameter_bytes = unavailable.clone();
            report.largest_execution_group_bytes = unavailable.clone();
            report.largest_adjacent_execution_groups_bytes = unavailable.clone();
            report.expert_parameter_bytes = unavailable;
        }
    }
}

fn record_rank_resources(
    report: &mut eredu_core::ModelResourceProfile,
    inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
    selected: &crate::SelectedExecution,
    mechanisms: &impl PreparationMechanismProvider,
) {
    use crate::configuration::{GgufModelConfig, SafetensorsModelConfig};
    let plan = inspection.architecture_plan();
    if !matches!(
        plan.model_kind(),
        crate::ModelKind::K2Horizon
            | crate::ModelKind::NemotronLabsDiffusion
            | crate::ModelKind::MuseGlimmer
            | crate::ModelKind::MuseSpark
    ) {
        return;
    }
    let project = || -> Result<eredu_core::SelectedRankResourceProfile, String> {
        let nemotron = plan
            .gguf_media_projector()
            .and_then(|projector| match projector.model() {
                crate::gguf_companion::GgufMediaProjectorConfig::NemotronLabsDiffusion {
                    config,
                    ..
                } => Some(config),
                _ => None,
            })
            .or_else(|| {
                match (
                    plan.safetensors_architecture().map(|p| p.model()),
                    plan.gguf_plan().map(|p| p.model()),
                ) {
                    (Some(SafetensorsModelConfig::NemotronLabsDiffusion(config)), None) => {
                        Some(config)
                    }
                    (None, Some(GgufModelConfig::NemotronLabsDiffusion(plan))) => {
                        Some(plan.text_config())
                    }
                    _ => None,
                }
            });
        let muse = match crate::replicated_text::composite_config(plan)
            .map_err(|error| error.to_string())?
        {
            Some(crate::replicated_text::CompositeConfig::Muse(config)) => Some(config),
            _ => None,
        };
        let (parameters, layers, kv_width) = if let Some(config) = muse {
            let config = crate::muse::with_checkpoint_formats(
                config,
                crate::replicated_text::selected_formats(selected.text_realization()),
            )?;
            let parameters = crate::muse::component_definition(&config, None)
                .map_err(|error| error.to_string())?
                .parameters()
                .clone();
            (
                parameters,
                config.num_hidden_layers as usize,
                (config.num_key_value_heads * config.head_dim) as usize,
            )
        } else if let Some(config) = nemotron {
            use crate::decoder::Config;
            let config = config
                .with_checkpoint_formats(crate::replicated_text::selected_formats(
                    selected.text_realization(),
                ))
                .map_err(|e| e.to_string())?;
            let parameters = if config.vision.is_some() {
                crate::nemotron_labs_diffusion::parallel::composite_parameter_description(&config)
            } else {
                crate::nemotron_labs_diffusion::parallel::parameter_description(&config)
            }
            .map_err(|e| e.to_string())?;
            (
                parameters,
                config.num_hidden_layers() as usize,
                (config.num_key_value_heads() * config.head_dim()) as usize,
            )
        } else {
            let args = match (
                plan.safetensors_architecture().map(|p| p.model()),
                plan.gguf_plan().map(|p| p.model()),
            ) {
                (Some(SafetensorsModelConfig::K2Horizon(args)), None)
                | (None, Some(GgufModelConfig::K2Horizon(args))) => args,
                _ => return Err("rank resource projection is unavailable for this family".into()),
            };
            let args = crate::replicated_text::selected_k2_horizon_args(
                args,
                selected.text_realization(),
            )?;
            (
                crate::k2_horizon::parameter_description(&args).map_err(|e| e.to_string())?,
                args.num_hidden_layers as usize,
                (args.num_key_value_heads * args.head_dim) as usize,
            )
        };
        let (resources, layout) = match selected.partition_parameter_resources(&parameters)? {
            Some((resources, layout)) => (resources, Some(layout)),
            None => (
                selected.parameter_resources().map_err(|e| e.to_string())?,
                None,
            ),
        };
        let scalar_bytes = selected
            .text_realization()
            .state()
            .floating_dtype()
            .ok_or_else(|| "selected KV dtype is missing".to_owned())?
            .bytes();
        let mut cache_bytes_per_position = 0u64;
        for layer in 0..layers {
            if selected.partition_groups().is_some_and(|groups| {
                !groups.iter().any(|group| {
                    group.group().as_str() == crate::decoder::TEXT_DECODER_EXECUTION_GROUP
                        && group.units().contains(&layer)
                })
            }) {
                continue;
            }
            let width = if let Some(layout) = &layout {
                layout
                    .tensor(&format!("model.layers.{layer}.self_attn.k_proj.weight"))
                    .ok_or_else(|| "local key projection is missing".to_owned())?
                    .local_shape()[0]
            } else {
                kv_width
            };
            cache_bytes_per_position += 2 * width as u64 * u64::from(scalar_bytes.get());
        }
        let source =
            crate::replicated_text::artifact::catalog(inspection).map_err(|e| e.to_string())?;
        let workspace = selected.parameter_materialization_workspace(
            source.as_ref(),
            layout.as_ref(),
            mechanisms,
        );
        Ok(eredu_core::SelectedRankResourceProfile {
            topology: selected.parallel_topology(),
            materialized_parameter_bytes: resources.parameter_bytes,
            pinned_parameter_bytes: resources.pinned_bytes,
            largest_execution_group_bytes: resources.largest_unit_bytes,
            largest_adjacent_execution_groups_bytes: resources.largest_adjacent_units_bytes,
            expert_parameter_bytes: resources.expert_bytes,
            cache_bytes_per_position,
            materialization_workspace: match workspace {
                Ok(value) => eredu_core::Observed::Available {
                    value,
                    kind: eredu_core::ObservationKind::Conservative,
                    source: "live source-recipe values and bounded compact-bank assembly".into(),
                },
                Err(error) => eredu_core::Observed::unavailable(error),
            },
        })
    };
    report.selected_rank = match project() {
        Ok(resources) => eredu_core::Observed::exact(
            resources,
            "selected physical parameter topology and rank-local head ownership",
        ),
        Err(error) => eredu_core::Observed::unavailable(error),
    };
}

fn record_discovery(
    report: &mut ModelInspectionReport,
    descriptor: eredu_core::ArchitectureDescriptor,
    selected: Option<&SelectedPreparation>,
    mechanisms: &impl PreparationMechanismProvider,
) {
    report.observation_support = Some(eredu_runtime::inspection::observation_support(
        &descriptor.observations,
        eredu_runtime::inspection::ObservationExecutionContext {
            selected: selected.is_some(),
            activation_inspection: selected
                .is_some_and(|s| s.session_capabilities().activation_inspection()),
            prediction_inspection: false,
            partitioned: selected.is_some_and(|s| s.execution().parallel_topology().is_some()),
            mechanisms: mechanisms.observation_mechanisms(),
        },
    ));
    report.architecture_descriptor = Some(descriptor);
    if let Some(support) = &mut report.observation_support {
        support.capture = mechanisms.capture_capabilities();
    }
}

fn submitted_format(path: &Path) -> ArtifactFormat {
    if path
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
    {
        ArtifactFormat::Gguf
    } else {
        ArtifactFormat::SafeTensors
    }
}

fn architecture_capabilities(
    inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
) -> Result<ArchitectureCapabilities, PreparationSelectionError> {
    match inspection.format() {
        ArtifactFormat::SafeTensors => prepared_safetensors_capabilities(
            inspection
                .architecture_plan()
                .safetensors_architecture()
                .ok_or(PreparationSelectionError::MissingArchitecturePlan {
                    format: ArtifactFormat::SafeTensors,
                })?,
        )
        .map_err(Into::into),
        ArtifactFormat::Gguf => inspection
            .architecture_plan()
            .gguf_plan()
            .map(prepared_gguf_capabilities)
            .ok_or(PreparationSelectionError::MissingArchitecturePlan {
                format: ArtifactFormat::Gguf,
            }),
        format => Err(PreparationSelectionError::MissingArchitecturePlan { format }),
    }
}

fn safetensors_processor(
    inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
    media: MediaFeatureAvailability,
) -> Option<(bool, MediaFeatureAvailability)> {
    (inspection.format() == ArtifactFormat::SafeTensors)
        .then(|| (inspection.architecture_plan().has_processor(), media))
}

fn record_media(
    report: &mut ModelInspectionReport,
    inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
    media: MediaFeatureAvailability,
) {
    record_media_companions(report, inspection);
    match inspection.format() {
        ArtifactFormat::SafeTensors => eredu_core::record_processor_inspection(
            report,
            inspection.architecture_plan().has_processor(),
            media,
        ),
        ArtifactFormat::Gguf => record_gguf_media(report, inspection, media),
        _ => {}
    }
}

fn record_media_companions(
    report: &mut ModelInspectionReport,
    inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
) {
    for missing in inspection.architecture_plan().missing_media_companions() {
        report.issue(
            InspectionIssueCode::MissingMediaCompanion,
            InspectionSeverity::Info,
            missing.to_string(),
            None,
        );
    }
}

fn record_gguf_media(
    report: &mut ModelInspectionReport,
    inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
    media: MediaFeatureAvailability,
) {
    if inspection.format() != ArtifactFormat::Gguf {
        return;
    }
    let Some(architecture) = inspection.architecture_plan().gguf_architecture() else {
        return;
    };
    let plan = gguf_composite_artifact_plan(architecture);
    let requirement = match plan.media_projector_requirement() {
        GgufMediaProjectorRequirement::NotApplicable => MediaProjectorRequirement::NotApplicable,
        GgufMediaProjectorRequirement::Optional => MediaProjectorRequirement::Optional,
        GgufMediaProjectorRequirement::Required => MediaProjectorRequirement::Required,
    };
    // Explicit component sources can satisfy media without a sibling GGUF.
    // Their retained configuration/processor is the authority for coverage.
    if !inspection.architecture_plan().media_companions().is_empty() {
        if let Some(modalities) = inspection.architecture_plan().muse_input_modalities() {
            report.expected_modalities = eredu_core::inspection::artifact_modalities(modalities);
            eredu_core::record_processor_inspection(
                report,
                inspection.architecture_plan().has_processor(),
                media,
            );
            return;
        }
    }
    let projector = inspection
        .validated_gguf()
        .and_then(|validated| {
            validated.companion(&eredu_core::ArtifactCompanionRole::MediaProjector)
        })
        .map(|companion| companion.path().to_owned());
    let composition = if eredu_core::record_media_projector_inspection(
        report,
        inspection.path(),
        requirement,
        projector,
    ) {
        GgufArtifactComposition::ValidatedMediaProjector
    } else {
        GgufArtifactComposition::ModelOnly
    };
    report.expected_modalities = eredu_core::inspection::artifact_modalities(
        inspection
            .architecture_plan()
            .muse_input_modalities()
            .unwrap_or_else(|| plan.input_modalities(composition)),
    );
    if composition == GgufArtifactComposition::ValidatedMediaProjector {
        report.multimodal = eredu_core::media_feature_readiness(&report.expected_modalities, media);
    }
}

fn reject_artifact(
    report: &mut ModelInspectionReport,
    path: &Path,
    format: ArtifactFormat,
    error: &eredu_core::artifact::ArtifactError,
) {
    use eredu_core::artifact::ArtifactError;

    if matches!(error, ArtifactError::MissingArtifact(_))
        || matches!(error, ArtifactError::Io(io) if io.kind() == std::io::ErrorKind::NotFound)
    {
        report.container = InspectionReadiness::Missing;
        report.architecture_support = InspectionReadiness::Unverified;
        report.structural_binding = InspectionReadiness::Missing;
        report.model_loadability = InspectionReadiness::Missing;
        report.requested_load = InspectionReadiness::Missing;
        report.issue(
            InspectionIssueCode::MissingCheckpointShard,
            InspectionSeverity::Error,
            error.to_string(),
            Some(path.to_owned()),
        );
        return;
    }
    if let ArtifactError::UnsupportedContainer(_) = error {
        report.container = InspectionReadiness::Unsupported;
        report.architecture_support = InspectionReadiness::Unverified;
        report.structural_binding = InspectionReadiness::Unsupported;
        report.model_loadability = InspectionReadiness::Unsupported;
        report.requested_load = InspectionReadiness::Unsupported;
        report.issue(
            InspectionIssueCode::InvalidContainer,
            InspectionSeverity::Error,
            error.to_string(),
            Some(path.to_owned()),
        );
        return;
    }
    if format == ArtifactFormat::SafeTensors {
        reject_safetensors_artifact(report, path, error);
    } else {
        eredu_core::reject_portable_artifact_inspection(report, path, error);
    }
}

fn reject_safetensors_artifact(
    report: &mut ModelInspectionReport,
    path: &Path,
    error: &eredu_core::artifact::ArtifactError,
) {
    use eredu_core::artifact::ArtifactError;

    let invalid_plan = matches!(error, ArtifactError::InvalidArchitecturePlan(_));
    let (code, container, architecture) = match error {
        ArtifactError::UnsupportedModelType(_) => (
            InspectionIssueCode::UnsupportedArchitecture,
            InspectionReadiness::Ready,
            InspectionReadiness::Unsupported,
        ),
        ArtifactError::SafetensorsShards(
            eredu_checkpoint::safetensors::SafetensorsShardError::MissingShard { .. },
        ) => (
            InspectionIssueCode::MissingCheckpointShard,
            InspectionReadiness::Missing,
            InspectionReadiness::Unverified,
        ),
        ArtifactError::Json(_) => (
            InspectionIssueCode::InvalidConfiguration,
            InspectionReadiness::Invalid,
            InspectionReadiness::Invalid,
        ),
        ArtifactError::InvalidArchitecturePlan(_) => (
            InspectionIssueCode::InvalidConfiguration,
            InspectionReadiness::Ready,
            InspectionReadiness::Invalid,
        ),
        ArtifactError::DuplicateTensor(_)
        | ArtifactError::SafetensorsShards(_)
        | ArtifactError::Catalog(_) => (
            InspectionIssueCode::InvalidContainer,
            InspectionReadiness::Invalid,
            InspectionReadiness::Unverified,
        ),
        _ => (
            InspectionIssueCode::InvalidContainer,
            InspectionReadiness::Invalid,
            InspectionReadiness::Invalid,
        ),
    };
    report.container = container;
    report.architecture_support = architecture;
    report.structural_binding = if invalid_plan {
        InspectionReadiness::Invalid
    } else {
        container
    };
    report.multimodal = if invalid_plan {
        InspectionReadiness::Invalid
    } else {
        InspectionReadiness::Unverified
    };
    report.model_loadability = if architecture == InspectionReadiness::Unsupported {
        InspectionReadiness::Unsupported
    } else if container == InspectionReadiness::Missing {
        InspectionReadiness::Missing
    } else {
        InspectionReadiness::Invalid
    };
    report.requested_load = report.model_loadability;
    report.issue(
        code,
        InspectionSeverity::Error,
        error.to_string(),
        Some(path.to_owned()),
    );
}

fn reject_selection(report: &mut ModelInspectionReport, error: PreparationSelectionError) {
    if let PreparationSelectionError::Admission(admission) = error {
        report.reject_preparation_admission(admission);
        return;
    }
    let invalid = matches!(
        error,
        PreparationSelectionError::ArchitectureCapabilities(_)
            | PreparationSelectionError::MissingArchitecturePlan { .. }
            | PreparationSelectionError::PredictionProjection(_)
    );
    if invalid {
        invalidate_architecture(report, error.to_string());
        return;
    }
    report.preparation_admission = None;
    report.requested_load = InspectionReadiness::Unsupported;
    let code = match &error {
        PreparationSelectionError::Request(NormalizedLoadRequestError::Quantization(_)) => {
            InspectionIssueCode::UnsupportedQuantizationRequest
        }
        PreparationSelectionError::Request(_)
        | PreparationSelectionError::PartitionedAdmission(_)
        | PreparationSelectionError::PartitionedMechanisms(_)
        | PreparationSelectionError::UnsupportedProductionRoute { .. } => {
            InspectionIssueCode::UnsupportedParallelTopology
        }
        PreparationSelectionError::Processor(_) | PreparationSelectionError::MissingProcessor => {
            InspectionIssueCode::MissingProcessor
        }
        _ => InspectionIssueCode::UnsupportedArchitecture,
    };
    report.issue(
        code,
        InspectionSeverity::Error,
        error.to_string(),
        Some(report.path.clone()),
    );
}

fn invalidate_architecture(report: &mut ModelInspectionReport, detail: String) {
    report.architecture_support = InspectionReadiness::Invalid;
    report.structural_binding = InspectionReadiness::Invalid;
    report.model_loadability = InspectionReadiness::Invalid;
    report.requested_load = InspectionReadiness::Invalid;
    report.preparation_admission = None;
    report.issue(
        InspectionIssueCode::InvalidConfiguration,
        InspectionSeverity::Error,
        detail,
        Some(report.path.clone()),
    );
}
