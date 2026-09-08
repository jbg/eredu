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
            .map_or(0, |capacity| capacity.get());
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
            let mut report = eredu_core::assemble_portable_model_inspection(
                &inspection,
                preparation.admission(),
                capabilities.input_modalities(),
                capabilities.embedded_draft_layers(),
                safetensors_processor(&inspection, media),
            );
            report.resources.embedded_draft_capacity = eredu_core::Observed::exact(
                embedded_capacity,
                "normalized architecture prediction contract",
            );
            record_gguf_media(&mut report, &inspection, media);
            record_discovery(&mut report, descriptor, Some(&preparation), mechanisms);
            ModelInspectionOutcome {
                report,
                selected: Some(SelectedModelInspection {
                    inspection,
                    preparation,
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
    let projector = inspection
        .validated_gguf()
        .and_then(|validated| validated.companion(&eredu_core::GgufCompanionRole::MediaProjector))
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
    report.expected_modalities =
        eredu_core::inspection::artifact_modalities(plan.input_modalities(composition));
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preparation_selection::tests::{
        inspected_config, prediction_config, BoundedIndependentAdapter,
    };

    #[test]
    fn missing_artifact_is_a_total_report_without_a_selection() {
        struct Unreachable;
        impl PreparationMechanismProvider for Unreachable {
            fn preparation_capabilities(&self) -> eredu_core::PreparationMechanismCapabilities {
                panic!("missing artifacts must not query backend facts")
            }
            fn supports_grouped_operation(
                &self,
                _: eredu_runtime::GroupedOperationRequirement,
            ) -> bool {
                panic!("missing artifacts must not query backend facts")
            }
            fn replicated_text_capabilities(
                &self,
                _: &eredu_runtime::ReplicatedTextRequirements,
                _: &eredu_runtime::ReplicatedTextSelectionRequest,
            ) -> eredu_runtime::BackendMechanismCapabilities {
                panic!("missing artifacts must not query backend facts")
            }
            fn processor_capabilities(&self) -> eredu_runtime::MediaPrimitiveCapabilities {
                panic!("missing artifacts must not query backend facts")
            }
            fn speculative_capabilities(&self) -> eredu_runtime::SpeculativeMechanismCapabilities {
                panic!("missing artifacts must not query backend facts")
            }
            fn communication_capabilities(&self) -> eredu_runtime::CommunicationCapabilities {
                panic!("missing artifacts must not query backend facts")
            }
        }

        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("absent.gguf");
        let outcome = inspect_model(
            &path,
            &NormalizedLoadRequest::default(),
            &Unreachable,
            MediaFeatureAvailability {
                image: false,
                audio: false,
            },
        );

        assert!(outcome.selected().is_none());
        assert_eq!(outcome.report().container, InspectionReadiness::Missing);
        assert_eq!(
            outcome.report().model_loadability,
            InspectionReadiness::Missing
        );
        assert_eq!(
            outcome.report().requested_load,
            InspectionReadiness::Missing
        );
    }

    #[test]
    fn inspection_reports_capacity_even_when_embedded_drafting_is_disabled_or_rejected() {
        let qwen = serde_json::json!({
            "model_type":"qwen3_next", "vocab_size":64, "hidden_size":32,
            "num_hidden_layers":4, "mtp_num_hidden_layers":1, "num_attention_heads":4,
            "num_key_value_heads":2, "head_dim":8, "max_position_embeddings":128,
            "linear_conv_kernel_dim":4, "linear_key_head_dim":8, "linear_value_head_dim":8,
            "linear_num_key_heads":2, "linear_num_value_heads":4, "intermediate_size":48,
            "moe_intermediate_size":16, "shared_expert_intermediate_size":24,
            "num_experts_per_tok":2, "num_experts":8,
            "layer_types":["linear_attention","linear_attention","linear_attention","full_attention"]
        });
        let dspark = serde_json::json!({
            "architectures":["DeepseekV4ForCausalLM"],"model_type":"deepseek_v4",
            "hidden_size":16,"moe_intermediate_size":8,"num_hidden_layers":2,
            "num_attention_heads":2,"num_key_value_heads":1,"head_dim":4,
            "qk_rope_head_dim":2,"q_lora_rank":2,"o_lora_rank":2,"o_groups":2,
            "vocab_size":32,"max_position_embeddings":128,"sliding_window":8,
            "compress_ratios":[0,0,0],"index_n_heads":2,"index_head_dim":4,"index_topk":1,
            "hc_mult":2,"hc_sinkhorn_iters":2,"n_routed_experts":2,"n_shared_experts":1,
            "num_experts_per_tok":1,"num_hash_layers":0,"scoring_func":"sqrtsoftplus",
            "topk_method":"noaux_tc","norm_topk_prob":true,"routed_scaling_factor":1.0,
            "swiglu_limit":4.0,"num_nextn_predict_layers":1,"dspark_block_size":4,
            "dspark_noise_token_id":0,"dspark_target_layer_ids":[0,1],"dspark_markov_rank":2
        });
        for (config, capacity) in [(prediction_config(), 1), (qwen, 1), (dspark, 4)] {
            let (_root, inspection) = inspected_config(config);
            for drafting in [
                eredu_runtime::DraftingLoadRequest::Disabled,
                eredu_runtime::DraftingLoadRequest::embedded(capacity + 1).unwrap(),
            ] {
                let request = NormalizedLoadRequest::default().with_drafting(drafting);
                let outcome = inspect_selected_model(
                    inspection.clone(),
                    &request,
                    &BoundedIndependentAdapter::default(),
                    MediaFeatureAvailability {
                        image: false,
                        audio: false,
                    },
                );
                assert_eq!(
                    outcome.report().resources.embedded_draft_layers.value(),
                    Some(&1)
                );
                assert_eq!(
                    outcome.report().resources.embedded_draft_capacity.value(),
                    Some(&capacity)
                );
                if drafting == eredu_runtime::DraftingLoadRequest::Disabled {
                    assert!(outcome.report().is_loadable(), "{:?}", outcome.report());
                } else {
                    assert!(!outcome.report().is_loadable());
                }
            }
        }
    }
}
