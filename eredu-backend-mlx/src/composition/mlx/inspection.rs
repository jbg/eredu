//! Side-effect-free MLX model artifact compatibility inspection.

use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    path::{Path, PathBuf},
};

use crate::composition::mlx::structural;
use eredu_architectures::GgufArchitecture;
use eredu_checkpoint::store::WeightStore;
use eredu_core::{
    ArtifactFormat, ArtifactModality, ArtifactTensorEncoding, InputModalities, InspectionIssue,
    InspectionIssueCode, InspectionReadiness, InspectionRequirement, InspectionSeverity,
    ModelInspectionReport, Observed,
};
use eredu_gguf::MetadataValue as GgufMetadataValue;
use safemlx::ops::GgufCheckpoint;
use serde_json::Value;

use super::*;
use eredu_checkpoint::store::SafetensorsWeightStore;

/// Options applied while inspecting a model artifact.
#[derive(Debug, Clone, Default)]
pub struct MlxInspectionOptions {
    /// The exact loading policy that admission should validate.
    pub load: ModelLoadOptions,
}

/// Inspects a local SafeTensors model directory or GGUF checkpoint without
/// instantiating a model, materializing tensor payloads, or creating an MLX
/// execution stream.
pub fn inspect_model(
    path: impl AsRef<Path>,
    options: MlxInspectionOptions,
) -> Result<ModelInspectionReport, Error> {
    let path = path.as_ref();
    let mut report = if is_gguf_file(path) {
        inspect_gguf(path, options)
    } else if path.is_dir() {
        inspect_safetensors(path, options)
    } else if !path.exists() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("model artifact does not exist: {}", path.display()),
        )
        .into());
    } else {
        return Err(Error::UnsupportedArchitecture(format!(
            "model artifact must be a SafeTensors directory or .gguf file: {}",
            path.display()
        )));
    };
    report.resources.model_family = report.model_family.clone();
    report.resources.architecture = report.architecture.clone();
    report.resources.tensor_count = report.tensor_count;
    report.resources.checkpoint_shards = report.checkpoint_shards;
    Ok(report)
}

fn is_gguf_file(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
}

fn record_embedded_drafting(
    report: &mut ModelInspectionReport,
    capabilities: eredu_architectures::preparation::ArchitectureCapabilities,
) {
    report.resources.embedded_draft_layers = capabilities.embedded_draft_layers().map_or_else(
        || Observed::unsupported("artifact convention does not expose embedded drafting"),
        |layers| Observed::exact(layers, "normalized architecture configuration"),
    );
}

fn resolve_projector(
    model: &Path,
    architecture: GgufArchitecture,
) -> Result<Option<(PathBuf, GgufCheckpoint, HashMap<String, GgufMetadataValue>)>, Error> {
    let requirements =
        eredu_architectures::configuration::gguf_companion_requirements(architecture)?;
    let mut companions = eredu_core::resolve_gguf_companions(model, &requirements)?;
    let Some(companion) = companions.remove(&eredu_core::GgufCompanionRole::MediaProjector) else {
        return Ok(None);
    };
    let path = companion.path().to_owned();
    let checkpoint = GgufCheckpoint::from_portable(companion.checkpoint().clone());
    let metadata = crate::backend::runtime::checkpoint::load::gguf_metadata(&checkpoint);
    Ok(Some((path, checkpoint, metadata)))
}

fn inspect_safetensors(path: &Path, options: MlxInspectionOptions) -> ModelInspectionReport {
    let mut report = ModelInspectionReport::unverified(path, ArtifactFormat::SafeTensors);
    let mut resolved_kind = None;
    let config_path = path.join("config.json");
    let config: Option<Value> = match std::fs::read_to_string(&config_path) {
        Ok(raw) => match serde_json::from_str(&raw) {
            Ok(value) => Some(value),
            Err(error) => {
                report.container = InspectionReadiness::Invalid;
                report.model_loadability = InspectionReadiness::Invalid;
                report.requested_load = InspectionReadiness::Invalid;
                report.issue(
                    InspectionIssueCode::InvalidConfiguration,
                    InspectionSeverity::Error,
                    format!("invalid config.json: {error}"),
                    Some(config_path.clone()),
                );
                None
            }
        },
        Err(error) => {
            report.container = if error.kind() == std::io::ErrorKind::NotFound {
                InspectionReadiness::Missing
            } else {
                InspectionReadiness::Invalid
            };
            report.model_loadability = report.container;
            report.requested_load = report.container;
            report.issue(
                InspectionIssueCode::InvalidConfiguration,
                InspectionSeverity::Error,
                format!("could not read config.json: {error}"),
                Some(config_path.clone()),
            );
            None
        }
    };

    if let Some(config) = &config {
        match eredu_architectures::configuration::resolve_model_config(config) {
            Ok(supported) => {
                report.model_family = Some(supported.kind.canonical_name().into());
                report.architecture = Some(supported.effective_model_type);
                match eredu_architectures::preparation::safetensors_capabilities(
                    supported.kind,
                    config,
                ) {
                    Ok(capabilities) => {
                        resolved_kind = Some(supported.kind);
                        record_embedded_drafting(&mut report, capabilities);
                        report.expected_modalities =
                            artifact_modalities(capabilities.input_modalities());
                        report.architecture_support = InspectionReadiness::Ready;
                        match structural::validate_safetensors_preparation(
                            supported.kind,
                            config,
                            options.load,
                        ) {
                            Ok(_) => report.requested_load = InspectionReadiness::Ready,
                            Err(error) => reject_load_policy(&mut report, &error),
                        }
                    }
                    Err(error) => {
                        report.architecture_support = InspectionReadiness::Invalid;
                        report.model_loadability = InspectionReadiness::Invalid;
                        report.structural_binding = InspectionReadiness::Invalid;
                        report.requested_load = InspectionReadiness::Invalid;
                        report.issue(
                            InspectionIssueCode::InvalidConfiguration,
                            InspectionSeverity::Error,
                            error.to_string(),
                            Some(config_path.clone()),
                        );
                    }
                }
            }
            Err(error) => {
                report.architecture_support = match &error {
                    eredu_core::artifact::ArtifactError::UnsupportedModelType(_) => {
                        InspectionReadiness::Unsupported
                    }
                    _ => InspectionReadiness::Invalid,
                };
                report.model_loadability = report.architecture_support;
                report.structural_binding = report.architecture_support;
                report.requested_load = report.model_loadability;
                report.issue(
                    match &error {
                        eredu_core::artifact::ArtifactError::UnsupportedModelType(_) => {
                            InspectionIssueCode::UnsupportedArchitecture
                        }
                        _ => InspectionIssueCode::InvalidConfiguration,
                    },
                    InspectionSeverity::Error,
                    error.to_string(),
                    Some(config_path),
                );
            }
        }
    }

    match SafetensorsWeightStore::open(path) {
        Ok(store) => {
            let keys = store.keys();
            let mut encodings = BTreeSet::new();
            let mut shards = BTreeSet::new();
            let mut catalog_error = None;
            let mut stored_tensor_bytes = Some(0_u64);
            let mut largest_stored_tensor_bytes = 0_u64;
            for key in &keys {
                match store.metadata(key) {
                    Ok(metadata) => {
                        encodings.insert(format!("{:?}", metadata.stored_dtype));
                        if let Some(shard) = metadata.backing_shard {
                            shards.insert(shard);
                        }
                        let bytes = metadata.encoded_byte_len;
                        stored_tensor_bytes =
                            stored_tensor_bytes.and_then(|total| total.checked_add(bytes));
                        largest_stored_tensor_bytes = largest_stored_tensor_bytes.max(bytes);
                    }
                    Err(error) => {
                        catalog_error = Some(error);
                        break;
                    }
                }
            }
            report.tensor_count = Some(keys.len());
            report.checkpoint_shards = Some(shards.len());
            report.tensor_encodings = encodings
                .into_iter()
                .map(|name| ArtifactTensorEncoding {
                    name,
                    ggml_type_code: None,
                })
                .collect();
            match stored_tensor_bytes {
                Some(total) if !keys.is_empty() => {
                    report.resources.stored_tensor_bytes =
                        Observed::exact(total, "validated SafeTensors tensor headers");
                    report.resources.largest_stored_tensor_bytes = Observed::exact(
                        largest_stored_tensor_bytes,
                        "validated SafeTensors tensor headers",
                    );
                }
                Some(_) => {}
                None => {
                    report.resources.stored_tensor_bytes =
                        Observed::unavailable("SafeTensors payload-byte total overflowed u64");
                    report.resources.largest_stored_tensor_bytes =
                        Observed::unavailable("SafeTensors payload-byte catalog was incomplete");
                }
            }
            if keys.is_empty() {
                report.container = InspectionReadiness::Invalid;
                report.model_loadability = InspectionReadiness::Invalid;
                report.requested_load = InspectionReadiness::Invalid;
                report.issue(
                    InspectionIssueCode::InvalidContainer,
                    InspectionSeverity::Error,
                    "SafeTensors checkpoint contains no tensors",
                    Some(path.to_path_buf()),
                );
            } else if let Some(error) = catalog_error {
                report.container = InspectionReadiness::Invalid;
                report.model_loadability = InspectionReadiness::Invalid;
                report.requested_load = InspectionReadiness::Invalid;
                let missing = matches!(
                    error,
                    eredu_checkpoint::store::StoreError::MissingShard { .. }
                );
                report.issue(
                    if missing {
                        InspectionIssueCode::MissingCheckpointShard
                    } else {
                        InspectionIssueCode::InvalidContainer
                    },
                    InspectionSeverity::Error,
                    error.to_string(),
                    Some(path.to_path_buf()),
                );
            } else if report.container == InspectionReadiness::Unverified {
                report.container = InspectionReadiness::Ready;
                if let (Some(kind), Some(config)) = (resolved_kind, config.as_ref()) {
                    apply_structural_validation(
                        &mut report,
                        structural::validate_safetensors(kind, config, &store, options.load),
                        path,
                    );
                }
            }
        }
        Err(error) => {
            report.container = InspectionReadiness::Invalid;
            report.model_loadability = InspectionReadiness::Invalid;
            report.requested_load = InspectionReadiness::Invalid;
            let missing = matches!(
                error,
                eredu_checkpoint::store::StoreError::MissingShard { .. }
            );
            report.issue(
                if missing {
                    InspectionIssueCode::MissingCheckpointShard
                } else {
                    InspectionIssueCode::InvalidContainer
                },
                InspectionSeverity::Error,
                error.to_string(),
                Some(path.to_path_buf()),
            );
        }
    }

    inspect_safetensors_media(&mut report, path);
    report
}

fn inspect_gguf(path: &Path, options: MlxInspectionOptions) -> ModelInspectionReport {
    let mut report = ModelInspectionReport::unverified(path, ArtifactFormat::Gguf);
    let portable = match eredu_architectures::configuration::inspect_artifact(path) {
        Ok(inspection) => inspection,
        Err(error) => {
            reject_portable_gguf(&mut report, path, &error);
            return report;
        }
    };
    let validated = portable
        .validated_gguf()
        .expect("GGUF inspection must expose its validated GGUF result");
    let gguf_architecture =
        GgufArchitecture::resolve(&portable.configuration().declared_model_type)
            .expect("GGUF inspection must resolve through the architecture registry");
    let checkpoint = GgufCheckpoint::from_portable(validated.checkpoint().clone());
    report.container = InspectionReadiness::Ready;
    report.model_family = Some(gguf_architecture.model_kind().canonical_name().into());
    report.architecture = Some(gguf_architecture.metadata_name().into());
    report.architecture_support = InspectionReadiness::Ready;
    report.checkpoint_shards = Some(checkpoint.catalog().shards().len());
    report.tensor_count = Some(checkpoint.catalog().logical_outputs().count());
    report.gguf_versions = Some(
        checkpoint
            .catalog()
            .shards()
            .iter()
            .map(|shard| shard.version())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect(),
    );
    report.tensor_encodings = checkpoint
        .catalog()
        .tensors()
        .map(|tensor| tensor.descriptor().ggml_type)
        .map(|encoding| (encoding.code(), format!("{encoding:?}")))
        .collect::<BTreeMap<_, _>>()
        .into_iter()
        .map(|(code, name)| ArtifactTensorEncoding {
            name,
            ggml_type_code: Some(code),
        })
        .collect();
    let mut stored_tensor_bytes = Some(0_u64);
    let mut largest_stored_tensor_bytes = 0_u64;
    for tensor in checkpoint.catalog().tensors() {
        let bytes = tensor.descriptor().byte_len;
        stored_tensor_bytes = stored_tensor_bytes.and_then(|total| total.checked_add(bytes));
        largest_stored_tensor_bytes = largest_stored_tensor_bytes.max(bytes);
    }
    match stored_tensor_bytes {
        Some(total) => {
            report.resources.stored_tensor_bytes =
                Observed::exact(total, "validated GGUF tensor descriptors");
            report.resources.largest_stored_tensor_bytes = Observed::exact(
                largest_stored_tensor_bytes,
                "validated GGUF tensor descriptors",
            );
        }
        None => {
            report.resources.stored_tensor_bytes =
                Observed::unavailable("GGUF payload-byte total overflowed u64");
            report.resources.largest_stored_tensor_bytes =
                Observed::unavailable("GGUF payload-byte catalog was incomplete");
        }
    }

    let metadata = crate::backend::runtime::checkpoint::load::gguf_metadata(&checkpoint);
    let composite_plan =
        eredu_architectures::preparation::gguf_composite_artifact_plan(gguf_architecture);
    let mut capabilities_valid = false;
    match eredu_architectures::preparation::gguf_capabilities(gguf_architecture, &checkpoint) {
        Ok(capabilities) => {
            capabilities_valid = true;
            record_embedded_drafting(&mut report, capabilities);
            apply_structural_validation(
                &mut report,
                structural::validate_gguf(gguf_architecture, &checkpoint, &metadata, options.load),
                path,
            );
        }
        Err(error) => {
            report.structural_binding = InspectionReadiness::Invalid;
            report.model_loadability = InspectionReadiness::Invalid;
            report.issue(
                InspectionIssueCode::InvalidConfiguration,
                InspectionSeverity::Error,
                error.to_string(),
                Some(path.to_path_buf()),
            );
        }
    }
    match structural::validate_gguf_preparation(gguf_architecture, &checkpoint, options.load) {
        Ok(()) => match validate_gguf_quantization_source(
            &checkpoint,
            &metadata,
            options.load.quantization,
        ) {
            Ok(()) => report.requested_load = InspectionReadiness::Ready,
            Err(error) => reject_load_policy(&mut report, &error),
        },
        Err(error) => reject_load_policy(&mut report, &error),
    }
    let composition =
        inspect_gguf_projector(&mut report, path, gguf_architecture, &checkpoint, &metadata);
    if capabilities_valid {
        report.expected_modalities =
            artifact_modalities(composite_plan.input_modalities(composition));
    }

    if let Err(error) = gguf_eos_token_ids(&metadata) {
        report.issue(
            InspectionIssueCode::InvalidConfiguration,
            InspectionSeverity::Error,
            error.to_string(),
            Some(path.to_path_buf()),
        );
        report.model_loadability = InspectionReadiness::Invalid;
        report.requested_load = InspectionReadiness::Invalid;
    }
    report
}

fn reject_portable_gguf(
    report: &mut ModelInspectionReport,
    path: &Path,
    error: &eredu_core::artifact::ArtifactError,
) {
    let detail = error.to_string();
    let (code, container, architecture, structural, type_code) = match error {
        eredu_core::artifact::ArtifactError::UnsupportedGgufArchitecture(name) => {
            report.architecture = Some(name.clone());
            (
                InspectionIssueCode::UnsupportedArchitecture,
                InspectionReadiness::Ready,
                InspectionReadiness::Unsupported,
                InspectionReadiness::Unsupported,
                None,
            )
        }
        eredu_core::artifact::ArtifactError::MissingGgufArchitecture => (
            InspectionIssueCode::InvalidConfiguration,
            InspectionReadiness::Ready,
            InspectionReadiness::Invalid,
            InspectionReadiness::Invalid,
            None,
        ),
        eredu_core::artifact::ArtifactError::InvalidArtifact(_)
        | eredu_core::artifact::ArtifactError::DuplicateTensor(_)
        | eredu_core::artifact::ArtifactError::Catalog(_) => (
            InspectionIssueCode::InvalidConfiguration,
            InspectionReadiness::Ready,
            InspectionReadiness::Unverified,
            InspectionReadiness::Invalid,
            None,
        ),
        _ => {
            let type_code = parse_unsupported_type_code(&detail);
            (
                if type_code.is_some() {
                    InspectionIssueCode::UnsupportedTensorEncoding
                } else {
                    InspectionIssueCode::InvalidContainer
                },
                InspectionReadiness::Invalid,
                InspectionReadiness::Unverified,
                InspectionReadiness::Unverified,
                type_code,
            )
        }
    };
    report.container = container;
    report.architecture_support = architecture;
    report.structural_binding = structural;
    report.model_loadability = if architecture == InspectionReadiness::Unsupported {
        InspectionReadiness::Unsupported
    } else {
        InspectionReadiness::Invalid
    };
    report.requested_load = report.model_loadability;
    report.issues.push(InspectionIssue {
        code,
        severity: InspectionSeverity::Error,
        detail,
        path: Some(path.to_path_buf()),
        metadata_key: None,
        tensor_name: None,
        tensor_type_code: type_code,
    });
}

fn apply_structural_validation(
    report: &mut ModelInspectionReport,
    validation: structural::StructuralValidation,
    path: &Path,
) {
    use structural::{StructuralIssueKind, StructuralValidation};

    let push = |report: &mut ModelInspectionReport,
                issue: structural::StructuralIssue,
                severity| {
        let code = match issue.kind {
            StructuralIssueKind::MissingTensor => InspectionIssueCode::MissingRequiredTensor,
            StructuralIssueKind::UnexpectedTensor => InspectionIssueCode::ConflictingTensorLayout,
            StructuralIssueKind::ConflictingLayout => InspectionIssueCode::ConflictingTensorLayout,
            StructuralIssueKind::ShapeMismatch => InspectionIssueCode::TensorShapeMismatch,
            StructuralIssueKind::UnsupportedEncoding => {
                InspectionIssueCode::UnsupportedTensorEncoding
            }
            StructuralIssueKind::CompanionMismatch => {
                InspectionIssueCode::QuantizationCompanionMismatch
            }
            StructuralIssueKind::InvalidGeometry => InspectionIssueCode::InvalidLayerOrExpertCount,
            StructuralIssueKind::ValidationUnavailable => {
                InspectionIssueCode::ValidationUnavailableUntilLoad
            }
        };
        report.issues.push(InspectionIssue {
            code,
            severity,
            detail: issue.detail,
            path: Some(path.to_path_buf()),
            metadata_key: issue.metadata_key,
            tensor_name: issue.tensor_name,
            tensor_type_code: issue.tensor_type_code,
        });
    };

    match validation {
        StructuralValidation::Exact => {
            if report.structural_binding == InspectionReadiness::Unverified {
                report.structural_binding = InspectionReadiness::Ready;
            }
            if report.model_loadability == InspectionReadiness::Unverified {
                report.model_loadability = InspectionReadiness::Ready;
            }
        }
        StructuralValidation::Invalid(issues) => {
            report.structural_binding = InspectionReadiness::Invalid;
            report.model_loadability = InspectionReadiness::Invalid;
            for issue in issues {
                push(report, issue, InspectionSeverity::Error);
            }
        }
        StructuralValidation::Unverified(issue) => {
            if matches!(
                report.structural_binding,
                InspectionReadiness::Ready | InspectionReadiness::Unverified
            ) {
                report.structural_binding = InspectionReadiness::Unverified;
            }
            if matches!(
                report.model_loadability,
                InspectionReadiness::Ready | InspectionReadiness::Unverified
            ) {
                report.model_loadability = InspectionReadiness::Unverified;
            }
            push(report, issue, InspectionSeverity::Warning);
        }
    }
}

fn inspect_gguf_projector(
    report: &mut ModelInspectionReport,
    path: &Path,
    architecture: GgufArchitecture,
    model_checkpoint: &GgufCheckpoint,
    model_metadata: &HashMap<String, GgufMetadataValue>,
) -> eredu_architectures::preparation::GgufArtifactComposition {
    use eredu_architectures::preparation::GgufArtifactComposition;

    let mut composition = GgufArtifactComposition::ModelOnly;
    match architecture {
        GgufArchitecture::Qwen3Vl | GgufArchitecture::Qwen3VlMoe => {
            match resolve_projector(path, architecture) {
                Ok(Some((projector, checkpoint, metadata))) => {
                    let validation = structural::validate_qwen3_vl_projector_gguf(
                        architecture,
                        model_checkpoint,
                        model_metadata,
                        &checkpoint,
                        &metadata,
                    );
                    let exact = matches!(validation, structural::StructuralValidation::Exact);
                    if exact {
                        composition = GgufArtifactComposition::ValidatedMediaProjector;
                    }
                    apply_structural_validation(report, validation, &projector);
                    report.multimodal = if !exact {
                        InspectionReadiness::Invalid
                    } else if cfg!(feature = "image") {
                        InspectionReadiness::Ready
                    } else {
                        InspectionReadiness::Unsupported
                    };
                    report.requirements.push(InspectionRequirement {
                        code: InspectionIssueCode::MissingMediaProjector,
                        readiness: if exact {
                            InspectionReadiness::Ready
                        } else {
                            InspectionReadiness::Invalid
                        },
                        detail: if exact {
                            "validated qwen3vl vision projector".into()
                        } else {
                            "qwen3vl vision projector is structurally incompatible".into()
                        },
                        path: Some(projector),
                    });
                }
                Ok(None) => reject_projector(
                    report,
                    path.to_path_buf(),
                    "Qwen3-VL preparation omitted its required media projector".into(),
                    true,
                ),
                Err(error) => reject_projector(report, path.to_path_buf(), error.to_string(), true),
            }
        }
        GgufArchitecture::Qwen35 | GgufArchitecture::Qwen35Moe => {
            match resolve_projector(path, architecture) {
                Ok(Some((projector_path, checkpoint, metadata))) => {
                    let validation = structural::validate_qwen35_projector_gguf(
                        model_checkpoint,
                        model_metadata,
                        &checkpoint,
                        &metadata,
                    );
                    let exact = matches!(validation, structural::StructuralValidation::Exact);
                    if exact {
                        composition = GgufArtifactComposition::ValidatedMediaProjector;
                    }
                    apply_structural_validation(report, validation, &projector_path);
                    report.multimodal = if !exact {
                        InspectionReadiness::Invalid
                    } else if cfg!(feature = "image") {
                        InspectionReadiness::Ready
                    } else {
                        InspectionReadiness::Unsupported
                    };
                    report.requirements.push(InspectionRequirement {
                        code: InspectionIssueCode::MissingMediaProjector,
                        readiness: if exact {
                            InspectionReadiness::Ready
                        } else {
                            InspectionReadiness::Invalid
                        },
                        detail: if exact {
                            "validated Qwen3.5 vision projector".into()
                        } else {
                            "Qwen3.5 vision projector is structurally incompatible".into()
                        },
                        path: Some(projector_path),
                    });
                }
                Ok(None) => {
                    report.multimodal = InspectionReadiness::Missing;
                    report.requirements.push(InspectionRequirement {
                        code: InspectionIssueCode::MissingMediaProjector,
                        readiness: InspectionReadiness::Missing,
                        detail: "Qwen3.5 text loading is available, but image/video input requires a sibling qwen35 mmproj GGUF".into(),
                        path: None,
                    });
                    report.issue(
                        InspectionIssueCode::MissingMediaProjector,
                        InspectionSeverity::Warning,
                        "Qwen3.5 has no sibling multimodal projector; text loading remains available",
                        Some(path.to_path_buf()),
                    );
                }
                Err(error) => reject_projector(report, path.to_path_buf(), error.to_string(), true),
            }
        }
        GgufArchitecture::Inkling => match resolve_projector(path, architecture) {
            Ok(Some((projector_path, checkpoint, metadata))) => {
                let validation = structural::validate_inkling_mmproj_gguf(
                    model_checkpoint,
                    model_metadata,
                    &checkpoint,
                    &metadata,
                );
                let exact = matches!(validation, structural::StructuralValidation::Exact);
                if exact {
                    composition = GgufArtifactComposition::ValidatedMediaProjector;
                }
                apply_structural_validation(report, validation, &projector_path);
                report.multimodal = if !exact {
                    InspectionReadiness::Invalid
                } else if cfg!(all(feature = "image", feature = "audio")) {
                    InspectionReadiness::Ready
                } else {
                    InspectionReadiness::Unsupported
                };
            }
            Ok(None) => {
                report.multimodal = InspectionReadiness::Missing;
                report.requirements.push(InspectionRequirement {
                    code: InspectionIssueCode::MissingMediaProjector,
                    readiness: InspectionReadiness::Missing,
                    detail: "Inkling text loading is available, but image/audio input requires a sibling mmproj GGUF".into(),
                    path: None,
                });
                report.issue(
                    InspectionIssueCode::MissingMediaProjector,
                    InspectionSeverity::Warning,
                    "Inkling has no sibling multimodal projector; text loading remains available",
                    Some(path.to_path_buf()),
                );
            }
            Err(error) => reject_projector(report, path.to_path_buf(), error.to_string(), true),
        },
        GgufArchitecture::Gemma4 => match resolve_projector(path, architecture) {
            Ok(Some((projector_path, checkpoint, metadata))) => {
                let validation = structural::validate_gemma4_mmproj_gguf(
                    model_checkpoint,
                    model_metadata,
                    &checkpoint,
                    &metadata,
                );
                let exact = matches!(validation, structural::StructuralValidation::Exact);
                if exact {
                    composition = GgufArtifactComposition::ValidatedMediaProjector;
                }
                apply_structural_validation(report, validation, &projector_path);
                report.multimodal = if !exact {
                    InspectionReadiness::Invalid
                } else if cfg!(any(feature = "image", feature = "audio")) {
                    InspectionReadiness::Ready
                } else {
                    InspectionReadiness::Unsupported
                };
                report.requirements.push(InspectionRequirement {
                    code: InspectionIssueCode::MissingMediaProjector,
                    readiness: if exact {
                        InspectionReadiness::Ready
                    } else {
                        InspectionReadiness::Invalid
                    },
                    detail: if exact {
                        "validated Gemma 4 vision/audio projector".into()
                    } else {
                        "Gemma 4 media projector is structurally incompatible".into()
                    },
                    path: Some(projector_path),
                });
            }
            Ok(None) => {
                report.multimodal = InspectionReadiness::Missing;
                report.requirements.push(InspectionRequirement {
                    code: InspectionIssueCode::MissingMediaProjector,
                    readiness: InspectionReadiness::Missing,
                    detail: "Gemma 4 text loading is available, but image/audio input requires a sibling mmproj GGUF".into(),
                    path: None,
                });
                report.issue(
                    InspectionIssueCode::MissingMediaProjector,
                    InspectionSeverity::Warning,
                    "Gemma 4 has no sibling multimodal projector; text loading remains available",
                    Some(path.to_path_buf()),
                );
            }
            Err(error) => reject_projector(report, path.to_path_buf(), error.to_string(), true),
        },
        GgufArchitecture::MuseGlimmer => match resolve_projector(path, architecture) {
            Ok(Some((projector_path, checkpoint, metadata))) => {
                let validation = structural::validate_muse_glimmer_projector_gguf(
                    model_checkpoint,
                    model_metadata,
                    &checkpoint,
                    &metadata,
                );
                let exact = matches!(validation, structural::StructuralValidation::Exact);
                if exact {
                    composition = GgufArtifactComposition::ValidatedMediaProjector;
                }
                apply_structural_validation(report, validation, &projector_path);
                report.multimodal = if !exact {
                    InspectionReadiness::Invalid
                } else if cfg!(feature = "image") {
                    InspectionReadiness::Ready
                } else {
                    InspectionReadiness::Unsupported
                };
                report.requirements.push(InspectionRequirement {
                    code: InspectionIssueCode::MissingMediaProjector,
                    readiness: if exact {
                        InspectionReadiness::Ready
                    } else {
                        InspectionReadiness::Invalid
                    },
                    detail: if exact {
                        "validated image-only Muse-Glimmer projector".into()
                    } else {
                        "Muse-Glimmer projector is structurally incompatible".into()
                    },
                    path: Some(projector_path),
                });
            }
            Ok(None) => {
                report.multimodal = InspectionReadiness::Missing;
                report.requirements.push(InspectionRequirement {
                        code: InspectionIssueCode::MissingMediaProjector,
                        readiness: InspectionReadiness::Missing,
                        detail: "Muse-Glimmer text loading is available, but image input requires the sibling mmproj-kquant.gguf".into(),
                        path: None,
                    });
                report.issue(
                    InspectionIssueCode::MissingMediaProjector,
                    InspectionSeverity::Warning,
                    "Muse-Glimmer has no sibling projector; text loading remains available",
                    Some(path.to_path_buf()),
                );
            }
            Err(error) => reject_projector(report, path.to_path_buf(), error.to_string(), false),
        },
        _ => report.multimodal = InspectionReadiness::NotApplicable,
    }
    composition
}

fn reject_projector(
    report: &mut ModelInspectionReport,
    path: PathBuf,
    detail: String,
    required_for_model: bool,
) {
    report.multimodal = InspectionReadiness::Missing;
    if required_for_model {
        report.model_loadability = InspectionReadiness::Missing;
        report.requested_load = InspectionReadiness::Missing;
    }
    report.requirements.push(InspectionRequirement {
        code: InspectionIssueCode::MissingMediaProjector,
        readiness: InspectionReadiness::Missing,
        detail: detail.clone(),
        path: Some(path.clone()),
    });
    report.issue(
        InspectionIssueCode::MissingMediaProjector,
        if required_for_model {
            InspectionSeverity::Error
        } else {
            InspectionSeverity::Warning
        },
        detail,
        Some(path),
    );
}

fn inspect_safetensors_media(report: &mut ModelInspectionReport, path: &Path) {
    if report.expected_modalities == [ArtifactModality::Text] {
        report.multimodal = InspectionReadiness::NotApplicable;
        return;
    }
    let sidecar = ["processor_config.json", "preprocessor_config.json"]
        .into_iter()
        .map(|name| path.join(name))
        .find(|candidate| candidate.exists());
    if let Some(sidecar) = sidecar {
        let features_available = report
            .expected_modalities
            .iter()
            .all(|modality| match modality {
                ArtifactModality::Text => true,
                ArtifactModality::Image | ArtifactModality::Video => {
                    cfg!(feature = "image")
                }
                ArtifactModality::Audio => cfg!(feature = "audio"),
            });
        report.multimodal = if features_available {
            InspectionReadiness::Ready
        } else {
            InspectionReadiness::Unsupported
        };
        report.requirements.push(InspectionRequirement {
            code: InspectionIssueCode::MissingProcessor,
            readiness: report.multimodal,
            detail: if features_available {
                "processor sidecar and required media build features are available".into()
            } else {
                "processor sidecar exists, but required image/audio processing features are not enabled".into()
            },
            path: Some(sidecar),
        });
    } else {
        report.multimodal = InspectionReadiness::Missing;
        report.issue(
            InspectionIssueCode::MissingProcessor,
            InspectionSeverity::Warning,
            "the resolved multimodal architecture has no processor_config.json or preprocessor_config.json sidecar",
            Some(path.to_path_buf()),
        );
    }
}

fn artifact_modalities(modalities: InputModalities) -> Vec<ArtifactModality> {
    [
        (modalities.text, ArtifactModality::Text),
        (modalities.image, ArtifactModality::Image),
        (modalities.video, ArtifactModality::Video),
        (modalities.audio, ArtifactModality::Audio),
    ]
    .into_iter()
    .filter_map(|(enabled, modality)| enabled.then_some(modality))
    .collect()
}

fn reject_load_policy(report: &mut ModelInspectionReport, error: &Error) {
    report.requested_load = InspectionReadiness::Unsupported;
    let (code, detail) = match error {
        Error::Quantization(detail) => (
            InspectionIssueCode::UnsupportedQuantizationRequest,
            detail.clone(),
        ),
        Error::Parallel(detail) => (
            InspectionIssueCode::UnsupportedParallelTopology,
            detail.clone(),
        ),
        Error::Artifact(eredu_core::artifact::ArtifactError::UnsupportedQuantizationPolicy(
            detail,
        )) => (
            InspectionIssueCode::UnsupportedQuantizationRequest,
            detail.clone(),
        ),
        Error::Artifact(eredu_core::artifact::ArtifactError::UnsupportedResidencyPolicy(
            detail,
        )) => (
            InspectionIssueCode::UnsupportedResidencyPolicy,
            detail.clone(),
        ),
        Error::UnsupportedArchitecture(detail)
            if detail.contains("residency")
                || detail.contains("stream")
                || detail.contains("expert cach") =>
        {
            (
                InspectionIssueCode::UnsupportedResidencyPolicy,
                detail.clone(),
            )
        }
        _ => (
            InspectionIssueCode::UnsupportedArchitecture,
            error.to_string(),
        ),
    };
    report.issue(
        code,
        InspectionSeverity::Error,
        detail,
        Some(report.path.clone()),
    );
}

fn parse_unsupported_type_code(detail: &str) -> Option<u32> {
    let marker = "unsupported GGML type ";
    let start = detail.find(marker)? + marker.len();
    detail[start..]
        .split(|character: char| !character.is_ascii_digit())
        .next()?
        .parse()
        .ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_minimal_llama_gguf(path: &Path, include_embedding_length: bool) {
        let mut metadata = BTreeMap::from([
            (
                "general.architecture".into(),
                eredu_gguf::MetadataValue::String("llama".into()),
            ),
            (
                "llama.block_count".into(),
                eredu_gguf::MetadataValue::Uint32(1),
            ),
        ]);
        if include_embedding_length {
            metadata.insert(
                "llama.embedding_length".into(),
                eredu_gguf::MetadataValue::Uint32(1),
            );
        }
        let data = 1.0_f32.to_le_bytes();
        eredu_gguf::Writer::default()
            .write(
                std::fs::File::create(path).unwrap(),
                &metadata,
                &[eredu_gguf::TensorInput {
                    name: "token_embd.weight",
                    dimensions: &[1],
                    ggml_type: eredu_gguf::GgmlType::F32,
                    data: &data,
                }],
            )
            .unwrap();
    }

    #[test]
    fn text_only_architecture_skips_processor_and_media_feature_requirements() {
        let mut report =
            ModelInspectionReport::unverified(Path::new("unused"), ArtifactFormat::SafeTensors);
        report.expected_modalities = artifact_modalities(InputModalities::TEXT);

        inspect_safetensors_media(&mut report, Path::new("unused"));

        assert_eq!(report.expected_modalities, [ArtifactModality::Text]);
        assert_eq!(report.multimodal, InspectionReadiness::NotApplicable);
        assert!(report.requirements.is_empty());
        assert!(report.issues.is_empty());
    }

    #[test]
    fn composite_gguf_plan_keeps_readiness_and_expected_modalities_consistent() {
        use eredu_architectures::preparation::{
            gguf_composite_artifact_plan, GgufArtifactComposition,
        };

        for (architecture, expected) in [
            (
                GgufArchitecture::Qwen35,
                vec![
                    ArtifactModality::Text,
                    ArtifactModality::Image,
                    ArtifactModality::Video,
                ],
            ),
            (
                GgufArchitecture::Gemma4,
                vec![
                    ArtifactModality::Text,
                    ArtifactModality::Image,
                    ArtifactModality::Audio,
                ],
            ),
            (
                GgufArchitecture::MuseGlimmer,
                vec![ArtifactModality::Text, ArtifactModality::Image],
            ),
        ] {
            let modalities = gguf_composite_artifact_plan(architecture)
                .input_modalities(GgufArtifactComposition::ValidatedMediaProjector);
            assert_eq!(artifact_modalities(modalities), expected);
        }
    }

    #[test]
    fn gguf_inspection_enriches_the_portable_validated_checkpoint() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("model.gguf");
        write_minimal_llama_gguf(&path, true);

        let report = inspect_model(&path, MlxInspectionOptions::default()).unwrap();

        assert_eq!(report.container, InspectionReadiness::Ready);
        assert_eq!(report.architecture_support, InspectionReadiness::Ready);
        assert_eq!(report.architecture.as_deref(), Some("llama"));
        assert_eq!(report.tensor_count, Some(1));
    }

    #[test]
    fn gguf_inspection_surfaces_the_portable_admission_floor() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("model.gguf");
        write_minimal_llama_gguf(&path, false);

        let report = inspect_model(&path, MlxInspectionOptions::default()).unwrap();

        assert_eq!(report.container, InspectionReadiness::Ready);
        assert_eq!(report.structural_binding, InspectionReadiness::Invalid);
        assert!(report
            .issues
            .iter()
            .any(|issue| issue.detail.contains("llama.embedding_length")));
    }
}
