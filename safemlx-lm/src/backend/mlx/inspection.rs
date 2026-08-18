//! Side-effect-free MLX model artifact compatibility inspection.

use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    path::{Path, PathBuf},
};

#[cfg(test)]
use crate::backend::mlx::resolve_model_config;
use crate::backend::mlx::structural::{self, GgufArchitectureValidation};
use safemlx::ops::GgufCheckpoint;
use safemlx_gguf::MetadataValue as GgufMetadataValue;
use safemlx_lm_core::{
    ArtifactFormat, ArtifactModality, ArtifactTensorEncoding, GgufArchitecture, InspectionIssue,
    InspectionIssueCode, InspectionReadiness, InspectionRequirement, InspectionSeverity,
    ModelInspectionReport, ModelKind, Observed,
};
use serde_json::Value;

use super::*;
use crate::{
    architectures::{
        gemma4::model as gemma4, inkling::model as inkling, muse_glimmer,
        qwen::vl::model as qwen3_vl,
    },
    backend::mlx::runtime::checkpoint::store::{SafetensorsWeightStore, WeightStore},
};

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
    report.resources.model_kind = report.model_kind;
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
        match crate::backend::mlx::resolve_model_config(config) {
            Ok(supported) => {
                resolved_kind = Some(supported.kind);
                report.model_kind = Some(supported.kind);
                report.architecture = Some(supported.effective_model_type);
                report.expected_modalities = modalities_for_safetensors(supported.kind, config);
                report.architecture_support = InspectionReadiness::Ready;
                match options.load.validate_preparation(
                    supported.kind,
                    None,
                    safemlx_lm_core::ArtifactFormat::SafeTensors,
                ) {
                    Ok(_) => report.requested_load = InspectionReadiness::Ready,
                    Err(error) => reject_load_policy(&mut report, &error),
                }
            }
            Err(error) => {
                report.architecture_support = match &error {
                    crate::backend::mlx::ModelConfigResolutionError::Loader(
                        Error::UnsupportedModelType(_),
                    ) => InspectionReadiness::Unsupported,
                    _ => InspectionReadiness::Invalid,
                };
                report.model_loadability = report.architecture_support;
                report.structural_binding = report.architecture_support;
                report.requested_load = report.model_loadability;
                report.issue(
                    match &error {
                        crate::backend::mlx::ModelConfigResolutionError::Loader(
                            Error::UnsupportedModelType(_),
                        ) => InspectionIssueCode::UnsupportedArchitecture,
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
                        let bytes = metadata.logical_byte_len as u64;
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
                    crate::backend::mlx::runtime::checkpoint::store::WeightStoreError::MissingShard { .. }
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
                crate::backend::mlx::runtime::checkpoint::store::WeightStoreError::MissingShard { .. }
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
    let checkpoint = match GgufCheckpoint::open(path) {
        Ok(checkpoint) => checkpoint,
        Err(error) => {
            let detail = error.to_string();
            let type_code = parse_unsupported_type_code(&detail);
            report.container = InspectionReadiness::Invalid;
            report.model_loadability = InspectionReadiness::Invalid;
            report.requested_load = InspectionReadiness::Invalid;
            report.text_generation = InspectionReadiness::Invalid;
            report.tokenizer = InspectionReadiness::Unverified;
            report.chat_template = InspectionReadiness::Unverified;
            report.semantic_streaming = InspectionReadiness::Unverified;
            report.native_tools = InspectionReadiness::Unverified;
            report.multimodal = InspectionReadiness::Unverified;
            report.issues.push(InspectionIssue {
                code: if type_code.is_some() {
                    InspectionIssueCode::UnsupportedTensorEncoding
                } else {
                    InspectionIssueCode::InvalidContainer
                },
                severity: InspectionSeverity::Error,
                detail,
                path: Some(path.to_path_buf()),
                metadata_key: None,
                tensor_name: None,
                tensor_type_code: type_code,
            });
            return report;
        }
    };
    report.container = InspectionReadiness::Ready;
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

    let metadata = crate::backend::mlx::runtime::checkpoint::load::gguf_metadata(&checkpoint);
    let architecture = match metadata.get("general.architecture") {
        Some(GgufMetadataValue::String(value)) => {
            report.architecture = Some(value.clone());
            value.clone()
        }
        Some(_) => {
            report.model_loadability = InspectionReadiness::Invalid;
            report.requested_load = InspectionReadiness::Invalid;
            report.issues.push(InspectionIssue {
                code: InspectionIssueCode::InvalidConfiguration,
                severity: InspectionSeverity::Error,
                detail: "GGUF metadata key general.architecture has the wrong type".into(),
                path: Some(path.to_path_buf()),
                metadata_key: Some("general.architecture".into()),
                tensor_name: None,
                tensor_type_code: None,
            });
            return report;
        }
        None => {
            report.model_loadability = InspectionReadiness::Invalid;
            report.requested_load = InspectionReadiness::Invalid;
            report.issues.push(InspectionIssue {
                code: InspectionIssueCode::InvalidConfiguration,
                severity: InspectionSeverity::Error,
                detail: "GGUF metadata is missing required key general.architecture".into(),
                path: Some(path.to_path_buf()),
                metadata_key: Some("general.architecture".into()),
                tensor_name: None,
                tensor_type_code: None,
            });
            return report;
        }
    };

    match GgufArchitecture::resolve(&architecture) {
        Ok(gguf_architecture) => {
            let kind = gguf_architecture.model_kind();
            report.model_kind = Some(kind);
            report.architecture_support = InspectionReadiness::Ready;
            report.expected_modalities = modalities_for_gguf(gguf_architecture);
            if let Err(error) = gguf_architecture.validate_catalog(&checkpoint, &metadata) {
                report.structural_binding = InspectionReadiness::Invalid;
                report.model_loadability = InspectionReadiness::Invalid;
                report.issue(
                    InspectionIssueCode::InvalidConfiguration,
                    InspectionSeverity::Error,
                    error.to_string(),
                    Some(path.to_path_buf()),
                );
            } else {
                apply_structural_validation(
                    &mut report,
                    structural::validate_gguf(
                        gguf_architecture,
                        &checkpoint,
                        &metadata,
                        options.load,
                    ),
                    path,
                );
            }
            match gguf_architecture.validate_load_policy(options.load) {
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
            inspect_gguf_projector(&mut report, path, gguf_architecture, &checkpoint, &metadata);
        }
        Err(error) => {
            report.architecture_support = InspectionReadiness::Unsupported;
            report.structural_binding = InspectionReadiness::Unsupported;
            report.model_loadability = InspectionReadiness::Unsupported;
            report.requested_load = InspectionReadiness::Unsupported;
            report.issue(
                InspectionIssueCode::UnsupportedArchitecture,
                InspectionSeverity::Error,
                error.to_string(),
                Some(path.to_path_buf()),
            );
            report.multimodal = InspectionReadiness::NotApplicable;
        }
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
            StructuralIssueKind::QuantizationCompanionMismatch => {
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
) {
    match architecture {
        GgufArchitecture::Qwen3Vl | GgufArchitecture::Qwen3VlMoe => {
            match qwen3_vl::find_qwen3_vl_mmproj(path) {
                Ok(projector) => match GgufCheckpoint::open(&projector) {
                    Ok(checkpoint) => {
                        let metadata =
                            crate::backend::mlx::runtime::checkpoint::load::gguf_metadata(
                                &checkpoint,
                            );
                        let validation = structural::validate_qwen3_vl_projector_gguf(
                            model_checkpoint,
                            model_metadata,
                            &checkpoint,
                            &metadata,
                        );
                        let exact = matches!(validation, structural::StructuralValidation::Exact);
                        apply_structural_validation(report, validation, &projector);
                        report.multimodal = if !exact {
                            InspectionReadiness::Invalid
                        } else if cfg!(feature = "image-processing") {
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
                    Err(error) => reject_projector(report, projector, error.to_string(), true),
                },
                Err(error) => reject_projector(report, path.to_path_buf(), error.to_string(), true),
            }
        }
        GgufArchitecture::Qwen35 | GgufArchitecture::Qwen35Moe => {
            match crate::architectures::qwen::hybrid::qwen3_5::open_sibling_mmproj(path) {
                Ok(Some(mmproj)) => {
                    let projector_path =
                        crate::backend::mlx::runtime::checkpoint::gguf::find_sibling_mmproj(
                            path, "qwen35",
                        )
                        .ok()
                        .flatten()
                        .unwrap_or_else(|| path.to_path_buf());
                    let validation = structural::validate_qwen35_projector_gguf(
                        model_checkpoint,
                        model_metadata,
                        &mmproj.checkpoint,
                        &mmproj.metadata,
                    );
                    let exact = matches!(validation, structural::StructuralValidation::Exact);
                    apply_structural_validation(report, validation, &projector_path);
                    report.multimodal = if !exact {
                        InspectionReadiness::Invalid
                    } else if cfg!(feature = "image-processing") {
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
        GgufArchitecture::Inkling => match inkling::open_sibling_mmproj(path) {
            Ok(Some(mmproj)) => {
                let projector_path =
                    crate::backend::mlx::runtime::checkpoint::gguf::find_sibling_mmproj(
                        path, "inkling",
                    )
                    .ok()
                    .flatten()
                    .unwrap_or_else(|| path.to_path_buf());
                let validation = structural::validate_inkling_mmproj_gguf(model_metadata, &mmproj);
                let exact = matches!(validation, structural::StructuralValidation::Exact);
                apply_structural_validation(report, validation, &projector_path);
                report.multimodal = if !exact {
                    InspectionReadiness::Invalid
                } else if cfg!(all(
                    feature = "image-processing",
                    feature = "audio-processing"
                )) {
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
        GgufArchitecture::Gemma4 => match gemma4::open_sibling_mmproj(path) {
            Ok(Some(mmproj)) => {
                let projector_path =
                    crate::backend::mlx::runtime::checkpoint::gguf::find_sibling_mmproj(
                        path, "gemma4",
                    )
                    .ok()
                    .flatten()
                    .unwrap_or_else(|| path.to_path_buf());
                let validation = structural::validate_gemma4_mmproj_gguf(
                    model_checkpoint,
                    model_metadata,
                    &mmproj,
                );
                let exact = matches!(validation, structural::StructuralValidation::Exact);
                apply_structural_validation(report, validation, &projector_path);
                report.multimodal = if !exact {
                    InspectionReadiness::Invalid
                } else if cfg!(any(
                    feature = "image-processing",
                    feature = "audio-processing"
                )) {
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
        GgufArchitecture::MuseGlimmer => match muse_glimmer::open_sibling_mmproj(path) {
            Ok(Some(mmproj)) => {
                let validation = structural::validate_muse_glimmer_projector_gguf(
                    model_checkpoint,
                    model_metadata,
                    &mmproj.checkpoint,
                    &mmproj.metadata,
                );
                let exact = matches!(validation, structural::StructuralValidation::Exact);
                apply_structural_validation(report, validation, &mmproj.path);
                report.multimodal = if !exact {
                    InspectionReadiness::Invalid
                } else if cfg!(feature = "image-processing") {
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
                    path: Some(mmproj.path),
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
                    cfg!(feature = "image-processing")
                }
                ArtifactModality::Audio => cfg!(feature = "audio-processing"),
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

fn modalities_for_safetensors(kind: ModelKind, config: &Value) -> Vec<ArtifactModality> {
    let mut modalities = BTreeSet::from([ArtifactModality::Text]);
    match kind {
        ModelKind::Gemma4 | ModelKind::Inkling => {
            modalities.insert(ArtifactModality::Image);
            modalities.insert(ArtifactModality::Audio);
        }
        ModelKind::Qwen3Vl | ModelKind::Qwen3VlMoe => {
            modalities.insert(ArtifactModality::Image);
            modalities.insert(ArtifactModality::Video);
        }
        ModelKind::MuseGlimmer => {
            modalities.insert(ArtifactModality::Image);
            modalities.insert(ArtifactModality::Video);
        }
        ModelKind::Qwen35
            if config.get("vision_config").is_some()
                || config
                    .get("text_config")
                    .and_then(|text| text.get("vision_config"))
                    .is_some() =>
        {
            modalities.insert(ArtifactModality::Image);
            modalities.insert(ArtifactModality::Video);
        }
        ModelKind::DeepSeekV3
        | ModelKind::DeepSeekV4
        | ModelKind::GptOss
        | ModelKind::KimiLinear
        | ModelKind::Llama
        | ModelKind::Lfm2
        | ModelKind::NemotronH
        | ModelKind::PersonaPlex
        | ModelKind::Qwen2
        | ModelKind::Qwen3
        | ModelKind::Qwen3Next
        | ModelKind::Qwen35 => {}
    }
    modalities.into_iter().collect()
}

fn modalities_for_gguf(architecture: GgufArchitecture) -> Vec<ArtifactModality> {
    match architecture {
        GgufArchitecture::Inkling => vec![
            ArtifactModality::Text,
            ArtifactModality::Image,
            ArtifactModality::Audio,
        ],
        GgufArchitecture::Qwen3Vl | GgufArchitecture::Qwen3VlMoe => vec![
            ArtifactModality::Text,
            ArtifactModality::Image,
            ArtifactModality::Video,
        ],
        GgufArchitecture::MuseGlimmer => vec![ArtifactModality::Text],
        _ => vec![ArtifactModality::Text],
    }
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
        Error::Artifact(
            safemlx_lm_core::artifact::ArtifactError::UnsupportedQuantizationPolicy(detail),
        ) => (
            InspectionIssueCode::UnsupportedQuantizationRequest,
            detail.clone(),
        ),
        Error::Artifact(safemlx_lm_core::artifact::ArtifactError::UnsupportedResidencyPolicy(
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
    use std::io::{Seek, Write};

    use safemlx::module::ModuleParameters;
    use safemlx_gguf::{GgmlType, MetadataArray, MetadataValue, TensorInput, Writer};
    use safetensors::tensor::{serialize_to_file, Dtype, TensorView};
    use serde_json::json;

    use super::*;
    use crate::{
        architectures::{deepseek_v4::model as deepseek_v4, moshi::personaplex},
        backend::mlx::{NonExpertWeightResidency, WeightResidency},
    };

    fn llama_config() -> Value {
        json!({
            "model_type": "llama",
            "hidden_size": 8,
            "num_hidden_layers": 1,
            "intermediate_size": 16,
            "num_attention_heads": 2,
            "rms_norm_eps": 0.00001,
            "vocab_size": 32,
            "num_key_value_heads": 2,
            "max_position_embeddings": 128,
            "head_dim": 4
        })
    }

    fn write_safetensors_dir(config: &Value) -> tempfile::TempDir {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(
            directory.path().join("config.json"),
            serde_json::to_vec(config).unwrap(),
        )
        .unwrap();
        let bytes = [0xff_u8; 4];
        let view = TensorView::new(Dtype::F32, vec![1], &bytes).unwrap();
        serialize_to_file(
            [("poison.weight", view)],
            None,
            &directory.path().join("model.safetensors"),
        )
        .unwrap();
        directory
    }

    fn write_typed_safetensors_dir(
        config: &Value,
        specs: &[(String, Vec<usize>, Dtype)],
    ) -> tempfile::TempDir {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(
            directory.path().join("config.json"),
            serde_json::to_vec(config).unwrap(),
        )
        .unwrap();
        let payloads = specs
            .iter()
            .map(|(_, shape, dtype)| {
                let bytes = match dtype {
                    Dtype::F8_E4M3 | Dtype::U8 | Dtype::I8 => 1,
                    Dtype::F16 | Dtype::BF16 => 2,
                    Dtype::F32 | Dtype::U32 => 4,
                    other => panic!("unsupported fixture dtype {other:?}"),
                };
                vec![0xff; shape.iter().product::<usize>() * bytes]
            })
            .collect::<Vec<_>>();
        let views = specs
            .iter()
            .zip(&payloads)
            .map(|((name, shape, dtype), payload)| {
                (
                    name.as_str(),
                    TensorView::new(*dtype, shape.clone(), payload).unwrap(),
                )
            })
            .collect::<Vec<_>>();
        serialize_to_file(views, None, &directory.path().join("model.safetensors")).unwrap();
        directory
    }

    fn affine_fixture_specs(
        specs: Vec<(String, Vec<usize>)>,
        group_size: usize,
        quantized: impl Fn(&str, &[usize]) -> bool,
    ) -> Vec<(String, Vec<usize>, Dtype)> {
        let mut encoded = Vec::new();
        for (name, shape) in specs {
            if !quantized(&name, &shape) {
                encoded.push((name, shape, Dtype::F32));
                continue;
            }
            let input = *shape.last().unwrap();
            assert_eq!(input % group_size, 0, "{name}");
            assert_eq!(input % 8, 0, "{name}");
            let mut packed = shape.clone();
            *packed.last_mut().unwrap() = input / 8;
            let mut companion = shape;
            *companion.last_mut().unwrap() = input / group_size;
            let prefix = name.trim_end_matches(".weight").to_owned();
            encoded.push((name, packed, Dtype::U32));
            encoded.push((format!("{prefix}.scales"), companion.clone(), Dtype::F32));
            encoded.push((format!("{prefix}.biases"), companion, Dtype::F32));
        }
        encoded
    }

    fn fp8_fixture_specs(
        specs: Vec<(String, Vec<usize>)>,
        quantized: impl Fn(&str, &[usize]) -> bool,
    ) -> Vec<(String, Vec<usize>, Dtype)> {
        let mut encoded = Vec::new();
        for (name, shape) in specs {
            if !quantized(&name, &shape) {
                encoded.push((name, shape, Dtype::F32));
                continue;
            }
            let rank = shape.len();
            assert!(rank >= 2, "{name}");
            let mut scale = shape.clone();
            scale[rank - 2] = scale[rank - 2].div_ceil(128);
            scale[rank - 1] = scale[rank - 1].div_ceil(128);
            let companion = if name.ends_with(".weight") {
                format!("{}.weight_scale_inv", name.trim_end_matches(".weight"))
            } else {
                format!("{name}_scale_inv")
            };
            encoded.push((name, shape, Dtype::F8_E4M3));
            encoded.push((companion, scale, Dtype::F32));
        }
        encoded
    }

    fn llama_safetensor_specs() -> Vec<(String, Vec<usize>)> {
        vec![
            ("model.embed_tokens.weight".into(), vec![32, 8]),
            ("model.norm.weight".into(), vec![8]),
            ("model.layers.0.input_layernorm.weight".into(), vec![8]),
            (
                "model.layers.0.post_attention_layernorm.weight".into(),
                vec![8],
            ),
            ("model.layers.0.self_attn.q_proj.weight".into(), vec![8, 8]),
            ("model.layers.0.self_attn.k_proj.weight".into(), vec![8, 8]),
            ("model.layers.0.self_attn.v_proj.weight".into(), vec![8, 8]),
            ("model.layers.0.self_attn.o_proj.weight".into(), vec![8, 8]),
            ("model.layers.0.mlp.gate_proj.weight".into(), vec![16, 8]),
            ("model.layers.0.mlp.up_proj.weight".into(), vec![16, 8]),
            ("model.layers.0.mlp.down_proj.weight".into(), vec![8, 16]),
        ]
    }

    fn write_complete_safetensors_dir(
        mutate: impl FnOnce(&mut Vec<(String, Vec<usize>)>),
    ) -> tempfile::TempDir {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(
            directory.path().join("config.json"),
            serde_json::to_vec(&llama_config()).unwrap(),
        )
        .unwrap();
        let mut specs = llama_safetensor_specs();
        mutate(&mut specs);
        let payloads = specs
            .iter()
            .map(|(_, shape)| vec![0xff; shape.iter().product::<usize>() * 4])
            .collect::<Vec<_>>();
        let views = specs
            .iter()
            .zip(&payloads)
            .map(|((name, shape), payload)| {
                (
                    name.as_str(),
                    TensorView::new(Dtype::F32, shape.clone(), payload).unwrap(),
                )
            })
            .collect::<Vec<_>>();
        serialize_to_file(views, None, &directory.path().join("model.safetensors")).unwrap();
        directory
    }

    fn personaplex_safetensor_specs() -> Vec<(String, Vec<usize>)> {
        let args = personaplex::model_args_7b_v1();
        let dim = args.dim as usize;
        let depth_dim = args.depformer_dim as usize;
        let temporal_feed_forward = args.dim_feedforward.unwrap_or(4 * args.dim) as usize;
        let temporal_hidden = if temporal_feed_forward == 4 * dim {
            11 * dim / 4
        } else {
            2 * temporal_feed_forward / 3
        };
        let depth_feed_forward = args
            .depformer_dim_feedforward
            .unwrap_or(4 * args.depformer_dim) as usize;
        let depth_hidden = if depth_feed_forward == 4 * depth_dim {
            11 * depth_dim / 4
        } else {
            2 * depth_feed_forward / 3
        };
        let mut specs = vec![
            (
                "text_emb.weight".into(),
                vec![args.text_card as usize + 1, dim],
            ),
            (
                "text_linear.weight".into(),
                vec![args.text_card as usize, dim],
            ),
            ("out_norm.alpha".into(), vec![1, dim]),
        ];
        for codebook in 0..args.n_q as usize {
            specs.push((
                format!("emb.{codebook}.weight"),
                vec![args.card as usize + 1, dim],
            ));
        }
        for layer in 0..args.num_layers as usize {
            let prefix = format!("transformer.layers.{layer}");
            specs.extend([
                (format!("{prefix}.norm1.alpha"), vec![1, dim]),
                (format!("{prefix}.norm2.alpha"), vec![1, dim]),
                (
                    format!("{prefix}.self_attn.in_proj_weight"),
                    vec![3 * dim, dim],
                ),
                (
                    format!("{prefix}.self_attn.out_proj.weight"),
                    vec![dim, dim],
                ),
                (
                    format!("{prefix}.gating.linear_in.weight"),
                    vec![2 * temporal_hidden, dim],
                ),
                (
                    format!("{prefix}.gating.linear_out.weight"),
                    vec![dim, temporal_hidden],
                ),
            ]);
        }
        for slice in 0..args.dep_q as usize {
            specs.extend([
                (
                    if slice == 0 {
                        "depformer_text_emb.weight".into()
                    } else {
                        format!("depformer_emb.{}.weight", slice - 1)
                    },
                    vec![
                        if slice == 0 {
                            args.text_card as usize + 1
                        } else {
                            args.card as usize + 1
                        },
                        depth_dim,
                    ],
                ),
                (format!("depformer_in.{slice}.weight"), vec![depth_dim, dim]),
                (
                    format!("linears.{slice}.weight"),
                    vec![args.card as usize, depth_dim],
                ),
            ]);
        }
        let depth_slices = args.dep_q as usize;
        for layer in 0..args.depformer_num_layers as usize {
            let prefix = format!("depformer.layers.{layer}");
            specs.extend([
                (format!("{prefix}.norm1.alpha"), vec![1, depth_dim]),
                (format!("{prefix}.norm2.alpha"), vec![1, depth_dim]),
                (
                    format!("{prefix}.self_attn.in_proj_weight"),
                    vec![depth_slices * 3 * depth_dim, depth_dim],
                ),
                (
                    format!("{prefix}.self_attn.out_proj.weight"),
                    vec![depth_slices * depth_dim, depth_dim],
                ),
            ]);
            for slice in 0..depth_slices {
                specs.extend([
                    (
                        format!("{prefix}.gating.{slice}.linear_in.weight"),
                        vec![2 * depth_hidden, depth_dim],
                    ),
                    (
                        format!("{prefix}.gating.{slice}.linear_out.weight"),
                        vec![depth_dim, depth_hidden],
                    ),
                ]);
            }
        }
        specs
    }

    fn write_sparse_personaplex_safetensors_dir(
        mutate: impl FnOnce(&mut Vec<(String, Vec<usize>)>),
    ) -> tempfile::TempDir {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(
            directory.path().join("config.json"),
            serde_json::to_vec(&json!({
                "model_type": "personaplex",
                "version": "7b-v1"
            }))
            .unwrap(),
        )
        .unwrap();
        let mut specs = personaplex_safetensor_specs();
        mutate(&mut specs);
        specs.sort_by(|left, right| left.0.cmp(&right.0));
        let mut offset = 0_u64;
        let mut header = serde_json::Map::new();
        for (name, shape) in specs {
            let bytes = shape.iter().try_fold(4_u64, |bytes, dimension| {
                bytes.checked_mul(*dimension as u64)
            });
            let bytes = bytes.expect("PersonaPlex fixture size");
            let end = offset
                .checked_add(bytes)
                .expect("PersonaPlex fixture offset");
            header.insert(
                name,
                json!({
                    "dtype": "F32",
                    "shape": shape,
                    "data_offsets": [offset, end]
                }),
            );
            offset = end;
        }
        let mut header = serde_json::to_vec(&Value::Object(header)).unwrap();
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let path = directory.path().join("model.safetensors");
        let mut file = std::fs::File::create(path).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header).unwrap();
        // The enormous released shapes are represented by sparse poisoned payload
        // holes. Header inspection must never fault these pages into memory.
        file.write_all(&[0xff]).unwrap();
        file.set_len(8 + header.len() as u64 + offset).unwrap();
        directory
    }

    fn qwen3_config(is_moe: bool) -> Value {
        json!({
            "model_type": "qwen3",
            "hidden_size": 32,
            "num_hidden_layers": 1,
            "intermediate_size": if is_moe { 0 } else { 64 },
            "num_attention_heads": 1,
            "rms_norm_eps": 0.000001,
            "vocab_size": 32,
            "num_key_value_heads": 1,
            "max_position_embeddings": 128,
            "rope_theta": 1_000_000.0,
            "head_dim": 32,
            "tie_word_embeddings": true,
            "rope_scaling": null,
            "moe_intermediate_size": if is_moe { 8 } else { 0 },
            "num_experts": if is_moe { 4 } else { 0 },
            "num_experts_per_tok": if is_moe { 2 } else { 0 },
            "norm_topk_prob": is_moe
        })
    }

    fn qwen2_config(tied: bool) -> Value {
        json!({
            "architectures": ["Qwen2ForCausalLM"],
            "model_type": "qwen2",
            "hidden_size": 8,
            "num_hidden_layers": 1,
            "intermediate_size": 16,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "rms_norm_eps": 0.000001,
            "vocab_size": 32,
            "max_position_embeddings": 128,
            "rope_theta": 1_000_000.0,
            "head_dim": 2,
            "tie_word_embeddings": tied,
            "attention_bias": true,
            "mlp_bias": false,
            "use_sliding_window": false
        })
    }

    fn qwen2_safetensor_specs(tied: bool) -> Vec<(String, Vec<usize>, Dtype)> {
        let mut specs = vec![
            ("model.embed_tokens.weight".into(), vec![32, 8], Dtype::F32),
            ("model.norm.weight".into(), vec![8], Dtype::F32),
            (
                "model.layers.0.input_layernorm.weight".into(),
                vec![8],
                Dtype::F32,
            ),
            (
                "model.layers.0.post_attention_layernorm.weight".into(),
                vec![8],
                Dtype::F32,
            ),
            (
                "model.layers.0.self_attn.q_proj.weight".into(),
                vec![8, 8],
                Dtype::F32,
            ),
            (
                "model.layers.0.self_attn.k_proj.weight".into(),
                vec![4, 8],
                Dtype::F32,
            ),
            (
                "model.layers.0.self_attn.v_proj.weight".into(),
                vec![4, 8],
                Dtype::F32,
            ),
            (
                "model.layers.0.self_attn.o_proj.weight".into(),
                vec![8, 8],
                Dtype::F32,
            ),
            (
                "model.layers.0.self_attn.q_proj.bias".into(),
                vec![8],
                Dtype::F32,
            ),
            (
                "model.layers.0.self_attn.k_proj.bias".into(),
                vec![4],
                Dtype::F32,
            ),
            (
                "model.layers.0.self_attn.v_proj.bias".into(),
                vec![4],
                Dtype::F32,
            ),
            (
                "model.layers.0.mlp.gate_proj.weight".into(),
                vec![16, 8],
                Dtype::F32,
            ),
            (
                "model.layers.0.mlp.up_proj.weight".into(),
                vec![16, 8],
                Dtype::F32,
            ),
            (
                "model.layers.0.mlp.down_proj.weight".into(),
                vec![8, 16],
                Dtype::F32,
            ),
        ];
        if !tied {
            specs.push(("lm_head.weight".into(), vec![32, 8], Dtype::F32));
        }
        specs
    }

    fn write_complete_qwen2_safetensors_dir(
        tied: bool,
        mutate: impl FnOnce(&mut Vec<(String, Vec<usize>, Dtype)>),
    ) -> tempfile::TempDir {
        let mut specs = qwen2_safetensor_specs(tied);
        mutate(&mut specs);
        write_typed_safetensors_dir(&qwen2_config(tied), &specs)
    }

    fn qwen3_safetensor_specs() -> Vec<(String, Vec<usize>)> {
        vec![
            ("model.embed_tokens.weight".into(), vec![32, 32]),
            ("model.norm.weight".into(), vec![32]),
            ("model.layers.0.input_layernorm.weight".into(), vec![32]),
            (
                "model.layers.0.post_attention_layernorm.weight".into(),
                vec![32],
            ),
            ("model.layers.0.self_attn.q_norm.weight".into(), vec![32]),
            ("model.layers.0.self_attn.k_norm.weight".into(), vec![32]),
            (
                "model.layers.0.self_attn.q_proj.weight".into(),
                vec![32, 32],
            ),
            (
                "model.layers.0.self_attn.k_proj.weight".into(),
                vec![32, 32],
            ),
            (
                "model.layers.0.self_attn.v_proj.weight".into(),
                vec![32, 32],
            ),
            (
                "model.layers.0.self_attn.o_proj.weight".into(),
                vec![32, 32],
            ),
            ("model.layers.0.mlp.gate_proj.weight".into(), vec![64, 32]),
            ("model.layers.0.mlp.up_proj.weight".into(), vec![64, 32]),
            ("model.layers.0.mlp.down_proj.weight".into(), vec![32, 64]),
        ]
    }

    fn write_complete_qwen3_safetensors_dir(
        mutate: impl FnOnce(&mut Vec<(String, Vec<usize>)>),
    ) -> tempfile::TempDir {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(
            directory.path().join("config.json"),
            serde_json::to_vec(&qwen3_config(false)).unwrap(),
        )
        .unwrap();
        let mut specs = qwen3_safetensor_specs();
        mutate(&mut specs);
        let payloads = specs
            .iter()
            .map(|(_, shape)| vec![0xff; shape.iter().product::<usize>() * 4])
            .collect::<Vec<_>>();
        let views = specs
            .iter()
            .zip(&payloads)
            .map(|((name, shape), payload)| {
                (
                    name.as_str(),
                    TensorView::new(Dtype::F32, shape.clone(), payload).unwrap(),
                )
            })
            .collect::<Vec<_>>();
        serialize_to_file(views, None, &directory.path().join("model.safetensors")).unwrap();
        directory
    }

    fn qwen3_vl_safetensors_config(is_moe: bool) -> Value {
        let model_type = if is_moe { "qwen3_vl_moe" } else { "qwen3_vl" };
        let mut text = qwen3_config(is_moe);
        text["model_type"] = json!(format!("{model_type}_text"));
        text["rope_scaling"] = json!({
            "rope_type": "default",
            "mrope_interleaved": true,
            "mrope_section": [6, 5, 5]
        });
        json!({
            "model_type": model_type,
            "image_token_id": 30,
            "video_token_id": 31,
            "tie_word_embeddings": true,
            "text_config": text,
            "vision_config": {
                "depth": 1,
                "hidden_size": 8,
                "hidden_act": "gelu_pytorch_tanh",
                "intermediate_size": 16,
                "num_heads": 2,
                "num_position_embeddings": 16,
                "in_channels": 3,
                "patch_size": 2,
                "spatial_merge_size": 2,
                "temporal_patch_size": 2,
                "out_hidden_size": 32,
                "deepstack_visual_indexes": [0]
            }
        })
    }

    fn qwen3_vl_safetensor_specs(is_moe: bool) -> Vec<(String, Vec<usize>, Dtype)> {
        let mut specs = qwen3_safetensor_specs();
        if is_moe {
            specs.retain(|(name, _)| !name.contains(".mlp."));
            specs.extend([
                ("model.layers.0.mlp.gate.weight".into(), vec![4, 32]),
                (
                    "model.layers.0.mlp.experts.gate_up_proj".into(),
                    vec![4, 16, 32],
                ),
                (
                    "model.layers.0.mlp.experts.down_proj".into(),
                    vec![4, 32, 8],
                ),
            ]);
        }
        let mut specs = specs
            .into_iter()
            .map(|(name, shape)| {
                let name = name
                    .strip_prefix("model.")
                    .map(|rest| format!("model.language_model.{rest}"))
                    .unwrap_or(name);
                (name, shape, Dtype::F32)
            })
            .collect::<Vec<_>>();
        let mut vision = vec![
            ("model.visual.pos_embed.weight".into(), vec![16, 8]),
            (
                "model.visual.patch_embed.proj.weight".into(),
                vec![8, 3, 2, 2, 2],
            ),
            ("model.visual.patch_embed.proj.bias".into(), vec![8]),
        ];
        let block = "model.visual.blocks.0";
        vision.extend([
            (format!("{block}.norm1.weight"), vec![8]),
            (format!("{block}.norm1.bias"), vec![8]),
            (format!("{block}.attn.qkv.weight"), vec![24, 8]),
            (format!("{block}.attn.qkv.bias"), vec![24]),
            (format!("{block}.attn.proj.weight"), vec![8, 8]),
            (format!("{block}.attn.proj.bias"), vec![8]),
            (format!("{block}.norm2.weight"), vec![8]),
            (format!("{block}.norm2.bias"), vec![8]),
            (format!("{block}.mlp.linear_fc1.weight"), vec![16, 8]),
            (format!("{block}.mlp.linear_fc1.bias"), vec![16]),
            (format!("{block}.mlp.linear_fc2.weight"), vec![8, 16]),
            (format!("{block}.mlp.linear_fc2.bias"), vec![8]),
            ("model.visual.merger.norm.weight".into(), vec![8]),
            ("model.visual.merger.norm.bias".into(), vec![8]),
            ("model.visual.merger.linear_fc1.weight".into(), vec![32, 32]),
            ("model.visual.merger.linear_fc1.bias".into(), vec![32]),
            ("model.visual.merger.linear_fc2.weight".into(), vec![32, 32]),
            ("model.visual.merger.linear_fc2.bias".into(), vec![32]),
            (
                "model.visual.deepstack_merger_list.0.norm.weight".into(),
                vec![32],
            ),
            (
                "model.visual.deepstack_merger_list.0.norm.bias".into(),
                vec![32],
            ),
            (
                "model.visual.deepstack_merger_list.0.linear_fc1.weight".into(),
                vec![32, 32],
            ),
            (
                "model.visual.deepstack_merger_list.0.linear_fc1.bias".into(),
                vec![32],
            ),
            (
                "model.visual.deepstack_merger_list.0.linear_fc2.weight".into(),
                vec![32, 32],
            ),
            (
                "model.visual.deepstack_merger_list.0.linear_fc2.bias".into(),
                vec![32],
            ),
        ]);
        specs.extend(
            vision
                .into_iter()
                .map(|(name, shape)| (name, shape, Dtype::F32)),
        );
        specs
    }

    fn write_complete_qwen3_vl_safetensors_dir(
        is_moe: bool,
        mutate: impl FnOnce(&mut Vec<(String, Vec<usize>, Dtype)>),
    ) -> tempfile::TempDir {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(
            directory.path().join("config.json"),
            serde_json::to_vec(&qwen3_vl_safetensors_config(is_moe)).unwrap(),
        )
        .unwrap();
        let mut specs = qwen3_vl_safetensor_specs(is_moe);
        mutate(&mut specs);
        let payloads = specs
            .iter()
            .map(|(_, shape, dtype)| {
                let bytes = match dtype {
                    Dtype::F32 => 4,
                    Dtype::F16 | Dtype::BF16 => 2,
                    Dtype::U8 => 1,
                    other => panic!("unsupported fixture dtype {other:?}"),
                };
                vec![0xff; shape.iter().product::<usize>() * bytes]
            })
            .collect::<Vec<_>>();
        let views = specs
            .iter()
            .zip(&payloads)
            .map(|((name, shape, dtype), payload)| {
                (
                    name.as_str(),
                    TensorView::new(*dtype, shape.clone(), payload).unwrap(),
                )
            })
            .collect::<Vec<_>>();
        serialize_to_file(views, None, &directory.path().join("model.safetensors")).unwrap();
        directory
    }

    fn qwen3_moe_safetensor_specs(split_experts: bool) -> Vec<(String, Vec<usize>)> {
        let mut specs = qwen3_safetensor_specs();
        specs.retain(|(name, _)| !name.starts_with("model.layers.0.mlp."));
        specs.push(("model.layers.0.mlp.gate.weight".into(), vec![4, 32]));
        if split_experts {
            for expert in 0..4 {
                specs.extend([
                    (
                        format!("model.layers.0.mlp.experts.{expert}.gate_proj.weight"),
                        vec![8, 32],
                    ),
                    (
                        format!("model.layers.0.mlp.experts.{expert}.up_proj.weight"),
                        vec![8, 32],
                    ),
                    (
                        format!("model.layers.0.mlp.experts.{expert}.down_proj.weight"),
                        vec![32, 8],
                    ),
                ]);
            }
        } else {
            specs.extend([
                (
                    "model.layers.0.mlp.experts.gate_up_proj".into(),
                    vec![4, 16, 32],
                ),
                (
                    "model.layers.0.mlp.experts.down_proj".into(),
                    vec![4, 32, 8],
                ),
            ]);
        }
        specs
    }

    fn write_complete_qwen3_moe_safetensors_dir(
        split_experts: bool,
        mutate: impl FnOnce(&mut Vec<(String, Vec<usize>)>),
    ) -> tempfile::TempDir {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(
            directory.path().join("config.json"),
            serde_json::to_vec(&qwen3_config(true)).unwrap(),
        )
        .unwrap();
        let mut specs = qwen3_moe_safetensor_specs(split_experts);
        mutate(&mut specs);
        let payloads = specs
            .iter()
            .map(|(_, shape)| vec![0xff; shape.iter().product::<usize>() * 4])
            .collect::<Vec<_>>();
        let views = specs
            .iter()
            .zip(&payloads)
            .map(|((name, shape), payload)| {
                (
                    name.as_str(),
                    TensorView::new(Dtype::F32, shape.clone(), payload).unwrap(),
                )
            })
            .collect::<Vec<_>>();
        serialize_to_file(views, None, &directory.path().join("model.safetensors")).unwrap();
        directory
    }

    fn lfm2_config(is_moe: bool) -> Value {
        json!({
            "model_type": if is_moe { "lfm2_moe" } else { "lfm2" },
            "vocab_size": 32,
            "hidden_size": 32,
            "intermediate_size": 48,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "max_position_embeddings": 128,
            "norm_eps": 0.00001,
            "layer_types": ["conv", "full_attention"],
            "conv_L_cache": 3,
            "conv_bias": false,
            "block_auto_adjust_ff_dim": false,
            "tie_word_embeddings": true,
            "moe_intermediate_size": if is_moe { 8 } else { 0 },
            "num_dense_layers": if is_moe { 1 } else { 0 },
            "num_experts": if is_moe { 4 } else { 0 },
            "num_experts_per_tok": if is_moe { 2 } else { 0 },
            "norm_topk_prob": is_moe,
            "use_expert_bias": is_moe
        })
    }

    fn lfm2_safetensor_specs() -> Vec<(String, Vec<usize>)> {
        vec![
            ("model.embed_tokens.weight".into(), vec![32, 32]),
            ("model.embedding_norm.weight".into(), vec![32]),
            ("model.layers.0.operator_norm.weight".into(), vec![32]),
            ("model.layers.0.ffn_norm.weight".into(), vec![32]),
            ("model.layers.0.conv.conv.weight".into(), vec![32, 1, 3]),
            ("model.layers.0.conv.in_proj.weight".into(), vec![96, 32]),
            ("model.layers.0.conv.out_proj.weight".into(), vec![32, 32]),
            ("model.layers.0.feed_forward.w1.weight".into(), vec![48, 32]),
            ("model.layers.0.feed_forward.w2.weight".into(), vec![32, 48]),
            ("model.layers.0.feed_forward.w3.weight".into(), vec![48, 32]),
            ("model.layers.1.operator_norm.weight".into(), vec![32]),
            ("model.layers.1.ffn_norm.weight".into(), vec![32]),
            (
                "model.layers.1.self_attn.q_proj.weight".into(),
                vec![32, 32],
            ),
            (
                "model.layers.1.self_attn.k_proj.weight".into(),
                vec![16, 32],
            ),
            (
                "model.layers.1.self_attn.v_proj.weight".into(),
                vec![16, 32],
            ),
            (
                "model.layers.1.self_attn.out_proj.weight".into(),
                vec![32, 32],
            ),
            (
                "model.layers.1.self_attn.q_layernorm.weight".into(),
                vec![8],
            ),
            (
                "model.layers.1.self_attn.k_layernorm.weight".into(),
                vec![8],
            ),
            ("model.layers.1.feed_forward.w1.weight".into(), vec![48, 32]),
            ("model.layers.1.feed_forward.w2.weight".into(), vec![32, 48]),
            ("model.layers.1.feed_forward.w3.weight".into(), vec![48, 32]),
        ]
    }

    fn write_complete_lfm2_safetensors_dir(
        mutate: impl FnOnce(&mut Vec<(String, Vec<usize>)>),
    ) -> tempfile::TempDir {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(
            directory.path().join("config.json"),
            serde_json::to_vec(&lfm2_config(false)).unwrap(),
        )
        .unwrap();
        let mut specs = lfm2_safetensor_specs();
        mutate(&mut specs);
        let payloads = specs
            .iter()
            .map(|(_, shape)| vec![0xff; shape.iter().product::<usize>() * 4])
            .collect::<Vec<_>>();
        let views = specs
            .iter()
            .zip(&payloads)
            .map(|((name, shape), payload)| {
                (
                    name.as_str(),
                    TensorView::new(Dtype::F32, shape.clone(), payload).unwrap(),
                )
            })
            .collect::<Vec<_>>();
        serialize_to_file(views, None, &directory.path().join("model.safetensors")).unwrap();
        directory
    }

    fn lfm2_moe_safetensor_specs(split_experts: bool) -> Vec<(String, Vec<usize>)> {
        let mut specs = lfm2_safetensor_specs();
        specs.retain(|(name, _)| !name.starts_with("model.layers.1.feed_forward."));
        specs.extend([
            (
                "model.layers.1.feed_forward.gate.weight".into(),
                vec![4, 32],
            ),
            ("model.layers.1.feed_forward.expert_bias".into(), vec![4]),
        ]);
        if split_experts {
            for expert in 0..4 {
                specs.extend([
                    (
                        format!("model.layers.1.feed_forward.experts.{expert}.w1.weight"),
                        vec![8, 32],
                    ),
                    (
                        format!("model.layers.1.feed_forward.experts.{expert}.w3.weight"),
                        vec![8, 32],
                    ),
                    (
                        format!("model.layers.1.feed_forward.experts.{expert}.w2.weight"),
                        vec![32, 8],
                    ),
                ]);
            }
        } else {
            specs.extend([
                (
                    "model.layers.1.feed_forward.experts.gate_up_proj".into(),
                    vec![4, 16, 32],
                ),
                (
                    "model.layers.1.feed_forward.experts.down_proj".into(),
                    vec![4, 32, 8],
                ),
            ]);
        }
        specs
    }

    fn write_complete_lfm2_moe_safetensors_dir(
        split_experts: bool,
        mutate: impl FnOnce(&mut Vec<(String, Vec<usize>)>),
    ) -> tempfile::TempDir {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(
            directory.path().join("config.json"),
            serde_json::to_vec(&lfm2_config(true)).unwrap(),
        )
        .unwrap();
        let mut specs = lfm2_moe_safetensor_specs(split_experts);
        mutate(&mut specs);
        let payloads = specs
            .iter()
            .map(|(_, shape)| vec![0xff; shape.iter().product::<usize>() * 4])
            .collect::<Vec<_>>();
        let views = specs
            .iter()
            .zip(&payloads)
            .map(|((name, shape), payload)| {
                (
                    name.as_str(),
                    TensorView::new(Dtype::F32, shape.clone(), payload).unwrap(),
                )
            })
            .collect::<Vec<_>>();
        serialize_to_file(views, None, &directory.path().join("model.safetensors")).unwrap();
        directory
    }

    fn deepseek_v3_config(native_fp8: bool) -> Value {
        let mut config = json!({
            "architectures": ["DeepseekV3ForCausalLM"],
            "model_type": "deepseek_v3",
            "hidden_size": 8,
            "intermediate_size": 16,
            "moe_intermediate_size": 4,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "vocab_size": 32,
            "rms_norm_eps": 0.000001,
            "max_position_embeddings": 128,
            "rope_theta": 10000,
            "q_lora_rank": 4,
            "kv_lora_rank": 4,
            "qk_nope_head_dim": 2,
            "qk_rope_head_dim": 2,
            "v_head_dim": 2,
            "first_k_dense_replace": 1,
            "moe_layer_freq": 1,
            "n_routed_experts": 4,
            "n_shared_experts": 1,
            "num_experts_per_tok": 2,
            "n_group": 2,
            "topk_group": 1,
            "topk_method": "noaux_tc",
            "scoring_func": "sigmoid",
            "norm_topk_prob": true,
            "routed_scaling_factor": 1.5,
            "num_nextn_predict_layers": 0,
            "tie_word_embeddings": false,
            "attention_bias": false,
            "attention_dropout": 0.0,
            "hidden_act": "silu"
        });
        if native_fp8 {
            config["quantization_config"] = json!({
                "activation_scheme": "dynamic",
                "fmt": "e4m3",
                "quant_method": "fp8",
                "weight_block_size": [128, 128]
            });
        }
        config
    }

    fn deepseek_v3_safetensor_specs(packed_experts: bool) -> Vec<(String, Vec<usize>)> {
        let mut specs = vec![
            ("model.embed_tokens.weight".into(), vec![32, 8]),
            ("model.norm.weight".into(), vec![8]),
            ("lm_head.weight".into(), vec![32, 8]),
        ];
        for layer in 0..2 {
            let prefix = format!("model.layers.{layer}");
            specs.extend([
                (format!("{prefix}.input_layernorm.weight"), vec![8]),
                (format!("{prefix}.post_attention_layernorm.weight"), vec![8]),
                (format!("{prefix}.self_attn.q_a_proj.weight"), vec![4, 8]),
                (format!("{prefix}.self_attn.q_a_layernorm.weight"), vec![4]),
                (format!("{prefix}.self_attn.q_b_proj.weight"), vec![8, 4]),
                (
                    format!("{prefix}.self_attn.kv_a_proj_with_mqa.weight"),
                    vec![6, 8],
                ),
                (format!("{prefix}.self_attn.kv_a_layernorm.weight"), vec![4]),
                (format!("{prefix}.self_attn.kv_b_proj.weight"), vec![8, 4]),
                (format!("{prefix}.self_attn.o_proj.weight"), vec![8, 4]),
            ]);
        }
        specs.extend([
            ("model.layers.0.mlp.gate_proj.weight".into(), vec![16, 8]),
            ("model.layers.0.mlp.up_proj.weight".into(), vec![16, 8]),
            ("model.layers.0.mlp.down_proj.weight".into(), vec![8, 16]),
            ("model.layers.1.mlp.gate.weight".into(), vec![4, 8]),
            (
                "model.layers.1.mlp.gate.e_score_correction_bias".into(),
                vec![4],
            ),
            (
                "model.layers.1.mlp.shared_experts.gate_proj.weight".into(),
                vec![4, 8],
            ),
            (
                "model.layers.1.mlp.shared_experts.up_proj.weight".into(),
                vec![4, 8],
            ),
            (
                "model.layers.1.mlp.shared_experts.down_proj.weight".into(),
                vec![8, 4],
            ),
        ]);
        if packed_experts {
            specs.extend([
                ("model.layers.1.mlp.experts.gate_proj".into(), vec![4, 4, 8]),
                ("model.layers.1.mlp.experts.up_proj".into(), vec![4, 4, 8]),
                ("model.layers.1.mlp.experts.down_proj".into(), vec![4, 8, 4]),
            ]);
        } else {
            for expert in 0..4 {
                specs.extend([
                    (
                        format!("model.layers.1.mlp.experts.{expert}.gate_proj.weight"),
                        vec![4, 8],
                    ),
                    (
                        format!("model.layers.1.mlp.experts.{expert}.up_proj.weight"),
                        vec![4, 8],
                    ),
                    (
                        format!("model.layers.1.mlp.experts.{expert}.down_proj.weight"),
                        vec![8, 4],
                    ),
                ]);
            }
        }
        specs
    }

    fn write_complete_deepseek_v3_safetensors_dir(
        packed_experts: bool,
        mutate: impl FnOnce(&mut Vec<(String, Vec<usize>)>),
    ) -> tempfile::TempDir {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(
            directory.path().join("config.json"),
            serde_json::to_vec(&deepseek_v3_config(false)).unwrap(),
        )
        .unwrap();
        let mut specs = deepseek_v3_safetensor_specs(packed_experts);
        mutate(&mut specs);
        let payloads = specs
            .iter()
            .map(|(_, shape)| vec![0xff; shape.iter().product::<usize>() * 4])
            .collect::<Vec<_>>();
        let views = specs
            .iter()
            .zip(&payloads)
            .map(|((name, shape), payload)| {
                (
                    name.as_str(),
                    TensorView::new(Dtype::F32, shape.clone(), payload).unwrap(),
                )
            })
            .collect::<Vec<_>>();
        serialize_to_file(views, None, &directory.path().join("model.safetensors")).unwrap();
        directory
    }

    fn write_complete_deepseek_v3_mtp_safetensors_dir(
        mutate: impl FnOnce(&mut Vec<(String, Vec<usize>)>),
    ) -> tempfile::TempDir {
        let directory = tempfile::tempdir().unwrap();
        let mut config = deepseek_v3_config(false);
        config["num_nextn_predict_layers"] = 1.into();
        std::fs::write(
            directory.path().join("config.json"),
            serde_json::to_vec(&config).unwrap(),
        )
        .unwrap();
        let mut specs = deepseek_v3_safetensor_specs(true);
        let decoder = specs
            .iter()
            .filter_map(|(name, shape)| {
                name.strip_prefix("model.layers.1.")
                    .map(|suffix| (format!("model.layers.2.{suffix}"), shape.clone()))
            })
            .collect::<Vec<_>>();
        specs.extend(decoder);
        specs.extend([
            ("model.layers.2.enorm.weight".into(), vec![8]),
            ("model.layers.2.hnorm.weight".into(), vec![8]),
            ("model.layers.2.eh_proj.weight".into(), vec![8, 16]),
            ("model.layers.2.shared_head.norm.weight".into(), vec![8]),
            ("model.layers.2.shared_head.head.weight".into(), vec![32, 8]),
        ]);
        mutate(&mut specs);
        let payloads = specs
            .iter()
            .map(|(_, shape)| vec![0xff; shape.iter().product::<usize>() * 4])
            .collect::<Vec<_>>();
        let views = specs
            .iter()
            .zip(&payloads)
            .map(|((name, shape), payload)| {
                (
                    name.as_str(),
                    TensorView::new(Dtype::F32, shape.clone(), payload).unwrap(),
                )
            })
            .collect::<Vec<_>>();
        serialize_to_file(views, None, &directory.path().join("model.safetensors")).unwrap();
        directory
    }

    fn gpt_oss_config() -> Value {
        json!({
            "model_type": "gpt_oss",
            "hidden_size": 32,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 32,
            "vocab_size": 32,
            "num_local_experts": 2,
            "num_experts_per_tok": 1,
            "rms_norm_eps": 0.00001,
            "sliding_window": 8,
            "max_position_embeddings": 128,
            "rope_theta": 150000.0,
            "rope_scaling": null,
            "layer_types": ["sliding_attention"],
            "quantization_config": { "quant_method": "mxfp4" },
            "swiglu_limit": 7.0
        })
    }

    fn gpt_oss_safetensor_specs() -> Vec<(String, Vec<usize>, Dtype)> {
        let mut specs = vec![
            ("model.embed_tokens.weight".into(), vec![32, 32], Dtype::F32),
            ("model.norm.weight".into(), vec![32], Dtype::F32),
            ("lm_head.weight".into(), vec![32, 32], Dtype::F32),
            (
                "model.layers.0.input_layernorm.weight".into(),
                vec![32],
                Dtype::F32,
            ),
            (
                "model.layers.0.post_attention_layernorm.weight".into(),
                vec![32],
                Dtype::F32,
            ),
            ("model.layers.0.self_attn.sinks".into(), vec![1], Dtype::F32),
        ];
        for projection in ["q", "k", "v", "o"] {
            specs.extend([
                (
                    format!("model.layers.0.self_attn.{projection}_proj.weight"),
                    vec![32, 32],
                    Dtype::F32,
                ),
                (
                    format!("model.layers.0.self_attn.{projection}_proj.bias"),
                    vec![32],
                    Dtype::F32,
                ),
            ]);
        }
        specs.extend([
            (
                "model.layers.0.mlp.router.weight".into(),
                vec![2, 32],
                Dtype::F32,
            ),
            ("model.layers.0.mlp.router.bias".into(), vec![2], Dtype::F32),
            (
                "model.layers.0.mlp.experts.gate_up_proj_blocks".into(),
                vec![2, 64, 1, 16],
                Dtype::U8,
            ),
            (
                "model.layers.0.mlp.experts.gate_up_proj_scales".into(),
                vec![2, 64, 1],
                Dtype::U8,
            ),
            (
                "model.layers.0.mlp.experts.gate_up_proj_bias".into(),
                vec![2, 64],
                Dtype::F32,
            ),
            (
                "model.layers.0.mlp.experts.down_proj_blocks".into(),
                vec![2, 32, 1, 16],
                Dtype::U8,
            ),
            (
                "model.layers.0.mlp.experts.down_proj_scales".into(),
                vec![2, 32, 1],
                Dtype::U8,
            ),
            (
                "model.layers.0.mlp.experts.down_proj_bias".into(),
                vec![2, 32],
                Dtype::F32,
            ),
        ]);
        specs
    }

    fn write_complete_gpt_oss_safetensors_dir(
        mutate: impl FnOnce(&mut Vec<(String, Vec<usize>, Dtype)>),
    ) -> tempfile::TempDir {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(
            directory.path().join("config.json"),
            serde_json::to_vec(&gpt_oss_config()).unwrap(),
        )
        .unwrap();
        let mut specs = gpt_oss_safetensor_specs();
        mutate(&mut specs);
        let payloads = specs
            .iter()
            .map(|(_, shape, dtype)| {
                let bytes = match dtype {
                    Dtype::U8 => 1,
                    Dtype::F32 => 4,
                    other => panic!("unsupported test dtype {other:?}"),
                };
                vec![0xff; shape.iter().product::<usize>() * bytes]
            })
            .collect::<Vec<_>>();
        let views = specs
            .iter()
            .zip(&payloads)
            .map(|((name, shape, dtype), payload)| {
                (
                    name.as_str(),
                    TensorView::new(*dtype, shape.clone(), payload).unwrap(),
                )
            })
            .collect::<Vec<_>>();
        serialize_to_file(views, None, &directory.path().join("model.safetensors")).unwrap();
        directory
    }

    fn kimi_linear_config() -> Value {
        json!({
            "model_type": "kimi_linear",
            "vocab_size": 32,
            "hidden_size": 8,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "intermediate_size": 16,
            "head_dim": 4,
            "model_max_length": 128,
            "rms_norm_eps": 0.00001,
            "rope_theta": 10000.0,
            "linear_attn_config": {
                "kda_layers": [1],
                "full_attn_layers": [2],
                "num_heads": 2,
                "head_dim": 4,
                "short_conv_kernel_size": 2
            },
            "num_experts": 4,
            "moe_intermediate_size": 8,
            "kv_lora_rank": 4,
            "q_lora_rank": null,
            "qk_nope_head_dim": 2,
            "qk_rope_head_dim": 2,
            "v_head_dim": 2,
            "mla_use_nope": true,
            "num_experts_per_token": 2,
            "num_shared_experts": 1,
            "moe_router_activation_func": "sigmoid",
            "moe_renormalize": true,
            "routed_scaling_factor": 1.0,
            "first_k_dense_replace": 1,
            "moe_layer_freq": 1,
            "use_grouped_topk": true,
            "num_expert_group": 1,
            "topk_group": 1,
            "tie_word_embeddings": false,
            "num_nextn_predict_layers": 0
        })
    }

    fn kimi_linear_safetensor_specs(packed_experts: bool) -> Vec<(String, Vec<usize>)> {
        let mut specs = vec![
            ("model.embed_tokens.weight".into(), vec![32, 8]),
            ("model.norm.weight".into(), vec![8]),
            ("lm_head.weight".into(), vec![32, 8]),
            ("model.layers.0.input_layernorm.weight".into(), vec![8]),
            (
                "model.layers.0.post_attention_layernorm.weight".into(),
                vec![8],
            ),
            ("model.layers.0.self_attn.q_proj.weight".into(), vec![8, 8]),
            ("model.layers.0.self_attn.k_proj.weight".into(), vec![8, 8]),
            ("model.layers.0.self_attn.v_proj.weight".into(), vec![8, 8]),
            (
                "model.layers.0.self_attn.q_conv1d.weight".into(),
                vec![8, 2],
            ),
            (
                "model.layers.0.self_attn.k_conv1d.weight".into(),
                vec![8, 2],
            ),
            (
                "model.layers.0.self_attn.v_conv1d.weight".into(),
                vec![8, 2],
            ),
            (
                "model.layers.0.self_attn.f_a_proj.weight".into(),
                vec![4, 8],
            ),
            (
                "model.layers.0.self_attn.f_b_proj.weight".into(),
                vec![8, 4],
            ),
            ("model.layers.0.self_attn.b_proj.weight".into(), vec![2, 8]),
            (
                "model.layers.0.self_attn.g_a_proj.weight".into(),
                vec![4, 8],
            ),
            (
                "model.layers.0.self_attn.g_b_proj.weight".into(),
                vec![8, 4],
            ),
            ("model.layers.0.self_attn.A_log".into(), vec![2]),
            ("model.layers.0.self_attn.dt_bias".into(), vec![8]),
            ("model.layers.0.self_attn.o_norm.weight".into(), vec![4]),
            ("model.layers.0.self_attn.o_proj.weight".into(), vec![8, 8]),
            ("model.layers.0.mlp.gate_proj.weight".into(), vec![16, 8]),
            ("model.layers.0.mlp.up_proj.weight".into(), vec![16, 8]),
            ("model.layers.0.mlp.down_proj.weight".into(), vec![8, 16]),
            ("model.layers.1.input_layernorm.weight".into(), vec![8]),
            (
                "model.layers.1.post_attention_layernorm.weight".into(),
                vec![8],
            ),
            ("model.layers.1.self_attn.q_proj.weight".into(), vec![8, 8]),
            (
                "model.layers.1.self_attn.kv_a_proj_with_mqa.weight".into(),
                vec![6, 8],
            ),
            (
                "model.layers.1.self_attn.kv_a_layernorm.weight".into(),
                vec![4],
            ),
            (
                "model.layers.1.self_attn.kv_b_proj.weight".into(),
                vec![8, 4],
            ),
            ("model.layers.1.self_attn.o_proj.weight".into(), vec![8, 4]),
            (
                "model.layers.1.block_sparse_moe.gate.weight".into(),
                vec![4, 8],
            ),
            (
                "model.layers.1.block_sparse_moe.gate.e_score_correction_bias".into(),
                vec![4],
            ),
            (
                "model.layers.1.block_sparse_moe.shared_experts.gate_proj.weight".into(),
                vec![8, 8],
            ),
            (
                "model.layers.1.block_sparse_moe.shared_experts.up_proj.weight".into(),
                vec![8, 8],
            ),
            (
                "model.layers.1.block_sparse_moe.shared_experts.down_proj.weight".into(),
                vec![8, 8],
            ),
        ];
        if packed_experts {
            specs.extend([
                (
                    "model.layers.1.block_sparse_moe.experts.gate_up_proj".into(),
                    vec![4, 16, 8],
                ),
                (
                    "model.layers.1.block_sparse_moe.experts.down_proj".into(),
                    vec![4, 8, 8],
                ),
            ]);
        } else {
            for expert in 0..4 {
                for (projection, shape) in
                    [("w1", vec![8, 8]), ("w2", vec![8, 8]), ("w3", vec![8, 8])]
                {
                    specs.push((
                        format!(
                            "model.layers.1.block_sparse_moe.experts.{expert}.{projection}.weight"
                        ),
                        shape,
                    ));
                }
            }
        }
        specs
    }

    fn write_complete_kimi_linear_safetensors_dir(
        packed_experts: bool,
        mutate: impl FnOnce(&mut Vec<(String, Vec<usize>)>),
    ) -> tempfile::TempDir {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(
            directory.path().join("config.json"),
            serde_json::to_vec(&kimi_linear_config()).unwrap(),
        )
        .unwrap();
        let mut specs = kimi_linear_safetensor_specs(packed_experts);
        mutate(&mut specs);
        let payloads = specs
            .iter()
            .map(|(_, shape)| vec![0xff; shape.iter().product::<usize>() * 4])
            .collect::<Vec<_>>();
        let views = specs
            .iter()
            .zip(&payloads)
            .map(|((name, shape), payload)| {
                (
                    name.as_str(),
                    TensorView::new(Dtype::F32, shape.clone(), payload).unwrap(),
                )
            })
            .collect::<Vec<_>>();
        serialize_to_file(views, None, &directory.path().join("model.safetensors")).unwrap();
        directory
    }

    fn kimi_linear_gguf_metadata() -> BTreeMap<String, MetadataValue> {
        BTreeMap::from([
            (
                "general.architecture".into(),
                MetadataValue::String("kimi-linear".into()),
            ),
            ("general.file_type".into(), MetadataValue::Uint32(0)),
            ("kimi-linear.block_count".into(), MetadataValue::Uint32(2)),
            (
                "kimi-linear.embedding_length".into(),
                MetadataValue::Uint32(8),
            ),
            (
                "kimi-linear.attention.head_count".into(),
                MetadataValue::Uint32(2),
            ),
            (
                "kimi-linear.attention.head_count_kv".into(),
                MetadataValue::Array(MetadataArray::Uint32(vec![0, 1])),
            ),
            (
                "kimi-linear.rope.dimension_count".into(),
                MetadataValue::Uint32(2),
            ),
            (
                "kimi-linear.attention.key_length_mla".into(),
                MetadataValue::Uint32(4),
            ),
            ("kimi-linear.vocab_size".into(), MetadataValue::Uint32(32)),
            (
                "kimi-linear.feed_forward_length".into(),
                MetadataValue::Uint32(16),
            ),
            (
                "kimi-linear.context_length".into(),
                MetadataValue::Uint32(128),
            ),
            (
                "kimi-linear.attention.layer_norm_rms_epsilon".into(),
                MetadataValue::Float32(0.00001),
            ),
            ("kimi-linear.kda.head_dim".into(), MetadataValue::Uint32(4)),
            (
                "kimi-linear.ssm.conv_kernel".into(),
                MetadataValue::Uint32(2),
            ),
            ("kimi-linear.expert_count".into(), MetadataValue::Uint32(4)),
            (
                "kimi-linear.expert_feed_forward_length".into(),
                MetadataValue::Uint32(8),
            ),
            (
                "kimi-linear.attention.kv_lora_rank".into(),
                MetadataValue::Uint32(4),
            ),
            (
                "kimi-linear.attention.value_length_mla".into(),
                MetadataValue::Uint32(2),
            ),
            (
                "kimi-linear.leading_dense_block_count".into(),
                MetadataValue::Uint32(1),
            ),
            (
                "kimi-linear.expert_used_count".into(),
                MetadataValue::Uint32(2),
            ),
        ])
    }

    fn kimi_linear_gguf_specs() -> Vec<(String, Vec<u64>, GgmlType)> {
        let matrix = |name: &str, mlx: &[u64]| {
            let mut dimensions = mlx.to_vec();
            dimensions.reverse();
            (name.into(), dimensions, GgmlType::F32)
        };
        let mut specs = vec![
            matrix("token_embd.weight", &[32, 8]),
            matrix("output_norm.weight", &[8]),
            matrix("output.weight", &[32, 8]),
        ];
        for layer in 0..2 {
            specs.extend([
                matrix(&format!("blk.{layer}.attn_norm.weight"), &[8]),
                matrix(&format!("blk.{layer}.ffn_norm.weight"), &[8]),
            ]);
        }
        specs.extend([
            matrix("blk.0.attn_q.weight", &[8, 8]),
            matrix("blk.0.attn_k.weight", &[8, 8]),
            matrix("blk.0.attn_v.weight", &[8, 8]),
            matrix("blk.0.ssm_conv1d_q.weight", &[8, 2]),
            matrix("blk.0.ssm_conv1d_k.weight", &[8, 2]),
            matrix("blk.0.ssm_conv1d_v.weight", &[8, 2]),
            matrix("blk.0.ssm_f_a.weight", &[4, 8]),
            matrix("blk.0.ssm_f_b.weight", &[8, 4]),
            matrix("blk.0.ssm_beta.weight", &[2, 8]),
            matrix("blk.0.ssm_g_a.weight", &[4, 8]),
            matrix("blk.0.ssm_g_b.weight", &[8, 4]),
            matrix("blk.0.ssm_a", &[2]),
            matrix("blk.0.ssm_dt.bias", &[8]),
            matrix("blk.0.ssm_norm.weight", &[4]),
            matrix("blk.0.attn_output.weight", &[8, 8]),
            matrix("blk.0.ffn_gate.weight", &[16, 8]),
            matrix("blk.0.ffn_up.weight", &[16, 8]),
            matrix("blk.0.ffn_down.weight", &[8, 16]),
            matrix("blk.1.attn_q.weight", &[8, 8]),
            matrix("blk.1.attn_kv_a_mqa.weight", &[6, 8]),
            matrix("blk.1.attn_kv_a_norm.weight", &[4]),
            matrix("blk.1.attn_kv_b.weight", &[8, 4]),
            matrix("blk.1.attn_output.weight", &[8, 4]),
            matrix("blk.1.ffn_gate_inp.weight", &[4, 8]),
            matrix("blk.1.exp_probs_b.bias", &[4]),
            matrix("blk.1.ffn_gate_shexp.weight", &[8, 8]),
            matrix("blk.1.ffn_up_shexp.weight", &[8, 8]),
            matrix("blk.1.ffn_down_shexp.weight", &[8, 8]),
            matrix("blk.1.ffn_gate_exps.weight", &[4, 8, 8]),
            matrix("blk.1.ffn_up_exps.weight", &[4, 8, 8]),
            matrix("blk.1.ffn_down_exps.weight", &[4, 8, 8]),
        ]);
        specs
    }

    fn write_complete_kimi_linear_gguf(path: &Path) {
        let specs = kimi_linear_gguf_specs();
        let payloads = specs
            .iter()
            .map(|(_, dimensions, encoding)| {
                let elements = dimensions.iter().product::<u64>();
                let (block, bytes) = encoding.block_and_bytes().unwrap();
                vec![0xff; (elements / block * bytes) as usize]
            })
            .collect::<Vec<_>>();
        let tensors = specs
            .iter()
            .zip(&payloads)
            .map(|((name, dimensions, ggml_type), data)| TensorInput {
                name,
                dimensions,
                ggml_type: *ggml_type,
                data,
            })
            .collect::<Vec<_>>();
        Writer::default()
            .write(
                std::fs::File::create(path).unwrap(),
                &kimi_linear_gguf_metadata(),
                &tensors,
            )
            .unwrap();
    }

    fn inkling_config() -> Value {
        json!({
            "model_type": "inkling_mm_model",
            "eos_token_id": 1,
            "text_config": {
                "torch_dtype": "float32",
                "hidden_size": 16,
                "num_hidden_layers": 2,
                "vocab_size": 32,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 8,
                "sliding_window_size": 4,
                "local_layer_ids": [0],
                "dense_mlp_idx": 1,
                "sconv_kernel_size": 3,
                "d_rel": 4,
                "rel_extent": 8,
                "intermediate_size": 8,
                "dense_intermediate_size": 16,
                "moe_intermediate_size": 8,
                "n_routed_experts": 2,
                "num_experts_per_tok": 1,
                "n_shared_experts": 1,
                "route_scale": 1.0,
                "use_sconv": true,
                "use_embed_norm": true,
                "shared_expert_sink": true,
                "use_gate_bias": true,
                "norm_after_topk": true,
                "use_global_scale": true,
                "gate_activation": "sigmoid",
                "hidden_act": "silu",
                "attention_dropout": 0.0,
                "q_bias": false,
                "o_bias": false,
                "logits_mup_width_multiplier": 2.0,
                "unpadded_vocab_size": 30
            }
        })
    }

    fn inkling_safetensor_specs() -> Vec<(String, Vec<usize>)> {
        let mut specs = vec![
            ("model.llm.embed.weight".into(), vec![32, 16]),
            ("model.llm.embed_norm.weight".into(), vec![16]),
            ("model.llm.norm.weight".into(), vec![16]),
            ("model.llm.unembed.weight".into(), vec![32, 16]),
        ];
        for layer in 0..2 {
            let prefix = format!("model.llm.layers.{layer}");
            let relative = if layer == 0 { 4 } else { 8 };
            specs.extend([
                (format!("{prefix}.attn_norm.weight"), vec![16]),
                (format!("{prefix}.mlp_norm.weight"), vec![16]),
                (format!("{prefix}.attn.wq_du.weight"), vec![16, 16]),
                (format!("{prefix}.attn.wk_dv.weight"), vec![8, 16]),
                (format!("{prefix}.attn.wv_dv.weight"), vec![8, 16]),
                (format!("{prefix}.attn.wr_du.weight"), vec![8, 16]),
                (format!("{prefix}.attn.wo_ud.weight"), vec![16, 16]),
                (format!("{prefix}.attn.q_norm.weight"), vec![8]),
                (format!("{prefix}.attn.k_norm.weight"), vec![8]),
                (
                    format!("{prefix}.attn.rel_logits_proj.proj"),
                    vec![4, relative],
                ),
                (format!("{prefix}.attn.k_sconv.weight"), vec![8, 1, 3]),
                (format!("{prefix}.attn.v_sconv.weight"), vec![8, 1, 3]),
                (format!("{prefix}.attn_sconv.weight"), vec![16, 1, 3]),
                (format!("{prefix}.mlp_sconv.weight"), vec![16, 1, 3]),
            ]);
        }
        specs.extend([
            ("model.llm.layers.0.mlp.w13_dn.weight".into(), vec![32, 16]),
            ("model.llm.layers.0.mlp.w2_md.weight".into(), vec![16, 16]),
            ("model.llm.layers.0.mlp.global_scale".into(), vec![1]),
            ("model.llm.layers.1.mlp.gate.weight".into(), vec![3, 16]),
            ("model.llm.layers.1.mlp.gate.bias".into(), vec![2]),
            ("model.llm.layers.1.mlp.gate.global_scale".into(), vec![1]),
            (
                "model.llm.layers.1.mlp.experts.w13_weight".into(),
                vec![2, 16, 16],
            ),
            (
                "model.llm.layers.1.mlp.experts.w2_weight".into(),
                vec![2, 16, 8],
            ),
            (
                "model.llm.layers.1.mlp.shared_experts.shared_w13_weight".into(),
                vec![1, 16, 16],
            ),
            (
                "model.llm.layers.1.mlp.shared_experts.shared_w2_weight".into(),
                vec![1, 16, 8],
            ),
        ]);
        specs
    }

    fn write_complete_inkling_safetensors_dir(
        mutate: impl FnOnce(&mut Vec<(String, Vec<usize>)>),
    ) -> tempfile::TempDir {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(
            directory.path().join("config.json"),
            serde_json::to_vec(&inkling_config()).unwrap(),
        )
        .unwrap();
        let mut specs = inkling_safetensor_specs();
        mutate(&mut specs);
        let payloads = specs
            .iter()
            .map(|(_, shape)| vec![0xff; shape.iter().product::<usize>() * 4])
            .collect::<Vec<_>>();
        let views = specs
            .iter()
            .zip(&payloads)
            .map(|((name, shape), payload)| {
                (
                    name.as_str(),
                    TensorView::new(Dtype::F32, shape.clone(), payload).unwrap(),
                )
            })
            .collect::<Vec<_>>();
        serialize_to_file(views, None, &directory.path().join("model.safetensors")).unwrap();
        directory
    }

    fn nemotron_h_config() -> Value {
        json!({
            "model_type": "nemotron_h",
            "vocab_size": 16,
            "hidden_size": 8,
            "intermediate_size": 12,
            "num_hidden_layers": 4,
            "hybrid_override_pattern": "M-E*",
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "max_position_embeddings": 64,
            "mamba_num_heads": 2,
            "mamba_head_dim": 4,
            "n_groups": 1,
            "ssm_state_size": 4,
            "conv_kernel": 3,
            "chunk_size": 2,
            "moe_intermediate_size": 6,
            "moe_shared_expert_intermediate_size": 10,
            "n_routed_experts": 2,
            "n_shared_experts": 1,
            "num_experts_per_tok": 2,
            "tie_word_embeddings": false,
            "torch_dtype": "float32"
        })
    }

    fn nemotron_h_safetensor_specs(packed_experts: bool) -> Vec<(String, Vec<usize>)> {
        let mut specs = vec![
            ("backbone.embeddings.weight".into(), vec![16, 8]),
            ("backbone.norm_f.weight".into(), vec![8]),
            ("lm_head.weight".into(), vec![16, 8]),
        ];
        for layer in 0..4 {
            specs.push((format!("backbone.layers.{layer}.norm.weight"), vec![8]));
        }
        specs.extend([
            ("backbone.layers.0.mixer.in_proj.weight".into(), vec![26, 8]),
            (
                "backbone.layers.0.mixer.conv1d.weight".into(),
                vec![16, 1, 3],
            ),
            ("backbone.layers.0.mixer.conv1d.bias".into(), vec![16]),
            ("backbone.layers.0.mixer.dt_bias".into(), vec![2]),
            ("backbone.layers.0.mixer.A_log".into(), vec![2]),
            ("backbone.layers.0.mixer.D".into(), vec![2]),
            ("backbone.layers.0.mixer.norm.weight".into(), vec![8]),
            ("backbone.layers.0.mixer.out_proj.weight".into(), vec![8, 8]),
            ("backbone.layers.1.mixer.up_proj.weight".into(), vec![12, 8]),
            (
                "backbone.layers.1.mixer.down_proj.weight".into(),
                vec![8, 12],
            ),
            ("backbone.layers.2.mixer.gate.weight".into(), vec![2, 8]),
            (
                "backbone.layers.2.mixer.gate.e_score_correction_bias".into(),
                vec![2],
            ),
            (
                "backbone.layers.2.mixer.shared_experts.up_proj.weight".into(),
                vec![10, 8],
            ),
            (
                "backbone.layers.2.mixer.shared_experts.down_proj.weight".into(),
                vec![8, 10],
            ),
            ("backbone.layers.3.mixer.q_proj.weight".into(), vec![8, 8]),
            ("backbone.layers.3.mixer.k_proj.weight".into(), vec![4, 8]),
            ("backbone.layers.3.mixer.v_proj.weight".into(), vec![4, 8]),
            ("backbone.layers.3.mixer.o_proj.weight".into(), vec![8, 8]),
        ]);
        if packed_experts {
            specs.extend([
                (
                    "backbone.layers.2.mixer.experts.up_proj".into(),
                    vec![2, 6, 8],
                ),
                (
                    "backbone.layers.2.mixer.experts.down_proj".into(),
                    vec![2, 8, 6],
                ),
            ]);
        } else {
            for expert in 0..2 {
                specs.extend([
                    (
                        format!("backbone.layers.2.mixer.experts.{expert}.up_proj.weight"),
                        vec![6, 8],
                    ),
                    (
                        format!("backbone.layers.2.mixer.experts.{expert}.down_proj.weight"),
                        vec![8, 6],
                    ),
                ]);
            }
        }
        specs
    }

    fn write_complete_nemotron_h_safetensors_dir(
        packed_experts: bool,
        mutate: impl FnOnce(&mut Vec<(String, Vec<usize>)>),
    ) -> tempfile::TempDir {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(
            directory.path().join("config.json"),
            serde_json::to_vec(&nemotron_h_config()).unwrap(),
        )
        .unwrap();
        let mut specs = nemotron_h_safetensor_specs(packed_experts);
        mutate(&mut specs);
        let payloads = specs
            .iter()
            .map(|(_, shape)| vec![0xff; shape.iter().product::<usize>() * 4])
            .collect::<Vec<_>>();
        let views = specs
            .iter()
            .zip(&payloads)
            .map(|((name, shape), payload)| {
                (
                    name.as_str(),
                    TensorView::new(Dtype::F32, shape.clone(), payload).unwrap(),
                )
            })
            .collect::<Vec<_>>();
        serialize_to_file(views, None, &directory.path().join("model.safetensors")).unwrap();
        directory
    }

    fn gemma4_config(is_moe: bool) -> Value {
        let mut config = json!({
            "model_type": "gemma4",
            "tie_word_embeddings": true,
            "text_config": {
                "model_type": "gemma4",
                "hidden_size": 8,
                "num_hidden_layers": 2,
                "intermediate_size": 16,
                "use_double_wide_mlp": true,
                "num_attention_heads": 2,
                "rms_norm_eps": 0.000001,
                "vocab_size": 32,
                "pad_token_id": 0,
                "num_key_value_heads": 1,
                "num_global_key_value_heads": 2,
                "max_position_embeddings": 128,
                "rope_theta": 10000.0,
                "head_dim": 4,
                "global_head_dim": 4,
                "attention_k_eq_v": true,
                "hidden_size_per_layer_input": 2,
                "vocab_size_per_layer_input": 16,
                "num_kv_shared_layers": 1,
                "layer_types": ["sliding_attention", "full_attention"],
                "sliding_window": 8,
                "enable_moe_block": is_moe
            }
        });
        if is_moe {
            config["text_config"]["num_experts"] = json!(2);
            config["text_config"]["top_k_experts"] = json!(1);
            config["text_config"]["moe_intermediate_size"] = json!(6);
        }
        config
    }

    fn gemma4_safetensor_specs(is_moe: bool) -> Vec<(String, Vec<usize>)> {
        let released =
            |name: &str| name.replacen("model.language_model.", "language_model.model.", 1);
        let mut specs = vec![
            (
                released("model.language_model.embed_tokens.weight"),
                vec![32, 8],
            ),
            (released("model.language_model.norm.weight"), vec![8]),
            (
                released("model.language_model.embed_tokens_per_layer.weight"),
                vec![16, 4],
            ),
            (
                released("model.language_model.per_layer_model_projection.weight"),
                vec![4, 8],
            ),
            (
                released("model.language_model.per_layer_projection_norm.weight"),
                vec![2],
            ),
        ];
        for layer in 0..2 {
            let prefix = format!("model.language_model.layers.{layer}");
            for name in [
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
                "pre_feedforward_layernorm.weight",
                "post_feedforward_layernorm.weight",
            ] {
                specs.push((released(&format!("{prefix}.{name}")), vec![8]));
            }
            specs.push((released(&format!("{prefix}.layer_scalar")), vec![1]));
            specs.extend([
                (
                    released(&format!("{prefix}.self_attn.q_proj.weight")),
                    vec![8, 8],
                ),
                (
                    released(&format!("{prefix}.self_attn.o_proj.weight")),
                    vec![8, 8],
                ),
                (
                    released(&format!("{prefix}.self_attn.q_norm.weight")),
                    vec![4],
                ),
            ]);
            if layer == 0 {
                specs.extend([
                    (
                        released(&format!("{prefix}.self_attn.k_proj.weight")),
                        vec![4, 8],
                    ),
                    (
                        released(&format!("{prefix}.self_attn.v_proj.weight")),
                        vec![4, 8],
                    ),
                    (
                        released(&format!("{prefix}.self_attn.k_norm.weight")),
                        vec![4],
                    ),
                ]);
            }
            let intermediate = if layer == 0 { 16 } else { 32 };
            specs.extend([
                (
                    released(&format!("{prefix}.mlp.gate_proj.weight")),
                    vec![intermediate, 8],
                ),
                (
                    released(&format!("{prefix}.mlp.up_proj.weight")),
                    vec![intermediate, 8],
                ),
                (
                    released(&format!("{prefix}.mlp.down_proj.weight")),
                    vec![8, intermediate],
                ),
                (
                    released(&format!("{prefix}.per_layer_input_gate.weight")),
                    vec![2, 8],
                ),
                (
                    released(&format!("{prefix}.per_layer_projection.weight")),
                    vec![8, 2],
                ),
                (
                    released(&format!("{prefix}.post_per_layer_input_norm.weight")),
                    vec![8],
                ),
            ]);
            if is_moe {
                specs.extend([
                    (
                        released(&format!("{prefix}.router.proj.weight")),
                        vec![2, 8],
                    ),
                    (released(&format!("{prefix}.router.scale")), vec![8]),
                    (
                        released(&format!("{prefix}.router.per_expert_scale")),
                        vec![2],
                    ),
                    (
                        released(&format!("{prefix}.post_feedforward_layernorm_1.weight")),
                        vec![8],
                    ),
                    (
                        released(&format!("{prefix}.pre_feedforward_layernorm_2.weight")),
                        vec![8],
                    ),
                    (
                        released(&format!("{prefix}.post_feedforward_layernorm_2.weight")),
                        vec![8],
                    ),
                    (
                        released(&format!("{prefix}.experts.switch_glu.gate_proj.weight")),
                        vec![2, 6, 8],
                    ),
                    (
                        released(&format!("{prefix}.experts.switch_glu.up_proj.weight")),
                        vec![2, 6, 8],
                    ),
                    (
                        released(&format!("{prefix}.experts.switch_glu.down_proj.weight")),
                        vec![2, 8, 6],
                    ),
                ]);
            }
        }
        specs
    }

    fn append_gemma4_clipped_specs(
        specs: &mut Vec<(String, Vec<usize>)>,
        prefix: &str,
        shape: Vec<usize>,
    ) {
        specs.push((format!("{prefix}.linear.weight"), shape));
        for suffix in ["input_min", "input_max", "output_min", "output_max"] {
            specs.push((format!("{prefix}.{suffix}"), vec![]));
        }
    }

    fn gemma4_multimodal_specs() -> Vec<(String, Vec<usize>)> {
        let mut specs = gemma4_safetensor_specs(false);
        let vision = "model.vision_tower";
        specs.extend([
            (
                format!("{vision}.patch_embedder.input_proj.weight"),
                vec![8, 12],
            ),
            (
                format!("{vision}.patch_embedder.position_embedding_table"),
                vec![2, 8, 8],
            ),
        ]);
        let vision_layer = format!("{vision}.encoder.layers.0");
        for name in [
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "pre_feedforward_layernorm.weight",
            "post_feedforward_layernorm.weight",
        ] {
            specs.push((format!("{vision_layer}.{name}"), vec![8]));
        }
        for (name, shape) in [
            ("self_attn.q_proj", vec![8, 8]),
            ("self_attn.k_proj", vec![8, 8]),
            ("self_attn.v_proj", vec![8, 8]),
            ("self_attn.o_proj", vec![8, 8]),
            ("mlp.gate_proj", vec![16, 8]),
            ("mlp.up_proj", vec![16, 8]),
            ("mlp.down_proj", vec![8, 16]),
        ] {
            append_gemma4_clipped_specs(&mut specs, &format!("{vision_layer}.{name}"), shape);
        }
        for name in ["self_attn.q_norm.weight", "self_attn.k_norm.weight"] {
            specs.push((format!("{vision_layer}.{name}"), vec![4]));
        }
        specs.push((
            "model.embed_vision.embedding_projection.weight".into(),
            vec![8, 8],
        ));

        let audio = "model.audio_tower";
        specs.extend([
            (
                format!("{audio}.subsample_conv_projection.layer0.conv.weight"),
                vec![2, 3, 3, 1],
            ),
            (
                format!("{audio}.subsample_conv_projection.layer0.norm.weight"),
                vec![2],
            ),
            (
                format!("{audio}.subsample_conv_projection.layer1.conv.weight"),
                vec![4, 3, 3, 2],
            ),
            (
                format!("{audio}.subsample_conv_projection.layer1.norm.weight"),
                vec![4],
            ),
            (
                format!("{audio}.subsample_conv_projection.input_proj_linear.weight"),
                vec![8, 128],
            ),
            (format!("{audio}.output_proj.weight"), vec![8, 8]),
            (format!("{audio}.output_proj.bias"), vec![8]),
        ]);
        let audio_layer = format!("{audio}.layers.0");
        for name in [
            "feed_forward1.pre_layer_norm.weight",
            "feed_forward1.post_layer_norm.weight",
            "norm_pre_attn.weight",
            "norm_post_attn.weight",
            "lconv1d.pre_layer_norm.weight",
            "lconv1d.conv_norm.weight",
            "feed_forward2.pre_layer_norm.weight",
            "feed_forward2.post_layer_norm.weight",
            "norm_out.weight",
        ] {
            specs.push((format!("{audio_layer}.{name}"), vec![8]));
        }
        for (name, shape) in [
            ("feed_forward1.ffw_layer_1", vec![32, 8]),
            ("feed_forward1.ffw_layer_2", vec![8, 32]),
            ("self_attn.q_proj", vec![8, 8]),
            ("self_attn.k_proj", vec![8, 8]),
            ("self_attn.v_proj", vec![8, 8]),
            ("self_attn.post", vec![8, 8]),
            ("lconv1d.linear_start", vec![16, 8]),
            ("lconv1d.linear_end", vec![8, 8]),
            ("feed_forward2.ffw_layer_1", vec![32, 8]),
            ("feed_forward2.ffw_layer_2", vec![8, 32]),
        ] {
            append_gemma4_clipped_specs(&mut specs, &format!("{audio_layer}.{name}"), shape);
        }
        specs.extend([
            (
                format!("{audio_layer}.self_attn.relative_k_proj.weight"),
                vec![8, 8],
            ),
            (format!("{audio_layer}.self_attn.per_dim_scale"), vec![4]),
            (
                format!("{audio_layer}.lconv1d.depthwise_conv1d.weight"),
                vec![8, 3, 1],
            ),
            (
                "model.embed_audio.embedding_projection.weight".into(),
                vec![8, 8],
            ),
        ]);
        specs
    }

    fn write_complete_gemma4_safetensors_dir(
        is_moe: bool,
        mutate: impl FnOnce(&mut Vec<(String, Vec<usize>)>),
    ) -> tempfile::TempDir {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(
            directory.path().join("config.json"),
            serde_json::to_vec(&gemma4_config(is_moe)).unwrap(),
        )
        .unwrap();
        let mut specs = gemma4_safetensor_specs(is_moe);
        mutate(&mut specs);
        let payloads = specs
            .iter()
            .map(|(_, shape)| vec![0xff; shape.iter().product::<usize>() * 4])
            .collect::<Vec<_>>();
        let views = specs
            .iter()
            .zip(&payloads)
            .map(|((name, shape), payload)| {
                (
                    name.as_str(),
                    TensorView::new(Dtype::F32, shape.clone(), payload).unwrap(),
                )
            })
            .collect::<Vec<_>>();
        serialize_to_file(views, None, &directory.path().join("model.safetensors")).unwrap();
        directory
    }

    fn write_complete_quantized_gemma4_safetensors_dir(
        mutate: impl FnOnce(&mut Vec<(String, Vec<usize>, Dtype)>),
    ) -> tempfile::TempDir {
        let directory = tempfile::tempdir().unwrap();
        let config = json!({
            "model_type": "gemma4",
            "tie_word_embeddings": true,
            "quantization": {"group_size": 32, "bits": 4, "mode": "affine"},
            "text_config": {
                "hidden_size": 32,
                "num_hidden_layers": 1,
                "intermediate_size": 32,
                "num_attention_heads": 1,
                "rms_norm_eps": 0.000001,
                "vocab_size": 32,
                "num_key_value_heads": 1,
                "max_position_embeddings": 128,
                "head_dim": 32,
                "layer_types": ["full_attention"]
            }
        });
        std::fs::write(
            directory.path().join("config.json"),
            serde_json::to_vec(&config).unwrap(),
        )
        .unwrap();
        let mut specs = Vec::new();
        let mut matrix = |name: &str| {
            specs.extend([
                (name.into(), vec![32, 4], Dtype::U32),
                (
                    name.trim_end_matches(".weight").to_string() + ".scales",
                    vec![32, 1],
                    Dtype::F32,
                ),
                (
                    name.trim_end_matches(".weight").to_string() + ".biases",
                    vec![32, 1],
                    Dtype::F32,
                ),
            ]);
        };
        matrix("language_model.model.embed_tokens.weight");
        for projection in ["q_proj", "k_proj", "v_proj", "o_proj"] {
            matrix(&format!(
                "language_model.model.layers.0.self_attn.{projection}.weight"
            ));
        }
        for projection in ["gate_proj", "up_proj", "down_proj"] {
            matrix(&format!(
                "language_model.model.layers.0.mlp.{projection}.weight"
            ));
        }
        specs.extend([
            (
                "language_model.model.norm.weight".into(),
                vec![32],
                Dtype::F32,
            ),
            (
                "language_model.model.layers.0.input_layernorm.weight".into(),
                vec![32],
                Dtype::F32,
            ),
            (
                "language_model.model.layers.0.post_attention_layernorm.weight".into(),
                vec![32],
                Dtype::F32,
            ),
            (
                "language_model.model.layers.0.pre_feedforward_layernorm.weight".into(),
                vec![32],
                Dtype::F32,
            ),
            (
                "language_model.model.layers.0.post_feedforward_layernorm.weight".into(),
                vec![32],
                Dtype::F32,
            ),
            (
                "language_model.model.layers.0.self_attn.q_norm.weight".into(),
                vec![32],
                Dtype::F32,
            ),
            (
                "language_model.model.layers.0.self_attn.k_norm.weight".into(),
                vec![32],
                Dtype::F32,
            ),
            (
                "language_model.model.layers.0.layer_scalar".into(),
                vec![1],
                Dtype::F32,
            ),
        ]);
        mutate(&mut specs);
        let payloads = specs
            .iter()
            .map(|(_, shape, _)| vec![0xff; shape.iter().product::<usize>() * 4])
            .collect::<Vec<_>>();
        let views = specs
            .iter()
            .zip(&payloads)
            .map(|((name, shape, dtype), payload)| {
                (
                    name.as_str(),
                    TensorView::new(*dtype, shape.clone(), payload).unwrap(),
                )
            })
            .collect::<Vec<_>>();
        serialize_to_file(views, None, &directory.path().join("model.safetensors")).unwrap();
        directory
    }

    fn qwen3_next_config() -> Value {
        json!({
            "model_type": "qwen3_next",
            "vocab_size": 32,
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "mtp_num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "max_position_embeddings": 64,
            "rms_norm_eps": 0.00001,
            "tie_word_embeddings": false,
            "linear_conv_kernel_dim": 3,
            "linear_key_head_dim": 4,
            "linear_value_head_dim": 4,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "intermediate_size": 0,
            "moe_intermediate_size": 8,
            "shared_expert_intermediate_size": 8,
            "num_experts_per_tok": 1,
            "num_experts": 2,
            "norm_topk_prob": true,
            "layer_types": ["linear_attention", "full_attention"]
        })
    }

    fn append_qwen3_next_block_specs(
        specs: &mut Vec<(String, Vec<usize>)>,
        prefix: &str,
        linear: bool,
        packed_experts: bool,
    ) {
        specs.extend([
            (format!("{prefix}.input_layernorm.weight"), vec![16]),
            (
                format!("{prefix}.post_attention_layernorm.weight"),
                vec![16],
            ),
        ]);
        if linear {
            specs.extend([
                (
                    format!("{prefix}.linear_attn.in_proj_qkvz.weight"),
                    vec![48, 16],
                ),
                (
                    format!("{prefix}.linear_attn.in_proj_ba.weight"),
                    vec![8, 16],
                ),
                (
                    format!("{prefix}.linear_attn.conv1d.weight"),
                    vec![32, 1, 3],
                ),
                (format!("{prefix}.linear_attn.dt_bias"), vec![4]),
                (format!("{prefix}.linear_attn.A_log"), vec![4]),
                (format!("{prefix}.linear_attn.norm.weight"), vec![4]),
                (
                    format!("{prefix}.linear_attn.out_proj.weight"),
                    vec![16, 16],
                ),
            ]);
        } else {
            specs.extend([
                (format!("{prefix}.self_attn.q_proj.weight"), vec![32, 16]),
                (format!("{prefix}.self_attn.k_proj.weight"), vec![8, 16]),
                (format!("{prefix}.self_attn.v_proj.weight"), vec![8, 16]),
                (format!("{prefix}.self_attn.o_proj.weight"), vec![16, 16]),
                (format!("{prefix}.self_attn.q_norm.weight"), vec![8]),
                (format!("{prefix}.self_attn.k_norm.weight"), vec![8]),
            ]);
        }
        specs.extend([
            (format!("{prefix}.mlp.gate.weight"), vec![2, 16]),
            (
                format!("{prefix}.mlp.shared_expert.gate_proj.weight"),
                vec![8, 16],
            ),
            (
                format!("{prefix}.mlp.shared_expert.up_proj.weight"),
                vec![8, 16],
            ),
            (
                format!("{prefix}.mlp.shared_expert.down_proj.weight"),
                vec![16, 8],
            ),
            (
                format!("{prefix}.mlp.shared_expert_gate.weight"),
                vec![1, 16],
            ),
        ]);
        if packed_experts {
            specs.extend([
                (
                    format!("{prefix}.mlp.experts.gate_up_proj"),
                    vec![2, 16, 16],
                ),
                (format!("{prefix}.mlp.experts.down_proj"), vec![2, 16, 8]),
            ]);
        } else {
            for expert in 0..2 {
                specs.extend([
                    (
                        format!("{prefix}.mlp.experts.{expert}.gate_proj.weight"),
                        vec![8, 16],
                    ),
                    (
                        format!("{prefix}.mlp.experts.{expert}.up_proj.weight"),
                        vec![8, 16],
                    ),
                    (
                        format!("{prefix}.mlp.experts.{expert}.down_proj.weight"),
                        vec![16, 8],
                    ),
                ]);
            }
        }
    }

    fn qwen3_next_safetensor_specs(packed_experts: bool) -> Vec<(String, Vec<usize>)> {
        let mut specs = vec![
            ("model.embed_tokens.weight".into(), vec![32, 16]),
            ("model.norm.weight".into(), vec![16]),
            ("lm_head.weight".into(), vec![32, 16]),
            ("mtp.pre_fc_norm_hidden.weight".into(), vec![16]),
            ("mtp.pre_fc_norm_embedding.weight".into(), vec![16]),
            ("mtp.fc.weight".into(), vec![16, 32]),
            ("mtp.norm.weight".into(), vec![16]),
        ];
        append_qwen3_next_block_specs(&mut specs, "model.layers.0", true, packed_experts);
        append_qwen3_next_block_specs(&mut specs, "model.layers.1", false, packed_experts);
        append_qwen3_next_block_specs(&mut specs, "mtp.layers.0", false, packed_experts);
        specs
    }

    fn write_complete_qwen3_next_safetensors_dir(
        packed_experts: bool,
        mutate: impl FnOnce(&mut Vec<(String, Vec<usize>)>),
    ) -> tempfile::TempDir {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(
            directory.path().join("config.json"),
            serde_json::to_vec(&qwen3_next_config()).unwrap(),
        )
        .unwrap();
        let mut specs = qwen3_next_safetensor_specs(packed_experts);
        mutate(&mut specs);
        let payloads = specs
            .iter()
            .map(|(_, shape)| vec![0xff; shape.iter().product::<usize>() * 4])
            .collect::<Vec<_>>();
        let views = specs
            .iter()
            .zip(&payloads)
            .map(|((name, shape), payload)| {
                (
                    name.as_str(),
                    TensorView::new(Dtype::F32, shape.clone(), payload).unwrap(),
                )
            })
            .collect::<Vec<_>>();
        serialize_to_file(views, None, &directory.path().join("model.safetensors")).unwrap();
        directory
    }

    fn qwen35_config(moe: bool) -> Value {
        let mut config = qwen3_next_config();
        config["model_type"] = json!(if moe {
            "qwen3_5_moe_text"
        } else {
            "qwen3_5_text"
        });
        config["intermediate_size"] = json!(if moe { 0 } else { 32 });
        config["moe_intermediate_size"] = json!(if moe { 8 } else { 0 });
        config["shared_expert_intermediate_size"] = json!(if moe { 8 } else { 0 });
        config["num_experts_per_tok"] = json!(if moe { 1 } else { 0 });
        config["num_experts"] = json!(if moe { 2 } else { 0 });
        config["norm_topk_prob"] = json!(moe);
        config
    }

    fn qwen35_safetensor_specs(moe: bool, packed_experts: bool) -> Vec<(String, Vec<usize>)> {
        let mut specs = qwen3_next_safetensor_specs(packed_experts);
        specs.retain(|(name, _)| {
            !name.ends_with("linear_attn.in_proj_qkvz.weight")
                && !name.ends_with("linear_attn.in_proj_ba.weight")
        });
        specs.extend([
            (
                "model.layers.0.linear_attn.in_proj_qkv.weight".into(),
                vec![32, 16],
            ),
            (
                "model.layers.0.linear_attn.in_proj_z.weight".into(),
                vec![16, 16],
            ),
            (
                "model.layers.0.linear_attn.in_proj_b.weight".into(),
                vec![4, 16],
            ),
            (
                "model.layers.0.linear_attn.in_proj_a.weight".into(),
                vec![4, 16],
            ),
        ]);
        if !moe {
            specs.retain(|(name, _)| !name.contains(".mlp."));
            for prefix in ["model.layers.0", "model.layers.1", "mtp.layers.0"] {
                specs.extend([
                    (format!("{prefix}.mlp.gate_proj.weight"), vec![32, 16]),
                    (format!("{prefix}.mlp.up_proj.weight"), vec![32, 16]),
                    (format!("{prefix}.mlp.down_proj.weight"), vec![16, 32]),
                ]);
            }
        }
        specs
    }

    fn write_complete_qwen35_safetensors_dir(
        moe: bool,
        packed_experts: bool,
        mutate: impl FnOnce(&mut Vec<(String, Vec<usize>)>),
    ) -> tempfile::TempDir {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(
            directory.path().join("config.json"),
            serde_json::to_vec(&qwen35_config(moe)).unwrap(),
        )
        .unwrap();
        let mut specs = qwen35_safetensor_specs(moe, packed_experts);
        mutate(&mut specs);
        let payloads = specs
            .iter()
            .map(|(_, shape)| vec![0xff; shape.iter().product::<usize>() * 4])
            .collect::<Vec<_>>();
        let views = specs
            .iter()
            .zip(&payloads)
            .map(|((name, shape), payload)| {
                (
                    name.as_str(),
                    TensorView::new(Dtype::F32, shape.clone(), payload).unwrap(),
                )
            })
            .collect::<Vec<_>>();
        serialize_to_file(views, None, &directory.path().join("model.safetensors")).unwrap();
        directory
    }

    fn write_gguf(
        path: &Path,
        architecture: &str,
        encoding: GgmlType,
        extra: impl IntoIterator<Item = (String, MetadataValue)>,
    ) {
        let mut metadata = BTreeMap::from([
            (
                "general.architecture".into(),
                MetadataValue::String(architecture.into()),
            ),
            ("general.file_type".into(), MetadataValue::Uint32(0)),
            (
                format!("{architecture}.block_count"),
                MetadataValue::Uint32(1),
            ),
            (
                format!("{architecture}.embedding_length"),
                MetadataValue::Uint32(8),
            ),
        ]);
        metadata.extend(extra);
        let (block, bytes) = encoding.block_and_bytes().unwrap();
        let payload = vec![0xff; bytes as usize];
        Writer::default()
            .write(
                std::fs::File::create(path).unwrap(),
                &metadata,
                &[TensorInput {
                    name: "token_embd.weight",
                    dimensions: &[block],
                    ggml_type: encoding,
                    data: &payload,
                }],
            )
            .unwrap();
    }

    fn llama_gguf_metadata() -> BTreeMap<String, MetadataValue> {
        BTreeMap::from([
            (
                "general.architecture".into(),
                MetadataValue::String("llama".into()),
            ),
            ("general.file_type".into(), MetadataValue::Uint32(0)),
            ("llama.block_count".into(), MetadataValue::Uint32(1)),
            ("llama.embedding_length".into(), MetadataValue::Uint32(32)),
            (
                "llama.attention.head_count".into(),
                MetadataValue::Uint32(4),
            ),
            (
                "llama.attention.head_count_kv".into(),
                MetadataValue::Uint32(2),
            ),
            (
                "llama.attention.key_length".into(),
                MetadataValue::Uint32(8),
            ),
            (
                "llama.feed_forward_length".into(),
                MetadataValue::Uint32(64),
            ),
            (
                "llama.attention.layer_norm_rms_epsilon".into(),
                MetadataValue::Float32(0.00001),
            ),
            ("llama.context_length".into(), MetadataValue::Uint32(128)),
            ("llama.vocab_size".into(), MetadataValue::Uint32(32)),
        ])
    }

    fn llama_gguf_specs() -> Vec<(String, Vec<u64>, GgmlType)> {
        vec![
            ("token_embd.weight".into(), vec![32, 32], GgmlType::F32),
            ("output_norm.weight".into(), vec![32], GgmlType::F32),
            ("blk.0.attn_norm.weight".into(), vec![32], GgmlType::F32),
            ("blk.0.ffn_norm.weight".into(), vec![32], GgmlType::F32),
            ("blk.0.attn_q.weight".into(), vec![32, 32], GgmlType::F32),
            ("blk.0.attn_k.weight".into(), vec![32, 16], GgmlType::F32),
            ("blk.0.attn_v.weight".into(), vec![32, 16], GgmlType::F32),
            (
                "blk.0.attn_output.weight".into(),
                vec![32, 32],
                GgmlType::F32,
            ),
            ("blk.0.ffn_gate.weight".into(), vec![32, 64], GgmlType::F32),
            ("blk.0.ffn_up.weight".into(), vec![32, 64], GgmlType::F32),
            ("blk.0.ffn_down.weight".into(), vec![64, 32], GgmlType::F32),
        ]
    }

    fn write_complete_gguf(
        path: &Path,
        mutate: impl FnOnce(&mut Vec<(String, Vec<u64>, GgmlType)>),
    ) {
        let mut specs = llama_gguf_specs();
        mutate(&mut specs);
        let payloads = specs
            .iter()
            .map(|(_, dimensions, encoding)| {
                let elements = dimensions.iter().product::<u64>();
                let (block, bytes) = encoding.block_and_bytes().unwrap();
                vec![0xff; (elements / block * bytes) as usize]
            })
            .collect::<Vec<_>>();
        let tensors = specs
            .iter()
            .zip(&payloads)
            .map(|((name, dimensions, ggml_type), data)| TensorInput {
                name,
                dimensions,
                ggml_type: *ggml_type,
                data,
            })
            .collect::<Vec<_>>();
        Writer::default()
            .write(
                std::fs::File::create(path).unwrap(),
                &llama_gguf_metadata(),
                &tensors,
            )
            .unwrap();
    }

    fn qwen3_gguf_metadata(is_moe: bool) -> BTreeMap<String, MetadataValue> {
        let architecture = if is_moe { "qwen3moe" } else { "qwen3" };
        let mut metadata = BTreeMap::from([
            (
                "general.architecture".into(),
                MetadataValue::String(architecture.into()),
            ),
            ("general.file_type".into(), MetadataValue::Uint32(0)),
            (
                format!("{architecture}.block_count"),
                MetadataValue::Uint32(1),
            ),
            (
                format!("{architecture}.embedding_length"),
                MetadataValue::Uint32(32),
            ),
            (
                format!("{architecture}.attention.head_count"),
                MetadataValue::Uint32(1),
            ),
            (
                format!("{architecture}.attention.head_count_kv"),
                MetadataValue::Uint32(1),
            ),
            (
                format!("{architecture}.attention.key_length"),
                MetadataValue::Uint32(32),
            ),
            (
                format!("{architecture}.attention.layer_norm_rms_epsilon"),
                MetadataValue::Float32(0.000001),
            ),
            (
                format!("{architecture}.context_length"),
                MetadataValue::Uint32(128),
            ),
            (
                format!("{architecture}.vocab_size"),
                MetadataValue::Uint32(32),
            ),
        ]);
        if is_moe {
            metadata.extend([
                (
                    format!("{architecture}.expert_feed_forward_length"),
                    MetadataValue::Uint32(8),
                ),
                (
                    format!("{architecture}.expert_count"),
                    MetadataValue::Uint32(4),
                ),
                (
                    format!("{architecture}.expert_used_count"),
                    MetadataValue::Uint32(2),
                ),
            ]);
        } else {
            metadata.insert(
                format!("{architecture}.feed_forward_length"),
                MetadataValue::Uint32(64),
            );
        }
        metadata
    }

    fn gemma4_gguf_metadata(is_moe: bool) -> BTreeMap<String, MetadataValue> {
        let mut metadata = BTreeMap::from([
            (
                "general.architecture".into(),
                MetadataValue::String("gemma4".into()),
            ),
            ("general.file_type".into(), MetadataValue::Uint32(0)),
            ("gemma4.block_count".into(), MetadataValue::Uint32(3)),
            ("gemma4.embedding_length".into(), MetadataValue::Uint32(8)),
            (
                "gemma4.feed_forward_length".into(),
                MetadataValue::Array(MetadataArray::Uint32(vec![12, 16, 16])),
            ),
            (
                "gemma4.attention.head_count".into(),
                MetadataValue::Uint32(2),
            ),
            (
                "gemma4.attention.head_count_kv".into(),
                MetadataValue::Array(MetadataArray::Uint32(vec![1, 2, 2])),
            ),
            (
                "gemma4.attention.key_length".into(),
                MetadataValue::Uint32(4),
            ),
            (
                "gemma4.attention.key_length_swa".into(),
                MetadataValue::Uint32(2),
            ),
            (
                "gemma4.attention.sliding_window_pattern".into(),
                MetadataValue::Array(MetadataArray::Bool(vec![true, false, false])),
            ),
            (
                "gemma4.attention.shared_kv_layers".into(),
                MetadataValue::Uint32(1),
            ),
            (
                "gemma4.attention.layer_norm_rms_epsilon".into(),
                MetadataValue::Float32(0.00001),
            ),
            (
                "gemma4.attention.sliding_window".into(),
                MetadataValue::Uint32(8),
            ),
            ("gemma4.context_length".into(), MetadataValue::Uint32(64)),
            ("gemma4.vocab_size".into(), MetadataValue::Uint32(16)),
        ]);
        if is_moe {
            metadata.extend([
                ("gemma4.expert_count".into(), MetadataValue::Uint32(2)),
                ("gemma4.expert_used_count".into(), MetadataValue::Uint32(1)),
                (
                    "gemma4.expert_feed_forward_length".into(),
                    MetadataValue::Uint32(4),
                ),
            ]);
        }
        metadata
    }

    fn gemma4_gguf_specs(is_moe: bool) -> Vec<(String, Vec<u64>, GgmlType)> {
        let mut specs = vec![
            ("token_embd.weight".into(), vec![8, 16], GgmlType::F32),
            ("output_norm.weight".into(), vec![8], GgmlType::F32),
            ("output.weight".into(), vec![8, 16], GgmlType::F32),
        ];
        for layer in 0..3 {
            let prefix = format!("blk.{layer}");
            let head_dim = if layer == 0 { 2 } else { 4 };
            let query = 2 * head_dim;
            let key_value = if layer == 0 { 2 } else { 8 };
            let intermediate = if layer == 0 { 12 } else { 16 };
            for name in [
                "attn_norm",
                "post_attention_norm",
                "ffn_norm",
                "post_ffw_norm",
            ] {
                specs.push((format!("{prefix}.{name}.weight"), vec![8], GgmlType::F32));
            }
            specs.extend([
                (
                    format!("{prefix}.layer_output_scale.weight"),
                    vec![1],
                    GgmlType::F32,
                ),
                (
                    format!("{prefix}.attn_q.weight"),
                    vec![8, query],
                    GgmlType::F32,
                ),
                (
                    format!("{prefix}.attn_output.weight"),
                    vec![query, 8],
                    GgmlType::F32,
                ),
                (
                    format!("{prefix}.attn_q_norm.weight"),
                    vec![head_dim],
                    GgmlType::F32,
                ),
                (
                    format!("{prefix}.ffn_gate.weight"),
                    vec![8, intermediate],
                    GgmlType::F32,
                ),
                (
                    format!("{prefix}.ffn_up.weight"),
                    vec![8, intermediate],
                    GgmlType::F32,
                ),
                (
                    format!("{prefix}.ffn_down.weight"),
                    vec![intermediate, 8],
                    GgmlType::F32,
                ),
            ]);
            if layer < 2 {
                specs.extend([
                    (
                        format!("{prefix}.attn_k.weight"),
                        vec![8, key_value],
                        GgmlType::F32,
                    ),
                    (
                        format!("{prefix}.attn_v.weight"),
                        vec![8, key_value],
                        GgmlType::F32,
                    ),
                    (
                        format!("{prefix}.attn_k_norm.weight"),
                        vec![head_dim],
                        GgmlType::F32,
                    ),
                ]);
            }
            if is_moe {
                specs.extend([
                    (
                        format!("{prefix}.ffn_gate_inp.weight"),
                        vec![8, 2],
                        GgmlType::F32,
                    ),
                    (
                        format!("{prefix}.ffn_gate_inp.scale"),
                        vec![8],
                        GgmlType::F32,
                    ),
                    (
                        format!("{prefix}.ffn_down_exps.scale"),
                        vec![2],
                        GgmlType::F32,
                    ),
                    (
                        format!("{prefix}.post_ffw_norm_1.weight"),
                        vec![8],
                        GgmlType::F32,
                    ),
                    (
                        format!("{prefix}.pre_ffw_norm_2.weight"),
                        vec![8],
                        GgmlType::F32,
                    ),
                    (
                        format!("{prefix}.post_ffw_norm_2.weight"),
                        vec![8],
                        GgmlType::F32,
                    ),
                    (
                        format!("{prefix}.ffn_down_exps.weight"),
                        vec![4, 8, 2],
                        GgmlType::F32,
                    ),
                ]);
                if layer == 1 {
                    specs.extend([
                        (
                            format!("{prefix}.ffn_gate_exps.weight"),
                            vec![8, 4, 2],
                            GgmlType::F32,
                        ),
                        (
                            format!("{prefix}.ffn_up_exps.weight"),
                            vec![8, 4, 2],
                            GgmlType::F32,
                        ),
                    ]);
                } else {
                    specs.push((
                        format!("{prefix}.ffn_gate_up_exps.weight"),
                        vec![8, 8, 2],
                        GgmlType::F32,
                    ));
                }
            }
        }
        specs
    }

    fn write_complete_gemma4_gguf(
        path: &Path,
        is_moe: bool,
        mutate: impl FnOnce(&mut Vec<(String, Vec<u64>, GgmlType)>),
    ) {
        let mut specs = gemma4_gguf_specs(is_moe);
        mutate(&mut specs);
        let payloads = specs
            .iter()
            .map(|(_, dimensions, encoding)| {
                let elements = dimensions.iter().product::<u64>();
                let (block, bytes) = encoding.block_and_bytes().unwrap();
                vec![0xff; (elements / block * bytes) as usize]
            })
            .collect::<Vec<_>>();
        let tensors = specs
            .iter()
            .zip(&payloads)
            .map(|((name, dimensions, ggml_type), data)| TensorInput {
                name,
                dimensions,
                ggml_type: *ggml_type,
                data,
            })
            .collect::<Vec<_>>();
        Writer::default()
            .write(
                std::fs::File::create(path).unwrap(),
                &gemma4_gguf_metadata(is_moe),
                &tensors,
            )
            .unwrap();
    }

    fn write_quantizable_gemma4_moe_gguf(path: &Path) {
        let mut metadata = gemma4_gguf_metadata(true);
        metadata.insert("gemma4.embedding_length".into(), MetadataValue::Uint32(32));
        metadata.insert(
            "gemma4.feed_forward_length".into(),
            MetadataValue::Array(MetadataArray::Uint32(vec![32, 32, 32])),
        );
        metadata.insert(
            "gemma4.attention.key_length".into(),
            MetadataValue::Uint32(16),
        );
        metadata.insert(
            "gemma4.attention.key_length_swa".into(),
            MetadataValue::Uint32(16),
        );
        metadata.insert(
            "gemma4.expert_feed_forward_length".into(),
            MetadataValue::Uint32(32),
        );
        let mut specs = gemma4_gguf_specs(true);
        for (name, dimensions, _) in &mut specs {
            *dimensions = if name.ends_with("layer_output_scale.weight") {
                vec![1]
            } else if name.ends_with("ffn_down_exps.scale") {
                vec![2]
            } else if name.ends_with("attn_q_norm.weight") || name.ends_with("attn_k_norm.weight") {
                vec![16]
            } else if name.ends_with("ffn_gate_inp.weight") {
                vec![32, 2]
            } else if name.ends_with("ffn_gate_up_exps.weight") {
                vec![32, 64, 2]
            } else if name.contains("_exps.weight") {
                vec![32, 32, 2]
            } else if name.ends_with("attn_k.weight") || name.ends_with("attn_v.weight") {
                let output = if name.starts_with("blk.0.") { 16 } else { 32 };
                vec![32, output]
            } else if name == "token_embd.weight" || name == "output.weight" {
                vec![32, 16]
            } else if dimensions.len() == 2 {
                vec![32, 32]
            } else {
                vec![32]
            };
        }
        let payloads = specs
            .iter()
            .map(|(_, dimensions, _)| vec![0u8; dimensions.iter().product::<u64>() as usize * 4])
            .collect::<Vec<_>>();
        let tensors = specs
            .iter()
            .zip(&payloads)
            .map(|((name, dimensions, ggml_type), data)| TensorInput {
                name,
                dimensions,
                ggml_type: *ggml_type,
                data,
            })
            .collect::<Vec<_>>();
        Writer::default()
            .write(std::fs::File::create(path).unwrap(), &metadata, &tensors)
            .unwrap();
    }

    fn inkling_gguf_metadata() -> BTreeMap<String, MetadataValue> {
        BTreeMap::from([
            (
                "general.architecture".into(),
                MetadataValue::String("inkling".into()),
            ),
            ("general.file_type".into(), MetadataValue::Uint32(0)),
            ("inkling.block_count".into(), MetadataValue::Uint32(2)),
            ("inkling.embedding_length".into(), MetadataValue::Uint32(32)),
            (
                "inkling.feed_forward_length".into(),
                MetadataValue::Uint32(64),
            ),
            (
                "inkling.expert_feed_forward_length".into(),
                MetadataValue::Uint32(32),
            ),
            (
                "inkling.attention.head_count".into(),
                MetadataValue::Uint32(4),
            ),
            (
                "inkling.attention.head_count_kv".into(),
                MetadataValue::Array(MetadataArray::Uint32(vec![2, 1])),
            ),
            (
                "inkling.attention.key_length".into(),
                MetadataValue::Uint32(8),
            ),
            (
                "inkling.attention.sliding_window".into(),
                MetadataValue::Uint32(8),
            ),
            (
                "inkling.attention.sliding_window_pattern".into(),
                MetadataValue::Array(MetadataArray::Bool(vec![true, false])),
            ),
            (
                "inkling.attention.layer_norm_rms_epsilon".into(),
                MetadataValue::Float32(0.000001),
            ),
            ("inkling.context_length".into(), MetadataValue::Uint32(128)),
            ("inkling.vocab_size".into(), MetadataValue::Uint32(64)),
            ("inkling.d_rel".into(), MetadataValue::Uint32(4)),
            ("inkling.rel_extent".into(), MetadataValue::Uint32(16)),
            ("inkling.shortconv_kernel".into(), MetadataValue::Uint32(4)),
            ("inkling.dense_block_count".into(), MetadataValue::Uint32(1)),
            ("inkling.expert_count".into(), MetadataValue::Uint32(4)),
            ("inkling.expert_used_count".into(), MetadataValue::Uint32(2)),
            (
                "inkling.expert_shared_count".into(),
                MetadataValue::Uint32(1),
            ),
            ("inkling.audio_token_id".into(), MetadataValue::Uint32(62)),
            ("inkling.image_token_id".into(), MetadataValue::Uint32(63)),
        ])
    }

    fn inkling_gguf_specs() -> Vec<(String, Vec<u64>, GgmlType)> {
        let mut specs = vec![
            ("token_embd.weight".into(), vec![32, 64], GgmlType::F32),
            ("token_embd_norm.weight".into(), vec![32], GgmlType::F32),
            ("output_norm.weight".into(), vec![32], GgmlType::F32),
            ("output.weight".into(), vec![32, 64], GgmlType::F32),
        ];
        for layer in 0..2 {
            let prefix = format!("blk.{layer}");
            let kv = if layer == 0 { 16 } else { 8 };
            let relative = if layer == 0 { 8 } else { 16 };
            specs.extend([
                (
                    format!("{prefix}.attn_norm.weight"),
                    vec![32],
                    GgmlType::F32,
                ),
                (format!("{prefix}.ffn_norm.weight"), vec![32], GgmlType::F32),
                (
                    format!("{prefix}.attn_q.weight"),
                    vec![32, 32],
                    GgmlType::F32,
                ),
                (
                    format!("{prefix}.attn_k.weight"),
                    vec![32, kv],
                    GgmlType::F32,
                ),
                (
                    format!("{prefix}.attn_v.weight"),
                    vec![32, kv],
                    GgmlType::F32,
                ),
                (
                    format!("{prefix}.attn_r.weight"),
                    vec![32, 16],
                    GgmlType::F32,
                ),
                (
                    format!("{prefix}.attn_output.weight"),
                    vec![32, 32],
                    GgmlType::F32,
                ),
                (
                    format!("{prefix}.attn_q_norm.weight"),
                    vec![8],
                    GgmlType::F32,
                ),
                (
                    format!("{prefix}.attn_k_norm.weight"),
                    vec![8],
                    GgmlType::F32,
                ),
                (
                    format!("{prefix}.attn_rel_proj.weight"),
                    vec![relative, 4],
                    GgmlType::F32,
                ),
                (
                    format!("{prefix}.shortconv_k.weight"),
                    vec![4, kv],
                    GgmlType::F32,
                ),
                (
                    format!("{prefix}.shortconv_v.weight"),
                    vec![4, kv],
                    GgmlType::F32,
                ),
                (
                    format!("{prefix}.shortconv_attn.weight"),
                    vec![4, 32],
                    GgmlType::F32,
                ),
                (
                    format!("{prefix}.shortconv_mlp.weight"),
                    vec![4, 32],
                    GgmlType::F32,
                ),
            ]);
        }
        specs.extend([
            ("blk.0.ffn_gate.weight".into(), vec![32, 64], GgmlType::F32),
            ("blk.0.ffn_up.weight".into(), vec![32, 64], GgmlType::F32),
            ("blk.0.ffn_down.weight".into(), vec![64, 32], GgmlType::F32),
            ("blk.0.ffn_gscale".into(), vec![1], GgmlType::F32),
            (
                "blk.1.ffn_gate_inp.weight".into(),
                vec![32, 5],
                GgmlType::F32,
            ),
            ("blk.1.ffn_exp_probs_b.bias".into(), vec![4], GgmlType::F32),
            ("blk.1.ffn_gscale".into(), vec![1], GgmlType::F32),
            (
                "blk.1.ffn_gate_exps.weight".into(),
                vec![32, 32, 4],
                GgmlType::F32,
            ),
            (
                "blk.1.ffn_up_exps.weight".into(),
                vec![32, 32, 4],
                GgmlType::F32,
            ),
            (
                "blk.1.ffn_down_exps.weight".into(),
                vec![32, 32, 4],
                GgmlType::F32,
            ),
            (
                "blk.1.ffn_gate_shexp.weight".into(),
                vec![32, 32, 1],
                GgmlType::F32,
            ),
            (
                "blk.1.ffn_up_shexp.weight".into(),
                vec![32, 32, 1],
                GgmlType::F32,
            ),
            (
                "blk.1.ffn_down_shexp.weight".into(),
                vec![32, 32, 1],
                GgmlType::F32,
            ),
        ]);
        specs
    }

    fn inkling_mmproj_metadata() -> BTreeMap<String, MetadataValue> {
        BTreeMap::from([
            (
                "general.architecture".into(),
                MetadataValue::String("clip".into()),
            ),
            ("general.file_type".into(), MetadataValue::Uint32(2)),
            ("clip.has_vision_encoder".into(), MetadataValue::Bool(true)),
            ("clip.has_audio_encoder".into(), MetadataValue::Bool(true)),
            (
                "clip.vision.projector_type".into(),
                MetadataValue::String("inkling".into()),
            ),
            (
                "clip.audio.projector_type".into(),
                MetadataValue::String("inkling".into()),
            ),
            (
                "clip.vision.projection_dim".into(),
                MetadataValue::Uint32(32),
            ),
            ("clip.vision.image_size".into(), MetadataValue::Uint32(40)),
            ("clip.vision.patch_size".into(), MetadataValue::Uint32(40)),
            (
                "clip.vision.embedding_length".into(),
                MetadataValue::Uint32(3),
            ),
            ("clip.vision.block_count".into(), MetadataValue::Uint32(4)),
            (
                "clip.audio.projection_dim".into(),
                MetadataValue::Uint32(32),
            ),
            (
                "clip.audio.embedding_length".into(),
                MetadataValue::Uint32(32),
            ),
            ("clip.audio.num_mel_bins".into(), MetadataValue::Uint32(80)),
        ])
    }

    fn inkling_mmproj_specs() -> Vec<(String, Vec<u64>, GgmlType)> {
        vec![
            (
                "a.dmel.embedding.weight".into(),
                vec![32, 1280],
                GgmlType::Q4_0,
            ),
            ("a.dmel.final_norm.weight".into(), vec![32], GgmlType::F32),
            (
                "v.hmlp.0.linear.weight".into(),
                vec![75, 128],
                GgmlType::F32,
            ),
            ("v.hmlp.0.norm.weight".into(), vec![128], GgmlType::F32),
            (
                "v.hmlp.1.linear.weight".into(),
                vec![512, 512],
                GgmlType::Q4_0,
            ),
            ("v.hmlp.1.norm.weight".into(), vec![512], GgmlType::F32),
            (
                "v.hmlp.2.linear.weight".into(),
                vec![8192, 4800],
                GgmlType::Q4_0,
            ),
            ("v.hmlp.2.norm.weight".into(), vec![4800], GgmlType::F32),
            (
                "v.hmlp.3.linear.weight".into(),
                vec![9600, 32],
                GgmlType::Q4_0,
            ),
            ("v.hmlp.final_norm.weight".into(), vec![32], GgmlType::F32),
        ]
    }

    fn write_inkling_gguf(
        path: &Path,
        metadata: &BTreeMap<String, MetadataValue>,
        mut specs: Vec<(String, Vec<u64>, GgmlType)>,
        mutate: impl FnOnce(&mut Vec<(String, Vec<u64>, GgmlType)>),
    ) {
        mutate(&mut specs);
        let payloads = specs
            .iter()
            .map(|(_, dimensions, encoding)| {
                let elements = dimensions.iter().product::<u64>();
                let (block, bytes) = encoding.block_and_bytes().unwrap();
                assert_eq!(elements % block, 0);
                vec![0xff; (elements / block * bytes) as usize]
            })
            .collect::<Vec<_>>();
        let tensors = specs
            .iter()
            .zip(&payloads)
            .map(|((name, dimensions, ggml_type), data)| TensorInput {
                name,
                dimensions,
                ggml_type: *ggml_type,
                data,
            })
            .collect::<Vec<_>>();
        Writer::default()
            .write(std::fs::File::create(path).unwrap(), metadata, &tensors)
            .unwrap();
    }

    fn nemotron_h_gguf_metadata(is_moe: bool) -> BTreeMap<String, MetadataValue> {
        let architecture = if is_moe {
            "nemotron_h_moe"
        } else {
            "nemotron_h"
        };
        let key = |suffix: &str| format!("{architecture}.{suffix}");
        let mut metadata = BTreeMap::from([
            (
                "general.architecture".into(),
                MetadataValue::String(architecture.into()),
            ),
            ("general.file_type".into(), MetadataValue::Uint32(0)),
            (key("block_count"), MetadataValue::Uint32(3)),
            (key("embedding_length"), MetadataValue::Uint32(8)),
            (key("attention.head_count"), MetadataValue::Uint32(2)),
            (
                key("attention.head_count_kv"),
                MetadataValue::Array(MetadataArray::Uint32(vec![0, 0, 1])),
            ),
            (key("attention.key_length"), MetadataValue::Uint32(4)),
            (
                key("attention.layer_norm_rms_epsilon"),
                MetadataValue::Float32(0.00001),
            ),
            (key("context_length"), MetadataValue::Uint32(64)),
            (key("vocab_size"), MetadataValue::Uint32(16)),
            (
                key("feed_forward_length"),
                MetadataValue::Array(MetadataArray::Uint32(vec![0, 12, 0])),
            ),
            (key("ssm.inner_size"), MetadataValue::Uint32(8)),
            (key("ssm.time_step_rank"), MetadataValue::Uint32(2)),
            (key("ssm.state_size"), MetadataValue::Uint32(4)),
            (key("ssm.group_count"), MetadataValue::Uint32(1)),
            (key("ssm.conv_kernel"), MetadataValue::Uint32(32)),
        ]);
        if is_moe {
            metadata.extend([
                (key("expert_count"), MetadataValue::Uint32(2)),
                (key("expert_shared_count"), MetadataValue::Uint32(1)),
                (key("expert_feed_forward_length"), MetadataValue::Uint32(6)),
                (
                    key("expert_shared_feed_forward_length"),
                    MetadataValue::Uint32(10),
                ),
                (key("expert_used_count"), MetadataValue::Uint32(2)),
            ]);
        }
        metadata
    }

    fn nemotron_h_gguf_specs(is_moe: bool) -> Vec<(String, Vec<u64>, GgmlType)> {
        let mut specs = vec![
            ("token_embd.weight".into(), vec![8, 16], GgmlType::F32),
            ("output_norm.weight".into(), vec![8], GgmlType::F32),
            ("output.weight".into(), vec![8, 16], GgmlType::F32),
        ];
        for layer in 0..3 {
            specs.push((
                format!("blk.{layer}.attn_norm.weight"),
                vec![8],
                GgmlType::F32,
            ));
        }
        specs.extend([
            ("blk.0.ssm_in.weight".into(), vec![8, 26], GgmlType::F32),
            (
                "blk.0.ssm_conv1d.weight".into(),
                vec![32, 16],
                GgmlType::F32,
            ),
            ("blk.0.ssm_dt.bias".into(), vec![2], GgmlType::F32),
            ("blk.0.ssm_a".into(), vec![2], GgmlType::F32),
            ("blk.0.ssm_d".into(), vec![2], GgmlType::F32),
            ("blk.0.ssm_norm.weight".into(), vec![8], GgmlType::F32),
            ("blk.0.ssm_out.weight".into(), vec![8, 8], GgmlType::F32),
            ("blk.2.attn_q.weight".into(), vec![8, 8], GgmlType::F32),
            ("blk.2.attn_k.weight".into(), vec![8, 4], GgmlType::F32),
            ("blk.2.attn_v.weight".into(), vec![8, 4], GgmlType::F32),
            ("blk.2.attn_output.weight".into(), vec![8, 8], GgmlType::F32),
        ]);
        if is_moe {
            specs.extend([
                (
                    "blk.1.ffn_gate_inp.weight".into(),
                    vec![8, 2],
                    GgmlType::F32,
                ),
                ("blk.1.exp_probs_b.bias".into(), vec![2], GgmlType::F32),
                (
                    "blk.1.ffn_up_shexp.weight".into(),
                    vec![8, 10],
                    GgmlType::F32,
                ),
                (
                    "blk.1.ffn_down_shexp.weight".into(),
                    vec![10, 8],
                    GgmlType::F32,
                ),
                (
                    "blk.1.ffn_up_exps.weight".into(),
                    vec![8, 6, 2],
                    GgmlType::F32,
                ),
                (
                    "blk.1.ffn_down_exps.weight".into(),
                    vec![6, 8, 2],
                    GgmlType::F32,
                ),
            ]);
        } else {
            specs.extend([
                ("blk.1.ffn_up.weight".into(), vec![8, 12], GgmlType::F32),
                ("blk.1.ffn_down.weight".into(), vec![12, 8], GgmlType::F32),
            ]);
        }
        specs
    }

    fn write_complete_nemotron_h_gguf(
        path: &Path,
        is_moe: bool,
        mutate: impl FnOnce(&mut Vec<(String, Vec<u64>, GgmlType)>),
    ) {
        let mut specs = nemotron_h_gguf_specs(is_moe);
        mutate(&mut specs);
        let payloads = specs
            .iter()
            .map(|(_, dimensions, encoding)| {
                let elements = dimensions.iter().product::<u64>();
                let (block, bytes) = encoding.block_and_bytes().unwrap();
                vec![0xff; (elements / block * bytes) as usize]
            })
            .collect::<Vec<_>>();
        let tensors = specs
            .iter()
            .zip(&payloads)
            .map(|((name, dimensions, ggml_type), data)| TensorInput {
                name,
                dimensions,
                ggml_type: *ggml_type,
                data,
            })
            .collect::<Vec<_>>();
        Writer::default()
            .write(
                std::fs::File::create(path).unwrap(),
                &nemotron_h_gguf_metadata(is_moe),
                &tensors,
            )
            .unwrap();
    }

    fn qwen3_gguf_specs(is_moe: bool) -> Vec<(String, Vec<u64>, GgmlType)> {
        let mut specs = vec![
            ("token_embd.weight".into(), vec![32, 32], GgmlType::F32),
            ("output_norm.weight".into(), vec![32], GgmlType::F32),
            ("blk.0.attn_norm.weight".into(), vec![32], GgmlType::F32),
            ("blk.0.ffn_norm.weight".into(), vec![32], GgmlType::F32),
            ("blk.0.attn_q_norm.weight".into(), vec![32], GgmlType::F32),
            ("blk.0.attn_k_norm.weight".into(), vec![32], GgmlType::F32),
            ("blk.0.attn_q.weight".into(), vec![32, 32], GgmlType::F32),
            ("blk.0.attn_k.weight".into(), vec![32, 32], GgmlType::F32),
            ("blk.0.attn_v.weight".into(), vec![32, 32], GgmlType::F32),
            (
                "blk.0.attn_output.weight".into(),
                vec![32, 32],
                GgmlType::F32,
            ),
        ];
        if is_moe {
            specs.extend([
                (
                    "blk.0.ffn_gate_inp.weight".into(),
                    vec![32, 4],
                    GgmlType::F32,
                ),
                (
                    "blk.0.ffn_gate_exps.weight".into(),
                    vec![32, 8, 4],
                    GgmlType::F32,
                ),
                (
                    "blk.0.ffn_up_exps.weight".into(),
                    vec![32, 8, 4],
                    GgmlType::F32,
                ),
                (
                    "blk.0.ffn_down_exps.weight".into(),
                    vec![8, 32, 4],
                    GgmlType::F32,
                ),
            ]);
        } else {
            specs.extend([
                ("blk.0.ffn_gate.weight".into(), vec![32, 64], GgmlType::F32),
                ("blk.0.ffn_up.weight".into(), vec![32, 64], GgmlType::F32),
                ("blk.0.ffn_down.weight".into(), vec![64, 32], GgmlType::F32),
            ]);
        }
        specs
    }

    fn write_complete_qwen3_gguf(
        path: &Path,
        is_moe: bool,
        mutate: impl FnOnce(&mut Vec<(String, Vec<u64>, GgmlType)>),
    ) {
        let mut specs = qwen3_gguf_specs(is_moe);
        mutate(&mut specs);
        let payloads = specs
            .iter()
            .map(|(_, dimensions, encoding)| {
                let elements = dimensions.iter().product::<u64>();
                let (block, bytes) = encoding.block_and_bytes().unwrap();
                vec![0xff; (elements / block * bytes) as usize]
            })
            .collect::<Vec<_>>();
        let tensors = specs
            .iter()
            .zip(&payloads)
            .map(|((name, dimensions, ggml_type), data)| TensorInput {
                name,
                dimensions,
                ggml_type: *ggml_type,
                data,
            })
            .collect::<Vec<_>>();
        Writer::default()
            .write(
                std::fs::File::create(path).unwrap(),
                &qwen3_gguf_metadata(is_moe),
                &tensors,
            )
            .unwrap();
    }

    fn qwen3_vl_gguf_metadata() -> BTreeMap<String, MetadataValue> {
        let mut tokens = (0..30)
            .map(|index| format!("token-{index}"))
            .collect::<Vec<_>>();
        tokens.extend(["<|image_pad|>".into(), "<|video_pad|>".into()]);
        BTreeMap::from([
            (
                "general.architecture".into(),
                MetadataValue::String("qwen3vl".into()),
            ),
            ("general.file_type".into(), MetadataValue::Uint32(0)),
            ("qwen3vl.block_count".into(), MetadataValue::Uint32(1)),
            ("qwen3vl.embedding_length".into(), MetadataValue::Uint32(32)),
            (
                "qwen3vl.feed_forward_length".into(),
                MetadataValue::Uint32(64),
            ),
            (
                "qwen3vl.attention.head_count".into(),
                MetadataValue::Uint32(1),
            ),
            (
                "qwen3vl.attention.head_count_kv".into(),
                MetadataValue::Uint32(1),
            ),
            (
                "qwen3vl.attention.key_length".into(),
                MetadataValue::Uint32(32),
            ),
            (
                "qwen3vl.attention.layer_norm_rms_epsilon".into(),
                MetadataValue::Float32(0.000001),
            ),
            ("qwen3vl.context_length".into(), MetadataValue::Uint32(128)),
            (
                "qwen3vl.rope.dimension_sections".into(),
                MetadataValue::Array(MetadataArray::Uint32(vec![6, 5, 5, 0])),
            ),
            (
                "qwen3vl.n_deepstack_layers".into(),
                MetadataValue::Uint32(1),
            ),
            (
                "tokenizer.ggml.tokens".into(),
                MetadataValue::Array(MetadataArray::String(tokens)),
            ),
            (
                "tokenizer.ggml.eos_token_id".into(),
                MetadataValue::Uint32(2),
            ),
        ])
    }

    fn qwen3_vl_mmproj_metadata() -> BTreeMap<String, MetadataValue> {
        BTreeMap::from([
            (
                "general.architecture".into(),
                MetadataValue::String("clip".into()),
            ),
            (
                "clip.projector_type".into(),
                MetadataValue::String("qwen3vl_merger".into()),
            ),
            (
                "clip.vision.embedding_length".into(),
                MetadataValue::Uint32(8),
            ),
            ("clip.vision.block_count".into(), MetadataValue::Uint32(1)),
            (
                "clip.vision.feed_forward_length".into(),
                MetadataValue::Uint32(16),
            ),
            (
                "clip.vision.attention.head_count".into(),
                MetadataValue::Uint32(2),
            ),
            ("clip.vision.patch_size".into(), MetadataValue::Uint32(2)),
            (
                "clip.vision.spatial_merge_size".into(),
                MetadataValue::Uint32(2),
            ),
            (
                "clip.vision.projection_dim".into(),
                MetadataValue::Uint32(32),
            ),
            (
                "clip.vision.is_deepstack_layers".into(),
                MetadataValue::Array(MetadataArray::Bool(vec![true])),
            ),
        ])
    }

    fn qwen3_vl_mmproj_specs() -> Vec<(String, Vec<u64>, GgmlType)> {
        let mut specs = vec![
            ("v.position_embd.weight".into(), vec![8, 16], GgmlType::F32),
            (
                "v.patch_embd.weight".into(),
                vec![2, 2, 3, 8],
                GgmlType::F32,
            ),
            (
                "v.patch_embd.weight.1".into(),
                vec![2, 2, 3, 8],
                GgmlType::F32,
            ),
            ("v.patch_embd.bias".into(), vec![8], GgmlType::F32),
        ];
        let prefix = "v.blk.0";
        specs.extend([
            (format!("{prefix}.ln1.weight"), vec![8], GgmlType::F32),
            (format!("{prefix}.ln1.bias"), vec![8], GgmlType::F32),
            (
                format!("{prefix}.attn_qkv.weight"),
                vec![8, 24],
                GgmlType::F32,
            ),
            (format!("{prefix}.attn_qkv.bias"), vec![24], GgmlType::F32),
            (
                format!("{prefix}.attn_out.weight"),
                vec![8, 8],
                GgmlType::F32,
            ),
            (format!("{prefix}.attn_out.bias"), vec![8], GgmlType::F32),
            (format!("{prefix}.ln2.weight"), vec![8], GgmlType::F32),
            (format!("{prefix}.ln2.bias"), vec![8], GgmlType::F32),
            (
                format!("{prefix}.ffn_up.weight"),
                vec![8, 16],
                GgmlType::F32,
            ),
            (format!("{prefix}.ffn_up.bias"), vec![16], GgmlType::F32),
            (
                format!("{prefix}.ffn_down.weight"),
                vec![16, 8],
                GgmlType::F32,
            ),
            (format!("{prefix}.ffn_down.bias"), vec![8], GgmlType::F32),
            ("v.post_ln.weight".into(), vec![8], GgmlType::F32),
            ("v.post_ln.bias".into(), vec![8], GgmlType::F32),
            ("mm.0.weight".into(), vec![32, 32], GgmlType::F32),
            ("mm.0.bias".into(), vec![32], GgmlType::F32),
            ("mm.2.weight".into(), vec![32, 32], GgmlType::F32),
            ("mm.2.bias".into(), vec![32], GgmlType::F32),
            ("v.deepstack.0.norm.weight".into(), vec![32], GgmlType::F32),
            ("v.deepstack.0.norm.bias".into(), vec![32], GgmlType::F32),
            (
                "v.deepstack.0.fc1.weight".into(),
                vec![32, 32],
                GgmlType::F32,
            ),
            ("v.deepstack.0.fc1.bias".into(), vec![32], GgmlType::F32),
            (
                "v.deepstack.0.fc2.weight".into(),
                vec![32, 32],
                GgmlType::F32,
            ),
            ("v.deepstack.0.fc2.bias".into(), vec![32], GgmlType::F32),
        ]);
        specs
    }

    fn write_gguf_specs(
        path: &Path,
        metadata: &BTreeMap<String, MetadataValue>,
        specs: &[(String, Vec<u64>, GgmlType)],
    ) {
        let payloads = specs
            .iter()
            .map(|(_, dimensions, encoding)| {
                let elements = dimensions.iter().product::<u64>();
                let (block, bytes) = encoding.block_and_bytes().unwrap();
                vec![0xff; (elements / block * bytes) as usize]
            })
            .collect::<Vec<_>>();
        let tensors = specs
            .iter()
            .zip(&payloads)
            .map(|((name, dimensions, ggml_type), data)| TensorInput {
                name,
                dimensions,
                ggml_type: *ggml_type,
                data,
            })
            .collect::<Vec<_>>();
        Writer::default()
            .write(std::fs::File::create(path).unwrap(), metadata, &tensors)
            .unwrap();
    }

    fn write_complete_qwen3_vl_gguf(
        directory: &Path,
        mutate_main: impl FnOnce(&mut Vec<(String, Vec<u64>, GgmlType)>),
        mutate_projector: impl FnOnce(&mut Vec<(String, Vec<u64>, GgmlType)>),
    ) -> (PathBuf, PathBuf) {
        let model = directory.join("qwen3vl-model.gguf");
        let projector = directory.join("mmproj-qwen3vl-F16.gguf");
        let mut main_specs = qwen3_gguf_specs(false);
        mutate_main(&mut main_specs);
        let mut projector_specs = qwen3_vl_mmproj_specs();
        mutate_projector(&mut projector_specs);
        write_gguf_specs(&model, &qwen3_vl_gguf_metadata(), &main_specs);
        write_gguf_specs(&projector, &qwen3_vl_mmproj_metadata(), &projector_specs);
        (model, projector)
    }

    fn qwen35_gguf_metadata(is_moe: bool) -> BTreeMap<String, MetadataValue> {
        let architecture = if is_moe { "qwen35moe" } else { "qwen35" };
        let key = |suffix: &str| format!("{architecture}.{suffix}");
        let mut metadata = BTreeMap::from([
            (
                "general.architecture".into(),
                MetadataValue::String(architecture.into()),
            ),
            ("general.file_type".into(), MetadataValue::Uint32(0)),
            (key("block_count"), MetadataValue::Uint32(2)),
            (key("nextn_predict_layers"), MetadataValue::Uint32(0)),
            (key("embedding_length"), MetadataValue::Uint32(32)),
            (key("attention.head_count"), MetadataValue::Uint32(2)),
            (key("attention.head_count_kv"), MetadataValue::Uint32(1)),
            (key("attention.key_length"), MetadataValue::Uint32(16)),
            (key("rope.dimension_count"), MetadataValue::Uint32(4)),
            (key("full_attention_interval"), MetadataValue::Uint32(2)),
            (key("vocab_size"), MetadataValue::Uint32(32)),
            (key("context_length"), MetadataValue::Uint32(128)),
            (
                key("attention.layer_norm_rms_epsilon"),
                MetadataValue::Float32(0.000001),
            ),
            (key("ssm.conv_kernel"), MetadataValue::Uint32(4)),
            (key("ssm.state_size"), MetadataValue::Uint32(8)),
            (key("ssm.group_count"), MetadataValue::Uint32(2)),
            (key("ssm.time_step_rank"), MetadataValue::Uint32(4)),
            (
                "tokenizer.ggml.tokens".into(),
                MetadataValue::Array(MetadataArray::String(
                    (0..28)
                        .map(|index| format!("token-{index}"))
                        .chain([
                            "<|vision_start|>".into(),
                            "<|vision_end|>".into(),
                            "<|image_pad|>".into(),
                            "<|video_pad|>".into(),
                        ])
                        .collect(),
                )),
            ),
            (
                "tokenizer.ggml.eos_token_id".into(),
                MetadataValue::Uint32(2),
            ),
        ]);
        if is_moe {
            metadata.extend([
                (key("expert_feed_forward_length"), MetadataValue::Uint32(16)),
                (
                    key("expert_shared_feed_forward_length"),
                    MetadataValue::Uint32(16),
                ),
                (key("expert_used_count"), MetadataValue::Uint32(1)),
                (key("expert_count"), MetadataValue::Uint32(2)),
            ]);
        } else {
            metadata.insert(key("feed_forward_length"), MetadataValue::Uint32(64));
        }
        metadata
    }

    fn qwen35_mmproj_metadata() -> BTreeMap<String, MetadataValue> {
        let mut metadata = qwen3_vl_mmproj_metadata();
        metadata.insert(
            "clip.vision.embedding_length".into(),
            MetadataValue::Uint32(32),
        );
        metadata.insert(
            "clip.vision.feed_forward_length".into(),
            MetadataValue::Uint32(64),
        );
        metadata.insert(
            "clip.vision.attention.head_count".into(),
            MetadataValue::Uint32(4),
        );
        metadata.insert(
            "clip.vision.is_deepstack_layers".into(),
            MetadataValue::Array(MetadataArray::Bool(vec![false])),
        );
        metadata
    }

    fn qwen35_mmproj_specs() -> Vec<(String, Vec<u64>, GgmlType)> {
        let mut specs = qwen3_vl_mmproj_specs();
        specs.retain(|(name, _, _)| !name.starts_with("v.deepstack."));
        for (name, dimensions, encoding) in &mut specs {
            *dimensions = match name.as_str() {
                "v.position_embd.weight" => vec![32, 16],
                "v.patch_embd.weight" | "v.patch_embd.weight.1" => vec![2, 2, 3, 32],
                "v.blk.0.attn_qkv.weight" => vec![32, 96],
                "v.blk.0.attn_qkv.bias" => vec![96],
                "v.blk.0.ffn_up.weight" => vec![32, 64],
                "v.blk.0.ffn_up.bias" => vec![64],
                "v.blk.0.ffn_down.weight" => vec![64, 32],
                "mm.0.weight" => vec![128, 128],
                "mm.0.bias" => vec![128],
                "mm.2.weight" => vec![128, 32],
                _ => dimensions
                    .iter()
                    .map(|dimension| if *dimension == 8 { 32 } else { *dimension })
                    .collect(),
            };
            if matches!(
                name.as_str(),
                "v.blk.0.attn_qkv.weight"
                    | "v.blk.0.attn_out.weight"
                    | "v.blk.0.ffn_up.weight"
                    | "v.blk.0.ffn_down.weight"
                    | "mm.0.weight"
                    | "mm.2.weight"
            ) {
                *encoding = GgmlType::Q8_0;
            }
        }
        specs
    }

    fn qwen35_gguf_specs(is_moe: bool) -> Vec<(String, Vec<u64>, GgmlType)> {
        let mut specs = vec![
            ("token_embd.weight".into(), vec![32, 32], GgmlType::F32),
            ("output_norm.weight".into(), vec![32], GgmlType::F32),
            ("output.weight".into(), vec![32, 32], GgmlType::F32),
        ];
        for layer in 0..2 {
            let prefix = format!("blk.{layer}");
            specs.extend([
                (
                    format!("{prefix}.attn_norm.weight"),
                    vec![32],
                    GgmlType::F32,
                ),
                (
                    format!("{prefix}.post_attention_norm.weight"),
                    vec![32],
                    GgmlType::F32,
                ),
            ]);
            if is_moe {
                specs.extend([
                    (
                        format!("{prefix}.ffn_gate_inp.weight"),
                        vec![32, 2],
                        GgmlType::F32,
                    ),
                    (
                        format!("{prefix}.ffn_gate_inp_shexp.weight"),
                        vec![32, 1],
                        GgmlType::F32,
                    ),
                    (
                        format!("{prefix}.ffn_gate_shexp.weight"),
                        vec![32, 16],
                        GgmlType::F32,
                    ),
                    (
                        format!("{prefix}.ffn_up_shexp.weight"),
                        vec![32, 16],
                        GgmlType::F32,
                    ),
                    (
                        format!("{prefix}.ffn_down_shexp.weight"),
                        vec![16, 32],
                        GgmlType::F32,
                    ),
                    (
                        format!("{prefix}.ffn_gate_exps.weight"),
                        vec![32, 16, 2],
                        GgmlType::F32,
                    ),
                    (
                        format!("{prefix}.ffn_up_exps.weight"),
                        vec![32, 16, 2],
                        GgmlType::F32,
                    ),
                    (
                        format!("{prefix}.ffn_down_exps.weight"),
                        vec![16, 32, 2],
                        GgmlType::F32,
                    ),
                ]);
            } else {
                specs.extend([
                    (
                        format!("{prefix}.ffn_gate.weight"),
                        vec![32, 64],
                        GgmlType::F32,
                    ),
                    (
                        format!("{prefix}.ffn_up.weight"),
                        vec![32, 64],
                        GgmlType::F32,
                    ),
                    (
                        format!("{prefix}.ffn_down.weight"),
                        vec![64, 32],
                        GgmlType::F32,
                    ),
                ]);
            }
        }
        specs.extend([
            ("blk.0.attn_qkv.weight".into(), vec![32, 64], GgmlType::F32),
            ("blk.0.attn_gate.weight".into(), vec![32, 32], GgmlType::F32),
            ("blk.0.ssm_beta.weight".into(), vec![32, 4], GgmlType::F32),
            ("blk.0.ssm_alpha.weight".into(), vec![32, 4], GgmlType::F32),
            ("blk.0.ssm_conv1d.weight".into(), vec![4, 64], GgmlType::F32),
            ("blk.0.ssm_dt.bias".into(), vec![4], GgmlType::F32),
            ("blk.0.ssm_a".into(), vec![4], GgmlType::F32),
            ("blk.0.ssm_norm.weight".into(), vec![8], GgmlType::F32),
            ("blk.0.ssm_out.weight".into(), vec![32, 32], GgmlType::F32),
            ("blk.1.attn_q.weight".into(), vec![32, 64], GgmlType::F32),
            ("blk.1.attn_k.weight".into(), vec![32, 16], GgmlType::F32),
            ("blk.1.attn_v.weight".into(), vec![32, 16], GgmlType::F32),
            (
                "blk.1.attn_output.weight".into(),
                vec![32, 32],
                GgmlType::F32,
            ),
            ("blk.1.attn_q_norm.weight".into(), vec![16], GgmlType::F32),
            ("blk.1.attn_k_norm.weight".into(), vec![16], GgmlType::F32),
        ]);
        specs
    }

    fn write_complete_qwen35_gguf(
        path: &Path,
        is_moe: bool,
        mutate: impl FnOnce(&mut Vec<(String, Vec<u64>, GgmlType)>),
    ) {
        let mut specs = qwen35_gguf_specs(is_moe);
        mutate(&mut specs);
        let payloads = specs
            .iter()
            .map(|(_, dimensions, encoding)| {
                let elements = dimensions.iter().product::<u64>();
                let (block, bytes) = encoding.block_and_bytes().unwrap();
                assert_eq!(elements % block, 0, "{dimensions:?} {encoding:?}");
                vec![0xff; (elements / block * bytes) as usize]
            })
            .collect::<Vec<_>>();
        let tensors = specs
            .iter()
            .zip(&payloads)
            .map(|((name, dimensions, ggml_type), data)| TensorInput {
                name,
                dimensions,
                ggml_type: *ggml_type,
                data,
            })
            .collect::<Vec<_>>();
        Writer::default()
            .write(
                std::fs::File::create(path).unwrap(),
                &qwen35_gguf_metadata(is_moe),
                &tensors,
            )
            .unwrap();
    }

    fn qwen3_next_gguf_metadata() -> BTreeMap<String, MetadataValue> {
        qwen35_gguf_metadata(true)
            .into_iter()
            .map(|(key, value)| {
                if key == "general.architecture" {
                    (key, MetadataValue::String("qwen3next".into()))
                } else {
                    (key.replacen("qwen35moe.", "qwen3next.", 1), value)
                }
            })
            .collect()
    }

    fn qwen3_next_gguf_specs() -> Vec<(String, Vec<u64>, GgmlType)> {
        let mut specs = qwen35_gguf_specs(true);
        specs.retain(|(name, _, _)| {
            ![
                "blk.0.attn_qkv.weight",
                "blk.0.attn_gate.weight",
                "blk.0.ssm_beta.weight",
                "blk.0.ssm_alpha.weight",
            ]
            .contains(&name.as_str())
        });
        specs.extend([
            ("blk.0.attn_qkvz.weight".into(), vec![32, 96], GgmlType::F32),
            ("blk.0.ssm_ba.weight".into(), vec![32, 8], GgmlType::F32),
        ]);
        specs
    }

    fn write_complete_qwen3_next_gguf(
        path: &Path,
        mutate_metadata: impl FnOnce(&mut BTreeMap<String, MetadataValue>),
        mutate: impl FnOnce(&mut Vec<(String, Vec<u64>, GgmlType)>),
    ) {
        let mut metadata = qwen3_next_gguf_metadata();
        mutate_metadata(&mut metadata);
        let mut specs = qwen3_next_gguf_specs();
        mutate(&mut specs);
        let payloads = specs
            .iter()
            .map(|(_, dimensions, encoding)| {
                let elements = dimensions.iter().product::<u64>();
                let (block, bytes) = encoding.block_and_bytes().unwrap();
                assert_eq!(elements % block, 0, "{dimensions:?} {encoding:?}");
                vec![0xff; (elements / block * bytes) as usize]
            })
            .collect::<Vec<_>>();
        let tensors = specs
            .iter()
            .zip(&payloads)
            .map(|((name, dimensions, ggml_type), data)| TensorInput {
                name,
                dimensions,
                ggml_type: *ggml_type,
                data,
            })
            .collect::<Vec<_>>();
        Writer::default()
            .write(std::fs::File::create(path).unwrap(), &metadata, &tensors)
            .unwrap();
    }

    fn lfm2_gguf_metadata(is_moe: bool) -> BTreeMap<String, MetadataValue> {
        let architecture = if is_moe { "lfm2moe" } else { "lfm2" };
        let mut metadata = BTreeMap::from([
            (
                "general.architecture".into(),
                MetadataValue::String(architecture.into()),
            ),
            ("general.file_type".into(), MetadataValue::Uint32(0)),
            (
                format!("{architecture}.block_count"),
                MetadataValue::Uint32(2),
            ),
            (
                format!("{architecture}.embedding_length"),
                MetadataValue::Uint32(32),
            ),
            (
                format!("{architecture}.feed_forward_length"),
                MetadataValue::Uint32(48),
            ),
            (
                format!("{architecture}.attention.head_count"),
                MetadataValue::Uint32(4),
            ),
            (
                format!("{architecture}.attention.head_count_kv"),
                MetadataValue::Array(MetadataArray::Uint32(vec![0, 2])),
            ),
            (
                format!("{architecture}.attention.layer_norm_rms_epsilon"),
                MetadataValue::Float32(0.00001),
            ),
            (
                format!("{architecture}.context_length"),
                MetadataValue::Uint32(128),
            ),
            (
                format!("{architecture}.shortconv.l_cache"),
                MetadataValue::Uint32(3),
            ),
            (
                format!("{architecture}.vocab_size"),
                MetadataValue::Uint32(32),
            ),
        ]);
        if is_moe {
            metadata.extend([
                (
                    format!("{architecture}.expert_feed_forward_length"),
                    MetadataValue::Uint32(8),
                ),
                (
                    format!("{architecture}.leading_dense_block_count"),
                    MetadataValue::Uint32(1),
                ),
                (
                    format!("{architecture}.expert_count"),
                    MetadataValue::Uint32(4),
                ),
                (
                    format!("{architecture}.expert_used_count"),
                    MetadataValue::Uint32(2),
                ),
                (
                    format!("{architecture}.expert_weights_norm"),
                    MetadataValue::Uint32(1),
                ),
            ]);
        }
        metadata
    }

    fn lfm2_gguf_specs(is_moe: bool) -> Vec<(String, Vec<u64>, GgmlType)> {
        let mut specs = vec![
            ("token_embd.weight".into(), vec![32, 32], GgmlType::F32),
            ("token_embd_norm.weight".into(), vec![32], GgmlType::F32),
            ("blk.0.attn_norm.weight".into(), vec![32], GgmlType::F32),
            ("blk.0.ffn_norm.weight".into(), vec![32], GgmlType::F32),
            (
                "blk.0.shortconv.conv.weight".into(),
                vec![3, 32],
                GgmlType::F32,
            ),
            (
                "blk.0.shortconv.in_proj.weight".into(),
                vec![32, 96],
                GgmlType::F32,
            ),
            (
                "blk.0.shortconv.out_proj.weight".into(),
                vec![32, 32],
                GgmlType::F32,
            ),
            ("blk.0.ffn_gate.weight".into(), vec![32, 48], GgmlType::F32),
            ("blk.0.ffn_down.weight".into(), vec![48, 32], GgmlType::F32),
            ("blk.0.ffn_up.weight".into(), vec![32, 48], GgmlType::F32),
            ("blk.1.attn_norm.weight".into(), vec![32], GgmlType::F32),
            ("blk.1.ffn_norm.weight".into(), vec![32], GgmlType::F32),
            ("blk.1.attn_q.weight".into(), vec![32, 32], GgmlType::F32),
            ("blk.1.attn_k.weight".into(), vec![32, 16], GgmlType::F32),
            ("blk.1.attn_v.weight".into(), vec![32, 16], GgmlType::F32),
            (
                "blk.1.attn_output.weight".into(),
                vec![32, 32],
                GgmlType::F32,
            ),
            ("blk.1.attn_q_norm.weight".into(), vec![8], GgmlType::F32),
            ("blk.1.attn_k_norm.weight".into(), vec![8], GgmlType::F32),
        ];
        if is_moe {
            specs.extend([
                (
                    "blk.1.ffn_gate_inp.weight".into(),
                    vec![32, 4],
                    GgmlType::F32,
                ),
                (
                    "blk.1.ffn_gate_exps.weight".into(),
                    vec![32, 8, 4],
                    GgmlType::F32,
                ),
                (
                    "blk.1.ffn_up_exps.weight".into(),
                    vec![32, 8, 4],
                    GgmlType::F32,
                ),
                (
                    "blk.1.ffn_down_exps.weight".into(),
                    vec![8, 32, 4],
                    GgmlType::F32,
                ),
                ("blk.1.ffn_exp_probs_b.bias".into(), vec![4], GgmlType::F32),
            ]);
        } else {
            specs.extend([
                ("blk.1.ffn_gate.weight".into(), vec![32, 48], GgmlType::F32),
                ("blk.1.ffn_down.weight".into(), vec![48, 32], GgmlType::F32),
                ("blk.1.ffn_up.weight".into(), vec![32, 48], GgmlType::F32),
            ]);
        }
        specs
    }

    fn write_complete_lfm2_gguf(
        path: &Path,
        is_moe: bool,
        mutate: impl FnOnce(&mut Vec<(String, Vec<u64>, GgmlType)>),
    ) {
        let mut specs = lfm2_gguf_specs(is_moe);
        mutate(&mut specs);
        let payloads = specs
            .iter()
            .map(|(_, dimensions, encoding)| {
                let elements = dimensions.iter().product::<u64>();
                let (block, bytes) = encoding.block_and_bytes().unwrap();
                vec![0xff; (elements / block * bytes) as usize]
            })
            .collect::<Vec<_>>();
        let tensors = specs
            .iter()
            .zip(&payloads)
            .map(|((name, dimensions, ggml_type), data)| TensorInput {
                name,
                dimensions,
                ggml_type: *ggml_type,
                data,
            })
            .collect::<Vec<_>>();
        Writer::default()
            .write(
                std::fs::File::create(path).unwrap(),
                &lfm2_gguf_metadata(is_moe),
                &tensors,
            )
            .unwrap();
    }

    fn deepseek4_gguf_metadata() -> BTreeMap<String, MetadataValue> {
        BTreeMap::from([
            (
                "general.architecture".into(),
                MetadataValue::String("deepseek4".into()),
            ),
            ("general.file_type".into(), MetadataValue::Uint32(36)),
            ("deepseek4.block_count".into(), MetadataValue::Uint32(3)),
            (
                "deepseek4.context_length".into(),
                MetadataValue::Uint32(128),
            ),
            (
                "deepseek4.embedding_length".into(),
                MetadataValue::Uint32(32),
            ),
            (
                "deepseek4.embedding_length_out".into(),
                MetadataValue::Uint32(64),
            ),
            (
                "deepseek4.attention.head_count".into(),
                MetadataValue::Uint32(4),
            ),
            (
                "deepseek4.attention.head_count_kv".into(),
                MetadataValue::Uint32(1),
            ),
            (
                "deepseek4.attention.key_length".into(),
                MetadataValue::Uint32(16),
            ),
            (
                "deepseek4.attention.layer_norm_rms_epsilon".into(),
                MetadataValue::Float32(0.000001),
            ),
            (
                "deepseek4.attention.q_lora_rank".into(),
                MetadataValue::Uint32(32),
            ),
            (
                "deepseek4.attention.output_lora_rank".into(),
                MetadataValue::Uint32(16),
            ),
            (
                "deepseek4.attention.output_group_count".into(),
                MetadataValue::Uint32(2),
            ),
            (
                "deepseek4.attention.sliding_window".into(),
                MetadataValue::Uint32(8),
            ),
            (
                "deepseek4.attention.compress_ratios".into(),
                MetadataValue::Array(MetadataArray::Uint32(vec![0, 4, 128])),
            ),
            (
                "deepseek4.attention.compress_rope_freq_base".into(),
                MetadataValue::Float32(160_000.0),
            ),
            (
                "deepseek4.attention.indexer.head_count".into(),
                MetadataValue::Uint32(4),
            ),
            (
                "deepseek4.attention.indexer.key_length".into(),
                MetadataValue::Uint32(8),
            ),
            (
                "deepseek4.attention.indexer.top_k".into(),
                MetadataValue::Uint32(2),
            ),
            (
                "deepseek4.rope.dimension_count".into(),
                MetadataValue::Uint32(8),
            ),
            (
                "deepseek4.rope.freq_base".into(),
                MetadataValue::Float32(10_000.0),
            ),
            (
                "deepseek4.hyper_connection.count".into(),
                MetadataValue::Uint32(2),
            ),
            (
                "deepseek4.hyper_connection.sinkhorn_iterations".into(),
                MetadataValue::Uint32(4),
            ),
            (
                "deepseek4.hyper_connection.epsilon".into(),
                MetadataValue::Float32(0.000001),
            ),
            (
                "deepseek4.expert_feed_forward_length".into(),
                MetadataValue::Uint32(32),
            ),
            ("deepseek4.expert_count".into(), MetadataValue::Uint32(2)),
            (
                "deepseek4.expert_shared_count".into(),
                MetadataValue::Uint32(1),
            ),
            (
                "deepseek4.expert_used_count".into(),
                MetadataValue::Uint32(1),
            ),
            (
                "deepseek4.expert_gating_func".into(),
                MetadataValue::Uint32(3),
            ),
            (
                "deepseek4.expert_weights_norm".into(),
                MetadataValue::Bool(true),
            ),
            (
                "deepseek4.expert_weights_scale".into(),
                MetadataValue::Float32(1.0),
            ),
            (
                "deepseek4.swiglu_clamp_exp".into(),
                MetadataValue::Array(MetadataArray::Float32(vec![7.0; 3])),
            ),
            (
                "deepseek4.swiglu_clamp_shexp".into(),
                MetadataValue::Array(MetadataArray::Float32(vec![7.0; 3])),
            ),
            (
                "deepseek4.hash_layer_count".into(),
                MetadataValue::Uint32(1),
            ),
            ("deepseek4.vocab_size".into(), MetadataValue::Uint32(32)),
        ])
    }

    fn deepseek4_gguf_specs() -> Vec<(String, Vec<u64>, GgmlType)> {
        let tensor = |name: String, mlx: &[u64], encoding| {
            let mut dimensions = mlx.to_vec();
            dimensions.reverse();
            (name, dimensions, encoding)
        };
        let f32 = |name: String, mlx: &[u64]| tensor(name, mlx, GgmlType::F32);
        let mut specs = vec![
            f32("token_embd.weight".into(), &[32, 32]),
            f32("output_norm.weight".into(), &[32]),
            f32("output.weight".into(), &[32, 32]),
            f32("output_hc_fn.weight".into(), &[2, 64]),
            f32("output_hc_base.weight".into(), &[2]),
            f32("output_hc_scale.weight".into(), &[1]),
        ];
        for (layer, ratio) in [0_u64, 4, 128].into_iter().enumerate() {
            let root = format!("blk.{layer}");
            specs.extend([
                f32(format!("{root}.attn_norm.weight"), &[32]),
                f32(format!("{root}.ffn_norm.weight"), &[32]),
                f32(format!("{root}.attn_sinks.weight"), &[4]),
                f32(format!("{root}.attn_q_a.weight"), &[32, 32]),
                f32(format!("{root}.attn_q_a_norm.weight"), &[32]),
                f32(format!("{root}.attn_q_b.weight"), &[64, 32]),
                f32(format!("{root}.attn_kv.weight"), &[16, 32]),
                f32(format!("{root}.attn_kv_a_norm.weight"), &[16]),
                f32(format!("{root}.attn_output_a.weight"), &[32, 32]),
                f32(format!("{root}.attn_output_b.weight"), &[32, 32]),
                f32(format!("{root}.hc_attn_fn.weight"), &[8, 64]),
                f32(format!("{root}.hc_attn_base.weight"), &[8]),
                f32(format!("{root}.hc_attn_scale.weight"), &[3]),
                f32(format!("{root}.hc_ffn_fn.weight"), &[8, 64]),
                f32(format!("{root}.hc_ffn_base.weight"), &[8]),
                f32(format!("{root}.hc_ffn_scale.weight"), &[3]),
                f32(format!("{root}.ffn_gate_inp.weight"), &[2, 32]),
                f32(format!("{root}.ffn_gate_shexp.weight"), &[32, 32]),
                f32(format!("{root}.ffn_down_shexp.weight"), &[32, 32]),
                f32(format!("{root}.ffn_up_shexp.weight"), &[32, 32]),
                tensor(
                    format!("{root}.ffn_gate_exps.weight"),
                    &[2, 32, 32],
                    GgmlType::MxFp4,
                ),
                tensor(
                    format!("{root}.ffn_down_exps.weight"),
                    &[2, 32, 32],
                    GgmlType::MxFp4,
                ),
                tensor(
                    format!("{root}.ffn_up_exps.weight"),
                    &[2, 32, 32],
                    GgmlType::MxFp4,
                ),
            ]);
            if layer == 0 {
                specs.push(tensor(
                    format!("{root}.ffn_gate_tid2eid.weight"),
                    &[32, 1],
                    GgmlType::I32,
                ));
            } else {
                specs.push(f32(format!("{root}.exp_probs_b.bias"), &[2]));
            }
            if ratio != 0 {
                let output = if ratio == 4 { 32 } else { 16 };
                specs.extend([
                    f32(format!("{root}.attn_compressor_kv.weight"), &[output, 32]),
                    f32(format!("{root}.attn_compressor_gate.weight"), &[output, 32]),
                    f32(
                        format!("{root}.attn_compressor_ape.weight"),
                        &[ratio, output],
                    ),
                    f32(format!("{root}.attn_compressor_norm.weight"), &[16]),
                ]);
            }
            if ratio == 4 {
                specs.extend([
                    f32(format!("{root}.indexer.attn_q_b.weight"), &[32, 32]),
                    f32(format!("{root}.indexer.proj.weight"), &[4, 32]),
                    f32(format!("{root}.indexer_compressor_kv.weight"), &[16, 32]),
                    f32(format!("{root}.indexer_compressor_gate.weight"), &[16, 32]),
                    f32(format!("{root}.indexer_compressor_ape.weight"), &[4, 16]),
                    f32(format!("{root}.indexer_compressor_norm.weight"), &[8]),
                ]);
            }
        }
        specs
    }

    fn write_complete_deepseek4_gguf(
        path: &Path,
        mutate_metadata: impl FnOnce(&mut BTreeMap<String, MetadataValue>),
        mutate_specs: impl FnOnce(&mut Vec<(String, Vec<u64>, GgmlType)>),
    ) {
        let mut metadata = deepseek4_gguf_metadata();
        mutate_metadata(&mut metadata);
        let mut specs = deepseek4_gguf_specs();
        mutate_specs(&mut specs);
        write_gguf_specs(path, &metadata, &specs);
    }

    fn deepseek2_gguf_metadata() -> BTreeMap<String, MetadataValue> {
        BTreeMap::from([
            (
                "general.architecture".into(),
                MetadataValue::String("deepseek2".into()),
            ),
            ("general.file_type".into(), MetadataValue::Uint32(0)),
            ("deepseek2.block_count".into(), MetadataValue::Uint32(2)),
            (
                "deepseek2.context_length".into(),
                MetadataValue::Uint32(128),
            ),
            (
                "deepseek2.embedding_length".into(),
                MetadataValue::Uint32(32),
            ),
            (
                "deepseek2.feed_forward_length".into(),
                MetadataValue::Uint32(64),
            ),
            (
                "deepseek2.attention.head_count".into(),
                MetadataValue::Uint32(2),
            ),
            (
                "deepseek2.attention.layer_norm_rms_epsilon".into(),
                MetadataValue::Float32(0.000001),
            ),
            (
                "deepseek2.rope.freq_base".into(),
                MetadataValue::Float32(10_000.0),
            ),
            (
                "deepseek2.rope.dimension_count".into(),
                MetadataValue::Uint32(8),
            ),
            (
                "deepseek2.attention.q_lora_rank".into(),
                MetadataValue::Uint32(32),
            ),
            (
                "deepseek2.attention.kv_lora_rank".into(),
                MetadataValue::Uint32(32),
            ),
            (
                "deepseek2.attention.key_length_mla".into(),
                MetadataValue::Uint32(40),
            ),
            (
                "deepseek2.attention.value_length_mla".into(),
                MetadataValue::Uint32(16),
            ),
            (
                "deepseek2.leading_dense_block_count".into(),
                MetadataValue::Uint32(1),
            ),
            ("deepseek2.expert_count".into(), MetadataValue::Uint32(4)),
            (
                "deepseek2.expert_shared_count".into(),
                MetadataValue::Uint32(1),
            ),
            (
                "deepseek2.expert_feed_forward_length".into(),
                MetadataValue::Uint32(32),
            ),
            (
                "deepseek2.expert_used_count".into(),
                MetadataValue::Uint32(2),
            ),
            (
                "deepseek2.expert_group_count".into(),
                MetadataValue::Uint32(2),
            ),
            (
                "deepseek2.expert_group_used_count".into(),
                MetadataValue::Uint32(1),
            ),
            (
                "deepseek2.expert_gating_func".into(),
                MetadataValue::Uint32(2),
            ),
            (
                "deepseek2.expert_weights_norm".into(),
                MetadataValue::Bool(true),
            ),
            (
                "deepseek2.expert_weights_scale".into(),
                MetadataValue::Float32(1.5),
            ),
            ("deepseek2.vocab_size".into(), MetadataValue::Uint32(32)),
        ])
    }

    fn deepseek2_gguf_specs(split_kv: bool) -> Vec<(String, Vec<u64>, GgmlType)> {
        let mut specs = vec![
            ("token_embd.weight".into(), vec![32, 32], GgmlType::F32),
            ("output_norm.weight".into(), vec![32], GgmlType::F32),
            ("output.weight".into(), vec![32, 32], GgmlType::F32),
        ];
        for layer in 0..2 {
            let prefix = format!("blk.{layer}");
            specs.extend([
                (
                    format!("{prefix}.attn_norm.weight"),
                    vec![32],
                    GgmlType::F32,
                ),
                (format!("{prefix}.ffn_norm.weight"), vec![32], GgmlType::F32),
                (
                    format!("{prefix}.attn_q_a.weight"),
                    vec![32, 32],
                    GgmlType::F32,
                ),
                (
                    format!("{prefix}.attn_q_a_norm.weight"),
                    vec![32],
                    GgmlType::F32,
                ),
                (
                    format!("{prefix}.attn_q_b.weight"),
                    vec![32, 80],
                    GgmlType::F32,
                ),
                (
                    format!("{prefix}.attn_kv_a_mqa.weight"),
                    vec![32, 40],
                    GgmlType::F32,
                ),
                (
                    format!("{prefix}.attn_kv_a_norm.weight"),
                    vec![32],
                    GgmlType::F32,
                ),
                (
                    format!("{prefix}.attn_output.weight"),
                    vec![32, 32],
                    GgmlType::F32,
                ),
            ]);
            if split_kv {
                specs.extend([
                    (
                        format!("{prefix}.attn_k_b.weight"),
                        vec![32, 32, 2],
                        GgmlType::F32,
                    ),
                    (
                        format!("{prefix}.attn_v_b.weight"),
                        vec![32, 16, 2],
                        GgmlType::F32,
                    ),
                ]);
            } else {
                specs.push((
                    format!("{prefix}.attn_kv_b.weight"),
                    vec![32, 96],
                    GgmlType::F32,
                ));
            }
        }
        specs.extend([
            ("blk.0.ffn_gate.weight".into(), vec![32, 64], GgmlType::F32),
            ("blk.0.ffn_up.weight".into(), vec![32, 64], GgmlType::F32),
            ("blk.0.ffn_down.weight".into(), vec![64, 32], GgmlType::F32),
            (
                "blk.1.ffn_gate_inp.weight".into(),
                vec![32, 4],
                GgmlType::F32,
            ),
            ("blk.1.exp_probs_b.bias".into(), vec![4], GgmlType::F32),
            (
                "blk.1.ffn_gate_shexp.weight".into(),
                vec![32, 32],
                GgmlType::F32,
            ),
            (
                "blk.1.ffn_up_shexp.weight".into(),
                vec![32, 32],
                GgmlType::F32,
            ),
            (
                "blk.1.ffn_down_shexp.weight".into(),
                vec![32, 32],
                GgmlType::F32,
            ),
            (
                "blk.1.ffn_gate_exps.weight".into(),
                vec![32, 32, 4],
                GgmlType::F32,
            ),
            (
                "blk.1.ffn_up_exps.weight".into(),
                vec![32, 32, 4],
                GgmlType::F32,
            ),
            (
                "blk.1.ffn_down_exps.weight".into(),
                vec![32, 32, 4],
                GgmlType::F32,
            ),
        ]);
        specs
    }

    fn write_complete_deepseek2_gguf(
        path: &Path,
        split_kv: bool,
        mutate: impl FnOnce(&mut Vec<(String, Vec<u64>, GgmlType)>),
    ) {
        let mut specs = deepseek2_gguf_specs(split_kv);
        mutate(&mut specs);
        let payloads = specs
            .iter()
            .map(|(_, dimensions, encoding)| {
                let elements = dimensions.iter().product::<u64>();
                let (block, bytes) = encoding.block_and_bytes().unwrap();
                vec![0xff; (elements / block * bytes) as usize]
            })
            .collect::<Vec<_>>();
        let tensors = specs
            .iter()
            .zip(&payloads)
            .map(|((name, dimensions, ggml_type), data)| TensorInput {
                name,
                dimensions,
                ggml_type: *ggml_type,
                data,
            })
            .collect::<Vec<_>>();
        Writer::default()
            .write(
                std::fs::File::create(path).unwrap(),
                &deepseek2_gguf_metadata(),
                &tensors,
            )
            .unwrap();
    }

    fn gpt_oss_gguf_metadata() -> BTreeMap<String, MetadataValue> {
        BTreeMap::from([
            (
                "general.architecture".into(),
                MetadataValue::String("gpt-oss".into()),
            ),
            ("general.file_type".into(), MetadataValue::Uint32(0)),
            ("gpt-oss.block_count".into(), MetadataValue::Uint32(1)),
            ("gpt-oss.embedding_length".into(), MetadataValue::Uint32(32)),
            (
                "gpt-oss.attention.head_count".into(),
                MetadataValue::Uint32(1),
            ),
            (
                "gpt-oss.attention.head_count_kv".into(),
                MetadataValue::Uint32(1),
            ),
            (
                "gpt-oss.attention.key_length".into(),
                MetadataValue::Uint32(32),
            ),
            (
                "gpt-oss.expert_feed_forward_length".into(),
                MetadataValue::Uint32(32),
            ),
            ("gpt-oss.expert_count".into(), MetadataValue::Uint32(2)),
            ("gpt-oss.expert_used_count".into(), MetadataValue::Uint32(1)),
            (
                "gpt-oss.attention.layer_norm_rms_epsilon".into(),
                MetadataValue::Float32(0.00001),
            ),
            (
                "gpt-oss.attention.sliding_window".into(),
                MetadataValue::Uint32(8),
            ),
            ("gpt-oss.context_length".into(), MetadataValue::Uint32(128)),
            ("gpt-oss.vocab_size".into(), MetadataValue::Uint32(32)),
        ])
    }

    fn gpt_oss_gguf_specs() -> Vec<(String, Vec<u64>, GgmlType)> {
        let mut specs = vec![
            ("token_embd.weight".into(), vec![32, 32], GgmlType::F32),
            ("output_norm.weight".into(), vec![32], GgmlType::F32),
            ("output.weight".into(), vec![32, 32], GgmlType::F32),
            ("blk.0.attn_norm.weight".into(), vec![32], GgmlType::F32),
            (
                "blk.0.attn_post_norm.weight".into(),
                vec![32],
                GgmlType::F32,
            ),
            ("blk.0.attn_sinks.weight".into(), vec![1], GgmlType::F32),
        ];
        for projection in ["q", "k", "v", "output"] {
            specs.extend([
                (
                    format!("blk.0.attn_{projection}.weight"),
                    vec![32, 32],
                    GgmlType::F32,
                ),
                (
                    format!("blk.0.attn_{projection}.bias"),
                    vec![32],
                    GgmlType::F32,
                ),
            ]);
        }
        specs.extend([
            (
                "blk.0.ffn_gate_inp.weight".into(),
                vec![32, 2],
                GgmlType::F32,
            ),
            ("blk.0.ffn_gate_inp.bias".into(), vec![2], GgmlType::F32),
            (
                "blk.0.ffn_gate_exps.weight".into(),
                vec![32, 32, 2],
                GgmlType::MxFp4,
            ),
            (
                "blk.0.ffn_gate_exps.bias".into(),
                vec![32, 2],
                GgmlType::F32,
            ),
            (
                "blk.0.ffn_up_exps.weight".into(),
                vec![32, 32, 2],
                GgmlType::MxFp4,
            ),
            ("blk.0.ffn_up_exps.bias".into(), vec![32, 2], GgmlType::F32),
            (
                "blk.0.ffn_down_exps.weight".into(),
                vec![32, 32, 2],
                GgmlType::MxFp4,
            ),
            (
                "blk.0.ffn_down_exps.bias".into(),
                vec![32, 2],
                GgmlType::F32,
            ),
        ]);
        specs
    }

    fn write_complete_gpt_oss_gguf(
        path: &Path,
        mutate: impl FnOnce(&mut Vec<(String, Vec<u64>, GgmlType)>),
    ) {
        let mut specs = gpt_oss_gguf_specs();
        mutate(&mut specs);
        let payloads = specs
            .iter()
            .map(|(_, dimensions, encoding)| {
                let elements = dimensions.iter().product::<u64>();
                let (block, bytes) = encoding.block_and_bytes().unwrap();
                assert_eq!(elements % block, 0);
                vec![0xff; (elements / block * bytes) as usize]
            })
            .collect::<Vec<_>>();
        let tensors = specs
            .iter()
            .zip(&payloads)
            .map(|((name, dimensions, ggml_type), data)| TensorInput {
                name,
                dimensions,
                ggml_type: *ggml_type,
                data,
            })
            .collect::<Vec<_>>();
        Writer::default()
            .write(
                std::fs::File::create(path).unwrap(),
                &gpt_oss_gguf_metadata(),
                &tensors,
            )
            .unwrap();
    }

    fn write_unknown_gguf_type(path: &Path, code: u32) {
        let mut file = std::fs::File::create(path).unwrap();
        file.write_all(b"GGUF").unwrap();
        file.write_all(&3_u32.to_le_bytes()).unwrap();
        file.write_all(&1_u64.to_le_bytes()).unwrap();
        file.write_all(&0_u64.to_le_bytes()).unwrap();
        file.write_all(&1_u64.to_le_bytes()).unwrap();
        file.write_all(b"x").unwrap();
        file.write_all(&1_u32.to_le_bytes()).unwrap();
        file.write_all(&1_u64.to_le_bytes()).unwrap();
        file.write_all(&code.to_le_bytes()).unwrap();
        file.write_all(&0_u64.to_le_bytes()).unwrap();
        let position = file.stream_position().unwrap();
        file.write_all(&vec![0; (64 - position) as usize]).unwrap();
    }

    #[test]
    fn loader_and_inspection_model_kind_dispatch_are_exhaustive() {
        for kind in ModelKind::ALL {
            let result = ModelLoadOptions::default().validate_preparation(
                kind,
                None,
                safemlx_lm_core::ArtifactFormat::SafeTensors,
            );
            assert_eq!(result.is_ok(), kind != ModelKind::PersonaPlex, "{kind:?}");
            assert_eq!(
                structural::safetensors_policy(kind),
                structural::StructuralValidationPolicy::Exact,
                "{kind:?}"
            );
        }
    }

    #[test]
    fn complete_sparse_personaplex_headers_are_structurally_exact_but_use_realtime_policy() {
        let directory = write_sparse_personaplex_safetensors_dir(|_| {});
        let report = inspect_model(directory.path(), MlxInspectionOptions::default()).unwrap();
        assert_eq!(report.container, InspectionReadiness::Ready);
        assert_eq!(report.architecture_support, InspectionReadiness::Ready);
        assert_eq!(report.structural_binding, InspectionReadiness::Ready);
        assert_eq!(report.model_loadability, InspectionReadiness::Ready);
        assert_eq!(report.requested_load, InspectionReadiness::Unsupported);
        assert!(!report.is_loadable());
        assert!(!report
            .issues
            .iter()
            .any(|issue| { issue.code == InspectionIssueCode::ValidationUnavailableUntilLoad }));
        structural::validate_safetensors_load_path(
            ModelKind::PersonaPlex,
            directory.path(),
            ModelLoadOptions::default(),
        )
        .unwrap();
    }

    #[test]
    fn personaplex_missing_and_misshaped_tensors_are_structured() {
        let missing = write_sparse_personaplex_safetensors_dir(|specs| {
            specs.retain(|(name, _)| name != "transformer.layers.31.self_attn.out_proj.weight");
        });
        let report = inspect_model(missing.path(), MlxInspectionOptions::default()).unwrap();
        assert_eq!(report.structural_binding, InspectionReadiness::Invalid);
        assert!(report.issues.iter().any(|issue| {
            issue.code == InspectionIssueCode::MissingRequiredTensor
                && issue.tensor_name.as_deref()
                    == Some("transformer.layers.31.self_attn.out_proj.weight")
        }));
        assert!(!report.is_loadable());

        let misshaped = write_sparse_personaplex_safetensors_dir(|specs| {
            let (_, shape) = specs
                .iter_mut()
                .find(|(name, _)| name == "depformer.layers.5.self_attn.in_proj_weight")
                .unwrap();
            shape[0] -= 1;
        });
        let report = inspect_model(misshaped.path(), MlxInspectionOptions::default()).unwrap();
        assert_eq!(report.structural_binding, InspectionReadiness::Invalid);
        assert!(report.issues.iter().any(|issue| {
            issue.code == InspectionIssueCode::TensorShapeMismatch
                && issue.tensor_name.as_deref()
                    == Some("depformer.layers.5.self_attn.in_proj_weight")
        }));
        assert!(!report.is_loadable());
    }

    #[test]
    fn personaplex_rejects_conflicting_pytorch_aliases() {
        let directory = write_sparse_personaplex_safetensors_dir(|specs| {
            specs.push((
                "transformer.layers.0.self_attn.in_proj.weight".into(),
                vec![12_288, 4_096],
            ));
        });
        let report = inspect_model(directory.path(), MlxInspectionOptions::default()).unwrap();
        assert_eq!(report.structural_binding, InspectionReadiness::Invalid);
        assert!(report.issues.iter().any(|issue| {
            issue.code == InspectionIssueCode::ConflictingTensorLayout
                && issue.tensor_name.as_deref()
                    == Some("transformer.layers.0.self_attn.in_proj.weight")
        }));
        assert!(!report.is_loadable());
    }

    #[test]
    fn gguf_architecture_resolution_covers_documented_dispatch() {
        for name in GgufArchitecture::SUPPORTED_NAMES
            .replace(", and ", ", ")
            .split(", ")
        {
            let architecture = GgufArchitecture::resolve(name).unwrap();
            assert_eq!(
                structural::gguf_policy(architecture),
                structural::StructuralValidationPolicy::Exact,
                "{name}"
            );
        }
    }

    #[test]
    fn unrelated_poison_safetensor_is_not_authoritatively_loadable() {
        let directory = write_safetensors_dir(&llama_config());
        let report = inspect_model(directory.path(), MlxInspectionOptions::default()).unwrap();
        assert_eq!(report.container, InspectionReadiness::Ready);
        assert_eq!(report.model_kind, Some(ModelKind::Llama));
        assert_eq!(report.structural_binding, InspectionReadiness::Invalid);
        assert_eq!(report.model_loadability, InspectionReadiness::Invalid);
        assert_eq!(report.requested_load, InspectionReadiness::Ready);
        assert_eq!(report.tensor_count, Some(1));
        assert_eq!(
            report.resources.stored_tensor_bytes,
            Observed::exact(4, "validated SafeTensors tensor headers")
        );
        assert_eq!(report.resources.model_kind, Some(ModelKind::Llama));
        assert_eq!(report.tensor_encodings[0].name, "F32");
        assert!(!report.is_loadable());
        assert!(
            report
                .issues
                .iter()
                .any(|issue| issue.code == InspectionIssueCode::MissingRequiredTensor),
            "{:#?}",
            report.issues
        );
    }

    #[test]
    fn complete_poisoned_llama_safetensors_headers_are_exactly_loadable() {
        let directory = write_complete_safetensors_dir(|_| {});
        let report = inspect_model(directory.path(), MlxInspectionOptions::default()).unwrap();
        assert_eq!(report.container, InspectionReadiness::Ready);
        assert_eq!(report.architecture_support, InspectionReadiness::Ready);
        assert_eq!(report.structural_binding, InspectionReadiness::Ready);
        assert_eq!(report.model_loadability, InspectionReadiness::Ready);
        assert!(report.is_loadable());
        structural::validate_safetensors_load_path(
            ModelKind::Llama,
            directory.path(),
            ModelLoadOptions::default(),
        )
        .unwrap();
    }

    #[test]
    fn complete_poisoned_dense_qwen3_safetensors_headers_are_exactly_loadable() {
        let directory = write_complete_qwen3_safetensors_dir(|_| {});
        let report = inspect_model(directory.path(), MlxInspectionOptions::default()).unwrap();
        assert_eq!(report.container, InspectionReadiness::Ready);
        assert_eq!(report.architecture_support, InspectionReadiness::Ready);
        assert_eq!(report.structural_binding, InspectionReadiness::Ready);
        assert_eq!(report.model_loadability, InspectionReadiness::Ready);
        assert!(report.is_loadable());
        structural::validate_safetensors_load_path(
            ModelKind::Qwen3,
            directory.path(),
            ModelLoadOptions::default(),
        )
        .unwrap();
    }

    #[test]
    fn qwen2_tied_and_untied_safetensors_catalogs_are_exactly_loadable() {
        for tied in [true, false] {
            let directory = write_complete_qwen2_safetensors_dir(tied, |_| {});
            let report = inspect_model(directory.path(), MlxInspectionOptions::default()).unwrap();
            assert_eq!(report.model_kind, Some(ModelKind::Qwen2));
            assert_eq!(report.structural_binding, InspectionReadiness::Ready);
            assert!(report.is_loadable());
            structural::validate_safetensors_load_path(
                ModelKind::Qwen2,
                directory.path(),
                ModelLoadOptions::default(),
            )
            .unwrap();
        }
    }

    #[test]
    fn qwen2_inspection_and_loader_preflight_reject_the_same_invalid_catalogs() {
        let cases = [
            write_complete_qwen2_safetensors_dir(false, |specs| {
                specs.retain(|(name, _, _)| name != "model.layers.0.self_attn.q_proj.bias");
            }),
            write_complete_qwen2_safetensors_dir(false, |specs| {
                specs.retain(|(name, _, _)| name != "model.layers.0.mlp.down_proj.weight");
            }),
            write_complete_qwen2_safetensors_dir(false, |specs| {
                specs.push((
                    "model.layers.0.self_attn.q_norm.weight".into(),
                    vec![2],
                    Dtype::F32,
                ));
            }),
            write_complete_qwen2_safetensors_dir(false, |specs| {
                let (_, shape, _) = specs
                    .iter_mut()
                    .find(|(name, _, _)| name == "model.layers.0.self_attn.k_proj.weight")
                    .unwrap();
                *shape = vec![8, 8];
            }),
            write_complete_qwen2_safetensors_dir(false, |specs| {
                let (_, _, dtype) = specs
                    .iter_mut()
                    .find(|(name, _, _)| name == "model.layers.0.self_attn.v_proj.bias")
                    .unwrap();
                *dtype = Dtype::I8;
            }),
        ];

        for (case, directory) in cases.into_iter().enumerate() {
            let report = inspect_model(directory.path(), MlxInspectionOptions::default()).unwrap();
            assert_eq!(
                report.structural_binding,
                InspectionReadiness::Invalid,
                "case {case}: {:#?}",
                report.issues
            );
            assert!(!report.is_loadable());
            assert!(structural::validate_safetensors_load_path(
                ModelKind::Qwen2,
                directory.path(),
                ModelLoadOptions::default(),
            )
            .is_err());
        }
    }

    #[test]
    fn complete_poisoned_qwen3_vl_safetensors_catalogs_are_exactly_loadable() {
        for (is_moe, kind) in [(false, ModelKind::Qwen3Vl), (true, ModelKind::Qwen3VlMoe)] {
            let directory = write_complete_qwen3_vl_safetensors_dir(is_moe, |_| {});
            let report = inspect_model(directory.path(), MlxInspectionOptions::default()).unwrap();
            assert_eq!(report.structural_binding, InspectionReadiness::Ready);
            assert_eq!(report.model_loadability, InspectionReadiness::Ready);
            assert!(
                report.is_loadable(),
                "is_moe={is_moe}: {:#?}",
                report.issues
            );
            structural::validate_safetensors_load_path(
                kind,
                directory.path(),
                ModelLoadOptions::default(),
            )
            .unwrap();
        }
    }

    #[test]
    fn qwen3_vl_vision_schedule_rejections_match_inspection_and_load_preflight() {
        for mutate in [
            |config: &mut Value| config["vision_config"]["window_size"] = json!(8),
            |config: &mut Value| config["vision_config"]["fullatt_block_indexes"] = json!([0]),
            |config: &mut Value| {
                config["vision_config"]["deepstack_visual_indexes"] = json!([0, 0])
            },
            |config: &mut Value| config["vision_config"]["deepstack_visual_indexes"] = json!([1]),
        ] {
            let directory = write_complete_qwen3_vl_safetensors_dir(false, |_| {});
            let mut config = qwen3_vl_safetensors_config(false);
            mutate(&mut config);
            std::fs::write(
                directory.path().join("config.json"),
                serde_json::to_vec(&config).unwrap(),
            )
            .unwrap();

            let report = inspect_model(directory.path(), MlxInspectionOptions::default()).unwrap();
            assert_eq!(report.model_loadability, InspectionReadiness::Invalid);
            assert!(!report.is_loadable());
            assert!(structural::validate_safetensors_load_path(
                ModelKind::Qwen3Vl,
                directory.path(),
                ModelLoadOptions::default(),
            )
            .is_err());
        }
    }

    #[test]
    fn qwen3_vl_moe_native_affine_text_and_dense_vision_are_exact() {
        let mut config = qwen3_vl_safetensors_config(true);
        config["text_config"]["moe_intermediate_size"] = json!(32);
        config["text_config"]["quantization"] = json!({
            "group_size": 32,
            "bits": 4,
            "mode": "affine"
        });
        let mut specs = qwen3_vl_safetensor_specs(true);
        specs.iter_mut().for_each(|(name, shape, _)| {
            if name.ends_with("experts.gate_up_proj") {
                *shape = vec![4, 64, 32];
            } else if name.ends_with("experts.down_proj") {
                *shape = vec![4, 32, 32];
            }
        });
        let specs = affine_fixture_specs(
            specs
                .into_iter()
                .map(|(name, shape, _)| (name, shape))
                .collect(),
            32,
            |name, shape| {
                name.starts_with("model.language_model.")
                    && shape.len() >= 2
                    && !name.ends_with(".mlp.gate.weight")
            },
        );
        let directory = write_typed_safetensors_dir(&config, &specs);
        let report = inspect_model(directory.path(), MlxInspectionOptions::default()).unwrap();
        assert_eq!(
            report.structural_binding,
            InspectionReadiness::Ready,
            "{:#?}",
            report.issues
        );
        assert!(report.is_loadable());
        structural::validate_safetensors_load_path(
            ModelKind::Qwen3VlMoe,
            directory.path(),
            ModelLoadOptions::default(),
        )
        .unwrap();
    }

    #[test]
    fn qwen3_vl_vision_tensor_failures_are_structured_and_fail_closed() {
        let missing = write_complete_qwen3_vl_safetensors_dir(false, |specs| {
            specs.retain(|(name, _, _)| name != "model.visual.blocks.0.attn.qkv.weight");
        });
        let report = inspect_model(missing.path(), MlxInspectionOptions::default()).unwrap();
        assert!(report.issues.iter().any(|issue| {
            issue.code == InspectionIssueCode::MissingRequiredTensor
                && issue.tensor_name.as_deref() == Some("model.visual.blocks.0.attn.qkv.weight")
        }));
        assert!(!report.is_loadable());

        let wrong_shape = write_complete_qwen3_vl_safetensors_dir(false, |specs| {
            specs
                .iter_mut()
                .find(|(name, _, _)| name == "model.visual.patch_embed.proj.weight")
                .unwrap()
                .1 = vec![8, 24];
        });
        let report = inspect_model(wrong_shape.path(), MlxInspectionOptions::default()).unwrap();
        assert!(report.issues.iter().any(|issue| {
            issue.code == InspectionIssueCode::TensorShapeMismatch
                && issue.tensor_name.as_deref() == Some("model.visual.patch_embed.proj.weight")
        }));
        assert!(!report.is_loadable());

        let quantized_vision = write_complete_qwen3_vl_safetensors_dir(false, |specs| {
            specs
                .iter_mut()
                .find(|(name, _, _)| name == "model.visual.merger.linear_fc1.weight")
                .unwrap()
                .2 = Dtype::U8;
        });
        let report =
            inspect_model(quantized_vision.path(), MlxInspectionOptions::default()).unwrap();
        assert!(report.issues.iter().any(|issue| {
            issue.code == InspectionIssueCode::UnsupportedTensorEncoding
                && issue.tensor_name.as_deref() == Some("model.visual.merger.linear_fc1.weight")
        }));
        assert!(!report.is_loadable());

        let unexpected = write_complete_qwen3_vl_safetensors_dir(false, |specs| {
            specs.push(("poison.weight".into(), vec![1], Dtype::F32));
        });
        let report = inspect_model(unexpected.path(), MlxInspectionOptions::default()).unwrap();
        assert!(report.issues.iter().any(|issue| {
            issue.code == InspectionIssueCode::ConflictingTensorLayout
                && issue.tensor_name.as_deref() == Some("poison.weight")
        }));
        assert!(!report.is_loadable());
    }

    #[test]
    fn qwen3_vl_moe_expert_layout_follows_residency_route() {
        let directory = write_complete_qwen3_vl_safetensors_dir(true, |specs| {
            specs.retain(|(name, _, _)| {
                !matches!(
                    name.as_str(),
                    "model.language_model.layers.0.mlp.experts.gate_up_proj"
                        | "model.language_model.layers.0.mlp.experts.down_proj"
                )
            });
            for expert in 0..4 {
                for (projection, shape) in [
                    ("gate_proj", vec![8, 32]),
                    ("up_proj", vec![8, 32]),
                    ("down_proj", vec![32, 8]),
                ] {
                    specs.push((
                        format!(
                            "model.language_model.layers.0.mlp.experts.{expert}.{projection}.weight"
                        ),
                        shape,
                        Dtype::F32,
                    ));
                }
            }
        });
        let resident = inspect_model(directory.path(), MlxInspectionOptions::default()).unwrap();
        assert!(!resident.is_loadable());
        assert!(resident.issues.iter().any(|issue| {
            issue.code == InspectionIssueCode::ConflictingTensorLayout
                && issue.tensor_name.as_deref().is_some_and(|name| {
                    name.starts_with("model.language_model.layers.0.mlp.experts.0")
                })
        }));

        let load = ModelLoadOptions::default()
            .with_weight_residency(WeightResidency::layerwise_host(Default::default()));
        let bounded = inspect_model(directory.path(), MlxInspectionOptions { load }).unwrap();
        assert!(bounded.is_loadable(), "{:#?}", bounded.issues);
        structural::validate_safetensors_load_path(ModelKind::Qwen3VlMoe, directory.path(), load)
            .unwrap();
    }

    #[test]
    fn complete_poisoned_dense_lfm2_safetensors_headers_are_exactly_loadable() {
        let directory = write_complete_lfm2_safetensors_dir(|_| {});
        let report = inspect_model(directory.path(), MlxInspectionOptions::default()).unwrap();
        assert_eq!(report.container, InspectionReadiness::Ready);
        assert_eq!(report.architecture_support, InspectionReadiness::Ready);
        assert_eq!(report.structural_binding, InspectionReadiness::Ready);
        assert_eq!(report.model_loadability, InspectionReadiness::Ready);
        assert!(report.is_loadable(), "{:#?}", report.issues);
        structural::validate_safetensors_load_path(
            ModelKind::Lfm2,
            directory.path(),
            ModelLoadOptions::default(),
        )
        .unwrap();
    }

    #[test]
    fn lfm2_mlx_shortconv_layout_is_admitted_and_normalized() {
        let directory = write_complete_lfm2_safetensors_dir(|specs| {
            let (_, shape) = specs
                .iter_mut()
                .find(|(name, _)| name == "model.layers.0.conv.conv.weight")
                .unwrap();
            *shape = vec![32, 3, 1];
        });
        let report = inspect_model(directory.path(), MlxInspectionOptions::default()).unwrap();
        assert!(report.is_loadable(), "{:#?}", report.issues);
        structural::validate_safetensors_load_path(
            ModelKind::Lfm2,
            directory.path(),
            ModelLoadOptions::default(),
        )
        .unwrap();

        let execution =
            safemlx::ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let weights =
            safemlx::ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let model = crate::architectures::lfm2::model::load_model(
            directory.path(),
            execution.stream(),
            weights.stream(),
        )
        .unwrap();
        assert_eq!(
            model.parameters().flatten()["model.layers.0.conv.conv.weight"].shape(),
            [32, 1, 3]
        );
    }

    #[test]
    fn complete_poisoned_gpt_oss_safetensors_headers_are_exactly_loadable() {
        let directory = write_complete_gpt_oss_safetensors_dir(|_| {});
        let report = inspect_model(directory.path(), MlxInspectionOptions::default()).unwrap();
        assert_eq!(report.structural_binding, InspectionReadiness::Ready);
        assert_eq!(report.model_loadability, InspectionReadiness::Ready);
        assert!(report.is_loadable(), "{:#?}", report.issues);
        assert!(structural::validate_safetensors_load_path(
            ModelKind::GptOss,
            directory.path(),
            ModelLoadOptions::default(),
        )
        .is_ok());
    }

    #[test]
    fn gpt_oss_schedule_inspection_and_loader_preflight_reject_identically() {
        for mutate in [
            |config: &mut Value| {
                config["layer_types"] = json!(["sliding_attention", "full_attention"])
            },
            |config: &mut Value| config["layer_types"] = json!(["unsupported_attention"]),
            |config: &mut Value| config["sliding_window"] = json!(0),
            |config: &mut Value| config["sliding_window"] = json!(-1),
            |config: &mut Value| config["sliding_window"] = json!(i64::from(i32::MAX) + 1),
        ] {
            let directory = write_complete_gpt_oss_safetensors_dir(|_| {});
            let mut config = gpt_oss_config();
            mutate(&mut config);
            std::fs::write(
                directory.path().join("config.json"),
                serde_json::to_vec(&config).unwrap(),
            )
            .unwrap();

            let report = inspect_model(directory.path(), MlxInspectionOptions::default()).unwrap();
            assert_eq!(report.model_loadability, InspectionReadiness::Invalid);
            assert!(!report.is_loadable());
            assert!(structural::validate_safetensors_load_path(
                ModelKind::GptOss,
                directory.path(),
                ModelLoadOptions::default(),
            )
            .is_err());
        }
    }

    #[test]
    fn gpt_oss_missing_native_mxfp4_companion_is_structured() {
        let directory = write_complete_gpt_oss_safetensors_dir(|specs| {
            specs.retain(|(name, _, _)| name != "model.layers.0.mlp.experts.gate_up_proj_scales");
        });
        let report = inspect_model(directory.path(), MlxInspectionOptions::default()).unwrap();
        let issue = report
            .issues
            .iter()
            .find(|issue| issue.code == InspectionIssueCode::QuantizationCompanionMismatch)
            .unwrap();
        assert_eq!(
            issue.tensor_name.as_deref(),
            Some("model.layers.0.mlp.experts.gate_up_proj_scales")
        );
        assert!(!report.is_loadable());
    }

    #[test]
    fn gpt_oss_unexpected_safetensor_is_not_a_false_positive() {
        let directory = write_complete_gpt_oss_safetensors_dir(|specs| {
            specs.push(("poison.weight".into(), vec![1], Dtype::F32));
        });
        let report = inspect_model(directory.path(), MlxInspectionOptions::default()).unwrap();
        let issue = report
            .issues
            .iter()
            .find(|issue| {
                issue.code == InspectionIssueCode::ConflictingTensorLayout
                    && issue.tensor_name.as_deref() == Some("poison.weight")
            })
            .unwrap();
        assert!(issue.detail.contains("unexpected tensor"));
        assert!(!report.is_loadable());
    }

    #[test]
    fn complete_poisoned_kimi_linear_safetensors_layouts_are_exactly_loadable() {
        for packed in [false, true] {
            let directory = write_complete_kimi_linear_safetensors_dir(packed, |specs| {
                specs.push(("model.mtp.poison.weight".into(), vec![1]));
            });
            let report = inspect_model(directory.path(), MlxInspectionOptions::default()).unwrap();
            assert_eq!(report.structural_binding, InspectionReadiness::Ready);
            assert_eq!(report.model_loadability, InspectionReadiness::Ready);
            assert!(
                report.is_loadable(),
                "packed={packed}: {:#?}",
                report.issues
            );
            assert!(structural::validate_safetensors_load_path(
                ModelKind::KimiLinear,
                directory.path(),
                ModelLoadOptions::default(),
            )
            .is_ok());
        }
    }

    #[test]
    fn kimi_schedule_rejections_match_inspection_and_load_preflight() {
        for mutate in [
            |config: &mut Value| config["linear_attn_config"]["full_attn_layers"] = json!([]),
            |config: &mut Value| config["linear_attn_config"]["full_attn_layers"] = json!([1]),
            |config: &mut Value| config["linear_attn_config"]["kda_layers"] = json!("invalid"),
            |config: &mut Value| config["first_k_dense_replace"] = json!(-1),
            |config: &mut Value| config["first_k_dense_replace"] = json!(3),
            |config: &mut Value| config["moe_layer_freq"] = json!(0),
        ] {
            let directory = write_complete_kimi_linear_safetensors_dir(false, |_| {});
            let mut config = kimi_linear_config();
            mutate(&mut config);
            std::fs::write(
                directory.path().join("config.json"),
                serde_json::to_vec(&config).unwrap(),
            )
            .unwrap();

            let report = inspect_model(directory.path(), MlxInspectionOptions::default()).unwrap();
            assert_eq!(report.model_loadability, InspectionReadiness::Invalid);
            assert!(!report.is_loadable());
            assert!(structural::validate_safetensors_load_path(
                ModelKind::KimiLinear,
                directory.path(),
                ModelLoadOptions::default(),
            )
            .is_err());
        }
    }

    #[test]
    fn complete_poisoned_kimi_linear_gguf_catalog_is_exact_without_reading_transition_values() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("kimi-linear.gguf");
        write_complete_kimi_linear_gguf(&path);
        let report = inspect_model(&path, MlxInspectionOptions::default()).unwrap();
        assert_eq!(
            report.structural_binding,
            InspectionReadiness::Ready,
            "{:#?}",
            report.issues
        );
        assert_eq!(report.model_loadability, InspectionReadiness::Ready);
        assert!(report.is_loadable());

        let checkpoint = GgufCheckpoint::open(&path).unwrap();
        let metadata = crate::backend::mlx::runtime::checkpoint::load::gguf_metadata(&checkpoint);
        assert_eq!(
            structural::validate_gguf(
                GgufArchitecture::KimiLinear,
                &checkpoint,
                &metadata,
                ModelLoadOptions::default(),
            ),
            structural::StructuralValidation::Exact
        );
    }

    #[test]
    fn kimi_linear_missing_split_expert_is_structured() {
        let directory = write_complete_kimi_linear_safetensors_dir(false, |specs| {
            specs.retain(|(name, _)| name != "model.layers.1.block_sparse_moe.experts.3.w2.weight");
        });
        let report = inspect_model(directory.path(), MlxInspectionOptions::default()).unwrap();
        let issue = report
            .issues
            .iter()
            .find(|issue| issue.code == InspectionIssueCode::MissingRequiredTensor)
            .unwrap();
        assert_eq!(
            issue.tensor_name.as_deref(),
            Some("model.layers.1.mlp.experts.3.w2.weight")
        );
        assert!(!report.is_loadable());
    }

    #[test]
    fn kimi_linear_wrong_kda_convolution_geometry_is_rejected() {
        let directory = write_complete_kimi_linear_safetensors_dir(false, |specs| {
            specs
                .iter_mut()
                .find(|(name, _)| name == "model.layers.0.self_attn.q_conv1d.weight")
                .unwrap()
                .1 = vec![8, 3];
        });
        let report = inspect_model(directory.path(), MlxInspectionOptions::default()).unwrap();
        let issue = report
            .issues
            .iter()
            .find(|issue| issue.code == InspectionIssueCode::TensorShapeMismatch)
            .unwrap();
        assert_eq!(
            issue.tensor_name.as_deref(),
            Some("model.layers.0.self_attn.q_conv1d.weight")
        );
        assert!(!report.is_loadable());
    }

    #[test]
    fn complete_inkling_safetensors_headers_are_exactly_loadable() {
        let directory = write_complete_inkling_safetensors_dir(|_| {});
        let report = inspect_model(directory.path(), MlxInspectionOptions::default()).unwrap();
        assert_eq!(report.structural_binding, InspectionReadiness::Ready);
        assert_eq!(report.model_loadability, InspectionReadiness::Ready);
        assert!(report.is_loadable(), "{:#?}", report.issues);
        structural::validate_safetensors_load_path(
            ModelKind::Inkling,
            directory.path(),
            ModelLoadOptions::default(),
        )
        .unwrap();
    }

    #[test]
    fn inkling_schedule_rejections_match_inspection_and_load_preflight() {
        let cases = [
            (
                "attention length",
                json!(["full_attention"]),
                None,
                None,
                None,
            ),
            (
                "attention conflict",
                json!(["full_attention", "full_attention"]),
                None,
                None,
                None,
            ),
            (
                "feed-forward conflict",
                Value::Null,
                Some(json!(["moe", "moe"])),
                None,
                None,
            ),
            (
                "invalid attention kind",
                json!(["sliding", "full_attention"]),
                None,
                None,
                None,
            ),
            ("zero window", Value::Null, None, Some(json!(0)), None),
            ("negative window", Value::Null, None, Some(json!(-1)), None),
            (
                "overflowing window",
                Value::Null,
                None,
                Some(json!(i64::from(i32::MAX) + 1)),
                None,
            ),
            (
                "duplicate local layer",
                Value::Null,
                None,
                None,
                Some(json!([0, 0])),
            ),
        ];

        for (name, layer_types, mlp_layer_types, window, local_layer_ids) in cases {
            let directory = write_complete_inkling_safetensors_dir(|_| {});
            let mut config = inkling_config();
            if !layer_types.is_null() {
                config["text_config"]["layer_types"] = layer_types;
            }
            if let Some(types) = mlp_layer_types {
                config["text_config"]["mlp_layer_types"] = types;
            }
            if let Some(window) = window {
                config["text_config"]["sliding_window_size"] = window;
            }
            if let Some(ids) = local_layer_ids {
                config["text_config"]["local_layer_ids"] = ids;
            }
            std::fs::write(
                directory.path().join("config.json"),
                serde_json::to_vec(&config).unwrap(),
            )
            .unwrap();

            let report = inspect_model(directory.path(), MlxInspectionOptions::default()).unwrap();
            assert_eq!(
                report.model_loadability,
                InspectionReadiness::Invalid,
                "{name}: {:#?}",
                report.issues
            );
            assert!(
                report
                    .issues
                    .iter()
                    .any(|issue| issue.code == InspectionIssueCode::InvalidConfiguration),
                "{name}: {:#?}",
                report.issues
            );
            assert!(
                structural::validate_safetensors_load_path(
                    ModelKind::Inkling,
                    directory.path(),
                    ModelLoadOptions::default(),
                )
                .is_err(),
                "{name}"
            );
        }
    }

    #[test]
    fn inkling_missing_layer_tensor_and_wrong_w13_shape_are_structured() {
        let missing = write_complete_inkling_safetensors_dir(|specs| {
            specs.retain(|(name, _)| name != "model.llm.layers.1.attn.wq_du.weight");
        });
        let report = inspect_model(missing.path(), MlxInspectionOptions::default()).unwrap();
        assert!(!report.is_loadable());
        assert!(report.issues.iter().any(|issue| {
            issue.code == InspectionIssueCode::MissingRequiredTensor
                && issue.tensor_name.as_deref() == Some("model.llm.layers.1.attn.wq_du.weight")
        }));

        let wrong = write_complete_inkling_safetensors_dir(|specs| {
            specs
                .iter_mut()
                .find(|(name, _)| name == "model.llm.layers.0.mlp.w13_dn.weight")
                .unwrap()
                .1 = vec![31, 16];
        });
        let report = inspect_model(wrong.path(), MlxInspectionOptions::default()).unwrap();
        assert!(!report.is_loadable());
        assert!(report.issues.iter().any(|issue| {
            issue.code == InspectionIssueCode::TensorShapeMismatch
                && issue.tensor_name.as_deref() == Some("model.llm.layers.0.mlp.w13_dn.weight")
        }));
    }

    #[test]
    fn inkling_alias_conflicts_and_missing_media_towers_are_structured() {
        let conflict = write_complete_inkling_safetensors_dir(|specs| {
            specs.push(("model.embed_tokens.weight".into(), vec![32, 16]));
        });
        let report = inspect_model(conflict.path(), MlxInspectionOptions::default()).unwrap();
        assert!(!report.is_loadable());
        assert!(report.issues.iter().any(|issue| {
            issue.code == InspectionIssueCode::ConflictingTensorLayout
                && issue.tensor_name.as_deref() == Some("model.embed_tokens.weight")
        }));

        let missing_media = write_complete_inkling_safetensors_dir(|_| {});
        let mut config = inkling_config();
        config["audio_config"] = json!({
            "text_hidden_size": 16,
            "num_codebooks": 2,
            "codebook_size": 8,
            "bias": false,
            "use_audio_norm": true,
            "audio_mode": "dmel"
        });
        std::fs::write(
            missing_media.path().join("config.json"),
            serde_json::to_vec(&config).unwrap(),
        )
        .unwrap();
        let report = inspect_model(missing_media.path(), MlxInspectionOptions::default()).unwrap();
        assert!(!report.is_loadable());
        assert!(report.issues.iter().any(|issue| {
            issue.code == InspectionIssueCode::MissingRequiredTensor
                && issue.tensor_name.as_deref() == Some("model.audio.encoder.weight")
        }));
    }

    #[test]
    fn complete_poisoned_nemotron_h_safetensors_layouts_are_exactly_loadable() {
        for packed in [false, true] {
            let directory = write_complete_nemotron_h_safetensors_dir(packed, |_| {});
            let report = inspect_model(directory.path(), MlxInspectionOptions::default()).unwrap();
            assert_eq!(report.structural_binding, InspectionReadiness::Ready);
            assert_eq!(report.model_loadability, InspectionReadiness::Ready);
            assert!(
                report.is_loadable(),
                "packed={packed}: {:#?}",
                report.issues
            );
            structural::validate_safetensors_load_path(
                ModelKind::NemotronH,
                directory.path(),
                ModelLoadOptions::default(),
            )
            .unwrap();
        }
    }

    #[test]
    fn nemotron_schedule_inspection_and_loader_preflight_reject_identically() {
        for mutate in [
            |config: &mut Value| config["hybrid_override_pattern"] = json!("M-E"),
            |config: &mut Value| config["sliding_window"] = json!(0),
            |config: &mut Value| config["sliding_window"] = json!(i64::from(i32::MAX) + 1),
        ] {
            let directory = write_complete_nemotron_h_safetensors_dir(false, |_| {});
            let mut config = nemotron_h_config();
            mutate(&mut config);
            std::fs::write(
                directory.path().join("config.json"),
                serde_json::to_vec(&config).unwrap(),
            )
            .unwrap();

            let report = inspect_model(directory.path(), MlxInspectionOptions::default()).unwrap();
            assert_eq!(report.model_loadability, InspectionReadiness::Invalid);
            assert!(!report.is_loadable());
            assert!(structural::validate_safetensors_load_path(
                ModelKind::NemotronH,
                directory.path(),
                ModelLoadOptions::default(),
            )
            .is_err());
        }
    }

    #[test]
    fn nemotron_h_missing_mamba_tensor_and_wrong_expert_shape_are_structured() {
        let missing = write_complete_nemotron_h_safetensors_dir(false, |specs| {
            specs.retain(|(name, _)| name != "backbone.layers.0.mixer.conv1d.weight");
        });
        let report = inspect_model(missing.path(), MlxInspectionOptions::default()).unwrap();
        assert!(!report.is_loadable());
        assert!(report.issues.iter().any(|issue| {
            issue.code == InspectionIssueCode::MissingRequiredTensor
                && issue.tensor_name.as_deref() == Some("backbone.layers.0.mixer.conv1d.weight")
        }));

        let wrong = write_complete_nemotron_h_safetensors_dir(false, |specs| {
            specs
                .iter_mut()
                .find(|(name, _)| name == "backbone.layers.2.mixer.experts.1.down_proj.weight")
                .unwrap()
                .1 = vec![8, 5];
        });
        let report = inspect_model(wrong.path(), MlxInspectionOptions::default()).unwrap();
        assert!(!report.is_loadable());
        assert!(report.issues.iter().any(|issue| {
            issue.code == InspectionIssueCode::TensorShapeMismatch
                && issue.tensor_name.as_deref()
                    == Some("backbone.layers.2.mixer.experts.1.down_proj.weight")
        }));
    }

    #[test]
    fn nemotron_h_mixed_expert_layout_and_incomplete_native_affine_are_rejected() {
        let mixed = write_complete_nemotron_h_safetensors_dir(true, |specs| {
            specs.push((
                "backbone.layers.2.mixer.experts.0.up_proj.weight".into(),
                vec![6, 8],
            ));
        });
        let report = inspect_model(mixed.path(), MlxInspectionOptions::default()).unwrap();
        assert!(!report.is_loadable());
        assert!(report.issues.iter().any(|issue| {
            issue.code == InspectionIssueCode::ConflictingTensorLayout
                && issue.tensor_name.as_deref()
                    == Some("backbone.layers.2.mixer.experts.0.up_proj.weight")
        }));

        let affine = write_complete_nemotron_h_safetensors_dir(false, |_| {});
        let mut config = nemotron_h_config();
        config["quantization"] = json!({
            "group_size": 32,
            "bits": 4,
            "mode": "affine"
        });
        std::fs::write(
            affine.path().join("config.json"),
            serde_json::to_vec(&config).unwrap(),
        )
        .unwrap();
        let report = inspect_model(affine.path(), MlxInspectionOptions::default()).unwrap();
        assert_eq!(report.structural_binding, InspectionReadiness::Invalid);
        assert!(!report.is_loadable());
        assert!(report.issues.iter().any(|issue| matches!(
            issue.code,
            InspectionIssueCode::TensorShapeMismatch
                | InspectionIssueCode::QuantizationCompanionMismatch
        )));
    }

    #[test]
    fn complete_poisoned_nemotron_h_gguf_catalogs_are_exactly_loadable() {
        let directory = tempfile::tempdir().unwrap();
        for (name, architecture, is_moe) in [
            ("nemotron-h.gguf", GgufArchitecture::NemotronH, false),
            ("nemotron-h-moe.gguf", GgufArchitecture::NemotronHMoe, true),
        ] {
            let path = directory.path().join(name);
            write_complete_nemotron_h_gguf(&path, is_moe, |_| {});
            let report = inspect_model(&path, MlxInspectionOptions::default()).unwrap();
            assert_eq!(report.structural_binding, InspectionReadiness::Ready);
            assert_eq!(report.model_loadability, InspectionReadiness::Ready);
            assert!(report.is_loadable(), "{:#?}", report.issues);

            let checkpoint = GgufCheckpoint::open(&path).unwrap();
            let metadata =
                crate::backend::mlx::runtime::checkpoint::load::gguf_metadata(&checkpoint);
            assert_eq!(
                structural::validate_gguf(
                    architecture,
                    &checkpoint,
                    &metadata,
                    ModelLoadOptions::default(),
                ),
                structural::StructuralValidation::Exact
            );
        }
    }

    #[test]
    fn nemotron_h_gguf_missing_tensor_and_grouped_expert_shape_are_structured() {
        let directory = tempfile::tempdir().unwrap();
        let missing = directory.path().join("missing-conv.gguf");
        write_complete_nemotron_h_gguf(&missing, false, |specs| {
            specs.retain(|(name, _, _)| name != "blk.0.ssm_conv1d.weight");
        });
        let report = inspect_model(&missing, MlxInspectionOptions::default()).unwrap();
        assert!(!report.is_loadable());
        assert!(report.issues.iter().any(|issue| {
            issue.code == InspectionIssueCode::MissingRequiredTensor
                && issue.tensor_name.as_deref() == Some("blk.0.ssm_conv1d.weight")
        }));

        let wrong = directory.path().join("wrong-expert.gguf");
        write_complete_nemotron_h_gguf(&wrong, true, |specs| {
            specs
                .iter_mut()
                .find(|(name, _, _)| name == "blk.1.ffn_down_exps.weight")
                .unwrap()
                .1 = vec![6, 7, 2];
        });
        let report = inspect_model(&wrong, MlxInspectionOptions::default()).unwrap();
        assert!(!report.is_loadable());
        assert!(report.issues.iter().any(|issue| {
            issue.code == InspectionIssueCode::TensorShapeMismatch
                && issue.tensor_name.as_deref() == Some("blk.1.ffn_down_exps.weight")
        }));
    }

    #[test]
    fn nemotron_h_gguf_rejects_quantized_recurrent_state_operation() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("quantized-conv.gguf");
        write_complete_nemotron_h_gguf(&path, false, |specs| {
            specs
                .iter_mut()
                .find(|(name, _, _)| name == "blk.0.ssm_conv1d.weight")
                .unwrap()
                .2 = GgmlType::Q4_0;
        });
        let report = inspect_model(&path, MlxInspectionOptions::default()).unwrap();
        assert!(!report.is_loadable());
        assert!(report.issues.iter().any(|issue| {
            issue.code == InspectionIssueCode::UnsupportedTensorEncoding
                && issue.tensor_name.as_deref() == Some("blk.0.ssm_conv1d.weight")
                && issue.tensor_type_code == Some(GgmlType::Q4_0.code())
        }));
    }

    #[test]
    fn nemotron_h_gguf_architecture_and_expert_catalog_must_agree() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("dense-with-experts.gguf");
        let mut metadata = nemotron_h_gguf_metadata(false);
        metadata.insert("nemotron_h.expert_count".into(), MetadataValue::Uint32(2));
        let specs = nemotron_h_gguf_specs(false);
        let payloads = specs
            .iter()
            .map(|(_, dimensions, encoding)| {
                let elements = dimensions.iter().product::<u64>();
                let (block, bytes) = encoding.block_and_bytes().unwrap();
                vec![0xff; (elements / block * bytes) as usize]
            })
            .collect::<Vec<_>>();
        let tensors = specs
            .iter()
            .zip(&payloads)
            .map(|((name, dimensions, ggml_type), data)| TensorInput {
                name,
                dimensions,
                ggml_type: *ggml_type,
                data,
            })
            .collect::<Vec<_>>();
        Writer::default()
            .write(std::fs::File::create(&path).unwrap(), &metadata, &tensors)
            .unwrap();
        let report = inspect_model(&path, MlxInspectionOptions::default()).unwrap();
        assert!(!report.is_loadable());
        assert!(report.issues.iter().any(|issue| {
            issue.code == InspectionIssueCode::InvalidLayerOrExpertCount
                && issue
                    .detail
                    .contains("architecture and expert tensors disagree")
        }));
    }

    #[test]
    fn complete_poisoned_gemma4_text_safetensors_layouts_are_exactly_loadable() {
        for is_moe in [false, true] {
            let directory = write_complete_gemma4_safetensors_dir(is_moe, |_| {});
            let report = inspect_model(directory.path(), MlxInspectionOptions::default()).unwrap();
            assert_eq!(report.structural_binding, InspectionReadiness::Ready);
            assert_eq!(report.model_loadability, InspectionReadiness::Ready);
            assert!(
                report.is_loadable(),
                "is_moe={is_moe}: {:#?}",
                report.issues
            );
            structural::validate_safetensors_load_path(
                ModelKind::Gemma4,
                directory.path(),
                ModelLoadOptions::default(),
            )
            .unwrap();
        }
    }

    #[test]
    fn gemma4_schedule_inspection_and_loader_preflight_reject_identically() {
        for mutate in [
            |config: &mut Value| {
                config["text_config"]["layer_types"] = json!(["sliding_attention"])
            },
            |config: &mut Value| {
                config["text_config"]["layer_types"] =
                    json!(["sliding_attention", "full_attention"]);
                config["text_config"]
                    .as_object_mut()
                    .unwrap()
                    .remove("sliding_window");
            },
            |config: &mut Value| config["text_config"]["sliding_window"] = json!(0),
            |config: &mut Value| config["text_config"]["sliding_window"] = json!(-1),
            |config: &mut Value| {
                config["text_config"]["sliding_window"] = json!(i64::from(i32::MAX) + 1)
            },
        ] {
            let directory = write_complete_gemma4_safetensors_dir(false, |_| {});
            let mut config = gemma4_config(false);
            mutate(&mut config);
            std::fs::write(
                directory.path().join("config.json"),
                serde_json::to_vec(&config).unwrap(),
            )
            .unwrap();

            let report = inspect_model(directory.path(), MlxInspectionOptions::default()).unwrap();
            assert_eq!(report.model_loadability, InspectionReadiness::Invalid);
            assert!(!report.is_loadable());
            assert!(structural::validate_safetensors_load_path(
                ModelKind::Gemma4,
                directory.path(),
                ModelLoadOptions::default(),
            )
            .is_err());
        }
    }

    #[test]
    fn gemma4_native_affine_catalog_and_companions_are_validated_exactly() {
        let complete = write_complete_quantized_gemma4_safetensors_dir(|_| {});
        let report = inspect_model(complete.path(), MlxInspectionOptions::default()).unwrap();
        assert_eq!(report.structural_binding, InspectionReadiness::Ready);
        assert!(report.is_loadable(), "{:#?}", report.issues);
        structural::validate_safetensors_load_path(
            ModelKind::Gemma4,
            complete.path(),
            ModelLoadOptions::default(),
        )
        .unwrap();

        let missing = write_complete_quantized_gemma4_safetensors_dir(|specs| {
            specs.retain(|(name, _, _)| {
                name != "language_model.model.layers.0.self_attn.q_proj.biases"
            });
        });
        let report = inspect_model(missing.path(), MlxInspectionOptions::default()).unwrap();
        assert!(!report.is_loadable());
        assert!(report.issues.iter().any(|issue| {
            issue.code == InspectionIssueCode::QuantizationCompanionMismatch
                && issue.tensor_name.as_deref()
                    == Some("language_model.model.layers.0.self_attn.q_proj.biases")
        }));
    }

    #[test]
    fn gemma4_missing_shared_kv_boundary_tensor_and_wrong_moe_shape_are_structured() {
        let missing = write_complete_gemma4_safetensors_dir(false, |specs| {
            specs.retain(|(name, _)| {
                name != "language_model.model.layers.0.self_attn.k_proj.weight"
            });
        });
        let report = inspect_model(missing.path(), MlxInspectionOptions::default()).unwrap();
        assert!(!report.is_loadable());
        assert!(report.issues.iter().any(|issue| {
            issue.code == InspectionIssueCode::MissingRequiredTensor
                && issue.tensor_name.as_deref()
                    == Some("language_model.model.layers.0.self_attn.k_proj.weight")
        }));

        let wrong = write_complete_gemma4_safetensors_dir(true, |specs| {
            specs
                .iter_mut()
                .find(|(name, _)| {
                    name == "language_model.model.layers.1.experts.switch_glu.down_proj.weight"
                })
                .unwrap()
                .1 = vec![2, 7, 6];
        });
        let report = inspect_model(wrong.path(), MlxInspectionOptions::default()).unwrap();
        assert!(!report.is_loadable());
        assert!(report.issues.iter().any(|issue| {
            issue.code == InspectionIssueCode::TensorShapeMismatch
                && issue.tensor_name.as_deref()
                    == Some("language_model.model.layers.1.experts.switch_glu.down_proj.weight")
        }));
    }

    #[test]
    fn gemma4_alias_conflicts_are_rejected_and_media_towers_are_exact() {
        let conflict = write_complete_gemma4_safetensors_dir(false, |specs| {
            specs.push((
                "model.language_model.layers.0.self_attn.q_proj.weight".into(),
                vec![8, 8],
            ));
        });
        let report = inspect_model(conflict.path(), MlxInspectionOptions::default()).unwrap();
        assert!(!report.is_loadable());
        assert!(report.issues.iter().any(|issue| {
            issue.code == InspectionIssueCode::ConflictingTensorLayout
                && issue.tensor_name.as_deref()
                    == Some("model.language_model.layers.0.self_attn.q_proj.weight")
        }));

        let mut config = gemma4_config(false);
        config["image_token_id"] = json!(30);
        config["audio_token_id"] = json!(31);
        config["vision_config"] = json!({
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "patch_size": 2,
            "pooling_kernel_size": 2,
            "position_embedding_size": 8,
            "rms_norm_eps": 0.000001
        });
        config["audio_config"] = json!({
            "hidden_size": 8,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "output_proj_dims": 8,
            "conv_kernel_size": 3,
            "attention_chunk_size": 4,
            "attention_context_left": 4,
            "attention_context_right": 0,
            "attention_invalid_logits_value": -1000000000.0,
            "attention_logit_cap": 50.0,
            "residual_weight": 0.5,
            "rms_norm_eps": 0.000001,
            "subsampling_conv_channels": [2, 4]
        });
        let specs = gemma4_multimodal_specs()
            .into_iter()
            .map(|(name, shape)| (name, shape, Dtype::F32))
            .collect::<Vec<_>>();
        let media = write_typed_safetensors_dir(&config, &specs);
        let report = inspect_model(media.path(), MlxInspectionOptions::default()).unwrap();
        assert_eq!(
            report.structural_binding,
            InspectionReadiness::Ready,
            "{:#?}",
            report.issues
        );
        assert!(report.is_loadable());
    }

    #[test]
    fn gemma4_fused_experts_follow_the_selected_residency_route() {
        let directory = write_complete_gemma4_safetensors_dir(true, |specs| {
            for layer in 0..2 {
                let released = format!("language_model.model.layers.{layer}.experts.switch_glu");
                specs.retain(|(name, _)| {
                    name != &format!("{released}.gate_proj.weight")
                        && name != &format!("{released}.up_proj.weight")
                });
                specs.push((
                    format!(
                        "model.language_model.layers.{layer}.experts.switch_glu.gate_up_proj.weight"
                    ),
                    vec![2, 12, 8],
                ));
            }
        });

        let resident = inspect_model(directory.path(), MlxInspectionOptions::default()).unwrap();
        assert!(!resident.is_loadable());
        let load =
            ModelLoadOptions::default().with_weight_residency(WeightResidency::layerwise_host(
                crate::backend::mlx::runtime::execution::layerwise::LayerwiseLoadOptions::default(),
            ));
        let bounded = inspect_model(directory.path(), MlxInspectionOptions { load }).unwrap();
        assert_eq!(bounded.structural_binding, InspectionReadiness::Ready);
        assert!(bounded.is_loadable(), "{:#?}", bounded.issues);
        structural::validate_safetensors_load_path(ModelKind::Gemma4, directory.path(), load)
            .unwrap();
    }

    #[test]
    fn complete_poisoned_qwen3_next_safetensors_layouts_are_exactly_loadable() {
        for packed in [false, true] {
            let directory = write_complete_qwen3_next_safetensors_dir(packed, |_| {});
            let report = inspect_model(directory.path(), MlxInspectionOptions::default()).unwrap();
            assert_eq!(report.structural_binding, InspectionReadiness::Ready);
            assert_eq!(report.model_loadability, InspectionReadiness::Ready);
            assert!(
                report.is_loadable(),
                "packed={packed}: {:#?}",
                report.issues
            );
            structural::validate_safetensors_load_path(
                ModelKind::Qwen3Next,
                directory.path(),
                ModelLoadOptions::default(),
            )
            .unwrap();
        }
    }

    #[test]
    fn qwen3_next_missing_fused_projection_and_wrong_expert_shape_are_structured() {
        let missing = write_complete_qwen3_next_safetensors_dir(false, |specs| {
            specs.retain(|(name, _)| name != "model.layers.0.linear_attn.in_proj_qkvz.weight");
        });
        let report = inspect_model(missing.path(), MlxInspectionOptions::default()).unwrap();
        assert!(!report.is_loadable());
        assert!(report.issues.iter().any(|issue| {
            issue.code == InspectionIssueCode::MissingRequiredTensor
                && issue.tensor_name.as_deref()
                    == Some("model.layers.0.linear_attn.in_proj_qkvz.weight")
        }));

        let wrong = write_complete_qwen3_next_safetensors_dir(true, |specs| {
            specs
                .iter_mut()
                .find(|(name, _)| name == "model.layers.1.mlp.experts.down_proj")
                .unwrap()
                .1 = vec![2, 15, 8];
        });
        let report = inspect_model(wrong.path(), MlxInspectionOptions::default()).unwrap();
        assert!(!report.is_loadable());
        assert!(report.issues.iter().any(|issue| {
            issue.code == InspectionIssueCode::TensorShapeMismatch
                && issue.tensor_name.as_deref() == Some("model.layers.1.mlp.experts.down_proj")
        }));
    }

    #[test]
    fn qwen3_next_mixed_projection_layout_is_rejected_and_native_fp8_is_exact() {
        let mixed = write_complete_qwen3_next_safetensors_dir(false, |specs| {
            specs.push((
                "model.layers.0.linear_attn.in_proj_qkv.weight".into(),
                vec![32, 16],
            ));
        });
        let report = inspect_model(mixed.path(), MlxInspectionOptions::default()).unwrap();
        assert!(!report.is_loadable());
        assert!(report.issues.iter().any(|issue| {
            issue.code == InspectionIssueCode::ConflictingTensorLayout
                && issue.tensor_name.as_deref()
                    == Some("model.layers.0.linear_attn.in_proj_qkv.weight")
        }));

        let mut config = qwen3_next_config();
        config["quantization_config"] = json!({
            "quant_method": "fp8",
            "fmt": "e4m3",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128]
        });
        let specs = fp8_fixture_specs(qwen3_next_safetensor_specs(false), |name, shape| {
            shape.len() >= 2
                && name != "model.embed_tokens.weight"
                && name != "lm_head.weight"
                && !name.ends_with(".mlp.gate.weight")
                && !name.ends_with(".mlp.shared_expert_gate.weight")
                && !name.ends_with(".linear_attn.in_proj_ba.weight")
                && !name.ends_with(".linear_attn.conv1d.weight")
        });
        let fp8 = write_typed_safetensors_dir(&config, &specs);
        let report = inspect_model(fp8.path(), MlxInspectionOptions::default()).unwrap();
        assert_eq!(
            report.structural_binding,
            InspectionReadiness::Ready,
            "{:#?}",
            report.issues
        );
        assert!(report.is_loadable());
    }

    #[test]
    fn complete_poisoned_qwen35_text_layouts_are_exactly_loadable() {
        for (moe, packed) in [(false, false), (true, false), (true, true)] {
            let directory = write_complete_qwen35_safetensors_dir(moe, packed, |specs| {
                if moe && !packed {
                    for (name, _) in specs {
                        if let Some(rest) = name.strip_prefix("model.") {
                            *name = format!("model.language_model.{rest}");
                        }
                    }
                }
            });
            for load in [
                ModelLoadOptions::default(),
                ModelLoadOptions::default()
                    .with_weight_residency(WeightResidency::layerwise_host(Default::default())),
            ] {
                let report =
                    inspect_model(directory.path(), MlxInspectionOptions { load }).unwrap();
                assert_eq!(report.structural_binding, InspectionReadiness::Ready);
                assert_eq!(report.model_loadability, InspectionReadiness::Ready);
                assert!(
                    report.is_loadable(),
                    "moe={moe} packed={packed}: {:#?}",
                    report.issues
                );
                structural::validate_safetensors_load_path(
                    ModelKind::Qwen35,
                    directory.path(),
                    load,
                )
                .unwrap();
            }
        }
    }

    #[test]
    fn qwen35_missing_split_projection_and_fused_layout_are_structured() {
        let missing = write_complete_qwen35_safetensors_dir(true, false, |specs| {
            specs.retain(|(name, _)| name != "model.layers.0.linear_attn.in_proj_z.weight");
        });
        let report = inspect_model(missing.path(), MlxInspectionOptions::default()).unwrap();
        assert!(!report.is_loadable());
        assert!(report.issues.iter().any(|issue| {
            issue.code == InspectionIssueCode::MissingRequiredTensor
                && issue.tensor_name.as_deref()
                    == Some("model.layers.0.linear_attn.in_proj_z.weight")
        }));

        let fused = write_complete_qwen35_safetensors_dir(true, false, |specs| {
            specs.retain(|(name, _)| {
                ![
                    "in_proj_qkv.weight",
                    "in_proj_z.weight",
                    "in_proj_b.weight",
                    "in_proj_a.weight",
                ]
                .iter()
                .any(|suffix| name.ends_with(suffix))
            });
            specs.extend([
                (
                    "model.layers.0.linear_attn.in_proj_qkvz.weight".into(),
                    vec![48, 16],
                ),
                (
                    "model.layers.0.linear_attn.in_proj_ba.weight".into(),
                    vec![8, 16],
                ),
            ]);
        });
        let report = inspect_model(fused.path(), MlxInspectionOptions::default()).unwrap();
        assert!(!report.is_loadable());
        assert!(report.issues.iter().any(|issue| {
            issue.code == InspectionIssueCode::ConflictingTensorLayout
                && issue.tensor_name.as_deref()
                    == Some("model.layers.0.linear_attn.in_proj_qkvz.weight")
        }));
    }

    #[test]
    fn qwen35_vision_and_native_fp8_are_exact_and_dense_sparse_is_invalid() {
        let mut config = qwen35_config(true);
        config["vision_config"] = json!({
            "depth": 1,
            "hidden_size": 8,
            "hidden_act": "silu",
            "intermediate_size": 16,
            "num_heads": 2,
            "num_position_embeddings": 16,
            "in_channels": 3,
            "patch_size": 2,
            "spatial_merge_size": 2,
            "temporal_patch_size": 1,
            "window_size": 8,
            "out_hidden_size": 16,
            "fullatt_block_indexes": [0],
            "deepstack_visual_indexes": []
        });
        let mut vision_specs = qwen35_safetensor_specs(true, false)
            .into_iter()
            .map(|(name, shape)| (name, shape, Dtype::F32))
            .collect::<Vec<_>>();
        vision_specs.extend(
            qwen3_vl_safetensor_specs(false)
                .into_iter()
                .filter(|(name, _, _)| {
                    name.starts_with("model.visual.") && !name.contains("deepstack_merger_list")
                })
                .map(|(name, mut shape, dtype)| {
                    let name = name.replacen("model.visual.", "visual.", 1);
                    if name == "visual.patch_embed.proj.weight" {
                        shape = vec![8, 3, 1, 2, 2];
                    } else if name == "visual.merger.linear_fc2.weight" {
                        shape = vec![16, 32];
                    } else if name == "visual.merger.linear_fc2.bias" {
                        shape = vec![16];
                    }
                    (name, shape, dtype)
                }),
        );
        let vision = write_typed_safetensors_dir(&config, &vision_specs);
        let report = inspect_model(vision.path(), MlxInspectionOptions::default()).unwrap();
        assert_eq!(
            report.structural_binding,
            InspectionReadiness::Ready,
            "{:#?}",
            report.issues
        );
        assert!(report.is_loadable());

        let mut config = qwen35_config(true);
        config["quantization_config"] = json!({
            "quant_method": "fp8",
            "fmt": "e4m3",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128]
        });
        let specs = fp8_fixture_specs(qwen35_safetensor_specs(true, false), |name, shape| {
            shape.len() >= 2
                && name != "model.embed_tokens.weight"
                && name != "lm_head.weight"
                && !name.ends_with(".mlp.gate.weight")
                && !name.ends_with(".mlp.shared_expert_gate.weight")
                && !name.ends_with(".linear_attn.conv1d.weight")
        });
        let fp8 = write_typed_safetensors_dir(&config, &specs);
        let report = inspect_model(fp8.path(), MlxInspectionOptions::default()).unwrap();
        assert_eq!(
            report.structural_binding,
            InspectionReadiness::Ready,
            "{:#?}",
            report.issues
        );
        assert!(report.is_loadable());

        let dense = write_complete_qwen35_safetensors_dir(false, false, |_| {});
        let load =
            ModelLoadOptions::default().with_weight_residency(WeightResidency::with_expert_cache(
                NonExpertWeightResidency::LayerwiseHost(Default::default()),
                Default::default(),
            ));
        let report = inspect_model(dense.path(), MlxInspectionOptions { load }).unwrap();
        assert!(!report.is_loadable());
        assert_eq!(report.structural_binding, InspectionReadiness::Invalid);
    }

    #[test]
    fn dense_lfm2_missing_shortconv_kernel_is_structured() {
        let directory = write_complete_lfm2_safetensors_dir(|specs| {
            specs.retain(|(name, _)| name != "model.layers.0.conv.conv.weight");
        });
        let report = inspect_model(directory.path(), MlxInspectionOptions::default()).unwrap();
        let issue = report
            .issues
            .iter()
            .find(|issue| issue.code == InspectionIssueCode::MissingRequiredTensor)
            .unwrap();
        assert_eq!(
            issue.tensor_name.as_deref(),
            Some("model.layers.0.conv.conv.weight")
        );
        assert!(!report.is_loadable());
    }

    #[test]
    fn dense_qwen3_missing_q_norm_has_a_structured_tensor_diagnostic() {
        let directory = write_complete_qwen3_safetensors_dir(|specs| {
            specs.retain(|(name, _)| name != "model.layers.0.self_attn.q_norm.weight");
        });
        let report = inspect_model(directory.path(), MlxInspectionOptions::default()).unwrap();
        let issue = report
            .issues
            .iter()
            .find(|issue| issue.code == InspectionIssueCode::MissingRequiredTensor)
            .unwrap();
        assert_eq!(
            issue.tensor_name.as_deref(),
            Some("model.layers.0.self_attn.q_norm.weight")
        );
        assert!(!report.is_loadable());
    }

    #[test]
    fn missing_layer_tensor_has_a_structured_tensor_diagnostic() {
        let directory = write_complete_safetensors_dir(|specs| {
            specs.retain(|(name, _)| name != "model.layers.0.self_attn.q_proj.weight");
        });
        let report = inspect_model(directory.path(), MlxInspectionOptions::default()).unwrap();
        let issue = report
            .issues
            .iter()
            .find(|issue| issue.code == InspectionIssueCode::MissingRequiredTensor)
            .unwrap();
        assert_eq!(
            issue.tensor_name.as_deref(),
            Some("model.layers.0.self_attn.q_proj.weight")
        );
        assert!(!report.is_loadable());
    }

    #[test]
    fn wrong_safetensors_shape_is_rejected_from_headers() {
        let directory = write_complete_safetensors_dir(|specs| {
            specs
                .iter_mut()
                .find(|(name, _)| name == "model.layers.0.self_attn.q_proj.weight")
                .unwrap()
                .1 = vec![7, 8];
        });
        let report = inspect_model(directory.path(), MlxInspectionOptions::default()).unwrap();
        let issue = report
            .issues
            .iter()
            .find(|issue| issue.code == InspectionIssueCode::TensorShapeMismatch)
            .unwrap();
        assert_eq!(
            issue.tensor_name.as_deref(),
            Some("model.layers.0.self_attn.q_proj.weight")
        );
        assert!(!report.is_loadable());
    }

    #[test]
    fn quantization_companion_mismatches_are_structured() {
        let mut config = llama_config();
        config["quantization"] = json!({
            "group_size": 32,
            "bits": 4,
            "mode": "affine"
        });
        let directory = write_safetensors_dir(&config);
        let report = inspect_model(directory.path(), MlxInspectionOptions::default()).unwrap();
        assert!(report
            .issues
            .iter()
            .any(|issue| issue.code == InspectionIssueCode::QuantizationCompanionMismatch));
        assert!(!report.is_loadable());
    }

    #[test]
    fn validation_unavailable_warning_is_always_fail_closed() {
        let directory = tempfile::tempdir().unwrap();
        let mut report =
            ModelInspectionReport::unverified(directory.path(), ArtifactFormat::SafeTensors);
        report.container = InspectionReadiness::Ready;
        report.architecture_support = InspectionReadiness::Ready;
        report.structural_binding = InspectionReadiness::Ready;
        report.model_loadability = InspectionReadiness::Ready;
        report.requested_load = InspectionReadiness::Ready;
        report.issue(
            InspectionIssueCode::ValidationUnavailableUntilLoad,
            InspectionSeverity::Warning,
            "validator intentionally unavailable",
            None,
        );
        assert!(!report.is_loadable());
    }

    #[test]
    fn qwen3_moe_packed_safetensors_route_is_exact() {
        let directory = write_complete_qwen3_moe_safetensors_dir(false, |_| {});
        let report = inspect_model(directory.path(), MlxInspectionOptions::default()).unwrap();
        assert_eq!(report.architecture_support, InspectionReadiness::Ready);
        assert_eq!(report.structural_binding, InspectionReadiness::Ready);
        assert_eq!(report.model_loadability, InspectionReadiness::Ready);
        assert_eq!(report.requested_load, InspectionReadiness::Ready);
        assert!(report.is_loadable());
    }

    #[test]
    fn qwen3_moe_split_safetensors_are_exact_for_all_residencies() {
        let directory = write_complete_qwen3_moe_safetensors_dir(true, |_| {});
        let resident = inspect_model(directory.path(), MlxInspectionOptions::default()).unwrap();
        assert_eq!(resident.structural_binding, InspectionReadiness::Ready);
        assert_eq!(resident.model_loadability, InspectionReadiness::Ready);
        assert!(resident.is_loadable());

        let bounded = inspect_model(
            directory.path(),
            MlxInspectionOptions {
                load: ModelLoadOptions::default()
                    .with_weight_residency(WeightResidency::layerwise_host(Default::default())),
            },
        )
        .unwrap();
        assert_eq!(bounded.structural_binding, InspectionReadiness::Ready);
        assert_eq!(bounded.model_loadability, InspectionReadiness::Ready);
        assert!(bounded.is_loadable());
    }

    #[test]
    fn qwen3_moe_separately_packed_safetensors_are_exact_for_all_residencies() {
        let directory = write_complete_qwen3_moe_safetensors_dir(false, |specs| {
            specs.retain(|(name, _)| !name.ends_with("experts.gate_up_proj"));
            specs.extend([
                (
                    "model.layers.0.mlp.experts.gate_proj".into(),
                    vec![4, 8, 32],
                ),
                ("model.layers.0.mlp.experts.up_proj".into(), vec![4, 8, 32]),
            ]);
        });
        let resident = inspect_model(directory.path(), MlxInspectionOptions::default()).unwrap();
        assert_eq!(resident.structural_binding, InspectionReadiness::Ready);
        assert_eq!(resident.model_loadability, InspectionReadiness::Ready);
        assert!(resident.is_loadable());

        let bounded = inspect_model(
            directory.path(),
            MlxInspectionOptions {
                load: ModelLoadOptions::default()
                    .with_weight_residency(WeightResidency::layerwise_host(Default::default())),
            },
        )
        .unwrap();
        assert_eq!(bounded.structural_binding, InspectionReadiness::Ready);
        assert_eq!(bounded.model_loadability, InspectionReadiness::Ready);
        assert!(bounded.is_loadable());
    }

    #[test]
    fn qwen3_moe_packed_shape_mismatch_is_structured() {
        let directory = write_complete_qwen3_moe_safetensors_dir(false, |specs| {
            specs
                .iter_mut()
                .find(|(name, _)| name.ends_with("experts.gate_up_proj"))
                .unwrap()
                .1 = vec![4, 16];
        });
        let report = inspect_model(directory.path(), MlxInspectionOptions::default()).unwrap();
        let issue = report
            .issues
            .iter()
            .find(|issue| issue.code == InspectionIssueCode::TensorShapeMismatch)
            .unwrap();
        assert_eq!(
            issue.tensor_name.as_deref(),
            Some("model.layers.0.mlp.experts.gate_up_proj")
        );
        assert!(!report.is_loadable());
    }

    #[test]
    fn lfm2_moe_packed_and_split_safetensors_routes_are_exact() {
        for split_experts in [false, true] {
            let directory = write_complete_lfm2_moe_safetensors_dir(split_experts, |_| {});
            let report = inspect_model(directory.path(), MlxInspectionOptions::default()).unwrap();
            assert_eq!(report.architecture_support, InspectionReadiness::Ready);
            assert_eq!(report.structural_binding, InspectionReadiness::Ready);
            assert_eq!(report.model_loadability, InspectionReadiness::Ready);
            assert_eq!(report.requested_load, InspectionReadiness::Ready);
            assert!(report.is_loadable());
        }
    }

    #[test]
    fn lfm2_moe_missing_split_expert_tensor_is_structured() {
        let directory = write_complete_lfm2_moe_safetensors_dir(true, |specs| {
            specs.retain(|(name, _)| name != "model.layers.1.feed_forward.experts.2.w2.weight");
        });
        let report = inspect_model(directory.path(), MlxInspectionOptions::default()).unwrap();
        let issue = report
            .issues
            .iter()
            .find(|issue| issue.code == InspectionIssueCode::MissingRequiredTensor)
            .unwrap();
        assert_eq!(
            issue.tensor_name.as_deref(),
            Some("model.layers.1.feed_forward.experts.2.w2.weight")
        );
        assert!(!report.is_loadable());
    }

    #[test]
    fn checkpoint_native_quantized_qwen3_and_lfm2_moe_are_exact() {
        for family in ["qwen3", "lfm2"] {
            let mut config = if family == "qwen3" {
                qwen3_config(true)
            } else {
                lfm2_config(true)
            };
            config["quantization"] = json!({
                "group_size": 16,
                "bits": 4,
                "mode": "affine"
            });
            let mut specs = if family == "qwen3" {
                config["moe_intermediate_size"] = json!(32);
                let mut specs = qwen3_moe_safetensor_specs(false);
                specs.iter_mut().for_each(|(name, shape)| {
                    if name.ends_with("experts.gate_up_proj") {
                        *shape = vec![4, 64, 32];
                    } else if name.ends_with("experts.down_proj") {
                        *shape = vec![4, 32, 32];
                    }
                });
                specs
            } else {
                config["moe_intermediate_size"] = json!(32);
                config["intermediate_size"] = json!(64);
                let mut specs = lfm2_moe_safetensor_specs(false);
                specs.iter_mut().for_each(|(name, shape)| {
                    if name.ends_with("experts.gate_up_proj") {
                        *shape = vec![4, 64, 32];
                    } else if name.ends_with("experts.down_proj") {
                        *shape = vec![4, 32, 32];
                    } else if name.ends_with("feed_forward.w1.weight")
                        || name.ends_with("feed_forward.w3.weight")
                    {
                        *shape = vec![64, 32];
                    } else if name.ends_with("feed_forward.w2.weight") {
                        *shape = vec![32, 64];
                    }
                });
                specs
            };
            specs.sort_by(|left, right| left.0.cmp(&right.0));
            let specs = affine_fixture_specs(specs, 16, |name, shape| {
                shape.len() >= 2
                    && !name.ends_with(".conv.conv.weight")
                    && !name.ends_with(".mlp.gate.weight")
                    && !name.ends_with(".feed_forward.gate.weight")
            });
            let directory = write_typed_safetensors_dir(&config, &specs);
            let report = inspect_model(directory.path(), MlxInspectionOptions::default()).unwrap();
            assert_eq!(report.architecture_support, InspectionReadiness::Ready);
            assert_eq!(
                report.structural_binding,
                InspectionReadiness::Ready,
                "{family}: {:#?}",
                report.issues
            );
            assert_eq!(report.model_loadability, InspectionReadiness::Ready);
            assert!(report.is_loadable());
        }
    }

    #[test]
    fn deepseek_v3_split_safetensors_route_is_exact() {
        let directory = write_complete_deepseek_v3_safetensors_dir(false, |_| {});
        let report = inspect_model(directory.path(), MlxInspectionOptions::default()).unwrap();
        assert_eq!(report.architecture_support, InspectionReadiness::Ready);
        assert_eq!(report.structural_binding, InspectionReadiness::Ready);
        assert_eq!(report.model_loadability, InspectionReadiness::Ready);
        assert!(report.is_loadable());
    }

    #[test]
    fn deepseek_embedded_mtp_catalog_is_exact_and_missing_fusion_fails_preflight() {
        let options = MlxInspectionOptions {
            load: ModelLoadOptions::default()
                .with_weight_residency(WeightResidency::layerwise_host(Default::default())),
        };
        let complete = write_complete_deepseek_v3_mtp_safetensors_dir(|_| {});
        let report = inspect_model(complete.path(), options.clone()).unwrap();
        assert_eq!(
            report.structural_binding,
            InspectionReadiness::Ready,
            "{:#?}",
            report.issues
        );
        assert!(report.is_loadable());

        let missing = write_complete_deepseek_v3_mtp_safetensors_dir(|specs| {
            specs.retain(|(name, _)| name != "model.layers.2.eh_proj.weight");
        });
        let report = inspect_model(missing.path(), options).unwrap();
        assert_eq!(report.structural_binding, InspectionReadiness::Invalid);
        assert!(report.issues.iter().any(|issue| {
            issue.code == InspectionIssueCode::MissingRequiredTensor
                && issue.tensor_name.as_deref() == Some("model.layers.2.eh_proj.weight")
        }));
    }

    #[test]
    fn unconfigured_deepseek_mtp_tensors_are_not_silently_ignored() {
        let directory = write_complete_deepseek_v3_safetensors_dir(false, |specs| {
            specs.push(("model.layers.2.eh_proj.weight".into(), vec![8, 16]));
        });
        let report = inspect_model(directory.path(), MlxInspectionOptions::default()).unwrap();
        assert_eq!(report.structural_binding, InspectionReadiness::Invalid);
        assert!(report.issues.iter().any(|issue| {
            issue.code == InspectionIssueCode::ConflictingTensorLayout
                && issue.tensor_name.as_deref() == Some("model.layers.2.eh_proj.weight")
        }));
    }

    #[test]
    fn deepseek_schedule_rejections_match_inspection_and_load_preflight() {
        for mutate in [
            |config: &mut Value| config["first_k_dense_replace"] = json!(-1),
            |config: &mut Value| config["first_k_dense_replace"] = json!(3),
            |config: &mut Value| config["moe_layer_freq"] = json!(0),
        ] {
            let directory = write_complete_deepseek_v3_safetensors_dir(false, |_| {});
            let mut config = deepseek_v3_config(false);
            mutate(&mut config);
            std::fs::write(
                directory.path().join("config.json"),
                serde_json::to_vec(&config).unwrap(),
            )
            .unwrap();

            let report = inspect_model(directory.path(), MlxInspectionOptions::default()).unwrap();
            assert_eq!(report.model_loadability, InspectionReadiness::Invalid);
            assert!(!report.is_loadable());
            assert!(structural::validate_safetensors_load_path(
                ModelKind::DeepSeekV3,
                directory.path(),
                ModelLoadOptions::default(),
            )
            .is_err());
        }
    }

    #[test]
    fn deepseek_v3_poison_only_checkpoint_is_not_loadable() {
        let directory = write_safetensors_dir(&deepseek_v3_config(false));
        let report = inspect_model(directory.path(), MlxInspectionOptions::default()).unwrap();
        assert_eq!(report.architecture_support, InspectionReadiness::Ready);
        assert_eq!(report.structural_binding, InspectionReadiness::Invalid);
        assert!(report
            .issues
            .iter()
            .any(|issue| issue.code == InspectionIssueCode::MissingRequiredTensor));
        assert!(report.issues.iter().any(|issue| {
            issue.code == InspectionIssueCode::ConflictingTensorLayout
                && issue.tensor_name.as_deref() == Some("poison.weight")
        }));
        assert!(!report.is_loadable());
    }

    #[test]
    fn deepseek_v3_packed_safetensors_are_residency_sensitive() {
        let directory = write_complete_deepseek_v3_safetensors_dir(true, |_| {});
        let resident = inspect_model(directory.path(), MlxInspectionOptions::default()).unwrap();
        assert_eq!(resident.structural_binding, InspectionReadiness::Invalid);
        assert!(resident
            .issues
            .iter()
            .any(|issue| issue.code == InspectionIssueCode::ConflictingTensorLayout));
        assert!(!resident.is_loadable());

        let bounded = inspect_model(
            directory.path(),
            MlxInspectionOptions {
                load: ModelLoadOptions::default()
                    .with_weight_residency(WeightResidency::layerwise_host(Default::default())),
            },
        )
        .unwrap();
        assert_eq!(bounded.structural_binding, InspectionReadiness::Ready);
        assert_eq!(bounded.model_loadability, InspectionReadiness::Ready);
        assert!(bounded.is_loadable());
    }

    #[test]
    fn deepseek_v3_bounded_route_accepts_projection_level_packed_and_split_mix() {
        let directory = write_complete_deepseek_v3_safetensors_dir(true, |specs| {
            specs.retain(|(name, _)| name != "model.layers.1.mlp.experts.up_proj");
            for expert in 0..4 {
                specs.push((
                    format!("model.layers.1.mlp.experts.{expert}.up_proj.weight"),
                    vec![4, 8],
                ));
            }
        });
        let report = inspect_model(
            directory.path(),
            MlxInspectionOptions {
                load: ModelLoadOptions::default()
                    .with_weight_residency(WeightResidency::layerwise_host(Default::default())),
            },
        )
        .unwrap();
        assert_eq!(report.structural_binding, InspectionReadiness::Ready);
        assert!(report.is_loadable());
    }

    #[test]
    fn deepseek_v3_missing_split_expert_and_wrong_packed_shape_are_structured() {
        let split = write_complete_deepseek_v3_safetensors_dir(false, |specs| {
            specs.retain(|(name, _)| name != "model.layers.1.mlp.experts.2.down_proj.weight");
        });
        let report = inspect_model(split.path(), MlxInspectionOptions::default()).unwrap();
        assert!(report.issues.iter().any(|issue| {
            issue.code == InspectionIssueCode::MissingRequiredTensor
                && issue.tensor_name.as_deref()
                    == Some("model.layers.1.mlp.experts.2.down_proj.weight")
        }));
        assert!(!report.is_loadable());

        let packed = write_complete_deepseek_v3_safetensors_dir(true, |specs| {
            specs
                .iter_mut()
                .find(|(name, _)| name == "model.layers.1.mlp.experts.gate_proj")
                .unwrap()
                .1 = vec![4, 4];
        });
        let report = inspect_model(
            packed.path(),
            MlxInspectionOptions {
                load: ModelLoadOptions::default()
                    .with_weight_residency(WeightResidency::layerwise_host(Default::default())),
            },
        )
        .unwrap();
        assert!(report.issues.iter().any(|issue| {
            issue.code == InspectionIssueCode::TensorShapeMismatch
                && issue.tensor_name.as_deref() == Some("model.layers.1.mlp.experts.gate_proj")
        }));
        assert!(!report.is_loadable());
    }

    #[test]
    fn checkpoint_native_deepseek_v3_fp8_is_exact() {
        let config = deepseek_v3_config(true);
        let specs = fp8_fixture_specs(deepseek_v3_safetensor_specs(false), |name, shape| {
            shape.len() >= 2
                && name != "model.embed_tokens.weight"
                && name != "lm_head.weight"
                && !name.ends_with(".mlp.gate.weight")
        });
        let directory = write_typed_safetensors_dir(&config, &specs);
        let report = inspect_model(directory.path(), MlxInspectionOptions::default()).unwrap();
        assert_eq!(report.architecture_support, InspectionReadiness::Ready);
        assert_eq!(
            report.structural_binding,
            InspectionReadiness::Ready,
            "{:#?}",
            report.issues
        );
        assert_eq!(report.model_loadability, InspectionReadiness::Ready);
        assert!(report.is_loadable());
    }

    #[test]
    fn safetensors_config_rejections_are_structured() {
        let unsupported = write_safetensors_dir(&json!({"model_type": "falcon"}));
        let report = inspect_model(unsupported.path(), MlxInspectionOptions::default()).unwrap();
        assert_eq!(report.model_loadability, InspectionReadiness::Unsupported);
        assert!(report
            .issues
            .iter()
            .any(|issue| issue.code == InspectionIssueCode::UnsupportedArchitecture));

        let invalid = write_safetensors_dir(&json!({"model_type": "llama", "hidden_size": "bad"}));
        let report = inspect_model(invalid.path(), MlxInspectionOptions::default()).unwrap();
        assert_eq!(report.model_loadability, InspectionReadiness::Invalid);
        assert!(report
            .issues
            .iter()
            .any(|issue| issue.code == InspectionIssueCode::InvalidConfiguration));
        assert!(!resolve_model_config(&json!({"model_type":"llama","hidden_size":"bad"})).is_ok());
        assert!(resolve_model_config(&json!({}))
            .unwrap_err()
            .to_string()
            .starts_with("invalid model config metadata:"));
    }

    #[test]
    fn missing_safetensors_index_shard_is_rejected_before_load() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(
            directory.path().join("config.json"),
            serde_json::to_vec(&llama_config()).unwrap(),
        )
        .unwrap();
        std::fs::write(
            directory.path().join("model.safetensors.index.json"),
            br#"{"weight_map":{"model.embed_tokens.weight":"missing-00001-of-00002.safetensors"}}"#,
        )
        .unwrap();
        let report = inspect_model(directory.path(), MlxInspectionOptions::default()).unwrap();
        assert_eq!(report.container, InspectionReadiness::Invalid);
        assert!(report
            .issues
            .iter()
            .any(|issue| issue.code == InspectionIssueCode::MissingCheckpointShard));
    }

    #[test]
    fn supported_and_unsupported_gguf_architectures_are_distinguished() {
        let directory = tempfile::tempdir().unwrap();
        let supported = directory.path().join("supported.gguf");
        write_gguf(
            &supported,
            "llama",
            GgmlType::Q4K,
            std::iter::empty::<(String, MetadataValue)>(),
        );
        let report = inspect_model(&supported, MlxInspectionOptions::default()).unwrap();
        assert_eq!(report.container, InspectionReadiness::Ready);
        assert_eq!(report.model_loadability, InspectionReadiness::Invalid);
        assert!(!report.is_loadable());
        assert_eq!(report.tensor_encodings[0].ggml_type_code, Some(12));

        let unsupported = directory.path().join("unsupported.gguf");
        write_gguf(
            &unsupported,
            "falcon",
            GgmlType::F16,
            std::iter::empty::<(String, MetadataValue)>(),
        );
        let report = inspect_model(&unsupported, MlxInspectionOptions::default()).unwrap();
        assert_eq!(report.model_loadability, InspectionReadiness::Unsupported);
        assert!(report
            .issues
            .iter()
            .any(|issue| issue.code == InspectionIssueCode::UnsupportedArchitecture));
    }

    #[test]
    fn complete_poisoned_llama_gguf_headers_are_exactly_loadable() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("complete.gguf");
        write_complete_gguf(&path, |_| {});
        let report = inspect_model(&path, MlxInspectionOptions::default()).unwrap();
        assert_eq!(report.structural_binding, InspectionReadiness::Ready);
        assert_eq!(report.model_loadability, InspectionReadiness::Ready);
        assert!(report.is_loadable());

        let checkpoint = GgufCheckpoint::open(&path).unwrap();
        let metadata = crate::backend::mlx::runtime::checkpoint::load::gguf_metadata(&checkpoint);
        assert_eq!(
            structural::validate_gguf(
                GgufArchitecture::Llama,
                &checkpoint,
                &metadata,
                ModelLoadOptions::default(),
            ),
            structural::StructuralValidation::Exact
        );
    }

    #[test]
    fn complete_poisoned_qwen3_gguf_headers_are_exactly_loadable() {
        let directory = tempfile::tempdir().unwrap();
        for (name, architecture, is_moe) in [
            ("qwen3.gguf", GgufArchitecture::Qwen3, false),
            ("qwen3moe.gguf", GgufArchitecture::Qwen3Moe, true),
        ] {
            let path = directory.path().join(name);
            write_complete_qwen3_gguf(&path, is_moe, |_| {});
            let report = inspect_model(&path, MlxInspectionOptions::default()).unwrap();
            assert_eq!(report.structural_binding, InspectionReadiness::Ready);
            assert_eq!(report.model_loadability, InspectionReadiness::Ready);
            assert!(report.is_loadable(), "{:#?}", report.issues);

            let checkpoint = GgufCheckpoint::open(&path).unwrap();
            let metadata =
                crate::backend::mlx::runtime::checkpoint::load::gguf_metadata(&checkpoint);
            assert_eq!(
                structural::validate_gguf(
                    architecture,
                    &checkpoint,
                    &metadata,
                    ModelLoadOptions::default(),
                ),
                structural::StructuralValidation::Exact
            );
        }
    }

    #[test]
    fn gemma4_moe_gguf_initializes_standalone_independent_expert_cache() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("gemma4-moe.gguf");
        write_quantizable_gemma4_moe_gguf(&path);
        let execution =
            safemlx::ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
        let weights =
            safemlx::ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let quantization =
            crate::backend::mlx::runtime::checkpoint::quantization::WeightQuantization::Affine(
                crate::backend::mlx::runtime::checkpoint::quantization::AffineQuantization::new(
                    32, 4,
                )
                .unwrap(),
            );
        let options = ModelLoadOptions::with_quantization(quantization)
            .with_weight_residency(WeightResidency::with_expert_cache(
            NonExpertWeightResidency::FullyResident,
            crate::backend::mlx::runtime::residency::expert_cache::ExpertCacheLoadOptions::default(
            ),
        ));
        let backend = crate::backend::mlx::MlxBackend::new(execution.stream(), weights.stream());
        let mut loaded = crate::load_model(&backend, &path, options)
            .unwrap()
            .into_inner()
            .into_complete()
            .unwrap();
        let crate::backend::mlx::Model::Gemma4(model) = &loaded else {
            panic!("expected Gemma 4 model");
        };
        let ordinary = model.metadata().materialization().unwrap();
        assert!(ordinary.transformed_weights > 0);
        assert!(ordinary.output_bytes < ordinary.source_bytes_read);
        let report = loaded.expert_cache_report().unwrap().unwrap();
        assert_eq!(report.owned_experts, 6);
        assert_eq!(report.host_resident_experts, 0);
        assert_eq!(report.device_resident_experts, 0);
        let experts = report.materialization.unwrap();
        assert_eq!(experts.transformed_weights, 18);
        assert!(experts.output_bytes < experts.source_bytes_read);
        let tokens = safemlx::Array::from_slice(&[1u32, 2], &[1, 2]);
        let parts =
            [crate::backend::mlx::runtime::media::input::InputPart::text_token_ids(&tokens)];
        let mut cache = loaded.new_cache();
        let logits = loaded
            .submit_prefill(
                crate::backend::mlx::runtime::media::input::ModelInput::new(&parts),
                &mut cache,
                execution.stream(),
            )
            .unwrap()
            .wait()
            .unwrap();
        assert_eq!(logits.shape(), &[1, 16]);
        assert!(
            loaded
                .expert_cache_report()
                .unwrap()
                .unwrap()
                .prefill
                .compact_banks
                > 0
        );
    }

    #[test]
    fn complete_poisoned_gemma4_gguf_catalogs_are_exactly_loadable() {
        let directory = tempfile::tempdir().unwrap();
        for (name, is_moe) in [("gemma4.gguf", false), ("gemma4-moe.gguf", true)] {
            let path = directory.path().join(name);
            write_complete_gemma4_gguf(&path, is_moe, |_| {});
            let report = inspect_model(&path, MlxInspectionOptions::default()).unwrap();
            assert_eq!(report.structural_binding, InspectionReadiness::Ready);
            assert_eq!(report.model_loadability, InspectionReadiness::Ready);
            assert!(
                report.is_loadable(),
                "is_moe={is_moe}: {:#?}",
                report.issues
            );

            let checkpoint = GgufCheckpoint::open(&path).unwrap();
            let metadata =
                crate::backend::mlx::runtime::checkpoint::load::gguf_metadata(&checkpoint);
            assert_eq!(
                structural::validate_gguf(
                    GgufArchitecture::Gemma4,
                    &checkpoint,
                    &metadata,
                    ModelLoadOptions::default(),
                ),
                structural::StructuralValidation::Exact
            );
            crate::architectures::gemma4::model::prepare_gemma4_gguf_checkpoint(
                &checkpoint,
                &metadata,
                None,
                None,
            )
            .unwrap();
        }
    }

    #[test]
    fn gemma4_gguf_missing_tensor_and_expert_shape_are_structured() {
        let directory = tempfile::tempdir().unwrap();
        let missing = directory.path().join("missing-q-norm.gguf");
        write_complete_gemma4_gguf(&missing, false, |specs| {
            specs.retain(|(name, _, _)| name != "blk.0.attn_q_norm.weight");
        });
        let report = inspect_model(&missing, MlxInspectionOptions::default()).unwrap();
        assert!(!report.is_loadable());
        assert!(report.issues.iter().any(|issue| {
            issue.code == InspectionIssueCode::MissingRequiredTensor
                && issue.tensor_name.as_deref() == Some("blk.0.attn_q_norm.weight")
        }));

        let wrong = directory.path().join("wrong-expert.gguf");
        write_complete_gemma4_gguf(&wrong, true, |specs| {
            specs
                .iter_mut()
                .find(|(name, _, _)| name == "blk.1.ffn_down_exps.weight")
                .unwrap()
                .1 = vec![4, 7, 2];
        });
        let report = inspect_model(&wrong, MlxInspectionOptions::default()).unwrap();
        assert!(!report.is_loadable());
        assert!(report.issues.iter().any(|issue| {
            issue.code == InspectionIssueCode::TensorShapeMismatch
                && issue.tensor_name.as_deref() == Some("blk.1.ffn_down_exps.weight")
        }));
    }

    #[test]
    fn gemma4_gguf_expert_layout_and_operation_encodings_are_validated() {
        let directory = tempfile::tempdir().unwrap();
        let mixed = directory.path().join("mixed-experts.gguf");
        write_complete_gemma4_gguf(&mixed, true, |specs| {
            specs.push((
                "blk.0.ffn_gate_exps.weight".into(),
                vec![8, 4, 2],
                GgmlType::F32,
            ));
        });
        let report = inspect_model(&mixed, MlxInspectionOptions::default()).unwrap();
        assert!(!report.is_loadable());
        assert!(report.issues.iter().any(|issue| {
            issue.code == InspectionIssueCode::ConflictingTensorLayout
                && issue.tensor_name.as_deref() == Some("blk.0.ffn_gate_up_exps.weight")
        }));

        let quantized_vector = directory.path().join("quantized-layer-scale.gguf");
        write_complete_gemma4_gguf(&quantized_vector, false, |specs| {
            let tensor = specs
                .iter_mut()
                .find(|(name, _, _)| name == "blk.0.layer_output_scale.weight")
                .unwrap();
            tensor.1 = vec![32];
            tensor.2 = GgmlType::Q4_0;
        });
        let report = inspect_model(&quantized_vector, MlxInspectionOptions::default()).unwrap();
        assert!(!report.is_loadable());
        assert!(report.issues.iter().any(|issue| {
            issue.code == InspectionIssueCode::UnsupportedTensorEncoding
                && issue.tensor_name.as_deref() == Some("blk.0.layer_output_scale.weight")
                && issue.tensor_type_code == Some(GgmlType::Q4_0.code())
        }));
    }

    #[test]
    fn complete_poisoned_inkling_gguf_catalog_is_exactly_loadable() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("inkling.gguf");
        write_inkling_gguf(
            &path,
            &inkling_gguf_metadata(),
            inkling_gguf_specs(),
            |specs| {
                specs
                    .iter_mut()
                    .find(|(name, _, _)| name == "blk.1.ffn_exp_probs_b.bias")
                    .unwrap()
                    .0 = "blk.1.exp_probs_b.bias".into();
            },
        );
        let report = inspect_model(&path, MlxInspectionOptions::default()).unwrap();
        assert_eq!(report.structural_binding, InspectionReadiness::Ready);
        assert_eq!(report.model_loadability, InspectionReadiness::Ready);
        assert_eq!(report.multimodal, InspectionReadiness::Missing);
        assert!(report.is_loadable(), "{:#?}", report.issues);

        let checkpoint = GgufCheckpoint::open(&path).unwrap();
        let metadata = crate::backend::mlx::runtime::checkpoint::load::gguf_metadata(&checkpoint);
        assert_eq!(
            structural::validate_gguf(
                GgufArchitecture::Inkling,
                &checkpoint,
                &metadata,
                ModelLoadOptions::default(),
            ),
            structural::StructuralValidation::Exact
        );
        crate::architectures::inkling::model::prepare_gguf_checkpoint_with_mmproj(
            &checkpoint,
            &metadata,
            None,
        )
        .unwrap();
    }

    #[test]
    fn inkling_gguf_schedule_rejections_match_inspection_and_load_preflight() {
        let cases = [
            (
                "pattern length",
                "inkling.attention.sliding_window_pattern",
                MetadataValue::Array(MetadataArray::Bool(vec![true])),
            ),
            (
                "pattern encoding",
                "inkling.attention.sliding_window_pattern",
                MetadataValue::Array(MetadataArray::Uint32(vec![1, 0])),
            ),
            (
                "zero window",
                "inkling.attention.sliding_window",
                MetadataValue::Uint32(0),
            ),
            (
                "overflowing window",
                "inkling.attention.sliding_window",
                MetadataValue::Uint32(i32::MAX as u32 + 1),
            ),
            (
                "dense layer overflow",
                "inkling.dense_block_count",
                MetadataValue::Uint32(3),
            ),
        ];
        let directory = tempfile::tempdir().unwrap();

        for (index, (name, key, value)) in cases.into_iter().enumerate() {
            let path = directory
                .path()
                .join(format!("invalid-schedule-{index}.gguf"));
            let mut metadata = inkling_gguf_metadata();
            metadata.insert(key.into(), value);
            write_inkling_gguf(&path, &metadata, inkling_gguf_specs(), |_| {});

            let report = inspect_model(&path, MlxInspectionOptions::default()).unwrap();
            assert_eq!(
                report.model_loadability,
                InspectionReadiness::Invalid,
                "{name}: {:#?}",
                report.issues
            );
            assert!(
                report.issues.iter().any(|issue| matches!(
                    issue.code,
                    InspectionIssueCode::InvalidConfiguration
                        | InspectionIssueCode::InvalidLayerOrExpertCount
                )),
                "{name}: {:#?}",
                report.issues
            );

            let checkpoint = GgufCheckpoint::open(&path).unwrap();
            let loaded_metadata =
                crate::backend::mlx::runtime::checkpoint::load::gguf_metadata(&checkpoint);
            assert!(
                matches!(
                    structural::validate_gguf(
                        GgufArchitecture::Inkling,
                        &checkpoint,
                        &loaded_metadata,
                        ModelLoadOptions::default(),
                    ),
                    structural::StructuralValidation::Invalid(_)
                ),
                "{name}"
            );
        }
    }

    #[test]
    fn inkling_gguf_missing_tensor_and_expert_shape_are_structured() {
        let directory = tempfile::tempdir().unwrap();
        let missing = directory.path().join("missing-shortconv.gguf");
        write_inkling_gguf(
            &missing,
            &inkling_gguf_metadata(),
            inkling_gguf_specs(),
            |specs| specs.retain(|(name, _, _)| name != "blk.0.shortconv_k.weight"),
        );
        let report = inspect_model(&missing, MlxInspectionOptions::default()).unwrap();
        assert!(!report.is_loadable());
        assert!(report.issues.iter().any(|issue| {
            issue.code == InspectionIssueCode::MissingRequiredTensor
                && issue.tensor_name.as_deref() == Some("blk.0.shortconv_k.weight")
        }));

        let wrong = directory.path().join("wrong-expert.gguf");
        write_inkling_gguf(
            &wrong,
            &inkling_gguf_metadata(),
            inkling_gguf_specs(),
            |specs| {
                specs
                    .iter_mut()
                    .find(|(name, _, _)| name == "blk.1.ffn_down_exps.weight")
                    .unwrap()
                    .1 = vec![32, 31, 4];
            },
        );
        let report = inspect_model(&wrong, MlxInspectionOptions::default()).unwrap();
        assert!(!report.is_loadable());
        assert!(report.issues.iter().any(|issue| {
            issue.code == InspectionIssueCode::TensorShapeMismatch
                && issue.tensor_name.as_deref() == Some("blk.1.ffn_down_exps.weight")
        }));
    }

    #[test]
    fn inkling_gguf_operation_and_paired_expert_encodings_are_validated() {
        let directory = tempfile::tempdir().unwrap();
        let convolution = directory.path().join("quantized-shortconv.gguf");
        write_inkling_gguf(
            &convolution,
            &inkling_gguf_metadata(),
            inkling_gguf_specs(),
            |specs| {
                let tensor = specs
                    .iter_mut()
                    .find(|(name, _, _)| name == "blk.0.shortconv_attn.weight")
                    .unwrap();
                tensor.1 = vec![32, 32];
                tensor.2 = GgmlType::Q4_0;
            },
        );
        let report = inspect_model(&convolution, MlxInspectionOptions::default()).unwrap();
        assert!(!report.is_loadable());
        assert!(report.issues.iter().any(|issue| {
            issue.code == InspectionIssueCode::UnsupportedTensorEncoding
                && issue.tensor_name.as_deref() == Some("blk.0.shortconv_attn.weight")
                && issue.tensor_type_code == Some(GgmlType::Q4_0.code())
        }));

        let paired = directory.path().join("mismatched-experts.gguf");
        write_inkling_gguf(
            &paired,
            &inkling_gguf_metadata(),
            inkling_gguf_specs(),
            |specs| {
                specs
                    .iter_mut()
                    .find(|(name, _, _)| name == "blk.1.ffn_gate_exps.weight")
                    .unwrap()
                    .2 = GgmlType::Q4_0;
                specs
                    .iter_mut()
                    .find(|(name, _, _)| name == "blk.1.ffn_up_exps.weight")
                    .unwrap()
                    .2 = GgmlType::Q8_0;
            },
        );
        let report = inspect_model(&paired, MlxInspectionOptions::default()).unwrap();
        assert!(!report.is_loadable());
        assert!(report.issues.iter().any(|issue| {
            issue.code == InspectionIssueCode::QuantizationCompanionMismatch
                && issue.tensor_name.as_deref() == Some("blk.1.ffn_gate_exps.weight")
        }));
    }

    #[test]
    fn inkling_sibling_mmproj_is_validated_before_model_admission() {
        let complete = tempfile::tempdir().unwrap();
        let model_path = complete.path().join("inkling.gguf");
        let projector_path = complete.path().join("mmproj-inkling-q4_0.gguf");
        write_inkling_gguf(
            &model_path,
            &inkling_gguf_metadata(),
            inkling_gguf_specs(),
            |_| {},
        );
        write_inkling_gguf(
            &projector_path,
            &inkling_mmproj_metadata(),
            inkling_mmproj_specs(),
            |_| {},
        );
        let report = inspect_model(&model_path, MlxInspectionOptions::default()).unwrap();
        assert!(report.is_loadable(), "{:#?}", report.issues);
        let checkpoint = GgufCheckpoint::open(&model_path).unwrap();
        let metadata = crate::backend::mlx::runtime::checkpoint::load::gguf_metadata(&checkpoint);
        let mmproj = crate::architectures::inkling::model::open_sibling_mmproj(&model_path)
            .unwrap()
            .unwrap();
        assert_eq!(
            structural::validate_inkling_mmproj_gguf(&metadata, &mmproj),
            structural::StructuralValidation::Exact
        );
        crate::architectures::inkling::model::prepare_gguf_checkpoint_with_mmproj(
            &checkpoint,
            &metadata,
            Some(&mmproj),
        )
        .unwrap();

        let incomplete = tempfile::tempdir().unwrap();
        let model_path = incomplete.path().join("inkling.gguf");
        let projector_path = incomplete.path().join("mmproj-inkling-q4_0.gguf");
        write_inkling_gguf(
            &model_path,
            &inkling_gguf_metadata(),
            inkling_gguf_specs(),
            |_| {},
        );
        write_inkling_gguf(
            &projector_path,
            &inkling_mmproj_metadata(),
            inkling_mmproj_specs(),
            |specs| specs.retain(|(name, _, _)| name != "v.hmlp.2.linear.weight"),
        );
        let report = inspect_model(&model_path, MlxInspectionOptions::default()).unwrap();
        assert!(!report.is_loadable());
        assert!(report.issues.iter().any(|issue| {
            issue.code == InspectionIssueCode::MissingRequiredTensor
                && issue.tensor_name.as_deref() == Some("v.hmlp.2.linear.weight")
        }));
    }

    #[test]
    fn complete_poisoned_qwen35_gguf_catalogs_are_exactly_loadable() {
        let directory = tempfile::tempdir().unwrap();
        for (name, architecture, is_moe) in [
            ("qwen35.gguf", GgufArchitecture::Qwen35, false),
            ("qwen35moe.gguf", GgufArchitecture::Qwen35Moe, true),
        ] {
            let path = directory.path().join(name);
            write_complete_qwen35_gguf(&path, is_moe, |_| {});
            let report = inspect_model(&path, MlxInspectionOptions::default()).unwrap();
            assert_eq!(report.structural_binding, InspectionReadiness::Ready);
            assert_eq!(report.model_loadability, InspectionReadiness::Ready);
            assert!(report.is_loadable(), "{:#?}", report.issues);

            let checkpoint = GgufCheckpoint::open(&path).unwrap();
            let metadata =
                crate::backend::mlx::runtime::checkpoint::load::gguf_metadata(&checkpoint);
            assert_eq!(
                structural::validate_gguf(
                    architecture,
                    &checkpoint,
                    &metadata,
                    ModelLoadOptions::default(),
                ),
                structural::StructuralValidation::Exact
            );
        }
    }

    #[test]
    fn qwen35_q8_sibling_projector_is_bound_and_validated() {
        let directory = tempfile::tempdir().unwrap();
        let model = directory.path().join("Qwen3.5-model-F16.gguf");
        let projector = directory.path().join("mmproj-Qwen3.5-Q8_0.gguf");
        write_complete_qwen35_gguf(&model, false, |_| {});
        write_gguf_specs(
            &projector,
            &qwen35_mmproj_metadata(),
            &qwen35_mmproj_specs(),
        );

        let report = inspect_model(&model, MlxInspectionOptions::default()).unwrap();
        assert_eq!(report.structural_binding, InspectionReadiness::Ready);
        assert_eq!(
            report.multimodal,
            if cfg!(feature = "image-processing") {
                InspectionReadiness::Ready
            } else {
                InspectionReadiness::Unsupported
            }
        );
        assert!(report.requirements.iter().any(|requirement| {
            requirement.code == InspectionIssueCode::MissingMediaProjector
                && requirement.readiness == InspectionReadiness::Ready
                && requirement.path.as_deref() == Some(projector.as_path())
        }));

        let checkpoint = GgufCheckpoint::open(&model).unwrap();
        let metadata = crate::backend::mlx::runtime::checkpoint::load::gguf_metadata(&checkpoint);
        let mmproj = crate::architectures::qwen::hybrid::qwen3_5::open_sibling_mmproj(&model)
            .unwrap()
            .unwrap();
        assert_eq!(
            structural::validate_qwen35_projector_gguf(
                &checkpoint,
                &metadata,
                &mmproj.checkpoint,
                &mmproj.metadata,
            ),
            structural::StructuralValidation::Exact
        );
        let execution =
            safemlx::ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let prepared = crate::architectures::qwen::hybrid::qwen3_5::prepare_qwen35_gguf_checkpoint(
            &checkpoint,
            &metadata,
            Some(&mmproj),
            execution.stream(),
        )
        .unwrap();
        assert_eq!(prepared.modalities.image_token_id, Some(30));
        assert_eq!(prepared.modalities.video_token_id, Some(31));
        let vision = prepared.modalities.vision_config.as_ref().unwrap();
        assert_eq!(vision.quantized_weight_configs.len(), 6);
        let store = crate::architectures::qwen::hybrid::layerwise::qwen_hybrid_gguf_store(
            &checkpoint,
            Some(&mmproj),
            Some(vision),
            1,
        )
        .unwrap();
        assert!(store.keys().contains(&"model.embed_tokens.weight".into()));
        assert!(store
            .keys()
            .contains(&"visual.blocks.0.attn.qkv.weight".into()));
        assert_eq!(store.diagnostics().unwrap().physical_reads, 0);

        let invalid_directory = tempfile::tempdir().unwrap();
        let invalid_model = invalid_directory.path().join("Qwen3.5-model-F16.gguf");
        let invalid_projector = invalid_directory.path().join("mmproj-Qwen3.5-Q8_0.gguf");
        write_complete_qwen35_gguf(&invalid_model, false, |_| {});
        let mut specs = qwen35_mmproj_specs();
        specs.retain(|(name, _, _)| name != "mm.2.weight");
        write_gguf_specs(&invalid_projector, &qwen35_mmproj_metadata(), &specs);
        let report = inspect_model(&invalid_model, MlxInspectionOptions::default()).unwrap();
        assert_eq!(report.multimodal, InspectionReadiness::Invalid);
        assert!(report.issues.iter().any(|issue| {
            issue.code == InspectionIssueCode::MissingRequiredTensor
                && issue.tensor_name.as_deref() == Some("mm.2.weight")
        }));
    }

    #[test]
    fn qwen35_gguf_missing_tensor_and_shape_mismatch_are_structured() {
        let directory = tempfile::tempdir().unwrap();
        let missing = directory.path().join("missing-qkv.gguf");
        write_complete_qwen35_gguf(&missing, false, |specs| {
            specs.retain(|(name, _, _)| name != "blk.0.attn_qkv.weight");
        });
        let report = inspect_model(&missing, MlxInspectionOptions::default()).unwrap();
        assert!(!report.is_loadable());
        assert!(report.issues.iter().any(|issue| {
            issue.code == InspectionIssueCode::MissingRequiredTensor
                && issue.tensor_name.as_deref() == Some("blk.0.attn_qkv.weight")
        }));

        let wrong = directory.path().join("wrong-expert.gguf");
        write_complete_qwen35_gguf(&wrong, true, |specs| {
            specs
                .iter_mut()
                .find(|(name, _, _)| name == "blk.1.ffn_down_exps.weight")
                .unwrap()
                .1 = vec![16, 31, 2];
        });
        let report = inspect_model(&wrong, MlxInspectionOptions::default()).unwrap();
        assert!(!report.is_loadable());
        assert!(report.issues.iter().any(|issue| {
            issue.code == InspectionIssueCode::TensorShapeMismatch
                && issue.tensor_name.as_deref() == Some("blk.1.ffn_down_exps.weight")
        }));
    }

    #[test]
    fn qwen35_gguf_operation_and_grouped_expert_encodings_are_validated() {
        let directory = tempfile::tempdir().unwrap();
        let conv = directory.path().join("quantized-conv.gguf");
        write_complete_qwen35_gguf(&conv, false, |specs| {
            let tensor = specs
                .iter_mut()
                .find(|(name, _, _)| name == "blk.0.ssm_conv1d.weight")
                .unwrap();
            tensor.1 = vec![32, 64];
            tensor.2 = GgmlType::Q4_0;
        });
        let report = inspect_model(&conv, MlxInspectionOptions::default()).unwrap();
        assert!(!report.is_loadable());
        assert!(report.issues.iter().any(|issue| {
            issue.code == InspectionIssueCode::UnsupportedTensorEncoding
                && issue.tensor_name.as_deref() == Some("blk.0.ssm_conv1d.weight")
                && issue.tensor_type_code == Some(GgmlType::Q4_0.code())
        }));

        let experts = directory.path().join("mismatched-experts.gguf");
        write_complete_qwen35_gguf(&experts, true, |specs| {
            specs
                .iter_mut()
                .find(|(name, _, _)| name == "blk.0.ffn_gate_exps.weight")
                .unwrap()
                .2 = GgmlType::Q4_0;
            specs
                .iter_mut()
                .find(|(name, _, _)| name == "blk.0.ffn_up_exps.weight")
                .unwrap()
                .2 = GgmlType::Q8_0;
        });
        let report = inspect_model(&experts, MlxInspectionOptions::default()).unwrap();
        assert!(!report.is_loadable());
        assert!(report.issues.iter().any(|issue| {
            issue.code == InspectionIssueCode::QuantizationCompanionMismatch
                && issue.tensor_name.as_deref() == Some("blk.0.ffn_gate_exps.weight")
        }));

        let grouped = directory.path().join("unaligned-value-head.gguf");
        write_complete_qwen35_gguf(&grouped, false, |specs| {
            specs
                .iter_mut()
                .find(|(name, _, _)| name == "blk.0.ssm_out.weight")
                .unwrap()
                .2 = GgmlType::Q5_0;
        });
        let report = inspect_model(&grouped, MlxInspectionOptions::default()).unwrap();
        assert!(!report.is_loadable());
        assert!(report.issues.iter().any(|issue| {
            issue.code == InspectionIssueCode::QuantizationCompanionMismatch
                && issue.tensor_name.as_deref() == Some("blk.0.ssm_out.weight")
                && issue.tensor_type_code == Some(GgmlType::Q5_0.code())
        }));
    }

    #[test]
    fn complete_poisoned_qwen3_next_gguf_catalog_is_exactly_loadable() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("qwen3next.gguf");
        write_complete_qwen3_next_gguf(&path, |_| {}, |_| {});
        let report = inspect_model(&path, MlxInspectionOptions::default()).unwrap();
        assert_eq!(report.structural_binding, InspectionReadiness::Ready);
        assert_eq!(report.model_loadability, InspectionReadiness::Ready);
        assert!(report.is_loadable(), "{:#?}", report.issues);

        let checkpoint = GgufCheckpoint::open(&path).unwrap();
        let metadata = crate::backend::mlx::runtime::checkpoint::load::gguf_metadata(&checkpoint);
        assert_eq!(
            structural::validate_gguf(
                GgufArchitecture::Qwen3Next,
                &checkpoint,
                &metadata,
                ModelLoadOptions::default(),
            ),
            structural::StructuralValidation::Exact
        );
    }

    #[test]
    fn qwen3_next_gguf_requires_fused_projection_catalog() {
        let directory = tempfile::tempdir().unwrap();
        let missing = directory.path().join("missing-qkvz.gguf");
        write_complete_qwen3_next_gguf(
            &missing,
            |_| {},
            |specs| {
                specs.retain(|(name, _, _)| name != "blk.0.attn_qkvz.weight");
            },
        );
        let report = inspect_model(&missing, MlxInspectionOptions::default()).unwrap();
        assert!(!report.is_loadable());
        assert!(report.issues.iter().any(|issue| {
            issue.code == InspectionIssueCode::MissingRequiredTensor
                && issue.tensor_name.as_deref() == Some("blk.0.attn_qkvz.weight")
        }));

        let mixed = directory.path().join("mixed-projections.gguf");
        write_complete_qwen3_next_gguf(
            &mixed,
            |_| {},
            |specs| {
                specs.push(("blk.0.attn_qkv.weight".into(), vec![32, 64], GgmlType::F32));
            },
        );
        let report = inspect_model(&mixed, MlxInspectionOptions::default()).unwrap();
        assert!(!report.is_loadable());
        assert!(report.issues.iter().any(|issue| {
            issue.code == InspectionIssueCode::ConflictingTensorLayout
                && issue.tensor_name.as_deref() == Some("blk.0.attn_qkv.weight")
        }));
    }

    #[test]
    fn qwen3_next_gguf_fused_affine_input_alignment_is_validated() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("unaligned-fused.gguf");
        write_complete_qwen3_next_gguf(
            &path,
            |metadata| {
                metadata.insert(
                    "qwen3next.embedding_length".into(),
                    MetadataValue::Uint32(16),
                );
            },
            |specs| {
                let tensor = specs
                    .iter_mut()
                    .find(|(name, _, _)| name == "blk.0.attn_qkvz.weight")
                    .unwrap();
                tensor.2 = GgmlType::Q4_0;
            },
        );
        let report = inspect_model(&path, MlxInspectionOptions::default()).unwrap();
        assert!(!report.is_loadable());
        assert!(report.issues.iter().any(|issue| {
            issue.code == InspectionIssueCode::QuantizationCompanionMismatch
                && issue.tensor_name.as_deref() == Some("blk.0.attn_qkvz.weight")
                && issue.tensor_type_code == Some(GgmlType::Q4_0.code())
        }));
    }

    #[test]
    fn complete_deepseek4_gguf_catalog_is_exactly_loadable() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("deepseek4.gguf");
        write_complete_deepseek4_gguf(&path, |_| {}, |_| {});
        let report = inspect_model(&path, MlxInspectionOptions::default()).unwrap();
        assert_eq!(report.structural_binding, InspectionReadiness::Ready);
        assert_eq!(report.model_loadability, InspectionReadiness::Ready);
        assert!(report.is_loadable(), "{:#?}", report.issues);

        let checkpoint = GgufCheckpoint::open(&path).unwrap();
        let metadata = crate::backend::mlx::runtime::checkpoint::load::gguf_metadata(&checkpoint);
        assert_eq!(
            structural::validate_gguf(
                GgufArchitecture::DeepSeek4,
                &checkpoint,
                &metadata,
                ModelLoadOptions::default(),
            ),
            structural::StructuralValidation::Exact
        );
        let args = deepseek_v4::model_args_from_gguf_catalog(&checkpoint, &metadata).unwrap();
        assert_eq!(args.compress_ratios, [0, 4, 128]);
        assert_eq!(args.num_hash_layers, 1);
        assert_eq!(args.quantized_weight_configs.as_ref().unwrap().len(), 9);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn canonical_deepseek4_gguf_binds_through_the_shared_metal_layerwise_loader() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("deepseek4.gguf");
        write_complete_deepseek4_gguf(&path, |_| {}, |_| {});
        let checkpoint = GgufCheckpoint::open(&path).unwrap();
        let metadata = crate::backend::mlx::runtime::checkpoint::load::gguf_metadata(&checkpoint);
        let context =
            safemlx::ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
        let weights =
            safemlx::ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        for residency in [
            WeightResidency::fully_resident(),
            WeightResidency::layerwise_host(
                crate::backend::mlx::runtime::execution::layerwise::LayerwiseLoadOptions::default(),
            ),
            WeightResidency::dense_disk_stream(crate::backend::mlx::runtime::residency::dense_stream::DenseDiskStreamLoadOptions::default()),
            WeightResidency::with_expert_cache(
                NonExpertWeightResidency::FullyResident,
                crate::backend::mlx::runtime::residency::expert_cache::ExpertCacheLoadOptions::default(),
            ),
        ] {
            let (model, _) =
                crate::architectures::deepseek_v4::layerwise::load_deepseek_v4_gguf_layerwise_model(
                    &checkpoint,
                    &metadata,
                    residency,
                    None,
                    context.stream(),
                    weights.stream(),
                )
                .unwrap();
            let cache = model
                .new_cache_with_options(crate::backend::mlx::runtime::cache::residency::CacheResidencyPolicy::Paged(
                    crate::backend::mlx::runtime::cache::residency::PagedCacheOptions::new(2, 1 << 20, 1 << 20, 1).unwrap(),
                ))
                .unwrap();
            assert!(cache.residency_report().unwrap().is_some());
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn canonical_deepseek4_gguf_preflights_every_cartesian_pipeline_rank() {
        use crate::backend::mlx::{DeviceAssignment, MlxParallelContext};

        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("deepseek4.gguf");
        write_complete_deepseek4_gguf(&path, |_| {}, |_| {});
        let execution =
            safemlx::ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
        let weights =
            safemlx::ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        for rank in 0..8 {
            let topology = MlxParallelContext::for_rank(
                rank,
                2,
                2,
                2,
                DeviceAssignment::new(safemlx::DeviceType::Gpu, 0),
            )
            .unwrap();
            for residency in [
                WeightResidency::fully_resident(),
                WeightResidency::with_expert_cache(
                    NonExpertWeightResidency::DenseDiskStream(
                        crate::backend::mlx::runtime::residency::dense_stream::DenseDiskStreamLoadOptions::new(u64::MAX, u64::MAX, 1, 1).unwrap(),
                    ),
                    crate::backend::mlx::runtime::residency::expert_cache::ExpertCacheLoadOptions::default(),
                ),
            ] {
                let loaded =
                    crate::architectures::distributed::pipeline::load_pipeline_model_with_options(
                        &path,
                        ModelLoadOptions::with_parallel(topology).with_weight_residency(residency),
                        execution.stream(),
                        weights.stream(),
                    )
                    .unwrap();
                assert_eq!(loaded.stage_info().topology, topology);
                assert!(!loaded.stage_info().global_layer_range.is_empty());
                let cache = loaded
                    .new_cache_with_options(crate::backend::mlx::runtime::cache::residency::CacheResidencyPolicy::Paged(
                        crate::backend::mlx::runtime::cache::residency::PagedCacheOptions::new(2, 1 << 20, 1 << 20, 1).unwrap(),
                    ))
                    .unwrap();
                assert!(loaded.cache_residency_report(&cache).unwrap().is_some());
            }
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn canonical_deepseek4_gguf_preflights_every_tensor_expert_rank() {
        use crate::backend::mlx::{DeviceAssignment, MlxParallelContext};

        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("deepseek4.gguf");
        write_complete_deepseek4_gguf(&path, |_| {}, |_| {});
        let execution =
            safemlx::ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
        let weights =
            safemlx::ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        for rank in 0..4 {
            let topology = MlxParallelContext::for_rank(
                rank,
                2,
                1,
                2,
                DeviceAssignment::new(safemlx::DeviceType::Gpu, 0),
            )
            .unwrap();
            for residency in [
                WeightResidency::fully_resident(),
                WeightResidency::with_expert_cache(
                    NonExpertWeightResidency::DenseDiskStream(
                        crate::backend::mlx::runtime::residency::dense_stream::DenseDiskStreamLoadOptions::new(u64::MAX, u64::MAX, 1, 1).unwrap(),
                    ),
                    crate::backend::mlx::runtime::residency::expert_cache::ExpertCacheLoadOptions::default(),
                ),
            ] {
                let loaded = crate::architectures::distributed::expert::
                    load_expert_parallel_model_with_options(
                        &path,
                        ModelLoadOptions::with_parallel(topology)
                            .with_weight_residency(residency),
                        execution.stream(),
                        weights.stream(),
                    )
                    .unwrap();
                assert_eq!(loaded.info().topology, topology);
                assert_eq!(loaded.info().assignment.local_expert_count(), 1);
                let cache = loaded
                    .new_cache_with_options(crate::backend::mlx::runtime::cache::residency::CacheResidencyPolicy::Paged(
                        crate::backend::mlx::runtime::cache::residency::PagedCacheOptions::new(2, 1 << 20, 1 << 20, 1).unwrap(),
                    ))
                    .unwrap();
                assert!(loaded.cache_residency_report(&cache).unwrap().is_some());
            }
        }
    }

    #[test]
    fn deepseek4_gguf_exact_catalog_rejects_missing_alias_and_extra_tensors() {
        let directory = tempfile::tempdir().unwrap();
        for (name, mutate) in [("missing.gguf", 0_u8), ("alias.gguf", 1), ("extra.gguf", 2)] {
            let path = directory.path().join(name);
            write_complete_deepseek4_gguf(
                &path,
                |_| {},
                |specs| match mutate {
                    0 => specs.retain(|(name, _, _)| name != "blk.1.indexer_compressor_kv.weight"),
                    1 => {
                        specs
                            .iter_mut()
                            .find(|(name, _, _)| name == "blk.1.indexer.proj.weight")
                            .unwrap()
                            .0 = "blk.1.indexer.weights_proj.weight".into();
                    }
                    _ => specs.push(("blk.0.legacy.weight".into(), vec![32], GgmlType::F32)),
                },
            );
            let report = inspect_model(&path, MlxInspectionOptions::default()).unwrap();
            assert!(!report.is_loadable(), "{name}: {:#?}", report.issues);
        }
    }

    #[test]
    fn deepseek4_gguf_requires_canonical_special_encodings_and_metadata_geometry() {
        let directory = tempfile::tempdir().unwrap();
        for (name, tensor_name, encoding) in [
            (
                "dense-experts.gguf",
                "blk.2.ffn_gate_exps.weight",
                GgmlType::F32,
            ),
            (
                "float-hash.gguf",
                "blk.0.ffn_gate_tid2eid.weight",
                GgmlType::F32,
            ),
        ] {
            let path = directory.path().join(name);
            write_complete_deepseek4_gguf(
                &path,
                |_| {},
                |specs| {
                    specs
                        .iter_mut()
                        .find(|(name, _, _)| name == tensor_name)
                        .unwrap()
                        .2 = encoding;
                },
            );
            let report = inspect_model(&path, MlxInspectionOptions::default()).unwrap();
            assert!(!report.is_loadable());
            assert!(report.issues.iter().any(|issue| {
                issue.code == InspectionIssueCode::UnsupportedTensorEncoding
                    && issue.tensor_name.as_deref() == Some(tensor_name)
            }));
        }

        let bad_output = directory.path().join("bad-output-width.gguf");
        write_complete_deepseek4_gguf(
            &bad_output,
            |metadata| {
                metadata.insert(
                    "deepseek4.embedding_length_out".into(),
                    MetadataValue::Uint32(32),
                );
            },
            |_| {},
        );
        let report = inspect_model(&bad_output, MlxInspectionOptions::default()).unwrap();
        assert!(!report.is_loadable());
        assert!(report.issues.iter().any(|issue| {
            issue.code == InspectionIssueCode::InvalidLayerOrExpertCount
                && issue.detail.contains("embedding_length_out")
        }));
    }

    #[test]
    fn complete_poisoned_deepseek2_gguf_catalogs_are_exactly_loadable() {
        let directory = tempfile::tempdir().unwrap();
        for (name, split_kv) in [("fused.gguf", false), ("split.gguf", true)] {
            let path = directory.path().join(name);
            write_complete_deepseek2_gguf(&path, split_kv, |_| {});
            let report = inspect_model(&path, MlxInspectionOptions::default()).unwrap();
            assert_eq!(report.structural_binding, InspectionReadiness::Ready);
            assert_eq!(report.model_loadability, InspectionReadiness::Ready);
            assert!(report.is_loadable(), "{:#?}", report.issues);

            let checkpoint = GgufCheckpoint::open(&path).unwrap();
            let metadata =
                crate::backend::mlx::runtime::checkpoint::load::gguf_metadata(&checkpoint);
            assert_eq!(
                structural::validate_gguf(
                    GgufArchitecture::DeepSeek2,
                    &checkpoint,
                    &metadata,
                    ModelLoadOptions::default(),
                ),
                structural::StructuralValidation::Exact
            );
        }
    }

    #[test]
    fn minimal_deepseek2_gguf_is_not_authoritatively_loadable() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("minimal.gguf");
        write_complete_deepseek2_gguf(&path, false, |specs| {
            specs.retain(|(name, _, _)| name == "token_embd.weight");
        });
        let report = inspect_model(&path, MlxInspectionOptions::default()).unwrap();
        assert!(!report.is_loadable());
        assert!(
            report
                .issues
                .iter()
                .any(|issue| issue.code == InspectionIssueCode::MissingRequiredTensor),
            "{:#?}",
            report.issues
        );
    }

    #[test]
    fn deepseek2_split_mla_shape_mismatch_is_structured() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("wrong-split-k.gguf");
        write_complete_deepseek2_gguf(&path, true, |specs| {
            specs
                .iter_mut()
                .find(|(name, _, _)| name == "blk.0.attn_k_b.weight")
                .unwrap()
                .1 = vec![32, 32];
        });
        let report = inspect_model(&path, MlxInspectionOptions::default()).unwrap();
        assert!(!report.is_loadable());
        assert!(report.issues.iter().any(|issue| {
            issue.code == InspectionIssueCode::TensorShapeMismatch
                && issue.tensor_name.as_deref() == Some("blk.0.attn_k_b.weight")
        }));
    }

    #[test]
    fn deepseek2_rejects_quantized_norm_but_accepts_quantized_expert_matrix() {
        let directory = tempfile::tempdir().unwrap();
        let invalid = directory.path().join("quantized-norm.gguf");
        write_complete_deepseek2_gguf(&invalid, false, |specs| {
            specs
                .iter_mut()
                .find(|(name, _, _)| name == "output_norm.weight")
                .unwrap()
                .2 = GgmlType::Q4_0;
        });
        let report = inspect_model(&invalid, MlxInspectionOptions::default()).unwrap();
        let issue = report
            .issues
            .iter()
            .find(|issue| {
                issue.code == InspectionIssueCode::UnsupportedTensorEncoding
                    && issue.tensor_name.as_deref() == Some("output_norm.weight")
            })
            .unwrap();
        assert_eq!(issue.tensor_type_code, Some(GgmlType::Q4_0.code()));
        assert!(!report.is_loadable());

        let valid = directory.path().join("quantized-expert.gguf");
        write_complete_deepseek2_gguf(&valid, false, |specs| {
            specs
                .iter_mut()
                .find(|(name, _, _)| name == "blk.1.ffn_gate_exps.weight")
                .unwrap()
                .2 = GgmlType::Q4_0;
        });
        let report = inspect_model(&valid, MlxInspectionOptions::default()).unwrap();
        assert!(report.is_loadable(), "{:#?}", report.issues);
    }

    #[test]
    fn deepseek2_rejects_unexpected_logical_tensors() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("extra.gguf");
        write_complete_deepseek2_gguf(&path, false, |specs| {
            specs.push(("poison.weight".into(), vec![32], GgmlType::F32));
        });
        let report = inspect_model(&path, MlxInspectionOptions::default()).unwrap();
        assert!(!report.is_loadable());
        assert!(report.issues.iter().any(|issue| {
            issue.code == InspectionIssueCode::ConflictingTensorLayout
                && issue.tensor_name.as_deref() == Some("poison.weight")
        }));
    }

    #[test]
    fn complete_poisoned_lfm2_gguf_headers_are_exactly_loadable() {
        let directory = tempfile::tempdir().unwrap();
        for (name, architecture, is_moe) in [
            ("lfm2.gguf", GgufArchitecture::Lfm2, false),
            ("lfm2moe.gguf", GgufArchitecture::Lfm2Moe, true),
        ] {
            let path = directory.path().join(name);
            write_complete_lfm2_gguf(&path, is_moe, |specs| {
                if is_moe {
                    specs
                        .iter_mut()
                        .find(|(name, _, _)| name == "blk.1.ffn_exp_probs_b.bias")
                        .unwrap()
                        .0 = "blk.1.exp_probs_b.bias".into();
                }
            });
            let report = inspect_model(&path, MlxInspectionOptions::default()).unwrap();
            assert_eq!(report.structural_binding, InspectionReadiness::Ready);
            assert_eq!(report.model_loadability, InspectionReadiness::Ready);
            assert!(report.is_loadable(), "{:#?}", report.issues);

            let checkpoint = GgufCheckpoint::open(&path).unwrap();
            let metadata =
                crate::backend::mlx::runtime::checkpoint::load::gguf_metadata(&checkpoint);
            assert_eq!(
                structural::validate_gguf(
                    architecture,
                    &checkpoint,
                    &metadata,
                    ModelLoadOptions::default(),
                ),
                structural::StructuralValidation::Exact
            );
        }
    }

    #[test]
    fn complete_poisoned_gpt_oss_gguf_headers_are_exactly_loadable() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("gpt-oss.gguf");
        write_complete_gpt_oss_gguf(&path, |_| {});
        let report = inspect_model(&path, MlxInspectionOptions::default()).unwrap();
        assert_eq!(report.structural_binding, InspectionReadiness::Ready);
        assert_eq!(report.model_loadability, InspectionReadiness::Ready);
        assert!(report.is_loadable(), "{:#?}", report.issues);

        let checkpoint = GgufCheckpoint::open(&path).unwrap();
        let metadata = crate::backend::mlx::runtime::checkpoint::load::gguf_metadata(&checkpoint);
        assert_eq!(
            structural::validate_gguf(
                GgufArchitecture::GptOss,
                &checkpoint,
                &metadata,
                ModelLoadOptions::default(),
            ),
            structural::StructuralValidation::Exact
        );
    }

    #[test]
    fn gpt_oss_requires_mxfp4_for_routed_gguf_operations() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("gpt-oss-q4-expert.gguf");
        write_complete_gpt_oss_gguf(&path, |specs| {
            specs
                .iter_mut()
                .find(|(name, _, _)| name == "blk.0.ffn_up_exps.weight")
                .unwrap()
                .2 = GgmlType::Q4_0;
        });
        let report = inspect_model(&path, MlxInspectionOptions::default()).unwrap();
        let issue = report
            .issues
            .iter()
            .find(|issue| {
                issue.code == InspectionIssueCode::UnsupportedTensorEncoding
                    && issue.tensor_name.as_deref() == Some("blk.0.ffn_up_exps.weight")
            })
            .unwrap();
        assert_eq!(issue.tensor_type_code, Some(GgmlType::Q4_0.code()));
        assert!(!report.is_loadable());
    }

    #[test]
    fn gpt_oss_gguf_accepts_the_supported_sink_alias_but_not_extra_tensors() {
        let directory = tempfile::tempdir().unwrap();
        let alias_path = directory.path().join("gpt-oss-sink-alias.gguf");
        write_complete_gpt_oss_gguf(&alias_path, |specs| {
            specs
                .iter_mut()
                .find(|(name, _, _)| name == "blk.0.attn_sinks.weight")
                .unwrap()
                .0 = "blk.0.attn_sinks".into();
        });
        let alias_report = inspect_model(&alias_path, MlxInspectionOptions::default()).unwrap();
        assert!(alias_report.is_loadable(), "{:#?}", alias_report.issues);

        let extra_path = directory.path().join("gpt-oss-extra.gguf");
        write_complete_gpt_oss_gguf(&extra_path, |specs| {
            specs.push(("poison.weight".into(), vec![1], GgmlType::F32));
        });
        let extra_report = inspect_model(&extra_path, MlxInspectionOptions::default()).unwrap();
        assert!(extra_report.issues.iter().any(|issue| {
            issue.code == InspectionIssueCode::ConflictingTensorLayout
                && issue.tensor_name.as_deref() == Some("poison.weight")
        }));
        assert!(!extra_report.is_loadable());
    }

    #[test]
    fn lfm2_quantized_shortconv_kernel_is_rejected_per_operation() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("lfm2-quantized-conv.gguf");
        write_complete_lfm2_gguf(&path, false, |specs| {
            let tensor = specs
                .iter_mut()
                .find(|(name, _, _)| name == "blk.0.shortconv.conv.weight")
                .unwrap();
            tensor.1 = vec![32, 3];
            tensor.2 = GgmlType::Q4_0;
        });
        let report = inspect_model(&path, MlxInspectionOptions::default()).unwrap();
        let issue = report
            .issues
            .iter()
            .find(|issue| issue.code == InspectionIssueCode::UnsupportedTensorEncoding)
            .unwrap();
        assert_eq!(
            issue.tensor_name.as_deref(),
            Some("blk.0.shortconv.conv.weight")
        );
        assert_eq!(issue.tensor_type_code, Some(GgmlType::Q4_0.code()));
        assert!(!report.is_loadable());
    }

    #[test]
    fn qwen3_moe_paired_expert_encoding_mismatch_is_rejected() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("qwen3moe-mismatched.gguf");
        write_complete_qwen3_gguf(&path, true, |specs| {
            specs
                .iter_mut()
                .find(|(name, _, _)| name == "blk.0.ffn_gate_exps.weight")
                .unwrap()
                .2 = GgmlType::Q4_0;
            specs
                .iter_mut()
                .find(|(name, _, _)| name == "blk.0.ffn_up_exps.weight")
                .unwrap()
                .2 = GgmlType::Q8_0;
        });
        let report = inspect_model(&path, MlxInspectionOptions::default()).unwrap();
        let issue = report
            .issues
            .iter()
            .find(|issue| issue.code == InspectionIssueCode::QuantizationCompanionMismatch)
            .unwrap();
        assert_eq!(
            issue.tensor_name.as_deref(),
            Some("blk.0.ffn_gate_exps.weight")
        );
        assert_eq!(issue.tensor_type_code, Some(GgmlType::Q4_0.code()));
        assert!(!report.is_loadable());
    }

    #[test]
    fn supported_gguf_encoding_on_an_unsupported_operation_is_rejected() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("quantized-norm.gguf");
        write_complete_gguf(&path, |specs| {
            specs
                .iter_mut()
                .find(|(name, _, _)| name == "output_norm.weight")
                .unwrap()
                .2 = GgmlType::Q4_0;
        });
        let report = inspect_model(&path, MlxInspectionOptions::default()).unwrap();
        let issue = report
            .issues
            .iter()
            .find(|issue| issue.code == InspectionIssueCode::UnsupportedTensorEncoding)
            .unwrap();
        assert_eq!(issue.tensor_name.as_deref(), Some("output_norm.weight"));
        assert_eq!(issue.tensor_type_code, Some(GgmlType::Q4_0.code()));
        assert!(!report.is_loadable());
    }

    #[test]
    fn unknown_gguf_tensor_type_has_a_numeric_structured_diagnostic() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("unknown.gguf");
        write_unknown_gguf_type(&path, 1234);
        let report = inspect_model(&path, MlxInspectionOptions::default()).unwrap();
        let issue = report
            .issues
            .iter()
            .find(|issue| issue.code == InspectionIssueCode::UnsupportedTensorEncoding)
            .unwrap();
        assert_eq!(issue.tensor_type_code, Some(1234));
    }

    #[test]
    fn missing_required_qwen_projector_blocks_model_admission() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("qwen.gguf");
        write_gguf(
            &path,
            "qwen3vl",
            GgmlType::F16,
            std::iter::empty::<(String, MetadataValue)>(),
        );
        let report = inspect_model(&path, MlxInspectionOptions::default()).unwrap();
        assert_eq!(report.model_loadability, InspectionReadiness::Missing);
        assert!(report
            .issues
            .iter()
            .any(|issue| issue.code == InspectionIssueCode::MissingMediaProjector));
    }

    #[test]
    fn complete_qwen3_vl_gguf_catalog_is_exact_and_shared_with_loader_preflight() {
        let directory = tempfile::tempdir().unwrap();
        let (model_path, projector_path) =
            write_complete_qwen3_vl_gguf(directory.path(), |_| {}, |_| {});
        let report = inspect_model(&model_path, MlxInspectionOptions::default()).unwrap();
        assert!(report.is_loadable(), "{:#?}", report.issues);
        assert_eq!(report.structural_binding, InspectionReadiness::Ready);
        assert!(report
            .issues
            .iter()
            .all(|issue| { issue.code != InspectionIssueCode::ValidationUnavailableUntilLoad }));

        let checkpoint = GgufCheckpoint::open(&model_path).unwrap();
        let projector = GgufCheckpoint::open(&projector_path).unwrap();
        let metadata = crate::backend::mlx::runtime::checkpoint::load::gguf_metadata(&checkpoint);
        let projector_metadata =
            crate::backend::mlx::runtime::checkpoint::load::gguf_metadata(&projector);
        let prepared = qwen3_vl::prepare_qwen3_vl_gguf_checkpoint(
            &checkpoint,
            &metadata,
            &projector,
            &projector_metadata,
        )
        .unwrap();
        assert_eq!(prepared.args.text_config.hidden_size, 32);
        assert_eq!(prepared.args.vision_config.hidden_size, 8);
        assert_eq!(prepared.args.vision_config.deepstack_layers(), vec![0]);
        assert_eq!(prepared.eos_token_ids, vec![2]);
    }

    #[test]
    fn qwen3_vl_missing_projector_tensor_has_structured_diagnostic() {
        let directory = tempfile::tempdir().unwrap();
        let (model_path, _) = write_complete_qwen3_vl_gguf(
            directory.path(),
            |_| {},
            |specs| {
                specs.retain(|(name, _, _)| name != "v.blk.0.attn_qkv.weight");
            },
        );
        let report = inspect_model(&model_path, MlxInspectionOptions::default()).unwrap();
        assert!(report.issues.iter().any(|issue| {
            issue.code == InspectionIssueCode::MissingRequiredTensor
                && issue.tensor_name.as_deref() == Some("v.blk.0.attn_qkv.weight")
        }));
        assert!(!report.is_loadable());
    }

    #[test]
    fn qwen3_vl_valid_projector_cannot_overwrite_invalid_text_catalog() {
        let directory = tempfile::tempdir().unwrap();
        let (model_path, _) = write_complete_qwen3_vl_gguf(
            directory.path(),
            |specs| {
                specs.retain(|(name, _, _)| name != "blk.0.attn_q_norm.weight");
            },
            |_| {},
        );
        let report = inspect_model(&model_path, MlxInspectionOptions::default()).unwrap();
        assert_eq!(report.structural_binding, InspectionReadiness::Invalid);
        assert_eq!(report.model_loadability, InspectionReadiness::Invalid);
        assert!(report.issues.iter().any(|issue| {
            issue.code == InspectionIssueCode::MissingRequiredTensor
                && issue.tensor_name.as_deref() == Some("blk.0.attn_q_norm.weight")
        }));
        assert!(!report.is_loadable());
    }

    #[test]
    fn qwen3_vl_projector_shape_and_operation_encoding_are_exact() {
        let shape_directory = tempfile::tempdir().unwrap();
        let (shape_model, _) = write_complete_qwen3_vl_gguf(
            shape_directory.path(),
            |_| {},
            |specs| {
                specs
                    .iter_mut()
                    .find(|(name, _, _)| name == "v.blk.0.attn_qkv.weight")
                    .unwrap()
                    .1 = vec![192];
            },
        );
        let shape_report = inspect_model(&shape_model, MlxInspectionOptions::default()).unwrap();
        assert!(shape_report.issues.iter().any(|issue| {
            issue.code == InspectionIssueCode::TensorShapeMismatch
                && issue.tensor_name.as_deref() == Some("v.blk.0.attn_qkv.weight")
        }));
        assert!(!shape_report.is_loadable());

        let encoding_directory = tempfile::tempdir().unwrap();
        let (encoding_model, _) = write_complete_qwen3_vl_gguf(
            encoding_directory.path(),
            |_| {},
            |specs| {
                specs
                    .iter_mut()
                    .find(|(name, _, _)| name == "v.post_ln.weight")
                    .map(|(_, shape, encoding)| {
                        *shape = vec![32];
                        *encoding = GgmlType::Q4_0;
                    })
                    .unwrap();
            },
        );
        let encoding_report =
            inspect_model(&encoding_model, MlxInspectionOptions::default()).unwrap();
        let issue = encoding_report
            .issues
            .iter()
            .find(|issue| issue.code == InspectionIssueCode::UnsupportedTensorEncoding)
            .unwrap();
        assert_eq!(issue.tensor_name.as_deref(), Some("v.post_ln.weight"));
        assert_eq!(issue.tensor_type_code, Some(GgmlType::Q4_0.code()));
        assert!(!encoding_report.is_loadable());
    }

    #[test]
    fn requested_quantization_and_residency_use_shared_loader_policy() {
        use crate::backend::mlx::runtime::{
            checkpoint::quantization::WeightQuantization, execution::layerwise::WeightResidency,
            residency::expert_cache::ExpertCacheLoadOptions,
        };

        let directory = tempfile::tempdir().unwrap();
        let gguf = directory.path().join("packed.gguf");
        write_gguf(
            &gguf,
            "llama",
            GgmlType::Q4_0,
            [("general.file_type".into(), MetadataValue::Uint32(2))],
        );
        let report = inspect_model(
            &gguf,
            MlxInspectionOptions {
                load: ModelLoadOptions::with_quantization(WeightQuantization::MxFp4),
            },
        )
        .unwrap();
        assert_eq!(report.requested_load, InspectionReadiness::Unsupported);
        assert!(report
            .issues
            .iter()
            .any(|issue| issue.code == InspectionIssueCode::UnsupportedQuantizationRequest));

        let safetensors = write_safetensors_dir(&llama_config());
        let report = inspect_model(
            safetensors.path(),
            MlxInspectionOptions {
                load: ModelLoadOptions::default().with_weight_residency(
                    WeightResidency::with_expert_cache(
                        NonExpertWeightResidency::LayerwiseHost(Default::default()),
                        ExpertCacheLoadOptions::default(),
                    ),
                ),
            },
        )
        .unwrap();
        assert_eq!(report.requested_load, InspectionReadiness::Unsupported);
        assert!(report
            .issues
            .iter()
            .any(|issue| issue.code == InspectionIssueCode::UnsupportedResidencyPolicy));

        let report = inspect_model(
            &gguf,
            MlxInspectionOptions {
                load: ModelLoadOptions::default().with_weight_residency(
                    WeightResidency::with_expert_cache(
                        NonExpertWeightResidency::LayerwiseHost(Default::default()),
                        ExpertCacheLoadOptions::default(),
                    ),
                ),
            },
        )
        .unwrap();
        assert_eq!(report.requested_load, InspectionReadiness::Unsupported);
        assert!(report
            .issues
            .iter()
            .any(|issue| issue.code == InspectionIssueCode::UnsupportedResidencyPolicy));
    }
}
