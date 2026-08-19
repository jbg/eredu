//! Side-effect-free MLX model artifact compatibility inspection.

use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    path::{Path, PathBuf},
};

#[cfg(test)]
use crate::backend::mlx::resolve_model_config;
use crate::backend::mlx::structural::{self, GgufArchitectureValidation};
use eredu_checkpoint::store::WeightStore;
use eredu_core::{
    ArtifactFormat, ArtifactModality, ArtifactTensorEncoding, GgufArchitecture, InspectionIssue,
    InspectionIssueCode, InspectionReadiness, InspectionRequirement, InspectionSeverity,
    ModelInspectionReport, ModelKind, Observed,
};
use eredu_gguf::MetadataValue as GgufMetadataValue;
use safemlx::ops::GgufCheckpoint;
use serde_json::Value;

use super::*;
use crate::{
    backend::mlx::runtime::checkpoint::store::SafetensorsWeightStore,
    composition::mlx_architectures::{
        gemma4::model as gemma4, inkling::model as inkling, muse_glimmer,
        qwen::vl::model as qwen3_vl,
    },
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
                    eredu_core::ArtifactFormat::SafeTensors,
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
                        } else if cfg!(feature = "mlx-image") {
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
            match crate::composition::mlx_architectures::qwen::hybrid::qwen3_5::open_sibling_mmproj(
                path,
            ) {
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
                    } else if cfg!(feature = "mlx-image") {
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
                } else if cfg!(all(feature = "mlx-image", feature = "mlx-audio")) {
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
                } else if cfg!(any(feature = "mlx-image", feature = "mlx-audio")) {
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
                } else if cfg!(feature = "mlx-image") {
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
                    cfg!(feature = "mlx-image")
                }
                ArtifactModality::Audio => cfg!(feature = "mlx-audio"),
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
