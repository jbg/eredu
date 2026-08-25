//! Side-effect-free MLX model artifact compatibility inspection.

use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
};

use crate::composition::mlx::structural;
use eredu_architectures::GgufArchitecture;
use eredu_core::{
    checkpoint::{TensorCatalog, TensorDtype},
    ArtifactFormat, ArtifactModality, ArtifactTensorEncoding, InputModalities, InspectionIssue,
    InspectionIssueCode, InspectionReadiness, InspectionRequirement, InspectionSeverity,
    ModelInspectionReport, Observed,
};
use safemlx::ops::GgufCheckpoint;

use super::*;

struct InspectedSafetensorsCatalog<'a>(&'a TensorCatalog);

impl eredu_checkpoint::validation::SafetensorsCatalog for InspectedSafetensorsCatalog<'_> {
    fn keys(&self) -> Vec<String> {
        self.0
            .descriptors()
            .map(|tensor| tensor.name.clone())
            .collect()
    }

    fn metadata(
        &self,
        key: &str,
    ) -> Result<eredu_checkpoint::validation::CatalogTensorMetadata, String> {
        let tensor = self
            .0
            .get(key)
            .ok_or_else(|| format!("unknown checkpoint tensor {key:?}"))?;
        Ok(eredu_checkpoint::validation::CatalogTensorMetadata {
            shape: tensor.shape.clone(),
            stored_dtype: inspected_stored_dtype(&tensor.dtype),
        })
    }
}

impl eredu_checkpoint::recipe::RecipeCatalog for InspectedSafetensorsCatalog<'_> {
    fn tensor_metadata(
        &self,
        key: &str,
    ) -> Result<eredu_checkpoint::store::TensorMetadata, eredu_checkpoint::store::StoreError> {
        let tensor =
            self.0
                .get(key)
                .ok_or_else(|| eredu_checkpoint::store::StoreError::UnknownTensor {
                    key: key.into(),
                })?;
        Ok(eredu_checkpoint::store::TensorMetadata {
            name: tensor.name.clone(),
            logical_shape: tensor.shape.clone(),
            physical_shape: tensor.shape.clone(),
            stored_dtype: inspected_stored_dtype(&tensor.dtype),
            encoded_byte_len: tensor.storage.as_ref().map_or(0, |storage| storage.length),
            backing_shard: tensor
                .storage
                .as_ref()
                .map(|storage| PathBuf::from(&storage.member)),
        })
    }
}

fn inspected_stored_dtype(dtype: &TensorDtype) -> eredu_checkpoint::StoredDtype {
    use eredu_checkpoint::StoredDtype;
    match dtype {
        TensorDtype::Bool => StoredDtype::Bool,
        TensorDtype::U8 => StoredDtype::U8,
        TensorDtype::I8 => StoredDtype::I8,
        TensorDtype::I16 => StoredDtype::I16,
        TensorDtype::U16 => StoredDtype::U16,
        TensorDtype::F16 => StoredDtype::F16,
        TensorDtype::Bf16 => StoredDtype::BF16,
        TensorDtype::I32 => StoredDtype::I32,
        TensorDtype::U32 => StoredDtype::U32,
        TensorDtype::F32 => StoredDtype::F32,
        TensorDtype::F64 => StoredDtype::F64,
        TensorDtype::I64 => StoredDtype::I64,
        TensorDtype::U64 => StoredDtype::U64,
        TensorDtype::Complex64 => StoredDtype::C64,
        TensorDtype::Encoded(name) if name == "F8_E4M3" => StoredDtype::F8E4M3,
        TensorDtype::Encoded(name) if name == "F4" => StoredDtype::F4,
        TensorDtype::Encoded(name) if name == "F8_E8M0" => StoredDtype::F8E8M0,
        TensorDtype::Encoded(name) if name == "F8_E5M2" => StoredDtype::F8E5M2,
        TensorDtype::Encoded(name) => StoredDtype::Other(name.clone()),
    }
}

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
        return Err(Error::ArchitectureModel(format!(
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

fn prepared_projector(validated: &eredu_core::ValidatedGguf) -> Option<PathBuf> {
    let companion = validated.companion(&eredu_core::GgufCompanionRole::MediaProjector)?;
    Some(companion.path().to_owned())
}

fn mark_portable_gguf_admitted(report: &mut ModelInspectionReport) {
    report.structural_binding = InspectionReadiness::Ready;
    report.model_loadability = InspectionReadiness::Ready;
}

fn inspect_safetensors(path: &Path, options: MlxInspectionOptions) -> ModelInspectionReport {
    let mut report = ModelInspectionReport::unverified(path, ArtifactFormat::SafeTensors);
    let portable = match eredu_architectures::configuration::inspect_artifact(path) {
        Ok(inspection) => inspection,
        Err(error) => {
            reject_portable_safetensors(&mut report, path, &error);
            return report;
        }
    };
    let configuration = portable.configuration();
    let architecture_plan = portable.architecture_plan();
    let catalog = portable.tensors();
    let inspected_catalog = InspectedSafetensorsCatalog(catalog);

    report.container = InspectionReadiness::Ready;
    report.model_family = Some(configuration.family.clone());
    report.architecture = Some(configuration.effective_model_type.clone());
    report.architecture_support = InspectionReadiness::Ready;
    report.tensor_count = Some(catalog.len());
    let mut shards = BTreeSet::new();
    let mut encodings = BTreeSet::new();
    let mut stored_tensor_bytes = Some(0_u64);
    let mut largest_stored_tensor_bytes = 0_u64;
    for tensor in catalog.descriptors() {
        encodings.insert(format!("{:?}", inspected_stored_dtype(&tensor.dtype)));
        if let Some(storage) = &tensor.storage {
            shards.insert(storage.member.clone());
            stored_tensor_bytes =
                stored_tensor_bytes.and_then(|total| total.checked_add(storage.length));
            largest_stored_tensor_bytes = largest_stored_tensor_bytes.max(storage.length);
        } else {
            stored_tensor_bytes = None;
        }
    }
    report.checkpoint_shards = Some(shards.len());
    report.tensor_encodings = encodings
        .into_iter()
        .map(|name| ArtifactTensorEncoding {
            name,
            ggml_type_code: None,
        })
        .collect();
    match stored_tensor_bytes {
        Some(total) => {
            report.resources.stored_tensor_bytes =
                Observed::exact(total, "authoritative SafeTensors tensor catalog");
            report.resources.largest_stored_tensor_bytes = Observed::exact(
                largest_stored_tensor_bytes,
                "authoritative SafeTensors tensor catalog",
            );
        }
        None => {
            report.resources.stored_tensor_bytes =
                Observed::unavailable("SafeTensors payload-byte catalog was incomplete");
            report.resources.largest_stored_tensor_bytes =
                Observed::unavailable("SafeTensors payload-byte catalog was incomplete");
        }
    }

    match eredu_architectures::preparation::prepared_safetensors_capabilities(
        architecture_plan
            .safetensors_architecture()
            .expect("SafeTensors inspection must retain its validated architecture plan"),
    ) {
        Ok(capabilities) => {
            record_embedded_drafting(&mut report, capabilities);
            report.expected_modalities = artifact_modalities(capabilities.input_modalities());
            apply_structural_validation(
                &mut report,
                structural::validate_safetensors(
                    architecture_plan.safetensors_architecture().expect(
                        "SafeTensors inspection must retain its validated architecture plan",
                    ),
                    &inspected_catalog,
                    options.load,
                ),
                path,
            );
            match options
                .load
                .preparation_policy()
                .and_then(|policy| structural::validate_inspected_preparation(&portable, policy))
            {
                Ok(()) => report.requested_load = InspectionReadiness::Ready,
                Err(error) => reject_load_policy(&mut report, &error),
            }
            inspect_safetensors_media(&mut report, architecture_plan);
        }
        Err(error) => {
            report.architecture_support = InspectionReadiness::Invalid;
            report.model_loadability = InspectionReadiness::Invalid;
            report.structural_binding = InspectionReadiness::Invalid;
            report.requested_load = InspectionReadiness::Invalid;
            report.multimodal = InspectionReadiness::Invalid;
            report.issue(
                InspectionIssueCode::InvalidConfiguration,
                InspectionSeverity::Error,
                error.to_string(),
                Some(path.join("config.json")),
            );
        }
    }
    report
}

fn reject_portable_safetensors(
    report: &mut ModelInspectionReport,
    path: &Path,
    error: &eredu_core::artifact::ArtifactError,
) {
    let detail = error.to_string();
    let invalid_plan = matches!(
        error,
        eredu_core::artifact::ArtifactError::InvalidArchitecturePlan(_)
    );
    let (code, container, architecture) = match error {
        eredu_core::artifact::ArtifactError::UnsupportedModelType(_) => (
            InspectionIssueCode::UnsupportedArchitecture,
            InspectionReadiness::Unverified,
            InspectionReadiness::Unsupported,
        ),
        eredu_core::artifact::ArtifactError::Io(error)
            if error.kind() == std::io::ErrorKind::NotFound =>
        {
            (
                InspectionIssueCode::MissingCheckpointShard,
                InspectionReadiness::Missing,
                InspectionReadiness::Unverified,
            )
        }
        eredu_core::artifact::ArtifactError::Json(_) => (
            InspectionIssueCode::InvalidConfiguration,
            InspectionReadiness::Invalid,
            InspectionReadiness::Invalid,
        ),
        eredu_core::artifact::ArtifactError::InvalidArchitecturePlan(_) => (
            InspectionIssueCode::InvalidConfiguration,
            InspectionReadiness::Ready,
            InspectionReadiness::Invalid,
        ),
        eredu_core::artifact::ArtifactError::DuplicateTensor(_)
        | eredu_core::artifact::ArtifactError::UnsafeShardPath(_)
        | eredu_core::artifact::ArtifactError::Catalog(_) => (
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
    } else {
        InspectionReadiness::Invalid
    };
    report.requested_load = report.model_loadability;
    report.issue(
        code,
        InspectionSeverity::Error,
        detail,
        Some(path.to_path_buf()),
    );
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
    let gguf_architecture = portable
        .architecture_plan()
        .gguf_architecture()
        .expect("GGUF inspection must retain its architecture-owned identity");
    let checkpoint = GgufCheckpoint::from_portable(validated.checkpoint().clone());
    report.container = InspectionReadiness::Ready;
    report.model_family = Some(gguf_architecture.model_kind().canonical_name().into());
    report.architecture = Some(gguf_architecture.metadata_name().into());
    report.architecture_support = InspectionReadiness::Ready;
    mark_portable_gguf_admitted(&mut report);
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
    let capabilities = eredu_architectures::preparation::prepared_gguf_capabilities(
        portable
            .architecture_plan()
            .gguf_plan()
            .expect("GGUF inspection must retain its validated architecture plan"),
    );
    record_embedded_drafting(&mut report, capabilities);
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
    let composition = inspect_gguf_projector(&mut report, path, gguf_architecture, validated);
    report.expected_modalities = artifact_modalities(composite_plan.input_modalities(composition));

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
        | eredu_core::artifact::ArtifactError::InvalidArchitecturePlan(_)
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
    validated: &eredu_core::ValidatedGguf,
) -> eredu_architectures::preparation::GgufArtifactComposition {
    use eredu_architectures::preparation::GgufArtifactComposition;

    let mut composition = GgufArtifactComposition::ModelOnly;
    match architecture {
        GgufArchitecture::Qwen3Vl | GgufArchitecture::Qwen3VlMoe => {
            match prepared_projector(validated) {
                Some(projector) => {
                    composition = GgufArtifactComposition::ValidatedMediaProjector;
                    report.multimodal = if cfg!(feature = "image") {
                        InspectionReadiness::Ready
                    } else {
                        InspectionReadiness::Unsupported
                    };
                    report.requirements.push(InspectionRequirement {
                        code: InspectionIssueCode::MissingMediaProjector,
                        readiness: InspectionReadiness::Ready,
                        detail: "portable admission validated the qwen3vl vision projector".into(),
                        path: Some(projector),
                    });
                }
                None => reject_projector(
                    report,
                    path.to_path_buf(),
                    "Qwen3-VL preparation omitted its required media projector".into(),
                    true,
                ),
            }
        }
        GgufArchitecture::Qwen35 | GgufArchitecture::Qwen35Moe => {
            match prepared_projector(validated) {
                Some(projector_path) => {
                    composition = GgufArtifactComposition::ValidatedMediaProjector;
                    report.multimodal = if cfg!(feature = "image") {
                        InspectionReadiness::Ready
                    } else {
                        InspectionReadiness::Unsupported
                    };
                    report.requirements.push(InspectionRequirement {
                        code: InspectionIssueCode::MissingMediaProjector,
                        readiness: InspectionReadiness::Ready,
                        detail: "portable admission validated the Qwen3.5 vision projector".into(),
                        path: Some(projector_path),
                    });
                }
                None => {
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
            }
        }
        GgufArchitecture::Inkling => match prepared_projector(validated) {
            Some(_projector_path) => {
                composition = GgufArtifactComposition::ValidatedMediaProjector;
                report.multimodal = if cfg!(all(feature = "image", feature = "audio")) {
                    InspectionReadiness::Ready
                } else {
                    InspectionReadiness::Unsupported
                };
            }
            None => {
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
        },
        GgufArchitecture::Gemma4 => match prepared_projector(validated) {
            Some(projector_path) => {
                composition = GgufArtifactComposition::ValidatedMediaProjector;
                report.multimodal = if cfg!(any(feature = "image", feature = "audio")) {
                    InspectionReadiness::Ready
                } else {
                    InspectionReadiness::Unsupported
                };
                report.requirements.push(InspectionRequirement {
                    code: InspectionIssueCode::MissingMediaProjector,
                    readiness: InspectionReadiness::Ready,
                    detail: "portable admission validated the Gemma 4 vision/audio projector"
                        .into(),
                    path: Some(projector_path),
                });
            }
            None => {
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
        },
        GgufArchitecture::MuseGlimmer => match prepared_projector(validated) {
            Some(projector_path) => {
                composition = GgufArtifactComposition::ValidatedMediaProjector;
                report.multimodal = if cfg!(feature = "image") {
                    InspectionReadiness::Ready
                } else {
                    InspectionReadiness::Unsupported
                };
                report.requirements.push(InspectionRequirement {
                    code: InspectionIssueCode::MissingMediaProjector,
                    readiness: InspectionReadiness::Ready,
                    detail: "portable admission validated the image-only Muse-Glimmer projector"
                        .into(),
                    path: Some(projector_path),
                });
            }
            None => {
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

fn inspect_safetensors_media(
    report: &mut ModelInspectionReport,
    plan: &eredu_architectures::processor_plan::ArtifactArchitecturePlan,
) {
    if report.expected_modalities == [ArtifactModality::Text] {
        report.multimodal = InspectionReadiness::NotApplicable;
        return;
    }
    if plan.has_processor() {
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
                "authoritative processor plan and required media build features are available"
                    .into()
            } else {
                "authoritative processor plan is available, but required image/audio processing features are not enabled".into()
            },
            path: None,
        });
    } else {
        report.multimodal = InspectionReadiness::Missing;
        report.issue(
            InspectionIssueCode::MissingProcessor,
            InspectionSeverity::Warning,
            "authoritative architecture inspection admitted no media processor",
            Some(report.path.clone()),
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
        Error::ArchitectureModel(detail)
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

    fn write_safetensors_fixture(root: &Path, config: &serde_json::Value) {
        use safetensors::tensor::{serialize_to_file, Dtype, TensorView};

        std::fs::write(
            root.join("config.json"),
            serde_json::to_vec(config).unwrap(),
        )
        .unwrap();
        let bytes = 0.0_f32.to_le_bytes();
        let view = TensorView::new(Dtype::F32, vec![1], &bytes).unwrap();
        serialize_to_file(
            [("model.language_model.embed_tokens.weight", view)],
            None,
            &root.join("model.safetensors"),
        )
        .unwrap();
    }

    fn qwen_vl_config() -> serde_json::Value {
        serde_json::json!({
            "model_type":"qwen3_vl", "image_token_id":61, "video_token_id":62,
            "vision_start_token_id":44, "vision_end_token_id":45,
            "text_config": {"model_type":"qwen3_vl_text", "hidden_size":32,
                "num_hidden_layers":1, "intermediate_size":64, "num_attention_heads":4,
                "num_key_value_heads":2, "head_dim":8, "rms_norm_eps":0.000001,
                "vocab_size":64, "max_position_embeddings":128, "tie_word_embeddings":true,
                "rope_scaling":{"mrope_section":[2,1,1]}},
            "vision_config":{"depth":1,"hidden_size":16,"intermediate_size":24,
                "num_heads":4,"num_position_embeddings":16,"in_channels":3,"patch_size":2,
                "spatial_merge_size":2,"temporal_patch_size":2,"out_hidden_size":32,
                "deepstack_visual_indexes":[0]}
        })
    }

    fn write_minimal_llama_gguf(
        path: &Path,
        include_embedding_length: bool,
        tokenizer_eos: Option<eredu_gguf::MetadataValue>,
    ) {
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
            metadata.extend([
                (
                    "llama.embedding_length".into(),
                    eredu_gguf::MetadataValue::Uint32(1),
                ),
                (
                    "llama.attention.head_count".into(),
                    eredu_gguf::MetadataValue::Uint32(1),
                ),
                (
                    "llama.feed_forward_length".into(),
                    eredu_gguf::MetadataValue::Uint32(1),
                ),
                (
                    "llama.attention.layer_norm_rms_epsilon".into(),
                    eredu_gguf::MetadataValue::Float32(1e-5),
                ),
                (
                    "llama.vocab_size".into(),
                    eredu_gguf::MetadataValue::Uint32(1),
                ),
                (
                    "llama.context_length".into(),
                    eredu_gguf::MetadataValue::Uint32(1),
                ),
            ]);
        }
        if let Some(tokenizer_eos) = tokenizer_eos {
            metadata.insert("tokenizer.ggml.eos_token_id".into(), tokenizer_eos);
        }
        let data = 1.0_f32.to_le_bytes();
        let tensor = |name, dimensions| eredu_gguf::TensorInput {
            name,
            dimensions,
            ggml_type: eredu_gguf::GgmlType::F32,
            data: &data,
        };
        let tensors = if include_embedding_length {
            vec![
                tensor("token_embd.weight", &[1, 1]),
                tensor("output_norm.weight", &[1]),
                tensor("blk.0.attn_norm.weight", &[1]),
                tensor("blk.0.ffn_norm.weight", &[1]),
                tensor("blk.0.attn_q.weight", &[1, 1]),
                tensor("blk.0.attn_k.weight", &[1, 1]),
                tensor("blk.0.attn_v.weight", &[1, 1]),
                tensor("blk.0.attn_output.weight", &[1, 1]),
                tensor("blk.0.ffn_gate.weight", &[1, 1]),
                tensor("blk.0.ffn_up.weight", &[1, 1]),
                tensor("blk.0.ffn_down.weight", &[1, 1]),
            ]
        } else {
            vec![tensor("token_embd.weight", &[1])]
        };
        eredu_gguf::Writer::default()
            .write(std::fs::File::create(path).unwrap(), &metadata, &tensors)
            .unwrap();
    }

    #[test]
    fn text_only_architecture_skips_processor_and_media_feature_requirements() {
        let mut report =
            ModelInspectionReport::unverified(Path::new("unused"), ArtifactFormat::SafeTensors);
        report.expected_modalities = artifact_modalities(InputModalities::TEXT);
        let plan = eredu_core::ModelConfigurationResolver::resolve_safetensors(
            &eredu_architectures::configuration::MODEL_CONFIGURATIONS,
            &serde_json::json!({
                "model_type": "llama",
                "hidden_size": 16,
                "num_hidden_layers": 2,
                "intermediate_size": 32,
                "num_attention_heads": 4,
                "rms_norm_eps": 0.00001,
                "vocab_size": 64
            }),
        )
        .unwrap()
        .architecture_plan;

        inspect_safetensors_media(&mut report, &plan);

        assert_eq!(report.expected_modalities, [ArtifactModality::Text]);
        assert_eq!(report.multimodal, InspectionReadiness::NotApplicable);
        assert!(report.requirements.is_empty());
        assert!(report.issues.is_empty());
    }

    #[test]
    fn safetensors_inspection_rejects_malformed_processor_sidecar() {
        let root = tempfile::tempdir().unwrap();
        write_safetensors_fixture(root.path(), &qwen_vl_config());
        std::fs::write(
            root.path()
                .join(eredu_architectures::processor_plan::PROCESSOR_CONFIG_FILENAME),
            b"{",
        )
        .unwrap();

        let report = inspect_model(root.path(), MlxInspectionOptions::default()).unwrap();

        assert_eq!(report.container, InspectionReadiness::Ready);
        assert_eq!(report.multimodal, InspectionReadiness::Invalid);
        assert_eq!(report.model_loadability, InspectionReadiness::Invalid);
        assert_eq!(report.requested_load, InspectionReadiness::Invalid);
        assert!(!report.is_loadable());
        assert!(report.issues.iter().any(|issue| {
            issue.code == InspectionIssueCode::InvalidConfiguration
                && issue.detail.contains("processor")
        }));
    }

    #[test]
    fn safetensors_inspection_rejects_wrong_family_processor_sidecar() {
        let root = tempfile::tempdir().unwrap();
        write_safetensors_fixture(root.path(), &qwen_vl_config());
        std::fs::write(
            root.path()
                .join(eredu_architectures::processor_plan::PROCESSOR_CONFIG_FILENAME),
            br#"{"image_processor":{},"video_processor":{}}"#,
        )
        .unwrap();

        let report = inspect_model(root.path(), MlxInspectionOptions::default()).unwrap();

        assert_eq!(report.container, InspectionReadiness::Ready);
        assert_eq!(report.multimodal, InspectionReadiness::Invalid);
        assert_eq!(report.model_loadability, InspectionReadiness::Invalid);
        assert_eq!(report.requested_load, InspectionReadiness::Invalid);
        assert!(!report.is_loadable());
        assert!(report.issues.iter().any(|issue| {
            issue.code == InspectionIssueCode::InvalidConfiguration
                && issue.detail.contains("processor")
        }));
    }

    #[test]
    fn composite_gguf_plan_keeps_readiness_and_expected_modalities_consistent() {
        use eredu_architectures::preparation::{
            gguf_composite_artifact_plan, GgufArtifactComposition,
        };

        for (architecture, expected) in [
            (
                GgufArchitecture::Inkling,
                vec![
                    ArtifactModality::Text,
                    ArtifactModality::Image,
                    ArtifactModality::Audio,
                ],
            ),
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
            let plan = gguf_composite_artifact_plan(architecture);
            assert_eq!(
                artifact_modalities(plan.input_modalities(GgufArtifactComposition::ModelOnly)),
                [ArtifactModality::Text]
            );
            let modalities =
                plan.input_modalities(GgufArtifactComposition::ValidatedMediaProjector);
            assert_eq!(artifact_modalities(modalities), expected);
        }
    }

    #[test]
    fn gguf_inspection_enriches_the_portable_validated_checkpoint() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("model.gguf");
        write_minimal_llama_gguf(&path, true, None);

        let report = inspect_model(&path, MlxInspectionOptions::default()).unwrap();

        assert_eq!(report.container, InspectionReadiness::Ready);
        assert_eq!(report.architecture_support, InspectionReadiness::Ready);
        assert_eq!(report.architecture.as_deref(), Some("llama"));
        assert_eq!(report.tensor_count, Some(11));
    }

    #[test]
    fn gguf_inspection_rejects_quantization_before_complete_tensor_parallel_loading() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("model.gguf");
        write_minimal_llama_gguf(&path, true, None);
        let topology = crate::backend::MlxParallelContext::for_rank(
            0,
            2,
            1,
            1,
            crate::backend::DeviceAssignment::new(safemlx::DeviceType::Cpu, 0),
        )
        .unwrap();
        let options = MlxInspectionOptions {
            load: ModelLoadOptions::with_quantization(eredu_checkpoint::WeightQuantization::MxFp4)
                .with_parallel_topology(topology),
        };

        let report = inspect_model(&path, options).unwrap();

        assert_eq!(report.requested_load, InspectionReadiness::Unsupported);
        assert!(report.issues.iter().any(|issue| {
            issue.code == InspectionIssueCode::UnsupportedQuantizationRequest
                && issue.detail.contains("complete tensor-parallel")
        }));
    }

    #[test]
    fn gguf_inspection_ignores_facade_owned_tokenizer_metadata() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("model.gguf");
        write_minimal_llama_gguf(&path, true, None);
        let baseline = inspect_model(&path, MlxInspectionOptions::default()).unwrap();

        write_minimal_llama_gguf(
            &path,
            true,
            Some(eredu_gguf::MetadataValue::String("not a token id".into())),
        );
        let with_tokenizer_metadata =
            inspect_model(&path, MlxInspectionOptions::default()).unwrap();

        assert_eq!(
            with_tokenizer_metadata.model_loadability,
            baseline.model_loadability
        );
        assert_eq!(
            with_tokenizer_metadata.requested_load,
            baseline.requested_load
        );
        assert_eq!(with_tokenizer_metadata.issues, baseline.issues);
    }

    #[test]
    fn gguf_inspection_surfaces_the_portable_admission_floor() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("model.gguf");
        write_minimal_llama_gguf(&path, false, None);

        let report = inspect_model(&path, MlxInspectionOptions::default()).unwrap();

        assert_eq!(report.container, InspectionReadiness::Ready);
        assert_eq!(report.structural_binding, InspectionReadiness::Invalid);
        assert!(report
            .issues
            .iter()
            .any(|issue| issue.detail.contains("llama.embedding_length")));
    }
}
