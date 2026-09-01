//! Adapter from checkpoint-probe artifacts to general parity observations.

use std::{
    fs,
    path::{Path, PathBuf},
};

use eredu_core::{
    ObservationSelector, ObservationSet, ObservationValue, TensorObservation, TensorObservationData,
};
use safetensors::{tensor::SafeTensors, Dtype};
use serde::{Deserialize, Serialize};

use crate::{
    compare_observations, LogitTolerance, ParityComparison, ParityPolicy, ParityReport, ParityRule,
};

const LOGIT_TENSORS: [&str; 2] = ["prefill.logits", "decode.logits"];

/// Configuration for text-checkpoint logit parity.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CheckpointParityOptions {
    /// Accepted vocabulary-logit differences.
    pub logits: LogitTolerance,
}

impl Default for CheckpointParityOptions {
    fn default() -> Self {
        Self {
            logits: LogitTolerance {
                relative_l2_max: 0.02,
                cosine_similarity_min: 0.999,
                top_k: 5,
                top_k_overlap_min: 4,
                require_unambiguous_argmax_match: true,
                argmax_margin_min: 0.0,
            },
        }
    }
}

/// General parity report plus source artifact identities.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CheckpointParityReport {
    /// Report schema version.
    pub format_version: u32,
    /// Stable evaluation kind.
    pub kind: String,
    /// Whether every identity and tensor comparison passed.
    pub passed: bool,
    /// Actual backend artifact report.
    pub actual: PathBuf,
    /// Reference artifact report.
    pub reference: PathBuf,
    /// Shared observation parity result.
    pub parity: ParityReport,
}

/// Compares two checkpoint-probe report/tensor pairs using general parity.
pub fn compare_checkpoint_artifacts(
    actual: impl AsRef<Path>,
    reference: impl AsRef<Path>,
    options: CheckpointParityOptions,
) -> Result<CheckpointParityReport, CheckpointParityError> {
    let actual = actual.as_ref();
    let reference = reference.as_ref();
    let actual_observations = read_checkpoint_observations(actual)?;
    let reference_observations = read_checkpoint_observations(reference)?;
    let policy = ParityPolicy {
        default: ParityComparison::Exact,
        rules: vec![ParityRule {
            selector: ObservationSelector::Prefix("logits".into()),
            comparison: ParityComparison::Logits {
                tolerance: options.logits,
            },
        }],
        require_same_paths: true,
    };
    let parity = compare_observations(&actual_observations, &reference_observations, &policy)?;
    Ok(CheckpointParityReport {
        format_version: 1,
        kind: "checkpoint_parity".into(),
        passed: parity.passed,
        actual: actual.into(),
        reference: reference.into(),
        parity,
    })
}

fn read_checkpoint_observations(path: &Path) -> Result<ObservationSet, CheckpointParityError> {
    let report: serde_json::Value = serde_json::from_slice(&fs::read(path)?)?;
    let input = json_integer_array(&report, &["input", "token_ids"], path)?;
    let fed = json_integer_array(&report, &["output", "fed_token_ids"], path)?;
    let tensor_path = report
        .pointer("/output/tensor_file")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| CheckpointParityError::InvalidArtifact {
            path: path.into(),
            message: "missing string output.tensor_file".into(),
        })?;
    let tensor_path = resolve_tensor_path(path, Path::new(tensor_path));
    let bytes = fs::read(&tensor_path)?;
    let tensors = SafeTensors::deserialize(&bytes)?;

    let mut observations = ObservationSet::new();
    observations.insert(
        "input.token_ids",
        ObservationValue::Tensor(TensorObservation::new(
            vec![input.len()],
            TensorObservationData::I64(input),
        )?),
    )?;
    observations.insert(
        "output.fed_token_ids",
        ObservationValue::Tensor(TensorObservation::new(
            vec![fed.len()],
            TensorObservationData::I64(fed),
        )?),
    )?;
    for name in LOGIT_TENSORS {
        let tensor = tensors.tensor(name)?;
        if tensor.dtype() != Dtype::F32 {
            return Err(CheckpointParityError::InvalidArtifact {
                path: tensor_path.clone(),
                message: format!("tensor {name:?} must be F32, got {:?}", tensor.dtype()),
            });
        }
        let (chunks, remainder) = tensor.data().as_chunks::<4>();
        if !remainder.is_empty() {
            return Err(CheckpointParityError::InvalidArtifact {
                path: tensor_path.clone(),
                message: format!("tensor {name:?} has a partial F32 value"),
            });
        }
        let values = chunks
            .iter()
            .map(|bytes| f32::from_le_bytes(*bytes))
            .collect();
        observations.insert(
            format!("logits.{name}"),
            ObservationValue::Tensor(TensorObservation::new(
                tensor.shape().to_vec(),
                TensorObservationData::F32(values),
            )?),
        )?;
    }
    Ok(observations)
}

fn json_integer_array(
    report: &serde_json::Value,
    path: &[&str],
    source: &Path,
) -> Result<Vec<i64>, CheckpointParityError> {
    let pointer = format!("/{}", path.join("/"));
    report
        .pointer(&pointer)
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| CheckpointParityError::InvalidArtifact {
            path: source.into(),
            message: format!("missing integer array {}", path.join(".")),
        })?
        .iter()
        .map(|value| {
            value
                .as_i64()
                .ok_or_else(|| CheckpointParityError::InvalidArtifact {
                    path: source.into(),
                    message: format!("{} contains a non-integer value", path.join(".")),
                })
        })
        .collect()
}

fn resolve_tensor_path(report: &Path, tensor: &Path) -> PathBuf {
    if tensor.is_absolute() || tensor.exists() {
        return tensor.into();
    }
    let relative = report
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(tensor);
    if relative.exists() {
        return relative;
    }
    report
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(tensor.file_name().unwrap_or_default())
}

/// Invalid checkpoint evidence or parity policy.
#[derive(Debug, thiserror::Error)]
pub enum CheckpointParityError {
    /// Artifact I/O failed.
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// Artifact JSON is invalid.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    /// SafeTensors data is invalid or incomplete.
    #[error(transparent)]
    SafeTensors(#[from] safetensors::SafeTensorError),
    /// Portable observation data is invalid.
    #[error(transparent)]
    Observation(#[from] eredu_core::ObservationError),
    /// General comparison failed before a report could be produced.
    #[error(transparent)]
    Parity(#[from] crate::ParityError),
    /// A required checkpoint artifact field is absent or malformed.
    #[error("invalid checkpoint artifact {}: {message}", path.display())]
    InvalidArtifact {
        /// Artifact containing the invalid data.
        path: PathBuf,
        /// Specific failure.
        message: String,
    },
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use safetensors::tensor::{serialize_to_file, TensorView};
    use tempfile::tempdir;

    use super::*;

    fn artifact(
        root: &Path,
        name: &str,
        input: &[i64],
        fed: &[i64],
        prefill: &[f32],
        decode: &[f32],
    ) -> PathBuf {
        let tensors = root.join(format!("{name}.safetensors"));
        let prefill_bytes = prefill
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let decode_bytes = decode
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let mut views = BTreeMap::new();
        views.insert(
            "prefill.logits",
            TensorView::new(Dtype::F32, vec![1, prefill.len()], &prefill_bytes).unwrap(),
        );
        views.insert(
            "decode.logits",
            TensorView::new(Dtype::F32, vec![1, decode.len()], &decode_bytes).unwrap(),
        );
        serialize_to_file(views, None, &tensors).unwrap();
        let report = root.join(format!("{name}.json"));
        fs::write(
            &report,
            serde_json::to_vec(&serde_json::json!({
                "input": {"token_ids": input},
                "output": {"fed_token_ids": fed, "tensor_file": tensors},
            }))
            .unwrap(),
        )
        .unwrap();
        report
    }

    #[test]
    fn checkpoint_adapter_uses_general_identity_and_logit_parity() {
        let root = tempdir().unwrap();
        let actual = artifact(
            root.path(),
            "actual",
            &[1, 2],
            &[3],
            &[0.1, 0.8, 0.2],
            &[0.7, 0.2, 0.1],
        );
        let reference = artifact(
            root.path(),
            "reference",
            &[1, 2],
            &[3],
            &[0.1, 0.8, 0.2],
            &[0.7, 0.2, 0.1],
        );
        assert!(
            compare_checkpoint_artifacts(actual, reference, CheckpointParityOptions::default())
                .unwrap()
                .passed
        );
    }

    #[test]
    fn checkpoint_adapter_reports_token_and_numeric_failures_together() {
        let root = tempdir().unwrap();
        let actual = artifact(
            root.path(),
            "actual",
            &[1, 9],
            &[3],
            &[0.9, 0.1, 0.0],
            &[0.7, 0.2, 0.1],
        );
        let reference = artifact(
            root.path(),
            "reference",
            &[1, 2],
            &[3],
            &[0.1, 0.8, 0.2],
            &[0.7, 0.2, 0.1],
        );
        let report =
            compare_checkpoint_artifacts(actual, reference, CheckpointParityOptions::default())
                .unwrap();
        assert!(!report.passed);
        assert!(report
            .parity
            .failures
            .iter()
            .any(|failure| failure.contains("input.token_ids")));
        assert!(report
            .parity
            .failures
            .iter()
            .any(|failure| failure.contains("prefill.logits")));
    }
}
