//! Architecture-owned plans for converting dense SafeTensors checkpoints.
//!
//! A concrete backend executes these plans literally. It does not infer
//! eligible matrices, canonicalize source names, derive companion names, or
//! choose how the converted architecture is represented in `config.json`.

use std::collections::BTreeSet;

use eredu_checkpoint::WeightQuantization;
use serde_json::Value;

/// One exact dense source and the physical tensors that replace it.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct SafetensorsQuantizationTarget {
    source_name: String,
    weight_name: String,
    scales_name: String,
    biases_name: Option<String>,
}

impl SafetensorsQuantizationTarget {
    /// Declares exact source and output identities for one quantized matrix.
    pub fn new(
        source_name: impl Into<String>,
        weight_name: impl Into<String>,
        scales_name: impl Into<String>,
        biases_name: Option<impl Into<String>>,
    ) -> Self {
        Self {
            source_name: source_name.into(),
            weight_name: weight_name.into(),
            scales_name: scales_name.into(),
            biases_name: biases_name.map(Into::into),
        }
    }

    /// Returns the exact dense source identity.
    pub fn source_name(&self) -> &str {
        &self.source_name
    }

    /// Returns the exact packed-weight output identity.
    pub fn weight_name(&self) -> &str {
        &self.weight_name
    }

    /// Returns the exact scale-companion output identity.
    pub fn scales_name(&self) -> &str {
        &self.scales_name
    }

    /// Returns the exact affine-bias output identity, when required.
    pub fn biases_name(&self) -> Option<&str> {
        self.biases_name.as_deref()
    }
}

/// A validated architecture decision for one complete checkpoint conversion.
#[derive(Debug, Clone, PartialEq)]
pub struct SafetensorsQuantizationPlan {
    quantization: WeightQuantization,
    targets: Vec<SafetensorsQuantizationTarget>,
    output_config: Value,
}

impl SafetensorsQuantizationPlan {
    /// Validates exact targets and the architecture-authored output config.
    pub fn new(
        quantization: impl Into<WeightQuantization>,
        targets: impl IntoIterator<Item = SafetensorsQuantizationTarget>,
        output_config: Value,
    ) -> Result<Self, SafetensorsQuantizationPlanError> {
        let quantization = quantization.into();
        quantization.validate().map_err(|error| {
            SafetensorsQuantizationPlanError::InvalidQuantization(error.to_string())
        })?;
        if quantization.gguf_iquant().is_some() {
            return Err(SafetensorsQuantizationPlanError::CheckpointNativeEncoding);
        }
        if !output_config.is_object() {
            return Err(SafetensorsQuantizationPlanError::InvalidOutputConfig);
        }

        let mut targets = targets.into_iter().collect::<Vec<_>>();
        if targets.is_empty() {
            return Err(SafetensorsQuantizationPlanError::EmptyTargets);
        }
        targets.sort_by(|left, right| left.source_name.cmp(&right.source_name));

        let mut sources = BTreeSet::new();
        let mut outputs = BTreeSet::new();
        for target in &targets {
            for (role, name) in [
                ("source", target.source_name()),
                ("weight", target.weight_name()),
                ("scales", target.scales_name()),
            ] {
                if name.trim().is_empty() {
                    return Err(SafetensorsQuantizationPlanError::EmptyIdentity { role });
                }
            }
            if let Some(name) = target.biases_name() {
                if name.trim().is_empty() {
                    return Err(SafetensorsQuantizationPlanError::EmptyIdentity { role: "biases" });
                }
            }
            if quantization.has_biases() != target.biases_name().is_some() {
                return Err(SafetensorsQuantizationPlanError::BiasMismatch {
                    target: target.source_name.clone(),
                    required: quantization.has_biases(),
                });
            }
            if !sources.insert(target.source_name.clone()) {
                return Err(SafetensorsQuantizationPlanError::DuplicateSource {
                    name: target.source_name.clone(),
                });
            }
            for name in std::iter::once(target.weight_name())
                .chain(std::iter::once(target.scales_name()))
                .chain(target.biases_name())
            {
                if !outputs.insert(name.to_owned()) {
                    return Err(SafetensorsQuantizationPlanError::DuplicateOutput {
                        name: name.to_owned(),
                    });
                }
            }
        }

        Ok(Self {
            quantization,
            targets,
            output_config,
        })
    }

    /// Returns the exact requested packed encoding.
    pub const fn quantization(&self) -> WeightQuantization {
        self.quantization
    }

    /// Returns exact conversion targets in deterministic source-name order.
    pub fn targets(&self) -> &[SafetensorsQuantizationTarget] {
        &self.targets
    }

    /// Returns the complete architecture-authored output `config.json` value.
    pub fn output_config(&self) -> &Value {
        &self.output_config
    }
}

/// Invalid architecture-owned SafeTensors conversion plan.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
pub enum SafetensorsQuantizationPlanError {
    /// The packed encoding itself is invalid.
    #[error("invalid checkpoint quantization: {0}")]
    InvalidQuantization(String),
    /// GGUF-native blocks cannot be produced in SafeTensors conversion.
    #[error("checkpoint-native GGUF encodings cannot be produced by SafeTensors conversion")]
    CheckpointNativeEncoding,
    /// Conversion without an exact target is not meaningful.
    #[error("SafeTensors quantization requires at least one exact target")]
    EmptyTargets,
    /// A physical identity was empty.
    #[error("SafeTensors quantization {role} identity cannot be empty")]
    EmptyIdentity {
        /// Identity role.
        role: &'static str,
    },
    /// A dense source was selected more than once.
    #[error("SafeTensors quantization source {name:?} is selected more than once")]
    DuplicateSource {
        /// Duplicate source identity.
        name: String,
    },
    /// Two targets publish the same physical tensor.
    #[error("SafeTensors quantization output {name:?} is produced more than once")]
    DuplicateOutput {
        /// Duplicate output identity.
        name: String,
    },
    /// Companion presence disagrees with the requested encoding.
    #[error("SafeTensors quantization source {target:?} affine-bias presence differs from encoding requirement {required}")]
    BiasMismatch {
        /// Dense source identity.
        target: String,
        /// Whether the encoding requires affine biases.
        required: bool,
    },
    /// The architecture did not provide a complete JSON object.
    #[error("SafeTensors quantization output config must be a JSON object")]
    InvalidOutputConfig,
}

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_checkpoint::AffineQuantization;
    use serde_json::json;

    #[test]
    fn plan_preserves_exact_nonconventional_companion_identities() {
        let plan = SafetensorsQuantizationPlan::new(
            AffineQuantization::default(),
            [SafetensorsQuantizationTarget::new(
                "released_legacy_weight",
                "runtime.matrix",
                "runtime.scale",
                Some("runtime.zero"),
            )],
            json!({"architecture_owned": true}),
        )
        .unwrap();
        let target = &plan.targets()[0];
        assert_eq!(target.source_name(), "released_legacy_weight");
        assert_eq!(target.weight_name(), "runtime.matrix");
        assert_eq!(target.scales_name(), "runtime.scale");
        assert_eq!(target.biases_name(), Some("runtime.zero"));
    }

    #[test]
    fn plan_rejects_companion_policy_drift_and_collisions() {
        let missing_bias = SafetensorsQuantizationPlan::new(
            AffineQuantization::default(),
            [SafetensorsQuantizationTarget::new(
                "source",
                "weight",
                "scale",
                None::<String>,
            )],
            json!({}),
        )
        .unwrap_err();
        assert!(matches!(
            missing_bias,
            SafetensorsQuantizationPlanError::BiasMismatch { .. }
        ));

        let collision = SafetensorsQuantizationPlan::new(
            WeightQuantization::MxFp4,
            [
                SafetensorsQuantizationTarget::new(
                    "first",
                    "first.weight",
                    "shared.scale",
                    None::<String>,
                ),
                SafetensorsQuantizationTarget::new(
                    "second",
                    "second.weight",
                    "shared.scale",
                    None::<String>,
                ),
            ],
            json!({}),
        )
        .unwrap_err();
        assert!(matches!(
            collision,
            SafetensorsQuantizationPlanError::DuplicateOutput { .. }
        ));
    }
}
