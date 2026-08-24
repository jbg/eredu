//! Backend-independent comparison of portable execution observations.

use std::collections::BTreeSet;

use eredu_core::{
    ObservationSelector, ObservationSet, ObservationValue, TensorObservation, TensorObservationData,
};
use serde::{Deserialize, Serialize};

/// Absolute and relative thresholds for ordinary floating-point observations.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct NumericTolerance {
    /// Largest accepted absolute element error.
    pub absolute_max: f64,
    /// Largest accepted L2 error relative to the reference norm.
    pub relative_l2_max: f64,
    /// Smallest accepted cosine similarity.
    pub cosine_similarity_min: f64,
}

impl NumericTolerance {
    /// Exact floating-point equality after finite-value validation.
    pub const fn exact() -> Self {
        Self {
            absolute_max: 0.0,
            relative_l2_max: 0.0,
            cosine_similarity_min: 1.0,
        }
    }

    fn validate(self) -> Result<(), ParityError> {
        if !self.absolute_max.is_finite() || self.absolute_max < 0.0 {
            return Err(ParityError::InvalidPolicy(
                "absolute_max must be finite and nonnegative".into(),
            ));
        }
        if !self.relative_l2_max.is_finite() || self.relative_l2_max < 0.0 {
            return Err(ParityError::InvalidPolicy(
                "relative_l2_max must be finite and nonnegative".into(),
            ));
        }
        if !self.cosine_similarity_min.is_finite()
            || !(-1.0..=1.0).contains(&self.cosine_similarity_min)
        {
            return Err(ParityError::InvalidPolicy(
                "cosine_similarity_min must be finite and within [-1, 1]".into(),
            ));
        }
        Ok(())
    }
}

/// Thresholds specialized for rows of vocabulary logits.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LogitTolerance {
    /// Largest accepted L2 error relative to the reference norm.
    pub relative_l2_max: f64,
    /// Smallest accepted cosine similarity.
    pub cosine_similarity_min: f64,
    /// Number of leading vocabulary entries considered for overlap.
    pub top_k: usize,
    /// Smallest accepted intersection of the two top-k sets.
    pub top_k_overlap_min: usize,
    /// Require equal argmax when the reference margin is unambiguous.
    pub require_unambiguous_argmax_match: bool,
    /// Reference top-1 minus top-2 margin above which argmax must agree.
    pub argmax_margin_min: f32,
}

impl LogitTolerance {
    fn validate(self) -> Result<(), ParityError> {
        NumericTolerance {
            absolute_max: 0.0,
            relative_l2_max: self.relative_l2_max,
            cosine_similarity_min: self.cosine_similarity_min,
        }
        .validate()?;
        if self.top_k == 0 {
            return Err(ParityError::InvalidPolicy(
                "logit top_k must be positive".into(),
            ));
        }
        if self.top_k_overlap_min > self.top_k {
            return Err(ParityError::InvalidPolicy(
                "top_k_overlap_min must not exceed top_k".into(),
            ));
        }
        if !self.argmax_margin_min.is_finite() || self.argmax_margin_min < 0.0 {
            return Err(ParityError::InvalidPolicy(
                "argmax_margin_min must be finite and nonnegative".into(),
            ));
        }
        Ok(())
    }
}

/// Comparison applied to selected observation paths.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "comparison", rename_all = "snake_case")]
pub enum ParityComparison {
    /// Values, variants, tensor shapes, and elements must be identical.
    Exact,
    /// Compare floating-point tensors or scalar values numerically.
    Numeric {
        /// Accepted numerical error.
        tolerance: NumericTolerance,
    },
    /// Compare each last-dimension vocabulary row using logit metrics.
    Logits {
        /// Accepted logit differences.
        tolerance: LogitTolerance,
    },
}

/// One selector-specific comparison rule.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParityRule {
    /// Observation paths covered by the rule.
    pub selector: ObservationSelector,
    /// Comparison applied to every selected path.
    pub comparison: ParityComparison,
}

/// Complete policy for comparing two observation sets.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParityPolicy {
    /// Comparison used when no rule matches a path.
    pub default: ParityComparison,
    /// Ordered overrides; a path matching multiple rules is invalid.
    pub rules: Vec<ParityRule>,
    /// Reject observations present on only one side.
    pub require_same_paths: bool,
}

impl ParityPolicy {
    /// Requires exact parity for every path and identical path sets.
    pub const fn exact() -> Self {
        Self {
            default: ParityComparison::Exact,
            rules: Vec::new(),
            require_same_paths: true,
        }
    }

    fn comparison_for(&self, path: &str) -> Result<&ParityComparison, ParityError> {
        let matches = self
            .rules
            .iter()
            .filter(|rule| rule.selector.matches(path))
            .collect::<Vec<_>>();
        match matches.as_slice() {
            [] => Ok(&self.default),
            [rule] => Ok(&rule.comparison),
            _ => Err(ParityError::AmbiguousRule(path.into())),
        }
    }

    fn validate(&self) -> Result<(), ParityError> {
        validate_comparison(&self.default)?;
        for rule in &self.rules {
            validate_comparison(&rule.comparison)?;
        }
        Ok(())
    }
}

fn validate_comparison(comparison: &ParityComparison) -> Result<(), ParityError> {
    match comparison {
        ParityComparison::Exact => Ok(()),
        ParityComparison::Numeric { tolerance } => tolerance.validate(),
        ParityComparison::Logits { tolerance } => tolerance.validate(),
    }
}

/// Common metrics for floating-point vectors.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NumericMetrics {
    /// Whether every value on both sides is finite.
    pub finite: bool,
    /// L2 norm of the difference divided by the reference L2 norm.
    pub relative_l2: Option<f64>,
    /// Cosine similarity.
    pub cosine_similarity: Option<f64>,
    /// Largest absolute element error.
    pub max_absolute_error: Option<f64>,
    /// Mean absolute element error.
    pub mean_absolute_error: Option<f64>,
}

/// Metrics for one vocabulary-logit row.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LogitRowMetrics {
    /// Flattened row ordinal.
    pub index: usize,
    /// Common numerical metrics.
    pub numeric: NumericMetrics,
    /// Number of shared entries in the configured top-k sets.
    pub top_k_overlap: usize,
    /// Actual top-1 vocabulary index.
    pub actual_argmax: Option<usize>,
    /// Reference top-1 vocabulary index.
    pub reference_argmax: Option<usize>,
    /// Reference top-1 minus top-2 logit margin.
    pub reference_argmax_margin: Option<f32>,
    /// Whether policy required argmax equality for this row.
    pub argmax_required: bool,
    /// Whether top-1 indices agree.
    pub argmax_match: bool,
    /// Whether this row passed.
    pub passed: bool,
}

/// Metrics emitted for one compared observation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ParityMetrics {
    /// Exact equality comparison.
    Exact,
    /// Ordinary numerical comparison.
    Numeric {
        /// Computed metrics.
        metrics: NumericMetrics,
    },
    /// Per-row vocabulary-logit comparison.
    Logits {
        /// Last logical dimension.
        vocabulary_size: usize,
        /// Metrics for every flattened leading row.
        rows: Vec<LogitRowMetrics>,
    },
}

/// Comparison result for one path.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObservationParity {
    /// Stable observation path.
    pub path: String,
    /// Whether the observation passed its rule.
    pub passed: bool,
    /// Shape or type failure when values were not comparable.
    pub failure: Option<String>,
    /// Metrics when comparison could be performed.
    pub metrics: Option<ParityMetrics>,
}

/// Complete backend/reference parity report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParityReport {
    /// Report schema version.
    pub format_version: u32,
    /// Whether every required observation passed.
    pub passed: bool,
    /// Path-level results in stable order.
    pub observations: Vec<ObservationParity>,
    /// Missing, unexpected, or invalid observations.
    pub failures: Vec<String>,
}

/// Compares two portable observation sets under one policy.
pub fn compare_observations(
    actual: &ObservationSet,
    reference: &ObservationSet,
    policy: &ParityPolicy,
) -> Result<ParityReport, ParityError> {
    policy.validate()?;
    let actual_paths = actual.iter().map(|(path, _)| path).collect::<BTreeSet<_>>();
    let reference_paths = reference
        .iter()
        .map(|(path, _)| path)
        .collect::<BTreeSet<_>>();
    let mut failures = Vec::new();
    if policy.require_same_paths {
        for path in reference_paths.difference(&actual_paths) {
            failures.push(format!("actual observations are missing {path:?}"));
        }
        for path in actual_paths.difference(&reference_paths) {
            failures.push(format!("actual observations unexpectedly contain {path:?}"));
        }
    }

    let mut observations = Vec::new();
    for path in actual_paths.intersection(&reference_paths) {
        let actual_value = actual.get(path).expect("path came from actual set");
        let reference_value = reference.get(path).expect("path came from reference set");
        let comparison = policy.comparison_for(path)?;
        let parity = compare_value(path, actual_value, reference_value, comparison)?;
        if !parity.passed {
            failures.push(format!("observation {path:?} failed parity"));
        }
        observations.push(parity);
    }
    Ok(ParityReport {
        format_version: 1,
        passed: failures.is_empty(),
        observations,
        failures,
    })
}

fn compare_value(
    path: &str,
    actual: &ObservationValue,
    reference: &ObservationValue,
    comparison: &ParityComparison,
) -> Result<ObservationParity, ParityError> {
    match comparison {
        ParityComparison::Exact => Ok(ObservationParity {
            path: path.into(),
            passed: actual == reference,
            failure: (actual != reference).then(|| "values differ".into()),
            metrics: Some(ParityMetrics::Exact),
        }),
        ParityComparison::Numeric { tolerance } => match numeric_values(actual, reference) {
            Ok((actual, reference)) => compare_numeric(path, &actual, &reference, *tolerance),
            Err(ParityError::Incomparable(failure)) => Ok(ObservationParity {
                path: path.into(),
                passed: false,
                failure: Some(failure),
                metrics: None,
            }),
            Err(error) => Err(error),
        },
        ParityComparison::Logits { tolerance } => match tensor_pair(actual, reference) {
            Ok((actual, reference)) => compare_logits(path, actual, reference, *tolerance),
            Err(ParityError::Incomparable(failure)) => Ok(ObservationParity {
                path: path.into(),
                passed: false,
                failure: Some(failure),
                metrics: None,
            }),
            Err(error) => Err(error),
        },
    }
}

fn numeric_values(
    actual: &ObservationValue,
    reference: &ObservationValue,
) -> Result<(Vec<f64>, Vec<f64>), ParityError> {
    match (actual, reference) {
        (ObservationValue::Float(actual), ObservationValue::Float(reference)) => {
            Ok((vec![*actual], vec![*reference]))
        }
        (ObservationValue::Tensor(actual), ObservationValue::Tensor(reference)) => {
            if actual.shape() != reference.shape() {
                return Err(ParityError::Incomparable(format!(
                    "tensor shapes differ: actual {:?}, reference {:?}",
                    actual.shape(),
                    reference.shape()
                )));
            }
            match (actual.data(), reference.data()) {
                (TensorObservationData::F32(actual), TensorObservationData::F32(reference)) => {
                    Ok((
                        actual.iter().map(|value| f64::from(*value)).collect(),
                        reference.iter().map(|value| f64::from(*value)).collect(),
                    ))
                }
                _ => Err(ParityError::Incomparable(
                    "numeric comparison requires matching F32 tensor values".into(),
                )),
            }
        }
        _ => Err(ParityError::Incomparable(
            "numeric comparison requires two floats or two F32 tensors".into(),
        )),
    }
}

fn tensor_pair<'a>(
    actual: &'a ObservationValue,
    reference: &'a ObservationValue,
) -> Result<(&'a TensorObservation, &'a TensorObservation), ParityError> {
    match (actual, reference) {
        (ObservationValue::Tensor(actual), ObservationValue::Tensor(reference)) => {
            Ok((actual, reference))
        }
        _ => Err(ParityError::Incomparable(
            "logit comparison requires two tensors".into(),
        )),
    }
}

fn compare_numeric(
    path: &str,
    actual: &[f64],
    reference: &[f64],
    tolerance: NumericTolerance,
) -> Result<ObservationParity, ParityError> {
    if actual.len() != reference.len() {
        return Err(ParityError::Incomparable(format!(
            "numeric value counts differ: actual {}, reference {}",
            actual.len(),
            reference.len()
        )));
    }
    let metrics = numeric_metrics(actual, reference);
    let passed = metrics.finite
        && metrics
            .max_absolute_error
            .is_some_and(|value| value <= tolerance.absolute_max)
        && metrics
            .relative_l2
            .is_some_and(|value| value <= tolerance.relative_l2_max)
        && metrics
            .cosine_similarity
            .is_some_and(|value| value >= tolerance.cosine_similarity_min);
    Ok(ObservationParity {
        path: path.into(),
        passed,
        failure: (!passed).then(|| "numeric thresholds failed".into()),
        metrics: Some(ParityMetrics::Numeric { metrics }),
    })
}

fn numeric_metrics(actual: &[f64], reference: &[f64]) -> NumericMetrics {
    let finite = actual
        .iter()
        .chain(reference)
        .all(|value| value.is_finite());
    if !finite {
        return NumericMetrics {
            finite: false,
            relative_l2: None,
            cosine_similarity: None,
            max_absolute_error: None,
            mean_absolute_error: None,
        };
    }
    let difference_squared = actual
        .iter()
        .zip(reference)
        .map(|(actual, reference)| (actual - reference).powi(2))
        .sum::<f64>();
    let actual_squared = actual.iter().map(|value| value * value).sum::<f64>();
    let reference_squared = reference.iter().map(|value| value * value).sum::<f64>();
    let dot = actual
        .iter()
        .zip(reference)
        .map(|(actual, reference)| actual * reference)
        .sum::<f64>();
    let absolute_errors = actual
        .iter()
        .zip(reference)
        .map(|(actual, reference)| (actual - reference).abs())
        .collect::<Vec<_>>();
    let denominator = actual_squared.sqrt() * reference_squared.sqrt();
    let cosine = if denominator == 0.0 {
        if actual == reference {
            1.0
        } else {
            0.0
        }
    } else {
        dot / denominator
    };
    NumericMetrics {
        finite: true,
        relative_l2: Some(difference_squared.sqrt() / reference_squared.sqrt().max(1e-12)),
        cosine_similarity: Some(cosine),
        max_absolute_error: Some(absolute_errors.iter().copied().fold(0.0, f64::max)),
        mean_absolute_error: Some(
            absolute_errors.iter().sum::<f64>() / absolute_errors.len().max(1) as f64,
        ),
    }
}

fn compare_logits(
    path: &str,
    actual: &TensorObservation,
    reference: &TensorObservation,
    tolerance: LogitTolerance,
) -> Result<ObservationParity, ParityError> {
    if actual.shape() != reference.shape() {
        return Ok(ObservationParity {
            path: path.into(),
            passed: false,
            failure: Some(format!(
                "tensor shapes differ: actual {:?}, reference {:?}",
                actual.shape(),
                reference.shape()
            )),
            metrics: None,
        });
    }
    let vocabulary_size = actual.shape().last().copied().unwrap_or(0);
    if vocabulary_size == 0 {
        return Err(ParityError::Incomparable(
            "logit tensor must have a nonempty last dimension".into(),
        ));
    }
    let (actual, reference) = match (actual.data(), reference.data()) {
        (TensorObservationData::F32(actual), TensorObservationData::F32(reference)) => {
            (actual, reference)
        }
        _ => {
            return Err(ParityError::Incomparable(
                "logit comparison requires F32 tensor values".into(),
            ))
        }
    };
    let mut rows = Vec::with_capacity(actual.len() / vocabulary_size);
    for (index, (actual, reference)) in actual
        .chunks_exact(vocabulary_size)
        .zip(reference.chunks_exact(vocabulary_size))
        .enumerate()
    {
        rows.push(compare_logit_row(index, actual, reference, tolerance));
    }
    let passed = rows.iter().all(|row| row.passed);
    Ok(ObservationParity {
        path: path.into(),
        passed,
        failure: (!passed).then(|| "logit thresholds failed".into()),
        metrics: Some(ParityMetrics::Logits {
            vocabulary_size,
            rows,
        }),
    })
}

fn compare_logit_row(
    index: usize,
    actual: &[f32],
    reference: &[f32],
    tolerance: LogitTolerance,
) -> LogitRowMetrics {
    let actual_f64 = actual
        .iter()
        .map(|value| f64::from(*value))
        .collect::<Vec<_>>();
    let reference_f64 = reference
        .iter()
        .map(|value| f64::from(*value))
        .collect::<Vec<_>>();
    let numeric = numeric_metrics(&actual_f64, &reference_f64);
    if !numeric.finite {
        return LogitRowMetrics {
            index,
            numeric,
            top_k_overlap: 0,
            actual_argmax: None,
            reference_argmax: None,
            reference_argmax_margin: None,
            argmax_required: false,
            argmax_match: false,
            passed: false,
        };
    }
    let top_count = tolerance.top_k.min(reference.len());
    let actual_top = top_indices(actual, top_count);
    let reference_order = top_indices(reference, top_count.max(2).min(reference.len()));
    let reference_top = &reference_order[..top_count];
    let top_k_overlap = actual_top
        .iter()
        .filter(|index| reference_top.contains(index))
        .count();
    let actual_argmax = actual_top.first().copied();
    let reference_argmax = reference_top.first().copied();
    let reference_argmax_margin = reference_argmax.map(|argmax| {
        let runner_up = reference_order.get(1).copied().unwrap_or(argmax);
        reference[argmax] - reference[runner_up]
    });
    let argmax_required = tolerance.require_unambiguous_argmax_match
        && reference_argmax_margin.is_some_and(|margin| margin > tolerance.argmax_margin_min);
    let argmax_match = actual_argmax == reference_argmax;
    let passed = numeric
        .relative_l2
        .is_some_and(|value| value <= tolerance.relative_l2_max)
        && numeric
            .cosine_similarity
            .is_some_and(|value| value >= tolerance.cosine_similarity_min)
        && top_k_overlap >= tolerance.top_k_overlap_min.min(top_count)
        && (!argmax_required || argmax_match);
    LogitRowMetrics {
        index,
        numeric,
        top_k_overlap,
        actual_argmax,
        reference_argmax,
        reference_argmax_margin,
        argmax_required,
        argmax_match,
        passed,
    }
}

fn top_indices(values: &[f32], count: usize) -> Vec<usize> {
    let mut indexes = (0..values.len()).collect::<Vec<_>>();
    indexes.sort_by(|left, right| {
        values[*right]
            .total_cmp(&values[*left])
            .then_with(|| left.cmp(right))
    });
    indexes.truncate(count);
    indexes
}

/// Invalid comparison policy or incompatible observations.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
pub enum ParityError {
    /// A comparison threshold is invalid.
    #[error("invalid parity policy: {0}")]
    InvalidPolicy(String),
    /// More than one rule matched the same path.
    #[error("multiple parity rules match observation {0:?}")]
    AmbiguousRule(String),
    /// Selected values cannot be compared by the requested method.
    #[error("incomparable observations: {0}")]
    Incomparable(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(shape: Vec<usize>, values: Vec<f32>) -> ObservationValue {
        ObservationValue::Tensor(
            TensorObservation::new(shape, TensorObservationData::F32(values)).unwrap(),
        )
    }

    fn tokens(values: Vec<i64>) -> ObservationValue {
        ObservationValue::Tensor(
            TensorObservation::new(vec![values.len()], TensorObservationData::I64(values)).unwrap(),
        )
    }

    #[test]
    fn exact_parity_handles_tokens_and_path_identity() {
        let mut actual = ObservationSet::new();
        actual.insert("decode.tokens", tokens(vec![1, 2])).unwrap();
        let reference = actual.clone();
        assert!(
            compare_observations(&actual, &reference, &ParityPolicy::exact())
                .unwrap()
                .passed
        );
    }

    #[test]
    fn logit_parity_computes_shared_metrics_once() {
        let mut actual = ObservationSet::new();
        actual
            .insert("model.logits", tensor(vec![1, 3], vec![0.0, 2.0, 1.0]))
            .unwrap();
        let mut reference = ObservationSet::new();
        reference
            .insert("model.logits", tensor(vec![1, 3], vec![0.0, 2.1, 0.9]))
            .unwrap();
        let policy = ParityPolicy {
            default: ParityComparison::Logits {
                tolerance: LogitTolerance {
                    relative_l2_max: 0.1,
                    cosine_similarity_min: 0.99,
                    top_k: 2,
                    top_k_overlap_min: 2,
                    require_unambiguous_argmax_match: true,
                    argmax_margin_min: 0.1,
                },
            },
            rules: Vec::new(),
            require_same_paths: true,
        };
        assert!(
            compare_observations(&actual, &reference, &policy)
                .unwrap()
                .passed
        );
    }
}
