//! General evaluation evidence and performance summaries.

use std::collections::BTreeMap;

use eredu_core::{
    ObservationSet, ObservationValue, RealtimeOutputFrame, TensorObservation, TensorObservationData,
};
use eredu_nn::Tensor;
use serde::{Deserialize, Serialize};

/// Portable evidence emitted by an evaluator, backend probe, or reference.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvaluationEvidence {
    /// Evidence schema version.
    pub format_version: u32,
    /// Extensible evaluation kind, such as `text_checkpoint` or `realtime_audio`.
    pub kind: String,
    /// Stable producer and environment fields.
    pub provenance: BTreeMap<String, String>,
    /// Path-addressed values available for comparison or inspection.
    pub observations: ObservationSet,
}

impl EvaluationEvidence {
    /// Creates version-one evidence with no provenance fields.
    pub fn new(kind: impl Into<String>, observations: ObservationSet) -> Self {
        Self {
            format_version: 1,
            kind: kind.into(),
            provenance: BTreeMap::new(),
            observations,
        }
    }

    /// Adds one stable provenance field.
    pub fn with_provenance(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.provenance.insert(key.into(), value.into());
        self
    }
}

/// Materializes a neutral tensor as F32 evaluation evidence.
pub fn observe_f32_tensor<T: Tensor>(
    tensor: &T,
    context: &T::Context,
) -> Result<TensorObservation, EvidenceError> {
    let shape = observation_shape(tensor.shape())?;
    let values = tensor.to_f32_vec(context)?;
    TensorObservation::new(shape, TensorObservationData::F32(values)).map_err(Into::into)
}

/// Materializes a neutral tensor as signed integer evaluation evidence.
pub fn observe_i32_tensor<T: Tensor>(
    tensor: &T,
    context: &T::Context,
) -> Result<TensorObservation, EvidenceError> {
    let shape = observation_shape(tensor.shape())?;
    let values = tensor
        .to_i32_vec(context)?
        .into_iter()
        .map(i64::from)
        .collect();
    TensorObservation::new(shape, TensorObservationData::I64(values)).map_err(Into::into)
}

fn observation_shape(shape: &[i32]) -> Result<Vec<usize>, EvidenceError> {
    shape
        .iter()
        .map(|dimension| {
            usize::try_from(*dimension).map_err(|_| EvidenceError::NegativeDimension(*dimension))
        })
        .collect()
}

/// Converts a completed portable realtime frame into general observations.
pub fn observe_realtime_frame(
    frame: &RealtimeOutputFrame,
) -> Result<ObservationSet, EvidenceError> {
    let mut observations = ObservationSet::new();
    insert_realtime_tokens(
        &mut observations,
        "tokens.text",
        frame.batch(),
        frame.text_tokens(),
    )?;
    insert_realtime_tokens(
        &mut observations,
        "tokens.audio_decisions",
        frame.batch(),
        frame.decision_audio_tokens(),
    )?;
    insert_realtime_tokens(
        &mut observations,
        "tokens.audio_sampled",
        frame.batch(),
        frame.sampled_audio_tokens(),
    )?;
    if let Some(tokens) = frame.output_audio_tokens() {
        insert_realtime_tokens(
            &mut observations,
            "tokens.audio_output",
            frame.batch(),
            tokens,
        )?;
    }
    for diagnostic in frame.diagnostics() {
        observations.insert(
            format!("decisions.{}.logits", diagnostic.prediction()),
            ObservationValue::Tensor(diagnostic.tensor().clone()),
        )?;
    }
    Ok(observations)
}

fn insert_realtime_tokens(
    observations: &mut ObservationSet,
    path: &str,
    batch: usize,
    tokens: &[i32],
) -> Result<(), EvidenceError> {
    if batch == 0 || !tokens.len().is_multiple_of(batch) {
        return Err(EvidenceError::RealtimeTokenShape {
            path: path.into(),
            batch,
            values: tokens.len(),
        });
    }
    observations.insert(
        path,
        ObservationValue::Tensor(TensorObservation::new(
            vec![batch, tokens.len() / batch],
            TensorObservationData::I64(tokens.iter().copied().map(i64::from).collect()),
        )?),
    )?;
    Ok(())
}

/// Summary of repeated operation latencies.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LatencySummary {
    /// Number of samples.
    pub samples: usize,
    /// Arithmetic mean in milliseconds.
    pub mean_ms: f64,
    /// Median in milliseconds.
    pub p50_ms: f64,
    /// 95th percentile in milliseconds.
    pub p95_ms: f64,
    /// Largest latency in milliseconds.
    pub max_ms: f64,
    /// Optional deadline in milliseconds.
    pub deadline_ms: Option<f64>,
    /// Samples strictly exceeding the deadline.
    pub deadline_misses: usize,
}

/// Summarizes finite, nonnegative latency samples.
pub fn summarize_latencies(
    samples_ms: &[f64],
    deadline_ms: Option<f64>,
) -> Result<LatencySummary, EvidenceError> {
    if samples_ms.is_empty() {
        return Err(EvidenceError::EmptyLatencies);
    }
    if samples_ms
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(EvidenceError::InvalidLatency);
    }
    if deadline_ms.is_some_and(|value| !value.is_finite() || value < 0.0) {
        return Err(EvidenceError::InvalidDeadline);
    }
    let mut ordered = samples_ms.to_vec();
    ordered.sort_by(f64::total_cmp);
    let percentile = |fraction: f64| {
        let index = ((ordered.len() - 1) as f64 * fraction).ceil() as usize;
        ordered[index]
    };
    Ok(LatencySummary {
        samples: ordered.len(),
        mean_ms: ordered.iter().sum::<f64>() / ordered.len() as f64,
        p50_ms: percentile(0.50),
        p95_ms: percentile(0.95),
        max_ms: *ordered.last().expect("samples are nonempty"),
        deadline_ms,
        deadline_misses: deadline_ms.map_or(0, |deadline| {
            ordered.iter().filter(|value| **value > deadline).count()
        }),
    })
}

/// Invalid evaluation evidence.
#[derive(Debug, thiserror::Error)]
pub enum EvidenceError {
    /// Backend-neutral tensor materialization failed.
    #[error(transparent)]
    Tensor(#[from] eredu_nn::Error),
    /// Portable observation construction failed.
    #[error(transparent)]
    Observation(#[from] eredu_core::ObservationError),
    /// Tensor shapes must not contain negative dimensions.
    #[error("observed tensor has negative dimension {0}")]
    NegativeDimension(i32),
    /// A realtime token vector cannot be represented with its declared batch.
    #[error("realtime observation {path:?} has {values} values for batch {batch}")]
    RealtimeTokenShape {
        /// Observation path.
        path: String,
        /// Declared batch.
        batch: usize,
        /// Token count.
        values: usize,
    },
    /// At least one latency is required.
    #[error("latency summary requires at least one sample")]
    EmptyLatencies,
    /// Latencies must be finite and nonnegative.
    #[error("latency samples must be finite and nonnegative")]
    InvalidLatency,
    /// A deadline must be finite and nonnegative.
    #[error("latency deadline must be finite and nonnegative")]
    InvalidDeadline,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn latency_summary_is_deterministic() {
        let summary = summarize_latencies(&[4.0, 1.0, 3.0, 2.0], Some(2.5)).unwrap();
        assert_eq!(summary.mean_ms, 2.5);
        assert_eq!(summary.p50_ms, 3.0);
        assert_eq!(summary.p95_ms, 4.0);
        assert_eq!(summary.deadline_misses, 2);
    }

    #[test]
    fn realtime_frames_use_the_general_observation_schema() {
        let diagnostic =
            eredu_core::RealtimeDecisionDiagnostics::new(0, vec![1, 3], vec![0.0, 2.0, 1.0])
                .unwrap();
        let frame = RealtimeOutputFrame::new(
            1,
            vec![7],
            vec![8, 9],
            vec![8],
            Some(vec![6]),
            vec![diagnostic],
        );
        let observations = observe_realtime_frame(&frame).unwrap();
        assert!(observations.get("tokens.text").is_some());
        assert!(observations.get("decisions.0.logits").is_some());
    }
}
