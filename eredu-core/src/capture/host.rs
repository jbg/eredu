//! Shared host transformations over prepaid observation payloads.

use super::*;
use crate::{TensorObservation, TensorObservationData};

/// F64 accumulation for a host observation reduction.
#[derive(Default)]
pub struct CompensatedSum {
    value: f64,
    correction: f64,
}
impl CompensatedSum {
    /// Accumulates one value while compensating finite rounding loss.
    pub fn add(&mut self, value: f64) {
        let next = self.value + value;
        if self.value.is_finite() && value.is_finite() {
            self.correction += if self.value.abs() >= value.abs() {
                (self.value - next) + value
            } else {
                (value - next) + self.value
            };
        } else {
            self.correction = 0.0;
        }
        self.value = next;
    }
    /// Returns the compensated result.
    pub fn finish(self) -> f64 {
        self.value + self.correction
    }
}

/// Finishes a reserved, already selected F32 activation.
pub fn transform_f32(
    values: Vec<f32>,
    shape: &[u64],
    transform: &CaptureTransform,
) -> Result<CapturePayload, CaptureError> {
    Ok(match transform {
        CaptureTransform::Slice
        | CaptureTransform::FullTensor
        | CaptureTransform::Preview { .. } => {
            let shape = if matches!(transform, CaptureTransform::Preview { .. }) {
                vec![values.len()]
            } else {
                shape
                    .iter()
                    .map(|n| usize::try_from(*n).map_err(|_| CaptureError::Overflow))
                    .collect::<Result<_, _>>()?
            };
            CapturePayload::Tensor(
                TensorObservation::new(shape, TensorObservationData::F32(values))
                    .map_err(|error| CaptureError::Invalid(error.to_string()))?,
            )
        }
        CaptureTransform::Summary => {
            let mut summary = CaptureSummary {
                elements: values.len() as u64,
                finite: 0,
                non_finite: 0,
                nan: 0,
                positive_infinity: 0,
                negative_infinity: 0,
                min: None,
                max: None,
                mean: None,
                rms: None,
            };
            let mut sum = CompensatedSum::default();
            let mut squares = CompensatedSum::default();
            for value in values {
                if value.is_finite() {
                    let value = f64::from(value);
                    summary.finite += 1;
                    summary.min = Some(summary.min.map_or(value, |min| min.min(value)));
                    summary.max = Some(summary.max.map_or(value, |max| max.max(value)));
                    sum.add(value);
                    squares.add(value * value);
                } else {
                    summary.non_finite += 1;
                    summary.nan += u64::from(value.is_nan());
                    summary.positive_infinity += u64::from(value == f32::INFINITY);
                    summary.negative_infinity += u64::from(value == f32::NEG_INFINITY);
                }
            }
            if summary.finite != 0 {
                summary.mean = Some(sum.finish() / summary.finite as f64);
                summary.rms = Some((squares.finish() / summary.finite as f64).sqrt());
            }
            CapturePayload::Summary(summary)
        }
        CaptureTransform::Histogram { edges } => {
            let mut histogram = CaptureHistogram {
                edges: edges.clone(),
                counts: vec![0; edges.len() - 1],
                below: 0,
                above: 0,
                non_finite: 0,
            };
            for value in values {
                if !value.is_finite() {
                    histogram.non_finite += 1;
                } else if value < edges[0] {
                    histogram.below += 1;
                } else if value > *edges.last().expect("admitted histogram") {
                    histogram.above += 1;
                } else {
                    let index = edges
                        .partition_point(|edge| *edge <= value)
                        .saturating_sub(1)
                        .min(histogram.counts.len() - 1);
                    histogram.counts[index] += 1;
                }
            }
            CapturePayload::Histogram(histogram)
        }
        _ => {
            return Err(CaptureError::Unsupported(
                "host activation transform".into(),
            ))
        }
    })
}

/// Reserves bounded host staging for local FP32 arithmetic and score ranking.
pub fn estimate_host_transform(
    workspace: CaptureWorkspaceModel,
    source: &[u64],
    selection: &CaptureSelection,
    slice: &ResolvedCaptureSlice,
) -> Result<CaptureUsage, CaptureError> {
    let mut usage = workspace.estimate(source, selection, slice)?;
    let bytes = match selection.transform {
        CaptureTransform::TopCandidates { .. } => source
            .last()
            .copied()
            .unwrap_or(0)
            .checked_mul((size_of::<f32>() + size_of::<usize>()) as u64),
        CaptureTransform::TokenScores { .. } => source
            .last()
            .copied()
            .unwrap_or(0)
            .checked_mul(size_of::<f32>() as u64),
        CaptureTransform::Summary | CaptureTransform::Histogram { .. } => {
            elements(&slice.shape)?.checked_mul(size_of::<f32>() as u64)
        }
        _ => Some(0),
    }
    .ok_or(CaptureError::Overflow)?;
    usage.host_bytes = usage
        .host_bytes
        .checked_add(bytes)
        .ok_or(CaptureError::Overflow)?;
    Ok(usage)
}

/// Summarizes a complete unfiltered row with score membership as separate evidence.
pub fn transform_scores(
    row: &[f32],
    transform: &CaptureTransform,
    domain: Option<CaptureTokenDomain<'_>>,
) -> Result<CapturePayload, CaptureError> {
    let width = u32::try_from(row.len()).map_err(|_| CaptureError::Overflow)?;
    if row.is_empty() || row.iter().any(|score| !score.is_finite()) {
        return Err(CaptureError::Invalid(
            "score capture requires a nonempty finite vocabulary row".into(),
        ));
    }
    let candidate = |id: u32| CaptureCandidate {
        token_id: id,
        score: row[id as usize],
        allowed: domain.is_none_or(|domain| domain.filter.allows(id)),
    };
    let stage = CandidateScoreStage::RawLogitsBeforeSampling;
    let domain = domain.map(|domain| domain.summary(width));
    let order = |a: u32, b: u32| {
        row[b as usize]
            .partial_cmp(&row[a as usize])
            .expect("validated finite scores")
            .then(a.cmp(&b))
    };
    Ok(match transform {
        CaptureTransform::TopCandidates { count } => {
            let count = usize::try_from(*count).map_err(|_| CaptureError::Overflow)?;
            if count == 0 || count > row.len() {
                return Err(CaptureError::Invalid(
                    "candidate count exceeds vocabulary".into(),
                ));
            }
            let mut ids: Vec<_> = (0..width).collect();
            if count < ids.len() {
                ids.select_nth_unstable_by(count, |a, b| order(*a, *b));
                ids.truncate(count);
            }
            ids.sort_unstable_by(|a, b| order(*a, *b));
            CapturePayload::Candidates(CaptureCandidates {
                stage,
                source: CandidateLogitsSource::Original,
                candidates: ids.into_iter().map(candidate).collect(),
                domain,
            })
        }
        CaptureTransform::TokenScores { token_ids, .. } => {
            let best = (0..width)
                .min_by(|a, b| order(*a, *b))
                .expect("nonempty scores");
            let runner_up = (0..width)
                .filter(|id| *id != best)
                .min_by(|a, b| order(*a, *b));
            let maximum = f64::from(row[best as usize]);
            let mut norm = CompensatedSum::default();
            for &value in row {
                norm.add((f64::from(value) - maximum).exp());
            }
            let log_partition = maximum + norm.finish().ln();
            let scores = token_ids
                .iter()
                .map(|&id| {
                    let score = *row.get(id as usize).ok_or_else(|| {
                        CaptureError::Invalid("captured token exceeds vocabulary".into())
                    })?;
                    Ok(CaptureTokenScore {
                        target: candidate(id),
                        log_probability: f64::from(score) - log_partition,
                        rank: 1 + row.iter().filter(|&&value| value > score).count() as u64,
                        strongest_alternative: (if id == best { runner_up } else { Some(best) })
                            .map(candidate),
                    })
                })
                .collect::<Result<_, CaptureError>>()?;
            CapturePayload::TokenScores(CaptureTokenScores {
                stage,
                source: CandidateLogitsSource::Original,
                vocabulary: u64::from(width),
                log_partition,
                scores,
                domain,
            })
        }
        _ => {
            return Err(CaptureError::Invalid(
                "capture transform does not select score data".into(),
            ))
        }
    })
}
