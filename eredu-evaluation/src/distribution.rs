//! Model-distribution comparison independent of backend and modality.

use serde::{Deserialize, Serialize};

/// Metrics comparing one candidate categorical distribution with a reference.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DistributionMetrics {
    /// KL(reference || candidate), in nats.
    pub kl_nats: f64,
    /// Entropy of the reference distribution, in nats.
    pub reference_entropy_nats: f64,
    /// Candidate minus reference negative log likelihood for an optional target.
    pub target_nll_delta_nats: Option<f64>,
    /// RMSE after removing the mean logit from each side.
    pub centered_logit_rmse: f64,
    /// Whether top-1 indices agree.
    pub top1_agreement: bool,
    /// Fractional overlap between the selected leading sets.
    pub top_k_overlap: f64,
}

/// Compares two complete categorical logit vectors.
pub fn compare_distributions(
    reference: &[f32],
    candidate: &[f32],
    target: Option<usize>,
    top_k: usize,
) -> Result<DistributionMetrics, DistributionError> {
    if reference.is_empty() {
        return Err(DistributionError::Empty);
    }
    if reference.len() != candidate.len() {
        return Err(DistributionError::Length {
            reference: reference.len(),
            candidate: candidate.len(),
        });
    }
    if top_k == 0 {
        return Err(DistributionError::ZeroTopK);
    }
    if let Some(target) = target {
        if target >= reference.len() {
            return Err(DistributionError::Target {
                target,
                classes: reference.len(),
            });
        }
    }
    if reference
        .iter()
        .chain(candidate)
        .any(|value| !value.is_finite())
    {
        return Err(DistributionError::NonFinite);
    }

    let reference_lse = logsumexp(reference);
    let candidate_lse = logsumexp(candidate);
    let mut kl = 0.0;
    let mut entropy = 0.0;
    for (&reference_logit, &candidate_logit) in reference.iter().zip(candidate) {
        let log_p = reference_logit as f64 - reference_lse;
        let log_q = candidate_logit as f64 - candidate_lse;
        let probability = log_p.exp();
        kl += probability * (log_p - log_q);
        entropy -= probability * log_p;
    }
    let reference_mean =
        reference.iter().map(|value| *value as f64).sum::<f64>() / reference.len() as f64;
    let candidate_mean =
        candidate.iter().map(|value| *value as f64).sum::<f64>() / candidate.len() as f64;
    let centered_logit_rmse = (reference
        .iter()
        .zip(candidate)
        .map(|(&left, &right)| {
            let delta = (left as f64 - reference_mean) - (right as f64 - candidate_mean);
            delta * delta
        })
        .sum::<f64>()
        / reference.len() as f64)
        .sqrt();
    let reference_top = top_indices(reference, top_k);
    let candidate_top = top_indices(candidate, top_k);
    let overlap = reference_top
        .iter()
        .filter(|index| candidate_top.contains(index))
        .count();
    Ok(DistributionMetrics {
        kl_nats: kl.max(0.0),
        reference_entropy_nats: entropy,
        target_nll_delta_nats: target.map(|target| {
            (candidate_lse - candidate[target] as f64) - (reference_lse - reference[target] as f64)
        }),
        centered_logit_rmse,
        top1_agreement: reference_top[0] == candidate_top[0],
        top_k_overlap: overlap as f64 / reference_top.len() as f64,
    })
}

fn logsumexp(values: &[f32]) -> f64 {
    let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    maximum
        + values
            .iter()
            .map(|value| (*value as f64 - maximum).exp())
            .sum::<f64>()
            .ln()
}

fn top_indices(values: &[f32], count: usize) -> Vec<usize> {
    let mut indices = (0..values.len()).collect::<Vec<_>>();
    indices.sort_by(|left, right| {
        values[*right]
            .total_cmp(&values[*left])
            .then_with(|| left.cmp(right))
    });
    indices.truncate(count.min(indices.len()));
    indices
}

/// Invalid categorical-distribution comparison.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
pub enum DistributionError {
    /// At least one class is required.
    #[error("distribution must contain at least one class")]
    Empty,
    /// Reference and candidate cardinalities differ.
    #[error("distribution lengths differ: reference {reference}, candidate {candidate}")]
    Length {
        /// Reference cardinality.
        reference: usize,
        /// Candidate cardinality.
        candidate: usize,
    },
    /// Leading-set size must be positive.
    #[error("distribution top_k must be positive")]
    ZeroTopK,
    /// Target index is outside the distribution.
    #[error("target index {target} is outside {classes} classes")]
    Target {
        /// Invalid target.
        target: usize,
        /// Distribution cardinality.
        classes: usize,
    },
    /// Logits must be finite.
    #[error("distribution logits must be finite")]
    NonFinite,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_distributions_have_exact_metrics() {
        let metrics =
            compare_distributions(&[0.0, 2.0, 1.0], &[0.0, 2.0, 1.0], Some(1), 2).unwrap();
        assert_eq!(metrics.kl_nats, 0.0);
        assert_eq!(metrics.target_nll_delta_nats, Some(0.0));
        assert_eq!(metrics.centered_logit_rmse, 0.0);
        assert!(metrics.top1_agreement);
        assert_eq!(metrics.top_k_overlap, 1.0);
    }
}
