//! Complete-source vocabulary reductions; no shard-local probability is global evidence.
use super::*;
use eredu_core::ObservationPosition;

fn invalid(message: &str) -> CaptureError {
    CaptureError::Invalid(message.into())
}

pub(super) fn is_vocabulary(transform: &CaptureTransform) -> bool {
    matches!(
        transform,
        CaptureTransform::TokenScores { .. } | CaptureTransform::TopCandidates { .. }
    )
}

fn complete_slice(slice: &ResolvedCaptureSlice, shape: &[u64]) -> bool {
    slice.shape == shape
        && slice.ends == shape
        && slice.starts.iter().all(|start| *start == 0)
        && slice.strides.iter().all(|stride| *stride == 1)
}

pub(super) fn validate_projection(
    transform: &CaptureTransform,
    projection: &CaptureSlicePartition,
) -> Result<(), CaptureError> {
    if !is_vocabulary(transform) {
        return Ok(());
    }
    let shape = projection.global_shape();
    let valid = !shape.contains(&0)
        && projection.local_shape() == shape
        && complete_slice(projection.global_slice(), shape)
        && projection.fragments().len() == 1
        && projection.fragments().iter().all(|fragment| {
            complete_slice(fragment.local(), shape) && complete_slice(fragment.destination(), shape)
        });
    if !valid {
        return Err(CaptureError::Unsupported(
            "vocabulary reduction requires one complete ordered global logits source".into(),
        ));
    }
    Ok(())
}

fn domain_valid(domain: Option<CandidateDomain>, vocabulary: u64) -> bool {
    domain.is_none_or(|domain| {
        domain.vocabulary == vocabulary
            && domain.allowed_tokens <= vocabulary
            && (!domain.constrained || domain.allowed_tokens < vocabulary)
    })
}

fn candidate_valid(
    candidate: &CaptureCandidate,
    domain: Option<CandidateDomain>,
    vocabulary: u64,
) -> bool {
    u64::from(candidate.token_id) < vocabulary
        && candidate.score.is_finite()
        && match domain {
            None => candidate.allowed,
            Some(domain) if domain.allowed_tokens == 0 => !candidate.allowed,
            Some(domain) if domain.allowed_tokens == vocabulary => candidate.allowed,
            Some(_) => true,
        }
}

fn probability_bounds_valid(
    target: f32,
    alternative: f32,
    probability: f64,
    vocabulary: u64,
) -> bool {
    // The strongest alternative and target together identify the global maximum.
    // Bound the shifted mass independently of log_partition: at extreme logits,
    // adding log mass to the maximum can round away even in F64.
    let target = f64::from(target);
    let alternative = f64::from(alternative);
    let delta = target - target.max(alternative);
    let maximum_log_mass = (vocabulary as f64).ln();
    let minimum_log_mass = (-(target - alternative).abs()).exp().ln_1p();
    // Permit F32 reduction error in the mass and F64 rounding of a large delta,
    // without making the bound depend on the unshifted score magnitude.
    let tolerance = 32.0 * f64::EPSILON * delta.abs().max(1.0)
        + 8.0 * f64::from(f32::EPSILON) * maximum_log_mass.max(1.0);
    probability >= delta - maximum_log_mass - tolerance
        && probability <= delta - minimum_log_mass + tolerance
}

/// Structural and arithmetic consistency only; independent numerical comparisons
/// establish that the producer evaluated the actual model distribution.
pub(super) fn validate_payload(
    selection: &CaptureSelection,
    position: ObservationPosition,
    shape: &[u64],
    payload: &CapturePayload,
) -> Result<(), CaptureError> {
    let vocabulary = *shape
        .last()
        .ok_or_else(|| invalid("scalar vocabulary source"))?;
    if vocabulary == 0 {
        return Err(invalid("empty vocabulary source"));
    }
    let expected_source = if position == ObservationPosition::AfterIntervention {
        CandidateLogitsSource::Effective
    } else {
        CandidateLogitsSource::Original
    };
    let valid = match (&selection.transform, payload) {
        (CaptureTransform::TopCandidates { count }, CapturePayload::Candidates(value)) => {
            value.source == expected_source
                && *count <= vocabulary
                && value.candidates.len() as u64 == *count
                && domain_valid(value.domain, vocabulary)
                && value
                    .candidates
                    .iter()
                    .enumerate()
                    .all(|(index, candidate)| {
                        candidate_valid(candidate, value.domain, vocabulary)
                            && value.candidates[..index]
                                .iter()
                                .all(|previous| previous.token_id != candidate.token_id)
                            && (index == 0 || value.candidates[index - 1].score >= candidate.score)
                    })
        }
        (CaptureTransform::TokenScores { token_ids }, CapturePayload::TokenScores(value)) => {
            value.source == expected_source
                && value.vocabulary == vocabulary
                && value.log_partition.is_finite()
                && value.scores.len() == token_ids.len()
                && domain_valid(value.domain, vocabulary)
                && value.scores.iter().zip(token_ids).all(|(score, id)| {
                    let target = &score.target;
                    // At very large F32 magnitudes, adding the small log mass can
                    // round in F64. Keep the independently stable log probability.
                    let tolerance = 32.0
                        * f64::EPSILON
                        * value
                            .log_partition
                            .abs()
                            .max(f64::from(target.score).abs())
                            .max(1.0);
                    candidate_valid(target, value.domain, vocabulary)
                        && target.token_id == *id
                        && score.log_probability.is_finite()
                        && score.log_probability <= 0.0
                        && score.rank > 0
                        && score.rank <= vocabulary
                        && ((f64::from(target.score) - value.log_partition) - score.log_probability)
                            .abs()
                            <= tolerance
                        && match &score.strongest_alternative {
                            None => {
                                vocabulary == 1 && score.rank == 1 && score.log_probability == 0.0
                            }
                            Some(other) => {
                                vocabulary > 1
                                    && other.token_id != *id
                                    && candidate_valid(other, value.domain, vocabulary)
                                    && ((other.score > target.score) == (score.rank > 1))
                                    && value.log_partition + tolerance >= f64::from(other.score)
                                    && probability_bounds_valid(
                                        target.score,
                                        other.score,
                                        score.log_probability,
                                        vocabulary,
                                    )
                            }
                        }
                })
        }
        _ => false,
    };
    if !valid {
        return Err(invalid(
            "vocabulary payload differs from its admitted reduction or score semantics",
        ));
    }
    Ok(())
}
