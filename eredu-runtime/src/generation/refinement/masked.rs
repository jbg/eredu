//! Scalar equations for confidence-based unmasking and float64 logit noise.

use eredu_core::refinement::MaskedSampling;

/// Invalid numerical input rejected before a transfer can be published.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum MaskedNumericalError {
    /// Rows and token positions must have complete matching geometry.
    #[error("masked refinement has inconsistent token or score geometry")]
    Geometry,
    /// Rows must contain a finite candidate and no NaN or positive infinity.
    #[error("invalid masked refinement position scores")]
    Scores,
    /// Random draws must belong to the open unit interval.
    #[error("masked refinement requires random draws in (0, 1)")]
    Random,
}

/// Samples one complete row without losing float64 noise precision.
/// Gumbel-power selection uses the equivalent log-domain comparison, avoiding
/// overflow in `exp(logits)` while retaining the published temperature equation.
pub fn sample_position(
    logits: &[f32],
    sampling: MaskedSampling,
    temperature: f32,
    mut draw: impl FnMut() -> f64,
) -> Result<u32, MaskedNumericalError> {
    validate_row(logits)?;
    if !temperature.is_finite() || temperature < 0.0 {
        return Err(MaskedNumericalError::Scores);
    }
    let uniform = |value: f64| {
        if value.is_finite() && value > 0.0 && value < 1.0 {
            Ok(value)
        } else {
            Err(MaskedNumericalError::Random)
        }
    };
    if temperature == 0.0 {
        return Ok(argmax(logits.iter().map(|x| f64::from(*x))));
    }
    match sampling {
        MaskedSampling::GumbelPower => {
            let mut values = Vec::with_capacity(logits.len());
            for score in logits {
                let u = uniform(draw())?;
                values.push(f64::from(*score) - f64::from(temperature) * (-u.ln()).ln());
            }
            Ok(argmax(values.into_iter()))
        }
        MaskedSampling::Categorical => {
            let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let weights: Vec<_> = logits
                .iter()
                .map(|x| ((f64::from(*x) - f64::from(maximum)) / f64::from(temperature)).exp())
                .collect();
            let total: f64 = weights.iter().sum();
            let selected = uniform(draw())? * total;
            let mut cumulative = 0.0;
            for (id, weight) in weights.iter().enumerate() {
                cumulative += weight;
                if selected < cumulative {
                    return Ok(id as u32);
                }
            }
            // Floating-point summation can leave the final interval one ulp short.
            Ok(weights
                .iter()
                .rposition(|x| *x > 0.0)
                .expect("finite candidate") as u32)
        }
    }
}

fn validate_row(row: &[f32]) -> Result<(), MaskedNumericalError> {
    if row.is_empty() || row.len() > u32::MAX as usize {
        return Err(MaskedNumericalError::Geometry);
    }
    if row.iter().any(|x| x.is_nan() || *x == f32::INFINITY) || !row.iter().any(|x| x.is_finite()) {
        return Err(MaskedNumericalError::Scores);
    }
    Ok(())
}

fn argmax(values: impl Iterator<Item = f64>) -> u32 {
    let mut best = 0;
    let mut maximum = f64::NEG_INFINITY;
    for (id, value) in values.enumerate() {
        if value > maximum {
            best = id as u32;
            maximum = value;
        }
    }
    best
}

/// Selects masked positions by the probability of their sampled token.
/// Threshold selection keeps the most confident masked position even when all
/// probabilities fall below the threshold. Ties retain increasing position order.
pub fn transfer_positions(
    logits: &[Vec<f32>],
    sampled: &[u32],
    eligible: &[bool],
    scheduled: u32,
    threshold: Option<f32>,
) -> Result<Vec<bool>, MaskedNumericalError> {
    if logits.len() != sampled.len()
        || logits.len() != eligible.len()
        || threshold.is_some_and(|x| !x.is_finite() || !(0.0..=1.0).contains(&x))
    {
        return Err(MaskedNumericalError::Geometry);
    }
    let mut positions = Vec::new();
    for (position, row) in logits.iter().enumerate() {
        validate_row(row)?;
        let score = *row
            .get(sampled[position] as usize)
            .ok_or(MaskedNumericalError::Geometry)?;
        if eligible[position] {
            let maximum = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let denominator: f64 = row
                .iter()
                .map(|x| (f64::from(*x) - f64::from(maximum)).exp())
                .sum();
            let confidence = (f64::from(score) - f64::from(maximum)).exp() / denominator;
            positions.push((position, confidence));
        }
    }
    positions.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    let mut mask = vec![false; logits.len()];
    for (rank, (position, confidence)) in positions.iter().enumerate() {
        mask[*position] = match threshold {
            Some(threshold) => rank == 0 || *confidence >= f64::from(threshold),
            None => rank < scheduled as usize,
        };
    }
    Ok(mask)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noisy_selection_and_confidence_use_distinct_distributions() {
        let mut draws = [0.001, 0.999].into_iter();
        let token = sample_position(&[2.0, 0.0], MaskedSampling::GumbelPower, 1.0, || {
            draws.next().unwrap()
        })
        .unwrap();
        assert_eq!(token, 1);
        // Confidence is the raw probability of the noisy winner, not the maximum probability.
        let mask = transfer_positions(
            &[vec![2.0, 0.0], vec![0.0, 0.0]],
            &[token, 0],
            &[true, true],
            0,
            Some(0.9),
        )
        .unwrap();
        assert_eq!(mask, [false, true]);
    }

    #[test]
    fn transfer_preserves_fixed_positions_and_makes_threshold_progress() {
        let logits = vec![vec![0.0, 0.0]; 4];
        assert_eq!(
            transfer_positions(&logits, &[0; 4], &[false, true, true, true], 2, None).unwrap(),
            [false, true, true, false]
        );
        assert_eq!(
            transfer_positions(&logits, &[0; 4], &[false, true, true, true], 0, Some(1.0)).unwrap(),
            [false, true, false, false]
        );
        assert_eq!(
            transfer_positions(&logits, &[0; 4], &[false; 4], 0, Some(1.0)).unwrap(),
            [false; 4]
        );
    }

    #[test]
    fn greedy_sampling_consumes_no_noise_and_invalid_rows_are_rejected() {
        assert_eq!(
            sample_position(
                &[1.0, 1.0, -1.0],
                MaskedSampling::GumbelPower,
                0.0,
                || panic!("greedy draw")
            )
            .unwrap(),
            0
        );
        assert_eq!(
            sample_position(&[f32::NAN, 0.0], MaskedSampling::GumbelPower, 0.0, || 0.5),
            Err(MaskedNumericalError::Scores)
        );
        assert_eq!(
            sample_position(
                &[f32::NEG_INFINITY],
                MaskedSampling::Categorical,
                1.0,
                || 0.5
            ),
            Err(MaskedNumericalError::Scores)
        );
    }
}
