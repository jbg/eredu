//! Host reference arithmetic for checking bounded component evidence.
//!
//! This is evaluation policy, not an inference primitive. Inputs must come from
//! effective parameters and the same forward's residual; checkpoint metadata or
//! another trial's normalization denominator is not sufficient. The decomposed
//! score is the affine readout before any nonlinear output transform.

use eredu_core::component::ComponentNormalizationKind;

/// Effective normalization parameters for a host reference calculation.
pub struct ReadoutNormalization<'a> {
    /// Centered LayerNorm or RMSNorm.
    pub kind: ComponentNormalizationKind,
    /// Epsilon inside the normalization square root.
    pub epsilon: f64,
    /// Effective gain, including any architecture offset.
    pub gain: &'a [f64],
    /// Effective additive normalization bias, if present.
    pub bias: Option<&'a [f64]>,
    /// Independent contiguous normalization groups.
    pub groups: usize,
}

/// Linearized readout at the measured residual, with bias kept separate.
#[derive(Debug, Clone, PartialEq)]
pub struct MeasuredReadout {
    /// Covector on unnormalized residual writes; LayerNorm centering is included.
    pub direction: Vec<f64>,
    /// Normalization and readout biases, once per score, not once per component.
    pub offset: f64,
}

/// Malformed or non-finite reference evidence is not measured zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum AttributionError {
    /// Input, gain, bias or write-column geometry does not agree.
    #[error("attribution geometry mismatch")]
    Geometry,
    /// Evidence or arithmetic is non-finite, or normalization is invalid.
    #[error("invalid or non-finite attribution evidence")]
    NonFinite,
}

/// Neumaier summation, retaining small signed terms between cancelling large terms.
pub fn signed_sum(values: impl IntoIterator<Item = f64>) -> f64 {
    let (mut sum, mut compensation) = (0.0_f64, 0.0_f64);
    for value in values {
        let next = sum + value;
        compensation += if sum.abs() >= value.abs() {
            (sum - next) + value
        } else {
            (value - next) + sum
        };
        sum = next;
    }
    sum + compensation
}

impl MeasuredReadout {
    /// Constructs an exact local covector for the measured forward's affine score.
    /// Pass a difference of readout rows and biases for a token-versus-token margin.
    /// This does not assume the denominator remains fixed under an intervention.
    pub fn new(
        residual: &[f64],
        readout: &[f64],
        readout_bias: f64,
        norm: ReadoutNormalization<'_>,
    ) -> Result<Self, AttributionError> {
        let width = residual.len();
        if width == 0
            || readout.len() != width
            || norm.gain.len() != width
            || norm.bias.is_some_and(|bias| bias.len() != width)
            || norm.groups == 0
            || width % norm.groups != 0
        {
            return Err(AttributionError::Geometry);
        }
        if !readout_bias.is_finite()
            || !norm.epsilon.is_finite()
            || norm.epsilon < 0.0
            || residual
                .iter()
                .chain(readout)
                .chain(norm.gain)
                .chain(norm.bias.unwrap_or_default())
                .any(|x| !x.is_finite())
        {
            return Err(AttributionError::NonFinite);
        }
        let mut direction = vec![0.0; width];
        let group_width = width / norm.groups;
        for group in 0..norm.groups {
            let start = group * group_width;
            let end = start + group_width;
            let center = if norm.kind == ComponentNormalizationKind::Layer {
                signed_sum(residual[start..end].iter().map(|x| x / group_width as f64))
            } else {
                0.0
            };
            let denominator = if norm.kind == ComponentNormalizationKind::Identity {
                1.0
            } else {
                (signed_sum(residual[start..end].iter().map(|x| {
                    (x - center).powi(2)
                        / if norm.kind == ComponentNormalizationKind::L2 {
                            1.0
                        } else {
                            group_width as f64
                        }
                })) + norm.epsilon)
                    .sqrt()
            };
            if !denominator.is_finite() || denominator <= 0.0 {
                return Err(AttributionError::NonFinite);
            }
            let centered_covector = if norm.kind == ComponentNormalizationKind::Layer {
                signed_sum((start..end).map(|i| readout[i] * norm.gain[i] / group_width as f64))
            } else {
                0.0
            };
            for i in start..end {
                // Center the gain-weighted row, not the raw readout before gain.
                direction[i] = (readout[i] * norm.gain[i] - centered_covector) / denominator;
            }
        }
        let offset = readout_bias
            + norm.bias.map_or(0.0, |bias| {
                signed_sum(readout.iter().zip(bias).map(|(w, b)| w * b))
            });
        if !offset.is_finite() || direction.iter().any(|x| !x.is_finite()) {
            return Err(AttributionError::NonFinite);
        }
        Ok(Self { direction, offset })
    }

    /// Signed scalar activation times an effective output-projection column.
    pub fn contribution(
        &self,
        activation: f64,
        write_column: &[f64],
    ) -> Result<f64, AttributionError> {
        if self.direction.len() != write_column.len() {
            return Err(AttributionError::Geometry);
        }
        if !activation.is_finite() || write_column.iter().any(|x| !x.is_finite()) {
            return Err(AttributionError::NonFinite);
        }
        let value =
            activation * signed_sum(self.direction.iter().zip(write_column).map(|(d, w)| d * w));
        if value.is_finite() {
            Ok(value)
        } else {
            Err(AttributionError::NonFinite)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signed_components_embedding_and_bias_reconstruct_declared_scores_and_margins() {
        let embedding = [1.1, -0.7, 0.2, 0.5];
        let columns = [
            [0.5, -0.3, 0.7, 0.1],
            [-0.2, 0.6, 0.4, -0.9],
            [0.3, 0.8, -0.1, 0.4],
        ];
        // Distinguishable signed scalars include a negative gated-unit value.
        let activations = [1.7, -0.8, 0.3];
        let gain = [0.4, 1.7, 0.8, 2.1];
        let bias = [0.1, -0.2, 0.4, 0.3];
        let target = [0.2, 0.5, -0.7, 0.9];
        let alternative = [-0.4, 0.2, 0.6, 0.1];
        let residual: Vec<_> = (0..4)
            .map(|i| embedding[i] + signed_sum((0..3).map(|u| activations[u] * columns[u][i])))
            .collect();
        for kind in [
            ComponentNormalizationKind::Identity,
            ComponentNormalizationKind::Layer,
            ComponentNormalizationKind::Rms,
            ComponentNormalizationKind::L2,
        ] {
            for groups in [1, 2] {
                for difference in [false, true] {
                    let weights: Vec<_> = (0..4)
                        .map(|i| target[i] - if difference { alternative[i] } else { 0.0 })
                        .collect();
                    let readout_bias = if difference { 0.6 - (-0.4) } else { 0.6 };
                    let readout = MeasuredReadout::new(
                        &residual,
                        &weights,
                        readout_bias,
                        ReadoutNormalization {
                            kind,
                            epsilon: 1e-5,
                            gain: &gain,
                            bias: Some(&bias),
                            groups,
                        },
                    )
                    .unwrap();
                    let sum =
                        readout.offset
                            + readout.contribution(1.0, &embedding).unwrap()
                            + signed_sum((0..3).map(|u| {
                                readout.contribution(activations[u], &columns[u]).unwrap()
                            }));
                    let mut reference = readout_bias;
                    for g in 0..groups {
                        let n = 4 / groups;
                        let x = &residual[g * n..(g + 1) * n];
                        let mean = if kind == ComponentNormalizationKind::Layer {
                            x.iter().sum::<f64>() / n as f64
                        } else {
                            0.0
                        };
                        let scale = if kind == ComponentNormalizationKind::Identity {
                            1.0
                        } else {
                            (x.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                                / if kind == ComponentNormalizationKind::L2 {
                                    1.0
                                } else {
                                    n as f64
                                }
                                + 1e-5)
                                .sqrt()
                        };
                        for (i, v) in x.iter().enumerate() {
                            let j = g * n + i;
                            reference += weights[j] * (gain[j] * (v - mean) / scale + bias[j]);
                        }
                    }
                    assert!(
                        (sum - reference).abs() < 1e-12,
                        "{kind:?}: {sum} versus {reference}"
                    );
                }
            }
        }
    }

    #[test]
    fn cancellation_preserves_small_signed_contributions() {
        assert_eq!(signed_sum([1e16, 1.0, -1e16, -0.25]), 0.75);
    }

    #[test]
    fn identity_readout_accepts_zero_residual_without_normalizing_it() {
        let readout = MeasuredReadout::new(
            &[0.0, 0.0],
            &[2.0, -3.0],
            0.25,
            ReadoutNormalization {
                kind: ComponentNormalizationKind::Identity,
                epsilon: 0.0,
                gain: &[1.0, 1.0],
                bias: None,
                groups: 1,
            },
        )
        .unwrap();
        assert_eq!(readout.contribution(1.0, &[0.0, 0.0]).unwrap(), 0.0);
        assert_eq!(readout.contribution(0.5, &[4.0, 1.0]).unwrap(), 2.5);
        assert_eq!(readout.offset, 0.25);
    }
}
