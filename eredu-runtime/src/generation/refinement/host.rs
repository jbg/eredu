//! Immutable host storage for the shared refinement policies.

use eredu_core::sampling::{
    host::{HostSampling, HostScores},
    host_random::HostRandomState,
};
use std::{num::NonZeroU32, sync::Arc};

use eredu_core::{
    refinement::{
        PositionScoreClass, RefinementBackend, RefinementFinalizationBackend, RefinementScoreRows,
        RefinementScoreSelection, RefinementSelectedScore,
    },
    sampling::{SamplingBackend, TokenDomain},
    BackendFailure, TokenFilter,
};

/// Adds the payload of an immutable host refinement value to a snapshot bound.
pub(crate) fn add_retained_value(
    total: Option<u64>,
    value: super::RefinementValue<'_, HostRefinement>,
) -> Option<u64> {
    use super::RefinementValue;
    let payload = match value {
        RefinementValue::Tokens(tokens) => (tokens.len() as u64).checked_mul(4)?,
        RefinementValue::Scores(scores) => scores.retained_bytes(),
        RefinementValue::Mask(mask) => mask.len() as u64,
        RefinementValue::Random(_) => 0,
    };
    total?.checked_add(payload)?.checked_add(64)
}

#[derive(Clone)]
pub(crate) struct ScoreMatrix {
    values: Arc<[f32]>,
    shape: [usize; 2],
    pub(crate) element: Option<eredu_nn::TensorElementType>,
}

impl ScoreMatrix {
    pub(crate) fn new(values: Vec<f32>, shape: [usize; 2]) -> Result<Self, BackendFailure> {
        let [positions, width] = shape;
        if positions == 0 || width == 0 || positions.checked_mul(width) != Some(values.len()) {
            return Err(HostSampling::error(
                "invalid refinement score geometry".into(),
            ));
        }
        Ok(Self {
            values: values.into(),
            shape,
            element: None,
        })
    }

    pub(crate) fn from_row(row: &HostScores) -> Self {
        Self {
            values: row.values().into(),
            shape: [1, row.values().len()],
            element: None,
        }
    }

    pub(crate) fn retained_bytes(&self) -> u64 {
        (std::mem::size_of::<f32>() * self.values.len()) as u64
    }

    pub(crate) fn row(&self, position: usize) -> Result<HostScores, BackendFailure> {
        let [rows, width] = self.shape;
        if position >= rows {
            return Err(HostSampling::error("invalid refinement position".into()));
        }
        HostScores::from_row(self.values[position * width..(position + 1) * width].to_vec())
    }

    pub(crate) fn read_selected(
        &self,
        selections: Vec<RefinementScoreSelection>,
    ) -> Result<RefinementScoreRows, BackendFailure> {
        selections
            .into_iter()
            .map(|selection| {
                let row = self.row(selection.position as usize)?;
                let (maximum, mass) = row.normalization()?;
                let log_mass = mass.ln();
                selection
                    .token_ids
                    .into_iter()
                    .map(|token_id| {
                        let score =
                            row.values()
                                .get(token_id as usize)
                                .copied()
                                .ok_or_else(|| {
                                    HostSampling::error(
                                        "refinement score ID outside vocabulary".into(),
                                    )
                                })?;
                        Ok(RefinementSelectedScore {
                            token_id,
                            score,
                            log_probability: (f64::from(score) - f64::from(maximum)) - log_mass,
                        })
                    })
                    .collect()
            })
            .collect()
    }
    pub(crate) fn values(&self) -> &[f32] {
        &self.values
    }

    fn entropies(&self) -> Result<Vec<f64>, BackendFailure> {
        (0..self.shape[0])
            .map(|position| {
                let row = self.row(position)?;
                let (maximum, mass) = row.normalization()?;
                let log_mass = mass.ln();
                Ok(row
                    .values()
                    .iter()
                    .filter(|score| score.is_finite())
                    .map(|&score| {
                        let log_probability = f64::from(score) - f64::from(maximum) - log_mass;
                        -log_probability.exp() * log_probability
                    })
                    .sum())
            })
            .collect()
    }
}

/// Host numerical adapter for both refinement policies.
pub(crate) struct HostRefinement;

impl RefinementBackend for HostRefinement {
    type Logits = ScoreMatrix;
    type Token = Arc<[u32]>;
    type RandomState = HostRandomState;
    type Context = ();
    type Error = BackendFailure;
    type Mask = Arc<[bool]>;
    type ConditioningFormat = ();

    fn read_tokens(tokens: &Self::Token, _: &()) -> Result<Vec<u32>, BackendFailure> {
        Ok(tokens.to_vec())
    }
    fn read_mask(mask: &Self::Mask, _: &()) -> Result<Vec<bool>, BackendFailure> {
        Ok(mask.to_vec())
    }
    fn masked_score_shape(scores: &Self::Logits) -> Option<[usize; 2]> {
        Some(scores.shape)
    }
    fn read_scores(scores: &Self::Logits, _: &()) -> Result<Vec<f32>, BackendFailure> {
        Ok(scores.values.to_vec())
    }
    fn masked_uniforms(
        count: usize,
        random: &mut HostRandomState,
        _: &(),
    ) -> Result<Vec<f64>, BackendFailure> {
        Ok((0..count).map(|_| random.open_unit_interval()).collect())
    }
    fn materialize_tokens(tokens: &[u32], _: &()) -> Result<Self::Token, BackendFailure> {
        Ok(tokens.into())
    }
    fn materialize_mask(flags: &[bool], _: &()) -> Result<Self::Mask, BackendFailure> {
        Ok(flags.into())
    }
    fn materialize_scores(
        scores: &[f32],
        shape: [usize; 2],
        _: &(),
    ) -> Result<ScoreMatrix, BackendFailure> {
        ScoreMatrix::new(scores.to_vec(), shape)
    }
    fn validate_refinement_logits(
        scores: &ScoreMatrix,
        canvas: NonZeroU32,
        domain: TokenDomain,
    ) -> Result<(), BackendFailure> {
        if scores.shape != [canvas.get() as usize, domain.cardinality()] {
            return Err(HostSampling::error(
                "scores differ from retained canvas geometry".into(),
            ));
        }
        Ok(())
    }
    fn settle_refinement(
        _: &Self::Token,
        _: &Self::Token,
        _: &ScoreMatrix,
        _: &ScoreMatrix,
        _: &Self::Mask,
        _: &HostRandomState,
        _: &(),
    ) -> Result<(), BackendFailure> {
        Ok(())
    }
    fn settle_initial_canvas(
        _: &Self::Token,
        _: &HostRandomState,
        _: &(),
    ) -> Result<(), BackendFailure> {
        Ok(())
    }

    fn pin_scores(
        scores: &ScoreMatrix,
        tokens: &Self::Token,
        pins: &Self::Mask,
        _: &(),
    ) -> Result<ScoreMatrix, BackendFailure> {
        let [rows, width] = scores.shape;
        if tokens.len() != rows || pins.len() != rows {
            return Err(HostSampling::error(
                "invalid refinement pin geometry".into(),
            ));
        }
        let mut pinned = scores.clone();
        let values = Arc::make_mut(&mut pinned.values);
        for ((row, &token), &pin) in values
            .chunks_exact_mut(width)
            .zip(tokens.iter())
            .zip(pins.iter())
        {
            if pin {
                row.fill(f32::NEG_INFINITY);
                *row.get_mut(token as usize).ok_or_else(|| {
                    HostSampling::error("pinned token outside vocabulary".into())
                })? = 0.;
            }
        }
        Ok(pinned)
    }

    fn scale_temperature(
        scores: &ScoreMatrix,
        temperature: f32,
        _: &(),
    ) -> Result<ScoreMatrix, BackendFailure> {
        let [rows, width] = scores.shape;
        let mut values = Vec::with_capacity(scores.values.len());
        for position in 0..rows {
            let row = scores.row(position)?;
            row.normalization()?;
            let scaled = HostSampling::scale_temperature(&row, temperature, &())?;
            values.extend_from_slice(scaled.values());
        }
        let mut scaled = ScoreMatrix::new(values, [rows, width])?;
        scaled.element = scores.element;
        Ok(scaled)
    }

    fn uniform_tokens(
        canvas: NonZeroU32,
        domain: TokenDomain,
        random: &mut HostRandomState,
        _: &(),
    ) -> Result<Self::Token, BackendFailure> {
        let count = domain.cardinality();
        let choices = std::num::NonZeroUsize::new(count)
            .filter(|_| u32::try_from(count - 1).is_ok())
            .ok_or_else(|| HostSampling::error("invalid refinement token domain".into()))?;
        Ok((0..canvas.get())
            .map(|_| random.index(choices) as u32)
            .collect())
    }

    fn categorical_tokens(
        scores: &ScoreMatrix,
        random: &mut HostRandomState,
        _: &(),
    ) -> Result<Self::Token, BackendFailure> {
        (0..scores.shape[0])
            .map(|position| {
                HostSampling::sample_processed(&scores.row(position)?, 1.0, Some(random), &())
            })
            .collect()
    }

    fn argmax_tokens(scores: &ScoreMatrix, _: &()) -> Result<Self::Token, BackendFailure> {
        (0..scores.shape[0])
            .map(|position| HostSampling::sample_processed(&scores.row(position)?, 0.0, None, &()))
            .collect()
    }

    fn entropy_bound_mask(
        scores: &ScoreMatrix,
        bound: f32,
        _: &(),
    ) -> Result<Self::Mask, BackendFailure> {
        let entropies = scores.entropies()?;
        let mut order = (0..entropies.len()).collect::<Vec<_>>();
        order.sort_unstable_by(|&a, &b| {
            entropies[a]
                .partial_cmp(&entropies[b])
                .expect("validated score rows have finite entropy")
                .then(a.cmp(&b))
        });
        let mut accepted = vec![false; entropies.len()];
        let mut prefix = 0.0;
        for position in order {
            accepted[position] = prefix <= f64::from(bound);
            prefix += entropies[position];
        }
        Ok(accepted.into())
    }

    fn select_tokens(
        mask: &Self::Mask,
        accepted: &Self::Token,
        other: &Self::Token,
        _: &(),
    ) -> Result<Self::Token, BackendFailure> {
        if mask.len() != accepted.len() || mask.len() != other.len() {
            return Err(HostSampling::error(
                "refinement selection geometry differs".into(),
            ));
        }
        Ok(mask
            .iter()
            .zip(accepted.iter().zip(other.iter()))
            .map(|(&accept, (&yes, &no))| if accept { yes } else { no })
            .collect())
    }

    fn retain_self_conditioning(
        scores: &ScoreMatrix,
        _: &(),
        _: &(),
    ) -> Result<ScoreMatrix, BackendFailure> {
        // The host cache retains the immutable matrix. The upload adapter
        // narrows it to the selected model's score dtype at invocation.
        Ok(scores.clone())
    }

    fn stable_and_confident(
        scores: &ScoreMatrix,
        argmax: &Self::Token,
        preceding: &[Self::Token],
        confidence_threshold: f32,
        pins: Option<&Self::Mask>,
        _: &(),
    ) -> Result<bool, BackendFailure> {
        if pins.is_some_and(|pins| pins.len() != scores.shape[0]) {
            return Err(HostSampling::error(
                "invalid refinement confidence mask".into(),
            ));
        }
        if preceding.iter().any(|previous| previous != argmax) {
            return Ok(false);
        }
        let mut sum = 0.0;
        let mut active = 0usize;
        for (position, entropy) in scores.entropies()?.into_iter().enumerate() {
            if !pins.is_some_and(|pins| pins[position]) {
                active += 1;
                sum += entropy;
            }
        }
        Ok(sum / (active.max(1) as f64) < f64::from(confidence_threshold))
    }
}

impl RefinementFinalizationBackend for HostRefinement {
    type CanvasScores = ScoreMatrix;
    type CanvasTokens = Arc<[u32]>;
    type PositionScores = HostScores;
    type FinalizationContext = ();
    type FinalizationError = BackendFailure;

    fn canvas_score_shape(scores: &ScoreMatrix) -> Option<[usize; 2]> {
        Some(scores.shape)
    }
    fn position_scores(
        scores: &ScoreMatrix,
        position: usize,
        _: &(),
    ) -> Result<HostScores, BackendFailure> {
        scores.row(position)
    }
    fn filter_position_scores(
        scores: &HostScores,
        filter: &TokenFilter,
        context: &(),
    ) -> Result<HostScores, BackendFailure> {
        HostSampling::apply_token_filter(scores, filter, context)
    }
    fn greedy_position(
        scores: &HostScores,
        context: &(),
    ) -> Result<Self::CanvasTokens, BackendFailure> {
        Ok(Arc::from([HostSampling::sample_processed(
            scores, 0.0, None, context,
        )?]))
    }
    fn validate_candidate(
        token: &Self::CanvasTokens,
        domain: TokenDomain,
        context: &(),
    ) -> Result<Self::CanvasTokens, BackendFailure> {
        HostSampling::validate_token(&scalar(token)?, domain, context)?;
        Ok(token.clone())
    }
    fn candidate_at(
        candidates: &Self::CanvasTokens,
        position: usize,
        _: &(),
    ) -> Result<Self::CanvasTokens, BackendFailure> {
        Ok(Arc::from([*candidates.get(position).ok_or_else(|| {
            HostSampling::error("invalid canvas position".into())
        })?]))
    }
    fn candidate_score_is_finite(
        candidate: &Self::CanvasTokens,
        scores: &HostScores,
        _: &(),
    ) -> Result<bool, BackendFailure> {
        scores
            .values()
            .get(scalar(candidate)? as usize)
            .map(|score| score.is_finite())
            .ok_or_else(|| HostSampling::error("candidate outside vocabulary".into()))
    }
    fn classify_position_scores(
        scores: &HostScores,
        _: &(),
    ) -> Result<PositionScoreClass, BackendFailure> {
        let row = scores.values();
        Ok(
            if row
                .iter()
                .any(|score| score.is_nan() || *score == f32::INFINITY)
            {
                PositionScoreClass::Invalid
            } else if row.iter().any(|score| score.is_finite()) {
                PositionScoreClass::HasFinite
            } else {
                PositionScoreClass::NoFinite
            },
        )
    }
}

pub(crate) fn scalar(token: &[u32]) -> Result<u32, BackendFailure> {
    let [id] = token else {
        return Err(HostSampling::error(
            "expected a single canonical token".into(),
        ));
    };
    Ok(*id)
}
