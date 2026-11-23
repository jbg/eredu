//! Numerical sampling over one owned host vocabulary row.

use std::sync::Arc;

use crate::{
    sampling::{PenaltyConfig, SamplingBackend, TokenDomain},
    BackendFailure, BackendFailureKind, TokenFilter,
};
use eredu_nn::mechanism_memory::{
    MechanismBacking, MechanismBytes, MechanismInvocation, MechanismMemoryContract,
    MechanismPlacement, MechanismStorage, MechanismStorageRole, SamplingMode, StorageRetention,
};

use super::host_random::HostRandomState;

/// One FP32 vocabulary row. Negative infinity denotes an excluded token.
#[derive(Clone)]
pub struct HostScores(Arc<Vec<f32>>);

impl HostScores {
    /// Takes ownership of a nonempty, token-addressable score row.
    pub fn from_row(scores: Vec<f32>) -> Result<Self, BackendFailure> {
        if scores.is_empty() || u32::try_from(scores.len() - 1).is_err() {
            return Err(invalid(
                "sampling requires a nonempty token-addressable score row",
            ));
        }
        Ok(Self(Arc::new(scores)))
    }

    /// Borrows the row in vocabulary order.
    pub fn values(&self) -> &[f32] {
        self.0.as_slice()
    }

    /// Estimates retained row storage.
    pub fn retained_bytes(&self) -> Option<u64> {
        (self.0.capacity() as u64)
            .checked_mul(size_of::<f32>() as u64)?
            .checked_add(
                (size_of::<Self>() + size_of::<Vec<f32>>() + 2 * size_of::<usize>()) as u64,
            )
    }

    /// Computes the log weights of the positive difference of two normalized rows.
    pub fn positive_difference(&self, draft: &Self) -> Result<Option<Self>, BackendFailure> {
        if self.0.len() != draft.0.len() {
            return Err(invalid("speculative vocabulary widths differ"));
        }
        let (target_max, target_mass) = self.normalization()?;
        let (draft_max, draft_mass) = draft.normalization()?;
        let mut nonempty = false;
        let scores = self
            .0
            .iter()
            .zip(draft.0.iter())
            .map(|(&target, &draft)| {
                let mass = weight(target, target_max) / target_mass
                    - weight(draft, draft_max) / draft_mass;
                if mass > 0.0 {
                    nonempty = true;
                    mass.ln() as f32
                } else {
                    f32::NEG_INFINITY
                }
            })
            .collect();
        nonempty.then(|| Self::from_row(scores)).transpose()
    }

    fn best(&self) -> Result<(u32, f32), BackendFailure> {
        let mut best = None;
        for (index, &score) in self.0.iter().enumerate() {
            if score == f32::NEG_INFINITY {
                continue;
            }
            if !score.is_finite() {
                return Err(invalid("non-finite score in the admitted vocabulary"));
            }
            if best.is_none_or(|(_, maximum)| score > maximum) {
                best = Some((index as u32, score));
            }
        }
        best.ok_or_else(|| invalid("no finite score in the admitted vocabulary"))
    }

    /// Returns the maximum and max-shifted probability mass.
    pub fn normalization(&self) -> Result<(f32, f64), BackendFailure> {
        let (_, maximum) = self.best()?;
        let mass = self.0.iter().map(|&score| weight(score, maximum)).sum();
        Ok((maximum, mass))
    }
}

fn weight(score: f32, maximum: f32) -> f64 {
    (f64::from(score) - f64::from(maximum)).exp()
}

fn invalid(detail: impl Into<Box<dyn std::error::Error + Send + Sync>>) -> BackendFailure {
    BackendFailure::new(
        BackendFailureKind::InvalidInput,
        std::io::Error::other(detail),
    )
}

/// Shared sampling arithmetic for a single host score row.
pub struct HostSampling;

impl SamplingBackend for HostSampling {
    type Logits = HostScores;
    type Token = u32;
    type RandomState = HostRandomState;
    type Context = ();
    type Error = BackendFailure;

    fn mechanism_memory(
        invocation: &MechanismInvocation,
    ) -> Result<MechanismMemoryContract, eredu_nn::Error> {
        let MechanismInvocation::Sampling {
            rows: 1,
            vocabulary,
            element,
            mode,
            top_k,
            top_p,
            min_p,
            penalties,
            ..
        } = *invocation
        else {
            return Err(eredu_nn::Error::backend(
                "host sampling expects one vocabulary row",
            ));
        };
        let values = invocation.logical_values()?;
        let mut storage = Vec::new();
        let mut scratch = |name: &str,
                           count: u64,
                           bytes: usize,
                           conditional: bool|
         -> Result<(), eredu_nn::Error> {
            let bytes = count
                .checked_mul(bytes as u64)
                .ok_or_else(|| eredu_nn::Error::backend("host sampling workspace overflows"))?;
            storage.push(MechanismStorage {
                name: name.into(),
                role: MechanismStorageRole::Scratch,
                payload: MechanismBytes {
                    lower: if conditional { 0 } else { bytes },
                    upper: Some(bytes),
                },
                capacity: MechanismBytes::unknown(if conditional { 0 } else { bytes }),
                backing: MechanismBacking::Invocation,
                placement: MechanismPlacement::Host,
                retention: StorageRetention::NativeCompletion,
                detail: "host vocabulary sampling buffer".into(),
            });
            Ok(())
        };
        scratch("host_score_row", vocabulary, size_of::<f32>(), false)?;
        if element != eredu_nn::TensorElementType::F32 {
            scratch("score_conversion", vocabulary, size_of::<f32>(), true)?;
        }
        // A closed token filter can copy the row even for greedy sampling.
        scratch("filtered_score_row", vocabulary, size_of::<f32>(), true)?;
        if penalties || mode != SamplingMode::Greedy || top_k > 0 || top_p || min_p {
            scratch("transformed_score_row", vocabulary, size_of::<f32>(), true)?;
        }
        if penalties {
            scratch("history_counts", vocabulary, size_of::<usize>(), true)?;
        }
        if top_k > 0 {
            scratch("top_k_threshold", vocabulary, size_of::<f32>(), false)?;
        }
        if top_p {
            scratch("top_p_order", vocabulary, size_of::<usize>(), false)?;
        }
        Ok(MechanismMemoryContract {
            values,
            storage,
            missing: vec!["allocator capacity, score extraction layout, controller mask backing and inline request state".into()],
        })
    }

    fn error(message: String) -> BackendFailure {
        invalid(message)
    }

    fn validate_token(token: &u32, domain: TokenDomain, _: &()) -> Result<u32, BackendFailure> {
        if (*token as usize) < domain.cardinality() {
            Ok(*token)
        } else {
            Err(invalid("token is outside the admitted vocabulary"))
        }
    }

    fn scale_temperature(
        logits: &HostScores,
        temperature: f32,
        _: &(),
    ) -> Result<HostScores, BackendFailure> {
        if !temperature.is_finite() || temperature <= 0.0 {
            return Err(invalid("sampling temperature must be finite and positive"));
        }
        if temperature == 1.0 {
            return Ok(logits.clone());
        }
        let mut output = logits.clone();
        for score in Arc::make_mut(&mut output.0) {
            if *score != f32::NEG_INFINITY {
                *score = (f64::from(*score) / f64::from(temperature)) as f32;
                if !score.is_finite() {
                    return Err(invalid("temperature scaling produced a non-finite score"));
                }
            }
        }
        Ok(output)
    }

    fn apply_penalties(
        logits: &HostScores,
        history: &[u32],
        penalties: PenaltyConfig,
        _: &(),
    ) -> Result<HostScores, BackendFailure> {
        if penalties.is_identity() || history.is_empty() || penalties.repeat_last_n == 0 {
            return Ok(logits.clone());
        }
        let start = usize::try_from(penalties.repeat_last_n)
            .map_or(0, |count| history.len().saturating_sub(count));
        let mut counts = vec![0usize; logits.0.len()];
        for &token in &history[start..] {
            if let Some(count) = counts.get_mut(token as usize) {
                *count += 1;
            }
        }
        let mut output = logits.clone();
        for (score, count) in Arc::make_mut(&mut output.0).iter_mut().zip(counts) {
            if count == 0 || *score == f32::NEG_INFINITY {
                continue;
            }
            *score = if *score > 0.0 {
                *score / penalties.repeat_penalty
            } else {
                *score * penalties.repeat_penalty
            };
            *score -= penalties.frequency_penalty * count as f32 + penalties.presence_penalty;
            if !score.is_finite() {
                return Err(invalid("penalties produced a non-finite score"));
            }
        }
        Ok(output)
    }

    fn apply_top_k(mut logits: HostScores, k: i32, _: &()) -> Result<HostScores, BackendFailure> {
        if k <= 0 || k as usize >= logits.0.len() {
            return Ok(logits);
        }
        logits.best()?;
        let mut ordered = (*logits.0).clone();
        let (_, threshold, _) = ordered.select_nth_unstable_by(k as usize - 1, |a, b| {
            b.partial_cmp(a).expect("validated scores")
        });
        for score in Arc::make_mut(&mut logits.0) {
            if *score < *threshold {
                *score = f32::NEG_INFINITY;
            }
        }
        Ok(logits)
    }

    fn apply_top_p(mut logits: HostScores, p: f32, _: &()) -> Result<HostScores, BackendFailure> {
        if p >= 1.0 {
            return Ok(logits);
        }
        let (max, mass) = logits.normalization()?;
        let mut indices = (0..logits.0.len()).collect::<Vec<_>>();
        let scores = Arc::make_mut(&mut logits.0);
        indices.sort_unstable_by(|&a, &b| {
            scores[b]
                .partial_cmp(&scores[a])
                .expect("validated scores")
                .then(a.cmp(&b))
        });
        let mut cumulative = 0.0;
        for index in indices {
            if cumulative > f64::from(p.max(0.0)) * mass {
                scores[index] = f32::NEG_INFINITY;
            } else {
                cumulative += weight(scores[index], max);
            }
        }
        Ok(logits)
    }

    fn apply_min_p(mut logits: HostScores, p: f32, _: &()) -> Result<HostScores, BackendFailure> {
        if p <= 0.0 {
            return Ok(logits);
        }
        let (_, maximum) = logits.best()?;
        for score in Arc::make_mut(&mut logits.0) {
            if weight(*score, maximum) < f64::from(p) {
                *score = f32::NEG_INFINITY;
            }
        }
        Ok(logits)
    }

    fn apply_token_filter(
        logits: &HostScores,
        filter: &TokenFilter,
        _: &(),
    ) -> Result<HostScores, BackendFailure> {
        let mut output = logits.clone();
        if let Some(mask) = filter
            .allowed_mask_for(logits.0.len())
            .map_err(BackendFailure::from_error)?
        {
            for (score, allowed) in Arc::make_mut(&mut output.0).iter_mut().zip(mask.iter()) {
                if !allowed {
                    *score = f32::NEG_INFINITY;
                }
            }
        }
        output.best()?;
        Ok(output)
    }

    fn apply_mirostat(
        logits: &HostScores,
        history: &[u32],
        penalties: PenaltyConfig,
        temperature: f32,
        mu: f32,
        context: &(),
    ) -> Result<HostScores, BackendFailure> {
        if !mu.is_finite() {
            return Err(invalid("Mirostat surprise cutoff must be finite"));
        }
        let logits = Self::apply_penalties(logits, history, penalties, context)?;
        let mut logits = Self::scale_temperature(&logits, temperature, context)?;
        let (best, _) = logits.best()?;
        let (max, mass) = logits.normalization()?;
        let cutoff = (-f64::from(mu)).exp2() * mass;
        for (id, score) in Arc::make_mut(&mut logits.0).iter_mut().enumerate() {
            if id != best as usize && weight(*score, max) < cutoff {
                *score = f32::NEG_INFINITY;
            }
        }
        Ok(logits)
    }

    fn sample_raw(
        logits: &HostScores,
        temperature: f32,
        random: Option<&mut HostRandomState>,
        context: &(),
    ) -> Result<u32, BackendFailure> {
        if temperature == 0.0 {
            return Ok(logits.best()?.0);
        }
        let scaled = Self::scale_temperature(logits, temperature, context)?;
        Self::sample_processed(&scaled, temperature, random, context)
    }

    fn sample_processed(
        logits: &HostScores,
        temperature: f32,
        random: Option<&mut HostRandomState>,
        _: &(),
    ) -> Result<u32, BackendFailure> {
        if temperature == 0.0 {
            return Ok(logits.best()?.0);
        }
        let (max, mass) = logits.normalization()?;
        let random =
            random.ok_or_else(|| invalid("stochastic sampling requires a random stream"))?;
        let draw = random.unit_interval() * mass;
        let mut cumulative = 0.0;
        let mut last = None;
        for (id, &score) in logits.0.iter().enumerate() {
            let mass = weight(score, max);
            if mass > 0.0 {
                last = Some(id as u32);
                cumulative += mass;
                if draw < cumulative {
                    return Ok(id as u32);
                }
            }
        }
        last.ok_or_else(|| invalid("no positive mass in the admitted vocabulary"))
    }

    fn token_id(token: &u32, _: &()) -> Result<u32, BackendFailure> {
        Ok(*token)
    }

    fn token_probability(logits: &HostScores, token: u32, _: &()) -> Result<f32, BackendFailure> {
        let (max, mass) = logits.normalization()?;
        let score = *logits
            .0
            .get(token as usize)
            .ok_or_else(|| invalid("token is outside the admitted vocabulary"))?;
        Ok((weight(score, max) / mass) as f32)
    }
}
