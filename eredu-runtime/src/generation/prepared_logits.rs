//! Exact immutable policy projection for the existing logit-processing worker.
use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Penalties {
    repeat: u32,
    last: i32,
    frequency: u32,
    presence: u32,
}
impl From<PenaltyConfig> for Penalties {
    fn from(value: PenaltyConfig) -> Self {
        Self {
            repeat: value.repeat_penalty.to_bits(),
            last: value.repeat_last_n,
            frequency: value.frequency_penalty.to_bits(),
            presence: value.presence_penalty.to_bits(),
        }
    }
}
impl Penalties {
    fn value(self) -> PenaltyConfig {
        PenaltyConfig {
            repeat_penalty: f32::from_bits(self.repeat),
            repeat_last_n: self.last,
            frequency_penalty: f32::from_bits(self.frequency),
            presence_penalty: f32::from_bits(self.presence),
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Standard {
    penalties: Penalties,
    top_k: i32,
    top_p: u32,
    min_p: u32,
}
impl Standard {
    pub(super) fn from_source(source: &GenerationSampler) -> Self {
        Self {
            penalties: source.penalty_config().into(),
            top_k: source.top_k,
            top_p: source.top_p.to_bits(),
            min_p: source.min_p.to_bits(),
        }
    }
    pub(super) fn process<B: SamplingBackend>(
        self,
        logits: &B::Logits,
        context: &B::Context,
        penalties: impl FnOnce(&B::Logits, PenaltyConfig, &B::Context) -> Result<B::Logits, B::Error>,
    ) -> Result<B::Logits, B::Error> {
        let logits = penalties(logits, self.penalties.value(), context)?;
        let logits = B::apply_top_k(logits, self.top_k, context)?;
        let logits = B::apply_top_p(logits, f32::from_bits(self.top_p), context)?;
        B::apply_min_p(logits, f32::from_bits(self.min_p), context)
    }
}
pub(super) fn default_process<B: SamplingBackend>(
    logits: &B::Logits,
    temperature: f32,
    context: &B::Context,
) -> Result<B::Logits, B::Error> {
    if temperature == 0.0 {
        Ok(logits.clone())
    } else {
        B::scale_temperature(logits, temperature, context)
    }
}
pub(super) fn finish_standard<B: SamplingBackend>(
    logits: B::Logits,
    temperature: f32,
    context: &B::Context,
) -> Result<B::Logits, B::Error> {
    if temperature == 0.0 {
        Ok(logits)
    } else {
        B::scale_temperature(&logits, temperature, context)
    }
}
pub(super) fn adaptive_process<B: SamplingBackend>(
    logits: &B::Logits,
    penalties: PenaltyConfig,
    temperature: f32,
    mu: f32,
    context: &B::Context,
    cutoff: impl FnOnce(&B::Logits, PenaltyConfig, f32, f32, &B::Context) -> Result<B::Logits, B::Error>,
) -> Result<B::Logits, B::Error> {
    if !temperature.is_finite() || temperature <= 0.0 {
        return Err(B::error(
            "Mirostat V2 requires a finite temperature greater than zero".into(),
        ));
    }
    cutoff(logits, penalties, temperature, mu, context)
}

#[derive(Debug)]
enum Source<'a> {
    Default(&'a DefaultSampler),
    Standard(&'a GenerationSampler),
    Adaptive(&'a MirostatV2Sampler),
}
/// Borrow of a concrete existing policy, with no callback or state clone. Only
/// the known ordinary implementations construct this view.
#[derive(Debug)]
pub struct PreparedLogitPolicy<'a> {
    source: Source<'a>,
}
impl<'a> PreparedLogitPolicy<'a> {
    pub(super) fn default(source: &'a DefaultSampler) -> Self {
        Self {
            source: Source::Default(source),
        }
    }
    pub(super) fn standard(source: &'a GenerationSampler) -> Self {
        Self {
            source: Source::Standard(source),
        }
    }
    pub(super) fn adaptive(source: &'a MirostatV2Sampler) -> Self {
        Self {
            source: Source::Adaptive(source),
        }
    }
    /// Freezes only actual scalar controls and this call's history extent. No
    /// history payload, RNG, callback, native value or authority is created.
    pub fn bind(
        self,
        temperature: f32,
        history_len: usize,
    ) -> Result<SpeculativeLogitProgram, PreparedLogitPolicyError> {
        let policy = match self.source {
            Source::Default(&DefaultSampler) => Policy::Default,
            Source::Standard(source) => Policy::Standard(Standard::from_source(source)),
            Source::Adaptive(source) => Policy::Adaptive {
                penalties: source.penalties.penalty_config().into(),
                mu: source.mu.to_bits(),
            },
        };
        // Workspace's selected scaling producer requires finite positive scale;
        // exact zero is the existing standard/default identity branch.
        if !temperature.is_finite()
            || temperature < 0.0
            || (matches!(policy, Policy::Adaptive { .. }) && temperature == 0.0)
        {
            return Err(PreparedLogitPolicyError::Temperature);
        }
        Ok(SpeculativeLogitProgram {
            policy,
            temperature: temperature.to_bits(),
            history_len,
        })
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Policy {
    Default,
    Standard(Standard),
    Adaptive { penalties: Penalties, mu: u32 },
}
/// Closed scalar descriptor from an actual immutable sampler-policy view.
/// IEEE bit fields retain exact policy values without heap allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpeculativeLogitProgram {
    policy: Policy,
    temperature: u32,
    history_len: usize,
}
/// Fixed preconstruction validation, independent of a backend diagnostic allocator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum PreparedLogitPolicyError {
    /// Arbitrary sampler callbacks have no declared deterministic projection.
    #[error("sampler has no prepared deterministic logit policy")]
    Unknown,
    /// The closed scaling/temperature producer requires its existing finite domain.
    #[error(
        "prepared logit policy requires a finite nonnegative temperature, positive for Mirostat"
    )]
    Temperature,
    /// Source geometry is not the exact selected one-sequence adaptive profile.
    #[error("prepared Mirostat logit policy requires exactly one sequence")]
    Shape,
    /// A different history slice extent cannot reuse an accepted descriptor.
    #[error("prepared logit policy history differs from its accepted extent")]
    History,
}
/// Preserves the underlying backend cause; no diagnostic is allocated by this shell.
#[derive(Debug, thiserror::Error)]
pub enum LogitProgramError<E> {
    /// Fixed source/extent refusal.
    #[error(transparent)]
    Policy(#[from] PreparedLogitPolicyError),
    /// Original primitive failure.
    #[error(transparent)]
    Backend(E),
}
impl SpeculativeLogitProgram {
    /// Exact history extent consumed by the shared penalty worker.
    pub fn history_len(self) -> usize {
        self.history_len
    }
    /// No numerical data is read during source validation.
    pub fn validate_layout(self, shape: &[i32]) -> Result<(), PreparedLogitPolicyError> {
        if matches!(self.policy, Policy::Adaptive { .. }) {
            let elements = shape
                .iter()
                .try_fold(1usize, |n, &d| n.checked_mul(usize::try_from(d).ok()?));
            if elements.is_none()
                || shape
                    .last()
                    .and_then(|&d| usize::try_from(d).ok())
                    .filter(|&n| n > 0)
                    != elements
            {
                return Err(PreparedLogitPolicyError::Shape);
            }
        }
        Ok(())
    }
    /// Runs the same ordinary immutable worker. Completion, source provenance
    /// and admission are supplied by the enclosing numerical phase.
    pub fn process<B: SamplingBackend>(
        self,
        logits: &B::Logits,
        history: &[u32],
        context: &B::Context,
    ) -> Result<B::Logits, LogitProgramError<B::Error>> {
        if history.len() != self.history_len {
            return Err(PreparedLogitPolicyError::History.into());
        }
        let temperature = f32::from_bits(self.temperature);
        let result = match self.policy {
            Policy::Default => default_process::<B>(logits, temperature, context),
            Policy::Standard(policy) => policy
                .process::<B>(logits, context, |logits, penalties, context| {
                    B::apply_penalties(logits, history, penalties, context)
                })
                .and_then(|logits| finish_standard::<B>(logits, temperature, context)),
            Policy::Adaptive { penalties, mu } => adaptive_process::<B>(
                logits,
                penalties.value(),
                temperature,
                f32::from_bits(mu),
                context,
                |logits, penalties, temperature, mu, context| {
                    B::apply_mirostat(logits, history, penalties, temperature, mu, context)
                },
            ),
        };
        result.map_err(LogitProgramError::Backend)
    }
}
