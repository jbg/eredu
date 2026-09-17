//! Closed projection of known greedy choice and unchanged commit behavior.
use super::{DefaultSampler, GenerationSampler, MirostatV2Sampler, SamplingBackend};

#[derive(Debug)]
enum Source<'a> {
    Default(&'a DefaultSampler),
    Standard(&'a GenerationSampler),
}
/// Actual known policy borrow. This independently qualifies greedy selection
/// and its inherited no-mutation speculative commit, never an arbitrary
/// `sample_processed` or `commit_token` override.
#[derive(Debug)]
pub struct PreparedGreedyPolicy<'a> {
    source: Source<'a>,
}
impl<'a> PreparedGreedyPolicy<'a> {
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
    /// Freezes the actual selected greedy worker. Nonzero or NaN temperature
    /// requires its independent explicit-key producer and cannot enter here.
    pub fn greedy(self, temperature: f32) -> Result<SpeculativeGreedyProgram, PreparedGreedyError> {
        if temperature != 0.0 {
            return Err(PreparedGreedyError::Temperature);
        }
        self.commit_without_mutation();
        Ok(SpeculativeGreedyProgram(()))
    }
    /// Same inherited no-op commit used by these actual ordinary policies.
    /// The caller still authenticates source/request and token domain.
    pub fn commit_without_mutation(self) {
        match self.source {
            Source::Default(&DefaultSampler) => (),
            Source::Standard(source) => {
                let _ = source;
            }
        }
    }
}
/// Closed exact greedy operation issued only by the known policy projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpeculativeGreedyProgram(());
impl SpeculativeGreedyProgram {
    /// Constructs the same argmax through the ordinary sampling backend.
    /// Completion and U32 observation remain the caller's admitted mechanism.
    pub fn construct<B: SamplingBackend>(
        self,
        logits: &B::Logits,
        context: &B::Context,
    ) -> Result<B::Token, B::Error> {
        B::sample_processed(logits, 0.0, None, context)
    }
}
/// Fixed qualification refusal, with no diagnostic allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum PreparedGreedyError {
    /// Exact stochastic branch requires its positive finite temperature domain.
    #[error("prepared categorical choice requires positive finite temperature")]
    StochasticTemperature,
    /// No exact projection of the actual choice and commit callbacks.
    #[error("sampler has no prepared greedy and unchanged-commit policy")]
    Unknown,
    /// Stochastic choice must consume its actual explicit-key producer.
    #[error("prepared greedy choice requires zero temperature")]
    Temperature,
}


/// Independent known-policy loan of the existing processed-logit categorical
/// worker. It does not certify the policy's commit callback or create a key.
#[derive(Debug)]
pub struct PreparedCategoricalPolicy<'a>(CategoricalSource<'a>);
#[derive(Debug)]
enum CategoricalSource<'a> {
    Default(&'a DefaultSampler),
    Standard(&'a GenerationSampler),
    Adaptive(&'a MirostatV2Sampler),
}
impl<'a> PreparedCategoricalPolicy<'a> {
    pub(super) fn default(source: &'a DefaultSampler) -> Self { Self(CategoricalSource::Default(source)) }
    pub(super) fn standard(source: &'a GenerationSampler) -> Self { Self(CategoricalSource::Standard(source)) }
    pub(super) fn adaptive(source: &'a MirostatV2Sampler) -> Self { Self(CategoricalSource::Adaptive(source)) }
    /// Selects the actual nonzero-temperature categorical branch. The explicit
    /// key and all native allocation authority remain separate inputs.
    pub fn bind(self, temperature: f32) -> Result<SpeculativeCategoricalProgram, PreparedGreedyError> {
        if !temperature.is_finite() || temperature <= 0.0 { return Err(PreparedGreedyError::StochasticTemperature); }
        match self.0 {
            CategoricalSource::Default(&DefaultSampler) => (),
            CategoricalSource::Standard(source) => { let _ = source; },
            CategoricalSource::Adaptive(source) => { let _ = source; },
        }
        Ok(SpeculativeCategoricalProgram(temperature.to_bits()))
    }
}
/// Closed ordinary categorical branch; values are already processed/scaled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpeculativeCategoricalProgram(u32);
impl SpeculativeCategoricalProgram {
    /// The same backend split-and-categorical worker used by ordinary policy.
    pub fn construct<B: SamplingBackend>(self, logits: &B::Logits,
        random: &mut B::RandomState, context: &B::Context) -> Result<B::Token, B::Error> {
        B::sample_processed(logits, f32::from_bits(self.0), Some(random), context)
    }
}
