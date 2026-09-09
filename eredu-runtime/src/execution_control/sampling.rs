//! Prospective sampling policy validated before native RNG preparation.
use eredu_core::{
    ModelRuntime, TextContinuationBoundary, TextGenerationBackend, TokenFilterController,
};
use serde::{Deserialize, Serialize};

/// Supported changes to future sampling only. Omitting `reseed` retains the exact
/// inherited RNG stream. Sampler strategy, adaptive counters, penalties and token
/// history remain unchanged; incompatible strategy transitions are not exposed.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct SamplingOverride {
    /// New temperature; zero selects greedy standard sampling.
    pub temperature: Option<f32>,
    /// Explicit new native RNG seed. Does not reset adaptive or penalty history.
    pub reseed: Option<u64>,
}

/// Native facts consumed by shared validation; contains no mutable native handle.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SamplingStateFacts {
    /// Current effective temperature.
    pub temperature: f32,
    /// The retained sampler requires strictly positive temperature (Mirostat).
    pub requires_positive_temperature: bool,
    /// An exact resumable RNG stream exists, including while temporarily greedy.
    pub has_rng: bool,
}

/// Validated native action. Only the shared policy constructs this value.
#[derive(Debug, Clone, Copy)]
pub struct ValidatedSamplingOverride {
    temperature: f32,
    reseed: Option<u64>,
}
impl ValidatedSamplingOverride {
    /// Effective future temperature.
    pub fn temperature(self) -> f32 {
        self.temperature
    }
    /// Explicit new seed, absent when the existing RNG must remain unchanged.
    pub fn reseed(self) -> Option<u64> {
        self.reseed
    }
}

/// Minimal native mechanism; ordinary sampling and snapshot state keep their
/// existing owners. Errors must preserve the previous logical sampling state.
pub trait TextSamplingControlBackend: TextGenerationBackend {
    /// Side-effect-free facts for the installed sampling state.
    fn sampling_control_facts(state: &Self::TextGenerationState) -> SamplingStateFacts;
    /// Prepares any replacement key and proves completion through the existing
    /// recovery owner before installing temperature/key together. Consumes no RNG
    /// draw, submits no model prediction and leaves all sampler history intact.
    fn install_sampling_override(
        runtime: &mut ModelRuntime<Self>,
        state: &mut Self::TextGenerationState,
        request: ValidatedSamplingOverride,
    ) -> Result<(), Self::Error>;
}

/// Invalid policy is rejected before any native operation or state change.
#[derive(Debug, thiserror::Error)]
pub enum SamplingOverrideError<E: std::error::Error + 'static> {
    /// Invalid or incompatible temperature/RNG policy.
    #[error("invalid sampling override: {0}")]
    Invalid(&'static str),
    /// Native replacement preparation failed, preserving the prior logical state.
    #[error("native sampling override failed: {0}")]
    Backend(#[source] E),
}

/// Shared prospective-temperature and RNG admission for ordinary and speculative runs.
pub fn validate_sampling_override<E: std::error::Error + 'static>(
    facts: SamplingStateFacts,
    request: SamplingOverride,
) -> Result<ValidatedSamplingOverride, SamplingOverrideError<E>> {
    let temperature = request.temperature.unwrap_or(facts.temperature);
    if !temperature.is_finite() || temperature < 0.0 {
        return Err(SamplingOverrideError::Invalid(
            "temperature must be finite and nonnegative",
        ));
    }
    if facts.requires_positive_temperature && temperature == 0.0 {
        return Err(SamplingOverrideError::Invalid(
            "adaptive sampling requires positive temperature",
        ));
    }
    if temperature > 0.0 && !facts.has_rng && request.reseed.is_none() {
        return Err(SamplingOverrideError::Invalid(
            "no inherited RNG exists; an explicit seed is required",
        ));
    }
    Ok(ValidatedSamplingOverride {
        temperature,
        reseed: request.reseed,
    })
}

/// Validates and applies one future sampling change at a proven native and record
/// boundary. Stochastic-to-greedy retains RNG; switching back resumes that stream.
/// A run created without an RNG needs an explicit seed to become stochastic.
pub fn apply_sampling_override<B: TextSamplingControlBackend, C: TokenFilterController>(
    boundary: &mut TextContinuationBoundary<'_, '_, B, C>,
    request: SamplingOverride,
) -> Result<SamplingStateFacts, SamplingOverrideError<B::Error>> {
    let (runtime, state, _) = boundary.mechanism_parts();
    apply_prepared_sampling_override(runtime, state, request)
}

/// Validates a prospective change on detached, independently prepared generation
/// state. Used during branch construction under its copy reservation, while the
/// loaded runtime is quiescent. Native installation must preserve the installed
/// model state and cooperate with its existing submission/completion owner.
pub fn apply_prepared_sampling_override<B: TextSamplingControlBackend>(
    runtime: &mut ModelRuntime<B>,
    state: &mut B::TextGenerationState,
    request: SamplingOverride,
) -> Result<SamplingStateFacts, SamplingOverrideError<B::Error>> {
    if let eredu_core::execution_control::ControlSupport::Unsupported { .. } =
        B::text_sampling_control_support(runtime)
    {
        return Err(SamplingOverrideError::Invalid(
            "loaded execution does not support sampling overrides",
        ));
    }
    let action = validate_sampling_override(B::sampling_control_facts(state), request)?;
    B::install_sampling_override(runtime, state, action).map_err(SamplingOverrideError::Backend)?;
    Ok(B::sampling_control_facts(state))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn check(
        facts: SamplingStateFacts,
        request: SamplingOverride,
    ) -> Result<ValidatedSamplingOverride, SamplingOverrideError<std::io::Error>> {
        validate_sampling_override(facts, request)
    }
    #[test]
    fn temperature_changes_preserve_rng_and_adaptation_unless_reseeding_is_explicit() {
        let initial = SamplingStateFacts {
            temperature: 0.0,
            requires_positive_temperature: false,
            has_rng: false,
        };
        let stochastic = SamplingOverride {
            temperature: Some(0.7),
            reseed: None,
        };
        assert!(check(initial, stochastic).is_err());
        let seeded = check(
            initial,
            SamplingOverride {
                reseed: Some(42),
                ..stochastic
            },
        )
        .unwrap();
        assert_eq!(seeded.reseed(), Some(42));
        assert_eq!(seeded.temperature(), 0.7);
        let retained = SamplingStateFacts {
            has_rng: true,
            ..initial
        };
        assert_eq!(check(retained, stochastic).unwrap().reseed(), None);
        for invalid in [f32::NAN, f32::INFINITY, -1.0] {
            assert!(check(
                retained,
                SamplingOverride {
                    temperature: Some(invalid),
                    reseed: None
                }
            )
            .is_err());
        }
        let adaptive = SamplingStateFacts {
            temperature: 0.8,
            requires_positive_temperature: true,
            has_rng: true,
        };
        assert!(check(
            adaptive,
            SamplingOverride {
                temperature: Some(0.0),
                reseed: None
            }
        )
        .is_err());
        assert_eq!(
            check(adaptive, SamplingOverride::default())
                .unwrap()
                .temperature(),
            0.8
        );
    }
}
