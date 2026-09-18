//! Prospective sampling policy validated before native RNG preparation.
use eredu_core::{
    ModelRuntime, TextContinuationBoundary, TextGenerationBackend, TokenFilterController,
};
pub use eredu_core::{
    SamplingOverride, SamplingOverrideError, SamplingStateFacts, TextSamplingControlBackend,
};

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
    boundary.sampling_boundary().apply(request)
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
    B::apply_sampling_override(runtime, state, None, request)
}

/// Shared policy validation before a backend prepares any native replacement.
/// This grants no storage or submission authority; the concrete installer must
/// admit its exact revised program before publishing the new facts.
pub fn prepare_sampling_override<B: TextSamplingControlBackend>(
    runtime: &ModelRuntime<B>,
    state: &B::TextGenerationState,
    request: SamplingOverride,
) -> Result<ValidatedSamplingOverride, SamplingOverrideError<B::Error>> {
    if let eredu_core::execution_control::ControlSupport::Unsupported { .. } =
        B::text_sampling_control_support(runtime)
    {
        return Err(SamplingOverrideError::Invalid(
            "loaded execution does not support sampling overrides",
        ));
    }
    validate_sampling_override(B::sampling_control_facts(state), request)
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
            assert!(
                check(
                    retained,
                    SamplingOverride {
                        temperature: Some(invalid),
                        reseed: None
                    }
                )
                .is_err()
            );
        }
        let adaptive = SamplingStateFacts {
            temperature: 0.8,
            requires_positive_temperature: true,
            has_rng: true,
        };
        assert!(
            check(
                adaptive,
                SamplingOverride {
                    temperature: Some(0.0),
                    reseed: None
                }
            )
            .is_err()
        );
        assert_eq!(
            check(adaptive, SamplingOverride::default())
                .unwrap()
                .temperature(),
            0.8
        );
    }
}
