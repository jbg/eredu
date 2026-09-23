use super::{DefaultSampler, GenerationSampler, MirostatV2Sampler};
use eredu_core::{TextGenerationConfig, TextSamplingStrategy};
use eredu_nn::{
    mechanism_memory::{MechanismInvocation, SamplingMode},
    Error, TensorElementType,
};

/// Describes a resolved generation policy's next sampling invocation.
///
/// `history` counts the history supplied to sampling, including speculative
/// proposals when applicable. It does not advance the sampler or reserve storage.
/// Token-filter controllers and custom sampler extensions require their own
/// implementation contracts; this describes only the selected standard policy.
pub fn sampling_invocation(
    config: &TextGenerationConfig,
    rows: u64,
    vocabulary: u64,
    element: TensorElementType,
    history: u64,
) -> Result<MechanismInvocation, Error> {
    let sampling = config.sampling();
    let sampler = GenerationSampler::from_resolved(sampling);
    let mode = match config.strategy() {
        TextSamplingStrategy::Standard => mode(sampling.temperature)?,
        TextSamplingStrategy::MirostatV2 { .. } => SamplingMode::MirostatV2,
    };
    describe(&sampler, rows, vocabulary, element, history, mode)
}

impl DefaultSampler {
    /// Describes the unfiltered sampling mechanism without drawing a token.
    pub fn sampling_invocation(
        &self,
        rows: u64,
        vocabulary: u64,
        element: TensorElementType,
        temperature: f32,
    ) -> Result<MechanismInvocation, Error> {
        checked(MechanismInvocation::Sampling {
            rows,
            vocabulary,
            history: 0,
            element,
            mode: mode(temperature)?,
            top_k: 0,
            top_p: false,
            min_p: false,
            penalties: false,
        })
    }
}

impl GenerationSampler {
    /// Describes the current policy and retained history, without changing either.
    /// Filtering still executes when temperature selects greedy sampling.
    pub fn sampling_invocation(
        &self,
        rows: u64,
        vocabulary: u64,
        element: TensorElementType,
        temperature: f32,
    ) -> Result<MechanismInvocation, Error> {
        describe(
            self,
            rows,
            vocabulary,
            element,
            self.generated_tokens.len() as u64,
            mode(temperature)?,
        )
    }
}

impl MirostatV2Sampler {
    /// Describes adaptive sampling without modifying surprise state or history.
    /// Standard top-k/top-p/min-p filters are not part of this execution path.
    pub fn sampling_invocation(
        &self,
        rows: u64,
        vocabulary: u64,
        element: TensorElementType,
    ) -> Result<MechanismInvocation, Error> {
        describe(
            &self.penalties,
            rows,
            vocabulary,
            element,
            self.generated_tokens().len() as u64,
            SamplingMode::MirostatV2,
        )
    }
}

fn mode(temperature: f32) -> Result<SamplingMode, Error> {
    if !temperature.is_finite() || temperature < 0.0 {
        return Err(Error::backend(
            "sampling temperature must be finite and nonnegative",
        ));
    }
    Ok(if temperature == 0.0 {
        SamplingMode::Greedy
    } else {
        SamplingMode::Categorical
    })
}

fn describe(
    sampler: &GenerationSampler,
    rows: u64,
    vocabulary: u64,
    element: TensorElementType,
    history: u64,
    mode: SamplingMode,
) -> Result<MechanismInvocation, Error> {
    let standard = mode != SamplingMode::MirostatV2;
    checked(MechanismInvocation::Sampling {
        rows,
        vocabulary,
        history,
        element,
        mode,
        top_k: if standard && sampler.top_k > 0 && (sampler.top_k as u64) < vocabulary {
            sampler.top_k as u64
        } else {
            0
        },
        top_p: standard && sampler.top_p < 1.0,
        min_p: standard && sampler.min_p > 0.0,
        penalties: history > 0 && !sampler.penalty_config().is_identity(),
    })
}

fn checked(invocation: MechanismInvocation) -> Result<MechanismInvocation, Error> {
    invocation.logical_values()?;
    Ok(invocation)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_keeps_filters_and_penalties_without_mutating_history() {
        let mut sampler = GenerationSampler {
            repeat_penalty: 1.1,
            repeat_last_n: 0,
            ..Default::default()
        };
        sampler.accept_token(9);
        let invocation = sampler
            .sampling_invocation(1, 128, TensorElementType::F32, 0.0)
            .unwrap();
        assert!(matches!(
            invocation,
            MechanismInvocation::Sampling {
                mode: SamplingMode::Greedy,
                history: 1,
                top_k: 40,
                top_p: true,
                min_p: true,
                penalties: true,
                ..
            }
        ));
        assert_eq!(sampler.generated_tokens(), &[9]);
    }

    #[test]
    fn adaptive_ignores_standard_filters_and_preserves_state() {
        let mut sampler = MirostatV2Sampler::default();
        sampler.penalties = GenerationSampler {
            repeat_penalty: 1.2,
            ..Default::default()
        };
        sampler.accept_token(4, 0.25).unwrap();
        let before = sampler.mu();
        let invocation = sampler
            .sampling_invocation(1, 128, TensorElementType::Bf16)
            .unwrap();
        assert!(matches!(
            invocation,
            MechanismInvocation::Sampling {
                mode: SamplingMode::MirostatV2,
                history: 1,
                top_k: 0,
                top_p: false,
                min_p: false,
                penalties: true,
                ..
            }
        ));
        assert_eq!(sampler.mu(), before);
        assert_eq!(sampler.generated_tokens(), &[4]);
    }

    #[test]
    fn disabled_filters_and_invalid_geometry_are_explicit() {
        let sampler = GenerationSampler {
            top_k: 999,
            top_p: 1.0,
            min_p: 0.0,
            ..Default::default()
        };
        assert!(matches!(
            sampler
                .sampling_invocation(1, 128, TensorElementType::F32, 1.0)
                .unwrap(),
            MechanismInvocation::Sampling {
                top_k: 0,
                top_p: false,
                min_p: false,
                penalties: false,
                ..
            }
        ));
        assert!(DefaultSampler
            .sampling_invocation(1, 128, TensorElementType::F32, f32::NAN)
            .is_err());
        assert!(DefaultSampler
            .sampling_invocation(u64::MAX, 128, TensorElementType::F32, 1.0)
            .is_err());
    }

    #[test]
    fn cold_config_and_live_policy_describe_the_same_mechanism() {
        let mut resolved =
            eredu_core::generation::resolve_generation_config(None, Default::default()).unwrap();
        resolved.temperature = 0.7;
        resolved.do_sample = true;
        resolved.repetition_penalty = 1.1;
        let config = TextGenerationConfig::new(resolved);
        let mut live = GenerationSampler::from_resolved(resolved);
        live.accept_token(3);
        live.accept_token(4);
        assert_eq!(
            sampling_invocation(&config, 1, 128, TensorElementType::F32, 2).unwrap(),
            live.sampling_invocation(1, 128, TensorElementType::F32, resolved.temperature)
                .unwrap()
        );
        let config = config.with_mirostat_v2(5.0, 0.1).unwrap();
        let mut adaptive = MirostatV2Sampler::default();
        adaptive.penalties = live;
        assert_eq!(
            sampling_invocation(&config, 1, 128, TensorElementType::F32, 2).unwrap(),
            adaptive
                .sampling_invocation(1, 128, TensorElementType::F32)
                .unwrap()
        );
    }
}
