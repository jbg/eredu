use super::*;
use eredu_runtime::{SamplingConfigurationError, SpeculativeSampler};

/// MLX sampling and randomness state for backend-generic text generation.
pub struct MlxTextGenerationState {
    pub(super) sampling: MlxTextSamplingState,
    pub(super) capture: Option<eredu_runtime::capture::CaptureSession>,
}

/// Native sampling component copied by the shared ordinary snapshot driver.
/// Capture ownership remains separate; this alone is not a generation snapshot.
pub struct MlxTextSamplingState {
    pub(super) temperature: f32,
    pub(super) prng: Option<RandomState>,
    pub(super) sampler: MlxTextSampler,
    pub(super) next_prediction: u64,
}

#[derive(Clone)]
pub(crate) enum MlxTextSampler {
    Standard(GenerationSampler),
    MirostatV2(MirostatV2Sampler),
}

impl MlxTextSampler {
    pub(crate) fn from_config(
        config: TextGenerationConfig,
    ) -> Result<Self, SamplingConfigurationError> {
        let sampling = config.sampling();
        Ok(match config.strategy() {
            TextSamplingStrategy::Standard => {
                Self::Standard(GenerationSampler::from_resolved(sampling))
            }
            TextSamplingStrategy::MirostatV2 { tau, eta } => {
                Self::MirostatV2(MirostatV2Sampler::new(tau, eta)?.penalties(
                    sampling.repetition_penalty,
                    sampling.repeat_last_n,
                    sampling.frequency_penalty,
                    sampling.presence_penalty,
                ))
            }
        })
    }

    fn sample(
        &mut self,
        logits: &MlxTensor,
        temperature: f32,
        random: Option<&mut RandomState>,
        stream: &Stream,
    ) -> Result<MlxTensor, Exception> {
        match self {
            Self::Standard(sampler) => {
                Sampler::<MlxSamplingBackend>::sample(sampler, logits, temperature, random, stream)
            }
            Self::MirostatV2(sampler) => {
                Sampler::<MlxSamplingBackend>::sample(sampler, logits, temperature, random, stream)
            }
        }
    }
}

impl SpeculativeSampler<MlxSamplingBackend> for MlxTextSampler {
    fn control_snapshot_bytes(&self) -> Option<u64> {
        let history = match self {
            Self::Standard(sampler) => sampler.generated_tokens(),
            Self::MirostatV2(sampler) => sampler.generated_tokens(),
        };
        (history.len() as u64)
            .checked_mul(4)?
            .checked_add(std::mem::size_of::<Self>() as u64)
    }

    fn supports_exact_optimistic_promotion(&self) -> bool {
        match self {
            Self::Standard(sampler) => {
                SpeculativeSampler::<MlxSamplingBackend>::supports_exact_optimistic_promotion(
                    sampler,
                )
            }
            Self::MirostatV2(sampler) => {
                SpeculativeSampler::<MlxSamplingBackend>::supports_exact_optimistic_promotion(
                    sampler,
                )
            }
        }
    }

    fn process_logits(
        &mut self,
        logits: &MlxTensor,
        temperature: f32,
        history: &[u32],
        stream: &Stream,
    ) -> Result<MlxTensor, Exception> {
        match self {
            Self::Standard(sampler) => SpeculativeSampler::<MlxSamplingBackend>::process_logits(
                sampler,
                logits,
                temperature,
                history,
                stream,
            ),
            Self::MirostatV2(sampler) => SpeculativeSampler::<MlxSamplingBackend>::process_logits(
                sampler,
                logits,
                temperature,
                history,
                stream,
            ),
        }
    }

    fn commit_token(
        &mut self,
        processed_logits: &MlxTensor,
        token: u32,
        stream: &Stream,
    ) -> Result<(), Exception> {
        match self {
            Self::Standard(sampler) => SpeculativeSampler::<MlxSamplingBackend>::commit_token(
                sampler,
                processed_logits,
                token,
                stream,
            ),
            Self::MirostatV2(sampler) => SpeculativeSampler::<MlxSamplingBackend>::commit_token(
                sampler,
                processed_logits,
                token,
                stream,
            ),
        }
    }
}

struct FilteredTextSampler<'a> {
    sampler: &'a mut MlxTextSampler,
    filter: &'a TokenFilter,
}

impl Sampler<MlxSamplingBackend> for FilteredTextSampler<'_> {
    fn sample(
        &mut self,
        logits: &MlxTensor,
        temperature: f32,
        random: Option<&mut RandomState>,
        stream: &Stream,
    ) -> Result<MlxTensor, Exception> {
        let logits = MlxSamplingBackend::apply_token_filter(logits, self.filter, stream)?;
        self.sampler.sample(&logits, temperature, random, stream)
    }
}

pub(super) fn sample_text_submission(
    session: &MlxModelSession,
    submission: Submission<MlxModelOutput, MlxSessionCompletion>,
    filter: &TokenFilter,
    state: &mut MlxTextGenerationState,
    stream: Stream,
) -> Result<Submission<MlxTextToken, MlxTextCompletion>, Error> {
    let operation = super::model_session::ResourceOperation::begin(submission.completion.owner())?;
    let sampled = (|| {
        let MlxTextSamplingState {
            temperature,
            prng,
            sampler,
            ..
        } = &mut state.sampling;
        let mut sampler = FilteredTextSampler { sampler, filter };
        let token = if session.synchronizes_sampling() {
            session
                .sample_under_submission(
                    submission.output.logits(),
                    1,
                    &mut sampler,
                    *temperature,
                    prng.as_mut(),
                    false,
                )?
                .token
                .try_index_device((.., 0), &stream)?
        } else {
            let logits = submission.output.logits().ok_or_else(|| {
                Error::Parallel("local text generation requires model logits".into())
            })?;
            Sampler::<MlxSamplingBackend>::sample(
                &mut sampler,
                logits,
                *temperature,
                prng.as_mut(),
                &stream,
            )?
            .into_array()
        };
        MlxCompletion::submission(token)
    })();
    let (sampled, recovery) = operation.finish(sampled)?;
    state.sampling.next_prediction = state
        .sampling
        .next_prediction
        .checked_add(1)
        .ok_or_else(|| Error::ArchitectureModel("text prediction overflow".into()))?;
    Ok(Submission {
        output: MlxTextToken {
            value: sampled.output,
            stream,
            owner: std::rc::Rc::clone(submission.completion.owner()),
        },
        completion: MlxTextCompletion {
            model: submission.completion,
            token: sampled.completion,
            recovery: std::cell::RefCell::new(Some(recovery)),
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forced_decision_advances_one_rng_draw_and_updates_ordinary_sampler_state_once() {
        let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let logits = MlxTensor::from_array(Array::from_slice(&[1.0f32, 0.2, -0.5, 2.0], &[1, 4]));
        let filter = TokenFilter::allowed(vec![false, false, true, false]).unwrap();
        let key = |random: &RandomState| {
            random
                .as_array()
                .evaluated()
                .unwrap()
                .as_slice::<u32>()
                .to_vec()
        };
        for adaptive in [false, true] {
            for temperature in if adaptive { vec![0.8] } else { vec![0.0, 0.8] } {
                let mut random = RandomState::with_seed(12345).unwrap();
                let mut expected_random = RandomState::with_seed(12345).unwrap();
                if temperature != 0.0 {
                    expected_random
                        .next_key(&stream)
                        .unwrap()
                        .evaluated()
                        .unwrap();
                }
                let mut sampler = if adaptive {
                    let mut sampler = MirostatV2Sampler::new(5.0, 0.3).unwrap();
                    sampler.accept_token(1, 0.25).unwrap();
                    MlxTextSampler::MirostatV2(sampler)
                } else {
                    MlxTextSampler::Standard(
                        GenerationSampler::new()
                            .top_k(4)
                            .top_p(1.0)
                            .min_p(0.0)
                            .penalties(1.2, 64, 0.1, 0.1)
                            .with_generated_tokens([1]),
                    )
                };
                let previous_mu = match &sampler {
                    MlxTextSampler::MirostatV2(s) => Some(s.mu()),
                    _ => None,
                };
                let token = FilteredTextSampler {
                    sampler: &mut sampler,
                    filter: &filter,
                }
                .sample(&logits, temperature, Some(&mut random), &stream)
                .unwrap();
                assert_eq!(MlxSamplingBackend::token_id(&token, &stream).unwrap(), 2);
                assert_eq!(key(&random), key(&expected_random));
                match &sampler {
                    MlxTextSampler::Standard(s) => assert_eq!(s.generated_tokens(), [1, 2]),
                    MlxTextSampler::MirostatV2(s) => {
                        assert_eq!(s.generated_tokens(), [1, 2]);
                        assert!((s.mu() - (previous_mu.unwrap() + 0.3 * 5.0)).abs() < 1e-6);
                    }
                }
            }
        }
    }
}
