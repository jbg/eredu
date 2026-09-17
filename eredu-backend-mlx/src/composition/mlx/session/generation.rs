use super::*;
use crate::backend::managed_memory::NativeMemoryRetention;

mod ordinary_sampler;
pub(super) use ordinary_sampler::MlxOrdinarySampler;
use ordinary_sampler::PreparedOrdinarySampler;

#[cfg(test)]
mod token_retention_tests;

/// MLX sampling and randomness state for backend-generic text generation.
pub struct MlxTextGenerationState {
    pub(super) sampling: MlxTextSamplingState,
    pub(super) capture: Option<eredu_runtime::capture::CaptureSession>,
    // Mutually exclusive with the legacy collector; no raw ledger export.
    pub(super) funded_capture: Option<super::model_session::text_capture::InstalledCapture>,
    // Declared last: sampler host history, RNG and capture payload retire before
    // closing future funding. Native work scopes retain unresolved operations.
    pub(super) funding: Option<eredu_runtime::working_memory::WorkingMemoryFundingRun>,
}

/// Native sampling component copied by the shared ordinary snapshot driver.
/// Capture ownership remains separate; this alone is not a generation snapshot.
pub struct MlxTextSamplingState {
    pub(super) temperature: f32,
    pub(super) prng: Option<RandomState>,
    pub(super) sampler: MlxOrdinarySampler,
    pub(super) next_prediction: u64,
    /// Parameter version bound on first submission and preserved by snapshots.
    pub(super) parameter_epoch: Option<u64>,
    /// Charges shared by prepared, copied and restored sampler owners.
    pub(super) inference_retention: eredu_runtime::working_memory::InferenceRetention,
    /// Unquoted authority for independently prepared or copied sampling payloads.
    pub(super) memory_retention: NativeMemoryRetention,
    /// Exact cold operation contract, distinct from historical storage charges.
    pub(super) quote: Option<super::model_session::text_quote::TextExecutionQuoteOwner>,
}

pub(crate) type MlxTextSampler = eredu_runtime::ConfiguredTextSampler;

/// Recovery retains sampler charges independently of the caller's state handle.
pub(super) struct TextInferenceRetention {
    _inference: eredu_runtime::working_memory::InferenceRetention,
    _memory: NativeMemoryRetention,
}

impl TextInferenceRetention {
    pub(super) fn new(
        inference: eredu_runtime::working_memory::InferenceRetention,
        memory: NativeMemoryRetention,
    ) -> Self {
        Self {
            _inference: inference,
            _memory: memory,
        }
    }
}
impl super::recovery::Retention for TextInferenceRetention {
    fn observe(&self, _: super::recovery::Status) {}
}

struct FilteredTextSampler<'a> {
    sampler: Option<PreparedOrdinarySampler<'a>>,
    filter: &'a TokenFilter,
}

impl FilteredTextSampler<'_> {
    fn sample_original(
        &mut self,
        logits: &MlxTensor,
        temperature: f32,
        random: Option<&mut RandomState>,
        context: &crate::backend::runtime::generation::OriginalSamplingContext<'_>,
    ) -> Result<MlxTensor, Error> {
        use crate::backend::runtime::generation::OriginalSamplingBackend;
        use eredu_runtime::SamplingBackend;
        let sampler = self
            .sampler
            .take()
            .ok_or(Error::PredictionScopeUnavailable)?;
        let logits = OriginalSamplingBackend::apply_token_filter(logits, self.filter, context)?;
        sampler.sample_with::<OriginalSamplingBackend>(&logits, temperature, random, context)
    }
}
impl Sampler<MlxSamplingBackend> for FilteredTextSampler<'_> {
    fn sample(
        &mut self,
        logits: &MlxTensor,
        temperature: f32,
        random: Option<&mut RandomState>,
        stream: &Stream,
    ) -> Result<MlxTensor, Exception> {
        // Consume before filtering: a failed filter cannot reuse the permit.
        let sampler = self
            .sampler
            .take()
            .ok_or_else(|| Exception::custom("sampler step was already consumed"))?;
        let logits = MlxSamplingBackend::apply_token_filter(logits, self.filter, stream)?;
        sampler.sample(&logits, temperature, random, stream)
    }
}

pub(super) fn sample_text_submission(
    session: &MlxModelSession,
    mut submission: Submission<MlxModelOutput, MlxSessionCompletion>,
    filter: &TokenFilter,
    state: &mut MlxTextGenerationState,
    stream: Stream,
) -> Result<Submission<MlxTextToken, MlxTextCompletion>, Error> {
    let host = state
        .capture
        .as_ref()
        .and_then(|capture| capture.ordinary_error_custody())
        .cloned();
    submission.completion.retain_ordinary_capture(host.clone());
    submission
        .completion
        .owner()
        .retain_inference(&state.sampling.inference_retention);
    submission
        .completion
        .owner()
        .retain_memory(&state.sampling.memory_retention);

    let operation =
        super::model_session::ResourceOperation::begin_prediction(submission.completion.owner())
            .map_err(|error| error.at_text_admission())?;

    let original = operation.sampling_context(&stream, submission.output.logits(), session.original_sampling_selection()).map_err(|error| error.at_text_admission())?;

    let sampled = (|| {
        let MlxTextSamplingState {
            temperature,
            prng,
            sampler,
            ..
        } = &mut state.sampling;
        // Reject history growth before filter construction or synchronized
        // native sampling; the host permit is separate from submission authority.
        let mut sampler = FilteredTextSampler {
            sampler: Some(sampler.prepare_sample()?),
            filter,
        };

        let token = if let Some(context) = &original {
            if session.synchronizes_sampling() {
                context.sample_synchronized(prng.as_mut(), |random| {
                    sampler.sample_original(submission.output.logits().ok_or(Error::PredictionScopeUnavailable)?,
                        *temperature,random,context)
                })?
            } else {
                sampler.sample_original(submission.output.logits().ok_or(Error::PredictionScopeUnavailable)?,
                    *temperature,prng.as_mut(),context)?.into_array()
            }
        } else if session.synchronizes_sampling() {
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

        if let Some(context) = original {
            return context.finish(token).map_err(|error| error.at_text_admission());
        }
        let random = state
            .sampling
            .prng
            .as_ref()
            .map(|random| random.as_array().clone());
        // Complete the next RNG root alongside this token. It belongs to the
        // sampling quote and must not leave a growing lazy sibling graph.
        MlxCompletion::submission_retaining_with_scope(
            token,
            random,
            &stream,
            submission.completion.owner().take_sampling_event_scope()?,
        )
    })();

    let (sampled, recovery) = operation.finish(sampled.map_err(|error| error.at_text_admission()))
        .map_err(|error| error.at_text_admission())?;
    submission
        .completion
        .owner()
        .retain_funded_array(&sampled.output);
    if let Some(random) = &state.sampling.prng {
        submission
            .completion
            .owner()
            .retain_funded_array(random.as_array());
    }

    state.sampling.next_prediction = state
        .sampling
        .next_prediction
        .checked_add(1)
        .ok_or_else(|| Error::ArchitectureModel("text prediction overflow".into()))?;

    Ok(Submission {
        output: MlxTextToken::new_with_sampling_source(
            sampled.output,
            stream,
            submission.completion.owner().clone(),
            submission.completion.owner().take_token_scalar_scope()?,
            host.clone(),
            sampled.completion.original_sampling_source(),
        ),
        completion: MlxTextCompletion {
            model: submission.completion,
            token: sampled.completion,
            recovery: std::cell::RefCell::new(Some(recovery)),
            observation: super::output_completion::Observation::with_ordinary_capture(host),
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
                    sampler: Some(PreparedOrdinarySampler::Unquoted(&mut sampler)),
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

    #[test]
    fn failed_filter_consumes_sampling_attempt_before_retry_can_touch_rng_or_history() {
        let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let logits = MlxTensor::from_array(Array::from_slice(&[1.0f32, 0.2, -0.5, 2.0], &[1, 4]));
        // Short masks are deliberately padded. This allows only an ID beyond
        // the score row, leaving no valid token after width normalization.
        let invalid_filter = TokenFilter::allowed(vec![false, false, false, false, true]).unwrap();
        let valid_filter = TokenFilter::allowed(vec![false, false, true, false]).unwrap();
        let key = |random: &RandomState| {
            random
                .as_array()
                .evaluated()
                .unwrap()
                .as_slice::<u32>()
                .to_vec()
        };
        let mut random = RandomState::with_seed(12345).unwrap();
        let mut expected_random = RandomState::with_seed(12345).unwrap();
        let initial_key = key(&random);
        let mut sampler = MlxTextSampler::Standard(
            GenerationSampler::new()
                .top_k(4)
                .top_p(1.0)
                .min_p(0.0)
                .penalties(1.2, 64, 0.1, 0.1)
                .with_generated_tokens([1]),
        );
        let initial_sampler = format!("{sampler:?}");
        let mut expected_sampler = sampler.clone();
        let mut prepared = FilteredTextSampler {
            sampler: Some(PreparedOrdinarySampler::Unquoted(&mut sampler)),
            filter: &invalid_filter,
        };
        assert!(
            prepared
                .sample(&logits, 0.8, Some(&mut random), &stream)
                .is_err()
        );
        assert!(prepared.sampler.is_none());
        assert_eq!(key(&random), initial_key);

        prepared.filter = &valid_filter;
        let retry = prepared
            .sample(&logits, 0.8, Some(&mut random), &stream)
            .err()
            .unwrap();
        assert!(
            retry
                .to_string()
                .contains("sampler step was already consumed")
        );
        assert_eq!(key(&random), initial_key);
        drop(prepared);
        assert_eq!(format!("{sampler:?}"), initial_sampler);

        let actual = FilteredTextSampler {
            sampler: Some(PreparedOrdinarySampler::Unquoted(&mut sampler)),
            filter: &valid_filter,
        }
        .sample(&logits, 0.8, Some(&mut random), &stream)
        .unwrap();
        let expected = FilteredTextSampler {
            sampler: Some(PreparedOrdinarySampler::Unquoted(&mut expected_sampler)),
            filter: &valid_filter,
        }
        .sample(&logits, 0.8, Some(&mut expected_random), &stream)
        .unwrap();
        assert_eq!(MlxSamplingBackend::token_id(&actual, &stream).unwrap(), 2);
        assert_eq!(MlxSamplingBackend::token_id(&expected, &stream).unwrap(), 2);
        assert_eq!(key(&random), key(&expected_random));
        assert_ne!(key(&random), initial_key);
        assert_eq!(format!("{sampler:?}"), format!("{expected_sampler:?}"));
        let MlxTextSampler::Standard(sampler) = sampler else {
            unreachable!("the fixture uses the ordinary standard sampler");
        };
        assert_eq!(sampler.generated_tokens(), [1, 2]);
    }
}
