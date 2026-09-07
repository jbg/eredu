use super::*;

/// MLX sampling and randomness state for backend-generic text generation.
pub struct MlxTextGenerationState {
    pub(super) temperature: f32,
    pub(super) prng: Option<RandomState>,
    pub(super) sampler: MlxTextSampler,
    pub(super) capture: Option<eredu_runtime::capture::CaptureSession>,
    pub(super) prediction_index: u64,
}

pub(super) enum MlxTextSampler {
    Standard(GenerationSampler),
    MirostatV2(MirostatV2Sampler),
}

impl MlxTextSampler {
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
        let MlxTextGenerationState {
            temperature,
            prng,
            sampler,
            ..
        } = state;
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
