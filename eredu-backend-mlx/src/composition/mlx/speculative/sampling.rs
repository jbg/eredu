use super::*;

/// MLX implementation of opaque speculative sampling operations.
#[derive(Clone)]
pub struct MlxSpeculativeSampling<S> {
    inner: S,
    capture: Option<std::rc::Rc<std::cell::RefCell<LogitCapture>>>,
}

impl<S> MlxSpeculativeSampling<S> {
    /// Wraps one runtime sampling policy for MLX execution.
    pub const fn new(inner: S) -> Self {
        Self {
            inner,
            capture: None,
        }
    }

    /// Returns the sampling policy after a test generation completes.
    #[cfg(test)]
    pub fn into_inner(self) -> S {
        self.inner
    }

    #[cfg(test)]
    pub const fn inner(&self) -> &S {
        &self.inner
    }
}

impl<S> SpeculativeSampling for MlxSpeculativeSampling<S>
where
    S: SpeculativeSampler<MlxSamplingBackend> + Clone,
{
    type Logits = Array;
    type Distribution = Array;
    type Seed = Array;
    type RandomState = RandomState;
    type DraftRandomness = Array;
    type RandomnessRoot = RandomState;
    type Context<'a>
        = SpeculativeExecutionStreams<'a>
    where
        Self: 'a;
    type Error = Exception;

    fn control_snapshot_bytes(
        &self,
        target: Option<&RandomState>,
        draft: Option<&Array>,
    ) -> Option<u64> {
        let mut bytes = self.inner.control_snapshot_bytes()?;
        for array in target.map(RandomState::as_array).into_iter().chain(draft) {
            bytes = bytes
                .checked_add(array.nbytes() as u64)?
                .checked_add(4096)?;
        }
        bytes.checked_add(std::mem::size_of::<Self>() as u64)
    }

    fn enable_control_capture(
        &mut self,
        plan: eredu_core::capture::AdmittedCapturePlan,
    ) -> Result<(), eredu_core::capture::CaptureError> {
        if plan.is_empty() {
            self.capture = None;
            return Ok(());
        }
        validate_control_capture(&plan)?;
        self.capture = Some(std::rc::Rc::new(std::cell::RefCell::new(LogitCapture {
            session: eredu_runtime::capture::CaptureSession::new(plan),
            records: Vec::new(),
        })));
        Ok(())
    }

    fn take_control_captures(
        &mut self,
    ) -> Vec<eredu_core::speculative::SpeculativePredictionCapture> {
        self.capture.as_ref().map_or_else(Vec::new, |capture| {
            std::mem::take(&mut capture.borrow_mut().records)
        })
    }

    fn supports_exact_optimistic_promotion(&self) -> bool {
        SpeculativeSampler::<MlxSamplingBackend>::supports_exact_optimistic_promotion(&self.inner)
    }

    fn grammar_is_complete(&mut self) -> Result<bool, Self::Error> {
        SpeculativeSampler::<MlxSamplingBackend>::grammar_is_complete(&mut self.inner)
    }

    fn prefix_is_complete(&self, history: &[u32]) -> Result<bool, Self::Error> {
        SpeculativeSampler::<MlxSamplingBackend>::prefix_is_complete(&self.inner, history)
    }

    fn randomness_root<'a>(
        seed: Option<Self::Seed>,
        _: Self::Context<'a>,
    ) -> Result<Self::RandomnessRoot, Self::Error>
    where
        Self: 'a,
    {
        seed.map(RandomState::from_key)
            .ok_or_else(|| Exception::custom("random operations require an explicit PRNG key"))
    }

    fn target_randomness_from_root<'a>(
        root: &mut Self::RandomnessRoot,
        context: Self::Context<'a>,
    ) -> Result<Self::RandomState, Self::Error>
    where
        Self: 'a,
    {
        root.next_key(context.target()).map(RandomState::from_key)
    }

    fn draft_randomness_from_root<'a>(
        root: &mut Self::RandomnessRoot,
        context: Self::Context<'a>,
    ) -> Result<Self::DraftRandomness, Self::Error>
    where
        Self: 'a,
    {
        let draft_key = root.next_key(context.target())?;
        let draft_key = if context.is_split() {
            if context.crosses_devices() {
                async_eval_with_event([&draft_key])?.synchronize()?;
                let copied = draft_key.copy(context.draft())?;
                async_eval_with_event([&copied])?.synchronize()?;
                copied
            } else {
                let _completion = context.wait_for_target_outputs([&draft_key])?;
                draft_key
            }
        } else {
            draft_key
        };
        Ok(draft_key)
    }

    fn draft_randomness_at<'a>(
        root: &Self::DraftRandomness,
        position: SpeculativeDraftRandomPosition,
        context: Self::Context<'a>,
    ) -> Result<Self::RandomState, Self::Error>
    where
        Self: 'a,
    {
        Ok(RandomState::from_key(crate::backend::random::split_key_at(
            root,
            position.get(),
            context.draft(),
        )?))
    }

    fn process_logits<'a>(
        &mut self,
        logits: &Self::Logits,
        temperature: f32,
        history: &[u32],
        placement: SamplingPlacement,
        context: Self::Context<'a>,
    ) -> Result<Self::Distribution, Self::Error>
    where
        Self: 'a,
    {
        if let Some(capture) = &self.capture {
            capture.borrow_mut().observe(
                logits,
                history.len() as u64,
                placement,
                sampling_stream(placement, context)?,
            )?;
        }
        SpeculativeSampler::<MlxSamplingBackend>::process_logits(
            &mut self.inner,
            &MlxTensor::from_array(logits.clone()),
            temperature,
            history,
            sampling_stream(placement, context)?,
        )
        .map(MlxTensor::into_array)
    }

    fn sample<'a>(
        &self,
        distribution: &Self::Distribution,
        temperature: f32,
        randomness: Option<&mut Self::RandomState>,
        placement: SamplingPlacement,
        context: Self::Context<'a>,
    ) -> Result<u32, Self::Error>
    where
        Self: 'a,
    {
        let stream = sampling_stream(placement, context)?;
        let token = SpeculativeSampler::<MlxSamplingBackend>::sample_processed(
            &self.inner,
            &MlxTensor::from_array(distribution.clone()),
            temperature,
            randomness,
            stream,
        )?;
        eval([token.as_array()])?;
        Ok(token.into_array().item::<u32>(stream))
    }

    fn probability_at<'a>(
        &self,
        distribution: &Self::Distribution,
        token: u32,
        placement: SamplingPlacement,
        context: Self::Context<'a>,
    ) -> Result<f32, Self::Error>
    where
        Self: 'a,
    {
        let stream = sampling_stream(placement, context)?;
        let probabilities = probabilities(distribution, stream)?;
        array_probability_at(&probabilities, token, stream)
    }

    fn sample_unit_interval<'a>(
        &self,
        randomness: Option<&mut Self::RandomState>,
        context: Self::Context<'a>,
    ) -> Result<f32, Self::Error>
    where
        Self: 'a,
    {
        uniform(randomness, context.target())
    }

    fn positive_probability_difference<'a>(
        &self,
        left: &Self::Distribution,
        right: &Self::Distribution,
        placement: SamplingPlacement,
        context: Self::Context<'a>,
    ) -> Result<Option<Self::Distribution>, Self::Error>
    where
        Self: 'a,
    {
        let stream = sampling_stream(placement, context)?;
        let left_probabilities = probabilities(left, stream)?;
        let right_probabilities = probabilities(right, stream)?;
        let difference = maximum(
            left_probabilities.subtract(&right_probabilities, stream)?,
            Array::from_f32(0.0),
            stream,
        )?;
        let mass = difference.sum(None, stream)?.item::<f32>(stream);
        if mass <= f32::EPSILON {
            Ok(None)
        } else {
            difference.log(stream).map(Some)
        }
    }

    fn update_sampler_state<'a>(
        &mut self,
        distribution: &Self::Distribution,
        token: u32,
        placement: SamplingPlacement,
        context: Self::Context<'a>,
    ) -> Result<(), Self::Error>
    where
        Self: 'a,
    {
        SpeculativeSampler::<MlxSamplingBackend>::commit_token(
            &mut self.inner,
            &MlxTensor::from_array(distribution.clone()),
            token,
            sampling_stream(placement, context)?,
        )
    }

    fn prepare_verification<'a>(
        &self,
        distributions: &mut [&mut Self::Distribution],
        temperature: f32,
        context: Self::Context<'a>,
    ) -> Result<(), Self::Error>
    where
        Self: 'a,
    {
        if temperature == 0.0 || !context.is_split() {
            return Ok(());
        }
        if context.crosses_devices() {
            async_eval_with_event(distributions.iter().map(|distribution| &**distribution))?
                .synchronize()?;
            for distribution in distributions {
                **distribution = distribution.copy(context.target())?;
            }
        } else {
            let _completion = context
                .wait_for_draft_outputs(distributions.iter().map(|distribution| &**distribution))?;
        }
        Ok(())
    }
}

fn sampling_stream<'a>(
    placement: SamplingPlacement,
    context: SpeculativeExecutionStreams<'a>,
) -> Result<&'a Stream, Exception> {
    match placement {
        SamplingPlacement::Target => Ok(context.target()),
        SamplingPlacement::Draft => Ok(context.draft()),
        _ => Err(Exception::custom(
            "unsupported speculative sampling placement requires an explicit stream",
        )),
    }
}

fn probabilities(logits: &Array, stream: &Stream) -> Result<Array, Exception> {
    softmax_axis(&logits.as_type::<f32>(stream)?, -1, true, stream)
}

pub(super) fn array_probability_at(
    probabilities: &Array,
    token: u32,
    stream: &Stream,
) -> Result<f32, Exception> {
    let vocabulary = probabilities.dim(-1);
    if vocabulary <= 0 || u64::from(token) >= vocabulary as u64 {
        return Err(Exception::custom(format!(
            "sampled token {token} exceeds vocabulary size {}",
            vocabulary
        )));
    }
    let token = i32::try_from(token)
        .map_err(|_| Exception::custom("sampled token exceeds the index domain"))?;
    let value = match probabilities.ndim() {
        2 => probabilities.try_index_device((0, token), stream)?,
        3 => probabilities.try_index_device((0, 0, token), stream)?,
        ndim => {
            return Err(Exception::custom(format!(
                "speculative distribution must be rank 2 or 3, got rank {ndim}"
            )))
        }
    };
    Ok(value.item::<f32>(stream))
}

fn uniform(state: Option<&mut RandomState>, stream: &Stream) -> Result<f32, Exception> {
    let state = state
        .ok_or_else(|| Exception::custom("stochastic speculative decoding requires a PRNG key"))?;
    let key = state.next_key(stream)?;
    Ok(random::uniform::<_, f32>(0.0, 1.0, &[1], &key, stream)?.item::<f32>(stream))
}

struct LogitCapture {
    session: eredu_runtime::capture::CaptureSession,
    records: Vec<eredu_core::speculative::SpeculativePredictionCapture>,
}
impl LogitCapture {
    fn observe(
        &mut self,
        logits: &Array,
        position: u64,
        placement: SamplingPlacement,
        stream: &Stream,
    ) -> Result<(), Exception> {
        use eredu_core::{
            capture::CapturePhase,
            speculative::{SpeculativeCaptureRole, SpeculativePredictionCapture},
        };
        let role = match placement {
            SamplingPlacement::Target => SpeculativeCaptureRole::Target,
            SamplingPlacement::Draft => SpeculativeCaptureRole::Draft,
            _ => {
                return Err(Exception::custom(
                    "unsupported speculative capture placement",
                ))
            }
        };
        let phase = if position == 0 {
            CapturePhase::Prefill
        } else {
            CapturePhase::Decode
        };
        self.session
            .begin_step(phase, position)
            .map_err(|e| Exception::custom(e.to_string()))?;
        // Sampling receives one vocabulary row. Normalize only its logical shape
        // to the ordinary model.logits catalog; the collector owns reservation,
        // evaluation, bounded transformation and host transfer.
        let values = logits.reshape(&[1, 1, -1], stream)?;
        let tensor = MlxTensor::from_array(values);
        let mut backend =
            crate::composition::mlx::session::bounded_capture::NativeCapture { stream };
        self.session
            .observe(
                &mut backend,
                eredu_core::MODEL_LOGITS_OBSERVATION_PATH,
                &tensor,
            )
            .map_err(|e| Exception::custom(e.to_string()))?;
        if let Some(capture) = self.session.take_step() {
            self.records.push(SpeculativePredictionCapture {
                role,
                position,
                capture,
            });
        }
        Ok(())
    }
}

pub(crate) fn validate_control_capture(
    plan: &eredu_core::capture::AdmittedCapturePlan,
) -> Result<(), eredu_core::capture::CaptureError> {
    if plan.is_empty() {
        return Ok(());
    }
    if plan.request().batch != 1 || plan.request().prompt_tokens != 1 {
        return Err(eredu_core::capture::CaptureError::Invalid(
            "speculative logits require one-row capture admission".into(),
        ));
    }
    if plan
        .plan()
        .selections
        .iter()
        .any(|s| s.path != eredu_core::MODEL_LOGITS_OBSERVATION_PATH)
    {
        return Err(eredu_core::capture::CaptureError::Unsupported(
                "controlled MLX speculation currently captures model.logits; layer captures require draft/verification phase attribution".into()));
    }
    Ok(())
}
