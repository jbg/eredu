use super::*;
use eredu_core::speculative::sampling::SpeculativeSamplingPrimitives;

#[derive(Clone)]
pub struct MlxSpeculativeSamplingPrimitives;

impl SpeculativeSamplingPrimitives for MlxSpeculativeSamplingPrimitives {
    type Backend = MlxSamplingBackend;
    type Value = Array;
    type Seed = Array;
    type DraftRandomness = Array;
    type RandomnessRoot = RandomState;
    type CaptureBinding =
        crate::composition::mlx::session::bounded_capture::MlxSpeculativeCaptureMechanisms;
    type Error = Exception;
    type Capture<'a> = crate::composition::mlx::session::bounded_capture::NativeCapture<'a>;

    fn wrap(value: &Array) -> MlxTensor {
        MlxTensor::from_array(value.clone())
    }

    fn unwrap(value: MlxTensor) -> Array {
        value.into_array()
    }

    fn vocabulary(value: &Array) -> usize {
        value.shape().last().copied().unwrap_or(0) as usize
    }

    fn candidate(token: u32) -> MlxTensor {
        MlxTensor::from_array(Array::from_slice(&[token], &[1]))
    }

    fn snapshot_randomness(target: Option<&RandomState>, draft: Option<&Array>) -> Option<u64> {
        target
            .map(RandomState::as_array)
            .into_iter()
            .chain(draft)
            .try_fold(0u64, |bytes, array| {
                bytes.checked_add(array.nbytes() as u64)?.checked_add(4096)
            })
    }

    fn continuation_randomness_bytes() -> u64 {
        2 * (4096 + 32)
    }

    fn from_error(error: impl std::error::Error + Send + Sync + 'static) -> Exception {
        Exception::from_source(error)
    }

    fn score_capture<'a>(
        stream: &'a Stream,
        domain: Option<eredu_core::capture::CaptureTokenDomain<'a>>,
    ) -> Self::Capture<'a> {
        crate::composition::mlx::session::bounded_capture::NativeCapture {
            partition: None,
            stream,
            domain,
        }
    }

    fn intervention_estimator(
    ) -> std::sync::Arc<dyn eredu_core::intervention::InterventionEstimator> {
        std::sync::Arc::new(
            crate::composition::mlx::session::intervention::NativeInterventionEstimator,
        )
    }

    fn capture_scores(value: &MlxTensor, stream: &Stream) -> Result<MlxTensor, Exception> {
        value
            .as_array()
            .reshape(&[1, 1, -1], stream)
            .map(MlxTensor::from_array)
    }

    fn restore_scores(
        value: MlxTensor,
        original: &MlxTensor,
        stream: &Stream,
    ) -> Result<MlxTensor, Exception> {
        value
            .into_array()
            .reshape(original.as_array().shape(), stream)
            .map(MlxTensor::from_array)
    }

    fn sample_token(token: MlxTensor, stream: &Stream) -> Result<u32, Exception> {
        eval([token.as_array()])?;
        Ok(token.into_array().item::<u32>(stream))
    }
    fn seed(seed: u64) -> Result<Array, Exception> {
        safemlx::random::key(seed)
    }

    fn randomness_root(seed: Option<Array>) -> Result<RandomState, Exception> {
        seed.map(RandomState::from_key)
            .ok_or_else(|| Exception::custom("random operations require an explicit PRNG key"))
    }

    fn target_randomness_from_root(
        root: &mut RandomState,
        stream: &Stream,
    ) -> Result<RandomState, Exception> {
        root.next_key(stream).map(RandomState::from_key)
    }

    fn draft_randomness_from_root(
        root: &mut RandomState,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        root.next_key(stream)
    }

    fn draft_key_root(key: &Array) -> &Array {
        key
    }

    fn copy_draft_key(key: &Array, destination: &Stream) -> Result<Array, Exception> {
        super::state_snapshot::copy_array(key, destination)
    }

    fn draft_randomness_at(
        root: &Array,
        position: SpeculativeDraftRandomPosition,
        stream: &Stream,
    ) -> Result<RandomState, Exception> {
        Ok(RandomState::from_key(crate::backend::random::split_key_at(
            root,
            position.get(),
            stream,
        )?))
    }

    fn probability_at(
        distribution: &Self::Value,
        token: u32,
        stream: &Stream,
    ) -> Result<f32, Self::Error> {
        let probabilities = probabilities(distribution, stream)?;
        array_probability_at(&probabilities, token, stream)
    }

    fn sample_unit_interval(
        randomness: Option<&mut RandomState>,
        stream: &Stream,
    ) -> Result<f32, Self::Error> {
        uniform(randomness, stream)
    }

    fn positive_probability_difference(
        left: &Self::Value,
        right: &Self::Value,
        stream: &Stream,
    ) -> Result<Option<Self::Value>, Self::Error> {
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

    fn distribution_root(value: &Array) -> &Array {
        value
    }

    fn copy_distributions(
        distributions: &mut [&mut Array],
        destination: &Stream,
    ) -> Result<(), Exception> {
        async_eval_with_event(distributions.iter().map(|distribution| &**distribution))?
            .synchronize()?;
        for distribution in distributions {
            **distribution = distribution.copy(destination)?;
        }
        Ok(())
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
