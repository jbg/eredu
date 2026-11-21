//! Native row-wise numerical mechanisms for iterative token refinement.

use std::num::NonZeroU32;

use eredu_core::refinement::RefinementBackend;
use eredu_core::sampling::{SamplingBackend, TokenDomain};
use safemlx::ops::indexing::TryIndexOp;
use safemlx::{error::Exception, Array, Dtype, Stream};

use super::MlxSamplingBackend;
use crate::{backend::random::RandomState, MlxTensor};

fn entropy(logits: &MlxTensor, stream: &Stream) -> Result<Array, Exception> {
    let scores = logits.as_array().as_dtype(Dtype::Float32, stream)?;
    let log_probs = scores.subtract(scores.logsumexp_axis(-1, true, stream)?, stream)?;
    let probs = log_probs.exp(stream)?;
    let contributions = safemlx::ops::r#where(
        probs.gt(Array::from_f32(0.0), stream)?,
        probs.multiply(log_probs, stream)?,
        Array::from_f32(0.0),
        stream,
    )?;
    contributions.sum_axis(-1, false, stream)?.negative(stream)
}

impl RefinementBackend for MlxSamplingBackend {
    type Logits = MlxTensor;
    type Token = MlxTensor;
    type RandomState = RandomState;
    type Context = Stream;
    type Error = Exception;
    type Mask = MlxTensor;
    type ConditioningFormat = Dtype;

    fn read_tokens(tokens: &MlxTensor, stream: &Stream) -> Result<Vec<u32>, Exception> {
        tokens
            .as_array()
            .as_dtype(Dtype::Uint32, stream)?
            .evaluated()?
            .try_to_vec::<u32>()
            .map_err(Exception::from_source)
    }

    fn read_mask(flags: &MlxTensor, _: &Stream) -> Result<Vec<bool>, Exception> {
        flags
            .as_array()
            .evaluated()?
            .try_to_vec::<bool>()
            .map_err(Exception::from_source)
    }

    fn masked_score_shape(scores: &MlxTensor) -> Option<[usize; 2]> {
        let [positions, width] = *scores.as_array().shape() else {
            return None;
        };
        (positions > 0 && width > 0).then(|| [positions as usize, width as usize])
    }

    fn read_scores(scores: &MlxTensor, stream: &Stream) -> Result<Vec<f32>, Exception> {
        scores
            .as_array()
            .as_dtype(Dtype::Float32, stream)?
            .evaluated()?
            .try_to_vec::<f32>()
            .map_err(Exception::from_source)
    }

    fn masked_uniforms(
        count: usize,
        random: &mut RandomState,
        stream: &Stream,
    ) -> Result<Vec<f64>, Exception> {
        // MLX has no float64 tensor operations. Draw 53 bits per host value.
        let count = i32::try_from(count)
            .map_err(|_| Exception::custom("masked noise exceeds addressable geometry"))?;
        let high_key = random.next_key(stream)?;
        let high =
            safemlx::random::randint::<_, i32>(0, 1_i32 << 26, &[count][..], &high_key, stream)?
                .evaluated()?
                .try_to_vec::<i32>()
                .map_err(|e| Exception::custom(e.to_string()))?;
        let low_key = random.next_key(stream)?;
        let low =
            safemlx::random::randint::<_, i32>(0, 1_i32 << 27, &[count][..], &low_key, stream)?
                .evaluated()?
                .try_to_vec::<i32>()
                .map_err(|e| Exception::custom(e.to_string()))?;
        Ok(high
            .into_iter()
            .zip(low)
            .map(|(high, low)| {
                let bits = ((high as u64) << 27) | low as u64;
                (bits as f64 + 1.0) / ((1_u64 << 53) as f64 + 2.0)
            })
            .collect())
    }

    fn materialize_tokens(tokens: &[u32], _: &Stream) -> Result<MlxTensor, Exception> {
        let length = i32::try_from(tokens.len()).map_err(Exception::from_source)?;
        let tokens: Vec<_> = tokens.iter().map(|&id| id as i32).collect();
        Ok(MlxTensor::from_array(Array::from_slice(&tokens, &[length])))
    }

    fn materialize_mask(flags: &[bool], _: &Stream) -> Result<MlxTensor, Exception> {
        let length = i32::try_from(flags.len()).map_err(Exception::from_source)?;
        Ok(MlxTensor::from_array(Array::from_slice(flags, &[length])))
    }

    fn materialize_scores(
        scores: &[f32],
        [positions, width]: [usize; 2],
        _: &Stream,
    ) -> Result<MlxTensor, Exception> {
        let positions = i32::try_from(positions).map_err(Exception::from_source)?;
        let width = i32::try_from(width).map_err(Exception::from_source)?;
        Ok(MlxTensor::from_array(Array::from_slice(
            scores,
            &[positions, width],
        )))
    }

    fn pin_scores(
        scores: &MlxTensor,
        tokens: &MlxTensor,
        pins: &MlxTensor,
        stream: &Stream,
    ) -> Result<MlxTensor, Exception> {
        let width = scores.as_array().dim(-1);
        let ids = Array::arange::<i32, i32>(None, width, None, stream)?;
        let selected = tokens
            .as_array()
            .reshape(&[-1, 1], stream)?
            .eq(ids, stream)?;
        let fixed = safemlx::ops::r#where(
            selected,
            Array::from_f32(0.0),
            Array::from_f32(f32::NEG_INFINITY),
            stream,
        )?;
        safemlx::ops::r#where(
            pins.as_array().reshape(&[-1, 1], stream)?,
            fixed,
            scores.as_array(),
            stream,
        )
        .map(MlxTensor::from_array)
    }

    fn validate_refinement_logits(
        logits: &MlxTensor,
        canvas: NonZeroU32,
        domain: TokenDomain,
    ) -> Result<(), Exception> {
        let shape = logits.as_array().shape();
        if shape != [canvas.get() as i32, domain.cardinality() as i32]
            || !matches!(
                logits.as_array().dtype(),
                Dtype::Float32 | Dtype::Float16 | Dtype::Bfloat16
            )
        {
            return Err(Exception::custom(
                "refinement requires a floating [canvas, vocabulary] score matrix",
            ));
        }
        Ok(())
    }

    fn scale_temperature(
        logits: &MlxTensor,
        temperature: f32,
        stream: &Stream,
    ) -> Result<MlxTensor, Exception> {
        let values = logits.as_array();
        let valid = values
            .is_nan(stream)?
            .logical_not(stream)?
            .logical_and(values.ne(Array::from_f32(f32::INFINITY), stream)?, stream)?
            .all(false, stream)?
            .logical_and(
                values
                    .is_finite(stream)?
                    .any_axis(-1, false, stream)?
                    .all(false, stream)?,
                stream,
            )?;
        if !valid.try_item::<bool>(stream)? {
            return Err(Exception::custom(
                "refinement rows require a finite candidate and forbid NaN or positive infinity",
            ));
        }
        <Self as SamplingBackend>::scale_temperature(logits, temperature, stream)
    }

    fn uniform_tokens(
        canvas: NonZeroU32,
        domain: TokenDomain,
        random: &mut RandomState,
        stream: &Stream,
    ) -> Result<MlxTensor, Exception> {
        let key = random.next_key(stream)?;
        safemlx::random::randint::<_, i32>(
            0,
            domain.cardinality() as i32,
            &[canvas.get() as i32][..],
            &key,
            stream,
        )
        .map(MlxTensor::from_array)
    }

    fn categorical_tokens(
        logits: &MlxTensor,
        random: &mut RandomState,
        stream: &Stream,
    ) -> Result<MlxTensor, Exception> {
        let key = random.next_key(stream)?;
        safemlx::random::categorical(
            logits.as_array().as_dtype(Dtype::Float32, stream)?,
            -1,
            None,
            &key,
            stream,
        )?
        .as_dtype(Dtype::Int32, stream)
        .map(MlxTensor::from_array)
    }

    fn argmax_tokens(logits: &MlxTensor, stream: &Stream) -> Result<MlxTensor, Exception> {
        safemlx::argmax_axis!(logits.as_array(), -1, stream = stream)?
            .as_dtype(Dtype::Int32, stream)
            .map(MlxTensor::from_array)
    }

    fn entropy_bound_mask(
        logits: &MlxTensor,
        bound: f32,
        stream: &Stream,
    ) -> Result<MlxTensor, Exception> {
        let entropies = entropy(logits, stream)?;
        let length = entropies.dim(0);
        // Unique integer ranks make tie ordering independent of the native
        // sort's stability. The temporary is canvas-sized, not vocabulary-sized.
        let row = entropies.reshape(&[1, length], stream)?;
        let column = entropies.reshape(&[length, 1], stream)?;
        let indices = Array::arange::<i32, i32>(None, length, None, stream)?;
        let earlier = indices
            .reshape(&[1, length], stream)?
            .lt(indices.reshape(&[length, 1], stream)?, stream)?;
        let precedes = row.lt(&column, stream)?.logical_or(
            row.eq(&column, stream)?.logical_and(earlier, stream)?,
            stream,
        )?;
        let ranks = precedes
            .as_dtype(Dtype::Int32, stream)?
            .sum_axis(-1, false, stream)?;
        let order = safemlx::ops::argsort_axis(&ranks, -1, stream)?;
        let sorted = safemlx::ops::indexing::take_along_axis(&entropies, &order, -1, stream)?;
        let accepted = sorted
            .cumsum(-1, None, None, stream)?
            .subtract(&sorted, stream)?
            .le(Array::from_f32(bound), stream)?;
        safemlx::ops::indexing::put_along_axis(&accepted, &order, &accepted, -1, stream)
            .map(MlxTensor::from_array)
    }

    fn select_tokens(
        mask: &MlxTensor,
        accepted: &MlxTensor,
        other: &MlxTensor,
        stream: &Stream,
    ) -> Result<MlxTensor, Exception> {
        safemlx::ops::r#where(
            mask.as_array(),
            accepted.as_array(),
            other.as_array(),
            stream,
        )
        .map(MlxTensor::from_array)
    }

    fn retain_self_conditioning(
        logits: &MlxTensor,
        format: &Dtype,
        stream: &Stream,
    ) -> Result<MlxTensor, Exception> {
        if !matches!(format, Dtype::Float32 | Dtype::Float16 | Dtype::Bfloat16) {
            return Err(Exception::custom(
                "refinement conditioning requires an embedding floating dtype",
            ));
        }
        logits
            .as_array()
            .as_dtype(*format, stream)
            .map(MlxTensor::from_array)
    }

    fn stable_and_confident(
        logits: &MlxTensor,
        argmax: &MlxTensor,
        preceding: &[MlxTensor],
        confidence_threshold: f32,
        pins: Option<&MlxTensor>,
        stream: &Stream,
    ) -> Result<bool, Exception> {
        let entropies = entropy(logits, stream)?;
        let mean = match pins {
            Some(pins) => {
                let active = pins.as_array().logical_not(stream)?;
                let count = active
                    .as_dtype(Dtype::Float32, stream)?
                    .sum(false, stream)?;
                let count = safemlx::ops::maximum(count, Array::from_f32(1.0), stream)?;
                safemlx::ops::r#where(active, entropies, Array::from_f32(0.0), stream)?
                    .sum(false, stream)?
                    .divide(count, stream)?
            }
            None => entropies.mean(false, stream)?,
        };
        let mut stop = mean.lt(Array::from_f32(confidence_threshold), stream)?;
        for old in preceding {
            stop = stop.logical_and(
                argmax
                    .as_array()
                    .eq(old.as_array(), stream)?
                    .all(false, stream)?,
                stream,
            )?;
        }
        stop.try_item(stream)
    }

    fn settle_refinement(
        canvas: &MlxTensor,
        argmax: &MlxTensor,
        scores: &MlxTensor,
        conditioning: &MlxTensor,
        acceptance: &MlxTensor,
        random: &RandomState,
        _: &Stream,
    ) -> Result<(), Exception> {
        safemlx::transforms::eval([
            canvas.as_array(),
            argmax.as_array(),
            scores.as_array(),
            conditioning.as_array(),
            acceptance.as_array(),
            random.as_array(),
        ])
    }

    fn settle_initial_canvas(
        canvas: &MlxTensor,
        random: &RandomState,
        _: &Stream,
    ) -> Result<(), Exception> {
        safemlx::transforms::eval([canvas.as_array(), random.as_array()])
    }
}

impl eredu_core::refinement::RefinementFinalizationBackend for MlxSamplingBackend {
    fn candidate_at(
        candidates: &MlxTensor,
        position: usize,
        stream: &Stream,
    ) -> Result<MlxTensor, Exception> {
        if candidates.as_array().shape().len() != 1 || position >= candidates.as_array().size() {
            return Err(Exception::custom("invalid selected canvas position"));
        }
        candidates
            .as_array()
            .try_index_device(position as i32, stream)
            .map(MlxTensor::from_array)
    }

    fn candidate_score_is_finite(
        candidate: &MlxTensor,
        scores: &MlxTensor,
        stream: &Stream,
    ) -> Result<bool, Exception> {
        let token = <Self as SamplingBackend>::token_id(candidate, stream)?;
        if scores.as_array().shape().len() != 2
            || scores.as_array().dim(0) != 1
            || token >= scores.as_array().dim(1) as u32
        {
            return Err(Exception::custom(
                "selected candidate lies outside position scores",
            ));
        }
        scores
            .as_array()
            .try_index_device((0, token as i32), stream)?
            .is_finite(stream)?
            .try_item(stream)
    }
    fn classify_position_scores(
        scores: &MlxTensor,
        stream: &Stream,
    ) -> Result<eredu_core::refinement::PositionScoreClass, Exception> {
        use eredu_core::refinement::PositionScoreClass;

        let values = scores.as_array();
        if values
            .is_nan(stream)?
            .logical_or(values.eq(Array::from_f32(f32::INFINITY), stream)?, stream)?
            .any(false, stream)?
            .try_item::<bool>(stream)?
        {
            return Ok(PositionScoreClass::Invalid);
        }
        let finite = values
            .is_finite(stream)?
            .any(false, stream)?
            .try_item::<bool>(stream)?;
        Ok(if finite {
            PositionScoreClass::HasFinite
        } else {
            PositionScoreClass::NoFinite
        })
    }
}
