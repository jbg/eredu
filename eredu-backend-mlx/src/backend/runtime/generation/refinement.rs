//! Native row-wise numerical mechanisms for iterative token refinement.

use std::num::NonZeroU32;

use eredu_runtime::{generation::refinement::RefinementBackend, SamplingBackend, TokenDomain};
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

    fn seed_canvas(
        noise: &MlxTensor,
        layout: &eredu_core::refinement::RefinementCanvas,
        stream: &Stream,
    ) -> Result<(MlxTensor, MlxTensor), Exception> {
        let values: Vec<i32> = layout
            .seed_tokens
            .iter()
            .map(|id| id.unwrap_or(0) as i32)
            .collect();
        let seeded: Vec<bool> = layout.seed_tokens.iter().map(Option::is_some).collect();
        let mut pins = vec![false; values.len()];
        for position in &layout.pinned_positions {
            pins[*position as usize] = true;
        }
        let shape = [values.len() as i32];
        let canvas = safemlx::ops::r#where(
            Array::from_slice(&seeded, &shape),
            Array::from_slice(&values, &shape),
            noise.as_array(),
            stream,
        )?;
        Ok((
            MlxTensor::from_array(canvas),
            MlxTensor::from_array(Array::from_slice(&pins, &shape)),
        ))
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

impl eredu_runtime::generation::refinement::finalization::RefinementFinalizationBackend
    for MlxSamplingBackend
{
    fn position_has_candidate(scores: &MlxTensor, stream: &Stream) -> Result<bool, Exception> {
        let values = scores.as_array();
        if values
            .is_nan(stream)?
            .logical_or(values.eq(Array::from_f32(f32::INFINITY), stream)?, stream)?
            .any(false, stream)?
            .try_item::<bool>(stream)?
        {
            return Err(Exception::custom("invalid finalization position scores"));
        }
        values
            .is_finite(stream)?
            .any(false, stream)?
            .try_item::<bool>(stream)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constrained_finalization_uses_each_final_position_domain() {
        use eredu_core::{refinement::PredictionScoreKind, TokenFilter};
        use eredu_runtime::generation::refinement::finalization::{
            finalize_refinement_position, FinalizationError,
        };
        let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let scores = MlxTensor::from_array(Array::from_slice(
            &[1.0f32, 4.0, 3.0, -2.0, 7.0, 2.0, 5.0, 1.0],
            &[2, 4],
        ));
        for (position, filter, expected) in [
            (0, TokenFilter::All, 1),
            (
                0,
                TokenFilter::allowed(vec![true, false, true, false]).unwrap(),
                2,
            ),
            (
                1,
                TokenFilter::allowed(vec![false, true, false, true]).unwrap(),
                1,
            ),
        ] {
            let finalized = finalize_refinement_position::<MlxSamplingBackend>(
                &scores,
                position,
                TokenDomain::new(4),
                &filter,
                &stream,
            )
            .unwrap();
            assert_eq!(
                <MlxSamplingBackend as SamplingBackend>::token_id(&finalized.token, &stream)
                    .unwrap(),
                expected
            );
            assert_eq!(finalized.scores.as_array().shape(), [1, 4]);
            assert_eq!(
                finalized.score_kind,
                PredictionScoreKind::ConstrainedFinalization
            );
        }
        let fixed = MlxTensor::from_array(Array::from_slice(
            &[f32::NEG_INFINITY, 0.0, f32::NEG_INFINITY, f32::NEG_INFINITY],
            &[1, 4],
        ));
        let incompatible = TokenFilter::allowed(vec![true, false, true, false]).unwrap();
        assert!(matches!(
            finalize_refinement_position::<MlxSamplingBackend>(
                &fixed,
                0,
                TokenDomain::new(4),
                &incompatible,
                &stream
            ),
            Err(FinalizationError::NoAdmissibleToken)
        ));
        let absent = TokenFilter::allowed(vec![false, false, false, false, true]).unwrap();
        assert!(matches!(
            finalize_refinement_position::<MlxSamplingBackend>(
                &scores,
                0,
                TokenDomain::new(4),
                &absent,
                &stream
            ),
            Err(FinalizationError::Backend(_))
        ));
        assert!(matches!(
            finalize_refinement_position::<MlxSamplingBackend>(
                &scores,
                2,
                TokenDomain::new(4),
                &TokenFilter::All,
                &stream
            ),
            Err(FinalizationError::Geometry)
        ));
    }

    #[test]
    fn native_entropy_acceptance_and_feedback_precision_match_position_policy() {
        let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let scores = MlxTensor::from_array(Array::from_slice(
            &[0.0f32, 0.0, 4.0, 0.0, 2.0, 0.0],
            &[3, 2],
        ));
        let mask = MlxSamplingBackend::entropy_bound_mask(&scores, 0.1, &stream).unwrap();
        assert_eq!(
            mask.as_array()
                .clone()
                .into_evaluated()
                .unwrap()
                .try_to_vec::<bool>()
                .unwrap(),
            [false, true, true]
        );
        let ties = MlxTensor::from_array(Array::from_slice(&[0.0f32; 6], &[3, 2]));
        let mask = MlxSamplingBackend::entropy_bound_mask(&ties, 0.1, &stream).unwrap();
        assert_eq!(
            mask.as_array()
                .clone()
                .into_evaluated()
                .unwrap()
                .try_to_vec::<bool>()
                .unwrap(),
            [true, false, false]
        );
        let feedback =
            MlxSamplingBackend::retain_self_conditioning(&scores, &Dtype::Bfloat16, &stream)
                .unwrap();
        assert_eq!(feedback.as_array().dtype(), Dtype::Bfloat16);
        let high = MlxTensor::from_array(Array::from_slice(&[20.0f32, 0.0, 20.0, 0.0], &[2, 2]));
        let argmax = MlxSamplingBackend::argmax_tokens(&high, &stream).unwrap();
        assert!(MlxSamplingBackend::stable_and_confident(
            &high,
            &argmax,
            std::slice::from_ref(&argmax),
            0.005,
            None,
            &stream
        )
        .unwrap());
        let changed = MlxTensor::from_array(Array::from_slice(&[1i32, 0], &[2]));
        assert!(!MlxSamplingBackend::stable_and_confident(
            &high,
            &argmax,
            &[changed],
            0.005,
            None,
            &stream
        )
        .unwrap());
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let scores = MlxTensor::from_array(Array::from_slice(&[invalid, invalid], &[1, 2]));
            assert!(
                <MlxSamplingBackend as RefinementBackend>::scale_temperature(&scores, 1.0, &stream)
                    .is_err()
            );
        }
    }
}
