//! Full-vocabulary reductions with bounded host materialization.
use super::*;
use safemlx::ops::indexing::TryIndexMutOp;

impl NativeCapture<'_> {
    pub(super) fn token_scores(
        &self,
        tensor: &MlxTensor,
        ids: &[u32],
    ) -> Result<CapturePayload, Exception> {
        let stream = self.stream;
        let vocabulary = *tensor
            .shape()
            .last()
            .ok_or_else(|| Exception::custom("scalar logits"))?;
        if vocabulary <= 0 || ids.iter().any(|id| *id >= vocabulary as u32) {
            return Err(Exception::custom("invalid selected score vocabulary"));
        }
        let row = tensor
            .as_array()
            .reshape(&[-1, vocabulary], stream)?
            .try_index_device((-1, ..), stream)?
            .as_dtype(Dtype::Float32, stream)?;
        if count_mask(row.is_finite(stream)?, stream)? != vocabulary as u64 {
            return Err(Exception::custom(
                "full-vocabulary scoring requires finite raw logits",
            ));
        }
        let maximum = scalar_f32(row.max(false, stream)?)?;
        let mut total = 0.0f64;
        let mut correction = 0.0f64;
        for start in (0..vocabulary).step_by(CHUNK as usize) {
            let end = start.saturating_add(CHUNK as i32).min(vocabulary);
            let chunk = row.try_index_device(start..end, stream)?;
            let sum = scalar_f32(
                chunk
                    .subtract(Array::from_f32(maximum as f32), stream)?
                    .exp(stream)?
                    .sum(false, stream)?,
            )?;
            let next = total + sum;
            correction += if total.abs() >= sum.abs() {
                (total - next) + sum
            } else {
                (sum - next) + total
            };
            total = next;
        }
        let log_mass = (total + correction).ln();
        let log_partition = maximum + log_mass;
        let candidate = |token_id, score| CaptureCandidate {
            token_id,
            score,
            allowed: self
                .domain
                .is_none_or(|domain| domain.filter.allows(token_id)),
        };
        let mut scores = Vec::with_capacity(ids.len());
        for &id in ids {
            let score = scalar_f32(row.try_index_device(id as i32, stream)?)? as f32;
            let rank = 1 + count_mask(row.gt(Array::from_f32(score), stream)?, stream)?;
            let strongest_alternative = if vocabulary == 1 {
                None
            } else {
                let mut alternatives = row.clone();
                alternatives.try_index_mut_device(
                    id as i32,
                    Array::from_f32(f32::NEG_INFINITY),
                    stream,
                )?;
                let winner = safemlx::ops::indexing::argmax(&alternatives, false, stream)?;
                #[cfg(test)]
                record_host_read(1);
                let winner = winner.evaluated()?.as_slice::<u32>()[0];
                let value = scalar_f32(row.try_index_device(winner as i32, stream)?)? as f32;
                Some(candidate(winner, value))
            };
            scores.push(CaptureTokenScore {
                target: candidate(id, score),
                log_probability: (f64::from(score) - maximum) - log_mass,
                rank,
                strongest_alternative,
            });
        }
        Ok(CapturePayload::TokenScores(CaptureTokenScores {
            stage: CandidateScoreStage::RawLogitsBeforeSampling,
            source: CandidateLogitsSource::Original,
            vocabulary: vocabulary as u64,
            log_partition,
            scores,
            domain: self.domain.map(|domain| domain.summary(vocabulary as u32)),
        }))
    }
}
