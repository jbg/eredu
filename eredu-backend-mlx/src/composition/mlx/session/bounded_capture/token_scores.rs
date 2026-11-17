//! Full-vocabulary reductions with bounded host materialization.
use super::*;
use safemlx::ops::indexing::TryIndexMutOp;

impl NativeCapture<'_> {
    pub(super) fn token_scores(
        &self,
        tensor: &MlxTensor,
        ids: &[u32],
        position: Option<u64>,
    ) -> Result<CapturePayload, Exception> {
        let stream = self.stream;
        let (row, vocabulary, maximum, log_mass) = super::super::score_readout::normalized_row(
            tensor.as_array(),
            ids,
            position,
            stream,
            |elements| {
                #[cfg(test)]
                record_host_read(elements);
                #[cfg(not(test))]
                let _ = elements;
            },
        )?;
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
