//! Full-vocabulary reductions with bounded host materialization.
use super::*;
use crate::backend::array_copy::{CaptureCompletion, TokenScoreProgram};

impl NativeCapture<'_> {
    pub(super) fn token_scores(
        &self,
        tensor: &MlxTensor,
        ids: &[u32],
    ) -> Result<CapturePayload, Exception> {
        let vocabulary = *tensor
            .shape()
            .last()
            .ok_or_else(|| Exception::custom("scalar logits"))?;
        if vocabulary <= 0 || ids.iter().any(|id| *id >= vocabulary as u32) {
            return Err(Exception::custom("invalid selected score vocabulary"));
        }
        let program = TokenScoreProgram::new(vocabulary, ids)
            .map_err(|cause| Exception::custom(cause.to_string()))?;
        let mut scores = Vec::with_capacity(ids.len());
        let log_partition = program
            .execute(
                tensor.as_array(),
                self.stream,
                CaptureCompletion::Ordinary,
                &mut |_| Ok(()),
                &mut |count| {
                    #[cfg(test)]
                    record_host_read(count);
                    #[cfg(not(test))]
                    let _ = count;
                },
                |id| self.domain.is_none_or(|domain| domain.filter.allows(id)),
                |score| {
                    scores.push(score);
                    Ok(())
                },
            )
            .map_err(|cause| Exception::custom(cause.to_string()))?;
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
