//! Borrow the actual completed B source; public replaceable input views cannot
//! manufacture this loan. Cloning the returned original owner shares custody.
use super::*;
impl MlxModelInput {
    pub(crate) fn original_prediction_source(
        &self,
        pool: &MemoryLedger,
    ) -> Result<&eredu_runtime::input::PreparedModelInputOwner<crate::MlxTensor>, WorkingMemoryError>
    {
        let cache = self
            .cache_identity
            .as_ref()
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        match &self.parts {
            super::super::super::super::pending_prompt::ModelInputParts::OriginalText(source) => {
                source.prediction_source(pool, &self.parts, cache)
            }
            super::super::super::super::pending_prompt::ModelInputParts::Original(_) => {
                let Some(input::OriginalMediaPacket::Original(packet)) = &self.original_media
                else {
                    return Err(WorkingMemoryError::IdentityMismatch);
                };
                packet
                    .validate_request_source(pool, &self.parts, cache)
                    .map_err(|_| WorkingMemoryError::IdentityMismatch)?;
                packet
                    .body
                    .prepared()
                    .ok_or(WorkingMemoryError::IdentityMismatch)
            }
            _ => Err(WorkingMemoryError::UnknownBound),
        }
    }
}
