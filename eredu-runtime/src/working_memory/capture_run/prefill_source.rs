//! Claim-free bootstrap from the original immutable scheduled bank.
use super::*;
use crate::inspection::{PrefillChunkRetentionContext, PreparedPrefillChunkRetention};

/// Borrowed original-bank custody, obtainable only through the closed funded
/// observer. It cannot be built from equal geometry, a reservation or bytes.
/// Creating a segment spends no frame/tensor claim and opens no funding scope.
pub struct CapturePrefillSourceBootstrap<'a> {
    source: &'a SharedCapturePlan,
    interventions: Option<&'a super::super::OriginalInterventionSource>,
    custody: &'a CaptureTensorCustody,
}
impl CapturePrefillSourceBootstrap<'_> {
    /// The actual immutable source; this borrow carries no admission override.
    pub fn source(&self) -> &SharedCapturePlan {
        self.source
    }
    /// Exact original intervention owner admitted with this capture bank.
    /// Its companion geometry is descriptive; this borrow issues no claim.
    pub fn intervention_source(&self) -> Option<&super::super::OriginalInterventionSource> {
        self.interventions
    }
    /// Bind this original scheduled account to the canonical runtime chunk and
    /// the backend's actual active scope. No native completion is inferred.
    pub fn begin_segment(
        self,
        native: &mut WorkingMemoryFundingScope,
        context: &PrefillChunkRetentionContext<'_>,
    ) -> Result<
        (
            crate::working_memory::CaptureSourceSegment,
            PreparedPrefillChunkRetention,
        ),
        WorkingMemoryError,
    > {
        let geometry = context.geometry();
        let admission = self.source.admission();
        let request = admission.request();
        let chunk = context.chunk();
        if geometry.batch_size == 0
            || geometry
                .cached_positions
                .checked_add(geometry.input_positions)
                .and_then(|n| n.checked_add(geometry.max_output_tokens))
                .is_none()
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        if request.batch != geometry.batch_size
            || request.prompt_tokens != geometry.input_positions
            || geometry.max_output_tokens > request.max_predictions
            || admission.text_origin().map(|o| o.cached_positions)
                != Some(geometry.cached_positions)
            || geometry.input_positions == 0
            || geometry.prefill_chunk_positions == 0
            || geometry.prefill_chunk_positions > geometry.input_positions
            || chunk.input.start >= geometry.input_positions
            || chunk.input.start % geometry.prefill_chunk_positions != 0
            || chunk.input.end
                != chunk
                    .input
                    .start
                    .saturating_add(geometry.prefill_chunk_positions)
                    .min(geometry.input_positions)
            || geometry.cached_positions.checked_add(chunk.input.start) != Some(chunk.position)
            || geometry
                .output
                .for_chunk(chunk.input.end == geometry.input_positions)
                != chunk.output
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        self.custody
            .begin_prefill_source_segment(self.source, self.interventions, native, context)
    }
}
impl PreparedCaptureRun {
    pub(crate) fn prefill_source_bootstrap(
        &self,
    ) -> Result<CapturePrefillSourceBootstrap<'_>, WorkingMemoryError> {
        self.custody.validate()?;
        if self.next != 0 {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        Ok(CapturePrefillSourceBootstrap {
            source: &self.source,
            interventions: self.interventions.as_ref(),
            custody: &self.custody,
        })
    }
}
impl<'a> CapturePrefillSourceBootstrap<'a> {
    pub(super) fn from_claim(claim: &'a CaptureStepClaim<'_>) -> Result<Self, WorkingMemoryError> {
        claim.custody.validate()?;
        if claim.phase != CapturePhase::Prefill || claim.prediction != 0 {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        Ok(Self {
            source: claim.source,
            interventions: claim.interventions,
            custody: &claim.custody,
        })
    }
}
