//! Original R diagnostics and the one consuming accepted sequence bank.
use super::*;
use crate::working_memory::InferenceTextPreparation;
use eredu_core::{GenerationSequencePreparation, RetainedGenerationSequence, TextStepContext};
mod decoder;
mod storage;
use decoder::DecoderBinding;
pub(in crate::working_memory) use decoder::OriginalTokenDomainBinding;
pub use decoder::{
    AggregateGenerationDecoderInput, LoadedGenerationDecoderInput, OriginalGenerationDecoderInput,
    OriginalGenerationDecoderSource,
};
pub(in crate::working_memory) use storage::SequenceExtractionError;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct SequenceBinding {
    pub(super) input: Option<super::token_input::TokenInputBinding>,
    maximum: usize,
    decoder: Option<DecoderBinding>,
    consumer: Option<eredu_core::GenerationSequenceConsumerLayout>,
    terminal_bytes: Option<usize>,
    eos_count: usize,
    // Equality only while the genuine synchronous claim keeps both borrows live.
    // Never dereferenced, exported, or carried into the retained result payload.
    request_address: usize,
    eos_address: usize,
    context: TextStepContext,
    bytes: u64,
}
impl SequenceBinding {
    fn prepare(
        claim: &GenerationSequencePreparation<'_, '_>,
        geometry: InferenceGeometry,
    ) -> Result<Self, WorkingMemoryError> {
        let request = claim.request();
        if u64::try_from(request.max_new_tokens()).ok() != Some(geometry.max_output_tokens)
            || claim.context().attempt() != 0
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let decoder = DecoderBinding::for_claim(claim)?;
        let terminal_bytes = DecoderBinding::terminal_text_bytes(claim)?;
        Ok(Self {
            input: super::token_input::TokenInputBinding::prepare(claim, geometry)?,
            decoder,
            maximum: request.max_new_tokens(),
            consumer: request.consumer_layout().copied(),
            terminal_bytes,
            eos_count: request.eos_token_ids().len(),
            request_address: std::ptr::from_ref(request) as usize,
            eos_address: request.eos_token_ids().as_ptr() as usize,
            context: claim.context().clone(),
            bytes: storage::required_bytes(
                request.max_new_tokens(),
                request.eos_token_ids().len(),
            )?
            .checked_add(
                u64::try_from(
                    request
                        .consumer_layout()
                        .map_or(0, |c| c.retention_peak_bytes()),
                )
                .map_err(|_| WorkingMemoryError::Overflow)?,
            )
            .and_then(|n| n.checked_add(decoder.map_or(0, |d| d.bytes)))
            .and_then(|n| n.checked_add(u64::try_from(terminal_bytes.unwrap_or(0)).ok()?))
            .ok_or(WorkingMemoryError::Overflow)?,
        })
    }
    fn validate(
        &self,
        claim: &GenerationSequencePreparation<'_, '_>,
    ) -> Result<(), WorkingMemoryError> {
        match (&self.input, claim.request().token_input()) {
            (Some(input), Some(_)) => input.validate_claim(claim)?,
            (None, None) => {}
            _ => return Err(WorkingMemoryError::IdentityMismatch),
        }
        if self.decoder != DecoderBinding::for_claim(claim)?
            || self.terminal_bytes != DecoderBinding::terminal_text_bytes(claim)?
            || self.consumer.as_ref() != claim.request().consumer_layout()
            || self.maximum != claim.request().max_new_tokens()
            || self.eos_count != claim.request().eos_token_ids().len()
            || self.request_address != std::ptr::from_ref(claim.request()) as usize
            || self.eos_address != claim.request().eos_token_ids().as_ptr() as usize
            || &self.context != claim.context()
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        Ok(())
    }
}
impl PreparedTextControlWorkspace {
    /// Prepares sequence-only original controls from the actual borrowed claim.
    /// No capture source is invented and no provider or payload is allocated.
    pub fn prepare_sequence(
        claim: &GenerationSequencePreparation<'_, '_>,
        geometry: InferenceGeometry,
        plan: &InferenceSpanWorkspacePlan,
        facts: TextHostControlFacts,
    ) -> Result<Self, WorkingMemoryError> {
        Self::prepare_controls(geometry, plan, facts)?.with_generation_sequence(claim)
    }

    /// Enriches the same capture controls before the original seal; no rebinding.
    pub fn with_generation_sequence(
        mut self,
        claim: &GenerationSequencePreparation<'_, '_>,
    ) -> Result<Self, WorkingMemoryError> {
        if self.binding.sequence.is_some() {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        self.binding.sequence = Some(SequenceBinding::prepare(claim, self.binding.geometry)?);
        Ok(self)
    }
    /// Concrete requested token/EOS layouts and named provider overlaps, or zero.
    pub fn sequence_storage_bytes(&self) -> u64 {
        self.binding
            .sequence
            .as_ref()
            .map_or(0, |sequence| sequence.bytes)
    }
}
/// One original sequence bank removed from the actual promoted span.
///
/// The bank is neither Clone nor an allocation allowance. Its private binding,
/// reservation and custody come only from `OwnedTextSpanWorkspace`; a consuming
/// attempt, including rejection or unwind, cannot recreate the bank. It performs
/// no native work and does not establish completion or full facade accounting.
///
/// ```compile_fail
/// fn duplicate(bank: eredu_runtime::working_memory::OriginalGenerationSequenceBank) {
///     let another = bank.clone();
/// }
/// ```
#[derive(Debug)]
#[must_use]
pub struct OriginalGenerationSequenceBank {
    #[cfg(test)]
    fail_terminal_reserve: bool,
    decoder: Option<OriginalGenerationDecoderSource>,
    binding: SequenceBinding,
    reservation: WorkingMemoryReservation,
    // Last: exact binding and reservation metadata retire before their custody.
    controls: OriginalTextControlGuard,
}
impl OwnedTextSpanWorkspace {
    /// Removes the one sealed sequence bank before any fallible validation.
    ///
    /// Absent or previously taken banks return None without allocating or cloning
    /// custody. This does not validate a caller or create a replacement bank.
    /// The returned bank owns every subsequent consuming attempt.
    pub fn take_generation_sequence_bank(&mut self) -> Option<OriginalGenerationSequenceBank> {
        if self.sequence_taken {
            return None;
        }
        let binding = self.workspace().control_binding()?.sequence.as_ref()?;
        let binding = binding.clone();
        self.sequence_taken = true;
        Some(OriginalGenerationSequenceBank {
            #[cfg(test)]
            fail_terminal_reserve: false,
            decoder: None,
            binding,
            reservation: self.reservation().clone(),
            controls: self.controls.clone(),
        })
    }
}
impl OriginalGenerationSequenceBank {
    #[cfg(test)]
    pub(in crate::working_memory) fn fail_terminal_reservation(mut self) -> Self {
        self.fail_terminal_reserve = true;
        self
    }

    /// Takes the unique pre-candidate source from its caller-owned staging slot
    /// only after matching this accepted bank. A mismatch consumes this bank
    /// into one owning core failure but leaves the unaccepted source untouched;
    /// its bytes cannot be charged to an unrelated original reservation.
    pub fn with_decoder_source(
        mut self,
        source: &mut Option<OriginalGenerationDecoderSource>,
    ) -> Result<Self, eredu_core::BackendFailure> {
        if self.decoder.is_some()
            || source.is_none()
            || source.as_ref().map(|source| source.binding) != self.binding.decoder
            || source
                .as_ref()
                .is_some_and(|source| source.validate_pool(&self.reservation.0.pool).is_err())
        {
            return Err(SequenceExtractionError::memory(
                WorkingMemoryError::IdentityMismatch,
                self,
            )
            .into_failure());
        }
        self.decoder = source.take();
        Ok(self)
    }
    /// Consumes the exact genuine claim after run binding and before Prompt.
    ///
    /// The returned sequence remains dormant. One failure retains the consumed
    /// bank and any partial provider through core's concrete source retirement;
    /// no native Box or replacement claim is created. All locks end before
    /// allocation, source conversion or payload destruction. A rejected claim
    /// terminally spends this bank and cannot grant a later retry.
    pub fn prepare(
        mut self,
        preparation: &InferenceTextPreparation,
        claim: GenerationSequencePreparation<'_, '_>,
    ) -> Result<RetainedGenerationSequence, eredu_core::BackendFailure> {
        let validate = || {
            self.binding.validate(&claim)?;
            if self.decoder.as_ref().map(|d| d.binding) != self.binding.decoder {
                return Err(WorkingMemoryError::IdentityMismatch);
            }
            let reservation = preparation.request().memory_reservation();
            if !reservation.0.same(&self.reservation.0)
                || preparation.request().geometry() != self.reservation.0.geometry
            {
                return Err(WorkingMemoryError::IdentityMismatch);
            }
            self.controls.validate_reservation(&self.reservation)?;
            preparation.claim_generation_sequence(claim.context())
        };
        if let Err(cause) = validate() {
            return Err(SequenceExtractionError::memory(cause, self).into_failure());
        }
        let sequence = match storage::construct(claim.request(), &mut self) {
            Ok(sequence) => sequence,
            Err(error) => return Err(error.with_bank(self).into_failure()),
        };
        if let Err(cause) = preparation.finish_generation_sequence() {
            return Err(SequenceExtractionError::prepared(cause, sequence, self).into_failure());
        }
        Ok(sequence)
    }
}
