//! Original input slot. No source pointer survives its consuming construction.
use super::*;
use eredu_core::{GenerationSequencePreparation, TokenInputRejection as Rejection};
use eredu_runtime::working_memory::{
    OriginalTokenInputBank, OwnedPromptTokenIds, OwnedTextSpanWorkspace,
};
use std::mem::size_of;

#[derive(Debug)]
enum Slot {
    Bank(OriginalTokenInputBank),
    Building,
    Ready(OwnedPromptTokenIds),
    Spent,
}
#[derive(Debug)]
pub(in crate::composition::mlx::session::model_session) struct InputQuotation {
    slot: RefCell<Slot>,
}
struct Checkout<'a>(&'a RefCell<Slot>);
impl Drop for Checkout<'_> {
    fn drop(&mut self) {
        if let Ok(mut slot) = self.0.try_borrow_mut() {
            if matches!(*slot, Slot::Building) {
                *slot = Slot::Spent;
            }
        }
    }
}
#[derive(Debug, thiserror::Error)]
#[error("original token input installation: {cause}")]
struct InstallationFailure {
    #[source]
    cause: Rejection,
    _input: OwnedPromptTokenIds,
}
impl InputQuotation {
    pub(super) fn from_span(span: &mut OwnedTextSpanWorkspace) -> Option<Self> {
        span.take_token_input_bank().map(|bank| Self {
            slot: RefCell::new(Slot::Bank(bank)),
        })
    }
    pub(super) fn prepare(
        &self,
        preparation: &InferenceTextPreparation,
        claim: &GenerationSequencePreparation<'_, '_>,
    ) -> Result<(), BackendFailure> {
        let previous = {
            let mut slot = self
                .slot
                .try_borrow_mut()
                .map_err(|_| Rejection::Busy.into_backend_failure())?;
            match &*slot {
                Slot::Bank(bank) => bank
                    .validate_claim(preparation, claim)
                    .map_err(|_| Rejection::IdentityMismatch.into_backend_failure())?,
                Slot::Building => return Err(Rejection::Busy.into_backend_failure()),
                _ => return Err(Rejection::Unavailable.into_backend_failure()),
            }
            std::mem::replace(&mut *slot, Slot::Building)
        };
        let checkout = Checkout(&self.slot);
        let Slot::Bank(bank) = previous else {
            unreachable!()
        };
        let input = bank.construct(preparation, claim)?;
        match self.slot.try_borrow_mut() {
            Ok(mut slot) => {
                *slot = Slot::Ready(input);
            }
            Err(_) => {
                return Err(BackendFailure::new(
                    BackendFailureKind::Busy,
                    InstallationFailure {
                        cause: Rejection::Busy,
                        _input: input,
                    },
                ));
            }
        }
        drop(checkout);
        Ok(())
    }
    #[cfg(test)]
    pub(super) fn is_unclaimed_for_test(&self) -> bool {
        matches!(*self.slot.borrow(), Slot::Bank(_))
    }
    pub(in crate::composition::mlx::session::model_session) fn take(
        &self,
    ) -> Result<OwnedPromptTokenIds, BackendFailure> {
        let previous = {
            let mut slot = self
                .slot
                .try_borrow_mut()
                .map_err(|_| Rejection::Busy.into_backend_failure())?;
            match &*slot {
                Slot::Ready(_) => {}
                Slot::Building => return Err(Rejection::Busy.into_backend_failure()),
                _ => return Err(Rejection::Unavailable.into_backend_failure()),
            }
            std::mem::replace(&mut *slot, Slot::Spent)
        };
        let Slot::Ready(input) = previous else {
            unreachable!()
        };
        Ok(input)
    }
}

/// Existing numerical/identity worker borrows either source while its owner lives.
#[derive(Debug)]
pub(in crate::composition::mlx::session::model_session) enum PromptIds {
    Legacy(Vec<u32>),
    Original(OwnedPromptTokenIds),
}
impl PromptIds {
    pub(in crate::composition::mlx::session::model_session) fn tokens(&self) -> &[u32] {
        match self {
            Self::Legacy(ids) => ids,
            Self::Original(input) => input.tokens(),
        }
    }
    pub(super) fn validate(
        &self,
        quote: &TextExecutionQuote,
        preparation: &InferenceTextPreparation,
    ) -> Result<(), Error> {
        match self {
            Self::Legacy(ids) => quote.validate_prompt(ids),
            Self::Original(input) => {
                input.validate(preparation).map_err(memory)?;
                if input.tokens().len() as u64 != quote.request.geometry().input_positions
                    || input.capacity_bytes() != quote.source_capacity_bytes
                {
                    return Err(memory(WorkingMemoryError::IdentityMismatch));
                }
                Ok(())
            }
        }
    }
}
/// Stored slot lies inside the quote's A. Name only additional construction,
/// transfer, error and cleanup representations, once in original A.
pub(super) fn control_bytes() -> Option<u64> {
    let error = BackendFailure::source_retention_peak_bytes::<InstallationFailure>()?;
    [
        super::preparation::original_input_borrow_control_bytes()?,
        size_of::<PromptIds>(),
        size_of::<Option<PromptIds>>(),
        size_of::<Result<MlxModelInput, BackendFailure>>(),
        size_of::<InputQuotation>(),
        size_of::<Slot>(),
        size_of::<Checkout<'_>>(),
        size_of::<std::cell::RefMut<'_, Slot>>(),
        size_of::<Result<(), WorkingMemoryError>>(),
        size_of::<Result<OwnedPromptTokenIds, BackendFailure>>(),
        size_of::<OwnedPromptTokenIds>(),
        size_of::<Result<(), BackendFailure>>(),
        error,
    ]
    .into_iter()
    .try_fold(0usize, usize::checked_add)
    .and_then(|n| u64::try_from(n).ok())
}

impl TextExecutionQuote {
    pub(in crate::composition::mlx::session::model_session) fn validate_original_prompt_preflight(
        &self,
        backend: &MlxBackend<'_>,
        request: &InferenceTextPreparation,
    ) -> Result<(), Rejection> {
        if self.session.get()
            || !self.context_pool.same_domain(backend.memory_pool())
            || self
                .request
                .validate_same_request(request.request())
                .is_err()
            || self.opening.require_sealed().is_err()
        {
            return Err(Rejection::IdentityMismatch);
        }
        Ok(())
    }
}
