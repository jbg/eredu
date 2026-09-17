//! One source-bound original host input allocation, separate from output R.
use super::*;
use crate::working_memory::{InferenceRequest, InferenceTextPreparation};
use eredu_core::{
    BackendFailure, BackendFailureKind, GenerationSequencePreparation, TokenIdsInputPlan,
};
use std::{alloc::Layout, collections::TryReserveError, mem::size_of};

/// Exact requested destination and actual closed construction/error controls.
/// This diagnostic allocates nothing and cannot construct a funded owner.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OriginalTokenInputLayout {
    count: usize,
    bytes: u64,
}
impl OriginalTokenInputLayout {
    /// Derives all facts from the actual positive immutable source.
    pub fn prepare(plan: &TokenIdsInputPlan<'_>) -> Result<Self, WorkingMemoryError> {
        let count = plan.tokens().len();
        let payload = Layout::array::<u32>(count)
            .map_err(|_| WorkingMemoryError::Overflow)?
            .size();
        let error = BackendFailure::source_retention_peak_bytes::<OriginalTokenInputFailure>()
            .ok_or(WorkingMemoryError::Overflow)?;
        let fixed = [
            size_of::<TokenIdsInputPlan<'static>>(),
            size_of::<Self>(),
            size_of::<TokenInputBinding>(),
            size_of::<Result<Option<TokenInputBinding>, WorkingMemoryError>>(),
            size_of::<Result<OriginalTokenInputLayout, WorkingMemoryError>>(),
            size_of::<OriginalTokenInputBank>(),
            size_of::<Option<OriginalTokenInputBank>>(),
            size_of::<OwnedPromptTokenIds>(),
            size_of::<Option<OwnedPromptTokenIds>>(),
            size_of::<Result<OwnedPromptTokenIds, BackendFailure>>(),
            size_of::<Vec<u32>>(),
            size_of::<TryReserveError>(),
            size_of::<Result<(), WorkingMemoryError>>(),
            error,
        ]
        .into_iter()
        .try_fold(payload, usize::checked_add)
        .ok_or(WorkingMemoryError::Overflow)?;
        Ok(Self {
            count,
            bytes: u64::try_from(fixed).map_err(|_| WorkingMemoryError::Overflow)?,
        })
    }
    /// Exact source token count, independent of caller spare capacity.
    pub const fn token_count(self) -> usize {
        self.count
    }
    /// Original protected input contribution, including its actual controls.
    pub const fn protected_bytes(self) -> u64 {
        self.bytes
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct TokenInputBinding {
    layout: OriginalTokenInputLayout,
    address: usize,
    request: usize,
    context: eredu_core::TextStepContext,
}
impl TokenInputBinding {
    pub(super) fn prepare(
        claim: &GenerationSequencePreparation<'_, '_>,
        geometry: InferenceGeometry,
    ) -> Result<Option<Self>, WorkingMemoryError> {
        let Some(plan) = claim.request().token_input() else {
            return Ok(None);
        };
        let layout = OriginalTokenInputLayout::prepare(plan)?;
        if geometry.batch_size != 1
            || geometry.input_positions != layout.count as u64
            || claim.context().attempt() != 0
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        Ok(Some(Self {
            layout,
            address: plan.tokens().as_ptr() as usize,
            request: std::ptr::from_ref(claim.request()) as usize,
            context: claim.context().clone(),
        }))
    }
    pub(super) fn bytes(&self) -> u64 {
        self.layout.bytes
    }
    pub(super) fn validate_claim(
        &self,
        claim: &GenerationSequencePreparation<'_, '_>,
    ) -> Result<(), WorkingMemoryError> {
        let Some(plan) = claim.request().token_input() else {
            return Err(WorkingMemoryError::IdentityMismatch);
        };
        if self.layout != OriginalTokenInputLayout::prepare(plan)?
            || self.address != plan.tokens().as_ptr() as usize
            || self.request != std::ptr::from_ref(claim.request()) as usize
            || &self.context != claim.context()
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        Ok(())
    }
    pub(super) fn validate(
        &self,
        claim: &GenerationSequencePreparation<'_, '_>,
        geometry: InferenceGeometry,
    ) -> Result<(), WorkingMemoryError> {
        if Self::prepare(claim, geometry)?.as_ref() != Some(self) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        Ok(())
    }
}

/// One non-Clone bank extracted only from the actual accepted original span.
#[derive(Debug)]
pub struct OriginalTokenInputBank {
    binding: TokenInputBinding,
    reservation: WorkingMemoryReservation,
    controls: OriginalTextControlGuard,
}
impl OwnedTextSpanWorkspace {
    /// Takes before fallible caller validation. Absence/replay clones no custody.
    pub fn take_token_input_bank(&mut self) -> Option<OriginalTokenInputBank> {
        if self.token_input_taken {
            return None;
        }
        let binding = self
            .workspace()
            .control_binding()?
            .sequence
            .as_ref()?
            .input
            .as_ref()?
            .clone();
        self.token_input_taken = true;
        Some(OriginalTokenInputBank {
            binding,
            reservation: self.reservation().clone(),
            controls: self.controls.clone(),
        })
    }
}
impl OriginalTokenInputBank {
    /// Borrows the exact original claim without consuming this bank, allocating,
    /// or cloning custody. Native callers use this before changing their slot.
    /// Construction repeats the check after the slot loan has ended.
    pub fn validate_claim(
        &self,
        preparation: &InferenceTextPreparation,
        claim: &GenerationSequencePreparation<'_, '_>,
    ) -> Result<(), WorkingMemoryError> {
        self.binding
            .validate(claim, preparation.request().geometry())?;
        let actual = preparation
            .request()
            .memory_reservation()
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        if !actual.0.same(&self.reservation.0) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        self.controls.validate_reservation(&self.reservation)?;
        preparation.validate_token_input_context(claim.context())
    }
    /// Allocates and fills once while this genuine synchronous source borrow is
    /// alive. Failure owns the actual construction and original custody.
    pub fn construct(
        self,
        preparation: &InferenceTextPreparation,
        claim: &GenerationSequencePreparation<'_, '_>,
    ) -> Result<OwnedPromptTokenIds, BackendFailure> {
        if let Err(cause) = self.validate_claim(preparation, claim) {
            return Err(OriginalTokenInputFailure::bank(cause, self));
        }
        // Move only: temporary source addresses do not enter the owned payload.
        let Self {
            binding,
            reservation,
            controls,
        } = self;
        let count = binding.layout.count;
        drop(binding);
        let mut owned = OwnedPromptTokenIds {
            tokens: Vec::new(),
            request: preparation.request().clone(),
            reservation,
            controls,
        };
        #[cfg(test)]
        let count = tests::reserve_count(count);
        if let Err(cause) = owned.tokens.try_reserve_exact(count) {
            return Err(OriginalTokenInputFailure::allocation(cause, owned));
        }
        owned.tokens.extend_from_slice(
            claim
                .request()
                .token_input()
                .expect("validated input")
                .tokens(),
        );
        #[cfg(test)]
        tests::after_fill(&owned);
        Ok(owned)
    }
}

/// Sole immutable host destination. No raw Vec, Clone, Weak or source swap.
///
/// ```compile_fail
/// fn duplicate(input: eredu_runtime::working_memory::OwnedPromptTokenIds) {
///     let other = input.clone();
/// }
/// ```
/// Payload precedes original custody; its native consumer only borrows the IDs.
#[derive(Debug)]
pub struct OwnedPromptTokenIds {
    tokens: Vec<u32>,
    request: InferenceRequest,
    reservation: WorkingMemoryReservation,
    controls: OriginalTextControlGuard,
}
impl OwnedPromptTokenIds {
    /// Exact immutable input, valid while this owner is borrowed.
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }
    /// Requested actual buffer capacity; no allocator/RSS claim.
    pub fn capacity_bytes(&self) -> u64 {
        self.tokens.capacity() as u64 * 4
    }
    /// Checks the same original preparation and source/account health.
    pub fn validate(
        &self,
        preparation: &InferenceTextPreparation,
    ) -> Result<(), WorkingMemoryError> {
        self.request.validate_same_request(preparation.request())?;
        self.controls.validate_reservation(&self.reservation)
    }
}
#[derive(Debug)]
enum FailureOwner {
    Bank(OriginalTokenInputBank),
    Input(OwnedPromptTokenIds),
}
#[derive(Debug, thiserror::Error)]
enum Cause {
    #[error("{0}")]
    Memory(#[source] WorkingMemoryError),
    #[error("{0}")]
    Allocation(#[source] TryReserveError),
}
/// Closed owning error; its actual buffer/bank and original account retire with
/// core's concrete source allocation. No consuming extraction or retry exists.
#[derive(Debug, thiserror::Error)]
#[error("original token input: {cause}")]
pub struct OriginalTokenInputFailure {
    #[source]
    cause: Cause,
    owner: FailureOwner,
}
impl OriginalTokenInputFailure {
    fn bank(cause: WorkingMemoryError, bank: OriginalTokenInputBank) -> BackendFailure {
        BackendFailure::new(
            BackendFailureKind::InvalidSession,
            Self {
                cause: Cause::Memory(cause),
                owner: FailureOwner::Bank(bank),
            },
        )
    }
    fn allocation(cause: TryReserveError, input: OwnedPromptTokenIds) -> BackendFailure {
        BackendFailure::new(
            BackendFailureKind::ResourceExhausted,
            Self {
                cause: Cause::Allocation(cause),
                owner: FailureOwner::Input(input),
            },
        )
    }
    /// Actual partial buffer, if allocation began; borrowed under its custody.
    pub fn partial_tokens(&self) -> Option<&[u32]> {
        match &self.owner {
            FailureOwner::Input(input) => Some(input.tokens()),
            FailureOwner::Bank(bank) => {
                let _ = &bank.binding;
                None
            }
        }
    }
}
#[cfg(test)]
pub(in crate::working_memory) mod tests {
    use super::*;
    use std::cell::Cell;
    thread_local! { static FAULT: Cell<u8> = const { Cell::new(0) }; }
    // Private, one-shot capacity overflow / post-fill unwind, not allocator OOM.
    pub(in crate::working_memory) fn fault<T>(mode: u8, f: impl FnOnce() -> T) -> T {
        struct Reset;
        impl Drop for Reset {
            fn drop(&mut self) {
                FAULT.with(|v| v.set(0));
            }
        }
        FAULT.with(|v| assert_eq!(v.replace(mode), 0));
        let _reset = Reset;
        f()
    }
    pub(super) fn reserve_count(count: usize) -> usize {
        FAULT.with(|v| {
            if v.get() == 1 {
                v.set(0);
                usize::MAX
            } else {
                count
            }
        })
    }
    pub(super) fn after_fill(_: &OwnedPromptTokenIds) {
        if FAULT.with(|v| {
            v.get() == 2 && {
                v.set(0);
                true
            }
        }) {
            panic!("original input after real fill");
        }
    }
}
