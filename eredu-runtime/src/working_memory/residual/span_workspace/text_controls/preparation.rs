//! Exactly two original preparation allocation roles, independent of output slots.
use super::*;
use crate::working_memory::{
    funding::RawSpanHostOwner, InferencePreparationStage, InferenceSamplerCompletion,
    InferenceTextPreparation,
};
use eredu_core::TextGenerationConfig;

/// Cold provider facts for the actual Prompt and Sampling Scope/node controls.
/// These are neither native completion nor an allocation grant. There is no
/// caller-supplied role count and no after-admission enlargement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TextPreparationScopeFacts {
    prompt: Option<u64>,
    sampling: Option<u64>,
}
impl TextPreparationScopeFacts {
    /// Describe each named provider peak; None remains an unknown original bound.
    pub const fn new(prompt: Option<u64>, sampling: Option<u64>) -> Self {
        Self { prompt, sampling }
    }
    /// Checked total; known partial terms cannot hide arithmetic overflow.
    pub fn total_bytes(self) -> Result<Option<u64>, WorkingMemoryError> {
        let sum = self
            .prompt
            .unwrap_or(0)
            .checked_add(self.sampling.unwrap_or(0))
            .ok_or(WorkingMemoryError::Overflow)?;
        Ok(self.prompt.zip(self.sampling).map(|_| sum))
    }
}

#[derive(Debug)]
enum PreparationScopeRole {
    Prompt,
    Sampling,
}
/// Opaque, non-Clone Send custody for one original preparation allocation role.
/// Only the accepted two-role bank constructs it. It exports neither a raw
/// guard, amount, scope, source tree nor native submission permission.
#[derive(Debug)]
pub struct OriginalPreparationScopeCustody {
    _role: PreparationScopeRole,
    _raw: RawSpanHostOwner,
}
impl OriginalPreparationScopeCustody {
    pub(super) fn sampling(controls: &OriginalTextControlGuard) -> Self {
        Self { _role: PreparationScopeRole::Sampling, _raw: controls.custody.raw().clone() }
    }
}

/// One non-refillable pair extracted from the accepted original span. A failed
/// native begin retains its same claimed stage and capsule outside this bank.
#[derive(Debug)]
pub struct OriginalTextPreparationScopes {
    binding: TextControlBinding,
    reservation: WorkingMemoryReservation,
    prompt_claimed: bool,
    sampling_claimed: bool,
    controls: OriginalTextControlGuard,
}
impl PreparedTextControlWorkspace {
    /// Price the optional single reseed role of a pending sampling extension.
    /// No prompt construction role is issued for this origin.
    pub fn with_sampling_reseed_scope(self, bytes: Option<u64>) -> Result<Self, WorkingMemoryError> {
        if self.binding.sampling_extension.is_none() { return Err(WorkingMemoryError::IdentityMismatch); }
        self.with_preparation_scopes(TextPreparationScopeFacts::new(Some(0), bytes))
    }
    /// Seal this fixed pair into original Q before reservation. The existing
    /// identity and exact plan/source association remain part of the binding.
    pub fn with_preparation_scopes(
        mut self,
        facts: TextPreparationScopeFacts,
    ) -> Result<Self, WorkingMemoryError> {
        cold_controls::<(Self, TextPreparationScopeFacts, Result<Self, WorkingMemoryError>)>(self.plan.metadata_funding().as_ref())?;
        if self.binding.preparation_scopes.is_some() {
            return Err(WorkingMemoryError::PreparationAlreadyStarted);
        }
        let native = facts.total_bytes()?;
        let controls = u64::try_from(control_bytes().ok_or(WorkingMemoryError::Overflow)?)
            .map_err(|_| WorkingMemoryError::Overflow)?;
        // Check every known partial term even if another required fact is unknown.
        let total = facts
            .prompt
            .unwrap_or(0)
            .checked_add(facts.sampling.unwrap_or(0))
            .and_then(|n| n.checked_add(controls))
            .and_then(|n| n.checked_add(self.binding.facts.admission.unwrap_or(0)))
            .ok_or(WorkingMemoryError::Overflow)?;
        self.binding.facts.admission = self.binding.facts.admission.zip(native).map(|_| total);
        self.binding.preparation_scopes = Some(facts);
        Ok(self)
    }
}
impl OwnedTextSpanWorkspace {
    /// Extract once; neither a diagnostic clone nor a guard can replace it.
    pub fn take_preparation_scopes(
        &mut self,
    ) -> Result<Option<OriginalTextPreparationScopes>, WorkingMemoryError> {
        let binding = self
            .workspace()
            .control_binding()
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        if binding.preparation_scopes.is_none() {
            return Ok(None);
        }
        if self.preparation_scopes_taken {
            return Err(WorkingMemoryError::PreparationAlreadyStarted);
        }
        self.controls.validate_reservation(self.reservation())?;
        let bank = OriginalTextPreparationScopes {
            binding: binding.clone(),
            reservation: self.reservation().clone(),
            prompt_claimed: false,
            sampling_claimed: false,
            controls: self.controls.clone(),
        };
        self.preparation_scopes_taken = true;
        Ok(Some(bank))
    }
}
impl OriginalTextPreparationScopes {
    /// Recheck this exact original preparation without consuming a role.
    /// Native retry calls this before extracting its retained pending owner.
    pub fn validate_preparation(
        &self,
        preparation: &InferenceTextPreparation,
    ) -> Result<(), WorkingMemoryError> {
        let reservation = preparation
            .request()
            .memory_reservation()
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        if !reservation.0.same(&self.reservation.0)
            || preparation.request().geometry() != self.binding.geometry
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        preparation.validate_scope_preparation()?;
        self.controls.validate_reservation(&self.reservation)
    }
    /// Claim Prompt once using its exact bound original preparation authority.
    pub fn claim_prompt(
        &mut self,
        preparation: &InferenceTextPreparation,
    ) -> Result<(InferencePreparationStage, OriginalPreparationScopeCustody), WorkingMemoryError>
    {
        self.validate_preparation(preparation)?;
        if self.prompt_claimed {
            return Err(WorkingMemoryError::PreparationAlreadyStarted);
        }
        let custody = OriginalPreparationScopeCustody {
            _role: PreparationScopeRole::Prompt,
            _raw: self.controls.custody.raw().clone(),
        };
        let stage = preparation.claim_prompt()?;
        self.prompt_claimed = true;
        Ok((stage, custody))
    }
    /// Claim Sampling once without replacing its original configuration.
    pub fn claim_sampling(
        &mut self,
        preparation: &InferenceTextPreparation,
        config: TextGenerationConfig,
    ) -> Result<(InferencePreparationStage, OriginalPreparationScopeCustody), WorkingMemoryError>
    {
        self.validate_preparation(preparation)?;
        if self.sampling_claimed {
            return Err(WorkingMemoryError::PreparationAlreadyStarted);
        }
        let custody = OriginalPreparationScopeCustody::sampling(&self.controls);
        let stage = preparation.claim_sampling(config)?;
        self.sampling_claimed = true;
        Ok((stage, custody))
    }
    /// Recheck the same still-claimed prompt before retrying an unconsumed native begin.
    pub fn validate_prompt(
        &self,
        preparation: &InferenceTextPreparation,
        stage: &InferencePreparationStage,
    ) -> Result<(), WorkingMemoryError> {
        self.validate_preparation(preparation)?;
        if !self.prompt_claimed {
            return Err(WorkingMemoryError::PreparationNotReady);
        }
        stage.validate_scope_claim(preparation, true)
    }
    /// Recheck sampling before its single host sampler construction.
    pub fn validate_sampling_stage(
        &self,
        preparation: &InferenceTextPreparation,
        stage: &InferencePreparationStage,
    ) -> Result<(), WorkingMemoryError> {
        self.validate_preparation(preparation)?;
        if !self.sampling_claimed {
            return Err(WorkingMemoryError::PreparationNotReady);
        }
        stage.validate_scope_claim(preparation, false)
    }
    /// Recheck the original finish-only sampler remainder; never reconstruct history.
    pub fn validate_sampling(
        &self,
        preparation: &InferenceTextPreparation,
        stage: &InferenceSamplerCompletion,
    ) -> Result<(), WorkingMemoryError> {
        self.validate_preparation(preparation)?;
        if !self.sampling_claimed {
            return Err(WorkingMemoryError::PreparationNotReady);
        }
        stage.validate_scope_claim(preparation)
    }
}
fn control_bytes() -> Option<usize> {
    [
        crate::working_memory::text_preparation::synchronization_control_bytes()?,
        size_of::<OriginalTextPreparationScopes>(),
        size_of::<Option<OriginalTextPreparationScopes>>(),
        size_of::<Result<Option<OriginalTextPreparationScopes>, WorkingMemoryError>>(),
        size_of::<TextPreparationScopeFacts>(),
        size_of::<TextControlBinding>(),
        size_of::<WorkingMemoryReservation>(),
        size_of::<OriginalTextControlGuard>(),
        // Each of the two roles has its own claim, returned pair and caller move.
        2usize.checked_mul(size_of::<OriginalPreparationScopeCustody>())?,
        2usize.checked_mul(size_of::<(
            InferencePreparationStage,
            OriginalPreparationScopeCustody,
        )>())?,
        2usize.checked_mul(size_of::<
            Result<
                (InferencePreparationStage, OriginalPreparationScopeCustody),
                WorkingMemoryError,
            >,
        >())?,
    ]
    .into_iter()
    .try_fold(0usize, usize::checked_add)
}
