//! Five fixed outer Scope roles issued only by the original active prediction.
use super::*;
use crate::working_memory::{
    InferenceTextStep, funding::RawSpanHostOwner, text_preparation::TextPreparationAuthority,
};

/// Cold concrete provider peaks for the five named outer prediction roles.
/// The population comes from the accepted request, never a caller count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TextPredictionScopeFacts {
    roles: [Option<u64>; 5],
}
impl TextPredictionScopeFacts {
    /// Native providers report actual complete role controls before admission.
    pub const fn new(
        model_execution: Option<u64>,
        sampling: Option<u64>,
        sampling_event: Option<u64>,
        model_validation: Option<u64>,
        token_scalar: Option<u64>,
    ) -> Self {
        Self {
            roles: [
                model_execution,
                sampling,
                sampling_event,
                model_validation,
                token_scalar,
            ],
        }
    }
    /// Number of distinct once-only outer roles in one accepted prediction.
    pub const fn role_count(self) -> usize {
        self.roles.len()
    }

    fn known_sum(self) -> Result<u64, WorkingMemoryError> {
        self.roles.into_iter().try_fold(0u64, |sum, value| {
            sum.checked_add(value.unwrap_or(0))
                .ok_or(WorkingMemoryError::Overflow)
        })
    }
    /// Unknown roles remain unknown even when the request emits no tokens.
    pub fn total_bytes(self) -> Result<Option<u64>, WorkingMemoryError> {
        let known = self.known_sum()?;
        Ok(self.roles.iter().all(Option::is_some).then_some(known))
    }
}

#[derive(Debug)]
enum Role {
    ModelExecution,
    Sampling,
    SamplingEvent,
    ModelValidation,
    TokenScalar,
}

/// Non-Clone Send native custody. No native/source/quote backreference or guard escape.
#[derive(Debug)]
pub struct OriginalPredictionNativeCustody {
    _raw: RawSpanHostOwner,
}
/// Independent last-field custody for the actual closed Rust Recovery allocation.
#[derive(Debug)]
pub struct OriginalPredictionRecoveryCustody {
    _raw: RawSpanHostOwner,
}
/// One original named role, with independently retiring native and Rust owners.
#[derive(Debug)]
pub struct OriginalPredictionScopeRole {
    _role: Role,
    native: OriginalPredictionNativeCustody,
    recovery: OriginalPredictionRecoveryCustody,
}
impl OriginalPredictionScopeRole {
    /// Consume the two prepriced lifetimes; neither result grants native work.
    pub fn into_custody(
        self,
    ) -> (
        OriginalPredictionNativeCustody,
        OriginalPredictionRecoveryCustody,
    ) {
        (self.native, self.recovery)
    }
}
/// The exact five once-only roles of one original active prediction.
#[derive(Debug)]
pub struct OriginalTextPredictionScopeSet {
    roles: [Option<OriginalPredictionScopeRole>; 5],
}
impl OriginalTextPredictionScopeSet {
    fn take(&mut self, ordinal: usize) -> Result<OriginalPredictionScopeRole, WorkingMemoryError> {
        self.roles[ordinal]
            .take()
            .ok_or(WorkingMemoryError::AlreadyStarted)
    }
    /// Take the enclosing model execution role.
    pub fn take_model_execution(
        &mut self,
    ) -> Result<OriginalPredictionScopeRole, WorkingMemoryError> {
        self.take(0)
    }
    /// Take prediction sampling, independently of startup Sampling preparation.
    pub fn take_sampling(&mut self) -> Result<OriginalPredictionScopeRole, WorkingMemoryError> {
        self.take(1)
    }
    /// Take the sampled token/RNG event publication role.
    pub fn take_sampling_event(
        &mut self,
    ) -> Result<OriginalPredictionScopeRole, WorkingMemoryError> {
        self.take(2)
    }
    /// Take the exact model completion's validation role.
    pub fn take_model_validation(
        &mut self,
    ) -> Result<OriginalPredictionScopeRole, WorkingMemoryError> {
        self.take(3)
    }
    /// Move the original token's scalar role into its shared observation owner.
    /// A copied token cannot acquire this role through its retained source owner.
    pub fn take_token_scalar(&mut self) -> Result<OriginalPredictionScopeRole, WorkingMemoryError> {
        self.take(4)
    }
}

/// One original bank. It authenticates actual active steps and never refunds an issue.
#[derive(Debug)]
pub struct OriginalTextPredictionScopes {
    binding: TextControlBinding,
    reservation: WorkingMemoryReservation,
    preparation: Option<Arc<TextPreparationAuthority>>,
    next_attempt: u64,
    // Last, after all bank controls and retained identity.
    controls: OriginalTextControlGuard,
}
impl PreparedTextControlWorkspace {
    /// The two actual prediction roles of a sampling extension. Model, model
    /// validation and token-scalar roles are absent from the resulting bank.
    pub fn with_sampling_prediction_scopes(
        self,
        sampling: Option<u64>,
        event: Option<u64>,
    ) -> Result<Self, WorkingMemoryError> {
        if self.binding.sampling_extension.is_none() {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        self.with_prediction_scopes(TextPredictionScopeFacts::new(
            Some(0),
            sampling,
            event,
            Some(0),
            Some(0),
        ))
    }
    /// Enrich original Q with the five roles for its own accepted output ceiling.
    pub fn with_prediction_scopes(
        mut self,
        facts: TextPredictionScopeFacts,
    ) -> Result<Self, WorkingMemoryError> {
        cold_controls::<(
            Self,
            TextPredictionScopeFacts,
            Result<Self, WorkingMemoryError>,
        )>(self.plan.metadata_funding().as_ref())?;
        if self.binding.prediction_scopes.is_some() {
            return Err(WorkingMemoryError::AlreadyStarted);
        }
        let native = facts.total_bytes()?;
        let neutral = u64::try_from(bank_control_bytes().ok_or(WorkingMemoryError::Overflow)?)
            .map_err(|_| WorkingMemoryError::Overflow)?;
        let issue = u64::try_from(issue_control_bytes().ok_or(WorkingMemoryError::Overflow)?)
            .map_err(|_| WorkingMemoryError::Overflow)?;
        let population = facts
            .known_sum()?
            .checked_add(issue)
            .and_then(|n| {
                n.checked_mul(
                    self.binding
                        .sampling_extension
                        .as_ref()
                        .map_or(self.binding.geometry.max_output_tokens, |origin| {
                            origin.end() - origin.first()
                        }),
                )
            })
            .and_then(|n| n.checked_add(neutral))
            .and_then(|n| n.checked_add(self.binding.facts.admission.unwrap_or(0)))
            .ok_or(WorkingMemoryError::Overflow)?;
        self.binding.facts.admission = self.binding.facts.admission.zip(native).map(|_| population);
        self.binding.prediction_scopes = Some(facts);
        Ok(self)
    }
}
impl OwnedTextSpanWorkspace {
    /// Extract the actual sealed bank once; absent legacy controls grant nothing.
    pub fn take_prediction_scopes(
        &mut self,
    ) -> Result<Option<OriginalTextPredictionScopes>, WorkingMemoryError> {
        let binding = self
            .workspace()
            .control_binding()
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        if binding.prediction_scopes.is_none() {
            return Ok(None);
        }
        if self.prediction_scopes_taken {
            return Err(WorkingMemoryError::AlreadyStarted);
        }
        self.controls.validate_reservation(self.reservation())?;
        let bank = OriginalTextPredictionScopes {
            binding: binding.clone(),
            reservation: self.reservation().clone(),
            preparation: None,
            next_attempt: binding
                .sampling_extension
                .as_ref()
                .map_or(0, |origin| origin.first()),
            controls: self.controls.clone(),
        };
        self.prediction_scopes_taken = true;
        Ok(Some(bank))
    }
}
impl OriginalTextPredictionScopes {
    fn role(&self, kind: Role, original_sampling: bool) -> Option<OriginalPredictionScopeRole> {
        if self.binding.sampling_extension.is_some()
            && !matches!(kind, Role::Sampling | Role::SamplingEvent)
        {
            return None;
        }
        if self.binding.sampling_extension.is_none()
            && !original_sampling
            && matches!(kind, Role::Sampling | Role::SamplingEvent)
        {
            return None;
        }
        Some(OriginalPredictionScopeRole {
            _role: kind,
            native: OriginalPredictionNativeCustody {
                _raw: self.controls.custody.raw().clone(),
            },
            recovery: OriginalPredictionRecoveryCustody {
                _raw: self.controls.custody.raw().clone(),
            },
        })
    }
    /// Issue once using genuine active-step authority. Receipts/numbers cannot call this.
    /// Preparing initial work and Started cached decode use the same exact run.
    /// Historical receipts cannot issue a role, even when cloned from this run.
    /// ```compile_fail
    /// use eredu_runtime::working_memory::{OriginalTextPredictionScopes, InferenceTextStepReceipt};
    /// fn receipt_is_not_a_step(bank: &mut OriginalTextPredictionScopes, receipt: &InferenceTextStepReceipt) {
    ///     let _ = bank.claim(receipt);
    /// }
    /// ```
    pub fn claim(
        &mut self,
        step: &InferenceTextStep,
    ) -> Result<OriginalTextPredictionScopeSet, WorkingMemoryError> {
        if step.request().geometry() != self.binding.geometry {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        self.controls.validate_reservation(&self.reservation)?;
        let (preparation, attempt) = if let Some(origin) = &self.binding.sampling_extension {
            let attempt = origin.validate_step(step)?;
            let (preparation, _) =
                step.original_scope_identity(origin.request().memory_reservation())?;
            (preparation, attempt)
        } else {
            step.original_scope_identity(&self.reservation)?
        };
        if self
            .preparation
            .as_ref()
            .is_some_and(|old| !Arc::ptr_eq(old, &preparation))
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        if attempt != self.next_attempt {
            return Err(WorkingMemoryError::TextStepOrdinalMismatch {
                expected: self.next_attempt,
                actual: attempt,
            });
        }
        if attempt >= self.binding.geometry.max_output_tokens {
            return Err(WorkingMemoryError::TextOutputAllowanceExceeded {
                issued: attempt,
                limit: self.binding.geometry.max_output_tokens,
            });
        }
        let next = attempt.checked_add(1).ok_or(WorkingMemoryError::Overflow)?;
        let original_sampling = step.original_sampling_is_current()?;
        let set = OriginalTextPredictionScopeSet {
            roles: [
                self.role(Role::ModelExecution, original_sampling),
                self.role(Role::Sampling, original_sampling),
                self.role(Role::SamplingEvent, original_sampling),
                self.role(Role::ModelValidation, original_sampling),
                self.role(Role::TokenScalar, original_sampling),
            ],
        };
        self.preparation = Some(preparation);
        self.next_attempt = next;
        Ok(set)
    }
}
fn bank_control_bytes() -> Option<usize> {
    [
        size_of::<OriginalTextPredictionScopes>(),
        size_of::<Option<OriginalTextPredictionScopes>>(),
        size_of::<Result<Option<OriginalTextPredictionScopes>, WorkingMemoryError>>(),
        size_of::<TextPredictionScopeFacts>(),
        size_of::<TextControlBinding>(),
        size_of::<WorkingMemoryReservation>(),
        size_of::<OriginalTextControlGuard>(),
    ]
    .into_iter()
    .try_fold(0usize, usize::checked_add)
}
fn issue_control_bytes() -> Option<usize> {
    [
        crate::working_memory::text_preparation::synchronization_control_bytes()?,
        // Complete array/return and individual role-construction/move controls.
        size_of::<Role>(),
        size_of::<&OriginalTextPredictionScopes>(),
        size_of::<Option<OriginalPredictionScopeRole>>(),
        size_of::<OriginalTextPredictionScopeSet>(),
        size_of::<Option<OriginalTextPredictionScopeSet>>(),
        size_of::<Result<OriginalTextPredictionScopeSet, WorkingMemoryError>>(),
        size_of::<(Arc<TextPreparationAuthority>, u64)>(),
        size_of::<Result<(Arc<TextPreparationAuthority>, u64), WorkingMemoryError>>(),
        size_of::<Result<(), WorkingMemoryError>>(),
    ]
    .into_iter()
    .try_fold(0usize, usize::checked_add)
}
