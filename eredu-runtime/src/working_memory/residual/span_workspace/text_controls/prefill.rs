//! Finite prefill controls from the same original text comparison and active run.
use super::*;
use crate::{
    prefill::{PrefillControlPlan, PrefillControlRole},
    working_memory::{InferenceRequest, InferenceTextStep, funding::RawSpanHostOwner},
};

/// Concrete provider controls for the existing seven local role kinds. These
/// facts are diagnostics, never a role, source pin or submission authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TextPrefillScopeFacts {
    plan: PrefillControlPlan,
    // Source, pre-boundary, outer, transaction, settlement, post-boundary, index.
    roles: [Option<u64>; 7],
    bank: Option<u64>,
    operations: Option<u64>,
    source_constructions: Option<HostSourceConstructionFacts>,
    host_destinations: Option<HostDestinationFacts>,
    output_sources: Option<HostSourceConstructionFacts>,
    root_capacity: u64,
    graph_bytes_per_completion: u64,
}
impl TextPrefillScopeFacts {
    /// The provider supplies exact concrete controls and the source/selection
    /// root capacity before Q comparison. An original request always retains
    /// its inner input transaction. Auxiliary/distributed profiles are separate.
    pub fn new(
        geometry: InferenceGeometry,
        roles: [Option<u64>; 7],
        bank: Option<u64>,
        root_capacity: u64,
        graph_bytes_per_completion: u64,
    ) -> Result<Self, WorkingMemoryError> {
        let value = Self {
            plan: PrefillControlPlan::new(geometry, true)?,
            roles,
            bank,
            operations: Some(0),
            source_constructions: None,
            host_destinations: None,
            output_sources: None,
            root_capacity,
            graph_bytes_per_completion,
        };
        value.total_bytes()?;
        value.graph_bytes()?;
        Ok(value)
    }
    /// Add source/selection-derived request-wide operation storage before Q.
    /// Unknown contributions remain unknown; these scalar diagnostics alone
    /// cannot issue a slot or certify a complete original request.
    pub fn with_operation_controls(
        mut self,
        bytes: Option<u64>,
    ) -> Result<Self, WorkingMemoryError> {
        self.operations = bytes;
        self.total_bytes()?;
        Ok(self)
    }
    /// Attach the separately funded source component. Its protected bytes enter
    /// Work through the host binding, not these role/operation bytes a second time.
    pub fn with_source_constructions(mut self, facts: Option<HostSourceConstructionFacts>) -> Self {
        self.source_constructions = facts;
        self
    }
    /// Attach the complete destination/source host component before admission.
    /// Its source sub-facts must match the separately retained source facts.
    pub fn with_host_destinations(mut self, facts: Option<HostDestinationFacts>) -> Self {
        self.host_destinations = facts;
        self
    }
    /// Independent finite immutable-output constructors retained across prefill
    /// and decode. The host binding prices this component exactly once.
    pub fn with_output_source_constructions(mut self, facts: HostSourceConstructionFacts) -> Self {
        self.output_sources = Some(facts);
        self
    }
    /// Complete disjoint host-destination contribution from the selected sources.
    pub const fn host_destination_facts(self) -> Option<HostDestinationFacts> {
        self.host_destinations
    }
    /// Source storage selected before acceptance; absence grants no constructor.
    pub const fn source_construction_facts(self) -> Option<HostSourceConstructionFacts> {
        self.source_constructions
    }
    /// Exact accepted operation component, separate from the seven span roles.
    pub const fn operation_control_bytes(self) -> Option<u64> {
        self.operations
    }
    /// Exact admitted local role geometry; no caller-selected issue count.
    pub const fn plan(self) -> PrefillControlPlan {
        self.plan
    }
    /// Maximum actual native-array slots at each completion, including future
    /// state/source fields. This is a handle bound, not numerical backing bytes.
    pub const fn root_capacity(self) -> u64 {
        self.root_capacity
    }
    /// All original collector buffers coexist; no scratch reuse or refill.
    /// The provider includes both exact native Graph block extents per site.
    pub fn graph_bytes(self) -> Result<u64, WorkingMemoryError> {
        self.graph_bytes_per_completion
            .checked_mul(self.plan.span_count())
            .ok_or(WorkingMemoryError::Overflow)
    }
    /// Residual physical arena ceiling after every collector block is reserved.
    /// This is not a bound on the other producers' required occupancy or a claim
    /// that their complete graph fits. They remain constrained by the same arena.
    pub fn graph_residual_ceiling(self, capacity: u64) -> Result<u64, WorkingMemoryError> {
        capacity
            .checked_sub(self.graph_bytes()?)
            .ok_or(WorkingMemoryError::UnknownBound)
    }
    fn populations(self) -> [u64; 7] {
        let n = self.plan.span_count();
        [
            u64::from(n != 0),
            n,
            n,
            n,
            n,
            n,
            u64::from(self.plan.geometry().output != eredu_core::OutputDemand::StateOnly),
        ]
    }
    fn known_bytes(self) -> Result<u64, WorkingMemoryError> {
        self.roles.into_iter().zip(self.populations()).try_fold(
            self.bank
                .unwrap_or(0)
                .checked_add(self.operations.unwrap_or(0))
                .ok_or(WorkingMemoryError::Overflow)?,
            |sum, (bytes, population)| {
                bytes
                    .unwrap_or(0)
                    .checked_mul(population)
                    .and_then(|n| n.checked_add(sum))
                    .ok_or(WorkingMemoryError::Overflow)
            },
        )
    }
    /// Known partial arithmetic is checked even when another role is unknown.
    pub fn total_bytes(self) -> Result<Option<u64>, WorkingMemoryError> {
        let total = self.known_bytes()?;
        let known = self.bank.is_some()
            && self.operations.is_some()
            && self
                .roles
                .into_iter()
                .zip(self.populations())
                .all(|(bytes, population)| population == 0 || bytes.is_some());
        Ok(known.then_some(total))
    }
}

/// Native Scope's last-field original custody; never grants a submission.
#[derive(Debug)]
pub struct OriginalPrefillNativeCustody {
    _raw: RawSpanHostOwner,
}
/// Rust recovery node's independent last-field original custody.
#[derive(Debug)]
pub struct OriginalPrefillRecoveryCustody {
    _raw: RawSpanHostOwner,
}
/// Completion-root payload and its control allocation retire before this owner.
#[derive(Debug)]
pub struct OriginalPrefillRootCustody {
    _raw: RawSpanHostOwner,
}
/// Private native weak projection's control allocation outlives the payload if
/// necessary. Its actual weak header must retire before this independent owner.
#[derive(Debug)]
pub struct OriginalPrefillRootProjectionCustody {
    _raw: RawSpanHostOwner,
}

/// Exactly one issued role. No Clone, refund, replacement grant or raw guard.
#[derive(Debug)]
pub struct OriginalPrefillScopeRole {
    role: PrefillControlRole,
    native: OriginalPrefillNativeCustody,
    recovery: OriginalPrefillRecoveryCustody,
    roots: OriginalPrefillRootCustody,
    projection: OriginalPrefillRootProjectionCustody,
}
impl OriginalPrefillScopeRole {
    /// Exact physical phase which this once-only owner was issued for.
    pub const fn role(&self) -> PrefillControlRole {
        self.role
    }
    /// Four actual independently retiring controls, not four numerical grants.
    pub fn into_custody(
        self,
    ) -> (
        OriginalPrefillNativeCustody,
        OriginalPrefillRecoveryCustody,
        OriginalPrefillRootCustody,
        OriginalPrefillRootProjectionCustody,
    ) {
        (self.native, self.recovery, self.roots, self.projection)
    }
}

/// Accepted original bank, still requiring the real first active text step.
#[derive(Debug)]
pub struct OriginalTextPrefillScopes {
    binding: TextControlBinding,
    reservation: WorkingMemoryReservation,
    claimed: bool,
    controls: OriginalTextControlGuard,
}
/// Once-only role construction cursor. The backend constructs all native slots
/// before source/encoder work, then enforces execution order in its private bank.
#[derive(Debug)]
pub struct OriginalTextPrefillScopeSet {
    facts: TextPrefillScopeFacts,
    next: u64,
    source_component_extracted: bool,
    request: InferenceRequest,
    controls: OriginalTextControlGuard,
}
impl OriginalTextPrefillScopeSet {
    /// Remaining original components; extracted sources keep their original account.
    /// Equality of another recipe is not authority.
    pub const fn facts(&self) -> TextPrefillScopeFacts {
        self.facts
    }
    /// Move the complete selected source component to an independent producer
    /// before any native role is issued. The actual accepted bank and account
    /// must match; neither capacity equality nor this receipt creates a bank.
    /// Remaining facts describe only the unextracted operation destinations.
    /// The original funding account and cumulative source spending are unchanged.
    pub fn take_source_component(
        &mut self,
        bank: &mut Option<OriginalHostDestinationBank>,
        expected: HostSourceConstructionFacts,
    ) -> Result<OriginalHostSourceBank, WorkingMemoryError> {
        self.validate_request(&self.request)?;
        if self.next != 0
            || self.source_component_extracted
            || self.facts.source_constructions != Some(expected)
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let facts = self
            .facts
            .host_destinations
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        let remaining = facts.without_source(expected)?;
        let actual = bank.as_ref().ok_or(WorkingMemoryError::IdentityMismatch)?;
        if !actual.belongs_to(&self.controls) || !actual.matches_facts(facts) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        // Every refusal precedes the paired move. No new role or source budget
        // can be recovered by dropping either resulting component.
        let sources = bank
            .as_mut()
            .expect("validated bank")
            .take_source_constructions()
            .expect("validated selected source component");
        let empty = remaining.capacity_bytes() == 0
            && remaining.maximum_attempts() == 0
            && remaining.maximum_partitions() == 0;
        self.source_component_extracted = true;
        self.facts.source_constructions = None;
        self.facts.host_destinations = (!empty).then_some(remaining);
        if empty {
            *bank = None;
        }
        Ok(sources)
    }
    /// Reattach an actual unspent component of the extracted source program
    /// before operation-bank construction. The same accepted account is required;
    /// facts alone cannot create or replenish a source. On refusal both owners
    /// remain with the caller, and successful restoration cannot be extracted again.
    pub fn restore_source_component(
        &mut self,
        bank: &mut Option<OriginalHostDestinationBank>,
        source: &mut Option<OriginalHostSourceBank>,
        expected: HostSourceConstructionFacts,
    ) -> Result<(), WorkingMemoryError> {
        self.validate_request(&self.request)?;
        if self.next != 0
            || !self.source_component_extracted
            || self.facts.source_constructions.is_some()
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let reservation = self.request.memory_reservation();
        let restored = OriginalHostDestinationBank::restore_source_component(
            bank,
            source,
            expected,
            self.facts.host_destinations,
            &self.controls,
            reservation,
        )?;
        self.facts.source_constructions = Some(expected);
        self.facts.host_destinations = Some(restored);
        Ok(())
    }
    /// Validate the actual request before consuming or retrying native slots.
    pub fn validate_request(&self, request: &InferenceRequest) -> Result<(), WorkingMemoryError> {
        self.request.validate_same_request(request)?;
        self.controls
            .validate_reservation(request.memory_reservation())
    }
    /// Validate installation against the actual selected session identity. This
    /// only compares the original request; it creates no binding or grant.
    pub fn validate_execution(
        &self,
        execution: &crate::working_memory::InferenceExecutionIdentity,
    ) -> Result<(), WorkingMemoryError> {
        self.request.validate(execution, self.facts.plan.geometry())
    }
    /// Construct one named role in the prepriced order. A mismatch leaves the
    /// cursor untouched. Previously issued owners can never be requested again.
    pub fn take_next(
        &mut self,
        role: PrefillControlRole,
    ) -> Result<OriginalPrefillScopeRole, WorkingMemoryError> {
        if self.facts.plan.role(self.next) != Some(role) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        self.controls
            .validate_reservation(self.request.memory_reservation())?;
        let original = OriginalPrefillScopeRole {
            role,
            native: OriginalPrefillNativeCustody {
                _raw: self.controls.custody.raw().clone(),
            },
            recovery: OriginalPrefillRecoveryCustody {
                _raw: self.controls.custody.raw().clone(),
            },
            roots: OriginalPrefillRootCustody {
                _raw: self.controls.custody.raw().clone(),
            },
            projection: OriginalPrefillRootProjectionCustody {
                _raw: self.controls.custody.raw().clone(),
            },
        };
        self.next += 1; // role() already proves next < the checked finite count.
        Ok(original)
    }
}
impl PreparedTextControlWorkspace {
    /// Add the complete selected local prefill component before original Q is
    /// sealed. Graph payload is already in the one original arena, not charged
    /// twice as new backing. This does not certify the remaining native graph.
    pub fn with_prefill_scopes(
        mut self,
        facts: TextPrefillScopeFacts,
    ) -> Result<Self, WorkingMemoryError> {
        if self.binding.prefill_scopes.is_some() {
            return Err(WorkingMemoryError::AlreadyStarted);
        }
        if facts.plan.geometry() != self.binding.geometry {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let graph = self
            .binding
            .graph_metadata
            .ok_or(WorkingMemoryError::UnknownBound)?;
        facts.graph_residual_ceiling(graph.capacity().get())?;
        let neutral = u64::try_from(bank_control_bytes().ok_or(WorkingMemoryError::Overflow)?)
            .map_err(|_| WorkingMemoryError::Overflow)?;
        let role = u64::try_from(role_control_bytes().ok_or(WorkingMemoryError::Overflow)?)
            .map_err(|_| WorkingMemoryError::Overflow)?;
        let added = role
            .checked_mul(facts.plan.scope_count())
            .and_then(|n| n.checked_add(neutral))
            .and_then(|n| n.checked_add(facts.known_bytes().ok()?))
            .ok_or(WorkingMemoryError::Overflow)?;
        let total = self
            .binding
            .facts
            .admission
            .unwrap_or(0)
            .checked_add(added)
            .ok_or(WorkingMemoryError::Overflow)?;
        self.binding.facts.admission = self
            .binding
            .facts
            .admission
            .zip(facts.total_bytes()?)
            .map(|_| total);
        if let Some(host) = facts.host_destinations {
            if host.source_constructions() != facts.source_constructions {
                return Err(WorkingMemoryError::IdentityMismatch);
            }
            self = self.with_host_destinations(host)?;
        } else if let Some(source) = facts.source_constructions {
            self = self.with_host_source_constructions(source)?;
        }
        if let Some(outputs) = facts.output_sources {
            self = self.with_output_source_constructions(outputs)?;
        }
        self.binding.prefill_scopes = Some(facts);
        Ok(self)
    }
}
impl OwnedTextSpanWorkspace {
    /// Extract once from the exact accepted workspace. No preparation token or
    /// retained receipt is sufficient to issue the first native role afterward.
    pub fn take_prefill_scopes(
        &mut self,
    ) -> Result<Option<OriginalTextPrefillScopes>, WorkingMemoryError> {
        let binding = self
            .workspace()
            .control_binding()
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        if binding.prefill_scopes.is_none() {
            return Ok(None);
        }
        if self.prefill_scopes_taken {
            return Err(WorkingMemoryError::AlreadyStarted);
        }
        self.controls.validate_reservation(self.reservation())?;
        let bank = OriginalTextPrefillScopes {
            binding: binding.clone(),
            reservation: self.reservation().clone(),
            claimed: false,
            controls: self.controls.clone(),
        };
        self.prefill_scopes_taken = true;
        Ok(Some(bank))
    }
}
impl OriginalTextPrefillScopes {
    /// Borrow the retained cold source facts. This cannot issue a role or grant
    /// native submission authority, including after the first bank was claimed.
    pub const fn facts(&self) -> Option<TextPrefillScopeFacts> {
        self.binding.prefill_scopes
    }
    /// Only the genuine first active step of this same original run can issue
    /// the complete bank. Decode, equal requests, receipts and retries cannot.
    /// ```compile_fail
    /// use eredu_runtime::working_memory::{OriginalTextPrefillScopes, InferenceTextStepReceipt};
    /// fn cannot_issue(bank: &mut OriginalTextPrefillScopes, receipt: &InferenceTextStepReceipt) {
    ///     let _ = bank.claim(receipt);
    /// }
    /// ```
    pub fn claim(
        &mut self,
        step: &InferenceTextStep,
    ) -> Result<OriginalTextPrefillScopeSet, WorkingMemoryError> {
        if step.request().geometry() != self.binding.geometry {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        self.controls.validate_reservation(&self.reservation)?;
        let (_, attempt) = step.original_scope_identity(&self.reservation)?;
        if attempt != 0 || self.claimed {
            return Err(WorkingMemoryError::AlreadyStarted);
        }
        let facts = self
            .binding
            .prefill_scopes
            .ok_or(WorkingMemoryError::UnknownBound)?;
        let set = OriginalTextPrefillScopeSet {
            facts,
            next: 0,
            source_component_extracted: false,
            request: step.request().clone(),
            controls: self.controls.clone(),
        };
        self.claimed = true;
        Ok(set)
    }
}
fn bank_control_bytes() -> Option<usize> {
    [
        crate::working_memory::text_preparation::synchronization_control_bytes()?,
        size_of::<OriginalTextPrefillScopes>(),
        size_of::<Option<OriginalTextPrefillScopes>>(),
        size_of::<Result<Option<OriginalTextPrefillScopes>, WorkingMemoryError>>(),
        size_of::<OriginalTextPrefillScopeSet>(),
        size_of::<(
            &mut OriginalTextPrefillScopeSet,
            &mut Option<OriginalHostDestinationBank>,
            HostSourceConstructionFacts,
        )>(),
        size_of::<Result<OriginalHostSourceBank, WorkingMemoryError>>(),
        size_of::<(
            &mut OriginalTextPrefillScopeSet,
            &mut Option<OriginalHostDestinationBank>,
            &mut Option<OriginalHostSourceBank>,
            HostSourceConstructionFacts,
        )>(),
        size_of::<(
            &mut Option<OriginalHostDestinationBank>,
            &mut Option<OriginalHostSourceBank>,
            HostSourceConstructionFacts,
            Option<HostDestinationFacts>,
            &OriginalTextControlGuard,
            &WorkingMemoryReservation,
        )>(),
        size_of::<(&OriginalHostSourceBank, &OriginalHostDestinationBank)>(),
        size_of::<(HostDestinationFacts, HostDestinationFacts)>(),
        size_of::<Result<HostDestinationFacts, WorkingMemoryError>>(),
        size_of::<Result<(), WorkingMemoryError>>(),
        size_of::<OriginalHostDestinationBank>(),
        size_of::<OriginalHostSourceBank>(),
        size_of::<(HostDestinationFacts, HostDestinationFacts, bool)>(),
        size_of::<Result<OriginalTextPrefillScopeSet, WorkingMemoryError>>(),
        size_of::<TextPrefillScopeFacts>(),
        size_of::<TextControlBinding>(),
        size_of::<InferenceRequest>(),
        size_of::<WorkingMemoryReservation>(),
        size_of::<OriginalTextControlGuard>(),
        size_of::<(
            Arc<crate::working_memory::text_preparation::TextPreparationAuthority>,
            u64,
        )>(),
        size_of::<
            Result<
                (
                    Arc<crate::working_memory::text_preparation::TextPreparationAuthority>,
                    u64,
                ),
                WorkingMemoryError,
            >,
        >(),
    ]
    .into_iter()
    .try_fold(0usize, usize::checked_add)
}
fn role_control_bytes() -> Option<usize> {
    [
        size_of::<PrefillControlRole>(),
        size_of::<OriginalPrefillScopeRole>(),
        size_of::<Option<OriginalPrefillScopeRole>>(),
        size_of::<Result<OriginalPrefillScopeRole, WorkingMemoryError>>(),
        size_of::<(
            OriginalPrefillNativeCustody,
            OriginalPrefillRecoveryCustody,
            OriginalPrefillRootCustody,
            OriginalPrefillRootProjectionCustody,
        )>(),
        size_of::<Result<(), WorkingMemoryError>>(),
    ]
    .into_iter()
    .try_fold(0usize, usize::checked_add)
}
