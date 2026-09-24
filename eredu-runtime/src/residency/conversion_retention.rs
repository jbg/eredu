//! Execution-scoped conversion admission, independent of native storage.
//!
//! A registry holds only weak execution groups and immutable parameter entries.
//! Clone one budget across every permanent unit, embedded owner and partition of
//! a selected execution. Independent executions must use independent group IDs.
//! Backends pair each claim with its native conversion; a registry or an entry
//! alone never grants permission to attach a conversion to a permanent owner.
//!
//! Lock order is registry, parameter entry, budget. Claim release takes only the
//! budget lock. No lock guard escapes this module; native creation, evaluation,
//! completion and destruction must happen outside these operations. A reservation
//! singles out one publisher while evaluation runs without locks. Invalidation
//! revokes that entry permanently and cancels its ticket, so a late publisher
//! cannot revive it. Replacement materializations register a new entry after
//! invalidation; old handles remain retired. Revoked native references must be
//! detached by their backend owners at a completion-safe boundary. Revocation
//! reports logical claim release, never physical reclamation.

use std::{
    collections::BTreeMap,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex, MutexGuard, Weak,
    },
};

use eredu_core::{
    residency::{
        ParameterConversionRetentionGroup, ParameterConversionRetentionPolicy,
        ParameterConversionRetentionPolicyReport, ParameterConversionRetentionReport,
        ParameterConversionRetentionUsage,
    },
    resources::ResourceIdentity,
    Observed,
};

/// Admission rejection; callers may still execute an ordinary temporary cast.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ConversionRetentionError {
    /// Identity namespaces and keys must be nonempty.
    #[error("conversion retention identity must have a nonempty scope and key")]
    InvalidIdentity,
    /// One live identity was registered with inconsistent immutable facts.
    #[error("conversion retention identity has conflicting policy or payload")]
    ConflictingIdentity,
    /// The effective policy excludes retention.
    #[error("parameter conversion retention is disabled")]
    Disabled,
    /// Retained plus reserved bytes would exceed the execution allowance.
    #[error("parameter conversion retention budget is exhausted")]
    Capacity,
    /// Payload or aggregate arithmetic cannot be represented.
    #[error("parameter conversion retention accounting overflow")]
    Overflow,
    /// Another caller is already preparing this immutable conversion.
    #[error("parameter conversion publication is already in progress")]
    Busy,
    /// Parameter replacement retired this entry or its pending reservation.
    #[error("parameter conversion entry was invalidated")]
    Invalidated,
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|error| error.into_inner())
}

fn validate(identity: &ResourceIdentity) -> Result<(), ConversionRetentionError> {
    if identity.scope.trim().is_empty() || identity.key.trim().is_empty() {
        Err(ConversionRetentionError::InvalidIdentity)
    } else {
        Ok(())
    }
}

/// Checked payload sizing before admission or native conversion creation.
pub fn conversion_retention_payload_bytes(
    elements: u64,
    bytes_per_element: u64,
) -> Result<u64, ConversionRetentionError> {
    elements
        .checked_mul(bytes_per_element)
        .ok_or(ConversionRetentionError::Overflow)
}

/// Weak lookup shared by backend materialization owners; never owns a model.
#[derive(Default)]
pub struct ConversionRetentionRegistry {
    state: Mutex<RegistryState>,
}

#[derive(Default)]
struct RegistryState {
    groups: BTreeMap<ParameterConversionRetentionGroup, Weak<BudgetInner>>,
    parameters: BTreeMap<ResourceIdentity, Weak<ParameterInner>>,
    allocations: BTreeMap<ResourceIdentity, Weak<AllocationInner>>,
}

/// Immutable observed backing identity and payload, shared across group claims.
#[derive(Clone)]
pub struct ConversionRetentionAllocation(Arc<AllocationInner>);

struct AllocationInner {
    identity: ResourceIdentity,
    payload_bytes: u64,
}

impl ConversionRetentionRegistry {
    /// Registers an actual evaluated backing, rejecting contradictory payload
    /// observations while any descriptor or claim for that backing is alive.
    /// This descriptor grants no permission to retain it in a group.
    pub fn allocation(
        &self,
        identity: ResourceIdentity,
        payload_bytes: u64,
    ) -> Result<ConversionRetentionAllocation, ConversionRetentionError> {
        validate(&identity)?;
        let mut registry = lock(&self.state);
        registry
            .allocations
            .retain(|_, value| value.strong_count() > 0);
        if let Some(existing) = registry.allocations.get(&identity).and_then(Weak::upgrade) {
            if existing.payload_bytes != payload_bytes {
                return Err(ConversionRetentionError::ConflictingIdentity);
            }
            return Ok(ConversionRetentionAllocation(existing));
        }
        let allocation = Arc::new(AllocationInner {
            identity: identity.clone(),
            payload_bytes,
        });
        registry
            .allocations
            .insert(identity, Arc::downgrade(&allocation));
        Ok(ConversionRetentionAllocation(allocation))
    }

    /// Creates or joins the selected execution's budget. Repeated identities
    /// must have the same selection facts; cloning does not multiply allowance.
    pub fn budget(
        &self,
        group: ParameterConversionRetentionGroup,
        policy: ParameterConversionRetentionPolicyReport,
    ) -> Result<ConversionRetentionBudget, ConversionRetentionError> {
        validate(&group.0)?;
        // Reports are public descriptive records; reject inconsistent hand-built
        // records rather than accidentally enabling an ineligible execution.
        let requested = match policy.source {
            eredu_core::residency::ParameterConversionRetentionPolicySource::ManagedDefault => None,
            eredu_core::residency::ParameterConversionRetentionPolicySource::Explicit => {
                Some(policy.requested)
            }
        };
        let resolved = ParameterConversionRetentionPolicyReport::resolve(
            requested,
            policy.eligibility.clone(),
        );
        if policy != resolved {
            return Err(ConversionRetentionError::ConflictingIdentity);
        }
        let mut registry = lock(&self.state);
        registry.groups.retain(|_, value| value.strong_count() > 0);
        if let Some(existing) = registry.groups.get(&group).and_then(Weak::upgrade) {
            if existing.policy != policy {
                return Err(ConversionRetentionError::ConflictingIdentity);
            }
            return Ok(ConversionRetentionBudget(existing));
        }
        let budget = Arc::new(BudgetInner {
            group: group.clone(),
            policy,
            state: Mutex::default(),
        });
        registry.groups.insert(group, Arc::downgrade(&budget));
        Ok(ConversionRetentionBudget(budget))
    }

    /// Registers immutable source identity and exact intended conversion payload.
    /// Aliases register the same identity. Names of mutable parameter slots are
    /// insufficient: replacement sources need distinct immutable identities.
    /// Registering a retired source creates a new materialization entry; it does
    /// not reactivate any old handle, claim or reservation.
    pub fn parameter(
        &self,
        identity: ResourceIdentity,
        payload_bytes: u64,
    ) -> Result<ConversionRetentionParameter, ConversionRetentionError> {
        validate(&identity)?;
        let mut registry = lock(&self.state);
        registry
            .parameters
            .retain(|_, value| value.strong_count() > 0);
        if let Some(existing) = registry.parameters.get(&identity).and_then(Weak::upgrade) {
            if existing.payload_bytes != payload_bytes {
                return Err(ConversionRetentionError::ConflictingIdentity);
            }
            if lock(&existing.state).active {
                return Ok(ConversionRetentionParameter(existing));
            }
        }
        let entry = Arc::new(ParameterInner {
            identity: identity.clone(),
            payload_bytes,
            state: Mutex::new(ParameterState {
                active: true,
                pending: None,
                allocation: None,
                claims: BTreeMap::new(),
            }),
        });
        registry.parameters.insert(identity, Arc::downgrade(&entry));
        Ok(ConversionRetentionParameter(entry))
    }
}

/// One selected execution's fixed policy and shared accounting authority.
#[derive(Clone)]
pub struct ConversionRetentionBudget(Arc<BudgetInner>);

struct BudgetInner {
    group: ParameterConversionRetentionGroup,
    policy: ParameterConversionRetentionPolicyReport,
    state: Mutex<BudgetState>,
}

#[derive(Default)]
struct BudgetState {
    retained: u64,
    reserved: u64,
    allocations: BTreeMap<ResourceIdentity, AllocationCharge>,
}

struct AllocationCharge {
    bytes: u64,
    bindings: u64,
}

impl BudgetInner {
    fn admit(&self, state: &BudgetState, bytes: u64) -> Result<(), ConversionRetentionError> {
        let total = state
            .retained
            .checked_add(state.reserved)
            .and_then(|total| total.checked_add(bytes))
            .ok_or(ConversionRetentionError::Overflow)?;
        match self.policy.effective {
            ParameterConversionRetentionPolicy::Disabled => Err(ConversionRetentionError::Disabled),
            ParameterConversionRetentionPolicy::Bounded { max_bytes } if total > max_bytes => {
                Err(ConversionRetentionError::Capacity)
            }
            _ => Ok(()),
        }
    }
}

impl std::fmt::Debug for ConversionRetentionBudget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConversionRetentionBudget")
            .field("group", &self.0.group)
            .field("policy", &self.0.policy)
            .finish_non_exhaustive()
    }
}

impl ConversionRetentionBudget {
    /// Scope shared across this execution's units and partitions.
    pub fn group(&self) -> &ParameterConversionRetentionGroup {
        &self.0.group
    }

    /// Drops the supplied owner's claims atomically with the returned ledger snapshot.
    /// Aliases outside this batch and other groups remain live. Rejection leaves
    /// the batch untouched. Native owners must detach paired storage only after
    /// establishing completion, and prevent concurrent publication during release.
    pub fn release_claims(
        &self,
        claims: &mut Vec<ConversionRetentionClaim>,
    ) -> Result<
        eredu_core::residency::ParameterConversionRetentionTrimReport,
        ConversionRetentionError,
    > {
        if claims
            .iter()
            .any(|claim| !Arc::ptr_eq(&claim.0.budget, &self.0))
        {
            return Err(ConversionRetentionError::ConflictingIdentity);
        }
        let mut ledger = lock(&self.0.state);
        if ledger.reserved != 0 {
            return Err(ConversionRetentionError::Busy);
        }
        let before_claims = ledger.allocations.len() as u64;
        let before_bytes = ledger.retained;
        let mut aliases = BTreeMap::<usize, usize>::new();
        for claim in claims.iter() {
            *aliases.entry(Arc::as_ptr(&claim.0) as usize).or_default() += 1;
        }
        for claim in claims.iter() {
            if aliases[&(Arc::as_ptr(&claim.0) as usize)] == Arc::strong_count(&claim.0) {
                claim.0.release_locked(&mut ledger);
            }
        }
        let remaining = ParameterConversionRetentionUsage {
            retained_claims: ledger.allocations.len() as u64,
            retained_payload_bytes: ledger.retained,
            reserved_payload_bytes: ledger.reserved,
            retained_backing_capacity_bytes: Observed::unavailable(
                "native backing capacity is not observed",
            ),
        };
        let report = eredu_core::residency::ParameterConversionRetentionTrimReport {
            group: self.group().clone(),
            released_claims: before_claims - remaining.retained_claims,
            released_payload_bytes: before_bytes - remaining.retained_payload_bytes,
            remaining,
            reclaimed_backing_bytes: Observed::unavailable(
                "other owners and native graphs may retain backing",
            ),
        };
        drop(ledger);
        claims.clear();
        Ok(report)
    }

    /// Coherent read-only accounting; never populates, settles or trims storage.
    pub fn report(&self) -> ParameterConversionRetentionReport {
        let state = lock(&self.0.state);
        ParameterConversionRetentionReport {
            group: self.0.group.clone(),
            policy: Observed::exact(self.0.policy.clone(), "retained execution selection"),
            usage: Observed::exact(
                ParameterConversionRetentionUsage {
                    retained_claims: state.allocations.len() as u64,
                    retained_payload_bytes: state.retained,
                    reserved_payload_bytes: state.reserved,
                    retained_backing_capacity_bytes: Observed::unavailable(
                        "native conversion backing capacity is not observed by the budget",
                    ),
                },
                "shared conversion admission ledger",
            ),
        }
    }
}

/// Registered immutable parameter; owning an entry alone retains no conversion.
#[derive(Clone)]
pub struct ConversionRetentionParameter(Arc<ParameterInner>);

struct ParameterInner {
    identity: ResourceIdentity,
    payload_bytes: u64,
    state: Mutex<ParameterState>,
}

struct ParameterState {
    active: bool,
    pending: Option<Pending>,
    allocation: Option<ConversionRetentionAllocation>,
    claims: BTreeMap<ParameterConversionRetentionGroup, Weak<ClaimInner>>,
}

struct Pending {
    ticket: Weak<()>,
    budget: Arc<BudgetInner>,
}

/// Either existing backing admitted to this group or exclusive creation authority.
pub enum ConversionRetentionAdmission {
    /// Pair this claim with the already published native allocation, only after
    /// checking the claim remains active under the backend's entry synchronization.
    Retained(ConversionRetentionClaim),
    /// Create and evaluate outside locks, then publish the actual allocation ID.
    Reserved(ConversionRetentionReservation),
}

impl ConversionRetentionParameter {
    /// Exact immutable source identity, independent of mutable slot names.
    pub fn identity(&self) -> &ResourceIdentity {
        &self.0.identity
    }

    /// Admits a group claim or reserves before creation. Concurrent creation
    /// returns `Busy`; it cannot double-charge or publish a competing conversion.
    /// Cross-group reuse passes the same admission ceiling as new retention.
    pub fn reserve(
        &self,
        budget: &ConversionRetentionBudget,
    ) -> Result<ConversionRetentionAdmission, ConversionRetentionError> {
        let mut entry = lock(&self.0.state);
        if !entry.active {
            return Err(ConversionRetentionError::Invalidated);
        }
        if entry.pending.is_some() {
            return Err(ConversionRetentionError::Busy);
        }
        if let Some(claim) = entry.claims.get(budget.group()).and_then(Weak::upgrade) {
            if !Arc::ptr_eq(&claim.budget, &budget.0) {
                return Err(ConversionRetentionError::ConflictingIdentity);
            }
            return Ok(ConversionRetentionAdmission::Retained(
                ConversionRetentionClaim(claim),
            ));
        }
        // Hold live claim metadata through admission so the final old claim
        // cannot disappear between inspecting the entry and joining its backing.
        let live_claims: Vec<_> = entry.claims.values().filter_map(Weak::upgrade).collect();
        entry.claims.retain(|_, claim| claim.strong_count() > 0);
        if live_claims.is_empty() {
            entry.allocation = None;
        }
        if let Some(allocation) = entry.allocation.clone() {
            let mut ledger = lock(&budget.0.state);
            let claim = self.claim(budget, &mut ledger, allocation)?;
            entry
                .claims
                .insert(budget.group().clone(), Arc::downgrade(&claim.0));
            drop(ledger);
            drop(live_claims);
            return Ok(ConversionRetentionAdmission::Retained(claim));
        }
        let mut ledger = lock(&budget.0.state);
        budget.0.admit(&ledger, self.0.payload_bytes)?;
        ledger.reserved += self.0.payload_bytes;
        let ticket = Arc::new(());
        entry.pending = Some(Pending {
            ticket: Arc::downgrade(&ticket),
            budget: budget.0.clone(),
        });
        Ok(ConversionRetentionAdmission::Reserved(
            ConversionRetentionReservation {
                parameter: self.clone(),
                budget: budget.clone(),
                ticket,
            },
        ))
    }

    fn claim(
        &self,
        budget: &ConversionRetentionBudget,
        ledger: &mut BudgetState,
        allocation: ConversionRetentionAllocation,
    ) -> Result<ConversionRetentionClaim, ConversionRetentionError> {
        if let Some(charge) = ledger.allocations.get_mut(&allocation.0.identity) {
            if charge.bytes != self.0.payload_bytes {
                return Err(ConversionRetentionError::ConflictingIdentity);
            }
            charge.bindings = charge
                .bindings
                .checked_add(1)
                .ok_or(ConversionRetentionError::Overflow)?;
        } else {
            budget.0.admit(ledger, self.0.payload_bytes)?;
            ledger.retained += self.0.payload_bytes;
            ledger.allocations.insert(
                allocation.0.identity.clone(),
                AllocationCharge {
                    bytes: self.0.payload_bytes,
                    bindings: 1,
                },
            );
        }
        Ok(ConversionRetentionClaim(Arc::new(ClaimInner {
            parameter: self.0.clone(),
            budget: budget.0.clone(),
            allocation,
            active: AtomicBool::new(true),
        })))
    }

    /// Retires this entry, cancels its reservation and revokes its claims once.
    /// Old aliases cannot republish. Other parameter entries and groups' claims
    /// for those entries remain intact, including when sharing physical backing.
    /// Backend synchronization must detach retired native values safely; this
    /// operation itself neither evaluates nor establishes native completion.
    pub fn invalidate(&self) {
        let mut entry = lock(&self.0.state);
        entry.active = false;
        if let Some(pending) = entry.pending.take() {
            lock(&pending.budget.state).reserved -= self.0.payload_bytes;
        }
        for claim in entry.claims.values().filter_map(Weak::upgrade) {
            claim.release();
        }
        entry.claims.clear();
        entry.allocation = None;
    }
}

/// Move-only reservation. Dropping, failure or invalidation restores capacity
/// exactly once. The controller performs no native operation; the backend may
/// create and evaluate the conversion while holding this ticket.
#[must_use = "drop to cancel, or publish after native evaluation succeeds"]
pub struct ConversionRetentionReservation {
    parameter: ConversionRetentionParameter,
    budget: ConversionRetentionBudget,
    ticket: Arc<()>,
}

impl ConversionRetentionReservation {
    /// Publishes the actual evaluated allocation identity. Validation failure
    /// cancels this reservation. Allocation identity must name the real backing,
    /// including across groups; parameter or owner identity is not a substitute.
    pub fn publish(
        self,
        allocation: ConversionRetentionAllocation,
    ) -> Result<ConversionRetentionClaim, ConversionRetentionError> {
        if allocation.0.payload_bytes != self.parameter.0.payload_bytes {
            return Err(ConversionRetentionError::ConflictingIdentity);
        }
        let mut entry = lock(&self.parameter.0.state);
        if !entry.active
            || !entry
                .pending
                .as_ref()
                .is_some_and(|pending| pending.ticket.ptr_eq(&Arc::downgrade(&self.ticket)))
        {
            return Err(ConversionRetentionError::Invalidated);
        }
        let mut ledger = lock(&self.budget.0.state);
        entry.pending = None;
        ledger.reserved -= self.parameter.0.payload_bytes;
        let claim = self
            .parameter
            .claim(&self.budget, &mut ledger, allocation.clone())?;
        entry.allocation = Some(allocation);
        entry
            .claims
            .insert(self.budget.group().clone(), Arc::downgrade(&claim.0));
        Ok(claim)
    }
}

impl Drop for ConversionRetentionReservation {
    fn drop(&mut self) {
        let mut entry = lock(&self.parameter.0.state);
        if entry
            .pending
            .as_ref()
            .is_some_and(|pending| pending.ticket.ptr_eq(&Arc::downgrade(&self.ticket)))
        {
            entry.pending = None;
            lock(&self.budget.0.state).reserved -= self.parameter.0.payload_bytes;
        }
    }
}

/// Shared group claim to one immutable conversion. Clones are aliases; the last
/// clone releases its binding. The last binding to an allocation frees its group
/// charge. Other groups retain their own independent claims to shared backing.
#[derive(Clone)]
pub struct ConversionRetentionClaim(Arc<ClaimInner>);

struct ClaimInner {
    parameter: Arc<ParameterInner>,
    budget: Arc<BudgetInner>,
    allocation: ConversionRetentionAllocation,
    active: AtomicBool,
}

impl ClaimInner {
    fn release(&self) {
        let mut ledger = lock(&self.budget.state);
        self.release_locked(&mut ledger);
    }
    fn release_locked(&self, ledger: &mut BudgetState) {
        if !self.active.swap(false, Ordering::AcqRel) {
            return;
        }
        let charge = ledger
            .allocations
            .get_mut(&self.allocation.0.identity)
            .expect("active allocation claim");
        charge.bindings -= 1;
        if charge.bindings == 0 {
            ledger.retained -= charge.bytes;
            ledger.allocations.remove(&self.allocation.0.identity);
        }
    }
}

impl Drop for ClaimInner {
    fn drop(&mut self) {
        self.release();
    }
}

impl ConversionRetentionClaim {
    /// Whether parameter invalidation has revoked this claim. Backends serialize
    /// native publication/attachment with invalidation; this query alone is not
    /// a native-completion fence or a lock across subsequent native operations.
    pub fn is_active(&self) -> bool {
        self.0.active.load(Ordering::Acquire)
    }
    /// Exact source to which this admitted conversion belongs.
    pub fn parameter(&self) -> &ResourceIdentity {
        &self.0.parameter.identity
    }
    /// Real conversion backing for physical residency deduplication.
    pub fn allocation(&self) -> &ResourceIdentity {
        &self.0.allocation.0.identity
    }
    /// Independent execution-budget scope of this retention claim.
    pub fn group(&self) -> &ParameterConversionRetentionGroup {
        &self.0.budget.group
    }
    /// Payload charged once per allocation in this group, never backing capacity.
    pub fn payload_bytes(&self) -> u64 {
        self.0.parameter.payload_bytes
    }
}

#[cfg(test)]
mod tests;
