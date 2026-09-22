//! Fixed mixed publication into the existing canonical registry.
//!
//! Only the consumed selected bank supplies production observations from its
//! retained neutral mechanism. That provider authenticates native birth, budget,
//! generation and charged capacity; keys and capacities alone cannot create
//! partition coverage. The existing-alias route recognizes only a published
//! native row and derives origin from that row, never from the caller. Raw
//! constructors remain private; the additional arbitrary fixtures are test-only.

use super::directory::PreparedNamespace;
use super::prepaid::{PrepaidHostOrigin, PrepaidStorageOrigin};
use super::*;
use funding::native_partition::NativePartition;
use std::mem::{align_of, size_of};
pub(super) mod copy;
pub(super) mod numerical;
use copy::PublicationOrigin;

pub(in crate::working_memory) struct NativeStorageWitness<K> {
    key: K,
    bytes: u64,
    origin: PrepaidStorageOrigin,
}

// The retained native mechanism authenticates this exact immutable key and
// charged capacity before constructing this closed input. It carries neither a
// partition nor a native owner, and cannot introduce a new canonical row.
pub(in crate::working_memory) struct ExistingNativeAlias<K> {
    key: K,
    bytes: u64,
    immutable: bool,
}

pub(in crate::working_memory) enum NativePublicationInput<K> {
    Ordinary(K, u64),
    // An actual explicit source owner, positively validated by the complete
    // inventory. May reuse an existing same-request prepaid immutable row;
    // absence still takes the ordinary source charge and creates no origin.
    SourceInventory(K, u64),
    Native(NativeStorageWitness<K>),
    Existing(ExistingNativeAlias<K>),
    // Positive actual-owner proof, but no claimed accounting provenance.
    ExistingPhysical(K, u64),
    CompletedNumerical(
        K,
        u64,
        crate::working_memory::OriginalNumericalBudgetCustody,
    ),
    ExistingSource(NativeStorageWitness<K>),
    // Ordinary load-time source payload, already fully paid in this pool.
    // Unlike SourceInventory this can never create a row or take native credit.
    RegisteredSource(K, u64),
}

impl<K> NativePublicationInput<K> {
    fn parts(&self) -> (&K, u64, Option<&PrepaidStorageOrigin>, bool, bool, bool) {
        match self {
            Self::Ordinary(key, bytes) | Self::SourceInventory(key, bytes) => {
                (key, *bytes, None, false, false, false)
            }
            Self::Native(witness) => (
                &witness.key,
                witness.bytes,
                Some(&witness.origin),
                false,
                witness.origin.immutable(),
                false,
            ),
            Self::ExistingPhysical(key, bytes) | Self::CompletedNumerical(key, bytes, _) => {
                (key, *bytes, None, true, false, false)
            }
            Self::Existing(alias) => (&alias.key, alias.bytes, None, true, alias.immutable, false),
            Self::RegisteredSource(key, bytes) => (key, *bytes, None, true, false, true),
            Self::ExistingSource(witness) => (
                &witness.key,
                witness.bytes,
                Some(&witness.origin),
                true,
                false,
                true,
            ),
        }
    }
}

struct Row<K: Ord + Send + Sync + 'static> {
    first_input: usize,
    // Provider keys must all retire before the result registration.
    key: Option<Arc<K>>,
    registry_key: Option<RegistryKey<K>>,
    bytes: u64,
    placement: Arc<eredu_core::MemoryPlacement>,
    // Sticky across duplicates: even a same-key birth witness cannot replace
    // the requirement for an already registered native allocation.
    existing_only: bool,
    existing_physical: bool,
    source_inventory: bool,
    registered_source: bool,
    immutable: bool,
    source: bool,
    debit: bool,
    native_debit: bool,
    funding_allowance_bytes: u64,
    locator: Option<EntryLocator>,
    output: Option<WorkingMemoryStorage<K>>,
    activation_pool: Option<MemoryLedger>,
    // Failed foreign-prefix keys must retire before their donor custody too.
    origin: Option<PrepaidStorageOrigin>,
    completed: Option<crate::working_memory::OriginalNumericalBudgetCustody>,
}

/// One terminal attempt. Inputs and partial preparation survive all refusals
/// and provider unwind. Construction is not a claim that control storage fits Q.
pub(in crate::working_memory) struct PreparedNativePublication<K: Ord + Send + Sync + 'static> {
    inputs: Vec<NativePublicationInput<K>>,
    placements: Vec<Arc<eredu_core::MemoryPlacement>>,
    rows: Vec<Row<K>>,
    node: Option<Box<RegistryBatch<K>>>,
    // Precharged outside Usage; only a successful first real row installs it.
    namespace: Option<PreparedNamespace>,
    terminal: bool,
    published: bool,
    slots: usize,
    exact_storage: bool,
    failure_site: &'static str,
    missing_existing_input: Option<usize>,
    partition: PublicationOrigin,
}

fn same_coverage(a: Option<&PrepaidStorageOrigin>, b: Option<&PrepaidStorageOrigin>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => a.same_origin(b),
        _ => false,
    }
}

impl<K: Clone + Ord + Send + Sync + 'static> PreparedNativePublication<K> {
    pub(in crate::working_memory) fn failure_site(&self) -> &'static str {
        self.failure_site
    }

    pub(in crate::working_memory) fn missing_existing_input(&self) -> Option<usize> {
        self.missing_existing_input
    }

    pub(in crate::working_memory) fn requested_control_bytes(
        slots: usize,
    ) -> Result<u64, WorkingMemoryError> {
        // Requested layouts only. Arc headers, allocator charge/spare capacity,
        // provider key payload and moves remain separately required inputs.
        let array = |size: usize, align: usize| {
            let bytes = size
                .checked_mul(slots)
                .ok_or(WorkingMemoryError::Overflow)?;
            std::alloc::Layout::from_size_align(bytes, align)
                .map_err(|_| WorkingMemoryError::Overflow)?;
            u64::try_from(bytes).map_err(|_| WorkingMemoryError::Overflow)
        };
        let values = [
            array(
                size_of::<NativePublicationInput<K>>(),
                align_of::<NativePublicationInput<K>>(),
            )?,
            array(size_of::<Row<K>>(), align_of::<Row<K>>())?,
            array(
                size_of::<Arc<eredu_core::MemoryPlacement>>(),
                align_of::<Arc<eredu_core::MemoryPlacement>>(),
            )?,
            // Include both requested buffers during Vec-to-Box conversion.
            array(
                size_of::<Option<(RegistryKey<K>, Entry)>>(),
                align_of::<Option<(RegistryKey<K>, Entry)>>(),
            )?,
            array(
                size_of::<Option<(RegistryKey<K>, Entry)>>(),
                align_of::<Option<(RegistryKey<K>, Entry)>>(),
            )?,
            array(size_of::<K>(), align_of::<K>())?,
            array(size_of::<K>(), align_of::<K>())?,
            array(size_of::<Registration<K>>(), align_of::<Registration<K>>())?,
            size_of::<Self>() as u64,
            size_of::<RegistryBatch<K>>() as u64,
            PreparedNamespace::requested_control_bytes::<K>()?,
        ];
        values.into_iter().try_fold(0u64, |a, b| {
            a.checked_add(b).ok_or(WorkingMemoryError::Overflow)
        })
    }

    pub(in crate::working_memory) fn prepare(
        partition: NativePartition,
        inputs: Vec<NativePublicationInput<K>>,
    ) -> Self {
        let slots = inputs.len();
        let placements = vec![Arc::clone(&partition.pool().0.host_placement); slots];
        Self {
            inputs,
            placements,
            rows: Vec::with_capacity(slots),
            node: Some(RegistryBatch::prepare_native(slots, partition.clone())),
            namespace: Some(PreparedNamespace::prepare::<K>(
                partition.namespace_metadata(),
            )),
            terminal: false,
            published: false,
            slots,
            exact_storage: false,
            failure_site: "registry preparation",
            missing_existing_input: None,
            partition: PublicationOrigin::Prepaid(PrepaidStorageOrigin::Native(partition)),
        }
    }

    pub(in crate::working_memory) fn publish(
        &mut self,
        scope: &WorkingMemoryFundingScope,
    ) -> Result<(), WorkingMemoryError> {
        self.publish_impl(Some(scope), None, None)
    }

    pub(in crate::working_memory) fn publish_original(
        &mut self,
        scope: &WorkingMemoryFundingScope,
        controls: &crate::working_memory::OriginalTextControlGuard,
        identity: &funding::native_partition::NativePublicationScopeIdentity,
    ) -> Result<(), WorkingMemoryError> {
        self.publish_impl(Some(scope), Some((controls, identity)), None)
    }

    pub(in crate::working_memory) fn publish_source(
        &mut self,
        controls: &crate::working_memory::OriginalHostSourceCustody,
        reservation: Option<&crate::working_memory::WorkingMemoryReservation>,
    ) -> Result<(), WorkingMemoryError> {
        if !self.partition.immutable() {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        self.publish_impl(controls.text_scope()?, None, Some((controls, reservation)))
    }

    fn publish_impl(
        &mut self,
        scope: Option<&WorkingMemoryFundingScope>,
        controls: Option<(
            &crate::working_memory::OriginalTextControlGuard,
            &funding::native_partition::NativePublicationScopeIdentity,
        )>,
        source: Option<(
            &crate::working_memory::OriginalHostSourceCustody,
            Option<&crate::working_memory::WorkingMemoryReservation>,
        )>,
    ) -> Result<(), WorkingMemoryError> {
        self.failure_site = "registry terminal attempt";
        if self.terminal {
            return Err(WorkingMemoryError::PreparationAlreadyStarted);
        }
        self.terminal = true;
        // Retain source accounting outside Usage. The source producer uses the
        // same publication transaction without fabricating a native scope.
        let source_account = source.map(|(controls, _)| controls.accounting());
        let numerical = self.partition.numerical();
        let (pool, account) = match source_account.as_ref() {
            Some(value) => (value.pool(), value.account()),
            None if numerical.is_some() => {
                let value = numerical.expect("numerical source");
                (value.pool(), value.account_id())
            }
            None => {
                let scope = scope.ok_or(WorkingMemoryError::IdentityMismatch)?;
                (scope.pool(), scope.id)
            }
        };
        let registration_account = match source {
            Some((controls, _)) => controls.funded_registration_account(),
            None if numerical.is_some() => None,
            None => Some(account),
        };
        // All provider clones and output shells are prepared outside Usage.
        // The complete input vector remains owned on partial clone failure.
        self.failure_site = "registry duplicate input coverage";
        if self.placements.len() != self.inputs.len() {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        for (first_input, input) in self.inputs.iter().enumerate() {
            let placement = &self.placements[first_input];
            placement.validate(pool.topology())?;
            let (key, bytes, origin, existing_only, immutable, source) = input.parts();
            let completed = match input {
                NativePublicationInput::CompletedNumerical(_, _, account) => Some(account),
                _ => None,
            };
            let source_inventory = matches!(input, NativePublicationInput::SourceInventory(..));
            let registered_source = matches!(input, NativePublicationInput::RegisteredSource(..));
            let existing_physical = matches!(input, NativePublicationInput::ExistingPhysical(..));
            if let Some(prior) = self.rows.iter_mut().find(|row| {
                row.key.as_ref().expect("staged key").as_ref().cmp(key) == Ordering::Equal
            }) {
                same_capacity(prior.bytes, bytes)?;
                if prior.placement != *placement {
                    return Err(WorkingMemoryError::IdentityMismatch);
                }
                let prior_native = prior.origin.is_some() || prior.existing_only;
                let incoming_native = origin.is_some() || existing_only;
                if !match (prior.completed.as_ref(), completed) {
                    (None, None) => true,
                    (Some(a), Some(b)) => a.same_account(b),
                    _ => false,
                } || prior_native != incoming_native
                    || prior.immutable != immutable
                    || prior.source != source
                    || prior.registered_source != registered_source
                    || prior.existing_physical != existing_physical
                {
                    return Err(WorkingMemoryError::IdentityMismatch);
                }
                if let (Some(a), Some(b)) = (prior.origin.as_ref(), origin) {
                    if !a.same_origin(b) {
                        return Err(WorkingMemoryError::IdentityMismatch);
                    }
                }
                if prior.origin.is_none() {
                    prior.origin = origin.cloned();
                }
                prior.existing_only |= existing_only;
                // Every occurrence must be an authenticated explicit source.
                prior.source_inventory &= source_inventory;
                continue;
            }
            let key = Arc::new(key.clone());
            let keys = if completed.is_some() {
                Vec::new()
            } else if self.exact_storage {
                let mut keys = crate::working_memory::qualified_storage::vector(1, true)?;
                keys.push(key.as_ref().clone());
                keys
            } else {
                vec![key.as_ref().clone()]
            };
            self.rows.push(Row {
                first_input,
                registry_key: Some(RegistryKey::Shared(key.clone())),
                output: Some(match self.partition.preparation() {
                    Some(host) => WorkingMemoryStorage::pending_prepared(keys, bytes, host),
                    None => WorkingMemoryStorage::pending(keys, bytes),
                }),
                key: Some(key),
                bytes,
                placement: Arc::clone(placement),
                existing_only,
                existing_physical,
                source_inventory,
                registered_source,
                immutable,
                source,
                debit: false,
                native_debit: false,
                funding_allowance_bytes: 0,
                origin: origin.cloned(),
                completed: completed.cloned(),
                locator: None,
                activation_pool: Some(pool.clone()),
            });
        }
        self.failure_site = "registry output ownership";
        for row in &mut self.rows {
            Arc::get_mut(&mut row.output.as_mut().expect("prepared output").0)
                .ok_or(WorkingMemoryError::IdentityMismatch)?;
        }
        self.failure_site = "registry publisher authority";
        let mut usage = pool
            .0
            .usage
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        pool.0.check_host_increment(&usage, 0)?;
        if let Some((controls, identity)) = controls {
            let scope = scope.ok_or(WorkingMemoryError::IdentityMismatch)?;
            if !scope
                .native_publication_identity
                .as_ref()
                .is_some_and(|actual| actual.same(identity))
            {
                return Err(WorkingMemoryError::IdentityMismatch);
            }
            controls
                .custody
                .validate_native_publication_locked(scope, &usage)?;
        }
        if let Some((controls, reservation)) = source {
            let accounting = source_account.as_ref().expect("source accounting retained");
            if !self.partition.same_source_account(accounting) {
                return Err(WorkingMemoryError::IdentityMismatch);
            }
            controls.validate_publication_locked(reservation, pool, &usage)?;
            // Text sources preserve their exact original source-scope check.
            // Speculative sources retain the accepted role account itself.
            if let Some(scope) = scope {
                self.partition.validate_publisher(scope, &usage)?;
            }
        } else if let Some(numerical) = numerical {
            numerical.validate_copy_source(pool, &usage)?;
        } else {
            if self.partition.immutable() {
                return Err(WorkingMemoryError::IdentityMismatch);
            }
            self.partition
                .validate_publisher(scope.ok_or(WorkingMemoryError::IdentityMismatch)?, &usage)?;
        }
        let state = registration_account
            .map(|id| {
                usage
                    .funding
                    .get(&id)
                    .ok_or(WorkingMemoryError::ExecutionFenced)
            })
            .transpose()?;
        if source.is_none() && numerical.is_none() {
            state
                .expect("funded publisher")
                .validate_native_publication(scope.ok_or(WorkingMemoryError::IdentityMismatch)?)?;
        }
        if registration_account.is_none()
            && numerical.is_none()
            && self.rows.iter().any(|row| {
                !row.immutable
                    || row.existing_only
                    || row.source
                    || row.origin.as_ref().is_none_or(|origin| {
                        !origin.immutable() || !self.partition.same_origin(origin)
                    })
            })
        {
            // This closed ticket-only route publishes actual immutable births
            // already paid by its source bank. No ordinary allocation or native
            // partition may borrow the ticket's protected remainder.
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        // Missing namespace is validated against the private empty candidate.
        // Existing-only observations still require a real canonical row. No
        // namespace becomes visible until every row/counter check succeeds.
        let missing_namespace = usage.storage.get(&TypeId::of::<K>()).is_none();
        let registry = if missing_namespace {
            self.namespace
                .as_ref()
                .ok_or(WorkingMemoryError::PreparationAlreadyStarted)?
                .registry::<K>()
        } else {
            usage
                .storage
                .get(&TypeId::of::<K>())
                .and_then(|value| value.downcast_ref::<Registry<K>>())
                .ok_or(WorkingMemoryError::IdentityMismatch)?
        };
        if let Some(numerical) = numerical {
            numerical::validate_rows(numerical, registry, &self.rows)?;
        }
        for row in &self.rows {
            if let Some(completed) = &row.completed {
                self.failure_site = "completed numerical account and placement";
                completed.validate_copy_source(pool, &usage)?;
                completed.validate_completed_allocations(
                    self.rows
                        .iter()
                        .filter(|other| {
                            other
                                .completed
                                .as_ref()
                                .is_some_and(|account| account.same_account(completed))
                        })
                        .map(|other| (other.bytes, other.placement.as_ref())),
                )?;
                // A canonical row for this identity would claim different
                // provenance. Never silently combine the two classifications.
                if registry
                    .locate(row.key.as_ref().expect("staged key").as_ref())
                    .is_some()
                {
                    return Err(WorkingMemoryError::IdentityMismatch);
                }
            }
        }
        let mut allocations = 0usize;
        let mut new_rows = 0usize;
        for row in &mut self.rows {
            if row.completed.is_some() {
                continue;
            }
            self.failure_site = "registry origin account and capacity";
            if let Some(origin) = &row.origin {
                origin.validate_pool(pool, &usage)?;
                origin.validate_capacity(row.bytes)?;
                let expected = match origin {
                    PrepaidStorageOrigin::Native(partition) => partition.placement(),
                    PrepaidStorageOrigin::Numerical(origin) => &origin.placement,
                    _ => &pool.0.host_placement,
                };
                let covered = match expected.kind() {
                    eredu_core::MemoryPlacementKind::Fixed(_) => row.placement == *expected,
                    eredu_core::MemoryPlacementKind::Possible { .. } => row
                        .placement
                        .domains()
                        .iter()
                        .all(|domain| expected.domains().contains(domain)),
                };
                if !covered {
                    return Err(WorkingMemoryError::IdentityMismatch);
                }
            }
            if let Some((locator, entry)) =
                registry.locate(row.key.as_ref().expect("staged key").as_ref())
            {
                self.failure_site = "registry canonical capacity and origin";
                same_capacity(entry.bytes, row.bytes)?;
                if entry.placement != row.placement {
                    return Err(WorkingMemoryError::IdentityMismatch);
                }
                validate_entry_origin(entry, &usage)?;
                if row.registered_source {
                    self.failure_site = "registry ordinary source alias origin";
                    // Recheck under the same commit lock: the source may have
                    // retired since preflight. Preserve its full ordinary charge.
                    // Neither a native witness nor constructor coverage can be
                    // substituted for this explicitly selected existing source.
                    if entry.owners == 0
                        || entry.prepaid.is_some()
                        || entry.funding.is_some()
                        || row.origin.is_some()
                    {
                        return Err(WorkingMemoryError::IdentityMismatch);
                    }
                } else if row.existing_physical {
                    self.failure_site = "registry canonical physical owner";
                    // The borrowed physical witness proves the same live owner,
                    // not its payer. Exact capacity and origin health were checked
                    // above. Preserve the canonical row and its full charge;
                    // only prepaid accounting custody needs an additional alias.
                    if let Some(canonical) = entry.prepaid.as_ref() {
                        canonical.validate_pool(pool, &usage)?;
                        canonical.validate_capacity(row.bytes)?;
                        row.origin = Some(canonical.clone());
                    }
                } else if row.source && entry.prepaid.is_none() && row.origin.is_some() {
                    self.failure_site = "registry fully charged source alias";
                    // The source identity was projected from this exact key by
                    // its selected mechanism before staging. Preserve the full
                    // ordinary charge and retain its genuine constructor too.
                    if entry.owners == 0
                        || entry.funding.is_some()
                        || row
                            .origin
                            .as_ref()
                            .is_none_or(|origin| origin.residual_source_charge().is_none())
                    {
                        return Err(WorkingMemoryError::IdentityMismatch);
                    }
                } else if row.existing_only {
                    self.failure_site = "registry canonical alias kind";
                    let canonical = entry
                        .prepaid
                        .as_ref()
                        .ok_or(WorkingMemoryError::IdentityMismatch)?;
                    canonical.validate_pool(pool, &usage)?;
                    canonical.validate_capacity(row.bytes)?;
                    if if row.source {
                        canonical.residual_source_charge().is_none()
                    } else {
                        !canonical.buffer_kind(row.immutable)
                    } {
                        return Err(WorkingMemoryError::IdentityMismatch);
                    }
                    if let Some(claimed) = &row.origin {
                        if !claimed.same_origin(canonical) {
                            return Err(WorkingMemoryError::IdentityMismatch);
                        }
                    } else {
                        // Clone only accounting custody into an empty field.
                        // It stays row-last through later validation refusal or
                        // provider unwind; no active alias retires under Usage.
                        row.origin = Some(canonical.clone());
                    }
                } else if row.source_inventory
                    && entry
                        .prepaid
                        .as_ref()
                        .is_some_and(|origin| origin.buffer_kind(true))
                {
                    // This is reuse only: the real source constructor already
                    // created and attached the canonical origin. Validate the
                    // exact live owner and publisher; no key/bytes pair can mint
                    // a new prepaid row or borrow another request's allowance.
                    let canonical = entry.prepaid.as_ref().expect("checked immutable origin");
                    canonical.validate_pool(pool, &usage)?;
                    canonical.validate_capacity(row.bytes)?;
                    canonical.validate_publisher(
                        scope.ok_or(WorkingMemoryError::IdentityMismatch)?,
                        &usage,
                    )?;
                    row.origin = Some(canonical.clone());
                    row.immutable = true;
                } else if !same_coverage(entry.prepaid.as_ref(), row.origin.as_ref()) {
                    self.failure_site = "registry canonical coverage mismatch";
                    return Err(WorkingMemoryError::IdentityMismatch);
                }
                entry
                    .owners
                    .checked_add(1)
                    .ok_or(WorkingMemoryError::Overflow)?;
                row.locator = Some(locator);
            } else {
                if row.existing_only {
                    self.missing_existing_input = Some(row.first_input);
                    self.failure_site = "registry immutable alias missing";
                    if !row.immutable {
                        self.failure_site = "registry native alias missing";
                    }
                    if row.existing_physical {
                        self.failure_site = "registry physical alias missing";
                    }
                    return Err(WorkingMemoryError::IdentityMismatch);
                }
                new_rows = new_rows
                    .checked_add(1)
                    .ok_or(WorkingMemoryError::Overflow)?;
                if let Some(origin) = &row.origin {
                    // An unpublished foreign birth cannot be relabeled by B.
                    if !self.partition.same_origin(origin) {
                        return Err(WorkingMemoryError::IdentityMismatch);
                    }
                } else {
                    allocations = allocations
                        .checked_add(1)
                        .ok_or(WorkingMemoryError::Overflow)?;
                }
            }
        }
        self.failure_site = "registry duplicate physical location";
        for (index, row) in self.rows.iter().enumerate() {
            if row.locator.is_some()
                && self.rows[..index]
                    .iter()
                    .any(|other| other.locator == row.locator)
            {
                return Err(WorkingMemoryError::IdentityMismatch);
            }
        }
        self.failure_site = "registry final accounting";
        // Scan only immutable prepared scalar facts. This avoids a second
        // unpriced transaction vector; validation precedes every slot mutation.
        for (slot, (domain, _)) in pool.topology().domains().enumerate() {
            let incremental = self
                .rows
                .iter()
                .filter(|row| {
                    row.completed.is_none()
                        && row.locator.is_none()
                        && (row.origin.is_none()
                            || matches!(&row.origin, Some(PrepaidStorageOrigin::Native(_))))
                        && row.placement.domains().contains(&domain)
                })
                .try_fold(0u64, |sum, row| {
                    sum.checked_add(row.bytes)
                        .ok_or(WorkingMemoryError::Overflow)
                })?;
            let native_incremental = self
                .rows
                .iter()
                .filter(|row| {
                    row.completed.is_none()
                        && row.locator.is_none()
                        && matches!(&row.origin, Some(PrepaidStorageOrigin::Native(_)))
                        && row.placement.domains().contains(&domain)
                })
                .try_fold(0u64, |sum, row| {
                    sum.checked_add(row.bytes)
                        .ok_or(WorkingMemoryError::Overflow)
                })?;
            let ordinary_incremental = incremental
                .checked_sub(native_incremental)
                .ok_or(WorkingMemoryError::Overflow)?;
            match state {
                Some(state) => {
                    state.domains[slot]
                        .native_registered
                        .checked_add(native_incremental)
                        .ok_or(WorkingMemoryError::Overflow)?;
                    if native_incremental != 0 {
                        state.domains[slot]
                            .native_held
                            .ok_or(WorkingMemoryError::IdentityMismatch)?
                            .checked_sub(native_incremental)
                            .ok_or(WorkingMemoryError::IdentityMismatch)?;
                    }
                    let available = if slot == pool.0.host_slot {
                        state.spendable_remaining()?
                    } else {
                        state.domains[slot]
                            .remaining
                            .checked_sub(state.domains[slot].native_held.unwrap_or(0))
                            .ok_or(WorkingMemoryError::Poisoned)?
                    };
                    if ordinary_incremental > available {
                        return Err(WorkingMemoryError::DomainAllowanceExceeded {
                            domain,
                            required_bytes: ordinary_incremental,
                            available_bytes: available,
                        });
                    }
                    state.domains[slot]
                        .remaining
                        .checked_sub(incremental)
                        .ok_or(WorkingMemoryError::Poisoned)?;
                    let placement_allowance = self
                        .rows
                        .iter()
                        .filter(|row| {
                            row.completed.is_none()
                                && row.locator.is_none()
                                && (row.origin.is_none()
                                    || matches!(&row.origin, Some(PrepaidStorageOrigin::Native(_))))
                                && row.placement.domains().contains(&domain)
                                && matches!(
                                    row.placement.kind(),
                                    eredu_core::MemoryPlacementKind::Possible { .. }
                                )
                        })
                        .map(|row| row.bytes)
                        .sum::<u64>();
                    let fixed_bytes = incremental
                        .checked_sub(placement_allowance)
                        .ok_or(WorkingMemoryError::Overflow)?;
                    let native_fixed = self
                        .rows
                        .iter()
                        .filter(|row| {
                            row.completed.is_none()
                                && row.locator.is_none()
                                && row.placement.domains().contains(&domain)
                                && matches!(
                                    row.placement.kind(),
                                    eredu_core::MemoryPlacementKind::Fixed(_)
                                )
                                && matches!(&row.origin, Some(PrepaidStorageOrigin::Native(_)))
                        })
                        .map(|row| row.bytes)
                        .sum::<u64>();
                    let held_fixed = state.domains[slot]
                        .native_held
                        .unwrap_or(0)
                        .checked_sub(state.domains[slot].native_held_allowance)
                        .ok_or(WorkingMemoryError::Poisoned)?;
                    let native_converted = if native_fixed > held_fixed {
                        native_fixed - held_fixed
                    } else {
                        0
                    };
                    let converted = state
                        .allocation_allowance(
                            slot,
                            fixed_bytes,
                            placement_allowance,
                            native_incremental,
                        )?
                        .max(native_converted);
                    state.domains[slot]
                        .remaining_charge
                        .placement_allowance_bytes
                        .checked_sub(placement_allowance)
                        .and_then(|bytes| bytes.checked_sub(converted))
                        .ok_or(WorkingMemoryError::IdentityMismatch)?;
                    usage.domains[slot]
                        .placement_allowances
                        .checked_sub(converted)
                        .ok_or(WorkingMemoryError::Poisoned)?;
                    let mut remaining_conversion = converted;
                    let mut native_conversion = native_converted;
                    for native_first in [true, false] {
                        for row in self.rows.iter_mut().filter(|row| {
                            row.completed.is_none()
                                && row.locator.is_none()
                                && (row.origin.is_none()
                                    || matches!(&row.origin, Some(PrepaidStorageOrigin::Native(_))))
                                && row.placement.domains().contains(&domain)
                                && matches!(
                                    row.placement.kind(),
                                    eredu_core::MemoryPlacementKind::Fixed(_)
                                )
                        }) {
                            let native =
                                matches!(&row.origin, Some(PrepaidStorageOrigin::Native(_)));
                            if native != native_first {
                                continue;
                            }
                            row.funding_allowance_bytes = if native {
                                row.bytes.min(native_conversion)
                            } else {
                                row.bytes.min(remaining_conversion)
                            };
                            if native {
                                native_conversion -= row.funding_allowance_bytes;
                            }
                            remaining_conversion -= row.funding_allowance_bytes;
                        }
                    }
                    if remaining_conversion != 0 {
                        return Err(WorkingMemoryError::IdentityMismatch);
                    }
                }
                None if incremental != 0 => return Err(WorkingMemoryError::IdentityMismatch),
                None => {}
            }
            usage.domains[slot]
                .reserved
                .checked_sub(incremental)
                .ok_or(WorkingMemoryError::Poisoned)?;
            usage.domains[slot]
                .registered
                .checked_add(incremental)
                .ok_or(WorkingMemoryError::Overflow)?;
        }
        let (allocations, registrations) = match state {
            Some(state) => (
                state
                    .allocations
                    .checked_add(allocations)
                    .ok_or(WorkingMemoryError::Overflow)?,
                state
                    .registrations
                    .checked_add(self.rows.len())
                    .ok_or(WorkingMemoryError::Overflow)?,
            ),
            None if allocations != 0 => return Err(WorkingMemoryError::IdentityMismatch),
            None => (0, 0),
        };
        for row in &mut self.rows {
            row.native_debit = row.completed.is_none()
                && row.locator.is_none()
                && matches!(&row.origin, Some(PrepaidStorageOrigin::Native(_)));
            row.debit = row.completed.is_none()
                && row.locator.is_none()
                && (row.origin.is_none() || row.native_debit);
        }
        let node = self
            .node
            .as_mut()
            .ok_or(WorkingMemoryError::PreparationAlreadyStarted)?;
        if missing_namespace && new_rows != 0 {
            usage
                .storage
                .install(self.namespace.take().expect("validated candidate"));
        }
        if self.rows.iter().any(|row| row.completed.is_none()) {
            let registry = usage
                .storage
                .get_mut(&TypeId::of::<K>())
                .and_then(|value| value.downcast_mut::<Registry<K>>())
                .expect("validated registry");
            // Commit uses only prevalidated locators, fixed slots and scalar writes.
            for (index, row) in self.rows.iter_mut().enumerate() {
                if row.completed.is_some() {
                    continue;
                }
                if let Some(locator) = row.locator {
                    registry.at_mut(locator).owners += 1;
                } else {
                    let funding = row.origin.is_none().then_some(account);
                    node.entries[index] = Some((
                        row.registry_key.take().expect("prepared registry key"),
                        Entry {
                            reset_layout_id: None,
                            funding_allowance_bytes: row.funding_allowance_bytes,
                            native_retired: false,
                            pending_allocation: false,
                            bytes: row.bytes,
                            placement: Arc::clone(&row.placement),
                            owners: 1,
                            funding,
                            prepaid: row.origin.take(),
                        },
                    ));
                }
            }
            if new_rows != 0 {
                registry.link(self.node.take().expect("prepared fixed batch"));
            }
        }
        if let Some(id) = registration_account {
            let state = usage
                .funding
                .get_mut(&id)
                .expect("validated funded publisher");

            state.allocations = allocations;
            state.registrations = registrations;
        }
        for (slot, (domain, _)) in pool.topology().domains().enumerate() {
            let incremental = self
                .rows
                .iter()
                .filter(|row| row.debit && row.placement.domains().contains(&domain))
                .map(|row| row.bytes)
                .sum::<u64>();
            let placement_allowance = self
                .rows
                .iter()
                .filter(|row| {
                    row.debit
                        && row.placement.domains().contains(&domain)
                        && matches!(
                            row.placement.kind(),
                            eredu_core::MemoryPlacementKind::Possible { .. }
                        )
                })
                .map(|row| row.bytes)
                .sum::<u64>();
            let converted = self
                .rows
                .iter()
                .filter(|row| row.debit && row.placement.domains().contains(&domain))
                .map(|row| row.funding_allowance_bytes)
                .sum::<u64>();
            let native_incremental = self
                .rows
                .iter()
                .filter(|row| row.native_debit && row.placement.domains().contains(&domain))
                .map(|row| row.bytes)
                .sum::<u64>();
            let native_allowance = self
                .rows
                .iter()
                .filter(|row| row.native_debit && row.placement.domains().contains(&domain))
                .map(|row| {
                    if matches!(
                        row.placement.kind(),
                        eredu_core::MemoryPlacementKind::Possible { .. }
                    ) {
                        row.bytes
                    } else {
                        row.funding_allowance_bytes
                    }
                })
                .sum::<u64>();
            if let Some(id) = registration_account {
                let balance = &mut usage
                    .funding
                    .get_mut(&id)
                    .expect("validated funding")
                    .domains[slot];
                balance.remaining -= incremental;
                balance.native_registered += native_incremental;
                balance.native_held_allowance -= native_allowance;
                if native_incremental != 0 {
                    *balance
                        .native_held
                        .as_mut()
                        .expect("validated native partition") -= native_incremental;
                }
                balance.remaining_charge.placement_allowance_bytes -=
                    placement_allowance + converted;
            }
            usage.domains[slot].placement_allowances -= converted;
            usage.domains[slot].reserved -= incremental;
            usage.domains[slot].registered += incremental;
        }
        for row in &mut self.rows {
            let output = Arc::get_mut(&mut row.output.as_mut().expect("prepared output").0)
                .expect("private registration");
            output.pool = row.activation_pool.take();
            output.funding = registration_account;
            output.completed_numerical = row.completed.take();
        }
        drop(usage);
        // Duplicate and original provider keys retire before outputs may escape.
        self.inputs.clear();
        for row in &mut self.rows {
            drop(row.registry_key.take());
            drop(row.key.take());
        }
        self.published = true;
        Ok(())
    }

    pub(in crate::working_memory) fn take(
        &mut self,
        index: usize,
    ) -> Option<WorkingMemoryStorage<K>> {
        self.published.then_some(())?;
        self.rows.get_mut(index)?.output.take()
    }

    pub(in crate::working_memory) fn prepare_slots(
        partition: NativePartition,
        slots: usize,
    ) -> Self {
        Self {
            inputs: Vec::with_capacity(slots),
            placements: Vec::with_capacity(slots),
            rows: Vec::with_capacity(slots),
            node: Some(RegistryBatch::prepare_native(slots, partition.clone())),
            namespace: Some(PreparedNamespace::prepare::<K>(
                partition.namespace_metadata(),
            )),
            terminal: false,
            published: false,
            slots,
            exact_storage: false,
            failure_site: "registry preparation",
            missing_existing_input: None,
            partition: PublicationOrigin::Prepaid(PrepaidStorageOrigin::Native(partition)),
        }
    }

    pub(in crate::working_memory) fn prepare_slots_exact(
        partition: NativePartition,
        slots: usize,
    ) -> Result<Self, WorkingMemoryError> {
        // The parameter retains original custody until every partial Vec/Box
        // has retired on failure. No provider callback or Usage loan occurs.
        let inputs = crate::working_memory::qualified_storage::vector(slots, true)?;
        let rows = crate::working_memory::qualified_storage::vector(slots, true)?;
        let placements = crate::working_memory::qualified_storage::vector(slots, true)?;
        let node = RegistryBatch::prepare_native_exact(slots, partition.clone())?;
        Ok(Self {
            inputs,
            placements,
            rows,
            node: Some(node),
            namespace: Some(PreparedNamespace::prepare::<K>(
                partition.namespace_metadata(),
            )),
            terminal: false,
            published: false,
            slots,
            exact_storage: true,
            failure_site: "registry preparation",
            missing_existing_input: None,
            partition: PublicationOrigin::Prepaid(PrepaidStorageOrigin::Native(partition)),
        })
    }

    pub(in crate::working_memory) fn prepare_source_exact(
        origin: PrepaidHostOrigin,
    ) -> Result<Self, WorkingMemoryError> {
        let inputs = crate::working_memory::qualified_storage::vector(1, true)?;
        let rows = crate::working_memory::qualified_storage::vector(1, true)?;
        let placements = crate::working_memory::qualified_storage::vector(1, true)?;
        let node = RegistryBatch::prepare_source_exact(1, origin.raw().clone())?;
        Ok(Self {
            inputs,
            placements,
            rows,
            node: Some(node),
            namespace: Some(PreparedNamespace::prepare_source::<K>(Some(
                origin.raw().clone(),
            ))),
            terminal: false,
            published: false,
            slots: 1,
            exact_storage: true,
            failure_site: "registry preparation",
            missing_existing_input: None,
            partition: PublicationOrigin::Prepaid(PrepaidStorageOrigin::Immutable(origin)),
        })
    }
    // Only the source worker holding the same successful attempt can supply
    // its mechanism's actual completed-copy identity/capacity here.
    pub(in crate::working_memory) fn push_source_birth(
        &mut self,
        key: K,
        bytes: u64,
    ) -> Result<(), WorkingMemoryError> {
        if !self.partition.immutable() || self.terminal || !self.inputs.is_empty() {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        self.partition.validate_capacity(bytes)?;
        self.placements.push(Arc::clone(
            &self.partition.source_host_pool()?.0.host_placement,
        ));
        self.inputs
            .push(NativePublicationInput::Native(NativeStorageWitness {
                key,
                bytes,
                origin: self.partition.prepaid()?.clone(),
            }));
        Ok(())
    }

    pub(in crate::working_memory) fn qualified_control_bytes(
        slots: usize,
        nested_key_bytes: u64,
    ) -> Result<u64, WorkingMemoryError> {
        use crate::working_memory::qualified_storage as storage;
        // The legacy requested bound includes both buffers at Vec-to-Box.
        // Exact native storage never makes that transition: retain one buffer.
        let requested = Self::requested_control_bytes(slots)?
            .checked_sub(storage::array_bytes::<Option<(RegistryKey<K>, Entry)>>(
                slots,
            )?)
            .ok_or(WorkingMemoryError::Overflow)?;
        let frames = storage::vector_control_bytes::<NativePublicationInput<K>>()?
            .checked_add(storage::vector_control_bytes::<
                Arc<eredu_core::MemoryPlacement>,
            >()?)
            .ok_or(WorkingMemoryError::Overflow)?
            .checked_add(storage::vector_control_bytes::<Row<K>>()?)
            .and_then(|n| {
                storage::vector_control_bytes::<Option<(RegistryKey<K>, Entry)>>()
                    .ok()
                    .and_then(|b| n.checked_add(b))
            })
            .ok_or(WorkingMemoryError::Overflow)?;
        let key_frame = storage::vector_control_bytes::<K>()?;
        let headers = storage::shared_header_bytes::<K>()?
            .checked_add(storage::shared_header_bytes::<Registration<K>>()?)
            .and_then(|n| n.checked_add(key_frame))
            // Input key, shared canonical key, registration key. Observation
            // describes into the input; it does not create a fourth live copy.
            .and_then(|n| {
                nested_key_bytes
                    .checked_mul(3)
                    .and_then(|keys| n.checked_add(keys))
            })
            .and_then(|n| n.checked_mul(u64::try_from(slots).ok()?))
            .ok_or(WorkingMemoryError::Overflow)?;
        // publish_impl retains this account alias and its source/scope
        // selectors through canonical commit. They are separate live caller
        // values, not fields of the requested registry/output destinations.
        let publication_frames = [
            size_of::<Option<&crate::working_memory::OriginalNumericalBudgetCustody>>(),
            size_of::<Option<u64>>(),
            size_of::<bool>(), // reached existing-physical classification
            size_of::<Option<&PrepaidStorageOrigin>>(), // canonical accounting loan
            size_of::<Option<&funding::FundingState>>(),
            size_of::<(u64, usize, usize)>(),
            size_of::<Option<crate::working_memory::OriginalHostMetadataCustody>>(),
            size_of::<Option<&WorkingMemoryFundingScope>>(),
            size_of::<
                Option<(
                    &crate::working_memory::OriginalTextControlGuard,
                    &funding::native_partition::NativePublicationScopeIdentity,
                )>,
            >(),
            size_of::<
                Option<(
                    &crate::working_memory::OriginalHostSourceCustody,
                    Option<&crate::working_memory::WorkingMemoryReservation>,
                )>,
            >(),
            size_of::<(&MemoryLedger, u64)>(),
            size_of::<Result<(), WorkingMemoryError>>(),
        ];
        let publication_frames = publication_frames
            .into_iter()
            .try_fold(
                std::mem::size_of_val(&publication_frames),
                usize::checked_add,
            )
            .and_then(|n| u64::try_from(n).ok())
            .ok_or(WorkingMemoryError::Overflow)?;
        requested
            .checked_add(headers)
            .and_then(|n| n.checked_add(frames))
            .and_then(|n| n.checked_add(publication_frames))
            .and_then(|n| {
                MemoryLedger::retained_source_inventory_control_bytes::<K>()
                    .and_then(|bytes| u64::try_from(bytes).ok())
                    .and_then(|bytes| n.checked_add(bytes))
            })
            .ok_or(WorkingMemoryError::Overflow)
    }

    #[cfg(test)]
    pub(in crate::working_memory) fn push_source(
        &mut self,
        key: &K,
        bytes: u64,
        identity: Option<&eredu_checkpoint::store::SourceStorageIdentity>,
        pool: &MemoryLedger,
    ) -> Result<usize, WorkingMemoryError>
    where
        K: crate::working_memory::GgufSourceStorageKey,
    {
        if key.gguf_source_identity() != identity {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        self.push_placed_source(
            key,
            bytes,
            |key| key.gguf_source_identity(),
            pool,
            pool.host_placement_handle(),
        )
    }

    pub(in crate::working_memory) fn push_placed_source<'a>(
        &mut self,
        key: &'a K,
        bytes: u64,
        project_identity: impl FnOnce(
            &'a K,
        )
            -> Option<&'a eredu_checkpoint::store::SourceStorageIdentity>,
        pool: &MemoryLedger,
        placement: Arc<eredu_core::MemoryPlacement>,
    ) -> Result<usize, WorkingMemoryError> {
        if self.terminal || self.inputs.len() == self.slots {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        placement.validate(pool.topology())?;
        // Identity comes from the exact key, never an independent equal-sized
        // source selected by the caller.
        let identity = project_identity(key);
        // Authenticate the same owning source before the transaction lock.
        // No native partition is substituted for its already admitted account.
        let origin = match identity {
            Some(identity) => crate::working_memory::gguf_source::SourceInventoryOrigin::inspect(
                identity, bytes, pool,
            )?,
            None => None,
        };
        let registered_source = identity.is_some() && origin.is_none();
        if registered_source {
            // A foreign original constructor is not an ordinary load source.
            if crate::working_memory::gguf_source::SourceInventoryOrigin::has_original_constructor(
                identity.expect("source identity checked"),
            ) {
                return Err(WorkingMemoryError::IdentityMismatch);
            }
            pool.validate_registered_ordinary_source(key, bytes)?;
        }
        let key = key.clone();
        let input = match origin {
            Some(origin) => NativePublicationInput::ExistingSource(NativeStorageWitness {
                key,
                bytes,
                origin: PrepaidStorageOrigin::Source(origin),
            }),
            None if registered_source => NativePublicationInput::RegisteredSource(key, bytes),
            None => NativePublicationInput::SourceInventory(key, bytes),
        };
        let index = self.inputs.len();
        self.inputs.push(input);
        self.placements.push(placement);
        Ok(index)
    }

    pub(in crate::working_memory) fn push_placed_observation(
        &mut self,
        observation: crate::working_memory::NativeStorageObservation<K>,
        placement: Arc<eredu_core::MemoryPlacement>,
    ) -> Result<usize, WorkingMemoryError> {
        use crate::working_memory::NativeStorageObservation;
        if self.terminal || self.inputs.len() == self.slots {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let input = match observation {
            NativeStorageObservation::Originating(key, bytes) => {
                NativePublicationInput::Native(NativeStorageWitness {
                    key,
                    bytes,
                    origin: self.partition.prepaid()?.clone(),
                })
            }
            NativeStorageObservation::Existing(key, bytes) => {
                NativePublicationInput::Existing(ExistingNativeAlias {
                    key,
                    bytes,
                    immutable: false,
                })
            }
            NativeStorageObservation::ExistingImmutable(key, bytes) => {
                NativePublicationInput::Existing(ExistingNativeAlias {
                    key,
                    bytes,
                    immutable: true,
                })
            }
            NativeStorageObservation::ExistingPhysical(key, bytes) => {
                NativePublicationInput::ExistingPhysical(key, bytes)
            }
            NativeStorageObservation::CompletedNumerical(key, bytes, account) => {
                NativePublicationInput::CompletedNumerical(key, bytes, account)
            }
            NativeStorageObservation::Ordinary(key, bytes) => {
                NativePublicationInput::Ordinary(key, bytes)
            }
            NativeStorageObservation::Empty => return Err(WorkingMemoryError::IdentityMismatch),
        };
        let index = self.inputs.len();
        self.inputs.push(input);
        self.placements.push(placement);
        Ok(index)
    }

    #[cfg(test)]
    pub(in crate::working_memory) fn push_observation(
        &mut self,
        observation: crate::working_memory::NativeStorageObservation<K>,
    ) -> Result<usize, WorkingMemoryError> {
        let placement = Arc::clone(&self.partition.fixture_pool()?.0.host_placement);
        self.push_placed_observation(observation, placement)
    }

    #[cfg(test)]
    pub(in crate::working_memory) fn control_capacities(&self) -> [usize; 3] {
        [
            self.inputs.capacity(),
            self.rows.capacity(),
            self.node
                .as_ref()
                .map_or(self.slots, |node| match &node.entries {
                    registry::RegistrySlots::Native(rows) => rows.capacity(),
                    registry::RegistrySlots::Ordinary(rows) => rows.len(),
                }),
        ]
    }

    pub(in crate::working_memory) fn input(
        &self,
        index: usize,
    ) -> Option<&WorkingMemoryStorage<K>> {
        self.published.then_some(())?;
        self.rows
            .iter()
            .find(|row| row.first_input == index)?
            .output
            .as_ref()
    }

    pub(in crate::working_memory) fn take_input(
        &mut self,
        index: usize,
    ) -> Option<WorkingMemoryStorage<K>> {
        self.published.then_some(())?;
        self.rows
            .iter_mut()
            .find(|row| row.first_input == index)?
            .output
            .take()
    }
}

#[cfg(test)]
pub(in crate::working_memory) fn test_native<K>(
    key: K,
    bytes: u64,
    origin: &NativePartition,
) -> NativePublicationInput<K> {
    NativePublicationInput::Native(NativeStorageWitness {
        key,
        bytes,
        origin: PrepaidStorageOrigin::Native(origin.clone()),
    })
}

#[cfg(test)]
pub(in crate::working_memory) fn test_existing_native<K>(
    key: K,
    bytes: u64,
) -> NativePublicationInput<K> {
    NativePublicationInput::Existing(ExistingNativeAlias {
        key,
        bytes,
        immutable: false,
    })
}

#[cfg(test)]
mod tests;
