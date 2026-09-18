//! Atomic registration of retained physical storage in a shared request pool.

use super::{WorkingMemoryError, WorkingMemoryFundingScope, WorkingMemoryPool, funding};
use std::{any::TypeId, borrow::Borrow, cmp::Ordering, collections::BTreeMap, sync::Arc};

pub(in crate::working_memory) mod bounded_pin;
pub(in crate::working_memory) mod bounded_publication;
pub(in crate::working_memory) mod capture_publication;
pub(in crate::working_memory) mod finite_pin;
pub use finite_pin::ExistingStoragePinLayout;
pub(in crate::working_memory) mod prepaid;
pub use capture_publication::{
    CapturePlanPublicationCause, CapturePlanStorageKey, PreparedCapturePlanPublication,
};
mod original_sources;
mod shared_filter;
use original_sources::OriginalSources;
pub use original_sources::{
    OriginalStorageSourcesError, OriginalStorageSourcesLayout, RetainedOriginalStorageSources,
};
mod host_transfer;
pub(in crate::working_memory) mod reset_layout;
pub(super) use host_transfer::{
    publish_dense_host_slots, dense_host_transfer_control_bytes,
};

#[cfg(test)]
mod pin_tests;
#[cfg(test)]
mod tests;

#[derive(Clone, Copy)]
enum InventoryAdmission<'a> {
    Register,
    Funded {
        scope: &'a WorkingMemoryFundingScope,
        registrations: usize,
    },
    Pin,
}

impl InventoryAdmission<'_> {
    fn capacity_matches(self, expected: u64, actual: u64) -> Result<(), WorkingMemoryError> {
        same_capacity(expected, actual).map_err(|error| self.inventory_error(error))
    }

    fn inventory_error(self, error: WorkingMemoryError) -> WorkingMemoryError {
        match (self, error) {
            (Self::Pin, WorkingMemoryError::StorageCapacityMismatch { .. }) => {
                WorkingMemoryError::IdentityMismatch
            }
            (_, error) => error,
        }
    }
}

#[derive(Debug)]
struct Entry {
    // Assigned once when an original reset pins this canonical entry.
    reset_layout_id: Option<u64>,
    bytes: u64,
    owners: usize,
    funding: Option<u64>,
    prepaid: Option<prepaid::PrepaidStorageOrigin>,
}

impl Entry {
    fn charged(&self) -> Option<(u64, Option<u64>)> {
        match &self.prepaid {
            None => Some((self.bytes, self.funding)),
            Some(origin) => origin.residual_source_charge().map(|bytes| (bytes, None)),
        }
    }
}

// Ordinary inventories keep their existing Send-only key contract. The closed
// host-table handoff uses a shared key staged outside both owner locks, so an
// unwinding provider comparison cannot destroy the last key owner inside them.
// Both forms use K's exact ordering and borrowed lookup in the same namespace.
enum RegistryKey<K> {
    Owned(K),
    Shared(Arc<dyn Borrow<K> + Send + Sync>),
}

impl<K> Borrow<K> for RegistryKey<K> {
    fn borrow(&self) -> &K {
        match self {
            Self::Owned(key) => key,
            Self::Shared(key) => key.as_ref().borrow(),
        }
    }
}

impl<K: Ord> PartialEq for RegistryKey<K> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl<K: Ord> Eq for RegistryKey<K> {}

impl<K: Ord> PartialOrd for RegistryKey<K> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<K: Ord> Ord for RegistryKey<K> {
    fn cmp(&self, other: &Self) -> Ordering {
        let key: &K = self.borrow();
        key.cmp(other.borrow())
    }
}

pub(in crate::working_memory) mod directory;
pub(in crate::working_memory) mod native_publication;
pub use native_publication::copy::{PreparedWorkspaceCopyPublication, WorkspaceCopyPublicationPlan};
mod registry;
mod source_inventory;
use registry::{EntryLocator, Registry, RegistryBatch, RegistryEntry};

impl WorkingMemoryPool {
    /// Charges a complete physical inventory before admitting work that uses it.
    /// Equal keys of the same Rust type identify the same backing allocation;
    /// independent inventories and cloned registration handles count it once.
    /// Registration and request admission compete atomically for pool capacity.
    /// A rejected inventory changes neither current usage nor the historical peak.
    ///
    /// The backend supplies certified allocation capacities and keeps identities
    /// valid by retaining their physical owners alongside this registration.
    /// Keys must distinguish independent native/source allocation namespaces and
    /// must not be reused while a registration is live. Every inventory in one
    /// domain must use the same key type. This is accounting for existing payloads,
    /// not permission to allocate new storage or evidence for a future workspace.
    /// The fixed `existing` baseline must be disjoint from registered storage.
    /// Registration may coexist with unquoted leases so a completed operation
    /// can account for its surviving storage before releasing its lease. It
    /// neither clears those admission blockers nor certifies their other work.
    /// It may also coexist with reservations within the shared capacity; callers
    /// must not use registration after allocation as permission to exceed an
    /// earlier reservation's bound.
    pub fn register_storage<K: Clone + Ord + Send + 'static>(
        &self,
        storage: impl IntoIterator<Item = (K, u64)>,
    ) -> Result<WorkingMemoryStorage<K>, WorkingMemoryError> {
        self.storage_grouped(storage, InventoryAdmission::Register)
    }

    /// Pins a complete inventory only if every unique identity is already
    /// registered in this pool with exactly the supplied capacity. Missing keys,
    /// foreign key namespaces and conflicting capacities reject the entire
    /// inventory with `IdentityMismatch`; no partial ownership is published.
    /// Equal keys share one pin, including zero-byte allocations.
    ///
    /// This never charges new bytes or changes reserved balances, historical
    /// peak or capacity ceilings. Pins preserve each allocation's existing
    /// funding origin until its last registration retires; they do not create
    /// another funding account or grant permission to allocate. Keep physical
    /// identities valid as required by [`Self::register_storage`]. Key cloning
    /// happens before accounting is locked; key destruction precedes refunds
    /// outside that lock, including when this pin is the final owner.
    pub fn pin_registered_storage<K: Clone + Ord + Send + 'static>(
        &self,
        storage: impl IntoIterator<Item = (K, u64)>,
    ) -> Result<WorkingMemoryStorage<K>, WorkingMemoryError> {
        let (unique, bytes) = prepare_inventory(storage)
            .map_err(|error| InventoryAdmission::Pin.inventory_error(error))?;
        // Preserve ordinary provider Clone callbacks before accounting is locked.
        // The finite owned-input route moves keys through the same commit worker.
        let keys = unique.keys().cloned().collect();
        let mut registration = WorkingMemoryStorage::pending(keys, bytes);
        let mut rows = finite_pin::ordinals(unique.values().copied(), unique.len(), false)?;
        self.commit_existing_pin(&mut registration, &mut rows)?;
        Ok(registration)
    }

    fn storage_grouped<K: Clone + Ord + Send + 'static>(
        &self,
        storage: impl IntoIterator<Item = (K, u64)>,
        admission: InventoryAdmission<'_>,
    ) -> Result<WorkingMemoryStorage<K>, WorkingMemoryError> {
        let (unique, bytes) =
            prepare_inventory(storage).map_err(|error| admission.inventory_error(error))?;
        // Key cloning is provider code; do it before touching shared accounting.
        let keys = unique.keys().cloned().collect();
        let mut registration = WorkingMemoryStorage::pending(keys, bytes);
        self.commit_storage_inventory(unique, admission)?;
        registration.activate(self.clone(), None);
        Ok(registration)
    }

    /// Atomically charges a complete inventory and returns one independent
    /// registration per unique allocation key. Retiring one handle releases
    /// only that key; surviving handles do not retain unrelated allocations.
    /// Equal keys share capacity with both grouped and individual registrations.
    /// Duplicate keys must have the same certified capacity.
    ///
    /// The identity, physical-lifetime and admission requirements of
    /// [`Self::register_storage`] also apply. Rejection publishes no handles and
    /// changes neither current usage nor historical peak. Provider key cloning
    /// and construction of the result map finish before accounting is locked.
    /// Returned map keys are separate provider-owned identity clones. If keys
    /// themselves pin physical storage, retain a registration until the
    /// corresponding returned map key has also retired.
    pub fn register_storage_individually<K: Clone + Ord + Send + 'static>(
        &self,
        storage: impl IntoIterator<Item = (K, u64)>,
    ) -> Result<BTreeMap<K, WorkingMemoryStorage<K>>, WorkingMemoryError> {
        self.storage_individually(storage, None)
    }

    /// Transfers a complete physical inventory from reserved funding into the
    /// shared storage registry atomically. Only previously unregistered keys
    /// consume the envelope; aliases transfer zero bytes. Total usage and peak
    /// do not increase. Rejection changes neither funding nor storage accounting.
    /// During a stamped span only its exact scope can publish on that account;
    /// sibling inventories reject even when their incremental charge is zero.
    ///
    /// The provider must retain this scope through complete native settlement
    /// and publication. Dropped handles return their allocation's credit to its
    /// originating envelope while that run or its work remains active, including
    /// quarantine after failed publication. After closure and certification,
    /// final physical retirement releases the charge instead. Returned identity
    /// keys obey the same pin-lifetime requirements as ordinary registrations.
    pub fn adopt_storage_individually<K: Clone + Ord + Send + 'static>(
        &self,
        scope: &WorkingMemoryFundingScope,
        storage: impl IntoIterator<Item = (K, u64)>,
    ) -> Result<BTreeMap<K, WorkingMemoryStorage<K>>, WorkingMemoryError> {
        scope.validate_domain(self)?;
        scope.validate_native_purpose()?;
        self.storage_individually(storage, Some(scope))
    }

    fn storage_individually<K: Clone + Ord + Send + 'static>(
        &self,
        storage: impl IntoIterator<Item = (K, u64)>,
        funding: Option<&WorkingMemoryFundingScope>,
    ) -> Result<BTreeMap<K, WorkingMemoryStorage<K>>, WorkingMemoryError> {
        let (unique, _) = prepare_inventory(storage)?;
        let mut registrations = unique
            .iter()
            .map(|(key, bytes)| {
                (
                    key.clone(),
                    WorkingMemoryStorage::pending(vec![key.clone()], *bytes),
                )
            })
            .collect::<BTreeMap<_, _>>();
        let admission = funding.map_or(InventoryAdmission::Register, |scope| {
            InventoryAdmission::Funded {
                scope,
                registrations: registrations.len(),
            }
        });
        self.commit_storage_inventory(unique, admission)?;
        for registration in registrations.values_mut() {
            registration.activate(self.clone(), funding.map(|scope| scope.id));
        }
        Ok(registrations)
    }

    fn commit_storage_inventory<K: Ord + Send + 'static>(
        &self,
        unique: BTreeMap<K, u64>,
        admission: InventoryAdmission<'_>,
    ) -> Result<(), WorkingMemoryError> {
        let (scope, registrations) = match admission {
            InventoryAdmission::Funded {
                scope,
                registrations,
            } => (Some(scope), registrations),
            _ => (None, 0),
        };
        let funding = scope.map(|scope| scope.id);
        // Equal incoming keys are not inserted into the registry. Their Drop
        // implementations may reenter accounting, so retain them until unlocked.
        let mut duplicate_keys = Vec::with_capacity(unique.len());
        let mut usage = self
            .0
            .usage
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        if let Some(id) = funding {
            scope.expect("funded admission").validate_native_purpose()?;
            let state = usage
                .funding
                .get(&id)
                .ok_or(WorkingMemoryError::ExecutionFenced)?;
            if state.native_scopes == 0 {
                return Err(WorkingMemoryError::ExecutionFenced);
            }
            state.validate_span_spend(scope)?;
        }
        if unique.is_empty() {
            return Ok(());
        }
        let registry = usage.storage.get(&TypeId::of::<K>()).map(|registry| {
            registry
                .downcast_ref::<Registry<K>>()
                .expect("typed storage registry")
        });
        let mut incremental = 0u64;
        let mut allocations = 0usize;
        for (key, bytes) in &unique {
            if let Some(prior) = registry.and_then(|registry| registry.get(key)) {
                admission.capacity_matches(prior.bytes, *bytes)?;
                if prior.prepaid.is_some() {
                    // Raw aliases do not invent coverage or replace its origin.
                    // Pure accounting pins retain their ordinary contract.
                    if let Some(scope) = scope {
                        validate_entry_origin(prior, &usage)?;
                        usage
                            .funding
                            .get(&scope.id)
                            .expect("validated scope")
                            .validate_native_publication(scope)?;
                    }
                }
                prior
                    .owners
                    .checked_add(1)
                    .ok_or(WorkingMemoryError::Overflow)?;
            } else {
                if matches!(admission, InventoryAdmission::Pin) {
                    return Err(WorkingMemoryError::IdentityMismatch);
                }
                allocations = allocations
                    .checked_add(1)
                    .ok_or(WorkingMemoryError::Overflow)?;
                incremental = incremental
                    .checked_add(*bytes)
                    .ok_or(WorkingMemoryError::Overflow)?;
            }
        }
        // Check the ordinary domain invariant for both paths. Funded adoption
        // shifts existing coverage and needs no additional pool capacity.
        let available = self.0.available(&usage, None)?;
        let available = if let Some(id) = funding {
            let state = usage.funding.get(&id).expect("validated funding scope");
            state
                .allocations
                .checked_add(allocations)
                .ok_or(WorkingMemoryError::Overflow)?;
            state
                .registrations
                .checked_add(registrations)
                .ok_or(WorkingMemoryError::Overflow)?;
            state.spendable_remaining()?
        } else {
            available
        };
        if incremental > available {
            return Err(WorkingMemoryError::BudgetExceeded {
                required_bytes: incremental,
                available_bytes: available,
            });
        }
        if let Some(id) = funding {
            let state = usage.funding.get_mut(&id).expect("validated funding scope");
            state.remaining -= incremental;
            state.allocations += allocations;
            state.registrations += registrations;
            usage.reserved -= incremental;
        }
        usage.registered += incremental;
        usage.peak = usage
            .peak
            .max(self.0.existing + usage.registered + usage.reserved);
        let registry = usage
            .storage
            .entry(TypeId::of::<K>())
            .or_insert_with(|| Box::new(Registry::<K>::new()))
            .downcast_mut::<Registry<K>>()
            .expect("typed storage registry");
        for (key, bytes) in unique {
            if let Some(entry) = registry.get_mut(&key) {
                entry.owners += 1;
                duplicate_keys.push(key);
            } else {
                registry.insert(
                    RegistryKey::Owned(key),
                    Entry {
                        reset_layout_id: None,
                        prepaid: None,
                        bytes,
                        owners: 1,
                        funding,
                    },
                );
            }
        }
        drop(usage);
        drop(duplicate_keys);
        Ok(())
    }
}

fn prepare_inventory<K: Ord>(
    storage: impl IntoIterator<Item = (K, u64)>,
) -> Result<(BTreeMap<K, u64>, u64), WorkingMemoryError> {
    let mut unique = BTreeMap::new();
    for (key, bytes) in storage {
        if let Some(prior) = unique.insert(key, bytes) {
            same_capacity(prior, bytes)?;
        }
    }
    let bytes = unique.values().try_fold(0u64, |total, bytes| {
        total
            .checked_add(*bytes)
            .ok_or(WorkingMemoryError::Overflow)
    })?;
    Ok((unique, bytes))
}

fn same_capacity(expected_bytes: u64, actual_bytes: u64) -> Result<(), WorkingMemoryError> {
    if expected_bytes == actual_bytes {
        Ok(())
    } else {
        Err(WorkingMemoryError::StorageCapacityMismatch {
            expected_bytes,
            actual_bytes,
        })
    }
}

#[derive(Debug)]
struct Registration<K: Ord + Send + 'static> {
    // Result owners are staged before admission and activated after a successful
    // commit. A rejected batch must not retire keys it never registered.
    pool: Option<WorkingMemoryPool>,
    keys: Vec<K>,
    bytes: u64,
    funding: Option<u64>,
    // Library-owned ordinary preparation, not part of any reset grant.
    original_sources: Option<OriginalSources>,
    // The prepared copy producer pays these key/Vec/Arc controls independently
    // of the physical B charge. Field-last through canonical row retirement.
    preparation: Option<eredu_core::HostPreparationAuthority>,
}

impl<K: Ord + Send + 'static> Drop for Registration<K> {
    fn drop(&mut self) {
        let Some(pool) = &self.pool else {
            return;
        };
        while let Some(key) = self.keys.pop() {
            let retired = {
                let mut usage = funding::lock_for_retirement(pool);
                let registry = usage
                    .storage
                    .get_mut(&TypeId::of::<K>())
                    .expect("live storage registration")
                    .downcast_mut::<Registry<K>>()
                    .expect("typed storage registry");
                let mut retired = registry.retire_owner(&key);
                if registry.is_empty() {
                    retired.namespace = usage.storage.remove(&TypeId::of::<K>());
                }
                retired
            };
            let released = retired
                .entry
                .as_ref()
                .and_then(|(_, entry)| entry.charged());
            // Identity keys can themselves pin inline physical payload bytes.
            // Destroy every key outside the lock, before refunding its charge.
            // A panic conservatively retains that charge. Reentrant registration
            // of the same identity may temporarily overlap it, never undercount it.
            retired.retire_with_key(key);
            if let Some((bytes, origin)) = released {
                let mut usage = funding::lock_for_retirement(pool);
                if let Some(id) = origin {
                    funding::retire_allocation(&mut usage, id, bytes);
                } else {
                    usage.registered -= bytes;
                }
            }
        }
        if let Some(id) = self.funding {
            let mut usage = funding::lock_for_retirement(pool);
            funding::retire_registration(&mut usage, id);
        }
    }
}

/// Shared accounting owner for one retained physical inventory. Keep it alive
/// until every allocation covered by it has retired or another registration
/// covers that allocation. Native owners must also keep its keys valid.
/// Clones retain the same registration; they grant no execution authority.
#[derive(Debug, Clone)]
#[must_use = "retain the registration alongside its physical storage owners"]
pub struct WorkingMemoryStorage<K: Ord + Send + 'static>(StorageOwner<K>);

// Deallocate the actual Arc header before Registration's source population
// releases its genuine ordinary lease. No Weak or raw owner escapes this type.
#[derive(Debug)]
struct StorageOwner<K: Ord + Send + 'static>(Option<Arc<Registration<K>>>);
impl<K: Ord + Send + 'static> Clone for StorageOwner<K> {
    fn clone(&self) -> Self {
        Self(Some(Arc::clone(
            self.0.as_ref().expect("live storage owner"),
        )))
    }
}
impl<K: Ord + Send + 'static> std::ops::Deref for StorageOwner<K> {
    type Target = Arc<Registration<K>>;
    fn deref(&self) -> &Self::Target {
        self.0.as_ref().expect("live storage owner")
    }
}
impl<K: Ord + Send + 'static> std::ops::DerefMut for StorageOwner<K> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.0.as_mut().expect("live storage owner")
    }
}
impl<K: Ord + Send + 'static> Drop for StorageOwner<K> {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            drop(Arc::into_inner(owner));
        }
    }
}

impl<K: Ord + Send + 'static> WorkingMemoryStorage<K> {
    /// Whether these live registrations cover exactly the same registered
    /// keys and capacity in the same pool. Extra original-source custody is
    /// deliberately excluded. This observation neither transfers ownership nor
    /// authorizes work: the caller must retain the other registration while
    /// the corresponding storage survives.
    pub fn same_registered_storage(&self, other: &Self) -> bool {
        self.0.original_sources.is_none()
            && other.0.original_sources.is_none()
            && self.0.pool.as_ref().zip(other.0.pool.as_ref())
                .is_some_and(|(a, b)| a.same_domain(b))
            && self.0.bytes == other.0.bytes
            && self.0.keys == other.0.keys
    }

    // The caller holds the same usage lock used for destination admission.
    // A live pin makes capacities immutable; check every origin's current
    // health rather than treating a cold identity snapshot as settled work.
    pub(super) fn validate_copy_source(
        &self,
        pool: &WorkingMemoryPool,
        usage: &super::Usage,
    ) -> Result<(), WorkingMemoryError> {
        if self
            .0
            .pool
            .as_ref()
            .is_none_or(|source| !source.same_domain(pool))
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        if let Some(sources) = &self.0.original_sources {
            sources.validate_in(pool, usage)?;
        }
        if self.0.keys.is_empty() {
            return Ok(());
        }
        let registry = usage
            .storage
            .get(&TypeId::of::<K>())
            .and_then(|registry| registry.downcast_ref::<Registry<K>>())
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        for key in &self.0.keys {
            let entry = registry
                .get(key)
                .filter(|entry| entry.owners != 0)
                .ok_or(WorkingMemoryError::IdentityMismatch)?;
            validate_entry_origin(entry, usage)?;
        }
        Ok(())
    }

    pub(super) fn host_source_key(
        &self,
        identity: &crate::HostMetadataKey,
    ) -> Result<&K, WorkingMemoryError>
    where
        K: super::HostSlotStorageKey,
    {
        self.0
            .keys
            .iter()
            .find(|key| key.host_slot_identity() == Some(identity))
            .ok_or(WorkingMemoryError::IdentityMismatch)
    }
    pub(super) fn locate_host_key(
        &self,
        usage: &super::Usage,
        key: &K,
        bytes: u64,
    ) -> Result<registry::EntryLocator, WorkingMemoryError> {
        let registry = usage
            .storage
            .get(&TypeId::of::<K>())
            .and_then(|r| r.downcast_ref::<Registry<K>>())
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        let (locator, entry) = registry
            .locate(key)
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        if entry.bytes != bytes {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        validate_entry_origin(entry, usage)?;
        Ok(locator)
    }
    pub(super) fn validate_host_key(
        &self,
        usage: &super::Usage,
        key: &K,
        bytes: u64,
    ) -> Result<(), WorkingMemoryError> {
        let registry = usage
            .storage
            .get(&TypeId::of::<K>())
            .and_then(|r| r.downcast_ref::<Registry<K>>())
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        let entry = registry
            .get(key)
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        if entry.bytes != bytes {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        validate_entry_origin(entry, usage)
    }

    pub(super) fn validate_host_source(
        &self,
        usage: &super::Usage,
        identity: &crate::HostMetadataKey,
        bytes: u64,
    ) -> Result<(), WorkingMemoryError>
    where
        K: super::HostSlotStorageKey,
    {
        let key = self
            .0
            .keys
            .iter()
            .find(|key| key.host_slot_identity() == Some(identity))
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        let registry = usage
            .storage
            .get(&TypeId::of::<K>())
            .and_then(|value| value.downcast_ref::<Registry<K>>())
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        let entry = registry
            .get(key)
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        if entry.bytes != bytes {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        validate_entry_origin(entry, usage)
    }

    fn pending(keys: Vec<K>, bytes: u64) -> Self {
        Self(StorageOwner(Some(Arc::new(Registration {
            pool: None,
            keys,
            bytes,
            funding: None,
            original_sources: None,
            preparation: None,
        }))))
    }

    fn pending_prepared(
        keys: Vec<K>,
        bytes: u64,
        host: &eredu_core::HostPreparationAuthority,
    ) -> Self {
        let mut value = Self::pending(keys, bytes);
        Arc::get_mut(&mut value.0)
            .expect("private prepared registration")
            .preparation = Some(host.clone());
        value
    }

    fn activate(&mut self, pool: WorkingMemoryPool, funding: Option<u64>) {
        // Staged handles have not been published or cloned. Activation does not
        // invoke key code, allocate payloads or repeat shared-accounting checks.
        let registration = Arc::get_mut(&mut self.0).expect("unpublished storage registration");
        registration.pool = Some(pool);
        registration.funding = funding;
    }

    /// Consumes a unique ordinary registration and a closed source population.
    /// On rejection the sources and their genuine lease stay in `sources`.
    /// Original physical bytes contribute to the report, never a second charge;
    /// newly constructed carrier controls remain excluded by that same lease.
    pub fn with_original_reset_sources(
        mut self,
        sources: &mut super::UnquotedOriginalSlotSources,
    ) -> Result<Self, WorkingMemoryError>
    where
        K: super::HostSlotStorageKey,
    {
        if sources.is_empty() {
            return Ok(self);
        }
        let pool = self
            .0
            .pool
            .as_ref()
            .ok_or(WorkingMemoryError::IdentityMismatch)?
            .clone();
        if !sources.same_domain(&pool)
            || self.0.original_sources.is_some()
            || Arc::strong_count(&self.0) != 1
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let mut bytes = self.0.bytes;
        for source in sources.sources() {
            let key = source.metadata().identity().registry_key();
            if self
                .0
                .keys
                .iter()
                .any(|ordinary| ordinary.host_slot_identity() == Some(key))
            {
                return Err(WorkingMemoryError::IdentityMismatch);
            }
            bytes = bytes
                .checked_add(
                    source
                        .metadata()
                        .capacity_bytes()
                        .ok_or(WorkingMemoryError::UnknownBound)?,
                )
                .ok_or(WorkingMemoryError::Overflow)?;
        }
        {
            let usage = pool
                .0
                .usage
                .lock()
                .map_err(|_| WorkingMemoryError::Poisoned)?;
            for source in sources.sources() {
                source.validate_in(&pool, &usage)?;
            }
        }
        let registration = Arc::get_mut(&mut self.0).expect("unpublished original-source carrier");
        registration.bytes = bytes;
        registration.original_sources = Some(OriginalSources::Unquoted(sources.take()));
        Ok(self)
    }

    /// Full unique capacity covered by this inventory, including aliases shared
    /// with other registrations. It is not an additional charge per handle.
    pub fn bytes(&self) -> u64 {
        self.0.bytes
    }
}

// Shared by ordinary source validation and the bounded existing-only commit.
fn validate_entry_origin(entry: &Entry, usage: &super::Usage) -> Result<(), WorkingMemoryError> {
    if entry.owners == 0 {
        return Err(WorkingMemoryError::IdentityMismatch);
    }
    if let Some(partition) = &entry.prepaid {
        partition.validate_origin(usage)?;
    }
    if let Some(id) = entry.funding {
        usage
            .funding
            .get(&id)
            .ok_or(WorkingMemoryError::IdentityMismatch)?
            .validate_registered_copy_origin()?;
    }
    Ok(())
}
