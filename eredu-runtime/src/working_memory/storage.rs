//! Atomic registration of retained physical storage in a shared request pool.

use super::{MemoryLedger, WorkingMemoryError, WorkingMemoryFundingScope, funding};
use std::{any::TypeId, borrow::Borrow, cmp::Ordering, sync::Arc};

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
mod host_slots;
mod host_transfer;
pub use host_slots::PreparedStorageHostSlots;
pub(in crate::working_memory) mod reset_layout;
pub(super) use host_transfer::{
    dense_host_transfer_control_bytes, ordinary_dense_host_publication_bytes,
    ordinary_dense_preparation_bytes, prepare_dense_host_metadata, publish_dense_host_slots,
};

#[cfg(test)]
mod pin_tests;
#[cfg(test)]
mod tests;

#[derive(Debug)]
struct Entry {
    // Assigned once when an original reset pins this canonical entry.
    reset_layout_id: Option<u64>,
    bytes: u64,
    placement: Arc<eredu_core::MemoryPlacement>,
    owners: usize,
    funding: Option<u64>,
    funding_allowance_bytes: u64,
    native_retired: bool,
    pending_allocation: bool,
    prepaid: Option<prepaid::PrepaidStorageOrigin>,
}

impl Entry {
    fn native_charge(
        &self,
    ) -> Option<(
        funding::native_partition::NativePartition,
        u64,
        Arc<eredu_core::MemoryPlacement>,
        u64,
    )> {
        match &self.prepaid {
            Some(prepaid::PrepaidStorageOrigin::Native(partition)) if !self.native_retired => {
                Some((
                    partition.clone(),
                    self.bytes,
                    Arc::clone(&self.placement),
                    self.funding_allowance_bytes,
                ))
            }
            _ => None,
        }
    }
    fn charged(&self) -> Option<(u64, Option<u64>, Arc<eredu_core::MemoryPlacement>, u64)> {
        match &self.prepaid {
            None => Some((
                self.bytes,
                self.funding,
                Arc::clone(&self.placement),
                self.funding_allowance_bytes,
            )),
            Some(origin) => origin
                .residual_source_charge()
                .map(|bytes| (bytes, None, Arc::clone(&self.placement), 0)),
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
pub use native_publication::copy::{
    PreparedWorkspaceCopyPublication, WorkspaceCopyPublicationPlan,
};
pub use native_publication::numerical::{
    NumericalStoragePublicationPlan, NumericalStorageRegistration,
    PreparedNumericalStoragePublication,
};
mod registry;
mod source_inventory;
use registry::{EntryLocator, Registry, RegistryBatch};

impl MemoryLedger {
    /// Observes the complete backing descriptor for a published storage identity.
    /// This allocation-free diagnostic neither pins that storage nor grants
    /// execution authority. Another owner may retire it after the observation.
    pub fn registered_allocation<K: Ord + Send + 'static>(
        &self,
        key: &K,
    ) -> Result<Option<StorageAllocation>, WorkingMemoryError> {
        let usage = self
            .0
            .usage
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        Ok(usage
            .storage
            .get(&TypeId::of::<K>())
            .and_then(|value| value.downcast_ref::<Registry<K>>())
            .and_then(|registry| registry.get(key))
            .filter(|entry| !entry.pending_allocation && !entry.native_retired)
            .map(|entry| StorageAllocation::new(entry.bytes, Arc::clone(&entry.placement))))
    }

    #[cfg(test)]
    pub(crate) fn registered_capacity_for_test<K: Ord + Send + 'static>(
        &self,
        key: &K,
    ) -> Result<Option<u64>, WorkingMemoryError> {
        let usage = self
            .0
            .usage
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        Ok(usage
            .storage
            .get(&TypeId::of::<K>())
            .and_then(|value| value.downcast_ref::<Registry<K>>())
            .and_then(|registry| registry.get(key))
            .map(|entry| entry.bytes))
    }

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
    #[cfg(test)]
    pub(crate) fn register_host_storage<K: Clone + Ord + Send + 'static>(
        &self,
        storage: impl IntoIterator<Item = (K, u64)>,
    ) -> Result<WorkingMemoryStorage<K>, WorkingMemoryError> {
        self.register_storage(storage.into_iter().map(|(key, bytes)| {
            (
                key,
                StorageAllocation::new(bytes, Arc::clone(&self.0.host_placement)),
            )
        }))
    }

    /// Pins a complete inventory only if every unique identity is already
    /// registered in this pool with exactly the supplied capacity. Missing keys,
    /// foreign key namespaces and conflicting capacities reject the entire
    /// inventory with `IdentityMismatch`; no partial ownership is published.
    /// Equal keys share one pin, including zero-byte allocations.
    ///
    /// Payload bytes remain charged once; the new pin's host bookkeeping is
    /// admitted atomically. Pins preserve each allocation's existing
    /// funding origin until its last registration retires; they do not create
    /// another funding account or grant permission to allocate. Keep physical
    /// identities valid as required by [`Self::register_storage`]. Key cloning
    /// happens before accounting is locked; key destruction precedes refunds
    /// outside that lock, including when this pin is the final owner.
    #[cfg(test)]
    pub(crate) fn pin_registered_storage<K: Clone + Ord + Send + 'static>(
        &self,
        storage: impl IntoIterator<Item = (K, u64)>,
    ) -> Result<WorkingMemoryStorage<K>, WorkingMemoryError> {
        let entries: Vec<_> = storage.into_iter().collect();
        StoragePublicationLayout::new(entries.len())?
            .fund(self)?
            .pin_registered_storage(entries)
    }
}

mod cold_metadata;
pub use cold_metadata::{StorageRegistrationValues, StorageRegistrations};
mod prepared;
pub use prepared::{PreparedStoragePublication, StoragePublicationLayout};
mod pending_allocation;
pub use pending_allocation::PendingStorageAllocation;
mod physical;
pub use physical::StorageAllocation;

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
    pool: Option<MemoryLedger>,
    keys: Vec<K>,
    bytes: Option<u64>,
    funding: Option<u64>,
    // Library-owned ordinary preparation, not part of any reset grant.
    original_sources: Option<OriginalSources>,
    completed_source: Option<super::workspace_copy::completed::CompletedSourceCustody>,
    completed_numerical: Option<super::OriginalNumericalBudgetCustody>,
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
            let pending_released = retired
                .entry
                .as_ref()
                .is_some_and(|(_, entry)| entry.pending_allocation);
            let native_released = retired
                .entry
                .as_ref()
                .and_then(|(_, entry)| entry.native_charge());
            // Identity keys can themselves pin inline physical payload bytes.
            // Destroy every key outside the lock, before refunding its charge.
            // A panic conservatively retains that charge. Reentrant registration
            // of the same identity may temporarily overlap it, never undercount it.
            retired.retire_with_key(key);
            if let Some((partition, bytes, placement, allowance)) = native_released {
                {
                    let mut usage = funding::lock_for_retirement(pool);
                    partition.retire_registered(&mut usage, bytes, &placement, allowance);
                }
                drop(partition);
            }
            if let Some((bytes, origin, placement, allowance)) = released {
                let mut usage = funding::lock_for_retirement(pool);
                funding::retire_placed_allocation_mode(
                    &mut usage,
                    origin,
                    bytes,
                    &placement,
                    allowance,
                    pending_released,
                );
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
    pub(in crate::working_memory) fn validate_completed_numerical_custody(
        &self,
        account: &super::OriginalNumericalBudgetCustody,
    ) -> Result<(), WorkingMemoryError> {
        if !self.0.keys.is_empty()
            || !self
                .0
                .completed_numerical
                .as_ref()
                .is_some_and(|origin| origin.same_account(account))
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let pool = self
            .0
            .pool
            .as_ref()
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        let usage = pool
            .0
            .usage
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        self.validate_copy_source(pool, &usage)
    }

    /// Called only by the successfully attached native sidecar after the
    /// backing's safe physical retirement. Stale accounting aliases keep the
    /// authenticated descriptor, but cannot keep or replay its physical charge.
    /// The closed numerical publisher also uses this to withdraw a fresh row
    /// after an abandoned attachment. Numerical origins only lose discovery;
    /// their existing native observer remains the sole payload refund owner.
    pub(in crate::working_memory) fn retire_native_backing(&self) {
        let Some(pool) = &self.0.pool else {
            return;
        };
        for key in &self.0.keys {
            let mut usage = funding::lock_for_retirement(pool);
            let native = {
                let Some(registry) = usage
                    .storage
                    .get_mut(&TypeId::of::<K>())
                    .and_then(|registry| registry.downcast_mut::<Registry<K>>())
                else {
                    continue;
                };
                let Some((locator, entry)) = registry.locate(key) else {
                    continue;
                };
                let native = entry.native_charge();
                if native.is_some()
                    || matches!(
                        &entry.prepaid,
                        Some(prepaid::PrepaidStorageOrigin::Numerical(_))
                    )
                {
                    registry.at_mut(locator).native_retired = true;
                }
                native
            };
            if let Some((partition, bytes, placement, allowance)) = native {
                partition.retire_registered(&mut usage, bytes, &placement, allowance);
                drop(usage);
                drop(partition);
            }
        }
    }

    /// Whether these live registrations cover exactly the same registered
    /// keys and capacity in the same pool. Extra original-source custody is
    /// deliberately excluded. This observation neither transfers ownership nor
    /// authorizes work: the caller must retain the other registration while
    /// the corresponding storage survives.
    pub fn same_registered_storage(&self, other: &Self) -> bool {
        self.0.original_sources.is_none()
            && other.0.original_sources.is_none()
            && self.0.completed_source.is_none()
            && other.0.completed_source.is_none()
            && self.0.completed_numerical.is_none()
            && other.0.completed_numerical.is_none()
            && self
                .0
                .pool
                .as_ref()
                .zip(other.0.pool.as_ref())
                .is_some_and(|(a, b)| a.same_ledger(b))
            && self.0.bytes == other.0.bytes
            && self.0.keys == other.0.keys
    }

    // The caller holds the same usage lock used for destination admission.
    // A live pin makes capacities immutable; check every origin's current
    // health rather than treating a cold identity snapshot as settled work.
    pub(super) fn validate_copy_source(
        &self,
        pool: &MemoryLedger,
        usage: &super::Usage,
    ) -> Result<(), WorkingMemoryError> {
        if self
            .0
            .pool
            .as_ref()
            .is_none_or(|source| !source.same_ledger(pool))
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        if let Some(sources) = &self.0.original_sources {
            sources.validate_in(pool, usage)?;
        }
        if let Some(source) = &self.0.completed_source {
            source.validate(pool, usage)?;
        }
        if let Some(source) = &self.0.completed_numerical {
            source.validate_copy_source(pool, usage)?;
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
        Self::pending_domains(keys, Some(bytes))
    }
    fn pending_domains(keys: Vec<K>, bytes: Option<u64>) -> Self {
        Self(StorageOwner(Some(Arc::new(Registration {
            pool: None,
            keys,
            bytes,
            funding: None,
            original_sources: None,
            completed_source: None,
            completed_numerical: None,
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

    fn activate(&mut self, pool: MemoryLedger, funding: Option<u64>) {
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
        if !sources.same_ledger(&pool)
            || self.0.original_sources.is_some()
            || Arc::strong_count(&self.0) != 1
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let mut bytes = self.0.bytes.ok_or(WorkingMemoryError::Overflow)?;
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
        registration.bytes = Some(bytes);
        registration.original_sources = Some(OriginalSources::Unquoted(sources.take()));
        Ok(self)
    }

    pub(in crate::working_memory) fn with_completed_source(
        mut self,
        custody: super::workspace_copy::completed::CompletedSourceCustody,
        bytes: Option<u64>,
    ) -> Result<Self, WorkingMemoryError> {
        let pool = self
            .0
            .pool
            .as_ref()
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        if self.0.completed_source.is_some() || Arc::strong_count(&self.0) != 1 {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        {
            let usage = pool
                .0
                .usage
                .lock()
                .map_err(|_| WorkingMemoryError::Poisoned)?;
            custody.validate(pool, &usage)?;
        }
        let registration = Arc::get_mut(&mut self.0).expect("private completed source pin");
        registration.bytes = registration
            .bytes
            .zip(bytes)
            .and_then(|(a, b)| a.checked_add(b));
        registration.completed_source = Some(custody);
        Ok(self)
    }

    /// Full unique capacity covered by this inventory, including aliases shared
    /// with other registrations. It is not an additional charge per handle.
    pub fn bytes(&self) -> Option<u64> {
        self.0.bytes
    }
}

// Shared by ordinary source validation and the bounded existing-only commit.
fn validate_entry_origin(entry: &Entry, usage: &super::Usage) -> Result<(), WorkingMemoryError> {
    if entry.owners == 0 || entry.native_retired || entry.pending_allocation {
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
