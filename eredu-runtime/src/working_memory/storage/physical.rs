//! One prepared publication transaction across every physical domain.
use super::*;
use crate::working_memory::{Usage, gguf_source::SourceInventoryOrigin};
use eredu_core::{DomainMemoryCharge, MemoryDomainId, MemoryPlacement, MemoryPlacementKind};

#[derive(Clone, Copy)]
pub(super) enum PhysicalFunding<'a> {
    Scope(&'a WorkingMemoryFundingScope),
    Allocation(&'a super::super::WorkingMemoryAllocationFunding),
}
impl PhysicalFunding<'_> {
    fn id(self) -> u64 {
        match self {
            Self::Scope(scope) => scope.id,
            Self::Allocation(source) => source.account(),
        }
    }
    fn validate(
        self,
        pool: &MemoryLedger,
        usage: &Usage,
        reserve_only: bool,
    ) -> Result<bool, WorkingMemoryError> {
        match self {
            Self::Scope(scope) => {
                scope.validate_ledger(pool)?;
                scope.validate_native_purpose()?;
                let state = usage
                    .funding
                    .get(&scope.id)
                    .ok_or(WorkingMemoryError::ExecutionFenced)?;
                let recovery = state.validate_storage_publication(scope)?;
                if reserve_only && recovery {
                    return Err(WorkingMemoryError::ExecutionFenced);
                }
                Ok(recovery)
            }
            Self::Allocation(source) => {
                source.validate_locked(pool, usage)?;
                Ok(false)
            }
        }
    }
}

/// Certified full backing capacity and placement, separate from its identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StorageAllocation {
    capacity_bytes: u64,
    placement: Arc<MemoryPlacement>,
}
impl StorageAllocation {
    /// Describes one backing. Views retain its full capacity and placement.
    pub fn new(capacity_bytes: u64, placement: Arc<MemoryPlacement>) -> Self {
        Self {
            capacity_bytes,
            placement,
        }
    }
    /// Full backing capacity, including portions outside any retained view.
    pub fn capacity_bytes(&self) -> u64 {
        self.capacity_bytes
    }
    /// Backend-established fixed placement or conservative candidate set.
    pub fn placement(&self) -> &MemoryPlacement {
        &self.placement
    }
    /// Retains the same immutable placement descriptor without copying it.
    pub fn placement_handle(&self) -> Arc<MemoryPlacement> {
        Arc::clone(&self.placement)
    }
    fn charge(&self, domain: MemoryDomainId) -> DomainMemoryCharge {
        if !self.placement.domains().contains(&domain) {
            return Default::default();
        }
        match self.placement.kind() {
            MemoryPlacementKind::Fixed(_) => DomainMemoryCharge {
                accounted_bytes: self.capacity_bytes,
                ..Default::default()
            },
            MemoryPlacementKind::Possible { .. } => DomainMemoryCharge {
                placement_allowance_bytes: self.capacity_bytes,
                ..Default::default()
            },
        }
    }
}
struct Row<K> {
    key: Option<K>,
    allocation: StorageAllocation,
    locator: Option<EntryLocator>,
    origin: Option<SourceInventoryOrigin>,
    converted_allowance: u64,
}

fn domain_increment<K>(
    rows: &[Row<K>],
    domain: MemoryDomainId,
    host: MemoryDomainId,
) -> Result<DomainMemoryCharge, WorkingMemoryError> {
    rows.iter().filter(|row| row.locator.is_none()).try_fold(
        DomainMemoryCharge::default(),
        |sum, row| {
            let charge = if let Some(origin) = &row.origin {
                if domain == host {
                    DomainMemoryCharge {
                        accounted_bytes: origin.residual(),
                        ..Default::default()
                    }
                } else {
                    Default::default()
                }
            } else {
                row.allocation.charge(domain)
            };
            Ok(sum.checked_add(charge)?)
        },
    )
}
fn domain_conversion<K>(rows: &[Row<K>], domain: MemoryDomainId) -> u64 {
    rows.iter()
        .filter(|row| row.allocation.placement.domains().contains(&domain))
        .map(|row| row.converted_allowance)
        .try_fold(0u64, u64::checked_add)
        .expect("prepared conversion")
}

pub(super) fn prepare<K: Ord>(
    pool: &MemoryLedger,
    storage: impl IntoIterator<Item = (K, StorageAllocation)>,
) -> Result<Vec<(K, StorageAllocation)>, WorkingMemoryError> {
    let mut unique: Vec<_> = storage.into_iter().collect();
    unique.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    for (_, allocation) in &unique {
        allocation.placement.validate(pool.topology())?;
    }
    for pair in unique.windows(2) {
        if pair[0].0 == pair[1].0 {
            same_capacity(pair[0].1.capacity_bytes, pair[1].1.capacity_bytes)?;
            if pair[0].1.placement != pair[1].1.placement {
                return Err(WorkingMemoryError::StoragePlacementMismatch);
            }
        }
    }
    unique.dedup_by(|a, b| a.0 == b.0);
    Ok(unique)
}

impl MemoryLedger {
    /// Publishes authenticated backings with explicit physical placement.
    /// All capacity, placement, ownership and funding checks precede one commit.
    #[cfg(test)]
    pub(crate) fn register_storage<K: Clone + Ord + Send + 'static>(
        &self,
        storage: impl IntoIterator<Item = (K, StorageAllocation)>,
    ) -> Result<WorkingMemoryStorage<K>, WorkingMemoryError> {
        let entries: Vec<_> = storage.into_iter().collect();
        StoragePublicationLayout::new(entries.len())?
            .fund(self)?
            .register_storage(entries)
    }
    pub(super) fn register_storage_prepared<K: Clone + Ord + Send + 'static>(
        &self,
        storage: impl IntoIterator<Item = (K, StorageAllocation)>,
        host: &eredu_core::HostPreparationAuthority,
    ) -> Result<WorkingMemoryStorage<K>, WorkingMemoryError> {
        let unique = prepare(self, storage)?;
        let bytes = unique
            .iter()
            .try_fold(0u64, |n, (_, a)| n.checked_add(a.capacity_bytes));
        let mut owner = WorkingMemoryStorage::pending_domains(
            unique.iter().map(|(key, _)| key.clone()).collect(),
            bytes,
        );
        Arc::get_mut(&mut owner.0).unwrap().preparation = Some(host.clone());
        self.commit_physical_inventory(unique, None, 0, Vec::new(), host)?;
        owner.activate(self.clone(), None);
        Ok(owner)
    }

    /// Independent owners allow a completed staging allocation to retire while
    /// its destination survives. Equal authenticated backings share one charge.
    #[cfg(test)]
    pub(crate) fn register_storage_individually<K: Clone + Ord + Send + 'static>(
        &self,
        storage: impl IntoIterator<Item = (K, StorageAllocation)>,
    ) -> Result<StorageRegistrations<K>, WorkingMemoryError> {
        let entries: Vec<_> = storage.into_iter().collect();
        StoragePublicationLayout::new(entries.len())?
            .fund(self)?
            .register_storage_individually(entries)
    }

    /// Converts already assigned domain allowances into retained storage.
    /// A device backing cannot consume an unrelated host-domain allowance.
    #[cfg(test)]
    pub(crate) fn adopt_storage_individually<K: Clone + Ord + Send + 'static>(
        &self,
        scope: &WorkingMemoryFundingScope,
        storage: impl IntoIterator<Item = (K, StorageAllocation)>,
    ) -> Result<StorageRegistrations<K>, WorkingMemoryError> {
        scope.validate_ledger(self)?;
        scope.validate_native_purpose()?;
        let entries: Vec<_> = storage.into_iter().collect();
        StoragePublicationLayout::new(entries.len())?
            .fund_from(scope)?
            .adopt_storage_individually(scope, entries)
    }

    pub(super) fn publish_physical_individually<K: Clone + Ord + Send + 'static>(
        &self,
        storage: impl IntoIterator<Item = (K, StorageAllocation)>,
        scope: Option<&WorkingMemoryFundingScope>,
        host: &eredu_core::HostPreparationAuthority,
    ) -> Result<StorageRegistrations<K>, WorkingMemoryError> {
        let unique = prepare(self, storage)?;
        let entries = unique
            .iter()
            .map(|(key, allocation)| {
                (
                    key.clone(),
                    WorkingMemoryStorage::pending(vec![key.clone()], allocation.capacity_bytes),
                )
            })
            .collect();
        let mut owners = StorageRegistrations::pending(entries)?;
        owners.retain_preparation(host);
        for owner in owners.values_mut() {
            Arc::get_mut(&mut owner.0).unwrap().preparation = Some(host.clone());
        }
        self.commit_physical_inventory(unique, scope, owners.len(), Vec::new(), host)?;
        for owner in owners.values_mut() {
            owner.activate(self.clone(), scope.map(|s| s.id));
        }
        Ok(owners)
    }

    pub(super) fn commit_physical_inventory<K: Ord + Send + 'static>(
        &self,
        unique: Vec<(K, StorageAllocation)>,
        scope: Option<&WorkingMemoryFundingScope>,
        registrations: usize,
        sources: Vec<(K, SourceInventoryOrigin)>,
        host: &eredu_core::HostPreparationAuthority,
    ) -> Result<(), WorkingMemoryError> {
        self.commit_physical_inventory_mode(
            unique,
            scope.map(PhysicalFunding::Scope),
            registrations,
            sources,
            host,
            false,
        )
    }
    pub(super) fn commit_physical_inventory_mode<K: Ord + Send + 'static>(
        &self,
        unique: Vec<(K, StorageAllocation)>,
        scope: Option<PhysicalFunding<'_>>,
        registrations: usize,
        mut sources: Vec<(K, SourceInventoryOrigin)>,
        host: &eredu_core::HostPreparationAuthority,
        reserve_only: bool,
    ) -> Result<(), WorkingMemoryError> {
        if reserve_only && !sources.is_empty() {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let mut rows: Vec<_> = unique
            .into_iter()
            .map(|(key, allocation)| {
                let origin = sources
                    .binary_search_by(|(candidate, _)| candidate.cmp(&key))
                    .ok()
                    .map(|index| sources.remove(index).1);
                Row {
                    key: Some(key),
                    allocation,
                    locator: None,
                    origin,
                    converted_allowance: 0,
                }
            })
            .collect();
        let mut namespace = Some(directory::PreparedNamespace::prepare_copy::<K>(host));
        let mut node = Some(RegistryBatch::prepare_copy_exact(rows.len(), host)?);
        let mut usage = self
            .0
            .usage
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        let recovery_only = scope
            .map(|source| source.validate(self, &usage, reserve_only))
            .transpose()?
            .unwrap_or(false);
        let missing = usage.storage.get(&TypeId::of::<K>()).is_none();
        let registry = usage
            .storage
            .get(&TypeId::of::<K>())
            .map(|r| {
                r.downcast_ref::<Registry<K>>()
                    .expect("typed storage namespace")
            })
            .unwrap_or_else(|| namespace.as_ref().unwrap().registry::<K>());
        let mut allocations = 0usize;
        for row in &mut rows {
            if let Some(origin) = &row.origin {
                origin.validate_pool(self)?;
            }
            if let Some((locator, prior)) = registry.locate(row.key.as_ref().unwrap()) {
                if recovery_only {
                    return Err(WorkingMemoryError::ExecutionFenced);
                }
                if reserve_only {
                    return Err(WorkingMemoryError::IdentityMismatch);
                }
                same_capacity(prior.bytes, row.allocation.capacity_bytes)?;
                if prior.placement != row.allocation.placement {
                    return Err(WorkingMemoryError::StoragePlacementMismatch);
                }
                validate_entry_origin(prior, &usage)?;
                if let (Some(origin), Some(prior)) = (&row.origin, &prior.prepaid) {
                    if !matches!(prior, prepaid::PrepaidStorageOrigin::Source(prior) if origin.same(prior))
                    {
                        return Err(WorkingMemoryError::IdentityMismatch);
                    }
                }
                prior
                    .owners
                    .checked_add(1)
                    .ok_or(WorkingMemoryError::Overflow)?;
                row.locator = Some(locator);
            } else {
                allocations = allocations
                    .checked_add(1)
                    .ok_or(WorkingMemoryError::Overflow)?;
                if row.origin.is_some()
                    && row.allocation.placement.as_ref() != self.0.host_placement.as_ref()
                {
                    return Err(WorkingMemoryError::StoragePlacementMismatch);
                }
            }
        }
        if let Some(scope) = scope {
            let state = usage
                .funding
                .get(&scope.id())
                .expect("validated funding scope");
            state
                .allocations
                .checked_add(allocations)
                .ok_or(WorkingMemoryError::Overflow)?;
            state
                .registrations
                .checked_add(registrations)
                .ok_or(WorkingMemoryError::Overflow)?;
        }
        for (slot, (domain, _)) in self.topology().domains().enumerate() {
            let current = &usage.domains[slot];
            let charge = domain_increment(&rows, domain, self.topology().host_domain())?;
            let increment = charge.total()?;
            let used = self.0.domains[slot]
                .existing
                .checked_add(current.registered)
                .and_then(|n| n.checked_add(current.reserved))
                .ok_or(WorkingMemoryError::Overflow)?;
            let limit = self.0.domain_capacity(&usage, domain, None)?;
            limit.check(domain, used, if scope.is_some() { 0 } else { increment })?;
            if !reserve_only || scope.is_none() {
                (if reserve_only {
                    current.reserved
                } else {
                    current.registered
                })
                .checked_add(increment)
                .ok_or(WorkingMemoryError::Overflow)?;
            }
            if let Some(scope) = scope {
                let state = usage.funding.get(&scope.id()).unwrap();
                let balance = &state.domains[slot];
                let protected = balance
                    .native_held
                    .unwrap_or(0)
                    .checked_add(if slot == usage.host_slot {
                        state.host_held
                    } else {
                        0
                    })
                    .ok_or(WorkingMemoryError::Overflow)?;
                let available = balance
                    .remaining
                    .checked_sub(protected)
                    .ok_or(WorkingMemoryError::Poisoned)?;
                if increment > available {
                    return Err(WorkingMemoryError::DomainAllowanceExceeded {
                        domain,
                        required_bytes: increment,
                        available_bytes: available,
                    });
                }
                current
                    .reserved
                    .checked_sub(increment)
                    .ok_or(WorkingMemoryError::Poisoned)?;
                let converted = state.allocation_allowance(
                    slot,
                    charge.accounted_bytes,
                    charge.placement_allowance_bytes,
                    0,
                )?;
                let debit = charge
                    .placement_allowance_bytes
                    .checked_add(converted)
                    .ok_or(WorkingMemoryError::Overflow)?;
                balance
                    .remaining_charge
                    .placement_allowance_bytes
                    .checked_sub(debit)
                    .ok_or(WorkingMemoryError::IdentityMismatch)?;
                current
                    .placement_allowances
                    .checked_sub(converted)
                    .ok_or(WorkingMemoryError::Poisoned)?;
                let mut rest = converted;
                for row in rows.iter_mut().filter(|row| row.locator.is_none()
                    && matches!(row.allocation.placement.kind(), MemoryPlacementKind::Fixed(id) if *id == domain)) {
                    row.converted_allowance = rest.min(row.allocation.capacity_bytes);
                    rest -= row.converted_allowance;
                }
                if rest != 0 {
                    return Err(WorkingMemoryError::IdentityMismatch);
                }
            } else {
                current
                    .placement_allowances
                    .checked_add(charge.placement_allowance_bytes)
                    .ok_or(WorkingMemoryError::Overflow)?;
            }
        }
        // No provider operation, allocation, fallible check or payload drop follows.
        if let Some(scope) = scope {
            let state = usage.funding.get_mut(&scope.id()).unwrap();
            state.allocations += allocations;
            state.registrations += registrations;
        }
        for (slot, (domain, _)) in self.topology().domains().enumerate() {
            let charge = domain_increment(&rows, domain, self.topology().host_domain())
                .expect("validated increment");
            let bytes = charge.total().expect("checked charge");
            let converted = domain_conversion(&rows, domain);
            if let Some(scope) = scope {
                let balance = &mut usage.funding.get_mut(&scope.id()).unwrap().domains[slot];
                balance.remaining -= bytes;
                balance.remaining_charge.placement_allowance_bytes -=
                    charge.placement_allowance_bytes + converted;
            }
            let current = &mut usage.domains[slot];
            if scope.is_some() {
                current.reserved -= bytes;
                current.placement_allowances -= converted;
            } else {
                current.placement_allowances += charge.placement_allowance_bytes;
            }
            if reserve_only {
                current.reserved += bytes;
            } else {
                current.registered += bytes;
            }
            current.peak = current
                .peak
                .max(self.0.domains[slot].existing + current.registered + current.reserved);
        }
        if missing && allocations != 0 {
            usage.storage.install(namespace.take().unwrap());
        }
        if !rows.is_empty() {
            let registry = usage
                .storage
                .get_mut(&TypeId::of::<K>())
                .unwrap()
                .downcast_mut::<Registry<K>>()
                .unwrap();
            for (slot, row) in rows.iter_mut().enumerate() {
                if let Some(locator) = row.locator {
                    registry.at_mut(locator).owners += 1;
                } else {
                    node.as_mut().unwrap().entries[slot] = Some((
                        RegistryKey::Owned(row.key.take().unwrap()),
                        Entry {
                            reset_layout_id: None,
                            bytes: row.allocation.capacity_bytes,
                            placement: Arc::clone(&row.allocation.placement),
                            owners: 1,
                            funding: scope.map(PhysicalFunding::id),
                            prepaid: row.origin.take().map(prepaid::PrepaidStorageOrigin::Source),
                            native_retired: false,
                            pending_allocation: reserve_only,
                            funding_allowance_bytes: row.converted_allowance,
                        },
                    ));
                }
            }
            if allocations != 0 {
                registry.link(node.take().unwrap());
            }
        }
        drop(usage);
        drop((rows, sources, namespace, node));
        Ok(())
    }
}

impl<K: Ord + Send + 'static> StorageRegistrations<K> {
    /// Complete managed host contribution of the ordinary publication worker.
    /// `population` is the deduplicated backing count; the other populations
    /// are actual capacities of its incoming descriptor and source vectors.
    /// Prepared native and workspace-copy publication have their own funded
    /// layouts and do not use this cold constructor.
    pub fn publication_control_bytes(
        population: usize,
        input_capacity: usize,
        source_capacity: usize,
    ) -> Result<u64, WorkingMemoryError> {
        [
            Self::retained_control_bytes(population)?,
            directory::PreparedNamespace::cold_control_bytes::<K>()?,
            RegistryBatch::<K>::cold_control_bytes(population)?,
            cold_metadata::vector_bytes::<Row<K>>(population)?,
            cold_metadata::vector_bytes::<(K, StorageAllocation)>(input_capacity)?,
            cold_metadata::vector_bytes::<(K, SourceInventoryOrigin)>(source_capacity)?,
        ]
        .into_iter()
        .try_fold(0u64, u64::checked_add)
        .ok_or(WorkingMemoryError::Overflow)
    }
}
