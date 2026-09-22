//! Fixed original reset account; no generic scope, refill, or public guard API.
use super::*;
use crate::working_memory::storage::reset_layout::ResetLayoutPin;
use eredu_core::{DomainMemoryRequirements, MemoryDomainId, MemoryLimit, MemoryLimits};

#[derive(Debug)]
pub(in crate::working_memory) struct Entry {
    id: u64,
    capacity: Arc<MemoryLimits>,
    active: bool,
    sources: Vec<(crate::HostMetadataKey, u64)>,
    next: Option<Box<Entry>>,
}
#[derive(Clone, Debug)]
pub(in crate::working_memory) struct Pending {
    id: u64,
    capacity: Arc<MemoryLimits>,
}
pub(in crate::working_memory) fn capacity(
    usage: &super::super::Usage,
    domain: MemoryDomainId,
) -> Result<MemoryLimit, WorkingMemoryError> {
    let mut limit = MemoryLimit::Unlimited;
    if let Some(pending) = &usage.reset_pending {
        limit = limit.minimum(pending.capacity.get(domain)?);
    }
    // Domain identities have already been validated against the immutable ledger.
    let slot = usage.domain_slot(domain)?;
    limit = limit.minimum(usage.domains[slot].reset_retiring_limit);
    let mut entry = usage.reset_entries.as_deref();
    while let Some(value) = entry {
        limit = limit.minimum(value.capacity.get(domain)?);
        entry = value.next.as_deref();
    }
    Ok(limit)
}
#[derive(Debug)]
struct Charge {
    pool: MemoryLedger,
    id: u64,
    requirements: Option<DomainMemoryRequirements>,
}
struct Retirement {
    entry: Option<Box<Entry>>,
    capacity: Arc<MemoryLimits>,
    active: bool,
}
impl Drop for Charge {
    fn drop(&mut self) {
        let retired = {
            let Ok(mut usage) = self.pool.0.usage.lock() else {
                return;
            };
            let pending = usage.reset_pending.as_ref().filter(|p| p.id == self.id);
            let mut lookup = usage.reset_entries.as_deref();
            let matched = loop {
                match lookup {
                    Some(entry) if entry.id == self.id => break Some(&entry.capacity),
                    Some(entry) => lookup = entry.next.as_deref(),
                    None => break None,
                }
            };
            let Some(capacity) = pending.map(|p| &p.capacity).or(matched) else {
                return;
            };
            let Some(count) = usage.reset_retiring_count.checked_add(1) else {
                return;
            };
            // Keep the ceiling while its paid entry allocation is destroyed.
            // Concurrent retirement conservatively retains the minimum until
            // the final retiring owner atomically refunds and clears it.
            let capacity = Arc::clone(capacity);
            let was_pending = pending.is_some();
            usage.reset_retiring_count = count;
            for (slot, (domain, _)) in self.pool.topology().domains().enumerate() {
                usage.domains[slot].reset_retiring_limit = usage.domains[slot]
                    .reset_retiring_limit
                    .minimum(capacity.get(domain).expect("validated reset limits"));
            }
            if was_pending {
                usage.reset_pending = None;
                Retirement {
                    entry: None,
                    capacity,
                    active: true,
                }
            } else {
                let mut link = &mut usage.reset_entries;
                loop {
                    if link.as_ref().is_some_and(|e| e.id == self.id) {
                        let mut entry = link.take().expect("matched reset entry");
                        *link = entry.next.take();
                        let active = entry.active;
                        break Retirement {
                            entry: Some(entry),
                            capacity,
                            active,
                        };
                    }
                    link = &mut link.as_mut().expect("located reset entry").next;
                }
            }
        };
        let Retirement {
            entry,
            capacity,
            active,
        } = retired;
        // Entry contains only scalar/payload-free keys. Its Box allocation
        // retires outside Usage, while every byte and ceiling remain live.
        drop(entry);
        drop(capacity);
        #[cfg(test)]
        super::tests::after_entry_retirement(&self.pool);
        let requirements = self.requirements.take().expect("live reset requirements");
        let host = self.pool.topology().host_domain();
        let host_charge = requirements.get(host).expect("same reset topology");
        if let Ok(mut usage) = self.pool.0.usage.lock() {
            for (slot, (domain_id, charge)) in requirements.iter().enumerate() {
                if domain_id == host {
                    continue;
                }
                let domain = &mut usage.domains[slot];
                domain.reserved -= charge.total().expect("accepted reset requirement");
                domain.placement_allowances -= charge.placement_allowance_bytes;
                domain.estimates -= charge.estimated_overhead_bytes;
                domain.headroom -= charge.headroom_bytes;
            }
        } else {
            return;
        }
        // The dense account vector is host storage. Keep the entire host charge
        // and retiring ceilings until that final allocation is gone, outside Usage.
        drop(requirements);
        if let Ok(mut usage) = self.pool.0.usage.lock() {
            let slot = usage.domain_slot(host).expect("same reset topology");
            let domain = &mut usage.domains[slot];
            domain.reserved -= host_charge.total().expect("accepted reset requirement");
            domain.placement_allowances -= host_charge.placement_allowance_bytes;
            domain.estimates -= host_charge.estimated_overhead_bytes;
            domain.headroom -= host_charge.headroom_bytes;
            if active {
                usage.reservations -= 1;
            }
            usage.reset_retiring_count -= 1;
            if usage.reset_retiring_count == 0 {
                for domain in &mut usage.domains {
                    domain.reset_retiring_limit = MemoryLimit::Unlimited;
                }
            }
        }
    }
}

struct Hold {
    // Only the shared layout entry survives success; never a predecessor table.
    layout: ResetLayoutPin,
    charge: Charge,
}
/// Crate-private fixed constructor/metadata custody; no arbitrary attachment API.
pub(crate) struct ResetCustody(Option<Arc<Hold>>);
impl std::fmt::Debug for ResetCustody {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResetCustody")
            .field("bytes", &self.bytes())
            .finish()
    }
}
impl Clone for ResetCustody {
    fn clone(&self) -> Self {
        Self(Some(Arc::clone(
            self.0.as_ref().expect("live reset custody"),
        )))
    }
}
impl Drop for ResetCustody {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            drop(Arc::into_inner(owner));
        }
    }
}
impl ResetCustody {
    fn hold(&self) -> &Hold {
        self.0.as_deref().expect("live reset custody")
    }
    pub(super) fn same_constructor(&self, other: &Self) -> bool {
        Arc::ptr_eq(
            self.0.as_ref().expect("live custody"),
            other.0.as_ref().expect("live custody"),
        )
    }
    pub(super) fn bytes(&self) -> u64 {
        self.hold()
            .charge
            .requirements
            .as_ref()
            .expect("live reset requirements")
            .get(self.pool().topology().host_domain())
            .expect("same topology")
            .total()
            .expect("accepted charge")
    }
    pub(crate) fn same_accounting_owner(
        &self,
        domain: &eredu_core::SharedStorageAccountingId,
    ) -> bool {
        self.hold().charge.pool.shared_storage_accounting_id() == domain
    }
    pub(super) fn pool(&self) -> &MemoryLedger {
        &self.hold().charge.pool
    }
    pub(super) fn validate_source(
        &self,
        pool: &MemoryLedger,
        metadata: &crate::HostSlotMetadata,
    ) -> Result<(), WorkingMemoryError> {
        if !self.pool().same_ledger(pool) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let usage = pool
            .0
            .usage
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        self.validate_source_in(pool, metadata, &usage)
    }
    pub(super) fn validate_source_in(
        &self,
        pool: &MemoryLedger,
        metadata: &crate::HostSlotMetadata,
        usage: &super::super::Usage,
    ) -> Result<(), WorkingMemoryError> {
        if !self.pool().same_ledger(pool) || !metadata.original_source_is_live() {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let mut entry = usage.reset_entries.as_deref();
        while let Some(value) = entry {
            if value.id == self.hold().charge.id {
                return if !value.active
                    && value.sources.iter().any(|(key, bytes)| {
                        key == metadata.identity().registry_key()
                            && Some(*bytes) == metadata.capacity_bytes()
                    }) {
                    Ok(())
                } else {
                    Err(WorkingMemoryError::IdentityMismatch)
                };
            }
            entry = value.next.as_deref();
        }
        Err(WorkingMemoryError::IdentityMismatch)
    }

    pub(super) fn finish(
        &self,
        sources: Vec<(crate::HostMetadataKey, u64)>,
    ) -> Result<(), WorkingMemoryError> {
        let mut usage = self
            .pool()
            .0
            .usage
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        let mut entry = usage.reset_entries.as_deref_mut();
        while let Some(value) = entry {
            if value.id == self.hold().charge.id {
                if !value.active || !value.sources.is_empty() {
                    return Err(WorkingMemoryError::IdentityMismatch);
                }
                value.sources = sources; // move only; old Vec has no allocation
                value.active = false;
                usage.reservations -= 1;
                return Ok(());
            }
            entry = value.next.as_deref_mut();
        }
        Err(WorkingMemoryError::IdentityMismatch)
    }
}

// Concrete source and pin representations, including their return/drop overlap.
/// Dense comparison vectors, resolution scratch and the retained shared limits.
/// The host-only reset carries no placement-estimate or diagnostic string rows.
pub(super) fn domain_control_bytes(topology: &eredu_core::MemoryTopology) -> Option<u64> {
    let per_domain = size_of::<eredu_core::DomainMemoryCharge>()
        .checked_mul(3)?
        .checked_add(size_of::<Option<MemoryLimit>>())?
        .checked_add(size_of::<MemoryLimit>())?;
    u64::try_from(
        per_domain
            .checked_mul(topology.len())?
            .checked_add(arc_bytes::<MemoryLimits>()?)?,
    )
    .ok()
}

pub(super) fn control_bytes<K: HostSlotStorageKey>(grouped: bool) -> Option<usize> {
    [
        arc_bytes::<Hold>()?,
        crate::working_memory::storage::reset_layout::retirement_control_bytes::<K>()?,
        size_of::<Retirement>(),
        size_of::<Hold>(),
        size_of::<Option<Hold>>(),
        size_of::<Charge>(),
        size_of::<eredu_core::DomainMemoryCharge>(),
        size_of::<Entry>(),
        size_of::<Box<Entry>>(),
        size_of::<Option<Box<Entry>>>(),
        size_of::<AdmissionFailure>(),
        size_of::<Result<Admission, AdmissionFailure>>(),
        size_of::<Admission>(),
        size_of::<Pending>(),
        size_of::<SourceOwner<K>>(),
        if grouped {
            size_of::<WorkingMemoryStorage<K>>()
        } else {
            0
        },
        size_of::<TableSourceCustody>(),
        size_of::<Option<TableSourceCustody>>(),
        size_of::<ResetLayoutPin>(),
        size_of::<Result<ResetLayoutPin, WorkingMemoryError>>(),
        crate::working_memory::storage::reset_layout::pair_control_bytes()?,
        size_of::<ResetCustody>(),
    ]
    .into_iter()
    .try_fold(0usize, usize::checked_add)
}

pub(super) enum TableSourceCustody {
    Ordinary(#[allow(dead_code)] Box<dyn Send + Sync>),
    Original(#[allow(dead_code)] crate::HostSlotMetadata),
    Registered(#[allow(dead_code)] ResetLayoutPin),
}
// Actual source cloning is allocation-free and happens outside Usage. The
// ordinary erasure allocation is delayed until after the original comparison.
enum SourceOwner<K: HostSlotStorageKey> {
    Ordinary(WorkingMemoryStorage<K>),
    Original(crate::HostSlotMetadata),
    Registered(ResetLayoutPin),
}
impl<K: HostSlotStorageKey> SourceOwner<K> {
    fn retain(self) -> TableSourceCustody {
        match self {
            Self::Ordinary(owner) => TableSourceCustody::Ordinary(Box::new(owner)),
            Self::Original(metadata) => TableSourceCustody::Original(metadata),
            Self::Registered(pin) => TableSourceCustody::Registered(pin),
        }
    }
}
pub(super) struct Admission {
    pub(super) source: TableSourceCustody,
    pub(super) custody: ResetCustody,
}

pub(super) struct AdmissionFailure {
    pub(super) cause: WorkingMemoryError,
    pub(super) entry: Option<Box<Entry>>,
    pub(super) source: Option<TableSourceCustody>,
    pub(super) custody: Option<ResetCustody>,
}
impl From<WorkingMemoryError> for AdmissionFailure {
    fn from(cause: WorkingMemoryError) -> Self {
        Self {
            cause,
            entry: None,
            source: None,
            custody: None,
        }
    }
}
impl std::fmt::Debug for AdmissionFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AdmissionFailure")
            .field("cause", &self.cause)
            .finish_non_exhaustive()
    }
}

pub(super) fn admit<K: HostSlotStorageKey>(
    pool: &MemoryLedger,
    source: &TableSource<'_, K>,
    source_table: (&crate::HostMetadataKey, u64),
    source_layout: (&crate::HostMetadataKey, u64),
    requirements: DomainMemoryRequirements,
    limits: MemoryLimits,
) -> Result<Admission, AdmissionFailure> {
    requirements
        .validate(pool.topology())
        .map_err(WorkingMemoryError::from)?;
    let capacity = Arc::new(limits);
    let (mut source_owner, mut layout_pin) = match source {
        TableSource::Ordinary { registration, .. } => (
            SourceOwner::Ordinary((*registration).clone()),
            ResetLayoutPin::prepare::<K>(pool, source_layout.0, source_layout.1)?,
        ),
        TableSource::Original { metadata, custody } => (
            SourceOwner::<K>::Original((*metadata).clone()),
            custody
                .hold()
                .layout
                .prepare_again::<K>(pool, source_layout.0, source_layout.1)?,
        ),
        TableSource::Registered { .. } => (
            SourceOwner::<K>::Registered(ResetLayoutPin::prepare::<K>(
                pool,
                source_table.0,
                source_table.1,
            )?),
            ResetLayoutPin::prepare::<K>(pool, source_layout.0, source_layout.1)?,
        ),
    };
    let charge = {
        let mut usage = pool
            .0
            .usage
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        if usage.reset_pending.is_some() {
            return Err(WorkingMemoryError::ResetAdmissionBusy.into());
        }
        let mut table_locator = None;
        let layout_locator = match source {
            TableSource::Ordinary {
                registration,
                table,
                layout,
            } => {
                registration.validate_copy_source(pool, &usage)?;
                registration.validate_host_key(&usage, table, source_table.1)?;
                Some(registration.locate_host_key(&usage, layout, source_layout.1)?)
            }
            TableSource::Original { metadata, custody } => {
                if metadata.identity().registry_key() != source_table.0
                    || metadata.capacity_bytes() != Some(source_table.1)
                {
                    return Err(WorkingMemoryError::IdentityMismatch.into());
                }
                custody.validate_source_in(pool, metadata, &usage)?;
                None
            }
            TableSource::Registered { table, layout } => {
                table_locator = Some(ResetLayoutPin::locate_existing::<K>(
                    &usage,
                    table,
                    source_table.1,
                )?);
                Some(ResetLayoutPin::locate_existing::<K>(
                    &usage,
                    layout,
                    source_layout.1,
                )?)
            }
        };
        if usage.unquoted_owners != 0 {
            return Err(WorkingMemoryError::UnknownBound.into());
        }
        let id = usage.next_reset;
        let next_id = id.checked_add(1).ok_or(WorkingMemoryError::Overflow)?;
        let commit = super::super::PreparedAccountCommit::prepare_reset(
            pool,
            &usage,
            &requirements,
            &capacity,
        )?;
        // Acquire exactly one existing layout owner after all budget checks.
        // No fallible work follows these scalar updates before Charge exists.
        match source {
            TableSource::Ordinary { registration, .. } => layout_pin.acquire_registered(
                registration,
                layout_locator.expect("ordinary layout located in same loan"),
                &mut usage,
            )?,
            TableSource::Original { .. } => layout_pin.acquire_again(&mut usage)?,
            TableSource::Registered { .. } => {
                let SourceOwner::Registered(table_pin) = &mut source_owner else {
                    unreachable!("same private source variant")
                };
                ResetLayoutPin::acquire_pair::<K>(
                    table_pin,
                    &mut layout_pin,
                    table_locator.expect("table located in same loan"),
                    layout_locator.expect("layout located in same loan"),
                    &mut usage,
                )?;
            }
        }
        commit.commit(pool, &mut usage);
        usage.next_reset = next_id;
        usage.reset_pending = Some(Pending {
            id,
            capacity: Arc::clone(&capacity),
        });
        Charge {
            pool: pool.clone(),
            id,
            requirements: Some(requirements),
        }
    };
    let id = charge.id;
    let mut entry = Some(Box::new(Entry {
        id,
        capacity,
        active: true,
        sources: Vec::new(),
        next: None,
    }));
    let owner = ResetCustody(Some(Arc::new(Hold {
        layout: layout_pin,
        charge,
    })));
    let source = source_owner.retain();
    #[cfg(test)]
    super::tests::before_entry_publication(pool);
    let installed = match pool.0.usage.lock() {
        Err(_) => Err(WorkingMemoryError::Poisoned),
        Ok(mut usage) => {
            if !usage.reset_pending.as_ref().is_some_and(|p| p.id == id) {
                Err(WorkingMemoryError::IdentityMismatch)
            } else {
                let mut node = entry.take().expect("unpublished fixed entry");
                node.next = usage.reset_entries.take();
                usage.reset_entries = Some(node);
                usage.reset_pending = None;
                Ok(())
            }
        }
    };
    if let Err(cause) = installed {
        return Err(AdmissionFailure {
            cause,
            entry,
            source: Some(source),
            custody: Some(owner),
        });
    }
    Ok(Admission {
        source,
        custody: owner,
    })
}

// Fixed source population, prepared after the same reset account accepts.
// Every pin is existing-only; no new physical registration or allowance.
pub(super) enum ChildSourceCustody {
    Original(#[allow(dead_code)] crate::HostSlotMetadata),
    Registered(#[allow(dead_code)] ResetLayoutPin),
}
pub(super) fn pin_child<K: HostSlotStorageKey>(
    pool: &MemoryLedger,
    metadata: &crate::HostSlotMetadata,
) -> Result<ChildSourceCustody, WorkingMemoryError> {
    if let Some(custody) = metadata.original_reset_custody() {
        custody.validate_source(pool, metadata)?;
        Ok(ChildSourceCustody::Original(metadata.clone()))
    } else {
        ResetLayoutPin::pin_existing_host::<K>(pool, metadata).map(ChildSourceCustody::Registered)
    }
}
pub(super) fn table_membership_bytes(count: usize) -> Option<usize> {
    std::alloc::Layout::array::<(crate::HostMetadataKey, u64)>(count)
        .ok()?
        .size()
        .checked_add(size_of::<Vec<(crate::HostMetadataKey, u64)>>())?
        .checked_add(size_of::<(crate::HostMetadataKey, u64)>())?
        .checked_add(size_of::<crate::HostMetadataKey>())?
        .checked_add(size_of::<Result<(), std::collections::TryReserveError>>())
}
