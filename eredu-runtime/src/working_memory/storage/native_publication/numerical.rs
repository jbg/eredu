//! Canonical identities for completed backings already paid by a numerical source.
use super::*;
use crate::working_memory::OriginalNumericalBudgetCustody;
use eredu_core::{HostMetadataFunding, HostPreparationAuthority, MemoryPlacement};
use std::{
    marker::PhantomData,
    sync::atomic::{AtomicBool, Ordering as AtomicOrdering},
};

#[derive(Clone, Debug)]
pub(in crate::working_memory) struct NumericalOrigin {
    pub(in crate::working_memory) account: OriginalNumericalBudgetCustody,
    pub(in crate::working_memory) bytes: u64,
    pub(in crate::working_memory) placement: Arc<MemoryPlacement>,
    host_controls: bool,
}
impl NumericalOrigin {
    pub(in crate::working_memory) fn same(&self, other: &Self) -> bool {
        self.account.same_account(&other.account)
            && self.bytes == other.bytes
            && self.placement == other.placement
            && self.host_controls == other.host_controls
    }
}

// The backend authenticates actual native identity/generation and these exact
// capacities. The directory compares all live rows of this canonical key type,
// including earlier batches, before committing any new row. Equal diagnostic
// names never establish coverage; account, key and placement must all agree.
pub(super) fn validate_rows<K: Clone + Ord + Send + Sync + 'static>(
    account: &OriginalNumericalBudgetCustody,
    registry: &Registry<K>,
    rows: &[Row<K>],
) -> Result<(), WorkingMemoryError> {
    for row in rows {
        let Some(PrepaidStorageOrigin::Numerical(origin)) = &row.origin else {
            return Err(WorkingMemoryError::IdentityMismatch);
        };
        if !origin.account.same_account(account)
            || row.completed.is_some()
            || row.existing_only
            || row.source
            || row.placement != origin.placement
            || (origin.host_controls && origin.placement != account.pool().0.host_placement)
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
    }
    let origins = origins(registry, rows, Some(account));
    let controls = control_origins(origins.clone()).try_fold(0u64, |sum, origin| {
        sum.checked_add(origin.bytes)
            .ok_or(WorkingMemoryError::Overflow)
    })?;
    account.validate_completed_metadata(controls)?;
    account.validate_completed_allocations(payload_origins(origins))
}

fn origins<'a, K: Clone + Ord + Send + Sync + 'static>(
    registry: &'a Registry<K>,
    rows: &'a [Row<K>],
    account: Option<&'a OriginalNumericalBudgetCustody>,
) -> impl Iterator<Item = &'a NumericalOrigin> + Clone {
    let prior = registry
        .entries()
        .filter(|entry| !entry.native_retired)
        .filter_map(move |entry| match &entry.prepaid {
            Some(PrepaidStorageOrigin::Numerical(origin))
                if account.is_some_and(|account| origin.account.same_account(account)) =>
            {
                Some(origin)
            }
            _ => None,
        });
    let fresh = rows
        .iter()
        .filter(move |row| {
            registry
                .get(row.key.as_ref().expect("staged key").as_ref())
                .is_none()
        })
        .filter_map(|row| match &row.origin {
            Some(PrepaidStorageOrigin::Numerical(origin)) => Some(origin),
            _ => None,
        });
    prior.chain(fresh)
}
fn control_origins<'a>(
    origins: impl Iterator<Item = &'a NumericalOrigin> + Clone,
) -> impl Iterator<Item = &'a NumericalOrigin> + Clone {
    origins.filter(|origin| origin.host_controls)
}
fn payload_origins<'a>(
    origins: impl Iterator<Item = &'a NumericalOrigin> + Clone,
) -> impl Iterator<Item = (u64, &'a MemoryPlacement)> + Clone {
    origins
        .filter(|origin| !origin.host_controls)
        .map(|origin| (origin.bytes, origin.placement.as_ref()))
}
fn validation_control_bytes<K: Clone + Ord + Send + Sync + 'static>() -> Option<usize> {
    // Only allocation-free iterator descriptors over empty borrowed inputs.
    // Actual validation uses these same factories and closure types.
    let registry = Registry::<K>::new();
    let origins = origins(&registry, &[], None);
    let controls = control_origins(origins.clone());
    let payload = payload_origins(origins.clone());
    let parts = [
        size_of::<Registry<K>>(),
        size_of::<&Row<K>>(),
        size_of::<&Entry>(),
        size_of::<Option<&NumericalOrigin>>(),
        size_of::<(&Registry<K>, &[Row<K>])>(),
        std::mem::size_of_val(&origins).checked_mul(2)?,
        std::mem::size_of_val(&controls),
        // The shared allocation validator retains and clones this same iterator
        // across placement/domain loops without allocating a descriptor vector.
        std::mem::size_of_val(&payload).checked_mul(4)?,
        size_of::<(u64, u64, u64, &MemoryPlacement)>(),
        size_of::<Result<(), WorkingMemoryError>>(),
    ];
    parts
        .into_iter()
        .try_fold(std::mem::size_of_val(&parts), usize::checked_add)
}

/// Prospective metadata for canonical discovery of actual completed numerical
/// backings. Payload and native Host controls remain charged to their original
/// numerical account; publication neither charges nor refunds them a second time.
#[must_use]
pub struct NumericalStoragePublicationPlan<K: Clone + Ord + Send + Sync + 'static> {
    slots: usize,
    bytes: usize,
    marker: PhantomData<fn() -> K>,
}
impl<K: Clone + Ord + Send + Sync + 'static> NumericalStoragePublicationPlan<K> {
    /// Count every observed row, including native Host-control identities.
    /// `nested_key_bytes` describes one actual owned key's out-of-line storage.
    pub fn new(slots: usize, nested_key_bytes: u64) -> Result<Self, WorkingMemoryError> {
        let parts = [
            validation_control_bytes::<K>().ok_or(WorkingMemoryError::Overflow)?,
            usize::try_from(PreparedNativePublication::<K>::qualified_control_bytes(
                slots,
                nested_key_bytes,
            )?)
            .map_err(|_| WorkingMemoryError::Overflow)?,
            HostPreparationAuthority::retention_bytes::<HostMetadataFunding>()
                .ok_or(WorkingMemoryError::Overflow)?,
            HostMetadataFunding::reservation_control_bytes(),
            size_of::<Self>(),
            size_of::<PreparedNumericalStoragePublication<K>>(),
            size_of::<NumericalStorageRegistration<K>>()
                .checked_mul(slots)
                .ok_or(WorkingMemoryError::Overflow)?,
            size_of::<NumericalOrigin>(),
            size_of::<PublicationOrigin>(),
            size_of::<(
                &OriginalNumericalBudgetCustody,
                &MemoryLedger,
                &HostMetadataFunding,
            )>(),
            size_of::<(&Registry<K>, &[Row<K>])>(),
            size_of::<std::slice::Iter<'_, Row<K>>>(),
            size_of::<Option<&WorkingMemoryStorage<K>>>(),
            size_of::<&OriginalNumericalBudgetCustody>(),
            size_of::<(&K, u64, Arc<MemoryPlacement>, bool)>(),
            size_of::<Result<Self, WorkingMemoryError>>(),
            size_of::<Result<PreparedNumericalStoragePublication<K>, WorkingMemoryError>>(),
            size_of::<Result<(), WorkingMemoryError>>(),
            size_of::<std::sync::MutexGuard<'_, crate::working_memory::Usage>>(),
            size_of::<std::sync::LockResult<std::sync::MutexGuard<'_, crate::working_memory::Usage>>>(
            ),
        ];
        let bytes = parts
            .into_iter()
            .try_fold(std::mem::size_of_val(&parts), usize::checked_add)
            .ok_or(WorkingMemoryError::Overflow)?;
        Ok(Self {
            slots,
            bytes,
            marker: PhantomData,
        })
    }
    /// Exact requested publication metadata, reserved by `prepare` on its payer.
    pub fn requested_bytes(&self) -> usize {
        self.bytes
    }
    /// Uses genuine completed numerical custody and the caller's independently
    /// admitted metadata payer. This grants no native work or additional payload.
    pub fn prepare(
        self,
        account: &OriginalNumericalBudgetCustody,
        ledger: &MemoryLedger,
        funding: &HostMetadataFunding,
    ) -> Result<PreparedNumericalStoragePublication<K>, WorkingMemoryError> {
        {
            let usage = ledger
                .0
                .usage
                .lock()
                .map_err(|_| WorkingMemoryError::Poisoned)?;
            account.validate_copy_source(ledger, &usage)?;
        }
        funding.reserve_metadata(self.bytes).map_err(|cause| {
            WorkingMemoryError::MetadataConstruction(
                eredu_nn::workspace::WorkspaceMetadataError::Funding(cause),
            )
        })?;
        let host = HostPreparationAuthority::retain(funding.clone());
        let inputs = crate::working_memory::qualified_storage::vector(self.slots, true)?;
        let placements = crate::working_memory::qualified_storage::vector(self.slots, true)?;
        let rows = crate::working_memory::qualified_storage::vector(self.slots, true)?;
        let node = RegistryBatch::prepare_copy_exact(self.slots, &host)?;
        let namespace = PreparedNamespace::prepare_copy::<K>(&host);
        Ok(PreparedNumericalStoragePublication {
            inner: PreparedNativePublication {
                inputs,
                placements,
                rows,
                node: Some(node),
                namespace: Some(namespace),
                terminal: false,
                published: false,
                slots: self.slots,
                exact_storage: true,
                failure_site: "numerical registry preparation",
                missing_existing_input: None,
                partition: PublicationOrigin::Numerical {
                    account: account.clone(),
                    host,
                },
            },
        })
    }
}

/// One terminal transaction through the existing canonical directory worker.
#[must_use]
pub struct PreparedNumericalStoragePublication<K: Clone + Ord + Send + Sync + 'static> {
    inner: PreparedNativePublication<K>,
}
impl<K: Clone + Ord + Send + Sync + 'static> PreparedNumericalStoragePublication<K> {
    /// The backend must authenticate this allocation identity, generation, full
    /// capacity and placement against the same actual completed numerical owner.
    pub fn push(
        &mut self,
        key: K,
        bytes: u64,
        placement: Arc<MemoryPlacement>,
    ) -> Result<usize, WorkingMemoryError> {
        self.push_kind(key, bytes, placement, false)
    }
    /// Registers the native allocation's separate Host-control identity against
    /// its numerical metadata allowance, never against its payload capacity.
    pub fn push_host_controls(&mut self, key: K, bytes: u64) -> Result<usize, WorkingMemoryError> {
        let placement = self
            .inner
            .partition
            .numerical()
            .expect("numerical source")
            .pool()
            .host_placement_handle();
        self.push_kind(key, bytes, placement, true)
    }
    fn push_kind(
        &mut self,
        key: K,
        bytes: u64,
        placement: Arc<MemoryPlacement>,
        host_controls: bool,
    ) -> Result<usize, WorkingMemoryError> {
        if self.inner.terminal || self.inner.inputs.len() == self.inner.slots {
            return Err(WorkingMemoryError::PreparationAlreadyStarted);
        }
        let origin = NumericalOrigin {
            account: self
                .inner
                .partition
                .numerical()
                .expect("numerical source")
                .clone(),
            bytes,
            placement: Arc::clone(&placement),
            host_controls,
        };
        let index = self.inner.inputs.len();
        self.inner
            .inputs
            .push(NativePublicationInput::Native(NativeStorageWitness {
                key,
                bytes,
                origin: PrepaidStorageOrigin::Numerical(origin),
            }));
        self.inner.placements.push(placement);
        Ok(index)
    }
    /// Validates every row and account before any visible registry mutation.
    pub fn publish(&mut self) -> Result<(), WorkingMemoryError> {
        self.inner.publish_impl(None, None, None)
    }
    /// Moves one distinct row into its actual physical attachment. Repeated
    /// observations have no second receipt. Existing rows yield a pin which
    /// cannot be armed as another independent physical-retirement signal.
    pub fn take_input(&mut self, index: usize) -> Option<NumericalStorageRegistration<K>> {
        if !self.inner.published {
            return None;
        }
        let row = self
            .inner
            .rows
            .iter()
            .find(|row| row.first_input == index)?;
        let fresh = row.locator.is_none();
        let registration = self.inner.take_input(index)?;
        Some(NumericalStorageRegistration {
            registration,
            attached: AtomicBool::new(false),
            fresh,
            account: self
                .inner
                .partition
                .numerical()
                .expect("numerical source")
                .clone(),
        })
    }
}

impl<K: Clone + Ord + Send + Sync + 'static> Drop for PreparedNumericalStoragePublication<K> {
    fn drop(&mut self) {
        if !self.inner.published {
            return;
        }
        let account = self.inner.partition.numerical().expect("numerical source");
        if std::thread::panicking() {
            account.quarantine();
            return;
        }
        // A caller can pin a published row before its physical attachment. If
        // that handoff is abandoned, withdrawing only our owner is insufficient:
        // those pins must no longer authenticate the unfinished publication.
        // This closed origin only marks directory availability; it cannot refund
        // numerical payload or affect an existing row borrowed by this attempt.
        for row in &self.inner.rows {
            if row.locator.is_none() {
                if let Some(registration) = &row.output {
                    registration.retire_native_backing();
                }
            }
        }
    }
}

/// Move-only receipt attached to the actual native backing. The native budget's
/// existing observer alone refunds numerical bytes; this receipt retires only
/// canonical discovery and the separately funded directory metadata.
#[must_use]
pub struct NumericalStorageRegistration<K: Ord + Send + 'static> {
    registration: WorkingMemoryStorage<K>,
    attached: AtomicBool,
    fresh: bool,
    account: OriginalNumericalBudgetCustody,
}
impl<K: Ord + Send + 'static> NumericalStorageRegistration<K> {
    /// Arm immediately before the actual backing attachment. An existing row
    /// already has its physical owner and cannot acquire another retirement hook.
    pub fn arm_attachment(&self) -> Result<(), WorkingMemoryError> {
        if !self.fresh || self.attached.swap(true, AtomicOrdering::AcqRel) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        Ok(())
    }
    /// Disarm only when the native attachment returned this same unused owner.
    /// Its destruction then withdraws this fresh publication, rejecting any
    /// surviving directory pins without refunding the live numerical payload.
    pub fn disarm_attachment(&self) {
        self.attached.store(false, AtomicOrdering::Release);
    }
}
impl<K: Ord + Send + 'static> Drop for NumericalStorageRegistration<K> {
    fn drop(&mut self) {
        // Every fresh receipt owns this row's publication lifetime. An armed
        // receipt ends with physical retirement; an unarmed one ends with a
        // failed/abandoned handoff. Both invalidate discovery, neither refunds
        // numerical bytes. Pins borrowed from existing rows are unaffected.
        if self.fresh {
            if std::thread::panicking() {
                self.account.quarantine();
            } else {
                self.registration.retire_native_backing();
            }
        }
    }
}

#[cfg(test)]
mod tests;
