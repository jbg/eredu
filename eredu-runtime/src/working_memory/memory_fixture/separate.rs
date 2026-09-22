use crate::working_memory::*;
use eredu_core::WorkspaceBound;
use std::sync::Arc;
// Device payload and host constructor budgets remain independent. Every account
// admits up to eight overlapping publications, each with three keys no larger than eight words.
pub(in crate::working_memory) fn device_location() -> eredu_core::MemoryLocation {
    eredu_core::MemoryLocation::Device(eredu_core::MemoryDeviceId {
        backend: "handoff-fixture",
        ordinal: 0,
    })
}
pub(in crate::working_memory) fn device_domain(pool: &MemoryLedger) -> eredu_core::MemoryDomainId {
    pool.topology().domain_for(device_location()).unwrap()
}
pub(in crate::working_memory) fn device_placement(
    pool: &MemoryLedger,
) -> Arc<eredu_core::MemoryPlacement> {
    Arc::new(eredu_core::MemoryPlacement::fixed(pool.topology(), device_domain(pool)).unwrap())
}
pub(in crate::working_memory) fn device_ledger(
    capacity: u64,
    existing: u64,
) -> Result<MemoryLedger, WorkingMemoryError> {
    static TOPOLOGY: std::sync::OnceLock<Arc<eredu_core::MemoryTopology>> =
        std::sync::OnceLock::new();
    let topology = Arc::clone(TOPOLOGY.get_or_init(|| {
        Arc::new(
            eredu_core::MemoryTopology::new(vec![
                eredu_core::MemoryDomainDescription {
                    name: "host".into(),
                    locations: vec![eredu_core::MemoryLocation::Host],
                },
                eredu_core::MemoryDomainDescription {
                    name: "device".into(),
                    locations: vec![device_location()],
                },
            ])
            .unwrap(),
        )
    }));
    let domain = topology.domain_for(device_location())?;
    let limits = eredu_core::MemoryLimits::resolve(
        &topology,
        [(domain, eredu_core::MemoryLimit::Finite(capacity))],
    )?;
    let mut baseline = eredu_core::DomainMemoryRequirements::zero(&topology);
    baseline.add_allocation(
        existing,
        &eredu_core::MemoryPlacement::fixed(&topology, domain)?,
    )?;
    MemoryLedger::new(topology, limits, baseline)
}
pub(in crate::working_memory) fn device_limits(
    pool: &MemoryLedger,
    capacity: u64,
) -> eredu_core::MemoryLimits {
    eredu_core::MemoryLimits::resolve(
        pool.topology(),
        [(
            device_domain(pool),
            eredu_core::MemoryLimit::Finite(capacity),
        )],
    )
    .unwrap()
}
pub(in crate::working_memory) fn host_publication_bytes() -> u64 {
    8 * (MemoryLedger::storage_metadata_control_bytes().unwrap()
        + crate::working_memory::StoragePublicationLayout::<[usize; 8]>::new(3)
            .unwrap()
            .requested_bytes())
}
pub(in crate::working_memory) trait DeviceLedgerFixture {
    fn device_used_bytes(&self) -> Result<u64, WorkingMemoryError>;
    fn device_capacity(&self) -> Result<u64, WorkingMemoryError>;
    fn device_peak_bytes(&self) -> Result<u64, WorkingMemoryError>;
    fn register_device_storage<K: Clone + Ord + Send + 'static>(
        &self,
        storage: impl IntoIterator<Item = (K, u64)>,
    ) -> Result<WorkingMemoryStorage<K>, WorkingMemoryError>;
    fn register_device_storage_individually<K: Clone + Ord + Send + 'static>(
        &self,
        storage: impl IntoIterator<Item = (K, u64)>,
    ) -> Result<crate::working_memory::StorageRegistrations<K>, WorkingMemoryError>;
}
impl DeviceLedgerFixture for MemoryLedger {
    fn register_device_storage_individually<K: Clone + Ord + Send + 'static>(
        &self,
        storage: impl IntoIterator<Item = (K, u64)>,
    ) -> Result<crate::working_memory::StorageRegistrations<K>, WorkingMemoryError> {
        let entries = storage.into_iter().collect::<Vec<_>>();
        crate::working_memory::StoragePublicationLayout::new(entries.len())?
            .fund(self)?
            .register_storage_individually(entries.into_iter().map(|(key, bytes)| {
                (
                    key,
                    crate::working_memory::StorageAllocation::new(bytes, device_placement(self)),
                )
            }))
    }
    fn device_peak_bytes(&self) -> Result<u64, WorkingMemoryError> {
        Ok(self.snapshot()?.domains[1].historical_peak_bytes)
    }
    fn device_used_bytes(&self) -> Result<u64, WorkingMemoryError> {
        Ok(self.snapshot()?.domains[1].current_charge_bytes)
    }
    fn device_capacity(&self) -> Result<u64, WorkingMemoryError> {
        match self.snapshot()?.domains[1].effective_limit {
            eredu_core::MemoryLimit::Finite(bytes) => Ok(bytes),
            eredu_core::MemoryLimit::Unlimited => Err(WorkingMemoryError::IdentityMismatch),
        }
    }
    fn register_device_storage<K: Clone + Ord + Send + 'static>(
        &self,
        storage: impl IntoIterator<Item = (K, u64)>,
    ) -> Result<WorkingMemoryStorage<K>, WorkingMemoryError> {
        let entries = storage.into_iter().collect::<Vec<_>>();
        crate::working_memory::StoragePublicationLayout::new(entries.len())?
            .fund(self)?
            .register_storage(entries.into_iter().map(|(key, bytes)| {
                (
                    key,
                    crate::working_memory::StorageAllocation::new(bytes, device_placement(self)),
                )
            }))
    }
}
pub(in crate::working_memory) trait DeviceFundingFixture {
    fn adopt_device_storage<K: Clone + Ord + Send + 'static>(
        &self,
        storage: impl IntoIterator<Item = (K, u64)>,
    ) -> Result<crate::working_memory::StorageRegistrations<K>, WorkingMemoryError>;
}
impl DeviceFundingFixture for crate::working_memory::WorkingMemoryFundingScope {
    fn adopt_device_storage<K: Clone + Ord + Send + 'static>(
        &self,
        storage: impl IntoIterator<Item = (K, u64)>,
    ) -> Result<crate::working_memory::StorageRegistrations<K>, WorkingMemoryError> {
        let entries = storage.into_iter().collect::<Vec<_>>();
        assert!(
            entries.len() <= 3 && std::mem::size_of::<K>() <= std::mem::size_of::<[usize; 8]>()
        );
        crate::working_memory::StoragePublicationLayout::new(entries.len())?
            .fund_from(self)?
            .adopt_storage_individually(
                self,
                entries.into_iter().map(|(key, bytes)| {
                    (
                        key,
                        crate::working_memory::StorageAllocation::new(
                            bytes,
                            device_placement(self.pool()),
                        ),
                    )
                }),
            )
    }
}

pub(in crate::working_memory) fn device_reservation(
    pool: &MemoryLedger,
    bytes: u64,
    capacity: u64,
) -> WorkingMemoryReservation {
    let mut admission =
        crate::working_memory::memory_fixture::host_admission(pool, host_publication_bytes());
    let mut requirements = eredu_core::DomainMemoryRequirements::zero(pool.topology());
    requirements
        .add_allocation(bytes, &device_placement(pool))
        .unwrap();
    let workspace = admission.state.execution_workspace.as_mut().unwrap();
    workspace.materialization = WorkspaceBound::bounded(bytes, "fixture device payload");
    workspace.physical_domains.as_mut().unwrap().materialization = requirements;
    admission.incremental_required_bytes = Some(bytes + host_publication_bytes());
    pool.reserve_with_capacity(
        &InferenceExecutionIdentity::default(),
        &admission,
        device_limits(pool, capacity),
    )
    .unwrap()
}
pub(in crate::working_memory) fn device_request_bytes(
    reservation: &WorkingMemoryReservation,
) -> Option<u64> {
    reservation
        .requirements()
        .get(device_domain(&reservation.0.pool))
        .ok()?
        .total()
        .ok()
}
