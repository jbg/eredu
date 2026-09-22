#![allow(dead_code)]
//! Explicit host allocation facts for neutral runtime fixtures.
use eredu_core::*;
use eredu_runtime::working_memory::*;
use std::sync::{Arc, OnceLock};

fn topology_owner() -> &'static Arc<MemoryTopology> {
    static TOPOLOGY: OnceLock<Arc<MemoryTopology>> = OnceLock::new();
    TOPOLOGY.get_or_init(|| {
        Arc::new(
            MemoryTopology::new(vec![MemoryDomainDescription {
                name: "host".into(),
                locations: vec![MemoryLocation::Host],
            }])
            .unwrap(),
        )
    })
}
pub fn topology() -> Arc<MemoryTopology> {
    Arc::clone(topology_owner())
}
pub fn topology_ref() -> &'static MemoryTopology {
    topology_owner()
}
pub fn placement_ref() -> &'static MemoryPlacement {
    static PLACEMENT: OnceLock<MemoryPlacement> = OnceLock::new();
    PLACEMENT.get_or_init(|| {
        MemoryPlacement::fixed(topology_ref(), topology_ref().host_domain()).unwrap()
    })
}
pub fn placement() -> Arc<MemoryPlacement> {
    let topology = topology();
    Arc::new(MemoryPlacement::fixed(&topology, topology.host_domain()).unwrap())
}
pub fn requirements(bytes: u64) -> DomainMemoryRequirements {
    let mut value = DomainMemoryRequirements::zero(&topology());
    value.add_allocation(bytes, &placement()).unwrap();
    value
}
fn fixed_bytes() -> u64 {
    let topology = topology();
    MemoryLedger::fixed_owner_bytes(
        &topology,
        &MemoryLimits::unlimited(&topology),
        &requirements(0),
    )
    .unwrap()
}
pub fn limits(capacity: u64) -> MemoryLimitDeclarations {
    let physical = if capacity == u64::MAX {
        capacity
    } else {
        capacity.checked_add(fixed_bytes()).unwrap()
    };
    MemoryLimitDeclarations::new([("host".into(), MemoryLimit::Finite(physical))])
}
pub fn resolved_limits(capacity: u64) -> MemoryLimits {
    limits(capacity).resolve(&topology()).unwrap()
}
pub fn host_ledger(capacity: u64, existing: u64) -> Result<MemoryLedger, WorkingMemoryError> {
    MemoryLedger::new(
        topology(),
        resolved_limits(capacity),
        requirements(existing),
    )
}
pub fn unlimited_ledger(existing: u64) -> MemoryLedger {
    MemoryLedger::new(
        topology(),
        MemoryLimits::unlimited(&topology()),
        requirements(existing),
    )
    .unwrap()
}
pub fn workspace(mut value: ExecutionWorkspaceEstimate) -> ExecutionWorkspaceEstimate {
    let fields = [
        &value.activations,
        &value.attention,
        &value.vocabulary,
        &value.state_update,
        &value.materialization,
        &value.retained,
    ];
    if let Some(bytes) = fields
        .map(|b| b.bytes())
        .into_iter()
        .collect::<Option<Vec<_>>>()
    {
        value.physical_domains = Some(DomainExecutionWorkspaceEstimate {
            geometry: value.geometry,
            activations: requirements(bytes[0]),
            attention: requirements(bytes[1]),
            vocabulary: requirements(bytes[2]),
            state_update: requirements(bytes[3]),
            materialization: requirements(bytes[4]),
            retained: requirements(bytes[5]),
        });
    }
    value
}
pub fn state(mut value: RuntimeStateEstimate) -> RuntimeStateEstimate {
    if let Some(w) = value.execution_workspace.take() {
        let geometry = w.geometry;
        value.execution_workspace = Some(workspace(w));
        let decoder = value
            .requested_state_bytes
            .checked_sub(value.multimodal_embedding_bytes)
            .and_then(|b| b.checked_sub(value.media_execution_workspace_bytes))
            .unwrap();
        value.physical_domains = Some(DomainRuntimeStateEstimate {
            geometry,
            decoder_state: requirements(decoder),
            media_embeddings: requirements(value.multimodal_embedding_bytes),
            media_workspace: requirements(value.media_execution_workspace_bytes),
        });
    }
    value
}
pub fn admission(mut value: Admission) -> Admission {
    value.state = state(value.state);
    value
}
pub trait LedgerFixture {
    fn funded_used_bytes(&self) -> Result<u64, WorkingMemoryError>;
    fn charged_bytes(&self) -> Result<u64, WorkingMemoryError>;
    fn payload_used_bytes(&self) -> Result<u64, WorkingMemoryError>;
    fn payload_peak_bytes(&self) -> Result<u64, WorkingMemoryError>;
    fn payload_effective_capacity(&self) -> Result<u64, WorkingMemoryError>;
}
impl LedgerFixture for MemoryLedger {
    fn funded_used_bytes(&self) -> Result<u64, WorkingMemoryError> {
        let row = &self.snapshot()?.domains[0];
        Ok(row
            .current_charge_bytes
            .checked_sub(fixed_bytes())
            .unwrap()
            .checked_sub(row.registry_metadata_bytes)
            .unwrap())
    }
    fn charged_bytes(&self) -> Result<u64, WorkingMemoryError> {
        Ok(self.snapshot()?.domains[0]
            .current_charge_bytes
            .checked_sub(fixed_bytes())
            .unwrap())
    }
    fn payload_used_bytes(&self) -> Result<u64, WorkingMemoryError> {
        let current = &self.snapshot()?.domains[0];
        Ok(current
            .current_charge_bytes
            .checked_sub(fixed_bytes())
            .and_then(|n| n.checked_sub(current.registry_metadata_bytes))
            .and_then(|n| n.checked_sub(current.reservation_control_bytes))
            .unwrap())
    }
    fn payload_peak_bytes(&self) -> Result<u64, WorkingMemoryError> {
        Ok(self.snapshot()?.domains[0]
            .historical_peak_bytes
            .checked_sub(fixed_bytes())
            .unwrap())
    }
    fn payload_effective_capacity(&self) -> Result<u64, WorkingMemoryError> {
        match self.snapshot()?.domains[0].effective_limit {
            MemoryLimit::Finite(bytes) => Ok(bytes.checked_sub(fixed_bytes()).unwrap()),
            MemoryLimit::Unlimited => Err(WorkingMemoryError::IdentityMismatch),
        }
    }
}

pub trait FundingFixture {
    fn adopt_host_storage_individually<K: Clone + Ord + Send + 'static>(
        &self,
        storage: impl IntoIterator<Item = (K, u64)>,
    ) -> Result<StorageRegistrations<K>, WorkingMemoryError>;
}
impl FundingFixture for WorkingMemoryFundingScope {
    fn adopt_host_storage_individually<K: Clone + Ord + Send + 'static>(
        &self,
        storage: impl IntoIterator<Item = (K, u64)>,
    ) -> Result<StorageRegistrations<K>, WorkingMemoryError> {
        let values = storage.into_iter().collect::<Vec<_>>();
        let grant = self.prepare_storage_metadata().map_err(funding_error)?;
        StoragePublicationLayout::new(values.len())?
            .prepare(self.pool(), &grant)?
            .adopt_storage_individually(
                self,
                values
                    .into_iter()
                    .map(|(key, bytes)| (key, StorageAllocation::new(bytes, placement()))),
            )
    }
}
fn funding_error(error: HostMetadataFundingError) -> WorkingMemoryError {
    match error {
        HostMetadataFundingError::Domain(error) => WorkingMemoryError::Domain(error),
        HostMetadataFundingError::Overflow => WorkingMemoryError::Overflow,
        error => WorkingMemoryError::MetadataConstruction(
            eredu_nn::workspace::WorkspaceMetadataError::Funding(error),
        ),
    }
}
pub trait StorageFixture {
    fn register_host_storage_individually<K: Clone + Ord + Send + 'static>(
        &self,
        storage: impl IntoIterator<Item = (K, u64)>,
    ) -> Result<StorageRegistrations<K>, WorkingMemoryError>;
    fn register_host_storage<K: Clone + Ord + Send + 'static>(
        &self,
        storage: impl IntoIterator<Item = (K, u64)>,
    ) -> Result<WorkingMemoryStorage<K>, WorkingMemoryError>;
    fn register_storage<K: Clone + Ord + Send + 'static>(
        &self,
        storage: impl IntoIterator<Item = (K, StorageAllocation)>,
    ) -> Result<WorkingMemoryStorage<K>, WorkingMemoryError>;
    fn pin_registered_storage<K: Clone + Ord + Send + 'static>(
        &self,
        storage: impl IntoIterator<Item = (K, u64)>,
    ) -> Result<WorkingMemoryStorage<K>, WorkingMemoryError>;
}
impl StorageFixture for MemoryLedger {
    fn register_host_storage_individually<K: Clone + Ord + Send + 'static>(
        &self,
        storage: impl IntoIterator<Item = (K, u64)>,
    ) -> Result<StorageRegistrations<K>, WorkingMemoryError> {
        let values = storage.into_iter().collect::<Vec<_>>();
        StoragePublicationLayout::new(values.len())?
            .fund(self)?
            .register_storage_individually(
                values
                    .into_iter()
                    .map(|(key, bytes)| (key, StorageAllocation::new(bytes, placement()))),
            )
    }
    fn register_host_storage<K: Clone + Ord + Send + 'static>(
        &self,
        storage: impl IntoIterator<Item = (K, u64)>,
    ) -> Result<WorkingMemoryStorage<K>, WorkingMemoryError> {
        let values = storage.into_iter().collect::<Vec<_>>();
        StoragePublicationLayout::new(values.len())?
            .fund(self)?
            .register_host_storage(values)
    }
    fn register_storage<K: Clone + Ord + Send + 'static>(
        &self,
        storage: impl IntoIterator<Item = (K, StorageAllocation)>,
    ) -> Result<WorkingMemoryStorage<K>, WorkingMemoryError> {
        let values = storage.into_iter().collect::<Vec<_>>();
        StoragePublicationLayout::new(values.len())?
            .fund(self)?
            .register_storage(values)
    }
    fn pin_registered_storage<K: Clone + Ord + Send + 'static>(
        &self,
        storage: impl IntoIterator<Item = (K, u64)>,
    ) -> Result<WorkingMemoryStorage<K>, WorkingMemoryError> {
        let values = storage.into_iter().collect::<Vec<_>>();
        StoragePublicationLayout::new(values.len())?
            .fund(self)?
            .pin_registered_storage(values)
    }
}

/// Includes the actual retained report and account controls, quoted through the
/// same admission constructor on an unlimited isolated ledger.
pub fn reservation_bytes(admission: &Admission) -> u64 {
    let mut admission = admission.clone();
    admission.memory_limits = Default::default();
    let pool = unlimited_ledger(0);
    pool.reserve(&InferenceExecutionIdentity::default(), &admission)
        .unwrap()
        .requirements()
        .get(pool.topology().host_domain())
        .unwrap()
        .total()
        .unwrap()
}
