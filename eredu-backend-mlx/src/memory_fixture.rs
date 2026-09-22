//! Native fixture limits resolved against the backend's actual immutable topology.
#![allow(dead_code)]
use eredu_core::*;
use eredu_runtime::working_memory::*;
use std::sync::Arc;

pub(crate) fn topology() -> Arc<MemoryTopology> {
    crate::memory_topology().unwrap()
}
pub(crate) fn host_requirements(bytes: u64) -> DomainMemoryRequirements {
    let topology = topology();
    let mut requirements = DomainMemoryRequirements::zero(&topology);
    requirements
        .add_allocation(
            bytes,
            &MemoryPlacement::fixed(&topology, topology.host_domain()).unwrap(),
        )
        .unwrap();
    requirements
}
pub(crate) fn native_requirements(bytes: u64) -> DomainMemoryRequirements {
    let domain = crate::backend::managed_memory::try_ledger().unwrap();
    let mut requirements = DomainMemoryRequirements::zero(domain.topology());
    let placement = crate::backend::managed_memory::cold_default_placement_handle().unwrap();
    requirements.add_allocation(bytes, &placement).unwrap();
    requirements
}
fn fixed_bytes() -> u64 {
    let topology = topology();
    MemoryLedger::fixed_owner_bytes(
        &topology,
        &MemoryLimits::unlimited(&topology),
        &host_requirements(0),
    )
    .unwrap()
}
pub(crate) fn limits(capacity: u64) -> MemoryLimitDeclarations {
    let topology = topology();
    let physical = if capacity == u64::MAX {
        capacity
    } else {
        capacity.checked_add(fixed_bytes()).unwrap()
    };
    let name = topology
        .description(topology.host_domain())
        .unwrap()
        .name
        .clone();
    MemoryLimitDeclarations::new([(name, MemoryLimit::Finite(physical))])
}
pub(crate) fn resolved_limits(capacity: u64) -> MemoryLimits {
    limits(capacity).resolve(&topology()).unwrap()
}
pub(crate) fn ledger(
    capacity: u64,
    existing_host: u64,
) -> Result<MemoryLedger, WorkingMemoryError> {
    crate::backend::managed_memory::try_ledger().map_err(|_| WorkingMemoryError::UnknownBound)?;
    MemoryLedger::new(
        topology(),
        resolved_limits(capacity),
        host_requirements(existing_host),
    )
}
pub(crate) fn workspace(value: ExecutionWorkspaceEstimate) -> ExecutionWorkspaceEstimate {
    workspace_in(value, native_requirements)
}
fn workspace_in(
    mut value: ExecutionWorkspaceEstimate,
    requirements: fn(u64) -> DomainMemoryRequirements,
) -> ExecutionWorkspaceEstimate {
    let fields = [
        &value.activations,
        &value.attention,
        &value.vocabulary,
        &value.state_update,
        &value.materialization,
        &value.retained,
    ];
    if let Some(bytes) = fields
        .map(|value| value.bytes())
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
pub(crate) fn state(value: RuntimeStateEstimate) -> RuntimeStateEstimate {
    state_in(value, native_requirements)
}
fn state_in(
    mut value: RuntimeStateEstimate,
    requirements: fn(u64) -> DomainMemoryRequirements,
) -> RuntimeStateEstimate {
    if let Some(workspace_value) = value.execution_workspace.take() {
        let geometry = workspace_value.geometry;
        value.execution_workspace = Some(workspace_in(workspace_value, requirements));
        let decoder = value
            .requested_state_bytes
            .checked_sub(value.multimodal_embedding_bytes)
            .and_then(|n| n.checked_sub(value.media_execution_workspace_bytes))
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
pub(crate) fn admission(mut value: Admission) -> Admission {
    value.state = state(value.state);
    value
}
pub(crate) fn host_admission(mut value: Admission) -> Admission {
    value.state = state_in(value.state, host_requirements);
    value
}
/// Constructor controls for these one-domain, zero-headroom request fixtures.
/// The account retains this floor while any funded backing or slot owner lives.
pub(crate) fn request_control_bytes(request: &InferenceRequest) -> u64 {
    let reservation = request.memory_reservation();
    assert_eq!(reservation.requirements().iter().len(), 1);
    host_total(reservation.requirements())
        .checked_sub(reservation.admission().incremental_required_bytes.unwrap())
        .unwrap()
}

pub(crate) trait LedgerFixture {
    /// Full committed host charge, including fixed and live bookkeeping.
    fn fixture_host_current(&self) -> Result<u64, WorkingMemoryError>;
    /// Live payload and complete funded accounts, excluding fixed owners and storage-directory metadata.
    fn fixture_funded_charge(&self) -> Result<u64, WorkingMemoryError>;
    fn fixture_host_charge(&self) -> Result<u64, WorkingMemoryError>;
    fn fixture_host_peak(&self) -> Result<u64, WorkingMemoryError>;
    fn fixture_host_limit(&self) -> Result<u64, WorkingMemoryError>;
}
impl LedgerFixture for MemoryLedger {
    fn fixture_funded_charge(&self) -> Result<u64, WorkingMemoryError> {
        let snapshot = self.snapshot()?;
        let host = snapshot
            .domains
            .iter()
            .find(|d| d.domain == self.topology().host_domain())
            .unwrap();
        host.current_charge_bytes
            .checked_sub(fixed_bytes())
            .and_then(|n| n.checked_sub(host.registry_metadata_bytes))
            .ok_or(WorkingMemoryError::Overflow)
    }
    fn fixture_host_current(&self) -> Result<u64, WorkingMemoryError> {
        let snapshot = self.snapshot()?;
        Ok(snapshot
            .domains
            .iter()
            .find(|domain| domain.domain == self.topology().host_domain())
            .ok_or(WorkingMemoryError::IdentityMismatch)?
            .current_charge_bytes)
    }
    fn fixture_host_charge(&self) -> Result<u64, WorkingMemoryError> {
        let snapshot = self.snapshot()?;
        let host = snapshot
            .domains
            .iter()
            .find(|value| value.domain == self.topology().host_domain())
            .unwrap();
        host.current_charge_bytes
            .checked_sub(fixed_bytes())
            .and_then(|bytes| bytes.checked_sub(host.registry_metadata_bytes))
            .and_then(|bytes| bytes.checked_sub(host.reservation_control_bytes))
            .ok_or(WorkingMemoryError::Overflow)
    }
    fn fixture_host_peak(&self) -> Result<u64, WorkingMemoryError> {
        let snapshot = self.snapshot()?;
        let host = snapshot
            .domains
            .iter()
            .find(|value| value.domain == self.topology().host_domain())
            .unwrap();
        host.historical_peak_bytes
            .checked_sub(fixed_bytes())
            .ok_or(WorkingMemoryError::Overflow)
    }
    fn fixture_host_limit(&self) -> Result<u64, WorkingMemoryError> {
        let snapshot = self.snapshot()?;
        let host = snapshot
            .domains
            .iter()
            .find(|value| value.domain == self.topology().host_domain())
            .unwrap();
        match host.effective_limit {
            MemoryLimit::Finite(bytes) => bytes
                .checked_sub(fixed_bytes())
                .ok_or(WorkingMemoryError::Overflow),
            MemoryLimit::Unlimited => Err(WorkingMemoryError::IdentityMismatch),
        }
    }
}

pub(crate) fn headroom(bytes: u64) -> MemoryHeadroomDeclarations {
    let topology = topology();
    MemoryHeadroomDeclarations::new([(
        topology
            .description(topology.host_domain())
            .unwrap()
            .name
            .clone(),
        bytes,
    )])
}

/// An ordinary unlimited reservation for fixtures which only validate binding
/// and completion ownership and execute no numerical operation.
pub(crate) fn empty_admitted_request(
    execution: &InferenceExecutionIdentity,
    geometry: InferenceGeometry,
) -> Result<InferenceRequest, WorkingMemoryError> {
    let topology = topology();
    let pool = MemoryLedger::new(
        topology.clone(),
        MemoryLimits::unlimited(&topology),
        DomainMemoryRequirements::zero(&topology),
    )?;
    let zero = || DomainMemoryRequirements::zero(&topology);
    let empty = || WorkspaceBound::bounded(0, "binding-only fixture performs no numerical work");
    let requested_positions = geometry
        .cached_positions
        .checked_add(geometry.input_positions)
        .and_then(|n| n.checked_add(geometry.max_output_tokens))
        .ok_or(WorkingMemoryError::Overflow)?;
    let admission = Admission {
        requested_positions,
        incremental_required_bytes: Some(0),
        memory_limits: Default::default(),
        additional_headroom: Default::default(),
        state: RuntimeStateEstimate {
            fixed_state_bytes: 0,
            bytes_per_position_per_batch: 0,
            context_state_bytes: 0,
            selected_state_backing: None,
            multimodal_embedding_bytes: 0,
            media_execution_workspace_bytes: 0,
            requested_state_bytes: 0,
            persistent_state_completeness: EstimationCompleteness::Complete,
            completeness: EstimationCompleteness::Complete,
            assumptions: StateMemoryAssumptions {
                floating_state_dtype_bytes: std::num::NonZeroU8::new(4).unwrap(),
                batch_size: geometry.batch_size,
                requested_positions,
                sliding_window_bounds: vec![],
                allocation_granularity: 1,
            },
            physical_domains: Some(DomainRuntimeStateEstimate {
                geometry,
                decoder_state: zero(),
                media_embeddings: zero(),
                media_workspace: zero(),
            }),
            execution_workspace: Some(ExecutionWorkspaceEstimate {
                geometry,
                activations: empty(),
                attention: empty(),
                vocabulary: empty(),
                state_update: empty(),
                materialization: empty(),
                retained: empty(),
                physical_domains: Some(DomainExecutionWorkspaceEstimate {
                    geometry,
                    activations: zero(),
                    attention: zero(),
                    vocabulary: zero(),
                    state_update: zero(),
                    materialization: zero(),
                    retained: zero(),
                }),
            }),
        },
    };
    Ok(pool.reserve(execution, &admission)?.into())
}

/// Test adapters expose preparation as a separate funded operation internally.
/// Transaction rejection assertions prepare explicitly before their snapshot.
pub(crate) trait StorageFixture {
    fn register_storage<K: Clone + Ord + Send + 'static>(
        &self,
        storage: impl IntoIterator<Item = (K, StorageAllocation)>,
    ) -> Result<WorkingMemoryStorage<K>, WorkingMemoryError>;
    fn register_host_storage<K: Clone + Ord + Send + 'static>(
        &self,
        storage: impl IntoIterator<Item = (K, u64)>,
    ) -> Result<WorkingMemoryStorage<K>, WorkingMemoryError>;
    fn register_storage_individually<K: Clone + Ord + Send + 'static>(
        &self,
        storage: impl IntoIterator<Item = (K, StorageAllocation)>,
    ) -> Result<StorageRegistrations<K>, WorkingMemoryError>;
    fn adopt_storage_individually<K: Clone + Ord + Send + 'static>(
        &self,
        scope: &WorkingMemoryFundingScope,
        storage: impl IntoIterator<Item = (K, StorageAllocation)>,
    ) -> Result<StorageRegistrations<K>, WorkingMemoryError>;
    fn pin_registered_storage<K: Ord + Send + 'static>(
        &self,
        storage: impl IntoIterator<Item = (K, u64)>,
    ) -> Result<WorkingMemoryStorage<K>, WorkingMemoryError>;
}
impl StorageFixture for MemoryLedger {
    fn register_storage<K: Clone + Ord + Send + 'static>(
        &self,
        storage: impl IntoIterator<Item = (K, StorageAllocation)>,
    ) -> Result<WorkingMemoryStorage<K>, WorkingMemoryError> {
        let entries: Vec<_> = storage.into_iter().collect();
        StoragePublicationLayout::new(entries.len())?
            .fund(self)?
            .register_storage(entries)
    }
    fn register_host_storage<K: Clone + Ord + Send + 'static>(
        &self,
        storage: impl IntoIterator<Item = (K, u64)>,
    ) -> Result<WorkingMemoryStorage<K>, WorkingMemoryError> {
        let entries: Vec<_> = storage.into_iter().collect();
        StoragePublicationLayout::new(entries.len())?
            .fund(self)?
            .register_host_storage(entries)
    }
    fn register_storage_individually<K: Clone + Ord + Send + 'static>(
        &self,
        storage: impl IntoIterator<Item = (K, StorageAllocation)>,
    ) -> Result<StorageRegistrations<K>, WorkingMemoryError> {
        let entries: Vec<_> = storage.into_iter().collect();
        StoragePublicationLayout::new(entries.len())?
            .fund(self)?
            .register_storage_individually(entries)
    }
    fn adopt_storage_individually<K: Clone + Ord + Send + 'static>(
        &self,
        scope: &WorkingMemoryFundingScope,
        storage: impl IntoIterator<Item = (K, StorageAllocation)>,
    ) -> Result<StorageRegistrations<K>, WorkingMemoryError> {
        let entries: Vec<_> = storage.into_iter().collect();
        StoragePublicationLayout::new(entries.len())?
            .fund_from(scope)?
            .adopt_storage_individually(scope, entries)
    }
    fn pin_registered_storage<K: Ord + Send + 'static>(
        &self,
        storage: impl IntoIterator<Item = (K, u64)>,
    ) -> Result<WorkingMemoryStorage<K>, WorkingMemoryError> {
        let entries: Vec<_> = storage.into_iter().collect();
        StoragePublicationLayout::new(entries.len())?
            .fund(self)?
            .pin_registered_storage(entries)
    }
}
pub(crate) trait FundingFixture {
    fn adopt_storage_individually<K: Clone + Ord + Send + 'static>(
        &self,
        storage: impl IntoIterator<Item = (K, StorageAllocation)>,
    ) -> Result<StorageRegistrations<K>, WorkingMemoryError>;
    fn adopt_host_storage_individually<K: Clone + Ord + Send + 'static>(
        &self,
        storage: impl IntoIterator<Item = (K, u64)>,
    ) -> Result<StorageRegistrations<K>, WorkingMemoryError>;
}
impl FundingFixture for WorkingMemoryFundingScope {
    fn adopt_storage_individually<K: Clone + Ord + Send + 'static>(
        &self,
        storage: impl IntoIterator<Item = (K, StorageAllocation)>,
    ) -> Result<StorageRegistrations<K>, WorkingMemoryError> {
        let entries: Vec<_> = storage.into_iter().collect();
        StoragePublicationLayout::new(entries.len())?
            .fund_from(self)?
            .adopt_storage_individually(self, entries)
    }
    fn adopt_host_storage_individually<K: Clone + Ord + Send + 'static>(
        &self,
        storage: impl IntoIterator<Item = (K, u64)>,
    ) -> Result<StorageRegistrations<K>, WorkingMemoryError> {
        self.adopt_storage_individually(storage.into_iter().map(|(key, bytes)| {
            (
                key,
                StorageAllocation::new(bytes, self.pool().host_placement_handle()),
            )
        }))
    }
}

/// A literal physical host limit; unlike payload fixtures, this adds no metadata allowance.
pub(crate) fn physical_host_limits(pool: &MemoryLedger, bytes: u64) -> MemoryLimits {
    let topology = pool.topology();
    MemoryLimitDeclarations::new([(
        topology
            .description(topology.host_domain())
            .unwrap()
            .name
            .clone(),
        MemoryLimit::Finite(bytes),
    )])
    .resolve(topology)
    .unwrap()
}

/// Exact generic native publication metadata for a bounded canonical row population.
pub(crate) fn publication_control_bytes(rows: usize) -> u64 {
    crate::backend::runtime::residency::storage::generic_storage_publication_layout(rows)
        .unwrap()
        .requested_bytes()
        .checked_add(MemoryLedger::storage_metadata_control_bytes().unwrap())
        .unwrap()
}
pub(crate) fn publication_copy_limits(
    pool: &MemoryLedger,
    roots: usize,
    physical_limit: u64,
) -> WorkspaceCopyLimits {
    let mut limits = WorkspaceCopyLimits::new(
        physical_host_limits(pool, physical_limit)
            .named(pool.topology())
            .unwrap(),
    );
    limits.additional_host_metadata_bytes =
        publication_control_bytes(roots.checked_mul(2).unwrap());
    limits
}
pub(crate) fn host_total(requirements: &DomainMemoryRequirements) -> u64 {
    requirements
        .get(topology().host_domain())
        .unwrap()
        .total()
        .unwrap()
}

/// Actual native ordinary authority constructor, independent of payload/publication.
pub(crate) fn native_owner_control_bytes() -> u64 {
    crate::backend::managed_memory::NativeMemoryOwner::preparation_bytes().unwrap()
}

/// Fixture metadata uses the same physical host account as production preparation.
pub(crate) fn parameter_context() -> (
    eredu_nn::workspace::WorkspaceContext,
    eredu_core::HostMetadataFunding,
) {
    let pool = crate::backend::managed_memory::try_ledger().unwrap();
    let funding = pool
        .prepare_workspace_metadata(
            &InferenceExecutionIdentity::default(),
            pool.configured_limits().clone(),
        )
        .unwrap();
    let context = eredu_nn::workspace::WorkspaceContext::new_with_metadata_funding(
        crate::backend::nn::workspace::MlxMetalWorkspaceMechanisms::current_host().unwrap(),
        funding.clone(),
    )
    .unwrap();
    (context, funding)
}
/// Uses the real prepared exchanges for fixture owners. Values are existing test
/// sources; this helper grants no execution, materialization or completion proof.
pub(crate) fn publish_parameters(
    rows: impl IntoIterator<Item = (String, crate::MlxTensor)>,
    active: bool,
    context: &eredu_nn::workspace::WorkspaceContext,
    funding: eredu_core::HostMetadataFunding,
    mut visit: impl FnMut(
        &mut dyn eredu_runtime::parameter_operations::ParameterPublication<crate::MlxTensor>,
    ) -> Result<bool, crate::backend::error::Error>,
) {
    use eredu_runtime::parameter_operations::{
        ParameterReplacementValues, PreparedParameterPublication,
    };
    let rows = rows.into_iter();
    let mut owned = context
        .metadata_vec(rows.size_hint().1.expect("finite fixture rows"))
        .unwrap();
    for (name, value) in rows {
        context.charge_metadata(name.capacity()).unwrap();
        owned.push((name, value));
    }
    let values =
        ParameterReplacementValues::from_prepared_rows(owned, funding.clone(), context).unwrap();
    let pool = crate::backend::managed_memory::try_ledger().unwrap();
    let mut prepared = PreparedParameterPublication::prepare(
        values,
        active,
        &mut visit,
        |value| {
            let clone = crate::backend::managed_memory::PreparedPublicationClone::prepare(&pool)
                .unwrap()
                .fill(value.as_array())
                .unwrap();
            Ok(crate::MlxTensor::from_array(clone))
        },
        context,
        funding,
    )
    .unwrap();
    prepared
        .validate(&mut visit, |a, b| {
            context
                .charge_metadata(safemlx::Array::descriptor_comparison_control_bytes().unwrap())?;
            Ok(a.as_array()
                .try_descriptor()
                .unwrap()
                .same_descriptor(b.as_array())
                .unwrap())
        })
        .unwrap();
    prepared.exchange(|v| {
        assert!(visit(v).unwrap());
    });
}

pub(crate) fn publish_model_parameters(
    model: &mut dyn crate::composition::mlx::replicated_text::ErasedReplicatedTextExecutable,
    rows: impl IntoIterator<Item = (String, crate::MlxTensor)>,
    active: bool,
) {
    let (context, funding) = parameter_context();
    let mut rows = Some(rows);
    model
        .with_parameter_publication(&context, &mut |visit| {
            publish_parameters(
                rows.take().expect("single fixture publication"),
                active,
                &context,
                funding.clone(),
                visit,
            );
            Ok(())
        })
        .unwrap();
    model.finalize_parameter_publication();
}
