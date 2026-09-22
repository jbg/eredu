//! Explicit host-only topology used by payload-focused neutral unit fixtures.
use super::*;
pub(crate) mod separate;

pub(crate) fn host_topology_ref() -> &'static Arc<MemoryTopology> {
    static TOPOLOGY: std::sync::OnceLock<Arc<MemoryTopology>> = std::sync::OnceLock::new();
    TOPOLOGY.get_or_init(|| {
        Arc::new(
            MemoryTopology::new(vec![eredu_core::MemoryDomainDescription {
                name: "host".into(),
                locations: vec![eredu_core::MemoryLocation::Host],
            }])
            .unwrap(),
        )
    })
}
pub(crate) fn host_topology() -> Arc<MemoryTopology> {
    Arc::clone(host_topology_ref())
}

pub(crate) fn host_placement() -> &'static eredu_core::MemoryPlacement {
    static PLACEMENT: std::sync::OnceLock<eredu_core::MemoryPlacement> = std::sync::OnceLock::new();
    PLACEMENT.get_or_init(|| {
        let topology = host_topology();
        eredu_core::MemoryPlacement::fixed(&topology, topology.host_domain()).unwrap()
    })
}
fn configuration(
    existing: u64,
) -> Result<
    (
        Arc<MemoryTopology>,
        MemoryLimits,
        DomainMemoryRequirements,
        u64,
    ),
    WorkingMemoryError,
> {
    let topology = host_topology();
    let limits = MemoryLimits::unlimited(&topology);
    let mut baseline = DomainMemoryRequirements::zero(&topology);
    baseline.add_allocation(
        existing,
        &eredu_core::MemoryPlacement::fixed(&topology, topology.host_domain())?,
    )?;
    let fixed = MemoryLedger::fixed_owner_bytes(&topology, &limits, &baseline)?;
    Ok((topology, limits, baseline, fixed))
}

pub(crate) fn host_ledger(
    capacity: u64,
    existing: u64,
) -> Result<MemoryLedger, WorkingMemoryError> {
    let (topology, _, baseline, fixed) = configuration(existing)?;
    let capacity = physical_limit(capacity, fixed)?;
    let limits = MemoryLimits::resolve(&topology, [(topology.host_domain(), capacity)])?;
    MemoryLedger::new(topology, limits, baseline)
}

/// Proves ordinary entry is no longer excluded, even when an exact payload
/// admission leaves no host capacity for the new owner's measured controls.
pub(crate) fn assert_unquoted_idle(pool: &MemoryLedger) {
    let before = pool.snapshot().unwrap();
    match pool.acquire_unquoted() {
        Ok(owner) => drop(owner),
        Err(WorkingMemoryError::Domain(MemoryDomainError::BudgetExceeded {
            domain,
            requested_bytes,
            ..
        })) => {
            assert_eq!(domain, pool.topology().host_domain());
            assert_eq!(
                requested_bytes,
                MemoryLedger::unquoted_owner_control_bytes().unwrap()
            );
            assert_eq!(pool.snapshot().unwrap(), before);
        }
        other => panic!("ordinary owner is still excluded: {other:?}"),
    }
    assert_eq!(pool.unquoted_owner_count().unwrap(), before.unquoted_owners);
}

fn physical_limit(payload: u64, fixed: u64) -> Result<MemoryLimit, WorkingMemoryError> {
    Ok(MemoryLimit::Finite(if payload == u64::MAX {
        u64::MAX
    } else {
        payload
            .checked_add(fixed)
            .ok_or(WorkingMemoryError::Overflow)?
    }))
}

pub(crate) fn host_limits(capacity: u64) -> eredu_core::MemoryLimitDeclarations {
    let (_, _, _, fixed) = configuration(0).expect("qualified fixture host topology");
    eredu_core::MemoryLimitDeclarations::new([(
        "host".into(),
        physical_limit(capacity, fixed).expect("fixture limit overflow"),
    )])
}

pub(crate) fn resolved_host_limits(pool: &MemoryLedger, capacity: u64) -> MemoryLimits {
    host_limits(capacity)
        .resolve(pool.topology())
        .expect("qualified fixture limit")
}

pub(crate) fn host_requirements(pool: &MemoryLedger, bytes: u64) -> DomainMemoryRequirements {
    let mut requirements = DomainMemoryRequirements::zero(pool.topology());
    requirements
        .add_allocation(bytes, &pool.host_placement_handle())
        .unwrap();
    requirements
}

impl WorkingMemoryFundingScope {
    pub(crate) fn adopt_host_storage_individually<K: Clone + Ord + Send + 'static>(
        &self,
        storage: impl IntoIterator<Item = (K, u64)>,
    ) -> Result<crate::working_memory::StorageRegistrations<K>, WorkingMemoryError> {
        let placement = self.pool().host_placement_handle();
        self.adopt_storage_individually(
            storage
                .into_iter()
                .map(|(key, bytes)| (key, StorageAllocation::new(bytes, Arc::clone(&placement)))),
        )
    }
}

pub(crate) fn host_workspace(
    pool: &MemoryLedger,
    geometry: eredu_core::InferenceGeometry,
    bytes: u64,
) -> eredu_core::DomainExecutionWorkspaceEstimate {
    eredu_core::DomainExecutionWorkspaceEstimate {
        geometry,
        activations: host_requirements(pool, bytes),
        attention: host_requirements(pool, 0),
        vocabulary: host_requirements(pool, 0),
        state_update: host_requirements(pool, 0),
        materialization: host_requirements(pool, 0),
        retained: host_requirements(pool, 0),
    }
}

pub(crate) fn empty_state(
    pool: &MemoryLedger,
    geometry: eredu_core::InferenceGeometry,
) -> eredu_core::DomainRuntimeStateEstimate {
    eredu_core::DomainRuntimeStateEstimate {
        geometry,
        decoder_state: host_requirements(pool, 0),
        media_embeddings: host_requirements(pool, 0),
        media_workspace: host_requirements(pool, 0),
    }
}

impl MemoryLedger {
    fn fixture_fixed_bytes(&self) -> Result<u64, WorkingMemoryError> {
        Self::fixed_owner_bytes(self.topology(), self.configured_limits(), &self.0.baseline)
    }
    pub(crate) fn payload_used_bytes(&self) -> Result<u64, WorkingMemoryError> {
        let host = self.topology().host_domain();
        let snapshot = self.snapshot()?;
        let current = snapshot
            .domains
            .iter()
            .find(|value| value.domain == host)
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        current
            .current_charge_bytes
            .checked_sub(self.fixture_fixed_bytes()?)
            .and_then(|bytes| bytes.checked_sub(current.registry_metadata_bytes))
            .and_then(|bytes| bytes.checked_sub(current.reservation_control_bytes))
            .ok_or(WorkingMemoryError::Overflow)
    }
    pub(crate) fn payload_peak_bytes(&self) -> Result<u64, WorkingMemoryError> {
        let host = self.topology().host_domain();
        let snapshot = self.snapshot()?;
        snapshot
            .domains
            .iter()
            .find(|value| value.domain == host)
            .ok_or(WorkingMemoryError::IdentityMismatch)?
            .historical_peak_bytes
            .checked_sub(self.fixture_fixed_bytes()?)
            .ok_or(WorkingMemoryError::Overflow)
    }
    pub(crate) fn payload_effective_limit(&self) -> Result<MemoryLimit, WorkingMemoryError> {
        let host = self.topology().host_domain();
        let snapshot = self.snapshot()?;
        match snapshot
            .domains
            .iter()
            .find(|value| value.domain == host)
            .ok_or(WorkingMemoryError::IdentityMismatch)?
            .effective_limit
        {
            MemoryLimit::Unlimited => Ok(MemoryLimit::Unlimited),
            MemoryLimit::Finite(value) => Ok(MemoryLimit::Finite(
                value
                    .checked_sub(self.fixture_fixed_bytes()?)
                    .ok_or(WorkingMemoryError::Overflow)?,
            )),
        }
    }
    pub(crate) fn payload_effective_capacity(&self) -> Result<u64, WorkingMemoryError> {
        match self.payload_effective_limit()? {
            MemoryLimit::Finite(bytes) => Ok(bytes),
            MemoryLimit::Unlimited => Err(WorkingMemoryError::IdentityMismatch),
        }
    }
}

/// Complete artificial host workspace; no native mechanism is implied.
pub(crate) fn host_admission(pool: &MemoryLedger, bytes: u64) -> eredu_core::Admission {
    use eredu_core::*;
    let geometry = InferenceGeometry {
        batch_size: 1,
        cached_positions: 0,
        input_positions: 1,
        max_output_tokens: 1,
        prefill_chunk_positions: 1,
        output: OutputDemand::LastPosition,
    };
    let mut state = estimate_runtime_state(
        &StateMemoryLayout::new(
            LayerSchedule::empty(),
            vec![],
            1,
            1,
            EstimationCompleteness::Complete,
        )
        .unwrap(),
        InputTokenCount::text(1),
        1,
        1,
        std::num::NonZeroU8::new(4).unwrap(),
    )
    .unwrap();
    state.physical_domains = Some(empty_state(pool, geometry));
    let bound = |bytes| WorkspaceBound::bounded(bytes, "explicit host fixture allocation");
    state = state
        .with_execution_workspace(ExecutionWorkspaceEstimate {
            geometry,
            activations: bound(bytes),
            attention: bound(0),
            vocabulary: bound(0),
            state_update: bound(0),
            materialization: bound(0),
            retained: bound(0),
            physical_domains: Some(host_workspace(pool, geometry, bytes)),
        })
        .unwrap();
    Admission {
        requested_positions: 2,
        state,
        incremental_required_bytes: Some(bytes),
        memory_limits: MemoryLimitDeclarations::default(),
        additional_headroom: MemoryHeadroomDeclarations::default(),
    }
}

impl MemoryLedger {
    pub(crate) fn register_host_storage_individually<K: Clone + Ord + Send + 'static>(
        &self,
        storage: impl IntoIterator<Item = (K, u64)>,
    ) -> Result<crate::working_memory::StorageRegistrations<K>, WorkingMemoryError> {
        let placement = self.host_placement_handle();
        self.register_storage_individually(
            storage
                .into_iter()
                .map(|(key, bytes)| (key, StorageAllocation::new(bytes, Arc::clone(&placement)))),
        )
    }
}

/// Attributes this synthetic fixture's explicitly declared buffers to host.
/// Missing component bounds remain missing and cannot authorize execution.
pub(crate) fn attribute_host_admission(
    pool: &MemoryLedger,
    mut admission: eredu_core::Admission,
) -> eredu_core::Admission {
    let state = &mut admission.state;
    if let Some(workspace) = state.execution_workspace.as_mut() {
        let bounds = [
            &workspace.activations,
            &workspace.attention,
            &workspace.vocabulary,
            &workspace.state_update,
            &workspace.materialization,
            &workspace.retained,
        ];
        if let Some(bytes) = bounds
            .map(|bound| bound.bytes())
            .into_iter()
            .collect::<Option<Vec<_>>>()
        {
            let mut physical = host_workspace(pool, workspace.geometry, 0);
            physical.activations = host_requirements(pool, bytes[0]);
            physical.attention = host_requirements(pool, bytes[1]);
            physical.vocabulary = host_requirements(pool, bytes[2]);
            physical.state_update = host_requirements(pool, bytes[3]);
            physical.materialization = host_requirements(pool, bytes[4]);
            physical.retained = host_requirements(pool, bytes[5]);
            workspace.physical_domains = Some(physical);
            let mut physical = empty_state(pool, workspace.geometry);
            let decoder = state
                .requested_state_bytes
                .checked_sub(state.multimodal_embedding_bytes)
                .and_then(|bytes| bytes.checked_sub(state.media_execution_workspace_bytes))
                .expect("fixture state components");
            physical.decoder_state = host_requirements(pool, decoder);
            physical.media_embeddings = host_requirements(pool, state.multimodal_embedding_bytes);
            physical.media_workspace =
                host_requirements(pool, state.media_execution_workspace_bytes);
            state.physical_domains = Some(physical);
        }
    }
    admission
}

/// Exact standalone report/account control quote; allocation requirements are
/// supplied separately so finite overlap tests include real host bookkeeping.
pub(crate) fn reservation_metadata_bytes(
    pool: &MemoryLedger,
    admission: &eredu_core::Admission,
    requirements: &DomainMemoryRequirements,
) -> u64 {
    let limits = admission.memory_limits.resolve(pool.topology()).unwrap();
    let retained = admission.clone();
    u64::try_from(
        super::reservation_metadata::constructor_bytes(pool.topology(), requirements, &limits)
            .unwrap(),
    )
    .unwrap()
    .checked_add(super::reservation_metadata::admission_backing_bytes(&retained).unwrap())
    .unwrap()
}
pub(crate) fn reservation_bytes(pool: &MemoryLedger, admission: &eredu_core::Admission) -> u64 {
    let requirements = admission
        .state
        .physical_domains
        .as_ref()
        .unwrap()
        .requirements()
        .unwrap()
        .checked_add(
            &admission
                .state
                .execution_workspace
                .as_ref()
                .unwrap()
                .physical_domains
                .as_ref()
                .unwrap()
                .requirements()
                .unwrap(),
        )
        .unwrap()
        .checked_add(
            &admission
                .additional_headroom
                .resolve(pool.topology())
                .unwrap(),
        )
        .unwrap();
    let bytes = requirements
        .iter()
        .try_fold(
            reservation_metadata_bytes(pool, admission, &requirements),
            |sum, (_, charge)| sum.checked_add(charge.total().unwrap()),
        )
        .unwrap();
    bytes
}
