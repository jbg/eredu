//! Explicit host placement for neutral architecture metadata fixtures.
use eredu_core::*;
use eredu_runtime::working_memory::{MemoryLedger, WorkingMemoryError};
use std::sync::Arc;

pub(crate) fn topology() -> &'static Arc<MemoryTopology> {
    static TOPOLOGY: std::sync::OnceLock<Arc<MemoryTopology>> = std::sync::OnceLock::new();
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
pub(crate) fn placement() -> &'static MemoryPlacement {
    static PLACEMENT: std::sync::OnceLock<MemoryPlacement> = std::sync::OnceLock::new();
    PLACEMENT.get_or_init(|| MemoryPlacement::fixed(topology(), topology().host_domain()).unwrap())
}
pub(crate) fn requirements(bytes: u64) -> DomainMemoryRequirements {
    let mut result = DomainMemoryRequirements::zero(topology());
    result.add_allocation(bytes, placement()).unwrap();
    result
}
pub(crate) fn used(pool: &MemoryLedger) -> Result<u64, WorkingMemoryError> {
    let snapshot = pool.snapshot()?;
    let host = snapshot
        .domains
        .iter()
        .find(|d| d.domain == pool.topology().host_domain())
        .unwrap();
    Ok(host
        .current_charge_bytes
        .checked_sub(host.fixed_baseline.total()?)
        .unwrap())
}
pub(crate) fn ledger(limit: u64, baseline: u64) -> Result<MemoryLedger, WorkingMemoryError> {
    let topology = Arc::clone(topology());
    let limits = MemoryLimits::resolve(
        &topology,
        [(topology.host_domain(), MemoryLimit::Finite(limit))],
    )?;
    let mut requirements = DomainMemoryRequirements::zero(&topology);
    requirements.add_allocation(
        baseline,
        &MemoryPlacement::fixed(&topology, topology.host_domain())?,
    )?;
    let fixed = MemoryLedger::fixed_owner_bytes(&topology, &limits, &requirements)?;
    let limits = MemoryLimits::resolve(
        &topology,
        [(
            topology.host_domain(),
            MemoryLimit::Finite(if limit == u64::MAX {
                limit
            } else {
                limit.checked_add(fixed).unwrap()
            }),
        )],
    )?;
    MemoryLedger::new(topology, limits, requirements)
}
pub(crate) fn limits(pool: &MemoryLedger, capacity: u64) -> MemoryLimits {
    MemoryLimits::resolve(
        pool.topology(),
        [(
            pool.topology().host_domain(),
            MemoryLimit::Finite(if capacity == u64::MAX {
                capacity
            } else {
                capacity
                    .checked_add(
                        pool.snapshot()
                            .unwrap()
                            .domains
                            .iter()
                            .find(|d| d.domain == pool.topology().host_domain())
                            .unwrap()
                            .fixed_baseline
                            .total()
                            .unwrap(),
                    )
                    .unwrap()
            }),
        )],
    )
    .unwrap()
}

/// Numerical fixtures execute eager host allocations within a declared finite
/// envelope and obtain the ordinary reservation/completion authority.
pub(crate) fn request(
    execution: &eredu_runtime::working_memory::InferenceExecutionIdentity,
    geometry: InferenceGeometry,
) -> Result<eredu_runtime::working_memory::InferenceRequest, WorkingMemoryError> {
    let pool = ledger(1 << 30, 0)?;
    request_in(&pool, execution, geometry)
}
pub(crate) fn request_in(
    pool: &MemoryLedger,
    execution: &eredu_runtime::working_memory::InferenceExecutionIdentity,
    geometry: InferenceGeometry,
) -> Result<eredu_runtime::working_memory::InferenceRequest, WorkingMemoryError> {
    let input = InputTokenCount::text(geometry.cached_positions + geometry.input_positions);
    let mut state = estimate_runtime_state(
        &StateMemoryLayout::new(
            LayerSchedule::empty(),
            vec![],
            1,
            1,
            EstimationCompleteness::Complete,
        )
        .unwrap(),
        input,
        geometry.max_output_tokens,
        geometry.batch_size,
        std::num::NonZeroU8::new(4).unwrap(),
    )
    .unwrap();
    state.physical_domains = Some(DomainRuntimeStateEstimate {
        geometry,
        decoder_state: requirements(0),
        media_embeddings: requirements(0),
        media_workspace: requirements(0),
    });
    let bound = |n| WorkspaceBound::bounded(n, "eager host numerical fixture allowance");
    state = state
        .with_execution_workspace(ExecutionWorkspaceEstimate {
            geometry,
            activations: bound(64 << 20),
            attention: bound(0),
            vocabulary: bound(0),
            state_update: bound(0),
            materialization: bound(0),
            retained: bound(0),
            physical_domains: Some(DomainExecutionWorkspaceEstimate {
                geometry,
                activations: requirements(64 << 20),
                attention: requirements(0),
                vocabulary: requirements(0),
                state_update: requirements(0),
                materialization: requirements(0),
                retained: requirements(0),
            }),
        })
        .unwrap();
    let admission = Admission {
        requested_positions: geometry.cached_positions
            + geometry.input_positions
            + geometry.max_output_tokens,
        state,
        incremental_required_bytes: Some(64 << 20),
        memory_limits: MemoryLimitDeclarations::default(),
        additional_headroom: MemoryHeadroomDeclarations::default(),
    };
    Ok(eredu_runtime::working_memory::InferenceRequest::from(
        pool.reserve(execution, &admission)?,
    ))
}
