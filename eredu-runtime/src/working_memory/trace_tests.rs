use super::*;
use eredu_core::{
    AdmissionRequest, AdmissionResult, CacheStateStrategy, EstimationCompleteness,
    InferenceGeometry, InputModalities, InputTokenCount, LayerSchedule, ModelCapabilities,
    Observed, OutputDemand, StateMemoryLayout,
};
use eredu_nn::{Error, Tensor, workspace::*};

#[derive(Debug)]
struct Facts {
    host: Option<u64>,
}
impl WorkspaceMechanisms for Facts {
    fn memory_topology(&self) -> Option<&eredu_core::MemoryTopology> {
        Some(crate::working_memory::memory_fixture::host_topology_ref())
    }
    fn output_placement(
        &self,
        _: WorkspaceOperationView<'_>,
        _: usize,
    ) -> Option<&eredu_core::MemoryPlacement> {
        Some(crate::working_memory::memory_fixture::host_placement())
    }
    fn scratch_placement(
        &self,
        _: WorkspaceOperationView<'_>,
    ) -> Option<&eredu_core::MemoryPlacement> {
        Some(crate::working_memory::memory_fixture::host_placement())
    }

    fn operation_bound(
        &self,
        op: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, Error> {
        Ok(Some(WorkspaceOperationBound {
            outputs: op
                .outputs
                .iter()
                .map(|l| l.bytes().map(WorkspaceOutputStorage::Allocate))
                .collect::<Result<_, _>>()?,
            scratch_bytes: 7,
            assumptions: "test mechanism: exact tensor bytes plus seven temporary bytes".into(),
        }))
    }
    fn host_workspace_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceHostBound>, Error> {
        Ok(self.host.map(|bytes| WorkspaceHostBound {
            bytes,
            assumptions: "test mechanism: disjoint per-operation host staging capacity".into(),
        }))
    }
}
fn geometry() -> InferenceGeometry {
    InferenceGeometry {
        batch_size: 1,
        cached_positions: 0,
        input_positions: 2,
        max_output_tokens: 1,
        prefill_chunk_positions: 2,
        output: OutputDemand::LastPosition,
    }
}
fn request(geometry: InferenceGeometry) -> AdmissionRequest {
    AdmissionRequest {
        input: InputTokenCount::text(geometry.input_positions),
        max_output_tokens: geometry.max_output_tokens,
        batch_size: geometry.batch_size,
        additional_headroom: Default::default(),
        memory_limits: Default::default(),
    }
}
fn capabilities() -> ModelCapabilities {
    ModelCapabilities {
        effective_model_type: "workspace-domain fixture".into(),
        native_max_context: Observed::exact(8, "fixture"),
        effective_max_context: Observed::exact(8, "fixture"),
        state_strategy: CacheStateStrategy::FullKv,
        modalities: InputModalities::TEXT,
        estimation: EstimationCompleteness::PersistentStateOnly,
    }
}
fn quote(
    geometry: InferenceGeometry,
    host: Option<u64>,
) -> Result<RuntimeStateEstimate, CapabilityError> {
    let context = WorkspaceContext::new(Facts { host });
    // This fixture quotes an equation over an already resident activation;
    // input construction and unloaded parameter seeds are outside its span.
    let input = WorkspaceTensor::existing(
        WorkspaceLayout::new(
            &[1, geometry.prefill_chunk_positions as i32, 3],
            WorkspaceDtype::Float32,
        )
        .unwrap(),
        &context,
    )
    .unwrap();
    context.begin_state_span([]).unwrap();
    let _ = input.square(&context).unwrap();
    let report = context.report(&[]).unwrap();
    let layout = StateMemoryLayout::new(
        LayerSchedule::empty(),
        vec![],
        1,
        1,
        EstimationCompleteness::Complete,
    )?;
    let mut state = eredu_core::estimate_runtime_state(
        &layout,
        request(geometry).input,
        geometry.max_output_tokens,
        1,
        std::num::NonZeroU8::new(4).unwrap(),
    )?;
    let empty = || eredu_core::DomainMemoryRequirements::zero(context.memory_topology().unwrap());
    state.physical_domains = Some(eredu_core::DomainRuntimeStateEstimate {
        geometry,
        decoder_state: empty(),
        media_embeddings: empty(),
        media_workspace: empty(),
    });
    let zero = || {
        WorkspaceBound::bounded(
            0,
            "test execution has no state or untraced preparation, sampling, materialization or retained resources",
        )
    };
    with_equation_workspace(
        state,
        ExecutionWorkspaceEstimate {
            geometry,
            activations: zero(),
            attention: zero(),
            vocabulary: zero(),
            state_update: zero(),
            materialization: zero(),
            retained: zero(),
            physical_domains: Some(eredu_core::DomainExecutionWorkspaceEstimate {
                geometry,
                activations: empty(),
                attention: empty(),
                vocabulary: empty(),
                state_update: empty(),
                materialization: empty(),
                retained: empty(),
            }),
        },
        &report,
    )
}
#[test]
fn missing_host_facts_reject_finite_and_unlimited_admission_despite_known_tensor_peak() {
    let geometry = geometry();
    for budget in [None, Some(u64::MAX)] {
        let mut request = request(geometry);
        request.memory_limits = budget
            .map(crate::working_memory::memory_fixture::host_limits)
            .unwrap_or_default();
        request.additional_headroom =
            eredu_core::MemoryHeadroomDeclarations::new([("host".into(), 1_000_000)]);
        let state = quote(geometry, None).unwrap();
        assert_eq!(
            state.completeness,
            EstimationCompleteness::PersistentStateOnly
        );
        let WorkspaceBound::Unknown { reason } =
            &state.execution_workspace.as_ref().unwrap().activations
        else {
            panic!("host coverage remains unknown")
        };
        assert!(reason.contains("managed-host operation indices: [0]"));
        assert!(matches!(
            eredu_core::apply_admission_policy(&capabilities(), request, state).unwrap(),
            AdmissionResult::Rejected(eredu_core::AdmissionRejection::EstimationUnsupported { .. })
        ));
    }
}
#[test]
fn unknown_host_workspace_cannot_reserve_capacity_or_be_fixed_by_smaller_chunks() {
    let geometry = geometry();
    let pool = crate::working_memory::memory_fixture::host_ledger(u64::MAX, 0).unwrap();
    let mut attempts = vec![];
    let result = crate::working_memory::plan_prefill(
        &crate::working_memory::InferenceExecutionIdentity::default(),
        &pool,
        &capabilities(),
        request(geometry),
        geometry,
        |g| {
            attempts.push(g.prefill_chunk_positions);
            quote(g, None)
        },
    );
    assert!(matches!(
        result,
        Err(crate::working_memory::PrefillPlanningError::Admission(
            eredu_core::AdmissionRejection::EstimationUnsupported { .. }
        ))
    ));
    assert_eq!(attempts, [2, 1]);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}
#[test]
fn complete_host_and_tensor_bounds_share_one_enforced_capacity_and_choose_a_smaller_chunk() {
    let geometry = geometry();
    let full = quote(geometry, Some(11)).unwrap();
    assert_eq!(
        full.execution_workspace
            .as_ref()
            .unwrap()
            .peak_bytes()
            .unwrap(),
        Some(42)
    ); // 24 tensor + 7 tensor scratch + 11 host
    let probe = crate::working_memory::memory_fixture::host_ledger(u64::MAX, 0).unwrap();
    let quote_bytes = |g| {
        let AdmissionResult::Admitted(mut admission) = eredu_core::apply_admission_policy(
            &capabilities(),
            request(g),
            quote(g, Some(11)).unwrap(),
        )
        .unwrap() else {
            panic!("complete attribution")
        };
        crate::working_memory::memory_fixture::reservation_bytes(&probe, &admission)
    };
    let first_bytes = quote_bytes(geometry);
    let second_bytes = quote_bytes(InferenceGeometry {
        prefill_chunk_positions: 1,
        ..geometry
    });
    let pool =
        crate::working_memory::memory_fixture::host_ledger(first_bytes + second_bytes, 0).unwrap();
    let identity = crate::working_memory::InferenceExecutionIdentity::default();
    let (first, reservation) = crate::working_memory::plan_prefill(
        &identity,
        &pool,
        &capabilities(),
        request(geometry),
        geometry,
        |g| quote(g, Some(11)),
    )
    .unwrap();
    assert_eq!(first.incremental_required_bytes, Some(42));
    assert_eq!(
        reservation
            .requirements()
            .get(crate::working_memory::memory_fixture::host_topology_ref().host_domain())
            .ok()
            .and_then(|charge| charge.total().ok()),
        Some(first_bytes)
    );
    let (second, second_reservation) = crate::working_memory::plan_prefill(
        &identity,
        &pool,
        &capabilities(),
        request(geometry),
        geometry,
        |g| quote(g, Some(11)),
    )
    .unwrap();
    assert_eq!(
        second
            .state
            .execution_workspace
            .unwrap()
            .geometry
            .prefill_chunk_positions,
        1
    );
    assert_eq!(
        second_reservation
            .requirements()
            .get(crate::working_memory::memory_fixture::host_topology_ref().host_domain())
            .ok()
            .and_then(|charge| charge.total().ok()),
        Some(second_bytes)
    );
    assert_eq!(pool.payload_used_bytes().unwrap(), 72);
    assert!(
        crate::working_memory::plan_prefill(
            &identity,
            &pool,
            &capabilities(),
            request(geometry),
            geometry,
            |g| quote(g, Some(11))
        )
        .is_err()
    );
    drop(reservation);
    assert_eq!(pool.payload_used_bytes().unwrap(), 30);
    drop(second_reservation);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn diagnostic_only_admission_with_unknown_host_cannot_be_reserved_for_execution() {
    let geometry = geometry();
    let mut request = request(geometry);
    let AdmissionResult::Admitted(mut admission) = eredu_core::apply_admission_policy(
        &capabilities(),
        request,
        quote(geometry, Some(11)).unwrap(),
    )
    .unwrap() else {
        panic!("complete diagnostic report")
    };
    let workspace = admission.state.execution_workspace.as_mut().unwrap();
    workspace.activations = WorkspaceBound::Unknown {
        reason: "host mechanism fact unavailable".into(),
    };
    workspace.physical_domains = None;
    let pool = crate::working_memory::memory_fixture::host_ledger(u64::MAX, 0).unwrap();
    assert!(matches!(
        pool.reserve(
            &crate::working_memory::InferenceExecutionIdentity::default(),
            &admission
        ),
        Err(crate::working_memory::WorkingMemoryError::UnknownBound)
    ));
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}
