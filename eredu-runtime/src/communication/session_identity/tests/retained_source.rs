use super::*;

#[test]
fn retained_communication_source_keeps_checked_waves_roles_and_exact_identity() {
    let completion = CommunicationCompletionPolicy::new(
        std::time::Duration::from_secs(1),
        CompletionCancellationMode::QuarantineUntilComplete,
    )
    .unwrap();
    let sum = CommunicationOperationRequirement::tensors(
        CommunicationOperation::AllReduceSum,
        [TensorDtype::F32],
        CommunicationTensorLimits::new(1, 2, 32, None).unwrap(),
        true,
    )
    .unwrap();
    let transfer = CommunicationOperationRequirement::tensors(
        CommunicationOperation::SendReceive,
        [TensorDtype::F32],
        CommunicationTensorLimits::new(1, 2, 32, None).unwrap(),
        true,
    )
    .unwrap();
    let route =
        CommunicationRouteDescriptor::new(CommunicationRouteId::new(19), 0, 0, 1, transfer.clone())
            .unwrap()
            .with_boundary_contract(
                RoleExactBoundaryContract::new(
                    "résidual-边界",
                    [BoundaryRoleContract::new("hidden", TensorDtype::F32, vec![2, 4]).unwrap()],
                )
                .unwrap(),
            )
            .unwrap();
    let mut proposals = proposals();
    for (rank, proposal) in proposals.iter_mut().enumerate() {
        let group = CommunicationGroupDescriptor::new(
            CollectiveGroupId::new(7),
            0,
            vec![0, 1],
            Some(rank),
            CommunicationGroupRequirements::new([sum.clone()]).unwrap(),
        )
        .unwrap();
        proposal.manifest = CommunicationManifest::new(2, rank, vec![group], vec![route.clone()])
            .unwrap()
            .with_completion_policy(completion);
    }
    let agreed = establish_communication_session(
        &transport(0, &proposals),
        &proposals[0].manifest,
        proposals[0].nonce,
    )
    .unwrap();
    let capability = CommunicationCapabilities::new([sum, transfer])
        .unwrap()
        .with_completion_capabilities(
            CommunicationCompletionCapabilities::new([
                CompletionCancellationMode::QuarantineUntilComplete,
            ])
            .unwrap(),
        )
        .with_boundary_framing([BoundaryFramingProtocol::RoleExactV1])
        .unwrap();
    let prepare = || {
        prepare_communication_realization(
            &proposals[0].manifest,
            agreed.manifests(),
            &capability,
            CommunicationTopologyCapabilities::RingWithWorldWaves,
        )
        .unwrap()
    };
    let source = prepare().into_retained_source(agreed.identity()).unwrap();
    let other = prepare().into_retained_source(agreed.identity()).unwrap();
    assert!(
        !source.same_source(&other),
        "equal declarations and setup identity cannot replace the actual source"
    );
    let alias = source.clone();
    struct Custody(std::sync::Arc<std::sync::atomic::AtomicUsize>);
    impl Drop for Custody {
        fn drop(&mut self) {
            self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
    }
    let retired = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let host = eredu_core::HostPreparationAuthority::retain(Custody(retired.clone()));
    let retained = source.with_host_preparation(&host).unwrap();
    assert!(source.same_source(&retained));
    assert!(
        retained.with_host_preparation(&host).is_none(),
        "custody cannot be overwritten"
    );
    let escaped = retained.clone();
    drop((retained, host));
    assert_eq!(retired.load(std::sync::atomic::Ordering::SeqCst), 0);

    assert!(source.same_source(&alias));
    assert_eq!(source.session_identity(), agreed.identity());
    assert_eq!(source.group(0).unwrap().0.members(), [0, 1]);
    // Full-world membership is directly reachable; topology capability does
    // not force an unnecessary world-wave realization.
    assert!(!source.group(0).unwrap().1);
    assert_eq!(
        source
            .route(0)
            .unwrap()
            .0
            .boundary_contract()
            .unwrap()
            .schema(),
        "résidual-边界"
    );
    // The two endpoints are neighbors in this actual two-rank ring.
    assert!(!source.route(0).unwrap().1);
    assert!(source.group(1).is_none() && source.route(1).is_none());
    let bytes = source.host_storage_bytes().unwrap();
    assert!(
        bytes
            > std::mem::size_of::<PreparedCommunicationRealization>()
                + 8 * std::mem::size_of::<usize>()
    );
    assert_eq!(alias.host_storage_bytes(), Some(bytes));
    let wrong = CommunicationSessionIdentity {
        digest: [9; 32],
        participants: 1,
    };
    assert!(prepare().into_retained_source(wrong).is_err());
    drop((source, other, agreed, proposals));
    assert_eq!(alias.manifest().rank(), 0);
    assert_eq!(
        alias
            .route(0)
            .unwrap()
            .0
            .requirement()
            .limits()
            .unwrap()
            .max_tensor_elements(),
        32
    );
    assert_eq!(alias.host_storage_bytes(), Some(bytes));
    assert!(escaped.same_source(&alias));
    assert_eq!(
        escaped
            .route(0)
            .unwrap()
            .0
            .boundary_contract()
            .unwrap()
            .schema(),
        "résidual-边界"
    );
    drop(escaped);
    assert_eq!(retired.load(std::sync::atomic::Ordering::SeqCst), 1);
}

#[test]
fn cold_model_control_driver_keeps_exact_group_route_order_and_rejects_foreign_trace() {
    use crate::{DistributedExecutionPhase, OpaqueFailureAgreement};
    use eredu_nn::workspace::*;
    #[derive(Debug)]
    struct Facts;
    impl WorkspaceMechanisms for Facts {
        fn operation_bound(
            &self,
            operation: &WorkspaceOperation,
        ) -> Result<Option<WorkspaceOperationBound>, eredu_nn::Error> {
            assert!(matches!(
                operation.kind,
                WorkspaceOperationKind::CommunicationControl(_)
            ));
            assert!(operation.inputs.is_empty() && operation.outputs.is_empty());
            Ok(Some(WorkspaceOperationBound {
                outputs: Vec::new(),
                scratch_bytes: 0,
                assumptions: "cold control marker owns no tensor storage".into(),
            }))
        }
    }
    let group_id = CollectiveGroupId::new(7);
    let route_id = CommunicationRouteId::new(19);
    let transfer = CommunicationOperationRequirement::tensors(
        CommunicationOperation::SendReceive,
        [TensorDtype::F32],
        CommunicationTensorLimits::new(1, 2, 32, None).unwrap(),
        true,
    )
    .unwrap();
    let agreement = CommunicationOperationRequirement::failure_agreement(true);
    let route = CommunicationRouteDescriptor::new(route_id, 0, 0, 1, transfer.clone()).unwrap();
    let completion = CommunicationCompletionPolicy::new(
        std::time::Duration::from_secs(1),
        CompletionCancellationMode::QuarantineUntilComplete,
    )
    .unwrap();
    let mut proposals = proposals();
    for (rank, proposal) in proposals.iter_mut().enumerate() {
        let group = CommunicationGroupDescriptor::new(
            group_id,
            0,
            vec![0, 1],
            Some(rank),
            CommunicationGroupRequirements::new([agreement.clone()]).unwrap(),
        )
        .unwrap();
        proposal.manifest = CommunicationManifest::new(2, rank, vec![group], vec![route.clone()])
            .unwrap()
            .with_completion_policy(completion);
    }
    let agreed = establish_communication_session(
        &transport(0, &proposals),
        &proposals[0].manifest,
        proposals[0].nonce,
    )
    .unwrap();
    let capability = CommunicationCapabilities::new([agreement, transfer])
        .unwrap()
        .with_completion_capabilities(
            CommunicationCompletionCapabilities::new([
                CompletionCancellationMode::QuarantineUntilComplete,
            ])
            .unwrap(),
        );
    let source = prepare_communication_realization(
        &proposals[0].manifest,
        agreed.manifests(),
        &capability,
        CommunicationTopologyCapabilities::RingWithWorldWaves,
    )
    .unwrap()
    .into_retained_source(agreed.identity())
    .unwrap();
    let context = WorkspaceContext::new(Facts);
    let communication =
        crate::working_memory::workspace_partition_communication(&source, &context).unwrap();
    let parallel = WorkspaceParallelContext::new(0, 2).unwrap();
    for (phase, success) in [
        (DistributedExecutionPhase::Execution, true),
        (
            DistributedExecutionPhase::BoundarySourceCompletion(route_id),
            true,
        ),
        (
            DistributedExecutionPhase::BoundarySourceReady(route_id),
            false,
        ),
    ] {
        assert_eq!(
            communication
                .agree_phase_with_parallel_context(
                    &mut OpaqueFailureAgreement,
                    group_id,
                    phase,
                    success,
                    &context,
                    Some(&parallel)
                )
                .unwrap(),
            success
        );
    }
    let report = context.report(&[]).unwrap();
    let actual = report
        .operations
        .iter()
        .map(|op| match op.kind {
            WorkspaceOperationKind::CommunicationControl(control) => control,
            _ => panic!("control path created an unrelated tensor operation"),
        })
        .collect::<Vec<_>>();
    assert_eq!(
        actual,
        [
            WorkspaceModelControl {
                group: group_id,
                phase: WorkspaceModelControlPhase::Execution
            },
            WorkspaceModelControl {
                group: group_id,
                phase: WorkspaceModelControlPhase::BoundarySourceCompletion { route: 19 }
            },
            WorkspaceModelControl {
                group: group_id,
                phase: WorkspaceModelControlPhase::BoundarySourceReady { route: 19 }
            },
        ]
    );
    assert_eq!(report.tensor_buffers.total_bytes, Some(0));
    communication.authority().ensure_active().unwrap();
    let foreign = WorkspaceContext::new(Facts);
    assert!(
        communication
            .agree_phase_with_parallel_context(
                &mut OpaqueFailureAgreement,
                group_id,
                DistributedExecutionPhase::Execution,
                true,
                &foreign,
                None
            )
            .is_err()
    );
    assert_eq!(foreign.operation_count(), 0);
}
