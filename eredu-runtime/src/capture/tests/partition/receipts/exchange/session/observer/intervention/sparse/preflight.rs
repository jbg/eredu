use super::*;

fn setup() -> (CaptureSession, HookTransport, SparseLayout) {
    let (capture, plan) = admission(0, false);
    let mut session = configured(capture, 9);
    session
        .enable_interventions(plan, Arc::new(Estimates))
        .unwrap();
    let transport = HookTransport {
        transport: transport(world(9), 0, Fault::None),
        hook: Some(transport(world(8), 0, Fault::None)),
        members: (0..8).collect(),
    };
    (session, transport, SparseLayout::new())
}

#[test]
fn sparse_preflight_rejects_incomplete_overlapping_ownership_and_spent_budget() {
    for fault in ["missing", "overlap", "peer", "budget"] {
        let (mut session, transport, mut layout) = setup();
        let epoch = DistributedCommitEpoch::FIRST;
        session
            .prepare_step_transaction(epoch, crate::ExpertPass::Prefill, 0)
            .unwrap();
        match fault {
            "missing" => {
                layout.0.remove(0);
            }
            "overlap" => {
                layout.0[0].coordinates = RoutedComponentCoordinateMap::new(
                    ComponentCoordinateMap::indices(5, vec![4, 1]).unwrap(),
                    layout.0[0].coordinates.units().clone(),
                );
            }
            "peer" => layout.0[1].source_peers = 4,
            "budget" => {
                let remaining = session.plan().plan().limits.cumulative.host_bytes
                    - session.cumulative_usage().host_bytes;
                session
                    .ledger
                    .reserve_quota(CaptureUsage {
                        host_bytes: remaining,
                        ..Default::default()
                    })
                    .unwrap();
            }
            _ => unreachable!(),
        }
        let backend = Backend::default();
        let result = session.prepare_partition_intervention(
            &transport,
            &layout,
            &backend,
            0,
            PartitionCaptureReceiptLimits {
                max_producers: 9,
                ..LIMITS
            },
        );
        let error = match result {
            Err(error) => error,
            Ok(_) => panic!("{fault} unexpectedly admitted"),
        };
        let message = error.to_string();
        assert!(
            match fault {
                "missing" => message.contains("incomplete"),
                "overlap" => message.contains("overlaps"),
                "peer" => message.contains("source geometry"),
                "budget" => matches!(
                    error,
                    PartitionCaptureExchangeError::Capture(CaptureError::Limit { .. })
                ),
                _ => false,
            },
            "{fault}: {error}"
        );
        assert_eq!(backend.transforms, 0);
        assert_eq!(backend.exported, 0);
        assert_eq!(transport.transport.calls.load(Ordering::SeqCst), 0);
        assert_eq!(
            transport
                .hook
                .as_ref()
                .unwrap()
                .calls
                .load(Ordering::SeqCst),
            0
        );
        session.finish_transaction(epoch, false);
        assert!(session
            .take_step()
            .unwrap()
            .interventions
            .iter()
            .all(|record| !matches!(
                record.outcome,
                InterventionOutcome::Applied | InterventionOutcome::Unmatched
            )));
    }
}

#[test]
fn sparse_authority_rejects_bypass_foreign_session_and_restore_replay() {
    let (mut session, transport, layout) = setup();
    let saved = session.checkpoint(&discovery(session.plan())).unwrap();
    let (mut foreign, _, _) = setup();
    let epoch = DistributedCommitEpoch::FIRST;
    session
        .prepare_step_transaction(epoch, crate::ExpertPass::Prefill, 0)
        .unwrap();
    foreign
        .prepare_step_transaction(epoch, crate::ExpertPass::Prefill, 0)
        .unwrap();
    let mut backend = Backend::default();
    let mut work = session
        .prepare_partition_intervention(
            &transport,
            &layout,
            &backend,
            0,
            PartitionCaptureReceiptLimits {
                max_producers: 9,
                ..LIMITS
            },
        )
        .unwrap();
    let input = value(vec![0, 4], vec![]);
    let counts = [0, 0, 0];
    let origins = crate::RoutedUnitOrigins::new(&counts, &[], 2).unwrap();
    let invocation = crate::RoutedUnitInvocation {
        input: &input,
        origins: Some(origins),
        unit_coordinates: Some(layout.0[0].coordinates.units()),
    };
    assert!(foreign
        .begin_partition_routed_intervention(&mut work, &mut backend, &invocation, true)
        .is_err());
    assert!(
        session
            .begin_partition_routed_intervention(&mut work, &mut backend, &invocation, true)
            .is_err(),
        "common coordination cannot be bypassed"
    );
    let used = session.cumulative_usage();
    session.finish_transaction(epoch, false);
    session.take_step().unwrap();
    session.restore(&saved).unwrap();
    assert_eq!(session.cumulative_usage(), used);
    session
        .prepare_step_transaction(epoch.next().unwrap(), crate::ExpertPass::Prefill, 0)
        .unwrap();
    assert!(session
        .begin_partition_routed_intervention(&mut work, &mut backend, &invocation, true)
        .is_err());
    assert!(session
        .finish_partition_routed_intervention::<std::io::Error, _>(&mut work, true)
        .is_err());
    assert!(session.complete_partition_intervention(work).is_err());
    assert_eq!(backend.transforms, 0);
    assert_eq!(transport.transport.calls.load(Ordering::SeqCst), 0);
    assert_eq!(
        transport
            .hook
            .as_ref()
            .unwrap()
            .calls
            .load(Ordering::SeqCst),
        0
    );
    assert!(session.cumulative_usage().host_bytes > used.host_bytes);
}
