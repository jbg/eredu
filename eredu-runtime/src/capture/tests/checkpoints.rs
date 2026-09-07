use super::*;

fn setup() -> (CaptureSession, CaptureDiscovery) {
    let (plan, catalog, mut support, capabilities) =
        fixture(CaptureTransform::Preview { max_elements: 2 });
    support.capture = capabilities.clone();
    let admitted = admit(plan, &catalog, &support, &capabilities).unwrap();
    (
        CaptureSession::new(admitted),
        CaptureDiscovery {
            artifact_identity: "source".into(),
            catalog,
            support,
        },
    )
}

fn run(session: &mut CaptureSession, prediction: u64) -> CapturedStep {
    session
        .begin_step(
            if prediction == 0 {
                CapturePhase::Prefill
            } else {
                CapturePhase::Decode
            },
            prediction,
        )
        .unwrap();
    // Missing captures remain explicit records; they are not fabricated values.
    session.take_step().unwrap()
}

#[test]
fn restore_rewinds_position_without_refunding_usage_or_repeating_delivery() {
    let (mut session, discovery) = setup();
    let initial = session.checkpoint(&discovery).unwrap();
    assert_eq!(initial.next_prediction(), 0);
    run(&mut session, 0);
    let saved = session.checkpoint(&discovery).unwrap();
    assert_eq!(saved.next_prediction(), 1);
    run(&mut session, 1);
    let spent = session.cumulative_usage();
    session.restore(&saved).unwrap();
    assert_eq!(session.cumulative_usage(), spent);
    assert!(session.take_step().is_none());
    let repeated = run(&mut session, 1);
    assert_eq!(repeated.prediction_index, 1);
    assert_eq!(
        repeated.cumulative_usage,
        spent.checked_add(repeated.step_usage).unwrap()
    );
    session.restore(&initial).unwrap();
    assert_eq!(session.checkpoint(&discovery).unwrap().next_prediction(), 0);
    assert_eq!(saved.next_prediction(), 1);
}

#[test]
fn records_and_source_identity_gate_checkpoint_and_restore() {
    let (mut session, discovery) = setup();
    let initial = session.checkpoint(&discovery).unwrap();
    let (mut other, _) = setup();
    assert!(other.restore(&initial).is_err());
    session.begin_step(CapturePhase::Prefill, 0).unwrap();
    assert!(session.checkpoint(&discovery).is_err());
    assert!(session.restore(&initial).is_err());
    session.take_step().unwrap();
    let mut incompatible = discovery.clone();
    incompatible.catalog.points[0].node_id = "another-node".into();
    assert!(session.checkpoint(&incompatible).is_err());
    session.restore(&initial).unwrap();
}

#[test]
fn forks_inherit_usage_and_recheck_future_geometry_and_source() {
    let (mut session, discovery) = setup();
    run(&mut session, 0);
    let checkpoint = session.checkpoint(&discovery).unwrap();
    let fork = |discovery: &CaptureDiscovery, limits: CaptureLimits| {
        checkpoint.fork(
            CaptureForkRequest {
                discovery,
                max_predictions: 10,
                limits,
                intervention: None,
            },
            |_, _, _| Ok(CaptureUsage::default()),
        )
    };
    let mut child = fork(&discovery, session.plan().plan().limits.clone()).unwrap();
    assert_eq!(child.cumulative_usage(), checkpoint.inherited_usage());
    assert!(child.restore(&checkpoint).is_err());
    run(&mut session, 1);
    assert_eq!(child.cumulative_usage(), checkpoint.inherited_usage());
    let child_step = run(&mut child, 1);
    assert_eq!(
        child_step.cumulative_usage,
        checkpoint
            .inherited_usage()
            .checked_add(child_step.step_usage)
            .unwrap()
    );
    let mut changed = discovery.clone();
    changed.artifact_identity = "other-source".into();
    assert!(fork(&changed, session.plan().plan().limits.clone()).is_err());
    let mut limits = session.plan().plan().limits.clone();
    limits.cumulative = checkpoint.inherited_usage();
    // Even with zero transformation costs, future envelopes require capacity.
    assert!(fork(&discovery, limits).is_err());
    assert!(checkpoint
        .fork(
            CaptureForkRequest {
                discovery: &discovery,
                max_predictions: 10,
                limits: session.plan().plan().limits.clone(),
                intervention: None,
            },
            |_, _, _| Err(CaptureError::Unsupported("estimate unavailable".into()))
        )
        .is_err());
}

#[test]
fn continuation_preflight_keeps_schedule_origin_and_only_charges_future_steps() {
    let (plan, catalog, support, caps) = fixture(CaptureTransform::Summary);
    let mut plan = plan;
    plan.selections[0].schedule = CaptureSchedule {
        first_prediction: 1,
        every: 3,
        ..Default::default()
    };
    let metadata = metadata_reservation(&plan.selections[0], &catalog.points[0]).unwrap();
    let inherited = CaptureUsage {
        captures: 1,
        ..Default::default()
    };
    let cost = CaptureUsage {
        captures: 1,
        ..Default::default()
    };
    // Next=3 retains absolute predictions 4 and 7 (not 3,6,9).
    plan.limits.cumulative = inherited
        .checked_add(metadata.checked_mul(7).unwrap())
        .unwrap()
        .checked_add(cost.checked_mul(2).unwrap())
        .unwrap();
    let admitted = admit(plan, &catalog, &support, &caps).unwrap();
    preflight_continuation(
        &admitted,
        &[],
        CaptureUsage::default(),
        &[],
        3,
        inherited,
        |_, _, _| Ok(cost),
    )
    .unwrap();
    assert!(preflight_continuation(
        &admitted,
        &[],
        CaptureUsage::default(),
        &[],
        2,
        inherited,
        |_, _, _| Ok(cost)
    )
    .is_err());
    assert_eq!(
        admitted.plan().selections[0]
            .schedule
            .count_and_last_from(CapturePhase::Decode, 3, 10)
            .unwrap(),
        Some((2, 7))
    );
}

#[test]
fn failed_reservations_do_not_become_checkpoints_after_drain() {
    let (mut session, discovery) = setup();
    let checkpoint = session.checkpoint(&discovery).unwrap();
    // Consume remaining capacity through the public ledger operation in this
    // internal fixture to exercise a partial begin_step failure.
    session
        .ledger
        .reserve(session.plan().plan().limits.per_step)
        .unwrap();
    assert!(session.begin_step(CapturePhase::Prefill, 0).is_err());
    assert!(session.take_step().is_none());
    assert!(session.checkpoint(&discovery).is_err());
    session.restore(&checkpoint).unwrap();
    assert!(session.begin_step(CapturePhase::Prefill, 0).is_err());
}
