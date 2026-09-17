use super::*;

fn source() -> (AdmittedCapturePlan, CaptureDiscovery) {
    let (raw, catalog, mut support, capabilities) =
        fixture(CaptureTransform::Preview { max_elements: 2 });
    support.capture = capabilities.clone();
    let plan = admit(raw, &catalog, &support, &capabilities).unwrap();
    (
        plan,
        CaptureDiscovery {
            artifact_identity: "source".into(),
            catalog,
            support,
        },
    )
}

#[test]
fn session_preflight_borrows_original_selections_and_rejects_before_estimation() {
    let (plan, discovery) = source();
    let calls = Cell::new(0);
    validate_session(&plan, &discovery, |shape, selection, slice| {
        assert!(std::ptr::eq(selection, &plan.plan().selections[0]));
        assert_eq!(slice.shape, shape);
        calls.set(calls.get() + 1);
        Ok(CaptureUsage::default())
    })
    .unwrap();
    assert_eq!(calls.get(), 2);
    for semantics in [true, false] {
        let mut changed = discovery.clone();
        if semantics {
            changed.catalog.points[0].meaning.push_str(" changed");
        } else {
            changed.support.capture.transformations.clear();
        }
        assert!(validate_session(&plan, &changed, |_, _, _| {
            calls.set(calls.get() + 1);
            Ok(CaptureUsage::default())
        })
        .is_err());
        assert_eq!(calls.get(), 2);
    }
}

#[test]
fn checkpoint_revalidation_preserves_live_source_usage_and_rejects_changed_facts() {
    let (plan, discovery) = source();
    let mut session = CaptureSession::new(plan);
    let selection = session.plan().plan().selections.as_ptr();
    session.begin_step(CapturePhase::Prefill, 0).unwrap();
    let mut backend = ProbeBackend {
        copies: Cell::new(0),
    };
    session
        .observe(&mut backend, "block.output", &vec![3, 4])
        .unwrap();
    let step = session.take_step().unwrap();
    assert!(step.records[0].payload.is_some());
    let usage = session.cumulative_usage();
    let saved = session.checkpoint(&discovery).unwrap();
    assert_eq!(saved.next_prediction(), 1);
    for semantics in [true, false] {
        let mut changed = discovery.clone();
        if semantics {
            changed.catalog.points[0].host_bytes = Some(99);
        } else {
            changed.support.points[0].decode =
                ObservationSupportStatus::Unverified("unknown".into());
        }
        assert!(session.checkpoint(&changed).is_err());
        assert_eq!(session.cumulative_usage(), usage);
        assert_eq!(session.plan().plan().selections.as_ptr(), selection);
        assert!(session.take_step().is_none());
    }
    session.restore(&saved).unwrap();
    assert_eq!(session.cumulative_usage(), usage);
    assert_eq!(backend.copies.get(), 1);
}
