use super::*;

#[test]
fn projection_shares_preflight_geometry_and_recounts_absolute_schedules() {
    let (mut plan, catalog, support, capabilities) = fixture(CaptureTransform::FullTensor);
    plan.selections[0].schedule.every = 2;
    let admitted = admit(plan, &catalog, &support, &capabilities).unwrap();
    let estimate = |shape: &[u64], _: &CaptureSelection, _: &ResolvedCaptureSlice| {
        let bytes = shape.iter().product::<u64>() * 4;
        Ok(CaptureUsage {
            captures: 1,
            retained_bytes: bytes,
            host_bytes: bytes,
            encoded_bytes: bytes * 8,
        })
    };
    preflight(&admitted, estimate).unwrap();
    let projection = project_capture_usage(&admitted, 0, |s, c, r| estimate(s, c, r).map(Some))
        .unwrap()
        .unwrap();
    let all = projection.outlook(0, 10).unwrap().unwrap();
    assert!(all.complete);
    assert_eq!(all.total.captures, 5); // prefill and predictions 2,4,6,8
    assert_eq!(all.total.retained_bytes, 48 + 4 * 16);
    assert_eq!(all.phases[0].retained_bytes, 48);
    assert_eq!(all.phases[1].retained_bytes, 16);
    let continued = projection.outlook(3, 8).unwrap().unwrap();
    assert_eq!(continued.total.captures, 2); // absolute 4,6; no frequency reset
    assert_eq!(continued.total.retained_bytes, 32);
    assert_eq!(
        continued.total.host_bytes,
        projection.per_step_metadata.host_bytes * 5 + 32
    );
    assert_eq!(
        projection.outlook(3, 3).unwrap().unwrap().total,
        CaptureUsage::default()
    );
    assert!(projection.outlook(0, 11).unwrap().is_none());
    let wire = serde_json::to_string(&projection).unwrap();
    let restored: CaptureUsageProjection = serde_json::from_str(&wire).unwrap();
    assert_eq!(restored.outlook(3, 8).unwrap().unwrap(), continued);
}

#[test]
fn uncovered_sources_remain_unknown_only_when_scheduled_and_skip_still_projects_values() {
    let (mut plan, catalog, support, capabilities) = fixture(CaptureTransform::FullTensor);
    plan.selections[0].schedule.first_prediction = 2;
    plan.limits.on_limit = CaptureLimitPolicy::Skip;
    let admitted = admit(plan, &catalog, &support, &capabilities).unwrap();
    let unknown = project_capture_usage(&admitted, 0, |_, _, _| Ok(None))
        .unwrap()
        .unwrap();
    assert!(unknown.outlook(0, 2).unwrap().unwrap().complete);
    assert!(!unknown.outlook(0, 3).unwrap().unwrap().complete);
    let known = project_capture_usage(&admitted, 0, |_, _, _| {
        Ok(Some(CaptureUsage {
            captures: 1,
            retained_bytes: 42,
            ..Default::default()
        }))
    })
    .unwrap()
    .unwrap();
    assert_eq!(
        known.outlook(0, 3).unwrap().unwrap().total.retained_bytes,
        42
    );
    let mut overflowing = known;
    overflowing.per_step_metadata.host_bytes = u64::MAX;
    assert!(overflowing.outlook(0, 3).is_err());
}
