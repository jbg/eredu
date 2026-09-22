use super::*;
use crate::*;

#[test]
fn all_transformations_revalidate_the_exact_retained_source() {
    let (admitted, discovery) = fixture();
    let source = SharedCapturePlan::new(admitted);
    let alias = source.clone();
    let capacity = source.capacity_bytes();
    let selections = source.admission().plan().selections.as_ptr();
    let points = source.admission().points().as_ptr();
    let digest = source.admission().identity().as_ptr();
    for _ in 0..3 {
        source.admission().revalidate(&discovery).unwrap();
        assert!(source.same_storage(&alias));
        assert_eq!(source.capacity_bytes(), capacity);
        assert_eq!(source.admission().plan().selections.as_ptr(), selections);
        assert_eq!(source.admission().points().as_ptr(), points);
        assert_eq!(source.admission().identity().as_ptr(), digest);
    }
    assert_eq!(
        source.admission().readmit(&discovery).unwrap().identity(),
        source.admission().identity()
    );
}

#[test]
fn every_selected_declaration_field_remains_part_of_admission_identity() {
    let (plan, discovery) = fixture();
    let mutations: &[fn(&mut ObservationPoint)] = &[
        |p| p.path.push_str(".changed"),
        |p| p.node_id.push_str(".changed"),
        |p| p.meaning.push_str(" changed"),
        |p| {
            p.value_type = ObservationValueType::RoutedUnits {
                routing: "changed".into(),
                geometry: RoutedUnitGeometry {
                    experts: 1,
                    units_per_expert: 1,
                    routes_per_token: 1,
                },
            }
        },
        |p| p.dtype = ObservationDtype::Unknown,
        |p| p.axes.as_mut().unwrap()[0].name.push_str(".changed"),
        |p| p.axes.as_mut().unwrap()[0].dimension = SymbolicDimension::Known(5),
        |p| p.prefill = !p.prefill,
        |p| p.decode = !p.decode,
        |p| p.requirements.clear(),
        |p| p.position = ObservationPosition::AfterIntervention,
        |p| p.retained_bytes = Some(19),
        |p| p.host_bytes = Some(23),
    ];
    for (index, mutation) in mutations.iter().enumerate() {
        let mut changed = discovery.clone();
        mutation(&mut changed.catalog.points[0]);
        assert!(plan.revalidate(&changed).is_err(), "mutation {index}");
        if let Ok(other) = plan.readmit(&changed) {
            assert_ne!(other.identity(), plan.identity(), "mutation {index}");
        }
    }
    let mut changed = discovery.clone();
    if let ObservationValueType::RoutedUnits { routing, geometry } =
        &mut changed.catalog.points[1].value_type
    {
        routing.push_str(".changed");
        geometry.units_per_expert += 1;
    } else {
        panic!("routed fixture");
    }
    assert!(plan.revalidate(&changed).is_err());
}

#[test]
fn current_transform_and_histogram_limits_share_initial_admission_checks() {
    let (plan, discovery) = fixture();
    for kind in &discovery.support.capture.transformations {
        let mut changed = discovery.clone();
        changed
            .support
            .capture
            .transformations
            .retain(|present| present != kind);
        assert!(matches!(
            plan.revalidate(&changed),
            Err(CaptureError::Unsupported(_))
        ));
        assert!(matches!(
            plan.readmit(&changed),
            Err(CaptureError::Unsupported(_))
        ));
    }
    let mut changed = discovery.clone();
    changed.support.capture.max_histogram_bins = 1;
    assert!(matches!(
        plan.revalidate(&changed),
        Err(CaptureError::Invalid(_))
    ));
    assert!(matches!(
        plan.readmit(&changed),
        Err(CaptureError::Invalid(_))
    ));
}

#[test]
fn enabled_phases_recheck_support_while_disabled_phases_remain_irrelevant() {
    let (original, mut discovery) = fixture();
    let mut raw = original.plan().clone();
    for s in &mut raw.selections {
        s.schedule.prefill = false;
    }
    discovery.support.points[0].prefill =
        ObservationSupportStatus::Unsupported("no prefill".into());
    let plan = raw
        .admit(
            &discovery.catalog,
            &discovery.support,
            &discovery.support.capture,
            original.request(),
        )
        .unwrap();
    plan.revalidate(&discovery).unwrap();
    discovery.support.points[0].decode =
        ObservationSupportStatus::Conditional("input dependent".into());
    plan.revalidate(&discovery).unwrap();
    for status in [
        ObservationSupportStatus::Unsupported("missing".into()),
        ObservationSupportStatus::Unverified("unknown".into()),
    ] {
        discovery.support.points[0].decode = status;
        assert!(matches!(
            plan.revalidate(&discovery),
            Err(CaptureError::Unsupported(_))
        ));
        assert!(plan.readmit(&discovery).is_err());
    }
    discovery.support.points.remove(0);
    assert!(matches!(
        plan.revalidate(&discovery),
        Err(CaptureError::Unsupported(_))
    ));
}

#[test]
fn discovery_order_and_unselected_facts_do_not_replace_selected_semantics() {
    let (plan, mut discovery) = fixture();
    discovery.catalog.points.reverse();
    discovery.support.points.reverse();
    let mut other = discovery.catalog.points[0].clone();
    other.path = "unselected".into();
    discovery.catalog.points.insert(0, other);
    discovery
        .support
        .capture
        .conditions
        .push("diagnostic".into());
    // Artifact/session binding and this conversion fact were not in readmit's
    // semantic digest; this borrowed method grants no new execution authority.
    discovery.artifact_identity = "different physical source".into();
    discovery.support.points[0].floating_to_f32 = false;
    plan.revalidate(&discovery).unwrap();
    assert_eq!(
        plan.readmit(&discovery).unwrap().identity(),
        plan.identity()
    );
    let mut shadow = discovery.catalog.points[1].clone();
    shadow.meaning.push_str(" shadow");
    discovery.catalog.points.insert(0, shadow);
    assert!(
        plan.revalidate(&discovery).is_err(),
        "same first-match lookup as admission"
    );
}

#[test]
fn ordinary_cached_and_independent_geometry_proofs_are_preserved() {
    let (original, discovery) = fixture();
    let raw = original.plan().clone();
    let origin = CaptureTextOrigin {
        cached_positions: 17,
    };
    let cached = raw
        .clone()
        .admit_with_text_origin(
            &discovery.catalog,
            &discovery.support,
            &discovery.support.capture,
            original.request(),
            origin,
        )
        .unwrap();
    cached.revalidate(&discovery).unwrap();
    assert_eq!(cached.text_origin(), Some(origin));
    assert_eq!(
        cached
            .geometry_at(CapturePhase::Decode, 2, None)
            .unwrap()
            .context,
        Some(22)
    );
    assert_eq!(
        cached.readmit(&discovery).unwrap().identity(),
        cached.identity()
    );
    let bounds = CaptureInvocationBounds {
        batch: 1,
        max_sequence: 3,
        max_context: Some(31),
        max_predictions: 5,
    };
    let independent = raw
        .admit_invocations(
            &discovery.catalog,
            &discovery.support,
            &discovery.support.capture,
            bounds,
        )
        .unwrap();
    independent.revalidate(&discovery).unwrap();
    assert_eq!(independent.invocation_bounds(), Some(bounds));
    assert!(independent
        .geometry_at(CapturePhase::Decode, 1, None)
        .is_err());
    assert_eq!(
        independent.readmit(&discovery).unwrap().identity(),
        independent.identity()
    );
}

#[test]
fn empty_admissions_still_validate_schema() {
    let (original, mut discovery) = fixture();
    let mut raw = original.plan().clone();
    raw.selections.clear();
    let plan = raw
        .admit(
            &discovery.catalog,
            &discovery.support,
            &discovery.support.capture,
            original.request(),
        )
        .unwrap();
    plan.revalidate(&discovery).unwrap();
    discovery.catalog.points.clear();
    discovery.support.points.clear();
    plan.revalidate(&discovery).unwrap();
    discovery.catalog.schema_version += 1;
    assert!(matches!(
        plan.revalidate(&discovery),
        Err(CaptureError::Invalid(_))
    ));
    discovery.catalog.schema_version -= 1;
    discovery.support.schema_version += 1;
    assert!(matches!(
        plan.revalidate(&discovery),
        Err(CaptureError::Invalid(_))
    ));
}

fn fixture() -> (AdmittedCapturePlan, CaptureDiscovery) {
    let point = ObservationPoint {
        path: crate::MODEL_LOGITS_OBSERVATION_PATH.into(),
        node_id: "decoder.output".into(),
        meaning: "nonzero raw output values".into(),
        value_type: ObservationValueType::Tensor,
        dtype: ObservationDtype::Floating,
        axes: Some(vec![TensorAxis {
            name: "width".into(),
            dimension: SymbolicDimension::Known(4),
        }]),
        prefill: true,
        decode: true,
        requirements: vec![ObservationRequirement::ActivationHooks],
        position: ObservationPosition::BeforeIntervention,
        retained_bytes: None,
        host_bytes: None,
    };
    let mut routed = point.clone();
    routed.path = "experts.units".into();
    routed.axes = None;
    routed.value_type = ObservationValueType::RoutedUnits {
        routing: "experts.dispatch".into(),
        geometry: RoutedUnitGeometry {
            experts: 4,
            units_per_expert: 3,
            routes_per_token: 2,
        },
    };
    let catalog = ObservationCatalog {
        schema_version: 1,
        points: vec![point, routed],
        completeness: DescriptionCompleteness::Complete,
    };
    let transforms = vec![
        CaptureTransform::FullTensor,
        CaptureTransform::Slice,
        CaptureTransform::Preview { max_elements: 2 },
        CaptureTransform::Summary,
        CaptureTransform::Histogram {
            edges: vec![-1.0, 0.0, 1.0],
        },
        CaptureTransform::TopCandidates { count: 2 },
        CaptureTransform::TokenScores {
            token_ids: vec![0, 3],
        },
        CaptureTransform::RoutedUnits,
    ];
    let capabilities = CaptureCapabilities {
        transformations: transforms.iter().map(CaptureTransform::kind).collect(),
        max_histogram_bins: 4,
        conditions: vec![],
    };
    let support = ObservationSupportReport {
        schema_version: 1,
        capture: capabilities.clone(),
        points: catalog
            .points
            .iter()
            .map(|p| ObservationSupport {
                path: p.path.clone(),
                prefill: ObservationSupportStatus::Supported,
                decode: ObservationSupportStatus::Supported,
                floating_to_f32: true,
            })
            .collect(),
    };
    let all = CaptureUsage {
        captures: u64::MAX,
        retained_bytes: u64::MAX,
        host_bytes: u64::MAX,
        encoded_bytes: u64::MAX,
    };
    let plan = CapturePlan {
        schema_version: 1,
        selections: transforms
            .into_iter()
            .enumerate()
            .map(|(index, transform)| CaptureSelection {
                id: format!("selection-{index}"),
                path: if matches!(transform, CaptureTransform::RoutedUnits) {
                    "experts.units".into()
                } else {
                    crate::MODEL_LOGITS_OBSERVATION_PATH.into()
                },
                schedule: CaptureSchedule::default(),
                slices: if matches!(transform, CaptureTransform::Slice) {
                    vec![CaptureSlice {
                        axis: "width".into(),
                        start: 1,
                        end: 4,
                        stride: 2,
                    }]
                } else {
                    vec![]
                },
                transform,
            })
            .collect(),
        limits: CaptureLimits {
            per_step: all,
            cumulative: all,
            on_limit: CaptureLimitPolicy::Fail,
        },
    }
    .admit(
        &catalog,
        &support,
        &capabilities,
        CaptureRequestShape {
            batch: 1,
            prompt_tokens: 3,
            max_predictions: 5,
        },
    )
    .unwrap();
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
fn borrowed_parts_preserve_exact_proof_and_the_discovery_validation_errors() {
    let (admitted, discovery) = fixture();
    let shared = SharedCapturePlan::new(admitted);
    let selections = shared.admission().plan().selections.as_ptr();
    let points = shared.admission().points().as_ptr();
    let digest = shared.admission().identity().as_ptr();
    let capacity = shared.capacity_bytes();
    shared
        .admission()
        .revalidate_parts(&discovery.catalog, &discovery.support)
        .unwrap();
    assert_eq!(shared.admission().plan().selections.as_ptr(), selections);
    assert_eq!(shared.admission().points().as_ptr(), points);
    assert_eq!(shared.admission().identity().as_ptr(), digest);
    assert_eq!(shared.capacity_bytes(), capacity);
    for mutation in [
        (|d: &mut CaptureDiscovery| d.catalog.points[0].meaning.push_str(" changed"))
            as fn(&mut CaptureDiscovery),
        |d| d.support.capture.transformations.clear(),
        |d| d.support.points[0].decode = ObservationSupportStatus::Unsupported("disabled".into()),
        |d| d.catalog.schema_version += 1,
    ] {
        let mut changed = discovery.clone();
        mutation(&mut changed);
        let ordinary = shared.admission().revalidate(&changed).unwrap_err();
        let borrowed = shared
            .admission()
            .revalidate_parts(&changed.catalog, &changed.support)
            .unwrap_err();
        assert_eq!(
            std::mem::discriminant(&ordinary),
            std::mem::discriminant(&borrowed)
        );
        assert_eq!(ordinary.to_string(), borrowed.to_string());
    }
}

#[test]
fn fixed_borrowed_revalidation_shares_transform_limits_and_enabled_phases() {
    let (plan, discovery) = fixture();
    let validate = |current: &CaptureDiscovery| {
        plan.revalidate_borrowed_declarations(
            &current.catalog,
            current.support.schema_version,
            &current.support.capture,
            |point, phase| {
                let Some(row) = current
                    .support
                    .points
                    .iter()
                    .find(|row| row.path == point.path)
                else {
                    return false;
                };
                let status = match phase {
                    CapturePhase::Prefill => &row.prefill,
                    CapturePhase::Decode => &row.decode,
                };
                matches!(
                    status,
                    ObservationSupportStatus::Supported | ObservationSupportStatus::Conditional(_)
                )
            },
        )
    };
    assert_eq!(validate(&discovery), Ok(()));
    for mutation in 0..5 {
        let mut current = discovery.clone();
        match mutation {
            0 => current.support.capture.transformations.clear(),
            1 => current.support.capture.max_histogram_bins = 0,
            2 => current.catalog.points[0].meaning.push_str(" stale"),
            3 => {
                current.support.points[0].decode =
                    ObservationSupportStatus::Unverified("unavailable".into())
            }
            4 => current.support.schema_version = 0,
            _ => unreachable!(),
        }
        assert!(validate(&current).is_err());
        assert!(plan.revalidate(&current).is_err());
    }
}

#[derive(Debug)]
struct AdmissionAccount {
    calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    refuse: usize,
}
impl crate::HostMetadataAccount for AdmissionAccount {
    fn reserve_metadata(&self, bytes: usize) -> Result<(), crate::HostMetadataFundingError> {
        use std::sync::atomic::Ordering::SeqCst;
        let index = self.calls.fetch_add(1, SeqCst);
        if index == self.refuse {
            Err(crate::HostMetadataFundingError::Capacity {
                required: bytes as u64,
                available: 0,
            })
        } else {
            Ok(())
        }
    }
}
fn paid_admission(
    raw: &CapturePlan,
    discovery: &CaptureDiscovery,
    request: CaptureRequestShape,
    refuse: usize,
) -> (Result<AdmittedCapturePlan, CaptureError>, usize) {
    use std::sync::{
        atomic::{AtomicUsize, Ordering::SeqCst},
        Arc,
    };
    let calls = Arc::new(AtomicUsize::new(0));
    let funding = crate::HostMetadataFunding::new(AdmissionAccount {
        calls: calls.clone(),
        refuse,
    })
    .unwrap();
    let result = raw.copy_with_funding(&funding).and_then(|plan| {
        plan.admit_with_funding(
            &discovery.catalog,
            &discovery.support,
            &discovery.support.capture,
            request,
            &funding,
        )
    });
    (result, calls.load(SeqCst))
}

#[test]
fn funded_admission_preserves_all_transforms_and_refuses_each_reached_producer() {
    let (ordinary, discovery) = fixture();
    let (paid, calls) = paid_admission(ordinary.plan(), &discovery, ordinary.request(), usize::MAX);
    let paid = paid.unwrap();
    assert_eq!(paid.identity(), ordinary.identity());
    assert_eq!(paid.plan(), ordinary.plan());
    assert_eq!(paid.points(), ordinary.points());
    assert_ne!(
        paid.plan().selections.as_ptr(),
        ordinary.plan().selections.as_ptr()
    );
    assert!(calls > 20);
    // Index zero constructs the test account, before the compiler invocation.
    for refusal in 1..calls {
        let (result, reached) =
            paid_admission(ordinary.plan(), &discovery, ordinary.request(), refusal);
        assert!(
            matches!(
                result,
                Err(CaptureError::AdmissionStorage(
                    CaptureAdmissionStorageError::Funding(
                        crate::HostMetadataFundingError::Capacity { available: 0, .. }
                    )
                ))
            ),
            "producer {refusal}: {result:?}"
        );
        assert_eq!(reached, refusal + 1, "producer after first refusal");
    }
}

#[test]
fn funded_diagnostics_keep_first_source_error_and_refuse_before_formatting() {
    let (ordinary, discovery) = fixture();
    let mut cases = Vec::new();
    let mut raw = ordinary.plan().clone();
    raw.schema_version += 1;
    cases.push(raw);
    let mut raw = ordinary.plan().clone();
    raw.selections[0].path = "missing target".into();
    cases.push(raw);
    let mut raw = ordinary.plan().clone();
    raw.selections[1].id = raw.selections[0].id.clone();
    raw.selections[2].path = "later missing target".into();
    cases.push(raw);
    let mut raw = ordinary.plan().clone();
    let duplicate = raw.selections[1].slices[0].clone();
    raw.selections[1].slices.push(duplicate);
    cases.push(raw);
    let mut raw = ordinary.plan().clone();
    raw.selections[6].transform = CaptureTransform::TokenScores {
        token_ids: vec![3, 3],
    };
    cases.push(raw);
    for raw in cases {
        let ordinary_error = raw
            .clone()
            .admit(
                &discovery.catalog,
                &discovery.support,
                &discovery.support.capture,
                ordinary.request(),
            )
            .unwrap_err();
        let (result, calls) = paid_admission(&raw, &discovery, ordinary.request(), usize::MAX);
        assert_eq!(result.unwrap_err().to_string(), ordinary_error.to_string());
        for refusal in 1..calls {
            let (result, reached) = paid_admission(&raw, &discovery, ordinary.request(), refusal);
            assert!(
                matches!(
                    result,
                    Err(CaptureError::AdmissionStorage(
                        CaptureAdmissionStorageError::Funding(_)
                    ))
                ),
                "producer {refusal}: {result:?}"
            );
            assert_eq!(reached, refusal + 1);
        }
    }
}
