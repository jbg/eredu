use super::*;
use crate::{DescriptionCompleteness, ObservationDtype, ObservationPosition, ObservationSupport};
use std::{
    panic::{catch_unwind, AssertUnwindSafe},
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
};

pub(super) struct PayloadRetired(pub Arc<AtomicBool>);
impl Drop for PayloadRetired {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

fn admitted() -> AdmittedCapturePlan {
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
        physical_native_limit: false,
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
    CapturePlan {
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
            physical_native_bytes: None,
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
    .unwrap()
}

#[test]
fn child_limit_copy_preserves_declarations_and_uses_canonical_identity() {
    let source = SharedCapturePlan::new(admitted());
    let mut limits = source.plan().limits.clone();
    limits.cumulative.captures = 41;
    let capabilities = CaptureCapabilities {
        transformations: vec![], max_histogram_bins: 0,
        physical_native_limit: false, conditions: vec![],
    };
    let copy = PreparedCapturePlanCopy::inspect_limit_revision(&source, limits.clone(), &capabilities).unwrap()
        .copy(HostPreparationAuthority::unmanaged()).unwrap();
    assert_eq!(copy.plan().limits, limits);
    assert_eq!(copy.plan().selections, source.plan().selections);
    assert_eq!(copy.admission().request(), source.request());
    assert_eq!(copy.admission().text_origin(), source.text_origin());
    assert_ne!(copy.admission().identity(), source.identity());
    assert!(copy.is_limit_revision_of(&source));
    assert!(!copy.is_limit_revision_of(&SharedCapturePlan::new(source.admission().clone())));
    let bytes = serde_json::to_vec(&(copy.plan(), &source.points, source.request())).unwrap();
    let reference: String = Sha256::digest(bytes).iter().map(|n| format!("{n:02x}")).collect();
    assert_eq!(copy.admission().identity(), reference);
    limits.physical_native_bytes = Some(1);
    assert!(matches!(PreparedCapturePlanCopy::inspect_limit_revision(&source, limits, &capabilities),
        Err(CapturePlanCopyError::Capability)));
}

fn grow<T>(values: &mut Vec<T>) -> u64 {
    let before = values.capacity();
    values.reserve_exact(before + 7);
    ((values.capacity() - before) * size_of::<T>()) as u64
}
fn grow_text(value: &mut String) -> u64 {
    let before = value.capacity();
    value.reserve_exact(before + 11);
    (value.capacity() - before) as u64
}
fn changed(
    source: &mut AdmittedCapturePlan,
    mutation: impl FnOnce(&mut AdmittedCapturePlan) -> u64,
) {
    let before = capacity(source).unwrap();
    let semantic = source.identity().to_owned();
    let delta = mutation(source);
    assert!(delta > 0);
    assert_eq!(capacity(source), before.checked_add(delta));
    assert_eq!(source.identity(), semantic);
}

#[test]
fn every_owned_buffer_counts_spare_capacity_without_changing_admission() {
    let mut source = admitted();
    changed(&mut source, |p| grow(&mut p.plan.selections));
    changed(&mut source, |p| grow(&mut p.points));
    changed(&mut source, |p| grow_text(&mut p.identity));
    for i in 0..source.plan.selections.len() {
        changed(&mut source, |p| grow_text(&mut p.plan.selections[i].id));
        changed(&mut source, |p| grow_text(&mut p.plan.selections[i].path));
        changed(&mut source, |p| grow(&mut p.plan.selections[i].slices));
        changed(&mut source, |p| grow_text(&mut p.points[i].path));
        changed(&mut source, |p| grow_text(&mut p.points[i].node_id));
        changed(&mut source, |p| grow_text(&mut p.points[i].meaning));
        changed(&mut source, |p| grow(&mut p.points[i].requirements));
        if source.points[i].axes.is_some() {
            changed(&mut source, |p| grow(p.points[i].axes.as_mut().unwrap()));
            changed(&mut source, |p| {
                grow_text(&mut p.points[i].axes.as_mut().unwrap()[0].name)
            });
        }
    }
    changed(&mut source, |p| {
        grow_text(&mut p.plan.selections[1].slices[0].axis)
    });
    changed(&mut source, |p| match &mut p.plan.selections[4].transform {
        CaptureTransform::Histogram { edges } => grow(edges),
        _ => panic!(),
    });
    changed(&mut source, |p| match &mut p.plan.selections[6].transform {
        CaptureTransform::TokenScores { token_ids } => grow(token_ids),
        _ => panic!(),
    });
    changed(&mut source, |p| match &mut p.points[7].value_type {
        ObservationValueType::RoutedUnits { routing, .. } => grow_text(routing),
        _ => panic!(),
    });
    let expected = capacity(&source);
    let shared = SharedCapturePlan::new(source);
    assert_eq!(shared.capacity_bytes(), expected);
}

#[test]
fn moving_and_aliasing_preserve_exact_plan_buffers_and_semantic_geometry() {
    let source = admitted();
    let selection = source.plan.selections.as_ptr();
    let point = source.points.as_ptr();
    let axis = source.points[0].axes.as_ref().unwrap().as_ptr();
    let name = source.points[0].meaning.as_ptr();
    let digest = source.identity.as_ptr();
    let shape = source.geometry_at(CapturePhase::Decode, 1, None).unwrap();
    let independent = SharedCapturePlan::new(source.clone());
    let shared = SharedCapturePlan::new(source);
    let alias = shared.clone();
    assert!(alias.same_storage(&shared));
    assert!(!alias.same_storage(&independent));
    assert_ne!(alias.storage_identity(), independent.storage_identity());
    assert_eq!(
        alias.admission().identity(),
        independent.admission().identity()
    );
    drop(shared);
    assert_eq!(alias.admission().plan().selections.as_ptr(), selection);
    assert_eq!(alias.admission().points().as_ptr(), point);
    assert_eq!(
        alias.admission().points()[0]
            .axes
            .as_ref()
            .unwrap()
            .as_ptr(),
        axis
    );
    assert_eq!(alias.admission().points()[0].meaning.as_ptr(), name);
    assert_eq!(alias.admission().identity().as_ptr(), digest);
    assert_eq!(
        alias
            .admission()
            .geometry_at(CapturePhase::Decode, 1, None)
            .unwrap(),
        shape
    );
}

struct Charge {
    count: Arc<AtomicUsize>,
    payload_retired: Arc<AtomicBool>,
}
impl Drop for Charge {
    fn drop(&mut self) {
        assert!(self.payload_retired.load(Ordering::SeqCst));
        self.count.fetch_add(1, Ordering::SeqCst);
    }
}
fn owner() -> (SharedCapturePlan, Arc<AtomicBool>, Arc<AtomicUsize>) {
    let retired = Arc::new(AtomicBool::new(false));
    let count = Arc::new(AtomicUsize::new(0));
    let shared = SharedCapturePlan(PlanOwner(Some(Arc::new(Inner {
        predecessor: None,
        plan: admitted(),
        retired: Some(PayloadRetired(retired.clone())),
        custody: SharedStorageCustody::new(),
        shell_retired: None,
        custody_retired: None,
        ordinary: Mutex::new(None),
    }))));
    (shared, retired, count)
}

#[test]
fn earlier_aliases_keep_per_domain_custody_until_all_nested_payload_retires() {
    let (shared, retired, count) = owner();
    let alias = shared.clone();
    let key = shared.storage_identity().clone();
    let domains = [
        SharedStorageDomain::default(),
        SharedStorageDomain::default(),
    ];
    for domain in &domains {
        assert!(shared
            .try_attach(domain, || Ok::<_, &'static str>(Box::new(Charge {
                count: count.clone(),
                payload_retired: retired.clone(),
            })))
            .unwrap());
        assert!(!alias
            .try_attach(domain, || -> Result<Box<dyn Send + Sync>, &'static str> {
                panic!("duplicate attachment must not acquire")
            })
            .unwrap());
    }
    drop(shared);
    assert!(!retired.load(Ordering::SeqCst));
    assert_eq!(count.load(Ordering::SeqCst), 0);
    assert!(alias.capacity_bytes().unwrap() > 0);
    drop(alias);
    assert!(retired.load(Ordering::SeqCst));
    assert_eq!(count.load(Ordering::SeqCst), 2);
    // A separately retained registry key never keeps the source/charge alive.
    drop(key);
    assert_eq!(count.load(Ordering::SeqCst), 2);
}

#[test]
fn rejected_and_panicking_attachment_preserve_existing_source_custody() {
    let (shared, retired, count) = owner();
    let first = SharedStorageDomain::default();
    shared
        .try_attach(&first, || {
            Ok::<_, &'static str>(Box::new(Charge {
                count: count.clone(),
                payload_retired: retired.clone(),
            }))
        })
        .unwrap();
    let second = SharedStorageDomain::default();
    assert!(matches!(
        shared.try_attach(&second, || Err("original typed rejection")),
        Err(SharedStorageAttachmentError::Provider(
            "original typed rejection"
        ))
    ));
    assert_eq!(count.load(Ordering::SeqCst), 0);
    assert!(catch_unwind(AssertUnwindSafe(|| {
        let _ = shared.try_attach(&second, || -> Result<Box<dyn Send + Sync>, &'static str> {
            panic!("provider unwind")
        });
    }))
    .is_err());
    assert!(matches!(
        shared.try_attach(&second, || Ok::<_, &'static str>(Box::new(()))),
        Err(SharedStorageAttachmentError::Poisoned)
    ));
    assert_eq!(count.load(Ordering::SeqCst), 0);
    drop(shared);
    assert!(retired.load(Ordering::SeqCst));
    assert_eq!(count.load(Ordering::SeqCst), 1);
}

#[test]
fn empty_admission_retains_its_inline_payload_and_digest_and_overflow_is_unknown() {
    let source = admitted();
    let none = CapturePlan::none()
        .admit(
            &ObservationCatalog {
                schema_version: 1,
                points: vec![],
                completeness: DescriptionCompleteness::Complete,
            },
            &ObservationSupportReport {
                schema_version: 1,
                capture: Default::default(),
                points: vec![],
            },
            &CaptureCapabilities::default(),
            source.request(),
        )
        .unwrap();
    let expected = size_of::<AdmittedCapturePlan>() as u64 + none.identity.capacity() as u64;
    assert_eq!(
        SharedCapturePlan::new(none).capacity_bytes(),
        Some(expected)
    );
    assert_eq!(bytes::<u32>(0), Some(0));
    if usize::BITS == 64 {
        assert_eq!(bytes::<u64>(usize::MAX), None);
    }
}

#[test]
fn escaped_source_and_concurrent_final_aliases_retire_shell_and_attachment_allocation_before_host()
{
    struct OrdinaryLease {
        shell: Arc<AtomicBool>,
        custody: Arc<AtomicBool>,
        drops: Arc<AtomicUsize>,
    }
    impl Drop for OrdinaryLease {
        fn drop(&mut self) {
            assert!(self.shell.load(Ordering::SeqCst));
            assert!(self.custody.load(Ordering::SeqCst));
            self.drops.fetch_add(1, Ordering::SeqCst);
        }
    }
    let shell = Arc::new(AtomicBool::new(false));
    let custody = Arc::new(AtomicBool::new(false));
    let drops = Arc::new(AtomicUsize::new(0));
    let source = SharedCapturePlan(PlanOwner(Some(Arc::new(Inner {
        predecessor: None,
        plan: admitted(),
        retired: None,
        custody: SharedStorageCustody::new(),
        shell_retired: Some(shell.clone()),
        custody_retired: Some(PayloadRetired(custody.clone())),
        ordinary: Mutex::new(None),
    }))));
    // Actual existing attachment storage must also be destroyed before the host.
    source
        .try_attach(&SharedStorageDomain::default(), || {
            Ok::<_, ()>(Box::new(17_u64))
        })
        .unwrap();
    let payload = source.capacity_bytes();
    let host = HostPreparationAuthority::retain(OrdinaryLease {
        shell: shell.clone(),
        custody: custody.clone(),
        drops: drops.clone(),
    });
    source.retain_host_preparation(&host).unwrap();
    drop(host);
    assert_eq!(
        source.capacity_bytes(),
        payload,
        "control is not payload diagnostic"
    );
    let escaped = source.clone();
    let alias = source.clone();
    drop(source);
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    let barrier = Arc::new(std::sync::Barrier::new(2));
    let other = barrier.clone();
    let thread = std::thread::spawn(move || {
        other.wait();
        drop(alias);
    });
    barrier.wait();
    drop(escaped);
    thread.join().unwrap();
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[test]
fn incoming_host_panic_keeps_pending_tail_on_consuming_retirement_path() {
    struct Earlier {
        allocation: Arc<AtomicBool>,
        observed: Arc<AtomicBool>,
        drops: Arc<AtomicUsize>,
    }
    impl Drop for Earlier {
        fn drop(&mut self) {
            self.observed
                .store(self.allocation.load(Ordering::SeqCst), Ordering::SeqCst);
            self.drops.fetch_add(1, Ordering::SeqCst);
        }
    }
    struct Panics;
    impl Drop for Panics {
        fn drop(&mut self) {
            std::panic::panic_any(73_u32);
        }
    }
    let allocation = Arc::new(AtomicBool::new(false));
    let observed = Arc::new(AtomicBool::new(false));
    let drops = Arc::new(AtomicUsize::new(0));
    let earlier = OrdinaryHostOwner(Some(Arc::new(OrdinaryHostNode {
        previous: None,
        _incoming: HostPreparationAuthority::retain(Earlier {
            allocation: allocation.clone(),
            observed: observed.clone(),
            drops: drops.clone(),
        }),
        allocation_retired: Some(allocation),
    })));
    let latest = OrdinaryHostOwner(Some(Arc::new(OrdinaryHostNode {
        previous: Some(earlier),
        _incoming: HostPreparationAuthority::retain(Panics),
        allocation_retired: None,
    })));
    let result = catch_unwind(AssertUnwindSafe(|| drop(latest)));
    assert_eq!(result.unwrap_err().downcast_ref::<u32>(), Some(&73));
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    assert!(
        observed.load(Ordering::SeqCst),
        "the remaining node's allocation retired before its token during unwind"
    );
}
