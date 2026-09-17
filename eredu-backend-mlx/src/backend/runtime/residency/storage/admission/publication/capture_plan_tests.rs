use super::*;
use eredu_core::{
    cache::LayerCachePolicy, capture::*, Admission, DescriptionCompleteness,
    EstimationCompleteness, ExecutionWorkspaceEstimate, InferenceGeometry, InputTokenCount,
    LayerSchedule, ObservationCatalog, ObservationDtype, ObservationPoint, ObservationPosition,
    ObservationRequirement, ObservationSupport, ObservationSupportReport, ObservationSupportStatus,
    ObservationValueType, OutputDemand, SharedStorageAttachmentError, StateMemoryLayout,
    SymbolicDimension, TensorAxis, WorkspaceBound,
};
use eredu_runtime::working_memory::{
    HostSlotStorageKey, InferenceExecutionIdentity, WorkingMemoryFundingRun,
    WorkingMemoryReservation,
};
use std::{
    cell::Cell,
    num::NonZeroU8,
    panic::{catch_unwind, AssertUnwindSafe},
};

fn admitted() -> AdmittedCapturePlan {
    let point = ObservationPoint {
        path: eredu_core::MODEL_LOGITS_OBSERVATION_PATH.into(),
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
    let mut plan = CapturePlan {
        schema_version: 1,
        selections: transforms
            .into_iter()
            .enumerate()
            .map(|(index, transform)| CaptureSelection {
                id: format!("selection-{index}"),
                path: if matches!(transform, CaptureTransform::RoutedUnits) {
                    "experts.units".into()
                } else {
                    eredu_core::MODEL_LOGITS_OBSERVATION_PATH.into()
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
    };
    plan.selections.reserve_exact(17);
    for selection in &mut plan.selections {
        selection.id.reserve_exact(83);
        selection.path.reserve_exact(83);
        selection.slices.reserve_exact(7);
        for slice in &mut selection.slices {
            slice.axis.reserve_exact(83);
        }
        match &mut selection.transform {
            CaptureTransform::Histogram { edges } => edges.reserve_exact(17),
            CaptureTransform::TokenScores { token_ids } => token_ids.reserve_exact(19),
            _ => {}
        }
    }
    plan.admit(
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

// Host-only transfer fixture: no model, native operation or compiler peak claim.
fn funding(
    pool: &WorkingMemoryPool,
    bytes: u64,
) -> (WorkingMemoryReservation, WorkingMemoryFundingRun) {
    let geometry = InferenceGeometry {
        batch_size: 1,
        cached_positions: 0,
        input_positions: 1,
        max_output_tokens: 0,
        prefill_chunk_positions: 1,
        output: OutputDemand::StateOnly,
    };
    let layout = StateMemoryLayout::new(
        LayerSchedule::new(1, vec![LayerCachePolicy::NoState]).unwrap(),
        vec![0],
        1,
        1,
        EstimationCompleteness::Complete,
    )
    .unwrap();
    let zero =
        || WorkspaceBound::bounded(0, "host-only capture plan publication has no native work");
    let state = eredu_core::estimate_runtime_state(
        &layout,
        InputTokenCount::text(1),
        0,
        1,
        NonZeroU8::new(4).unwrap(),
    )
    .unwrap()
    .with_execution_workspace(ExecutionWorkspaceEstimate {
        geometry,
        activations: zero(),
        attention: zero(),
        vocabulary: zero(),
        state_update: zero(),
        materialization: zero(),
        retained: WorkspaceBound::bounded(bytes, "exact retained capture plan publication credit"),
    })
    .unwrap();
    pool.reserve_with_capacity(
        &InferenceExecutionIdentity::default(),
        &Admission {
            requested_positions: 1,
            state,
            incremental_required_bytes: bytes,
            available_memory_bytes: None,
        },
        bytes,
    )
    .unwrap()
    .into_funding()
    .unwrap()
}

fn plan() -> SharedCapturePlan {
    SharedCapturePlan::new(admitted())
}
fn inventory(plans: impl IntoIterator<Item = SharedCapturePlan>) -> RetainedStorage {
    let mut result = RetainedStorage::default();
    for plan in plans {
        result.include_capture_plan(plan).unwrap();
    }
    result
}
fn reclaim(pool: &WorkingMemoryPool, expected: u64) {
    crate::backend::ordinary_retirement::reclaim_all();
    assert_eq!(pool.used_bytes().unwrap(), expected);
}
fn cause<'a, T: std::error::Error + 'static>(
    mut error: &'a (dyn std::error::Error + 'static),
) -> Option<&'a T> {
    loop {
        if let Some(found) = error.downcast_ref::<T>() {
            return Some(found);
        }
        error = error.source()?;
    }
}
fn assert_payload(source: &SharedCapturePlan) {
    let selections = &source.admission().plan().selections;
    assert_eq!(selections.len(), 8);
    assert_eq!(selections[1].slices[0].axis, "width");
    assert!(matches!(&selections[4].transform,
        CaptureTransform::Histogram { edges } if edges == &[-1.0, 0.0, 1.0]));
    assert!(matches!(&selections[6].transform,
        CaptureTransform::TokenScores { token_ids } if token_ids == &[0, 3]));
}
thread_local! { static HOUSEKEEPING: Cell<usize> = const { Cell::new(0) }; }
fn housekeeping() {
    HOUSEKEEPING.with(|count| count.set(count.get() + 1));
}
struct Hook;
impl Hook {
    fn install() -> Self {
        safemlx::register_thread_runtime_housekeeping(housekeeping);
        HOUSEKEEPING.with(|count| count.set(0));
        Self
    }
}
impl Drop for Hook {
    fn drop(&mut self) {
        safemlx::unregister_thread_runtime_housekeeping(housekeeping);
    }
}

#[test]
fn capture_plan_inventory_preserves_nested_spare_capacity_and_is_cold() {
    let pool = WorkingMemoryPool::new(1 << 20, 0).unwrap();
    let loading = NativeMemoryOwner::acquire(&pool).unwrap();
    let source = plan();
    let alias = source.clone();
    let independent = SharedCapturePlan::new(source.admission().clone());
    assert_eq!(
        source.admission().identity(),
        independent.admission().identity()
    );
    assert_ne!(source.storage_identity(), independent.storage_identity());
    let selections = &source.admission().plan().selections;
    assert!(selections.capacity() > selections.len());
    assert!(selections[0].id.capacity() > selections[0].id.len());
    assert!(selections[1].slices.capacity() > selections[1].slices.len());
    assert!(selections[1].slices[0].axis.capacity() > selections[1].slices[0].axis.len());
    assert!(matches!(&selections[4].transform,
        CaptureTransform::Histogram { edges } if edges.capacity() > edges.len()));
    assert!(matches!(&selections[6].transform,
        CaptureTransform::TokenScores { token_ids } if token_ids.capacity() > token_ids.len()));
    let pointer = selections.as_ptr();
    let bytes = source.capacity_bytes().unwrap();
    let independent_bytes = independent.capacity_bytes().unwrap();
    assert!(bytes > independent_bytes);
    let hook = Hook::install();
    let mut storage = inventory([source.clone(), alias.clone()]);
    storage
        .merge(inventory([alias.clone(), independent.clone()]))
        .unwrap();
    assert_eq!(storage.capture_plans.len(), 2);
    assert_eq!(
        storage.byte_bound().unwrap(),
        Some(bytes + independent_bytes)
    );
    let entries = storage.storage_entries().unwrap();
    assert_eq!(entries.len(), 2);
    assert!(entries
        .iter()
        .all(|(key, _)| matches!(key, StorageIdentity::CapturePlan(_))
            && key.host_slot_identity().is_none()));
    let registered = inventory([source.clone()]).register(&pool).unwrap();
    let publication = storage.publish_unquoted(&loading).unwrap();
    let repeated = inventory([source.clone()])
        .publish_unquoted(&loading)
        .unwrap();
    let pinned = inventory([source.clone()]).pin_registered(&pool).unwrap();
    assert_eq!(pinned.bytes(), bytes);
    assert_eq!(HOUSEKEEPING.with(Cell::get), 0);
    drop(hook);
    drop((registered, pinned, repeated, source, loading));
    reclaim(&pool, bytes + independent_bytes);
    assert_eq!(alias.admission().plan().selections.as_ptr(), pointer);
    assert_payload(&alias);
    drop(alias);
    reclaim(&pool, independent_bytes);
    drop(independent);
    // An idle publication record contains no cycle back to either shared plan.
    reclaim(&pool, 0);
    drop(publication);
}

#[test]
fn capture_plan_attachments_retain_earlier_aliases_once_in_each_domain() {
    let a = WorkingMemoryPool::new(1 << 20, 0).unwrap();
    let b = WorkingMemoryPool::new(1 << 20, 0).unwrap();
    let loading_a = NativeMemoryOwner::acquire(&a).unwrap();
    let loading_b = NativeMemoryOwner::acquire(&b).unwrap();
    let source = plan();
    let alias = source.clone();
    let key = source.storage_identity().clone();
    let bytes = source.capacity_bytes().unwrap();
    let pa = inventory([source.clone()])
        .publish_unquoted(&loading_a)
        .unwrap();
    let pb = inventory([source.clone()])
        .publish_unquoted(&loading_b)
        .unwrap();
    let repeat = inventory([alias.clone()])
        .publish_unquoted(&loading_a)
        .unwrap();
    assert!(!source
        .try_attach(
            a.shared_storage_domain(),
            || -> Result<Box<dyn Send + Sync>, WorkingMemoryError> {
                panic!("published domain must not acquire twice")
            }
        )
        .unwrap());
    drop((pa, pb, repeat, loading_a, loading_b, source));
    reclaim(&a, bytes);
    reclaim(&b, bytes);
    assert_eq!(a.unquoted_owner_count().unwrap(), 0);
    assert_eq!(b.unquoted_owner_count().unwrap(), 0);
    assert_payload(&alias);
    drop(alias);
    reclaim(&a, 0);
    reclaim(&b, 0);
    assert_eq!(a.peak_bytes().unwrap(), bytes);
    assert_eq!(b.peak_bytes().unwrap(), bytes);
    drop(key);
}

#[test]
fn capture_plan_exact_registration_rejects_one_short_and_unknown_before_accounting() {
    let source_pool = WorkingMemoryPool::new(1 << 20, 0).unwrap();
    let source_authority = NativeMemoryOwner::acquire(&source_pool).unwrap();
    let source = plan();
    let bytes = source.capacity_bytes().unwrap();
    let short = WorkingMemoryPool::new(bytes - 1, 0).unwrap();
    let exact = WorkingMemoryPool::new(bytes, 0).unwrap();
    let short_owner = NativeMemoryOwner::acquire(&short).unwrap();
    let exact_owner = NativeMemoryOwner::acquire(&exact).unwrap();
    let error = inventory([source.clone()])
        .publish_unquoted(&short_owner)
        .unwrap_err();
    assert_eq!(
        cause::<WorkingMemoryError>(&error),
        Some(&WorkingMemoryError::BudgetExceeded {
            required_bytes: bytes,
            available_bytes: bytes - 1,
        })
    );
    assert_eq!(
        (short.used_bytes().unwrap(), short.peak_bytes().unwrap()),
        (0, 0)
    );
    let mut unknown = inventory([source.clone()]);
    unknown.mark_incomplete();
    let mut merged = inventory([source.clone()]);
    merged.merge(unknown).unwrap();
    assert_eq!(merged.byte_bound().unwrap(), None);
    let error = merged.publish_unquoted(&exact_owner).unwrap_err();
    assert_eq!(
        cause::<WorkingMemoryError>(&error),
        Some(&WorkingMemoryError::UnknownBound)
    );
    assert_eq!(
        (exact.used_bytes().unwrap(), exact.peak_bytes().unwrap()),
        (0, 0)
    );
    let publication = inventory([source.clone()])
        .publish_unquoted(&exact_owner)
        .unwrap();
    assert_eq!(exact.used_bytes().unwrap(), bytes);
    assert_payload(&source);
    drop((
        publication,
        source_authority,
        exact_owner,
        short_owner,
        source,
    ));
    reclaim(&exact, 0);
    reclaim(&short, 0);
}

#[test]
fn funded_capture_plan_exact_credit_follows_aliases_after_request_retirement() {
    let source_pool = WorkingMemoryPool::new(1 << 20, 0).unwrap();
    let loading = NativeMemoryOwner::acquire(&source_pool).unwrap();
    let source = plan();
    let bytes = source.capacity_bytes().unwrap();
    let alias = source.clone();
    let key = source.storage_identity().clone();
    let pool = WorkingMemoryPool::new(bytes, 0).unwrap();
    let (reservation, run) = funding(&pool, bytes);
    let scope = run.scope().unwrap();
    let publication = inventory([source.clone()]).publish_funded(&scope).unwrap();
    let duplicate = inventory([alias.clone()]).publish_funded(&scope).unwrap();
    assert_eq!(pool.used_bytes().unwrap(), bytes);
    assert_eq!(
        scope.adopt_storage_individually([(37_u32, 1)]).unwrap_err(),
        WorkingMemoryError::BudgetExceeded {
            required_bytes: 1,
            available_bytes: 0
        }
    );
    scope.certify().unwrap();
    drop((reservation, run, publication, duplicate, source, loading));
    reclaim(&pool, bytes);
    assert_payload(&alias);
    assert_eq!(alias.storage_identity(), &key);
    drop(alias);
    reclaim(&pool, 0);
    assert_eq!(pool.peak_bytes().unwrap(), bytes);
    drop(key);
}

#[test]
fn funded_capture_plan_short_credit_rejects_without_attachment_or_lost_funding() {
    let source_pool = WorkingMemoryPool::new(1 << 20, 0).unwrap();
    let loading = NativeMemoryOwner::acquire(&source_pool).unwrap();
    let source = plan();
    let bytes = source.capacity_bytes().unwrap();
    let pool = WorkingMemoryPool::new(bytes - 1, 0).unwrap();
    let (reservation, run) = funding(&pool, bytes - 1);
    let scope = run.scope().unwrap();
    let error = inventory([source.clone()])
        .publish_funded(&scope)
        .unwrap_err();
    assert_eq!(
        cause::<WorkingMemoryError>(&error),
        Some(&WorkingMemoryError::BudgetExceeded {
            required_bytes: bytes,
            available_bytes: bytes - 1,
        })
    );
    // Atomic rejection leaves all credit usable. There is no submitted native work.
    let credit = scope
        .adopt_storage_individually([(37_u32, bytes - 1)])
        .unwrap();
    assert_payload(&source);
    scope.certify().unwrap();
    drop((credit, reservation, run));
    reclaim(&pool, 0);
    let exact = WorkingMemoryPool::new(bytes, 0).unwrap();
    let owner = NativeMemoryOwner::acquire(&exact).unwrap();
    let publication = inventory([source.clone()])
        .publish_unquoted(&owner)
        .unwrap();
    drop((publication, owner, source, loading));
    reclaim(&exact, 0);
}

fn ordered_plans() -> (SharedCapturePlan, SharedCapturePlan) {
    let first = plan();
    let second = plan();
    if first.storage_identity() < second.storage_identity() {
        (first, second)
    } else {
        (second, first)
    }
}
fn poison(source: &SharedCapturePlan) {
    assert!(catch_unwind(AssertUnwindSafe(|| {
        let _ = source.try_attach(
            &eredu_core::SharedStorageDomain::default(),
            || -> Result<Box<dyn Send + Sync>, WorkingMemoryError> {
                panic!("test custody failure")
            },
        );
    }))
    .is_err());
}

#[test]
fn later_capture_plan_attachment_failure_preserves_first_charge_and_typed_cause() {
    let pool = WorkingMemoryPool::new(1 << 20, 0).unwrap();
    let loading = NativeMemoryOwner::acquire(&pool).unwrap();
    let (first, failed) = ordered_plans();
    let first_bytes = first.capacity_bytes().unwrap();
    let total = first_bytes + failed.capacity_bytes().unwrap();
    poison(&failed);
    let error = inventory([first.clone(), failed.clone()])
        .publish_unquoted(&loading)
        .unwrap_err();
    assert!(matches!(
        cause::<SharedStorageAttachmentError<WorkingMemoryError>>(&error),
        Some(SharedStorageAttachmentError::Poisoned)
    ));
    reclaim(&pool, first_bytes);
    assert_eq!(pool.peak_bytes().unwrap(), total);
    assert_eq!(pool.unquoted_owner_count().unwrap(), 1);
    assert_payload(&first);
    assert_payload(&failed);
    drop((error, failed, loading));
    reclaim(&pool, first_bytes);
    drop(first);
    reclaim(&pool, 0);
}

#[test]
fn failed_funded_capture_plan_attachment_retains_uncertified_envelope() {
    let source_pool = WorkingMemoryPool::new(1 << 20, 0).unwrap();
    let loading = NativeMemoryOwner::acquire(&source_pool).unwrap();
    let (first, failed) = ordered_plans();
    let total = first.capacity_bytes().unwrap() + failed.capacity_bytes().unwrap();
    let pool = WorkingMemoryPool::new(total, 0).unwrap();
    let (reservation, run) = funding(&pool, total);
    let scope = run.scope().unwrap();
    poison(&failed);
    let error = inventory([first.clone(), failed.clone()])
        .publish_funded(&scope)
        .unwrap_err();
    assert!(matches!(
        cause::<SharedStorageAttachmentError<WorkingMemoryError>>(&error),
        Some(SharedStorageAttachmentError::Poisoned)
    ));
    reclaim(&pool, total);
    assert_payload(&first);
    assert_payload(&failed);
    // No certification after partial attachment and no refund at final retirement.
    drop(scope);
    drop((error, run, reservation, first, failed, loading));
    reclaim(&pool, total);
    assert_eq!(pool.peak_bytes().unwrap(), total);
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
}

#[test]
fn capture_none_still_publishes_its_actual_retained_admission_payload() {
    let pool = WorkingMemoryPool::new(1 << 20, 0).unwrap();
    let loading = NativeMemoryOwner::acquire(&pool).unwrap();
    let capabilities = CaptureCapabilities::default();
    let admitted = CapturePlan::none()
        .admit(
            &ObservationCatalog {
                schema_version: 1,
                points: vec![],
                completeness: DescriptionCompleteness::Complete,
            },
            &ObservationSupportReport {
                schema_version: 1,
                capture: capabilities.clone(),
                points: vec![],
            },
            &capabilities,
            CaptureRequestShape {
                batch: 1,
                prompt_tokens: 3,
                max_predictions: 1,
            },
        )
        .unwrap();
    assert!(admitted.is_empty());
    let source = SharedCapturePlan::new(admitted);
    let bytes = source.capacity_bytes().unwrap();
    assert!(bytes >= std::mem::size_of::<AdmittedCapturePlan>() as u64 + 64);
    let publication = inventory([source.clone()])
        .publish_unquoted(&loading)
        .unwrap();
    reclaim(&pool, bytes);
    drop((source, loading));
    // Publication itself must not keep even the empty semantic plan alive.
    reclaim(&pool, 0);
    drop(publication);
}
