use super::*;
use crate::input::host::{HostInputPart, HostTensorValues, HostTensorView, PreparedHostInputPlan};
use eredu_core::{InputModality, InputPayloadKind};
fn source(pool: &WorkingMemoryPool) -> OriginalPreparedHostInput {
    let ids = [1_u32, 2, 3];
    let parts = [HostInputPart {
        modality: InputModality::Text,
        kind: InputPayloadKind::TokenIds,
        payload: HostTensorView {
            shape: &[1, 3],
            values: HostTensorValues::U32(&ids),
        },
        metadata: &[],
        extents: &[],
    }];
    pool.compile_prepared_host_input(PreparedHostInputPlan::prepare(&parts).unwrap())
        .unwrap()
}
fn sizes() -> (u64, u64) {
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let source = source(&pool);
    let recipe = PreparedCompositeSemanticRecipe::new(
        &(),
        &source,
        3,
        CompositeSemanticCoordinates::ThreeAxesAndPrefix,
    )
    .unwrap();
    (source.original_bytes(), recipe.required_bytes().unwrap())
}
fn binding() -> MediaSessionBinding {
    MediaSessionBinding {
        execution: InferenceExecutionIdentity::default(),
        revision: super::super::InferenceRetention::new().revision().clone(),
        control: Arc::new(()),
        frontier: 0,
    }
}
#[test]
fn independent_media_binding_requires_the_actual_initialized_cache_revision() {
    let execution = InferenceExecutionIdentity::default();
    let retention = super::super::InferenceRetention::new();
    let foreign = super::super::InferenceRetention::new();
    let binding = MediaSessionBinding {
        execution: execution.clone(),
        revision: retention.revision().clone(),
        control: Arc::new(()),
        frontier: 0,
    };
    assert!(binding.matches_retained_state(&execution, &retention, 0));
    assert!(!binding.matches_retained_state(&execution, &foreign, 0));
    assert!(foreign.initialized_revision().is_none(), "comparison cannot mint an identity");
    let _ = foreign.revision();
    assert!(!binding.matches_retained_state(&execution, &foreign, 0));
    assert!(!binding.matches_retained_state(&execution, &retention, 1));
    assert!(!binding.matches_retained_state(&InferenceExecutionIdentity::default(), &retention, 0));
}
#[test]
fn exact_semantic_capacity_and_short_rejection_preserve_original_host_residence() {
    let (host, bytes) = sizes();
    for short in [true, false] {
        let pool = WorkingMemoryPool::new(host + bytes - u64::from(short), 0).unwrap();
        let input = source(&pool);
        let recipe = PreparedCompositeSemanticRecipe::new(
            &(),
            &input,
            3,
            CompositeSemanticCoordinates::ThreeAxesAndPrefix,
        )
        .unwrap();
        let result = recipe.allocate(&pool);
        if short {
            let error = result.err().expect("short rejected before reserve");
            assert!(
                matches!(error.accounting_failure(),Some(WorkingMemoryError::BudgetExceeded {required_bytes,available_bytes}) if *required_bytes==bytes && *available_bytes==bytes-1)
            );
            assert_eq!(error.retained_bytes(), 0);
            assert_eq!(pool.used_bytes().unwrap(), host);
        } else {
            let mut builder = result.ok().unwrap();
            assert!(matches!(
                pool.acquire_unquoted(),
                Err(WorkingMemoryError::ReservedWorkActive)
            ));
            builder.records_mut()[0].end = 3;
            builder
                .coordinates_mut()
                .copy_from_slice(&[0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 0, 0]);
            let output = builder.finish().ok().unwrap();
            drop(pool.acquire_unquoted().unwrap());
            drop(input);
            assert_eq!(output.records()[0].end, 3);
            assert_eq!(output.coordinates()[2], 2);
            assert_eq!(pool.used_bytes().unwrap(), host + bytes);
            drop(output);
            assert_eq!(pool.used_bytes().unwrap(), 0);
            continue;
        }
        drop(input);
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}
#[test]
fn both_actual_reserve_failures_retain_real_prefix_and_original_source() {
    let (host, bytes) = sizes();
    for at in 0..2 {
        let pool = WorkingMemoryPool::new(host + bytes, 0).unwrap();
        let input = source(&pool);
        let layout = PreparedCompositeSemanticLayout::for_source(
            &input,
            3,
            CompositeSemanticCoordinates::ThreeAxesAndPrefix,
        )
        .unwrap();
        let error = pool
            .prepare_composite_semantic_storage_inner(&input, &(), layout, Some(at))
            .err()
            .unwrap();
        assert_eq!(error.failed_buffer(), Some(at));
        assert_eq!(error.retained_bytes(), bytes);
        assert_eq!(error.retained_heap_bytes() == 0, at == 0);
        drop(input);
        assert_eq!(pool.used_bytes().unwrap(), host + bytes);
        drop(error);
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}
#[test]
fn foreign_source_and_ordinary_owner_reject_before_semantic_destinations() {
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let input = source(&pool);
    let foreign = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let make = || {
        PreparedCompositeSemanticRecipe::new(&(), &input, 3, CompositeSemanticCoordinates::Ordinary)
            .unwrap()
    };
    let error = make().allocate(&foreign).err().unwrap();
    assert_eq!(
        error.accounting_failure(),
        Some(&WorkingMemoryError::IdentityMismatch)
    );
    assert_eq!(error.retained_bytes(), 0);
    assert_eq!(foreign.used_bytes().unwrap(), 0);
    let ordinary = pool.acquire_unquoted().unwrap();
    let error = make().allocate(&pool).err().unwrap();
    assert_eq!(
        error.accounting_failure(),
        Some(&WorkingMemoryError::UnknownBound)
    );
    assert_eq!(error.retained_bytes(), 0);
    drop(ordinary);
}
#[test]
fn unwind_before_finish_refunds_after_all_fixed_destinations_retire() {
    let (host, bytes) = sizes();
    let pool = WorkingMemoryPool::new(host + bytes, 0).unwrap();
    let input = source(&pool);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let builder = PreparedCompositeSemanticRecipe::new(
            &(),
            &input,
            3,
            CompositeSemanticCoordinates::ThreeAxesAndPrefix,
        )
        .unwrap()
        .allocate(&pool)
        .ok()
        .unwrap();
        assert_eq!(builder.payload.buffers.coordinates.len(), 12);
        panic!("after actual fixed destinations");
    }));
    assert!(result.is_err());
    assert_eq!(pool.used_bytes().unwrap(), host);
    drop(input);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
#[test]
fn final_arc_settlement_poison_retains_both_buffers_and_never_false_refunds() {
    let (host, bytes) = sizes();
    let pool = WorkingMemoryPool::new(host + bytes, 0).unwrap();
    let input = source(&pool);
    let builder = PreparedCompositeSemanticRecipe::new(
        &(),
        &input,
        3,
        CompositeSemanticCoordinates::ThreeAxesAndPrefix,
    )
    .unwrap()
    .allocate(&pool)
    .ok()
    .unwrap();
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _guard = pool.0.usage.lock().unwrap();
        panic!("poison only after two real reserves");
    }));
    let error = builder.finish().err().unwrap();
    assert_eq!(
        error.accounting_failure(),
        Some(&WorkingMemoryError::Poisoned)
    );
    assert_eq!(error.retained_bytes(), bytes);
    assert!(error.retained_heap_bytes() > 0);
    drop(input);
    drop(error);
    let usage = pool.0.usage.lock().unwrap_err().into_inner();
    assert_eq!(usage.reserved, host + bytes);
}
#[test]
fn concurrent_final_bound_aliases_preserve_payload_and_custody_until_last_owner() {
    let (host, bytes) = sizes();
    let pool = WorkingMemoryPool::new(host + bytes, 0).unwrap();
    let input = source(&pool);
    let owner = PreparedCompositeSemanticRecipe::new(
        &(),
        &input,
        3,
        CompositeSemanticCoordinates::ThreeAxesAndPrefix,
    )
    .unwrap()
    .allocate(&pool)
    .ok()
    .unwrap()
    .finish()
    .ok()
    .unwrap()
    .bind(binding());
    let last = owner.clone();
    let aliases = (0..8).map(|_| owner.clone()).collect::<Vec<_>>();
    drop(input);
    drop(owner);
    std::thread::scope(|scope| {
        for alias in aliases {
            scope.spawn(move || drop(alias));
        }
    });
    assert_eq!(pool.used_bytes().unwrap(), host + bytes);
    assert_eq!(last.source().parts().len(), 1);
    drop(last);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn equal_numeric_frontier_never_substitutes_for_actual_revision_and_session_identity() {
    let first = binding();
    let alias = first.duplicate();
    assert!(first.matches(&alias));
    let mut other = first.duplicate();
    other.revision = super::super::InferenceRetention::new().revision().clone();
    assert_eq!(first.frontier(), other.frontier());
    assert!(first.same_origin(&other));
    assert!(!first.matches(&other));
    let mut other = first.duplicate();
    other.control = Arc::new(());
    assert!(!first.same_origin(&other));
    let mut other = first.duplicate();
    other.execution = InferenceExecutionIdentity::default();
    assert!(!first.same_origin(&other));
}
