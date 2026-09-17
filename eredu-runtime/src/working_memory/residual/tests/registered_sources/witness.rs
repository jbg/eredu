//! A later installation keeps accounting pins without retaining its numerical quote.
use super::*;
use crate::working_memory::RegisteredInferenceSourceWitness;

#[test]
fn unobserved_extraction_returns_none_and_retires_residual_quote_custody() {
    let pool = WorkingMemoryPool::new(1000, 0).unwrap();
    let original = pool.register_storage([(1u32, 64)]).unwrap();
    let quote = replacement_quote(&pool, geometry(), 0).into_incremental();
    drop(original);
    assert_eq!(pool.used_bytes().unwrap(), 64);
    assert!(quote.into_registered_source_witness().is_none());
    assert_eq!(pool.used_bytes().unwrap(), 0);
    assert!(matches!(
        pool.pin_registered_storage([(1u32, 64)]),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
}

#[test]
fn witness_keeps_only_attached_sources_after_full_quote_retirement() {
    let pool = WorkingMemoryPool::new(1000, 0).unwrap();
    let original = pool.register_storage([(1u32, 64)]).unwrap();
    let source = pool.register_storage([(2u32, 24), (3, 0)]).unwrap();
    let quote = replacement_quote(&pool, geometry(), 0)
        .into_incremental()
        .with_registered_sources(source.clone())
        .unwrap()
        .with_registered_sources(source.clone())
        .unwrap();
    drop((original, source));
    assert_eq!(pool.used_bytes().unwrap(), 88);
    let witness: RegisteredInferenceSourceWitness = quote.into_registered_source_witness().unwrap();
    assert_eq!(pool.used_bytes().unwrap(), 24);
    assert!(matches!(
        pool.pin_registered_storage([(1u32, 64)]),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    let before = (used(&pool), pool.effective_capacity().unwrap());
    for _ in 0..3 {
        witness.validate(&pool).unwrap();
    }
    assert_eq!((used(&pool), pool.effective_capacity().unwrap()), before);
    assert_eq!(
        pool.pin_registered_storage([(2u32, 24), (3, 0)])
            .unwrap()
            .bytes(),
        24
    );
    drop(witness);
    assert_eq!(pool.used_bytes().unwrap(), 0);
    assert!(matches!(
        pool.pin_registered_storage([(3u32, 0)]),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
}

#[test]
fn healthy_settled_source_can_close_while_its_witness_remains_valid() {
    let pool = WorkingMemoryPool::new(1000, 0).unwrap();
    let execution = InferenceExecutionIdentity::default();
    let (metadata, run) = funded(&pool, &execution, 24, 1000);
    let scope = run.scope().unwrap();
    let source = scope
        .adopt_storage_individually([(2u32, 24)])
        .unwrap()
        .remove(&2)
        .unwrap();
    let original = pool.register_storage([(1u32, 64)]).unwrap();
    let quote = replacement_quote(&pool, geometry(), 0)
        .into_incremental()
        .with_registered_sources(source.clone())
        .unwrap();
    let witness = quote.into_registered_source_witness().unwrap();
    witness.validate(&pool).unwrap();
    // This fixture submits no native work. Its actual source registration has
    // already been published, so the original scope can explicitly certify.
    scope.certify().unwrap();
    run.close().unwrap();
    drop((metadata, original, source));
    assert_eq!(pool.used_bytes().unwrap(), 24);
    witness.validate(&pool).unwrap();
    drop(witness);
    assert_eq!(pool.used_bytes().unwrap(), 0);
    drop(pool.acquire_unquoted().unwrap());
}

#[test]
fn source_quarantine_after_extraction_rejects_later_install_validation() {
    for bytes in [0, 24] {
        let pool = WorkingMemoryPool::new(1000, 0).unwrap();
        let execution = InferenceExecutionIdentity::default();
        let (metadata, run) = funded(&pool, &execution, bytes, 1000);
        let scope = run.scope().unwrap();
        let source = scope
            .adopt_storage_individually([(2u32, bytes)])
            .unwrap()
            .remove(&2)
            .unwrap();
        let original = pool.register_storage([(1u32, 64)]).unwrap();
        let healthy = pool.register_storage([(4u32, 8)]).unwrap();
        let quote = replacement_quote(&pool, geometry(), 0)
            .into_incremental()
            .with_registered_sources(healthy.clone())
            .unwrap()
            .with_registered_sources(source.clone())
            .unwrap();
        let witness = quote.into_registered_source_witness().unwrap();
        drop((original, healthy, source));
        witness.validate(&pool).unwrap();
        // An independently owned source operation fails during Prompt/Sampling.
        // Neither successful extraction nor a healthy earlier origin masks it.
        drop(scope);
        let before = (used(&pool), pool.effective_capacity().unwrap());
        assert!(matches!(
            witness.validate(&pool),
            Err(WorkingMemoryError::ExecutionFenced)
        ));
        assert_eq!((used(&pool), pool.effective_capacity().unwrap()), before);
        assert_eq!(
            pool.pin_registered_storage([(2u32, bytes)])
                .unwrap()
                .bytes(),
            bytes
        );
        drop((witness, run, metadata));
        // No witness Drop can certify the failed source or refund its envelope.
        assert_eq!(pool.used_bytes().unwrap(), bytes);
        assert!(matches!(
            pool.acquire_unquoted(),
            Err(WorkingMemoryError::ReservedWorkActive)
        ));
    }
}

#[test]
fn empty_and_zero_byte_sources_keep_their_domain_without_a_new_identity() {
    for empty in [false, true] {
        let pool = WorkingMemoryPool::new(1000, 0).unwrap();
        let foreign = WorkingMemoryPool::new(1000, 0).unwrap();
        let original = pool.register_storage([(1u32, 64)]).unwrap();
        let source = if empty {
            pool.register_storage::<u64>([]).unwrap()
        } else {
            pool.register_storage([(9u64, 0)]).unwrap()
        };
        let quote = replacement_quote(&pool, geometry(), 0)
            .into_incremental()
            .with_registered_sources(source)
            .unwrap();
        drop(original);
        let witness = quote.into_registered_source_witness().unwrap();
        let before = (used(&pool), used(&foreign));
        assert!(matches!(
            witness.validate(&foreign),
            Err(WorkingMemoryError::IdentityMismatch)
        ));
        witness.validate(&pool.clone()).unwrap();
        assert_eq!((used(&pool), used(&foreign)), before);
        assert_eq!(pool.used_bytes().unwrap(), 0);
        if !empty {
            assert_eq!(pool.pin_registered_storage([(9u64, 0)]).unwrap().bytes(), 0);
        }
        drop(witness);
        assert!(matches!(
            pool.pin_registered_storage([(9u64, 0)]),
            Err(WorkingMemoryError::IdentityMismatch)
        ));
    }
}

#[test]
fn post_reservation_witness_does_not_retrofit_pins_into_existing_native_scopes() {
    let pool = WorkingMemoryPool::new(1000, 0).unwrap();
    let original = pool.register_storage([(1u32, 64)]).unwrap();
    let quote = replacement_quote(&pool, geometry(), 0).into_incremental();
    let (_, reservation, accepted) = reserve(
        &pool,
        &quote,
        200,
        &InferenceExecutionIdentity::default(),
        &[],
    )
    .unwrap();
    let (metadata, run) = reservation.into_funding().unwrap();
    let native = run.scope().unwrap();
    // Actual additional registration becomes available after original admission.
    let source = pool.register_storage([(2u32, 24)]).unwrap();
    let witness = accepted
        .with_registered_sources(source.clone())
        .unwrap()
        .into_registered_source_witness()
        .unwrap();
    drop((quote, original, source));
    assert_eq!(pool.used_bytes().unwrap(), 184);
    witness.validate(&pool).unwrap();
    drop(witness);
    // The old run/scope still pins its original 64-byte source and 96-byte
    // envelope, but never acquired ownership of the late source registration.
    assert_eq!(pool.used_bytes().unwrap(), 160);
    assert!(matches!(
        pool.pin_registered_storage([(2u32, 24)]),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    run.close().unwrap();
    native.certify().unwrap();
    drop(metadata);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
