use super::super::tests::{settle, source};
use super::*;

#[test]
fn native_source_exact_short_and_foreign_admission_preserve_real_original_owners() {
    COMPILES.set(0);
    let materializer = MlxPreparedInputMaterializer::prepare().unwrap();
    let sample_pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
    let sample = source(&sample_pool, 16);
    let i = sample.original_bytes();
    let b = materializer
        .plan(&sample)
        .unwrap()
        .required_bytes()
        .unwrap();
    drop(sample);
    drop(sample_pool);
    let short = crate::memory_fixture::ledger(i + b - 1, 0).unwrap();
    let input = source(&short, 16);
    let error = materializer
        .plan(&input)
        .unwrap()
        .materialize(&short)
        .unwrap_err();
    assert!(
        matches!(error.accounting_failure(),Some(WorkingMemoryError::Domain(eredu_core::MemoryDomainError::BudgetExceeded { requested_bytes: required_bytes, limit_bytes, existing_bytes, .. })) if *required_bytes==b&&(*limit_bytes - *existing_bytes)==b-1)
    );
    assert_eq!(error.retained_bytes(), 0);
    assert_eq!(COMPILES.get(), 0);
    assert_eq!(short.fixture_host_charge().unwrap(), i);
    drop(error);
    drop(input);
    let exact = crate::memory_fixture::ledger(i + b, 0).unwrap();
    let input = source(&exact, 16);
    let native = materializer
        .plan(&input)
        .unwrap()
        .materialize(&exact)
        .unwrap();
    assert_eq!(native.original_bytes(), b);
    assert_eq!(native.slot_count(), input.slot_count());
    assert_eq!(exact.fixture_host_charge().unwrap(), i + b);
    let other = crate::memory_fixture::ledger(i + b, 0).unwrap();
    let error = materializer
        .plan(&input)
        .unwrap()
        .materialize(&other)
        .unwrap_err();
    assert_eq!(
        error.accounting_failure(),
        Some(&WorkingMemoryError::IdentityMismatch)
    );
    assert_eq!(other.fixture_host_charge().unwrap(), 0);
    drop(error);
    drop(native);
    settle(&exact, 0, i);
    drop(input);
    assert_eq!(exact.fixture_host_charge().unwrap(), 0);
}

#[test]
fn native_source_real_vector_reserve_failure_holds_arena_until_error_and_reclaimer_retire() {
    let materializer = MlxPreparedInputMaterializer::prepare().unwrap();
    let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
    let input = source(&pool, 16);
    let plan = materializer.plan(&input).unwrap();
    let required = plan.required_bytes().unwrap();
    FAIL_VECTOR_RESERVE.set(true);
    let error = plan.materialize(&pool).unwrap_err();
    assert!(matches!(
        error.0.compiler_failure(),
        Some(MlxNativeInputCause::Slots(_))
    ));
    assert_eq!(error.retained_bytes(), required);
    assert_eq!(
        pool.fixture_host_charge().unwrap(),
        input.original_bytes() + required
    );
    drop(error);
    settle(&pool, 0, input.original_bytes());
    let lease = pool.acquire_unquoted().unwrap();
    let before = COMPILES.get();
    let error = materializer
        .plan(&input)
        .unwrap()
        .materialize(&pool)
        .unwrap_err();
    assert_eq!(
        error.accounting_failure(),
        Some(&WorkingMemoryError::UnknownBound)
    );
    assert_eq!(COMPILES.get(), before);
    assert_eq!(pool.unquoted_owner_count().unwrap(), 1);
    drop(error);
    drop(lease);
    drop(input);
    assert_eq!(pool.fixture_host_charge().unwrap(), 0);
}

#[test]
fn native_source_equal_content_is_not_packet_authority_and_wrong_slot_does_no_clone() {
    let materializer = MlxPreparedInputMaterializer::prepare().unwrap();
    let root = tempfile::tempdir().unwrap();
    crate::tests::distributed_pipeline_ring::write_qwen3_vl_component_fixture(
        root.path(),
        false,
        false,
    );
    let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
    let first = source(&pool, 64);
    let equal = source(&pool, 64);
    assert_eq!(first.content_digest(), equal.content_digest());
    assert!(!first.same_source(&equal));
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let backend = MlxBackend::new(&stream, &stream).with_memory_ledger(pool.clone());
    let config = super::super::tests::cold_config(&backend, root.path(), 0);
    let semantics = config
        .prepared_sources()
        .plan_original_media_semantics(&first)
        .unwrap()
        .compile(&pool)
        .unwrap();
    let native = materializer
        .plan(&equal)
        .unwrap()
        .materialize(&pool)
        .unwrap();
    let held = native.original_bytes();
    let model = backend.prepare_model_borrowed(&config).unwrap();
    let runtime = ModelRuntime::from_prepared(backend, model).unwrap();
    CLONES.set(0);
    let error =
        MlxModelInput::from_original_native_input_with_semantics(&runtime, native, semantics)
            .unwrap_err();
    let MlxHostInputUploadError::Native {
        native_input,
        semantics,
        cause,
    } = error
    else {
        panic!("owning source rejection")
    };
    assert_eq!(cause, WorkingMemoryError::IdentityMismatch);
    assert!(semantics.is_some());
    assert_eq!(native_input.original_bytes(), held);
    assert_eq!(CLONES.get(), 0);
    let ordinary = NativeMemoryOwner::acquire_typed(&pool).unwrap();
    let wrong = first.slot(0).unwrap();
    assert!(native_input
        .clone_slot(0, wrong.values, wrong.shape)
        .is_err());
    let second = equal.slot(1).unwrap();
    assert!(native_input
        .clone_slot(0, second.values, second.shape)
        .is_err());
    assert_eq!(CLONES.get(), 0);
    drop(ordinary);
}

#[test]
fn native_source_all_slots_have_independent_completed_values_and_aliases_keep_original_charge() {
    let materializer = MlxPreparedInputMaterializer::prepare().unwrap();
    let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
    let input = source(&pool, 16);
    let native = materializer
        .plan(&input)
        .unwrap()
        .materialize(&pool)
        .unwrap();
    let original = pool.fixture_host_charge().unwrap();
    let b = native.original_bytes();
    let owner = NativeMemoryOwner::acquire_typed(&pool).unwrap(); // C wrappers are ordinary
    let mut aliases = Vec::new();
    let mut identities = std::collections::BTreeSet::new();
    for index in 0..input.slot_count() {
        let slot = input.slot(index).unwrap();
        let value = native.clone_slot(index, slot.values, slot.shape).unwrap();
        owner.retain_array(&value).unwrap();
        let expected = match slot.values {
            HostTensorValues::U32(v) => v.iter().map(|v| f64::from(*v)).collect::<Vec<_>>(),
            HostTensorValues::I32(v) => v.iter().map(|v| f64::from(*v)).collect(),
            HostTensorValues::F32(v) => v.iter().map(|v| f64::from(*v)).collect(),
            _ => unreachable!(),
        };
        assert_eq!(super::super::tests::array_values(&value), expected);
        let info = native.0.storage().leaves[index].allocation_info().unwrap();
        assert!(identities.insert(info.identity()));
        assert!(info.bytes() > 0);
        aliases.push(value);
    }
    drop(native);
    drop(input);
    assert_eq!(pool.fixture_host_charge().unwrap(), b);
    assert!(b < original);
    assert!(pool.unquoted_owner_count().unwrap() > 0);
    drop(owner);
    assert!(pool.unquoted_owner_count().unwrap() > 0);
    while aliases.len() > 1 {
        aliases.pop();
        safemlx::reclaim_allocation_owners();
        assert_eq!(pool.fixture_host_charge().unwrap(), b);
    }
    drop(aliases);
    settle(&pool, 0, 0);
}

#[test]
fn original_native_qwen_vl_matches_full_reference_in_three_residencies_and_cached_decodes() {
    if !crate::composition::mlx::session::model_session::original_host_input::tests::admitted::enter(
    ) {
        return;
    }
    let root = tempfile::tempdir().unwrap();
    crate::tests::distributed_pipeline_ring::write_qwen3_vl_component_fixture(
        root.path(),
        false,
        false,
    );
    super::super::tests::same_mode(root.path(), 64, 3);
}
#[test]
fn original_native_conditional_qwen_matches_full_reference_in_three_residencies_and_cached_decodes()
{
    if !crate::composition::mlx::session::model_session::original_host_input::tests::admitted::enter(
    ) {
        return;
    }
    let root = tempfile::tempdir().unwrap();
    crate::tests::distributed_pipeline_ring::write_qwen35_conditional_component_fixture(
        root.path(),
        false,
    );
    super::super::tests::same_mode(root.path(), 16, 3);
}
#[test]
fn original_native_sources_enter_the_same_prepared_iterator_and_manual_core_driver() {
    if !crate::composition::mlx::session::model_session::original_host_input::tests::admitted::enter(
    ) {
        return;
    }
    super::super::tests::core_driver_family(2, false);
    super::super::tests::core_driver_family(2, true);
}

#[cfg(test)]
#[allow(unused_imports)]
use crate::memory_fixture::LedgerFixture;
