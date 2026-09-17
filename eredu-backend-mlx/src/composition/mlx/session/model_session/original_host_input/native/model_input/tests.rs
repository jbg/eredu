use super::super::super::tests::{cold_config, core_driver_family, same_mode, settle, source};
use super::*;

#[test]
fn full_original_input_qwen_vl_matches_all_states_and_three_cached_decodes_across_residencies() {
    let root = tempfile::tempdir().unwrap();
    crate::tests::distributed_pipeline_ring::write_qwen3_vl_component_fixture(
        root.path(),
        false,
        false,
    );
    same_mode(root.path(), 64, 4);
}
#[test]
fn full_original_input_conditional_qwen_matches_all_states_and_three_cached_decodes_across_residencies(
) {
    let root = tempfile::tempdir().unwrap();
    crate::tests::distributed_pipeline_ring::write_qwen35_conditional_component_fixture(
        root.path(),
        false,
    );
    same_mode(root.path(), 16, 4);
}
#[test]
fn full_original_input_both_selected_families_use_same_iterator_and_manual_driver() {
    core_driver_family(3, false);
    core_driver_family(3, true);
}
#[test]
fn full_original_input_exact_short_foreign_and_partial_constructor_preserve_b() {
    let root = tempfile::tempdir().unwrap();
    crate::tests::distributed_pipeline_ring::write_qwen3_vl_component_fixture(
        root.path(),
        false,
        false,
    );
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let materializer = MlxPreparedInputMaterializer::prepare().unwrap();
    let measure = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let input = source(&measure, 64);
    let backend = MlxBackend::new(&stream, &stream).with_memory_pool(measure.clone());
    let config = cold_config(&backend, root.path(), 0);
    let a = config
        .prepared_sources()
        .plan_original_media_semantics(&input)
        .unwrap()
        .compile(&measure)
        .unwrap();
    let i = input.original_bytes();
    let abytes = a.original_bytes();
    let b = materializer
        .model_input_plan(&a)
        .unwrap()
        .required_bytes()
        .unwrap();
    drop(a);
    drop(input);
    drop(backend);
    drop(config);
    for short in [true, false] {
        let pool = WorkingMemoryPool::new(i + abytes + b - u64::from(short), 0).unwrap();
        let input = source(&pool, 64);
        let backend = MlxBackend::new(&stream, &stream).with_memory_pool(pool.clone());
        let config = cold_config(&backend, root.path(), 0);
        let a = config
            .prepared_sources()
            .plan_original_media_semantics(&input)
            .unwrap()
            .compile(&pool)
            .unwrap();
        COMPILES.set(0);
        let plan = materializer.model_input_plan(&a).unwrap();
        assert_eq!(plan.required_bytes().unwrap(), b);
        let result = plan.materialize(&pool);
        if short {
            let error = result.unwrap_err();
            assert_eq!(error.retained_bytes(), 0);
            assert_eq!(COMPILES.get(), 0);
            assert!(
                matches!(error.accounting_failure(),Some(WorkingMemoryError::BudgetExceeded{required_bytes,available_bytes}) if *required_bytes==b&&*available_bytes==b-1)
            );
        } else {
            let full = result.unwrap();
            assert_eq!(full.original_bytes(), b);
            assert_eq!(COMPILES.get(), 1);
            let cache = full.0.storage().cache().unwrap().clone();
            let parts = full.0.storage().parts().unwrap().clone();
            assert!(cache.original_source().unwrap().same_source(&input));
            let before = pool.used_bytes().unwrap();
            drop(full);
            assert_eq!(pool.used_bytes().unwrap(), before);
            assert_eq!(parts.as_ref().len(), input.parts().len());
            drop(parts);
            drop(cache);
            settle(&pool, 0, i + abytes);
            FAIL_VECTOR_RESERVE.set(true);
            let error = materializer
                .model_input_plan(&a)
                .unwrap()
                .materialize(&pool)
                .unwrap_err();
            assert_eq!(error.retained_bytes(), b);
            assert!(matches!(
                error.0.compiler_failure(),
                Some(PreparedModelInputSourceError::Native(
                    MlxNativeInputCause::Slots(_)
                ))
            ));
            drop(error);
            settle(&pool, 0, i + abytes);
        }
        let foreign = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
        let before = COMPILES.get();
        assert_eq!(
            materializer
                .model_input_plan(&a)
                .unwrap()
                .materialize(&foreign)
                .unwrap_err()
                .accounting_failure(),
            Some(&WorkingMemoryError::IdentityMismatch)
        );
        assert_eq!(COMPILES.get(), before);
        drop(a);
        drop(input);
        settle(&pool, 0, 0);
    }
}
#[test]
fn full_original_cache_ordinary_publication_authenticates_residence_without_duplicate_registration()
{
    use crate::backend::runtime::residency::storage::RetainedStorage;
    let root = tempfile::tempdir().unwrap();
    crate::tests::distributed_pipeline_ring::write_qwen3_vl_component_fixture(
        root.path(),
        false,
        false,
    );
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let input = source(&pool, 64);
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let backend = MlxBackend::new(&stream, &stream).with_memory_pool(pool.clone());
    let config = cold_config(&backend, root.path(), 0);
    let a = config
        .prepared_sources()
        .plan_original_media_semantics(&input)
        .unwrap()
        .compile(&pool)
        .unwrap();
    let full = MlxPreparedInputMaterializer::prepare()
        .unwrap()
        .model_input_plan(&a)
        .unwrap()
        .materialize(&pool)
        .unwrap();
    let cache = full.0.storage().cache().unwrap().clone();
    let held = pool.used_bytes().unwrap();
    let abytes = a.original_bytes();
    let inventory = || {
        let mut s = RetainedStorage::default();
        s.include_metadata(eredu_runtime::SharedHostMetadata::Input(cache.clone()))
            .unwrap();
        s
    };
    let owner = NativeMemoryOwner::acquire_typed(&pool).unwrap();
    owner
        .retain_metadata(&eredu_runtime::SharedHostMetadata::Input(cache.clone()))
        .unwrap();
    let publication = inventory().publish_unquoted(&owner).unwrap();
    assert_eq!(pool.used_bytes().unwrap(), held);
    assert!(inventory().register(&pool).is_err());
    let other = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let foreign = NativeMemoryOwner::acquire_typed(&other).unwrap();
    assert!(inventory().publish_unquoted(&foreign).is_err());
    assert_eq!(other.used_bytes().unwrap(), 0);
    drop(inventory);
    drop(cache);
    drop(full);
    drop(a);
    drop(input);
    assert_eq!(pool.used_bytes().unwrap(), held - abytes);
    drop(publication);
    drop(owner);
    crate::backend::ordinary_retirement::reclaim_all();
    settle(&pool, 0, 0);
}

fn error_source<'a, T: std::error::Error + 'static>(
    mut error: &'a (dyn std::error::Error + 'static),
) -> Option<&'a T> {
    loop {
        if let Some(value) = error.downcast_ref::<T>() {
            return Some(value);
        }
        error = error.source()?;
    }
}
#[test]
fn full_original_input_binding_rejects_substitution_busy_poison_and_stale_frontier_without_refund()
{
    use eredu_core::Completion;
    for case in 0..5 {
        let root = tempfile::tempdir().unwrap();
        crate::tests::distributed_pipeline_ring::write_qwen3_vl_component_fixture(
            root.path(),
            false,
            false,
        );
        let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
        let input = source(&pool, 64);
        let equal = source(&pool, 64);
        assert_eq!(input.content_digest(), equal.content_digest());
        assert!(!input.same_source(&equal));
        let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let backend = MlxBackend::new(&stream, &stream).with_memory_pool(pool.clone());
        let config = cold_config(&backend, root.path(), 0);
        let a = config
            .prepared_sources()
            .plan_original_media_semantics(&input)
            .unwrap()
            .compile(&pool)
            .unwrap();
        let full = MlxPreparedInputMaterializer::prepare()
            .unwrap()
            .model_input_plan(&a)
            .unwrap()
            .materialize(&pool)
            .unwrap();
        let b = full.original_bytes();
        let a = if case == 0 {
            drop(a);
            config
                .prepared_sources()
                .plan_original_media_semantics(&equal)
                .unwrap()
                .compile(&pool)
                .unwrap()
        } else {
            a
        };
        let model = backend.prepare_model_borrowed(&config).unwrap();
        let mut runtime = ModelRuntime::from_prepared(backend, model).unwrap();
        if case != 1 {
            runtime
                .session()
                .payload
                .model
                .erased()
                .prepare_completed_media_binding_fixture()
                .unwrap();
        }
        if case == 4 {
            let ordinary = MlxModelInput::from_original_host_input(&runtime, &input).unwrap();
            let output = runtime.prefill(ordinary).unwrap();
            output.completion.wait().unwrap();
            drop(output);
        }
        let lease = if case == 2 {
            Some(
                runtime
                    .session()
                    .authority
                    .borrow_mut()
                    .begin_submission()
                    .unwrap(),
            )
        } else {
            None
        };
        if case == 3 {
            runtime.session().poison.set(true);
        }
        let used = pool.used_bytes().unwrap();
        let owners = pool.unquoted_owner_count().unwrap();
        let error = full.bind(&runtime, a).unwrap_err();
        assert_eq!(error.retained_bytes(), b);
        assert_eq!(pool.used_bytes().unwrap(), used);
        assert_eq!(pool.unquoted_owner_count().unwrap(), owners);
        if case == 4 {
            assert!(
                error_source::<OriginalCompositeSemanticStorageError>(&error)
                    .unwrap()
                    .semantic_failure()
                    .is_some()
            );
        } else {
            assert!(error_source::<WorkingMemoryError>(&error).is_some());
        }
        drop(lease);
        if case == 3 {
            runtime.session().poison.set(false);
        }
        drop(error);
        drop(runtime);
        drop(input);
        drop(equal);
        settle(&pool, 0, 0);
    }
}
#[test]
fn full_original_input_frontier_error_representation_retains_actual_box_until_b_retirement() {
    // A corrupted negative native frontier is not reachable through the valid
    // fixture driver. Exercise its exact pinned conversion/error representation
    // under one completed B; the actual mechanism call is source-audited.
    let root = tempfile::tempdir().unwrap();
    crate::tests::distributed_pipeline_ring::write_qwen3_vl_component_fixture(
        root.path(),
        false,
        false,
    );
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let input = source(&pool, 64);
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let backend = MlxBackend::new(&stream, &stream).with_memory_pool(pool.clone());
    let config = cold_config(&backend, root.path(), 0);
    let a = config
        .prepared_sources()
        .plan_original_media_semantics(&input)
        .unwrap()
        .compile(&pool)
        .unwrap();
    let full = MlxPreparedInputMaterializer::prepare()
        .unwrap()
        .model_input_plan(&a)
        .unwrap()
        .materialize(&pool)
        .unwrap();
    let held = pool.used_bytes().unwrap();
    let conversion = u64::try_from(-1i32).unwrap_err();
    assert!(size_of::<std::num::TryFromIntError>() > 0);
    let error = MlxPreparedModelInputBindError {
        cause: CompletedMediaBindingError::mechanism(
            a,
            eredu_runtime::replicated_session::MediaSemanticBindingError::Mechanism(Error::Other(
                Box::new(conversion),
            )),
        ),
        input: full,
    };
    drop(input);
    assert_eq!(pool.used_bytes().unwrap(), held);
    assert!(error_source::<std::num::TryFromIntError>(&error).is_some());
    drop(error);
    settle(&pool, 0, 0);
}
