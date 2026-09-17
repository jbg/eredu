//! Actual selected executables retain and publish their original path source.
use super::*;
use eredu_runtime::{
    working_memory::WorkingMemoryPool, SharedHostMetadata, SharedLayeredObservationPaths,
};

fn reclaim() {
    crate::backend::nn::shared::MlxNeuralBackend::reclaim_retired_resources();
    safemlx::reclaim_allocation_owners();
}
fn settle(pool: &WorkingMemoryPool, bytes: u64) {
    crate::backend::submission_recovery::wait_for_retirement(|| {
        reclaim();
        pool.used_bytes().unwrap() == bytes && pool.unquoted_owner_count().unwrap() == 0
    });
}
fn paths(runtime: &ModelRuntime<MlxBackend<'_>>) -> SharedLayeredObservationPaths {
    let executable = runtime.session().payload.model.erased();
    let source = executable.shared_observation_paths().unwrap().clone();
    let inventory = executable.retained_idle_auxiliary_storage().unwrap();
    let actual = inventory
        .metadata_sources()
        .filter_map(|owner| match owner {
            SharedHostMetadata::ObservationPaths(paths) => Some(paths),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(actual.len(), 1);
    assert!(source.same_storage(actual[0]));
    assert_eq!(source.identity(), actual[0].identity());
    source
}
fn strings(source: &SharedLayeredObservationPaths) -> Vec<String> {
    let mut result = Vec::new();
    for group in 0..source.group_count() {
        if let Some(path) = source.group_input(group) {
            result.push(path.to_owned());
        }
        for unit in 0..source.unit_count(group).unwrap() {
            let (input, output) = source.unit_paths(group, unit).unwrap();
            assert!(!input.is_empty() && !output.is_empty());
            result.extend([input.to_owned(), output.to_owned()]);
        }
        if let Some(path) = source.group_output(group) {
            result.push(path.to_owned());
        }
    }
    assert!(!result.is_empty());
    result
}

#[test]
fn loaded_native_paths_share_prepublication_aliases_and_outlive_prepared_model_and_session() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let backend = MlxBackend::new(&stream, &stream).with_memory_pool(pool.clone());
    let artifact = crate::composition::mlx::replicated_text::tests::tiny_artifact("llama", true);
    let mut model =
        eredu_core::load_model(&backend, artifact.path(), crate::MlxLoadRequest::default())
            .unwrap();
    assert!(model.memory_owner().is_some());
    let original = model
        .executable_mut()
        .erased()
        .shared_observation_paths()
        .unwrap()
        .clone();
    let old_alias = original.clone();
    let expected = strings(&original);
    let retained = original.capacity_bytes().unwrap();
    assert!(retained > expected.iter().map(String::len).sum::<usize>() as u64);
    let runtime = ModelRuntime::from_prepared(backend, model).unwrap();
    crate::backend::submission_recovery::wait_for_retirement(|| {
        reclaim();
        pool.unquoted_owner_count().unwrap() == 0
    });
    assert!(runtime.session().payload.model.has_published_idle_storage());
    assert!(runtime.session().payload._memory_owner.is_none());
    let published = paths(&runtime);
    assert!(original.same_storage(&published));
    assert_eq!(strings(&published), expected);
    let mut inventory = runtime
        .session()
        .payload
        .model
        .erased()
        .retained_idle_auxiliary_storage()
        .unwrap();
    let bytes = inventory.byte_bound().unwrap().unwrap();
    inventory
        .include_metadata(SharedHostMetadata::ObservationPaths(old_alias.clone()))
        .unwrap();
    assert_eq!(inventory.byte_bound().unwrap(), Some(bytes));
    drop((inventory, published, original, runtime));
    stream.synchronize().unwrap();
    settle(&pool, retained);
    assert_eq!(strings(&old_alias), expected);
    drop(old_alias);
    settle(&pool, 0);
}

#[cfg(all(feature = "metal", target_vendor = "apple", not(feature = "cuda")))]
mod metal {
    use super::*;
    use crate::composition::mlx::session::model_session::{
        disk_layerwise_tests as disk, host_layerwise_tests as host,
    };
    use crate::tests::support::path_instrumentation;

    fn quiescent(runtime: &ModelRuntime<MlxBackend<'_>>) {
        crate::backend::submission_recovery::wait_for_retirement(|| {
            reclaim();
            runtime.session().payload.active_owner_count() == 1
        });
        runtime.session().ensure_no_submission_in_flight().unwrap();
    }

    #[test]
    fn resident_host_and_disk_paths_keep_identity_through_ordinary_and_controlled_execution() {
        let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
        let mut expected_tokens = None;
        for route in 0..3 {
            for controlled in [false, true] {
                let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
                let (mut runtime, _artifact) = match route {
                    0 => host::runtime(&stream, &pool, None),
                    1 => host::runtime(&stream, &pool, Some(1)),
                    _ => disk::load_runtime(&stream, &pool, true),
                };
                let source = paths(&runtime);
                let source_strings = strings(&source);
                let retained = source.capacity_bytes().unwrap();
                let input = disk::tokens();
                let prior_peak = pool.peak_bytes().unwrap();
                let (capacity, _) = disk::exact_capacity(&runtime, &pool, &input, 0.0, 1);
                let output = disk::outputs(
                    &mut runtime,
                    input,
                    disk::config(0.0, 1, capacity),
                    controlled,
                );
                let tokens = disk::token_ids(&output);
                if let Some(expected) = &expected_tokens {
                    assert_eq!(&tokens, expected);
                } else {
                    expected_tokens = Some(tokens);
                }
                drop(output);
                quiescent(&runtime);
                let before = path_instrumentation::snapshot();
                let inventory_before = pool.used_bytes().unwrap();
                let alias = paths(&runtime);
                assert!(source.same_storage(&alias));
                assert_eq!(strings(&alias), source_strings);
                assert_eq!(path_instrumentation::snapshot(), before);
                assert_eq!(pool.used_bytes().unwrap(), inventory_before);
                assert!(pool.peak_bytes().unwrap() <= prior_peak.max(capacity));
                drop((alias, runtime));
                stream.synchronize().unwrap();
                settle(&pool, retained);
                drop(source);
                settle(&pool, 0);
            }
        }
    }

    #[test]
    fn actual_composite_executables_have_independent_equal_path_sources_and_charges() {
        let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
        let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
        let artifact = host::family_artifact("gemma4");
        let source_stream =
            Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let load = || {
            let backend = MlxBackend::new(&stream, &source_stream).with_memory_pool(pool.clone());
            let model =
                eredu_core::load_model(&backend, artifact.path(), crate::MlxLoadRequest::default())
                    .unwrap();
            ModelRuntime::from_prepared(backend, model).unwrap()
        };
        let a = load();
        let b = load();
        let first = paths(&a);
        let second = paths(&b);
        assert!(!first.same_storage(&second));
        assert_ne!(first.identity(), second.identity());
        assert_eq!(strings(&first), strings(&second));
        let a_bytes = first.capacity_bytes().unwrap();
        let b_bytes = second.capacity_bytes().unwrap();
        assert!(a_bytes > 0 && b_bytes > 0);
        drop((a, b));
        stream.synchronize().unwrap();
        settle(&pool, a_bytes + b_bytes);
        drop(first);
        settle(&pool, b_bytes);
        drop(second);
        settle(&pool, 0);
    }
}

#[cfg(all(feature = "metal", target_vendor = "apple", not(feature = "cuda")))]
mod blueprint;

#[cfg(all(feature = "metal", target_vendor = "apple", not(feature = "cuda")))]
mod prepared;

#[cfg(all(feature = "metal", target_vendor = "apple", not(feature = "cuda")))]
mod capture_quote;

#[cfg(all(feature = "metal", target_vendor = "apple", not(feature = "cuda")))]
mod validation;
