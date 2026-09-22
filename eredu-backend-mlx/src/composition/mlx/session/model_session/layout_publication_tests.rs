use super::*;
use eredu_runtime::{working_memory::MemoryLedger, SharedHostMetadata, SharedStateLayout};

fn reclaim() {
    safemlx::memory::clear_cache();
    crate::backend::nn::shared::MlxNeuralBackend::reclaim_retired_resources();
    safemlx::reclaim_allocation_owners();
}

fn settle_bytes(pool: &MemoryLedger, bytes: u64) {
    let mut last = None;
    crate::backend::submission_recovery::wait_for_retirement(|| {
        reclaim();
        let actual = pool.fixture_host_charge().unwrap();
        if last != Some(actual) {
            eprintln!(
                "retirement target {bytes}, observed {actual}, unquoted {:?}",
                pool.unquoted_owner_count()
            );
            last = Some(actual);
        }
        actual == bytes
    });
}

fn fixture() -> (
    Stream,
    MemoryLedger,
    ModelRuntime<MlxBackend<'static>>,
    tempfile::TempDir,
) {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
    let backend = MlxBackend::new(&stream, &stream).with_memory_ledger(pool.clone());
    let root = crate::composition::mlx::replicated_text::tests::tiny_artifact("llama", true);
    let model =
        eredu_core::load_model(&backend, root.path(), crate::MlxLoadRequest::default()).unwrap();
    let runtime = ModelRuntime::from_prepared(backend, model).unwrap();
    crate::backend::submission_recovery::wait_for_retirement(|| {
        reclaim();
        pool.unquoted_owner_count().unwrap() == 0
    });
    (stream, pool, runtime, root)
}

fn actual_layout(runtime: &ModelRuntime<MlxBackend<'_>>) -> SharedStateLayout {
    let storage = runtime
        .session()
        .payload
        .model
        .erased()
        .retained_idle_auxiliary_storage()
        .unwrap();
    let layouts = storage
        .metadata_sources()
        .filter_map(|source| match source {
            SharedHostMetadata::Layout(layout) => Some(layout.clone()),
            SharedHostMetadata::Input(_) | SharedHostMetadata::ObservationPaths(_) => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(layouts.len(), 1);
    layouts.into_iter().next().unwrap()
}

#[test]
fn fresh_loaded_model_publishes_actual_layout_before_loading_owner_retires() {
    let (stream, pool, runtime, _root) = fixture();
    assert!(runtime.session().payload._memory_owner.is_none());
    assert!(runtime.session().payload.model.has_published_idle_storage());
    let layout = actual_layout(&runtime);
    let alias = actual_layout(&runtime);
    assert!(layout.same_storage(&alias));
    let layout_bytes = layout.capacity_bytes().unwrap();
    assert!(layout_bytes > 0);
    assert!(!layout.layout().layers().is_empty());

    let inventory = runtime.session().payload.retained_idle_storage().unwrap();
    assert_eq!(inventory.decoder_state_bytes().unwrap(), Some(0));
    let (mut nonstate, decoder) = inventory.into_parts();
    nonstate.merge(decoder).unwrap();
    let bytes = nonstate.byte_bound().unwrap().unwrap();
    assert!(
        bytes > layout_bytes,
        "the fixture also retains actual model weights"
    );
    // Source preparation metadata has independent paid ownership in addition
    // to the physical inventory. Repeated aliases do not change either charge.
    let charge = pool.snapshot().unwrap();
    assert!(pool.fixture_host_charge().unwrap() >= bytes);
    nonstate
        .include_metadata(SharedHostMetadata::Layout(alias.clone()))
        .unwrap();
    assert_eq!(nonstate.byte_bound().unwrap(), Some(bytes));
    assert_eq!(pool.snapshot().unwrap(), charge);
    drop((nonstate, alias, runtime));
    stream.synchronize().unwrap();
    settle_bytes(&pool, layout_bytes);
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
    drop(layout);
    settle_bytes(&pool, 0);
}

#[test]
fn loaded_state_tables_keep_exact_source_charges_through_last_metadata_alias() {
    use crate::composition::mlx::replicated_text::tests::{
        qwen_hybrid_config, routed_deepseek_v4_config, tiny_artifact, tiny_heterogeneous_artifact,
    };
    use eredu_runtime::{HostSlotAttachmentError, HostSlotMetadata};
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    for (root, fixed_tables) in [
        (tiny_artifact("llama", true), false),
        (tiny_heterogeneous_artifact(qwen_hybrid_config()), true),
        (
            tiny_heterogeneous_artifact(routed_deepseek_v4_config()),
            false,
        ),
    ] {
        let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
        let backend = MlxBackend::new(&stream, &stream).with_memory_ledger(pool.clone());
        let model = eredu_core::load_model(&backend, root.path(), crate::MlxLoadRequest::default())
            .unwrap();
        let runtime = ModelRuntime::from_prepared(backend, model).unwrap();
        // Public loading settles its already-retained numerical helpers before
        // publication. Deliberately unfinalized low-level inventories retain
        // their separate Unknown coverage in retained_rotary tests.
        crate::backend::submission_recovery::wait_for_retirement(|| {
            reclaim();
            pool.unquoted_owner_count().unwrap() == 0
        });
        assert!(runtime.session().payload._memory_owner.is_none());
        assert!(runtime.session().payload.model.has_published_idle_storage());
        let mut auxiliary = runtime
            .session()
            .payload
            .model
            .erased()
            .retained_idle_auxiliary_storage()
            .unwrap();
        let tokens: Vec<HostSlotMetadata> = auxiliary.slot_metadata_sources().cloned().collect();
        let layout = actual_layout(&runtime);
        assert_eq!(
            tokens.len(),
            1 + if fixed_tables {
                layout.layout().len()
            } else {
                0
            }
        );
        let table_bytes: u64 = tokens.iter().map(|t| t.capacity_bytes().unwrap()).sum();
        assert!(table_bytes > 0);
        let before = auxiliary.byte_bound().unwrap();
        for token in &tokens {
            auxiliary.include_slot_metadata(token.clone()).unwrap();
        }
        assert_eq!(auxiliary.byte_bound().unwrap(), before);
        let all = runtime.session().payload.retained_idle_storage().unwrap();
        assert_eq!(all.decoder_state_bytes().unwrap(), Some(0));
        assert!(pool.fixture_host_charge().unwrap() >= all.nonstate_bytes().unwrap().unwrap());
        drop((all, auxiliary, layout, runtime));
        settle_bytes(&pool, table_bytes);
        for token in &tokens {
            let error = token
                .try_attach(
                    pool.shared_storage_accounting_id(),
                    || -> Result<Box<dyn Send + Sync>, std::convert::Infallible> {
                        panic!("retired table must not acquire new custody")
                    },
                )
                .unwrap_err();
            assert!(matches!(error, HostSlotAttachmentError::Retired));
        }
        let remaining = tokens
            .iter()
            .find(|token| token.capacity_bytes().unwrap() > 0)
            .unwrap()
            .clone();
        let remaining_bytes = remaining.capacity_bytes().unwrap();
        drop(tokens);
        settle_bytes(&pool, remaining_bytes);
        drop(remaining);
        settle_bytes(&pool, 0);
    }
}

#[cfg(test)]
#[allow(unused_imports)]
use crate::memory_fixture::LedgerFixture;
