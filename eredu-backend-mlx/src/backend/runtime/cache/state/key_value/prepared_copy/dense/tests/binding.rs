//! Final binding uses actual settled copies, without starting a resumed run.

use super::*;
use crate::memory_fixture::LedgerFixture;
use crate::{
    backend::{error::Error, runtime::cache::kv::KeyValueCache},
    composition::mlx::{loading, replicated_text::PreparedDenseControlBindingError, Executable},
};
use eredu_core::PreparedInputIdentity;
use eredu_runtime::{
    replicated_session::{ReplicatedTextControlState, ReplicatedTextSessionError},
    PreparedInputCacheIdentity, SharedPreparedInputCacheIdentity,
};

fn load(root: &std::path::Path, stream: &Stream) -> Executable {
    let source_stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let inspection = eredu_architectures::configuration::inspect_artifact(root).unwrap();
    let options = crate::MlxLoadRequest::default();
    let selected = loading::select_preparation(&inspection, options.clone()).unwrap();
    let plan = eredu_core::plan_model_preparation(
        inspection,
        options.normalized().preparation_policy().unwrap(),
        selected.session_capabilities(),
    )
    .unwrap();
    let (sources, _) = loading::prepare_selected_sources(plan, selected, None).unwrap();
    loading::materialize_model_plan(sources, None, stream, &source_stream)
        .unwrap()
        .into_executable()
}

fn populated_source(
    model: &Executable,
    pool: &MemoryLedger,
    stream: &Stream,
    positions: usize,
) -> MlxKeyValueState {
    let owner = NativeMemoryOwner::acquire(pool).unwrap();
    let source = model.erased().prepare_resident_decoder_copy().unwrap();
    let layout = source.shared_layout().unwrap().as_ref().clone();
    let mut state = MlxKeyValueState::device(layout).unwrap();
    for index in 0..state.layers.len() {
        let LayerCachePolicy::KeyValue {
            num_key_value_heads,
            head_dim,
            ..
        } = state.layout.layout().layer(index).unwrap()
        else {
            panic!("the dense fixture requires actual ordinary KV layers");
        };
        let shape = [
            1,
            num_key_value_heads.get() as i32,
            positions as i32,
            head_dim.get() as i32,
        ];
        let count = shape.iter().product::<i32>() as usize;
        let values = (0..count)
            .map(|i| 1.0 + index as f32 + i as f32 / 17.0)
            .collect::<Vec<_>>();
        let keys = Array::from_slice(&values, &shape);
        let values = Array::from_slice(&values.iter().map(|x| -x).collect::<Vec<_>>(), &shape);
        let MlxKeyValueLayerState::Device(cache) = &mut state.layers.slots_mut()[index] else {
            panic!("resident fixture")
        };
        cache.update_and_fetch(keys, values, stream).unwrap();
    }
    publish_source(&state, &owner);
    drop(owner);
    settle(pool, pool.fixture_funded_charge().unwrap());
    state
}

fn prompt(label: &str, positions: usize) -> SharedPreparedInputCacheIdentity {
    use eredu_core::{
        checkpoint::TensorDtype, InputModality, InputPartDescriptor, InputPayloadKind,
        InputTensorIdentity,
    };
    SharedPreparedInputCacheIdentity::new(
        PreparedInputCacheIdentity::new(
            PreparedInputIdentity::new(vec![InputPartDescriptor::new(
                InputModality::Text,
                InputPayloadKind::TokenIds,
                InputTensorIdentity::new(TensorDtype::U32, vec![1, positions]).unwrap(),
                [],
            )
            .unwrap()])
            .unwrap(),
            label.to_owned(),
        )
        .unwrap(),
    )
}

fn published(
    source: &MlxKeyValueState,
    pool: &MemoryLedger,
    stream: &Stream,
) -> (
    PublishedDenseResidentKvState,
    InferencePromptCompletion,
    InferenceTextPreparation,
    WorkingMemoryFundingRun,
) {
    let plan = source.prepare_resident_copy().unwrap();
    let quoted = quote(&plan, pool);
    // Reserve the exact P+N payload; a roomy application ceiling allows another
    // installed branch to remain alive without testing capacity succession here.
    let (preparation, run) = fresh(pool, quoted.bytes, u64::MAX).unwrap();
    let (slots, native) = preparation
        .claim_prompt()
        .unwrap()
        .construct_dense_decoder(quoted.host, &run, quoted.complete)
        .unwrap();
    let roots = RefCell::new(Vec::new());
    let completed = plan.copy_dense_retained(slots, stream, &roots).unwrap();
    let pointer = completed.layers.get(0).unwrap() as *const MlxKeyValueLayerState;
    publish_arrays(&completed, &native, &roots);
    let (published, completion) = completed.publish_for_control().unwrap();
    assert_eq!(published.state.layers.slots().as_ptr(), pointer);
    assert!(published.state.inference_retention.is_empty());
    roots.borrow_mut().clear();
    native.certify().unwrap();
    (published, completion, preparation, run)
}

fn contract(error: Error) -> String {
    let Error::Other(error) = error else {
        panic!("original typed neutral error required")
    };
    let error = error
        .downcast::<ReplicatedTextSessionError<eredu_nn::Error, Error, Error>>()
        .unwrap();
    let ReplicatedTextSessionError::Contract(message) = *error else {
        panic!("contract error required")
    };
    message
}

#[test]
fn binding_preserves_published_table_values_frontier_prompt_and_fresh_retention() {
    let stream = metal();
    let artifact = crate::composition::mlx::replicated_text::tests::tiny_artifact("llama", false);
    let mut model = load(artifact.path(), &stream);
    let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
    let origin = model.erased().resident_control_origin().unwrap();
    // Finish unquoted source preparation before either funded branch exists.
    let sources = [
        populated_source(&model, &pool, &stream, 3),
        populated_source(&model, &pool, &stream, 5),
    ];
    let mut previous_prompt: Option<SharedPreparedInputCacheIdentity> = None;
    for (positions, mut source) in [3, 5].into_iter().zip(sources) {
        let (old_sampler, old_preparation, old_run) = sampler(&pool);
        source.inference_retention.admit(old_preparation.request());
        let expected = values(&source.prepare_resident_copy().unwrap());
        assert!(expected.iter().flatten().any(|x| *x != 0.0));
        let identity = prompt(&format!("branch-{positions}"), positions);
        let identity_alias = identity.clone();
        let identity_pointer = identity.as_ref() as *const PreparedInputCacheIdentity;
        let (state, completion, preparation, run) = published(&source, &pool, &stream);
        let table = state.state.layer_slot_metadata().clone();
        let allocation_ids = operands(&state.state.prepare_resident_copy().unwrap())
            .iter()
            .map(|array| array.allocation_info().unwrap().unwrap().identity())
            .collect::<Vec<_>>();
        let before = (usage(&pool), model.erased().state_snapshot());
        let mut slot = model
            .erased()
            .bind_prepared_dense_control_state(&origin, state, Some(identity))
            .unwrap();
        assert_eq!(usage(&pool), before.0);
        assert_eq!(
            model.erased().state_snapshot(),
            before.1,
            "binding is not installation"
        );
        let saved = slot
            .downcast_ref::<ReplicatedTextControlState<MlxKeyValueState>>()
            .unwrap();
        let held = saved.shared_prompt_input_identity().unwrap();
        assert_eq!(
            held.as_ref() as *const PreparedInputCacheIdentity,
            identity_pointer
        );
        assert!(held.same_storage(&identity_alias));
        assert_eq!(
            model.erased().resident_copy_input_identity().unwrap(),
            previous_prompt
        );
        completion.finish().unwrap();
        preparation.bind_prompt().unwrap();
        model
            .erased_mut()
            .exchange_native_control_state(slot.as_mut())
            .unwrap();
        model
            .erased()
            .validate_text_frontier(positions as u64)
            .unwrap();
        assert!(model
            .erased()
            .retained_inference_authority()
            .unwrap()
            .is_empty());
        assert!(
            source.inference_retention.admission().is_some(),
            "old source grant remains only with source"
        );
        assert!(model
            .erased()
            .resident_copy_input_identity()
            .unwrap()
            .unwrap()
            .same_storage(&identity_alias));
        let mut inventory = model.erased().retained_decoder_state_storage().unwrap();
        inventory
            .merge(model.erased().retained_idle_auxiliary_storage().unwrap())
            .unwrap();
        assert!(inventory
            .slot_metadata_sources()
            .any(|metadata| metadata.same_storage(&table)));
        let plan = model.erased().prepare_resident_decoder_copy().unwrap();
        let mut installed = Vec::new();
        plan.dense_key_value()
            .expect("resident KV fixture")
            .visit_operands(&mut |array| installed.push(array));
        assert_eq!(
            installed
                .iter()
                .map(|array| array.evaluated().unwrap().try_to_vec::<f32>().unwrap())
                .collect::<Vec<_>>(),
            expected
        );
        assert_eq!(
            installed
                .iter()
                .map(|array| array.allocation_info().unwrap().unwrap().identity())
                .collect::<Vec<_>>(),
            allocation_ids
        );
        drop((
            installed,
            plan,
            inventory,
            slot,
            table,
            source,
            old_sampler,
            old_preparation,
            old_run,
            preparation,
            run,
        ));
        previous_prompt = Some(identity_alias);
    }
    drop(model);
    settle(&pool, 0);
}

#[test]
fn foreign_and_invalidated_origins_reject_settled_owner_without_refunding_escaped_roots() {
    let stream = metal();
    let artifact = crate::composition::mlx::replicated_text::tests::tiny_artifact("llama", false);
    let mut model = load(artifact.path(), &stream);
    let foreign = load(artifact.path(), &stream);
    for stale in [false, true] {
        let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
        let source = populated_source(&model, &pool, &stream, 3);
        let origin = if stale {
            model.erased().resident_control_origin().unwrap()
        } else {
            foreign.erased().resident_control_origin().unwrap()
        };
        // Cold validation and final binding use the same exact identity fence.
        if stale {
            model
                .erased()
                .validate_resident_control_origin(&origin)
                .unwrap();
            crate::memory_fixture::publish_model_parameters(
                model.erased_mut(),
                std::iter::empty(),
                false,
            );
        }
        assert!(model
            .erased()
            .validate_resident_control_origin(&origin)
            .is_err());
        let (state, completion, preparation, run) = published(&source, &pool, &stream);
        let table = state.state.layer_slot_metadata().clone();
        let raw = Array::clone(operands(&state.state.prepare_resident_copy().unwrap())[0]);
        let raw_bytes = {
            let info = raw.allocation_info().unwrap().unwrap();
            (info.bytes() as u64)
                .checked_add(info.host_control_bytes() as u64)
                .unwrap()
        };
        let before = model.erased().state_snapshot();
        let error = model
            .erased()
            .bind_prepared_dense_control_state(&origin, state, None)
            .err()
            .unwrap();
        assert!(contract(error).contains("different executable"));
        assert_eq!(model.erased().state_snapshot(), before);
        assert!(matches!(
            table.try_attach(pool.shared_storage_accounting_id(), || Ok::<_, Infallible>(
                Box::new(())
            )),
            Err(HostSlotAttachmentError::Retired)
        ));
        let account_controls = crate::memory_fixture::request_control_bytes(preparation.request());
        drop((source, completion, preparation, run));
        settle(
            &pool,
            table.capacity_bytes().unwrap() + raw_bytes + account_controls,
        );
        assert!(!raw
            .evaluated()
            .unwrap()
            .try_to_vec::<f32>()
            .unwrap()
            .is_empty());
        drop(table);
        settle(&pool, raw_bytes + account_controls);
        drop(raw);
        settle(&pool, 0);
    }
}

#[test]
fn selected_geometry_mismatch_consumes_settled_owner_without_installing() {
    let stream = metal();
    let artifact = crate::composition::mlx::replicated_text::tests::tiny_artifact("llama", false);
    let model = load(artifact.path(), &stream);
    let origin = model.erased().resident_control_origin().unwrap();
    let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
    let loading = NativeMemoryOwner::acquire(&pool).unwrap();
    let source = super::state(&stream); // three layers and different selected KV geometry
    publish_source(&source, &loading);
    drop(loading);
    settle(&pool, pool.fixture_funded_charge().unwrap());
    let (state, completion, preparation, run) = published(&source, &pool, &stream);
    let before = model.erased().state_snapshot();
    let message = contract(
        model
            .erased()
            .bind_prepared_dense_control_state(&origin, state, None)
            .err()
            .unwrap(),
    );
    assert!(message.contains("layout"), "{message}");
    assert_eq!(model.erased().state_snapshot(), before);
    drop((source, completion, preparation, run));
    settle(&pool, 0);
}

#[test]
fn actual_hybrid_executable_rejects_dense_kv_representation_without_installing() {
    let stream = metal();
    let artifact = crate::composition::mlx::replicated_text::tests::tiny_artifact("llama", false);
    let model = load(artifact.path(), &stream);
    let origin = model.erased().resident_control_origin().unwrap();
    let hybrid_artifact =
        crate::composition::mlx::replicated_text::tests::tiny_heterogeneous_artifact(
            crate::composition::mlx::replicated_text::tests::qwen_hybrid_config(),
        );
    let hybrid = load(hybrid_artifact.path(), &stream);
    let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
    let source = populated_source(&model, &pool, &stream, 3);
    let (state, completion, preparation, run) = published(&source, &pool, &stream);
    let before = hybrid.erased().state_snapshot();
    let error = hybrid
        .erased()
        .bind_prepared_dense_control_state(&origin, state, None)
        .err()
        .unwrap();
    let Error::Other(error) = error else {
        panic!("typed native state representation rejection required")
    };
    assert!(matches!(
        error.downcast_ref::<PreparedDenseControlBindingError>(),
        Some(PreparedDenseControlBindingError::UnsupportedStateType)
    ));
    assert_eq!(hybrid.erased().state_snapshot(), before);
    drop((source, completion, preparation, run));
    settle(&pool, 0);
}
