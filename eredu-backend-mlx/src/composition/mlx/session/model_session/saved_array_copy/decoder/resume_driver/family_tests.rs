//! Actual existing family fixtures through the same dense saved-source driver.
use super::super::super::tests::{reclaim, settle, words, AllTokens, Runtime};
use super::tests::{required_capacity, saved_source, source_config};
use super::*;
use crate::composition::mlx::{
    replicated_text::tests::{tiny_artifact, tiny_packed_safetensors_artifact},
    session::model_session::host_layerwise_tests::family_artifact,
};
use eredu_core::{
    cache::LayerCachePolicy, AttentionPolicy, GenerationCancellationToken, TextGeneration,
    TextGenerationDriver, TokenOutput,
};
use eredu_runtime::working_memory::WorkingMemoryPool;

#[derive(Clone, Copy, Debug)]
enum Family {
    RoutedQwen3,
    SlidingGemma4,
    PackedQwen3,
}
impl Family {
    fn artifact(self) -> tempfile::TempDir {
        match self {
            Self::RoutedQwen3 => tiny_artifact("qwen3_moe", true),
            Self::SlidingGemma4 => family_artifact("gemma4"),
            Self::PackedQwen3 => tiny_packed_safetensors_artifact("qwen3"),
        }
    }
    fn adaptive(self) -> bool {
        matches!(self, Self::RoutedQwen3)
    }
}

fn family_runtime(pool: &WorkingMemoryPool, family: Family) -> (Runtime, tempfile::TempDir) {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    let weights = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let backend = MlxBackend::new(&stream, &weights).with_memory_pool(pool.clone());
    let artifact = family.artifact();
    let prepared =
        eredu_core::load_model(&backend, artifact.path(), crate::MlxLoadRequest::default())
            .unwrap_or_else(|error| panic!("{family:?} public loading: {error}"));
    let runtime = ModelRuntime::from_prepared(backend, prepared).unwrap();
    crate::backend::submission_recovery::wait_for_retirement(|| {
        reclaim();
        pool.unquoted_owner_count().unwrap() == 0
    });
    assert!(runtime.session().payload.model.has_published_idle_storage());
    if matches!(family, Family::PackedQwen3) {
        let selected = runtime
            .session()
            .payload
            .model
            .inference_blueprint()
            .unwrap()
            .selected();
        let head = selected
            .text_realization()
            .parameters()
            .iter()
            .find(|parameter| parameter.name() == "lm_head.weight")
            .expect("packed untied fixture must select a distinct output projection");
        assert!(matches!(
            head.executable(),
            eredu_checkpoint::LinearFormat::Affine(_)
        ));
        assert!(matches!(
            head.source_encoding(),
            eredu_checkpoint::SourceTensorEncoding::Safetensors(eredu_checkpoint::StoredDtype::U32)
        ));
    }
    (runtime, artifact)
}

#[derive(Debug, PartialEq)]
struct Values {
    sampler: String,
    key: Vec<u32>,
    pending: Vec<u32>,
    native: Vec<Vec<f32>>,
}
fn values(saved: &MlxSavedTextComponents) -> Values {
    let saved = saved.funded_source().unwrap();
    let mut native = Vec::new();
    saved
        .decoder
        .native
        .prepare_copy()
        .unwrap()
        .visit_operands(&mut |array| {
            native.push(array.evaluated().unwrap().try_to_vec::<f32>().unwrap());
        });
    Values {
        sampler: format!("{:?}", saved.sampling.sampler.as_sampler()),
        key: words(saved.sampling.arrays.key.as_ref().unwrap()),
        pending: words(saved.sampling.arrays.pending.as_ref().unwrap()),
        native,
    }
}

fn parity(family: Family) {
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (mut runtime, artifact) = family_runtime(&pool, family);
    let (saved, expected) = saved_source(&mut runtime, family.adaptive());
    let source = saved.funded_source().unwrap();
    assert_eq!(
        (source.sampling.frontier(), source.sampling.next_prediction),
        (6, 2)
    );
    let prepared = source.decoder.native.prepare_copy().unwrap();
    let layout = prepared
        .shared_layout()
        .expect("selected native representation retains its exact shared layout")
        .clone();
    assert!((0..layout.layout().len()).all(|layer| matches!(
        layout.layout().layer(layer),
        Some(LayerCachePolicy::KeyValue { .. } | LayerCachePolicy::KeyValueWithFixedState { .. })
    )));
    if matches!(family, Family::SlidingGemma4) {
        assert_eq!(layout.layout().len(), 3);
        let fixed = layout.layout().layer(0).unwrap().fixed_state();
        assert_eq!(
            fixed.len(),
            1,
            "the optional prefix role is a real retained slot"
        );
        assert_eq!(
            fixed[0].role,
            eredu_core::cache::StateTensorRole::PrefixEmbedding
        );
        assert_eq!(
            fixed[0].presence,
            eredu_core::cache::StateTensorPresence::Optional
        );
        assert!(layout.layout().layer(1).unwrap().fixed_state().is_empty());
        for (index, expected) in [
            AttentionPolicy::Full,
            AttentionPolicy::sliding(8).unwrap(),
            AttentionPolicy::Full,
        ]
        .into_iter()
        .enumerate()
        {
            let Some(
                LayerCachePolicy::KeyValue { attention, .. }
                | LayerCachePolicy::KeyValueWithFixedState { attention, .. },
            ) = layout.layout().layer(index)
            else {
                panic!("Gemma fixture KV policy")
            };
            assert_eq!(*attention, expected);
        }
    }
    drop(prepared);
    // Registered children are pinned before a later copy. Grouped saved
    // outer/child tables retain authenticated host holds rather than fake keys.
    {
        let plan = source.decoder.native.prepare_copy().unwrap();
        let mut children = RetainedStorage::default();
        plan.visit_registered_child_metadata(&mut |metadata| {
            children
                .include_slot_metadata(metadata.clone())
                .map_err(Error::from)
        })
        .unwrap();
        drop(children.pin_registered(&pool).unwrap());
    }
    let original = values(&saved);
    assert!(
        original.native.iter().flatten().any(|value| *value != 0.0),
        "{family:?} native source must be nonzero"
    );
    let (required, exact) = required_capacity(
        &runtime,
        &saved,
        source_config(family.adaptive(), 3, u64::MAX),
    );
    // One common finite ceiling covers both successive resumptions and their
    // existing target state. The separate Llama test checks exact/-1 admission.
    let capacity = exact.checked_add(required.checked_mul(3).unwrap()).unwrap();
    let config = source_config(family.adaptive(), 3, capacity);
    let cancellation = GenerationCancellationToken::new();
    let ordinary = {
        let mut run = TextGeneration::resume_saved(&mut runtime, &saved, config, &cancellation)
            .unwrap_or_else(|error| panic!("{family:?} ordinary resume: {error}"))
            .expect("positive output allowance");
        run.by_ref()
            .map(|token| token.unwrap().token_id().unwrap())
            .collect::<Vec<_>>()
    };
    assert_eq!(ordinary, expected, "{family:?} ordinary future");
    reclaim();
    let controlled = {
        let mut driver = TextGenerationDriver::new(&mut runtime);
        let mut state = driver
            .resume_saved(&saved, config, AllTokens, &cancellation)
            .unwrap_or_else(|error| panic!("{family:?} controlled resume: {error}"))
            .expect("positive output allowance");
        // Check immediately after installation, before ordinary forward
        // publication could hide a missing empty-child registration.
        {
            let model = driver.runtime().session().payload.model.erased();
            let mut all = model.retained_decoder_state_storage().unwrap();
            all.merge(model.retained_idle_auxiliary_storage().unwrap())
                .unwrap();
            drop(all.pin_registered(&pool).expect("fresh actual layer and empty-child identities were published before first prediction"));
        }
        let mut ids = Vec::new();
        while let Some(output) = driver.advance(&mut state).unwrap() {
            assert_eq!(
                output.output().step_receipt().unwrap().attempt(),
                ids.len() as u64
            );
            ids.push(output.token_id());
            // Controlled advancement requires delivery and exact completion
            // before the next step, even when capture is disabled.
            assert!(driver.take_completed_step(&mut state).unwrap().is_none());
        }
        ids
    };
    assert_eq!(controlled, expected, "{family:?} controlled future");
    assert_eq!(
        values(&saved),
        original,
        "{family:?} immutable saved source"
    );
    assert_eq!(
        (source.sampling.frontier(), source.sampling.next_prediction),
        (6, 2)
    );
    let current = runtime
        .session()
        .payload
        .model
        .erased()
        .prepare_resident_decoder_copy()
        .unwrap();
    let mut shapes = Vec::new();
    current.visit_operands(&mut |array| shapes.push(array.shape().to_vec()));
    assert_eq!(shapes.len(), layout.layout().len() * 2);
    for index in 0..layout.layout().len() {
        let Some(
            LayerCachePolicy::KeyValue { attention, .. }
            | LayerCachePolicy::KeyValueWithFixedState { attention, .. },
        ) = layout.layout().layer(index)
        else {
            unreachable!()
        };
        let retained = attention
            .window()
            .map_or(9usize, |window| 9usize.min(window.get() as usize - 1));
        assert_eq!(
            shapes[index * 2][2],
            retained as i32,
            "{family:?} final key cache {index}"
        );
        assert_eq!(
            shapes[index * 2 + 1][2],
            retained as i32,
            "{family:?} final value cache {index}"
        );
    }
    assert!(runtime
        .session()
        .payload
        .model
        .erased()
        .state_snapshot()
        .iter()
        .all(|(position, _)| *position == 9));
    drop(current);
    {
        let model = runtime.session().payload.model.erased();
        let mut all = model.retained_decoder_state_storage().unwrap();
        all.merge(model.retained_idle_auxiliary_storage().unwrap())
            .unwrap();
        drop(
            all.pin_registered(&pool)
                .expect("fresh actual layer and empty-child identities were published"),
        );
    }
    assert_eq!(pool.effective_capacity().unwrap(), capacity);
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
    drop((layout, saved, runtime, artifact));
    settle(&pool, 0);
}

#[test]
fn routed_qwen3_moe_tied_resume_matches_managed_ordinary_and_controlled_future() {
    parity(Family::RoutedQwen3);
}
#[test]
fn gemma4_composite_sliding_resume_crosses_window_with_matching_future() {
    parity(Family::SlidingGemma4);
}
#[test]
fn affine_packed_qwen3_readout_resume_matches_managed_future() {
    parity(Family::PackedQwen3);
}
