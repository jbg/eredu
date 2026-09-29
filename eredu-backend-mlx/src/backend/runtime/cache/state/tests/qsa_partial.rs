use super::*;
use crate::backend::runtime::cache::residency::{
    load_prompt_cache_state_tensors, open_prompt_cache,
};
use eredu_architectures::qwen4_exp::qsa::{QsaBlockPosition, QsaPartialStateSpec};
use eredu_core::cache::{PromptCacheModelIdentity, PromptCacheStateSegment, PromptCacheTopology};
use eredu_core::{AttentionPolicy, LayerSchedule};
use eredu_nn::{RotaryPosition, Tensor, TensorElementType};

fn exercise(stream: &Stream) {
    let spec = QsaPartialStateSpec::new(
        3,
        2,
        2,
        TensorElementType::F32,
        TensorElementType::F32,
        7,
        10,
    )
    .unwrap();
    let workspace = spec.required_workspace::<MlxTensor>(2).unwrap();
    let policies = LayerSchedule::new(
        1,
        vec![LayerCachePolicy::key_value_with_state(
            AttentionPolicy::Full,
            1,
            2,
            spec.policies(),
            vec![],
        )
        .unwrap()],
    )
    .unwrap();
    let layout = StateLayout::new(policies.clone()).unwrap();
    let options = eredu_runtime::PagedCacheOptions::new(2, 512, 262144, 1)
        .unwrap()
        .with_full_attention(true);
    let mut state = MlxHybridState::paged(
        layout.clone(),
        CacheResidencyManager::new(options.clone()).unwrap(),
        None,
        &[],
    )
    .unwrap();
    let cosine: MlxTensor = Array::from_slice(&[0.6f32, -0.8], &[1, 1, 1, 2]).into();
    let sine: MlxTensor = Array::from_slice(&[0.8f32, 0.6], &[1, 1, 1, 2]).into();
    let key: MlxTensor = Array::from_slice(&[0.7f32, -0.3], &[1, 2]).into();
    let kv: MlxTensor = Array::from_slice(&[0.1f32, 0.2, 0.3, 0.4], &[2, 1, 1, 2]).into();
    let mut partial = spec
        .read::<MlxNeuralBackend, _>(&mut state.layers_mut()[0], 2, workspace, stream)
        .unwrap();
    for position in 0..2 {
        AttentionCache::update_for_attention(
            &mut state.layers_mut()[0],
            kv.clone(),
            kv.clone(),
            stream,
        )
        .unwrap();
        for (lane, part) in partial.iter_mut().enumerate() {
            part.push(
                &key,
                position,
                lane == 0 || position == 0,
                RotaryPosition::Embeddings {
                    cosine: &cosine,
                    sine: &sine,
                },
                stream,
            )
            .unwrap();
        }
    }
    spec.write::<MlxNeuralBackend, _>(&mut state.layers_mut()[0], &partial, workspace, stream)
        .unwrap();
    let checkpoint = state.deep_clone_state().unwrap();
    let descriptor = PromptCacheDescriptor::new(
        "fixture",
        "fixture",
        "weights",
        "prefix",
        "layout",
        1,
        0,
        1,
        2,
        policies.clone(),
        vec![0],
        vec![PromptCacheStateSegment::new("state", 0..1).unwrap()],
        0,
        PromptCacheTopology::default(),
    )
    .unwrap();
    let identity = PromptCacheModelIdentity::new(
        "fixture",
        "fixture",
        "layout",
        1,
        0,
        1,
        0,
        PromptCacheTopology::default(),
        policies,
        vec![0],
        descriptor.state_segments().to_vec(),
    )
    .unwrap();
    let directory = tempfile::tempdir().unwrap();
    let destination = directory.path().join("qsa-partial");
    state
        .save_prompt_cache(
            &destination,
            descriptor.clone(),
            &[7, 8],
            &PromptCacheOptions::default(),
        )
        .unwrap();
    let (manager, manifest) =
        open_prompt_cache(&destination, &descriptor, &identity, &[7, 8], options).unwrap();
    assert_eq!(manifest.state_tensors.len(), 4);
    let arrays = load_prompt_cache_state_tensors(&destination, &manifest, stream).unwrap();
    let mut restored = MlxHybridState::paged(layout, manager, None, &[]).unwrap();
    restored
        .restore_prompt_cache_state(arrays, 2, &[0])
        .unwrap();
    let mut tails = spec
        .read::<MlxNeuralBackend, _>(&mut restored.layers_mut()[0], 2, workspace, stream)
        .unwrap();
    assert_eq!(tails[0].tail(), [0, 1]);
    assert_eq!(tails[1].tail(), [0]);
    assert_eq!(tails[1].last_seen(), Some(1));
    let block = tails[0]
        .push(&key, 2, true, RotaryPosition::Offset(999), stream)
        .unwrap()
        .unwrap();
    assert_eq!(block.positions, [0, 1, 2]);
    assert_eq!(
        block.raw_keys.to_f32_vec(stream).unwrap(),
        [0.7, -0.3, 0.7, -0.3, 0.7, -0.3]
    );
    match block.first_position {
        QsaBlockPosition::Embeddings { cosine: c, sine: s } => {
            assert_eq!(
                c.to_f32_vec(stream).unwrap(),
                cosine.to_f32_vec(stream).unwrap()
            );
            assert_eq!(
                s.to_f32_vec(stream).unwrap(),
                sine.to_f32_vec(stream).unwrap()
            );
        }
        _ => panic!("first-token media rotary provenance was lost"),
    }
    spec.write::<MlxNeuralBackend, _>(&mut state.layers_mut()[0], &tails, workspace, stream)
        .unwrap();
    state.restore_checkpoint(&checkpoint, stream).unwrap();
    let rollback = spec
        .read::<MlxNeuralBackend, _>(&mut state.layers_mut()[0], 2, workspace, stream)
        .unwrap();
    assert_eq!(rollback[0].tail(), [0, 1]);
    assert_eq!(rollback[0].last_seen(), Some(1));
}

#[test]
fn qsa_partial_components_survive_native_prompt_restore_and_rollback() {
    exercise(&Stream::new_with_device(&safemlx::Device::new(
        safemlx::DeviceType::Cpu,
        0,
    )));
}
#[test]
#[ignore = "requires Metal device access"]
fn qsa_partial_components_survive_native_prompt_restore_and_rollback_metal() {
    exercise(&Stream::new_with_device(&safemlx::Device::new(
        safemlx::DeviceType::Gpu,
        0,
    )));
}
