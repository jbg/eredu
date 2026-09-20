use super::*;
use eredu_runtime::working_memory::WorkingMemoryPool;
use std::{
    collections::BTreeMap,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

#[derive(Debug)]
struct Retired(Arc<AtomicBool>);

impl Drop for Retired {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}

fn watch_backing(array: &Array) -> Arc<AtomicBool> {
    let retired = Arc::new(AtomicBool::new(false));
    array
        .retain_deferred_allocation_owner(Retired(retired.clone()))
        .unwrap();
    retired
}

fn reclaim() {
    crate::backend::nn::shared::MlxNeuralBackend::reclaim_retired_resources();
    safemlx::reclaim_allocation_owners();
}

fn await_retirement(probes: &[Arc<AtomicBool>]) {
    crate::backend::submission_recovery::wait_for_retirement(|| {
        reclaim();
        probes.iter().all(|probe| probe.load(Ordering::Acquire))
    });
}

fn config(temperature: f32) -> TextGenerationConfig {
    TextGenerationConfig::new(
        eredu_core::resolve_generation_config(
            None,
            eredu_core::GenerationConfigOverrides {
                temperature: Some(temperature),
                max_new_tokens: Some(4),
                top_k: Some(0),
                top_p: Some(1.0),
                min_p: Some(0.0),
                repetition_penalty: Some(1.0),
                ..Default::default()
            },
        )
        .unwrap(),
    )
    .with_seed(12345)
}

fn emitted_tokens_retire_parents(
    stream: &Stream,
    target: &str,
    token_capacity: Option<u64>,
    key_capacity: u64,
) {
    let source = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    for temperature in [0.0, 0.7] {
        let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
        let backend = MlxBackend::new(stream, &source).with_memory_pool(pool.clone());
        let root = crate::composition::mlx::replicated_text::tests::tiny_artifact("llama", true);
        let model = eredu_core::load_model(&backend, root.path(), crate::MlxLoadRequest::default())
            .unwrap();
        let mut runtime = ModelRuntime::from_prepared(backend, model).unwrap();
        let mut state =
            MlxBackend::start_text_generation(runtime.backend(), config(temperature)).unwrap();
        assert_eq!(state.sampling.prng.is_some(), temperature > 0.0);
        let mut tokens = Vec::new();
        let mut token_backings = BTreeMap::new();
        let mut token_retirement = Vec::new();
        let mut parent_retirement = Vec::new();
        let mut expected_ids = Vec::new();

        for step in 0..4 {
            // Use the real ordinary model and text submission/completion path.
            // This fixture supplies decode IDs independently so retaining all
            // previous token handles cannot be confused with pending inputs.
            let model = runtime
                .decode(Array::from_slice(&[step + 1_u32], &[1, 1]))
                .unwrap();
            let logits = model.output.logits().unwrap().as_array();
            parent_retirement.push(watch_backing(logits));
            let expected_greedy = if temperature == 0.0 {
                let evaluated = logits.evaluated().unwrap();
                let values = evaluated.as_slice::<f32>();
                assert!(values.iter().all(|value| value.is_finite()));
                assert!(values.iter().any(|value| value.abs() > 1e-6));
                Some(
                    values
                        .iter()
                        .enumerate()
                        .rev()
                        .max_by(|(_, left), (_, right)| left.total_cmp(right))
                        .unwrap()
                        .0 as u32,
                )
            } else {
                None
            };
            if temperature > 0.0 {
                let random = state.sampling.prng.as_ref().unwrap();
                // This works on an unmaterialized slice as well as the initial
                // seed, replaced by this stochastic step. Attaching the probe
                // does not evaluate the RNG graph. Greedy sampling has no key.
                parent_retirement.push(watch_backing(random.as_array()));
            }
            let token = sample_text_submission(
                runtime.session(),
                model,
                &TokenFilter::All,
                &mut state,
                stream.clone(),
            )
            .unwrap()
            .wait()
            .unwrap();
            let id = token.token_id().unwrap();
            assert!(id < 64);
            if let Some(expected) = expected_greedy {
                assert_eq!(id, expected);
            }
            token.value.evaluated().unwrap();
            assert_eq!(token.value.shape(), &[1]);
            assert_eq!(token.value.dtype(), Dtype::Uint32);
            let allocation = token.value.allocation_info().unwrap().unwrap();
            let bytes = u64::try_from(allocation.bytes()).unwrap();
            assert!(bytes >= 4);
            if let Some(bound) = token_capacity {
                assert!(bytes <= bound);
            }
            assert!(token_backings
                .insert(allocation.identity(), bytes)
                .is_none());
            token_retirement.push(watch_backing(&token.value));
            let clone = token.clone();
            assert_eq!(clone.value.allocation_info().unwrap(), Some(allocation));
            drop(token);
            tokens.push(clone);
            expected_ids.push(id);

            // Waiting and dropping exact completion releases model logits and
            // prior key backing even while every emitted token remains live.
            await_retirement(&parent_retirement);
            assert!(token_retirement
                .iter()
                .all(|probe| !probe.load(Ordering::Acquire)));
            assert_eq!(token_backings.len(), tokens.len());
            if let Some(bound) = token_capacity {
                assert!(token_backings.values().sum::<u64>() <= bound * tokens.len() as u64);
            }
        }

        // The current RNG root belongs to sampler state, not any emitted token.
        // Native indexing can materialize its two words independently of the
        // four-word split parent; do not infer its capacity from that parent.
        if let Some(random) = &state.sampling.prng {
            random.as_array().evaluated().unwrap();
            let allocation = random.as_array().allocation_info().unwrap().unwrap();
            assert_eq!(random.as_array().shape(), &[2]);
            assert_eq!(random.as_array().dtype(), Dtype::Uint32);
            let bytes = u64::try_from(allocation.bytes()).unwrap();
            eprintln!(
                "retained RNG after four samples: target={target}, temperature={temperature}, \
                 dtype={:?}, bytes={bytes}, upper_bound={key_capacity}",
                random.as_array().dtype()
            );
            assert!(bytes >= 2 * std::mem::size_of::<u32>() as u64);
            assert!(bytes <= key_capacity);
            assert!(!token_backings.contains_key(&allocation.identity()));
            parent_retirement.push(watch_backing(random.as_array()));
        } else {
            assert_eq!(temperature, 0.0);
            eprintln!(
                "retained RNG after four samples: target={target}, temperature=0, no RNG root"
            );
        }
        let payload_retired = runtime.session().test_payload_retirement_probe();
        drop((state, runtime));
        await_retirement(&parent_retirement);
        crate::backend::submission_recovery::wait_for_retirement(|| {
            reclaim();
            payload_retired()
        });
        for (token, expected) in tokens.iter().zip(expected_ids) {
            assert_eq!(token.token_id().unwrap(), expected);
        }
        assert!(token_retirement
            .iter()
            .all(|probe| !probe.load(Ordering::Acquire)));
        drop(tokens);
        await_retirement(&token_retirement);
        crate::backend::submission_recovery::wait_for_retirement(|| {
            reclaim();
            pool.unquoted_owner_count().unwrap() == 0 && pool.used_bytes().unwrap() == 0
        });
    }
}

#[test]
fn cpu_retained_tokens_release_model_logits_and_previous_rng_backing() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    // The CPU fixture allows the full four-word split capacity even if native
    // indexing materializes only the current two-word key.
    emitted_tokens_retire_parents(&stream, "CPU", None, 16);
}

#[test]
#[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
fn metal_emitted_greedy_and_stochastic_tokens_fit_individual_backing_bounds() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    let allocation = crate::backend::nn::workspace::NativeAllocationFacts::current_host().unwrap();
    emitted_tokens_retire_parents(
        &stream,
        "Metal",
        Some(allocation.buffer_capacity(4).unwrap()),
        allocation.buffer_capacity(16).unwrap(),
    );
}
