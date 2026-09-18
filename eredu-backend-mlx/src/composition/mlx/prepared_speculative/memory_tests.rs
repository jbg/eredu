use super::*;
use crate::backend::managed_memory::NativeMemoryOwner;
use eredu_core::{
    cache::LayerCachePolicy, Admission, EstimationCompleteness, ExecutionWorkspaceEstimate,
    InferenceGeometry, InputTokenCount, LayerSchedule, OutputDemand, StateMemoryLayout,
    TextGenerationBackend, TextGenerationConfig, TokenFilter, TokenFilterController,
    WorkspaceBound,
};
use eredu_runtime::working_memory::{
    InferenceExecutionIdentity, WorkingMemoryError, WorkingMemoryPool,
};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

fn zero_admission() -> Admission {
    let geometry = InferenceGeometry {
        batch_size: 1,
        cached_positions: 0,
        input_positions: 1,
        max_output_tokens: 0,
        prefill_chunk_positions: 1,
        output: OutputDemand::StateOnly,
    };
    let layout = StateMemoryLayout::new(
        LayerSchedule::new(1, vec![LayerCachePolicy::NoState]).unwrap(),
        vec![0],
        1,
        1,
        EstimationCompleteness::Complete,
    )
    .unwrap();
    let zero = || WorkspaceBound::bounded(0, "stateless speculative ownership fixture");
    let state = eredu_core::estimate_runtime_state(
        &layout,
        InputTokenCount::text(1),
        0,
        1,
        std::num::NonZeroU8::new(4).unwrap(),
    )
    .unwrap()
    .with_execution_workspace(ExecutionWorkspaceEstimate {
        geometry,
        activations: zero(),
        attention: zero(),
        vocabulary: zero(),
        state_update: zero(),
        materialization: zero(),
        retained: zero(),
    })
    .unwrap();
    Admission {
        requested_positions: 1,
        state,
        incremental_required_bytes: 0,
        available_memory_bytes: None,
    }
}

fn config(temperature: f32) -> TextGenerationConfig {
    TextGenerationConfig::new(
        eredu_core::resolve_generation_config(
            None,
            eredu_core::GenerationConfigOverrides {
                temperature: Some(temperature),
                max_new_tokens: Some(2),
                ..Default::default()
            },
        )
        .unwrap(),
    )
    .with_seed(19)
}

fn settle(pool: &WorkingMemoryPool, expected: usize) {
    crate::backend::submission_recovery::wait_for_retirement(|| {
        MlxNeuralBackend::reclaim_retired_resources();
        safemlx::reclaim_allocation_owners();
        pool.unquoted_owner_count().unwrap() == expected
    });
}

fn assert_unquoted(pool: &WorkingMemoryPool, expected: usize) {
    settle(pool, expected);
    assert!(matches!(
        pool.reserve(&InferenceExecutionIdentity::default(), &zero_admission()),
        Err(WorkingMemoryError::UnknownBound)
    ));
}

fn memory_error<'a>(
    error: &'a (dyn std::error::Error + 'static),
) -> Option<&'a WorkingMemoryError> {
    let mut current = error;
    loop {
        if let Some(memory) = current.downcast_ref::<WorkingMemoryError>() {
            return Some(memory);
        }
        current = current.source()?;
    }
}

struct WatchConstraint(Arc<AtomicUsize>);
impl Clone for WatchConstraint {
    fn clone(&self) -> Self {
        self.0.fetch_add(1, Ordering::SeqCst);
        Self(self.0.clone())
    }
}
impl TokenFilterController for WatchConstraint {
    type Error = std::convert::Infallible;
    fn current_filter(&mut self) -> Result<TokenFilter, Self::Error> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(TokenFilter::All)
    }
    fn commit_token(&mut self, _: u32) -> Result<(), Self::Error> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
    fn is_complete(&mut self) -> Result<bool, Self::Error> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(false)
    }
}
impl SpeculativeTokenFilterController for WatchConstraint {
    type PreparedGrammar = eredu_core::speculative::NoPreparedGrammar;
    fn filter_at(&self, _: &[u32]) -> Result<TokenFilter, Self::Error> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(TokenFilter::All)
    }
    fn prefix_is_complete(&self, _: &[u32]) -> Result<bool, Self::Error> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(false)
    }
}

struct UnenteredSemantic;
impl SemanticState for UnenteredSemantic {
    fn fork_owned(
        &self,
    ) -> Result<eredu_core::SemanticStateOwner, eredu_core::SpeculativeOutputError> {
        panic!("semantic factory entered");
    }
    fn push_token(&mut self, _: u32) -> Result<bool, eredu_core::SpeculativeOutputError> {
        panic!("semantic decoder entered");
    }
    fn finish(
        &mut self,
        _: eredu_core::generation::FinishReason,
    ) -> Result<(), eredu_core::SpeculativeOutputError> {
        panic!("semantic finish entered");
    }
    fn cancel(&mut self) -> Result<(), eredu_core::SpeculativeOutputError> {
        panic!("semantic cancellation entered");
    }
    fn take_events(&mut self) -> eredu_core::SpeculativeBuffer<SemanticEvent> {
        panic!("semantic publication entered");
    }
}

#[test]
fn prepared_speculation_rejects_a_reserved_domain_before_lane_setup_and_respects_other_domains() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let reserved_pool = WorkingMemoryPool::new(0, 0).unwrap();
    let other_pool = WorkingMemoryPool::new(0, 0).unwrap();
    let reservation = reserved_pool
        .reserve(&InferenceExecutionIdentity::default(), &zero_admission())
        .unwrap();
    for same_domain in [true, false] {
        let selected_pool = if same_domain {
            &reserved_pool
        } else {
            &other_pool
        };
        let model_pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
        let loader = MlxBackend::new(&stream, &stream).with_memory_pool(model_pool);
        let root = super::super::replicated_text::tests::tiny_artifact("llama", true);
        let model =
            eredu_core::load_model(&loader, root.path(), crate::MlxLoadRequest::default()).unwrap();
        let prompt = MlxBackend::prepare_text_prompt(&loader, vec![1, 2, 3]).unwrap();
        let mut runtime = ModelRuntime::from_prepared(
            MlxBackend::new(&stream, &stream).with_memory_pool(selected_pool.clone()),
            model,
        )
        .unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let lane = SpeculativeGenerationLane::new(
            prompt,
            config(0.7),
            SpeculativeConfig {
                max_tokens: 2,
                temperature: 0.7,
                ..Default::default()
            },
            WatchConstraint(calls.clone()),
            Box::new(UnenteredSemantic),
            GenerationCancellationToken::new(),
            Box::new(|_| panic!("event callback entered")),
        );
        let request =
            SpeculativeGenerationBatchRequest::new(SpeculativeDraft::Embedded, vec![lane], [0; 32]);
        let error = MlxBackend::with_speculative_execution(
            &mut runtime,
            request,
            eredu_runtime::speculative::RunSpeculativeGeneration::default(),
        )
        .err()
        .unwrap();
        if same_domain {
            assert_eq!(
                memory_error(&error),
                Some(&WorkingMemoryError::ReservedWorkActive)
            );
        } else {
            // This ordinary fixture has no embedded head. Reaching its typed
            // target rejection proves another domain's reservation did not win.
            assert!(memory_error(&error).is_none());
            assert!(error.to_string().contains("prediction-extension contract"));
        }
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        settle(selected_pool, 0);
        assert_eq!(selected_pool.used_bytes().unwrap(), 0);
        assert_eq!(selected_pool.peak_bytes().unwrap(), 0);
        drop(runtime);
    }
    drop(reservation);
}

#[test]
fn bound_greedy_and_stochastic_sampler_clones_keep_their_domain_owner() {
    for temperature in [0.0, 0.7] {
        let pool = WorkingMemoryPool::new(0, 0).unwrap();
        let owner = NativeMemoryOwner::acquire(&pool).unwrap();
        let (seed, policy) = MlxSpeculativeSession::prepare_mlx_speculative_sampling(
            config(temperature),
            WatchConstraint(Arc::new(AtomicUsize::new(0))),
            &owner,
        )
        .unwrap();
        assert_eq!(seed.is_some(), temperature > 0.0);
        let sampler = MlxSpeculativeSampling::new(policy).with_memory_owner(&owner);
        let clone = sampler.clone();
        drop((sampler, seed, owner));
        assert_unquoted(&pool, 1);
        let history = match clone.inner().policy() {
            MlxTextSampler::Standard(policy) => policy.generated_tokens(),
            MlxTextSampler::MirostatV2(policy) => policy.generated_tokens(),
        };
        assert!(history.is_empty());
        drop(clone);
        settle(&pool, 0);
    }
}

#[test]
fn copied_speculative_tensor_has_distinct_backing_and_retains_fresh_authority_through_raw_escape() {
    for device in [safemlx::DeviceType::Cpu, safemlx::DeviceType::Gpu] {
        if device == safemlx::DeviceType::Gpu && !cfg!(feature = "metal") {
            continue;
        }
        let stream = Stream::new_with_device(&safemlx::Device::new(device, 0));
        let pool = WorkingMemoryPool::new(0, 0).unwrap();
        let owner = NativeMemoryOwner::acquire(&pool).unwrap();
        let context = SpeculativeExecutionStreams::single(&stream).with_memory_owner(&owner);
        let source =
            MlxTensor::from_array(Array::from_slice(&[0.25_f32, -1.5, 2.75, 4.0], &[1, 4]));
        source.as_array().evaluated().unwrap();
        let source_allocation = source.as_array().allocation_info().unwrap().unwrap();
        let copied = MlxEmbeddedPredictionMechanisms::control_tensor_snapshot(&source, context)
            .unwrap()
            .unwrap();
        let escaped = copied.into_array();
        let copied_allocation = escaped.allocation_info().unwrap().unwrap();
        assert_ne!(source_allocation.identity(), copied_allocation.identity());
        assert_eq!(
            escaped.evaluated().unwrap().as_slice::<f32>(),
            &[0.25, -1.5, 2.75, 4.0]
        );
        assert_unquoted(&pool, 2);
        let alias = escaped.try_index_device((.., 1..), &stream).unwrap();
        alias.evaluated().unwrap();
        assert_eq!(alias.allocation_info().unwrap(), Some(copied_allocation));
        drop((source, owner, escaped));
        assert_unquoted(&pool, 1);
        assert_eq!(
            alias.evaluated().unwrap().as_slice::<f32>(),
            &[-1.5, 2.75, 4.0]
        );
        drop(alias);
        settle(&pool, 0);
    }
}

#[test]
fn speculative_control_seed_rejects_same_domain_reservation_before_allocation() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let pool = WorkingMemoryPool::new(0, 0).unwrap();
    let reservation = pool
        .reserve(&InferenceExecutionIdentity::default(), &zero_admission())
        .unwrap();
    let context = SpeculativeExecutionStreams::single(&stream).with_memory_pool(&pool);
    let error =
        <MlxSpeculativeSampling<MlxTextSampler> as SpeculativeSampling>::control_seed(19, context)
            .err()
            .unwrap();
    assert_eq!(
        memory_error(&error),
        Some(&WorkingMemoryError::ReservedWorkActive)
    );
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
    assert_eq!(pool.used_bytes().unwrap(), 0);
    assert_eq!(pool.peak_bytes().unwrap(), 0);
    drop(reservation);
}

fn target_state(
    populated: bool,
    stream: &Stream,
) -> crate::backend::runtime::cache::state::MlxKeyValueState {
    use crate::backend::runtime::cache::{kv::KeyValueCache, state::MlxKeyValueState};
    use eredu_runtime::{LayerRuntimeState, StateLayout};

    let layout = StateLayout::new(
        LayerSchedule::new(
            1,
            vec![LayerCachePolicy::key_value(eredu_core::AttentionPolicy::Full, 1, 2).unwrap()],
        )
        .unwrap(),
    )
    .unwrap();
    let mut state = MlxKeyValueState::device(layout).unwrap();
    if populated {
        let keys = Array::from_slice(&[0.25_f32, -1.5, 2.75, 4.0, 6.25, -8.5], &[1, 1, 3, 2]);
        let values = Array::from_slice(&[1.25_f32, 2.5, -3.75, 5.0, 7.25, 9.5], &[1, 1, 3, 2]);
        state
            .layer(0)
            .unwrap()
            .update_and_fetch(keys, values, stream)
            .unwrap();
    }
    state
}

fn target_state_values(
    state: &crate::backend::runtime::cache::state::MlxKeyValueState,
    stream: &Stream,
) -> Vec<Vec<f32>> {
    state
        .retained_arrays()
        .into_iter()
        .map(|array| {
            array
                .contiguous(false, stream)
                .unwrap()
                .evaluated()
                .unwrap()
                .as_slice::<f32>()
                .to_vec()
        })
        .collect()
}

#[test]
fn empty_and_populated_embedded_state_copies_keep_fresh_authority_through_clone_and_restore() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    for populated in [false, true] {
        let pool = WorkingMemoryPool::new(0, 0).unwrap();
        let owner = NativeMemoryOwner::acquire(&pool).unwrap();
        let source = target_state(populated, &stream);
        assert_eq!(source.retained_arrays().is_empty(), !populated);
        let expected = target_state_values(&source, &stream);
        let copied = {
            let context = SpeculativeExecutionStreams::single(&stream).with_memory_owner(&owner);
            copy_control_state(&source, context).unwrap().unwrap()
        };
        assert_eq!(source.offset(), if populated { 3 } else { 0 });
        assert_eq!(copied.offset(), source.offset());
        assert_eq!(target_state_values(&copied, &stream), expected);
        assert_eq!(copied.retained_arrays().is_empty(), !populated);
        for (original, copy) in source
            .retained_arrays()
            .into_iter()
            .zip(copied.retained_arrays())
        {
            assert_ne!(
                original.allocation_info().unwrap().unwrap().identity(),
                copy.allocation_info().unwrap().unwrap().identity(),
            );
        }
        assert_unquoted(&pool, 2);

        let clone = copied.clone();
        drop((source, copied, owner));
        assert_unquoted(&pool, 1);
        assert_eq!(target_state_values(&clone, &stream), expected);

        let mut restored = target_state(false, &stream);
        restored.restore_checkpoint(&clone, &stream).unwrap();
        assert_eq!(restored.offset(), clone.offset());
        assert_eq!(target_state_values(&restored, &stream), expected);
        drop(clone);
        // In the empty case no native allocation can carry this authority.
        assert_unquoted(&pool, 1);
        drop(restored);
        settle(&pool, 0);
        drop(
            pool.reserve(&InferenceExecutionIdentity::default(), &zero_admission())
                .unwrap(),
        );
    }
}

#[test]
fn embedded_state_copy_rejects_a_reserved_domain_without_changing_empty_or_populated_source() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    for populated in [false, true] {
        let pool = WorkingMemoryPool::new(0, 0).unwrap();
        let source = target_state(populated, &stream);
        let expected = target_state_values(&source, &stream);
        let allocations = source
            .retained_arrays()
            .into_iter()
            .map(|array| array.allocation_info().unwrap().unwrap())
            .collect::<Vec<_>>();
        let offset = source.offset();
        let reservation = pool
            .reserve(&InferenceExecutionIdentity::default(), &zero_admission())
            .unwrap();
        let context = SpeculativeExecutionStreams::single(&stream).with_memory_pool(&pool);
        let error = copy_control_state(&source, context).unwrap_err();
        assert_eq!(
            memory_error(&error),
            Some(&WorkingMemoryError::ReservedWorkActive)
        );
        assert_eq!(source.offset(), offset);
        assert_eq!(target_state_values(&source, &stream), expected);
        assert_eq!(
            source
                .retained_arrays()
                .into_iter()
                .map(|array| array.allocation_info().unwrap().unwrap())
                .collect::<Vec<_>>(),
            allocations,
        );
        assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
        assert_eq!(pool.used_bytes().unwrap(), 0);
        assert_eq!(pool.peak_bytes().unwrap(), 0);
        drop(reservation);
    }
}
