use super::*;
use eredu_runtime::{GenerationSampler, working_memory::WorkingMemoryPool};

type Sampling = MlxSpeculativeSampling<GenerationSampler>;

fn key_words(state: &MlxSpeculativeRandomState) -> Vec<u32> {
    state
        .value
        .ordinary()
        .expect("ordinary fixture key")
        .as_array()
        .evaluated()
        .unwrap()
        .as_slice::<u32>()
        .to_vec()
}

fn zero_admission() -> eredu_core::Admission {
    use eredu_core::{
        EstimationCompleteness, ExecutionWorkspaceEstimate, InferenceGeometry, InputTokenCount,
        LayerSchedule, OutputDemand, StateMemoryLayout, WorkspaceBound, cache::LayerCachePolicy,
    };
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
    let zero = || WorkspaceBound::bounded(0, "stateless speculative RNG ownership fixture");
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
    eredu_core::Admission {
        requested_positions: 1,
        state,
        incremental_required_bytes: 0,
        available_memory_bytes: None,
    }
}

fn assert_reserved(error: &Exception) {
    use eredu_runtime::working_memory::WorkingMemoryError;
    let mut source: &(dyn std::error::Error + 'static) = error;
    loop {
        if let Some(memory) = source.downcast_ref::<WorkingMemoryError>() {
            assert_eq!(*memory, WorkingMemoryError::ReservedWorkActive);
            return;
        }
        source = source
            .source()
            .expect("preserve the domain admission error");
    }
}

#[test]
fn control_seed_clones_retain_an_independent_preparation_owner() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let pool = WorkingMemoryPool::new(4096, 0).unwrap();
    let owner = NativeMemoryOwner::acquire(&pool).unwrap();
    let context = SpeculativeExecutionStreams::single(&stream).with_memory_owner(&owner);
    let seed = Sampling::control_seed(73, context).unwrap();
    let snapshot = seed.clone();
    assert_eq!(pool.unquoted_owner_count().unwrap(), 2);
    drop(owner);
    drop(seed);
    assert_eq!(pool.unquoted_owner_count().unwrap(), 1);
    assert_eq!(
        snapshot.as_array().evaluated().unwrap().as_slice::<u32>(),
        &[0, 73]
    );
    drop(snapshot);
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
}

#[test]
fn target_draft_and_position_clones_keep_ownership_and_randomness_after_root_drop() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let pool = WorkingMemoryPool::new(4096, 0).unwrap();
    let owner = NativeMemoryOwner::acquire(&pool).unwrap();
    let context = SpeculativeExecutionStreams::single(&stream).with_memory_owner(&owner);
    let seed = Sampling::control_seed(91, context).unwrap();
    let mut root = Sampling::randomness_root(Some(seed), context).unwrap();
    let mut target = Sampling::target_randomness_from_root(&mut root, context).unwrap();
    let draft = Sampling::draft_randomness_from_root(&mut root, context).unwrap();
    let draft_snapshot = draft.clone();
    let position =
        Sampling::draft_randomness_at(&draft, SpeculativeDraftRandomPosition::new(3), context)
            .unwrap();
    let position_again = Sampling::draft_randomness_at(
        &draft_snapshot,
        SpeculativeDraftRandomPosition::new(3),
        context,
    )
    .unwrap();
    let different_position = Sampling::draft_randomness_at(
        &draft_snapshot,
        SpeculativeDraftRandomPosition::new(4),
        context,
    )
    .unwrap();
    let mut target_snapshot = target.clone();
    drop(root);
    drop(draft);
    drop(draft_snapshot);
    let sampler = Sampling::new(GenerationSampler::new()).with_memory_owner(&owner);
    let value = sampler
        .sample_unit_interval(Some(&mut target), context)
        .unwrap();
    let cloned_value = sampler
        .sample_unit_interval(Some(&mut target_snapshot), context)
        .unwrap();
    assert_eq!(value, cloned_value);
    assert!((0.0..1.0).contains(&value));
    assert_eq!(key_words(&position), key_words(&position_again));
    assert_ne!(key_words(&position), key_words(&different_position));
    drop(owner);
    drop(sampler);
    drop(target);
    drop(target_snapshot);
    drop(position);
    drop(different_position);
    assert_eq!(pool.unquoted_owner_count().unwrap(), 2);
    drop(position_again);
    crate::backend::submission_recovery::reap();
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
}

#[test]
fn raw_seed_is_bound_before_derivation_and_preserves_its_existing_allocation() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let pool = WorkingMemoryPool::new(4096, 0).unwrap();
    let context = SpeculativeExecutionStreams::single(&stream).with_memory_pool(&pool);
    let value = random::key(109).unwrap();
    let identity = value.allocation_info().unwrap().unwrap();
    let seed = Sampling::seed_from_array(value);
    assert_eq!(
        seed.as_array().allocation_info().unwrap().unwrap(),
        identity
    );
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
    let mut root = Sampling::randomness_root(Some(seed), context).unwrap();
    assert_eq!(pool.unquoted_owner_count().unwrap(), 1);
    let target = Sampling::target_randomness_from_root(&mut root, context).unwrap();
    drop(root);
    assert_eq!(pool.unquoted_owner_count().unwrap(), 1);
    assert_eq!(key_words(&target).len(), 2);
    drop(target);
    crate::backend::submission_recovery::reap();
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
}

#[test]
fn failed_key_derivation_preserves_source_ownership_and_retires_after_error() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let pool = WorkingMemoryPool::new(4096, 0).unwrap();
    let owner = NativeMemoryOwner::acquire(&pool).unwrap();
    let context = SpeculativeExecutionStreams::single(&stream).with_memory_owner(&owner);
    let seed = Sampling::control_seed(113, context).unwrap();
    let mut root = Sampling::randomness_root(Some(seed), context).unwrap();
    let draft = Sampling::draft_randomness_from_root(&mut root, context).unwrap();
    assert!(
        Sampling::draft_randomness_at(
            &draft,
            SpeculativeDraftRandomPosition::new(usize::MAX),
            context,
        )
        .is_err()
    );
    // The failure did not consume or mutate the reusable position root.
    let next =
        Sampling::draft_randomness_at(&draft, SpeculativeDraftRandomPosition::new(1), context)
            .unwrap();
    assert_eq!(key_words(&next).len(), 2);
    drop(owner);
    drop(root);
    drop(draft);
    assert_eq!(pool.unquoted_owner_count().unwrap(), 2);
    drop(next);
    crate::backend::submission_recovery::reap();
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
}

#[test]
fn entering_another_domain_checks_reservation_before_mutating_source_randomness() {
    use eredu_runtime::working_memory::InferenceExecutionIdentity;
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let pool_a = WorkingMemoryPool::new(4096, 0).unwrap();
    let pool_b = WorkingMemoryPool::new(4096, 0).unwrap();
    let owner_a = NativeMemoryOwner::acquire(&pool_a).unwrap();
    let context_a = SpeculativeExecutionStreams::single(&stream).with_memory_owner(&owner_a);
    let context_b = SpeculativeExecutionStreams::single(&stream).with_memory_pool(&pool_b);
    let seed = Sampling::seed_from_array(random::key(127).unwrap()).with_memory_owner(&owner_a);
    let mut root = Sampling::randomness_root(Some(seed), context_a).unwrap();
    let draft = Sampling::draft_randomness_from_root(&mut root, context_a).unwrap();
    let original = key_words(&root);
    let sampler = Sampling::new(GenerationSampler::new());
    let distribution = MlxSpeculativeDistribution::new(
        Array::from_slice(&[0.0f32, 1.0], &[1, 2]),
        NativeMemoryRetention::default(),
    );
    let reserved = pool_b
        .reserve(&InferenceExecutionIdentity::default(), &zero_admission())
        .unwrap();

    assert_reserved(&Sampling::target_randomness_from_root(&mut root, context_b).unwrap_err());
    assert_reserved(&Sampling::draft_randomness_from_root(&mut root, context_b).unwrap_err());
    assert_reserved(
        &Sampling::draft_randomness_at(&draft, SpeculativeDraftRandomPosition::new(2), context_b)
            .unwrap_err(),
    );
    assert_reserved(
        &sampler
            .sample_unit_interval(Some(&mut root), context_b)
            .unwrap_err(),
    );
    assert_reserved(
        &sampler
            .sample(
                &distribution,
                1.0,
                Some(&mut root),
                SamplingPlacement::Target,
                context_b,
            )
            .unwrap_err(),
    );
    assert_reserved(
        &sampler
            .sample(
                &distribution,
                0.0,
                None,
                SamplingPlacement::Target,
                context_b,
            )
            .unwrap_err(),
    );
    assert_eq!(key_words(&root), original);
    assert!(!root.memory_retention.covers_pool(&pool_b));
    assert_eq!(pool_b.unquoted_owner_count().unwrap(), 0);
    drop(reserved);

    let mut target = Sampling::target_randomness_from_root(&mut root, context_b).unwrap();
    let mut child = Sampling::target_randomness_from_root(&mut target, context_b).unwrap();
    for _ in 0..3 {
        sampler
            .sample_unit_interval(Some(&mut child), context_b)
            .unwrap();
        assert_eq!(pool_b.unquoted_owner_count().unwrap(), 1);
    }
    drop(owner_a);
    drop(root);
    drop(draft);
    drop(target);
    assert_eq!(pool_a.unquoted_owner_count().unwrap(), 1);
    assert_eq!(pool_b.unquoted_owner_count().unwrap(), 1);
    drop(child);
    crate::backend::submission_recovery::wait_for_retirement(|| {
        pool_a.unquoted_owner_count().unwrap() == 0 && pool_b.unquoted_owner_count().unwrap() == 0
    });
}

#[test]
fn sampling_merges_sampler_domain_before_acquiring_another_lease() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let pool_a = WorkingMemoryPool::new(4096, 0).unwrap();
    let pool_b = WorkingMemoryPool::new(4096, 0).unwrap();
    let owner_a = NativeMemoryOwner::acquire(&pool_a).unwrap();
    let owner_b = NativeMemoryOwner::acquire(&pool_b).unwrap();
    let context_a = SpeculativeExecutionStreams::single(&stream).with_memory_owner(&owner_a);
    let context_b = SpeculativeExecutionStreams::single(&stream).with_memory_pool(&pool_b);
    let seed = Sampling::seed_from_array(random::key(131).unwrap()).with_memory_owner(&owner_a);
    let mut root = Sampling::randomness_root(Some(seed), context_a).unwrap();
    let sampler = Sampling::new(GenerationSampler::new()).with_memory_owner(&owner_b);
    for _ in 0..3 {
        sampler
            .sample_unit_interval(Some(&mut root), context_b)
            .unwrap();
        assert_eq!(pool_b.unquoted_owner_count().unwrap(), 1);
    }
    drop(owner_a);
    drop(owner_b);
    drop(sampler);
    assert_eq!(pool_a.unquoted_owner_count().unwrap(), 1);
    assert_eq!(pool_b.unquoted_owner_count().unwrap(), 1);
    drop(root);
    crate::backend::submission_recovery::wait_for_retirement(|| {
        pool_a.unquoted_owner_count().unwrap() == 0 && pool_b.unquoted_owner_count().unwrap() == 0
    });
}
