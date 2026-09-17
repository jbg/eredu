//! Real admitted model sources and the production numerical compiler. No test
//! account, raw numerical-value constructor or test-sized native arena is used.
use super::*;
use crate::composition::mlx::speculative::autoregressive::AutoregressiveSourcePair;
use crate::backend::{
    MlxAcceleratorFamily, MlxBackend, MlxDeviceIdentity,
    managed_memory::gpu_stream::PreparedExecutionStreams,
    nn::shared::MlxNeuralBackend,
    random::{RandomState, split_key_at},
};
use crate::composition::mlx::{
    loading::{MlxModelConfig, MlxPreparationMechanisms},
    session::MlxModelSession,
};
use eredu_core::{
    DevicePlan, DraftPlacementPlan, DraftingPlan, ExecutionPlan, ExternalDraftArtifact,
    ModelLoadingBackend, SpeculativeConfig, SpeculativeDraftRandomPosition,
    SpeculativeSchedulerOptions, TokenizerCompatibilityProof,
};
use eredu_runtime::{
    speculative::autoregressive::AutoregressiveSchedulePlan, working_memory::WorkingMemoryPool,
};
use std::num::{NonZeroU64, NonZeroUsize};

mod token_input;
mod cpu_normalize;
mod cpu_greedy;
mod cpu_token_filter;
mod cpu_sampling_filters;
mod cpu_capture;
mod cpu_probability;
mod cpu_row;
mod cpu_key;
mod cpu_split;
mod cpu_uniform;
mod cpu_difference;
mod cpu_categorical;
mod registered_range;

const SEED: u64 = 0x1234_5678_9abc_def0;
// This ceiling is compared by the existing account. Each phase derives its
// physical, Graph, Record, pipeline and control requirements before admission.
pub(in crate::composition::mlx::speculative) const REQUEST_CEILING: u64 = 1 << 30;

fn reclaim() {
    MlxNeuralBackend::reclaim_retired_resources();
    safemlx::reclaim_allocation_owners();
}
pub(in crate::composition::mlx::speculative) fn settle(pool: &WorkingMemoryPool, bytes: u64) {
    crate::backend::submission_recovery::wait_for_retirement(|| {
        reclaim();
        pool.used_bytes().unwrap() == bytes && pool.unquoted_owner_count().unwrap() == 0
    });
    assert_eq!(pool.used_bytes().unwrap(), bytes);
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
}
pub(in crate::composition::mlx::speculative) fn admitted_backend(pool: &WorkingMemoryPool) -> MlxBackend<'static> {
    let streams = PreparedExecutionStreams::for_factory(pool)
        .unwrap()
        .unwrap();
    let identity = MlxDeviceIdentity::from_realized_device(
        &safemlx::Device::new(safemlx::DeviceType::Gpu, 0),
        Some(MlxAcceleratorFamily::Metal),
    )
    .unwrap();
    MlxBackend::for_prepared_execution_plan(streams, identity)
}
pub(in crate::composition::mlx::speculative) fn load(backend: &MlxBackend<'_>, config: &MlxModelConfig) -> MlxModelSession {
    let prepared = backend.prepare_model_borrowed(config).unwrap();
    let session = MlxModelSession::from_model(
        prepared.into_inner(),
        config.prepared_sources().selected().session_capabilities(),
    )
    .unwrap();
    crate::backend::submission_recovery::wait_for_retirement(|| {
        reclaim();
        backend.memory_pool().unquoted_owner_count().unwrap() == 0
    });
    session
}
// Post-completion observations are test oracles, never producers or grants.
// The numerical compiler has already waited and drained its exact Record.
fn key_words(key: &OriginalNumericalKey) -> [u32; 2] {
    ordinary_words(&key.value().value().array)
}
fn ordinary_words(array: &Array) -> [u32; 2] {
    array
        .evaluated()
        .unwrap()
        .as_slice::<u32>()
        .try_into()
        .unwrap()
}

pub(in crate::composition::mlx::speculative) fn source_configs(
    backend: &MlxBackend<'_>,
    path: &std::path::Path,
) -> (
    MlxModelConfig,
    MlxModelConfig,
    eredu_runtime::SelectedSpeculativeRealization,
) {
    let target_inspection =
        eredu_core::inspect_artifact(path, backend.configuration_resolver()).unwrap();
    let target_config = eredu_core::prepare_inspected_model_config(
        backend,
        target_inspection.clone(),
        crate::MlxLoadRequest::default(),
    )
    .unwrap();
    let plan = ExecutionPlan::fully_resident(DevicePlan::new("mlx", "gpu:0").unwrap())
        .with_drafting(DraftingPlan::External {
            model: path.display().to_string(),
            placement: DraftPlacementPlan::Target,
            max_draft_tokens: 1,
            lookahead: false,
            adaptive_lookahead: false,
        });
    let draft = eredu_architectures::prepare_execution_plan_draft(
        &plan,
        &target_inspection,
        ExternalDraftArtifact {
            preparation: eredu_architectures::prepare_external_draft(path).unwrap(),
            // Identical caller-supplied tokenizer mapping for both fixtures.
            tokenizer_compatibility: TokenizerCompatibilityProof::prove([7; 32], [7; 32]).unwrap(),
        },
        &MlxPreparationMechanisms::new(
            &crate::composition::mlx::replicated_text::GROUPED_OPERATION_CAPABILITIES,
        ),
        |descriptor, transforms| {
            if transforms
                && crate::composition::mlx::replicated_text::supports_transform(descriptor)
            {
                Some(eredu_runtime::WeightLoweringKind::Transform)
            } else if !transforms
                && crate::composition::mlx::replicated_text::supports_direct(descriptor)
            {
                Some(eredu_runtime::WeightLoweringKind::Direct)
            } else {
                None
            }
        },
    )
    .unwrap();
    let eredu_architectures::PreparedExternalDraft::Autoregressive(draft) = draft.preparation
    else {
        panic!("fixture must select an independent ordinary decoder");
    };
    let (sources, selected, _) = draft.into_parts();
    let draft_config = MlxModelConfig {
        sources,
        rank_context: None,
    };
    (target_config, draft_config, selected)
}

#[test]
#[ignore = "requires native Metal execution"]
fn original_numerical_seed_split_position_and_uniform_use_real_sources() {
    let artifact = tempfile::tempdir().unwrap();
    crate::tests::distributed_pipeline_ring::write_fixture(artifact.path());
    let pool = crate::tests::support::test_utils::initialize_original_sources();
    let backend = admitted_backend(&pool);
    let initial = pool.used_bytes().unwrap();
    let (target_config, draft_config, selected) = source_configs(&backend, artifact.path());
    let target = load(&backend, &target_config);
    let draft = load(&backend, &draft_config);
    let loaded = pool.used_bytes().unwrap();
    assert!(loaded > initial);
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
    let config = SpeculativeConfig {
        max_tokens: 2,
        max_draft_tokens: 1,
        temperature: 0.7,
        eos_token_ids: Vec::new(),
    };
    let schedule = AutoregressiveSchedulePlan::new(
        &selected,
        NonZeroUsize::new(1).unwrap(),
        NonZeroU64::new(2).unwrap(),
        NonZeroU64::new(32).unwrap(),
        &config,
        SpeculativeSchedulerOptions::default(),
    )
    .unwrap();
    let (observed_keys, draws) = {
        let pair = AutoregressiveSourcePair::prepare(
            target.original_model_source().unwrap(),
            draft.original_model_source().unwrap(),
            &schedule,
            &pool,
            REQUEST_CEILING,
        )
        .unwrap();
        let environment = backend.original_copy_environment().unwrap();
        pair.request().validate_execution(target.original_model_source().unwrap().erased().inference_execution_identity()).unwrap();
        assert!(pair.request().validate_execution(&eredu_runtime::working_memory::InferenceExecutionIdentity::default()).is_err());
        // The common numerical source retains the actual AR-tagged request, but
        // no AR model occurrence is installed. Embedded will use this same
        // numerical binding with its own exact tagged request and role driver.
        let context = SpeculativeExecutionStreams::single(backend.stream())
            .with_original_numerical_sources(pair.numerical_sources(), &environment)
            .unwrap();
        assert!(context.original_execution().is_none());
        assert!(context.original_numerical().is_some());
        eprintln!("original numerical: create-key");
        let zero = create_key(0, context).unwrap();
        assert_eq!(key_words(&zero), [0, 0]);
        let root = create_key(SEED, context).unwrap();
        let mut state = root.clone();
        let initial_key = key_words(&root);
        eprintln!("original numerical: sequential split");
        let next = next_key(&mut state, context).unwrap_or_else(|error| {
            let mut source: &dyn std::error::Error = &error;
            eprintln!("next key failed: {source}");
            while let Some(next) = source.source() {
                eprintln!("caused by: {next}");
                source = next;
            }
            panic!("next key failed: {error:?}");
        });
        let after_split = key_words(&state);
        let returned_split = key_words(&next);
        assert_ne!(after_split, returned_split);
        eprintln!("original numerical: positional key");
        let copied_root=copy_key_to(&root,SamplingPlacement::Target,SamplingPlacement::Draft,context).unwrap();
        assert_eq!(key_words(&copied_root),initial_key);
        assert_ne!(root.value().value().array.try_allocation_info().unwrap().unwrap().identity(),
            copied_root.value().value().array.try_allocation_info().unwrap().unwrap().identity());
        let at = key_at(&copied_root, SpeculativeDraftRandomPosition::new(3), context).unwrap();
        drop(copied_root);
        let positional = key_words(&at);
        assert_eq!(key_words(&root), initial_key);
        eprintln!("original numerical: sequential uniform");
        let draws = [
            sample_unit_interval(&mut state, context).unwrap(),
            sample_unit_interval(&mut state, context).unwrap(),
        ];
        assert!(draws.iter().all(|x| (0.0..1.0).contains(x)));
        let after_uniform = key_words(&state);
        let observed = [
            initial_key,
            after_split,
            returned_split,
            positional,
            after_uniform,
        ];

        // Closing issuance refuses before publishing a new key. A refused
        // phase retains its typed error and does not mutate the old state.
        pair.request().close().unwrap();
        let before_failure = key_words(&state);
        let error = next_key(&mut state, context).unwrap_err();
        assert_eq!(key_words(&state), before_failure);
        drop(error);
        let charged = pool.used_bytes().unwrap();
        drop((zero, root, next, at));
        reclaim();
        assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
        // Retired native values release their actual accounts. Closing issuance
        // remains monotone and cannot be undone by reclaiming those owners.
        assert!(pool.used_bytes().unwrap() < charged);
        let retired_key = key_words(&state);
        let refused = next_key(&mut state, context).unwrap_err();
        assert_eq!(key_words(&state), retired_key);
        drop(refused);
        drop(pair);
        reclaim();
        // An escaped completed key retains its own accepted account, even when
        // the source-pair wrapper and the other phases have been dropped.
        assert!(pool.used_bytes().unwrap() > loaded);
        drop(state);
        (observed, draws)
    };
    settle(&pool, loaded);

    // The ordinary oracle starts after every original request/value retires.
    // The same split/table/uniform primitives use the same actual supplied stream.
    let stream = backend.stream();
    let mut reference = RandomState::with_seed(SEED).unwrap();
    assert_eq!(ordinary_words(reference.as_array()), observed_keys[0]);
    let root = reference.as_array().clone();
    let next = reference.next_key(stream).unwrap();
    assert_eq!(ordinary_words(reference.as_array()), observed_keys[1]);
    assert_eq!(ordinary_words(&next), observed_keys[2]);
    assert_eq!(
        ordinary_words(&split_key_at(&root, 3, stream).unwrap()),
        observed_keys[3]
    );
    for actual in draws {
        let expected = reference.uniform_unit_interval(stream).unwrap();
        assert_eq!(expected.evaluated().unwrap().as_slice::<f32>(), &[actual]);
    }
    assert_eq!(ordinary_words(reference.as_array()), observed_keys[4]);
    drop((root, next, reference, target, draft));
    drop(schedule);
    drop((target_config, draft_config, selected));
    settle(&pool, initial);
}

mod categorical;

mod snapshot;

mod adaptive;

mod cpu_copy;
mod cpu_pending_input;
