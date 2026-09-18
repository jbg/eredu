//! Original RNG snapshots consume real completed keys and independently publish
//! copied backings through the same production copy and sampler callbacks.
use super::*;
use eredu_core::HostPreparationAuthority;
use eredu_runtime::DefaultSampler;

fn snapshot_host(
    sampler: &MlxSpeculativeSampling<DefaultSampler>,
    target: &MlxSpeculativeRandomState,
    draft: &MlxSpeculativeSeed,
    funding: &HostMetadataFunding,
) -> HostPreparationAuthority {
    let bytes = sampler
        .original_snapshot_metadata(Some(target), Some(draft))
        .unwrap()
        .checked_add(
            HostPreparationAuthority::retention_bytes::<HostMetadataFunding>().unwrap(),
        )
        .unwrap();
    funding.reserve_metadata(bytes).unwrap();
    HostPreparationAuthority::retain(funding.clone())
}
fn allocation(key: &OriginalNumericalKey) -> safemlx::AllocationIdentity {
    key.value()
        .value()
        .array
        .try_descriptor()
        .unwrap()
        .facts()
        .allocation()
        .unwrap()
        .identity()
}
#[test]
#[ignore = "requires native Metal execution"]
fn original_rng_snapshot_copies_alias_backing_and_keeps_request_identity() {
    let artifact = tempfile::tempdir().unwrap();
    crate::tests::distributed_pipeline_ring::write_fixture(artifact.path());
    let pool = crate::tests::support::test_utils::initialize_original_sources();
    let backend = admitted_backend(&pool);
    let initial = pool.used_bytes().unwrap();
    let (target_config, draft_config, selected) = source_configs(&backend, artifact.path());
    let target = load(&backend, &target_config);
    let draft = load(&backend, &draft_config);
    let loaded = pool.used_bytes().unwrap();
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
    let escaped = {
        let pair = AutoregressiveSourcePair::prepare(
            target.original_model_source().unwrap(),
            draft.original_model_source().unwrap(),
            &schedule,
            &pool,
            REQUEST_CEILING,
        )
        .unwrap();
        let environment = backend.original_copy_environment().unwrap();
        let context = SpeculativeExecutionStreams::single(backend.stream())
            .with_original_sources(&pair, &environment)
            .unwrap();
        let sampler = MlxSpeculativeSampling::prepare(DefaultSampler, context, None).unwrap();
        let root = create_key(SEED, context).unwrap();
        // A selected positional key owns the full split-table backing. Copying
        // its logical two words must produce a separate completed allocation.
        let positional = key_at(&root, SpeculativeDraftRandomPosition::new(3), context).unwrap();
        let mut target_key = MlxSpeculativeRandomState {
            value: KeyValue::Original(positional),
            memory_retention: NativeMemoryRetention::default(),
        };
        let draft_key = MlxSpeculativeSeed {
            value: KeyValue::Original(root),
            memory_retention: NativeMemoryRetention::default(),
        };
        let source_words = key_words(target_key.value.original().unwrap());
        let source_id = allocation(target_key.value.original().unwrap());
        let draft_id = allocation(draft_key.value.original().unwrap());
        let physical = sampler
            .original_snapshot_bytes(Some(&target_key), Some(&draft_key))
            .unwrap();
        assert!(
            physical
                > sampler
                    .original_snapshot_metadata(Some(&target_key), Some(&draft_key))
                    .unwrap() as u64
        );
        let host = snapshot_host(&sampler, &target_key, &draft_key, pair.metadata_funding());
        let (saved_sampler, saved_target, saved_draft) = sampler
            .copy_snapshot(Some(&target_key), Some(&draft_key), host)
            .unwrap();
        let mut saved_target = saved_target.unwrap();
        let saved_draft = saved_draft.unwrap();
        assert_eq!(
            key_words(saved_target.value.original().unwrap()),
            source_words
        );
        assert_eq!(
            key_words(saved_draft.value.original().unwrap()),
            key_words(draft_key.value.original().unwrap())
        );
        assert_ne!(
            allocation(saved_target.value.original().unwrap()),
            source_id
        );
        assert_ne!(allocation(saved_draft.value.original().unwrap()), draft_id);
        assert_ne!(
            allocation(saved_target.value.original().unwrap()),
            allocation(saved_draft.value.original().unwrap())
        );
        // Recursively snapshot an already published copy. This takes the existing
        // registered-source route, rather than reusing its ancestor's Q as origin.
        let host = snapshot_host(
            &saved_sampler,
            &saved_target,
            &saved_draft,
            pair.metadata_funding(),
        );
        let (again_sampler, again_target, again_draft) = saved_sampler
            .copy_snapshot(Some(&saved_target), Some(&saved_draft), host)
            .unwrap();
        let again_target = again_target.unwrap();
        let again_draft = again_draft.unwrap();
        assert_ne!(
            allocation(again_target.value.original().unwrap()),
            allocation(saved_target.value.original().unwrap())
        );
        assert_eq!(
            key_words(again_target.value.original().unwrap()),
            source_words
        );
        let a = sample_unit_interval(target_key.value.original_mut().unwrap(), context).unwrap();
        let b = sample_unit_interval(saved_target.value.original_mut().unwrap(), context).unwrap();
        assert_eq!(a, b);
        assert_eq!(
            key_words(target_key.value.original().unwrap()),
            key_words(saved_target.value.original().unwrap())
        );
        assert_eq!(
            key_words(again_target.value.original().unwrap()),
            source_words
        );
        // Deliberately share H across two actual request issuances. The closed
        // request identity still refuses the foreign key before copying anything.
        let foreign = AutoregressiveSourcePair::prepare_funded(
            target.original_model_source().unwrap(),
            draft.original_model_source().unwrap(),
            &schedule,
            &pool,
            REQUEST_CEILING,
            pair.metadata_funding().clone(),
        )
        .unwrap();
        let foreign_context = SpeculativeExecutionStreams::single(backend.stream())
            .with_original_sources(&foreign, &environment)
            .unwrap();
        let foreign_key = MlxSpeculativeSeed {
            value: KeyValue::Original(create_key(SEED, foreign_context).unwrap()),
            memory_retention: NativeMemoryRetention::default(),
        };
        assert!(
            sampler
                .original_snapshot_metadata(Some(&target_key), Some(&foreign_key))
                .is_none()
        );
        let host = snapshot_host(&sampler, &target_key, &draft_key, pair.metadata_funding());
        assert!(
            sampler
                .copy_snapshot(Some(&target_key), Some(&foreign_key), host)
                .is_err()
        );
        assert_eq!(
            key_words(again_target.value.original().unwrap()),
            source_words
        );
        drop((
            foreign_key,
            foreign,
            target_key,
            draft_key,
            saved_target,
            saved_draft,
            saved_sampler,
            sampler,
            again_sampler,
            again_draft,
        ));
        assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
        drop(pair);
        reclaim();
        assert!(pool.used_bytes().unwrap() > loaded);
        assert_eq!(
            key_words(again_target.value.original().unwrap()),
            source_words
        );
        again_target
    };
    drop(escaped);
    settle(&pool, loaded);
    drop((target, draft));
    drop(schedule);
    drop((target_config, draft_config, selected));
    settle(&pool, initial);
}
