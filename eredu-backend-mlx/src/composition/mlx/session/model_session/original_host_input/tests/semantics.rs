use super::*;
use eredu_runtime::input::OriginalModelInputBackend as _;
#[test]
fn original_semantics_qwen_vl_upload_preserves_source_and_requires_execution_producer() {
    if !crate::composition::mlx::session::model_session::original_host_input::tests::admitted::enter(
    ) {
        return;
    }
    let root = tempfile::tempdir().unwrap();
    crate::tests::distributed_pipeline_ring::write_qwen3_vl_component_fixture(
        root.path(),
        false,
        false,
    );
    same_mode(root.path(), 64, 2);
}
#[test]
fn original_semantics_conditional_qwen_upload_preserves_source_and_requires_execution_producer() {
    if !crate::composition::mlx::session::model_session::original_host_input::tests::admitted::enter(
    ) {
        return;
    }
    let root = tempfile::tempdir().unwrap();
    crate::tests::distributed_pipeline_ring::write_qwen35_conditional_component_fixture(
        root.path(),
        false,
    );
    same_mode(root.path(), 16, 2);
}
#[test]
fn original_semantic_upload_requires_complete_execution_producer() {
    if !crate::composition::mlx::session::model_session::original_host_input::tests::admitted::enter(
    ) {
        return;
    }
    core_driver(true);
}
#[test]
fn compiled_source_graph_pool_equal_content_and_current_revision_reject_before_upload() {
    if !crate::composition::mlx::session::model_session::original_host_input::tests::admitted::enter(
    ) {
        return;
    }
    let _hooks = Hooks;
    let root = tempfile::tempdir().unwrap();
    crate::tests::distributed_pipeline_ring::write_qwen3_vl_component_fixture(
        root.path(),
        false,
        false,
    );
    let pool = crate::tests::support::test_utils::initialize_original_sources();
    let input = source(&pool, 64);
    let equal = source(&pool, 64);
    assert_eq!(input.content_digest(), equal.content_digest());
    assert!(!input.same_source(&equal));
    let backend = admitted::backend(&pool);
    let config = cold_config(&backend, root.path(), 0);
    let independent = cold_config(&backend, root.path(), 0);
    let compile = || {
        config
            .prepared_sources()
            .plan_original_media_semantics(&input)
            .unwrap()
            .compile(&pool)
            .unwrap()
    };
    let wrong_graph = independent
        .prepared_sources()
        .plan_original_media_semantics(&input)
        .unwrap()
        .compile(&pool)
        .unwrap();
    let wrong_pool = compile();
    let wrong_source = compile();
    let current = compile();
    let stale = compile();
    let executing = compile();
    let completed = MlxPreparedInputMaterializer::prepare()
        .unwrap()
        .model_input_plan(&executing)
        .unwrap()
        .materialize(&pool)
        .unwrap();
    let model = backend.prepare_model_borrowed(&config).unwrap();
    let mut runtime = ModelRuntime::from_prepared(backend, model).unwrap();
    SLOTS.set(0);
    assert!(matches!(
        MlxModelInput::from_original_host_input_with_semantics(&runtime, wrong_graph),
        Err(MlxHostInputUploadError::Semantics(_))
    ));
    assert_eq!(SLOTS.get(), 0);
    let foreign = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
    assert!(matches!(
        upload_with_semantics(&foreign, &input, Some((&runtime, wrong_pool))),
        Err(MlxHostInputUploadError::Semantics(_))
    ));
    assert_eq!(foreign.unquoted_owner_count().unwrap(), 0);
    assert_eq!(SLOTS.get(), 0);
    // Actual native typed binder checks the real source owner before the first
    // lowerer tensor call, even when all fingerprints and values are identical.
    assert!(matches!(
        upload_with_semantics(&pool, &equal, Some((&runtime, wrong_source))),
        Err(MlxHostInputUploadError::Operation(_))
    ));
    assert_eq!(SLOTS.get(), 0);
    let prompt = MlxModelInput::from_original_host_input_with_semantics(&runtime, current)
        .unwrap()
        .with_prefill_chunk_positions(2.try_into().unwrap());
    assert_eq!(SLOTS.get(), 5);
    assert!(prompt.original_media.is_some());
    // Every actual payload/metadata slot is complete before this packet escapes.
    prompt.with_borrowed(|i| {
        for p in i.parts {
            for v in std::iter::once(p.payload().value()).chain(p.metadata().values()) {
                assert!(v.try_allocation_info().unwrap().is_some());
            }
        }
    });
    admitted::rejects_upload(&mut runtime, &prompt);
    drop(prompt);
    safemlx::memory::clear_cache();
    safemlx::reclaim_allocation_owners();
    runtime
        .session()
        .payload
        .model
        .erased()
        .prepare_completed_media_binding_fixture()
        .unwrap();
    let executing = completed.bind(&runtime, executing).unwrap();
    admitted::collect(&mut runtime, executing, 1, true);
    SLOTS.set(0);
    let rejected = MlxModelInput::from_original_host_input_with_semantics(&runtime, stale);
    assert!(
        matches!(&rejected, Err(MlxHostInputUploadError::Semantics(cause))
        if cause.accounting_failure() == Some(&WorkingMemoryError::ReservedWorkActive)),
        "retained admitted state excludes an ordinary upload before any native slot: {rejected:?}"
    );
    assert_eq!(SLOTS.get(), 0);
}
#[test]
fn independent_original_source_retains_semantics_and_executes_actual_copied_slots() {
    if !crate::composition::mlx::session::model_session::original_host_input::tests::admitted::enter(
    ) {
        return;
    }
    pending_copy(false);
}
pub(super) fn pending_copy(native_source: bool) {
    if !crate::composition::mlx::session::model_session::original_host_input::tests::admitted::enter(
    ) {
        return;
    }
    use eredu_core::PendingTextInput;
    use eredu_runtime::execution_control::TextSnapshotBackend;
    let root = tempfile::tempdir().unwrap();
    crate::tests::distributed_pipeline_ring::write_qwen3_vl_component_fixture(
        root.path(),
        false,
        false,
    );
    let pool = crate::tests::support::test_utils::initialize_original_sources();
    let source = source(&pool, 64);
    let backend = admitted::backend(&pool);
    let config = cold_config(&backend, root.path(), 0);
    let semantic = config
        .prepared_sources()
        .plan_original_media_semantics(&source)
        .unwrap()
        .compile(&pool)
        .unwrap();
    let copied_semantics = config
        .prepared_sources()
        .plan_original_media_semantics(&source)
        .unwrap()
        .compile(&pool)
        .unwrap();
    let materializer = MlxPreparedInputMaterializer::prepare().unwrap();
    if native_source {
        let leaves = materializer
            .plan(&source)
            .unwrap()
            .materialize(&pool)
            .unwrap();
        assert_eq!(leaves.source().content_digest(), source.content_digest());
        drop(leaves);
    }
    let completed = materializer
        .model_input_plan(&semantic)
        .unwrap()
        .materialize(&pool)
        .unwrap();
    let copied = materializer
        .model_input_plan(&copied_semantics)
        .unwrap()
        .materialize(&pool)
        .unwrap();
    let model = backend.prepare_model_borrowed(&config).unwrap();
    let mut runtime = ModelRuntime::from_prepared(backend, model).unwrap();
    runtime
        .session()
        .payload
        .model
        .erased()
        .prepare_completed_media_binding_fixture()
        .unwrap();
    let prompt = completed
        .bind(&runtime, semantic)
        .unwrap()
        .with_prefill_chunk_positions(2.try_into().unwrap());
    let copied = copied
        .bind(&runtime, copied_semantics)
        .unwrap()
        .with_prefill_chunk_positions(2.try_into().unwrap());
    let alias = prompt.clone();
    assert!(alias.original_media.is_some());
    let raw = prompt.with_borrowed(|view| MlxModelInput::from(view));
    assert!(raw.original_media.is_none());
    let replacement = baseline(&pool, 64);
    fn assert_relabel_clears_packet<'a>(
        mut view: crate::backend::runtime::media::input::ModelInput<'a>,
        parts: &'a [crate::backend::runtime::media::InputPart],
    ) {
        view.parts = parts;
        let relabelled = MlxModelInput::from(view);
        assert!(relabelled.original_media.is_none());
    }
    replacement.with_borrowed(|replacement| {
        prompt.with_borrowed(|view| assert_relabel_clears_packet(view, replacement.parts))
    });
    drop(raw);
    drop(replacement);
    let relabelled = alias
        .with_semantic_content_fingerprint("ordinary relabelled content")
        .unwrap();
    assert!(relabelled.original_media.is_none());
    drop(relabelled);
    assert!(copied.original_media.is_some());
    assert_eq!(
        MlxBackend::original_model_input_semantics(&prompt)
            .unwrap()
            .layout()
            .positions(),
        9
    );
    assert_eq!(
        MlxBackend::original_model_input_semantics(&copied)
            .unwrap()
            .layout()
            .positions(),
        9
    );
    assert_eq!(
        MlxBackend::continuation_input_tokens(Some(PendingTextInput::Prefill(&copied)), 3),
        Some(11)
    );
    prompt.with_borrowed(|a| {
        copied.with_borrowed(|b| {
            assert_eq!(a.parts.len(), b.parts.len());
            for (a, b) in a.parts.iter().zip(b.parts) {
                assert_eq!(a.extents(), b.extents());
                assert_eq!(a.modality(), b.modality());
                assert_eq!(a.payload().kind(), b.payload().kind());
                for (a, b) in std::iter::once(a.payload().value())
                    .chain(a.metadata().values())
                    .zip(std::iter::once(b.payload().value()).chain(b.metadata().values()))
                {
                    assert_eq!(array_values(a), array_values(b));
                    assert_ne!(
                        a.allocation_info().unwrap().unwrap().identity(),
                        b.allocation_info().unwrap().unwrap().identity()
                    );
                }
            }
        })
    });
    drop(prompt);
    drop(source);
    safemlx::memory::clear_cache();
    safemlx::reclaim_allocation_owners();
    let values = admitted::collect(&mut runtime, copied, 1, true).0;
    drop((runtime, config));
    crate::backend::submission_recovery::wait_for_retirement(|| {
        crate::backend::ordinary_retirement::reclaim_all();
        safemlx::memory::clear_cache();
        safemlx::reclaim_allocation_owners();
        pool.unquoted_owner_count().unwrap() == 0
    });
    let expected = admitted::run(root.path(), 0, 64, true, true).0;
    close(&values[0], &expected[0]);
}
#[test]
fn compiled_cancellation_before_and_inside_media_matches_ordinary_prefix_and_cached_decode() {
    if !crate::composition::mlx::session::model_session::original_host_input::tests::admitted::enter(
    ) {
        return;
    }
    compare_cancellation(2, &[0], &[1, 2]);
}
pub(crate) fn compare_cancellation(source_mode: usize, modes: &[usize], boundaries: &[usize]) {
    use eredu_core::{
        ControlledTextGeneration, GenerationSequenceRequest, TextGeneration, TextGenerationInput,
        TokenFilter,
    };
    assert!(matches!(source_mode, 2 | 4));
    if !crate::composition::mlx::session::model_session::original_host_input::tests::admitted::enter(
    ) {
        return;
    }
    for conditional in [false, true] {
        let root = tempfile::tempdir().unwrap();
        if conditional {
            crate::tests::distributed_pipeline_ring::write_qwen35_conditional_component_fixture(
                root.path(),
                false,
            );
        } else {
            crate::tests::distributed_pipeline_ring::write_qwen3_vl_component_fixture(
                root.path(),
                false,
                false,
            );
        }
        for &mode in modes {
            for &boundary in boundaries {
                let mut reports = Vec::new();
                for manual in [false, true] {
                    let (mut runtime, prompt) =
                        admitted::prepare(root.path(), mode, if conditional { 16 } else { 64 });
                    let prompt = prompt.with_prefill_chunk_positions(2.try_into().unwrap());
                    let before = runtime.session().payload.model.erased().state_snapshot();
                    let cancel = eredu_core::GenerationCancellationToken::new();
                    if boundary == 0 {
                        cancel.cancel();
                    }
                    let ((), roots) = crate::tests::support::media_completion::observe(
                        Some((cancel.clone(), boundary)),
                        || {
                            if manual {
                                let mut generation =
                                    ControlledTextGeneration::from_input_with_sequence(
                                        &mut runtime,
                                        TextGenerationInput::OriginalPrepared(prompt),
                                        admitted::config(4),
                                        admitted::All,
                                        None,
                                        GenerationSequenceRequest::new(4, &[]),
                                    )
                                    .unwrap();
                                assert!(generation.next_cancellable(&cancel).is_none());
                            } else {
                                let mut generation = TextGeneration::from_input_with_sequence(
                                    &mut runtime,
                                    TextGenerationInput::OriginalPrepared(prompt),
                                    admitted::config(4),
                                    TokenFilter::All,
                                    None,
                                    GenerationSequenceRequest::new(4, &[]),
                                )
                                .unwrap();
                                assert!(generation.next_cancellable(&cancel).is_none());
                            }
                        },
                    );
                    assert_eq!(roots.len(), boundary * 2);
                    if boundary == 0 {
                        assert_eq!(
                            runtime.session().payload.model.erased().state_snapshot(),
                            before
                        );
                        continue;
                    }
                    assert!(roots
                        .iter()
                        .filter(|r| r.after)
                        .all(|r| r.ready.iter().all(|v| *v)));
                    let partial = admitted::snapshot(&runtime);
                    // Cancellation terminates that source. The completed prefix is
                    // extended by a fresh, authenticated one-token prefill request.
                    let continued = admitted::continue_text(&mut runtime);
                    reports.push((partial, continued));
                }
                if boundary != 0 {
                    assert_eq!(reports[0], reports[1]);
                }
            }
        }
    }
}
