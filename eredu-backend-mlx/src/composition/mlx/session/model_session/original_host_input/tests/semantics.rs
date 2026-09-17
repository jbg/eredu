use super::*;
#[test]
fn original_semantics_qwen_vl_match_full_state_across_all_residencies_and_cached_decodes() {
    let root = tempfile::tempdir().unwrap();
    crate::tests::distributed_pipeline_ring::write_qwen3_vl_component_fixture(
        root.path(),
        false,
        false,
    );
    same_mode(root.path(), 64, 2);
}
#[test]
fn original_semantics_conditional_qwen_match_full_state_across_all_residencies_and_cached_decodes()
{
    let root = tempfile::tempdir().unwrap();
    crate::tests::distributed_pipeline_ring::write_qwen35_conditional_component_fixture(
        root.path(),
        false,
    );
    same_mode(root.path(), 16, 2);
}
#[test]
fn original_semantic_upload_enters_actual_core_prepared_iterator_and_manual_driver() {
    core_driver(true);
}
#[test]
fn compiled_source_graph_pool_equal_content_and_current_revision_reject_before_upload() {
    let _hooks = Hooks;
    let root = tempfile::tempdir().unwrap();
    crate::tests::distributed_pipeline_ring::write_qwen3_vl_component_fixture(
        root.path(),
        false,
        false,
    );
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let input = source(&pool, 64);
    let equal = source(&pool, 64);
    assert_eq!(input.content_digest(), equal.content_digest());
    assert!(!input.same_source(&equal));
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let backend = MlxBackend::new(&stream, &stream).with_memory_pool(pool.clone());
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
    let model = backend.prepare_model_borrowed(&config).unwrap();
    let mut runtime = ModelRuntime::from_prepared(backend, model).unwrap();
    SLOTS.set(0);
    assert!(matches!(
        MlxModelInput::from_original_host_input_with_semantics(&runtime, wrong_graph),
        Err(MlxHostInputUploadError::Semantics(_))
    ));
    assert_eq!(SLOTS.get(), 0);
    let foreign = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
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
    let first = runtime.prefill(prompt).unwrap();
    first.completion.wait().unwrap();
    drop(first);
    SLOTS.set(0);
    assert!(matches!(
        MlxModelInput::from_original_host_input_with_semantics(&runtime, stale),
        Err(MlxHostInputUploadError::Operation(_))
    ));
    assert_eq!(SLOTS.get(), 0);
}
#[test]
fn independent_pending_copy_clears_compiled_packet_and_executes_actual_copied_slots() {
    pending_copy(false);
}
pub(super) fn pending_copy(native_source: bool) {
    use eredu_core::{PendingTextInput, PreparedControlInputBackend};
    use eredu_runtime::execution_control::TextSnapshotBackend;
    let root = tempfile::tempdir().unwrap();
    crate::tests::distributed_pipeline_ring::write_qwen3_vl_component_fixture(
        root.path(),
        false,
        false,
    );
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let source = source(&pool, 64);
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let backend = MlxBackend::new(&stream, &stream).with_memory_pool(pool.clone());
    let config = cold_config(&backend, root.path(), 0);
    let semantic = config
        .prepared_sources()
        .plan_original_media_semantics(&source)
        .unwrap()
        .compile(&pool)
        .unwrap();
    let native = if native_source {
        Some(
            MlxPreparedInputMaterializer::prepare()
                .unwrap()
                .plan(&source)
                .unwrap()
                .materialize(&pool)
                .unwrap(),
        )
    } else {
        None
    };
    let model = backend.prepare_model_borrowed(&config).unwrap();
    let mut runtime = ModelRuntime::from_prepared(backend, model).unwrap();
    let prompt = match native {
        Some(native) => {
            MlxModelInput::from_original_native_input_with_semantics(&runtime, native, semantic)
        }
        None => MlxModelInput::from_original_host_input_with_semantics(&runtime, semantic),
    }
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
    let prepared = MlxBackend::prepare_control_input(&runtime, prompt).unwrap();
    let (prompt, attribution) = MlxBackend::consume_control_input(&runtime, prepared).unwrap();
    assert_eq!(attribution.attribution().decoder_positions, 9);
    let Some(PendingTextInput::Prefill(copied)) =
        MlxBackend::copy_pending_input(&mut runtime, Some(PendingTextInput::Prefill(&prompt)))
            .unwrap()
    else {
        panic!("actual pending media copy")
    };
    assert!(copied.original_media.is_none());
    assert_eq!(copied.cache_identity(), prompt.cache_identity());
    assert_eq!(copied.controlled_decoder_positions(), Some(9));
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
    drop(attribution);
    input::reset_original_semantic_preparations();
    let result = runtime.prefill(copied).unwrap();
    result.completion.wait().unwrap();
    assert_eq!(
        input::original_semantic_preparations(),
        1,
        "copied arrays use actual ordinary admission"
    );
    let values = array_values(result.output.logits().unwrap().as_array());
    let expected = run_mode(root.path(), 0, 0, 64).0;
    close(&values, &expected[0]);
}
#[test]
fn compiled_cancellation_before_and_inside_media_matches_ordinary_prefix_and_cached_decode() {
    compare_cancellation(2, &[0], &[1, 2]);
}
pub(crate) fn compare_cancellation(source_mode: usize, modes: &[usize], boundaries: &[usize]) {
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
        let hidden = if conditional { 16 } else { 64 };
        for &mode in modes {
            for &boundary in boundaries {
                let mut reports = Vec::new();
                for compiled in [false, true] {
                    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
                    let source = source(&pool, hidden);
                    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
                    let backend = MlxBackend::new(&stream, &stream).with_memory_pool(pool.clone());
                    let config = cold_config(&backend, root.path(), mode);
                    let semantic = compiled.then(|| {
                        config
                            .prepared_sources()
                            .plan_original_media_semantics(&source)
                            .unwrap()
                            .compile(&pool)
                            .unwrap()
                    });
                    let full = (compiled && source_mode == 4).then(|| {
                        MlxPreparedInputMaterializer::prepare()
                            .unwrap()
                            .model_input_plan(semantic.as_ref().unwrap())
                            .unwrap()
                            .materialize(&pool)
                            .unwrap()
                    });
                    let model = backend.prepare_model_borrowed(&config).unwrap();
                    let mut runtime = ModelRuntime::from_prepared(backend, model).unwrap();
                    if full.is_some() {
                        runtime
                            .session()
                            .payload
                            .model
                            .erased()
                            .prepare_completed_media_binding_fixture()
                            .unwrap();
                    }
                    let prompt = match semantic {
                        Some(s) => match full {
                            Some(full) => Ok(full.bind(&runtime, s).unwrap()),
                            None => {
                                MlxModelInput::from_original_host_input_with_semantics(&runtime, s)
                            }
                        },
                        None => MlxModelInput::from_original_host_input(&runtime, &source),
                    }
                    .unwrap()
                    .with_prefill_chunk_positions(2.try_into().unwrap());
                    let cancel = eredu_core::GenerationCancellationToken::new();
                    input::reset_original_semantic_preparations();
                    crate::tensor::reset_prepared_rotary_calls();
                    let before = runtime.session().payload.model.erased().state_snapshot();
                    if boundary == 0 {
                        cancel.cancel();
                    }
                    let (result, roots) = crate::tests::support::media_completion::observe(
                        Some((cancel.clone(), boundary)),
                        || runtime.prefill_cancellable(prompt, &cancel),
                    );
                    assert!(result.unwrap().is_none());
                    assert_eq!(roots.len(), boundary * 2);
                    assert_eq!(
                        crate::tensor::prepared_rotary_calls(),
                        usize::from(compiled && source_mode == 4 && boundary != 0)
                    );
                    if boundary == 0 {
                        assert_eq!(
                            runtime.session().payload.model.erased().state_snapshot(),
                            before
                        );
                        continue;
                    }
                    assert_eq!(
                        input::original_semantic_preparations(),
                        usize::from(!compiled)
                    );
                    assert!(!roots[0].shapes.is_empty());
                    assert!(roots
                        .iter()
                        .filter(|r| r.after)
                        .all(|r| r.ready.iter().all(|v| *v)));
                    let mut logits = Vec::new();
                    let mut states = Vec::new();
                    for token in [None, Some(4_u32), Some(5), Some(6)] {
                        if let Some(token) = token {
                            let next = runtime
                                .decode(Array::from_slice(&[token], &[1, 1]))
                                .unwrap();
                            next.completion.wait().unwrap();
                            logits.push(array_values(next.output.logits().unwrap().as_array()));
                        }
                        let target = runtime
                            .session_mut()
                            .neutral_prediction_target_mut()
                            .unwrap();
                        states.push((
                            target.retained_numeric_state_snapshot().unwrap().unwrap(),
                            target.fixed_numeric_state_snapshot().unwrap(),
                        ));
                    }
                    assert!(states.iter().all(|s| !s.0.is_empty()));
                    assert_eq!(
                        crate::tensor::prepared_rotary_calls(),
                        usize::from(compiled && source_mode == 4)
                    );
                    reports.push((logits, states));
                }
                if boundary == 0 {
                    continue;
                }
                for (a, b) in reports[0].0.iter().zip(&reports[1].0) {
                    close(a, b);
                }
                for ((av, af), (bv, bf)) in reports[0].1.iter().zip(&reports[1].1) {
                    assert_eq!(av.len(), bv.len());
                    for ((ashape, a), (bshape, b)) in av.iter().zip(bv) {
                        assert_eq!(ashape, bshape);
                        close(
                            &a.iter().map(|v| f64::from(*v)).collect::<Vec<_>>(),
                            &b.iter().map(|v| f64::from(*v)).collect::<Vec<_>>(),
                        );
                    }
                    assert_eq!(af.len(), bf.len());
                    for ((al, ar, ashape, a), (bl, br, bshape, b)) in af.iter().zip(bf) {
                        assert_eq!((al, ar, ashape), (bl, br, bshape));
                        close(
                            &a.iter().map(|v| f64::from(*v)).collect::<Vec<_>>(),
                            &b.iter().map(|v| f64::from(*v)).collect::<Vec<_>>(),
                        );
                    }
                }
            }
        }
    }
}
