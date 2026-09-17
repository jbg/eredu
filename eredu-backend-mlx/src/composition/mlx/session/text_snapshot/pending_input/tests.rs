use super::*;
use eredu_core::{InputExtent, InputMetadataKey, PreparedControlInputBackend};
use eredu_runtime::{working_memory::WorkingMemoryPool, PreparedInputInspector};
use safemlx::{ops::indexing::TryIndexOp, Device, DeviceType};
use std::cell::Cell;

thread_local! {
    static COPIED: Cell<usize> = const { Cell::new(0) };
    static FAIL_AT: Cell<Option<(usize, bool)>> = const { Cell::new(None) };
    static ESCAPED: RefCell<Option<Array>> = const { RefCell::new(None) };
}
#[derive(Debug, thiserror::Error)]
#[error("injected pending-input copy failure after a real slot")]
struct CopyFault;
pub(super) fn after_slot(count: usize, result: &Array) -> Result<(), Error> {
    COPIED.set(count);
    if let Some((at, panic)) = FAIL_AT.get() {
        if at == count {
            FAIL_AT.set(None);
            ESCAPED.with_borrow_mut(|slot| *slot = Some(result.clone()));
            assert!(!panic, "injected pending-copy unwind after real slot");
            return Err(Error::Other(Box::new(CopyFault)));
        }
    }
    Ok(())
}
struct Hook;
impl Drop for Hook {
    fn drop(&mut self) {
        FAIL_AT.set(None);
        ESCAPED.with_borrow_mut(|slot| {
            slot.take();
        });
    }
}
fn with_runtime<R>(
    run: impl FnOnce(&mut ModelRuntime<MlxBackend<'_>>, &Stream, &WorkingMemoryPool) -> R,
) -> R {
    let root = tempfile::tempdir().unwrap();
    crate::tests::distributed_pipeline_ring::write_qwen3_vl_component_fixture(
        root.path(),
        false,
        false,
    );
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let backend = MlxBackend::new(&stream, &stream).with_memory_pool(pool.clone());
    let model =
        eredu_core::load_model(&backend, root.path(), crate::MlxLoadRequest::default()).unwrap();
    let mut runtime = ModelRuntime::from_prepared(backend, model).unwrap();
    run(&mut runtime, &stream, &pool)
}
fn raw(stream: &Stream) -> MlxModelInput {
    let text = Array::from_slice(&[9_u32, 1, 2, 8], &[1, 4])
        .try_index_device((.., 1..3), stream)
        .unwrap();
    let pixels = Array::from_slice(
        &(0..16 * 24)
            .map(|i| (i as f32 - 171.) / 383.)
            .collect::<Vec<_>>(),
        &[16, 24],
    );
    let grids = Array::from_slice(&[1_i32, 4, 4, 1, 4, 4], &[1, 6]);
    let image = |start| {
        input::input_part(
            InputModality::Image,
            input::InputPayload::Tensor(
                pixels
                    .try_index_device((.., start..start + 12), stream)
                    .unwrap(),
            ),
            [(
                InputMetadataKey::PatchGrid,
                grids
                    .try_index_device((.., if start == 0 { 0..3 } else { 3..6 }), stream)
                    .unwrap(),
            )],
            [InputExtent::PatchGrid {
                time: 1,
                height: 4,
                width: 4,
            }],
        )
        .unwrap()
    };
    let parts = [
        input::token_ids_part(&text).unwrap(),
        image(0),
        image(12),
        input::input_part(
            InputModality::Text,
            input::InputPayload::Embeddings(Array::from_slice(
                &(0..128)
                    .map(|i| (i as f32 - 63.) / 129.)
                    .collect::<Vec<_>>(),
                &[1, 2, 64],
            )),
            [],
            [],
        )
        .unwrap(),
        input::token_ids_part(&Array::from_slice(&[3_u32], &[1, 1])).unwrap(),
    ];
    MlxModelInput::from(input::ModelInput::new(&parts))
        .with_semantic_content_fingerprint("actual ordered multi-view pending source")
        .unwrap()
        .with_prefill_chunk_positions(2.try_into().unwrap())
}
fn source(runtime: &ModelRuntime<MlxBackend<'_>>, stream: &Stream) -> MlxModelInput {
    let prepared = MlxBackend::prepare_control_input(runtime, raw(stream)).unwrap();
    let (prompt, attribution) = MlxBackend::consume_control_input(runtime, prepared).unwrap();
    assert_eq!(attribution.attribution().decoder_positions, 13);
    assert_eq!(attribution.attribution().canonical_token_ids, [1, 2, 3]);
    prompt
}
fn values(array: &Array) -> Vec<f64> {
    let a = array.evaluated().unwrap();
    match array.dtype() {
        Dtype::Float32 => a.try_iter::<f32>().unwrap().map(f64::from).collect(),
        Dtype::Int32 => a.try_iter::<i32>().unwrap().map(f64::from).collect(),
        Dtype::Uint32 => a.try_iter::<u32>().unwrap().map(f64::from).collect(),
        other => panic!("unexpected fixture dtype {other:?}"),
    }
}
fn settle(stream: &Stream, pool: &WorkingMemoryPool, expected: usize) {
    stream.synchronize().unwrap();
    crate::backend::submission_recovery::wait_for_retirement(|| {
        safemlx::reclaim_allocation_owners();
        pool.unquoted_owner_count().unwrap() == expected
    });
}

#[test]
fn pending_media_copy_preserves_all_slots_views_extents_and_authenticated_growth() {
    with_runtime(|runtime, stream, _| {
        let source = source(runtime, stream);
        let before = runtime.session().test_state_presence();
        let plan = PromptCopyPlan::prepare(&source).unwrap();
        assert_eq!(plan.slots, 7);
        assert!(plan.estimate().copy_bytes > plan.estimate().retained_bytes);
        use eredu_core::execution_control::{SnapshotLimits, SnapshotResourceKind};
        use eredu_runtime::execution_control::SnapshotBudget;
        let estimate = plan.estimate();
        for short_retained in [true, false] {
            let budget = SnapshotBudget::new(SnapshotLimits {
                max_snapshots: 1,
                max_branches: 0,
                retained_bytes: estimate.retained_bytes - u64::from(short_retained),
                cumulative_copy_bytes: estimate.copy_bytes - u64::from(!short_retained),
            });
            COPIED.set(0);
            assert!(budget
                .reserve(SnapshotResourceKind::Snapshot, Some(estimate))
                .is_err());
            assert_eq!(budget.usage().cumulative_copy_bytes, 0);
            assert_eq!(COPIED.get(), 0);
        }
        let budget = SnapshotBudget::new(SnapshotLimits {
            max_snapshots: 1,
            max_branches: 0,
            retained_bytes: estimate.retained_bytes,
            cumulative_copy_bytes: estimate.copy_bytes,
        });
        let paid = budget
            .reserve(SnapshotResourceKind::Snapshot, Some(estimate))
            .unwrap();
        let copied = plan.copy(runtime).unwrap();
        assert_eq!(COPIED.get(), 7);
        assert_eq!(budget.usage().cumulative_copy_bytes, estimate.copy_bytes);
        assert_eq!(copied.controlled_decoder_positions(), Some(13));
        assert_eq!(copied.cache_identity(), source.cache_identity());
        for predictions in [0, 1, 4] {
            assert_eq!(
                MlxBackend::continuation_input_tokens(
                    Some(PendingTextInput::Prefill(&copied)),
                    predictions
                ),
                Some(if predictions == 0 {
                    0
                } else {
                    13 + predictions - 1
                })
            );
        }
        assert_eq!(
            MlxBackend::continuation_input_tokens(
                Some(PendingTextInput::Prefill(&copied)),
                u64::MAX
            ),
            None
        );
        source.with_borrowed(|original| {
            copied.with_borrowed(|copy| {
                assert_eq!(
                    copy.prefill_chunk_positions(),
                    original.prefill_chunk_positions()
                );
                assert_eq!(copy.parts.len(), original.parts.len());
                let mut all_destinations = std::collections::HashSet::new();
                for (a, b) in original.parts.iter().zip(copy.parts) {
                    assert_eq!(
                        a.descriptor(&|v| input::MlxInputInspector.identity(v))
                            .unwrap(),
                        b.descriptor(&|v| input::MlxInputInspector.identity(v))
                            .unwrap()
                    );
                    assert_eq!(a.extents(), b.extents());
                    for (a, b) in std::iter::once(a.payload().value())
                        .chain(a.metadata().values())
                        .zip(std::iter::once(b.payload().value()).chain(b.metadata().values()))
                    {
                        assert_eq!(values(a), values(b));
                        let source = a.allocation_info().unwrap().unwrap().identity();
                        let target = b.allocation_info().unwrap().unwrap().identity();
                        assert_ne!(source, target);
                        assert!(
                            all_destinations.insert(target),
                            "one independent allocation per logical slot"
                        );
                    }
                }
                let a = original.parts[1].payload().value();
                let b = original.parts[2].payload().value();
                assert_eq!(
                    a.allocation_info().unwrap().unwrap().identity(),
                    b.allocation_info().unwrap().unwrap().identity()
                );
                assert_ne!(
                    values(a),
                    values(b),
                    "different offsets on one source backing retain different values"
                );
                assert_eq!(all_destinations.len(), 7);
            })
        });
        assert_eq!(
            runtime.session().test_state_presence(),
            before,
            "copy is not model execution"
        );
        drop(copied);
        drop(paid);
        assert_eq!(budget.usage().retained_bytes, 0);
        assert_eq!(budget.usage().cumulative_copy_bytes, estimate.copy_bytes);
    });
}

#[test]
fn pending_media_copy_rejects_raw_or_relabelled_source_before_any_slot() {
    with_runtime(|runtime, stream, pool| {
        let original = source(runtime, stream);
        let relabelled = original
            .clone()
            .with_semantic_content_fingerprint("new source without consumed attribution")
            .unwrap();
        let raw = raw(stream);
        for prompt in [&raw, &relabelled] {
            COPIED.set(0);
            let before = pool.unquoted_owner_count().unwrap();
            assert!(MlxBackend::estimate_pending_input(
                runtime,
                Some(PendingTextInput::Prefill(prompt))
            )
            .unwrap()
            .is_none());
            assert!(MlxBackend::copy_pending_input(
                runtime,
                Some(PendingTextInput::Prefill(prompt))
            )
            .is_err());
            assert_eq!(COPIED.get(), 0);
            assert_eq!(pool.unquoted_owner_count().unwrap(), before);
            assert_eq!(
                MlxBackend::continuation_input_tokens(Some(PendingTextInput::Prefill(prompt)), 3),
                None
            );
        }
        let legacy = MlxModelInput::from(input::ModelInput::new(&[input::token_ids_part(
            &Array::from_slice(&[1_u32, 2], &[1, 2]),
        )
        .unwrap()]));
        let copy =
            MlxBackend::copy_pending_input(runtime, Some(PendingTextInput::Prefill(&legacy)))
                .unwrap()
                .unwrap();
        assert!(matches!(copy, PendingTextInput::Prefill(_)));
        assert_eq!(
            MlxBackend::continuation_input_tokens(Some(PendingTextInput::Prefill(&legacy)), 3),
            Some(4)
        );
    });
}

#[test]
fn pending_media_copy_failure_and_unwind_keep_partial_backing_until_final_alias() {
    for panic in [false, true] {
        let _hook = Hook;
        let (pool, stream, escaped, error) = with_runtime(|runtime, stream, pool| {
            let source = source(runtime, stream);
            let before = runtime.session().test_state_presence();
            FAIL_AT.set(Some((2, panic)));
            COPIED.set(0);
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                PromptCopyPlan::prepare(&source).unwrap().copy(runtime)
            }));
            let error = if panic {
                assert!(result.is_err());
                None
            } else {
                let error = result.unwrap().err().unwrap();
                assert!(error
                    .to_string()
                    .contains("injected pending-input copy failure"));
                Some(error)
            };
            assert_eq!(COPIED.get(), 2);
            assert_eq!(runtime.session().test_state_presence(), before);
            let escaped = ESCAPED.with_borrow_mut(Option::take).unwrap();
            assert!(!values(&escaped).is_empty());
            (pool.clone(), stream.clone(), escaped, error)
        });
        settle(&stream, &pool, 1);
        drop(escaped);
        if error.is_some() {
            settle(&stream, &pool, 1);
        }
        drop(error);
        settle(&stream, &pool, 0);
    }
}

#[test]
fn copied_media_prompt_can_retire_before_escaped_completed_metadata_array() {
    let (pool, stream, escaped, expected) = with_runtime(|runtime, stream, pool| {
        let source = source(runtime, stream);
        let copied = PromptCopyPlan::prepare(&source)
            .unwrap()
            .copy(runtime)
            .unwrap();
        let escaped = copied
            .with_borrowed(|view| view.parts[1].metadata()[&InputMetadataKey::PatchGrid].clone());
        let expected = values(&escaped);
        drop((copied, source));
        (pool.clone(), stream.clone(), escaped, expected)
    });
    settle(&stream, &pool, 1);
    assert_eq!(values(&escaped), expected);
    drop(escaped);
    settle(&stream, &pool, 0);
}

// Genuine core comparison and original pool reservation for a separate fixed
// text fixture. It authorizes no media or copy work; this test only exercises
// rejection of that real foreign reservation before any ordinary acquisition.
fn text_reservation() -> eredu_runtime::working_memory::InferenceRequest {
    use eredu_core::*;
    use eredu_runtime::working_memory::InferenceExecutionIdentity;
    let geometry = InferenceGeometry {
        batch_size: 1,
        cached_positions: 0,
        input_positions: 3,
        max_output_tokens: 1,
        prefill_chunk_positions: 2,
        output: OutputDemand::LastPosition,
    };
    let request = AdmissionRequest {
        input: InputTokenCount::text(3),
        max_output_tokens: 1,
        batch_size: 1,
        safety_reserve_bytes: 0,
        application_memory_budget_bytes: None,
        require_complete_estimate: true,
    };
    let layout = StateMemoryLayout::new(
        LayerSchedule::empty(),
        vec![],
        1,
        1,
        EstimationCompleteness::Complete,
    )
    .unwrap();
    let bound = || WorkspaceBound::bounded(16, "fixed text authority rejection fixture only");
    let state = estimate_runtime_state(
        &layout,
        request.input,
        1,
        1,
        std::num::NonZeroU8::new(4).unwrap(),
    )
    .unwrap()
    .with_execution_workspace(ExecutionWorkspaceEstimate {
        geometry,
        activations: bound(),
        attention: bound(),
        vocabulary: bound(),
        state_update: bound(),
        materialization: bound(),
        retained: bound(),
    })
    .unwrap();
    let capabilities = ModelCapabilities {
        effective_model_type: "fixed text authority rejection fixture".into(),
        native_max_context: Observed::exact(8, "fixture"),
        effective_max_context: Observed::exact(8, "fixture"),
        state_strategy: CacheStateStrategy::FullKv,
        modalities: InputModalities::TEXT,
        estimation: EstimationCompleteness::Complete,
    };
    let AdmissionResult::Admitted(admission) =
        apply_admission_policy(&capabilities, request, state, None).unwrap()
    else {
        panic!("genuine fixed text admission");
    };
    WorkingMemoryPool::new(admission.incremental_required_bytes, 0)
        .unwrap()
        .reserve(&InferenceExecutionIdentity::default(), &admission)
        .unwrap()
        .into()
}

#[test]
fn pending_media_copy_rejects_actual_reserved_request_even_without_quote_marker() {
    with_runtime(|runtime, stream, pool| {
        let original = source(runtime, stream);
        let prompt = original.with_inference_request(text_reservation());
        // with_inference_request clears the quote marker in the real constructor.
        assert!(prompt.has_original_input_custody());
        COPIED.set(0);
        let before = pool.unquoted_owner_count().unwrap();
        assert!(MlxBackend::estimate_pending_input(
            runtime,
            Some(PendingTextInput::Prefill(&prompt))
        )
        .unwrap()
        .is_none());
        let error =
            MlxBackend::copy_pending_input(runtime, Some(PendingTextInput::Prefill(&prompt)))
                .err()
                .unwrap();
        let Error::Other(cause) = error else {
            panic!("typed unknown bound")
        };
        assert!(matches!(
            cause.downcast_ref::<eredu_runtime::working_memory::WorkingMemoryError>(),
            Some(eredu_runtime::working_memory::WorkingMemoryError::UnknownBound)
        ));
        assert_eq!(COPIED.get(), 0);
        assert_eq!(pool.unquoted_owner_count().unwrap(), before);
    });
}

#[test]
fn copied_media_prefill_and_three_cached_decodes_match_every_retained_state_value() {
    use eredu_core::Completion;
    let mut reports = Vec::new();
    for copied in [false, true] {
        reports.push(with_runtime(|runtime, stream, _| {
            let original = source(runtime, stream);
            let prompt = if copied {
                PromptCopyPlan::prepare(&original)
                    .unwrap()
                    .copy(runtime)
                    .unwrap()
            } else {
                original
            };
            let mut outputs = Vec::new();
            let mut states = Vec::new();
            let first = runtime.prefill(prompt).unwrap();
            first.completion.wait().unwrap();
            outputs.push(values(first.output.logits().unwrap().as_array()));
            drop(first);
            {
                let target = runtime
                    .session_mut()
                    .neutral_prediction_target_mut()
                    .unwrap();
                states.push((
                    target.retained_numeric_state_snapshot().unwrap().unwrap(),
                    target.fixed_numeric_state_snapshot().unwrap(),
                ));
            }
            for id in [4_u32, 5, 6] {
                let next = runtime.decode(Array::from_slice(&[id], &[1, 1])).unwrap();
                next.completion.wait().unwrap();
                outputs.push(values(next.output.logits().unwrap().as_array()));
                drop(next);
                let target = runtime
                    .session_mut()
                    .neutral_prediction_target_mut()
                    .unwrap();
                states.push((
                    target.retained_numeric_state_snapshot().unwrap().unwrap(),
                    target.fixed_numeric_state_snapshot().unwrap(),
                ));
            }
            assert!(outputs[0].iter().any(|value| value.abs() > 1e-5));
            assert!(states.iter().all(|(arrays, _)| !arrays.is_empty()));
            (outputs, states)
        }));
    }
    let close = |actual: &[f64], expected: &[f64]| {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert!(
                (actual - expected).abs() <= 3e-4 + 3e-4 * expected.abs(),
                "{actual} != {expected}"
            );
        }
    };
    assert_eq!(reports[0].0.len(), 4);
    for (actual, expected) in reports[1].0.iter().zip(&reports[0].0) {
        close(actual, expected);
    }
    assert_eq!(reports[0].1.len(), reports[1].1.len());
    for ((arrays, fixed), (reference_arrays, reference_fixed)) in
        reports[1].1.iter().zip(&reports[0].1)
    {
        assert_eq!(arrays.len(), reference_arrays.len());
        for ((shape, actual), (reference_shape, expected)) in arrays.iter().zip(reference_arrays) {
            assert_eq!(shape, reference_shape);
            close(
                &actual.iter().copied().map(f64::from).collect::<Vec<_>>(),
                &expected.iter().copied().map(f64::from).collect::<Vec<_>>(),
            );
        }
        assert_eq!(fixed.len(), reference_fixed.len());
        for (
            (layer, role, shape, actual),
            (reference_layer, reference_role, reference_shape, expected),
        ) in fixed.iter().zip(reference_fixed)
        {
            assert_eq!(
                (layer, role, shape),
                (reference_layer, reference_role, reference_shape)
            );
            close(
                &actual.iter().copied().map(f64::from).collect::<Vec<_>>(),
                &expected.iter().copied().map(f64::from).collect::<Vec<_>>(),
            );
        }
    }
}
