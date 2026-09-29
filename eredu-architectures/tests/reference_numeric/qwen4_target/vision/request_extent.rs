//! Whole media requests retain finite history while target rows remain chunk-sized.
use super::*;
use eredu_architectures::composite_execution::{CompositeArchitecture, PreparedCompositeInput};
type Model = ConditionalModel<NumericBackend>;

fn rebound(target: &PreparedTarget, chunk: i32, rows: usize) -> PreparedTarget {
    let path = target
        .artifact()
        .source_provenance("model.embed_tokens.weight")
        .unwrap()
        .backing_shard
        .unwrap();
    let header = eredu_architectures::qwen4_exp::prepared::GgufTargetPlan::prepare(
        &eredu_gguf::Checkpoint::open(path).unwrap(),
    )
    .unwrap();
    let mut limits = target.spec().limits;
    limits.qsa.batch = 1;
    limits.qsa.tokens = chunk;
    limits.invocation_tokens = chunk as usize;
    limits.lookup_rows = rows;
    header.bind(target.artifact().clone(), limits).unwrap()
}
fn selected(
    target: &PreparedTarget,
    ingress: &MediaIngress,
    residency: LayerWeightResidency,
) -> eredu_architectures::qwen4_exp::prepared::SelectedConditionalExecution {
    let rows = target
        .row_lookups(
            RowLookupLimits {
                requests: target.spec().limits.lookup_rows,
                rows_per_acquisition: 2,
                acquisition_bytes: 128,
                host_bytes: 1 << 16,
                output_bytes: 32768,
            },
            eredu_core::residency::ResidencyPolicy::Cacheable,
        )
        .unwrap();
    assert!(rows
        .entries()
        .values()
        .all(|entry| entry.limits().requests == target.spec().limits.lookup_rows));
    let rows = SelectedRowLookupPlans::select(
        rows.descriptors().clone(),
        ParameterBankLoadOptions::new(
            eredu_core::residency::OffloadConfig::new(Some(4096), Some(0), 1).unwrap(),
            1 << 20,
            1 << 20,
        )
        .unwrap(),
        0,
        &cold::Support,
    )
    .unwrap();
    let mut streams = stream_bindings(target.spec());
    for binding in &mut streams {
        binding.lanes = 1;
    }
    let plan = target
        .execution_plan(streams, rows)
        .unwrap()
        .with_vision(ingress.vision().clone())
        .unwrap();
    let facts = cold::capabilities(plan.requirements(), None);
    let request = eredu_architectures::routed_text::RoutedTextSelectionRequest::new(
        ReplicatedTextSelectionRequest::new(residency, CacheResidencyPolicy::Device)
            .with_session(eredu_core::SessionCapabilities::new(false, true, true))
            .with_exact_completion(true),
        WeightResidency::with_layers(residency),
    )
    .unwrap();
    plan.select(
        &request,
        &facts,
        None,
        &processor_request(),
        &processor_facts(),
    )
    .unwrap()
}

#[test]
fn single_lane_long_media_request_keeps_rows_and_assembly_chunk_bounded() {
    let (_directory, reference_target, reference_ingress) = setup();
    let ctx = NumericContext {
        bind_checkpoint_values: true,
        ..Default::default()
    };
    let inspector = Inspector(0.into());
    let mut parts = prompt(&inspector).into_parts();
    parts.push(text(&[5; 23]));
    let source = input(parts, &inspector);
    let prepared = reference_ingress
        .admission_config()
        .admit(&source, &inspector)
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    let expected = reference(&reference_target, &reference_ingress, &prepared, &ctx);
    let target = rebound(&reference_target, 8, 16);
    let ingress = target
        .media_ingress(reference_ingress.vision().clone())
        .unwrap();
    assert_eq!(target.spec().limits.qsa.batch, 1);
    assert_eq!(target.spec().limits.invocation_tokens, 8);
    assert_eq!(ingress.admission_config().maximum_request_tokens(), 128);
    assert_eq!(ingress.admission_config().maximum_chunk_tokens(), 8);
    let admitted = ingress
        .admission_config()
        .admit(&source, &inspector)
        .unwrap();
    assert_eq!(admitted.token_ids().len(), 41);
    let proof = admitted.into_composite();
    let cursor_request =
        <Model as CompositeArchitecture<NumericBackend, State>>::prepare_prefill_request(
            source.clone(),
            proof,
            &ctx,
        )
        .unwrap();
    for residency in [
        LayerWeightResidency::FullyResident,
        LayerWeightResidency::LayerwiseHost(LayerwiseLoadOptions::new(
            eredu_core::residency::OffloadConfig::new(Some(1 << 24), Some(1 << 24), 1).unwrap(),
        )),
        LayerWeightResidency::DenseDiskStream(
            DenseDiskStreamLoadOptions::new(1 << 24, 0, 0, 0).unwrap(),
        ),
    ] {
        let (handoff, store, _) = selected(&target, &ingress, residency)
            .prepare_composite::<NumericBackend, State>(&ctx)
            .unwrap();
        let mut mechanisms = if residency.is_fully_resident() {
            NumericReplicatedMechanisms::with_bound_checkpoint(store)
        } else {
            NumericReplicatedMechanisms::with_bounded_checkpoint(store)
        };
        mechanisms.fixture_state_factory = Some(gguf::state_from_layout);
        reset_reference_stage_evidence("qwen4-long-single-lane");
        macro_rules! exercise {
            ($value:expr) => {{
                let mut session = $value;
                for chunk in [5, 8] {
                    session.reset(&ctx).unwrap();
                    assert!(session
                        .start_prefill(Ok(cursor_request.clone()), 9, None, &ctx)
                        .is_err());
                    let mut cursor = session
                        .start_prefill(Ok(cursor_request.clone()), chunk, None, &ctx)
                        .unwrap();
                    session
                        .advance_prefill(&mut cursor, &ctx, &mut NoopObserver)
                        .unwrap();
                    assert_eq!(cursor.position(), chunk);
                    let saved_cursor = cursor.clone();
                    let frozen = session.capture_control_state(&ctx).unwrap();
                    let mut fork = session.copy_control_state(&frozen, &ctx).unwrap();
                    let encoded = vision_acquisitions();
                    session
                        .advance_prefill(&mut cursor, &ctx, &mut NoopObserver)
                        .unwrap();
                    let advanced = session.checkpoint(&ctx).unwrap();
                    session.exchange_control_state(&mut fork, &ctx).unwrap();
                    cursor = saved_cursor;
                    session
                        .advance_prefill(&mut cursor, &ctx, &mut NoopObserver)
                        .unwrap();
                    session::assert_checkpoint(&session.checkpoint(&ctx).unwrap(), &advanced);
                    let last = session
                        .finish_prefill(
                            &mut cursor,
                            &ctx,
                            &mut FailAt("model.visual.blocks.1.output"),
                        )
                        .unwrap();
                    assert_eq!(cursor.position(), 41);
                    assert_tensor_close(&last, &expected.0[0], "single-lane long request prefill");
                    assert_eq!(
                        vision_acquisitions(),
                        encoded,
                        "encoder output survives cursor fork and restore"
                    );
                    for step in 0..16 {
                        let token = input(vec![text(&[step % 12])], &inspector);
                        let proof = ingress
                            .admission_config()
                            .admit(&token, &inspector)
                            .unwrap()
                            .into_composite();
                        let logits = session
                            .decode_input(
                                PreparedCompositeInput::new(&token, &proof).unwrap(),
                                &ctx,
                            )
                            .unwrap();
                        assert_tensor_close(
                            &logits,
                            &expected.0[step as usize + 1],
                            "single-lane cached decode",
                        );
                    }
                    session::assert_checkpoint(&session.checkpoint(&ctx).unwrap(), &expected.1);
                }
                Ok::<_, String>(())
            }};
        }
        eredu_architectures::prepared_execution::construct_selected_routed_session(
            handoff,
            mechanisms,
            &ctx,
            |_, _, rows| {
                let rows = rows.unwrap();
                assert!(rows
                    .prepared()
                    .entries()
                    .values()
                    .all(|entry| entry.limits().requests == 16));
                let providers = rows
                    .prepared()
                    .bind(
                        rows.prepared()
                            .entries()
                            .iter()
                            .map(|(id, entry)| {
                                (
                                    id.clone(),
                                    super::super::super::row_bank::SourceRows::new(entry),
                                )
                            })
                            .collect(),
                    )
                    .unwrap();
                Ok::<_, String>(
                    (
                        BTreeMap::<
                            RoutedBankId,
                            (NumericGroupedBankMechanism, NumericIndexedMovement),
                        >::new(),
                        Some(providers),
                    ),
                )
            },
            (),
            |_, value, _| exercise!(value),
            |_, value, _| exercise!(value),
        )
        .unwrap();
    }
    // Metadata-only total-extent rejection precedes original-ID inspection or
    // any request tensor/embedding allocation.
    let too_long = input(vec![text(&[3; 129])], &inspector);
    let before = (
        inspector.0.get(),
        ctx.integer_tensor_constructions.get(),
        ctx.embedding_forward_calls.get(),
    );
    assert!(ingress
        .admission_config()
        .admit(&too_long, &inspector)
        .is_err());
    assert_eq!(
        before,
        (
            inspector.0.get(),
            ctx.integer_tensor_constructions.get(),
            ctx.embedding_forward_calls.get()
        )
    );
}

#[test]
fn retained_media_proofs_preserve_their_chunk_authority_across_same_geometry_models() {
    let (_directory, original, original_ingress) = setup();
    let consumer_target = rebound(&original, 8, 128);
    let consumer = consumer_target
        .media_ingress(original_ingress.vision().clone())
        .unwrap();
    let ctx = NumericContext::default();
    let inspector = Inspector(0.into());
    let source = input(vec![text(&[3, 4])], &inspector);
    for (producer_chunk, allowed) in [(32, false), (5, true)] {
        // Keep executable geometry identical while changing only the outer
        // source-free request admission contract. A retained proof must not
        // authorize a larger chunk than its consumer allows.
        let mut spec = consumer_target.spec().clone();
        spec.limits.qsa.tokens = producer_chunk;
        spec.limits.invocation_tokens = producer_chunk as usize;
        let producer = eredu_architectures::qwen4_exp::media::MediaAdmissionConfig::new(
            &spec,
            consumer.vision().config(),
            consumer.vision().media_tokens(),
        )
        .unwrap();
        assert_eq!(producer.identity(), consumer.admission_config().identity());
        let admitted = producer.admit(&source, &inspector).unwrap();
        let selected = selected(
            &consumer_target,
            &consumer,
            LayerWeightResidency::FullyResident,
        );
        assert_eq!(
            selected
                .encoder_memory::<NumericBackend>(&admitted, eredu_nn::TensorElementType::F32, &ctx)
                .is_ok(),
            allowed
        );
        let prepared = admitted.prepare(&ctx).unwrap();
        let composite = admitted.into_composite();
        let mut model = Model::new(
            consumer_target.bound_spec().unwrap(),
            consumer.clone(),
            consumer.vision().config().clone(),
            &ctx,
        )
        .unwrap();
        for owned in [false, true] {
            let mut state =
                gguf::state_from_layout(&consumer_target.spec().state_layout().unwrap()).unwrap();
            let saved = state.clone();
            let before = (
                ctx.integer_tensor_constructions.get(),
                ctx.embedding_forward_calls.get(),
            );
            let result = if owned {
                model.begin_composite_forward(
                    PreparedCompositeInput::new(&source, &composite).unwrap(),
                    &mut state,
                    &ctx,
                )
            } else {
                model.begin_forward(
                    ConditionalInput::MediaChunk {
                        prepared: &prepared,
                        range: 0..2,
                        projected: None,
                    },
                    &mut state,
                    &ctx,
                )
            };
            assert_eq!(result.is_ok(), allowed, "owned={owned}");
            if !allowed {
                assert_eq!(
                    before,
                    (
                        ctx.integer_tensor_constructions.get(),
                        ctx.embedding_forward_calls.get()
                    )
                );
                session::assert_checkpoint(&state, &saved);
            }
        }
    }
}
