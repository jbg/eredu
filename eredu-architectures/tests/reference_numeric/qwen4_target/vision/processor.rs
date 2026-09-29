//! Shared prepared-input adapter, exact host proof, and ordinary session lifecycle.
use super::*;
use eredu_architectures::composite_execution::{CompositeArchitecture, PreparedCompositeInput};
use eredu_nn::EmbeddingOperator;
type Model = ConditionalModel<NumericBackend>;
type Plan = <Model as CompositeArchitecture<NumericBackend, State>>::InputPartPlan;

fn admit(
    ingress: &MediaIngress,
    input: &PreparedModelInput<NumericTensor>,
    inspector: &Inspector,
) -> eredu_architectures::media_plan::AdmittedCompositeInput<Plan> {
    <Model as CompositeArchitecture<NumericBackend, State>>::admit_prepared_input(
        ingress.admission_config(),
        input,
        inspector,
    )
    .unwrap()
}

// Older family fixtures intentionally inspect their synthetic tensors as F32.
// This fixture uses the same allocations but must preserve integer provenance exactly.
#[derive(Default)]
struct RawMechanisms(NumericProcessorMechanisms);
impl std::ops::Deref for RawMechanisms {
    type Target = NumericProcessorMechanisms;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl PreparedInputInspector<NumericTensor> for RawMechanisms {
    fn identity(
        &self,
        value: &NumericTensor,
    ) -> Result<eredu_core::InputTensorIdentity, eredu_core::PreparedInputError> {
        Inspector(0.into()).identity(value)
    }
    fn i32_values(&self, value: &NumericTensor) -> Result<Vec<i32>, eredu_core::CapabilityError> {
        Inspector(0.into()).i32_values(value)
    }
    fn bool_values(&self, value: &NumericTensor) -> Result<Vec<bool>, eredu_core::CapabilityError> {
        self.0.bool_values(value)
    }
}
impl eredu_architectures::processor_execution::ProcessorMechanisms for RawMechanisms {
    type Tensor = NumericTensor;
    type Error = String;
    fn tensor_u32(&mut self, values: &[u32], shape: &[usize]) -> Result<NumericTensor, String> {
        eredu_architectures::processor_execution::ProcessorMechanisms::tensor_u32(
            &mut self.0,
            values,
            shape,
        )
    }
    fn tensor_f32(&mut self, values: &[f32], shape: &[usize]) -> Result<NumericTensor, String> {
        eredu_architectures::processor_execution::ProcessorMechanisms::tensor_f32(
            &mut self.0,
            values,
            shape,
        )
    }
    fn tensor_i32(&mut self, values: &[i32], shape: &[usize]) -> Result<NumericTensor, String> {
        eredu_architectures::processor_execution::ProcessorMechanisms::tensor_i32(
            &mut self.0,
            values,
            shape,
        )
    }
}

fn buffer_budget() -> eredu_runtime::processor_resources::ProcessorRequestBudget {
    eredu_runtime::processor_resources::ProcessorRequestBudget {
        decoded_input_bytes: 1 << 20,
        host_buffer_bytes: 1 << 20,
        output_tensor_bytes: 1 << 20,
        planning_items: 4096,
        decoder_positions: 64,
    }
}

fn processor_json() -> Vec<u8> {
    serde_json::to_vec(&serde_json::json!({
        "size": {"shortest_edge":16,"longest_edge":64},
        "patch_size":2,"temporal_patch_size":2,"merge_size":2,
        "image_mean":[0.,0.,0.],"image_std":[1.,1.,1.],"min_frames":4,"max_frames":768
    }))
    .unwrap()
}
fn raw_request(extra_tokens: usize) -> eredu_core::TokenizedMultimodalRequest {
    use eredu_core::{
        Media, MultimodalRequest, MultimodalSegment as S, RgbImage, Video, VideoSampling,
    };
    let image = |w, h| RgbImage::new([64, 128, 192].repeat((w * h) as usize), w, h).unwrap();
    MultimodalRequest::new(vec![
        S::TokenIds(vec![3]),
        S::Media(Media::Image(image(8, 8))),
        S::Media(Media::Video(
            Video::new(vec![image(4, 8); 4], Some(2.), VideoSampling::All).unwrap(),
        )),
        S::TokenIds(vec![6; 1 + extra_tokens]),
    ])
    .unwrap()
    .tokenize::<String>(|_| unreachable!())
    .unwrap()
}

pub(super) fn run_processor(target: &PreparedTarget, ingress: &MediaIngress) {
    let ctx = NumericContext {
        bind_checkpoint_values: true,
        ..Default::default()
    };
    let artifacts = processor_json();
    let mut malformed: serde_json::Value = serde_json::from_slice(&artifacts).unwrap();
    malformed["patch_size"] = 3.into();
    assert!(ingress
        .vision()
        .clone()
        .with_processor_json(Some(&serde_json::to_vec(&malformed).unwrap()), None)
        .is_err());
    let vision = if ingress.vision().processor().is_some() {
        ingress.vision().clone()
    } else {
        ingress
            .vision()
            .clone()
            .with_processor_json(Some(&artifacts), Some(&artifacts))
            .unwrap()
    };
    let owned_ingress = target.media_ingress(vision).unwrap();
    let ingress = &owned_ingress;
    let inspector = Inspector(0.into());
    let source = prompt(&inspector);
    let proof = admit(ingress, &source, &inspector);
    let prepared = ingress
        .admission_config()
        .admit(&source, &inspector)
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    let expected = reference(target, ingress, &prepared, &ctx);
    let mut long_parts = source.clone().into_parts();
    long_parts.push(text(&[5; 23]));
    let long_source = input(long_parts, &inspector);
    let long_proof = admit(ingress, &long_source, &inspector);
    let long_prepared = ingress
        .admission_config()
        .admit(&long_source, &inspector)
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    let long_expected = reference(target, ingress, &long_prepared, &ctx);
    assert!(
        <Model as CompositeArchitecture<NumericBackend, State>>::prepare_prefill_request(
            source.clone(),
            long_proof.clone(),
            &ctx,
        )
        .is_err(),
        "cursor construction checks the exact prepared input before allocating request products"
    );
    assert_eq!(
        <Model as CompositeArchitecture<NumericBackend, State>>::prefill_chunk_limit(&long_proof)
            .unwrap(),
        Some(32),
    );
    let chunk_request =
        <Model as CompositeArchitecture<NumericBackend, State>>::prepare_prefill_request(
            long_source.clone(),
            long_proof,
            &ctx,
        )
        .unwrap();
    let mut encoder = tower(ingress, &ctx);
    let projected = prepared
        .with_vision_input(|v| encoder.forward(v.unwrap(), &ctx))
        .unwrap()
        .embeddings;
    let mut offset = 0;
    let projected_parts = source
        .parts()
        .iter()
        .zip(proof.parts())
        .map(|(part, plan)| {
            let eredu_architectures::media_plan::QwenInputPartPlan::Media { ingress, shape } =
                plan.shared()
            else {
                return part.clone();
            };
            let positions = shape.decoder_positions as i32;
            let value = projected
                .index(
                    &[
                        Index::Full,
                        Index::Range(offset, offset + positions),
                        Index::Full,
                    ],
                    &ctx,
                )
                .unwrap();
            offset += positions;
            PreparedInputPart::new(
                part.modality(),
                P::Embeddings(value),
                [
                    (
                        K::PatchGrid,
                        part.metadata_value(K::PatchGrid).unwrap().clone(),
                    ),
                    (
                        K::OriginalTokenIds,
                        NumericTensor::from_i32_slice(
                            &vec![ingress.placeholder_token_id as i32; positions as usize],
                            &[1, positions],
                            &ctx,
                        )
                        .unwrap(),
                    ),
                ],
            )
            .unwrap()
        })
        .collect();
    let projected_source = input(projected_parts, &inspector);
    let projected_proof = admit(ingress, &projected_source, &inspector);
    let mut embedding_model =
        TargetModel::<NumericBackend>::new(target.bound_spec().unwrap(), &ctx).unwrap();
    let static_modules = <TargetModel<NumericBackend> as LayeredArchitecture<
        NumericBackend,
        State,
    >>::static_modules_mut(&mut embedding_model);
    static_modules.visit_parameters_mut(&mut TargetBind {
        ordinary: target.static_parameters(),
        experts: vec![],
        ctx: &ctx,
    });
    let zero_id = NumericTensor::from_i32_slice(&[0], &[1, 1], &ctx).unwrap();
    let projected_text = input(
        vec![PreparedInputPart::new(
            M::Text,
            P::Embeddings(static_modules.embeddings.forward(&zero_id, &ctx).unwrap()),
            [(K::OriginalTokenIds, zero_id)],
        )
        .unwrap()],
        &inspector,
    );
    let projected_text_proof = admit(ingress, &projected_text, &inspector);
    let tokens =
        <Model as CompositeArchitecture<NumericBackend, State>>::prepared_prediction_token_ids(
            PreparedCompositeInput::new(&source, &proof).unwrap(),
            &ctx,
        )
        .unwrap();
    assert_tensor_exact(
        &tokens,
        prepared.token_ids(),
        "processor prediction original IDs",
    );
    let numeric = Model::new(
        target.bound_spec().unwrap(),
        ingress.clone(),
        ingress.vision().config().clone(),
        &ctx,
    )
    .unwrap();
    assert!(
        <Model as CompositeArchitecture<NumericBackend, State>>::should_execute_prepared_group(
            &numeric,
            1,
            PreparedCompositeInput::new(&source, &proof).unwrap(),
        )
    );
    assert_eq!(
        <Model as CompositeArchitecture<NumericBackend, State>>::prepared_group_boundary_sequence(
            &numeric,
            1,
            PreparedCompositeInput::new(&source, &proof).unwrap(),
        )
        .unwrap(),
        8
    );
    assert!(
        !<Model as CompositeArchitecture<NumericBackend, State>>::should_execute_prepared_group(
            &numeric,
            1,
            PreparedCompositeInput::new(&projected_source, &projected_proof).unwrap(),
        )
    );
    let missing_ids = input(
        vec![PreparedInputPart::new(
            M::Text,
            P::Embeddings(NumericTensor::new([1, 1, 32], vec![0.1; 32])),
            [],
        )
        .unwrap()],
        &inspector,
    );
    assert!(
        <Model as CompositeArchitecture<NumericBackend, State>>::admit_prepared_input(
            ingress.admission_config(),
            &missing_ids,
            &inspector,
        )
        .is_err()
    );

    let foreign_cursor = std::cell::RefCell::new(None);
    for residency in [
        LayerWeightResidency::FullyResident,
        LayerWeightResidency::LayerwiseHost(LayerwiseLoadOptions::new(
            eredu_core::residency::OffloadConfig::new(Some(1 << 24), Some(1 << 24), 1).unwrap(),
        )),
        LayerWeightResidency::DenseDiskStream(
            DenseDiskStreamLoadOptions::new(1 << 24, 0, 0, 0).unwrap(),
        ),
    ] {
        let plan = gguf::plan(target)
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
        let raw_policy = processor_request().with_raw_media(true);
        let missing = eredu_runtime::MediaPrimitiveCapabilities::new(
            [M::Text, M::Image, M::Video],
            [M::Text, M::Image, M::Video],
            [M::Text, M::Image, M::Video],
            [],
            i32::MAX as u64,
        );
        assert!(plan
            .clone()
            .select(&request, &facts, None, &raw_policy, &missing)
            .is_err());
        let no_artifacts = target
            .media_ingress(
                ingress
                    .vision()
                    .clone()
                    .with_processor_json(None, None)
                    .unwrap(),
            )
            .unwrap();
        let no_plan = gguf::plan(target)
            .with_vision(no_artifacts.vision().clone())
            .unwrap();
        assert!(no_plan
            .clone()
            .select(&request, &facts, None, &raw_policy, &processor_facts())
            .is_err());
        let text_only = no_plan
            .select(
                &request,
                &facts,
                None,
                &eredu_runtime::ProcessorSelectionRequest::new([M::Text]),
                &processor_facts(),
            )
            .unwrap();
        let mut text_mechanisms = RawMechanisms::default();
        let tokens =
            eredu_core::MultimodalRequest::new(vec![eredu_core::MultimodalSegment::TokenIds(
                vec![3, 4],
            )])
            .unwrap()
            .tokenize::<String>(|_| unreachable!())
            .unwrap();
        let text_input = text_only
            .processor()
            .prepare(&tokens, &mut text_mechanisms, buffer_budget(), &mut |_| {
                Ok::<_, String>(vec![])
            })
            .unwrap();
        assert_eq!(
            text_only
                .processor()
                .admit(&text_input, &inspector)
                .unwrap()
                .token_ids(),
            [3, 4]
        );
        let before = text_mechanisms.tensors;
        let invalid_ids =
            eredu_core::MultimodalRequest::new(vec![eredu_core::MultimodalSegment::TokenIds(
                vec![u32::MAX],
            )])
            .unwrap()
            .tokenize::<String>(|_| unreachable!())
            .unwrap();
        assert!(matches!(
            text_only.processor().prepare(
                &invalid_ids,
                &mut text_mechanisms,
                buffer_budget(),
                &mut |_| Ok::<_, String>(vec![])
            ),
            Err(eredu_architectures::processor_execution::ProcessorExecutionError::Admission(_))
        ));
        assert_eq!(text_mechanisms.tensors, before);
        let inspected = inspector.0.get();
        assert!(text_only.processor().admit(&source, &inspector).is_err());
        assert_eq!(
            inspector.0.get(),
            inspected,
            "unselected modality rejects before metadata reads"
        );
        let unprojected = plan
            .clone()
            .select(
                &request,
                &facts,
                None,
                &eredu_runtime::ProcessorSelectionRequest::new([M::Text, M::Image, M::Video]),
                &processor_facts(),
            )
            .unwrap();
        assert!(unprojected
            .processor()
            .admit(&projected_source, &inspector)
            .is_err());
        assert_eq!(
            inspector.0.get(),
            inspected,
            "unselected embeddings reject before metadata reads"
        );
        let selected = plan
            .clone()
            .select(&request, &facts, None, &raw_policy, &processor_facts())
            .unwrap();
        assert!(selected.processor().realization().raw_media());
        let image_only = plan
            .clone()
            .select(
                &request,
                &facts,
                None,
                &eredu_runtime::ProcessorSelectionRequest::new([M::Image]).with_raw_media(true),
                &processor_facts(),
            )
            .unwrap();
        assert!(image_only
            .processor()
            .realization()
            .modalities()
            .contains(&M::Text));
        let image_request =
            eredu_core::MultimodalRequest::new(vec![eredu_core::MultimodalSegment::Media(
                eredu_core::Media::Image(eredu_core::RgbImage::new(vec![128; 48], 4, 4).unwrap()),
            )])
            .unwrap()
            .tokenize::<String>(|_| unreachable!())
            .unwrap();
        let mut image_mechanisms = RawMechanisms::default();
        let image_input = image_only
            .processor()
            .prepare(
                &image_request,
                &mut image_mechanisms,
                buffer_budget(),
                &mut |_| Ok::<_, String>(vec![]),
            )
            .unwrap();
        assert_eq!(
            image_only
                .processor()
                .admit(&image_input, &inspector)
                .unwrap()
                .token_ids(),
            [14, 12, 15]
        );

        let prepared_only = plan
            .select(
                &request,
                &facts,
                None,
                &processor_request(),
                &processor_facts(),
            )
            .unwrap();
        let mut raw_mechanisms = RawMechanisms::default();
        assert!(prepared_only
            .processor()
            .prepare(
                &raw_request(0),
                &mut raw_mechanisms,
                buffer_budget(),
                &mut |_| Ok::<_, String>(vec![4])
            )
            .is_err());
        assert_eq!(raw_mechanisms.tensors, 0);
        // Ordinary selected raw preparation needs no accounting budget.
        let mut ordinary_mechanisms = RawMechanisms::default();
        let ordinary = selected
            .processor()
            .prepared_processor(None)
            .prepare(&raw_request(0), &mut ordinary_mechanisms, &mut |_| {
                Ok::<_, String>(vec![4])
            })
            .unwrap();
        assert!(!ordinary.parts().is_empty());
        assert!(ordinary_mechanisms.tensors > 0);
        let mut timestamps = 0;
        let raw = selected
            .processor()
            .prepare(
                &raw_request(0),
                &mut raw_mechanisms,
                buffer_budget(),
                &mut |_| {
                    timestamps += 1;
                    Ok::<_, String>(vec![3 + timestamps])
                },
            )
            .unwrap();
        assert_eq!(timestamps, 2);
        let report = *raw.resources();
        assert_eq!(report.decoded_input_bytes, 584);
        assert_eq!(report.output_tensor_bytes, 2380);
        assert_eq!(report.decoder_positions, 16);
        assert!(report.host_buffer_bytes >= 2380 + 3840);
        assert!(report.host_buffer_bytes < 8192);
        assert!(report.planning_items > 0);
        // Reject plan storage before timestamp tokenization or native construction.
        let mut no_plan = buffer_budget();
        no_plan.planning_items = 0;
        let before = raw_mechanisms.tensors;
        let mut callbacks = 0;
        let error = selected
            .processor()
            .prepare(&raw_request(0), &mut raw_mechanisms, no_plan, &mut |_| {
                callbacks += 1;
                Ok::<_, String>(vec![4])
            })
            .unwrap_err();
        assert!(matches!(
            error,
            eredu_architectures::processor_execution::ProcessorExecutionError::Resources(
                eredu_runtime::processor_resources::ProcessorResourceError::Exhausted {
                    resource: "planning items",
                    ..
                }
            )
        ));
        assert_eq!(callbacks, 0);
        assert_eq!(raw_mechanisms.tensors, before);
        let exact = eredu_runtime::processor_resources::ProcessorRequestBudget {
            decoded_input_bytes: report.decoded_input_bytes,
            host_buffer_bytes: report.host_buffer_bytes,
            output_tensor_bytes: report.output_tensor_bytes,
            planning_items: report.planning_items,
            decoder_positions: report.decoder_positions,
        };
        let replay = selected
            .processor()
            .prepare(&raw_request(0), &mut raw_mechanisms, exact, &mut |_| {
                Ok::<_, String>(vec![4])
            })
            .unwrap();
        assert_eq!(*replay.resources(), report);
        for field in 0..5 {
            let mut denied = exact;
            match field {
                0 => denied.decoded_input_bytes -= 1,
                1 => denied.host_buffer_bytes -= 1,
                2 => denied.output_tensor_bytes -= 1,
                3 => denied.planning_items -= 1,
                _ => denied.decoder_positions -= 1,
            }
            let before = raw_mechanisms.tensors;
            let error = selected
                .processor()
                .prepare(&raw_request(0), &mut raw_mechanisms, denied, &mut |_| {
                    Ok::<_, String>(vec![4])
                })
                .unwrap_err();
            assert!(matches!(
                error,
                eredu_architectures::processor_execution::ProcessorExecutionError::Resources(_)
            ));
            assert_eq!(raw_mechanisms.tensors, before);
        }

        // Ordinary construction erases family type but retains exactly the same
        // request validation and optional coarse limits as the typed processor.
        let retained = selected.processor().prepared_processor(Some(exact));
        let retained_input = retained
            .prepare_budgeted(&raw_request(0), &mut raw_mechanisms, &mut |_| {
                Ok::<_, String>(vec![4])
            })
            .unwrap();
        assert_eq!(*retained_input.resources(), report);
        let plain = retained
            .prepare(&raw_request(0), &mut raw_mechanisms, &mut |_| {
                Ok::<_, String>(vec![4])
            })
            .unwrap();
        assert_eq!(plain.identity(), retained_input.identity());
        for field in 0..5 {
            let mut denied = exact;
            match field {
                0 => denied.decoded_input_bytes -= 1,
                1 => denied.host_buffer_bytes -= 1,
                2 => denied.output_tensor_bytes -= 1,
                3 => denied.planning_items -= 1,
                _ => denied.decoder_positions -= 1,
            }
            let processor = selected.processor().prepared_processor(Some(denied));
            let before = raw_mechanisms.tensors;
            let error = if field == 3 {
                processor
                    .prepare_with_observer(
                        &raw_request(0),
                        &mut raw_mechanisms,
                        &mut |_| Ok::<_, String>(vec![4]),
                        &mut eredu_runtime::NoopObserver,
                    )
                    .unwrap_err()
            } else if field == 4 {
                processor
                    .prepare_with_admission(
                        &raw_request(0),
                        &mut raw_mechanisms,
                        &mut |_| Ok::<_, String>(vec![4]),
                        ingress.admission_config(),
                        buffer_budget(),
                    )
                    .unwrap_err()
            } else {
                processor
                    .prepare(&raw_request(0), &mut raw_mechanisms, &mut |_| {
                        Ok::<_, String>(vec![4])
                    })
                    .unwrap_err()
            };
            assert!(matches!(
                error,
                eredu_architectures::processor_execution::ProcessorExecutionError::Resources(_)
            ));
            assert_eq!(raw_mechanisms.tensors, before);
        }
        for processor in [
            image_only
                .processor()
                .prepared_processor(Some(buffer_budget())),
            prepared_only
                .processor()
                .prepared_processor(Some(buffer_budget())),
        ] {
            let before = raw_mechanisms.tensors;
            let error = processor
                .prepare(&raw_request(0), &mut raw_mechanisms, &mut |_| {
                    panic!("unselected raw modality reached timestamp callback");
                    #[allow(unreachable_code)]
                    Ok::<_, String>(vec![4])
                })
                .unwrap_err();
            assert!(matches!(
                error,
                eredu_architectures::processor_execution::ProcessorExecutionError::Plan(_)
            ));
            assert_eq!(raw_mechanisms.tensors, before);
        }
        for mode in [
            Intervention::Shape,
            Intervention::Dtype,
            Intervention::Grid,
            Intervention::Token,
            Intervention::Values,
        ] {
            let result = retained.prepare_with_observer(
                &raw_request(0),
                &mut raw_mechanisms,
                &mut |_| Ok::<_, String>(vec![4]),
                &mut mode.clone(),
            );
            if matches!(mode, Intervention::Values) {
                let changed = result.unwrap();
                assert_eq!(changed.identity(), retained_input.identity());
                assert_ne!(
                    changed.parts()[2].payload().value().data,
                    retained_input.parts()[2].payload().value().data
                );
                selected.processor().admit(&changed, &inspector).unwrap();
            } else {
                assert!(
                    result.is_err(),
                    "invalid intervention was admitted: {mode:?}"
                );
            }
        }

        let raw_proof = selected.processor().admit(&raw, &inspector).unwrap();
        let encoder_memory = selected
            .encoder_memory::<NumericBackend>(&raw_proof, eredu_nn::TensorElementType::F32, &ctx)
            .unwrap();

        assert!(encoder_memory.host_setup_logical_bytes > 0);
        assert!(!encoder_memory.missing.is_empty());
        assert!(encoder_memory
            .invocations
            .iter()
            .any(|invocation| invocation.repetitions == 2));
        // NumericBackend has no native storage facts; retain those unknowns.
        assert!(encoder_memory
            .invocations
            .iter()
            .all(|invocation| !invocation.memory.missing.is_empty()));
        let request_rotary = selected
            .request_rotary_memory::<NumericBackend>(&raw_proof, &ctx)
            .unwrap();
        assert_eq!(
            request_rotary
                .values
                .iter()
                .find(|v| v.name == "cosine")
                .unwrap()
                .shape,
            vec![raw_proof.token_ids().len() as u64, 6]
        );
        assert!(!request_rotary.missing.is_empty());
        assert!(encoder_memory
            .invocations
            .iter()
            .any(|group| matches!(&group.invocation,
            eredu_nn::mechanism_memory::MechanismInvocation::MultiAxisRotary { position_shape, .. }
                if position_shape.last() == Some(&2))));
        let raw_prepared = raw_proof.prepare(&ctx).unwrap();
        assert_eq!(
            raw_proof.token_ids(),
            // The 64-pixel video budget covers four frames: each resizes to 4x4.
            &[3, 14, 12, 12, 12, 12, 15, 4, 14, 13, 15, 5, 14, 13, 15, 6]
        );
        for part in raw
            .parts()
            .iter()
            .filter(|p| matches!(p.modality(), M::Image | M::Video))
        {
            let row = &part.payload().value().data[..24];
            for channel in 0..3 {
                for value in &row[channel * 8..(channel + 1) * 8] {
                    assert!((*value - [64., 128., 192.][channel] / 255.).abs() < 1e-6);
                }
            }
        }
        let raw_expected = reference(target, ingress, &raw_prepared, &ctx);
        let raw_cursor_request =
            <Model as CompositeArchitecture<NumericBackend, State>>::prepare_prefill_request(
                raw.clone(),
                raw_proof.into_composite(),
                &ctx,
            )
            .unwrap();
        let before = raw_mechanisms.tensors;
        let error = selected
            .processor()
            .prepare(
                &raw_request(64),
                &mut raw_mechanisms,
                buffer_budget(),
                &mut |_| Ok::<_, String>(vec![4]),
            )
            .unwrap_err();
        assert!(matches!(
            error,
            eredu_architectures::processor_execution::ProcessorExecutionError::Resources(_)
        ));
        assert_eq!(
            raw_mechanisms.tensors, before,
            "host admission rejects before native tensors"
        );
        let (handoff, store, _processor) = selected
            .prepare_composite::<NumericBackend, State>(&ctx)
            .unwrap();
        let mut mechanisms = if residency.is_fully_resident() {
            NumericReplicatedMechanisms::with_bound_checkpoint(store)
        } else {
            NumericReplicatedMechanisms::with_bounded_checkpoint(store)
        };
        mechanisms.fixture_state_factory = Some(gguf::state_from_layout);
        reset_reference_stage_evidence("qwen4-processor");
        macro_rules! exercise {
            ($session:expr) => {{
                let mut session = $session;
                let empty = session.checkpoint(&ctx).unwrap();
                if let Some(cursor) = foreign_cursor.borrow_mut().as_mut() {
                    assert!(session
                        .advance_prefill(cursor, &ctx, &mut eredu_runtime::NoopObserver)
                        .is_err());
                    session::assert_checkpoint(&session.checkpoint(&ctx).unwrap(), &empty);
                }

                let inspected = inspector.0.get();
                for path in ["model.visual.blocks.1.output", "model.layers.0.mixer.write"] {
                    assert!(session
                        .prefill_input_with_observer(
                            PreparedCompositeInput::new(&source, &proof).unwrap(),
                            &ctx,
                            &mut FailAt(path),
                        )
                        .is_err());
                    session::assert_checkpoint(&session.checkpoint(&ctx).unwrap(), &empty);
                }
                let first = session
                    .prefill_input(PreparedCompositeInput::new(&source, &proof).unwrap(), &ctx)
                    .unwrap();
                assert_tensor_close(&first, &expected.0[0], "processor initial media");
                assert_eq!(
                    inspector.0.get(),
                    inspected,
                    "processor execution cannot re-inspect metadata"
                );
                let checkpoint = session.checkpoint(&ctx).unwrap();
                let vision_reads = vision_acquisitions();
                assert!(session
                    .prefill_input(PreparedCompositeInput::new(&source, &proof).unwrap(), &ctx)
                    .is_err());
                session::assert_checkpoint(&session.checkpoint(&ctx).unwrap(), &checkpoint);
                let frozen = session.capture_control_state(&ctx).unwrap();
                let mut fork = session.copy_control_state(&frozen, &ctx).unwrap();
                let one = input(vec![text(&[3])], &inspector);
                let admitted = admit(ingress, &one, &inspector);
                let a = session
                    .decode_input(PreparedCompositeInput::new(&one, &admitted).unwrap(), &ctx)
                    .unwrap();
                session.exchange_control_state(&mut fork, &ctx).unwrap();
                let b = session
                    .decode_input(PreparedCompositeInput::new(&one, &admitted).unwrap(), &ctx)
                    .unwrap();
                assert_tensor_exact(&a, &b, "processor control fork");
                session.rollback(checkpoint, &ctx).unwrap();
                for step in 0..16 {
                    let one = input(vec![text(&[step % 12])], &inspector);
                    let admitted = admit(ingress, &one, &inspector);
                    let value = session
                        .decode_input_with_observer(
                            PreparedCompositeInput::new(&one, &admitted).unwrap(),
                            &ctx,
                            &mut FailAt("model.visual.blocks.1.output"),
                        )
                        .unwrap();
                    assert_tensor_close(
                        &value,
                        &expected.0[step as usize + 1],
                        "processor cached media delta",
                    );
                }
                session::assert_checkpoint(&session.checkpoint(&ctx).unwrap(), &expected.1);
                assert_eq!(
                    vision_acquisitions(),
                    vision_reads,
                    "processor text decode skips tower"
                );
                session.reset(&ctx).unwrap();
                let (first, capture, _) = session
                    .prefill_input_prediction_target_observed(
                        PreparedCompositeInput::new(&projected_source, &projected_proof).unwrap(),
                        &ctx,
                        &mut FailAt("model.visual.blocks.1.output"),
                    )
                    .unwrap();
                assert_eq!(capture.shape(), [1, 18, 2, 32]);
                assert!(capture.data.iter().any(|value| value.abs() > 0.01));
                let last = first.dim(1) - 1;
                let first = first
                    .index(
                        &[Index::Full, Index::Range(last, last + 1), Index::Full],
                        &ctx,
                    )
                    .unwrap();
                assert_tensor_close(&first, &expected.0[0], "processor projected image/video");
                let next = session
                    .decode_input_with_observer(
                        PreparedCompositeInput::new(&projected_text, &projected_text_proof)
                            .unwrap(),
                        &ctx,
                        &mut FailAt("model.visual.blocks.1.output"),
                    )
                    .unwrap();
                assert_tensor_close(
                    &next,
                    &expected.0[1],
                    "processor projected text after media",
                );
                assert_eq!(
                    vision_acquisitions(),
                    vision_reads,
                    "projected inputs skip tower acquisitions"
                );
                session.reset(&ctx).unwrap();
                let oversized = input(vec![text(&[3; 33])], &inspector);
                let admitted = admit(ingress, &oversized, &inspector);
                let error = session
                    .prefill_input(
                        PreparedCompositeInput::new(&oversized, &admitted).unwrap(),
                        &ctx,
                    )
                    .unwrap_err();
                assert_media_error(&error);
                session::assert_checkpoint(&session.checkpoint(&ctx).unwrap(), &empty);
                for chunk_tokens in [5, 32] {
                    session.reset(&ctx).unwrap();
                    assert!(session
                        .start_prefill(Ok(chunk_request.clone()), 0, None, &ctx)
                        .is_err());
                    assert!(session
                        .start_prefill(Ok(chunk_request.clone()), 33, None, &ctx)
                        .is_err());
                    let mut cursor = session
                        .start_prefill(
                            Ok(chunk_request.clone()),
                            chunk_tokens,
                            Some(long_source.cache_identity("long cursor input").unwrap()),
                            &ctx,
                        )
                        .unwrap();
                    let empty = session.checkpoint(&ctx).unwrap();
                    if chunk_tokens == 5 {
                        assert!(session
                            .advance_prefill(
                                &mut cursor,
                                &ctx,
                                &mut FailAt("model.visual.blocks.1.output")
                            )
                            .is_err());
                        assert_eq!(cursor.position(), 0);
                        session::assert_checkpoint(&session.checkpoint(&ctx).unwrap(), &empty);
                    }
                    let completed_before =
                        REFERENCE_STAGE_EVIDENCE.with(|e| e.borrow().request_completions.len());
                    let step = session
                        .advance_prefill(&mut cursor, &ctx, &mut eredu_runtime::NoopObserver)
                        .unwrap();
                    assert_eq!(step.range, 0..chunk_tokens);
                    assert!(session.committed_prompt_input_identity().is_none());
                    let encoded = vision_acquisitions();
                    REFERENCE_STAGE_EVIDENCE.with(|e| {
                        assert!(
                            e.borrow().request_completions[completed_before..]
                                .iter()
                                .any(|shape| shape == &[1, 8, 32]),
                            "encoder result completes before cursor publication"
                        )
                    });
                    *foreign_cursor.borrow_mut() = Some(cursor.clone());
                    let saved = cursor.clone();
                    let frozen = session.capture_control_state(&ctx).unwrap();
                    let mut copied = session.copy_control_state(&frozen, &ctx).unwrap();
                    let prefix = session.checkpoint(&ctx).unwrap();
                    REFERENCE_STAGE_EVIDENCE
                        .with(|e| e.borrow_mut().fail_request_completion = Some(vec![1, 8, 32]));
                    assert!(session
                        .advance_prefill(&mut cursor, &ctx, &mut eredu_runtime::NoopObserver)
                        .is_err());
                    assert_eq!(cursor.position(), chunk_tokens);
                    session::assert_checkpoint(&session.checkpoint(&ctx).unwrap(), &prefix);
                    REFERENCE_STAGE_EVIDENCE
                        .with(|e| assert!(e.borrow().fail_request_completion.is_none()));
                    assert!(session
                        .advance_prefill(
                            &mut cursor,
                            &ctx,
                            &mut FailAt("model.layers.0.mixer.write")
                        )
                        .is_err());
                    assert_eq!(cursor.position(), chunk_tokens);
                    session::assert_checkpoint(&session.checkpoint(&ctx).unwrap(), &prefix);
                    session
                        .advance_prefill(&mut cursor, &ctx, &mut eredu_runtime::NoopObserver)
                        .unwrap();
                    let advanced = session.checkpoint(&ctx).unwrap();
                    let mut stale = saved.clone();
                    assert!(session
                        .advance_prefill(&mut stale, &ctx, &mut eredu_runtime::NoopObserver)
                        .is_err());
                    session::assert_checkpoint(&session.checkpoint(&ctx).unwrap(), &advanced);
                    session.exchange_control_state(&mut copied, &ctx).unwrap();
                    cursor = saved;
                    let last = session
                        .finish_prefill(
                            &mut cursor,
                            &ctx,
                            &mut FailAt("model.visual.blocks.1.output"),
                        )
                        .unwrap();
                    assert!(cursor.is_complete());
                    assert_eq!(cursor.position(), 41);
                    assert_eq!(
                        session.committed_prompt_input_identity(),
                        Some(&long_source.cache_identity("long cursor input").unwrap())
                    );
                    assert_tensor_close(&last, &long_expected.0[0], "encoder-once chunked prefill");
                    assert!(session
                        .advance_prefill(&mut cursor, &ctx, &mut eredu_runtime::NoopObserver)
                        .is_err());
                    assert_eq!(
                        vision_acquisitions(),
                        encoded,
                        "encoder must run once across committed chunks and restore"
                    );
                    for step in 0..16 {
                        let one = input(vec![text(&[step % 12])], &inspector);
                        let admitted = admit(ingress, &one, &inspector);
                        let value = session
                            .decode_input(
                                PreparedCompositeInput::new(&one, &admitted).unwrap(),
                                &ctx,
                            )
                            .unwrap();
                        assert_tensor_close(
                            &value,
                            &long_expected.0[step as usize + 1],
                            "long chunked cached decode",
                        );
                    }
                    session::assert_checkpoint(
                        &session.checkpoint(&ctx).unwrap(),
                        &long_expected.1,
                    );
                }
                // Prediction prefill exposes the same forward continuation as
                // the ordinary cursor. Media spanning a target chunk boundary
                // must reuse its completed encoder products in the next chunk.
                session.reset(&ctx).unwrap();
                let mut prediction_continuation = None;
                let mut prediction_last = None;
                let mut encoded_once = None;
                for start in (0..16).step_by(5) {
                    let end = (start + 5).min(16);
                    let (logits, capture, forward) =
                        eredu_runtime::prefill::ChunkedPrefillRequest::<
                            eredu_architectures::composite_execution::PreparedCompositeArchitecture<
                                Model,
                            >,
                            NumericBackend,
                            State,
                        >::with_chunk(
                            &raw_cursor_request,
                            start..end,
                            prediction_continuation.as_ref(),
                            |input| session.prefill_input_prediction_target(input, &ctx),
                        )
                        .unwrap()
                        .unwrap();
                    assert_eq!(capture.dim(1) as usize, end - start);
                    let next = eredu_runtime::prefill::ChunkedPrefillRequest::<
                        eredu_architectures::composite_execution::PreparedCompositeArchitecture<
                            Model,
                        >,
                        NumericBackend,
                        State,
                    >::continuation(&raw_cursor_request, &forward);
                    assert!(next
                        .as_ref()
                        .unwrap()
                        .data
                        .iter()
                        .any(|value| value.abs() > 0.01));
                    if let Some(count) = encoded_once {
                        assert_eq!(
                            vision_acquisitions(),
                            count,
                            "prediction chunks reuse encoder continuation"
                        );
                    } else {
                        encoded_once = Some(vision_acquisitions());
                    }
                    prediction_continuation = Some(next);
                    prediction_last = Some(logits);
                }
                let logits = prediction_last.unwrap();
                let last = logits
                    .index(
                        &[
                            Index::Full,
                            Index::Range(logits.dim(1) - 1, logits.dim(1)),
                            Index::Full,
                        ],
                        &ctx,
                    )
                    .unwrap();
                assert_tensor_close(&last, &raw_expected.0[0], "prediction media continuation");
                session.reset(&ctx).unwrap();
                let mut cursor = session
                    .start_prefill(Ok(raw_cursor_request.clone()), 5, None, &ctx)
                    .unwrap();
                let value = session
                    .finish_prefill(&mut cursor, &ctx, &mut eredu_runtime::NoopObserver)
                    .unwrap();
                assert_tensor_close(&value, &raw_expected.0[0], "selected raw processor prefill");
                for step in 0..16 {
                    let one = input(vec![text(&[step % 12])], &inspector);
                    let proof = _processor.admit(&one, &inspector).unwrap().into_composite();
                    let value = session
                        .decode_input(PreparedCompositeInput::new(&one, &proof).unwrap(), &ctx)
                        .unwrap();
                    assert_tensor_close(
                        &value,
                        &raw_expected.0[step as usize + 1],
                        "selected raw processor decode",
                    );
                }
                session::assert_checkpoint(&session.checkpoint(&ctx).unwrap(), &raw_expected.1);
                Ok::<_, String>(())
            }};
        }
        eredu_architectures::prepared_execution::construct_selected_routed_session(
            handoff,
            mechanisms,
            &ctx,
            |_, _, rows| {
                let rows = rows.unwrap();
                let providers = rows
                    .prepared()
                    .bind(
                        rows.prepared()
                            .entries()
                            .iter()
                            .map(|(id, e)| {
                                (
                                    id.clone(),
                                    super::super::super::row_bank::SourceRows::new(e),
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
            |_, session, _| exercise!(session),
            |_, session, _| exercise!(session),
        )
        .unwrap();
        REFERENCE_STAGE_EVIDENCE.with(|e| {
            if !residency.is_fully_resident() {
                assert_eq!(e.borrow().peak_bound_units, 1);
            }
        });
    }
}

#[test]
fn media_policy_and_processor_selection_need_no_readable_artifact() {
    use eredu_architectures::qwen4_exp::media::MediaAdmissionConfig;
    let (dir, target, ingress) = setup();
    let spec = target.spec().clone();
    let vision = ingress.vision().config().clone();
    let media = ingress.vision().media_tokens().clone();
    let processor = ingress.vision().processor().cloned();
    let expected_identity = ingress.admission_config().identity().to_owned();
    let expected_requirements = ingress.processor_requirements().unwrap();
    // Keep only geometry: neither this policy nor its selected processor retains
    // a readable source. The source directory is removed before either is built.
    drop(ingress);
    drop(target);
    dir.close().unwrap();
    let policy = MediaAdmissionConfig::new(&spec, &vision, &media).unwrap();
    assert_eq!(policy.identity(), expected_identity);
    assert_eq!(
        policy.processor_requirements(processor.as_ref()).unwrap(),
        expected_requirements
    );
    let selected = policy
        .select_processor(processor.as_ref(), &processor_request(), &processor_facts())
        .unwrap();
    let inspector = Inspector(0.into());
    let request = prompt(&inspector);
    let admitted = selected.admit(&request, &inspector).unwrap();
    let prepared = admitted.prepare(&NumericContext::default()).unwrap();
    assert_eq!(prepared.token_ids().dim(1), 18);

    let mut wrong_vision = vision.clone();
    wrong_vision.out_hidden_size += 1;
    assert!(MediaAdmissionConfig::new(&spec, &wrong_vision, &media).is_err());
    let mut wrong_media = media.clone();
    wrong_media.image = u32::MAX;
    assert!(MediaAdmissionConfig::new(&spec, &vision, &wrong_media).is_err());
    wrong_vision = vision.clone();
    wrong_vision.patch_size = 0;
    assert!(MediaAdmissionConfig::new(&spec, &wrong_vision, &media).is_err());
    wrong_vision = vision.clone();
    let mut layers = vision.layer_schedule.iter().cloned().collect::<Vec<_>>();
    layers[0].deepstack_merger = Some(0);
    wrong_vision.layer_schedule =
        eredu_core::attention::LayerSchedule::new(layers.len(), layers).unwrap();
    assert!(MediaAdmissionConfig::new(&spec, &wrong_vision, &media).is_err());
    wrong_media = media.clone();
    wrong_media.video = wrong_media.image;
    assert!(MediaAdmissionConfig::new(&spec, &vision, &wrong_media).is_err());
}

#[derive(Debug, Clone)]
enum Intervention {
    Shape,
    Dtype,
    Grid,
    Token,
    Values,
}
impl
    eredu_runtime::ActivationObserver<
        NumericTensor,
        eredu_architectures::processor_execution::ProcessorExecutionError<String, String>,
    > for Intervention
{
    fn observe(
        &mut self,
        _: &str,
        _: &NumericTensor,
    ) -> Result<(), eredu_architectures::processor_execution::ProcessorExecutionError<String, String>>
    {
        Ok(())
    }
    fn intervene(
        &mut self,
        path: &str,
        value: &NumericTensor,
    ) -> Result<
        Option<NumericTensor>,
        eredu_architectures::processor_execution::ProcessorExecutionError<String, String>,
    > {
        let first = path == format!("{}.0", eredu_core::PROCESSOR_OUTPUT_OBSERVATION_PATH);
        let mut changed = value.clone();
        match self {
            Self::Shape if first => {
                changed.shape = vec![1, 2];
                changed.data = vec![3., 4.];
                changed.exact_i32 = Some(std::sync::Arc::new(vec![3, 4]));
            }
            Self::Dtype if first => changed.dtype = eredu_core::checkpoint::TensorDtype::F32,
            Self::Grid if path.ends_with("metadata.PatchGrid") => {
                changed.data[0] = 0.;
                changed.exact_i32 = Some(std::sync::Arc::new(vec![0, 2, 2]));
            }
            Self::Token if first => {
                changed.data[0] = 1_000_000.;
                changed.exact_i32 = Some(std::sync::Arc::new(vec![1_000_000]));
            }
            Self::Values if value.dtype == eredu_core::checkpoint::TensorDtype::F32 => {
                changed.data[0] += 0.25
            }
            _ => return Ok(None),
        }
        Ok(Some(changed))
    }
}
