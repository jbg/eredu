include!("component_capture/routed.rs");
include!("component_capture/additive.rs");

fn component_capture_prompt_tokens() -> Vec<u32> {
    if std::env::var_os(COMPONENT_CAPTURE_MEDIA).is_some() {
        vec![1, 2, 42, 42]
    } else {
        vec![1, 2]
    }
}

fn component_capture_prompt(runtime: &ModelRuntime<MlxBackend<'_>>) -> MlxModelInput {
    use eredu_core::TextGenerationBackend as _;
    if std::env::var_os(COMPONENT_CAPTURE_MEDIA).is_none() {
        return MlxBackend::prepare_text_prompt(
            runtime.backend(),
            component_capture_prompt_tokens(),
        )
        .unwrap();
    }
    let tokens = Array::from_slice(&[1u32, 2], &[1, 2]);
    let gemma = runtime
        .session()
        .effective_model_type()
        .starts_with("gemma4");
    let patch_width = std::env::var(COMPONENT_CAPTURE_PATCH_WIDTH)
        .map(|value| value.parse::<i32>().unwrap())
        .unwrap_or(12);
    let pixel_shape = if gemma {
        vec![1, 8, patch_width]
    } else {
        vec![8, patch_width]
    };
    let pixels = Array::from_slice(
        &(0..8 * patch_width)
            .map(|index| (index as f32 - 41.0) / 97.0)
            .collect::<Vec<_>>(),
        &pixel_shape,
    );
    let grid = Array::from_slice(&[1i32, 2, 4], &[1, 3]);
    if gemma {
        let positions = Array::from_slice(
            &(0..8).flat_map(|i| [i / 4, i % 4]).collect::<Vec<i32>>(),
            &[1, 8, 2],
        );
        let parts = vec![
            text_input_part(&tokens),
            input_part(
                InputModality::Image,
                InputPayload::Tensor(pixels),
                [
                    (InputMetadataKey::PatchGrid, grid),
                    (InputMetadataKey::PatchPositions, positions),
                ],
                [InputExtent::PatchGrid {
                    time: 1,
                    height: 2,
                    width: 4,
                }],
            ),
        ];
        return synthetic_prediction_input(&parts, &component_capture_prompt_tokens());
    }
    let parts = vec![
        text_input_part(&tokens),
        input_part(
            InputModality::Image,
            InputPayload::Tensor(pixels),
            [(InputMetadataKey::PatchGrid, grid)],
            [],
        ),
    ];
    synthetic_prediction_input(&parts, &component_capture_prompt_tokens())
}

fn component_capture_generation<'a, 'world>(
    runtime: &'a mut ModelRuntime<MlxBackend<'world>>,
    sampling: eredu_core::ResolvedGenerationConfig,
    options: Option<eredu_core::TextPreparationOptions>,
) -> Result<ComponentState<'a, 'world>, ComponentPreparationError> {
    component_generation_start_with_tokens(
        runtime,
        options,
        TextGenerationConfig::new(sampling),
        &component_capture_prompt_tokens(),
        std::env::var_os(COMPONENT_CAPTURE_MEDIA).is_some(),
    )
}

fn verify_loaded_component_capture(
    runtime: &mut ModelRuntime<MlxBackend<'_>>,
    reference: &mut ModelRuntime<MlxBackend<'_>>,
    stream: &Stream,
    loop_normalization: Option<(&str, &str)>,
    required_unit_points: &[&str],
    stream_readout: Option<&eredu_core::component::ComponentReadout>,
) {
    use eredu_core::{ObservationSupportStatus, TextGenerationBackend as _, capture::*};
    runtime.synchronize().unwrap();
    let count = if std::env::var_os(COMPONENT_CAPTURE_MEDIA).is_some() {
        <MlxBackend as eredu_core::ModelCapabilityBackend>::count_prepared_input(
            runtime,
            &component_capture_prompt(runtime),
        )
        .unwrap()
    } else {
        use eredu_runtime::working_memory::OriginalTokenizerBackend;
        let tokens = component_capture_prompt_tokens();
        let model = runtime.session().original_model_source().unwrap();
        let (_tokenizer, prepared) = crate::tests::support::plain_controller::source(
            runtime.backend().memory_ledger(),
            model.erased().inference_execution_identity(),
        );
        let prompt = MlxBackend::prepare_semantic_prompt(
            runtime,
            &prepared,
            &eredu_core::TokenIdsInputPlan::new(&tokens).unwrap(),
            None,
        )
        .unwrap();
        <MlxBackend as eredu_core::ModelCapabilityBackend>::count_prepared_input(runtime, &prompt)
            .unwrap()
    };
    assert_eq!(count.text_tokens, 2);
    assert_eq!(
        count.model_positions,
        component_capture_prompt_tokens().len() as u64
    );
    assert_eq!(count.kind, eredu_core::ObservationKind::Exact);
    if std::env::var_os(COMPONENT_CAPTURE_MEDIA).is_some() {
        assert_eq!(count.media_positions, 2);
        assert!(count.media_execution_workspace_bytes() >= 96 * 4);
    } else {
        assert_eq!(count.media_positions, 0);
    }
    let discovery = MlxBackend::capture_discovery(runtime).unwrap();
    let selections = discovery
        .catalog
        .points
        .iter()
        .filter(|point| matches!(point.value_type, eredu_core::ObservationValueType::Tensor))
        .filter(|point| {
            discovery.support.points.iter().any(|support| {
                support.path == point.path
                    && ((support.prefill == ObservationSupportStatus::Supported
                        && support.decode == ObservationSupportStatus::Supported)
                        || (std::env::var_os(COMPONENT_CAPTURE_MEDIA).is_some()
                            && point.path.contains(".deepstack.")
                            && matches!(
                                support.prefill,
                                ObservationSupportStatus::Supported
                                    | ObservationSupportStatus::Conditional(_)
                            )))
            })
        })
        .map(|point| {
            let support = discovery
                .support
                .points
                .iter()
                .find(|support| support.path == point.path)
                .unwrap();
            assert!(
                matches!(
                    support.prefill,
                    ObservationSupportStatus::Supported | ObservationSupportStatus::Conditional(_)
                ),
                "{}",
                point.path
            );
            CaptureSelection {
                id: point.path.clone(),
                path: point.path.clone(),
                schedule: CaptureSchedule {
                    decode: support.decode == ObservationSupportStatus::Supported,
                    ..CaptureSchedule::default()
                },
                slices: vec![],
                transform: CaptureTransform::FullTensor,
            }
        })
        .collect::<Vec<_>>();
    assert!(
        selections.len() >= 4,
        "actual and effective FFN/attention components"
    );
    for path in [
        "readout.embedding",
        "readout.embedding.effective",
        "readout.residual",
        "readout.residual.effective",
        "readout.normalized",
        "readout.normalized.effective",
        "readout.linear",
        "readout.linear.effective",
        "model.logits",
    ]
    .into_iter()
    .chain(required_unit_points.iter().copied())
    .chain(
        loop_normalization
            .into_iter()
            .flat_map(|(input, output)| [input, output]),
    ) {
        assert!(
            selections.iter().any(|selection| selection.path == path),
            "loaded global observation support for {path}: {:?}",
            discovery
                .support
                .points
                .iter()
                .find(|point| point.path == path)
        );
    }
    if std::env::var_os(COMPONENT_CAPTURE_MEDIA).is_some() {
        for point in discovery
            .catalog
            .points
            .iter()
            .filter(|point| point.path.ends_with(".deepstack.input"))
        {
            let prefix = point.path.strip_suffix("input").unwrap();
            for suffix in [
                "input",
                "output",
                "output.effective",
                "residual",
                "residual.effective",
            ] {
                let path = format!("{prefix}{suffix}");
                let selected = selections
                    .iter()
                    .find(|selection| selection.path == path)
                    .unwrap_or_else(|| panic!("missing prepared media component {path}"));
                assert!(selected.schedule.prefill);
                assert!(!selected.schedule.decode);
            }
        }
    }
    if let Some(readout) = stream_readout {
        let path = readout.projection_input.as_ref().unwrap();
        assert!(
            selections.iter().any(|selection| &selection.path == path),
            "loaded stream readout must expose its actual projection input: {:?}",
            discovery
                .support
                .points
                .iter()
                .find(|point| &point.path == path)
        );
    }
    let usage = CaptureUsage {
        captures: 100_000,
        retained_bytes: 512 << 20,
        // Includes globally charged receipt decoding on every receiver. The
        // expanded normalized/readout plan has many small replicated records;
        // decoder envelope reservations dominate its actual tensor payloads.
        // Eight-rank TP/EP/PP cases price every producer on every receiver,
        // including the parser's 128-byte bound per encoded byte. Their global
        // work scales quadratically with participant count. Reserve four times
        // the four-rank allowance for these cases; no physical allocation is implied.
        host_bytes: 4 << 30,
        encoded_bytes: 512 << 20,
    }
    .checked_mul(
        if std::env::var(CARTESIAN_AXES).as_deref() == Ok("tp-pp-ep") {
            4
        } else {
            1
        },
    )
    .unwrap();
    let plan = CapturePlan {
        schema_version: 1,
        selections,
        limits: CaptureLimits {
            per_step: usage,
            cumulative: usage.checked_mul(8).unwrap(),
            on_limit: CaptureLimitPolicy::Fail,
        },
    }
    .admit(
        &discovery.catalog,
        &discovery.support,
        &discovery.support.capture,
        CaptureRequestShape {
            batch: 1,
            prompt_tokens: component_capture_prompt_tokens().len() as u64,
            max_predictions: 3,
        },
    )
    .unwrap();
    let mut skip = plan.plan().clone();
    skip.limits.on_limit = CaptureLimitPolicy::Skip;
    skip.limits.per_step.captures = 0;
    let skip = skip
        .admit(
            &discovery.catalog,
            &discovery.support,
            &discovery.support.capture,
            plan.request(),
        )
        .unwrap();
    MlxBackend::validate_text_capture(runtime, &skip).unwrap();
    // The independently loaded oracle keeps its completed parameter custody
    // through capture; reset only its mutable generation state.
    reference.reset().unwrap();
    reference.synchronize().unwrap();
    let sampling = eredu_core::resolve_generation_config(
        None,
        eredu_core::GenerationConfigOverrides {
            max_new_tokens: Some(3),
            temperature: Some(0.0),
            ..Default::default()
        },
    )
    .unwrap();
    let run = |runtime: &mut ModelRuntime<MlxBackend<'_>>, plan: &AdmittedCapturePlan| {
        let mut generation = component_capture_generation(
            runtime,
            sampling,
            Some(eredu_core::TextPreparationOptions {
                capture: Some(SharedCapturePlan::new(plan.clone())),
                interventions: None,
            }),
        )
        .unwrap();
        let mut tokens = Vec::new();
        let mut steps = Vec::new();
        for _ in 0..3 {
            tokens.push(generation.next().unwrap().unwrap().token_id());
            steps.push(generation.take_captured_delivery().unwrap().unwrap());
        }
        (tokens, steps)
    };
    let expected = run(reference, &plan);
    if std::env::var_os(COMPONENT_CAPTURE_MEDIA).is_some() {
        verify_component_prefill_geometry_rejection(runtime, &plan, sampling);
    }
    verify_cold_component_preparation(runtime, &plan, sampling, stream);
    let actual = run(runtime, &plan);
    if let Some(readout) = stream_readout {
        v4_components::verify_partitioned_readout(runtime, readout, &actual.1);
    }
    let additive_targets = actual.1[0]
        .partitions
        .iter()
        .filter(|evidence| evidence.combination == PartitionCaptureCombination::SumF64ToF32)
        .map(|evidence| &plan.plan().selections[evidence.context.selection_index].path)
        .filter(|path| {
            discovery.catalog.get(path).unwrap().position
                == eredu_core::ObservationPosition::BeforeIntervention
        })
        .cloned()
        .collect::<Vec<_>>();
    eprintln!(
        "global component capture: {} selections, prefill usage {:?}",
        plan.plan().selections.len(),
        actual.1[0].step_usage
    );
    assert_eq!(actual.0, expected.0);
    let run_identity = actual.1[0].partitions[0].context.run_identity.clone();
    for (prediction, (actual, expected)) in actual.1.iter().zip(&expected.1).enumerate() {
        if let Some((input, output)) = loop_normalization {
            let values = |path| {
                let record = actual
                    .records
                    .iter()
                    .find(|record| record.path == path)
                    .unwrap();
                let Some(value) = record.payload.as_ref().and_then(CapturePayload::as_tensor)
                else {
                    panic!("loop normalization tensor")
                };
                let eredu_core::TensorObservationData::F32(values) = value.data() else {
                    panic!("F32 loop normalization")
                };
                values
            };
            let (before, after) = (values(input), values(output));
            assert!(before.iter().zip(after).any(|(a, b)| (a - b).abs() > 1e-4));
            for (before, after) in before.chunks_exact(32).zip(after.chunks_exact(32)) {
                let rms = (before.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>() / 32.0 + 1e-6)
                    .sqrt();
                for (index, (&input, &output)) in before.iter().zip(after).enumerate() {
                    let expected = f64::from(input) / rms * (0.8 + index as f64 * 0.005);
                    assert!((f64::from(output) - expected).abs() < 2e-4);
                }
            }
        }
        assert_eq!(actual.prediction_index, prediction as u64);
        assert_eq!(actual.records.len(), expected.records.len());
        assert_eq!(
            actual.partitions.len(),
            actual
                .records
                .iter()
                .filter(|record| record.payload.is_some())
                .count()
        );
        for evidence in &actual.partitions {
            assert_eq!(evidence.context.run_identity, run_identity);
            assert_eq!(evidence.context.prediction, prediction as u64);
            assert!(evidence.context.forward_epoch > 0);
        }
        let mut deviations = Vec::new();
        for (actual, expected) in actual.records.iter().zip(&expected.records) {
            let path = &actual.path;
            assert_eq!(actual.path, expected.path);
            assert_eq!(actual.outcome, expected.outcome);
            if matches!(
                actual.outcome,
                CaptureOutcome::Skipped {
                    reason: CaptureSkipReason::Schedule
                }
            ) {
                assert!(actual.payload.is_none() && expected.payload.is_none());
                continue;
            }
            assert_eq!(actual.outcome, CaptureOutcome::Captured);
            let (Some(actual), Some(expected)) = (
                actual.payload.as_ref().and_then(CapturePayload::as_tensor),
                expected
                    .payload
                    .as_ref()
                    .and_then(CapturePayload::as_tensor),
            ) else {
                panic!("component capture must produce global tensors")
            };
            assert_eq!(actual.shape(), expected.shape());
            let (
                eredu_core::TensorObservationData::F32(actual),
                eredu_core::TensorObservationData::F32(expected),
            ) = (actual.data(), expected.data())
            else {
                panic!("F32 fixture")
            };
            assert!(expected.iter().any(|value| value.abs() > 1e-7));
            if let Some((index, (actual, expected))) =
                actual
                    .iter()
                    .zip(expected)
                    .enumerate()
                    .find(|(_, (actual, expected))| {
                        (*actual - *expected).abs() > 2e-4 + 2e-4 * expected.abs()
                    })
            {
                deviations.push(format!("{path}[{index}]: {actual} != {expected}"));
            }
        }
        assert!(
            deviations.is_empty(),
            "component prediction {prediction} deviations: {deviations:#?}"
        );
    }
    runtime.synchronize().unwrap();
    reference.synchronize().unwrap();
    // Select only the final seam: success must not depend on an upstream
    // readout capture having forced every rank's lazy vocabulary projection.
    let mut logits_plan = plan.plan().clone();
    logits_plan
        .selections
        .retain(|selection| selection.path == eredu_core::MODEL_LOGITS_OBSERVATION_PATH);
    assert_eq!(logits_plan.selections.len(), 1);
    let logits_plan = logits_plan
        .admit(
            &discovery.catalog,
            &discovery.support,
            &discovery.support.capture,
            plan.request(),
        )
        .unwrap();
    runtime.reset().unwrap();
    let logits_only = run(runtime, &logits_plan);
    assert_eq!(logits_only.0, expected.0);
    for (actual, expected) in logits_only.1.iter().zip(&expected.1) {
        assert_eq!(actual.partitions[0].producers.len(), 1);
        assert!(actual.capture_seconds > 0.0);
        let expected = expected
            .records
            .iter()
            .find(|record| record.path == eredu_core::MODEL_LOGITS_OBSERVATION_PATH)
            .unwrap();
        let (Some(actual), Some(expected)) = (
            actual.records[0]
                .payload
                .as_ref()
                .and_then(CapturePayload::as_tensor),
            expected
                .payload
                .as_ref()
                .and_then(CapturePayload::as_tensor),
        ) else {
            panic!("final logits")
        };
        assert_eq!(actual.shape(), expected.shape());
        let (
            eredu_core::TensorObservationData::F32(actual),
            eredu_core::TensorObservationData::F32(expected),
        ) = (actual.data(), expected.data())
        else {
            panic!("F32 final logits")
        };
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() <= 2e-4 + 2e-4 * expected.abs());
        }
    }
    runtime.synchronize().unwrap();
    // Each reduction is the only selected hook. Compare with independently
    // exported complete final rows, never a softmax over truncated candidates.
    for transform in [
        CaptureTransform::TokenScores {
            token_ids: vec![0, 3],
        },
        CaptureTransform::TopCandidates { count: 3 },
    ] {
        let mut reduced_plan = logits_plan.plan().clone();
        reduced_plan.selections[0].transform = transform;
        let reduced_plan = reduced_plan
            .admit(
                &discovery.catalog,
                &discovery.support,
                &discovery.support.capture,
                plan.request(),
            )
            .unwrap();
        runtime.reset().unwrap();
        let reduced = run(runtime, &reduced_plan);
        assert_eq!(reduced.0, expected.0);
        for (step, raw) in reduced.1.iter().zip(&logits_only.1) {
            assert_eq!(step.partitions[0].producers.len(), 1);
            assert_eq!(step.records[0].outcome, CaptureOutcome::Captured);
            let Some(raw) = raw.records[0]
                .payload
                .as_ref()
                .and_then(CapturePayload::as_tensor)
            else {
                panic!("raw final scores")
            };
            let eredu_core::TensorObservationData::F32(values) = raw.data() else {
                panic!("F32 scores")
            };
            let vocabulary = *raw.shape().last().unwrap();
            let row = &values[values.len() - vocabulary..];
            let maximum = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
            let log_mass = row
                .iter()
                .map(|score| (f64::from(*score) - maximum).exp())
                .sum::<f64>()
                .ln();
            match step.records[0].payload.as_ref().unwrap() {
                CapturePayload::TokenScores(value) => {
                    assert_eq!(value.vocabulary, vocabulary as u64);
                    assert!((value.log_partition - maximum - log_mass).abs() < 2e-5);
                    for (score, id) in value.scores.iter().zip([0u32, 3]) {
                        assert_eq!(score.target.token_id, id);
                        assert!((score.target.score - row[id as usize]).abs() < 2e-5);
                        assert!(
                            (score.log_probability
                                - (f64::from(row[id as usize]) - maximum - log_mass))
                                .abs()
                                < 2e-5
                        );
                        assert_eq!(
                            score.rank,
                            1 + row
                                .iter()
                                .filter(|other| **other > row[id as usize])
                                .count() as u64
                        );
                        let alternative = score.strongest_alternative.as_ref().unwrap();
                        assert_ne!(alternative.token_id, id);
                        let expected = row
                            .iter()
                            .enumerate()
                            .filter(|(index, _)| *index != id as usize)
                            .map(|(_, score)| *score)
                            .fold(f32::NEG_INFINITY, f32::max);
                        assert!((alternative.score - expected).abs() < 2e-5);
                        assert!(
                            (alternative.score - row[alternative.token_id as usize]).abs() < 2e-5
                        );
                    }
                }
                CapturePayload::Candidates(value) => {
                    assert_eq!(value.candidates.len(), 3);
                    let mut ordered = row.to_vec();
                    ordered.sort_by(|left, right| right.total_cmp(left));
                    for (index, candidate) in value.candidates.iter().enumerate() {
                        assert!((candidate.score - ordered[index]).abs() < 2e-5);
                        assert!((candidate.score - row[candidate.token_id as usize]).abs() < 2e-5);
                        assert!(
                            value.candidates[..index]
                                .iter()
                                .all(|other| other.token_id != candidate.token_id)
                        );
                    }
                }
                _ => panic!("bounded vocabulary reduction"),
            }
        }
        runtime.synchronize().unwrap();
    }
    let mut preview_plan = plan.plan().clone();
    for selection in &mut preview_plan.selections {
        selection.transform = CaptureTransform::Preview { max_elements: 7 };
    }
    let preview_plan = preview_plan
        .admit(
            &discovery.catalog,
            &discovery.support,
            &discovery.support.capture,
            plan.request(),
        )
        .unwrap();
    runtime.reset().unwrap();
    let previews = run(runtime, &preview_plan);
    assert_eq!(previews.0, expected.0);
    for (preview, complete) in previews.1.iter().zip(&expected.1) {
        for (preview, complete) in preview.records.iter().zip(&complete.records) {
            assert_eq!(preview.path, complete.path);
            assert_eq!(preview.selected_shape, complete.selected_shape);
            if matches!(
                complete.outcome,
                CaptureOutcome::Skipped {
                    reason: CaptureSkipReason::Schedule
                }
            ) {
                assert_eq!(preview.outcome, complete.outcome);
                assert!(preview.payload.is_none() && complete.payload.is_none());
                continue;
            }
            let (Some(prefix), Some(full)) = (
                preview.payload.as_ref().and_then(CapturePayload::as_tensor),
                complete
                    .payload
                    .as_ref()
                    .and_then(CapturePayload::as_tensor),
            ) else {
                panic!("preview tensors")
            };
            let (
                eredu_core::TensorObservationData::F32(prefix),
                eredu_core::TensorObservationData::F32(full),
            ) = (prefix.data(), full.data())
            else {
                panic!("F32 preview")
            };
            let count = full.len().min(7);
            assert_eq!(prefix.len(), count);
            assert_eq!(
                preview.outcome,
                if count < full.len() {
                    CaptureOutcome::Truncated {
                        available_elements: full.len() as u64,
                        emitted_elements: count as u64,
                    }
                } else {
                    CaptureOutcome::Captured
                }
            );
            for (actual, expected) in prefix.iter().zip(full) {
                assert!((actual - expected).abs() <= 2e-4 + 2e-4 * expected.abs());
            }
        }
    }
    runtime.synchronize().unwrap();
    verify_cold_component_preparation(runtime, &plan, sampling, stream);
    verify_component_capture_branches(runtime, &plan, sampling);
    verify_loaded_component_interventions(
        runtime,
        reference,
        &plan,
        sampling,
        required_unit_points,
    );
    verify_additive_component_interventions(runtime, reference, &plan, sampling, &additive_targets);
    verify_additive_component_transforms(runtime, reference, &plan, sampling, &additive_targets);
    runtime.reset().unwrap();
    let skipped = run(runtime, &skip);
    assert_eq!(
        skipped.0, expected.0,
        "skipping capture preserves ordinary sampled tokens"
    );
    for step in skipped.1 {
        assert!(step.partitions.is_empty());
        assert!(step.records.iter().all(|record| record.payload.is_none()
            && matches!(
                record.outcome,
                CaptureOutcome::Skipped {
                    reason: CaptureSkipReason::Limit {
                        budget: CaptureBudget::Captures,
                        cumulative: false
                    } | CaptureSkipReason::Schedule
                }
            )));
        assert_eq!(step.step_usage.captures, 0);
        assert!(
            step.step_usage.host_bytes > 0 && step.step_usage.retained_bytes > 0,
            "mandatory coordination and completed preparation remain charged"
        );
    }
    verify_loaded_routed_capture(runtime, reference, &plan, sampling);
    verify_cold_preparation_fencing(runtime);
}

fn verify_component_prefill_geometry_rejection(
    runtime: &mut ModelRuntime<MlxBackend<'_>>,
    plan: &eredu_core::capture::AdmittedCapturePlan,
    sampling: eredu_core::ResolvedGenerationConfig,
) {
    use eredu_core::{TextGenerationBackend as _, capture::CaptureError};
    let discovery = MlxBackend::capture_discovery(runtime).unwrap();
    let mut wrong = plan.request();
    wrong.prompt_tokens -= 1;
    let wrong = plan
        .plan()
        .clone()
        .admit(
            &discovery.catalog,
            &discovery.support,
            &discovery.support.capture,
            wrong,
        )
        .unwrap();
    let before = runtime
        .session()
        .original_model_source()
        .unwrap()
        .erased()
        .fixed_numeric_state_snapshot()
        .unwrap();
    {
        let error = match component_capture_generation(
            runtime,
            sampling,
            Some(eredu_core::TextPreparationOptions {
                capture: Some(eredu_core::capture::SharedCapturePlan::new(wrong)),
                interventions: None,
            }),
        ) {
            Err(error) => error,
            Ok(mut generation) => match generation.next().unwrap() {
                Ok(_) => panic!("mismatched media capture input entered model execution"),
                Err(error) => error,
            },
        };
        let mut source: &(dyn std::error::Error + 'static) = &error;
        loop {
            if let Some(CaptureError::Invalid(message)) = source.downcast_ref::<CaptureError>() {
                assert!(
                    message.contains("prepared prefill geometry [1, 4]"),
                    "{message}"
                );
                break;
            }
            source = source
                .source()
                .expect("prefill rejection retains its neutral geometry cause");
        }
    }
    runtime.synchronize().unwrap();
    assert_eq!(
        runtime
            .session()
            .original_model_source()
            .unwrap()
            .erased()
            .fixed_numeric_state_snapshot()
            .unwrap(),
        before
    );
}

fn verify_cold_preparation_fencing(runtime: &mut ModelRuntime<MlxBackend<'_>>) {
    use eredu_core::run_preparation::*;
    let rank = eredu_core::DistributedSession::descriptor(
        <MlxBackend<'_> as eredu_core::DistributedBackend>::distributed_session(runtime.session())
            .unwrap(),
    )
    .rank();
    let before = runtime.text_preparation_usage().unwrap();
    let stage = if rank == 0 {
        TextPreparationStage::Request
    } else {
        TextPreparationStage::Prompt
    };
    let error = runtime
        .agree_text_preparation(stage, TextPreparationStatus::Ready)
        .unwrap_err();
    let mut cause: &(dyn std::error::Error + 'static) = &error;
    loop {
        if let Some(eredu_runtime::run_preparation::TextPreparationAgreementError::Protocol(_)) =
            cause.downcast_ref::<eredu_runtime::run_preparation::TextPreparationAgreementError>()
        {
            break;
        }
        cause = cause.source().expect("typed protocol mismatch");
    }
    let after = runtime.text_preparation_usage().unwrap();
    assert_eq!(after.attempts, before.attempts + 1);
    assert!(
        runtime
            .agree_text_preparation(TextPreparationStage::Request, TextPreparationStatus::Ready)
            .is_err()
    );
    assert_eq!(runtime.text_preparation_usage().unwrap(), after);
    assert!(
        runtime.synchronize().is_err(),
        "completion cannot clear a communication fence"
    );
    assert!(
        runtime.reset().is_err(),
        "reset cannot restore a fenced transport"
    );
}

fn verify_cold_component_preparation(
    runtime: &mut ModelRuntime<MlxBackend<'_>>,
    plan: &eredu_core::capture::AdmittedCapturePlan,
    sampling: eredu_core::generation::ResolvedGenerationConfig,
    stream: &Stream,
) {
    use eredu_core::{TextGenerationBackend as _, capture::*, run_preparation::*};
    let rank = eredu_core::DistributedSession::descriptor(
        <MlxBackend<'_> as eredu_core::DistributedBackend>::distributed_session(runtime.session())
            .unwrap(),
    )
    .rank();
    let cached_positions = runtime
        .session()
        .original_model_source()
        .unwrap()
        .erased()
        .retained_inference_authority()
        .unwrap()
        .admission()
        .map_or(0, |admission| admission.position());
    let discovery = MlxBackend::capture_discovery(runtime).unwrap();
    let plan = plan
        .plan()
        .clone()
        .admit_with_text_origin(
            &discovery.catalog,
            &discovery.support,
            &discovery.support.capture,
            plan.request(),
            CaptureTextOrigin { cached_positions },
        )
        .unwrap();
    let before = runtime
        .session()
        .original_model_source()
        .unwrap()
        .erased()
        .state_snapshot();
    let numeric = runtime
        .session()
        .original_model_source()
        .unwrap()
        .erased()
        .fixed_numeric_state_snapshot()
        .unwrap();
    let initial_usage = runtime.text_preparation_usage().unwrap();
    // Empty token geometry is rejected before any distributed admission. The
    // phase-agreement checks below use valid source-bound input on every rank.
    assert!(matches!(
        eredu_core::TokenIdsInputPlan::new(&[]),
        Err(eredu_core::TokenInputRejection::Empty),
    ));
    let mut expected_attempts = 0;
    // The shared startup driver agrees admission before prompt construction.
    // Every failed phase retains all preceding agreement attempts.
    for (stage, attempts) in [
        (TextPreparationStage::Prompt, 2),
        (TextPreparationStage::Sampling, 3),
        (TextPreparationStage::Instrumentation, 4),
    ] {
        let ids = component_capture_prompt_tokens();
        let mut expected_native = None;
        let (_prompt_failure, _sampling_failure) = if rank == 0
            && matches!(
                stage,
                TextPreparationStage::Prompt | TextPreparationStage::Sampling
            ) {
            let original = Array::from_slice(&[1.0_f32, 2.0], &[2])
                .reshape(&[3], stream)
                .unwrap_err();
            expected_native = Some((original.what().to_owned(), original.location()));
            if stage == TextPreparationStage::Prompt {
                (
                    Some(MlxBackend::fail_next_prompt_for_test(original.into())),
                    None,
                )
            } else {
                (
                    None,
                    Some(MlxBackend::fail_next_sampling_for_test(original.into())),
                )
            }
        } else {
            (None, None)
        };
        let _instrumentation_failure =
            (rank == 0 && stage == TextPreparationStage::Instrumentation).then(|| {
                MlxBackend::fail_next_instrumentation_for_test(CaptureError::Limit {
                    budget: CaptureBudget::Retention,
                    cumulative: false,
                })
            });
        let options = (stage == TextPreparationStage::Instrumentation).then(|| {
            eredu_core::TextPreparationOptions {
                capture: Some(SharedCapturePlan::new(plan.clone())),
                interventions: None,
            }
        });
        let error: Box<dyn std::error::Error> = {
            match component_generation_start_with_tokens(
                runtime,
                options,
                TextGenerationConfig::new(sampling),
                &ids,
                false,
            ) {
                Err(error) => Box::new(error),
                Ok(_) => panic!("rejected {stage:?} entered generation"),
            }
        };
        if rank != 0 {
            let mut source = error.as_ref();
            loop {
                if let Some(rejected) = source.downcast_ref::<TextPreparationRejected>() {
                    assert_eq!((rejected.stage, rejected.rank), (stage, 0));
                    break;
                }
                source = source.source().unwrap_or_else(|| {
                    panic!("peer {stage:?} rejection retains typed cause: {error:?}")
                });
            }
        } else {
            match stage {
                TextPreparationStage::Instrumentation => {
                    let mut source = error.as_ref();
                    loop {
                        if let Some(CaptureError::Limit {
                            budget: CaptureBudget::Retention,
                            cumulative: false,
                        }) = source.downcast_ref::<CaptureError>()
                        {
                            break;
                        }
                        source = source.source().unwrap_or_else(|| {
                            panic!("instrumentation rejection retains retention cause: {error:?}")
                        });
                    }
                }
                TextPreparationStage::Prompt | TextPreparationStage::Sampling => {
                    let mut source = error.as_ref();
                    loop {
                        if let Some(crate::backend::error::Error::Exception(native)) =
                            source.downcast_ref::<crate::backend::error::Error>()
                        {
                            let expected = expected_native.as_ref().unwrap();
                            assert_eq!(native.what(), expected.0);
                            assert_eq!(native.location(), expected.1);
                            break;
                        }
                        source = source.source().unwrap_or_else(|| {
                            panic!("original native {stage:?} cause: {error:?}")
                        });
                    }
                }
                _ => unreachable!(),
            }
        }
        runtime.synchronize().unwrap();
        assert_eq!(
            runtime
                .session()
                .original_model_source()
                .unwrap()
                .erased()
                .state_snapshot(),
            before
        );
        assert_eq!(
            runtime
                .session()
                .original_model_source()
                .unwrap()
                .erased()
                .fixed_numeric_state_snapshot()
                .unwrap(),
            numeric
        );
        expected_attempts += attempts;
        assert_eq!(
            runtime.text_preparation_usage().unwrap().attempts - initial_usage.attempts,
            expected_attempts
        );
    }
    let usage = runtime.text_preparation_usage().unwrap();
    assert_eq!(usage.attempts - initial_usage.attempts, expected_attempts);
    assert!(usage.retained_bytes > initial_usage.retained_bytes);
    assert!(usage.host_bytes > initial_usage.host_bytes);
}

#[derive(Debug, Clone, Default, PartialEq)]
struct ComponentCaptureController {
    tokens: [u32; 32],
    committed: usize,
}
impl eredu_core::TokenFilterController for ComponentCaptureController {
    type Error = std::convert::Infallible;
    fn inference_workspace_is_run_owned(&self) -> bool {
        true
    }
    fn inference_workspace(&self, _: u64) -> Option<eredu_core::TextControllerWorkspace<'_>> {
        Some(eredu_core::TextControllerWorkspace {
            filter: eredu_core::TextFilterWorkspace::Exact(&TokenFilter::All),
            additional_host_bytes: 0,
        })
    }
    fn current_filter(&mut self) -> Result<TokenFilter, Self::Error> {
        Ok(TokenFilter::All)
    }
    fn commit_token(&mut self, token: u32) -> Result<(), Self::Error> {
        self.tokens[self.committed] = token;
        self.committed += 1;
        Ok(())
    }
    fn is_complete(&mut self) -> Result<bool, Self::Error> {
        Ok(false)
    }
}
impl eredu_runtime::execution_control::SnapshotTokenController for ComponentCaptureController {
    fn snapshot_storage_bytes(&self) -> Option<u64> {
        Some(std::mem::size_of::<Self>() as u64)
    }
    fn fork_snapshot(&self) -> Result<Self, String> {
        Ok(self.clone())
    }
    fn original_snapshot_storage_bytes(&self) -> Option<u64> {
        self.snapshot_storage_bytes()
    }
    fn fork_original_snapshot(&self) -> Option<Self> {
        Some(self.clone())
    }
}
type ComponentState<'run, 'model> =
    eredu_core::ControlledTextGeneration<'run, MlxBackend<'model>, ComponentCaptureController>;
type ComponentSnapshot<'model> = (
    eredu_runtime::execution_control::TextContinuationSnapshot<
        MlxBackend<'model>,
        ComponentCaptureController,
    >,
    (eredu_core::RetainedGenerationSequence, f32),
);
type ComponentBranch<'model> = (
    eredu_core::TextGenerationBranch<MlxBackend<'model>, ComponentCaptureController>,
    (eredu_core::RetainedGenerationSequence, f32),
);
type ComponentProvider = crate::tests::support::original_snapshot::NativeSnapshotProvider;
use eredu_evaluation::execution_control::ContinuationSnapshotProvider as _;
fn component_capture_start<'run, 'model>(
    runtime: &'run mut ModelRuntime<MlxBackend<'model>>,
    plan: &eredu_core::capture::AdmittedCapturePlan,
    sampling: eredu_core::ResolvedGenerationConfig,
) -> (ComponentState<'run, 'model>, ComponentProvider) {
    runtime.reset().unwrap();
    component_state_start(runtime, Some(plan), sampling)
}
fn component_snapshot_limits() -> eredu_core::execution_control::SnapshotLimits {
    eredu_core::execution_control::SnapshotLimits {
        max_snapshots: 8,
        max_branches: 8,
        retained_bytes: 512 << 20,
        cumulative_copy_bytes: 4 << 30,
    }
}
fn component_snapshot_budget() -> eredu_runtime::execution_control::SnapshotBudget {
    eredu_runtime::execution_control::SnapshotBudget::new(component_snapshot_limits())
}
fn component_snapshot_sampling() -> eredu_core::ResolvedGenerationConfig {
    eredu_core::resolve_generation_config(
        None,
        eredu_core::GenerationConfigOverrides {
            max_new_tokens: Some(20),
            temperature: Some(0.0),
            ..Default::default()
        },
    )
    .unwrap()
}
// Retain the real installed prefix, pending source, sampler and host sequence;
// source preparation and copying do not execute or reset the model frontier.
#[track_caller]
fn component_compatibility_snapshot<'model>(
    runtime: &mut ModelRuntime<MlxBackend<'model>>,
) -> ComponentSnapshot<'model> {
    let (mut state, mut provider) =
        component_state_start(runtime, None, component_snapshot_sampling());
    component_capture_snapshot(&mut state, &mut provider, &component_snapshot_budget())
}
fn component_snapshot_compatible<'model>(
    runtime: &ModelRuntime<MlxBackend<'model>>,
    saved: &ComponentSnapshot<'model>,
) -> bool {
    saved
        .0
        .native_continuation_growth(runtime, saved.0.next_prediction())
        .is_ok()
}
fn component_exchange_same_frontier(runtime: &mut ModelRuntime<MlxBackend<'_>>) {
    let (mut state, mut provider) =
        component_state_start(runtime, None, component_snapshot_sampling());
    let saved = component_capture_snapshot(&mut state, &mut provider, &component_snapshot_budget());
    let mut child = component_capture_fork(
        &mut state,
        &mut provider,
        &saved,
        &eredu_core::OriginalTextResumeOptions::new(eredu_core::OriginalTextResumeKind::Branch),
    );
    component_capture_exchange(&mut state, &mut provider, &mut child);
}
#[track_caller]
fn component_state_start<'run, 'model>(
    runtime: &'run mut ModelRuntime<MlxBackend<'model>>,
    plan: Option<&eredu_core::capture::AdmittedCapturePlan>,
    sampling: eredu_core::ResolvedGenerationConfig,
) -> (ComponentState<'run, 'model>, ComponentProvider) {
    component_state_start_with_tokens(
        runtime,
        plan,
        sampling,
        &component_capture_prompt_tokens(),
        std::env::var_os(COMPONENT_CAPTURE_MEDIA).is_some(),
    )
}
#[track_caller]
fn component_state_start_with_tokens<'run, 'model>(
    runtime: &'run mut ModelRuntime<MlxBackend<'model>>,
    plan: Option<&eredu_core::capture::AdmittedCapturePlan>,
    sampling: eredu_core::ResolvedGenerationConfig,
    tokens: &[u32],
    media: bool,
) -> (ComponentState<'run, 'model>, ComponentProvider) {
    let pool = runtime.backend().memory_ledger().clone();
    let config = TextGenerationConfig::new(sampling);
    let options = plan.map(|plan| eredu_core::TextPreparationOptions {
        capture: Some(eredu_core::capture::SharedCapturePlan::new(plan.clone())),
        interventions: None,
    });
    let mut state =
        component_generation_start_with_tokens(runtime, options, config.clone(), tokens, media)
            .unwrap();
    let sequence = state
        .take_prepared_sequence()
        .unwrap()
        .prepare_storage()
        .unwrap();
    (state, ComponentProvider::new(sequence, config, pool))
}

type ComponentPreparationError = eredu_core::ControlledTextGenerationError<
    crate::backend::error::Error,
    std::convert::Infallible,
>;

fn component_generation_start_with_tokens<'run, 'model>(
    runtime: &'run mut ModelRuntime<MlxBackend<'model>>,
    options: Option<eredu_core::TextPreparationOptions>,
    config: TextGenerationConfig,
    tokens: &[u32],
    media: bool,
) -> Result<ComponentState<'run, 'model>, ComponentPreparationError> {
    runtime.synchronize().unwrap();
    let pool = runtime.backend().memory_ledger();
    crate::backend::submission_recovery::wait_for_retirement(|| {
        safemlx::memory::clear_cache();
        crate::backend::nn::shared::MlxNeuralBackend::reclaim_retired_resources();
        safemlx::reclaim_allocation_owners();
        pool.unquoted_owner_count().unwrap() == 0
    });
    let max = config
        .sampling()
        .max_new_tokens
        .expect("bounded component fixture");
    let consumer = eredu_core::GenerationSequenceConsumerLayout::for_driver_types::<
        ComponentProvider,
        crate::backend::error::Error,
        crate::backend::error::Error,
    >()
    .unwrap();
    let sequence = eredu_core::GenerationSequenceRequest::new(max, &[]).with_consumer(&consumer);
    if media {
        let prompt = component_capture_prompt(runtime);
        let (prompt, _custody, _) = crate::tests::support::original_input::prepare(runtime, prompt);
        eredu_core::ControlledTextGeneration::from_input_with_sequence(
            runtime,
            eredu_core::TextGenerationInput::OriginalPrepared(prompt),
            config,
            ComponentCaptureController::default(),
            options,
            sequence,
        )
    } else {
        eredu_core::ControlledTextGeneration::from_token_ids_with_sequence(
            runtime,
            eredu_core::TokenIdsInputPlan::new(tokens).unwrap(),
            config,
            ComponentCaptureController::default(),
            options,
            sequence,
        )
    }
}

#[track_caller]
fn component_capture_step<'model>(
    state: &mut ComponentState<'_, 'model>,
    provider: &mut ComponentProvider,
) -> (u32, eredu_core::capture::SharedCapturedStep) {
    let token = state.next().unwrap().unwrap().token_id();
    let capture = state.take_captured_delivery().unwrap().unwrap();
    provider.observe_token(token);
    (token, capture)
}
fn component_capture_snapshot<'model>(
    state: &mut ComponentState<'_, 'model>,
    provider: &mut ComponentProvider,
    budget: &eredu_runtime::execution_control::SnapshotBudget,
) -> ComponentSnapshot<'model> {
    provider
        .capture(&mut state.snapshot_source().unwrap(), budget, Some(4096))
        .unwrap()
}
fn component_capture_fork<'model>(
    state: &mut ComponentState<'_, 'model>,
    provider: &mut ComponentProvider,
    saved: &ComponentSnapshot<'model>,
    options: &eredu_core::OriginalTextResumeOptions<'_>,
) -> ComponentBranch<'model> {
    state
        .fork_completed(
            |runtime| provider.resume(runtime, &saved.0, &saved.1, options),
            |error| panic!("component fork boundary: {error:?}"),
        )
        .unwrap()
        .unwrap()
}
fn component_capture_restore<'model>(
    state: &mut ComponentState<'_, 'model>,
    provider: &mut ComponentProvider,
    saved: &ComponentSnapshot<'model>,
) {
    let mut host = state
        .replace_completed(
            |runtime| {
                provider
                    .resume(
                        runtime,
                        &saved.0,
                        &saved.1,
                        &eredu_core::OriginalTextResumeOptions::new(
                            eredu_core::OriginalTextResumeKind::Restore,
                        ),
                    )
                    .map(|result| {
                        result.map(|(state, displaced, host)| {
                            drop(displaced);
                            (state, host)
                        })
                    })
            },
            |error| panic!("component restore boundary: {error:?}"),
        )
        .unwrap()
        .unwrap();
    provider.exchange_host(&mut host);
}
fn component_capture_exchange<'model>(
    state: &mut ComponentState<'_, 'model>,
    provider: &mut ComponentProvider,
    branch: &mut ComponentBranch<'model>,
) {
    state.exchange_branch(&mut branch.0).unwrap();
    provider.exchange_host(&mut branch.1);
}

fn same_component_values(
    actual: &(u32, eredu_core::capture::SharedCapturedStep),
    expected: &(u32, eredu_core::capture::SharedCapturedStep),
) {
    assert_eq!(actual.0, expected.0);
    assert_eq!(actual.1.prediction_index, expected.1.prediction_index);
    assert_eq!(actual.1.records.len(), expected.1.records.len());
    for (actual, expected) in actual.1.records.iter().zip(&expected.1.records) {
        assert_eq!(actual.path, expected.path);
        assert_eq!(actual.outcome, expected.outcome);
        assert_eq!(actual.payload, expected.payload);
    }
}

fn verify_component_capture_branches(
    runtime: &mut ModelRuntime<MlxBackend<'_>>,
    plan: &eredu_core::capture::AdmittedCapturePlan,
    sampling: eredu_core::ResolvedGenerationConfig,
) {
    use eredu_core::{
        OriginalTextResumeKind, OriginalTextResumeOptions, execution_control::SnapshotLimits,
    };
    use eredu_runtime::execution_control::SnapshotBudget;
    let (mut state, mut provider) = component_capture_start(runtime, plan, sampling);
    let prefix = component_capture_step(&mut state, &mut provider);
    let budget = SnapshotBudget::new(SnapshotLimits {
        max_snapshots: 2,
        max_branches: 3,
        retained_bytes: 64 << 20,
        cumulative_copy_bytes: 512 << 20,
    });
    let saved = component_capture_snapshot(&mut state, &mut provider, &budget);
    let request = |session_id| {
        let mut options = OriginalTextResumeOptions::new(OriginalTextResumeKind::Branch);
        options.session_id = Some(session_id);
        options.capture_limits = Some(&plan.plan().limits);
        options
    };
    let mut left = component_capture_fork(
        &mut state,
        &mut provider,
        &saved,
        &request("component-left"),
    );
    let mut right = component_capture_fork(
        &mut state,
        &mut provider,
        &saved,
        &request("component-right"),
    );
    let mut skipped_limits = plan.plan().limits.clone();
    skipped_limits.on_limit = eredu_core::capture::CaptureLimitPolicy::Skip;
    skipped_limits.per_step.captures = 0;
    let mut skip_request = request("component-skipped");
    skip_request.capture_limits = Some(&skipped_limits);
    let mut skipped_child =
        component_capture_fork(&mut state, &mut provider, &saved, &skip_request);
    let baseline = component_capture_step(&mut state, &mut provider);
    assert_eq!(
        baseline.1.partitions[0].context.run_identity,
        prefix.1.partitions[0].context.run_identity
    );
    component_capture_exchange(&mut state, &mut provider, &mut left);
    let first_left = component_capture_step(&mut state, &mut provider);
    same_component_values(&first_left, &baseline);
    assert_ne!(
        first_left.1.partitions[0].context.run_identity,
        baseline.1.partitions[0].context.run_identity
    );
    component_capture_exchange(&mut state, &mut provider, &mut left);
    component_capture_exchange(&mut state, &mut provider, &mut right);
    let first_right = component_capture_step(&mut state, &mut provider);
    same_component_values(&first_right, &baseline);
    assert_ne!(
        first_right.1.partitions[0].context.run_identity,
        first_left.1.partitions[0].context.run_identity
    );
    component_capture_exchange(&mut state, &mut provider, &mut right);
    component_capture_exchange(&mut state, &mut provider, &mut skipped_child);
    let skipped = component_capture_step(&mut state, &mut provider);
    assert_eq!(skipped.0, baseline.0);
    assert!(skipped.1.partitions.is_empty());
    assert_eq!(skipped.1.records.len(), baseline.1.records.len());
    for (record, baseline) in skipped.1.records.iter().zip(&baseline.1.records) {
        use eredu_core::capture::{CaptureBudget, CaptureOutcome, CaptureSkipReason};
        assert_eq!(record.path, baseline.path);
        assert!(record.payload.is_none());
        let reason = if matches!(
            baseline.outcome,
            CaptureOutcome::Skipped {
                reason: CaptureSkipReason::Schedule,
            }
        ) {
            CaptureSkipReason::Schedule
        } else {
            assert_eq!(baseline.outcome, CaptureOutcome::Captured);
            CaptureSkipReason::Limit {
                budget: CaptureBudget::Captures,
                cumulative: false,
            }
        };
        assert_eq!(record.outcome, CaptureOutcome::Skipped { reason });
    }
    component_capture_exchange(&mut state, &mut provider, &mut skipped_child);
    let copy_before = budget.usage().cumulative_copy_bytes;
    component_capture_restore(&mut state, &mut provider, &saved);
    let replay = component_capture_step(&mut state, &mut provider);
    same_component_values(&replay, &baseline);
    // Restoration keeps the capture run while consuming a later forward epoch.
    assert_eq!(
        replay.1.partitions[0].context.run_identity,
        baseline.1.partitions[0].context.run_identity
    );
    assert!(
        replay.1.partitions[0].context.forward_epoch
            > baseline.1.partitions[0].context.forward_epoch
    );
    assert!(replay.1.cumulative_usage.encoded_bytes > baseline.1.cumulative_usage.encoded_bytes);
    assert!(budget.usage().cumulative_copy_bytes > copy_before);
}

fn verify_loaded_component_interventions(
    runtime: &mut ModelRuntime<MlxBackend<'_>>,
    reference: &mut ModelRuntime<MlxBackend<'_>>,
    capture: &eredu_core::capture::AdmittedCapturePlan,
    sampling: eredu_core::ResolvedGenerationConfig,
    required_points: &[&str],
) {
    use eredu_core::{
        ObservationSupportStatus, TextGenerationBackend as _, capture::*, intervention::*,
    };
    let discovery = MlxBackend::intervention_discovery(runtime).unwrap();
    let targets = discovery
        .points
        .iter()
        .filter(|point| {
            point.routing.is_none()
                && point.routed_units.is_none()
                && point
                    .axes
                    .last()
                    .is_some_and(|axis| axis.name == "component")
                && point.prefill == ObservationSupportStatus::Supported
                && point.decode == ObservationSupportStatus::Supported
        })
        .enumerate()
        .filter(|(index, point)| *index < 2 || required_points.contains(&point.path.as_str()))
        .map(|(_, point)| point)
        .collect::<Vec<_>>();
    assert!(
        !targets.is_empty(),
        "loaded component intervention capability"
    );
    let mut observations = capture.plan().clone();
    observations
        .selections
        .retain(|selection| selection.path == eredu_core::MODEL_LOGITS_OBSERVATION_PATH);
    let capture_discovery = MlxBackend::capture_discovery(runtime).unwrap();
    for path in required_points {
        let point = capture_discovery.catalog.get(path).unwrap();
        if point.position == eredu_core::ObservationPosition::BeforeIntervention
            && point
                .axes
                .as_ref()
                .and_then(|axes| axes.last())
                .is_some_and(|axis| axis.name == "component")
        {
            assert!(
                targets.iter().any(|target| target.path == *path),
                "required scalar capture must also participate in native mask trials: {path}"
            );
        }
    }
    let capture = observations
        .admit(
            &capture_discovery.catalog,
            &capture_discovery.support,
            &capture_discovery.support.capture,
            capture.request(),
        )
        .unwrap();
    for keep in [false, true] {
        let mut operations = Vec::new();
        for (index, point) in targets.iter().enumerate() {
            let eredu_core::SymbolicDimension::Known(width) = point.axes.last().unwrap().dimension
            else {
                panic!("known components")
            };
            for (phase, position) in [(CapturePhase::Prefill, 1), (CapturePhase::Decode, 0)] {
                operations.push(InterventionOperation {
                    id: format!("component-{index}-{phase:?}"),
                    target: point.path.clone(),
                    schedule: CaptureSchedule {
                        prefill: phase == CapturePhase::Prefill,
                        decode: phase == CapturePhase::Decode,
                        ..Default::default()
                    },
                    slices: vec![CaptureSlice {
                        axis: "sequence".into(),
                        start: position,
                        end: position + 1,
                        stride: 1,
                    }],
                    action: InterventionAction::MaskComponents {
                        dtype: InterventionDtype::Float32,
                        indices: vec![0, (width / 2) as u32, (width - 1) as u32],
                        keep_selected: keep,
                    },
                    evidence: InterventionEvidence::Preview { max_elements: 128 },
                });
            }
            operations.push(InterventionOperation {
                id: format!("scale-{index}"),
                target: point.path.clone(),
                schedule: CaptureSchedule::default(),
                slices: vec![],
                action: InterventionAction::Scale {
                    dtype: InterventionDtype::Float32,
                    factor: 0.7,
                },
                evidence: InterventionEvidence::Preview { max_elements: 128 },
            });
        }
        if std::env::var_os(COMPONENT_CAPTURE_MEDIA).is_some() {
            for (index, target) in discovery
                .points
                .iter()
                .filter(|point| {
                    point.path == "readout.embedding"
                        || point.path.ends_with(".deepstack.output")
                        || point.path.ends_with(".per_layer.output")
                })
                .map(|point| point.path.as_str())
                .enumerate()
            {
                operations.push(InterventionOperation {
                    id: format!("media-{index}"),
                    target: target.into(),
                    schedule: CaptureSchedule {
                        prefill: true,
                        decode: false,
                        ..Default::default()
                    },
                    slices: vec![CaptureSlice {
                        axis: "sequence".into(),
                        start: 2,
                        end: 4,
                        stride: 1,
                    }],
                    action: if keep {
                        InterventionAction::Scale {
                            dtype: InterventionDtype::Float32,
                            factor: -0.5,
                        }
                    } else {
                        InterventionAction::Zero {
                            dtype: InterventionDtype::Float32,
                        }
                    },
                    evidence: InterventionEvidence::Preview { max_elements: 128 },
                });
            }
        }
        let plan = InterventionPlan {
            schema_version: INTERVENTION_SCHEMA_VERSION,
            operations,
        };
        let run = |runtime: &mut ModelRuntime<MlxBackend<'_>>| {
            runtime.reset().unwrap();
            let admitted = plan
                .clone()
                .admit(
                    &MlxBackend::intervention_discovery(runtime).unwrap(),
                    capture.request(),
                    "native-component-trial",
                )
                .unwrap();
            let interventions = runtime
                .backend()
                .memory_ledger()
                .compile_intervention_source(
                    eredu_core::intervention::PreparedInterventionPlanCopy::inspect(&admitted)
                        .unwrap(),
                )
                .unwrap()
                .plan()
                .clone();
            let mut generation = component_capture_generation(
                runtime,
                sampling,
                Some(eredu_core::TextPreparationOptions {
                    capture: Some(eredu_core::capture::SharedCapturePlan::new(capture.clone())),
                    interventions: Some(interventions),
                }),
            )
            .unwrap();
            (0..3)
                .map(|_| {
                    let token = generation.next().unwrap().unwrap().token_id();
                    let step = generation.take_captured_delivery().unwrap().unwrap();
                    (token, step)
                })
                .collect::<Vec<_>>()
        };
        if keep {
            verify_component_intervention_branches(runtime, &capture, &plan, sampling);
        }
        let expected = run(reference);
        let actual = run(runtime);
        let close = |actual: &CaptureRecord, expected: &CaptureRecord| {
            assert_eq!(actual.path, expected.path);
            assert_eq!(actual.position, expected.position);
            assert_eq!(actual.outcome, expected.outcome);
            match (&actual.payload, &expected.payload) {
                (None, None) => (),
                (Some(actual), Some(expected)) => {
                    let actual = actual.as_tensor().expect("actual tensor evidence");
                    let expected = expected.as_tensor().expect("expected tensor evidence");
                    assert_eq!(actual.shape(), expected.shape());
                    let (
                        eredu_core::TensorObservationData::F32(actual),
                        eredu_core::TensorObservationData::F32(expected),
                    ) = (actual.data(), expected.data())
                    else {
                        panic!("float fixture")
                    };
                    for (actual, expected) in actual.iter().zip(expected) {
                        assert!(
                            (actual - expected).abs() <= 3e-4 + 3e-4 * expected.abs(),
                            "global intervention {actual} != {expected}"
                        );
                    }
                }
                _ => panic!("matching tensor evidence"),
            }
        };
        for (actual, expected) in actual.iter().zip(&expected) {
            assert_eq!(actual.0, expected.0);
            assert_eq!(actual.1.prediction_index, expected.1.prediction_index);
            for (actual, expected) in actual.1.records.iter().zip(&expected.1.records) {
                close(actual, expected);
            }
            for (actual, expected) in actual.1.interventions.iter().zip(&expected.1.interventions) {
                assert_eq!(actual.outcome, expected.outcome);
                assert_eq!(actual.operation_id, expected.operation_id);
                for (actual, expected) in actual.evidence.iter().zip(&expected.evidence) {
                    close(actual, expected);
                }
            }
        }
    }
    runtime.synchronize().unwrap();
    reference.synchronize().unwrap();
}

fn verify_component_intervention_branches(
    runtime: &mut ModelRuntime<MlxBackend<'_>>,
    capture: &eredu_core::capture::AdmittedCapturePlan,
    plan: &eredu_core::intervention::InterventionPlan,
    sampling: eredu_core::ResolvedGenerationConfig,
) {
    use eredu_core::{
        OriginalTextResumeKind, OriginalTextResumeOptions, execution_control::SnapshotLimits,
        intervention::*,
    };
    use eredu_runtime::execution_control::SnapshotBudget;
    let (mut state, mut provider) = component_capture_start(runtime, capture, sampling);
    let _prefix = component_capture_step(&mut state, &mut provider);
    let budget = SnapshotBudget::new(SnapshotLimits {
        max_snapshots: 3,
        max_branches: 2,
        retained_bytes: 128 << 20,
        cumulative_copy_bytes: 1 << 30,
    });
    let saved = component_capture_snapshot(&mut state, &mut provider, &budget);
    let mut future = plan.clone();
    future
        .operations
        .retain(|operation| operation.schedule.decode);
    for operation in &mut future.operations {
        operation.schedule.prefill = false;
        operation.schedule.first_prediction = 1;
    }
    let mut alternate = future.clone();
    for operation in &mut alternate.operations {
        if let InterventionAction::MaskComponents { keep_selected, .. } = &mut operation.action {
            *keep_selected = !*keep_selected;
        }
    }
    let request = |session_id, intervention| {
        let mut options = OriginalTextResumeOptions::new(OriginalTextResumeKind::Branch);
        options.session_id = Some(session_id);
        options.capture_limits = Some(&capture.plan().limits);
        options.intervention = Some(intervention);
        options
    };
    let mut left = component_capture_fork(
        &mut state,
        &mut provider,
        &saved,
        &request("intervention-left", &future),
    );
    let mut right = component_capture_fork(
        &mut state,
        &mut provider,
        &saved,
        &request("intervention-right", &alternate),
    );
    let baseline = component_capture_step(&mut state, &mut provider);
    component_capture_exchange(&mut state, &mut provider, &mut left);
    let left_saved = component_capture_snapshot(&mut state, &mut provider, &budget);
    let first_left = component_capture_step(&mut state, &mut provider);
    assert!(
        first_left
            .1
            .interventions
            .iter()
            .all(|record| record.outcome == InterventionOutcome::Applied)
    );
    component_capture_restore(&mut state, &mut provider, &left_saved);
    let replay_left = component_capture_step(&mut state, &mut provider);
    same_component_values(&replay_left, &first_left);
    for (actual, expected) in replay_left
        .1
        .interventions
        .iter()
        .zip(&first_left.1.interventions)
    {
        assert_eq!(actual.outcome, expected.outcome);
        for (actual, expected) in actual.evidence.iter().zip(&expected.evidence) {
            assert_eq!(actual.payload, expected.payload);
        }
    }
    assert!(replay_left.1.cumulative_usage.host_bytes > first_left.1.cumulative_usage.host_bytes);
    component_capture_exchange(&mut state, &mut provider, &mut left);
    component_capture_exchange(&mut state, &mut provider, &mut right);
    let first_right = component_capture_step(&mut state, &mut provider);
    assert!(
        first_right
            .1
            .interventions
            .iter()
            .all(|record| record.outcome == InterventionOutcome::Applied)
    );
    assert_ne!(
        first_right.1.records[0].payload, first_left.1.records[0].payload,
        "different masks recompute the prediction"
    );
    assert_ne!(
        first_right.1.partitions[0].context.run_identity,
        first_left.1.partitions[0].context.run_identity
    );
    component_capture_exchange(&mut state, &mut provider, &mut right);
    component_capture_restore(&mut state, &mut provider, &saved);
    let replay = component_capture_step(&mut state, &mut provider);
    same_component_values(&replay, &baseline);
    assert!(
        replay.1.interventions.is_empty(),
        "child changes do not contaminate the parent"
    );
    assert!(replay.1.cumulative_usage.host_bytes > baseline.1.cumulative_usage.host_bytes);
}
