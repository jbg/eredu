// The opaque expert fixture enters through the same admitted text driver as
// applications. File-cache replacement happens only between completed runs;
// the next run prepares its own exact source at the restored native frontier.
const ADMITTED_EXPERT_PREFIX: [u32; 2] = [1, 2];
const ADMITTED_EXPERT_PREDICTIONS: usize = 4;

fn verify_admitted_expert_session(
    runtime: &mut ModelRuntime<MlxBackend<'_>>,
    checkpoint: &Path,
    reference_options: MlxLoadRequest,
    stream: &Stream,
    cache_root: &Path,
    rank: usize,
    layers: usize,
    tolerance: f32,
) {
    use eredu_core::{TextGenerationBackend as _, capture::*};

    fn run(
        runtime: &mut ModelRuntime<MlxBackend<'_>>,
        tokens: &[u32],
        predictions: usize,
    ) -> Vec<(u32, SharedCapturedStep)> {
        let discovery = MlxBackend::capture_discovery(runtime).unwrap();
        let cached_positions = runtime
            .session()
            .original_model_source()
            .unwrap()
            .erased()
            .original_text_frontier()
            .unwrap()
            .expect("expert fixture owns actual cache state");
        let usage = CaptureUsage {
            captures: 8,
            retained_bytes: 8 << 20,
            host_bytes: 256 << 20,
            encoded_bytes: 8 << 20,
        };
        let plan = CapturePlan {
            schema_version: 1,
            selections: vec![CaptureSelection {
                id: "expert-final-logits".into(),
                path: eredu_core::MODEL_LOGITS_OBSERVATION_PATH.into(),
                schedule: CaptureSchedule::default(),
                slices: Vec::new(),
                transform: CaptureTransform::FullTensor,
            }],
            limits: CaptureLimits {
                per_step: usage,
                cumulative: usage.checked_mul(predictions as u64).unwrap(),
                on_limit: CaptureLimitPolicy::Fail,
            },
        }
        .admit_with_text_origin(
            &discovery.catalog,
            &discovery.support,
            &discovery.support.capture,
            CaptureRequestShape {
                batch: 1,
                prompt_tokens: tokens.len() as u64,
                max_predictions: predictions as u64,
            },
            CaptureTextOrigin { cached_positions },
        )
        .unwrap();
        let sampling = eredu_core::resolve_generation_config(
            None,
            eredu_core::GenerationConfigOverrides {
                temperature: Some(0.0),
                max_new_tokens: Some(predictions),
                ..Default::default()
            },
        )
        .unwrap();
        let (mut state, mut provider) =
            component_state_start_with_tokens(runtime, Some(&plan), sampling, tokens, false);
        (0..predictions)
            .map(|_| component_capture_step(&mut state, &mut provider))
            .collect()
    }

    fn values(step: &SharedCapturedStep) -> &[f32] {
        assert_eq!(step.outcome, CaptureStepOutcome::Committed);
        assert_eq!(step.records.len(), 1);
        let record = &step.records[0];
        assert_eq!(record.outcome, CaptureOutcome::Captured);
        let tensor = record
            .payload
            .as_ref()
            .and_then(CapturePayload::as_tensor)
            .unwrap();
        let eredu_core::TensorObservationData::F32(values) = tensor.data() else {
            panic!("expert fixture must retain the actual F32 logits");
        };
        assert!(!values.is_empty());
        assert!(values.iter().all(|value| value.is_finite()));
        assert!(values.iter().any(|value| *value != 0.0));
        values
    }

    fn close(
        actual: &(u32, SharedCapturedStep),
        expected: &(u32, SharedCapturedStep),
        tolerance: f32,
    ) {
        assert_eq!(actual.0, expected.0);
        let actual = values(&actual.1);
        let expected = values(&expected.1);
        // The first one-token continuation uses the same cached position as
        // decode in the uninterrupted oracle; compare its complete final row.
        assert_eq!(actual.len(), expected.len());
        assert!(
            actual
                .iter()
                .zip(expected)
                .all(|(a, e)| (a - e).abs() <= tolerance),
            "expert logits differ: actual={actual:?}, expected={expected:?}"
        );
    }

    let backend = prepared_component_backend(stream);
    let prepared = eredu_core::prepare_inspected_model(
        &backend,
        component_fixture_inspection(checkpoint),
        reference_options,
    )
    .unwrap();
    let mut reference = ModelRuntime::from_prepared(backend, prepared).unwrap();
    let prefix_tokens = ADMITTED_EXPERT_PREFIX;
    let expected = run(&mut reference, &prefix_tokens, ADMITTED_EXPERT_PREDICTIONS);
    reference.synchronize().unwrap();

    let before = crate::tests::support::path_instrumentation::snapshot().forwards;
    let exchanges_before =
        crate::tests::support::path_instrumentation::variable_all_to_all_submissions();
    let prefix = run(runtime, &prefix_tokens, 1).pop().unwrap();
    close(&prefix, &expected[0], tolerance);
    runtime.synchronize().unwrap();
    let descriptor = PromptCacheDescriptor::from_model_identity(
        runtime.session().prompt_cache_model_identity().unwrap(),
        "opaque-ring-expert-fixture",
        format!("tokens:{prefix_tokens:?}"),
        1,
    )
    .unwrap();
    let root = cache_root.join(format!("rank-{rank}"));
    let saved = {
        let (backend, session) = runtime.parts_mut();
        session
            .save_prompt_cache(
                backend,
                &root,
                descriptor.clone(),
                &prefix_tokens,
                &PromptCacheOptions::default(),
            )
            .unwrap()
    };
    assert_eq!(saved.total_prefix_tokens, prefix_tokens.len());
    assert!(!saved.blocks.is_empty());
    assert!(saved.blocks.iter().all(|block| block.logical_bytes > 0));
    let baseline = run(runtime, &[prefix.0], 1).pop().unwrap();
    close(&baseline, &expected[1], tolerance);
    runtime.synchronize().unwrap();
    let restored = {
        let (backend, session) = runtime.parts_mut();
        session
            .load_prompt_cache(backend, &root, &descriptor, &prefix_tokens)
            .unwrap()
    };
    assert_eq!(restored.total_prefix_tokens, saved.total_prefix_tokens);
    assert_eq!(restored.blocks, saved.blocks);
    let continued = run(runtime, &[prefix.0], ADMITTED_EXPERT_PREDICTIONS - 1);
    assert_eq!(values(&continued[0].1), values(&baseline.1));
    assert_eq!(continued[0].0, baseline.0);
    for (actual, expected) in continued.iter().zip(&expected[1..]) {
        close(actual, expected, tolerance);
    }
    assert!(
        continued[1..]
            .iter()
            .all(|(_, step)| step.phase == CapturePhase::Decode)
    );
    runtime.synchronize().unwrap();
    assert_eq!(
        crate::tests::support::path_instrumentation::snapshot().forwards - before,
        5,
        "admitted prefill, retained/restored one-token continuation and two cached decodes",
    );
    assert_eq!(
        crate::tests::support::path_instrumentation::variable_all_to_all_submissions()
            - exchanges_before,
        5 * layers * 8,
        "every expert layer must execute its exact eight owner/count exchanges per forward",
    );
}
