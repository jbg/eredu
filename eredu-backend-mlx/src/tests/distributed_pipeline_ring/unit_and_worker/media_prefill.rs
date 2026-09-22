mod retained_media_prefill_tests {
    include!("media_prefill/prepared_capture.rs");
    use super::*;
    use crate::backend::error::Error;
    use crate::tests::support::media_completion::{self, Sample};
    use eredu_core::Completion;

    fn weights(mode: usize) -> MlxLoadRequest {
        let residency = match mode {
            0 => WeightResidency::fully_resident(),
            1 => WeightResidency::layerwise_host(LayerwiseLoadOptions::new(
                OffloadConfig::new(Some(1 << 26), Some(1 << 26), 1).unwrap(),
            )),
            2 => WeightResidency::dense_disk_stream(
                DenseDiskStreamLoadOptions::new(1 << 26, 1 << 26, 1, 1).unwrap(),
            ),
            _ => unreachable!(),
        };
        MlxLoadRequest::from_normalized(
            eredu_runtime::NormalizedLoadRequest::default().with_weight_residency(residency),
        )
    }
    fn input(chunk: Option<u64>, invalid: bool) -> MlxModelInput {
        let tokens = Array::from_slice(&[if invalid { 64_u32 } else { 1 }, 2], &[1, 2]);
        let pixels = Array::from_slice(
            &(0..16 * 12)
                .map(|i| (i as f32 - 73.0) / 193.0)
                .collect::<Vec<_>>(),
            &[16, 12],
        );
        let grid = Array::from_slice(&[1_i32, 4, 4], &[1, 3]);
        let tail = Array::from_slice(&[3_u32], &[1, 1]);
        let parts = [
            text_input_part(&tokens),
            input_part(
                InputModality::Image,
                InputPayload::Tensor(pixels),
                [(InputMetadataKey::PatchGrid, grid)],
                [],
            ),
            text_input_part(&tail),
        ];
        let input = synthetic_prediction_input(&parts, &[1, 2, 42, 42, 42, 42, 3]);
        match chunk {
            Some(chunk) => input.with_prefill_chunk_positions(chunk.try_into().unwrap()),
            None => input,
        }
    }
    fn values(array: &Array) -> Vec<f32> {
        array.evaluated().unwrap().as_slice::<f32>().to_vec()
    }
    fn close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (i, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= 3e-4 + 3e-4 * expected.abs(),
                "value {i}: {actual} != {expected}"
            );
        }
    }
    type Arrays = Vec<(Vec<i32>, Vec<f32>)>;
    type Fixed = Vec<(
        usize,
        eredu_core::cache::StateTensorRole,
        Vec<i32>,
        Vec<f32>,
    )>;
    fn same_arrays(actual: &Arrays, expected: &Arrays) {
        assert_eq!(actual.len(), expected.len());
        for ((shape, values), (expected_shape, expected_values)) in actual.iter().zip(expected) {
            assert_eq!(shape, expected_shape);
            close(values, expected_values);
        }
    }
    fn fixed_equal(actual: &Fixed, expected: &Fixed) {
        assert_eq!(actual.len(), expected.len());
        for ((layer, role, shape, data), (el, er, es, ed)) in actual.iter().zip(expected) {
            assert_eq!((layer, role, shape), (el, er, es));
            close(data, ed);
        }
    }
    #[derive(Default)]
    struct Observer {
        chunks: Vec<eredu_runtime::prefill::PrefillChunk>,
        outputs: Vec<Vec<i32>>,
    }
    impl eredu_runtime::ActivationObserver<MlxTensor, Error> for Observer {
        fn requires_sequence_readout(&self) -> bool {
            false
        }
        fn begin_prefill_chunk(
            &mut self,
            chunk: &eredu_runtime::prefill::PrefillChunk,
        ) -> Result<(), Error> {
            self.chunks.push(chunk.clone());
            Ok(())
        }
        fn observe(&mut self, path: &str, value: &MlxTensor) -> Result<(), Error> {
            if path == eredu_core::MODEL_LOGITS_OBSERVATION_PATH {
                self.outputs.push(value.as_array().shape().to_vec());
            }
            Ok(())
        }
    }
    struct Report {
        logits: Vec<f32>,
        cached: Vec<Vec<f32>>,
        state: Arrays,
        final_state: Arrays,
        fixed: Fixed,
        final_fixed: Fixed,
        observer: Observer,
        roots: Vec<Sample>,
    }
    fn run(path: &Path, mode: usize, chunk: Option<u64>) -> Report {
        run_prompt(path, mode, input(chunk, false))
    }
    fn run_prompt(path: &Path, mode: usize, prompt: MlxModelInput) -> Report {
        let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let backend = crate::native::backend(&stream, &stream);
        let model = load_model(&backend, path, weights(mode))
            .unwrap()
            .into_inner();
        let mut session = MlxModelSession::from_model(
            model,
            eredu_core::SessionCapabilities::new(true, true, true),
        )
        .unwrap();
        let mut observer = Observer::default();
        let (submission, roots) = media_completion::observe(None, || {
            session
                .submit_prefill_with_observer(&backend, prompt, &mut observer)
                .unwrap()
        });
        submission.completion.wait().unwrap();
        assert_eq!(submission.output.shape(), &[1, 64]);
        let logits = values(&submission.output);
        assert!(logits.iter().any(|value| value.abs() > 1e-5));
        drop(submission);
        let target = session.neutral_prediction_target_mut().unwrap();
        let state = target
            .retained_numeric_state_snapshot()
            .expect("actual composite retained arrays")
            .unwrap();
        let fixed = target.fixed_numeric_state_snapshot().unwrap();
        assert!(!state.is_empty());
        let mut cached = Vec::new();
        for token in [4, 5, 6] {
            let submission = session.submit_token_decode(&backend, token).unwrap();
            submission.completion.wait().unwrap();
            cached.push(values(submission.output.logits().unwrap().as_array()));
        }
        let target = session.neutral_prediction_target_mut().unwrap();
        let final_state = target.retained_numeric_state_snapshot().unwrap().unwrap();
        let final_fixed = target.fixed_numeric_state_snapshot().unwrap();
        Report {
            logits,
            cached,
            state,
            final_state,
            fixed,
            final_fixed,
            observer,
            roots,
        }
    }
    fn check(path: &Path) {
        for mode in 0..3 {
            let full = run(path, mode, None);
            let split = run(path, mode, Some(2));
            close(&split.logits, &full.logits);
            assert_eq!(split.cached.len(), 3);
            for (actual, expected) in split.cached.iter().zip(&full.cached) {
                close(actual, expected);
            }
            same_arrays(&split.state, &full.state);
            same_arrays(&split.final_state, &full.final_state);
            fixed_equal(&split.fixed, &full.fixed);
            fixed_equal(&split.final_fixed, &full.final_fixed);
            assert_eq!(
                split
                    .observer
                    .chunks
                    .iter()
                    .map(|c| c.input.clone())
                    .collect::<Vec<_>>(),
                vec![0..2, 2..4, 4..6, 6..7]
            );
            assert!(split.observer.chunks[..3]
                .iter()
                .all(|c| c.output == eredu_core::OutputDemand::StateOnly));
            assert_eq!(
                split.observer.outputs,
                vec![vec![1, 1, 64]],
                "only the final decoder row reaches the vocabulary observation"
            );
            assert_eq!(split.roots.len(), 8);
            let first = &split.roots[..2];
            assert!(!first[0].after && first[1].after);
            assert!(!first[0].shapes.is_empty());
            assert_eq!(first[0].shapes, first[1].shapes);
            assert!(
                first[0].ready.iter().any(|ready| !ready),
                "future-only media must be lazy before first text-only span completion"
            );
            assert!(
                first[1].ready.iter().all(|ready| *ready),
                "first completion must settle even unused future roots"
            );
            assert!(split
                .roots
                .iter()
                .filter(|sample| sample.after)
                .all(|sample| sample.ready.iter().all(|ready| *ready)));
            assert!(
                full.roots.is_empty(),
                "ordinary full reference retains its old completion route"
            );
        }
    }
    #[test]
    fn media_capable_selected_model_keeps_existing_token_only_prefill_path() {
        let root = tempfile::tempdir().unwrap();
        write_qwen3_vl_component_fixture(root.path(), false, false);
        let prompt = |chunk| {
            let tokens = Array::from_slice(&[1_u32, 2, 3, 4, 5], &[1, 5]);
            let input = synthetic_prediction_input(&[text_input_part(&tokens)], &[1, 2, 3, 4, 5]);
            match chunk {
                Some(n) => {
                    input.with_prefill_chunk_positions(std::num::NonZeroU64::new(n).unwrap())
                }
                None => input,
            }
        };
        for mode in 0..3 {
            let full = run_prompt(root.path(), mode, prompt(None));
            let split = run_prompt(root.path(), mode, prompt(Some(2)));
            assert!(full.roots.is_empty() && split.roots.is_empty());
            assert_eq!(
                split
                    .observer
                    .chunks
                    .iter()
                    .map(|c| c.input.clone())
                    .collect::<Vec<_>>(),
                vec![0..2, 2..4, 4..5]
            );
            close(&split.logits, &full.logits);
            for (actual, expected) in split.cached.iter().zip(&full.cached) {
                close(actual, expected);
            }
            same_arrays(&split.state, &full.state);
            same_arrays(&split.final_state, &full.final_state);
            fixed_equal(&split.fixed, &full.fixed);
            fixed_equal(&split.final_fixed, &full.final_fixed);
        }
    }
    #[test]
    fn media_capable_selected_model_preserves_actual_pure_text_capture() {
        use eredu_core::{capture::*, TextGenerationBackend as _};
        let root = tempfile::tempdir().unwrap();
        write_qwen3_vl_component_fixture(root.path(), false, false);
        let sampling = eredu_core::resolve_generation_config(
            None,
            eredu_core::GenerationConfigOverrides {
                do_sample: Some(false),
                max_new_tokens: Some(3),
                ..Default::default()
            },
        )
        .unwrap();
        for mode in 0..3 {
            let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
            let mut results = Vec::new();
            let mut states = Vec::new();
            for captured in [false, true] {
                let backend = crate::native::backend(&stream, &stream);
                let model = load_model(&backend, root.path(), weights(mode)).unwrap();
                let mut runtime = ModelRuntime::from_prepared(backend, model).unwrap();
                let discovery = MlxBackend::capture_discovery(&runtime).unwrap();
                let mut raw = CapturePlan::none();
                raw.selections.push(CaptureSelection {
                    id: "text-logits".into(),
                    path: eredu_core::MODEL_LOGITS_OBSERVATION_PATH.into(),
                    schedule: CaptureSchedule::default(),
                    slices: vec![],
                    transform: CaptureTransform::FullTensor,
                });
                let limit = CaptureUsage {
                    captures: 16,
                    retained_bytes: 1 << 24,
                    host_bytes: 1 << 24,
                    encoded_bytes: 1 << 24,
                };
                raw.limits.per_step = limit;
                raw.limits.cumulative = limit;
                let plan = raw
                    .admit(
                        &discovery.catalog,
                        &discovery.support,
                        &discovery.support.capture,
                        CaptureRequestShape {
                            batch: 1,
                            prompt_tokens: 5,
                            max_predictions: 3,
                        },
                    )
                    .unwrap();
                let tokens = Array::from_slice(&[1_u32, 2, 3, 4, 5], &[1, 5]);
                let prompt =
                    synthetic_prediction_input(&[text_input_part(&tokens)], &[1, 2, 3, 4, 5])
                        .with_prefill_chunk_positions(std::num::NonZeroU64::new(2).unwrap());
                let ((ids, captures), roots) = media_completion::observe(None, || {
                    let mut generation = eredu_core::ControlledTextGeneration::from_prompt(
                        &mut runtime,
                        prompt,
                        TextGenerationConfig::new(sampling),
                        AllowAllTokens,
                    )
                    .unwrap();
                    if captured {
                        generation.enable_capture(plan).unwrap();
                    }
                    let mut ids = Vec::new();
                    let mut captures = Vec::new();
                    for _ in 0..3 {
                        ids.push(generation.next().unwrap().unwrap().token_id());
                        if captured {
                            captures.push(generation.take_captured_delivery().unwrap().unwrap());
                        }
                    }
                    assert!(generation.next().is_none());
                    (ids, captures)
                });
                assert!(
                    roots.is_empty(),
                    "pure text must retain its established captured source route"
                );
                if captured {
                    assert_eq!(captures.len(), 3);
                    for (index, step) in captures.iter().enumerate() {
                        assert_eq!(step.records.len(), 1);
                        assert_eq!(step.records[0].outcome, CaptureOutcome::Captured);
                        let Some(CapturePayload::Tensor(tensor)) = &step.records[0].payload else {
                            panic!("actual tensor capture")
                        };
                        assert_eq!(tensor.shape(), &[1, if index == 0 { 5 } else { 1 }, 64]);
                        let eredu_core::TensorObservationData::F32(values) = tensor.data() else {
                            panic!("F32 logits")
                        };
                        assert!(values.iter().any(|v| v.abs() > 1e-7));
                    }
                }
                results.push(ids);
                states.push(snapshot(runtime.session_mut()));
            }
            assert_eq!(results[0], results[1]);
            same_arrays(&states[0].0, &states[1].0);
            fixed_equal(&states[0].1, &states[1].1);
        }
    }
    #[test]
    fn native_qwen_vl_media_spans_complete_future_roots_and_match_full_state_all_residencies() {
        for routed in [false, true] {
            let root = tempfile::tempdir().unwrap();
            write_qwen3_vl_component_fixture(root.path(), routed, false);
            check(root.path());
        }
    }
    #[test]
    fn native_conditional_qwen_media_spans_match_full_state_all_residencies() {
        for routed in [false, true] {
            let root = tempfile::tempdir().unwrap();
            write_qwen35_conditional_component_fixture(root.path(), routed);
            check(root.path());
        }
    }
    fn snapshot(session: &mut MlxModelSession) -> (Arrays, Fixed) {
        let target = session.neutral_prediction_target_mut().unwrap();
        (
            target.retained_numeric_state_snapshot().unwrap().unwrap(),
            target.fixed_numeric_state_snapshot().unwrap(),
        )
    }
    fn cancelled(
        path: &Path,
        mode: usize,
        chunk: u64,
        boundary: usize,
    ) -> (Arrays, Fixed, Vec<Vec<f32>>) {
        let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let backend = crate::native::backend(&stream, &stream);
        let model = load_model(&backend, path, weights(mode))
            .unwrap()
            .into_inner();
        let mut session = MlxModelSession::from_model(
            model,
            eredu_core::SessionCapabilities::new(true, true, true),
        )
        .unwrap();
        let before = snapshot(&mut session);
        let cancellation = GenerationCancellationToken::new();
        if boundary == 0 {
            cancellation.cancel();
        }
        let (result, roots) =
            media_completion::observe(Some((cancellation.clone(), boundary)), || {
                session.prefill_cancellable(&backend, input(Some(chunk), false), &cancellation)
            });
        assert!(result.unwrap().is_none());
        crate::backend::submission_recovery::wait_for_retirement(|| {
            session.neutral_prediction_target_mut().is_ok()
        });
        assert_eq!(
            roots.len(),
            boundary * 2,
            "no later encoder/span work after cancellation"
        );
        let (arrays, fixed) = snapshot(&mut session);
        if boundary == 0 {
            same_arrays(&arrays, &before.0);
            fixed_equal(&fixed, &before.1);
            return (arrays, fixed, Vec::new());
        }
        assert!(roots[1].ready.iter().all(|ready| *ready));
        let completed = (chunk * boundary as u64).min(7);
        session
            .neutral_prediction_target_mut()
            .unwrap()
            .validate_text_frontier(completed)
            .unwrap();
        let mut cached = Vec::new();
        // Ordinary cancelled-prefix decoding is existing compatibility behavior;
        // this source is terminal and no original allowance is replenished.
        for token in [4, 5, 6] {
            let submission = session.submit_token_decode(&backend, token).unwrap();
            submission.completion.wait().unwrap();
            cached.push(values(submission.output.logits().unwrap().as_array()));
        }
        (arrays, fixed, cached)
    }
    #[test]
    fn native_media_cancellation_preserves_equal_prefix_across_span_boundaries() {
        let root = tempfile::tempdir().unwrap();
        write_qwen3_vl_component_fixture(root.path(), false, false);
        for mode in 0..3 {
            cancelled(root.path(), mode, 2, 0);
            // Same prefix ending inside media, independently partitioned 2+2 vs4.
            let a = cancelled(root.path(), mode, 2, 2);
            let b = cancelled(root.path(), mode, 4, 1);
            same_arrays(&a.0, &b.0);
            fixed_equal(&a.1, &b.1);
            assert_eq!(a.2.len(), 3);
            for (actual, expected) in a.2.iter().zip(&b.2) {
                close(actual, expected);
            }
            let prefix = cancelled(root.path(), mode, 2, 1);
            assert!(!prefix.0.is_empty());
        }
    }
    #[test]
    fn native_media_exact_request_is_retained_and_foreign_or_replayed_request_is_rejected() {
        use eredu_runtime::working_memory::InferenceExecutionIdentity;
        let root = tempfile::tempdir().unwrap();
        write_qwen3_vl_component_fixture(root.path(), false, false);
        let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let backend = crate::native::backend(&stream, &stream);
        let model = load_model(&backend, root.path(), weights(0)).unwrap();
        let mut runtime = ModelRuntime::from_prepared(backend, model).unwrap();
        let (prompt, _source_custody, attribution) =
            crate::tests::support::original_input::prepare(&runtime, input(Some(2), false));
        let execution = runtime
            .session_mut()
            .neutral_prediction_target_mut()
            .unwrap()
            .inference_execution_identity()
            .clone();
        let geometry = eredu_core::InferenceGeometry {
            batch_size: 1,
            cached_positions: 0,
            input_positions: attribution.decoder_positions,
            max_output_tokens: 2,
            prefill_chunk_positions: 2,
            output: eredu_core::OutputDemand::LastPosition,
        };
        let config = TextGenerationConfig::new(
            eredu_core::resolve_generation_config(
                None,
                eredu_core::GenerationConfigOverrides {
                    do_sample: Some(false),
                    max_new_tokens: Some(2),
                    ..Default::default()
                },
            )
            .unwrap(),
        );
        let foreign = crate::memory_fixture::empty_admitted_request(
            &InferenceExecutionIdentity::default(),
            geometry,
        )
        .unwrap();
        let before = snapshot(runtime.session_mut());
        let (error, roots) = media_completion::observe(None, || {
            eredu_core::TextGeneration::from_input_with_options(
                &mut runtime,
                eredu_core::TextGenerationInput::OriginalPrepared(
                    prompt.clone().with_inference_request(foreign),
                ),
                config.clone(),
                Default::default(),
            )
            .err()
            .expect("foreign request")
        });
        assert!(!error.to_string().is_empty());
        assert!(roots.is_empty());
        assert_eq!(snapshot(runtime.session_mut()), before);
        let (request, replay) = {
            let mut driver = eredu_core::TextGenerationDriver::new(&mut runtime);
            let mut continuation = driver
                .start_input(
                    eredu_core::TextGenerationInput::OriginalPrepared(prompt),
                    config,
                    AllowAllTokens,
                )
                .unwrap();
            let (request, replay) = {
                let mut boundary = driver.quiescent(&mut continuation).unwrap();
                let Some(eredu_core::PendingTextInput::Prefill(prompt)) = boundary.parts().2 else {
                    panic!("pending original prompt")
                };
                (
                    prompt.with_borrowed(|input| input.inference_request().unwrap().clone()),
                    prompt.clone(),
                )
            };
            assert!(driver.advance(&mut continuation).unwrap().is_some());
            (request, replay)
        };
        let target = runtime
            .session_mut()
            .neutral_prediction_target_mut()
            .unwrap();
        assert!(target
            .retained_inference_authority()
            .unwrap()
            .requests()
            .any(|retained| retained.validate_same_request(&request).is_ok()));
        assert!(matches!(
            eredu_runtime::prefill::PrefillDriver::<MlxTensor, crate::backend::MlxCompletion>::new(
                &execution,
                request.clone(),
                request.geometry(),
                eredu_core::GenerationCancellationToken::new()
            ),
            Err(eredu_runtime::working_memory::WorkingMemoryError::AlreadyStarted)
        ));
        let before = snapshot(runtime.session_mut());
        let (result, roots) = media_completion::observe(None, || runtime.prefill(replay));
        assert!(result.is_err());
        assert!(roots.is_empty());
        assert_eq!(snapshot(runtime.session_mut()), before);
    }
    #[test]
    fn native_media_first_validation_failure_preserves_state_and_never_publishes_ready_source() {
        let root = tempfile::tempdir().unwrap();
        write_qwen3_vl_component_fixture(root.path(), false, false);
        let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let backend = crate::native::backend(&stream, &stream);
        let model = load_model(&backend, root.path(), weights(0))
            .unwrap()
            .into_inner();
        let mut session = MlxModelSession::from_model(
            model,
            eredu_core::SessionCapabilities::new(true, true, true),
        )
        .unwrap();
        let before = snapshot(&mut session);
        let mut observer = Observer::default();
        let (error, roots) = media_completion::observe(None, || {
            session
                .submit_prefill_with_observer(&backend, input(Some(2), true), &mut observer)
                .err()
                .unwrap()
        });
        assert!(error.to_string().contains("outside 0..64"));
        assert!(error.model_state_preserved());
        assert_eq!(observer.chunks.len(), 1);
        assert!(!roots.is_empty());
        assert!(
            roots.iter().all(|sample| !sample.after),
            "failed validation cannot establish source Ready"
        );
        crate::backend::submission_recovery::wait_for_retirement(|| {
            session.neutral_prediction_target_mut().is_ok()
        });
        assert_eq!(snapshot(&mut session), before);
    }
    #[test]
    fn ordinary_core_prepared_media_iterator_and_manual_advance_share_native_path() {
        let root = tempfile::tempdir().unwrap();
        write_qwen3_vl_component_fixture(root.path(), false, false);
        let sampling = eredu_core::resolve_generation_config(
            None,
            eredu_core::GenerationConfigOverrides {
                do_sample: Some(false),
                max_new_tokens: Some(3),
                ..Default::default()
            },
        )
        .unwrap();
        for mode in 0..3 {
            let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
            let mut outputs = Vec::new();
            let mut states = Vec::new();
            for manual in [false, true] {
                let backend = crate::native::backend(&stream, &stream);
                let model = load_model(&backend, root.path(), weights(mode)).unwrap();
                let mut runtime = ModelRuntime::from_prepared(backend, model).unwrap();
                let prompt = input(Some(2), false);
                let expected_identity = prompt.cache_identity().unwrap().clone();
                let (ids, roots) = media_completion::observe(None, || {
                    if manual {
                        let mut generation = eredu_core::ControlledTextGeneration::from_input(
                            &mut runtime,
                            eredu_core::TextGenerationInput::Prepared(prompt),
                            TextGenerationConfig::new(sampling),
                            AllowAllTokens,
                        )
                        .unwrap();
                        let mut ids = Vec::new();
                        while let Some(next) =
                            generation.next_cancellable(&GenerationCancellationToken::new())
                        {
                            ids.push(next.unwrap().token_id());
                        }
                        ids
                    } else {
                        eredu_core::TextGeneration::from_prompt(
                            &mut runtime,
                            prompt,
                            TextGenerationConfig::new(sampling),
                        )
                        .unwrap()
                        .map(|next| next.unwrap().token_id().unwrap())
                        .collect::<Vec<_>>()
                    }
                });
                assert_eq!(ids.len(), 3);
                assert_eq!(roots.len(), 8);
                assert_eq!(
                    runtime
                        .session_mut()
                        .neutral_prediction_target_mut()
                        .unwrap()
                        .resident_copy_input_identity()
                        .unwrap()
                        .as_ref()
                        .map(AsRef::as_ref),
                    Some(&expected_identity)
                );
                outputs.push(ids);
                states.push(snapshot(runtime.session_mut()));
            }
            assert_eq!(outputs[0], outputs[1]);
            same_arrays(&states[0].0, &states[1].0);
            fixed_equal(&states[0].1, &states[1].1);
        }
    }

    #[test]
    fn native_media_rejects_foreign_identity_required_capture_and_managed_preparation() {
        struct Required;
        impl eredu_runtime::ActivationObserver<MlxTensor, Error> for Required {
            fn requires_sequence_readout(&self) -> bool {
                false
            }
            fn requires_prepared_traversal(&self) -> bool {
                true
            }
            fn observe(&mut self, _: &str, _: &MlxTensor) -> Result<(), Error> {
                panic!("rejected before observations")
            }
        }
        let root = tempfile::tempdir().unwrap();
        write_qwen3_vl_component_fixture(root.path(), false, false);
        let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let backend = crate::native::backend(&stream, &stream);
        let model = load_model(&backend, root.path(), weights(0)).unwrap();
        let mut runtime = ModelRuntime::from_prepared(backend, model).unwrap();
        let before = snapshot(runtime.session_mut());
        let (backend, session) = runtime.parts_mut();
        let (error, roots) = media_completion::observe(None, || {
            session
                .submit_prefill_with_observer(backend, input(Some(2), false), &mut Required)
                .err()
                .unwrap()
        });
        assert!(error.model_state_preserved());
        assert!(roots.is_empty());
        assert!(error
            .to_string()
            .contains("media capture/transaction attribution is not admitted"));
        assert_eq!(snapshot(session), before);
        let other = synthetic_prediction_input(
            &[text_input_part(&Array::from_slice(&[1_u32], &[1, 1]))],
            &[1],
        );
        let foreign = other.cache_identity().unwrap();
        let prompt = input(Some(2), false).with_borrowed(|actual| {
            MlxModelInput::from(
                crate::backend::runtime::media::input::ModelInput::with_cache_identity(
                    actual.parts,
                    foreign,
                )
                .with_prefill_chunk_positions(2.try_into().unwrap()),
            )
        });
        let (error, roots) = media_completion::observe(None, || {
            session
                .submit_prefill_with_observer(backend, prompt, &mut Observer::default())
                .err()
                .unwrap()
        });
        assert!(error.model_state_preserved());
        assert!(roots.is_empty());
        assert!(error
            .to_string()
            .contains("prepared-input cache identity differs"));
        assert_eq!(snapshot(session), before);
        let sampling = eredu_core::resolve_generation_config(
            None,
            eredu_core::GenerationConfigOverrides {
                do_sample: Some(false),
                max_new_tokens: Some(3),
                ..Default::default()
            },
        )
        .unwrap();
        let config = TextGenerationConfig::new(sampling).with_inference_policy(
            eredu_core::TextInferencePolicy {
                prefill_chunk_positions: Some(2.try_into().unwrap()),
                memory_limits: crate::memory_fixture::limits(1 << 28),
                submission_tracking_capacity_bytes: None,
                graph_metadata_capacity_bytes: None,
            },
        );
        let (error, roots) = media_completion::observe(None, || {
            eredu_core::TextGeneration::from_prompt(&mut runtime, input(Some(2), false), config)
                .err()
                .unwrap()
        });
        assert!(roots.is_empty());
        assert!(error.to_string().contains("bound"));
        assert_eq!(snapshot(runtime.session_mut()), before);
    }
    #[test]
    fn prepared_control_media_attribution_keeps_actual_nonzero_frontier_and_source_transfer() {
        use eredu_core::PromptTokenAttribution;
        for conditional in [false, true] {
            let root = tempfile::tempdir().unwrap();
            if conditional {
                write_qwen35_conditional_component_fixture(root.path(), false);
            } else {
                write_qwen3_vl_component_fixture(root.path(), false, false);
            }
            for mode in 0..3 {
                let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
                let backend = crate::native::backend(&stream, &stream);
                let model = load_model(&backend, root.path(), weights(mode)).unwrap();
                let mut runtime = ModelRuntime::from_prepared(backend, model).unwrap();
                let prefix = Array::from_slice(&[4_u32, 5, 6], &[1, 3]);
                let prefix = synthetic_prediction_input(&[text_input_part(&prefix)], &[4, 5, 6]);
                runtime.prefill(prefix).unwrap().wait().unwrap();
                let before = snapshot(runtime.session_mut());
                let raw = input(Some(2), false);
                assert_eq!(raw.controlled_decoder_positions(), None);
                let ((prompt, _source_custody, attribution), roots) =
                    media_completion::observe(None, || {
                        crate::tests::support::original_input::prepare(&runtime, raw)
                    });
                assert!(
                    roots.is_empty(),
                    "attribution reads do not run the media encoder"
                );
                let source = &attribution;
                assert_eq!(source.opening_position, 3);
                assert_eq!(source.decoder_positions, 7);
                assert_eq!(source.canonical_token_ids, [1, 2, 3]);
                assert_eq!(
                    source
                        .segments
                        .iter()
                        .map(|s| s.plan.decoder_range)
                        .collect::<Vec<_>>(),
                    [[0, 2], [2, 6], [6, 7]]
                );
                assert!(matches!(
                    source.segments[1].tokens,
                    PromptTokenAttribution::NotTokenized
                ));
                assert_eq!(source.complete_token_ids(), None);
                assert_eq!(source.input_range(0).unwrap(), [3, 10]);
                assert_eq!(source.input_range(3).unwrap(), [12, 13]);
                assert_eq!(snapshot(runtime.session_mut()), before);
                assert_eq!(
                    crate::tests::support::original_input::attribution(&prompt).decoder_positions,
                    7
                );
                assert_eq!(
                    crate::tests::support::original_input::attribution(&prompt.clone())
                        .decoder_positions,
                    7
                );
                let relabeled = prompt
                    .clone()
                    .with_semantic_content_fingerprint("independent relabel")
                    .unwrap();
                assert_eq!(relabeled.controlled_decoder_positions(), None);
                assert_eq!(attribution.opening_position, 3);
                assert_eq!(snapshot(runtime.session_mut()), before);
            }
        }
    }

    #[test]
    fn original_media_source_keeps_native_rows_and_state_in_controlled_and_uninterrupted_generation(
    ) {
        let root = tempfile::tempdir().unwrap();
        write_qwen3_vl_component_fixture(root.path(), false, false);
        let sampling = eredu_core::resolve_generation_config(
            None,
            eredu_core::GenerationConfigOverrides {
                do_sample: Some(false),
                max_new_tokens: Some(4),
                ..Default::default()
            },
        )
        .unwrap();
        for mode in 0..3 {
            let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
            let mut ids = Vec::new();
            let mut states = Vec::new();
            for carrier in [false, true] {
                let backend = crate::native::backend(&stream, &stream);
                let model = load_model(&backend, root.path(), weights(mode)).unwrap();
                let mut runtime = ModelRuntime::from_prepared(backend, model).unwrap();
                let (prompt, _source_custody, _) =
                    crate::tests::support::original_input::prepare(&runtime, input(Some(2), false));
                let (tokens, roots) = media_completion::observe(None, || {
                    let config = TextGenerationConfig::new(sampling);
                    if carrier {
                        eredu_core::ControlledTextGeneration::from_input(
                            &mut runtime,
                            eredu_core::TextGenerationInput::OriginalPrepared(prompt),
                            config,
                            AllowAllTokens,
                        )
                        .unwrap()
                        .map(|token| token.unwrap().token_id())
                        .collect::<Vec<_>>()
                    } else {
                        eredu_core::TextGeneration::from_input_with_options(
                            &mut runtime,
                            eredu_core::TextGenerationInput::OriginalPrepared(prompt),
                            config,
                            Default::default(),
                        )
                        .unwrap()
                        .map(|token| token.unwrap().token_id().unwrap())
                        .collect::<Vec<_>>()
                    }
                });
                assert_eq!(tokens.len(), 4);
                assert_eq!(roots.len(), 8);
                ids.push(tokens);
                states.push(snapshot(runtime.session_mut()));
            }
            assert_eq!(ids[0], ids[1]);
            same_arrays(&states[0].0, &states[1].0);
            fixed_equal(&states[0].1, &states[1].1);
        }
    }

    #[test]
    fn prepared_control_rejects_foreign_and_equal_frontier_restored_native_source() {
        if !crate::tests::support::native_process::enter("prepared-media-exchange") {
            return;
        }
        crate::tests::support::test_utils::initialize_original_sources();
        let root = tempfile::tempdir().unwrap();
        write_qwen3_vl_component_fixture(root.path(), false, false);
        let make = || {
            let plan = eredu_core::ExecutionPlan::fully_resident(
                eredu_core::DevicePlan::new("mlx", "cpu:0").unwrap(),
            );
            let factory = crate::MlxBackendFactory::default();
            let selected = eredu_core::select_execution_plan_target(
                &factory,
                &plan,
                component_fixture_inspection(root.path()),
            )
            .unwrap();
            eredu_core::realize_execution_plan_target(&factory, &plan, selected)
                .unwrap()
                .into_runtime()
                .unwrap()
        };
        let mut runtime = make();
        let mut foreign = make();
        let (source, _custody, _) =
            crate::tests::support::original_input::prepare(&runtime, input(Some(2), false));
        let config = || {
            TextGenerationConfig::new(
                eredu_core::resolve_generation_config(
                    None,
                    eredu_core::GenerationConfigOverrides {
                        max_new_tokens: Some(3),
                        ..Default::default()
                    },
                )
                .unwrap(),
            )
        };
        assert!(eredu_core::TextGeneration::from_input_with_options(
            &mut foreign,
            eredu_core::TextGenerationInput::OriginalPrepared(source),
            config(),
            Default::default()
        )
        .is_err());
        let (source, _custody, _) =
            crate::tests::support::original_input::prepare(&runtime, input(Some(2), false));
        let before = snapshot(runtime.session_mut());
        component_exchange_same_frontier(&mut runtime);
        assert_eq!(
            snapshot(runtime.session_mut()),
            before,
            "equal numeric frontier is not old revision authority"
        );
        assert!(eredu_core::TextGeneration::from_input_with_options(
            &mut runtime,
            eredu_core::TextGenerationInput::OriginalPrepared(source),
            config(),
            Default::default()
        )
        .is_err());
    }

    #[test]
    fn prepared_control_signed_ids_and_projected_text_have_distinct_real_attribution() {
        use eredu_core::PromptTokenAttribution;
        for conditional in [false, true] {
            let root = tempfile::tempdir().unwrap();
            if conditional {
                write_qwen35_conditional_component_fixture(root.path(), false);
            } else {
                write_qwen3_vl_component_fixture(root.path(), false, false);
            }
            let fixture: serde_json::Value =
                serde_json::from_slice(&std::fs::read(root.path().join("config.json")).unwrap())
                    .unwrap();
            let hidden =
                i32::try_from(fixture["text_config"]["hidden_size"].as_u64().unwrap()).unwrap();
            let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
            let backend = crate::native::backend(&stream, &stream);
            let model = load_model(&backend, root.path(), weights(0)).unwrap();
            let runtime = ModelRuntime::from_prepared(backend, model).unwrap();
            for projected in [false, true] {
                let ids = Array::from_slice(&[1_i32, 2], &[1, 2]);
                let mut parts = vec![text_input_part(&ids)];
                if projected {
                    parts.push(input_part(
                        InputModality::Text,
                        InputPayload::Embeddings(Array::from_slice(
                            &(0..2 * hidden)
                                .map(|i| (i as f32 - 7.0) / 17.0)
                                .collect::<Vec<_>>(),
                            &[1, 2, hidden],
                        )),
                        [],
                        [],
                    ));
                }
                let prompt = MlxModelInput::from(
                    crate::backend::runtime::media::input::ModelInput::new(&parts),
                );
                let prompt = if projected {
                    prompt
                        .with_semantic_content_fingerprint("actual projected fixture")
                        .unwrap()
                } else {
                    prompt
                };
                let (prompt, _source_custody, value) =
                    crate::tests::support::original_input::prepare(&runtime, prompt);
                assert_eq!(value.canonical_token_ids, [1, 2]);
                assert_eq!(value.decoder_positions, if projected { 4 } else { 2 });
                if projected {
                    assert!(matches!(
                        value.segments[1].tokens,
                        PromptTokenAttribution::NotTokenized
                    ));
                    assert_eq!(value.complete_token_ids(), None);
                } else {
                    assert_eq!(value.complete_token_ids(), Some([1, 2].as_slice()));
                }
                assert!(
                    prompt.cache_identity().is_some(),
                    "signed text gained identity from actual canonical values"
                );
                assert_eq!(
                    crate::tests::support::original_input::attribution(&prompt).decoder_positions,
                    if projected { 4 } else { 2 }
                );
            }
        }
    }
}
