mod prepared_capture {
    use super::*;
    use eredu_core::{
        capture::*, PreparedControlInput, PreparedControlInputBackend, TextGenerationBackend,
    };
    fn prompt(family: usize, chunk: Option<u64>) -> MlxModelInput {
        prompt_length(family, chunk, false)
    }
    fn prompt_length(family: usize, chunk: Option<u64>, long: bool) -> MlxModelInput {
        let tokens = Array::from_slice(&[1_u32, 2], &[1, 2]);
        let tail = if long {
            Array::from_slice(&[3_u32, 4], &[1, 2])
        } else {
            Array::from_slice(&[3_u32], &[1, 1])
        };
        let media = if family == 4 {
            input_part(
                InputModality::Audio,
                InputPayload::Tensor(Array::from_slice(&[0_u32, 1, 2, 3], &[1, 2, 2])),
                [],
                [],
            )
        } else {
            let width = if family == 3 { 48 } else { 12 };
            let shape = if family == 3 {
                vec![1, 8, width]
            } else {
                vec![8, width]
            };
            let pixels = Array::from_slice(
                &(0..8 * width)
                    .map(|i| (i as f32 - 41.) / 197.)
                    .collect::<Vec<_>>(),
                &shape,
            );
            let grid = Array::from_slice(&[1_i32, 2, 4], &[1, 3]);
            let mut metadata = vec![(InputMetadataKey::PatchGrid, grid)];
            if family == 3 {
                metadata.push((
                    InputMetadataKey::PatchPositions,
                    Array::from_slice(
                        &(0..8).flat_map(|i| [i / 4, i % 4]).collect::<Vec<i32>>(),
                        &[1, 8, 2],
                    ),
                ));
            }
            input_part(
                InputModality::Image,
                InputPayload::Tensor(pixels),
                metadata,
                [InputExtent::PatchGrid {
                    time: 1,
                    height: 2,
                    width: 4,
                }],
            )
        };
        let parts = [text_input_part(&tokens), media, text_input_part(&tail)];
        let input = MlxModelInput::from(crate::backend::runtime::media::input::ModelInput::new(
            &parts,
        ))
        .with_semantic_content_fingerprint(format!("prepared capture family {family}"))
        .unwrap();
        match chunk {
            Some(n) => input.with_prefill_chunk_positions(n.try_into().unwrap()),
            None => input,
        }
    }
    fn fixture(path: &Path, family: usize) {
        match family {
            0 => write_qwen3_vl_component_fixture(path, false, false),
            1 => write_qwen35_conditional_component_fixture(path, false),
            2 => write_muse_glimmer_component_fixture(path, false, false),
            3 => {
                write_gemma4_multimodal_tensor_parallel_fixture(path);
                initialize_gemma_capture_fixture(path);
            }
            4 => write_inkling_multimodal_fixture(path),
            _ => unreachable!(),
        }
    }
    // The shared Gemma geometry fixture intentionally stores zeros. Capture
    // parity needs distinct rows/channels and valid clipping ranges so omitted
    // projections and media contributions cannot pass as identical zero output.
    fn initialize_gemma_capture_fixture(path: &Path) {
        let file = path.join("model.safetensors");
        let bytes = std::fs::read(&file).unwrap();
        let source = safetensors::SafeTensors::deserialize(&bytes).unwrap();
        let tensors = source
            .tensors()
            .into_iter()
            .map(|(name, tensor)| {
                assert_eq!(tensor.dtype(), Dtype::F32);
                let seed = name
                    .bytes()
                    .fold(0_u32, |s, b| s.wrapping_mul(31).wrapping_add(b.into()));
                let data = (0..tensor.shape().iter().product::<usize>())
                    .flat_map(|i| {
                        let delta = ((i * 7 + (seed % 97) as usize) % 41) as f32 - 20.0;
                        let value = if name.ends_with("input_min") || name.ends_with("output_min") {
                            -8.0
                        } else if name.ends_with("input_max") || name.ends_with("output_max") {
                            8.0
                        } else if name.contains("norm") && name.ends_with("weight") {
                            0.9 + delta * 0.002
                        } else {
                            delta * 0.008
                        };
                        value.to_le_bytes()
                    })
                    .collect::<Vec<_>>();
                (name, tensor.shape().to_vec(), data)
            })
            .collect::<Vec<_>>();
        let views = tensors.iter().map(|(name, shape, data)| {
            (
                name.as_str(),
                TensorView::new(Dtype::F32, shape.clone(), data).unwrap(),
            )
        });
        serialize_to_file(views, None, &file).unwrap();
    }
    fn plan(empty: bool) -> CapturePlan {
        let usage = CaptureUsage {
            captures: 128,
            retained_bytes: 1 << 30,
            host_bytes: 1 << 30,
            encoded_bytes: 1 << 30,
        };
        let selection = |id: &str, sliced| CaptureSelection {
            id: id.into(),
            path: eredu_core::MODEL_LOGITS_OBSERVATION_PATH.into(),
            schedule: CaptureSchedule {
                decode: false,
                ..Default::default()
            },
            slices: if sliced {
                vec![CaptureSlice {
                    axis: "sequence".into(),
                    start: 1,
                    end: 5,
                    stride: 2,
                }]
            } else {
                vec![]
            },
            transform: if sliced {
                CaptureTransform::Slice
            } else {
                CaptureTransform::FullTensor
            },
        };
        CapturePlan {
            schema_version: 1,
            selections: if empty {
                vec![]
            } else {
                vec![selection("whole", false), selection("strided", true)]
            },
            limits: CaptureLimits {
                per_step: usage,
                cumulative: usage,
                physical_native_bytes: None,
                on_limit: CaptureLimitPolicy::Fail,
            },
        }
    }
    fn source(
        runtime: &ModelRuntime<MlxBackend<'_>>,
        input: &impl PreparedControlInput,
        empty: bool,
    ) -> SharedCapturePlan {
        source_plan(runtime, input, plan(empty))
    }
    fn source_plan(
        runtime: &ModelRuntime<MlxBackend<'_>>,
        input: &impl PreparedControlInput,
        selected: CapturePlan,
    ) -> SharedCapturePlan {
        let discovery = MlxBackend::capture_discovery(runtime).unwrap();
        let a = input.attribution();
        SharedCapturePlan::new(
            selected
                .admit_with_text_origin(
                    &discovery.catalog,
                    &discovery.support,
                    &discovery.support.capture,
                    CaptureRequestShape {
                        batch: a.batch,
                        prompt_tokens: a.decoder_positions,
                        max_predictions: 4,
                    },
                    CaptureTextOrigin {
                        cached_positions: a.opening_position,
                    },
                )
                .unwrap(),
        )
    }
    struct RetainedFrame(SharedCapturedStep);
    impl std::ops::Deref for RetainedFrame {
        type Target = CapturedStep;
        fn deref(&self) -> &CapturedStep {
            self.0.as_step()
        }
    }
    fn retained(delivery: SharedCapturedStep) -> RetainedFrame {
        let frame = delivery;
        RetainedFrame(frame)
    }
    struct Captured {
        ids: Vec<u32>,
        state: (Arrays, Fixed),
        frames: Vec<RetainedFrame>,
        roots: Vec<Sample>,
    }
    fn run(
        path: &Path,
        family: usize,
        mode: usize,
        chunk: Option<u64>,
        captured: Option<bool>,
        manual: bool,
        cancel: Option<usize>,
    ) -> Captured {
        run_plan(
            path,
            family,
            mode,
            chunk,
            captured.map(plan),
            manual,
            cancel,
            false,
        )
    }
    fn run_plan(
        path: &Path,
        family: usize,
        mode: usize,
        chunk: Option<u64>,
        selected: Option<CapturePlan>,
        manual: bool,
        cancel: Option<usize>,
        long: bool,
    ) -> Captured {
        let captured = selected.is_some();
        let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let backend = crate::native::backend(&stream, &stream);
        let model = load_model(&backend, path, weights(mode)).unwrap();
        let mut runtime = ModelRuntime::from_prepared(backend, model).unwrap();
        let config = TextGenerationConfig::new(
            eredu_core::resolve_generation_config(
                None,
                eredu_core::GenerationConfigOverrides {
                    do_sample: Some(false),
                    max_new_tokens: Some(4),
                    ..Default::default()
                },
            )
            .unwrap(),
        );
        let input = MlxBackend::prepare_control_input(&runtime, prompt_length(family, chunk, long))
            .unwrap();
        assert_eq!(
            input.attribution().decoder_positions,
            if long { 6 } else { 5 }
        );
        let source = selected.map(|plan| source_plan(&runtime, &input, plan));
        let input = if let Some(source) = &source {
            MlxBackend::bind_control_input_capture(&runtime, input, config, source.clone())
                .unwrap_or_else(|error| {
                    panic!("family {family}, mode {mode}, chunk {chunk:?}: {error:?}")
                })
        } else {
            input
        };
        let input = MlxBackend::consume_control_input(&runtime, input)
            .unwrap()
            .0;
        let cancellation = GenerationCancellationToken::new();
        if cancel == Some(0) {
            cancellation.cancel();
        }
        let ((ids, frames), roots) =
            media_completion::observe(cancel.map(|n| (cancellation.clone(), n)), || {
                let mut ids = Vec::new();
                let mut frames = Vec::new();
                if manual {
                    let mut generation = eredu_core::ControlledTextGeneration::from_input(
                        &mut runtime,
                        eredu_core::TextGenerationInput::Prepared(input),
                        config,
                        AllowAllTokens,
                    )
                    .unwrap();
                    if let Some(source) = source {
                        generation.enable_prepared_capture(source).unwrap();
                    }
                    while let Some(token) = generation.next_cancellable(&cancellation) {
                        ids.push(token.unwrap().token_id());
                        if captured {
                            assert!(generation.capture_pending());
                            frames.push(retained(
                                generation.take_captured_delivery().unwrap().unwrap(),
                            ));
                            assert!(!generation.capture_pending());
                        }
                    }
                } else {
                    let mut generation =
                        eredu_core::TextGeneration::from_prompt(&mut runtime, input, config)
                            .unwrap();
                    if let Some(source) = source {
                        generation.enable_prepared_capture(source).unwrap();
                    }
                    while let Some(token) = generation.next_cancellable(&cancellation) {
                        ids.push(token.unwrap().token_id().unwrap());
                        if captured {
                            assert!(generation.capture_pending());
                            frames.push(retained(
                                generation.take_captured_delivery().unwrap().unwrap(),
                            ));
                            assert!(!generation.capture_pending());
                        }
                    }
                }
                (ids, frames)
            });
        crate::backend::submission_recovery::wait_for_retirement(|| {
            runtime
                .session_mut()
                .neutral_prediction_target_mut()
                .is_ok()
        });
        let state = snapshot(runtime.session_mut());
        Captured {
            ids,
            state,
            frames,
            roots,
        }
    }

    fn globals(rows: u64, candidates_only: bool) -> CapturePlan {
        let mut out = plan(true);
        let mut add = |id: &str, transform: CaptureTransform, slices: Vec<CaptureSlice>| {
            out.selections.push(CaptureSelection {
                id: id.into(),
                path: eredu_core::MODEL_LOGITS_OBSERVATION_PATH.into(),
                schedule: CaptureSchedule {
                    decode: false,
                    ..Default::default()
                },
                slices,
                transform,
            });
        };
        if !candidates_only {
            add("reference", CaptureTransform::FullTensor, vec![]);
            add("summary", CaptureTransform::Summary, vec![]);
            add(
                "histogram",
                CaptureTransform::Histogram {
                    edges: vec![-1000., -0.25, 0.25, 1000.],
                },
                vec![],
            );
            add(
                "preview",
                CaptureTransform::Preview { max_elements: 9 },
                vec![CaptureSlice {
                    axis: "sequence".into(),
                    start: 1,
                    end: rows,
                    stride: 2,
                }],
            );
            add(
                "empty preview",
                CaptureTransform::Preview { max_elements: 0 },
                vec![],
            );
        }
        add(
            "candidates",
            CaptureTransform::TopCandidates { count: 3 },
            vec![],
        );
        out
    }
    fn floats(payload: &CapturePayload) -> (&[usize], &[f32]) {
        let tensor = payload.as_tensor().expect("floating tensor");
        let eredu_core::TensorObservationData::F32(values) = tensor.data() else {
            panic!("floating data")
        };
        (tensor.shape(), values)
    }
    fn validate_global(frame: &CapturedStep, rows: usize) {
        assert_eq!(frame.outcome, CaptureStepOutcome::Committed);
        assert_eq!(frame.prediction_index, 0);
        assert_eq!(frame.records.len(), 6);
        let (shape, values) = floats(frame.records[0].payload.as_ref().unwrap());
        assert_eq!(shape[1], rows);
        assert!(values.iter().all(|v| v.is_finite()));
        assert!(values.iter().any(|v| v.abs() > 1e-7));
        let CapturePayload::Summary(summary) = frame.records[1].payload.as_ref().unwrap() else {
            panic!("summary")
        };
        assert_eq!(
            (summary.elements, summary.finite, summary.non_finite),
            (values.len() as u64, values.len() as u64, 0)
        );
        let min = values.iter().copied().reduce(f32::min).unwrap() as f64;
        let max = values.iter().copied().reduce(f32::max).unwrap() as f64;
        let mean = values.iter().map(|&v| v as f64).sum::<f64>() / values.len() as f64;
        let rms =
            (values.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / values.len() as f64).sqrt();
        for (a, b) in [
            (summary.min.unwrap(), min),
            (summary.max.unwrap(), max),
            (summary.mean.unwrap(), mean),
            (summary.rms.unwrap(), rms),
        ] {
            assert!(
                (a - b).abs() <= 3e-4 + 3e-4 * b.abs(),
                "statistic {a} != {b}"
            );
        }
        let CapturePayload::Histogram(hist) = frame.records[2].payload.as_ref().unwrap() else {
            panic!("histogram")
        };
        let mut expected = vec![0; hist.edges.len() - 1];
        let (mut below, mut above) = (0, 0);
        for &v in values {
            if v < hist.edges[0] {
                below += 1;
            } else if v > *hist.edges.last().unwrap() {
                above += 1;
            } else {
                let i = hist
                    .edges
                    .windows(2)
                    .position(|b| v >= b[0] && v < b[1])
                    .unwrap_or(expected.len() - 1);
                expected[i] += 1;
            }
        }
        assert_eq!(
            (&hist.counts, hist.below, hist.above, hist.non_finite),
            (&expected, below, above, 0)
        );
        let selected = (1..rows)
            .step_by(2)
            .flat_map(|row| values[row * shape[2]..(row + 1) * shape[2]].iter().copied())
            .collect::<Vec<_>>();
        let (prefix_shape, prefix) = floats(frame.records[3].payload.as_ref().unwrap());
        assert_eq!(prefix_shape, [9]);
        close(prefix, &selected[..9]);
        assert_eq!(
            frame.records[3].outcome,
            CaptureOutcome::Truncated {
                available_elements: selected.len() as u64,
                emitted_elements: 9
            }
        );
        let (empty_shape, empty) = floats(frame.records[4].payload.as_ref().unwrap());
        assert_eq!(empty_shape, [0]);
        assert!(empty.is_empty());
        assert_eq!(
            frame.records[4].outcome,
            CaptureOutcome::Truncated {
                available_elements: values.len() as u64,
                emitted_elements: 0
            }
        );
        let CapturePayload::Candidates(candidate) = frame.records[5].payload.as_ref().unwrap()
        else {
            panic!("candidates")
        };
        assert_eq!(
            candidate.stage,
            CandidateScoreStage::RawLogitsBeforeSampling
        );
        assert_eq!(candidate.source, CandidateLogitsSource::Original);
        assert_eq!(candidate.candidates.len(), 3);
        let last = &values[values.len() - shape[2]..];
        for value in &candidate.candidates {
            close(&[value.score], &[last[value.token_id as usize]]);
            assert!(value.allowed);
        }
        assert!(candidate
            .candidates
            .windows(2)
            .all(|w| w[0].score >= w[1].score));
        assert_eq!(
            candidate.domain, None,
            "core AllowAllTokens does not claim tokenizer provenance"
        );
    }
    fn same_global(actual: &CapturedStep, expected: &CapturedStep) {
        assert_eq!(actual.records.len(), expected.records.len());
        for (a, b) in actual.records.iter().zip(&expected.records) {
            assert_eq!(
                (&a.outcome, &a.source_shape, &a.selected_shape),
                (&b.outcome, &b.source_shape, &b.selected_shape)
            );
            match (a.payload.as_ref().unwrap(), b.payload.as_ref().unwrap()) {
                (CapturePayload::Summary(a), CapturePayload::Summary(b)) => {
                    assert_eq!(
                        (a.elements, a.finite, a.non_finite),
                        (b.elements, b.finite, b.non_finite)
                    );
                    for (a, b) in [
                        (a.min, b.min),
                        (a.max, b.max),
                        (a.mean, b.mean),
                        (a.rms, b.rms),
                    ] {
                        let (a, b) = (a.unwrap(), b.unwrap());
                        assert!((a - b).abs() <= 3e-4 + 3e-4 * b.abs());
                    }
                }
                (CapturePayload::Histogram(a), CapturePayload::Histogram(b)) => assert_eq!(a, b),
                (CapturePayload::Candidates(a), CapturePayload::Candidates(b)) => {
                    assert_eq!(
                        (&a.stage, &a.source, &a.domain),
                        (&b.stage, &b.source, &b.domain)
                    );
                    for (a, b) in a.candidates.iter().zip(&b.candidates) {
                        assert_eq!((a.token_id, a.allowed), (b.token_id, b.allowed));
                        close(&[a.score], &[b.score]);
                    }
                }
                (a, b) => {
                    let (as_, av) = floats(a);
                    let (bs, bv) = floats(b);
                    assert_eq!(as_, bs);
                    close(av, bv);
                }
            }
        }
    }

    #[test]
    fn native_prepared_media_global_late_failure_retains_native_cause_and_committed_prefix() {
        fn exception<'a>(
            error: &'a (dyn std::error::Error + 'static),
        ) -> Option<&'a safemlx::error::Exception> {
            let mut current = Some(error);
            while let Some(error) = current {
                if let Some(value) = error.downcast_ref() {
                    return Some(value);
                }
                current = error.source();
            }
            None
        }
        let root = tempfile::tempdir().unwrap();
        fixture(root.path(), 0);
        for mode in 0..3 {
            for transform in [
                CaptureTransform::Summary,
                CaptureTransform::Histogram {
                    edges: vec![-1000., 0., 1000.],
                },
                CaptureTransform::Preview { max_elements: 7 },
                CaptureTransform::TopCandidates { count: 3 },
            ] {
                let candidate = matches!(transform, CaptureTransform::TopCandidates { .. });
                let baseline = run_plan(
                    root.path(),
                    0,
                    mode,
                    Some(2),
                    None,
                    true,
                    Some(if candidate { 2 } else { 1 }),
                    true,
                );
                let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
                let pool =
                    eredu_runtime::working_memory::WorkingMemoryPool::new(u64::MAX, 0).unwrap();
                let backend = MlxBackend::new(&stream, &stream).with_memory_pool(pool.clone());
                let model = load_model(&backend, root.path(), weights(mode)).unwrap();
                let mut runtime = ModelRuntime::from_prepared(backend, model).unwrap();
                let config = TextGenerationConfig::new(
                    eredu_core::resolve_generation_config(
                        None,
                        eredu_core::GenerationConfigOverrides {
                            do_sample: Some(false),
                            max_new_tokens: Some(4),
                            ..Default::default()
                        },
                    )
                    .unwrap(),
                );
                let input =
                    MlxBackend::prepare_control_input(&runtime, prompt_length(0, Some(2), true))
                        .unwrap();
                let mut selected = plan(true);
                selected.selections.push(CaptureSelection {
                    id: "failed global".into(),
                    path: eredu_core::MODEL_LOGITS_OBSERVATION_PATH.into(),
                    schedule: CaptureSchedule {
                        decode: false,
                        ..Default::default()
                    },
                    slices: vec![],
                    transform,
                });
                let source = source_plan(&runtime, &input, selected);
                let input =
                    MlxBackend::bind_control_input_capture(&runtime, input, config, source.clone())
                        .unwrap();
                let input = MlxBackend::consume_control_input(&runtime, input)
                    .unwrap()
                    .0;
                let native_error = Array::from_slice(&[1.0f32], &[1])
                    .reshape(&[2], &stream)
                    .unwrap_err();
                let expected = (native_error.what().to_owned(), native_error.location());
                let fault = crate::composition::mlx::fixture_bounded_capture::fail_transform_after(
                    if candidate { 0 } else { 1 },
                    native_error,
                );
                let mut generation = eredu_core::ControlledTextGeneration::from_input(
                    &mut runtime,
                    eredu_core::TextGenerationInput::Prepared(input),
                    config,
                    AllowAllTokens,
                )
                .unwrap();
                generation.enable_prepared_capture(source).unwrap();
                let error = match generation.next() {
                    Some(Err(error)) => error,
                    _ => panic!("selected global failure must not emit a token"),
                };
                let native = exception(&error).expect("original native cause");
                assert_eq!(native.what(), expected.0);
                assert_eq!(native.location(), expected.1);
                assert!(generation.capture_pending());
                let frame = retained(
                    generation
                        .take_captured_delivery()
                        .unwrap()
                        .expect("aborted logical frame"),
                );
                assert!(!generation.capture_pending());
                assert_eq!(frame.outcome, CaptureStepOutcome::Aborted);
                assert!(frame.records[0].payload.is_none());
                assert!(
                    frame.cumulative_usage.host_bytes > 0
                        && frame.cumulative_usage.retained_bytes > 0
                );
                assert!(matches!(
                    frame.records[0].outcome,
                    CaptureOutcome::Failed {
                        reason: CaptureFailureReason::Native,
                        ..
                    }
                ));
                drop(fault);
                drop(generation);
                crate::backend::submission_recovery::wait_for_retirement(|| {
                    runtime
                        .session_mut()
                        .neutral_prediction_target_mut()
                        .is_ok()
                });
                let state = snapshot(runtime.session_mut());
                same_arrays(&state.0, &baseline.state.0);
                fixed_equal(&state.1, &baseline.state.1);
                assert_eq!(
                    exception(&error).unwrap().what(),
                    expected.0,
                    "escaped error survives session observation"
                );
                // The standalone original native error is a separate lifetime.
                // Retire it and every session owner before testing the failed
                // frame's diagnostic/metadata authority alone.
                drop(error);
                drop(runtime);
                drop(state);
                stream.synchronize().unwrap();
                crate::backend::submission_recovery::wait_for_retirement(|| {
                    safemlx::try_retire_completed_submissions();
                    safemlx::reclaim_allocation_owners();
                    pool.unquoted_owner_count().unwrap() == 1
                });
                let alias = frame.0.clone();
                drop(frame);
                assert_eq!(pool.unquoted_owner_count().unwrap(), 1);
                assert!(
                    matches!(&alias.records()[0].outcome,CaptureOutcome::Failed{message,..}if !message.is_empty())
                );
                drop(alias);
                crate::backend::submission_recovery::wait_for_retirement(|| {
                    safemlx::try_retire_completed_submissions();
                    safemlx::reclaim_allocation_owners();
                    pool.unquoted_owner_count().unwrap() == 0
                });
            }
        }
    }
    #[test]
    fn native_prepared_media_global_transforms_match_full_rows_and_state_all_families_residencies()
    {
        for family in 0..5 {
            let root = tempfile::tempdir().unwrap();
            fixture(root.path(), family);
            for mode in 0..3 {
                for long in [false, true] {
                    let rows = if long { 6 } else { 5 };
                    let baseline =
                        run_plan(root.path(), family, mode, None, None, false, None, long);
                    let full = run_plan(
                        root.path(),
                        family,
                        mode,
                        None,
                        Some(globals(rows, false)),
                        false,
                        None,
                        long,
                    );
                    let chunked = run_plan(
                        root.path(),
                        family,
                        mode,
                        Some(2),
                        Some(globals(rows, false)),
                        true,
                        None,
                        long,
                    );
                    assert_eq!(baseline.ids.len(), 4);
                    for value in [&full, &chunked] {
                        assert_eq!(value.ids, baseline.ids);
                        same_arrays(&value.state.0, &baseline.state.0);
                        fixed_equal(&value.state.1, &baseline.state.1);
                        assert_eq!(value.frames.len(), 4);
                        validate_global(&value.frames[0], rows as usize);
                        assert!(value
                            .roots
                            .iter()
                            .filter(|r| r.after)
                            .all(|r| r.ready.iter().all(|v| *v)));
                    }
                    same_global(&chunked.frames[0], &full.frames[0]);
                }
            }
        }
    }
    #[test]
    fn native_prepared_media_candidates_only_terminal_and_cancellation_keep_actual_prefix() {
        use crate::backend::array_copy::CandidateExtraction;
        for family in 0..5 {
            let root = tempfile::tempdir().unwrap();
            fixture(root.path(), family);
            for mode in 0..3 {
                let full = run_plan(
                    root.path(),
                    family,
                    mode,
                    None,
                    Some(globals(6, true)),
                    false,
                    None,
                    true,
                );
                CandidateExtraction::reset_test_counts();
                let chunked = run_plan(
                    root.path(),
                    family,
                    mode,
                    Some(2),
                    Some(globals(6, true)),
                    true,
                    None,
                    true,
                );
                assert_eq!(
                    CandidateExtraction::test_counts().0,
                    1,
                    "one actual terminal extraction"
                );
                assert_eq!(full.ids, chunked.ids);
                same_arrays(&chunked.state.0, &full.state.0);
                fixed_equal(&chunked.state.1, &full.state.1);
                same_global(&chunked.frames[0], &full.frames[0]);
            }
            for boundary in [0, 1, 2] {
                let ordinary = run_plan(
                    root.path(),
                    family,
                    0,
                    Some(2),
                    None,
                    true,
                    Some(boundary),
                    true,
                );
                CandidateExtraction::reset_test_counts();
                let captured = run_plan(
                    root.path(),
                    family,
                    0,
                    Some(2),
                    Some(globals(6, true)),
                    true,
                    Some(boundary),
                    true,
                );
                assert_eq!(CandidateExtraction::test_counts().0, 0);
                assert!(captured.ids.is_empty() && captured.frames.is_empty());
                same_arrays(&captured.state.0, &ordinary.state.0);
                fixed_equal(&captured.state.1, &ordinary.state.1);
            }
        }
    }
    #[test]
    fn native_prepared_media_full_slice_empty_and_run_step_parity_all_five_families_residencies() {
        for family in 0..5 {
            let root = tempfile::tempdir().unwrap();
            fixture(root.path(), family);
            for mode in 0..3 {
                let baseline = run(root.path(), family, mode, None, None, false, None);
                let full = run(root.path(), family, mode, None, Some(false), false, None);
                let chunked = run(root.path(), family, mode, Some(2), Some(false), true, None);
                let empty = run(root.path(), family, mode, Some(2), Some(true), true, None);
                assert_eq!(baseline.ids.len(), 4);
                for candidate in [&full, &chunked, &empty] {
                    assert_eq!(candidate.ids, baseline.ids, "family{family} mode{mode}");
                    same_arrays(&candidate.state.0, &baseline.state.0);
                    fixed_equal(&candidate.state.1, &baseline.state.1);
                    assert_eq!(candidate.frames.len(), 4);
                }
                assert!(empty.frames.iter().all(|f| f.records.is_empty()));
                for candidate in [&full, &chunked] {
                    let frame = &candidate.frames[0];
                    assert_eq!(frame.prediction_index, 0);
                    assert_eq!(frame.outcome, CaptureStepOutcome::Committed);
                    assert_eq!(frame.records.len(), 2);
                    for (index, record) in frame.records.iter().enumerate() {
                        assert_eq!(record.outcome, CaptureOutcome::Captured);
                        let tensor = record.payload.as_ref().unwrap().as_tensor().unwrap();
                        assert_eq!(tensor.shape()[1], if index == 0 { 5 } else { 2 });
                        let eredu_core::TensorObservationData::F32(v) = tensor.data() else {
                            panic!("float scores")
                        };
                        assert!(v.iter().all(|x| x.is_finite()));
                        assert!(
                            v.iter().any(|x| x.abs() > 1e-7),
                            "family {family}, mode {mode}, record {index}"
                        );
                    }
                }
                for (a, b) in full.frames[0]
                    .records
                    .iter()
                    .zip(&chunked.frames[0].records)
                {
                    let a = a.payload.as_ref().unwrap().as_tensor().unwrap();
                    let b = b.payload.as_ref().unwrap().as_tensor().unwrap();
                    assert_eq!(a.shape(), b.shape());
                    let (
                        eredu_core::TensorObservationData::F32(a),
                        eredu_core::TensorObservationData::F32(b),
                    ) = (a.data(), b.data())
                    else {
                        unreachable!()
                    };
                    close(a, b);
                }
                assert!(
                    chunked.roots.len() >= 6,
                    "three actual spans retain their complete media roots"
                );
            }
        }
    }
    #[test]
    fn native_prepared_media_capture_cancellation_emits_no_token_or_frame_and_preserves_same_prefix(
    ) {
        let root = tempfile::tempdir().unwrap();
        fixture(root.path(), 0);
        for boundary in [0, 1, 2] {
            let ordinary = run(root.path(), 0, 0, Some(2), None, true, Some(boundary));
            let captured = run(
                root.path(),
                0,
                0,
                Some(2),
                Some(false),
                true,
                Some(boundary),
            );
            assert!(captured.ids.is_empty());
            assert!(captured.frames.is_empty());
            same_arrays(&ordinary.state.0, &captured.state.0);
            fixed_equal(&ordinary.state.1, &captured.state.1);
            assert_eq!(ordinary.roots.len(), captured.roots.len());
        }
    }
    #[test]
    fn native_prepared_media_capture_rejects_equal_content_source_before_encoder() {
        let root = tempfile::tempdir().unwrap();
        fixture(root.path(), 0);
        let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let backend = crate::native::backend(&stream, &stream);
        let model = load_model(&backend, root.path(), weights(0)).unwrap();
        let mut runtime = ModelRuntime::from_prepared(backend, model).unwrap();
        let config = TextGenerationConfig::new(
            eredu_core::resolve_generation_config(
                None,
                eredu_core::GenerationConfigOverrides {
                    max_new_tokens: Some(4),
                    ..Default::default()
                },
            )
            .unwrap(),
        );
        let input = MlxBackend::prepare_control_input(&runtime, prompt(0, Some(2))).unwrap();
        let source = source(&runtime, &input, false);
        let substitute = SharedCapturePlan::new(source.admission().clone());
        assert!(!source.same_storage(&substitute));
        let input =
            MlxBackend::bind_control_input_capture(&runtime, input, config, source).unwrap();
        let prompt = MlxBackend::consume_control_input(&runtime, input)
            .unwrap()
            .0;
        let (_, roots) = media_completion::observe(None, || {
            let mut generation = eredu_core::ControlledTextGeneration::from_prompt(
                &mut runtime,
                prompt,
                config,
                AllowAllTokens,
            )
            .unwrap();
            assert!(generation.enable_prepared_capture(substitute).is_err());
        });
        assert!(roots.is_empty());
    }
    #[test]
    fn native_prepared_capture_rejects_existing_mismatching_request_without_replacement_or_encoder()
    {
        use eredu_runtime::working_memory::{InferenceExecutionIdentity, InferenceRequest};
        let root = tempfile::tempdir().unwrap();
        fixture(root.path(), 0);
        let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let backend = crate::native::backend(&stream, &stream);
        let model = load_model(&backend, root.path(), weights(0)).unwrap();
        let mut runtime = ModelRuntime::from_prepared(backend, model).unwrap();
        let execution = runtime
            .session_mut()
            .neutral_prediction_target_mut()
            .unwrap()
            .inference_execution_identity()
            .clone();
        let config = TextGenerationConfig::new(
            eredu_core::resolve_generation_config(
                None,
                eredu_core::GenerationConfigOverrides {
                    max_new_tokens: Some(4),
                    ..Default::default()
                },
            )
            .unwrap(),
        );
        let expected = eredu_core::InferenceGeometry {
            batch_size: 1,
            cached_positions: 0,
            input_positions: 5,
            max_output_tokens: 4,
            prefill_chunk_positions: 2,
            output: eredu_core::OutputDemand::Sequence,
        };
        for wrong_identity in [false, true] {
            let mut geometry = expected;
            if !wrong_identity {
                geometry.output = eredu_core::OutputDemand::LastPosition;
            }
            let supplied_execution = if wrong_identity {
                InferenceExecutionIdentity::default()
            } else {
                execution.clone()
            };
            let request =
                InferenceRequest::without_memory_budget(&supplied_execution, geometry).unwrap();
            let pending = prompt(0, Some(2)).with_inference_request(request.clone());
            let input = MlxBackend::prepare_control_input(&runtime, pending).unwrap();
            let source = source(&runtime, &input, false);
            let before = snapshot(runtime.session_mut());
            let (result, roots) = media_completion::observe(None, || {
                MlxBackend::bind_control_input_capture(&runtime, input, config, source)
            });
            assert!(result.is_err());
            assert!(roots.is_empty());
            assert_eq!(snapshot(runtime.session_mut()), before);
            assert_eq!(request.geometry(), geometry);
            // A rejected attachment neither consumed nor replaced the original
            // ordinary request. Genuine preparation can still claim its start.
            if wrong_identity {
                assert!(request.validate(&execution, expected).is_err());
            }
            let prepared = request
                .prepare_text(&supplied_execution, geometry, config)
                .unwrap();
            drop(prepared);
        }
    }

    #[test]
    fn native_prepared_media_empty_and_tensor_frames_keep_actual_pool_owner_after_runtime() {
        use eredu_runtime::working_memory::WorkingMemoryPool;
        for empty in [false, true] {
            let root = tempfile::tempdir().unwrap();
            fixture(root.path(), 0);
            let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
            let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
            let frame = {
                let backend = MlxBackend::new(&stream, &stream).with_memory_pool(pool.clone());
                let model = load_model(&backend, root.path(), weights(0)).unwrap();
                let mut runtime = ModelRuntime::from_prepared(backend, model).unwrap();
                let config = TextGenerationConfig::new(
                    eredu_core::resolve_generation_config(
                        None,
                        eredu_core::GenerationConfigOverrides {
                            do_sample: Some(false),
                            max_new_tokens: Some(4),
                            ..Default::default()
                        },
                    )
                    .unwrap(),
                );
                let input =
                    MlxBackend::prepare_control_input(&runtime, prompt(0, Some(2))).unwrap();
                let source = source(&runtime, &input, empty);
                let input =
                    MlxBackend::bind_control_input_capture(&runtime, input, config, source.clone())
                        .unwrap();
                let input = MlxBackend::consume_control_input(&runtime, input)
                    .unwrap()
                    .0;
                let mut generation = eredu_core::ControlledTextGeneration::from_input(
                    &mut runtime,
                    eredu_core::TextGenerationInput::Prepared(input),
                    config,
                    AllowAllTokens,
                )
                .unwrap();
                generation.enable_prepared_capture(source).unwrap();
                let mut frame = None;
                for prediction in 0..2 {
                    assert!(generation.next().unwrap().is_ok());
                    assert!(generation.capture_pending());
                    assert!(generation.capture_pending());
                    let value = retained(generation.take_captured_delivery().unwrap().unwrap());
                    assert_eq!(value.prediction_index, prediction);
                    assert_eq!(value.outcome, CaptureStepOutcome::Committed);
                    assert_eq!(value.records.is_empty(), empty);
                    assert!(!generation.capture_pending());
                    frame = Some(value.0);
                }
                drop(generation);
                drop(runtime);
                frame.unwrap()
            };
            stream.synchronize().unwrap();
            crate::backend::submission_recovery::wait_for_retirement(|| {
                safemlx::try_retire_completed_submissions();
                safemlx::reclaim_allocation_owners();
                pool.unquoted_owner_count().unwrap() == 1
            });
            let alias = frame.clone();
            drop(frame);
            assert_eq!(pool.unquoted_owner_count().unwrap(), 1);
            assert_eq!(alias.phase(), CapturePhase::Decode);
            drop(alias);
            crate::backend::submission_recovery::wait_for_retirement(|| {
                safemlx::try_retire_completed_submissions();
                safemlx::reclaim_allocation_owners();
                pool.unquoted_owner_count().unwrap() == 0
            });
        }
    }
    #[test]
    fn native_prepared_media_error_alone_retains_actual_host_after_all_frames_and_sessions() {
        fn exception<'a>(
            error: &'a (dyn std::error::Error + 'static),
        ) -> Option<&'a safemlx::error::Exception> {
            let mut current = Some(error);
            while let Some(error) = current {
                if let Some(value) = error.downcast_ref() {
                    return Some(value);
                }
                current = error.source();
            }
            None
        }
        let root = tempfile::tempdir().unwrap();
        fixture(root.path(), 0);
        for mode in 0..3 {
            let baseline = run_plan(root.path(), 0, mode, Some(2), None, true, Some(1), true);
            let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
            let pool = eredu_runtime::working_memory::WorkingMemoryPool::new(u64::MAX, 0).unwrap();
            let backend = MlxBackend::new(&stream, &stream).with_memory_pool(pool.clone());
            let model = load_model(&backend, root.path(), weights(mode)).unwrap();
            let mut runtime = ModelRuntime::from_prepared(backend, model).unwrap();
            let config = TextGenerationConfig::new(
                eredu_core::resolve_generation_config(
                    None,
                    eredu_core::GenerationConfigOverrides {
                        do_sample: Some(false),
                        max_new_tokens: Some(4),
                        ..Default::default()
                    },
                )
                .unwrap(),
            );
            let input =
                MlxBackend::prepare_control_input(&runtime, prompt_length(0, Some(2), true))
                    .unwrap();
            let mut selected = plan(true);
            selected.selections.push(CaptureSelection {
                id: "failed global".into(),
                path: eredu_core::MODEL_LOGITS_OBSERVATION_PATH.into(),
                schedule: CaptureSchedule {
                    decode: false,
                    ..Default::default()
                },
                slices: vec![],
                transform: CaptureTransform::Summary,
            });
            let source = source_plan(&runtime, &input, selected);
            let input =
                MlxBackend::bind_control_input_capture(&runtime, input, config, source.clone())
                    .unwrap();
            let input = MlxBackend::consume_control_input(&runtime, input)
                .unwrap()
                .0;
            let native_error = Array::from_slice(&[1.0f32], &[1])
                .reshape(&[2], &stream)
                .unwrap_err();
            let expected = (native_error.what().to_owned(), native_error.location());
            let fault = crate::composition::mlx::fixture_bounded_capture::fail_transform_after(
                1,
                native_error,
            );
            let mut generation = eredu_core::ControlledTextGeneration::from_input(
                &mut runtime,
                eredu_core::TextGenerationInput::Prepared(input),
                config,
                AllowAllTokens,
            )
            .unwrap();
            generation.enable_prepared_capture(source).unwrap();
            let error = match generation.next() {
                Some(Err(error)) => error,
                _ => panic!("selected global failure must not emit a token"),
            };
            let native = exception(&error).expect("original native cause");
            assert_eq!(native.what(), expected.0);
            assert_eq!(native.location(), expected.1);
            assert!(generation.capture_pending());
            let frame = retained(
                generation
                    .take_captured_delivery()
                    .unwrap()
                    .expect("aborted logical frame"),
            );
            assert!(!generation.capture_pending());
            assert_eq!(frame.outcome, CaptureStepOutcome::Aborted);
            assert!(frame.records[0].payload.is_none());
            assert!(
                frame.cumulative_usage.host_bytes > 0 && frame.cumulative_usage.retained_bytes > 0
            );
            assert!(matches!(
                frame.records[0].outcome,
                CaptureOutcome::Failed {
                    reason: CaptureFailureReason::Native,
                    ..
                }
            ));
            drop(fault);
            drop(generation);
            crate::backend::submission_recovery::wait_for_retirement(|| {
                runtime
                    .session_mut()
                    .neutral_prediction_target_mut()
                    .is_ok()
            });
            let state = snapshot(runtime.session_mut());
            same_arrays(&state.0, &baseline.state.0);
            fixed_equal(&state.1, &baseline.state.1);
            assert_eq!(
                exception(&error).unwrap().what(),
                expected.0,
                "escaped error survives session observation"
            );
            // Error custody is independent of every completed/failed frame.
            drop(frame);
            drop(runtime);
            drop(state);
            stream.synchronize().unwrap();
            crate::backend::submission_recovery::wait_for_retirement(|| {
                safemlx::try_retire_completed_submissions();
                safemlx::reclaim_allocation_owners();
                pool.unquoted_owner_count().unwrap() == 1
            });
            assert_eq!(exception(&error).unwrap().what(), expected.0);
            assert_eq!(exception(&error).unwrap().location(), expected.1);
            drop(error);
            crate::backend::submission_recovery::wait_for_retirement(|| {
                safemlx::try_retire_completed_submissions();
                safemlx::reclaim_allocation_owners();
                pool.unquoted_owner_count().unwrap() == 0
            });
        }
    }
}
