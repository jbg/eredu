use super::*;

fn selection(transform: CaptureTransform) -> CaptureSelection {
    CaptureSelection {
        id: "test".into(),
        path: "test".into(),
        schedule: CaptureSchedule::default(),
        slices: vec![],
        transform,
    }
}

fn whole(shape: &[u64]) -> ResolvedCaptureSlice {
    ResolvedCaptureSlice {
        starts: vec![0; shape.len()],
        ends: shape.to_vec(),
        strides: vec![1; shape.len()],
        shape: shape.to_vec(),
    }
}

#[test]
fn bounded_native_summary_handles_empty_nonfinite_extremes_and_chunks() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let empty = Array::from_slice(&[] as &[f32], &[0]);
    let result = summary(&empty, &stream).unwrap();
    assert_eq!(result.elements, 0);
    assert_eq!(result.mean, None);
    assert_eq!(result.rms, None);
    let values = [1.0f32, 2.0, 3.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY];
    let result = summary(&Array::from_slice(&values, &[6]), &stream).unwrap();
    assert_eq!(
        (
            result.elements,
            result.finite,
            result.nan,
            result.positive_infinity,
            result.negative_infinity
        ),
        (6, 3, 1, 1, 1)
    );
    assert_eq!(result.non_finite, 3);
    assert_eq!(result.min, Some(1.0));
    assert_eq!(result.max, Some(3.0));
    assert!((result.mean.unwrap() - 2.0).abs() < 1e-6);
    assert!((result.rms.unwrap() - (14.0f64 / 3.0).sqrt()).abs() < 1e-6);
    let extremes = Array::from_slice(&[f32::MAX, -f32::MAX], &[2]);
    let result = summary(&extremes, &stream).unwrap();
    assert_eq!(result.mean, Some(0.0));
    assert_eq!(result.rms, Some(f64::from(f32::MAX)));
    let result = summary(
        &Array::from_slice(&[f32::NAN, f32::INFINITY], &[2]),
        &stream,
    )
    .unwrap();
    assert_eq!(result.min, None);
    assert_eq!(result.mean, None);
    let values: Vec<f32> = (0..3001).map(|n| n as f32 / 3000.0).collect();
    let result = summary(&Array::from_slice(&values, &[3001]), &stream).unwrap();
    let mean = values.iter().map(|x| f64::from(*x)).sum::<f64>() / values.len() as f64;
    assert!((result.mean.unwrap() - mean).abs() < 1e-6);
}

#[test]
fn bounded_native_histogram_edge_and_nonfinite_semantics() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let values = Array::from_slice(
        &[-1.0f32, 0.0, 0.5, 1.0, 2.0, 3.0, f32::NAN, f32::INFINITY],
        &[8],
    );
    let result = histogram(&values, &[0.0, 1.0, 2.0], &stream).unwrap();
    assert_eq!(result.counts, [2, 2]);
    assert_eq!((result.below, result.above, result.non_finite), (1, 1, 2));
}

#[test]
fn bounded_native_integer_boolean_statistics_and_empty_histograms() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let integers = Array::from_slice(&[-2i32, 0, 2, 4], &[4]);
    let result = summary(&integers, &stream).unwrap();
    assert_eq!(
        (result.elements, result.finite, result.non_finite),
        (4, 4, 0)
    );
    assert_eq!(
        (result.min, result.max, result.mean),
        (Some(-2.0), Some(4.0), Some(1.0))
    );
    assert!((result.rms.unwrap() - 6.0f64.sqrt()).abs() < 1e-6);
    assert_eq!(
        histogram(&integers, &[-2.0, 0.0, 4.0], &stream)
            .unwrap()
            .counts,
        [1, 3]
    );
    let booleans = Array::from_slice(&[true, false, true, false], &[4]);
    let result = summary(&booleans, &stream).unwrap();
    assert_eq!(result.mean, Some(0.5));
    let empty = Array::from_slice(&[] as &[f32], &[0]);
    let histogram = histogram(&empty, &[0.0, 1.0], &stream).unwrap();
    assert_eq!(histogram.counts, [0]);
    assert_eq!(
        (histogram.below, histogram.above, histogram.non_finite),
        (0, 0, 0)
    );
}

#[test]
fn bounded_native_slices_and_previews_preserve_large_integer_ids() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let source = MlxTensor::from_array(Array::from_slice(
        &[u64::MAX, 1, u64::MAX - 1, 3, u64::MAX - 2, 5],
        &[2, 3],
    ));
    let mut native = NativeCapture {
        stream: &stream,
        domain: None,
    };
    let slice = ResolvedCaptureSlice {
        starts: vec![0, 0],
        ends: vec![2, 3],
        strides: vec![1, 2],
        shape: vec![2, 2],
    };
    let CapturePayload::Tensor(tensor) = native
        .transform(&source, &selection(CaptureTransform::Slice), &slice)
        .unwrap()
    else {
        panic!()
    };
    assert_eq!(tensor.shape(), [2, 2]);
    assert_eq!(
        tensor.data(),
        &TensorObservationData::U64(vec![u64::MAX, u64::MAX - 1, 3, 5])
    );
    let CapturePayload::Tensor(tensor) = native
        .transform(
            &source,
            &selection(CaptureTransform::Preview { max_elements: 2 }),
            &whole(&[2, 3]),
        )
        .unwrap()
    else {
        panic!()
    };
    assert_eq!(
        tensor.data(),
        &TensorObservationData::U64(vec![u64::MAX, 1])
    );
    let CapturePayload::Tensor(tensor) = native
        .transform(
            &source,
            &selection(CaptureTransform::Preview { max_elements: 0 }),
            &whole(&[2, 3]),
        )
        .unwrap()
    else {
        panic!()
    };
    assert!(tensor.data().is_empty());
}

#[test]
fn bounded_native_estimate_scales_host_transfer_with_preview_or_reduction() {
    let request = whole(&[1, 1_000_000]);
    let full = estimate_shape(
        &[1, 1_000_000],
        &selection(CaptureTransform::FullTensor),
        &request,
    )
    .unwrap();
    let preview = estimate_shape(
        &[1, 1_000_000],
        &selection(CaptureTransform::Preview { max_elements: 8 }),
        &request,
    )
    .unwrap();
    let reduced = estimate_shape(
        &[1, 1_000_000],
        &selection(CaptureTransform::Summary),
        &request,
    )
    .unwrap();
    assert_eq!(preview.host_bytes, 128);
    assert!(reduced.host_bytes < full.host_bytes / 50);
    assert!(estimate_shape(
        &[u64::MAX, 2],
        &selection(CaptureTransform::Summary),
        &request
    )
    .is_err());
}

#[test]
fn bounded_native_candidate_estimate_charges_the_row_not_the_prefill_source() {
    let vocabulary = 262_144;
    let request = selection(CaptureTransform::TopCandidates { count: 16 });
    let decode = estimate_shape(&[1, 1, vocabulary], &request, &whole(&[1, 1, vocabulary])).unwrap();
    let prefill = estimate_shape(
        &[1, 4096, vocabulary],
        &request,
        &whole(&[1, 4096, vocabulary]),
    )
    .unwrap();
    assert_eq!(prefill.retained_bytes, decode.retained_bytes);
    assert!(prefill.retained_bytes < 8 * 1024 * 1024);
}

#[test]
fn bounded_native_candidates_use_last_prediction_raw_scores() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let tensor = MlxTensor::from_array(Array::from_slice(
        &[99.0f32, 100.0, 0.0, -1.0, 4.0, 2.0],
        &[1, 2, 3],
    ));
    let mut native = NativeCapture {
        stream: &stream,
        domain: None,
    };
    let request = selection(CaptureTransform::TopCandidates { count: 2 });
    let CapturePayload::Candidates(result) = native
        .transform(&tensor, &request, &whole(&[1, 2, 3]))
        .unwrap()
    else {
        panic!()
    };
    assert_eq!(result.stage, CandidateScoreStage::RawLogitsBeforeSampling);
    assert_eq!(result.source, CandidateLogitsSource::Original);
    assert_eq!(result.domain, None);
    assert_eq!(
        result.candidates,
        [
            CaptureCandidate {
                token_id: 1,
                score: 4.0,
                allowed: true,
            },
            CaptureCandidate {
                token_id: 2,
                score: 2.0,
                allowed: true,
            }
        ]
    );
    assert_eq!(
        native
            .estimate(&tensor, &request, &whole(&[1, 2, 3]))
            .unwrap()
            .host_bytes,
        120
    );
}

#[test]
fn bounded_native_transforms_never_materialize_the_full_source_for_a_preview_or_reduction() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let source = MlxTensor::from_array(Array::from_slice(&vec![1.0f32; 16384], &[128, 128]));
    let mut native = NativeCapture {
        stream: &stream,
        domain: None,
    };
    for transform in [
        CaptureTransform::Summary,
        CaptureTransform::Histogram {
            edges: vec![0.0, 1.0, 2.0],
        },
        CaptureTransform::Preview { max_elements: 16 },
    ] {
        HOST_READS.with(|reads| reads.set((0, 0)));
        native
            .transform(&source, &selection(transform), &whole(&[128, 128]))
            .unwrap();
        let (total, largest) = HOST_READS.with(std::cell::Cell::get);
        assert!(
            total < 1024,
            "a summary/preview must not copy a full 16384-value tensor"
        );
        assert!(largest <= 16, "hidden full-tensor host materialization");
    }
    HOST_READS.with(|reads| reads.set((0, 0)));
    native
        .transform(
            &source,
            &selection(CaptureTransform::FullTensor),
            &whole(&[128, 128]),
        )
        .unwrap();
    assert_eq!(HOST_READS.with(std::cell::Cell::get), (16384, 16384));
}

#[test]
fn original_and_effective_candidate_records_share_the_exact_domain() {
    use eredu_core::*;
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let validity = TokenFilter::allowed(vec![true, false, true, true]).unwrap();
    let filter = TokenFilter::allowed(vec![true, false, true, false]).unwrap();
    for (position, source, logits, expected_ids) in [
        (
            ObservationPosition::BeforeIntervention,
            CandidateLogitsSource::Original,
            [0.0f32, 8.0, 4.0, 1.0],
            [1, 2],
        ),
        (
            ObservationPosition::AfterIntervention,
            CandidateLogitsSource::Effective,
            [4.0f32, 0.0, 1.0, 8.0],
            [3, 0],
        ),
    ] {
        let path = MODEL_LOGITS_OBSERVATION_PATH.to_owned();
        let catalog = ObservationCatalog {
            schema_version: 1,
            completeness: DescriptionCompleteness::Complete,
            points: vec![ObservationPoint {
                path: path.clone(),
                node_id: "output".into(),
                meaning: "logits".into(),
                value_type: ObservationValueType::Tensor,
                dtype: ObservationDtype::Floating,
                axes: Some(vec![TensorAxis {
                    name: "vocabulary".into(),
                    dimension: SymbolicDimension::Known(4),
                }]),
                prefill: true,
                decode: true,
                requirements: vec![],
                position,
                retained_bytes: None,
                host_bytes: None,
            }],
        };
        let support = ObservationSupportReport {
            schema_version: 1,
            capture: capabilities(),
            points: vec![ObservationSupport {
                path: path.clone(),
                prefill: ObservationSupportStatus::Supported,
                decode: ObservationSupportStatus::Supported,
                floating_to_f32: true,
            }],
        };
        let usage = CaptureUsage {
            captures: 4,
            retained_bytes: 1 << 20,
            host_bytes: 1 << 20,
            encoded_bytes: 1 << 20,
        };
        let mut selected = selection(CaptureTransform::TopCandidates { count: 2 });
        selected.path = path.clone();
        let plan = CapturePlan {
            schema_version: CAPTURE_SCHEMA_VERSION,
            selections: vec![selected],
            limits: CaptureLimits {
                per_step: usage,
                cumulative: usage,
                physical_native_bytes: None,
                on_limit: CaptureLimitPolicy::Fail,
            },
        }
        .admit(
            &catalog,
            &support,
            &support.capture,
            CaptureRequestShape {
                batch: 1,
                prompt_tokens: 1,
                max_predictions: 1,
            },
        )
        .unwrap();
        let mut capture = eredu_runtime::capture::CaptureSession::new(plan);
        capture.begin_step(CapturePhase::Prefill, 0).unwrap();
        let tensor = MlxTensor::from_array(Array::from_slice(&logits, &[4]));
        let mut native = NativeCapture {
            stream: &stream,
            domain: Some(CaptureTokenDomain {
                filter: &filter,
                tokenizer_validity: &validity,
            }),
        };
        capture.observe(&mut native, &path, &tensor).unwrap();
        let step = capture.take_step().unwrap();
        let Some(CapturePayload::Candidates(result)) = &step.records[0].payload else {
            panic!()
        };
        assert_eq!(result.source, source);
        assert_eq!(
            result.domain,
            Some(CandidateDomain {
                allowed_tokens: 2,
                vocabulary: 4,
                constrained: true
            })
        );
        assert_eq!(
            result
                .candidates
                .iter()
                .map(|c| c.token_id)
                .collect::<Vec<_>>(),
            expected_ids
        );
        assert!(!result.candidates[0].allowed);
        assert!(result.candidates[1].allowed);
        assert_eq!(result.candidates[0].score, 8.0);
    }
}
