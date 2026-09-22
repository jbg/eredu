use super::*;
use std::mem::size_of;
fn point(path: &str, batch_axis: bool, width: usize) -> ObservationPoint {
    let mut axes = vec![];
    if batch_axis {
        axes.push(TensorAxis {
            name: "batch".into(),
            dimension: SymbolicDimension::Batch,
        });
    }
    axes.push(TensorAxis {
        name: "sequence".into(),
        dimension: SymbolicDimension::Sequence,
    });
    axes.push(TensorAxis {
        name: "hidden".into(),
        dimension: SymbolicDimension::Known(width),
    });
    ObservationPoint {
        path: path.into(),
        node_id: path.into(),
        meaning: "causal decoder rows".into(),
        value_type: ObservationValueType::Tensor,
        dtype: ObservationDtype::Floating,
        axes: Some(axes),
        prefill: true,
        decode: true,
        requirements: vec![ObservationRequirement::ActivationHooks],
        position: ObservationPosition::BeforeIntervention,
        retained_bytes: None,
        host_bytes: None,
    }
}
fn bind(
    transforms: &[CaptureTransform],
    n: u64,
    chunk: u64,
    batch: u64,
    batch_axis: bool,
    slices: &[CaptureSlice],
    limits: CaptureLimits,
) -> OrdinaryPrefillCapture {
    let candidate = transforms.iter().any(|t| {
        matches!(
            t,
            CaptureTransform::TopCandidates { .. } | CaptureTransform::TokenScores { .. }
        )
    });
    let path = if candidate {
        MODEL_LOGITS_OBSERVATION_PATH
    } else {
        "block.output"
    };
    let point = point(path, batch_axis, 4);
    let caps = CaptureCapabilities {
        transformations: vec![
            CaptureTransformKind::Summary,
            CaptureTransformKind::Histogram,
            CaptureTransformKind::Preview,
            CaptureTransformKind::TopCandidates,
            CaptureTransformKind::TokenScores,
            CaptureTransformKind::FullTensor,
        ],
        max_histogram_bins: 128,
        conditions: vec![],
    };
    let support = ObservationSupportReport {
        schema_version: 1,
        capture: caps.clone(),
        points: vec![ObservationSupport {
            path: path.into(),
            prefill: ObservationSupportStatus::Supported,
            decode: ObservationSupportStatus::Supported,
            floating_to_f32: true,
        }],
    };
    let catalog = ObservationCatalog {
        schema_version: 1,
        points: vec![point],
        completeness: DescriptionCompleteness::Complete,
    };
    let plan = CapturePlan {
        schema_version: 1,
        selections: transforms
            .iter()
            .enumerate()
            .map(|(i, t)| CaptureSelection {
                id: format!("selection{i}"),
                path: path.into(),
                schedule: CaptureSchedule {
                    decode: false,
                    ..Default::default()
                },
                slices: slices.to_vec(),
                transform: t.clone(),
            })
            .collect(),
        limits,
    };
    let source = SharedCapturePlan::new(
        plan.admit_with_text_origin(
            &catalog,
            &support,
            &caps,
            CaptureRequestShape {
                batch,
                prompt_tokens: n,
                max_predictions: 3,
            },
            CaptureTextOrigin {
                cached_positions: 7,
            },
        )
        .unwrap(),
    );
    let mut paths = source_paths();
    Arc::get_mut(&mut paths.0).unwrap().media_prefill =
        vec![PrefillObservationDeclaration::prepared_media_decoder(
            path.into(),
            usize::from(batch_axis),
            if candidate {
                PrefillReadoutStage::VocabularyScores
            } else {
                PrefillReadoutStage::BeforeReadout
            },
        )]
        .into_boxed_slice();
    let selected = paths.prepare_media_capture_selection(&source).unwrap();
    let output = selected.physical_output(OutputDemand::LastPosition);
    OrdinaryPrefillCapture::new(
        selected,
        InferenceGeometry {
            batch_size: batch,
            cached_positions: 7,
            input_positions: n,
            max_output_tokens: 3,
            prefill_chunk_positions: chunk,
            output,
        },
    )
    .unwrap()
}
#[test]
fn prepared_media_preview_keeps_bounded_transform_and_tensor_fragment_views() {
    use crate::capture::CapturePrefillObservationPolicy;
    for maximum in [0, 7] {
        let binding = bind(
            &[CaptureTransform::Preview {
                max_elements: maximum,
            }],
            5,
            2,
            2,
            true,
            &[],
            limits(),
        );
        let mut paths = source_paths();
        Arc::get_mut(&mut paths.0).unwrap().media_prefill =
            vec![PrefillObservationDeclaration::prepared_media_decoder(
                "block.output".into(),
                1,
                PrefillReadoutStage::BeforeReadout,
            )]
            .into_boxed_slice();
        let selected = paths
            .prepare_media_capture_selection(binding.source())
            .unwrap();
        let policy = CapturePrefillObservationPolicy::from_bound(
            selected.bind_geometry(binding.geometry()).unwrap(),
        )
        .unwrap();
        let row = policy.row(0).unwrap();
        assert!(row.transform_plan().is_some());
        let assembly = row.assembly().unwrap();
        assert_eq!(assembly.logical_geometry().elements(), maximum as usize);
        let mut destinations = Vec::new();
        for chunk in 0..assembly.chunk_count() {
            let fragment = assembly.fragment(chunk).unwrap();
            destinations.extend(
                (0..fragment.selected_elements())
                    .filter_map(|index| fragment.mapping_at(index))
                    .map(|mapping| mapping.destination_index()),
            );
        }
        destinations.sort_unstable();
        assert_eq!(destinations, (0..maximum as usize).collect::<Vec<_>>());
    }
}
fn source_paths() -> SharedLayeredObservationPaths {
    super::super::source()
}
fn tensor(
    start: u64,
    end: u64,
    batch: u64,
    batch_axis: bool,
    terminal_only: bool,
) -> TensorObservation {
    let rows = if terminal_only {
        end - 1..end
    } else {
        start..end
    };
    let mut shape = vec![];
    if batch_axis {
        shape.push(batch as usize);
    }
    shape.extend([(rows.end - rows.start) as usize, 4]);
    let values = (0..batch)
        .flat_map(|b| {
            rows.clone()
                .flat_map(move |r| (0..4).map(move |c| (1000 * b + 10 * r + c + 1) as f32))
        })
        .collect();
    TensorObservation::new(shape, TensorObservationData::F32(values)).unwrap()
}
fn with_nonfinite(tensor: TensorObservation, enabled: bool) -> TensorObservation {
    if !enabled {
        return tensor;
    }
    let (shape, data) = tensor.into_parts();
    let TensorObservationData::F32(mut values) = data else {
        unreachable!()
    };
    for value in &mut values {
        *value = match *value {
            1. => f32::NAN,
            12. => f32::INFINITY,
            23. => f32::NEG_INFINITY,
            other => other,
        };
    }
    TensorObservation::new(shape, TensorObservationData::F32(values)).unwrap()
}
fn values(t: &TensorObservation) -> &[f32] {
    let TensorObservationData::F32(v) = t.data() else {
        panic!("f32")
    };
    v
}
fn selected(t: &TensorObservation, s: &ResolvedCaptureSlice) -> Vec<f32> {
    // Independent coordinate filtering over the full source flatten order.
    values(t)
        .iter()
        .enumerate()
        .filter_map(|(flat, &value)| {
            let mut remainder = flat;
            for axis in (0..t.shape().len()).rev() {
                let coordinate = remainder % t.shape()[axis];
                remainder /= t.shape()[axis];
                if coordinate < (s.starts[axis] as usize)
                    || coordinate >= s.ends[axis] as usize
                    || (coordinate - s.starts[axis] as usize) % (s.strides[axis] as usize) != 0
                {
                    return None;
                }
            }
            Some(value)
        })
        .collect()
}
fn result(t: &CaptureTransform, v: &[f32]) -> CapturePayload {
    match t {
        CaptureTransform::Summary => {
            let finite = v
                .iter()
                .copied()
                .filter(|v| v.is_finite())
                .collect::<Vec<_>>();
            CapturePayload::Summary(CaptureSummary {
                elements: v.len() as u64,
                finite: finite.len() as u64,
                non_finite: (v.len() - finite.len()) as u64,
                nan: v.iter().filter(|v| v.is_nan()).count() as u64,
                positive_infinity: v.iter().filter(|&&v| v == f32::INFINITY).count() as u64,
                negative_infinity: v.iter().filter(|&&v| v == f32::NEG_INFINITY).count() as u64,
                min: finite.iter().copied().reduce(f32::min).map(f64::from),
                max: finite.iter().copied().reduce(f32::max).map(f64::from),
                mean: (!finite.is_empty())
                    .then(|| finite.iter().map(|&v| v as f64).sum::<f64>() / finite.len() as f64),
                rms: (!finite.is_empty()).then(|| {
                    (finite.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / finite.len() as f64)
                        .sqrt()
                }),
            })
        }
        CaptureTransform::Histogram { edges } => {
            let mut h = CaptureHistogram {
                edges: edges.clone(),
                counts: vec![0; edges.len() - 1],
                below: 0,
                above: 0,
                non_finite: 0,
            };
            for &v in v {
                if !v.is_finite() {
                    h.non_finite += 1;
                } else if v < edges[0] {
                    h.below += 1;
                } else if v > *edges.last().unwrap() {
                    h.above += 1;
                } else {
                    let index = edges
                        .windows(2)
                        .position(|e| v >= e[0] && v < e[1])
                        .unwrap_or(edges.len() - 2);
                    h.counts[index] += 1;
                }
            }
            CapturePayload::Histogram(h)
        }
        CaptureTransform::Preview { max_elements } => {
            let count = v.len().min(*max_elements as usize);
            CapturePayload::Tensor(
                TensorObservation::new(
                    vec![count],
                    TensorObservationData::F32(v[..count].to_vec()),
                )
                .unwrap(),
            )
        }
        _ => panic!("separate candidates"),
    }
}
#[derive(Default)]
struct TransformBackend {
    calls: Vec<(u64, u64, u64)>,
    candidate_rows: Vec<usize>,
    validations: Cell<usize>,
    nonfinite: bool,
    bad_dtype: Option<usize>,
    fail: Option<usize>,
}
impl CaptureBackend for TransformBackend {
    type Tensor = TensorObservation;
    type Error = Failure;
    fn shape(&self, t: &Self::Tensor) -> Result<Vec<u64>, Failure> {
        Ok(t.shape().iter().map(|&n| n as u64).collect())
    }
    fn source_dtype(&self, _: &Self::Tensor) -> Option<TensorDtype> {
        Some(TensorDtype::F32)
    }
    fn estimate(
        &self,
        _: &Self::Tensor,
        _: &CaptureSelection,
        _: &ResolvedCaptureSlice,
    ) -> Result<CaptureUsage, CaptureError> {
        Ok(CaptureUsage::default())
    }
    fn transform(
        &mut self,
        t: &Self::Tensor,
        s: &CaptureSelection,
        slice: &ResolvedCaptureSlice,
    ) -> Result<CapturePayload, Failure> {
        Ok(result(&s.transform, &selected(t, slice)))
    }
    fn validate_capture_prefill_transform_source(
        &self,
        t: &Self::Tensor,
        f: &CapturePrefillTransformFragment<'_, '_>,
    ) -> Option<Result<TensorDtype, Failure>> {
        self.validations.set(self.validations.get() + 1);
        Some(
            if t.shape()
                .iter()
                .map(|&n| n as u64)
                .eq(f.source_shape().iter().copied())
            {
                Ok(if self.bad_dtype == Some(f.chunk_index() as usize) {
                    TensorDtype::I32
                } else {
                    TensorDtype::F32
                })
            } else {
                Err(Failure)
            },
        )
    }
    fn estimate_capture_prefill_transform(
        &self,
        f: &CapturePrefillTransformFragment<'_, '_>,
    ) -> Result<CaptureUsage, CaptureError> {
        Ok(CaptureUsage {
            captures: 1,
            host_bytes: 4 * f.selected_elements() + 32 * f.source_shape().len() as u64,
            retained_bytes: 8 * f.selected_elements(),
            encoded_bytes: 99,
        })
    }
    fn capture_prefill_transform(
        &mut self,
        t: &Self::Tensor,
        f: &CapturePrefillTransformFragment<'_, '_>,
    ) -> Option<Result<CapturePayload, Failure>> {
        self.calls
            .push((f.input().start, f.input().end, f.selected_elements()));
        Some(if self.fail == Some(f.chunk_index() as usize) {
            Err(Failure)
        } else {
            let slice = ResolvedCaptureSlice {
                starts: f.starts().to_vec(),
                ends: f.ends().to_vec(),
                strides: f.strides().to_vec(),
                shape: f.selected_shape().to_vec(),
            };
            Ok(result(
                &f.plan().selection().transform,
                &selected(t, &slice),
            ))
        })
    }
    fn estimate_capture_prefill_candidates(
        &self,
        g: &CaptureCandidateGeometry<'_>,
    ) -> Result<CaptureUsage, CaptureError> {
        Ok(CaptureUsage {
            captures: 1,
            host_bytes: (g.count() * size_of::<CaptureCandidate>()) as u64,
            retained_bytes: 32 * g.vocabulary() as u64,
            encoded_bytes: 1024,
        })
    }
    fn capture_prefill_candidates(
        &mut self,
        t: &Self::Tensor,
        g: &CaptureCandidateGeometry<'_>,
    ) -> Option<Result<CaptureCandidates, Failure>> {
        self.candidate_rows.push(t.shape()[1]);
        if t.shape() != g.source_shape() {
            return Some(Err(Failure));
        }
        let v = &values(t)[values(t).len() - g.vocabulary()..];
        let mut ids = (0..v.len()).collect::<Vec<_>>();
        ids.sort_by(|&a, &b| v[b].total_cmp(&v[a]));
        Some(Ok(CaptureCandidates {
            stage: CandidateScoreStage::RawLogitsBeforeSampling,
            source: CandidateLogitsSource::Original,
            candidates: ids[..g.count()]
                .iter()
                .map(|&i| CaptureCandidate {
                    token_id: i as u32,
                    score: v[i],
                    allowed: i % 2 == 1,
                })
                .collect(),
            domain: Some(CandidateDomain {
                allowed_tokens: 2,
                vocabulary: 4,
                constrained: true,
            }),
        }))
    }
}
fn run_global(
    binding: OrdinaryPrefillCapture,
    backend: &mut TransformBackend,
    batch_axis: bool,
    omit: Option<usize>,
    cancel: Option<usize>,
    finish: bool,
) -> (CaptureSession, bool) {
    run_global_host(
        binding,
        backend,
        batch_axis,
        omit,
        cancel,
        finish,
        &HostPreparationAuthority::default(),
    )
}
fn run_global_host(
    binding: OrdinaryPrefillCapture,
    backend: &mut TransformBackend,
    batch_axis: bool,
    omit: Option<usize>,
    cancel: Option<usize>,
    finish: bool,
    host: &HostPreparationAuthority,
) -> (CaptureSession, bool) {
    let g = binding.geometry();
    let path = binding.source().admission().plan().selections[0]
        .path
        .clone();
    let mut session = CaptureSession::with_ordinary_prefill(binding, host).unwrap();
    let mut epoch = DistributedCommitEpoch::FIRST;
    let mut ok = true;
    for i in 0..g.input_positions.div_ceil(g.prefill_chunk_positions) {
        if cancel == Some(i as usize) {
            ok = false;
            break;
        }
        let start = i * g.prefill_chunk_positions;
        let end = (start + g.prefill_chunk_positions).min(g.input_positions);
        let output = g.output.for_chunk(end == g.input_positions);
        let chunk = PrefillChunk {
            input: start..end,
            position: g.cached_positions + start,
            output,
        };
        session.begin_ordinary_prefill(&chunk).unwrap();
        if session
            .prepare_ordinary_prefill(epoch, crate::ExpertPass::Prefill)
            .is_err()
        {
            ok = false;
            break;
        }
        if omit != Some(i as usize) && (path != "model.logits" || output != OutputDemand::StateOnly)
        {
            let observed = with_nonfinite(
                tensor(
                    start,
                    end,
                    g.batch_size,
                    batch_axis,
                    path == MODEL_LOGITS_OBSERVATION_PATH && output == OutputDemand::LastPosition,
                ),
                backend.nonfinite,
            );
            if session.observe(backend, &path, &observed).is_err() {
                session.finish_ordinary_chunk(epoch, false);
                ok = false;
                break;
            }
        }
        if session.complete_ordinary_prefill(epoch).is_err() {
            session.finish_ordinary_chunk(epoch, false);
            ok = false;
            break;
        }
        assert!(session.take_test_frame().is_none());
        session.finish_ordinary_chunk(epoch, true);
        assert!(session.take_test_frame().is_none());
        epoch = epoch.next().unwrap();
    }
    session.finish_ordinary_prefill(ok && finish);
    (session, ok)
}
fn assert_payload(actual: &CapturePayload, expected: &CapturePayload) {
    match (actual, expected) {
        (CapturePayload::Summary(a), CapturePayload::Summary(b)) => {
            assert_eq!(
                (
                    a.elements,
                    a.finite,
                    a.non_finite,
                    a.nan,
                    a.positive_infinity,
                    a.negative_infinity,
                    a.min,
                    a.max
                ),
                (
                    b.elements,
                    b.finite,
                    b.non_finite,
                    b.nan,
                    b.positive_infinity,
                    b.negative_infinity,
                    b.min,
                    b.max
                )
            );
            for (a, b) in [(a.mean, b.mean), (a.rms, b.rms)] {
                match (a, b) {
                    (Some(a), Some(b)) => assert!((a - b).abs() < 1e-10),
                    (None, None) => {}
                    _ => panic!("finite statistics"),
                }
            }
        }
        _ if actual.as_tensor().is_some() => assert_eq!(actual.as_tensor(), expected.as_tensor()),
        _ => assert_eq!(actual, expected),
    }
}
#[test]
fn media_global_reductions_and_prefix_match_whole_selected_nonleading_rows() {
    for n in [5, 6] {
        for batch_axis in [false, true] {
            let batch = if batch_axis { 2 } else { 1 };
            for transform in [
                CaptureTransform::Summary,
                CaptureTransform::Histogram {
                    edges: vec![0., 12., 34., 1002., 1054.],
                },
                CaptureTransform::Preview { max_elements: 0 },
                CaptureTransform::Preview { max_elements: 7 },
                CaptureTransform::Preview { max_elements: 100 },
            ] {
                for zero in [false, true] {
                    let slices = vec![
                        CaptureSlice {
                            axis: "sequence".into(),
                            start: 1,
                            end: if zero { 1 } else { n },
                            stride: 2,
                        },
                        CaptureSlice {
                            axis: "hidden".into(),
                            start: 0,
                            end: 4,
                            stride: 2,
                        },
                    ];
                    let binding = bind(
                        &[transform.clone()],
                        n,
                        2,
                        batch,
                        batch_axis,
                        &slices,
                        limits(),
                    );
                    let point = &binding.source().admission().points()[0];
                    let selection = &binding.source().admission().plan().selections[0];
                    let whole = tensor(0, n, batch, batch_axis, false);
                    let slice = resolve_slice(
                        point,
                        selection,
                        &whole.shape().iter().map(|&n| n as u64).collect::<Vec<_>>(),
                    )
                    .unwrap();
                    let selected = selected(&whole, &slice);
                    let expected = result(&transform, &selected);
                    let mut backend = TransformBackend::default();
                    let (mut session, ok) =
                        run_global(binding, &mut backend, batch_axis, None, None, true);
                    assert!(ok);
                    let step = session.take_test_frame().unwrap();
                    assert_eq!(step.outcome, CaptureStepOutcome::Committed);
                    assert_eq!(step.records.len(), 1);
                    assert_payload(step.records[0].payload.as_ref().unwrap(), &expected);
                    let emitted = expected
                        .as_tensor()
                        .map(|tensor| tensor.shape().iter().product::<usize>())
                        .unwrap_or(selected.len());
                    let expected_outcome = if emitted < selected.len() {
                        CaptureOutcome::Truncated {
                            available_elements: selected.len() as u64,
                            emitted_elements: emitted as u64,
                        }
                    } else {
                        CaptureOutcome::Captured
                    };
                    assert_eq!(step.records[0].outcome, expected_outcome);
                    assert_eq!(backend.validations.get(), n.div_ceil(2) as usize);
                    if zero {
                        assert!(backend.calls.is_empty());
                    }
                }
            }
        }
    }
}
#[test]
fn media_global_terminal_candidates_preserve_domain_without_extra_sequence_readout() {
    for mixed in [false, true] {
        let mut transforms = vec![CaptureTransform::TopCandidates { count: 3 }];
        if mixed {
            transforms.push(CaptureTransform::Preview { max_elements: 5 });
        }
        let binding = bind(&transforms, 6, 2, 1, true, &[], limits());
        assert_eq!(
            binding.geometry().output,
            if mixed {
                OutputDemand::Sequence
            } else {
                OutputDemand::LastPosition
            }
        );
        let mut backend = TransformBackend::default();
        let (mut session, ok) = run_global(binding, &mut backend, true, None, None, true);
        assert!(ok);
        assert_eq!(backend.candidate_rows, [if mixed { 2 } else { 1 }]);
        let step = session.take_test_frame().unwrap();
        let CapturePayload::Candidates(c) = step.records[0].payload.as_ref().unwrap() else {
            panic!("candidates")
        };
        assert_eq!(
            c.candidates
                .iter()
                .map(|c| (c.token_id, c.score, c.allowed))
                .collect::<Vec<_>>(),
            [(3, 54., true), (2, 53., false), (1, 52., true)]
        );
        assert_eq!(
            c.domain,
            Some(CandidateDomain {
                allowed_tokens: 2,
                vocabulary: 4,
                constrained: true
            })
        );
        if mixed {
            assert_eq!(
                backend.calls.iter().map(|c| (c.0, c.1)).collect::<Vec<_>>(),
                [(0, 2), (2, 4), (4, 6)]
            );
        }
    }
}
#[test]
fn media_global_failure_cancel_missing_hook_and_final_failure_never_publish_success() {
    for transform in [
        CaptureTransform::Summary,
        CaptureTransform::Histogram {
            edges: vec![0., 20., 60.],
        },
        CaptureTransform::Preview { max_elements: 7 },
    ] {
        for fault in 0..6 {
            let binding = bind(&[transform.clone()], 6, 2, 1, false, &[], limits());
            let mut backend = TransformBackend::default();
            backend.fail = (fault == 0).then_some(1);
            backend.bad_dtype = (fault == 1).then_some(2);
            let (mut session, ok) = run_global(
                binding,
                &mut backend,
                false,
                (fault == 2).then_some(1),
                (fault == 3).then_some(1),
                fault != 4,
            );
            if fault == 5 {
                assert!(ok);
            } else if fault != 4 {
                assert!(!ok);
            }
            let step = session.take_test_frame().unwrap();
            assert_eq!(
                step.outcome,
                if fault == 5 {
                    CaptureStepOutcome::Committed
                } else {
                    CaptureStepOutcome::Aborted
                }
            );
            if fault != 5 {
                assert!(step.records[0].payload.is_none());
            }
            assert!(session.cumulative_usage().captures >= 1);
        }
    }
}

#[test]
fn media_global_full_allowance_exact_short_skip_and_nonrefund() {
    for transform in [
        CaptureTransform::Summary,
        CaptureTransform::Histogram {
            edges: vec![0., 20., 60.],
        },
        CaptureTransform::Preview { max_elements: 7 },
        CaptureTransform::TopCandidates { count: 3 },
    ] {
        let candidate = matches!(transform, CaptureTransform::TopCandidates { .. });
        let (mut complete, ok) = run_global(
            bind(&[transform.clone()], 6, 2, 1, candidate, &[], limits()),
            &mut TransformBackend::default(),
            candidate,
            None,
            None,
            true,
        );
        assert!(ok);
        let charged = complete.take_test_frame().unwrap().cumulative_usage;
        assert_eq!(
            complete.cumulative_usage(),
            charged,
            "draining never refunds"
        );
        let mut exact = limits();
        exact.per_step = charged;
        exact.cumulative = charged;
        let (mut full, ok) = run_global(
            bind(&[transform.clone()], 6, 2, 1, candidate, &[], exact.clone()),
            &mut TransformBackend::default(),
            candidate,
            None,
            None,
            true,
        );
        assert!(ok);
        assert_eq!(full.take_test_frame().unwrap().cumulative_usage, charged);
        for dimension in 0..4 {
            for skip in [false, true] {
                let mut short = exact.clone();
                for usage in [&mut short.per_step, &mut short.cumulative] {
                    let field = match dimension {
                        0 => &mut usage.captures,
                        1 => &mut usage.host_bytes,
                        2 => &mut usage.retained_bytes,
                        _ => &mut usage.encoded_bytes,
                    };
                    assert!(*field > 0);
                    *field -= 1;
                }
                if skip {
                    short.on_limit = CaptureLimitPolicy::Skip;
                }
                let mut backend = TransformBackend::default();
                let (mut run, ok) = run_global(
                    bind(&[transform.clone()], 6, 2, 1, candidate, &[], short),
                    &mut backend,
                    candidate,
                    None,
                    None,
                    true,
                );
                assert_eq!(ok, skip, "transform{transform:?} dimension{dimension}");
                assert!(backend.calls.is_empty() && backend.candidate_rows.is_empty());
                if skip {
                    assert!(matches!(
                        run.take_test_frame().unwrap().records[0].outcome,
                        CaptureOutcome::Skipped {
                            reason: CaptureSkipReason::Limit { .. }
                        }
                    ));
                }
            }
        }
        let (mut rejected, ok) = run_global(
            bind(&[transform.clone()], 6, 2, 1, candidate, &[], exact),
            &mut TransformBackend::default(),
            candidate,
            None,
            None,
            false,
        );
        assert!(ok);
        assert_eq!(rejected.cumulative_usage(), charged);
        let step = rejected.take_test_frame().unwrap();
        assert_eq!(step.outcome, CaptureStepOutcome::Aborted);
        assert!(step.records[0].payload.is_none());
        assert_eq!(rejected.cumulative_usage(), charged);
    }
}

#[test]
fn media_global_no_overlap_still_validates_precision_and_cancelled_candidates_never_run() {
    for transform in [
        CaptureTransform::Summary,
        CaptureTransform::Histogram {
            edges: vec![0., 60.],
        },
        CaptureTransform::Preview { max_elements: 0 },
    ] {
        let slices = [CaptureSlice {
            axis: "sequence".into(),
            start: 4,
            end: 6,
            stride: 1,
        }];
        let mut backend = TransformBackend {
            bad_dtype: Some(0),
            ..Default::default()
        };
        let (mut run, ok) = run_global(
            bind(&[transform], 6, 2, 1, false, &slices, limits()),
            &mut backend,
            false,
            None,
            None,
            true,
        );
        assert!(!ok);
        assert_eq!(backend.validations.get(), 1);
        assert!(backend.calls.is_empty());
        assert_eq!(
            run.take_test_frame().unwrap().outcome,
            CaptureStepOutcome::Aborted
        );
    }
    for boundary in [0, 1, 2] {
        let mut backend = TransformBackend::default();
        let (mut run, ok) = run_global(
            bind(
                &[CaptureTransform::TopCandidates { count: 4 }],
                6,
                2,
                1,
                true,
                &[],
                limits(),
            ),
            &mut backend,
            true,
            None,
            Some(boundary),
            true,
        );
        assert!(!ok);
        assert!(backend.candidate_rows.is_empty());
        if boundary == 0 {
            assert!(run.take_test_frame().is_none());
        } else {
            let step = run.take_test_frame().unwrap();
            assert_eq!(step.outcome, CaptureStepOutcome::Aborted);
            assert!(step.records[0].payload.is_none());
        }
    }
}

#[test]
fn media_global_nonfinite_reductions_match_whole_source_counts_and_moments() {
    for transform in [
        CaptureTransform::Summary,
        CaptureTransform::Histogram {
            edges: vec![0., 20., 40., 60.],
        },
    ] {
        let expected = result(
            &transform,
            values(&with_nonfinite(tensor(0, 6, 1, false, false), true)),
        );
        for chunk in [1, 2, 6] {
            let mut backend = TransformBackend {
                nonfinite: true,
                ..Default::default()
            };
            let (mut run, ok) = run_global(
                bind(&[transform.clone()], 6, chunk, 1, false, &[], limits()),
                &mut backend,
                false,
                None,
                None,
                true,
            );
            assert!(ok);
            let frame = run.take_test_frame().unwrap();
            assert_payload(frame.records[0].payload.as_ref().unwrap(), &expected);
        }
    }
}

#[test]
fn media_global_each_owned_payload_and_failed_diagnostic_keeps_frame_host_after_session() {
    struct Retired(Arc<AtomicUsize>);
    impl Drop for Retired {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    for transform in [
        CaptureTransform::Summary,
        CaptureTransform::Histogram {
            edges: vec![0., 12., 99.],
        },
        CaptureTransform::Preview { max_elements: 0 },
        CaptureTransform::TopCandidates { count: 2 },
    ] {
        for fail in [false, true] {
            let retired = Arc::new(AtomicUsize::new(0));
            let host = HostPreparationAuthority::retain(Retired(retired.clone()));
            let candidate = matches!(transform, CaptureTransform::TopCandidates { .. });
            let binding = bind(&[transform.clone()], 5, 2, 1, candidate, &[], limits());
            let mut backend = TransformBackend {
                fail: if fail { Some(0) } else { None },
                ..Default::default()
            };
            let (mut session, ok) =
                run_global_host(binding, &mut backend, candidate, None, None, true, &host);
            // Candidate extraction uses its own backend method; its standalone
            // native error path is covered by the real native failure test.
            if !candidate {
                assert_eq!(ok, !fail)
            }
            let frame = session.take_test_frame().unwrap();
            if !ok {
                assert!(
                    matches!(&frame.records[0].outcome,CaptureOutcome::Failed{message,..}if !message.is_empty())
                )
            } else {
                assert!(frame.records[0].payload.is_some())
            }
            let alias = frame.0.clone();
            drop(frame);
            drop((session, host));
            assert_eq!(retired.load(Ordering::SeqCst), 0);
            assert!(!alias.records()[0].path.is_empty());
            drop(alias);
            assert_eq!(retired.load(Ordering::SeqCst), 1);
        }
    }
}

#[test]
fn selected_token_scores_bind_terminal_readout_but_full_capture_keeps_sequence() {
    for prompt in [5, 6] {
        for mixed in [false, true] {
            let mut transforms = vec![CaptureTransform::TokenScores {
                token_ids: vec![2, 0],
            }];
            if mixed {
                transforms.push(CaptureTransform::FullTensor);
            }
            let binding = bind(&transforms, prompt, 2, 1, true, &[], limits());
            assert_eq!(
                binding.geometry().output,
                if mixed {
                    OutputDemand::Sequence
                } else {
                    OutputDemand::LastPosition
                }
            );
            // This is the same architecture-owned selection used before vocabulary
            // projection; the native fixture separately exercises both executions.
            let policy = crate::capture::CapturePrefillObservationPolicy::new(
                binding.source(),
                binding.geometry(),
            )
            .unwrap();
            let row = policy.row(0).unwrap();
            assert!(row.token_scores().is_some());
            assert!(row.assembly().is_none());
        }
    }
}
