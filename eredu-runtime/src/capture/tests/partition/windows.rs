//! The real collector and window worker share the existing numerical backend.
use super::*;
use crate::inspection::{with_speculative_activation, SpeculativeActivationObserver};
use eredu_core::{intervention::*, speculative::*};
use std::{
    cell::{RefCell, RefMut},
    rc::Rc,
};

struct Borrowed<'a>(RefMut<'a, Backend>);
impl CaptureBackend for Borrowed<'_> {
    type Tensor = Value;
    type Error = std::io::Error;
    fn shape(&self, v: &Value) -> Result<Vec<u64>, Self::Error> {
        self.0.shape(v)
    }
    fn source_dtype(&self, v: &Value) -> Option<TensorDtype> {
        self.0.source_dtype(v)
    }
    fn estimate(
        &self,
        v: &Value,
        s: &CaptureSelection,
        r: &ResolvedCaptureSlice,
    ) -> Result<CaptureUsage, CaptureError> {
        self.0.estimate(v, s, r)
    }
    fn transform(
        &mut self,
        v: &Value,
        s: &CaptureSelection,
        r: &ResolvedCaptureSlice,
    ) -> Result<CapturePayload, Self::Error> {
        self.0.transform(v, s, r)
    }
}
fn unused<T>() -> Result<T, std::io::Error> {
    Err(std::io::Error::other("test has no intervention admission"))
}
impl InterventionBackend for Borrowed<'_> {
    fn intervention_dtype(&self, _: &Value) -> Result<InterventionDtype, std::io::Error> {
        unused()
    }
    fn validate_intervention_geometry(
        &self,
        _: &[u64],
        _: &ResolvedCaptureSlice,
    ) -> Result<(), CaptureError> {
        Err(CaptureError::Unsupported(
            "test has no intervention admission".into(),
        ))
    }
    fn select_region(
        &mut self,
        _: &Value,
        _: &ResolvedCaptureSlice,
    ) -> Result<Value, std::io::Error> {
        unused()
    }
    fn update_region(
        &mut self,
        _: &Value,
        _: &ResolvedCaptureSlice,
        _: &Value,
    ) -> Result<Value, std::io::Error> {
        unused()
    }
    fn zeros(&mut self, _: &[u64], _: InterventionDtype) -> Result<Value, std::io::Error> {
        unused()
    }
    fn scale(&mut self, _: &Value, _: f32) -> Result<Value, std::io::Error> {
        unused()
    }
    fn fill_masked(&mut self, _: &Value, _: &[bool], _: f32) -> Result<Value, std::io::Error> {
        unused()
    }
    fn mask_components(&mut self, _: &Value, _: &[u32], _: bool) -> Result<Value, std::io::Error> {
        unused()
    }
    fn realize_tensor(&mut self, _: &InterventionTensor) -> Result<Value, std::io::Error> {
        unused()
    }
    fn add(&mut self, _: &Value, _: &Value) -> Result<Value, std::io::Error> {
        unused()
    }
    fn fill_columns(&mut self, _: &Value, _: &[u32], _: f32) -> Result<Value, std::io::Error> {
        unused()
    }
}
struct Provider(Rc<RefCell<Backend>>);
impl CaptureBackendProvider for Provider {
    type Tensor = Value;
    type Error = std::io::Error;
    type Backend<'a> = Borrowed<'a>;
    fn backend(&mut self) -> Borrowed<'_> {
        Borrowed(self.0.borrow_mut())
    }
}
fn map(e: &CaptureExecutionError<std::io::Error>) -> String {
    e.to_string()
}
type Observer =
    SpeculativeCaptureObserver<Provider, fn(&CaptureExecutionError<std::io::Error>) -> String>;

fn collector(
    total: u64,
    shifted: bool,
    limit: u64,
    policy: CaptureLimitPolicy,
    stride: bool,
) -> (Observer, Rc<RefCell<Backend>>, CaptureDiscovery) {
    collector_limits(total, shifted, limit, policy, stride, None)
}
fn collector_limits(
    total: u64,
    shifted: bool,
    limit: u64,
    policy: CaptureLimitPolicy,
    stride: bool,
    host: Option<u64>,
) -> (Observer, Rc<RefCell<Backend>>, CaptureDiscovery) {
    collector_config(total, shifted, limit, policy, stride, host, 0, false, None)
}
fn collector_config(
    total: u64,
    shifted: bool,
    limit: u64,
    policy: CaptureLimitPolicy,
    stride: bool,
    host: Option<u64>,
    layout: usize,
    empty: bool,
    encoded: Option<u64>,
) -> (Observer, Rc<RefCell<Backend>>, CaptureDiscovery) {
    let (mut plan, mut catalog, mut support, caps) = fixture(CaptureTransform::Summary);
    if layout != 0 {
        let row = TensorAxis {
            name: "sequence".into(),
            dimension: SymbolicDimension::Sequence,
        };
        let fixed = |name: &str, n| TensorAxis {
            name: name.into(),
            dimension: SymbolicDimension::Known(n),
        };
        catalog.points[0].axes = Some(if layout == 1 {
            vec![fixed("left", 2), row, fixed("right", 2)]
        } else {
            vec![fixed("hidden", 4), row]
        });
    }
    support.capture = caps.clone();
    let original = plan.selections.pop().unwrap();
    let mut scopes = Vec::new();
    for (lane, scope) in [
        SpeculativeCaptureScope::Target,
        SpeculativeCaptureScope::Prediction { depth: 0 },
    ]
    .into_iter()
    .enumerate()
    {
        for (kind, transform) in [
            CaptureTransform::Summary,
            CaptureTransform::Histogram {
                edges: vec![-10., 0., 10., 100., 2000.],
            },
        ]
        .into_iter()
        .enumerate()
        {
            let mut selection = original.clone();
            selection.id = format!("lane{lane}-{kind}");
            selection.transform = transform;
            if stride && (lane == 0 || !shifted || total > 1) {
                selection.slices = vec![CaptureSlice {
                    axis: "sequence".into(),
                    start: 1,
                    end: total - u64::from(lane == 1 && shifted),
                    stride: 2,
                }];
            }
            if empty {
                selection.slices = vec![CaptureSlice {
                    axis: "sequence".into(),
                    start: 1,
                    end: 1,
                    stride: 1,
                }];
            }
            plan.selections.push(selection);
            scopes.push(scope);
        }
    }
    plan.limits.cumulative.captures = limit;
    plan.limits.per_step.captures = 100;
    plan.limits.on_limit = policy;
    if let Some(host) = host {
        plan.limits.cumulative.host_bytes = host;
    }
    if let Some(encoded) = encoded {
        plan.limits.cumulative.encoded_bytes = encoded;
    }
    let admitted = plan
        .admit_invocations(
            &catalog,
            &support,
            &caps,
            CaptureInvocationBounds {
                batch: 1,
                max_sequence: total,
                max_context: None,
                max_predictions: 1,
            },
        )
        .unwrap();
    let backend = Rc::new(RefCell::new(Backend::default()));
    let observer = SpeculativeCaptureObserver::new(
        CaptureSession::new(eredu_core::capture::SharedCapturePlan::new(admitted)),
        Provider(backend.clone()),
        map as fn(&CaptureExecutionError<std::io::Error>) -> String,
        SpeculativeRequestId::new(0),
        scopes,
        vec![],
    )
    .unwrap();
    (
        observer,
        backend,
        CaptureDiscovery {
            artifact_identity: "source".into(),
            catalog,
            support,
        },
    )
}
fn origin() -> SpeculativeActivationOrigin {
    SpeculativeActivationOrigin {
        request: SpeculativeRequestId::new(0),
        committed_tokens: 0,
        prediction: 0,
        prefix_digest: [3; 32],
        optimistic: false,
    }
}
fn begin(observer: &mut Observer, total: u64, shifted: bool) {
    observer.set_activation_origin(Some(origin()));
    observer.set_prefill_reduction_geometry(SpeculativePrefillReductionGeometry {
        target_sequence: total,
        prediction_sequence: total - u64::from(shifted),
    });
}
fn values(start: u64, end: u64, seed: bool) -> Value {
    Value {
        shape: vec![end - start, 4],
        data: TensorObservationData::F32(
            (start * 4..end * 4)
                .map(|i| i as f32 + if seed { 1001. } else { 1. })
                .collect(),
        ),
    }
}
fn span(
    observer: &mut Observer,
    total: u64,
    start: u64,
    end: u64,
    shifted: bool,
    seed: bool,
) -> Result<(), String> {
    let skip = u64::from(seed && shifted && start == 0);
    let width = end - start - skip;
    if width == 0 {
        return Ok(());
    }
    let hidden = if seed && shifted {
        start.saturating_sub(1)
    } else {
        start
    };
    let phase = if seed {
        SpeculativeActivationPhase::PredictionPrefill
    } else {
        SpeculativeActivationPhase::TargetPrefill
    };
    observer.set_prefill_span(Some(SpeculativePrefillSpan {
        prompt_tokens: total,
        input_start: start,
        input_end: end,
        position: start,
        hidden_start: hidden,
        token_start: start + skip,
        sequence: width,
        seed_start: hidden,
    }));
    let input = values(hidden, hidden + width, seed);
    let result = with_speculative_activation(Some(observer), phase, width as usize, |o| {
        o.unwrap().observe("block.output", &input)
    });
    observer.set_prefill_span(None);
    result
}
fn all(observer: &mut Observer, total: u64, chunk: u64, shifted: bool) -> Result<(), String> {
    begin(observer, total, shifted);
    for start in (0..total).step_by(chunk as usize) {
        let end = (start + chunk).min(total);
        span(observer, total, start, end, shifted, false)?;
        span(observer, total, start, end, shifted, true)?;
    }
    observer.complete_prefill_reductions()
}
fn drain(observer: &mut Observer) -> Vec<SpeculativeActivationCapture> {
    std::iter::from_fn(|| observer.take_activation_capture()).collect()
}
fn reference(
    observer: &Observer,
    entry: &SpeculativePrefillReduction,
    seed: bool,
) -> CapturePayload {
    let plan = observer.session().plan();
    let selection = &plan.plan().selections[entry.selection_index];
    let point = &plan.points()[entry.selection_index];
    let value = values(0, entry.logical_sequence, seed);
    let slice = resolve_slice(point, selection, &value.shape).unwrap();
    Backend::default()
        .transform(&value, selection, &slice)
        .unwrap()
}

#[test]
fn logical_reductions_preserve_interleaved_windows_strides_and_terminal_attribution() {
    for (total, chunk, shifted, stride) in [
        (5, 2, true, false),
        (5, 2, true, true),
        (6, 2, false, false),
        (1, 1, true, false),
        (5, 1, true, false),
    ] {
        let (mut observer, backend, discovery) =
            collector(total, shifted, 100, CaptureLimitPolicy::Fail, stride);
        let saved = observer.checkpoint(&discovery).unwrap();
        all(&mut observer, total, chunk, shifted).unwrap();
        let mut physical = drain(&mut observer);
        assert!(physical.iter().all(|p| p.prefill_reductions.is_none()));
        assert!(observer.session().checkpoint(&discovery).is_err());
        let used = observer.session().cumulative_usage();
        observer.finish_prefill_reductions(true);
        physical.extend(drain(&mut observer));
        let report = physical
            .last()
            .unwrap()
            .prefill_reductions
            .as_ref()
            .unwrap();
        assert_eq!(report.logical_invocation, 0);
        assert_eq!(report.origin, origin());
        assert!(serde_json::to_vec(report).unwrap().len() as u64 <= report.charged.encoded_bytes);
        for entry in &report.records {
            if entry.logical_sequence == 0 {
                assert_eq!(entry.status, SpeculativePrefillReductionStatus::Skipped);
                continue;
            }
            assert_eq!(entry.status, SpeculativePrefillReductionStatus::Complete);
            assert_eq!(entry.covered_sequence, entry.logical_sequence);
            let expected = reference(
                &observer,
                entry,
                entry.phase == SpeculativeActivationPhase::PredictionPrefill,
            );
            match (entry.record.payload.as_ref().unwrap(), expected) {
                (CapturePayload::Summary(actual), CapturePayload::Summary(expected)) => {
                    assert_eq!(
                        (actual.elements, actual.finite, actual.min, actual.max),
                        (
                            expected.elements,
                            expected.finite,
                            expected.min,
                            expected.max
                        )
                    );
                    assert!((actual.mean.unwrap() - expected.mean.unwrap()).abs() < 1e-10);
                    assert!((actual.rms.unwrap() - expected.rms.unwrap()).abs() < 1e-10);
                }
                (CapturePayload::Histogram(actual), CapturePayload::Histogram(expected)) => {
                    assert_eq!(*actual, expected)
                }
                _ => panic!("wrong reduction"),
            }
            assert_eq!(
                entry.record.source_shape.as_ref().unwrap()[0],
                entry.logical_sequence
            );
        }
        for (id, p) in physical.iter().enumerate() {
            assert_eq!(p.invocation, id as u64);
            assert!(p.completed);
            assert!(p.captures.as_step().invocation.unwrap().sequence <= chunk);
        }
        assert!(backend.borrow().transforms > 0);
        observer.set_activation_origin(None);
        observer.restore(&saved).unwrap();
        assert_eq!(observer.session().cumulative_usage(), used);
        all(&mut observer, total, chunk, shifted).unwrap();
        observer.finish_prefill_reductions(true);
        let repeated = drain(&mut observer);
        assert_eq!(
            repeated
                .last()
                .unwrap()
                .prefill_reductions
                .as_ref()
                .unwrap()
                .logical_invocation,
            physical.len() as u64
        );
    }
}

#[test]
fn reduction_cancellation_late_failure_and_quota_skip_never_publish_partial_totals() {
    for cancel_after_complete in [false, true] {
        let (mut observer, _, _) = collector(5, true, 100, CaptureLimitPolicy::Fail, false);
        if cancel_after_complete {
            all(&mut observer, 5, 2, true).unwrap();
        } else {
            begin(&mut observer, 5, true);
            span(&mut observer, 5, 0, 2, true, false).unwrap();
        }
        let used = observer.session().cumulative_usage();
        observer.finish_prefill_reductions(false);
        let records = drain(&mut observer);
        let report = records.last().unwrap().prefill_reductions.as_ref().unwrap();
        assert!(report
            .records
            .iter()
            .all(|r| r.status != SpeculativePrefillReductionStatus::Complete
                && r.record.payload.is_none()));
        assert_eq!(observer.session().cumulative_usage(), used);
        assert!(records[0].completed);
    }
    let (mut observer, backend, _) = collector(5, true, 2, CaptureLimitPolicy::Skip, false);
    all(&mut observer, 5, 2, true).unwrap();
    observer.finish_prefill_reductions(true);
    let records = drain(&mut observer);
    let report = records.last().unwrap().prefill_reductions.as_ref().unwrap();
    assert!(report
        .records
        .iter()
        .all(|r| r.status == SpeculativePrefillReductionStatus::Skipped
            && r.record.payload.is_none()));
    assert_eq!(backend.borrow().transforms, 2);
    assert_eq!(observer.session().cumulative_usage().captures, 2);
}

#[test]
fn reduction_missing_duplicate_and_corrupt_hooks_fail_without_complete_companion() {
    for mode in 0..4 {
        let (mut observer, backend, _) = collector(5, true, 100, CaptureLimitPolicy::Fail, false);
        begin(&mut observer, 5, true);
        span(&mut observer, 5, 0, 2, true, false).unwrap();
        let before = backend.borrow().transforms;
        let failure = match mode {
            0 => span(&mut observer, 5, 0, 2, true, false),
            1 => observer.complete_prefill_reductions(),
            2 => {
                backend.borrow_mut().corrupt_reduction = true;
                span(&mut observer, 5, 0, 2, true, true)
            }
            _ => {
                backend.borrow_mut().fail = true;
                span(&mut observer, 5, 0, 2, true, true)
            }
        };
        assert!(failure.is_err());
        if mode < 2 {
            assert_eq!(backend.borrow().transforms, before);
        }
        observer.finish_prefill_reductions(false);
        let records = drain(&mut observer);
        assert!(records
            .last()
            .unwrap()
            .prefill_reductions
            .as_ref()
            .unwrap()
            .records
            .iter()
            .all(|r| r.status != SpeculativePrefillReductionStatus::Complete));
        assert!(observer.session().cumulative_usage().captures >= 2);
    }
}

#[test]
fn reduction_exact_host_limit_and_original_source_retire_without_refund() {
    let (mut measured, _, _) = collector(5, true, 100, CaptureLimitPolicy::Fail, false);
    all(&mut measured, 5, 2, true).unwrap();
    let exact = measured.session().cumulative_usage();
    measured.finish_prefill_reductions(true);
    for encoded in [false, true] {
        let required = if encoded {
            exact.encoded_bytes
        } else {
            exact.host_bytes
        };
        for (limit, allowed) in [(required, true), (required - 1, false)] {
            let (mut observer, backend, _) = collector_config(
                5,
                true,
                100,
                CaptureLimitPolicy::Fail,
                false,
                (!encoded).then_some(limit),
                0,
                false,
                encoded.then_some(limit),
            );
            let result = all(&mut observer, 5, 2, true);
            assert_eq!(result.is_ok(), allowed);
            let spent = observer.session().cumulative_usage();
            observer.finish_prefill_reductions(allowed);
            assert_eq!(observer.session().cumulative_usage(), spent);
            assert!(
                if encoded {
                    spent.encoded_bytes
                } else {
                    spent.host_bytes
                } <= limit
            );
            assert!(backend.borrow().transforms > 0);
            if allowed {
                assert_eq!(spent, exact);
                let records = drain(&mut observer);
                let report = records.last().unwrap().prefill_reductions.as_ref().unwrap();
                let mut exact_wire = report.as_reductions().clone();
                // Already complete: the accepting serializer needs no terminal delta.
                exact_wire.charged.encoded_bytes = serde_json::to_vec(report).unwrap().len() as u64;
                // Changing the charged number can alter its own decimal width. Stabilize it.
                for _ in 0..3 {
                    exact_wire.charged.encoded_bytes =
                        serde_json::to_vec(&exact_wire).unwrap().len() as u64;
                }
                assert!(crate::capture::encoded::prefill_reductions_fit_encoding(
                    &exact_wire
                ));
                exact_wire.charged.encoded_bytes -= 1;
                assert!(!crate::capture::encoded::prefill_reductions_fit_encoding(
                    &exact_wire
                ));
            } else {
                assert!(drain(&mut observer)
                    .iter()
                    .filter_map(|r| r.prefill_reductions.as_ref())
                    .flat_map(|r| &r.records)
                    .all(|r| r.status != SpeculativePrefillReductionStatus::Complete));
            }
        }
    }

    struct Retired {
        plan: crate::capture::tests::PlanRetirementProbe,
        drops: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }
    impl Drop for Retired {
        fn drop(&mut self) {
            assert!(
                self.plan.is_retired(),
                "actual source must retire before final host custody"
            );
            self.drops.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
    }
    let (mut observer, _, _) = collector(5, true, 100, CaptureLimitPolicy::Fail, false);
    let source = crate::capture::tests::plan_retirement_probe(&observer.session().plan);
    let drops = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let authority = eredu_core::HostPreparationAuthority::retain(Retired {
        plan: source.clone(),
        drops: drops.clone(),
    });
    observer
        .session()
        .retain_host_preparation(&authority)
        .unwrap();
    drop(authority);
    begin(&mut observer, 5, true);
    span(&mut observer, 5, 0, 2, true, false).unwrap();
    assert_eq!(drops.load(std::sync::atomic::Ordering::SeqCst), 0);
    assert!(!source.is_retired());
    drop(observer);
    assert_eq!(drops.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert!(source.is_retired());
}

#[test]
fn logical_reduction_nonleading_axes_empty_slices_and_nonfinite_values_match_full_selection() {
    for layout in 0..3 {
        for empty in [false, true] {
            for stride in [false, true] {
                let (mut observer, backend, discovery) = collector_config(
                    5,
                    false,
                    100,
                    CaptureLimitPolicy::Fail,
                    stride,
                    None,
                    layout,
                    empty,
                    None,
                );
                let shape = match layout {
                    0 => vec![5, 4],
                    1 => vec![2, 5, 2],
                    _ => vec![4, 5],
                };
                let axis = usize::from(layout != 0);
                let global = Value {
                    shape: shape.clone(),
                    data: TensorObservationData::F32(
                        (0..20)
                            .map(|i| match i {
                                0 => f32::NAN,
                                1 => f32::INFINITY,
                                2 => f32::NEG_INFINITY,
                                3 => -0.0,
                                _ => i as f32 - 10.,
                            })
                            .collect(),
                    ),
                };
                let expected: Vec<_> = observer
                    .session()
                    .plan()
                    .plan()
                    .selections
                    .iter()
                    .map(|selection| {
                        let slice =
                            resolve_slice(&discovery.catalog.points[0], selection, &shape).unwrap();
                        Backend::default()
                            .transform(&global, selection, &slice)
                            .unwrap()
                    })
                    .collect();
                begin(&mut observer, 5, false);
                for (start, end) in [(0, 2), (2, 4), (4, 5)] {
                    let mut local_shape = shape.clone();
                    local_shape[axis] = end - start;
                    let mut selected = ResolvedCaptureSlice {
                        starts: vec![0; shape.len()],
                        ends: shape.clone(),
                        strides: vec![1; shape.len()],
                        shape: local_shape.clone(),
                    };
                    selected.starts[axis] = start;
                    selected.ends[axis] = end;
                    let TensorObservationData::F32(data) = &global.data else {
                        unreachable!()
                    };
                    let local = Value {
                        shape: local_shape,
                        data: TensorObservationData::F32(
                            selected_indices(&shape, &selected)
                                .into_iter()
                                .map(|i| data[i])
                                .collect(),
                        ),
                    };
                    for phase in [
                        SpeculativeActivationPhase::TargetPrefill,
                        SpeculativeActivationPhase::PredictionPrefill,
                    ] {
                        observer.set_prefill_span(Some(SpeculativePrefillSpan {
                            prompt_tokens: 5,
                            input_start: start,
                            input_end: end,
                            position: start,
                            hidden_start: start,
                            token_start: start,
                            sequence: end - start,
                            seed_start: start,
                        }));
                        with_speculative_activation(
                            Some(&mut observer),
                            phase,
                            (end - start) as usize,
                            |o| o.unwrap().observe("block.output", &local),
                        )
                        .unwrap();
                        observer.set_prefill_span(None);
                    }
                }
                observer.complete_prefill_reductions().unwrap();
                observer.finish_prefill_reductions(true);
                let records = drain(&mut observer);
                let report = records.last().unwrap().prefill_reductions.as_ref().unwrap();
                for entry in &report.records {
                    assert_eq!(entry.status, SpeculativePrefillReductionStatus::Complete);
                    assert_eq!(entry.covered_sequence, 5);
                    match (
                        entry.record.payload.as_ref().unwrap(),
                        &expected[entry.selection_index],
                    ) {
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
                                    (None, None) => (),
                                    _ => panic!("wrong finite aggregate"),
                                }
                            }
                        }
                        (CapturePayload::Histogram(a), CapturePayload::Histogram(b)) => {
                            assert_eq!(a, b)
                        }
                        _ => panic!("wrong payload"),
                    }
                }
                if empty {
                    assert_eq!(backend.borrow().transforms, 0);
                }
            }
        }
    }
}

fn emit_target(
    observer: &mut Observer,
    start: u64,
    value: &Value,
    hook: bool,
) -> Result<(), String> {
    let width = value.shape[0];
    observer.set_prefill_span(Some(SpeculativePrefillSpan {
        prompt_tokens: 5,
        input_start: start,
        input_end: start + width,
        position: start,
        hidden_start: start,
        token_start: start,
        sequence: width,
        seed_start: start,
    }));
    let result = with_speculative_activation(
        Some(observer),
        SpeculativeActivationPhase::TargetPrefill,
        width as usize,
        |o| {
            if hook {
                o.unwrap().observe("block.output", value)
            } else {
                Ok(())
            }
        },
    );
    observer.set_prefill_span(None);
    result
}

#[test]
fn logical_reduction_rejects_missing_hook_changed_source_and_foreign_origin() {
    for mode in 0..5 {
        let (mut observer, backend, _) = collector(5, true, 100, CaptureLimitPolicy::Fail, false);
        begin(&mut observer, 5, true);
        span(&mut observer, 5, 0, 2, true, false).unwrap();
        let prior = backend.borrow().transforms;
        let mut value = values(2, 4, false);
        if mode == 1 {
            value.data = TensorObservationData::I64((9..17).collect());
        }
        if mode == 2 {
            value.shape[1] = 5;
        }
        if mode == 3 {
            let mut foreign = origin();
            foreign.prefix_digest[0] ^= 1;
            observer.set_activation_origin(Some(foreign));
        }
        let result = emit_target(
            &mut observer,
            if mode == 4 { 3 } else { 2 },
            &value,
            mode != 0,
        );
        assert!(result.is_err(), "mode {mode}");
        if mode != 1 {
            assert_eq!(backend.borrow().transforms, prior);
        }
        observer.finish_prefill_reductions(false);
        let records = drain(&mut observer);
        let report = records.last().unwrap().prefill_reductions.as_ref().unwrap();
        assert!(report.records.iter().all(|entry| entry.status
            != SpeculativePrefillReductionStatus::Complete
            && entry.record.payload.is_none()));
    }
}

#[test]
fn logical_reduction_integer_rounding_and_late_native_limit_preserve_consumed_work() {
    let data = (0..20).map(|i| 16_777_210i64 + i).collect::<Vec<_>>();
    let global = Value {
        shape: vec![5, 4],
        data: TensorObservationData::I64(data.clone()),
    };
    let (mut observer, _, discovery) = collector(5, false, 100, CaptureLimitPolicy::Fail, false);
    let expected = observer
        .session()
        .plan()
        .plan()
        .selections
        .iter()
        .map(|selection| {
            let selected =
                resolve_slice(&discovery.catalog.points[0], selection, &global.shape).unwrap();
            Backend::default()
                .transform(&global, selection, &selected)
                .unwrap()
        })
        .collect::<Vec<_>>();
    begin(&mut observer, 5, false);
    for (start, end) in [(0, 2), (2, 4), (4, 5)] {
        let local = Value {
            shape: vec![end - start, 4],
            data: TensorObservationData::I64(
                data[(start * 4) as usize..(end * 4) as usize].to_vec(),
            ),
        };
        for phase in [
            SpeculativeActivationPhase::TargetPrefill,
            SpeculativeActivationPhase::PredictionPrefill,
        ] {
            observer.set_prefill_span(Some(SpeculativePrefillSpan {
                prompt_tokens: 5,
                input_start: start,
                input_end: end,
                position: start,
                hidden_start: start,
                token_start: start,
                sequence: end - start,
                seed_start: start,
            }));
            with_speculative_activation(Some(&mut observer), phase, (end - start) as usize, |o| {
                o.unwrap().observe("block.output", &local)
            })
            .unwrap();
            observer.set_prefill_span(None);
        }
    }
    observer.complete_prefill_reductions().unwrap();
    observer.finish_prefill_reductions(true);
    let records = drain(&mut observer);
    for entry in &records
        .last()
        .unwrap()
        .prefill_reductions
        .as_ref()
        .unwrap()
        .records
    {
        assert_eq!(entry.record.source_dtype, Some(TensorDtype::I64));
        match (
            entry.record.payload.as_ref().unwrap(),
            &expected[entry.selection_index],
        ) {
            (CapturePayload::Summary(a), CapturePayload::Summary(b)) => {
                assert_eq!(
                    (a.elements, a.finite, a.min, a.max, a.mean),
                    (b.elements, b.finite, b.min, b.max, b.mean)
                );
                assert!((a.rms.unwrap() - b.rms.unwrap()).abs() < 1e-7);
            }
            (CapturePayload::Histogram(a), CapturePayload::Histogram(b)) => assert_eq!(a, b),
            _ => panic!("wrong integer reduction"),
        }
    }
    let (mut observer, backend, _) = collector(5, true, 3, CaptureLimitPolicy::Fail, false);
    let result = all(&mut observer, 5, 2, true);
    assert!(result.is_err());
    assert_eq!(backend.borrow().transforms, 3);
    let used = observer.session().cumulative_usage();
    assert_eq!(used.captures, 3);
    observer.finish_prefill_reductions(false);
    assert_eq!(observer.session().cumulative_usage(), used);
    assert!(drain(&mut observer)
        .iter()
        .filter_map(|r| r.prefill_reductions.as_ref())
        .flat_map(|r| &r.records)
        .all(|r| r.status != SpeculativePrefillReductionStatus::Complete));
}

#[test]
fn logical_reduction_generated_refusal_and_unwind_keep_terminal_evidence() {
    let (mut unopened, backend, _) = collector(5, true, 100, CaptureLimitPolicy::Fail, false);
    begin(&mut unopened, 5, true);
    unopened.finish_prefill_reductions(false);
    assert!(drain(&mut unopened).is_empty());
    assert_eq!(backend.borrow().transforms, 0);
    assert_eq!(
        unopened.session().cumulative_usage(),
        CaptureUsage::default()
    );
    for unwind in [false, true] {
        let (mut observer, backend, _) = collector(5, true, 100, CaptureLimitPolicy::Fail, false);
        begin(&mut observer, 5, true);
        observer.set_prefill_span(Some(SpeculativePrefillSpan {
            prompt_tokens: 5,
            input_start: 0,
            input_end: 2,
            position: 0,
            hidden_start: 0,
            token_start: 0,
            sequence: 2,
            seed_start: 0,
        }));
        let mut factories = 0;
        let attempt = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            with_speculative_activation(
                Some(&mut observer),
                SpeculativeActivationPhase::TargetPrefill,
                2,
                |o| {
                    if unwind {
                        std::panic::panic_any(12345u32);
                    }
                    o.unwrap().observe_generated(
                        "block.output",
                        &values(0, 2, false),
                        &GeneratedCaptureSource {
                            creation_bytes: 0,
                            source_dtype: Some(TensorDtype::F32),
                        },
                        &mut || {
                            factories += 1;
                            Ok(values(0, 2, false))
                        },
                    )
                },
            )
        }));
        if unwind {
            assert_eq!(*attempt.unwrap_err().downcast::<u32>().unwrap(), 12345);
        } else {
            assert!(attempt.unwrap().is_err());
            assert!(matches!(
                observer.take_activation_error(),
                Some(SpeculativeControlError::Capture(CaptureError::Unsupported(
                    _
                )))
            ));
        }
        assert_eq!(factories, 0);
        assert_eq!(backend.borrow().transforms, 0);
        let spent = observer.session().cumulative_usage();
        observer.set_prefill_span(None);
        observer.finish_prefill_reductions(false);
        assert_eq!(observer.session().cumulative_usage(), spent);
        let physical = drain(&mut observer);
        assert_eq!(physical.len(), 1);
        assert!(!physical[0].completed);
        assert!(physical[0]
            .prefill_reductions
            .as_ref()
            .unwrap()
            .records
            .iter()
            .all(
                |entry| entry.status != SpeculativePrefillReductionStatus::Complete
                    && entry.record.payload.is_none()
            ));
    }
}

mod preview;
