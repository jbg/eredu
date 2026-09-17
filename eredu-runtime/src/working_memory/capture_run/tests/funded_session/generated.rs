use super::*;
use eredu_nn::{
    BlockFp8InputReconstructionPlan, GeneratedTensorProgram, GeneratedTensorSourceRole,
    RetainedGeneratedTensorFactory,
};
use std::{rc::Rc, sync::Arc};

#[derive(Clone)]
struct Value {
    shape: Vec<i32>,
    dtype: TensorDtype,
    data: Rc<Vec<f32>>,
}
fn value(shape: &[i32], dtype: TensorDtype) -> Value {
    Value {
        shape: shape.into(),
        dtype,
        data: Rc::new(
            (0..shape.iter().product::<i32>())
                .map(|i| i as f32 * 0.25 - 1.0)
                .collect(),
        ),
    }
}
#[derive(Debug, thiserror::Error)]
enum Outer {
    #[error(transparent)]
    Capture(#[from] FundedCaptureError<MechanismError>),
    #[error("original factory failure")]
    Factory(Arc<()>),
}
struct Factory<'a> {
    prototype: &'a Value,
    compact: Value,
    scales: Value,
    calls: usize,
    visits: usize,
    fail: Option<usize>,
    panic: bool,
    wrong_dtype: bool,
    cause: Arc<()>,
}
impl<'a> Factory<'a> {
    fn new(prototype: &'a Value) -> Self {
        let plan = BlockFp8InputReconstructionPlan::new(&prototype.shape).unwrap();
        Self {
            prototype,
            compact: value(&plan.values_shape(), TensorDtype::U8),
            scales: value(&plan.scales_shape(), TensorDtype::F32),
            calls: 0,
            visits: 0,
            fail: None,
            panic: false,
            wrong_dtype: false,
            cause: Arc::new(()),
        }
    }
    fn source(&self) -> GeneratedCaptureSource {
        crate::capture::generated_capture_source(
            &BlockFp8InputReconstructionPlan::new(&self.prototype.shape)
                .unwrap()
                .logical_capture_source()
                .unwrap(),
        )
    }
}
impl RetainedGeneratedTensorFactory<Value, Outer> for Factory<'_> {
    fn program(&self) -> GeneratedTensorProgram<'_> {
        GeneratedTensorProgram::BlockFp8Input(
            BlockFp8InputReconstructionPlan::new(&self.prototype.shape).unwrap(),
        )
    }
    fn visit_sources(
        &mut self,
        sink: &mut dyn FnMut(GeneratedTensorSourceRole, &Value) -> Result<(), Outer>,
    ) -> Result<(), Outer> {
        self.visits += 1;
        sink(GeneratedTensorSourceRole::CompactValues, &self.compact)?;
        sink(GeneratedTensorSourceRole::BlockScales, &self.scales)
    }
    fn generate(
        &mut self,
        sink: &mut dyn FnMut(&Value) -> Result<(), Outer>,
    ) -> Result<Value, Outer> {
        self.calls += 1;
        let mut last = None;
        // Protocol fixture: numerical reconstruction is covered by actual NN/
        // native producer tests; these seven owned values exercise claim custody.
        for index in 1..=7 {
            let output = value(
                &self.prototype.shape,
                if self.wrong_dtype {
                    TensorDtype::F16
                } else {
                    TensorDtype::F32
                },
            );
            sink(&output)?;
            if self.fail == Some(index) {
                assert!(!self.panic, "after retained factory output");
                return Err(Outer::Factory(self.cause.clone()));
            }
            last = Some(output);
        }
        Ok(last.unwrap())
    }
}
struct GeneratedBackend {
    scope: WorkingMemoryFundingScope,
    roots: Vec<Value>,
    preflights: usize,
    transforms: usize,
    retain_failure: Option<usize>,
    reject_preflight: bool,
}
impl ScheduledCaptureBackend for GeneratedBackend {
    type Tensor = Value;
    type Error = MechanismError;
    fn validate_source(
        &self,
        value: &Value,
        geometry: &CaptureTensorGeometry<'_>,
    ) -> Result<TensorDtype, MechanismError> {
        if value
            .shape
            .iter()
            .map(|&n| n as usize)
            .ne(geometry.source_shape().iter().copied())
        {
            return Err(MechanismError::Shape);
        }
        Ok(value.dtype.clone())
    }
    fn estimate(
        &self,
        _: &Value,
        g: &CaptureTensorGeometry<'_>,
    ) -> Result<CaptureUsage, CaptureError> {
        Ok(usage(g.elements()))
    }
    fn estimate_generated(
        &self,
        p: &Value,
        _: &GeneratedCaptureSource,
        g: &CaptureTensorGeometry<'_>,
    ) -> Result<CaptureUsage, CaptureError> {
        ScheduledCaptureBackend::estimate(self, p, g)
    }
    fn preflight_generated(
        &mut self,
        _: &Value,
        _: &GeneratedCaptureSource,
        _: GeneratedTensorProgram<'_>,
        _: bool,
        c: &CaptureTensorClaim<'_, '_>,
    ) -> Result<(), FundedCaptureError<MechanismError>> {
        self.preflights += 1;
        c.validate_native_scope(&self.scope)
            .map_err(CaptureRunHostError::from)?;
        if self.reject_preflight {
            return Err(FundedCaptureError::Backend(MechanismError::Sentinel(11)));
        }
        Ok(())
    }
    fn retain_generated_source(
        &mut self,
        _: &Value,
        _: GeneratedTensorSourceRole,
        v: &Value,
        c: &CaptureTensorClaim<'_, '_>,
    ) -> Result<(), FundedCaptureError<MechanismError>> {
        self.retain_generated_output(v, c)
    }
    fn retain_generated_output(
        &mut self,
        v: &Value,
        c: &CaptureTensorClaim<'_, '_>,
    ) -> Result<(), FundedCaptureError<MechanismError>> {
        self.roots.push(v.clone());
        c.validate_native_scope(&self.scope)
            .map_err(CaptureRunHostError::from)?;
        if self.retain_failure == Some(self.roots.len()) {
            return Err(FundedCaptureError::Backend(MechanismError::Sentinel(12)));
        }
        Ok(())
    }
    fn transform(
        &mut self,
        v: &Value,
        c: CaptureTensorClaim<'_, '_>,
    ) -> Result<ClaimedCaptureTensor, MechanismError> {
        self.transforms += 1;
        let mut b = c.prepare()?;
        while b.initialized_count() < b.len() {
            b.push_f32(v.data[b.initialized_count()]).unwrap();
        }
        Ok(b.finish().unwrap())
    }
}
impl CaptureBackend for GeneratedBackend {
    type Tensor = Value;
    type Error = MechanismError;
    fn shape(&self, v: &Value) -> Result<Vec<u64>, MechanismError> {
        Ok(v.shape.iter().map(|&n| n as u64).collect())
    }
    fn source_dtype(&self, v: &Value) -> Option<TensorDtype> {
        Some(v.dtype.clone())
    }
    fn estimate(
        &self,
        _: &Value,
        s: &CaptureSelection,
        sl: &ResolvedCaptureSlice,
    ) -> Result<CaptureUsage, CaptureError> {
        let n = sl.shape.iter().product::<u64>();
        let n = match s.transform {
            CaptureTransform::Preview { max_elements } => n.min(max_elements),
            _ => n,
        };
        Ok(usage(n as usize))
    }
    fn transform(
        &mut self,
        v: &Value,
        s: &CaptureSelection,
        sl: &ResolvedCaptureSlice,
    ) -> Result<CapturePayload, MechanismError> {
        let (shape, n) = match s.transform {
            CaptureTransform::Preview { max_elements } => {
                let n = (sl.shape.iter().product::<u64>()).min(max_elements) as usize;
                (vec![n], n)
            }
            _ => (
                sl.shape.iter().map(|&n| n as usize).collect(),
                sl.shape.iter().product::<u64>() as usize,
            ),
        };
        Ok(CapturePayload::Tensor(
            TensorObservation::new(shape, TensorObservationData::F32(v.data[..n].to_vec()))
                .unwrap(),
        ))
    }
}
fn selected(mut raw: CapturePlan) -> SharedCapturePlan {
    for s in &mut raw.selections {
        s.schedule = CaptureSchedule::default();
    }
    admit(raw, point(), 4, false)
}
fn setup(
    source: &SharedCapturePlan,
) -> (
    WorkingMemoryPool,
    WorkingMemoryReservation,
    WorkingMemoryFundingRun,
    FundedCaptureSession,
    GeneratedBackend,
) {
    let h = plan(source).initialization_peak_bytes();
    let pool = WorkingMemoryPool::new(h, 0).unwrap();
    let (r, run) = fresh(&pool, h);
    let funded = session(source, &pool, &r, &run);
    let scope = run.scope().unwrap();
    let backend = GeneratedBackend {
        scope,
        roots: vec![],
        preflights: 0,
        transforms: 0,
        retain_failure: None,
        reject_preflight: false,
    };
    (pool, r, run, funded, backend)
}
fn invoke(
    s: &mut FundedCaptureSession,
    b: &mut GeneratedBackend,
    p: &Value,
    f: &mut Factory<'_>,
    path: &str,
) -> Result<(), Outer> {
    let e = DistributedCommitEpoch::new(1).unwrap();
    let source = f.source();
    s.with_observer(b, 0, &Outer::Capture, |o| {
        let guard = crate::inspection::ObservationTransactionGuard::new(o, e);
        guard.observer.prepare_transaction(e, ExpertPass::Prefill)?;
        guard
            .observer
            .observe_generated_retained(path, p, &source, f)?;
        guard.observer.complete_transaction(e)?;
        guard.finish(true);
        Ok(())
    })
    .unwrap()
}
fn finish_backend(b: GeneratedBackend) {
    let GeneratedBackend { scope, roots, .. } = b;
    drop(roots);
    scope.certify().unwrap();
}

#[test]
fn multiple_generated_selections_match_legacy_usage_and_generate_once() {
    let source = selected(raw());
    let (pool, r, run, mut funded, mut backend) = setup(&source);
    let p = value(&[5, 2], TensorDtype::Bf16);
    let mut f = Factory::new(&p);
    let declared = f.source();
    invoke(&mut funded, &mut backend, &p, &mut f, "block.output").unwrap();
    assert_eq!(
        (f.calls, f.visits, backend.transforms, backend.roots.len()),
        (1, 1, 3, 9)
    );
    let actual = funded.take_shared_step().unwrap().unwrap();
    let mut legacy = CaptureSession::from_shared_plan(source.clone());
    let e = DistributedCommitEpoch::new(1).unwrap();
    legacy
        .prepare_step_transaction(e, ExpertPass::Prefill, 0)
        .unwrap();
    let mut calls = 0;
    legacy
        .observe_generated(
            &mut backend,
            "block.output",
            &p,
            &declared,
            &mut || {
                calls += 1;
                Ok::<_, String>(value(&p.shape, TensorDtype::F32))
            },
            &|e| e.to_string(),
        )
        .unwrap();
    legacy.complete_transaction(e).unwrap();
    legacy.finish_transaction(e, true);
    let expected = legacy.take_step().unwrap();
    let mut a = serde_json::to_value(&actual).unwrap();
    let mut b = serde_json::to_value(expected).unwrap();
    a.as_object_mut().unwrap().remove("capture_seconds");
    b.as_object_mut().unwrap().remove("capture_seconds");
    assert_eq!(a, b);
    assert_eq!(calls, 1);
    let escaped = actual.clone();
    let h = plan(&source).initialization_peak_bytes();
    finish_backend(backend);
    drop((actual, f, funded, run, r));
    assert_eq!(pool.used_bytes().unwrap(), h);
    drop(escaped);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
#[test]
fn limit_skip_unselected_and_zero_extent_keep_factory_lazy_but_preview_zero_does_not() {
    for mode in 0..4 {
        let mut raw = raw();
        raw.selections.truncate(1);
        if mode == 0 {
            raw.limits.per_step.captures = 0;
            raw.limits.on_limit = CaptureLimitPolicy::Skip;
        }
        if mode == 2 {
            raw.selections[0].slices = vec![CaptureSlice {
                axis: "width".into(),
                start: 0,
                end: 0,
                stride: 1,
            }];
        }
        if mode == 3 {
            raw.selections[0].transform = CaptureTransform::Preview { max_elements: 0 };
        }
        let source = selected(raw);
        let (_, r, run, mut funded, mut backend) = setup(&source);
        let p = value(&[5, 2], TensorDtype::F16);
        let mut f = Factory::new(&p);
        invoke(
            &mut funded,
            &mut backend,
            &p,
            &mut f,
            if mode == 1 {
                "unselected"
            } else {
                "block.output"
            },
        )
        .unwrap();
        assert_eq!(f.calls, usize::from(mode == 3));
        assert_eq!(f.visits, usize::from(mode == 3));
        assert_eq!(backend.transforms, usize::from(mode == 3));
        if mode < 2 {
            assert_eq!(backend.preflights, 0);
            assert!(backend.roots.is_empty());
        } else {
            assert!(funded.usage().retained_bytes >= f.source().creation_bytes);
        }
        let step = funded.take_shared_step().unwrap().unwrap();
        if mode >= 2 {
            assert_eq!(
                step.records()[0]
                    .payload
                    .as_ref()
                    .unwrap()
                    .as_tensor()
                    .unwrap()
                    .data()
                    .len(),
                0
            );
        }
        finish_backend(backend);
        drop((step, f, funded, run, r));
    }
}
#[test]
fn factory_error_and_unwind_keep_each_prior_output_and_do_not_refund_claim() {
    for panic in [false, true] {
        for stop in [1, 4, 7] {
            let source = selected(raw());
            let (pool, r, run, mut funded, mut backend) = setup(&source);
            let p = value(&[5, 2], TensorDtype::F32);
            let mut f = Factory::new(&p);
            f.fail = Some(stop);
            f.panic = panic;
            let cause = f.cause.clone();
            let result = catch_unwind(AssertUnwindSafe(|| {
                invoke(&mut funded, &mut backend, &p, &mut f, "block.output")
            }));
            if panic {
                assert!(result.is_err());
            } else {
                assert!(
                    matches!(result.unwrap(),Err(Outer::Factory(ref c))if Arc::ptr_eq(c,&cause))
                );
            }
            assert_eq!(backend.roots.len(), 2 + stop);
            let weak = Rc::downgrade(&backend.roots.last().unwrap().data);
            assert_eq!(funded.spent_steps(), 1);
            assert!(funded.usage().retained_bytes >= f.source().creation_bytes);
            let step = funded.take_shared_step().unwrap().unwrap();
            assert_eq!(step.outcome(), CaptureStepOutcome::Aborted);
            drop((step, f, funded));
            assert!(weak.upgrade().is_some());
            // An abandoned original native scope remains quarantined independently.
            drop(backend);
            drop((run, r));
            assert!(pool.used_bytes().unwrap() > 0);
            assert!(weak.upgrade().is_none());
        }
    }
}
#[test]
fn foreign_scope_and_preflight_failure_reject_before_factory_but_spend_once() {
    for foreign in [false, true] {
        let source = selected(raw());
        let (pool, r, run, mut funded, mut backend) = setup(&source);
        let foreign_owner = if foreign {
            let p = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
            let (r2, run2) = fresh(&p, 64);
            let old = std::mem::replace(&mut backend.scope, run2.scope().unwrap());
            old.certify().unwrap();
            Some((p, r2, run2))
        } else {
            backend.reject_preflight = true;
            None
        };
        let p = value(&[5, 2], TensorDtype::F32);
        let mut f = Factory::new(&p);
        assert!(invoke(&mut funded, &mut backend, &p, &mut f, "block.output").is_err());
        assert_eq!((f.calls, f.visits), (0, 0));
        assert!(backend.roots.is_empty());
        assert_eq!(funded.spent_steps(), 1);
        assert!(funded.usage().captures > 0);
        drop(funded.take_shared_step().unwrap());
        finish_backend(backend);
        drop((f, funded, run, r, foreign_owner));
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}
#[test]
fn retention_error_keeps_new_root_and_generated_precision_mismatch_is_typed() {
    for mismatch in [false, true] {
        let source = selected(raw());
        let (_, r, run, mut funded, mut backend) = setup(&source);
        let p = value(&[5, 2], TensorDtype::F32);
        let mut f = Factory::new(&p);
        f.wrong_dtype = mismatch;
        if !mismatch {
            backend.retain_failure = Some(5);
        }
        let error = invoke(&mut funded, &mut backend, &p, &mut f, "block.output").unwrap_err();
        assert_eq!(backend.roots.len(), if mismatch { 9 } else { 5 });
        assert_eq!(backend.transforms, 0);
        assert!(matches!(error, Outer::Capture(_)));
        drop(error);
        drop(funded.take_shared_step().unwrap());
        finish_backend(backend);
        drop((f, funded, run, r));
    }
}
