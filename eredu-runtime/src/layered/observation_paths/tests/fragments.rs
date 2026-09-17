//! Bound semantic fixture + real original host bank. No native capability claim.
use super::*;
use crate::{capture::*, prefill::PrefillChunk, working_memory::*, ActivationObserver, ExpertPass};
use eredu_core::{capture::*, checkpoint::TensorDtype, *};
use eredu_nn::{
    BlockFp8InputReconstructionPlan, GeneratedTensorProgram, GeneratedTensorSourceRole,
    RetainedGeneratedTensorFactory,
};
use std::num::NonZeroU8;

fn geometry() -> InferenceGeometry {
    InferenceGeometry {
        batch_size: 1,
        cached_positions: 7,
        input_positions: 3,
        max_output_tokens: 3,
        prefill_chunk_positions: 2,
        output: OutputDemand::LastPosition,
    }
}
fn admitted(preview: u64, captures: u64, limit: CaptureLimitPolicy) -> SharedCapturePlan {
    admitted_slice(preview, captures, limit, false)
}
fn admitted_slice(
    preview: u64,
    captures: u64,
    limit: CaptureLimitPolicy,
    empty: bool,
) -> SharedCapturePlan {
    let usage = CaptureUsage {
        captures,
        retained_bytes: u64::MAX,
        host_bytes: u64::MAX,
        encoded_bytes: u64::MAX,
    };
    let point = ObservationPoint {
        path: "layer.0.output".into(),
        node_id: "layer.0".into(),
        meaning: "declared fixture rows".into(),
        value_type: ObservationValueType::Tensor,
        dtype: ObservationDtype::Floating,
        axes: Some(vec![
            TensorAxis {
                name: "heads".into(),
                dimension: SymbolicDimension::Known(2),
            },
            TensorAxis {
                name: "sequence".into(),
                dimension: SymbolicDimension::Sequence,
            },
            TensorAxis {
                name: "width".into(),
                dimension: SymbolicDimension::Known(2),
            },
        ]),
        prefill: true,
        decode: true,
        requirements: vec![ObservationRequirement::ActivationHooks],
        position: ObservationPosition::BeforeIntervention,
        retained_bytes: None,
        host_bytes: None,
    };
    let support = ObservationSupportReport {
        schema_version: 1,
        capture: Default::default(),
        points: vec![ObservationSupport {
            path: point.path.clone(),
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
    let mut plan = CapturePlan::none();
    plan.selections = [
        CaptureTransform::FullTensor,
        CaptureTransform::Preview {
            max_elements: preview,
        },
    ]
    .into_iter()
    .enumerate()
    .map(|(i, transform)| CaptureSelection {
        id: i.to_string(),
        path: "layer.0.output".into(),
        schedule: CaptureSchedule::default(),
        slices: if empty {
            vec![CaptureSlice {
                axis: "sequence".into(),
                start: 0,
                end: 0,
                stride: 1,
            }]
        } else {
            vec![]
        },
        transform,
    })
    .collect();
    plan.limits = CaptureLimits {
        per_step: usage,
        cumulative: usage,
        physical_native_bytes: None,
        on_limit: limit,
    };
    SharedCapturePlan::new(
        plan.admit_with_text_origin(
            &catalog,
            &support,
            &CaptureCapabilities {
                transformations: vec![
                    CaptureTransformKind::FullTensor,
                    CaptureTransformKind::Preview,
                ],
                ..Default::default()
            },
            CaptureRequestShape {
                batch: 1,
                prompt_tokens: 3,
                max_predictions: 3,
            },
            CaptureTextOrigin {
                cached_positions: 7,
            },
        )
        .unwrap(),
    )
}
fn selected(source: &SharedCapturePlan) -> PreparedCaptureSelection {
    // The immutable declaration belongs to this test architecture/path owner;
    // actual family collection/revalidation has independent companion coverage.
    let mut paths = super::source();
    Arc::get_mut(&mut paths.0).unwrap().prefill =
        vec![PrefillObservationDeclaration::causal_ordinary_text(
            "layer.0.output".into(),
            1,
            PrefillReadoutStage::BeforeReadout,
        )]
        .into_boxed_slice();
    paths.prepare_capture_selection(source).unwrap()
}
fn fresh(
    source: &SharedCapturePlan,
    shortage: u64,
) -> (
    WorkingMemoryPool,
    WorkingMemoryReservation,
    WorkingMemoryFundingRun,
) {
    let h = CaptureRunHostPlan::prepare(source)
        .unwrap()
        .initialization_peak_bytes();
    fresh_capacity(h, shortage)
}
fn fresh_capacity(h: u64, shortage: u64) -> (WorkingMemoryPool,WorkingMemoryReservation,WorkingMemoryFundingRun) {
    let pool = WorkingMemoryPool::new(h, 0).unwrap();
    let layout = StateMemoryLayout::new(
        LayerSchedule::new(1, vec![cache::LayerCachePolicy::NoState]).unwrap(),
        vec![0],
        1,
        1,
        EstimationCompleteness::Complete,
    )
    .unwrap();
    let state = estimate_runtime_state(
        &layout,
        InputTokenCount::text(10),
        3,
        1,
        NonZeroU8::new(4).unwrap(),
    )
    .unwrap();
    let bound = |n| WorkspaceBound::bounded(n, "finite host-only fragment observer fixture");
    let state = state
        .with_execution_workspace(ExecutionWorkspaceEstimate {
            geometry: geometry(),
            activations: bound(h - shortage),
            attention: bound(0),
            vocabulary: bound(0),
            state_update: bound(0),
            materialization: bound(0),
            retained: bound(0),
        })
        .unwrap();
    let (r, run) = pool
        .reserve_with_capacity(
            &InferenceExecutionIdentity::default(),
            &Admission {
                state,
                requested_positions: 13,
                incremental_required_bytes: h - shortage,
                available_memory_bytes: None,
            },
            pool.effective_capacity().unwrap(),
        )
        .unwrap()
        .into_funding()
        .unwrap();
    (pool, r, run)
}
fn bank(
    source: &SharedCapturePlan,
    r: &WorkingMemoryReservation,
    run: &WorkingMemoryFundingRun,
) -> FundedCaptureSession {
    run.prepare_capture_run(r, CaptureRunHostPlan::prepare(source).unwrap())
        .unwrap()
        .into_capture_session()
        .unwrap()
}
#[derive(Clone)]
struct Value {
    shape: Vec<i32>,
    values: Arc<Vec<f32>>,
}
fn value(start: u64, rows: u64) -> Value {
    Value {
        shape: vec![2, rows as i32, 2],
        values: Arc::new(
            (0..2)
                .flat_map(|h| {
                    (0..rows).flat_map(move |r| {
                        (0..2).map(move |c| (h * 100 + (start + r) * 10 + c) as f32)
                    })
                })
                .collect(),
        ),
    }
}
#[derive(Debug, thiserror::Error)]
enum Native {
    #[error("bad actual source")]
    Shape,
    #[error(transparent)]
    Host(#[from] CaptureRunHostError),
}
#[derive(Debug, thiserror::Error)]
enum Error {
    #[error(transparent)]
    Capture(#[from] FundedCaptureError<Native>),
    #[error("original factory failure")]
    Factory(Arc<()>),
}
struct Backend {
    scope: WorkingMemoryFundingScope,
    roots: Vec<Value>,
    transforms: usize,
    preflights: usize,
}
impl ScheduledCaptureBackend for Backend {
    type Tensor = Value;
    type Error = Native;
    fn validate_routed_prefill_source(
        &self, source: &RoutedUnitCaptureSource<'_, Value>, fragment: &CaptureRoutedPrefillFragment<'_, '_>,
    ) -> Result<TensorDtype, FundedCaptureError<Native>> {
        if source.values.shape != [3, 5] || source.source_groups.shape != [fragment.source_tokens() as i32, 3] {
            return Err(FundedCaptureError::Backend(Native::Shape));
        }
        Ok(TensorDtype::F32)
    }
    fn estimate_routed_prefill(&self, geometry: &CaptureRoutedUnitsGeometry<'_>) -> Result<CaptureUsage, CaptureError> {
        Ok(CaptureUsage { captures: 1, retained_bytes: 65536, host_bytes: 65536,
            encoded_bytes: 65536 + geometry.elements() as u64 })
    }
    fn transform_routed_prefill(
        &mut self, source: &RoutedUnitCaptureSource<'_, Value>,
        mut writer: CaptureRoutedPrefillWriter<'_, '_, '_, '_>,
    ) -> Result<(), FundedCaptureError<Native>> {
        self.transforms += 1;
        for slot in 0..3 {
            if !writer.fragment().selects(source.token_offset, slot) { continue; }
            let expert = source.source_groups.values[source.token_offset as usize * 3 + slot as usize] as u64;
            writer.begin_row(source.token_offset, slot, expert, source.coefficients.values[slot as usize])
                .map_err(CaptureRunHostError::from)?;
            for unit in [1,3] {
                writer.push_f32(source.values.values[slot as usize*5+unit])
                    .map_err(CaptureRunHostError::from)?;
            }
            writer.finish_row().map_err(CaptureRunHostError::from)?;
        }
        writer.source_chunk(source.token_offset, source.token_offset + 1).map_err(CaptureRunHostError::from)?;
        writer.finish().map_err(CaptureRunHostError::from)?;
        Ok(())
    }
    fn validate_routed_invocation_source(
        &self, source: &RoutedUnitCaptureSource<'_, Value>, geometry: &CaptureRoutedUnitsGeometry<'_>,
    ) -> Result<TensorDtype, FundedCaptureError<Native>> {
        if source.values.shape != [3, 5]
            || source.source_groups.shape != [geometry.source_shape()[0] as i32, 3] {
            return Err(FundedCaptureError::Backend(Native::Shape));
        }
        Ok(TensorDtype::F32)
    }
    fn transform_routed_batch(
        &mut self, source: &RoutedUnitCaptureSource<'_, Value>,
        mut writer: CaptureRoutedBatchWriter<'_, '_>,
    ) -> Result<(), FundedCaptureError<Native>> {
        writer.validate_native_scope(&self.scope).map_err(CaptureRunHostError::from)?;
        self.transforms += 1;
        for slot in 0..3 {
            if !writer.selects(source.token_offset, slot) { continue; }
            let expert = source.source_groups.values[source.token_offset as usize * 3 + slot as usize] as u64;
            writer.begin_row(source.token_offset, slot, expert, source.coefficients.values[slot as usize])
                .map_err(CaptureRunHostError::from)?;
            for unit in [1, 3] {
                writer.push_f32(source.values.values[slot as usize * 5 + unit])
                    .map_err(CaptureRunHostError::from)?;
            }
            writer.finish_row().map_err(CaptureRunHostError::from)?;
        }
        writer.source_chunk(source.token_offset, source.token_offset + 1).map_err(CaptureRunHostError::from)?;
        writer.finish().map_err(CaptureRunHostError::from)?;
        Ok(())
    }
    fn validate_source(
        &self,
        value: &Value,
        geometry: &CaptureTensorGeometry<'_>,
    ) -> Result<TensorDtype, Native> {
        if value
            .shape
            .iter()
            .map(|&n| n as usize)
            .ne(geometry.source_shape().iter().copied())
        {
            return Err(Native::Shape);
        }
        Ok(TensorDtype::F32)
    }
    fn estimate(
        &self,
        _: &Value,
        geometry: &CaptureTensorGeometry<'_>,
    ) -> Result<CaptureUsage, CaptureError> {
        Ok(usage(geometry.elements()))
    }
    fn transform(
        &mut self,
        value: &Value,
        claim: CaptureTensorClaim<'_, '_>,
    ) -> Result<ClaimedCaptureTensor, Native> {
        let mut output = claim.prepare()?;
        for &v in value.values.iter().take(output.len()) {
            output.push_f32(v).map_err(CaptureRunHostError::from)?;
        }
        Ok(output.finish().unwrap())
    }
    fn validate_prefill_source(
        &self,
        value: &Value,
        f: &CapturePrefillFragment<'_, '_>,
    ) -> Result<TensorDtype, FundedCaptureError<Native>> {
        if value
            .shape
            .iter()
            .map(|&n| n as usize)
            .ne(f.source_shape().iter().copied())
        {
            return Err(FundedCaptureError::Backend(Native::Shape));
        }
        Ok(TensorDtype::F32)
    }
    fn estimate_prefill(
        &self,
        geometry: &CaptureTensorGeometry<'_>,
    ) -> Result<CaptureUsage, CaptureError> {
        Ok(usage(geometry.elements()))
    }
    fn transform_prefill_fragment(
        &mut self,
        value: &Value,
        claim: CapturePrefillFragmentClaim<'_, '_, '_, '_>,
    ) -> Result<(), FundedCaptureError<Native>> {
        self.transforms += 1;
        let fragment = claim.fragment();
        let mut writer = claim.prepare()?;
        for map in fragment.mappings() {
            writer.push_f32(value.values[map.source_index()])?;
        }
        writer.finish()?;
        Ok(())
    }
    fn preflight_generated_fragment(
        &mut self,
        _: &Value,
        _: &GeneratedCaptureSource,
        _: GeneratedTensorProgram<'_>,
        _: bool,
        claim: &CapturePrefillFragmentClaim<'_, '_, '_, '_>,
    ) -> Result<(), FundedCaptureError<Native>> {
        claim
            .validate_native_scope(&self.scope)
            .map_err(CaptureRunHostError::from)?;
        self.preflights += 1;
        Ok(())
    }
    fn retain_generated_fragment_source(
        &mut self,
        _: &Value,
        _: GeneratedTensorSourceRole,
        value: &Value,
        claim: &CapturePrefillFragmentClaim<'_, '_, '_, '_>,
    ) -> Result<(), FundedCaptureError<Native>> {
        self.roots.push(value.clone());
        claim
            .validate_native_scope(&self.scope)
            .map_err(CaptureRunHostError::from)?;
        Ok(())
    }
    fn retain_generated_fragment_output(
        &mut self,
        value: &Value,
        claim: &CapturePrefillFragmentClaim<'_, '_, '_, '_>,
    ) -> Result<(), FundedCaptureError<Native>> {
        self.roots.push(value.clone());
        claim
            .validate_native_scope(&self.scope)
            .map_err(CaptureRunHostError::from)?;
        Ok(())
    }
}
fn usage(n: usize) -> CaptureUsage {
    CaptureUsage {
        captures: 1,
        retained_bytes: n as u64 * 4,
        host_bytes: n as u64 * 4,
        encoded_bytes: 4096,
    }
}
fn backend(run: &WorkingMemoryFundingRun) -> Backend {
    Backend {
        scope: run.scope().unwrap(),
        roots: vec![],
        transforms: 0,
        preflights: 0,
    }
}
fn epoch(k: u64) -> DistributedCommitEpoch {
    DistributedCommitEpoch::new(k + 1).unwrap()
}
fn chunk(k: u64) -> PrefillChunk {
    let start = k * 2;
    let end = (start + 2).min(3);
    PrefillChunk {
        input: start..end,
        position: 7 + start,
        output: OutputDemand::LastPosition.for_chunk(end == 3),
    }
}
fn enter(o: &mut dyn ActivationObserver<Value, Error>, k: u64) {
    o.begin_prefill_chunk(&chunk(k)).unwrap();
    o.prepare_transaction(epoch(k), ExpertPass::Prefill)
        .unwrap();
}
fn commit(o: &mut dyn ActivationObserver<Value, Error>, k: u64) {
    o.complete_transaction(epoch(k)).unwrap();
    o.finish_transaction(epoch(k), true);
}
fn values(record: &CaptureRecord) -> &[f32] {
    let Some(CapturePayload::SharedTensor(t)) = &record.payload else {
        panic!("shared tensor");
    };
    let TensorObservationData::F32(v) = t.data() else {
        panic!("f32");
    };
    v
}

#[test]
fn bound_fragments_keep_one_frame_full_quota_and_batch_scatter_then_decode() {
    let source = admitted(5, 100, CaptureLimitPolicy::Fail);
    let selected = selected(&source);
    let bound = selected.bind_geometry(geometry()).unwrap();
    let (pool, r, run) = fresh(&source, 0);
    let mut bank = bank(&source, &r, &run);
    let mut native = backend(&run);
    let used = pool.used_bytes().unwrap();
    bank.with_prefill_observer(&mut native, bound, &Error::Capture, |o| {
        assert!(!o.requires_sequence_readout());
        for k in 0..2 {
            enter(o, k);
            o.observe("layer.0.output", &value(k * 2, (3 - k * 2).min(2)))
                .unwrap();
            commit(o, k);
        }
        o.finish_prefill(true);
    })
    .unwrap();
    assert_eq!(bank.spent_steps(), 1);
    assert_eq!(pool.used_bytes().unwrap(), used);
    assert_eq!(bank.usage().captures, 2);
    let frame = bank.take_shared_step().unwrap().unwrap();
    assert_eq!(frame.outcome(), CaptureStepOutcome::Committed);
    assert_eq!(
        values(&frame.records()[0]),
        &[0., 1., 10., 11., 20., 21., 100., 101., 110., 111., 120., 121.]
    );
    assert_eq!(values(&frame.records()[1]), &[0., 1., 10., 11., 20.]);
    bank.with_observer(&mut native, 1, &Error::Capture, |o| {
        o.prepare_transaction(epoch(2), ExpertPass::Decode).unwrap();
        o.observe("layer.0.output", &value(3, 1)).unwrap();
        o.complete_transaction(epoch(2)).unwrap();
        o.finish_transaction(epoch(2), true);
    })
    .unwrap();
    assert_eq!(
        bank.take_shared_step().unwrap().unwrap().prediction_index(),
        1
    );
    drop((frame, bank, r));
    native.scope.certify().unwrap();
    drop(run);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn missing_and_duplicate_zero_hooks_abort_without_refunding_full_row() {
    for duplicate in [false, true] {
        let source = admitted(0, 100, CaptureLimitPolicy::Fail);
        let selected = selected(&source);
        let (pool, r, run) = fresh(&source, 0);
        let mut bank = bank(&source, &r, &run);
        let mut native = backend(&run);
        bank.with_prefill_observer(
            &mut native,
            selected.bind_geometry(geometry()).unwrap(),
            &Error::Capture,
            |o| {
                enter(o, 0);
                o.observe("layer.0.output", &value(0, 2)).unwrap();
                if duplicate {
                    assert!(o.observe("layer.0.output", &value(0, 2)).is_err());
                } else {
                    commit(o, 0);
                    enter(o, 1);
                    assert!(o.complete_transaction(epoch(1)).is_err());
                }
            },
        )
        .unwrap();
        assert_eq!(bank.usage().captures, 2);
        assert_eq!(
            bank.take_shared_step().unwrap().unwrap().outcome(),
            CaptureStepOutcome::Aborted
        );
        drop((bank, r));
        native.scope.certify().unwrap();
        drop(run);
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}

#[test]
fn exact_source_and_annotation_reject_before_claiming() {
    let source = admitted(5, 100, CaptureLimitPolicy::Fail);
    let foreign = admitted(5, 100, CaptureLimitPolicy::Fail);
    let selection = selected(&source);
    let other = selected(&foreign);
    let (_pool, r, run) = fresh(&source, 0);
    let mut bank = bank(&source, &r, &run);
    let mut native = backend(&run);
    assert!(bank
        .with_prefill_observer(
            &mut native,
            other.bind_geometry(geometry()).unwrap(),
            &Error::Capture,
            |_| panic!("foreign callback")
        )
        .is_err());
    assert_eq!(bank.spent_steps(), 0);
    bank.with_prefill_observer(
        &mut native,
        selection.bind_geometry(geometry()).unwrap(),
        &Error::Capture,
        |o| {
            assert!(o
                .prepare_transaction(epoch(0), ExpertPass::Prefill)
                .is_err());
        },
    )
    .unwrap();
    assert_eq!(bank.spent_steps(), 0);
    assert!(!bank.has_pending_step());
    drop((bank, r));
    native.scope.certify().unwrap();
    drop(run);
}

#[test]
fn full_row_skip_and_fail_do_not_repeat_quota_or_native_transforms() {
    for limit in [CaptureLimitPolicy::Skip, CaptureLimitPolicy::Fail] {
        let source = admitted(5, 0, limit);
        let selected = selected(&source);
        let (_pool, r, run) = fresh(&source, 0);
        let mut bank = bank(&source, &r, &run);
        let mut native = backend(&run);
        bank.with_prefill_observer(
            &mut native,
            selected.bind_geometry(geometry()).unwrap(),
            &Error::Capture,
            |o| {
                for k in 0..2 {
                    enter(o, k);
                    let result = o.observe("layer.0.output", &value(k * 2, (3 - k * 2).min(2)));
                    if limit == CaptureLimitPolicy::Fail {
                        assert!(result.is_err());
                        return;
                    }
                    result.unwrap();
                    commit(o, k);
                }
                o.finish_prefill(true);
            },
        )
        .unwrap();
        assert_eq!(bank.usage().captures, 0);
        assert_eq!(native.transforms, 0);
        let frame = bank.take_shared_step().unwrap().unwrap();
        assert_eq!(
            frame.outcome(),
            if limit == CaptureLimitPolicy::Skip {
                CaptureStepOutcome::Committed
            } else {
                CaptureStepOutcome::Aborted
            }
        );
        drop((frame, bank, r));
        native.scope.certify().unwrap();
        drop(run);
    }
}

struct Factory<'a> {
    prototype: &'a Value,
    calls: usize,
    visits: usize,
    fail: bool,
    panic: bool,
    cause: Arc<()>,
}
impl Factory<'_> {
    fn descriptor(&self) -> GeneratedCaptureSource {
        generated_capture_source(
            &BlockFp8InputReconstructionPlan::new(&self.prototype.shape)
                .unwrap()
                .logical_capture_source()
                .unwrap(),
        )
    }
}
impl RetainedGeneratedTensorFactory<Value, Error> for Factory<'_> {
    fn program(&self) -> GeneratedTensorProgram<'_> {
        GeneratedTensorProgram::BlockFp8Input(
            BlockFp8InputReconstructionPlan::new(&self.prototype.shape).unwrap(),
        )
    }
    fn visit_sources(
        &mut self,
        sink: &mut dyn FnMut(GeneratedTensorSourceRole, &Value) -> Result<(), Error>,
    ) -> Result<(), Error> {
        self.visits += 1;
        sink(GeneratedTensorSourceRole::CompactValues, self.prototype)?;
        sink(GeneratedTensorSourceRole::BlockScales, self.prototype)
    }
    fn generate(
        &mut self,
        sink: &mut dyn FnMut(&Value) -> Result<(), Error>,
    ) -> Result<Value, Error> {
        self.calls += 1;
        for i in 0..7 {
            sink(self.prototype)?;
            if i == 2 {
                assert!(!self.panic, "after third retained output");
                if self.fail {
                    return Err(Error::Factory(self.cause.clone()));
                }
            }
        }
        Ok(self.prototype.clone())
    }
}
#[test]
fn generated_full_creation_is_charged_per_selection_factory_once_per_chunk_even_preview_zero() {
    let source = admitted(0, 100, CaptureLimitPolicy::Fail);
    let selected = selected(&source);
    let bound = selected.bind_geometry(geometry()).unwrap();
    let (_pool, r, run) = fresh(&source, 0);
    let mut bank = bank(&source, &r, &run);
    let mut native = backend(&run);
    bank.with_prefill_observer(&mut native, bound, &Error::Capture, |o| {
        for k in 0..2 {
            enter(o, k);
            let v = value(k * 2, (3 - k * 2).min(2));
            let mut f = Factory {
                prototype: &v,
                calls: 0,
                visits: 0,
                fail: false,
                panic: false,
                cause: Arc::new(()),
            };
            o.observe_generated_retained("layer.0.output", &v, &f.descriptor(), &mut f)
                .unwrap();
            assert_eq!((f.calls, f.visits), (1, 1));
            commit(o, k);
        }
        o.finish_prefill(true);
    })
    .unwrap();
    assert_eq!(native.preflights, 4);
    assert_eq!(native.transforms, 4);
    assert_eq!(native.roots.len(), 18);
    assert_eq!(bank.usage().captures, 2);
    let policy = CapturePrefillObservationPolicy::from_bound(bound).unwrap();
    let mut expected = CaptureLedger::new(source.admission());
    expected.begin_step();
    CaptureObservationStep::new(source.admission(), CapturePhase::Prefill, 0)
        .unwrap()
        .reserve_metadata(&mut expected)
        .unwrap();
    for i in 0..2 {
        let row = policy.row(i).unwrap();
        let mut progress = row.initial_progress();
        assert_eq!(
            row.begin_hook(&mut progress, &chunk(0), "layer.0.output")
                .unwrap(),
            CapturePrefillHookDecision::First
        );
        row.reserve_first(
            &mut progress,
            &mut expected,
            row.full_generated_usage(usage(row.assembly().unwrap().logical_geometry().elements()))
                .unwrap(),
        )
        .unwrap();
    }
    assert_eq!(bank.usage(), expected.total());
    let frame = bank.take_shared_step().unwrap().unwrap();
    assert!(values(&frame.records()[1]).is_empty());
    drop((frame, bank, r));
    native.scope.certify().unwrap();
    drop(run);
}

#[test]
fn generated_failure_and_unwind_preserve_original_cause_partial_roots_and_aborted_frame() {
    for panic in [false, true] {
        let source = admitted(0, 100, CaptureLimitPolicy::Fail);
        let selected = selected(&source);
        let (_pool, r, run) = fresh(&source, 0);
        let mut bank = bank(&source, &r, &run);
        let mut native = backend(&run);
        let cause = Arc::new(());
        let v = value(0, 2);
        let result = catch_unwind(AssertUnwindSafe(|| {
            bank.with_prefill_observer(
                &mut native,
                selected.bind_geometry(geometry()).unwrap(),
                &Error::Capture,
                |o| {
                    enter(o, 0);
                    let mut f = Factory {
                        prototype: &v,
                        calls: 0,
                        visits: 0,
                        fail: true,
                        panic,
                        cause: cause.clone(),
                    };
                    let result =
                        o.observe_generated_retained("layer.0.output", &v, &f.descriptor(), &mut f);
                    if let Err(Error::Factory(actual)) = result {
                        assert!(Arc::ptr_eq(&actual, &cause));
                    } else {
                        panic!("original factory error");
                    }
                },
            )
            .unwrap()
        }));
        assert_eq!(result.is_err(), panic);
        assert_eq!(native.roots.len(), 5);
        assert_eq!(bank.usage().captures, 1);
        assert_eq!(
            bank.take_shared_step().unwrap().unwrap().outcome(),
            CaptureStepOutcome::Aborted
        );
        drop((bank, r));
        native.scope.certify().unwrap();
        drop(run);
    }
}

#[test]
fn final_outer_abort_keeps_completed_shared_buffer_and_logical_charge() {
    let source = admitted(5, 100, CaptureLimitPolicy::Fail);
    let selected = selected(&source);
    let (pool, r, run) = fresh(&source, 0);
    let mut bank = bank(&source, &r, &run);
    let mut native = backend(&run);
    bank.with_prefill_observer(
        &mut native,
        selected.bind_geometry(geometry()).unwrap(),
        &Error::Capture,
        |o| {
            for k in 0..2 {
                enter(o, k);
                o.observe("layer.0.output", &value(k * 2, (3 - k * 2).min(2)))
                    .unwrap();
                commit(o, k);
            }
            o.finish_prefill(false);
        },
    )
    .unwrap();
    let frame = bank.take_shared_step().unwrap().unwrap();
    assert_eq!(frame.outcome(), CaptureStepOutcome::Aborted);
    assert_eq!(frame.step_usage().captures, 2);
    let pointer = values(&frame.records()[0]).as_ptr();
    let alias = frame.clone();
    drop((frame, bank, r));
    native.scope.certify().unwrap();
    drop(run);
    assert!(pool.used_bytes().unwrap() > 0);
    assert_eq!(values(&alias.records()[0]).as_ptr(), pointer);
    drop(alias);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn enlarged_observer_controls_fit_exact_original_h_and_reject_one_short() {
    let source = admitted(5, 100, CaptureLimitPolicy::Fail);
    for shortage in [0, 1] {
        let (pool, r, run) = fresh(&source, shortage);
        let before = pool.used_bytes().unwrap();
        let result = run.prepare_capture_run(&r, CaptureRunHostPlan::prepare(&source).unwrap());
        if shortage == 1 {
            assert!(matches!(
                result,
                Err(CaptureRunHostError::Memory(
                    WorkingMemoryError::BudgetExceeded { .. }
                ))
            ));
            assert_eq!(pool.used_bytes().unwrap(), before);
        } else {
            drop(result.unwrap());
        }
        drop((r, run));
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}

#[test]
fn overquota_and_true_empty_slices_never_invoke_generated_factory() {
    for empty in [false, true] {
        let source = admitted_slice(
            0,
            if empty { 100 } else { 0 },
            CaptureLimitPolicy::Skip,
            empty,
        );
        let selected = selected(&source);
        let (_pool, r, run) = fresh(&source, 0);
        let mut bank = bank(&source, &r, &run);
        let mut native = backend(&run);
        bank.with_prefill_observer(
            &mut native,
            selected.bind_geometry(geometry()).unwrap(),
            &Error::Capture,
            |o| {
                for k in 0..2 {
                    enter(o, k);
                    let v = value(k * 2, (3 - k * 2).min(2));
                    let mut f = Factory {
                        prototype: &v,
                        calls: 0,
                        visits: 0,
                        fail: false,
                        panic: false,
                        cause: Arc::new(()),
                    };
                    o.observe_generated_retained("layer.0.output", &v, &f.descriptor(), &mut f)
                        .unwrap();
                    assert_eq!((f.calls, f.visits), (0, 0));
                    commit(o, k);
                }
                o.finish_prefill(true);
            },
        )
        .unwrap();
        assert_eq!(native.preflights, 0);
        assert_eq!(native.transforms, 0);
        assert!(native.roots.is_empty());
        assert_eq!(bank.usage().captures, if empty { 2 } else { 0 });
        let frame = bank.take_shared_step().unwrap().unwrap();
        assert_eq!(frame.outcome(), CaptureStepOutcome::Committed);
        if empty {
            assert!(values(&frame.records()[0]).is_empty());
        }
        drop((frame, bank, r));
        native.scope.certify().unwrap();
        drop(run);
    }
}

mod routed;
