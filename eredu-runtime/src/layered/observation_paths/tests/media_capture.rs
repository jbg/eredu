//! Collector contract tests. Actual selected native ingress is covered separately.
use super::*;
use crate::{
    capture::{CaptureSession, OrdinaryPrefillCapture},
    prefill::PrefillChunk,
};
use eredu_core::{capture::*, checkpoint::TensorDtype, *};
use std::cell::Cell;
fn paths() -> SharedLayeredObservationPaths {
    let mut value = source();
    Arc::get_mut(&mut value.0).unwrap().media_prefill =
        vec![PrefillObservationDeclaration::prepared_media_decoder(
            "block.output".into(),
            0,
            PrefillReadoutStage::BeforeReadout,
        )]
        .into_boxed_slice();
    value
}
fn limits() -> CaptureLimits {
    let usage = CaptureUsage {
        captures: 10,
        host_bytes: 1 << 20,
        retained_bytes: 1 << 20,
        encoded_bytes: 1 << 20,
    };
    CaptureLimits {
        per_step: usage,
        cumulative: usage,
        on_limit: CaptureLimitPolicy::Fail,
    }
}
fn binding(sliced: bool, empty: bool, limits: CaptureLimits) -> OrdinaryPrefillCapture {
    binding_selected(sliced, empty, false, limits)
}
fn binding_selected(
    sliced: bool,
    empty: bool,
    zero: bool,
    limits: CaptureLimits,
) -> OrdinaryPrefillCapture {
    let point = ObservationPoint {
        path: "block.output".into(),
        node_id: "block".into(),
        meaning: "decoder output".into(),
        value_type: ObservationValueType::Tensor,
        dtype: ObservationDtype::Floating,
        axes: Some(vec![
            TensorAxis {
                name: "sequence".into(),
                dimension: SymbolicDimension::Sequence,
            },
            TensorAxis {
                name: "hidden".into(),
                dimension: SymbolicDimension::Known(4),
            },
        ]),
        prefill: true,
        decode: true,
        requirements: vec![ObservationRequirement::ActivationHooks],
        position: ObservationPosition::BeforeIntervention,
        retained_bytes: None,
        host_bytes: None,
    };
    let caps = CaptureCapabilities {
        transformations: vec![
            CaptureTransformKind::FullTensor,
            CaptureTransformKind::Slice,
        ],
        max_histogram_bins: 0,
        conditions: vec![],
    };
    let support = ObservationSupportReport {
        schema_version: 1,
        capture: caps.clone(),
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
    let selection = CaptureSelection {
        id: "rows".into(),
        path: "block.output".into(),
        schedule: CaptureSchedule {
            // The strided slice addresses multiple prompt rows; decode hooks
            // expose one row. Full/zero selections still exercise decode mode.
            decode: !sliced,
            ..Default::default()
        },
        slices: if zero {
            vec![CaptureSlice {
                axis: "sequence".into(),
                start: 1,
                end: 1,
                stride: 1,
            }]
        } else if sliced {
            vec![
                CaptureSlice {
                    axis: "sequence".into(),
                    start: 1,
                    end: 5,
                    stride: 2,
                },
                CaptureSlice {
                    axis: "hidden".into(),
                    start: 1,
                    end: 4,
                    stride: 2,
                },
            ]
        } else {
            vec![]
        },
        transform: if sliced || zero {
            CaptureTransform::Slice
        } else {
            CaptureTransform::FullTensor
        },
    };
    let plan = CapturePlan {
        schema_version: 1,
        selections: if empty { vec![] } else { vec![selection] },
        limits,
    };
    let source = SharedCapturePlan::new(
        plan.admit_with_text_origin(
            &catalog,
            &support,
            &caps,
            CaptureRequestShape {
                batch: 1,
                prompt_tokens: 5,
                max_predictions: 3,
            },
            CaptureTextOrigin {
                cached_positions: 0,
            },
        )
        .unwrap(),
    );
    let selected = paths().prepare_media_capture_selection(&source).unwrap();
    OrdinaryPrefillCapture::new(
        selected,
        InferenceGeometry {
            batch_size: 1,
            cached_positions: 0,
            input_positions: 5,
            max_output_tokens: 3,
            prefill_chunk_positions: 2,
            output: OutputDemand::LastPosition,
        },
    )
    .unwrap()
}
#[derive(Debug, thiserror::Error)]
#[error("fragment failure")]
struct Failure;
#[derive(Default)]
struct Backend {
    validates: Cell<usize>,
    copies: Cell<usize>,
    fail: Option<usize>,
    dtype: Option<TensorDtype>,
}
impl CaptureBackend for Backend {
    type Tensor = TensorObservation;
    type Error = Failure;
    fn shape(&self, v: &Self::Tensor) -> Result<Vec<u64>, Failure> {
        Ok(v.shape().iter().map(|n| *n as u64).collect())
    }
    fn estimate(
        &self,
        _: &Self::Tensor,
        _: &CaptureSelection,
        _: &ResolvedCaptureSlice,
    ) -> Result<CaptureUsage, CaptureError> {
        unreachable!()
    }
    fn transform(
        &mut self,
        _: &Self::Tensor,
        _: &CaptureSelection,
        _: &ResolvedCaptureSlice,
    ) -> Result<CapturePayload, Failure> {
        unreachable!()
    }
    fn validate_capture_prefill_source(
        &self,
        v: &Self::Tensor,
        f: &CapturePrefillFragment<'_, '_>,
    ) -> Option<Result<TensorDtype, Failure>> {
        self.validates.set(self.validates.get() + 1);
        Some(if v.shape() == f.source_shape() {
            Ok(self.dtype.clone().unwrap_or(TensorDtype::F32))
        } else {
            Err(Failure)
        })
    }
    fn estimate_capture_prefill_fragment(
        &self,
        f: &CapturePrefillFragment<'_, '_>,
    ) -> Result<CaptureUsage, CaptureError> {
        Ok(CaptureUsage {
            host_bytes: (f.output_elements() * 4) as u64,
            retained_bytes: (f.source_shape().iter().product::<usize>() * 4) as u64,
            ..Default::default()
        })
    }
    fn capture_prefill_fragment(
        &mut self,
        v: &Self::Tensor,
        f: &CapturePrefillFragment<'_, '_>,
    ) -> Option<Result<TensorObservation, Failure>> {
        let count = self.copies.get() + 1;
        self.copies.set(count);
        if self.fail == Some(count) {
            return Some(Err(Failure));
        }
        let TensorObservationData::F32(data) = v.data() else {
            unreachable!()
        };
        let a = f.selection_axis(0).unwrap();
        let b = f.selection_axis(1).unwrap();
        let values = (a.start()..a.end())
            .step_by(a.stride())
            .flat_map(|r| {
                (b.start()..b.end())
                    .step_by(b.stride())
                    .map(move |c| data[r * 4 + c])
            })
            .collect();
        Some(Ok(TensorObservation::new(
            f.selected_shape().to_vec(),
            TensorObservationData::F32(values),
        )
        .unwrap()))
    }
}
fn source_tensor(start: u64, end: u64) -> TensorObservation {
    TensorObservation::new(
        vec![(end - start) as usize, 4],
        TensorObservationData::F32(
            (start..end)
                .flat_map(|r| (0..4).map(move |c| (10 * r + c + 1) as f32))
                .collect(),
        ),
    )
    .unwrap()
}
fn run(
    binding: OrdinaryPrefillCapture,
    backend: &mut Backend,
    omit: Option<usize>,
    commit: bool,
) -> (CaptureSession, bool) {
    let mut session =
        CaptureSession::with_ordinary_prefill(binding, &HostPreparationAuthority::default())
            .unwrap();
    let success = advance(&mut session, backend, omit, commit);
    (session, success)
}
fn advance(
    session: &mut CaptureSession,
    backend: &mut Backend,
    omit: Option<usize>,
    commit: bool,
) -> bool {
    advance_from_epoch(
        session,
        backend,
        omit,
        commit,
        DistributedCommitEpoch::FIRST,
    )
}
fn advance_from_epoch(
    session: &mut CaptureSession,
    backend: &mut Backend,
    omit: Option<usize>,
    commit: bool,
    mut epoch: DistributedCommitEpoch,
) -> bool {
    let mut success = true;
    for (index, (start, end)) in [(0, 2), (2, 4), (4, 5)].into_iter().enumerate() {
        let chunk = PrefillChunk {
            input: start..end,
            position: start,
            output: OutputDemand::LastPosition.for_chunk(end == 5),
        };
        session.begin_ordinary_prefill(&chunk).unwrap();
        if session
            .prepare_ordinary_prefill(epoch, crate::ExpertPass::Prefill)
            .is_err()
        {
            success = false;
            break;
        }
        if omit != Some(index)
            && session
                .observe(backend, "block.output", &source_tensor(start, end))
                .is_err()
        {
            session.finish_ordinary_chunk(epoch, false);
            success = false;
            break;
        }
        if session.complete_ordinary_prefill(epoch).is_err() {
            session.finish_ordinary_chunk(epoch, false);
            success = false;
            break;
        }
        assert!(
            session.take_test_frame().is_none(),
            "physical completion cannot publish the logical frame"
        );
        session.finish_ordinary_chunk(epoch, true);
        assert!(session.take_test_frame().is_none());
        epoch = epoch.next().unwrap();
    }
    session.finish_ordinary_prefill(success && commit);
    success
}
#[test]
fn prepared_media_one_frame_scattered_slice_empty_owner_and_terminal_barrier() {
    for sliced in [false, true] {
        let mut backend = Backend::default();
        let (mut session, ok) = run(binding(sliced, false, limits()), &mut backend, None, true);
        assert!(ok);
        let step = session.take_test_frame().unwrap();
        assert_eq!(step.records.len(), 1);
        assert_eq!(step.records[0].outcome, CaptureOutcome::Captured);
        let value = step.records[0]
            .payload
            .as_ref()
            .unwrap()
            .as_tensor()
            .unwrap();
        let expected = if sliced {
            vec![12., 14., 32., 34.]
        } else {
            (0..5)
                .flat_map(|r| (0..4).map(move |c| (10 * r + c + 1) as f32))
                .collect()
        };
        assert_eq!(value.data(), &TensorObservationData::F32(expected));
        assert_eq!(backend.validates.get(), 3);
        assert_eq!(backend.copies.get(), if sliced { 2 } else { 3 });
    }
    let mut backend = Backend::default();
    let (mut session, ok) = run(binding(false, true, limits()), &mut backend, None, true);
    assert!(ok);
    assert!(session.take_test_frame().unwrap().records.is_empty());
    assert_eq!(backend.copies.get(), 0);
}
#[test]
fn prepared_media_full_logical_quota_exact_short_skip_and_failure_keep_consumption() {
    let (mut complete, _) = run(
        binding(false, false, limits()),
        &mut Backend::default(),
        None,
        true,
    );
    let charged = complete.take_test_frame().unwrap().cumulative_usage;
    let mut exact = limits();
    exact.per_step = charged;
    exact.cumulative = charged;
    let (mut full, ok) = run(
        binding(false, false, exact.clone()),
        &mut Backend::default(),
        None,
        true,
    );
    assert!(ok);
    assert_eq!(full.take_test_frame().unwrap().cumulative_usage, charged);
    for skip in [false, true] {
        let mut short = exact.clone();
        short.per_step.host_bytes -= 1;
        short.cumulative.host_bytes -= 1;
        if skip {
            short.on_limit = CaptureLimitPolicy::Skip;
        }
        let mut backend = Backend::default();
        let (mut run, ok) = run(binding(false, false, short), &mut backend, None, true);
        assert_eq!(ok, skip);
        assert_eq!(backend.copies.get(), 0);
        if skip {
            assert!(matches!(
                run.take_test_frame().unwrap().records[0].outcome,
                CaptureOutcome::Skipped {
                    reason: CaptureSkipReason::Limit { .. }
                }
            ));
        }
    }
    for omit in [Some(1), None] {
        let mut backend = Backend {
            fail: if omit.is_none() { Some(2) } else { None },
            ..Default::default()
        };
        let (run, ok) = run(binding(false, false, limits()), &mut backend, omit, true);
        assert!(!ok);
        assert_eq!(run.cumulative_usage(), charged);
    }
    let (mut run, ok) = run(
        binding(false, false, limits()),
        &mut Backend::default(),
        None,
        false,
    );
    assert!(ok);
    assert_eq!(
        run.take_test_frame().unwrap().outcome,
        CaptureStepOutcome::Aborted
    );
    assert_eq!(run.cumulative_usage(), charged);
}
#[test]
fn prepared_media_equal_content_is_not_the_same_source_or_path_binding() {
    let one = binding(false, false, limits());
    let two = binding(false, false, limits());
    assert!(!one.source().same_storage(two.source()));
    assert!(one
        .validate(two.source(), &paths(), one.geometry())
        .is_err());
    let text = paths().prepare_capture_selection(one.source());
    assert!(
        matches!(text, Err(PreparedCaptureSelectionError::Undeclared { .. })),
        "media declarations cannot authorize ordinary text by shape"
    );
}

fn discovery(binding: &OrdinaryPrefillCapture) -> CaptureDiscovery {
    let points = binding.source().admission().points().to_vec();
    let support = points
        .iter()
        .map(|point| ObservationSupport {
            path: point.path.clone(),
            prefill: ObservationSupportStatus::Supported,
            decode: ObservationSupportStatus::Supported,
            floating_to_f32: true,
        })
        .collect();
    CaptureDiscovery {
        artifact_identity: "actual test source".into(),
        catalog: ObservationCatalog {
            schema_version: 1,
            points,
            completeness: DescriptionCompleteness::Complete,
        },
        support: ObservationSupportReport {
            schema_version: 1,
            points: support,
            capture: CaptureCapabilities {
                transformations: vec![
                    CaptureTransformKind::FullTensor,
                    CaptureTransformKind::Slice,
                ],
                max_histogram_bins: 0,
                conditions: vec![],
            },
        },
    }
}
#[test]
fn prepared_capture_checkpoint_authenticates_same_run_and_derived_child_not_equal_fresh_owners() {
    let bound = binding(false, false, limits());
    let discovery = discovery(&bound);
    let run =
        CaptureSession::with_ordinary_prefill(bound.clone(), &HostPreparationAuthority::default())
            .unwrap();
    let saved = run.checkpoint(&discovery).unwrap();
    saved.validate_pending_capture_destination(&run).unwrap();
    let unrelated =
        CaptureSession::with_ordinary_prefill(bound, &HostPreparationAuthority::default()).unwrap();
    assert!(
        saved
            .validate_pending_capture_destination(&unrelated)
            .is_err(),
        "same source and geometry do not prove child construction"
    );
    let foreign = CaptureSession::with_ordinary_prefill(
        binding(false, false, limits()),
        &HostPreparationAuthority::default(),
    )
    .unwrap()
    .checkpoint(&discovery)
    .unwrap();
    assert!(foreign.validate_pending_capture_destination(&run).is_err());
    let usage = saved.inherited_usage();
    let child = saved
        .fork(
            crate::capture::CaptureForkRequest {
                discovery: &discovery,
                max_predictions: 3,
                limits: limits(),
                intervention: None,
            },
            |_, _, _| Ok(CaptureUsage::default()),
        )
        .unwrap();
    saved.validate_pending_capture_destination(&child).unwrap();
    assert!(foreign
        .validate_pending_capture_destination(&child)
        .is_err());
    assert_eq!(child.cumulative_usage(), usage);
    assert_eq!(run.cumulative_usage(), usage);
    assert!(!child
        .shared_plan_source()
        .same_storage(run.shared_plan_source()));
}

#[test]
fn prepared_checkpoint_lineage_rejects_shared_plan_different_run_ledger_and_saved_frontier() {
    let binding = binding(false, false, limits());
    let discovery = discovery(&binding);
    let mut run = CaptureSession::with_ordinary_prefill(
        binding.clone(),
        &HostPreparationAuthority::default(),
    )
    .unwrap();
    let before = run.checkpoint(&discovery).unwrap();
    assert!(advance(&mut run, &mut Backend::default(), None, true));
    let frame = run.take_test_frame().unwrap();
    assert!(frame.cumulative_usage.host_bytes > 0);
    let completed = run.checkpoint(&discovery).unwrap();
    assert!(
        completed
            .validate_pending_capture_destination(&run)
            .is_err(),
        "completed checkpoint has no pending prefill"
    );
    run.restore(&before).unwrap();
    let after_restore = run.checkpoint(&discovery).unwrap();
    assert_eq!(after_restore.inherited_usage(), frame.cumulative_usage);
    let child = after_restore
        .fork(
            crate::capture::CaptureForkRequest {
                discovery: &discovery,
                max_predictions: 3,
                limits: limits(),
                intervention: None,
            },
            |_, _, _| Ok(CaptureUsage::default()),
        )
        .unwrap();
    after_restore
        .validate_pending_capture_destination(&child)
        .unwrap();
    assert!(
        before.validate_pending_capture_destination(&child).is_err(),
        "same run/source/frontier, different saved consumption"
    );
    assert!(
        completed
            .validate_pending_capture_destination(&child)
            .is_err(),
        "different saved frontier"
    );
    let unrelated =
        CaptureSession::with_ordinary_prefill(binding, &HostPreparationAuthority::default())
            .unwrap();
    let lower = unrelated.checkpoint(&discovery).unwrap();
    assert!(lower.inherited_usage().host_bytes < after_restore.inherited_usage().host_bytes);
    let lower_child = lower
        .fork(
            crate::capture::CaptureForkRequest {
                discovery: &discovery,
                max_predictions: 3,
                limits: limits(),
                intervention: None,
            },
            |_, _, _| Ok(CaptureUsage::default()),
        )
        .unwrap();
    assert!(
        after_restore
            .validate_pending_capture_destination(&lower_child)
            .is_err(),
        "same shared plan from a different lower-ledger run"
    );
    assert_eq!(run.cumulative_usage(), frame.cumulative_usage);
    assert_eq!(child.cumulative_usage(), frame.cumulative_usage);
    let mut used_child = child;
    assert!(advance(
        &mut used_child,
        &mut Backend::default(),
        None,
        true
    ));
    let consumed = used_child.take_test_frame().unwrap().cumulative_usage;
    assert!(after_restore
        .validate_pending_capture_destination(&used_child)
        .is_err());
    assert_eq!(used_child.cumulative_usage(), consumed);
}

#[test]
fn prepared_media_zero_length_validates_every_source_precision_and_escaped_tensor_retains_custody()
{
    let mut backend = Backend::default();
    let (mut session, success) = run(
        binding_selected(false, false, true, limits()),
        &mut backend,
        None,
        true,
    );
    assert!(success);
    assert_eq!(backend.validates.get(), 3);
    assert_eq!(backend.copies.get(), 0);
    let step = session.take_test_frame().unwrap();
    assert_eq!(step.records[0].outcome, CaptureOutcome::Captured);
    assert_eq!(
        step.records[0]
            .payload
            .as_ref()
            .unwrap()
            .as_tensor()
            .unwrap()
            .shape(),
        &[0, 4]
    );
    let mut wrong = Backend {
        dtype: Some(TensorDtype::I32),
        ..Default::default()
    };
    let (failed, success) = run(
        binding_selected(false, false, true, limits()),
        &mut wrong,
        None,
        true,
    );
    assert!(!success);
    assert_eq!(wrong.validates.get(), 1);
    assert_eq!(wrong.copies.get(), 0);
    assert!(failed.cumulative_usage().host_bytes > 0);
    struct Retired(Arc<AtomicUsize>);
    impl Drop for Retired {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    let retired = Arc::new(AtomicUsize::new(0));
    let host = HostPreparationAuthority::retain(Retired(retired.clone()));
    let mut session =
        CaptureSession::with_ordinary_prefill(binding(false, false, limits()), &host).unwrap();
    drop(host);
    assert!(advance(&mut session, &mut Backend::default(), None, true));
    let frame = session.take_test_frame().unwrap();
    let CapturePayload::SharedTensor(tensor) = frame.records[0].payload.as_ref().unwrap() else {
        panic!("closed tensor owner")
    };
    let escaped = tensor.clone();
    drop(frame);
    drop(session);
    assert_eq!(retired.load(Ordering::SeqCst), 0);
    drop(escaped);
    assert_eq!(retired.load(Ordering::SeqCst), 1);
}

// The test adapter only borrows the shared DTO. It never clones or exports it.
struct RetainedFrame(SharedCapturedStep);
impl std::ops::Deref for RetainedFrame {
    type Target = CapturedStep;
    fn deref(&self) -> &CapturedStep {
        self.0.as_step()
    }
}
trait RetainedDrain {
    fn take_test_frame(&mut self) -> Option<RetainedFrame>;
}
impl RetainedDrain for CaptureSession {
    fn take_test_frame(&mut self) -> Option<RetainedFrame> {
        self.take_shared_step().map(RetainedFrame)
    }
}
mod frame_custody;
mod observer_lifetime;
mod transforms;
