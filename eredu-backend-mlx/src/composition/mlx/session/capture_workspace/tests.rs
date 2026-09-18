mod invocation;
mod checkpoint;
mod generated;
mod histogram;
use super::*;
use eredu_core::*;
use eredu_nn::{Tensor, workspace::*};
use std::cell::Cell;
fn admitted_limited(
    axes: Vec<SymbolicDimension>,
    transform: CaptureTransform,
    slices: Vec<CaptureSlice>,
    captures: u64,
    limit: CaptureLimitPolicy,
) -> AdmittedCapturePlan {
    let point = ObservationPoint {
        path: "block.output".into(),
        node_id: "block".into(),
        meaning: "actual activation".into(),
        value_type: ObservationValueType::Tensor,
        dtype: ObservationDtype::Floating,
        axes: Some(
            axes.into_iter()
                .enumerate()
                .map(|(index, dimension)| TensorAxis {
                    name: format!("axis{index}"),
                    dimension,
                })
                .collect(),
        ),
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
    let usage = CaptureUsage {
        captures,
        retained_bytes: u64::MAX,
        host_bytes: u64::MAX,
        encoded_bytes: u64::MAX,
    };
    CapturePlan {
        schema_version: 1,
        selections: vec![CaptureSelection {
            id: "tensor".into(),
            path: "block.output".into(),
            schedule: CaptureSchedule {
                prefill: false,
                ..Default::default()
            },
            slices,
            transform,
        }],
        limits: CaptureLimits {
            per_step: usage,
            cumulative: usage,
            physical_native_bytes: None,
            on_limit: limit,
        },
    }
    .admit(
        &catalog,
        &support,
        &CaptureCapabilities {
            transformations: vec![
                CaptureTransformKind::Preview,
                CaptureTransformKind::Slice,
                CaptureTransformKind::FullTensor,
                CaptureTransformKind::Summary,
                CaptureTransformKind::Histogram,
            ],
            max_histogram_bins: 2,
            physical_native_limit: false,
            conditions: vec![],
        },
        CaptureRequestShape {
            batch: 1,
            prompt_tokens: 3,
            max_predictions: 4,
        },
    )
    .unwrap()
}
fn admitted(
    axes: Vec<SymbolicDimension>,
    transform: CaptureTransform,
    slices: Vec<CaptureSlice>,
) -> AdmittedCapturePlan {
    admitted_limited(axes, transform, slices, 10, CaptureLimitPolicy::Fail)
}
#[derive(Clone, Copy, Debug, Default)]
struct Facts {
    missing_tensor: bool,
    missing_host: bool,
}
impl WorkspaceMechanisms for Facts {
    fn operation_bound(&self, op: &WorkspaceOperation) -> Result<Option<WorkspaceOperationBound>> {
        let storage = match &op.kind {
            WorkspaceOperationKind::StaticSlice { .. } => {
                WorkspaceOutputStorage::AliasInput(0)
            }
            WorkspaceOperationKind::View("reshape") => {
                WorkspaceOutputStorage::AllocateOrAliasInputs {
                    bytes: op.outputs[0].bytes()?,
                    inputs: vec![0],
                }
            }
            WorkspaceOperationKind::Elementwise("capture_cast_f32") => {
                if self.missing_tensor {
                    return Ok(None);
                }
                WorkspaceOutputStorage::Allocate(op.outputs[0].bytes()?)
            }
            _ => return Ok(None),
        };
        Ok(Some(WorkspaceOperationBound {
            outputs: vec![storage],
            scratch_bytes: 0,
            assumptions:
                "neutral test facts: alias selection, possible reshape copy, independent cast"
                    .into(),
        }))
    }
    fn host_workspace_bound(&self, op: &WorkspaceOperation) -> Result<Option<WorkspaceHostBound>> {
        if self.missing_host
            && matches!(
                op.kind,
                WorkspaceOperationKind::Elementwise("capture_cast_f32")
            )
        {
            return Ok(None);
        }
        Ok(Some(WorkspaceHostBound {
            bytes: 0,
            assumptions: "neutral test mechanism has no numerical host staging".into(),
        }))
    }
}
const FACT_ASSUMPTIONS: &[u8] = b"neutral fixture exact output allocation and alias facts";

// The metadata-recording branch uses finite fact emission, with the same
// physical effects as this fixture's ordinary mechanism.
impl WorkspaceFactMechanisms for Facts {
    type Error = std::convert::Infallible;
    fn operation_facts(
        &self,
        op: WorkspaceOperationView<'_>,
    ) -> std::result::Result<Option<WorkspaceOperationFacts>, Self::Error> {
        let aliases = match op.kind {
            WorkspaceOperationKindView::StaticSlice { .. } => 0,
            WorkspaceOperationKindView::View("reshape") => 1,
            WorkspaceOperationKindView::Elementwise("capture_cast_f32") if !self.missing_tensor => {
                0
            }
            _ => return Ok(None),
        };
        Ok(Some(WorkspaceOperationFacts {
            layout: WorkspaceEffectLayout {
                outputs: 1,
                aliases,
                assumption_bytes: FACT_ASSUMPTIONS.len(),
            },
            scratch_bytes: 0,
        }))
    }
    fn write_operation_facts(
        &self,
        op: WorkspaceOperationView<'_>,
        destination: WorkspaceEffectDestination<'_>,
    ) -> std::result::Result<Option<WorkspaceOperationFacts>, Self::Error> {
        let Some(facts) = self.operation_facts(op)? else {
            return Ok(None);
        };
        destination.validate(facts.layout).unwrap();
        destination.assumptions.copy_from_slice(FACT_ASSUMPTIONS);
        destination.outputs[0] = match op.kind {
            WorkspaceOperationKindView::StaticSlice { .. } => {
                WorkspaceOutputEffect::AliasInput(0)
            }
            WorkspaceOperationKindView::View("reshape") => {
                destination.aliases[0] = 0;
                WorkspaceOutputEffect::AllocateOrAliasInputs {
                    bytes: op.outputs.get(0).unwrap().bytes().unwrap(),
                    alias_start: 0,
                    alias_count: 1,
                }
            }
            _ => WorkspaceOutputEffect::Allocate(op.outputs.get(0).unwrap().bytes().unwrap()),
        };
        Ok(Some(facts))
    }
    fn host_facts(
        &self,
        op: WorkspaceOperationView<'_>,
    ) -> std::result::Result<Option<WorkspaceHostFacts>, Self::Error> {
        if self.missing_host
            && matches!(
                op.kind,
                WorkspaceOperationKindView::Elementwise("capture_cast_f32")
            )
        {
            return Ok(None);
        }
        Ok(Some(WorkspaceHostFacts {
            bytes: 0,
            assumption_bytes: FACT_ASSUMPTIONS.len(),
        }))
    }
    fn write_host_facts(
        &self,
        op: WorkspaceOperationView<'_>,
        destination: WorkspaceHostDestination<'_>,
    ) -> std::result::Result<Option<WorkspaceHostFacts>, Self::Error> {
        let Some(facts) = self.host_facts(op)? else {
            return Ok(None);
        };
        destination.validate(facts).unwrap();
        destination.assumptions.copy_from_slice(FACT_ASSUMPTIONS);
        Ok(Some(facts))
    }
}

fn imported(
    context: &WorkspaceContext,
    shape: &[i32],
    dtype: WorkspaceDtype,
    bytes: Option<u64>,
) -> (WorkspaceTensor, WorkspaceExistingStorage) {
    let storage = WorkspaceExistingStorage::new(bytes, context);
    let value = WorkspaceTensor::existing_with_storage(
        WorkspaceLayout::new(shape, dtype).unwrap(),
        &storage,
        context,
    )
    .unwrap();
    (value, storage)
}
fn casts(report: &WorkspaceTraceReport) -> usize {
    report
        .operations
        .iter()
        .filter(|op| {
            matches!(
                op.kind,
                WorkspaceOperationKind::Elementwise("capture_cast_f32")
            )
        })
        .count()
}

fn geometry() -> InferenceGeometry {
    InferenceGeometry {
        batch_size: 1,
        cached_positions: 0,
        input_positions: 3,
        max_output_tokens: 4,
        prefill_chunk_positions: 2,
        output: OutputDemand::LastPosition,
    }
}
fn source(
    shape: &[usize],
    transform: CaptureTransform,
    slices: Vec<CaptureSlice>,
) -> SharedCapturePlan {
    SharedCapturePlan::new(admitted(
        shape
            .iter()
            .copied()
            .map(SymbolicDimension::Known)
            .collect(),
        transform,
        slices,
    ))
}
fn prefill(start: u64, end: u64) -> InferenceWorkspaceSpan {
    InferenceWorkspaceSpan::Prefill(eredu_runtime::prefill::PrefillChunk {
        input: start..end,
        position: start,
        output: if end == 3 {
            OutputDemand::LastPosition
        } else {
            OutputDemand::StateOnly
        },
    })
}
fn decode(p: u64) -> InferenceWorkspaceSpan {
    InferenceWorkspaceSpan::Decode {
        index: p - 1,
        position: 3 + p - 1,
        output: OutputDemand::LastPosition,
    }
}
fn begin(o: &mut CaptureWorkspaceObserver<'_>, context: &WorkspaceContext) {
    assert!(!o.requires_sequence_readout());
    assert!(
        !o.begin_span(geometry(), &prefill(0, 2), 0, context)
            .unwrap()
    );
    let metadata = o.ledger.total();
    assert!(metadata.host_bytes > 0);
    assert!(
        !o.begin_span(geometry(), &prefill(2, 3), 0, context)
            .unwrap()
    );
    assert_eq!(
        o.ledger.total(),
        metadata,
        "p0 is one logical row across both chunks"
    );
    assert!(o.begin_span(geometry(), &decode(1), 1, context).unwrap());
    assert_eq!(o.ledger.total(), metadata.checked_mul(2).unwrap());
}
fn typed<E: std::error::Error + 'static>(
    error: &eredu_nn::Error,
    test: impl Fn(&E) -> bool,
) -> bool {
    let mut current: Option<&(dyn std::error::Error + 'static)> = Some(error);
    while let Some(e) = current {
        if let Some(e) = e.downcast_ref::<E>() {
            return test(e);
        }
        current = e.source();
    }
    false
}

#[test]
fn closed_future_capture_keeps_imported_backing_and_roots_through_whole_span() {
    let cases = [
        (vec![2, 3], CaptureTransform::FullTensor, vec![], vec![2, 3]),
        (
            vec![2, 3],
            CaptureTransform::Slice,
            vec![CaptureSlice {
                axis: "axis1".into(),
                start: 1,
                end: 3,
                stride: 1,
            }],
            vec![2, 2],
        ),
        (
            vec![2, 3],
            CaptureTransform::Preview { max_elements: 3 },
            vec![],
            vec![3],
        ),
        (vec![], CaptureTransform::FullTensor, vec![], vec![]),
        (
            vec![2, 3],
            CaptureTransform::Preview { max_elements: 0 },
            vec![],
            vec![0],
        ),
    ];
    for (shape, transform, slices, expected) in cases {
        let source = source(&shape, transform, slices);
        let context = WorkspaceContext::new(Facts::default());
        let (mut observer, h) =
            CaptureWorkspaceObserver::new(&source, geometry(), &context).unwrap();
        assert!(h.initialization_peak_bytes() > 0);
        assert!(std::ptr::eq(h.source(), &source));
        begin(&mut observer, &context);
        let dimensions = shape.iter().map(|&n| n as i32).collect::<Vec<_>>();
        let (value, storage) = imported(&context, &dimensions, WorkspaceDtype::Float32, Some(4096));
        let borrowed = WorkspaceBorrowedStorage::new(&context, [&storage]).unwrap();
        context.set_borrowed_storage(borrowed.clone()).unwrap();
        context.begin_state_span([&value]).unwrap();
        observer.observe("block.output", &value).unwrap();
        assert_eq!(observer.roots.last().unwrap().shape(), expected);
        assert_eq!(observer.ledger.total().captures, 1);
        let duplicate = observer.observe("block.output", &value).unwrap_err();
        assert!(typed(&duplicate, |e: &CaptureProtocolError| matches!(
            e,
            CaptureProtocolError::Duplicate
        )));
        drop(value);
        drop(storage);
        let mut roots = Vec::new();
        observer.visit_retained(&mut |v| roots.push(v.clone()));
        assert_eq!(roots.len(), 2);
        let report = context.report(&roots).unwrap();
        assert_eq!(
            casts(&report),
            usize::from(expected.iter().product::<i32>() != 0)
        );
        assert!(report.state.as_ref().unwrap().retained_bytes.unwrap() >= 4096);
        assert!(
            report
                .residual
                .as_ref()
                .unwrap()
                .borrowed_storage
                .same_identity(&borrowed)
        );
        drop(roots);
        observer.end_span(&decode(1), &context).unwrap();
        assert!(observer.roots.is_empty());
        context
            .begin_state_span(std::iter::empty::<&WorkspaceTensor>())
            .unwrap();
        assert_eq!(
            context.report(&[]).unwrap().state.unwrap().retained_bytes,
            Some(0)
        );
    }
}

#[test]
fn actual_source_context_shape_and_dtype_reject_before_quota_or_trace() {
    let source = source(&[2, 3], CaptureTransform::FullTensor, vec![]);
    for wrong in 0..3 {
        let context = WorkspaceContext::new(Facts::default());
        let other = WorkspaceContext::new(Facts::default());
        let (mut observer, _) =
            CaptureWorkspaceObserver::new(&source, geometry(), &context).unwrap();
        begin(&mut observer, &context);
        let (value, _) = imported(
            if wrong == 0 { &other } else { &context },
            if wrong == 1 { &[2, 4] } else { &[2, 3] },
            if wrong == 2 {
                WorkspaceDtype::Int32
            } else {
                WorkspaceDtype::Float32
            },
            Some(4096),
        );
        context
            .begin_state_span(std::iter::empty::<&WorkspaceTensor>())
            .unwrap();
        let quota = observer.ledger.total();
        assert!(observer.observe("block.output", &value).is_err());
        assert_eq!(observer.ledger.total(), quota);
        assert!(context.report(&[]).unwrap().operations.is_empty());
        assert!(observer.roots.is_empty());
    }
    let context = WorkspaceContext::new(Facts::default());
    let mut wrong = geometry();
    wrong.cached_positions = 1;
    assert!(CaptureWorkspaceObserver::new(&source, wrong, &context).is_err());
    let (mut observer, _) = CaptureWorkspaceObserver::new(&source, geometry(), &context).unwrap();
    assert!(
        observer
            .begin_span(geometry(), &decode(1), 1, &context)
            .is_err()
    );
    assert!(
        observer
            .begin_span(geometry(), &prefill(1, 3), 0, &context)
            .is_err()
    );
}

#[test]
fn missing_selected_tensor_host_or_imported_facts_never_become_complete() {
    for missing in 0..3 {
        let source = source(&[2, 3], CaptureTransform::FullTensor, vec![]);
        let context = WorkspaceContext::new(Facts {
            missing_tensor: missing == 0,
            missing_host: missing == 1,
        });
        let (mut observer, _) =
            CaptureWorkspaceObserver::new(&source, geometry(), &context).unwrap();
        begin(&mut observer, &context);
        let (value, _) = imported(
            &context,
            &[2, 3],
            WorkspaceDtype::Float32,
            if missing == 2 { None } else { Some(4096) },
        );
        context.begin_state_span([&value]).unwrap();
        observer.observe("block.output", &value).unwrap();
        let report = context.report(&observer.roots).unwrap();
        match missing {
            0 => assert!(report.tensor_buffers.total_bytes.is_none()),
            1 => assert!(report.host_workspace_bytes.is_none()),
            _ => assert!(report.state.as_ref().unwrap().retained_bytes.is_none()),
        }
        observer.end_span(&decode(1), &context).unwrap();
    }
}

#[test]
fn selected_generated_source_is_typed_unfinished_and_unselected_factory_stays_lazy() {
    let source = source(&[2, 3], CaptureTransform::FullTensor, vec![]);
    let context = WorkspaceContext::new(Facts::default());
    let (mut observer, _) = CaptureWorkspaceObserver::new(&source, geometry(), &context).unwrap();
    let (value, _) = imported(&context, &[2, 3], WorkspaceDtype::Float32, Some(24));
    context.begin_state_span([&value]).unwrap();
    let calls = Cell::new(0);
    let mut factory = || {
        calls.set(calls.get() + 1);
        Ok(value.clone())
    };
    let generated = GeneratedCaptureSource {
        creation_bytes: 24,
        source_dtype: Some(eredu_core::checkpoint::TensorDtype::F32),
    };
    assert!(
        !observer
            .begin_span(geometry(), &prefill(0, 2), 0, &context)
            .unwrap()
    );
    observer
        .observe_generated("block.output", &value, &generated, &mut factory)
        .unwrap();
    assert_eq!(calls.get(), 0, "scheduled-off selected factory stays lazy");
    assert!(
        !observer
            .begin_span(geometry(), &prefill(2, 3), 0, &context)
            .unwrap()
    );
    observer
        .begin_span(geometry(), &decode(1), 1, &context)
        .unwrap();
    observer
        .observe_generated("unrelated", &value, &generated, &mut factory)
        .unwrap();
    let quota = observer.ledger.total();
    let failure = observer
        .observe_generated("block.output", &value, &generated, &mut factory)
        .unwrap_err();
    assert!(typed(&failure, |e: &CaptureProtocolError| matches!(
        e,
        CaptureProtocolError::GeneratedSource
    )));
    assert_eq!(calls.get(), 0);
    assert_eq!(observer.ledger.total(), quota);
    assert!(context.report(&[]).unwrap().operations.is_empty());
}

#[test]
fn cumulative_skip_fail_and_candidate_retries_do_not_spend_other_ledgers_or_trace_skips() {
    for limit in [CaptureLimitPolicy::Skip, CaptureLimitPolicy::Fail] {
        let source = SharedCapturePlan::new(admitted_limited(
            vec![SymbolicDimension::Known(2), SymbolicDimension::Known(3)],
            CaptureTransform::FullTensor,
            vec![],
            1,
            limit,
        ));
        for _candidate in 0..2 {
            let context = WorkspaceContext::new(Facts::default());
            let (mut observer, _) =
                CaptureWorkspaceObserver::new(&source, geometry(), &context).unwrap();
            begin(&mut observer, &context);
            let (value, _) = imported(&context, &[2, 3], WorkspaceDtype::Float32, Some(24));
            context.begin_state_span([&value]).unwrap();
            observer.observe("block.output", &value).unwrap();
            assert_eq!(observer.ledger.total().captures, 1);
            assert_eq!(casts(&context.report(&observer.roots).unwrap()), 1);
            observer.end_span(&decode(1), &context).unwrap();
            observer
                .begin_span(geometry(), &decode(2), 2, &context)
                .unwrap();
            context.begin_state_span([&value]).unwrap();
            let result = observer.observe("block.output", &value);
            if limit == CaptureLimitPolicy::Fail {
                assert!(typed(&result.unwrap_err(), |e: &CaptureError| matches!(
                    e,
                    CaptureError::Limit {
                        cumulative: true,
                        ..
                    }
                )));
            } else {
                result.unwrap();
            }
            assert_eq!(observer.ledger.total().captures, 1);
            assert!(context.report(&[]).unwrap().operations.is_empty());
            assert!(observer.roots.is_empty());
        }
    }
}

#[test]
fn actual_transfer_counter_preserves_skip_empty_and_cumulative_capture_rules() {
    for checked in [false, true] {
        for maximum in [0, 1] {
            for empty in [false, true] {
                let source = SharedCapturePlan::new(admitted_limited(
                    vec![
                        SymbolicDimension::Known(1),
                        SymbolicDimension::Known(1),
                        SymbolicDimension::Known(8),
                    ],
                    if empty {
                        CaptureTransform::Preview { max_elements: 0 }
                    } else {
                        CaptureTransform::FullTensor
                    },
                    vec![],
                    maximum,
                    CaptureLimitPolicy::Skip,
                ));
                let context = if checked {
                    WorkspaceContext::new_recording_facts(Facts::default())
                } else {
                    WorkspaceContext::new(Facts::default())
                };
                let transfers = Cell::new(CaptureNativePopulation::default());
                let (mut observer, _) = CaptureWorkspaceObserver::prepare(
                    &source,
                    geometry(),
                    &context,
                    None,
                    Some(&transfers),
                    None,
                    None,
                )
                .unwrap();
                if checked {
                    assert!(context.metadata_census().unwrap().context_bytes() > 0);
                }
                begin(&mut observer, &context);
                let (value, _backing) =
                    imported(&context, &[1, 1, 8], WorkspaceDtype::Float32, Some(32));
                context.begin_state_span([&value]).unwrap();
                observer.observe("block.output", &value).unwrap();
                assert_eq!(
                    transfers.get().publications,
                    maximum as usize,
                    "an accepted empty tensor still performs the actual source/output completion pair"
                );
                let consumption = observer.ledger.total();
                let duplicate = observer.observe("block.output", &value);
                if maximum != 0 {
                    assert!(typed(
                        &duplicate.unwrap_err(),
                        |cause: &CaptureProtocolError| matches!(
                            cause,
                            CaptureProtocolError::Duplicate
                        )
                    ));
                }
                assert_eq!(transfers.get().publications, maximum as usize);
                assert_eq!(
                    observer.ledger.total(),
                    consumption,
                    "the same hook cannot repeat a consumed transfer"
                );
                let report = context.finish_report(&observer.roots).unwrap();
                if maximum == 0 {
                    assert!(report.operations.is_empty());
                }
                observer.end_span(&decode(1), &context).unwrap();
                observer
                    .begin_span(geometry(), &decode(2), 2, &context)
                    .unwrap();
                assert_eq!(
                    transfers.get().publications,
                    0,
                    "each genuine span has its own completion population"
                );
                context.begin_state_span([&value]).unwrap();
                observer.observe("block.output", &value).unwrap();
                assert_eq!(
                    transfers.get().publications,
                    0,
                    "the prior accepted value still consumes cumulative quota"
                );
                assert_eq!(observer.ledger.total().captures, maximum);
            }
        }
    }
}

mod vocabulary;
