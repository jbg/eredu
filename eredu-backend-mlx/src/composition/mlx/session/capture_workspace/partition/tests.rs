#[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
#[test]
fn projected_prefill_quotes_actual_local_sources_and_preserves_raw_additive_reductions() {
    use super::*;
    use crate::backend::nn::workspace::{
        MlxCpuMatmulMechanism, MlxCpuWorkspaceMechanisms, MlxMetalWorkspaceMechanisms,
    };
    use eredu_core::*;
    use eredu_nn::workspace::*;
    use eredu_runtime::capture::partition::{
        PartitionCaptureProducer, PartitionCaptureReceiptLimits,
    };
    let native = MlxMetalWorkspaceMechanisms::current_host().unwrap();
    let selected =
        MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles).unwrap();
    let cpu = MlxCpuWorkspaceMechanisms::new(native.allocation(), selected);
    for transform in [
        CaptureTransform::Slice,
        CaptureTransform::Preview { max_elements: 5 },
        CaptureTransform::Summary,
        CaptureTransform::Histogram {
            edges: vec![-1.0, 0.25, 2.0],
        },
    ] {
        for sum in [false, true] {
            let source = projected_source(transform.clone());
            let slice = ResolvedCaptureSlice {
                starts: vec![0, 1, 1],
                ends: vec![2, 3, 7],
                strides: vec![1, 1, 2],
                shape: vec![2, 2, 3],
            };
            let combination = if sum {
                PartitionCaptureCombination::SumF64ToF32
            } else {
                PartitionCaptureCombination::Disjoint
            };
            let producer = |rank, range| PartitionCaptureProducer {
                rank,
                projection: CaptureContiguousProjectionPlan::prepare(
                    &[2, 3, 7],
                    &slice,
                    2,
                    range,
                    1,
                )
                .unwrap()
                .construct(),
            };
            let producers = if sum {
                vec![producer(0, 0..7), producer(1, 0..7)]
            } else {
                vec![producer(0, 3..7), producer(1, 0..3)]
            };
            let context = PartitionCaptureContext {
                artifact_identity: "fixture".into(),
                execution_identity: "retained".into(),
                run_identity: "source".into(),
                overlay_identity: None,
                capture_plan_identity: source.admission().identity().into(),
                selection_index: 0,
                phase: CapturePhase::Prefill,
                prediction: 0,
                forward_epoch: 1,
                invocation: None,
                invocation_window: None,
            };
            let mut ledger = CaptureLedger::new(source.admission());
            ledger.begin_step();
            let limits = PartitionCaptureReceiptLimits {
                max_producers: 2,
                max_fragments: 2,
                max_record_bytes: 64 << 10,
            };
            let receipt = if sum {
                PartitionCaptureReceiptPlan::new_sum(
                    source.clone(),
                    context,
                    producers,
                    2,
                    limits,
                    &mut ledger,
                )
            } else {
                PartitionCaptureReceiptPlan::new(
                    source.clone(),
                    context,
                    producers,
                    2,
                    limits,
                    &mut ledger,
                )
            }
            .unwrap();
            let inference = InferenceGeometry {
                batch_size: 1,
                cached_positions: 2,
                input_positions: 3,
                max_output_tokens: 4,
                prefill_chunk_positions: 1,
                output: OutputDemand::LastPosition,
            };
            let context = WorkspaceContext::new(cpu);
            let equation =
                PartitionPrefillEquation::prepare(&receipt, 0, 0, inference, &context).unwrap();
            let native = equation.native_source(eredu_core::checkpoint::TensorDtype::F32);
            assert_eq!(
                native.local_shape,
                if sum { &[2, 3, 7][..] } else { &[2, 3, 4][..] }
            );
            assert_eq!(
                native.transform,
                if sum
                    && matches!(
                        transform,
                        CaptureTransform::Summary | CaptureTransform::Histogram { .. }
                    )
                {
                    &CaptureTransform::Slice
                } else {
                    &transform
                }
            );
            assert!(native.estimate.capture.retained_bytes > 0);
            assert_eq!(native.estimate.generated_creation_bytes, 0);
            for k in 0..3 {
                context.begin_span();
                let mut roots = vec![];
                let value = WorkspaceTensor::existing(
                    context
                        .layout(&[2, 1, if sum { 7 } else { 4 }], WorkspaceDtype::Float32)
                        .unwrap()
                        .with_representation(Some(WorkspaceRepresentation::new(
                            WorkspaceFloatingType::Float32,
                            true,
                        ))),
                    &context,
                )
                .unwrap();
                let (dtype, population) = equation
                    .trace_fragment(k, &value, &context, &mut roots)
                    .unwrap();
                assert_eq!(dtype, WorkspaceFloatingType::Float32);
                let report = context.report(&roots).unwrap();
                assert!(report.unpriced_operations.is_empty());
                assert_eq!(population.publications == 0, k == 0);
                assert_eq!(roots.is_empty(), k == 0);
                if k != 0 {
                    assert!(population.completions > 0);
                    assert!(!report.operations.is_empty());
                }
            }
            assert!(std::ptr::eq(
                equation.source.admission(),
                receipt.shared_plan_source().admission()
            ));
            assert!(std::ptr::eq(
                equation.source.projection(),
                receipt.producer(0).unwrap()
            ));
        }
    }
}

#[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
fn projected_source(
    transform: eredu_core::capture::CaptureTransform,
) -> eredu_core::capture::SharedCapturePlan {
    use super::*;
    use eredu_core::*;
    let point = ObservationPoint {
        path: "block.output".into(),
        node_id: "block".into(),
        meaning: "projected activation".into(),
        value_type: ObservationValueType::Tensor,
        dtype: ObservationDtype::Floating,
        axes: Some(
            [
                SymbolicDimension::Known(2),
                SymbolicDimension::Sequence,
                SymbolicDimension::Known(7),
            ]
            .into_iter()
            .enumerate()
            .map(|(i, dimension)| TensorAxis {
                name: format!("axis{i}"),
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
    let maximum = CaptureUsage {
        captures: u64::MAX,
        retained_bytes: u64::MAX,
        host_bytes: u64::MAX,
        encoded_bytes: u64::MAX,
    };
    let declaration = CapturePlan {
        schema_version: 1,
        selections: vec![CaptureSelection {
            id: "projected".into(),
            path: "block.output".into(),
            schedule: CaptureSchedule {
                prefill: true,
                decode: false,
                ..Default::default()
            },
            transform: transform.clone(),
            slices: vec![
                CaptureSlice {
                    axis: "axis1".into(),
                    start: 1,
                    end: 3,
                    stride: 1,
                },
                CaptureSlice {
                    axis: "axis2".into(),
                    start: 1,
                    end: 7,
                    stride: 2,
                },
            ],
        }],
        limits: CaptureLimits {
            per_step: maximum,
            cumulative: maximum,
            on_limit: CaptureLimitPolicy::Fail,
        },
    };
    let caps = CaptureCapabilities {
        transformations: vec![
            CaptureTransformKind::Slice,
            CaptureTransformKind::Preview,
            CaptureTransformKind::Summary,
            CaptureTransformKind::Histogram,
        ],
        max_histogram_bins: 2,
        conditions: vec![],
    };
    SharedCapturePlan::new(
        declaration
            .admit_with_text_origin(
                &catalog,
                &support,
                &caps,
                CaptureRequestShape {
                    batch: 1,
                    prompt_tokens: 3,
                    max_predictions: 4,
                },
                CaptureTextOrigin {
                    cached_positions: 2,
                },
            )
            .unwrap(),
    )
}

#[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
#[test]
fn contiguous_cold_observer_preserves_source_scalar_and_local_completion_population() {
    use super::*;
    use crate::backend::nn::workspace::{
        MlxCpuMatmulMechanism, MlxCpuWorkspaceMechanisms, MlxMetalWorkspaceMechanisms,
    };
    use eredu_core::*;
    use eredu_nn::workspace::*;
    let native = MlxMetalWorkspaceMechanisms::current_host().unwrap();
    let selected =
        MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles).unwrap();
    let cpu = MlxCpuWorkspaceMechanisms::new(native.allocation(), selected);
    let inference = InferenceGeometry {
        batch_size: 1,
        cached_positions: 2,
        input_positions: 3,
        max_output_tokens: 4,
        prefill_chunk_positions: 1,
        output: OutputDemand::LastPosition,
    };
    for transform in [CaptureTransform::Slice, CaptureTransform::Summary] {
        for produces in [false, true] {
            for empty in [false, true] {
                let source = projected_source(transform.clone());
                let context = WorkspaceContext::new(cpu);
                let range = if empty { 0..1 } else { 3..7 };
                let projection =
                    observer::projection(source.admission(), 0, 2, range, &context).unwrap();
                assert_eq!(projection.fragments().is_empty(), empty);
                let transfers = Cell::new(CaptureNativePopulation::default());
                let scalars = [Cell::new(None)];
                let mut roots = vec![];
                let value = WorkspaceTensor::existing(
                    context
                        .layout(&[2, 1, if empty { 1 } else { 4 }], WorkspaceDtype::Float32)
                        .unwrap()
                        .with_representation(Some(WorkspaceRepresentation::new(
                            WorkspaceFloatingType::Float32,
                            true,
                        ))),
                    &context,
                )
                .unwrap();
                context.begin_state_span([&value]).unwrap();
                observer::trace(
                    &source,
                    0,
                    &projection,
                    PartitionCaptureCombination::Disjoint,
                    0,
                    produces,
                    inference,
                    1,
                    1,
                    &value,
                    &context,
                    &mut roots,
                    Some(&transfers),
                    Some(&scalars),
                )
                .unwrap();
                assert_eq!(scalars[0].get(), Some(WorkspaceFloatingType::Float32));
                let population = transfers.get();
                if !produces || empty {
                    assert_eq!(
                        (
                            population.publications,
                            population.completions,
                            population.retained_roots
                        ),
                        (0, 1, 1)
                    );
                    assert_eq!(roots.len(), 1);
                    assert!(context.report(&roots).unwrap().operations.is_empty());
                } else {
                    assert!(population.publications > 0);
                    assert!(population.completions > 0);
                    let report = context.report(&roots).unwrap();
                    assert!(report.unpriced_operations.is_empty());
                    assert!(!report.operations.is_empty());
                }
                // Geometry equality cannot substitute for the native scalar witness.
                let opaque = WorkspaceTensor::existing(
                    context
                        .layout(&[2, 1, if empty { 1 } else { 4 }], WorkspaceDtype::Float32)
                        .unwrap(),
                    &context,
                )
                .unwrap();
                assert!(
                    observer::trace(
                        &source,
                        0,
                        &projection,
                        PartitionCaptureCombination::Disjoint,
                        0,
                        produces,
                        inference,
                        1,
                        1,
                        &opaque,
                        &context,
                        &mut roots,
                        Some(&transfers),
                        Some(&scalars)
                    )
                    .is_err()
                );
                let wrong = WorkspaceTensor::existing(
                    context
                        .layout(&[2, 2, if empty { 1 } else { 4 }], WorkspaceDtype::Float32)
                        .unwrap()
                        .with_representation(Some(WorkspaceRepresentation::new(
                            WorkspaceFloatingType::Float32,
                            true,
                        ))),
                    &context,
                )
                .unwrap();
                assert!(
                    observer::trace(
                        &source,
                        0,
                        &projection,
                        PartitionCaptureCombination::Disjoint,
                        0,
                        produces,
                        inference,
                        1,
                        1,
                        &wrong,
                        &context,
                        &mut roots,
                        Some(&transfers),
                        Some(&scalars)
                    )
                    .is_err()
                );
            }
        }
    }
}
