use super::*;
use crate::*;

fn fixture(
    dimensions: Vec<SymbolicDimension>,
    transform: CaptureTransform,
    slices: &[(usize, u64, u64, u64)],
    batch: u64,
    prompt: u64,
    cached: u64,
) -> (AdmittedCapturePlan, CaptureDiscovery, InferenceGeometry) {
    let point = ObservationPoint {
        path: "body.output".into(),
        node_id: "body".into(),
        meaning: "declared operation rows".into(),
        value_type: ObservationValueType::Tensor,
        dtype: ObservationDtype::Floating,
        axes: Some(
            dimensions
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
    let capabilities = CaptureCapabilities {
        transformations: vec![
            CaptureTransformKind::FullTensor,
            CaptureTransformKind::Slice,
            CaptureTransformKind::Preview,
            CaptureTransformKind::Summary,
            CaptureTransformKind::Histogram,
        ],
        max_histogram_bins: 2,
        ..Default::default()
    };
    let discovery = CaptureDiscovery {
        artifact_identity: "row-geometry-fixture".into(),
        catalog: ObservationCatalog {
            schema_version: 1,
            points: vec![point],
            completeness: DescriptionCompleteness::Complete,
        },
        support: ObservationSupportReport {
            schema_version: 1,
            capture: capabilities,
            points: vec![ObservationSupport {
                path: "body.output".into(),
                prefill: ObservationSupportStatus::Supported,
                decode: ObservationSupportStatus::Supported,
                floating_to_f32: true,
            }],
        },
    };
    let usage = CaptureUsage {
        captures: 32,
        retained_bytes: u64::MAX,
        host_bytes: u64::MAX,
        encoded_bytes: u64::MAX,
    };
    let plan = CapturePlan {
        schema_version: 1,
        selections: vec![CaptureSelection {
            id: "rows".into(),
            path: "body.output".into(),
            schedule: CaptureSchedule {
                decode: false,
                ..Default::default()
            },
            slices: slices
                .iter()
                .map(|&(axis, start, end, stride)| CaptureSlice {
                    axis: format!("axis{axis}"),
                    start,
                    end,
                    stride,
                })
                .collect(),
            transform,
        }],
        limits: CaptureLimits {
            per_step: usage,
            cumulative: usage,
            on_limit: CaptureLimitPolicy::Fail,
        },
    };
    let request = CaptureRequestShape {
        batch,
        prompt_tokens: prompt,
        max_predictions: 1,
    };
    let admitted = plan
        .admit_with_text_origin(
            &discovery.catalog,
            &discovery.support,
            &discovery.support.capture,
            request,
            CaptureTextOrigin {
                cached_positions: cached,
            },
        )
        .unwrap();
    let inference = InferenceGeometry {
        batch_size: batch,
        cached_positions: cached,
        input_positions: prompt,
        max_output_tokens: 1,
        prefill_chunk_positions: prompt.min(3),
        output: OutputDemand::LastPosition,
    };
    (admitted, discovery, inference)
}
fn coordinates(mut index: usize, shape: &[usize]) -> Vec<usize> {
    let mut result = vec![0; shape.len()];
    for axis in (0..shape.len()).rev() {
        result[axis] = index % shape[axis];
        index /= shape[axis];
    }
    result
}
// Independent reference uses the existing whole-source slice resolver and
// filters coordinates of a nonzero full tensor. It never computes fragment
// slice intersections or the production scatter formula.
fn compare_full_reference(plan: &CapturePrefillRowAssembly<'_>) {
    let full_shape = plan.logical_geometry().source_shape();
    let size = full_shape.iter().product::<usize>();
    let values = (0..size)
        .map(|i| i as f32 * 0.25 - 1.375)
        .collect::<Vec<_>>();
    let admitted = plan.logical_geometry().admission();
    let selection = &admitted.plan().selections[0];
    let slice = resolve_slice(
        &admitted.points()[0],
        selection,
        &full_shape.iter().map(|&d| d as u64).collect::<Vec<_>>(),
    )
    .unwrap();
    let selected = |coordinate: &[usize]| {
        coordinate.iter().enumerate().all(|(i, &v)| {
            let v = v as u64;
            v >= slice.starts[i]
                && v < slice.ends[i]
                && (v - slice.starts[i]) % slice.strides[i] == 0
        })
    };
    let mut expected = values
        .iter()
        .enumerate()
        .filter(|(i, _)| selected(&coordinates(*i, full_shape)))
        .map(|(_, &v)| v)
        .collect::<Vec<_>>();
    if let CaptureTransform::Preview { max_elements } = selection.transform {
        expected.truncate(usize::try_from(max_elements).unwrap_or(usize::MAX));
    }
    let mut actual = vec![None; expected.len()];
    let mut visited = 0;
    for i in 0..plan.chunk_count() {
        let fragment = plan.fragment(i).unwrap();
        let physical = values
            .iter()
            .enumerate()
            .filter(|(flat, _)| {
                let row = coordinates(*flat, full_shape)[plan.sequence_axis()] as u64;
                fragment.input().contains(&row)
            })
            .map(|(_, &v)| v)
            .collect::<Vec<_>>();
        assert_eq!(
            physical.len(),
            fragment.source_shape().iter().product::<usize>()
        );
        assert!(std::ptr::eq(fragment.assembly(), plan));
        assert!(fragment.mapping_at(fragment.output_elements()).is_none());
        assert!(fragment.mapping_at(usize::MAX).is_none());
        let mut mappings = fragment.mappings();
        assert_eq!(mappings.len(), fragment.output_elements());
        let mut previous = None;
        for entry in mappings.by_ref() {
            assert_eq!(fragment.mapping_at(entry.selected_index()), Some(entry));
            assert!(entry.source_index() < physical.len());
            assert!(entry.selected_index() < fragment.selected_elements());
            assert!(previous.is_none_or(|p| p < entry.destination_index()));
            previous = Some(entry.destination_index());
            assert!(
                actual[entry.destination_index()]
                    .replace(physical[entry.source_index()])
                    .is_none(),
                "a logical value was mapped twice"
            );
            visited += 1;
        }
        assert!(mappings.next().is_none());
        assert_eq!(mappings.len(), 0);
    }
    assert_eq!(visited, expected.len());
    assert_eq!(
        actual.into_iter().map(Option::unwrap).collect::<Vec<_>>(),
        expected
    );
}

#[test]
fn full_rows_scatter_across_batch_and_heads_with_nonzero_origin() {
    let (source, _, geometry) = fixture(
        vec![
            SymbolicDimension::Batch,
            SymbolicDimension::Known(3),
            SymbolicDimension::Sequence,
            SymbolicDimension::Known(4),
        ],
        CaptureTransform::FullTensor,
        &[],
        2,
        7,
        11,
    );
    let plan = CapturePrefillRowAssembly::prepare(&source, 0, geometry).unwrap();
    assert_eq!(plan.sequence_axis(), 2);
    assert_eq!(plan.selected_shape(), &[2, 3, 7, 4]);
    assert_eq!(plan.selected_elements(), 168);
    assert_eq!(plan.chunk_count(), 3);
    for (index, range, position, output) in [
        (0, 0..3, 11, OutputDemand::StateOnly),
        (1, 3..6, 14, OutputDemand::StateOnly),
        (2, 6..7, 17, OutputDemand::LastPosition),
    ] {
        let f = plan.fragment(index).unwrap();
        assert!(f.matches_chunk(&range, position, output));
        assert!(!f.matches_chunk(&range, position + 1, output));
        assert_eq!(f.source_shape()[2] as u64, range.end - range.start);
    }
    let first = plan
        .fragment(0)
        .unwrap()
        .mappings()
        .map(|e| e.destination_index())
        .collect::<Vec<_>>();
    assert_eq!(first[11], 11);
    assert_eq!(first[12], 28); // scatter into next head, not append after row 2
    compare_full_reference(&plan);
}

#[test]
fn global_strides_cross_chunks_and_preview_cuts_the_final_selected_order() {
    for width in [1, 2, 4, 9] {
        for transform in [
            CaptureTransform::Slice,
            CaptureTransform::Preview { max_elements: 11 },
        ] {
            let (source, _, mut geometry) = fixture(
                vec![
                    SymbolicDimension::Batch,
                    SymbolicDimension::Known(3),
                    SymbolicDimension::Sequence,
                    SymbolicDimension::Known(5),
                ],
                transform,
                &[(1, 0, 3, 2), (2, 1, 9, 2), (3, 1, 5, 2)],
                2,
                9,
                5,
            );
            geometry.prefill_chunk_positions = width;
            let plan = CapturePrefillRowAssembly::prepare(&source, 0, geometry).unwrap();
            assert_eq!(plan.selected_shape(), &[2, 2, 4, 2]);
            assert_eq!(plan.selected_elements(), 32);
            compare_full_reference(&plan);
        }
    }
}

#[test]
fn preview_zero_and_cutoffs_do_not_restart_for_each_chunk_or_batch() {
    for maximum in [0, 1, 7, 17, 100, usize::MAX as u64] {
        let (source, _, geometry) = fixture(
            vec![
                SymbolicDimension::Batch,
                SymbolicDimension::Sequence,
                SymbolicDimension::Known(3),
            ],
            CaptureTransform::Preview {
                max_elements: maximum,
            },
            &[],
            2,
            7,
            3,
        );
        let plan = CapturePrefillRowAssembly::prepare(&source, 0, geometry).unwrap();
        assert_eq!(plan.selected_elements(), 42);
        assert_eq!(
            plan.logical_geometry().shape(),
            &[usize::try_from(maximum.min(42)).unwrap()]
        );
        if maximum == 0 {
            for i in 0..plan.chunk_count() {
                let f = plan.fragment(i).unwrap();
                assert_eq!(f.output_elements(), 0);
                assert!(f.selected_elements() > 0);
            }
        }
        compare_full_reference(&plan);
    }
}

#[test]
fn empty_intersections_slices_and_zero_fixed_axes_produce_no_indices() {
    let (source, _, geometry) = fixture(
        vec![SymbolicDimension::Sequence, SymbolicDimension::Known(2)],
        CaptureTransform::Slice,
        &[(0, 5, 7, 1)],
        1,
        7,
        0,
    );
    let plan = CapturePrefillRowAssembly::prepare(&source, 0, geometry).unwrap();
    assert_eq!(plan.fragment(0).unwrap().selected_elements(), 0);
    assert_eq!(plan.fragment(1).unwrap().output_elements(), 2);
    assert_eq!(plan.fragment(2).unwrap().output_elements(), 2);
    compare_full_reference(&plan);
    for (dimension, slices) in [(2, vec![(0, 3, 3, 1)]), (0, vec![])] {
        let transform = if slices.is_empty() {
            CaptureTransform::FullTensor
        } else {
            CaptureTransform::Slice
        };
        let (source, _, geometry) = fixture(
            vec![
                SymbolicDimension::Sequence,
                SymbolicDimension::Known(dimension),
            ],
            transform,
            &slices,
            1,
            7,
            0,
        );
        let plan = CapturePrefillRowAssembly::prepare(&source, 0, geometry).unwrap();
        assert_eq!(plan.logical_geometry().elements(), 0);
        for i in 0..plan.chunk_count() {
            assert_eq!(plan.fragment(i).unwrap().mappings().len(), 0);
        }
        compare_full_reference(&plan);
    }
}

#[test]
fn actual_source_and_schedule_stay_bound_without_changing_admission_identity() {
    let (source, discovery, geometry) = fixture(
        vec![
            SymbolicDimension::Batch,
            SymbolicDimension::Sequence,
            SymbolicDimension::Known(2),
        ],
        CaptureTransform::FullTensor,
        &[],
        2,
        7,
        13,
    );
    let shared = SharedCapturePlan::new(source);
    let alias = shared.clone();
    let identity = shared.admission().identity().to_owned();
    let pointer = shared.admission().plan().selections.as_ptr();
    let capacity = shared.capacity_bytes();
    let plan = CapturePrefillRowAssembly::prepare(shared.admission(), 0, geometry).unwrap();
    let independent = shared.admission().readmit(&discovery).unwrap();
    assert_eq!(independent.identity(), identity);
    assert!(std::ptr::eq(
        plan.logical_geometry().admission(),
        alias.admission()
    ));
    assert!(!std::ptr::eq(
        plan.logical_geometry().admission(),
        &independent
    ));
    for _ in 0..32 {
        for i in 0..plan.chunk_count() {
            let f = plan.fragment(i).unwrap();
            assert_eq!(f.mappings().count(), f.output_elements());
        }
    }
    assert_eq!(shared.admission().identity(), identity);
    assert_eq!(shared.admission().plan().selections.as_ptr(), pointer);
    assert_eq!(shared.capacity_bytes(), capacity);
    assert!(shared.same_storage(&alias));
    for changed in [
        InferenceGeometry {
            batch_size: 1,
            ..geometry
        },
        InferenceGeometry {
            cached_positions: 0,
            ..geometry
        },
        InferenceGeometry {
            input_positions: 6,
            ..geometry
        },
        InferenceGeometry {
            max_output_tokens: 2,
            ..geometry
        },
    ] {
        assert!(matches!(
            CapturePrefillRowAssembly::prepare(shared.admission(), 0, changed),
            Err(CapturePrefillGeometryError::RequestMismatch)
        ));
    }
    for width in [0, 8] {
        assert!(matches!(
            CapturePrefillRowAssembly::prepare(
                shared.admission(),
                0,
                InferenceGeometry {
                    prefill_chunk_positions: width,
                    ..geometry
                }
            ),
            Err(CapturePrefillGeometryError::Schedule)
        ));
    }
    let sequence = CapturePrefillRowAssembly::prepare(
        shared.admission(),
        0,
        InferenceGeometry {
            output: OutputDemand::Sequence,
            ..geometry
        },
    )
    .unwrap();
    assert_eq!(
        sequence.fragment(0).unwrap().output_demand(),
        OutputDemand::Sequence
    );
    assert_eq!(
        plan.fragment(0).unwrap().output_demand(),
        OutputDemand::StateOnly
    );
    assert!(matches!(
        plan.fragment(plan.chunk_count()),
        Err(CapturePrefillGeometryError::Chunk { .. })
    ));
}

#[test]
fn unsupported_semantic_axes_and_independent_invocations_remain_typed() {
    for dimension in [
        SymbolicDimension::Context,
        SymbolicDimension::MediaPositions,
        SymbolicDimension::Unknown,
    ] {
        let (source, _, geometry) = fixture(
            vec![SymbolicDimension::Sequence, dimension],
            CaptureTransform::FullTensor,
            &[],
            1,
            3,
            2,
        );
        assert!(matches!(
            CapturePrefillRowAssembly::prepare(&source, 0, geometry),
            Err(CapturePrefillGeometryError::UnsupportedAxis { axis: 1 })
        ));
    }
    for (dimensions, count) in [
        (vec![SymbolicDimension::Known(2)], 0),
        (
            vec![SymbolicDimension::Sequence, SymbolicDimension::TokenRows],
            2,
        ),
        (
            vec![SymbolicDimension::Sequence, SymbolicDimension::Sequence],
            2,
        ),
    ] {
        let (source, _, geometry) = fixture(dimensions, CaptureTransform::FullTensor, &[], 1, 3, 2);
        assert!(
            matches!(CapturePrefillRowAssembly::prepare(&source,0,geometry),Err(CapturePrefillGeometryError::SequenceAxes{actual}) if actual==count)
        );
    }
    let (batched, _, geometry) = fixture(
        vec![SymbolicDimension::TokenRows],
        CaptureTransform::FullTensor,
        &[],
        2,
        3,
        2,
    );
    assert!(matches!(
        CapturePrefillRowAssembly::prepare(&batched, 0, geometry),
        Err(CapturePrefillGeometryError::UnsupportedAxis { axis: 0 })
    ));
    let (source, discovery, geometry) = fixture(
        vec![SymbolicDimension::Sequence],
        CaptureTransform::FullTensor,
        &[],
        1,
        3,
        2,
    );
    let independent = source
        .plan()
        .clone()
        .admit_invocations(
            &discovery.catalog,
            &discovery.support,
            &discovery.support.capture,
            CaptureInvocationBounds {
                batch: 1,
                max_sequence: 3,
                max_context: Some(5),
                max_predictions: 1,
            },
        )
        .unwrap();
    assert!(matches!(
        CapturePrefillRowAssembly::prepare(&independent, 0, geometry),
        Err(CapturePrefillGeometryError::Invocation)
    ));
    assert!(matches!(
        CapturePrefillRowAssembly::prepare(&source, 1, geometry),
        Err(CapturePrefillGeometryError::Tensor(
            CaptureTensorGeometryError::SelectionMissing { index: 1 }
        ))
    ));
    let mut raw = source.plan().clone();
    raw.selections[0].schedule.prefill = false;
    let inactive = raw
        .admit_with_text_origin(
            &discovery.catalog,
            &discovery.support,
            &discovery.support.capture,
            source.request(),
            source.text_origin().unwrap(),
        )
        .unwrap();
    assert!(matches!(
        CapturePrefillRowAssembly::prepare(&inactive, 0, geometry),
        Err(CapturePrefillGeometryError::Tensor(
            CaptureTensorGeometryError::Inactive
        ))
    ));
    let (summary, _, geometry) = fixture(
        vec![SymbolicDimension::Sequence],
        CaptureTransform::Summary,
        &[],
        1,
        3,
        2,
    );
    assert!(matches!(
        CapturePrefillRowAssembly::prepare(&summary, 0, geometry),
        Err(CapturePrefillGeometryError::Tensor(
            CaptureTensorGeometryError::Unsupported
        ))
    ));
}

#[test]
fn checked_frontier_overflow_is_rejected_from_a_real_valid_capture_admission() {
    // Capture's last actual p0 fits MAX. Inference's separate F+prompt+M
    // frontier does not. No fabricated AdmittedCapturePlan or giant allocation.
    let (source, _, geometry) = fixture(
        vec![SymbolicDimension::Sequence],
        CaptureTransform::FullTensor,
        &[],
        1,
        3,
        u64::MAX - 3,
    );
    assert!(matches!(
        CapturePrefillRowAssembly::prepare(&source, 0, geometry),
        Err(CapturePrefillGeometryError::Overflow)
    ));
}

#[cfg(target_pointer_width = "64")]
#[test]
fn extreme_real_extents_and_strides_use_checked_indices_without_allocating_payload() {
    let width = usize::MAX / 7;
    let (source, _, geometry) = fixture(
        vec![SymbolicDimension::Known(width), SymbolicDimension::Sequence],
        CaptureTransform::Slice,
        &[(0, (width - 1) as u64, width as u64, 1)],
        1,
        7,
        0,
    );
    let plan = CapturePrefillRowAssembly::prepare(&source, 0, geometry).unwrap();
    for i in 0..plan.chunk_count() {
        let f = plan.fragment(i).unwrap();
        let physical_rows = (f.input().end - f.input().start) as usize;
        let entries = f.mappings().collect::<Vec<_>>();
        assert_eq!(entries.len(), physical_rows);
        assert_eq!(entries[0].source_index(), (width - 1) * physical_rows);
        assert_eq!(
            entries.last().unwrap().source_index(),
            width * physical_rows - 1
        );
    }
    let prompt = u64::MAX - 2;
    let (source, _, mut geometry) = fixture(
        vec![SymbolicDimension::Sequence],
        CaptureTransform::Slice,
        &[(0, prompt - 1, prompt, u64::MAX)],
        1,
        prompt,
        0,
    );
    geometry.prefill_chunk_positions = prompt - 1;
    let plan = CapturePrefillRowAssembly::prepare(&source, 0, geometry).unwrap();
    assert_eq!(plan.chunk_count(), 2);
    assert_eq!(plan.fragment(0).unwrap().output_elements(), 0);
    let last = plan.fragment(1).unwrap();
    assert_eq!(last.input(), &(prompt - 1..prompt));
    assert_eq!(last.position(), prompt - 1);
    let element = last.mappings().next().unwrap();
    assert_eq!(
        (element.source_index(), element.destination_index()),
        (0, 0)
    );
    assert!(matches!(
        plan.fragment(u64::MAX),
        Err(CapturePrefillGeometryError::Chunk { .. })
    ));
}

fn compare_axis_cartesian(plan: &CapturePrefillRowAssembly<'_>) {
    for chunk in 0..plan.chunk_count() {
        let fragment = plan.fragment(chunk).unwrap();
        assert!(fragment
            .selection_axis(fragment.source_shape().len())
            .is_none());
        assert!(fragment.selection_axis(usize::MAX).is_none());
        let axes = (0..fragment.source_shape().len())
            .map(|i| fragment.selection_axis(i).unwrap())
            .collect::<Vec<_>>();
        for (i, axis) in axes.iter().enumerate() {
            assert!(axis.stride() > 0);
            assert!(axis.end() <= fragment.source_shape()[i]);
            assert_eq!(
                (axis.end() - axis.start()).div_ceil(axis.stride()),
                axis.elements()
            );
            assert_eq!(axis.elements(), fragment.selected_shape()[i]);
        }
        // Enumerate the real physical rectangle independently; no construction
        // of the production intersection or destination formula is repeated.
        let mut selected = (0..fragment.source_shape().iter().product::<usize>())
            .filter(|&flat| {
                coordinates(flat, fragment.source_shape())
                    .iter()
                    .zip(&axes)
                    .all(|(&c, a)| {
                        c >= a.start() && c < a.end() && (c - a.start()) % a.stride() == 0
                    })
            })
            .collect::<Vec<_>>();
        assert_eq!(selected.len(), fragment.selected_elements());
        selected.truncate(fragment.output_elements());
        assert_eq!(
            selected,
            fragment
                .mappings()
                .map(|m| m.source_index())
                .collect::<Vec<_>>()
        );
    }
}
#[test]
fn canonical_axis_slices_match_physical_cartesian_mapping_and_global_preview() {
    for transform in [
        CaptureTransform::Slice,
        CaptureTransform::Preview { max_elements: 0 },
        CaptureTransform::Preview { max_elements: 9 },
        CaptureTransform::Preview { max_elements: 999 },
    ] {
        let (source, _, mut geometry) = fixture(
            vec![
                SymbolicDimension::Batch,
                SymbolicDimension::Known(3),
                SymbolicDimension::Sequence,
                SymbolicDimension::Known(4),
            ],
            transform,
            &[(1, 1, 3, 1), (2, 1, 7, 2), (3, 0, 4, 2)],
            2,
            7,
            19,
        );
        for width in [1, 2, 3, 7] {
            geometry.prefill_chunk_positions = width;
            let plan = CapturePrefillRowAssembly::prepare(&source, 0, geometry).unwrap();
            compare_axis_cartesian(&plan);
            compare_full_reference(&plan);
        }
    }
    for (last, slice) in [(4, vec![(1, 3, 3, 1)]), (0, vec![])] {
        let transform = if slice.is_empty() {
            CaptureTransform::FullTensor
        } else {
            CaptureTransform::Slice
        };
        let (source, _, geometry) = fixture(
            vec![
                SymbolicDimension::Batch,
                SymbolicDimension::Sequence,
                SymbolicDimension::Known(last),
            ],
            transform,
            &slice,
            2,
            7,
            9,
        );
        let plan = CapturePrefillRowAssembly::prepare(&source, 0, geometry).unwrap();
        compare_axis_cartesian(&plan);
    }
}
#[test]
fn canonical_axis_keeps_extreme_stride_and_checked_exclusive_endpoint() {
    let stride = usize::MAX as u64;
    let (source, _, geometry) = fixture(
        vec![SymbolicDimension::Sequence, SymbolicDimension::Known(7)],
        CaptureTransform::Slice,
        &[(1, 6, 7, stride)],
        1,
        7,
        1,
    );
    let plan = CapturePrefillRowAssembly::prepare(&source, 0, geometry).unwrap();
    let fragment = plan.fragment(0).unwrap();
    let axis = fragment.selection_axis(1).unwrap();
    assert_eq!(
        (axis.start(), axis.end(), axis.stride(), axis.elements()),
        (6, 7, usize::MAX, 1)
    );
    compare_axis_cartesian(&plan);
}

#[test]
fn independent_windows_preserve_global_strides_and_nonzero_values_for_all_transforms() {
    let (ordinary, discovery, _) = fixture(
        vec![
            SymbolicDimension::Known(2),
            SymbolicDimension::Sequence,
            SymbolicDimension::Known(4),
        ],
        CaptureTransform::FullTensor,
        &[(1, 1, 11, 3), (2, 1, 4, 2)],
        1,
        11,
        0,
    );
    let mut capabilities = discovery.support.capture.clone();
    capabilities
        .transformations
        .push(CaptureTransformKind::Histogram);
    capabilities.max_histogram_bins = 3;
    for transform in [
        CaptureTransform::FullTensor,
        CaptureTransform::Slice,
        CaptureTransform::Preview { max_elements: 5 },
        CaptureTransform::Summary,
        CaptureTransform::Histogram {
            edges: vec![-20.0, 0.0, 20.0, 100.0],
        },
    ] {
        let mut raw = ordinary.plan().clone();
        raw.selections[0].transform = transform.clone();
        let source = raw
            .admit_invocations(
                &discovery.catalog,
                &discovery.support,
                &capabilities,
                CaptureInvocationBounds {
                    batch: 1,
                    max_sequence: 11,
                    max_context: None,
                    max_predictions: 1,
                },
            )
            .unwrap();
        for input in [0..1, 1..2, 2..5, 5..9, 9..11] {
            let physical = CaptureInvocationShape {
                batch: 1,
                sequence: input.end - input.start,
                context: None,
            };
            let window = CaptureInvocationWindow {
                logical_sequence: 11,
                start: input.start,
            };
            let actual = vec![2, physical.sequence, 4];
            let oracle = window
                .project(
                    physical,
                    &source.points()[0],
                    &source.plan().selections[0],
                    &actual,
                )
                .unwrap();
            let (shape, starts, ends, strides, count) = match transform {
                CaptureTransform::Summary => {
                    let value = CaptureSummaryGeometry::prepare_window(
                        &source,
                        0,
                        CapturePhase::Prefill,
                        0,
                        physical,
                        window,
                    )
                    .unwrap();
                    (
                        value.source_shape().to_vec(),
                        value.starts().to_vec(),
                        value.ends().to_vec(),
                        value.strides().to_vec(),
                        value.elements(),
                    )
                }
                CaptureTransform::Histogram { .. } => {
                    let value = CaptureHistogramGeometry::prepare_window(
                        &source,
                        0,
                        CapturePhase::Prefill,
                        0,
                        physical,
                        window,
                    )
                    .unwrap();
                    assert_eq!(value.edges(), &[-20.0, 0.0, 20.0, 100.0]);
                    (
                        value.source_shape().to_vec(),
                        value.starts().to_vec(),
                        value.ends().to_vec(),
                        value.strides().to_vec(),
                        value.elements(),
                    )
                }
                _ => {
                    let value = CaptureTensorGeometry::prepare_window(
                        &source,
                        0,
                        CapturePhase::Prefill,
                        0,
                        physical,
                        window,
                    )
                    .unwrap();
                    (
                        value.source_shape().to_vec(),
                        value.starts().to_vec(),
                        value.ends().to_vec(),
                        value.strides().to_vec(),
                        value.elements(),
                    )
                }
            };
            assert_eq!(shape, [2, physical.sequence as usize, 4]);
            let values = |starts: &[u64], ends: &[u64], strides: &[u64]| {
                let mut out = Vec::new();
                for head in (starts[0]..ends[0]).step_by(strides[0] as usize) {
                    for row in (starts[1]..ends[1]).step_by(strides[1] as usize) {
                        for column in (starts[2]..ends[2]).step_by(strides[2] as usize) {
                            let global = (head * 11 + input.start + row) * 4 + column;
                            out.push(global as f32 * 0.375 - 12.5);
                        }
                    }
                }
                if let CaptureTransform::Preview { max_elements } = transform {
                    out.truncate(max_elements as usize);
                }
                out
            };
            let expected = oracle
                .fragments()
                .first()
                .map_or_else(Vec::new, |fragment| {
                    let local = fragment.local();
                    values(&local.starts, &local.ends, &local.strides)
                });
            let actual = values(&starts, &ends, &strides);
            assert_eq!(actual, expected);
            assert_eq!(count, actual.len());
        }
        assert!(CaptureTensorGeometry::prepare_window(
            &source,
            0,
            CapturePhase::Prefill,
            0,
            CaptureInvocationShape {
                batch: 1,
                sequence: 2,
                context: None
            },
            CaptureInvocationWindow {
                logical_sequence: 11,
                start: 10
            },
        )
        .is_err());
    }
}

#[test]
fn partition_prefill_rows_keep_original_spatial_projection_and_global_preview_order() {
    use crate::component::ComponentCoordinateMap;
    for transform in [
        CaptureTransform::Slice,
        CaptureTransform::Preview { max_elements: 5 },
    ] {
        let (source, _, mut inference) = fixture(
            vec![
                SymbolicDimension::Batch,
                SymbolicDimension::Sequence,
                SymbolicDimension::Known(7),
            ],
            transform.clone(),
            &[(1, 1, 5, 2), (2, 1, 7, 2)],
            2,
            5,
            4,
        );
        inference.prefill_chunk_positions = 2;
        let shape = [2, 5, 7];
        let selected =
            resolve_slice(&source.points()[0], &source.plan().selections[0], &shape).unwrap();
        for (range, combination) in [
            (0..3, PartitionCaptureCombination::Disjoint),
            (3..7, PartitionCaptureCombination::Disjoint),
            (0..7, PartitionCaptureCombination::SumF64ToF32),
        ] {
            let members = ComponentCoordinateMap::range(7, range).unwrap();
            let projection = CaptureSlicePartition::new(&shape, &selected, 2, &members, 1).unwrap();
            let plan = CapturePrefillRowAssembly::prepare_partition(
                &source,
                0,
                inference,
                &projection,
                0,
                combination,
            )
            .unwrap();
            assert!(std::ptr::eq(plan.logical_geometry().admission(), &source));
            assert_eq!(
                plan.logical_geometry().source_shape(),
                projection
                    .local_shape()
                    .iter()
                    .map(|n| *n as usize)
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                plan.logical_geometry().starts(),
                projection.fragments()[0].local().starts
            );
            let full_shape = plan.logical_geometry().source_shape();
            let full: Vec<f32> = (0..full_shape.iter().product::<usize>())
                .map(|n| n as f32 * 0.125 + 0.75)
                .collect();
            let local = projection.fragments()[0].local();
            let mut expected: Vec<_> = full
                .iter()
                .enumerate()
                .filter(|(flat, _)| {
                    coordinates(*flat, full_shape)
                        .iter()
                        .enumerate()
                        .all(|(axis, n)| {
                            *n as u64 >= local.starts[axis]
                                && (*n as u64) < local.ends[axis]
                                && (*n as u64 - local.starts[axis]) % local.strides[axis] == 0
                        })
                })
                .map(|(_, n)| *n)
                .collect();
            if let CaptureTransform::Preview { max_elements } = transform {
                expected.truncate(max_elements as usize);
            }
            let mut output = vec![None; expected.len()];
            for index in 0..plan.chunk_count() {
                let fragment = plan.fragment(index).unwrap();
                let physical: Vec<_> = full
                    .iter()
                    .enumerate()
                    .filter(|(flat, _)| {
                        fragment
                            .input()
                            .contains(&(coordinates(*flat, full_shape)[1] as u64))
                    })
                    .map(|(_, n)| *n)
                    .collect();
                assert_eq!(
                    physical.len(),
                    fragment.source_shape().iter().product::<usize>()
                );
                assert_eq!(fragment.position(), 4 + index * 2);
                for entry in fragment.mappings() {
                    assert!(output[entry.destination_index()]
                        .replace(physical[entry.source_index()])
                        .is_none());
                }
            }
            assert_eq!(
                output.into_iter().map(Option::unwrap).collect::<Vec<_>>(),
                expected
            );
            assert!(CapturePrefillRowAssembly::prepare_partition(
                &source,
                0,
                inference,
                &projection,
                1,
                combination
            )
            .is_err());
        }
        let temporal = CaptureSlicePartition::new(
            &shape,
            &selected,
            1,
            &ComponentCoordinateMap::range(5, 0..3).unwrap(),
            1,
        )
        .unwrap();
        assert!(matches!(
            CapturePrefillRowAssembly::prepare_partition(
                &source,
                0,
                inference,
                &temporal,
                0,
                PartitionCaptureCombination::Disjoint
            ),
            Err(CapturePrefillGeometryError::UnsupportedAxis { axis: 1 })
        ));
    }
}

#[test]
fn partition_nonlinear_windows_keep_local_shapes_and_additive_raw_terms() {
    use crate::component::ComponentCoordinateMap;
    for transform in [
        CaptureTransform::Summary,
        CaptureTransform::Histogram {
            edges: vec![-1.0, 0.25, 2.0],
        },
    ] {
        let (source, _, mut inference) = fixture(
            vec![
                SymbolicDimension::Batch,
                SymbolicDimension::Sequence,
                SymbolicDimension::Known(7),
            ],
            transform.clone(),
            &[(1, 1, 5, 2), (2, 1, 7, 2)],
            2,
            5,
            4,
        );
        inference.prefill_chunk_positions = 2;
        let shape = [2, 5, 7];
        let selected =
            resolve_slice(&source.points()[0], &source.plan().selections[0], &shape).unwrap();
        for (range, combination) in [
            (0..3, PartitionCaptureCombination::Disjoint),
            (3..7, PartitionCaptureCombination::Disjoint),
            (0..7, PartitionCaptureCombination::SumF64ToF32),
        ] {
            let projection = CaptureSlicePartition::new(
                &shape,
                &selected,
                2,
                &ComponentCoordinateMap::range(7, range).unwrap(),
                1,
            )
            .unwrap();
            let plan = CapturePrefillTransformPlan::prepare_partition(
                &source,
                0,
                inference,
                &projection,
                0,
                combination,
            )
            .unwrap();
            assert!(std::ptr::eq(plan.admission(), &source));
            assert_eq!(plan.selection().transform, transform);
            assert_eq!(&plan.window().source()[..3], projection.local_shape());
            assert_eq!(
                &plan.window().selected()[..3],
                projection.fragments()[0].local().shape
            );
            let mut count = 0;
            for index in 0..plan.chunk_count() {
                let fragment = plan.fragment(index).unwrap();
                count += fragment.selected_elements();
                let mut physical = projection.local_shape().to_vec();
                physical[1] = (5 - index * 2).min(2);
                assert_eq!(fragment.source_shape(), physical);
                assert_eq!(fragment.position(), 4 + index * 2);
                let local = projection.fragments()[0].local();
                assert_eq!(fragment.starts()[2], local.starts[2]);
                assert_eq!(fragment.strides()[2], local.strides[2]);
                if combination == PartitionCaptureCombination::Disjoint {
                    match transform {
                        CaptureTransform::Summary => {
                            let geometry = CaptureSummaryGeometry::prepare_partition(
                                &source,
                                0,
                                CapturePhase::Prefill,
                                0,
                                None,
                                &projection,
                                0,
                            )
                            .unwrap()
                            .fragment(&fragment)
                            .unwrap();
                            assert_eq!(geometry.elements() as u64, fragment.selected_elements());
                            assert_eq!(
                                geometry.source_shape(),
                                physical.iter().map(|n| *n as usize).collect::<Vec<_>>()
                            );
                        }
                        CaptureTransform::Histogram { .. } => {
                            let geometry = CaptureHistogramGeometry::prepare_partition(
                                &source,
                                0,
                                CapturePhase::Prefill,
                                0,
                                None,
                                &projection,
                                0,
                            )
                            .unwrap()
                            .fragment(&fragment)
                            .unwrap();
                            assert_eq!(geometry.elements() as u64, fragment.selected_elements());
                            assert_eq!(
                                geometry.source_shape(),
                                physical.iter().map(|n| *n as usize).collect::<Vec<_>>()
                            );
                        }
                        _ => unreachable!(),
                    }
                }
            }
            assert_eq!(
                count,
                projection.fragments()[0]
                    .local()
                    .shape
                    .iter()
                    .product::<u64>()
            );
            if combination == PartitionCaptureCombination::SumF64ToF32 {
                let raw = CapturePrefillRowAssembly::prepare_additive_transform_partition(
                    &source,
                    0,
                    inference,
                    &projection,
                    0,
                )
                .unwrap();
                assert!(std::ptr::eq(raw.logical_geometry().admission(), &source));
                assert_eq!(
                    raw.logical_geometry().native_transform(),
                    &CaptureTransform::Slice
                );
                let source_shape = raw.logical_geometry().source_shape();
                let full: Vec<f32> = (0..source_shape.iter().product::<usize>())
                    .map(|n| n as f32 * 0.125 - 2.25)
                    .collect();
                let local = projection.fragments()[0].local();
                let expected: Vec<_> = full
                    .iter()
                    .enumerate()
                    .filter(|(flat, _)| {
                        coordinates(*flat, source_shape)
                            .iter()
                            .enumerate()
                            .all(|(axis, n)| {
                                *n as u64 >= local.starts[axis]
                                    && (*n as u64) < local.ends[axis]
                                    && (*n as u64 - local.starts[axis]) % local.strides[axis] == 0
                            })
                    })
                    .map(|(_, n)| *n)
                    .collect();
                let mut output = vec![None; expected.len()];
                for index in 0..raw.chunk_count() {
                    let fragment = raw.fragment(index).unwrap();
                    let physical: Vec<_> = full
                        .iter()
                        .enumerate()
                        .filter(|(flat, _)| {
                            fragment
                                .input()
                                .contains(&(coordinates(*flat, source_shape)[1] as u64))
                        })
                        .map(|(_, n)| *n)
                        .collect();
                    for entry in fragment.mappings() {
                        assert!(output[entry.destination_index()]
                            .replace(physical[entry.source_index()])
                            .is_none());
                    }
                }
                assert_eq!(
                    output.into_iter().map(Option::unwrap).collect::<Vec<_>>(),
                    expected
                );
            }
            assert!(CapturePrefillTransformPlan::prepare_partition(
                &source,
                0,
                inference,
                &projection,
                1,
                combination
            )
            .is_err());
        }
        let temporal = CaptureSlicePartition::new(
            &shape,
            &selected,
            1,
            &ComponentCoordinateMap::range(5, 0..3).unwrap(),
            1,
        )
        .unwrap();
        assert!(CapturePrefillTransformPlan::prepare_partition(
            &source,
            0,
            inference,
            &temporal,
            0,
            PartitionCaptureCombination::Disjoint
        )
        .is_err());
    }
}
