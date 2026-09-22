//! Exact static rectangles through the ordinary CPU Slice source.
use super::*;

pub(super) fn inspect(
    operation: WorkspaceOperationView<'_>,
    mechanism: MlxCpuWorkspaceMechanisms,
) -> facts::FactResult<Option<OperationPlan>> {
    let WorkspaceOperationKindView::StaticSlice { strides, .. } = operation.kind else {
        return Ok(None);
    };
    if !super::super::basic::is_static_slice(operation) {
        return Err(MlxWorkspaceFactError::descriptor(
            "CPU static slice coordinates differ",
        ));
    }
    let input = operation.inputs.get(0).expect("qualified slice input");
    let output = operation.outputs.get(0).expect("qualified slice output");
    let rank = input.shape().len();
    if !(1..=4).contains(&rank)
        || input.shape().iter().chain(output.shape()).any(|&n| n < 0)
        || strides.iter().any(|&n| n <= 0)
    {
        return Ok(None);
    }
    if input.elements()? > i32::MAX as u64 {
        return Ok(None);
    }
    let empty = output.elements()? == 0;
    let representation = input.representation();
    let dtype = if input.dtype() == WorkspaceDtype::Float32 {
        let Some(representation) = representation else {
            return Ok(None);
        };
        if !empty {
            let Some(source_strides) = views::physical_strides(input, representation) else {
                return Ok(None);
            };
            // Floating consumers retain the exact selected stride facts. Unit axes
            // observe only coordinate zero; every remaining slot must fit them.
            let mut span = 1u64;
            for axis in 0..rank {
                let step = if output.shape()[axis] == 1 {
                    1
                } else {
                    match u64::try_from(source_strides[axis])
                        .ok()
                        .and_then(|s| s.checked_mul(strides[axis] as u64))
                    {
                        Some(step) if step > 0 && step <= u32::MAX as u64 => step,
                        _ => return Ok(None),
                    }
                };
                span = match (output.shape()[axis] as u64 - 1)
                    .checked_mul(step)
                    .and_then(|n| span.checked_add(n))
                {
                    Some(span) if span <= i64::MAX as u64 => span,
                    _ => return Ok(None),
                };
            }
        }
        representation.dtype()
    } else {
        if representation.is_some() || output.representation().is_some() {
            return Ok(None);
        }
        // Integer/boolean/byte layouts already state their physical scalar
        // width. Slice is a dtype-preserving alias: its fixed native controls
        // depend on rank, never inferred floating strides. The actual Slice
        // guard independently checks offset + selected span against the exact
        // retained input backing before sharing it. This internal plan field
        // is unused for nonfloating outputs and publishes no representation.
        WorkspaceFloatingType::Float32
    };
    // MLX returns the input directly for a complete positive-stride slice.
    // The C result still owns a handle, but creates no Slice/Eval/task.
    let whole = input.shape() == output.shape();
    let mut population = CpuPopulation::default();
    if !whole {
        let Some(source) = OperationEvent::cpu_slice_layout(rank, empty, false) else {
            return Ok(None);
        };
        if source.backing_births() != usize::from(empty && mechanism.allocation.cpu_header)
            || population.copy(source, 1).is_none()
        {
            return Ok(None);
        }
        // Empty Slice constructs Data with the compiled allocator's zero-size
        // backing. Nonempty Slice shares the original Data.
        population.maximum_captures = usize::from(empty);
    }
    let frames = [
        if representation.is_some() && !empty {
            views::physical_stride_control_bytes()
                .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?
        } else {
            0
        },
        crate::tensor::narrow::control_bytes(rank)
            .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?,
        size_of::<MlxCpuWorkspaceMechanisms>(),
        size_of::<u64>(),
        size_of::<WorkspaceOperationView<'_>>(),
        size_of::<WorkspaceLayoutView<'_>>() * 2,
        size_of::<WorkspaceRepresentation>(),
        size_of::<Option<WorkspaceRepresentation>>(),
        size_of::<WorkspaceDtype>(),
        size_of::<WorkspaceFloatingType>(),
        size_of::<CpuCopyEvalLayout>(),
        size_of::<Option<CpuCopyEvalLayout>>(),
        size_of::<CpuPopulation>(),
        size_of::<OperationPlan>(),
        size_of::<Option<OperationPlan>>(),
        size_of::<facts::FactResult<Option<OperationPlan>>>(),
        size_of::<usize>() * 4,
        size_of::<u64>() * 2,
        size_of::<bool>() * 3,
        size_of::<std::ops::Range<usize>>(),
        size_of::<std::iter::Rev<std::ops::Range<usize>>>(),
        size_of::<std::slice::Iter<'_, i32>>(),
        size_of::<std::iter::Chain<std::slice::Iter<'_, i32>, std::slice::Iter<'_, i32>>>(),
        size_of::<(&safemlx::Array, &[i32], &[i32], &[i32], &safemlx::Stream)>(),
        size_of::<Result<safemlx::Array, safemlx::error::Exception>>(),
        size_of::<[i32; 4]>(),
        size_of::<Option<i32>>(),
        size_of::<&[i32]>() * 3,
        size_of::<[i64; 4]>(),
        size_of::<Option<[i64; 4]>>(),
        size_of::<[u64; 4]>(),
        size_of::<u64>() * 3,
        size_of::<Option<u64>>(),
        size_of::<usize>(),
    ];
    population.controls = frames.into_iter().try_fold(
        population
            .controls
            .checked_add(size_of_val(&frames))
            .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?,
        |n, b| {
            n.checked_add(b)
                .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)
        },
    )?;
    let output_bytes = if empty && !whole {
        mechanism.allocation.fixed_buffer_capacity(0)?
    } else {
        0
    };
    Ok(Some(OperationPlan {
        dtype,
        population,
        alias_input: (!empty || whole).then_some(0),
        output_bytes,
        scratch_bytes: 0,
        rank,
        parameter_shells: usize::from(whole),
        seeds: 0,
        validations: 0,
    }))
}

/// A rectangular alias can introduce gaps between rows. Compare the retained
/// input's actual row-major strides multiplied by each exact positive slice
/// step with selected dense strides; singleton output axes impose no constraint.
pub(super) fn representation(
    operation: WorkspaceOperationView<'_>,
    input: WorkspaceRepresentation,
) -> WorkspaceRepresentation {
    let source = operation.inputs.get(0).expect("qualified static slice");
    let output = operation.outputs.get(0).expect("qualified static slice");
    let WorkspaceOperationKindView::StaticSlice { strides, .. } = operation.kind else {
        unreachable!("qualified static slice")
    };
    if source.shape() == output.shape() {
        return input;
    }
    if output.shape().contains(&0) {
        return WorkspaceRepresentation::new(input.dtype(), true);
    }
    let source_strides =
        views::physical_strides(source, input).expect("qualified slice source strides");
    let mut selected = [1u64; 4];
    let mut dense_stride = 1u64;
    let mut rows = true;
    for axis in (0..source.shape().len()).rev() {
        if output.shape()[axis] > 1 {
            selected[axis] = source_strides[axis] as u64 * strides[axis] as u64;
            if selected[axis] != dense_stride {
                rows = false;
            }
        }
        dense_stride *= output.shape()[axis] as u64;
    }
    let last = source.shape().len() - 1;
    WorkspaceRepresentation::new(input.dtype(), rows)
        .with_last_axis_contiguous(output.shape()[last] <= 1 || selected[last] == 1)
        .with_element_strides(&selected[..source.shape().len()])
        .expect("qualified bounded slice strides")
}

#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
mod tests {
    use super::*;
    use eredu_nn::Tensor;
    #[test]
    fn capture_slice_preserves_exact_coordinates_source_alias_and_row_gaps() {
        let ordinary = MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let choice =
            MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles).unwrap();
        let cpu = MlxCpuWorkspaceMechanisms::new(ordinary.allocation(), choice);
        for dtype in [
            WorkspaceFloatingType::Float32,
            WorkspaceFloatingType::Float16,
            WorkspaceFloatingType::Bfloat16,
        ] {
            let context = WorkspaceContext::new(cpu);
            let storage =
                eredu_nn::workspace::WorkspaceExistingStorage::try_new(Some(4096), &context)
                    .unwrap();
            let source = WorkspaceTensor::existing_with_storage(
                context
                    .layout(&[2, 3, 7], WorkspaceDtype::Float32)
                    .unwrap()
                    .with_representation(Some(WorkspaceRepresentation::new(dtype, true))),
                &storage,
                &context,
            )
            .unwrap();
            context.begin_state_span([&source]).unwrap();
            let selected = source
                .static_slice(&[0, 1, 2], &[2, 3, 6], &[1, 1, 1], &context)
                .unwrap();
            assert_eq!(selected.shape(), [2, 2, 4]);
            let rep = selected.layout().representation().unwrap();
            assert_eq!(rep.dtype(), dtype);
            assert!(!rep.row_contiguous());
            assert!(rep.last_axis_contiguous());
            let report = context.report(std::slice::from_ref(&selected)).unwrap();
            assert_eq!(report.tensor_buffers.total_bytes, Some(0));
            assert_eq!(report.state.as_ref().unwrap().retained_bytes, Some(4096));
            let op = report.operations[0].as_view();
            let plan = cpu.plan(op).unwrap().unwrap();
            assert_eq!(plan.alias_input, Some(0));
            assert_eq!(plan.population.births, 0);
            assert_eq!(plan.population.primitives, 1);
            assert_eq!(plan.seeds, 0);
            assert_eq!(plan.validations, 0);
            assert_eq!(plan.output_bytes, 0);
            assert_eq!(plan.scratch_bytes, 0);
            assert!(plan.population.extents > 0 && plan.population.controls > 0);
            let WorkspaceOperationKindView::StaticSlice {
                starts,
                ends,
                strides,
            } = op.kind
            else {
                panic!("slice coordinates lost");
            };
            assert_eq!(starts, [0, 1, 2]);
            assert_eq!(ends, [2, 3, 6]);
            assert_eq!(strides, [1, 1, 1]);
            // A single row's contiguous interval remains row-major despite its
            // offset within the unchanged larger backing owner.
            let row = source
                .static_slice(&[1, 2, 2], &[2, 3, 6], &[1, 1, 1], &context)
                .unwrap();
            assert!(row.layout().representation().unwrap().row_contiguous());
        }
        let source = WorkspaceLayoutView::new(&[2, 3, 7], WorkspaceDtype::Float32)
            .unwrap()
            .with_representation(Some(WorkspaceRepresentation::new(
                WorkspaceFloatingType::Float32,
                true,
            )));
        let output = WorkspaceLayoutView::new(&[2, 2, 4], WorkspaceDtype::Float32).unwrap();
        let outputs = [output];
        let inputs = [source];
        let base = WorkspaceOperationView {
            kind: WorkspaceOperationKindView::StaticSlice {
                starts: &[0, 1, 2],
                ends: &[2, 3, 6],
                strides: &[1, 1, 1],
            },
            inputs: WorkspaceLayoutList::Views(&inputs),
            outputs: WorkspaceLayoutList::Views(&outputs),
        };
        let malformed = WorkspaceOperationView {
            kind: WorkspaceOperationKindView::StaticSlice {
                starts: &[0, 1],
                ends: &[2, 3, 6],
                strides: &[1, 1, 1],
            },
            ..base
        };
        assert!(cpu.plan(malformed).is_err());
        let erased = [source.with_representation(None)];
        assert!(cpu
            .plan(WorkspaceOperationView {
                inputs: WorkspaceLayoutList::Views(&erased),
                ..base
            })
            .unwrap()
            .is_none());
        let strided = [
            source.with_representation(Some(WorkspaceRepresentation::new(
                WorkspaceFloatingType::Float32,
                false,
            ))),
        ];
        assert!(cpu
            .plan(WorkspaceOperationView {
                inputs: WorkspaceLayoutList::Views(&strided),
                ..base
            })
            .unwrap()
            .is_none());
        let step_output = [WorkspaceLayoutView::new(&[2, 2, 2], WorkspaceDtype::Float32).unwrap()];
        let stepped = WorkspaceOperationView {
            kind: WorkspaceOperationKindView::StaticSlice {
                starts: &[0, 1, 2],
                ends: &[2, 3, 6],
                strides: &[1, 1, 2],
            },
            outputs: WorkspaceLayoutList::Views(&step_output),
            ..base
        };
        let selected = cpu.plan(stepped).unwrap().unwrap();
        assert_eq!(selected.alias_input, Some(0));
        let facts = cpu.output_representation(stepped, 0).unwrap();
        assert!(!facts.row_contiguous());
        assert!(!facts.last_axis_contiguous());
        let generic = WorkspaceOperationView {
            kind: WorkspaceOperationKindView::Index { selected_axes: 0 },
            ..base
        };
        assert!(cpu.plan(generic).unwrap().is_none());
    }
    #[test]
    fn positive_capture_slice_quotes_exact_steps_and_preserves_cast_alias() {
        let ordinary = MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let choice =
            MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles).unwrap();
        let cpu = MlxCpuWorkspaceMechanisms::new(ordinary.allocation(), choice);
        for (ends, steps, shape, row, last) in [
            ([2, 1, 3], [1, 1, 2], [2, 1, 2], false, false),
            ([2, 1, 1], [1, 1, 2], [2, 1, 1], false, true),
        ] {
            let context = WorkspaceContext::new(cpu);
            let source = WorkspaceTensor::existing(
                context
                    .layout(&[2, 1, 4], WorkspaceDtype::Float32)
                    .unwrap()
                    .with_representation(Some(WorkspaceRepresentation::new(
                        WorkspaceFloatingType::Float32,
                        true,
                    ))),
                &context,
            )
            .unwrap();
            context.begin_state_span([&source]).unwrap();
            let selected = source
                .static_slice(&[0, 0, 0], &ends, &steps, &context)
                .unwrap();
            assert_eq!(selected.shape(), shape);
            let representation = selected.layout().representation().unwrap();
            assert_eq!(
                (
                    representation.row_contiguous(),
                    representation.last_axis_contiguous()
                ),
                (row, last)
            );
            let mut outputs = context.metadata_vec(1).unwrap();
            outputs.push(context.layout(&shape, WorkspaceDtype::Float32).unwrap());
            let output = context
                .execute(
                    WorkspaceOperationKind::Elementwise("capture_cast_f32"),
                    &[&selected],
                    outputs,
                )
                .unwrap()
                .remove(0);
            assert_eq!(output.layout().representation(), Some(representation));
            let report = context.report(&[output]).unwrap();
            assert!(report.unpriced_operations.is_empty());
            assert!(report.unpriced_host_operations.is_empty());
            assert_eq!(report.tensor_buffers.total_bytes, Some(0));
            let plan = cpu.plan(report.operations[0].as_view()).unwrap().unwrap();
            assert_eq!(
                (
                    plan.population.primitives,
                    plan.population.births,
                    plan.alias_input
                ),
                (1, 0, Some(0))
            );
        }
    }
    #[test]
    fn stepped_preview_flatten_uses_actual_native_alias_or_copy_and_retains_strides() {
        let ordinary = MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let choice =
            MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles).unwrap();
        let cpu = MlxCpuWorkspaceMechanisms::new(ordinary.allocation(), choice);
        for (width, copied) in [(4, false), (5, true)] {
            let context = WorkspaceContext::new(cpu);
            let storage = WorkspaceExistingStorage::try_new(Some(4096), &context).unwrap();
            let source = WorkspaceTensor::existing_with_storage(
                context
                    .layout(&[2, 1, width], WorkspaceDtype::Float32)
                    .unwrap()
                    .with_representation(Some(WorkspaceRepresentation::new(
                        WorkspaceFloatingType::Float32,
                        true,
                    ))),
                &storage,
                &context,
            )
            .unwrap();
            context.begin_state_span([&source]).unwrap();
            let selected = source
                .static_slice(&[0, 0, 0], &[2, 1, 3], &[1, 1, 2], &context)
                .unwrap();
            let selected_rep = selected.layout().representation().unwrap();
            assert_eq!(selected_rep.element_stride_at(3, 0), Some(width as u64));
            assert_eq!(selected_rep.element_stride_at(3, 2), Some(2));
            let flat = selected.reshape(&[4], &context).unwrap();
            let flat_rep = flat.layout().representation().unwrap();
            assert_eq!(flat_rep.row_contiguous(), copied);
            assert_eq!(
                flat_rep.element_stride_at(1, 0),
                if copied { None } else { Some(2) }
            );
            let preview = flat.static_slice(&[0], &[3], &[1], &context).unwrap();
            let mut outputs = context.metadata_vec(1).unwrap();
            outputs.push(context.layout(&[3], WorkspaceDtype::Float32).unwrap());
            let output = context
                .execute(
                    WorkspaceOperationKind::Elementwise("capture_cast_f32"),
                    &[&preview],
                    outputs,
                )
                .unwrap()
                .remove(0);
            assert_eq!(
                output.layout().representation(),
                preview.layout().representation()
            );
            let report = context.report(&[output]).unwrap();
            assert!(report.unpriced_operations.is_empty());
            assert!(report.unpriced_host_operations.is_empty());
            let reshape = cpu.plan(report.operations[1].as_view()).unwrap().unwrap();
            assert_eq!(reshape.population.births, usize::from(copied));
            assert_eq!(reshape.alias_input, if copied { None } else { Some(0) });
            assert_eq!(reshape.output_bytes > 0, copied);
            let erase = [report.operations[1]
                .as_view()
                .inputs
                .get(0)
                .unwrap()
                .with_representation(Some(WorkspaceRepresentation::new(
                    WorkspaceFloatingType::Float32,
                    false,
                )))];
            let operation = WorkspaceOperationView {
                inputs: WorkspaceLayoutList::Views(&erase),
                ..report.operations[1].as_view()
            };
            assert!(
                cpu.plan(operation).unwrap().is_none(),
                "shape alone cannot recover selected strides"
            );
        }
    }
    #[test]
    fn narrow_axis_quotes_exact_multi_position_batched_slice_and_preserves_custody() {
        let ordinary = MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let choice =
            MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles).unwrap();
        let cpu = MlxCpuWorkspaceMechanisms::new(ordinary.allocation(), choice);
        for (start, end) in [(3, 5), (4, 5), (0, 5)] {
            let context = WorkspaceContext::new(cpu);
            let storage = WorkspaceExistingStorage::try_new(Some(4096), &context).unwrap();
            let source = WorkspaceTensor::existing_with_storage(
                context
                    .layout(&[2, 5, 3], WorkspaceDtype::Float32)
                    .unwrap()
                    .with_representation(Some(WorkspaceRepresentation::new(
                        WorkspaceFloatingType::Float32,
                        true,
                    ))),
                &storage,
                &context,
            )
            .unwrap();
            context.begin_state_span([&source]).unwrap();
            let selected = source.narrow_axis(1, start, end, &context).unwrap();
            assert_eq!(selected.shape(), [2, end - start, 3]);
            let representation = selected.layout().representation().unwrap();
            assert_eq!(representation.row_contiguous(), start == 0 && end == 5);
            assert!(representation.last_axis_contiguous());
            let report = context.finish_report(&[source.clone(), selected]).unwrap();
            assert_eq!(report.operations.len(), 1);
            let WorkspaceOperationKindView::StaticSlice {
                starts,
                ends,
                strides,
            } = report.operations[0].as_view().kind
            else {
                panic!("narrow axis lost exact coordinates");
            };
            assert_eq!(starts, [0, start, 0]);
            assert_eq!(ends, [2, end, 3]);
            assert_eq!(strides, [1, 1, 1]);
            assert_eq!(report.state.as_ref().unwrap().retained_bytes, Some(4096));
            assert_eq!(report.state.as_ref().unwrap().displaced_bytes, Some(0));
            assert_eq!(report.tensor_buffers.total_bytes, Some(0));
            assert!(report.unpriced_operations.is_empty());
            assert!(report.unpriced_host_operations.is_empty());
            let plan = cpu.plan(report.operations[0].as_view()).unwrap().unwrap();
            assert_eq!(plan.population.births, 0);
            assert_eq!(plan.alias_input, Some(0));
            assert_eq!(
                plan.population.primitives,
                usize::from(start != 0 || end != 5)
            );
            let recipe = SpeculativeNumericalRecipe::inspect_cpu_capture(
                &report, 2, 0, ordinary, cpu, &context,
            )
            .unwrap();
            assert_eq!(recipe.kernels, 0);
        }
    }
}

#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
mod index_tests {
    use super::*;
    use eredu_nn::{Index, Tensor};

    #[test]
    fn cpu_paged_range_index_retains_key_scalar_strides_and_exact_source_backing() {
        let ordinary = MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let choice =
            MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles).unwrap();
        let cpu = MlxCpuWorkspaceMechanisms::new(ordinary.allocation(), choice);
        let context = WorkspaceContext::new(cpu);
        let storage =
            eredu_nn::workspace::WorkspaceExistingStorage::try_new(Some(8192), &context).unwrap();
        let input = WorkspaceTensor::existing_with_storage(
            context
                .layout(&[1, 2, 5, 4], WorkspaceDtype::Float32)
                .unwrap()
                .with_representation(Some(WorkspaceRepresentation::new(
                    WorkspaceFloatingType::Float32,
                    true,
                ))),
            &storage,
            &context,
        )
        .unwrap();
        context.begin_state_span([&input]).unwrap();
        let page = input
            .index(
                &[Index::Full, Index::Full, Index::Range(1, 4), Index::Full],
                &context,
            )
            .unwrap();
        let representation = page
            .layout()
            .representation()
            .expect("exact paged key source");
        assert_eq!(representation.dtype(), WorkspaceFloatingType::Float32);
        assert!(!representation.row_contiguous());
        assert!(representation.last_axis_contiguous());
        assert_eq!(representation.element_stride_at(4, 1), Some(20));
        assert_eq!(representation.element_stride_at(4, 2), Some(4));
        let report = context.report(std::slice::from_ref(&page)).unwrap();
        assert_eq!(report.tensor_buffers.total_bytes, Some(0));
        assert_eq!(report.state.as_ref().unwrap().retained_bytes, Some(8192));
        let op = report.operations[0].as_view();
        let WorkspaceOperationKindView::StaticSlice {
            starts,
            ends,
            strides,
        } = op.kind
        else {
            panic!("actual coordinates lost")
        };
        assert_eq!(starts, [0, 0, 1, 0]);
        assert_eq!(ends, [1, 2, 4, 4]);
        assert_eq!(strides, [1, 1, 1, 1]);
        let plan = cpu.plan(op).unwrap().unwrap();
        assert_eq!(plan.alias_input, Some(0));
        assert_eq!(plan.population.births, 0);
        assert_eq!(plan.population.primitives, 1);
        // The isolated CPU range consumer reads this same exact event and
        // retains its separate Slice source, without a rank-count fallback.
        let range =
            SpeculativeNumericalRecipe::inspect_cpu_range(&report, ordinary, &context).unwrap();
        assert!(range.graph_capacity > 0 && range.record_capacity > 0);
        let unknown = [op.inputs.get(0).unwrap().with_representation(None)];
        assert!(cpu
            .plan(WorkspaceOperationView {
                inputs: WorkspaceLayoutList::Views(&unknown),
                ..op
            })
            .unwrap()
            .is_none());
        // The old rank-count-only event does not receive a fabricated geometry.
        assert!(cpu
            .plan(WorkspaceOperationView {
                kind: WorkspaceOperationKindView::Index { selected_axes: 0 },
                ..op
            })
            .unwrap()
            .is_none());
    }
}

#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
mod scalar_tests;

#[cfg(all(test, target_vendor = "apple", not(feature = "cuda")))]
mod empty_tests;
