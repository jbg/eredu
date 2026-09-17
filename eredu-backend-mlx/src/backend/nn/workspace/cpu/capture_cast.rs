//! Shared CPU AsType source for actual capture/intervention F32 conversion.
use super::*;

pub(super) fn inspect(
    operation: WorkspaceOperationView<'_>,
    mechanism: MlxCpuWorkspaceMechanisms,
) -> facts::FactResult<Option<OperationPlan>> {
    if !matches!(
        operation.kind,
        WorkspaceOperationKindView::Elementwise("capture_cast_f32")
            | WorkspaceOperationKindView::CastFloating(WorkspaceFloatingType::Float32)
    ) {
        return Ok(None);
    }
    if operation.inputs.len() != 1 || operation.outputs.len() != 1 {
        return Err(MlxWorkspaceFactError::descriptor(
            "CPU capture cast input/output population differs",
        ));
    }
    let input = operation.inputs.get(0).expect("one cast source");
    let output = operation.outputs.get(0).expect("one cast output");
    if input.shape() != output.shape()
        || input.dtype() != WorkspaceDtype::Float32
        || output.dtype() != WorkspaceDtype::Float32
    {
        return Err(MlxWorkspaceFactError::descriptor(
            "CPU capture cast changes floating geometry",
        ));
    }
    let rank = input.shape().len();
    if rank > 4 || input.shape().iter().any(|&n| n <= 0) {
        return Ok(None);
    }
    let Some(representation) = input.representation() else {
        return Ok(None);
    };
    let source = match representation.dtype() {
        WorkspaceFloatingType::Float32 => safemlx::Dtype::Float32,
        WorkspaceFloatingType::Float16 => safemlx::Dtype::Float16,
        WorkspaceFloatingType::Bfloat16 => safemlx::Dtype::Bfloat16,
    };
    let alias = source == safemlx::Dtype::Float32;
    // Same-type AsType returns the existing source, preserving all layout facts.
    // Half conversion uses CopyType::Vector on a row-contiguous input. Other
    // actual layouts need their output layout evidence before joining capture.
    if !alias && !representation.row_contiguous() {
        return Ok(None);
    }
    let elements = usize::try_from(input.elements()?)?;
    if elements > i32::MAX as usize {
        return Ok(None);
    }
    let mut population = CpuPopulation::default();
    if !alias {
        let Some(native) =
            OperationEvent::cpu_cast_layout(source, safemlx::Dtype::Float32, rank, elements, false)
        else {
            return Ok(None);
        };
        if native.backing_births() != 1 || population.copy(native, 1).is_none() {
            return Ok(None);
        }
    }
    let frames = [
        safemlx::Array::as_dtype_control_bytes()
            .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?,
        size_of::<WorkspaceOperationView<'_>>(),
        size_of::<WorkspaceLayoutView<'_>>() * 2,
        size_of::<MlxCpuWorkspaceMechanisms>(),
        size_of::<WorkspaceRepresentation>(),
        size_of::<Option<WorkspaceRepresentation>>(),
        size_of::<WorkspaceFloatingType>(),
        size_of::<safemlx::Dtype>(),
        size_of::<CpuPopulation>(),
        size_of::<CpuCopyEvalLayout>(),
        size_of::<Option<CpuCopyEvalLayout>>(),
        size_of::<OperationPlan>(),
        size_of::<Option<OperationPlan>>(),
        size_of::<facts::FactResult<Option<OperationPlan>>>(),
        size_of::<std::slice::Iter<'_, i32>>(),
        size_of::<bool>(),
        size_of::<usize>() * 2,
        size_of::<u64>(),
        size_of::<Option<()>>(),
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
    let output_bytes = if alias {
        0
    } else {
        mechanism
            .allocation
            .fixed_buffer_capacity(output.bytes()?)?
    };
    Ok(Some(OperationPlan {
        dtype: WorkspaceFloatingType::Float32,
        population,
        alias_input: alias.then_some(0),
        output_bytes,
        scratch_bytes: 0,
        rank,
        // Native dtype equality still returns one C result handle. Converted
        // results already have that shell in their one primitive population.
        parameter_shells: usize::from(alias),
        seeds: 0,
        validations: 0,
    }))
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
    fn capture_conversion_requires_physical_precision_and_preserves_alias_or_copy() {
        let ordinary = MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let Some(choice) =
            MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles)
        else {
            assert!(std::env::var_os("EREDU_REQUIRE_QUALIFIED_RECORD_LAYOUT").is_none());
            return;
        };
        let cpu = MlxCpuWorkspaceMechanisms::new(ordinary.allocation(), choice);
        let output = WorkspaceLayoutView::new(&[1, 1, 7], WorkspaceDtype::Float32).unwrap();
        for kind in [
            WorkspaceOperationKindView::Elementwise("capture_cast_f32"),
            WorkspaceOperationKindView::CastFloating(WorkspaceFloatingType::Float32),
        ] {
            for dtype in [
                WorkspaceFloatingType::Float32,
                WorkspaceFloatingType::Float16,
                WorkspaceFloatingType::Bfloat16,
            ] {
                let inputs =
                    [output.with_representation(Some(WorkspaceRepresentation::new(dtype, true)))];
                let outputs = [output];
                let op = WorkspaceOperationView {
                    kind,
                    inputs: WorkspaceLayoutList::Views(&inputs),
                    outputs: WorkspaceLayoutList::Views(&outputs),
                };
                let plan = cpu.plan(op).unwrap().unwrap();
                let alias = dtype == WorkspaceFloatingType::Float32;
                assert_eq!(plan.alias_input, alias.then_some(0));
                assert_eq!(plan.population.primitives, usize::from(!alias));
                assert_eq!(plan.population.births, usize::from(!alias));
                assert_eq!(plan.parameter_shells, usize::from(alias));
                assert_eq!(plan.scratch_bytes, 0);
                assert!(plan.population.controls > 0);
                if alias {
                    assert_eq!(plan.output_bytes, 0);
                } else {
                    assert!(plan.output_bytes >= 28);
                    assert!(plan.population.extents > 0);
                }
                // Exercise the existing neutral cast consumer: it must keep an
                // F32 alias outside new-backing accounting and retain a real
                // independent result for an actual half conversion.
                let context = WorkspaceContext::new(cpu);
                let value = WorkspaceTensor::existing(
                    context
                        .layout(&[1, 1, 7], WorkspaceDtype::Float32)
                        .unwrap()
                        .with_representation(Some(WorkspaceRepresentation::new(dtype, true))),
                    &context,
                )
                .unwrap();
                context.begin_span();
                let result = value
                    .cast_floating(WorkspaceFloatingType::Float32, &context)
                    .unwrap();
                assert_eq!(result.shape(), value.shape());
                assert_eq!(
                    result.layout().representation().unwrap().dtype(),
                    WorkspaceFloatingType::Float32
                );
                let report = context.report(&[result]).unwrap();
                if alias {
                    assert_eq!(report.tensor_buffers.retained_bytes, Some(0));
                } else {
                    assert!(report.tensor_buffers.retained_bytes.unwrap() >= 28);
                }
            }
        }
        let outputs = [output];
        for representation in [
            None,
            Some(WorkspaceRepresentation::new(
                WorkspaceFloatingType::Bfloat16,
                false,
            )),
        ] {
            let inputs = [output.with_representation(representation)];
            assert!(cpu
                .plan(WorkspaceOperationView {
                    kind: WorkspaceOperationKindView::Elementwise("capture_cast_f32"),
                    inputs: WorkspaceLayoutList::Views(&inputs),
                    outputs: WorkspaceLayoutList::Views(&outputs)
                })
                .unwrap()
                .is_none());
        }
        let strided = WorkspaceRepresentation::new(WorkspaceFloatingType::Float32, false);
        let inputs = [output.with_representation(Some(strided))];
        assert_eq!(
            cpu.plan(WorkspaceOperationView {
                kind: WorkspaceOperationKindView::Elementwise("capture_cast_f32"),
                inputs: WorkspaceLayoutList::Views(&inputs),
                outputs: WorkspaceLayoutList::Views(&outputs)
            })
            .unwrap()
            .unwrap()
            .alias_input,
            Some(0)
        );
        let wrong = [WorkspaceLayoutView::new(&[1, 7, 1], WorkspaceDtype::Float32).unwrap()];
        assert!(cpu
            .plan(WorkspaceOperationView {
                kind: WorkspaceOperationKindView::Elementwise("capture_cast_f32"),
                inputs: WorkspaceLayoutList::Views(&inputs),
                outputs: WorkspaceLayoutList::Views(&wrong)
            })
            .is_err());
    }
}
