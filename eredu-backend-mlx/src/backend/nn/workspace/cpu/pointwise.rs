//! Actual portable Tensor pointwise calls lowered through shared CPU workers.
use super::*;
use safemlx::CpuUnaryOperation;

pub(super) fn inspect(
    operation: WorkspaceOperationView<'_>,
    mechanism: MlxCpuWorkspaceMechanisms,
) -> facts::FactResult<Option<OperationPlan>> {
    let scalar = matches!(
        operation.kind,
        WorkspaceOperationKindView::Elementwise("multiply_scalar")
    );
    let (binary, unary) = match operation.kind {
        WorkspaceOperationKindView::Elementwise("add") => (Some(CpuBinaryOperation::Add), None),
        WorkspaceOperationKindView::Elementwise("subtract") => {
            (Some(CpuBinaryOperation::Subtract), None)
        }
        WorkspaceOperationKindView::Elementwise("multiply" | "multiply_scalar") => {
            (Some(CpuBinaryOperation::Multiply), None)
        }
        WorkspaceOperationKindView::Elementwise("divide") => {
            (Some(CpuBinaryOperation::Divide), None)
        }
        WorkspaceOperationKindView::Elementwise("square") => {
            (None, Some(CpuUnaryOperation::Square))
        }
        WorkspaceOperationKindView::Elementwise("tanh") => (None, Some(CpuUnaryOperation::Tanh)),
        WorkspaceOperationKindView::Elementwise("exp") => {
            (None, Some(CpuUnaryOperation::Exponential))
        }
        WorkspaceOperationKindView::Elementwise("log") => (None, Some(CpuUnaryOperation::Log)),
        _ => return Ok(None),
    };
    if operation.outputs.len() != 1
        || operation.inputs.len() != if binary.is_some() && !scalar { 2 } else { 1 }
    {
        return Err(MlxWorkspaceFactError::descriptor(
            "CPU pointwise source input/output population differs",
        ));
    }
    let output = operation.outputs.get(0).expect("one pointwise output");
    let first = operation.inputs.get(0).expect("pointwise source input");
    let Some(input_dtype) = first.representation().map(|r| r.dtype()) else {
        return Ok(None);
    };
    // multiply_scalar's ordinary eager F32 scalar promotes either half input
    // to F32. Quote the real input cast; do not relabel the half source or round
    // the F32 result back to its input representation.
    let right_dtype = match operation.inputs.get(1) {
        Some(right) => {
            let Some(representation) = right.representation() else {
                return Ok(None);
            };
            Some(representation.dtype())
        }
        None => None,
    };
    let dtype = if scalar {
        WorkspaceFloatingType::Float32
    } else {
        right_dtype.map_or(input_dtype, |right| {
            super::super::representation::promote(input_dtype, right)
        })
    };
    let native_input = match input_dtype {
        WorkspaceFloatingType::Float32 => safemlx::Dtype::Float32,
        WorkspaceFloatingType::Bfloat16 => safemlx::Dtype::Bfloat16,
        WorkspaceFloatingType::Float16 => safemlx::Dtype::Float16,
    };
    let native_dtype = match dtype {
        WorkspaceFloatingType::Float32 => safemlx::Dtype::Float32,
        WorkspaceFloatingType::Bfloat16 => safemlx::Dtype::Bfloat16,
        WorkspaceFloatingType::Float16 => safemlx::Dtype::Float16,
    };
    let rank = output.shape().len();
    if rank > 4
        || output.dtype() != WorkspaceDtype::Float32
        || output.shape().iter().any(|&n| n <= 0)
    {
        return Ok(None);
    }
    let elements = usize::try_from(output.elements()?)?;
    if elements > i32::MAX as usize {
        return Ok(None);
    }
    // Binary General dispatch already quotes rank-bounded collapse and
    // iterator controls for strided operands, including the eager scalar path.
    // Complete F32 rows may have gaps in either operand; promotion casts use
    // the same General-copy source and the consumer validates actual spans.
    let strided_rows = binary.is_some() && dtype == WorkspaceFloatingType::Float32;
    for input in operation.inputs.iter() {
        if input.dtype() != WorkspaceDtype::Float32
            || input.shape().len() > rank
            || input.shape().iter().any(|&n| n <= 0)
            || !input
                .representation()
                .is_some_and(|r| r.row_contiguous() || (strided_rows && r.last_axis_contiguous()))
        {
            return Ok(None);
        }
    }
    if let Some(right) = operation.inputs.get(1) {
        let shape = WorkspaceBroadcastShape::new(first.shape(), right.shape())
            .map_err(|_| MlxWorkspaceFactError::descriptor("CPU pointwise broadcast differs"))?;
        if !shape.dimensions().eq(output.shape().iter().copied()) {
            return Err(MlxWorkspaceFactError::descriptor(
                "CPU pointwise output shape differs",
            ));
        }
    } else if first.shape() != output.shape() {
        return Err(MlxWorkspaceFactError::descriptor(
            "CPU unary output shape differs",
        ));
    }
    let mut population = CpuPopulation::default();
    let mut cast_births = 0usize;
    let mut cast_bytes = 0u64;
    let source = (|| {
        if scalar {
            population.copy(
                OperationEvent::cpu_cast_layout(native_input, native_dtype, rank, elements, false)?,
                1,
            )?;
            population.copy(
                OperationEvent::cpu_cast_layout(native_dtype, native_dtype, 0, 1, false)?,
                1,
            )?;
            population.copy(
                OperationEvent::cpu_broadcast_alias_layout(rank, rank, false)?,
                1,
            )?;
            population.copy(
                OperationEvent::cpu_broadcast_alias_layout(0, rank, false)?,
                1,
            )?;
        }
        for input in operation.inputs.iter() {
            let source_dtype = input.representation()?.dtype();
            if !scalar && source_dtype != dtype {
                // Native binary promotion casts before broadcast. A stored
                // scalar converts one element, not an output-sized replica;
                // a differing full operand owns its own complete cast backing.
                let from = match source_dtype {
                    WorkspaceFloatingType::Float32 => safemlx::Dtype::Float32,
                    WorkspaceFloatingType::Float16 => safemlx::Dtype::Float16,
                    WorkspaceFloatingType::Bfloat16 => safemlx::Dtype::Bfloat16,
                };
                let count = usize::try_from(input.elements().ok()?).ok()?;
                let cast = OperationEvent::cpu_cast_layout(
                    from,
                    native_dtype,
                    input.shape().len(),
                    count,
                    false,
                )?;
                if dtype != WorkspaceFloatingType::Float32 || cast.backing_births() != 1 {
                    return None;
                }
                cast_births = cast_births.checked_add(1)?;
                cast_bytes = cast_bytes.checked_add(
                    mechanism
                        .allocation
                        .fixed_buffer_capacity(u64::try_from(count).ok()?.checked_mul(4)?)
                        .ok()?,
                )?;
                population.copy(cast, 1)?;
            }
            if input.shape() != output.shape() {
                population.copy(
                    OperationEvent::cpu_broadcast_alias_layout(input.shape().len(), rank, false)?,
                    1,
                )?;
            }
        }
        if let Some(binary) = binary {
            population.binary(OperationEvent::cpu_binary_layout(
                binary,
                native_dtype,
                rank,
                elements,
                false,
            )?)?;
        } else {
            population.unary(OperationEvent::cpu_unary_layout(
                unary?,
                native_dtype,
                rank,
                false,
            )?)?;
        }
        Some(())
    })();
    if source.is_none() || population.births != if scalar { 3 } else { 1 + cast_births } {
        return Ok(None);
    }
    let frames = [
        size_of::<OperationPlan>(),
        size_of::<Option<OperationPlan>>(),
        size_of::<CpuPopulation>() * 2,
        size_of::<WorkspaceOperationView<'_>>(),
        size_of::<WorkspaceLayoutView<'_>>() * 3,
        size_of::<MlxCpuWorkspaceMechanisms>(),
        size_of::<WorkspaceFloatingType>() * 3,
        size_of::<Option<WorkspaceFloatingType>>(),
        size_of::<safemlx::Dtype>() * 3,
        size_of::<WorkspaceRepresentation>(),
        size_of::<Option<WorkspaceRepresentation>>(),
        size_of::<Result<u64, MlxWorkspaceFactError>>(),
        size_of::<WorkspaceBroadcastShape<'_>>(),
        size_of::<Option<CpuBinaryOperation>>(),
        size_of::<Option<CpuUnaryOperation>>(),
        size_of::<CpuCopyEvalLayout>(),
        size_of::<Option<CpuCopyEvalLayout>>(),
        size_of::<safemlx::CpuBinaryEvalLayout>(),
        size_of::<Option<safemlx::CpuBinaryEvalLayout>>(),
        size_of::<safemlx::CpuUnaryEvalLayout>(),
        size_of::<Option<safemlx::CpuUnaryEvalLayout>>(),
        size_of::<Option<()>>(),
        size_of::<usize>() * 6,
        size_of::<u64>() * 3,
        size_of::<bool>() * 2,
        size_of::<Option<WorkspaceLayoutView<'_>>>(),
        size_of::<Result<u64, eredu_nn::workspace::WorkspaceLayoutError>>(),
        size_of::<Option<u64>>(),
        size_of::<std::ops::RangeInclusive<usize>>(),
        size_of::<std::iter::Enumerate<eredu_nn::workspace::WorkspaceLayoutIter<'_>>>(),
        size_of::<std::slice::Iter<i32>>(),
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
    let output_bytes = mechanism
        .allocation
        .fixed_buffer_capacity(facts::mul(output.elements()?, 4)?)?;
    let scalar_bytes = mechanism.allocation.fixed_buffer_capacity(4)?;
    Ok(Some(OperationPlan {
        alias_input: None,
        seeds: usize::from(scalar),
        validations: 0,
        dtype,
        population,
        rank,
        parameter_shells: 0,
        // Identity casts may alias; half input casts really widen to F32.
        // Both use the full-input F32 copy envelope, scalar copy and real seed.
        scratch_bytes: facts::add(
            cast_bytes,
            if scalar {
                facts::add(output_bytes, facts::mul(scalar_bytes, 2)?)?
            } else {
                0
            },
        )?,
        output_bytes,
    }))
}

#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
mod strided_tests;

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
    fn cpu_strided_binary_rows_have_a_complete_same_worker_plan() {
        let ordinary = MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let selected =
            MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles).unwrap();
        let cpu = MlxCpuWorkspaceMechanisms::new(ordinary.allocation(), selected);
        let context = WorkspaceContext::new(cpu);
        let representation = WorkspaceRepresentation::new(WorkspaceFloatingType::Float32, false)
            .with_last_axis_contiguous(true)
            .with_element_strides(&[1, 48, 1])
            .unwrap();
        let value = || {
            WorkspaceTensor::existing(
                context
                    .layout(&[1, 2, 16], WorkspaceDtype::Float32)
                    .unwrap()
                    .with_representation(Some(representation)),
                &context,
            )
            .unwrap()
        };
        let left = value();
        let right = value();
        context.begin_span();
        let output = left.multiply(&right, &context).unwrap();
        let report = context.report(&[output]).unwrap();
        let plan = cpu.plan(report.operations[0].as_view()).unwrap().unwrap();
        assert_eq!(plan.population.births, 1);
        assert_eq!(plan.scratch_bytes, 0);
        SpeculativeNumericalRecipe::inspect_cpu_equations(&report, ordinary, cpu, &context)
            .unwrap();
    }

    #[test]
    fn cpu_same_f16_pointwise_keeps_half_rounding_source_without_promoted_copies() {
        let ordinary = MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let selected =
            MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles).unwrap();
        let cpu = MlxCpuWorkspaceMechanisms::new(ordinary.allocation(), selected);
        let context = WorkspaceContext::new(cpu);
        let known = Some(WorkspaceRepresentation::new(
            WorkspaceFloatingType::Float16,
            true,
        ));
        let value = |shape: &[i32]| {
            WorkspaceTensor::existing(
                context
                    .layout(shape, WorkspaceDtype::Float32)
                    .unwrap()
                    .with_representation(known),
                &context,
            )
            .unwrap()
        };
        let input = value(&[2, 19]);
        let gain = value(&[19]);
        context.begin_span();
        let product = input.multiply(&gain, &context).unwrap();
        let square = product.square(&context).unwrap();
        let bounded = square.tanh(&context).unwrap();
        let output = bounded.add(&input, &context).unwrap();
        assert_eq!(output.layout().representation(), known);
        let report = context.report(&[output]).unwrap();
        assert_eq!(report.operations.len(), 4);
        for (index, operation) in report.operations.iter().enumerate() {
            let plan = cpu.plan(operation.as_view()).unwrap().unwrap();
            assert_eq!(plan.dtype, WorkspaceFloatingType::Float16);
            assert_eq!(plan.population.births, 1);
            assert_eq!(plan.population.primitives, if index == 0 { 2 } else { 1 });
            assert_eq!(
                plan.scratch_bytes, 0,
                "same precision needs no promotion copy"
            );
        }
        SpeculativeNumericalRecipe::inspect_cpu_equations(&report, ordinary, cpu, &context)
            .unwrap();
        let mut missing = report.operations[0].clone();
        missing.inputs[0] = missing.inputs[0].clone().with_representation(None);
        assert!(cpu.plan(missing.as_view()).unwrap().is_none());
    }
}
