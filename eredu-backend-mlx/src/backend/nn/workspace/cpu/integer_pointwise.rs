//! I32 arithmetic and broadcasts through the existing typed CPU binary worker.
use super::*;
pub(super) fn inspect(
    operation: WorkspaceOperationView<'_>,
    mechanism: MlxCpuWorkspaceMechanisms,
) -> facts::FactResult<Option<OperationPlan>> {
    let binary = match operation.kind {
        WorkspaceOperationKindView::Elementwise("add") => CpuBinaryOperation::Add,
        WorkspaceOperationKindView::Elementwise("subtract") => CpuBinaryOperation::Subtract,
        WorkspaceOperationKindView::Elementwise("multiply") => CpuBinaryOperation::Multiply,
        _ => return Ok(None),
    };
    if operation.inputs.len() != 2 || operation.outputs.len() != 1 {
        return Ok(None);
    }
    let output = operation.outputs.get(0).expect("one integer output");
    let rank = output.shape().len();
    if output.dtype() != WorkspaceDtype::Int32
        || output.representation().is_some()
        || rank > 4
        || output.shape().iter().any(|&n| n <= 0)
        || operation.inputs.iter().any(|input| {
            input.dtype() != WorkspaceDtype::Int32
                || input.representation().is_some()
                || input.shape().len() > rank
                || input.shape().iter().any(|&n| n <= 0)
        })
    {
        return Ok(None);
    }
    let elements = usize::try_from(output.elements()?)?;
    if elements == 0 || elements > i32::MAX as usize {
        return Ok(None);
    }
    let left = operation.inputs.get(0).expect("two integer inputs");
    let right = operation.inputs.get(1).expect("two integer inputs");
    let shape = WorkspaceBroadcastShape::new(left.shape(), right.shape())
        .map_err(|_| MlxWorkspaceFactError::descriptor("CPU integer broadcast differs"))?;
    if !shape.dimensions().eq(output.shape().iter().copied()) {
        return Err(MlxWorkspaceFactError::descriptor(
            "CPU integer output shape differs",
        ));
    }
    let mut population = CpuPopulation::default();
    // The ordinary frontend keeps both I32 operands, then broadcasts only
    // changed shapes. Native dispatch validates their actual strides and
    // backing spans; the logical dtype supplies no floating-layout evidence.
    for input in operation.inputs.iter() {
        if input.shape() != output.shape() {
            let Some(alias) =
                OperationEvent::cpu_broadcast_alias_layout(input.shape().len(), rank, false)
            else {
                return Ok(None);
            };
            if alias.backing_births() != 0 || population.copy(alias, 1).is_none() {
                return Ok(None);
            }
        }
    }
    let Some(native) =
        OperationEvent::cpu_binary_layout(binary, safemlx::Dtype::Int32, rank, elements, false)
    else {
        return Ok(None);
    };
    if population.binary(native).is_none() || population.births != 1 {
        return Ok(None);
    }
    let frames = [
        size_of::<WorkspaceOperationView<'_>>(),
        size_of::<WorkspaceLayoutView<'_>>() * 4,
        size_of::<Option<WorkspaceLayoutView<'_>>>(),
        size_of::<MlxCpuWorkspaceMechanisms>(),
        size_of::<CpuPopulation>(),
        size_of::<OperationPlan>(),
        size_of::<Option<OperationPlan>>(),
        size_of::<facts::FactResult<Option<OperationPlan>>>(),
        size_of::<safemlx::CpuBinaryEvalLayout>(),
        size_of::<Option<safemlx::CpuBinaryEvalLayout>>(),
        size_of::<CpuBinaryOperation>(),
        size_of::<CpuCopyEvalLayout>(),
        size_of::<Option<CpuCopyEvalLayout>>(),
        size_of::<WorkspaceBroadcastShape<'_>>(),
        size_of::<eredu_nn::workspace::WorkspaceLayoutIter<'_>>(),
        size_of::<std::slice::Iter<'_, i32>>(),
        size_of::<Result<WorkspaceBroadcastShape<'_>, eredu_nn::workspace::WorkspaceShapeError>>(),
        size_of::<facts::FactResult<WorkspaceBroadcastShape<'_>>>(),
        size_of_val(&shape.dimensions()),
        size_of::<(&safemlx::Array, &safemlx::Array, &safemlx::Stream)>(),
        size_of::<Result<safemlx::Array, safemlx::error::Exception>>(),
        size_of::<usize>() * 3,
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
    Ok(Some(OperationPlan {
        alias_input: None,
        seeds: 0,
        validations: 0,
        dtype: WorkspaceFloatingType::Float32,
        population,
        rank,
        parameter_shells: 0,
        output_bytes: mechanism
            .allocation
            .fixed_buffer_capacity(facts::mul(output.elements()?, 4)?)?,
        scratch_bytes: 0,
    }))
}

#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
#[path = "integer_pointwise/source_tests.rs"]
mod source_tests;

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
    fn integer_status_sources_cover_addition_and_completed_alias() {
        let ordinary = MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let cpu = MlxCpuWorkspaceMechanisms::new(
            ordinary.allocation(),
            MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles).unwrap(),
        );
        for add in [false, true] {
            let context = WorkspaceContext::new(cpu);
            let left = WorkspaceTensor::existing(
                context.layout(&[1], WorkspaceDtype::Int32).unwrap(),
                &context,
            )
            .unwrap();
            let right = WorkspaceTensor::existing(
                context.layout(&[1], WorkspaceDtype::Int32).unwrap(),
                &context,
            )
            .unwrap();
            context.begin_span();
            let output = if add {
                left.add(&right, &context).unwrap()
            } else {
                left
            };
            let report = context.finish_report(&[output]).unwrap();
            let recipe = SpeculativeNumericalRecipe::inspect_cpu_outputs(
                &report, 1, ordinary, cpu, &context,
            )
            .unwrap();
            assert_eq!(recipe.completion.graph.primitives(), usize::from(add));
            assert_eq!(recipe.storage.maximum_births(), usize::from(add));
            assert_eq!(recipe.completion.traversal.limits().roots, 1);
            assert_eq!(recipe.kernels, 0);
        }
    }
}
