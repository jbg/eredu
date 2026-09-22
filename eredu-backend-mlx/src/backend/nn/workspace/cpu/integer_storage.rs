//! Scalar row overwrite used by packed I32 status. Structural views and
//! StaticSlice use their shared typed producers; SliceUpdate authenticates
//! both complete rows before dispatch without floating representation.
use super::*;
pub(super) fn inspect(
    operation: WorkspaceOperationView<'_>,
    mechanism: MlxCpuWorkspaceMechanisms,
) -> facts::FactResult<Option<OperationPlan>> {
    let Some([output]) = operation.outputs.array() else {
        return Ok(None);
    };
    if output.dtype() != WorkspaceDtype::Int32
        || output.representation().is_some()
        || operation
            .inputs
            .iter()
            .any(|v| v.dtype() != WorkspaceDtype::Int32 || v.representation().is_some())
    {
        return Ok(None);
    }
    let mut population = CpuPopulation::default();
    let (rank, alias, shells, bytes) = match operation.kind {
        WorkspaceOperationKindView::StaticSliceUpdate {
            starts,
            ends,
            strides,
        } => {
            let Some([input, update]) = operation.inputs.array() else {
                return Ok(None);
            };
            // This ordinary status worker writes one scalar into a column.
            // The native source independently verifies the physical complete
            // rows and matching dtype before its two existing copy tasks.
            if input.shape().len() != 2
                || input.shape()[1] != 1
                || input.shape()[0] <= 0
                || update.shape() != [1, 1]
                || output.shape() != input.shape()
                || starts.len() != 2
                || ends.len() != 2
                || strides != [1, 1]
                || starts[1] != 0
                || ends[1] != 1
                || starts[0] < 0
                || ends[0] != starts[0] + 1
                || ends[0] > input.shape()[0]
            {
                return Ok(None);
            }
            let whole = input.shape() == update.shape();
            if !whole {
                let Some(native) = OperationEvent::cpu_static_update_layout(
                    2,
                    usize::try_from(input.elements()?)?,
                    1,
                    false,
                ) else {
                    return Ok(None);
                };
                if native.backing_births() != 1 || population.copy(native, 2).is_none() {
                    return Ok(None);
                }
            }
            population.controls = population
                .controls
                .checked_add(
                    safemlx::Array::static_slice_update_control_bytes()
                        .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?,
                )
                .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
            (
                2,
                whole.then_some(1),
                usize::from(whole),
                if whole {
                    0
                } else {
                    mechanism
                        .allocation
                        .fixed_buffer_capacity(output.bytes()?)?
                },
            )
        }
        _ => return Ok(None),
    };
    let frames = [
        size_of::<WorkspaceOperationView<'_>>(),
        size_of::<MlxCpuWorkspaceMechanisms>(),
        size_of::<WorkspaceLayoutView<'_>>() * 4,
        size_of::<Option<[WorkspaceLayoutView<'_>; 1]>>() * 2,
        size_of::<Option<[WorkspaceLayoutView<'_>; 2]>>(),
        size_of::<CpuPopulation>(),
        size_of::<OperationPlan>(),
        size_of::<Option<OperationPlan>>(),
        size_of::<facts::FactResult<Option<OperationPlan>>>(),
        size_of::<CpuCopyEvalLayout>(),
        size_of::<Option<CpuCopyEvalLayout>>(),
        size_of::<(usize, Option<usize>, usize, u64)>(),
        size_of::<usize>() * 4,
        size_of::<bool>() * 2,
        size_of::<Option<usize>>(),
        size_of::<u64>(),
        size_of::<Result<usize, std::num::TryFromIntError>>(),
        size_of::<std::slice::Iter<'_, i32>>(),
        size_of::<[i32; 2]>(),
        size_of::<&[i32]>() * 3,
    ];
    population.controls = frames
        .into_iter()
        .try_fold(
            population
                .controls
                .checked_add(size_of_val(&frames))
                .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?,
            usize::checked_add,
        )
        .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
    Ok(Some(OperationPlan {
        dtype: WorkspaceFloatingType::Float32,
        population,
        alias_input: alias,
        output_bytes: bytes,
        scratch_bytes: 0,
        rank,
        parameter_shells: shells,
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
    use crate::backend::nn::logical_collective::{self, packed};
    use eredu_nn::Tensor;
    #[test]
    fn packed_integer_status_quotes_actual_pack_and_extraction_without_float_facts() {
        let ordinary = MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let cpu = MlxCpuWorkspaceMechanisms::new(
            ordinary.allocation(),
            MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles).unwrap(),
        );
        for world in [4, 8, 12] {
            for slot in [0, world / 2, world - 1] {
                let context = WorkspaceContext::new(cpu);
                let input = WorkspaceTensor::existing(
                    context.layout(&[1], WorkspaceDtype::Int32).unwrap(),
                    &context,
                )
                .unwrap();
                context.begin_span();
                let packed = packed::pack(
                    &logical_collective::Workspace(&context),
                    &input,
                    slot,
                    world,
                )
                .unwrap();
                assert_eq!(packed.shape(), [world as i32, 1]);
                assert_eq!(packed.layout().representation(), None);
                let layout = packed.layout().clone();
                let report = context.finish_report(&[packed]).unwrap();
                for operation in &report.operations {
                    assert!(
                        cpu.plan(operation.as_view()).unwrap().is_some(),
                        "{:?}",
                        operation.kind
                    );
                }
                let recipe = SpeculativeNumericalRecipe::inspect_cpu_outputs(
                    &report, 1, ordinary, cpu, &context,
                )
                .unwrap();
                assert_eq!(recipe.kernels, 0);
                assert!(recipe.storage.maximum_births() >= 2);
                let received = WorkspaceTensor::existing(layout, &context).unwrap();
                context.begin_span();
                let result =
                    packed::sum_result(&logical_collective::Workspace(&context), &received, slot)
                        .unwrap();
                assert_eq!(result.shape(), [1]);
                assert_eq!(result.layout().representation(), None);
                let report = context.finish_report(&[result]).unwrap();
                let recipe = SpeculativeNumericalRecipe::inspect_cpu_outputs(
                    &report, 1, ordinary, cpu, &context,
                )
                .unwrap();
                assert_eq!(recipe.storage.maximum_births(), 0);
                assert_eq!(recipe.kernels, 0);
            }
        }
    }
}
